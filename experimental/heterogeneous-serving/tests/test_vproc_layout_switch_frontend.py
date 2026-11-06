"""In-engine layout switch, driver side (non-transparent H <-> S-dp of ONE engine over all four hosts).

- ``BlockController.switch_layout`` over in-process fake kernels and frontends (no jax, no Ray): protocol
  order, built flags, admission blocked during the switch and resumed after, validation, drain failure
  (engine keeps serving), failure after the hosts moved (engine left stopped, no rebuild, ``rebuild()``
  brings it back).
- The same over a real 4 x 4 CPU block of Ray host kernels with ``SyntheticFrontend`` actors: H -> S-dp ->
  H -> S-dp, tokens against the oracle after every switch, one runtime, same kernel pids, same frontend actor.
- ``VllmFrontendCore.switch_layout`` with fake executor/core/schedulers shaped like vLLM's ``Scheduler`` and
  tpu-inference's ``DPScheduler`` (neither vLLM nor torch constructs on CPU here): KV-init sequence and
  segments, scheduler swap, parked schedulers, DP capacity accounting, failure isolation, shutdown.
- Static checks of every vLLM/tpu-inference attribute the frontend touches against the fork sources.
"""
import ast
import asyncio
import contextlib
import dataclasses
import os
import sys
import threading
import time
import types
import unittest
from collections import deque
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from hetero.vproc import frontend
from hetero.vproc.controller import (BlockController, EngineNotFound, KernelDead, LayoutSwitchFailed,
                                     Timeouts)
from hetero.vproc.endpoints import InMemoryEndpoints
from hetero.vproc.synthetic_engine import expected_tokens
from hetero.vproc.types import IN_ENGINE_LAYOUTS, EngineSpec
from tests.test_vproc_controller import FakeFrontend, FakeKernel, LoggedEndpoints, Cluster
from tests.test_vproc_frontend import FakeDriver, FakeVllmFrontend, stub_vllm

FAST = Timeouts(describe_s=2, health_s=0.5, stop_s=10, reshard_s=10, start_s=10, step_s=10,
                drain_wait_s=5, serve_wait_s=10, layout_s=10)
FAKE_REQUEST = {"prompts": [[1, 2, 3], [4, 5]]}
REQUEST = {"prompt": [[1, 2, 3], [4, 5], [9]], "steps": 6}
SEED = 7
H, SDP = (16, 1), (8, 2)


def engine_a(layout=H, seed=1):
    return EngineSpec("A", "synthetic", (0, 1, 2, 3), layout[0], {"seed": seed}, 0, layout[1])


# ------------------------------------------------------------------------------ controller fakes


class LKernel(FakeKernel):
    """FakeKernel with ``switch_engine_layout``: ``built`` on the first visit of a (tp, dp), all hosts agree."""

    def __init__(self, rank, cluster):
        super().__init__(rank, cluster)
        self.visited = set()
        self.gate = None
        self.lie_built = None
        self.drift_after_sdp_calls = None  # tokens change from this S-dp serve on (tests the "later visits equal" gate)
        self.sdp_calls = 0

    def start_engine(self, spec, resident=None):
        result = super().start_engine(spec, resident)
        self.visited = {spec.layout}
        return result

    def engine_call(self, method, *args):
        result = super().engine_call(method, *args)
        if self.spec.layout == SDP:
            self.sdp_calls += 1
            if self.drift_after_sdp_calls is not None and self.sdp_calls > self.drift_after_sdp_calls:
                result = {"tokens": [[(t + 1) % 50 for t in row] for row in result["tokens"]]}
        return result

    def switch_engine_layout(self, target):
        self._enter("switch_engine_layout")
        assert self.spec is not None and target.engine_id == self.spec.engine_id
        self.cluster.record("switch_engine_layout", self.rank, target.layout)
        if self.cluster.switch_gate is not None:
            self.cluster.switch_gate.wait(10)
        t0 = self.now()
        built = target.layout not in self.visited
        self.visited.add(target.layout)
        self.spec = target
        if self.lie_built is not None:
            built = self.lie_built
        return {"t0": t0, "t1": self.now(), "built": built, "layout": target.layout,
                "segments": {"reshard": 0.001, "compile": 0.002 if built else 0.0}, "compiles": 3 if built else 0}


class LFrontend(FakeFrontend):
    def __init__(self, *args):
        super().__init__(*args)
        self.draining, self.idle_answer, self.switches, self.served_layouts = False, True, [], []
        self.fail_switch = None
        self.visited = {self.spec.layout}

    def drain(self, timeout_s=0):
        self.draining = True
        return super().drain(timeout_s)

    def await_idle(self, timeout_s=0):
        self._enter()
        return {"idle": self.idle_answer, "inflight": self.inflight}

    def resume_admission(self):
        self._enter()
        self.draining = False
        self.cluster.record("fe.resume", self.spec.engine_id)
        return {"draining": False}

    def switch_layout(self, target, built):
        self._enter()
        self.cluster.record("fe.switch_layout", self.spec.engine_id, target.layout, built)
        if self.fail_switch is not None:
            raise self.fail_switch
        if not self.draining or self.inflight:
            raise RuntimeError("switch_layout needs a drained, idle frontend")
        if built == (target.layout in self.visited) and not built:
            raise ValueError("built=False for a layout that was never built")
        self.visited.add(target.layout)
        self.spec = target
        self.switches.append((target.layout, built))
        return {"built": built, "layout": list(target.layout), "compiles": 1 if built else 0,
                "segments": {"kv_init": 0.01 if built else 0.0, "swap": 0.001, "total": 0.02}}

    def generate(self, request):
        self.served_layouts.append(self.spec.layout)
        return super().generate(request)

    def identity(self):
        body = super().identity()
        body["tp_size"], body["dp_size"] = self.spec.tp_size, self.spec.dp_size
        return body


class LCluster(Cluster):
    def __init__(self, hosts=4):
        self.switch_gate = None
        super().__init__(hosts)

    def spawn(self):
        self.runtimes += 1
        self.runtime_id = f"runtime-{self.runtimes}"
        self.kernels = [LKernel(r, self) for r in range(self.hosts)]
        self.all_kernels.append(self.kernels)
        return self.kernels

    def make_frontend(self, spec, refs, settings, name):
        self.record("fe.create", spec.engine_id)
        front = LFrontend(spec, refs, settings, name, self, list(self.kernels))
        self.frontends.append(front)
        self.by_name[name] = front
        return front


def make_fake():
    cluster = LCluster()
    endpoints = LoggedEndpoints(cluster)
    controller = BlockController(cluster.kernels, endpoints, respawn=cluster.spawn, timeouts=FAST,
                                 engine_model={"seed": 1}, frontend_factory=cluster.make_frontend)
    return cluster, endpoints, controller


class ControllerLayoutSwitchFakes(unittest.TestCase):
    def test_h_sdp_h_sdp_protocol_flags_and_identity(self):
        cluster, endpoints, ctl = make_fake()
        deploy = ctl.deploy([engine_a(H)])
        self.assertTrue(deploy["ok"])
        runtime, front = ctl.runtime_ids(), ctl.frontends["A"]
        ops_after_deploy = list(endpoints.ops())
        pids = [d.pid for d in ctl.describe()]

        reports = [ctl.switch_layout("A", target) for target in ("S-dp", "H", "S-dp")]
        self.assertEqual([r["built"] for r in reports], [True, False, False])
        self.assertEqual([r["target_layout"] for r in reports], [[8, 2], [16, 1], [8, 2]])
        self.assertEqual([r["source_layout"] for r in reports], [[16, 1], [8, 2], [16, 1]])
        for report in reports:
            self.assertEqual([p["name"] for p in report["phases"]],
                             ["drain", "worker_switch", "frontend_switch", "publish"])
            self.assertTrue(all(p["ok"] for p in report["phases"]))
            self.assertEqual(report["kind"], "layout_switch")
            self.assertLess(abs(report["unaccounted_s"]), 0.1)
            self.assertEqual(set(report["worker_segments"]), {"0", "1", "2", "3"})
            self.assertEqual(report["runtime_ids"], runtime)
            self.assertEqual(report["host_pids"], pids)
            self.assertEqual(report["generation"], 0)
        self.assertEqual(reports[0]["compiles"], 3 + 1)  # kernels (max over hosts) + the frontend's own
        self.assertEqual(reports[1]["compiles"], 0)
        self.assertEqual(reports[0]["frontend_segments"].get("kv_init"), 0.01)
        # same engine id, same endpoint, same frontend object, same published identity: no withdraw/publish
        self.assertIs(ctl.frontends["A"], front)
        self.assertEqual(endpoints.ops(), ops_after_deploy)
        self.assertEqual(set(endpoints.published), {"A"})
        self.assertEqual(ctl.engines["A"].layout, SDP)
        self.assertEqual(ctl.frontend_specs["A"].layout, SDP)
        self.assertEqual(ctl.last_layout[1][0].layout, SDP)
        self.assertEqual((reports[0]["layout_id"], reports[0]["layout_digest"]),
                         (deploy["layout_id"], deploy["layout_digest"]))
        self.assertEqual(len(cluster.frontends), 1)
        self.assertEqual(cluster.events("start", "stop"), [e for e in cluster.log if e[0] in ("start", "stop")])
        self.assertEqual(len(cluster.events("start")), 4)  # only the deploy built engines
        self.assertEqual(front.switches, [(SDP, True), (H, False), (SDP, False)])
        self.assertEqual(ctl.measure.switches[-3:], reports)
        self.assertEqual(ctl.measure.report()["verdict"]["status"], "passed")
        # serving continues on the new layout, same engine id
        out = ctl.serve("A", FAKE_REQUEST)
        self.assertEqual(out["serve"]["hosts"], [0, 1, 2, 3])
        self.assertEqual(front.served_layouts[-1], SDP)

    def test_deploy_can_start_in_sdp_and_target_forms(self):
        _, _, ctl = make_fake()
        ctl.deploy([engine_a(SDP)])  # tp 8 x dp 2 == 4 hosts x 4 chips: the chips check uses spec.chips
        self.assertEqual(ctl.engines["A"].layout, SDP)
        for target, expected in (((16, 1), H), (engine_a(SDP), SDP), ("H", H), ((8, 2), SDP)):
            ctl.switch_layout("A", target)
            self.assertEqual(ctl.engines["A"].layout, expected)
        self.assertTrue(ctl._same_engine(engine_a(H), engine_a(H)))
        self.assertFalse(ctl._same_engine(engine_a(H), engine_a(SDP)))
        self.assertFalse(ctl._same_engine(engine_a(SDP), dataclasses.replace(engine_a(SDP), tp_size=8, dp_size=1)))

    def test_noop_and_validation_leave_the_engine_serving(self):
        cluster, endpoints, ctl = make_fake()
        ctl.deploy([engine_a(H)])
        self.assertTrue(ctl.switch_layout("A", "H")["noop"])
        bad = [("S-block", ValueError), ((8, 1), ValueError), ((16, 2), ValueError), ((3, 5), ValueError),
               (dataclasses.replace(engine_a(SDP), hosts=(0, 1), tp_size=4), ValueError),
               (dataclasses.replace(engine_a(SDP), model={"seed": 9}), ValueError)]
        for target, error in bad:
            with self.assertRaises(error, msg=str(target)):
                ctl.switch_layout("A", target)
        with self.assertRaises(EngineNotFound):
            ctl.switch_layout("nope", "S-dp")
        self.assertEqual(cluster.events("switch_engine_layout"), [])
        self.assertEqual(ctl.engines["A"].layout, H)
        ctl.serve("A", FAKE_REQUEST)

    def test_admission_is_blocked_during_the_switch_and_resumes_after(self):
        cluster, _, ctl = make_fake()
        ctl.deploy([engine_a(H)])
        front = ctl.frontends["A"]
        cluster.switch_gate = threading.Event()
        report, served = {}, {}
        switcher = threading.Thread(target=lambda: report.update(ctl.switch_layout("A", "S-dp")))
        switcher.start()
        deadline = time.time() + 5
        while not cluster.events("switch_engine_layout") and time.time() < deadline:
            time.sleep(0.01)
        self.assertTrue(cluster.events("switch_engine_layout"))
        served_event = threading.Event()

        def serve():
            served.update(ctl.serve("A", FAKE_REQUEST))
            served_event.set()

        client = threading.Thread(target=serve)
        client.start()
        self.assertFalse(served_event.wait(0.4))  # held by the admission gate while the hosts switch
        self.assertEqual(cluster.events("engine_call"), [])
        cluster.switch_gate.set()
        switcher.join(10)
        client.join(10)
        self.assertTrue(served_event.is_set())
        self.assertTrue(report["ok"])
        self.assertEqual(front.served_layouts, [SDP])  # served after, on the new layout
        self.assertFalse(front.draining)
        self.assertIn(("fe.resume", "A"), cluster.log)

    def test_drain_failure_before_the_hosts_moved_keeps_the_engine_serving(self):
        cluster, endpoints, ctl = make_fake()
        ctl.deploy([engine_a(H)])
        front = ctl.frontends["A"]
        front.idle_answer = False
        with self.assertRaisesRegex(RuntimeError, "not idle"):
            ctl.switch_layout("A", "S-dp")
        self.assertEqual(cluster.events("switch_engine_layout"), [])
        self.assertIn("A", ctl.engines)
        self.assertFalse(front.draining)  # admission resumed
        self.assertFalse(ctl.measure.switches[-1]["ok"])
        self.assertFalse(ctl.measure.switches[-1]["stopped"])
        front.idle_answer = True
        ctl.serve("A", FAKE_REQUEST)
        self.assertTrue(ctl.switch_layout("A", "S-dp")["ok"])

    def stopped_engine(self, break_what):
        cluster, endpoints, ctl = make_fake()
        ctl.deploy([engine_a(H)])
        front = ctl.frontends["A"]
        break_what(cluster, front, ctl)
        with self.assertRaises(LayoutSwitchFailed) as caught:
            ctl.switch_layout("A", "S-dp")
        return cluster, endpoints, ctl, front, caught.exception

    def check_left_stopped(self, cluster, endpoints, ctl, front, error):
        self.assertFalse(error.report["ok"])
        self.assertTrue(error.report["stopped"])
        self.assertEqual(ctl.engines, {})
        self.assertEqual(ctl.frontends, {})
        self.assertTrue(front.dead)
        self.assertEqual(endpoints.published, {})
        self.assertEqual(endpoints.ops()[-1], "withdraw")
        self.assertEqual(ctl.measure.rebuilds, [])  # never an automatic rebuild
        self.assertFalse(ctl.down)
        with self.assertRaises(EngineNotFound):
            ctl.serve("A", FAKE_REQUEST)
        self.assertFalse(ctl._blocked)  # admission is not left blocked
        self.assertFalse(ctl.measure.switches[-1]["ok"])
        # ... and it stays stopped until the operator rebuilds: the rebuild redeploys the layout recorded
        rebuild = ctl.rebuild("operator")
        self.assertTrue(rebuild["ok"])
        self.assertEqual(ctl.engines["A"].layout, H)
        self.assertEqual(ctl.engines["A"].generation, 1)
        ctl.serve("A", FAKE_REQUEST)

    def test_failing_worker_switch_leaves_the_engine_stopped_without_rebuild(self):
        def fail(cluster, front, ctl):
            cluster.fail[("switch_engine_layout", 2)] = RuntimeError("worker reshard failed")
        cluster, endpoints, ctl, front, error = self.stopped_engine(fail)
        self.assertEqual([p["name"] for p in error.report["phases"]], ["drain", "worker_switch"])
        self.assertFalse(error.report["phases"][1]["ok"])
        self.check_left_stopped(cluster, endpoints, ctl, front, error)

    def test_dead_host_during_the_worker_switch_is_reported_not_rebuilt(self):
        def fail(cluster, front, ctl):
            cluster.fail[("switch_engine_layout", 1)] = KernelDead("host 1 is gone")
        cluster, endpoints, ctl, front, error = self.stopped_engine(fail)
        self.assertEqual(ctl.measure.rebuilds, [])
        self.assertTrue(error.report["stopped"])

    def test_failing_frontend_switch_leaves_the_engine_stopped(self):
        def fail(cluster, front, ctl):
            front.fail_switch = RuntimeError("scheduler build failed")
        cluster, endpoints, ctl, front, error = self.stopped_engine(fail)
        self.assertEqual([p["name"] for p in error.report["phases"]],
                         ["drain", "worker_switch", "frontend_switch"])
        self.check_left_stopped(cluster, endpoints, ctl, front, error)

    def test_hosts_that_disagree_on_built_fail_the_switch(self):
        def lie(cluster, front, ctl):
            cluster.kernels[3].lie_built = False  # the others say built=True (first visit), host 3 says it is not
        cluster, endpoints, ctl, front, error = self.stopped_engine(lie)
        self.assertIn("disagree", error.report["error"])
        self.assertIn("built_by_host", error.report["phases"][1])


# ------------------------------------------------------------------------------ real Ray block


class ControllerLayoutSwitchRay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import ray
        if ray.is_initialized():
            ray.shutdown()
        ray.init(address="local", num_cpus=8, include_dashboard=False, logging_level="ERROR")
        from hetero.vproc.host_kernel import spawn_block
        cls.actors = spawn_block(4, 4, name_prefix="vproc-kernel-e0", actor_options={"namespace": "omp_hetero"})
        cls.endpoints = InMemoryEndpoints()
        cls.ctl = BlockController(
            cls.actors, cls.endpoints, engine_model={"seed": SEED},
            timeouts=Timeouts(health_s=10, start_s=300, reshard_s=300, stop_s=120, step_s=120, layout_s=300))

    @classmethod
    def tearDownClass(cls):
        import ray
        try:
            cls.ctl.close()
        finally:
            ray.shutdown()

    def serve_and_check(self, label):
        spec = self.ctl.engines["A"]
        out = self.ctl.serve("A", REQUEST)
        self.assertTrue(out["serve"]["ranks_agree"], label)
        check = self.ctl.measure.check_tokens(f"{label}/A", out["tokens"], expected_tokens(spec, REQUEST))
        self.assertTrue(check["equal"], check)

    def test_h_sdp_h_sdp_on_a_cpu_block(self):
        import ray
        ctl = self.ctl
        spec = EngineSpec("A", "synthetic", (0, 1, 2, 3), 16, {"seed": SEED})
        ctl.deploy([spec])
        runtime, pids = ctl.runtime_ids(), [d.pid for d in ctl.describe()]
        actor = ray.get_actor("vproc-A-g0", namespace="omp_hetero")
        self.serve_and_check("H")
        reports = []
        for label, target in (("S-dp", "S-dp"), ("H", "H"), ("S-dp-again", "S-dp"), ("H-again", "H")):
            report = ctl.switch_layout("A", target)
            reports.append(report)
            self.assertTrue(report["ok"], report)
            self.serve_and_check(label)
            identity = ray.get(actor.identity.remote(), timeout=30)
            self.assertEqual((identity["tp_size"], identity["dp_size"]), tuple(IN_ENGINE_LAYOUTS[target]))
            self.assertEqual(ray.get(actor.status.remote())["state"], "ready")
        self.assertEqual([r["built"] for r in reports], [True, False, False, False])
        self.assertEqual(reports[0]["frontend_segments"].keys() >= {"verify", "swap"}, True)
        self.assertEqual(ctl.runtime_ids(), runtime)
        self.assertEqual([d.pid for d in ctl.describe()], pids)
        self.assertEqual(ray.get_actor("vproc-A-g0", namespace="omp_hetero")._actor_id, actor._actor_id)
        self.assertEqual(ctl.engines["A"].generation, 0)
        for report in reports:
            self.assertEqual([p["name"] for p in report["phases"]],
                             ["drain", "worker_switch", "frontend_switch", "publish"])
            self.assertEqual(report["runtime_ids"], runtime)
        print("compiles per switch:", [r["compiles"] for r in reports], flush=True)
        self.assertEqual(self.endpoints.ops().count("publish"), 1)  # the deploy's only
        self.assertEqual(ctl.measure.report()["verdict"]["status"], "passed")

    def test_requests_during_a_switch_wait_and_succeed_after(self):
        ctl = self.ctl
        if "A" not in ctl.engines:
            ctl.deploy([EngineSpec("A", "synthetic", (0, 1, 2, 3), 16, {"seed": SEED})])
        target = "S-dp" if ctl.engines["A"].layout == H else "H"
        results = []

        def clients():
            for _ in range(8):
                results.append(ctl.serve("A", REQUEST))

        thread = threading.Thread(target=clients)
        thread.start()
        report = ctl.switch_layout("A", target)
        thread.join(60)
        self.assertTrue(report["ok"])
        self.assertEqual(len(results), 8)
        self.assertTrue(all(r["serve"]["ranks_agree"] for r in results))
        for result in results:  # token content is layout independent (exact integer forward)
            self.assertEqual(result["tokens"], expected_tokens(ctl.engines["A"], REQUEST))


# ------------------------------------------------------------------------------ vLLM frontend fakes


class FakeBlockPool:
    def __init__(self, blocks):
        self.num_gpu_blocks, self.free = blocks, blocks - 1

    def get_num_free_blocks(self):
        return self.free


class FakeScheduler:
    """Shaped like vllm.v1.core.sched.scheduler.Scheduler (the attributes the frontend reads)."""

    def __init__(self, blocks):
        self.kv_cache_manager = types.SimpleNamespace(block_pool=FakeBlockPool(blocks))
        self.running, self.waiting = [], []
        self.busy = False
        self.resets, self.shutdowns = 0, 0

    def has_requests(self):
        return self.busy or bool(self.running or self.waiting)

    def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
        self.resets += 1
        return True

    def shutdown(self):
        self.shutdowns += 1


class FakeDPScheduler:
    """Shaped like tpu_inference DPScheduler: no kv_cache_manager/running/waiting; per-rank configs, counts, stats."""

    def __init__(self, per_rank_blocks, dp):
        self.per_rank_kv_cache_configs = [types.SimpleNamespace(num_blocks=per_rank_blocks) for _ in range(dp)]
        self._pending_new_requests = []
        self.log_stats, self.usage = True, 0.0
        self.counts, self.busy = (0, 0), False
        self.resets, self.shutdowns = 0, 0

    def get_request_counts(self):
        return self.counts

    def make_stats(self, *args):
        return types.SimpleNamespace(kv_cache_usage=self.usage)

    def has_requests(self):
        return self.busy or any(self.counts)

    def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
        self.resets += 1
        return True

    def shutdown(self):
        self.shutdowns += 1


class FakeLayoutExecutor:
    def __init__(self, hosts, bump):
        self.timings, self._hosts, self._bump = {}, hosts, bump
        self.calls, self.started = [], []
        self.vllm_config = None
        self.fail_memory = False

    def get_kv_cache_specs(self):
        self.calls.append("get_kv_cache_specs")
        time.sleep(0.002)
        return [{}, {}]

    def determine_available_memory(self):
        self.calls.append("determine_available_memory")
        time.sleep(0.003)
        if self.fail_memory:
            raise MemoryError("profile failed")
        return [1000, 1000]

    def initialize_from_config(self, configs):
        self.calls.append("initialize_from_config")
        self.timings["kv_alloc"] = 0.25
        self.warm()

    def warm(self):
        self._hosts.compiles[0] += self._bump
        self._hosts.compiles[1] += self._bump - 1
        self.timings["warmup"] = 0.5

    def start_warmup(self):
        self.started.append("start")
        self.warm()

    def wait_for_warmup(self):
        self.started.append("wait")


def fake_config(dp, blocks=0, prefix=False):
    attrs = {name: types.SimpleNamespace(tag=f"{name}-dp{dp}") for name in (
        "model_config", "lora_config", "load_config", "parallel_config", "device_config",
        "speculative_config", "observability_config")}
    return types.SimpleNamespace(
        dp=dp, cache_config=types.SimpleNamespace(block_size=16, num_gpu_blocks=blocks,
                                                  enable_prefix_caching=prefix),
        scheduler_config=types.SimpleNamespace(async_scheduling=False), max_concurrent_batches=1, **attrs)


class FakeCore:
    def __init__(self, scheduler, executor):
        self.scheduler, self.model_executor = scheduler, executor
        self.batch_queue, self.batch_queue_size, self.log_stats = None, 1, True
        self.structured_output_manager = "som-initial"
        self.request_block_hasher, self.async_scheduling = None, False
        self.available_gpu_memory_for_kv_cache = 111
        self.vllm_config = None
        self.step_fn = None
        self.initialized = []

    def step(self):
        return None

    def _initialize_kv_caches(self, config):
        executor = self.model_executor
        executor.get_kv_cache_specs()
        self.available_gpu_memory_for_kv_cache = 1000 * (1 + config.dp)
        executor.determine_available_memory()
        config.cache_config.num_gpu_blocks = 64 * config.dp
        executor.initialize_from_config([])
        self.initialized.append(config)
        return types.SimpleNamespace(kv_cache_groups=[object()], num_blocks=64 * config.dp)


class SwitchFakeFrontend(FakeVllmFrontend):
    """VllmFrontendCore over fake pieces; layouts H (stock Scheduler) and S-dp (DPScheduler)."""

    def __init__(self, spec, kernels, settings=None):
        super().__init__(spec, kernels, settings)
        self.executor = FakeLayoutExecutor(self.hosts, self.BUMP)
        self.built_schedulers = []
        self.fail_scheduler = False
        self.wrong_scheduler = False

    def _verify_kernels(self, handles, timeout_s=60.0, expect_layout=None):
        self.verified = getattr(self, "verified", []) + [expect_layout]
        return {"runtime_id": "rt", "pids": [1, 2]}

    def _make_config(self, spec=None):
        spec = spec or self.spec
        return fake_config(spec.dp_size)

    def _make_llm_engine(self, config):
        self.executor.warm()
        self._executor = self.executor
        self._retarget_executor(config)  # as Executor.__init__ sets them
        core = FakeCore(FakeScheduler(64), self.executor)
        core.vllm_config = config
        self.core = core
        engine = types.SimpleNamespace(engine_core=types.SimpleNamespace(engine_core=core), vllm_config=config,
                                       model_config=config.model_config,
                                       observability_config=config.observability_config,
                                       has_unfinished_requests=lambda: False)
        return engine

    def _make_driver(self, llm_engine, loop):
        self.driver = SwitchDriver()
        return self.driver

    def _make_scheduler(self, config, kv_cache_config):
        if self.fail_scheduler:
            raise RuntimeError("DPScheduler worker did not start")
        if config.dp > 1 and not self.wrong_scheduler:
            scheduler = FakeDPScheduler(kv_cache_config.num_blocks // config.dp, config.dp)
        else:
            scheduler = FakeScheduler(kv_cache_config.num_blocks)
        self.built_schedulers.append(scheduler)
        return scheduler, f"som-dp{config.dp}", None


class SwitchDriver(FakeDriver):
    def __init__(self):
        super().__init__()
        self.lock = threading.RLock()
        self.published = 0

    @contextlib.contextmanager
    def priority_lock(self):
        with self.lock:
            yield

    def _publish_snapshot(self):
        self.published += 1


def sw_spec(layout=H):
    return EngineSpec("v-0", "vllm", (0, 1), layout[0], {}, 3, layout[1])


class VllmFrontendLayoutSwitch(unittest.TestCase):
    def build(self, **settings):
        fe = SwitchFakeFrontend(sw_spec(H), ["k0", "k1"], {"layout_id": "L", "layout_digest": "D",
                                                          "switch_idle_timeout_s": 0.3, **settings})
        asyncio.run(fe.initialize())
        return fe

    def switch(self, fe, layout, built):
        return asyncio.run(fe.switch_layout(sw_spec(layout), built))

    def test_first_visit_runs_the_kv_init_sequence_and_swaps_the_scheduler(self):
        fe = self.build()
        core, executor = fe.core, fe.executor
        stock, config_h = core.scheduler, fe.vllm_config
        self.assertEqual(os.environ.get("TPU_MULTIPROCESS_DP"), "0")
        before = list(fe.hosts.compiles)
        report = self.switch(fe, SDP, True)
        self.assertTrue(report["built"])
        self.assertEqual(report["layout"], [8, 2])
        for name in ("verify", "kv_spec", "available_memory", "kv_init", "warmup", "scheduler_build", "swap",
                     "total"):
            self.assertIn(name, report["segments"])
        seg = report["segments"]
        self.assertEqual((seg["kv_init"], seg["warmup"]), (0.25, 0.5))
        self.assertGreater(seg["kv_spec"], 0.0)
        self.assertGreater(seg["available_memory"], 0.0)
        self.assertGreaterEqual(seg["total"], seg["kv_spec"] + seg["available_memory"])
        self.assertEqual(report["compiles"], 3)  # max over the two hosts: 3 and 2
        self.assertEqual(executor.calls, ["get_kv_cache_specs", "determine_available_memory",
                                          "initialize_from_config"])
        self.assertNotIn("get_kv_cache_specs", executor.__dict__)  # timing wrappers removed
        # EngineCore state of the target layout, the previous layout parked intact
        self.assertIsInstance(core.scheduler, FakeDPScheduler)
        self.assertIs(fe.scheduler, core.scheduler)
        self.assertEqual(core.structured_output_manager, "som-dp2")
        self.assertIs(core.vllm_config, fe.vllm_config)
        self.assertIsNot(fe.vllm_config, config_h)
        self.assertEqual(fe.vllm_config.cache_config.num_gpu_blocks, 128)
        self.assertEqual(config_h.cache_config.num_gpu_blocks, 0)  # the parked config is untouched
        self.assertIs(executor.vllm_config, fe.vllm_config)
        self.assertEqual(executor.parallel_config.tag, "parallel_config-dp2")
        self.assertEqual(core.available_gpu_memory_for_kv_cache, 3000)
        self.assertEqual(stock.shutdowns, 0)  # parked, alive
        self.assertEqual(sorted(fe._layouts), [SDP, H])
        self.assertEqual(report["parked"], [[16, 1]])
        self.assertEqual((fe.spec.layout, fe.instance["tp_size"], fe.instance["dp_size"]), (SDP, 8, 2))
        self.assertEqual(fe.identity()["dp_size"], 2)
        self.assertGreaterEqual(fe.driver.published, 1)
        self.assertEqual(fe.verified[-1], SDP)
        self.assertFalse(fe._switching)
        self.assertEqual(fe.state, "ready")
        self.assertEqual(before[0] + 3, fe.hosts.compiles[0])

    def test_revisit_swaps_the_parked_scheduler_resets_its_prefix_cache_builds_nothing(self):
        fe = self.build()
        core, stock = fe.core, fe.core.scheduler
        h_memory = core.available_gpu_memory_for_kv_cache
        self.switch(fe, SDP, True)
        dp = core.scheduler
        built_before = len(core.initialized)
        report = self.switch(fe, H, False)
        self.assertFalse(report["built"])
        self.assertIs(core.scheduler, stock)  # the very object that was parked, not a new one
        self.assertEqual((stock.resets, dp.resets), (1, 0))
        self.assertEqual(len(core.initialized), built_before)  # no KV init
        self.assertEqual(core.available_gpu_memory_for_kv_cache, h_memory)  # H's value, not the target's
        self.assertEqual(report["segments"]["kv_init"], 0.0)
        self.assertEqual(report["segments"]["warmup"], 0.0)
        self.assertEqual(report["compiles"], 0)
        self.assertEqual(core.structured_output_manager, "som-initial")
        report = self.switch(fe, SDP, False)
        self.assertIs(core.scheduler, dp)
        self.assertEqual(dp.resets, 1)
        self.assertEqual(core.available_gpu_memory_for_kv_cache, 3000)
        self.assertEqual((stock.shutdowns, dp.shutdowns), (0, 0))
        self.assertEqual(len(fe.built_schedulers), 1)

    def test_rebuilding_a_visited_layout_replaces_and_stops_the_parked_scheduler(self):
        fe = self.build()
        self.switch(fe, SDP, True)
        first_dp = fe.core.scheduler
        self.switch(fe, H, False)
        self.switch(fe, SDP, True)  # the hosts rebuilt it (e.g. they restarted): a fresh scheduler
        self.assertIsNot(fe.core.scheduler, first_dp)
        self.assertEqual(first_dp.shutdowns, 1)

    def test_capacity_status_and_idleness_for_both_scheduler_kinds(self):
        fe = self.build()
        status = fe.status()
        self.assertEqual((status["total_blocks"], status["usable_blocks"], status["layout"]), (100, 99, [16, 1]))
        self.assertEqual(frontend.scheduler_capacity(fe.core.scheduler)["pools"], 1)
        self.switch(fe, SDP, True)
        dp = fe.core.scheduler
        capacity = frontend.scheduler_capacity(dp)
        self.assertEqual((capacity["total_blocks"], capacity["pools"], capacity["free_blocks"]), (128, 2, 126))
        dp.usage, dp.counts = 0.5, (3, 4)
        dp._pending_new_requests = [object()]
        capacity = frontend.scheduler_capacity(dp)
        self.assertEqual((capacity["num_running"], capacity["num_waiting"], capacity["free_blocks"]), (3, 5, 63))
        dp.usage, dp.counts, dp._pending_new_requests = 0.0, (0, 0), []
        fe.driver.snapshot = {"total_blocks": 128, "free_blocks": 126, "num_running": 0, "num_waiting": 0,
                              "pools": 2}
        kv = fe.kv_capacity()
        self.assertEqual((kv["usable_blocks"], kv["usable_tokens"]), (126, 126 * 16))
        self.assertTrue(fe.idle_snapshot()["all_blocks_free"])  # each DP rank keeps its own null block
        idle = asyncio.run(fe.await_idle(1.0))
        self.assertTrue(idle["idle"])
        self.assertEqual(idle["free_blocks"], 126)
        dp.busy = True
        self.assertFalse(asyncio.run(fe.await_idle(0.1))["idle"])
        dp.busy = False

    def test_generate_and_admission_across_switches(self):
        fe = self.build()
        with stub_vllm():
            self.assertEqual(asyncio.run(fe.generate(prompt_token_ids=[1, 2], max_tokens=3))["output_tokens"], 3)
            self.switch(fe, SDP, True)
            self.assertEqual(asyncio.run(fe.generate(prompt_token_ids=[1, 2], max_tokens=3))["output_tokens"], 3)
            self.switch(fe, H, False)
            out = asyncio.run(fe.generate({"prompt": [[1], [2]], "steps": 2}))
            self.assertEqual(len(out["tokens"]), 2)
            fe._switching = True
            with self.assertRaisesRegex(RuntimeError, "switching"):
                asyncio.run(fe.generate(prompt_token_ids=[1], max_tokens=1))
            fe._switching = False

    def test_refusals_leave_the_engine_untouched_and_unpoisoned(self):
        fe = self.build()
        core = fe.core
        with self.assertRaisesRegex(ValueError, "no parked scheduler"):
            self.switch(fe, SDP, False)  # never built
        with self.assertRaisesRegex(ValueError, "already runs"):
            self.switch(fe, H, True)
        with self.assertRaisesRegex(ValueError, "spans"):
            asyncio.run(fe.switch_layout(dataclasses.replace(sw_spec(H), tp_size=4), True))
        with self.assertRaisesRegex(ValueError, "except for"):
            asyncio.run(fe.switch_layout(dataclasses.replace(sw_spec(SDP), model={"x": 1}), True))
        core.scheduler.busy = True  # work in flight: not idle
        with self.assertRaisesRegex(RuntimeError, "not idle"):
            self.switch(fe, SDP, True)
        core.scheduler.busy = False
        self.assertIsNone(fe._poisoned)
        self.assertFalse(fe._switching)
        self.assertEqual(core.initialized.__len__(), 0)
        self.assertEqual(fe.state, "ready")

    def test_a_failed_build_poisons_and_restores_the_active_layout(self):
        for what in ("memory", "scheduler", "wrong_scheduler"):
            with self.subTest(what):
                fe = self.build()
                core, executor, config = fe.core, fe.executor, fe.vllm_config
                attrs = {n: getattr(executor, n, None) for n in fe._EXECUTOR_CONFIG_ATTRS}
                stock = core.scheduler
                if what == "memory":
                    executor.fail_memory = True
                elif what == "scheduler":
                    fe.fail_scheduler = True
                else:
                    fe.wrong_scheduler = True
                with self.assertRaises((MemoryError, RuntimeError)):
                    self.switch(fe, SDP, True)
                self.assertIsNotNone(fe._poisoned)
                self.assertIn("layout switch to (8, 2) failed", fe._poisoned)
                self.assertEqual(fe.state, "poisoned")
                self.assertFalse(fe.health()["ok"])
                self.assertFalse(fe._switching)
                self.assertIs(core.scheduler, stock)
                self.assertIs(core.vllm_config, config)
                self.assertEqual(core.available_gpu_memory_for_kv_cache, 111)  # KV init overwrote it; restored
                self.assertEqual({n: getattr(executor, n, None) for n in fe._EXECUTOR_CONFIG_ATTRS}, attrs)
                self.assertNotIn("determine_available_memory", executor.__dict__)
                self.assertEqual(sorted(fe._layouts), [H])
                for scheduler in fe.built_schedulers:  # a scheduler built for a failed switch is stopped
                    self.assertEqual(scheduler.shutdowns, 1)

    def test_timing_wrappers_and_rollback_restore_exactly_what_they_replaced(self):
        fe = self.build()
        executor = fe.executor
        custom = lambda: [{}, {}]  # an instance-level override that must survive
        executor.get_kv_cache_specs = custom
        with fe._timed_executor(executor, {"kv_spec": 0.0, "available_memory": 0.0}):
            self.assertIsNot(executor.get_kv_cache_specs, custom)
        self.assertIs(executor.get_kv_cache_specs, custom)
        self.assertNotIn("determine_available_memory", executor.__dict__)
        del executor.__dict__["speculative_config"]  # absent before the failed build
        executor.fail_memory = True
        with self.assertRaises(MemoryError):
            self.switch(fe, SDP, True)
        self.assertNotIn("speculative_config", executor.__dict__)
        self.assertIs(executor.get_kv_cache_specs, custom)

    def test_llm_engine_config_references_follow_the_swap(self):
        fe = self.build()
        self.switch(fe, SDP, True)
        engine = fe.llm_engine
        self.assertIs(engine.vllm_config, fe.vllm_config)
        self.assertEqual(engine.model_config.tag, "model_config-dp2")
        self.assertEqual(engine.observability_config.tag, "observability_config-dp2")

    def test_shutdown_stops_the_active_and_every_parked_scheduler(self):
        fe = self.build()
        stock = fe.core.scheduler
        self.switch(fe, SDP, True)
        dp = fe.core.scheduler
        snapshot = asyncio.run(fe.shutdown())
        self.assertEqual((stock.shutdowns, dp.shutdowns), (1, 1))
        self.assertEqual(snapshot["schedulers_shutdown"], 2)
        self.assertEqual(fe._layouts, {})
        self.assertEqual(fe.state, "stopped")

    def test_failing_scheduler_shutdown_is_reported_not_raised(self):
        fe = self.build()
        stock = fe.core.scheduler
        stock.shutdown = lambda: (_ for _ in ()).throw(OSError("pipe closed"))
        self.switch(fe, SDP, True)
        snapshot = asyncio.run(fe.shutdown())
        self.assertIn("OSError", snapshot["scheduler_shutdown_errors"][0])

    def test_tpu_multiprocess_dp_must_be_off(self):
        saved = os.environ.get("TPU_MULTIPROCESS_DP")
        os.environ["TPU_MULTIPROCESS_DP"] = "1"
        try:
            fe = SwitchFakeFrontend(sw_spec(H), ["k0", "k1"], {"layout_id": "L", "layout_digest": "D"})
            with self.assertRaisesRegex(RuntimeError, "TPU_MULTIPROCESS_DP"):
                asyncio.run(fe.initialize())
        finally:
            if saved is None:
                os.environ.pop("TPU_MULTIPROCESS_DP", None)
            else:
                os.environ["TPU_MULTIPROCESS_DP"] = saved
            os.environ["TPU_MULTIPROCESS_DP"] = "0"


class SyntheticFrontendRefusals(unittest.TestCase):
    def test_synthetic_switch_requires_a_visited_layout_for_built_false(self):
        spec = EngineSpec("s-0", "synthetic", (0, 1), 8, {}, 0)
        fe = frontend.SyntheticFrontendCore(spec, ["k0", "k1"], {"layout_id": "L", "layout_digest": "D"})
        fe._init_result = {}
        with self.assertRaisesRegex(ValueError, "never built"):
            asyncio.run(fe.switch_layout(dataclasses.replace(spec, tp_size=4, dp_size=2), False))
        with self.assertRaisesRegex(ValueError, "already runs"):
            asyncio.run(fe.switch_layout(spec, True))


# ------------------------------------------------------------------------------ the G3 gate script


def load_gate():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "dp_switch_gate.py"
    spec = importlib.util.spec_from_file_location("dp_switch_gate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dp_switch_gate"] = module
    spec.loader.exec_module(module)
    return module


def fake_tokens(spec, request):
    return [[(sum(p) + i) % 50 for i in range(4)] for p in request["prompts"]]


class DpSwitchGateScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = load_gate()

    def run_fake_gate(self, reps=2, sdp_reference=None, stop=False, drift_after=None):
        gate = self.gate
        cluster, endpoints, ctl = make_fake()
        for kernel in cluster.kernels:
            kernel.drift_after_sdp_calls = drift_after
        spec = engine_a(H)

        def reference_for(s):
            if s.layout == SDP and sdp_reference is not None:
                return sdp_reference
            return fake_tokens(s, FAKE_REQUEST)

        evidence = gate.new_evidence(types.SimpleNamespace(reps=reps, run_id="t", stop_on_mismatch=stop), "fake")
        try:
            gate.run_gate(reps, ctl, spec, FAKE_REQUEST, reference_for, evidence, stop)
        except gate.GateStop as error:
            evidence["failure"] = f"GateStop: {error}"
        gate.summarize(evidence)
        return evidence, ctl

    def test_compare_tokens_reports_per_prompt_positions(self):
        compare = self.gate.compare_tokens
        self.assertTrue(compare([[1, 2], [3]], [[1, 2], [3]])["equal"])
        out = compare([[1, 2, 9], [3, 4], [5]], [[1, 2, 3], [3, 4, 4], [5]])
        self.assertFalse(out["equal"])
        self.assertEqual([(m["prompt"], m["first_diff"], m["got_len"], m["ref_len"]) for m in out["mismatched"]],
                         [(0, 2, 3, 3), (1, 2, 2, 3)])
        self.assertEqual(compare([[1]], [[1], [2]])["mismatched"][0]["got_len"], None)

    def test_gate_passes_on_the_fake_block_and_records_the_sequence(self):
        evidence, ctl = self.run_fake_gate(reps=2)
        self.assertTrue(evidence["ok"], evidence["failure"] or evidence["summary"])
        summary = evidence["summary"]
        self.assertEqual(summary["built_flags"], [True, False, False, False, False, False])
        self.assertEqual([(s["target"], s["built"]) for s in summary["switches"]][:3],
                         [([8, 2], True), ([16, 1], False), ([8, 2], False)])
        self.assertEqual(summary["visits"], 7)  # deploy + 6 switches
        self.assertTrue(summary["sdp_equals_s_reference"] and summary["sdp_visits_identical"])
        self.assertTrue(all(c["ranks_agree"] for c in evidence["checks"]))
        self.assertEqual(evidence["first_sdp"], "sdp0_S-dp")
        self.assertTrue(summary["runtime_unchanged"] and summary["pids_unchanged"] and summary["frontend_unchanged"])
        first = evidence["switches"][0]
        for key in ("controller_segments", "worker_segments", "frontend_segments", "compiles", "hbm", "built"):
            self.assertIn(key, first)
        self.assertIsNone(first["cache_entries_added"])  # no cache dir given
        self.assertEqual([r["cache_hit"] for r in evidence["switches"]], [False] + [True] * 5)
        self.assertEqual(ctl.measure.report()["verdict"]["status"], "passed")

    def test_ranks_agree_none_is_not_applicable_but_false_fails(self):
        evidence, _ = self.run_fake_gate(reps=1)
        for check in evidence["checks"]:
            check["ranks_agree"] = None  # the vLLM frontend has one output path and reports None
        self.gate.summarize(evidence)
        self.assertTrue(evidence["ok"], evidence["summary"])
        evidence["checks"][1]["ranks_agree"] = False
        self.gate.summarize(evidence)
        self.assertFalse(evidence["summary"]["ranks_agree"])
        self.assertFalse(evidence["ok"])

    def test_sdp_mismatch_with_the_s_reference_is_recorded_and_the_gate_keeps_going(self):
        wrong = fake_tokens(None, FAKE_REQUEST)
        wrong[1][2] = (wrong[1][2] + 7) % 50
        evidence, _ = self.run_fake_gate(reps=1, sdp_reference=wrong)
        summary = evidence["summary"]
        self.assertFalse(summary["sdp_equals_s_reference"])
        self.assertEqual(len(evidence["switches"]), 4)  # it did not stop
        self.assertEqual(summary["sdp_mismatches"][0]["prompts"], [(1, 2)])
        self.assertTrue(all(len(m["prompts"]) == 1 for m in summary["sdp_mismatches"]))
        self.assertTrue(summary["sdp_visits_identical"])
        self.assertTrue(evidence["ok"])  # identical visits, H correct: the mismatch is a finding, not a failure

    def test_stop_on_mismatch_stops_at_the_first_one(self):
        wrong = fake_tokens(None, FAKE_REQUEST)
        wrong[0][0] = (wrong[0][0] + 1) % 50
        evidence, _ = self.run_fake_gate(reps=2, sdp_reference=wrong, stop=True)
        self.assertIn("GateStop", evidence["failure"])
        self.assertEqual(len(evidence["switches"]), 1)
        self.assertFalse(evidence["ok"])

    def test_a_later_sdp_visit_that_differs_from_the_first_fails_the_gate(self):
        evidence, ctl = self.run_fake_gate(reps=2, drift_after=1)
        summary = evidence["summary"]
        self.assertFalse(summary["sdp_visits_identical"])
        self.assertFalse(evidence["ok"])
        later = [c for c in evidence["checks"] if c.get("equal_first_sdp") is False]
        self.assertEqual(len(later), 2)
        self.assertEqual(later[0]["first_sdp_label"], "sdp0_S-dp")
        self.assertEqual(ctl.measure.report()["verdict"]["status"], "failed")

    def test_dry_run_builds_the_submit_payload(self):
        import contextlib
        import io
        import json
        args = self.gate.parse(["--dry-run", "--reps", "2", "--run-id", "unit"])
        if not args.inventory.is_file():
            self.skipTest(f"{args.inventory} not found")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.gate.submit(args)
        body = json.loads(out.getvalue())
        self.assertIn("scripts/dp_switch_gate.py --remote --reps 2 --run-id unit", body["entrypoint"])
        self.assertEqual(body["env"]["TPU_MULTIPROCESS_DP"], "0")
        self.assertEqual(body["reference_for"], {"H": "tp16", "S-dp": "tp8"})
        self.assertEqual(body["engine"]["sequence"], ["S-dp", "H", "S-dp", "H", "S-dp", "H"])
        self.assertEqual(body["prompts"], 32)

    def test_local_run_passes_and_writes_evidence(self):
        import json
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "G3-local.json"
            done = subprocess.run([sys.executable, str(Path(self.gate.__file__)), "--local", "--reps", "1",
                                   "--output", str(output)], capture_output=True, text=True, timeout=240,
                                  env={**os.environ, "JAX_PLATFORMS": "cpu"})
            self.assertEqual(done.returncode, 0, done.stdout[-2000:] + done.stderr[-2000:])
            evidence = json.loads(output.read_text())
        self.assertTrue(evidence["ok"], evidence["failure"])
        self.assertEqual(evidence["summary"]["built_flags"], [True, False, False, False])
        self.assertEqual(evidence["report"]["verdict"]["status"], "passed")


# ------------------------------------------------------------------------------ fork sources


FORKS = Path.home() / "heterogeneous-serving-tpu"
DP_SOURCE = FORKS / "tpu-inference/tpu_inference/core/sched/dp_scheduler.py"
CORE_SOURCE = FORKS / "vllm/vllm/v1/engine/core.py"


def _class_methods(path, name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name)
    methods = {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    attrs = {t.attr for n in ast.walk(cls) if isinstance(n, ast.Assign) for t in n.targets
             if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self"}
    attrs |= {n.target.attr for n in ast.walk(cls) if isinstance(n, ast.AnnAssign)
              and isinstance(n.target, ast.Attribute)}
    return methods, attrs


@unittest.skipUnless(DP_SOURCE.is_file() and CORE_SOURCE.is_file(), "fork sources not found")
class ForkSurface(unittest.TestCase):
    def test_dp_scheduler_exposes_what_the_frontend_reads(self):
        methods, attrs = _class_methods(DP_SOURCE, "DPScheduler")
        for name in ("get_request_counts", "make_stats", "has_finished_requests", "get_num_unfinished_requests",
                     "reset_prefix_cache", "shutdown", "finish_requests"):
            self.assertIn(name, methods)
        for name in ("per_rank_kv_cache_configs", "_pending_new_requests", "log_stats", "dp_size",
                     "assigned_dp_rank", "processes"):
            self.assertIn(name, attrs)
        self.assertNotIn("kv_cache_manager", attrs)  # why scheduler_capacity has a DP branch
        self.assertNotIn("running", attrs)
        self.assertIn("def update_vllm_config_for_dp_scheduler", DP_SOURCE.read_text())

    def test_engine_core_fields_the_swap_replaces_exist(self):
        methods, attrs = _class_methods(CORE_SOURCE, "EngineCore")
        for name in ("scheduler", "vllm_config", "structured_output_manager", "batch_queue_size", "batch_queue",
                     "request_block_hasher", "step_fn", "async_scheduling", "available_gpu_memory_for_kv_cache",
                     "log_stats", "model_executor"):
            self.assertIn(name, attrs)
        self.assertIn("_initialize_kv_caches", methods)
        source = CORE_SOURCE.read_text()
        for needle in ("resolve_kv_cache_block_sizes", "get_request_block_hasher", "include_finished_set",
                       "StructuredOutputManager(vllm_config)", "get_scheduler_cls()"):
            self.assertIn(needle, source)


if __name__ == "__main__":
    unittest.main()
