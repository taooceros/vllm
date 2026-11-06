"""Frontends (hetero/vproc/frontend.py): the gateway-facing actor of one virtual-process engine.

- SyntheticFrontend over a real 2-host CPU block of Ray host kernels: protocol, one fan-out at a time,
  poisoning, health during a long call, drain/resume.
- VllmFrontendCore with fake engine pieces and a stubbed ``vllm`` (no vLLM, no TPU): build segments, compile
  counts, poisoning from the shared ViewRpc and from a failed step loop.
- ViewRpc (vllm_executor): serialization and poisoning without Ray.
"""
import asyncio
import contextlib
import os
import sys
import threading
import time
import types
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from hetero.vproc import frontend, vllm_executor
from hetero.vproc.synthetic_engine import expected_tokens
from hetero.vproc.types import EngineSpec

NAMESPACE = "omp_hetero"
SEED = 11
REQUEST = {"prompt": [[1, 2, 3], [4, 5]], "steps": 5}


# ------------------------------------------------------------------------------ ViewRpc


class RecordingTransport:
    """Records submit/gather events in order; ``gather`` can sleep, fail, or block on an event."""

    def __init__(self):
        self.events, self.lock = [], threading.Lock()
        self.active = self.max_active = 0
        self.delay = 0.0
        self.fail_gather = None
        self.fail_submit_on = None
        self.gate = None

    def submit(self, host, envelope):
        if self.fail_submit_on == host:
            raise ConnectionError(f"host {host} is gone")
        with self.lock:
            self.events.append(("submit", host, envelope["rpc"]))
        return (host, envelope["rpc"])

    def gather(self, refs, timeout):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.events.append(("gather", None, refs[0][1], timeout))
        try:
            if self.gate is not None:
                self.gate.wait(5)
            time.sleep(self.delay)
            if self.fail_gather:
                raise self.fail_gather
            return [{"result": (host, rpc)} for host, rpc in refs]
        finally:
            with self.lock:
                self.active -= 1

    def health(self, hosts, timeout):
        return [{"ok": True} for _ in hosts]


class ViewRpcTests(unittest.TestCase):
    def test_concurrent_calls_run_one_fanout_at_a_time_in_one_order(self):
        transport = RecordingTransport()
        transport.delay = 0.01
        rpc = vllm_executor.ViewRpc([0, 1, 2], transport)
        threads = [threading.Thread(target=rpc.call, args=(f"m{i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        self.assertEqual(transport.max_active, 1)
        self.assertEqual((rpc.fanouts, rpc.max_active), (8, 1))
        order = {h: [e[2] for e in transport.events if e[0] == "submit" and e[1] == h] for h in (0, 1, 2)}
        self.assertEqual(order[0], order[1])  # same order on every host
        self.assertEqual(order[1], order[2])
        # no submit of fan-out k+1 before the gather of fan-out k
        kinds = [e[0] for e in transport.events]
        self.assertEqual(kinds, (["submit"] * 3 + ["gather"]) * 8)

    def test_call_async_keeps_its_slot_until_gathered(self):
        transport = RecordingTransport()
        transport.gate = threading.Event()
        rpc = vllm_executor.ViewRpc([0, 1], transport)
        first = rpc.call_async("first")
        started = threading.Event()
        results = []

        def second():
            started.set()
            results.append(rpc.call("second"))

        t = threading.Thread(target=second)
        t.start()
        started.wait(2)
        time.sleep(0.1)
        self.assertNotIn(("submit", 0, "second"), transport.events)  # held back by the in-flight fan-out
        transport.gate.set()
        self.assertEqual(first.result(5), [(0, "first"), (1, "first")])
        t.join(5)
        self.assertEqual(results, [[(0, "second"), (1, "second")]])

    def test_gather_failure_poisons_once_and_blocks_further_fanouts(self):
        transport, reasons = RecordingTransport(), []
        rpc = vllm_executor.ViewRpc([0, 1], transport, timeout_s=7, on_poison=reasons.append)
        self.assertEqual(rpc.call("ok"), [(0, "ok"), (1, "ok")])
        transport.fail_gather = TimeoutError("host 1 hung")
        with self.assertRaises(TimeoutError):
            rpc.call("stuck")
        self.assertEqual(len(reasons), 1)
        self.assertIn("stuck", reasons[0])
        self.assertIn("TimeoutError", reasons[0])
        before = len(transport.events)
        for attempt in (lambda: rpc.call("later"), lambda: rpc.call_async("later")):
            with self.assertRaises(vllm_executor.ViewPoisoned):
                attempt()
        self.assertEqual(len(transport.events), before)  # nothing reached any host
        self.assertEqual(len(reasons), 1)
        self.assertEqual(transport.events[-1][3], 7)  # the configured timeout reached gather

    def test_submit_failure_on_one_host_poisons(self):
        transport, reasons = RecordingTransport(), []
        rpc = vllm_executor.ViewRpc([0, 1, 2], transport, on_poison=reasons.append)
        transport.fail_submit_on = 1
        with self.assertRaises(ConnectionError):
            rpc.call("m")
        self.assertIn("submit", reasons[0])
        self.assertEqual(rpc.fanouts, 1)
        with self.assertRaises(vllm_executor.ViewPoisoned):
            rpc.call("n")
        # the slot was released: a poisoned view fails fast instead of deadlocking
        self.assertTrue(rpc._slot.acquire(timeout=1))
        rpc._slot.release()

    def test_a_broken_observer_does_not_hide_the_failure(self):
        transport = RecordingTransport()
        transport.fail_gather = RuntimeError("boom")

        def observer(reason):
            raise ValueError("observer broke")

        rpc = vllm_executor.ViewRpc([0], transport, on_poison=observer)
        with self.assertRaises(RuntimeError):
            rpc.call("m")
        self.assertIn("boom", rpc.poisoned)


# ------------------------------------------------------------------ SyntheticFrontend (Ray)


def _spec(hosts=(0, 1), engine_id="fe-0", generation=0):
    return EngineSpec(engine_id, "synthetic", hosts, 4 * len(hosts), {"seed": SEED}, generation)


class SyntheticFrontendRay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import ray
        if ray.is_initialized():  # an earlier test module (test_gateway_lifecycle) leaves its session open
            ray.shutdown()
        ray.init(address="local", num_cpus=6, include_dashboard=False, logging_level="ERROR")
        from hetero.vproc.host_kernel import spawn_block
        cls.kernels = spawn_block(2, 4, name_prefix="fe-kernel", actor_options={"namespace": NAMESPACE},
                                  preinit=None)
        cls.refs = [{"name": f"fe-kernel-{i}", "namespace": NAMESPACE} for i in range(2)]
        cls.spec = _spec()
        ray.get([k.start_engine.remote(cls.spec, None) for k in cls.kernels], timeout=300)
        cls.counter = 0

    @classmethod
    def tearDownClass(cls):
        import ray
        ray.shutdown()

    def make(self, **settings):
        type(self).counter += 1
        spec = _spec(generation=type(self).counter)
        settings = {"layout_id": "vproc-test", "layout_digest": "d" * 8, **settings}
        # the engine on the kernels is generation 0; verification must notice the mismatch for g>0
        return spec, frontend.SyntheticFrontend.options(
            name=f"vproc-{spec.engine_id}-g{spec.generation}", namespace=NAMESPACE).remote(
            spec, list(self.refs), settings)

    def ready(self, **settings):
        import ray
        spec = _spec()
        type(self).counter += 1
        actor = frontend.SyntheticFrontend.options(
            name=f"vproc-fe-0-t{type(self).counter}", namespace=NAMESPACE).remote(
            spec, list(self.refs), {"layout_id": "vproc-test", "layout_digest": "d" * 8, **settings})
        init = ray.get(actor.initialize.remote(), timeout=60)
        return actor, init

    def test_initialize_verifies_the_kernels_and_reports_segments(self):
        import ray
        actor, init = self.ready()
        self.assertEqual(init["compiles"], 0)
        self.assertIn("verify", init["segments"])
        self.assertEqual(len(init["pids"]), 2)
        self.assertEqual(ray.get(actor.initialize.remote()), init)  # idempotent
        self.assertTrue(ray.get(actor.health.remote())["ok"])

    def test_initialize_refuses_kernels_running_another_generation(self):
        import ray
        spec, actor = self.make()  # generation >= 1; the kernels run generation 0
        with self.assertRaises(Exception) as caught:
            ray.get(actor.initialize.remote(), timeout=60)
        self.assertIn("generation", str(caught.exception))
        health = ray.get(actor.health.remote())
        self.assertFalse(health["ok"])
        self.assertIn("initialize failed", health["poisoned"])

    def test_kernels_are_selected_by_host_rank(self):
        spec = _spec(hosts=(1, 3))
        core = frontend.SyntheticFrontendCore(spec, ["k0", "k1", "k2", "k3"], {})
        self.assertEqual(core.kernels, ["k1", "k3"])
        with self.assertRaises(ValueError):
            frontend.SyntheticFrontendCore(spec, ["k0", "k1"], {})

    def test_request_form_matches_the_oracle(self):
        import ray
        actor, _ = self.ready()
        out = ray.get(actor.generate.remote(dict(REQUEST)), timeout=60)
        self.assertEqual(out["tokens"], expected_tokens(self.spec, REQUEST))
        self.assertTrue(out["ranks_agree"])
        self.assertEqual(out["output_token_ids"], out["tokens"][0])
        self.assertEqual(out["finish_reason"], "length")

    def test_gateway_form_status_identity_and_adoption(self):
        import ray
        actor, _ = self.ready(endpoint_port=8123, node_ip="10.1.2.3")
        out = ray.get(actor.generate.remote(prompt_token_ids=[1, 2, 3], max_tokens=5, request_id="r1"), timeout=60)
        self.assertEqual(out["request_id"], "r1")
        self.assertEqual(out["output_token_ids"], expected_tokens(self.spec, {"prompt": [[1, 2, 3]], "steps": 5})[0])
        status = ray.get(actor.status.remote())
        for key in ("generation", "state", "total_blocks", "free_blocks", "block_size", "num_running",
                    "num_waiting"):
            self.assertIn(key, status)
        self.assertEqual(status["state"], "ready")
        identity = ray.get(actor.identity.remote())
        self.assertEqual((identity["layout_id"], identity["layout_digest"], identity["generation"]),
                         ("vproc-test", "d" * 8, 0))
        self.assertEqual((identity["node_ip"], identity["endpoint_port"]), ("10.1.2.3", 8123))
        ray.get(actor.adopt_identity.remote("vproc-next", "e" * 8))
        identity = ray.get(actor.identity.remote())
        self.assertEqual((identity["layout_id"], identity["layout_digest"]), ("vproc-next", "e" * 8))

    def test_concurrent_requests_are_one_fanout_at_a_time(self):
        import ray
        actor, _ = self.ready()
        refs = [actor.generate.remote(prompt_token_ids=[i + 1, 2], max_tokens=4) for i in range(8)]
        results = ray.get(refs, timeout=120)
        self.assertEqual(len(results), 8)
        stats = ray.get(actor.fanout_stats.remote())
        self.assertEqual((stats["fanouts"], stats["max_active"], stats["overlaps"]), (8, 1, 0))
        starts = [row["start"] for row in stats["log"]]
        self.assertEqual(starts, sorted(starts))

    def test_health_and_status_answer_during_a_long_fanout(self):
        import ray
        actor, _ = self.ready()
        long = actor.call.remote("idle", 2.0)
        time.sleep(0.3)
        start = time.time()
        health = ray.get(actor.health.remote(), timeout=10)
        status = ray.get(actor.status.remote(), timeout=10)
        self.assertLess(time.time() - start, 1.0)
        self.assertTrue(health["ok"])
        self.assertEqual(status["state"], "ready")
        rows = ray.get(long, timeout=30)
        self.assertEqual(len(rows), 2)

    def test_failed_fanout_poisons_and_stops_all_dispatch(self):
        import ray
        actor, _ = self.ready()
        with self.assertRaises(Exception):
            ray.get(actor.call.remote("no_such_engine_method"), timeout=60)
        health = ray.get(actor.health.remote())
        self.assertFalse(health["ok"])
        self.assertIn("no_such_engine_method", health["poisoned"])
        self.assertEqual(ray.get(actor.status.remote())["state"], "poisoned")
        before = ray.get(actor.fanout_stats.remote())["fanouts"]
        with self.assertRaises(Exception) as caught:
            ray.get(actor.generate.remote(dict(REQUEST)), timeout=30)
        self.assertIn("poisoned", str(caught.exception))
        with self.assertRaises(Exception):
            ray.get(actor.call.remote("step", dict(REQUEST)), timeout=30)
        # a poisoned frontend dispatches nothing: the refused calls never reached the fan-out thread
        self.assertEqual(ray.get(actor.fanout_stats.remote())["fanouts"], before)
        ray.get(actor.resume_admission.remote())
        self.assertEqual(ray.get(actor.status.remote())["state"], "poisoned")  # resume does not heal
        self.assertTrue(ray.get(actor.shutdown.remote())["poisoned"])

    def test_drain_then_resume_admission_serves_again(self):
        import ray
        actor, _ = self.ready()
        drained = ray.get(actor.drain.remote(10.0), timeout=30)
        self.assertTrue(drained["drained"])
        with self.assertRaises(Exception) as caught:
            ray.get(actor.generate.remote(dict(REQUEST)), timeout=30)
        self.assertIn("draining", str(caught.exception))
        ray.get(actor.resume_admission.remote())
        self.assertEqual(ray.get(actor.status.remote())["state"], "ready")
        out = ray.get(actor.generate.remote(dict(REQUEST)), timeout=60)
        self.assertEqual(out["tokens"], expected_tokens(self.spec, REQUEST))

    def test_unsupported_methods_are_refused_without_touching_the_kernels(self):
        import ray
        actor, _ = self.ready()
        for name in ("omp_kv_refs", "migration_candidates", "quiesce"):
            with self.assertRaises(Exception) as caught:
                ray.get(getattr(actor, name).remote(), timeout=30)
            self.assertIn("not supported", str(caught.exception))
        stats = ray.get(actor.fanout_stats.remote())
        self.assertEqual(stats["fanouts"], 0)
        self.assertTrue(ray.get(actor.health.remote())["ok"])


# --------------------------------------------------------------- VllmFrontendCore (fakes)


class FakeHosts:
    """Stands in for HostKernel handles behind ViewRpc: compile_count answers a running counter."""

    def __init__(self):
        self.compiles, self.log = [0, 0], []


class FakeTransport:
    def __init__(self, hosts):
        self.hosts = hosts

    def submit(self, host, envelope):
        method = envelope["rpc"]
        self.hosts.log.append((host, getattr(method, "__name__", method)))
        if callable(method):
            return {"result": self.hosts.compiles[host]}
        return {"result": (method, host)}

    def gather(self, refs, timeout):
        return list(refs)

    def health(self, hosts, timeout):
        return [{"ok": True} for _ in hosts]


class FakeExecutor:
    def __init__(self, hosts, bump):
        self.timings, self._hosts, self._bump = {"kv_alloc": 0.25, "warmup": 0.5}, hosts, bump
        self.started = []

    def warm(self):
        """The warm-up compiles; host 1 compiled one program fewer than host 0."""
        self._hosts.compiles[0] += self._bump
        self._hosts.compiles[1] += self._bump - 1

    def start_warmup(self):
        self.started.append("start")
        self.warm()

    def wait_for_warmup(self):
        self.started.append("wait")


class FakeDriver:
    def __init__(self):
        self.started, self.error, self.snapshot = False, None, {"total_blocks": 100, "free_blocks": 99,
                                                                 "num_running": 0, "num_waiting": 0}

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def stats(self):
        return {"error": self.error}

    def capacity_snapshot(self):
        return self.snapshot

    async def generate(self, rid, inputs, params):
        tokens = list(range(1, params.max_tokens + 1))
        out = types.SimpleNamespace(token_ids=tokens, text="t", finish_reason="length")
        return types.SimpleNamespace(prompt_token_ids=list(inputs["prompt_token_ids"]), outputs=[out]), 0.01


class FakeVllmFrontend(frontend.VllmFrontendCore):
    BUMP = 3

    def __init__(self, spec, kernels, settings=None):
        super().__init__(spec, kernels, settings)
        self.hosts = FakeHosts()
        self.executor = FakeExecutor(self.hosts, self.BUMP)
        self.fail_build = False

    def _handles(self):
        return [0, 1]

    def _verify_kernels(self, handles, timeout_s=60.0):
        return {"runtime_id": "rt", "pids": [1, 2]}

    def _make_rpc(self, hosts):
        return vllm_executor.ViewRpc(hosts, FakeTransport(self.hosts), on_poison=self._poison)

    def _make_config(self):
        cache = types.SimpleNamespace(block_size=16)
        return types.SimpleNamespace(cache_config=cache)

    def _make_llm_engine(self, config):
        if self.fail_build:
            raise MemoryError("KV allocation failed")
        if not self.frontend_settings.get("defer_warmup"):
            self.executor.warm()  # vLLM warms up inside the LLMEngine constructor (Executor.initialize_from_config)
        core = types.SimpleNamespace(scheduler=types.SimpleNamespace(), model_executor=self.executor)
        return types.SimpleNamespace(engine_core=types.SimpleNamespace(engine_core=core))

    def _executor_of(self, llm_engine):
        return self.executor

    def _make_driver(self, llm_engine, loop):
        self.driver = FakeDriver()
        return self.driver

    def _config_evidence(self, vllm_config):
        return {"block_size": 16}

    def _model_build(self):
        return "fake"


@contextlib.contextmanager
def stub_vllm():
    saved = {n: sys.modules.get(n) for n in ("vllm", "vllm.inputs")}
    vllm, inputs = types.ModuleType("vllm"), types.ModuleType("vllm.inputs")
    vllm.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    inputs.TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids}
    sys.modules["vllm"], sys.modules["vllm.inputs"] = vllm, inputs
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def vllm_spec():
    return EngineSpec("v-0", "vllm", (0, 1), 8, {}, 3)


class VllmFrontendFakes(unittest.TestCase):
    def build(self, **settings):
        return FakeVllmFrontend(vllm_spec(), ["k0", "k1"], {"layout_id": "L", "layout_digest": "D", **settings})

    def test_initialize_reports_segments_and_per_frontend_compiles(self):
        fe = self.build()
        init = asyncio.run(fe.initialize())
        self.assertEqual(fe.state, "ready")
        self.assertTrue(fe.driver.started)
        for name in ("config", "engine_init", "kv_alloc", "warmup", "build", "start_loop", "total"):
            self.assertIn(name, init["segments"])
        self.assertEqual((init["segments"]["kv_alloc"], init["segments"]["warmup"]), (0.25, 0.5))
        self.assertGreaterEqual(init["segments"]["engine_init"], 0.0)
        self.assertEqual(init["compiles"], 3)  # max over hosts of (after - before): hosts compiled 3 and 2
        self.assertEqual(fe.executor.started, [])  # defer_warmup off: the executor warms up by itself
        self.assertEqual(init["runtime_id"], "rt")
        self.assertEqual(asyncio.run(fe.initialize()), init)

    def test_defer_warmup_is_driven_by_the_frontend_inside_initialize(self):
        fe = self.build(defer_warmup=True)
        init = asyncio.run(fe.initialize())
        self.assertEqual(fe.executor.started, ["start", "wait"])
        self.assertEqual(init["compiles"], 3)  # counted whether the executor or the frontend warms up

    def test_identity_carries_the_layout_and_generation_of_the_spec(self):
        fe = self.build(endpoint_port=9000, node_ip="1.2.3.4")
        asyncio.run(fe.initialize())
        identity = fe.identity()
        self.assertEqual((identity["layout_id"], identity["layout_digest"], identity["generation"]),
                         ("L", "D", 3))
        self.assertEqual((identity["node_ip"], identity["endpoint_port"], identity["hosts"]),
                         ("1.2.3.4", 9000, [0, 1]))

    def test_identity_feeds_the_gateway_descriptor(self):
        """endpoints.engine_descriptor needs node_ip and a nonzero port from the frontend, its settings
        (``endpoint_port``) or the endpoints' default; without any of them publishing must fail loudly."""
        from hetero.vproc.endpoints import EndpointError, engine_descriptor
        spec = vllm_spec()
        configured = self.build(endpoint_port=9000, node_ip="1.2.3.4")
        row = engine_descriptor(spec, "L", "D", {"actor_name": "vproc-v-0-g3", "namespace": NAMESPACE,
                                                 "identity": configured.identity()})
        self.assertEqual((row["endpoint_host"], row["endpoint_port"], row["generation"], row["tp_size"]),
                         ("1.2.3.4", 9000, 3, 8))
        self.assertEqual((row["engine_actor_name"], row["ray_namespace"]), ("vproc-v-0-g3", NAMESPACE))
        bare = {"actor_name": "vproc-v-0-g3", "namespace": NAMESPACE, "identity": self.build(node_ip="1.2.3.4").identity()}
        with self.assertRaises(EndpointError):  # no port anywhere: frontend_settings defaults to empty
            engine_descriptor(spec, "L", "D", bare)
        self.assertEqual(engine_descriptor(spec, "L", "D", bare, endpoint_port=8000)["endpoint_port"], 8000)

    def test_failed_initialize_poisons(self):
        fe = self.build()
        fe.fail_build = True
        with self.assertRaises(MemoryError):
            asyncio.run(fe.initialize())
        health = fe.health()
        self.assertFalse(health["ok"])
        self.assertIn("KV allocation failed", health["poisoned"])
        self.assertEqual(fe.status()["state"], "poisoned")
        with self.assertRaises(frontend.FrontendPoisoned):
            asyncio.run(fe.initialize())

    def test_generate_both_forms_and_batch_prompts(self):
        fe = self.build()
        with stub_vllm():
            async def run():
                await fe.initialize()
                single = await fe.generate(prompt_token_ids=[5, 6], max_tokens=3, request_id="a")
                many = await fe.generate({"prompt": [[1], [2, 3]], "steps": 2, "request_id": "b"})
                return single, many
            single, many = asyncio.run(run())
        self.assertEqual((single["request_id"], single["output_token_ids"]), ("a", [1, 2, 3]))
        self.assertEqual(many["tokens"], [[1, 2], [1, 2]])
        self.assertEqual([r["request_id"] for r in many["results"]], ["b-0", "b-1"])
        self.assertEqual(fe.status()["served"], 3)

    def test_shared_viewrpc_failure_poisons_the_frontend(self):
        fe = self.build()
        asyncio.run(fe.initialize())
        fe._rpc.transport.gather = lambda refs, timeout: (_ for _ in ()).throw(TimeoutError("host 0 hung"))
        with self.assertRaises(TimeoutError):
            fe._rpc.call("execute_model")
        health = fe.health()
        self.assertFalse(health["ok"])
        self.assertIn("execute_model", health["poisoned"])
        with stub_vllm(), self.assertRaises(frontend.FrontendPoisoned):
            asyncio.run(fe.generate(prompt_token_ids=[1], max_tokens=1))

    def test_step_loop_error_is_noticed_by_health(self):
        fe = self.build()
        asyncio.run(fe.initialize())
        self.assertTrue(fe.health()["ok"])
        fe.driver.error = "RuntimeError('step failed')"
        health = fe.health()
        self.assertFalse(health["ok"])
        self.assertIn("step loop failed", health["poisoned"])

    def test_stock_scheduler_features_are_unavailable_not_broken(self):
        fe = self.build()
        asyncio.run(fe.initialize())
        self.assertIsNone(asyncio.run(fe.start_iteration_capture()))
        self.assertEqual(asyncio.run(fe.stop_iteration_capture())["audit"]["supported"], False)
        with self.assertRaises(NotImplementedError):
            fe.cancel_request(request_id="x")
        with self.assertRaises(NotImplementedError):
            asyncio.run(fe.omp_kv_refs())

    def test_shutdown_detaches_and_reports_poison(self):
        fe = self.build()
        asyncio.run(fe.initialize())
        key = fe._rpc_key
        fe._poison("test")
        snapshot = asyncio.run(fe.shutdown())
        self.assertEqual(snapshot["poisoned"], "test")
        self.assertEqual(fe.state, "stopped")
        self.assertNotIn(key, vllm_executor._SHARED)


if __name__ == "__main__":
    unittest.main()
