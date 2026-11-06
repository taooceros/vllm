"""Serving frontends: the gateway-facing Ray actor of one virtual-process engine (plan C3/C5/C6).

Host kernels (``HostKernel``) only run SPMD calls. They do not speak the gateway's engine protocol,
construct an EngineCore, or notice that a peer died. A *frontend* is one actor per engine
generation that does all three:

- ``VllmFrontend``: owns the driver-side vLLM ``LLMEngine`` (EngineCore + scheduler, in this process,
  ``multiprocess_mode=False``) over ``VllmViewExecutor``, which addresses only the view's host
  kernels. ``initialize()`` builds it: KV allocation, warm-up, then the step loop; it returns when the
  engine can serve.
- ``SyntheticFrontend``: the same protocol over ``SyntheticEngine`` (``engine_call("step", ...)``), for
  every local test of the controller and the gateway path.

Both subclass the real ``hetero.engine_actor.HeteroEngine`` class (not a copy), so ``status``, ``drain``,
``await_idle``, ``idle_snapshot``, ``resume_admission``, ``adopt_identity``, ``start_serving`` /
``stop_serving`` and ``generate`` of the vLLM frontend are the code the gateway already runs against.
What changes is engine construction (no placement group, ``VllmViewExecutor`` over the view's kernels),
identity/generation, poisoning, and the methods that need worker RPCs this adapter does not have.

Rules (acceptance of this module):
- One in-flight fan-out per engine, in the same order on every host. ``ViewRpc`` (vLLM) and
  ``SyntheticFrontendCore._fanout`` (synthetic) both serialize; ``fanout_stats()`` exposes the count and
  the maximum concurrency actually observed.
- A failed or timed-out fan-out *poisons* the frontend: ``health()["ok"]`` is False with a reason,
  ``status()["state"]`` is ``"poisoned"`` (so the gateway stops admitting), and every later dispatch
  raises ``FrontendPoisoned`` immediately. The controller tears the whole block down on a poisoned
  frontend; the frontend itself never retries and never touches the kernels again.
- Naming/spawning belong to the controller: ``controller.frontend_name(spec)`` (``vproc-<id>-g<generation>``)
  in namespace ``omp_hetero`` and ``controller.default_frontend_factory(spec, kernel_refs, settings, name)``,
  which constructs ``SyntheticFrontend``/``VllmFrontend`` as ``cls(spec, kernel_refs, settings)``.
  ``kernel_refs`` are ``{"name", "namespace"}`` of ALL kernels of the block, indexed by host rank; a
  frontend selects ``kernel_refs[h] for h in spec.hosts``. ``settings`` carries ``layout_id`` and
  ``layout_digest`` (identity must agree with the published layout, ``controller._publish``).
- ``health()`` is what ``controller.ensure_healthy`` polls: ``ok`` False or a truthy ``poisoned`` rebuilds
  the block.

- ``switch_layout(target, built)`` (``BlockController.switch_layout``): the engine's hosts moved to another
  (tp_size, dp_size) of the same chips; the frontend follows without being rebuilt. Idle only. First visit of
  a layout (``built=True``): ``engine_config(target)``, the KV-init sequence of ``EngineCore._initialize_kv_caches``
  through the existing executor (specs, available memory, ``initialize_from_config``, warm-up), the target
  scheduler as ``EngineCore.__init__`` builds it (tpu-inference's ``DPScheduler`` for dp_size > 1), then a swap
  of ``EngineCore.scheduler``/``vllm_config``/``structured_output_manager``/batch queue/``step_fn``/block hasher/
  ``available_gpu_memory_for_kv_cache`` and of the executor's config attributes, under the engine lock. A
  revisit (``built=False``) only swaps the parked layout back and resets its prefix cache (the hosts
  reallocated its KV empty). Previous layouts' schedulers stay parked and alive (DPScheduler children are
  processes); ``shutdown`` stops all of them. ``scheduler_capacity``/``_VllmDriver``/``await_idle``/
  ``kv_capacity`` work for both the stock Scheduler and DPScheduler (which has no block pool to read). A failure
  after the hosts moved poisons the frontend; a refusal before (not idle, unknown parked layout) does not.

Gateway protocol surface and its sources: see ``GATEWAY_METHODS``.

Without vLLM this module imports (ray is required; vLLM, jax and tpu-inference are lazy).
"""

from __future__ import annotations

import asyncio
from collections import deque
import contextlib
import dataclasses
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

import ray

from ..engine_actor import HeteroEngine
from ..inproc_engine import InprocAsyncEngine
from .endpoints import FRONTEND_NAMESPACE, layout_digest
from .engine_vllm import compile_count
from .types import EngineSpec
from .vllm_executor import (RayTransport, ViewRpc, executor_additional_config, register_view_rpc,
                            unregister_view_rpc)

# method -> where the gateway or the deployment code calls it (hetero/gateway/*.py, hetero/deployment.py)
GATEWAY_METHODS = {
    "identity": "gateway/service.py:301,410; deployment.py:759 (instance_id, layout_id, generation, node_ip, "
                "layout_digest, state)",
    "status": "gateway/core.py:143; service.py:307 (generation, state, total_blocks, free_blocks, block_size, "
              "num_running, num_waiting)",
    "generate": "gateway/core.py:424 (kwargs prompt, prompt_token_ids, max_tokens, temperature, request_id, "
                "ignore_eos)",
    "drain": "gateway/service.py:413 ({drained, inflight})",
    "await_idle": "gateway/core.py:907; service.py:417 ({idle, ...})",
    "idle_snapshot": "gateway/service.py:419 ({all_blocks_free, ...})",
    "resume_admission": "gateway/core.py:926",
    "start_iteration_capture": "gateway/core.py:933; service.py:348",
    "stop_iteration_capture": "gateway/core.py:943",
    "start_serving": "deployment.py:291", "stop_serving": "deployment.py:789",
    "shutdown": "deployment.py:792", "adopt_identity": "engine_actor.py:285 (kept engines, transition.py)",
}

# Inherited from HeteroEngine but needing worker RPCs (omp_*), KV transport or a migration-aware scheduler
# that the unmodified vLLM + tpu-inference engine does not provide. Refused up front: letting them reach the
# kernels would fail a fan-out and poison a healthy engine.
UNSUPPORTED = (
    "omp_kv_notify", "omp_kv_refs", "migration_candidates", "quiesce", "unquiesce", "release_source",
    "reserve_continuation", "prepare_continuation", "commit_continuation", "await_continuation",
    "abort_continuation", "cancel_continuation", "kv_layout", "export_request_kv", "migration_state",
    "generate_qualification", "await_qualification_checkpoint", "next_token_diagnostic",
    "worker_evidence", "worker_init_phases", "snapshot_weights", "weight_fingerprint", "worker_kv_evidence",
)

_RawEngine = HeteroEngine.__ray_metadata__.modified_class.__ray_actor_class__


class FrontendPoisoned(RuntimeError):
    """A fan-out of this engine failed; the engine's hosts may be out of step. Tear the block down."""


def frontend_actor_name(spec: EngineSpec) -> str:
    return f"vproc-{spec.engine_id}-g{spec.generation}"


def _refuse(name: str):
    def refused(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{name} is not supported by {type(self).__name__} "
                                  "(needs worker RPCs/migration support this engine does not have)")
    refused.__name__ = name
    return refused


class _FrontendCore(_RawEngine):
    """Shared state, poisoning and the protocol methods that do not depend on the engine kind."""

    def __init__(self, spec: EngineSpec, kernels: Sequence[Any], settings: Optional[Dict[str, Any]] = None):
        settings = dict(settings or {})
        if max(spec.hosts) >= len(kernels):
            raise ValueError(f"engine {spec.engine_id} uses hosts {spec.hosts}, but only {len(kernels)} kernels "
                             "were given (kernels are indexed by host rank)")
        kernels = [kernels[h] for h in spec.hosts]
        layout_id = settings.get("layout_id", "vproc")
        instance = {"instance_id": spec.engine_id, "tp_size": spec.tp_size,
                    "host_ips": list(settings.get("host_ips", ())), "hosts": [], "intended_runtime_env": {}}
        super().__init__(instance=instance, layout_id=layout_id,
                         layout_digest=settings.get("layout_digest") or layout_digest(layout_id, [spec]),
                         engine_settings=self._engine_settings(spec, settings))
        self.spec = spec
        self.kernels = list(kernels)
        self.frontend_settings = settings
        self.generation = int(spec.generation)  # the registry row's generation (HeteroEngine uses time())
        if settings.get("node_ip"):
            self.node_ip = settings["node_ip"]
        self.endpoint_port = settings.get("endpoint_port")
        self._poisoned: Optional[str] = None
        self._poison_lock = threading.Lock()
        self._init_result: Optional[Dict[str, Any]] = None
        self._switching = False  # a layout switch is running: no admission (``_admit``)
        self._layout_switches: List[Dict[str, Any]] = []
        self._visited_layouts = {spec.layout}

    # -- construction hooks ------------------------------------------------------------
    @staticmethod
    def _engine_settings(spec: EngineSpec, settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return settings.get("engine_settings")

    def _handles(self) -> List[Any]:
        return [ray.get_actor(k["name"], namespace=k.get("namespace")) if isinstance(k, dict) else k
                for k in self.kernels]

    def _verify_kernels(self, handles: Sequence[Any], timeout_s: float = 60.0,
                        expect_layout: Optional[tuple] = None) -> Dict[str, Any]:
        """Every kernel answers, sits at its rank, runs this engine generation (in ``expect_layout`` =
        (tp_size, dp_size) when given), shares one runtime."""
        rows = ray.get([h.describe.remote() for h in handles], timeout=timeout_s)
        problems = []
        for position, (host, row) in enumerate(zip(self.spec.hosts, rows)):
            if row.rank != host:
                problems.append(f"kernel {position} is rank {row.rank}, spec wants host {host}")
            engine = row.engine
            if engine is None or engine.engine_id != self.spec.engine_id \
                    or engine.generation != self.spec.generation:
                running = "no engine" if engine is None else f"engine {engine.engine_id} generation {engine.generation}"
                problems.append(f"rank {row.rank} runs {running}, not engine {self.spec.engine_id} "
                                f"generation {self.spec.generation}")
            elif expect_layout is not None and tuple(engine.layout) != tuple(expect_layout):
                problems.append(f"rank {row.rank} runs layout {tuple(engine.layout)}, not {tuple(expect_layout)}")
        if len({row.runtime_id for row in rows}) != 1:
            problems.append(f"kernels disagree on runtime_id: {sorted({row.runtime_id for row in rows})}")
        if problems:
            raise RuntimeError("; ".join(problems))
        return {"runtime_id": rows[0].runtime_id, "pids": [row.pid for row in rows]}

    # -- poisoning ---------------------------------------------------------------------
    def _poison(self, reason: str) -> None:
        with self._poison_lock:
            if self._poisoned is not None:
                return
            self._poisoned = reason
        self.state = "poisoned"

    def _check(self) -> None:
        if self._poisoned is not None:
            raise FrontendPoisoned(f"{self.instance_id} is poisoned: {self._poisoned}")

    def _admit(self) -> None:
        self._check()
        if self._switching:
            raise RuntimeError(f"{self.instance_id} is switching its in-engine layout and refuses new work")
        if self._draining:
            raise RuntimeError(f"{self.instance_id} is draining and refuses new work")
        if self.state != "ready":
            raise RuntimeError(f"{self.instance_id} is {self.state}, not ready")

    def resume_admission(self) -> Dict[str, Any]:
        """``HeteroEngine.resume_admission`` clears the drain latch but leaves ``state == "drained"``,
        which ``_admit`` refuses; a healthy engine is ``ready`` again, a poisoned one stays poisoned."""
        result = super().resume_admission()
        if self._poisoned is None and self.state in ("drained", "draining"):
            self.state = "ready"
        return result

    def health(self) -> Dict[str, Any]:
        """Answers while generate/initialize run (separate from the fan-out path)."""
        self._observe_failures()
        return {"ok": self._poisoned is None and self.state != "stopped", "poisoned": self._poisoned,
                "inflight": self._inflight, "state": self.state}

    def _observe_failures(self) -> None:
        pass

    # -- identity -----------------------------------------------------------------------
    def identity(self) -> Dict[str, Any]:
        body = super().identity()
        body["generation"] = int(self.spec.generation)
        body["endpoint_port"] = self.endpoint_port
        body["frontend"] = type(self).__name__
        body["hosts"] = list(self.spec.hosts)
        body["tp_size"], body["dp_size"] = int(self.spec.tp_size), int(self.spec.dp_size)
        return body

    # -- in-engine layout switch (non-transparent; ``BlockController.switch_layout``) ----------
    LAYOUT_SEGMENTS = ("verify", "kv_spec", "available_memory", "kv_init", "warmup", "scheduler_build",
                       "swap")

    def _validate_layout_target(self, target: EngineSpec) -> None:
        current = self.spec
        same = (target.engine_id, target.kind, target.hosts, target.model, target.generation) == \
               (current.engine_id, current.kind, current.hosts, current.model, current.generation)
        if not same:
            raise ValueError(f"{self.instance_id}: layout target must equal the engine spec except for "
                             f"tp_size/dp_size ({target} vs {current})")
        if target.chips != current.chips:
            raise ValueError(f"{self.instance_id}: layout {target.layout} spans {target.chips} chips, the "
                             f"engine spans {current.chips}")
        if target.layout == current.layout:
            raise ValueError(f"{self.instance_id} already runs layout {target.layout}")

    def _precheck_layout(self, target: EngineSpec, built: bool) -> None:
        """Refusals that leave the engine untouched (no poisoning)."""

    async def _apply_layout(self, target: EngineSpec, built: bool) -> Dict[str, Any]:
        raise NotImplementedError

    def _retarget(self, target: EngineSpec) -> None:
        """The engine serves ``target``'s layout from now on (identity, instance row, visited set)."""
        self.spec = target
        self.instance["tp_size"], self.instance["dp_size"] = target.tp_size, target.dp_size
        self._visited_layouts.add(target.layout)

    async def switch_layout(self, target: EngineSpec, built: bool) -> Dict[str, Any]:
        """Follow the hosts into ``target``'s (tp_size, dp_size). Requires an idle engine. ``built`` is
        what the hosts reported (``switch_engine_layout``): True on the first visit of the layout (the
        frontend then builds the layout's KV and scheduler), False when the layout's scheduler is parked
        here (the frontend swaps it in and resets its prefix cache). Returns ``{"built", "layout",
        "segments", "compiles", ...}``; ``segments`` names are ``LAYOUT_SEGMENTS`` plus ``total``.
        Refusals before anything changed raise without poisoning; a failure after the hosts moved poisons
        the frontend (the engine cannot serve until the controller rebuilds it)."""
        self._check()
        if self._init_result is None:
            raise RuntimeError(f"{self.instance_id} is not initialized")
        if self._switching:
            raise RuntimeError(f"{self.instance_id}: a layout switch is already running")
        built = bool(built)
        self._validate_layout_target(target)
        self._precheck_layout(target, built)
        self._switching = True
        started = time.time()
        try:
            idle = await self.await_idle(float(self.frontend_settings.get("switch_idle_timeout_s", 30.0)))
            if not idle["idle"]:
                raise RuntimeError(f"{self.instance_id} is not idle, cannot switch layout: {idle}")
            try:
                report = await self._apply_layout(target, built)
            except BaseException as error:  # noqa: BLE001
                self._poison(f"layout switch to {target.layout} failed: {type(error).__name__}: {error}")
                raise
        finally:
            self._switching = False
        report["segments"]["total"] = time.time() - started
        report.update(built=built, layout=list(target.layout), tp_size=target.tp_size, dp_size=target.dp_size,
                      instance_id=self.instance_id, generation=int(self.spec.generation))
        self._layout_switches.append({k: report[k] for k in ("built", "layout", "segments", "compiles")})
        return report

    # -- request-dict form (BlockController): generate(request) -> {"tokens": ...} --------
    @staticmethod
    def _prompts(request: Dict[str, Any]) -> List[List[int]]:
        prompts = request.get("prompt", request.get("prompt_token_ids"))
        if prompts is None:
            raise ValueError("request needs 'prompt' (list of token-id lists)")
        if prompts and isinstance(prompts[0], int):
            prompts = [prompts]
        return [[int(t) for t in row] for row in prompts]

    @staticmethod
    def _merge(results: List[Dict[str, Any]], **extra: Any) -> Dict[str, Any]:
        first = dict(results[0])
        first.update(tokens=[r["output_token_ids"] for r in results], results=results, **extra)
        return first

    def _heartbeat_state(self) -> str:
        return self.state

    async def shutdown(self) -> Dict[str, Any]:
        snapshot = await super().shutdown()
        if self.state == "stopped" and self._poisoned is not None:
            snapshot["poisoned"] = self._poisoned
        return snapshot


for _name in UNSUPPORTED:
    setattr(_FrontendCore, _name, _refuse(_name))


# ================================================================================= synthetic


class SyntheticFrontendCore(_FrontendCore):
    """``SyntheticEngine`` behind the gateway protocol. Generation is one serialized fan-out of
    ``engine_call("step", {"prompt": [[int]], "steps": n})`` to every kernel of the view.

    ``generate(request)`` (BlockController form): ``request = {"prompt": [[int]], "steps": int}`` ->
    HeteroEngine-shaped result of the first prompt plus ``tokens`` (all prompts), ``ranks_agree``.
    ``generate(prompt_token_ids=..., max_tokens=...)`` (gateway form): the same result for one prompt.
    ``settings``: ``kv_blocks`` (default 1024) and ``block_size`` (16) are the synthetic KV figures the
    gateway's capacity accounting reads; ``fanout_timeout_s`` (default 120); plus ``layout_id``,
    ``layout_digest``, ``node_ip``, ``endpoint_port``, ``host_ips``.
    """

    def __init__(self, spec: EngineSpec, kernels: Sequence[Any], settings: Optional[Dict[str, Any]] = None):
        super().__init__(spec, kernels, settings)
        s = self.frontend_settings
        self._timeout_s = float(s.get("fanout_timeout_s", 120.0))
        self._kv_blocks = int(s.get("kv_blocks", 1024))
        self._block_size = int(s.get("block_size", 16))
        self._hosts: List[Any] = []
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vproc-fanout")
        self._order = asyncio.Lock()
        self._stats = threading.Lock()
        self._active = 0
        self._max_active = 0
        self._fanout_log: List[Dict[str, Any]] = []
        self._fanouts = 0
        self._queued = 0

    # -- lifecycle ---------------------------------------------------------------------
    async def initialize(self) -> Dict[str, Any]:
        """Verify the kernels (rank, engine id and generation, one runtime). Nothing heavy runs."""
        if self._init_result is not None:
            return self._init_result
        self._check()
        started = time.time()
        loop = asyncio.get_running_loop()

        def verify() -> Dict[str, Any]:
            self._hosts = self._handles()
            return self._verify_kernels(self._hosts)

        try:
            evidence = await loop.run_in_executor(None, verify)
        except BaseException as error:  # noqa: BLE001
            self._poison(f"initialize failed: {type(error).__name__}: {error}")
            raise
        self.state = "ready"
        self._init_result = {"segments": {"verify": time.time() - started}, "compiles": 0, **evidence}
        return self._init_result

    # -- the one dispatch path -----------------------------------------------------------
    def _fanout_blocking(self, method: str, args: Sequence[Any]) -> List[Any]:
        """Runs on the single fan-out thread: submit to every kernel in rank order, gather, check."""
        started = time.time()
        with self._stats:
            self._active += 1
            self._fanouts += 1
            self._max_active = max(self._max_active, self._active)
            seq = self._fanouts
        try:
            self._check()
            refs = [host.engine_call.remote(method, *args) for host in self._hosts]
            rows = ray.get(refs, timeout=self._timeout_s)
            if method == "step":
                tokens = [row.get("tokens") for row in rows]
                if any(t != tokens[0] for t in tokens):
                    raise RuntimeError("ranks returned different tokens: the view is out of step")
            return rows
        except FrontendPoisoned:
            raise
        except BaseException as error:
            self._poison(f"fan-out {seq} engine_call({method!r}) failed: {type(error).__name__}: {error}")
            raise
        finally:
            with self._stats:
                self._active -= 1
                self._fanout_log.append({"seq": seq, "method": method, "start": started, "end": time.time()})
                del self._fanout_log[:-1000]

    async def _fanout(self, method: str, *args: Any) -> List[Any]:
        loop = asyncio.get_running_loop()
        self._queued += 1
        try:
            async with self._order:  # FIFO: the order of arrival is the order on every host
                self._check()
                return await loop.run_in_executor(self._pool, self._fanout_blocking, method, args)
        finally:
            self._queued -= 1

    async def call(self, method: str, *args: Any) -> List[Any]:
        """Serialized ``engine_call(method, *args)`` on every kernel (rank order); poisons on failure."""
        if self._init_result is None:
            raise RuntimeError(f"{self.instance_id} is not initialized")
        return await self._fanout(method, *args)

    def fanout_stats(self) -> Dict[str, Any]:
        with self._stats:
            log = list(self._fanout_log)
            overlaps = sum(1 for a, b in zip(log, log[1:]) if b["start"] < a["end"])
            return {"fanouts": self._fanouts, "active": self._active, "max_active": self._max_active,
                    "overlaps": overlaps, "log": log}

    # -- generation ------------------------------------------------------------------------
    @staticmethod
    def _shape(rid: str, instance_id: str, prompt: List[int], tokens: List[int], started: float,
               ended: float) -> Dict[str, Any]:
        return {"migrated_transaction": None, "request_id": rid, "instance_id": instance_id,
                "prompt_tokens": len(prompt), "output_tokens": len(tokens), "output_token_ids": list(tokens),
                "text": " ".join(str(t) for t in tokens), "finish_reason": "length",
                "ttft_s": round(ended - started, 6), "elapsed_s": round(ended - started, 4),
                "started_at_s": started, "finished_at_s": ended}

    async def generate(self, request: Optional[Dict[str, Any]] = None, *, prompt: Optional[str] = None,
                       prompt_token_ids: Optional[Sequence[int]] = None, max_tokens: int = 16,
                       temperature: float = 0.0, seed: Optional[int] = None, request_id: Optional[str] = None,
                       ignore_eos: bool = False) -> Dict[str, Any]:
        self._admit()
        if request is not None:
            prompts, steps = self._prompts(request), int(request.get("steps", request.get("max_tokens", 16)))
            payload = {"prompt": prompts, "steps": steps}
            rid = request.get("request_id") or f"{self.instance_id}-{uuid.uuid4().hex[:12]}"
        else:
            if prompt_token_ids is not None:
                ids = [int(t) for t in prompt_token_ids]
            elif prompt is not None:
                ids = list(prompt.encode())
            else:
                raise ValueError("either prompt or prompt_token_ids is required")
            prompts, steps = [ids], int(max_tokens)
            payload = {"prompt": prompts, "steps": steps}
            rid = request_id or f"{self.instance_id}-{uuid.uuid4().hex[:12]}"
        self._inflight += 1
        started = time.time()
        try:
            rows = await self._fanout("step", payload)
        finally:
            self._inflight -= 1
        ended = time.time()
        self._served += 1
        tokens = rows[0]["tokens"]
        results = [self._shape(rid, self.instance_id, p, t, started, ended) for p, t in zip(prompts, tokens)]
        out = self._merge(results, ranks_agree=True, forward_s=rows[0].get("forward_s"),
                          decode_s=rows[0].get("decode_s"))
        return out

    async def generate_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [await self.generate(**request) for request in requests]

    # -- capacity / idleness (synthetic KV: nothing is ever held) ------------------------------
    def kv_capacity(self) -> Dict[str, Any]:
        running = min(1, self._active)
        return {"block_size": self._block_size, "total_blocks": self._kv_blocks, "free_blocks": self._kv_blocks,
                "usable_blocks": self._kv_blocks - 1, "usable_tokens": (self._kv_blocks - 1) * self._block_size,
                "num_running": running, "num_waiting": max(0, self._inflight - running)}

    def status(self) -> Dict[str, Any]:
        body = super().status()
        body.update(self.kv_capacity())
        body["poisoned"] = self._poisoned
        body["switching"] = self._switching
        return body

    async def await_idle(self, timeout_s: float = 300.0) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            if self._poisoned is not None:
                raise RuntimeError(f"engine step loop failed: {self._poisoned}")
            idle = self._inflight == 0 and self._queued == 0 and self._active == 0
            capacity = self.kv_capacity()
            snapshot = {"instance_id": self.instance_id, "idle": idle, "inflight": self._inflight,
                        "num_running": capacity["num_running"], "num_waiting": capacity["num_waiting"],
                        "free_blocks": capacity["free_blocks"], "pending_batches": self._queued,
                        "scheduler_has_work": not idle, "draining": self._draining}
            if idle or time.monotonic() >= deadline:
                return snapshot
            await asyncio.sleep(0.02)

    async def drain(self, timeout_s: float = 300.0) -> Dict[str, Any]:
        result = await super().drain(timeout_s)
        if self._poisoned is not None:
            self.state = "poisoned"
        return result

    async def start_iteration_capture(self) -> None:
        return None

    async def stop_iteration_capture(self) -> Dict[str, Any]:
        return {"rows": [], "audit": {"supported": False, "reason": "synthetic engine has no scheduler"}}

    def cancel_request(self, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError("synthetic frontend serves one fan-out per request; nothing to cancel")

    def _precheck_layout(self, target: EngineSpec, built: bool) -> None:
        if not built and target.layout not in self._visited_layouts:
            raise ValueError(f"{self.instance_id}: layout {target.layout} was never built here (built=False)")

    async def _apply_layout(self, target: EngineSpec, built: bool) -> Dict[str, Any]:
        """Synthetic engines hold no scheduler or KV: check that every kernel runs ``target``, then adopt it."""
        loop = asyncio.get_running_loop()
        started = time.time()
        evidence = await loop.run_in_executor(
            None, lambda: self._verify_kernels(self._hosts, expect_layout=target.layout))
        segments = {name: 0.0 for name in self.LAYOUT_SEGMENTS}
        segments["verify"] = time.time() - started
        step = time.time()
        self._retarget(target)
        segments["swap"] = time.time() - step
        return {"segments": segments, "compiles": 0, **evidence}

    async def shutdown(self) -> Dict[str, Any]:
        self.state = "stopped"
        self._pool.shutdown(wait=False)
        return {"instance_id": self.instance_id, "served": self._served, "poisoned": self._poisoned}


# ====================================================================================== vLLM


_MISSING = object()


def scheduler_capacity(scheduler: Any) -> Dict[str, Any]:
    """KV capacity and request counts of a scheduler, for the stock vLLM ``Scheduler`` and for tpu-inference's
    ``DPScheduler`` (no ``kv_cache_manager``/``running``/``waiting``: its per-rank schedulers live in forked
    processes). ``pools`` = independent block pools (1, or the DP size; each keeps its own null block).
    DP: ``total_blocks`` = sum of the per-rank KV configs, ``free_blocks`` is derived from the ranks' averaged
    ``kv_cache_usage`` (``make_stats``; exact when all ranks are equally loaded, always exact when idle),
    ``num_waiting`` includes requests held back by DP prefill batching."""
    pool = getattr(getattr(scheduler, "kv_cache_manager", None), "block_pool", None)
    if pool is not None:
        return {"total_blocks": int(pool.num_gpu_blocks), "free_blocks": int(pool.get_num_free_blocks()),
                "num_running": len(scheduler.running), "num_waiting": len(scheduler.waiting), "pools": 1}
    configs = getattr(scheduler, "per_rank_kv_cache_configs", None)
    if not configs:
        raise TypeError(f"{type(scheduler).__name__} exposes neither a block pool nor per-rank KV configs")
    running, waiting = scheduler.get_request_counts()
    waiting += len(getattr(scheduler, "_pending_new_requests", ()) or ())
    pools = len(configs)
    total = sum(int(c.num_blocks) for c in configs)
    usage = 0.0
    stats = scheduler.make_stats() if getattr(scheduler, "log_stats", False) else None
    if stats is not None:
        usage = min(1.0, max(0.0, float(stats.kv_cache_usage)))
    return {"total_blocks": total, "free_blocks": int(round((1.0 - usage) * (total - pools))),
            "num_running": int(running), "num_waiting": int(waiting), "pools": pools}


class _VllmDriver(InprocAsyncEngine):
    """``InprocAsyncEngine`` whose published capacity also works for ``DPScheduler``."""

    def _publish_snapshot(self) -> None:
        snapshot = scheduler_capacity(self.engine_core.scheduler)
        snapshot.update(published_at=time.time(), step_count=self._step_count)
        self._capacity_snapshot = snapshot


@dataclasses.dataclass
class _Layout:
    """What one in-engine layout owns inside the EngineCore: its config and scheduler (DPScheduler children
    stay alive while parked) and the EngineCore fields derived from them."""

    spec: EngineSpec
    config: Any
    scheduler: Any
    structured_output_manager: Any = None
    kv_cache_config: Any = None
    request_block_hasher: Any = None
    batch_queue_size: int = 1
    available_memory: int = -1
    visits: int = 1


class VllmFrontendCore(_FrontendCore):
    """Unmodified vLLM + tpu-inference serving through the view's kernels.

    ``initialize()``: verify kernels, build the ``EngineCore`` with ``VllmViewExecutor`` (determine memory,
    KV allocation, warm-up), start the in-process step loop, return ``{"segments", "compiles"}`` where
    segments are ``config``, ``engine_init`` (rest of the constructor), ``kv_alloc``, ``warmup``,
    ``start_loop`` and ``total`` seconds, and ``compiles`` the backend compiles on the kernels during it
    (max over hosts). Every fan-out of this engine, the executor's and the frontend's own, goes through
    one ``ViewRpc``: serialized, poisoned on failure.

    ``settings``: ``layout_id``, ``layout_digest``, ``node_ip``, ``endpoint_port``, ``host_ips``;
    ``rpc_timeout_s`` (default 900, per fan-out); ``defer_warmup`` (default False; with True the
    frontend starts and awaits warm-up itself, still inside ``initialize``); ``scheduler_cls`` (default
    None = vLLM's own; ``"hetero.scheduler.HeteroScheduler"`` only together with async scheduling, which
    ``engine_config`` turns off; with a stock scheduler iteration capture and migration are unavailable);
    ``engine_settings`` (EngineSettings overrides for identity()).
    """

    def __init__(self, spec: EngineSpec, kernels: Sequence[Any], settings: Optional[Dict[str, Any]] = None):
        super().__init__(spec, kernels, settings)
        self._rpc: Optional[ViewRpc] = None
        self._rpc_key = f"{spec.engine_id}-g{spec.generation}-{uuid.uuid4().hex[:8]}"
        self._executor: Any = None
        self._layouts: Dict[tuple, _Layout] = {}  # layout key (tp, dp) -> its parked or active scheduler
        self._current: Optional[_Layout] = None

    @staticmethod
    def _engine_settings(spec: EngineSpec, settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        from .engine_vllm import DEFAULT_ENGINE_ARGS
        args = {**DEFAULT_ENGINE_ARGS, **spec.model}
        extra = args.get("additional_config") or {}
        mapped = {"model": args["model"], "dtype": args["dtype"], "kv_cache_dtype": args["kv_cache_dtype"],
                  "activation_dtype": extra.get("activation_dtype"),
                  "matmul_precision": extra.get("matmul_precision", "default"),
                  "max_model_len": args["max_model_len"], "max_num_seqs": args["max_num_seqs"],
                  "max_num_batched_tokens": args["max_num_batched_tokens"],
                  "gpu_memory_utilization": args["gpu_memory_utilization"],
                  "enable_prefix_caching": args["enable_prefix_caching"],
                  "enforce_eager": args["enforce_eager"], "seed": args["seed"]}
        mapped.update(settings.get("engine_settings") or {})
        return mapped

    # -- construction hooks (tests replace these) ---------------------------------------------
    def _make_rpc(self, hosts: List[Any]) -> ViewRpc:
        return ViewRpc(hosts, RayTransport(), float(self.frontend_settings.get("rpc_timeout_s", 900.0)),
                       on_poison=self._poison)

    def _make_config(self, spec: Optional[EngineSpec] = None) -> Any:
        from .engine_vllm import engine_config
        spec = spec or self.spec
        config = engine_config(spec)
        config.additional_config.update(executor_additional_config(
            [k for k in self.kernels if isinstance(k, dict)] or [],
            timeout_s=float(self.frontend_settings.get("rpc_timeout_s", 900.0)),
            defer_warmup=bool(self.frontend_settings.get("defer_warmup", False)), rpc_key=self._rpc_key))
        if self.frontend_settings.get("scheduler_cls"):
            if spec.dp_size > 1:
                raise ValueError("a scheduler_cls override cannot serve a data-parallel layout (DPScheduler)")
            config.scheduler_config.scheduler_cls = self.frontend_settings["scheduler_cls"]
        return config

    def _make_llm_engine(self, config: Any) -> Any:
        from vllm.v1.engine.llm_engine import LLMEngine
        from .vllm_executor import VllmViewExecutor
        from ..engine_actor import _supported_kwargs
        return LLMEngine(**_supported_kwargs(LLMEngine.__init__, vllm_config=config,
                                             executor_class=VllmViewExecutor, log_stats=True,
                                             multiprocess_mode=False))

    def _make_driver(self, llm_engine: Any, loop: asyncio.AbstractEventLoop) -> Any:
        return _VllmDriver(llm_engine, loop)

    @staticmethod
    def _executor_of(llm_engine: Any) -> Any:
        return llm_engine.engine_core.engine_core.model_executor

    # -- lifecycle ---------------------------------------------------------------------
    @staticmethod
    def _dp_env() -> None:
        """``TPU_MULTIPROCESS_DP`` must be off: tpu_platform would turn a DP layout into one process per DP
        rank (tpu_platform.py:246-264) instead of the single-process SPMD ``data`` mesh axis."""
        value = os.environ.get("TPU_MULTIPROCESS_DP", "")
        if value.lower() not in ("", "0", "false"):
            raise RuntimeError(f"TPU_MULTIPROCESS_DP={value!r}: in-engine data parallelism is single-process SPMD")
        os.environ["TPU_MULTIPROCESS_DP"] = "0"

    def _build(self) -> Dict[str, Any]:
        os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")  # engine_actor.py:344
        self._dp_env()
        segments: Dict[str, float] = {}
        started = time.time()
        hosts = self._handles()
        evidence = self._verify_kernels(hosts)
        self._rpc = self._make_rpc(hosts)
        register_view_rpc(self._rpc_key, self._rpc)
        before = self._rpc.call(compile_count)
        step = time.time()
        config = self._make_config()
        segments["config"] = time.time() - step
        step = time.time()
        llm_engine = self._make_llm_engine(config)
        constructor = time.time() - step
        self._executor = executor = self._executor_of(llm_engine)
        if self.frontend_settings.get("defer_warmup", False):
            step = time.time()
            executor.start_warmup()
            executor.wait_for_warmup()
            constructor += time.time() - step
        kv, warmup = executor.timings.get("kv_alloc", 0.0), executor.timings.get("warmup", 0.0)
        segments.update(engine_init=max(0.0, constructor - kv - warmup), kv_alloc=kv, warmup=warmup)
        after = self._rpc.call(compile_count)
        segments["build"] = time.time() - started
        return {"config": config, "llm_engine": llm_engine, "segments": segments, "evidence": evidence,
                "compiles": max(a - b for a, b in zip(after, before))}

    async def initialize(self) -> Dict[str, Any]:
        if self._init_result is not None:
            return self._init_result
        self._check()
        loop = asyncio.get_running_loop()
        started = time.time()
        try:
            built = await loop.run_in_executor(None, self._build)
            step = time.time()
            self.vllm_config, self.llm_engine = built["config"], built["llm_engine"]
            self.engine = self._make_driver(self.llm_engine, loop)
            self._bind_engine_internals()
            self._current = self._capture_layout()
            self._layouts = {self.spec.layout: self._current}
            self.engine.start()
            self.state = "ready"
            segments = dict(built["segments"], start_loop=time.time() - step, total=time.time() - started)
            self._init_evidence = {
                "status": "initialized", "instance_id": self.instance_id, "init_s": round(segments["total"], 2),
                "engine_settings": self.settings.to_dict(), "kv_capacity": self.kv_capacity(),
                "resolved_engine_config": self._config_evidence(self.vllm_config)}
        except BaseException as error:  # noqa: BLE001
            self._poison(f"initialize failed: {type(error).__name__}: {error}")
            raise
        self._init_result = {"segments": segments, "compiles": built["compiles"], **built["evidence"]}
        return self._init_result

    def _install_iteration_hooks(self) -> None:
        return None  # stock scheduler: capture/migration unavailable; see start_iteration_capture

    # -- capacity / idleness (stock Scheduler and DPScheduler) -------------------------------
    def _capacity(self) -> Dict[str, Any]:
        snapshot = self.engine.capacity_snapshot() if self.engine is not None else {}
        if snapshot:
            return snapshot
        with self._step_fence():
            return dict(scheduler_capacity(self.scheduler), published_at=time.time())

    def kv_capacity(self) -> Dict[str, Any]:
        snapshot = self._capacity()
        total, free = int(snapshot["total_blocks"]), int(snapshot["free_blocks"])
        pools = int(snapshot.get("pools", 1))
        block_size = int(self.vllm_config.cache_config.block_size)
        return {"block_size": block_size, "total_blocks": total, "free_blocks": free,
                "usable_blocks": total - pools, "usable_tokens": (total - pools) * block_size,
                "num_running": int(snapshot["num_running"]), "num_waiting": int(snapshot["num_waiting"])}

    def idle_snapshot(self) -> Dict[str, Any]:
        snapshot = super().idle_snapshot()
        pools = int(self._capacity().get("pools", 1))
        snapshot["all_blocks_free"] = snapshot["free_blocks"] >= snapshot["total_blocks"] - pools
        return snapshot

    def status(self) -> Dict[str, Any]:
        body = super().status()
        body["poisoned"] = self._poisoned
        body["switching"] = self._switching
        body["layout"] = list(self.spec.layout)
        return body

    async def await_idle(self, timeout_s: float = 300.0) -> Dict[str, Any]:
        """``HeteroEngine.await_idle`` with scheduler counts that also work for ``DPScheduler``."""
        deadline = time.monotonic() + timeout_s
        while True:
            with self._step_fence():
                error = self.engine.stats()["error"]
                if error:
                    raise RuntimeError(f"engine step loop failed: {error}")
                capacity = scheduler_capacity(self.scheduler)
                pending = len(self.engine_core.batch_queue or ())
                native_work = bool(self.scheduler.has_requests())
                idle = (self._inflight == 0 and not self.llm_engine.has_unfinished_requests()
                        and not native_work and pending == 0)
                snapshot = {"instance_id": self.instance_id, "idle": idle, "inflight": self._inflight,
                            "num_running": capacity["num_running"], "num_waiting": capacity["num_waiting"],
                            "free_blocks": capacity["free_blocks"], "pending_batches": pending,
                            "scheduler_has_work": native_work, "draining": self._draining}
            if idle or time.monotonic() >= deadline:
                return snapshot
            await asyncio.sleep(0.05)

    # -- in-engine layout switch ----------------------------------------------------------------
    def _capture_layout(self) -> _Layout:
        """The active layout as the EngineCore holds it right now."""
        core = self.engine_core
        return _Layout(
            spec=self.spec, config=self.vllm_config, scheduler=core.scheduler,
            structured_output_manager=getattr(core, "structured_output_manager", None),
            request_block_hasher=getattr(core, "request_block_hasher", None),
            batch_queue_size=int(getattr(core, "batch_queue_size", 1) or 1),
            available_memory=int(getattr(core, "available_gpu_memory_for_kv_cache", -1)))

    def _precheck_layout(self, target: EngineSpec, built: bool) -> None:
        if self._current is None:
            raise RuntimeError(f"{self.instance_id} has no active layout (not initialized)")
        if not built and target.layout not in self._layouts:
            raise ValueError(f"{self.instance_id}: no parked scheduler for layout {target.layout} "
                             f"(have {sorted(self._layouts)}); the hosts must report built=True")

    @contextlib.contextmanager
    def _timed_executor(self, executor: Any, segments: Dict[str, float]):
        """Attribute the executor's KV-init calls to ``segments`` (instance-level wrappers, removed after)."""
        names = {"get_kv_cache_specs": "kv_spec", "determine_available_memory": "available_memory"}
        # exactly what was in the instance dict (pre-existing instance overrides survive), nothing else
        saved = {method: executor.__dict__.get(method, _MISSING) for method in names}
        for method, name in names.items():
            original = getattr(executor, method)

            def wrapped(*args: Any, _original: Any = original, _name: str = name, **kwargs: Any) -> Any:
                step = time.time()
                try:
                    return _original(*args, **kwargs)
                finally:
                    segments[_name] += time.time() - step
            setattr(executor, method, wrapped)
        try:
            yield
        finally:
            for method, value in saved.items():
                if value is _MISSING:
                    executor.__dict__.pop(method, None)
                else:
                    setattr(executor, method, value)

    def _make_scheduler(self, config: Any, kv_cache_config: Any) -> tuple:
        """The target scheduler, constructed exactly as ``EngineCore.__init__`` does (core.py:137-160, 213-224).
        Returns ``(scheduler, structured_output_manager, request_block_hasher)``."""
        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core.kv_cache_utils import (get_request_block_hasher, init_none_hash,
                                                 resolve_kv_cache_block_sizes)
        from vllm.v1.structured_output import StructuredOutputManager
        core = self.engine_core
        manager = StructuredOutputManager(config)
        scheduler_cls = config.scheduler_config.get_scheduler_cls()
        if len(kv_cache_config.kv_cache_groups) == 0 and config.scheduler_config.enable_chunked_prefill:
            config.scheduler_config.enable_chunked_prefill = False
        block_size, hash_block_size = resolve_kv_cache_block_sizes(kv_cache_config, config)
        scheduler = scheduler_cls(
            vllm_config=config, kv_cache_config=kv_cache_config, structured_output_manager=manager,
            include_finished_set=False, log_stats=core.log_stats, block_size=block_size,
            hash_block_size=hash_block_size)
        hasher = None
        if config.cache_config.enable_prefix_caching:
            caching_hash_fn = get_hash_fn_by_name(config.cache_config.prefix_caching_hash_algo)
            init_none_hash(caching_hash_fn)
            hasher = get_request_block_hasher(hash_block_size, caching_hash_fn)
        return scheduler, manager, hasher

    _EXECUTOR_CONFIG_ATTRS = ("vllm_config", "model_config", "cache_config", "lora_config", "load_config",
                              "parallel_config", "scheduler_config", "device_config", "speculative_config",
                              "observability_config")

    def _build_layout(self, target: EngineSpec, segments: Dict[str, float]) -> _Layout:
        """First visit of ``target``: the hosts hold its runner and weights; allocate its KV and warm it up
        through the existing executor (``EngineCore._initialize_kv_caches``: specs -> available memory ->
        ``initialize_from_config`` -> warm-up), then build its scheduler. The executor is pointed at the target
        config for the duration (its warm-up bookkeeping writes into it); the active layout's executor
        attributes, ``available_gpu_memory_for_kv_cache`` (KV init overwrites it) and scheduler are restored
        or left alone on any failure, and the active layout's value is parked before KV init runs."""
        core, executor = self.engine_core, self._executor
        saved_memory = int(getattr(core, "available_gpu_memory_for_kv_cache", -1))
        if self._current is not None:
            self._current.available_memory = saved_memory
        saved_attrs = {name: getattr(executor, name, _MISSING) for name in self._EXECUTOR_CONFIG_ATTRS}
        scheduler = None
        try:
            config = self._make_config(target)
            self._retarget_executor(config)
            timings = getattr(executor, "timings", {})
            timings.pop("kv_alloc", None)
            timings.pop("warmup", None)
            step = time.time()
            with self._timed_executor(executor, segments):
                kv_cache_config = core._initialize_kv_caches(config)
                if self.frontend_settings.get("defer_warmup", False):
                    executor.start_warmup()
                    executor.wait_for_warmup()
            segments["kv_init"] = timings.get("kv_alloc", 0.0)
            segments["warmup"] = timings.get("warmup", 0.0)
            if "kv_alloc" not in timings:  # an executor that does not report: whatever is left of the call
                segments["kv_init"] = max(0.0, time.time() - step - segments["kv_spec"]
                                          - segments["available_memory"] - segments["warmup"])
            target_memory = int(getattr(core, "available_gpu_memory_for_kv_cache", -1))
            step = time.time()
            scheduler, manager, hasher = self._make_scheduler(config, kv_cache_config)
            if target.dp_size > 1 and "DPScheduler" not in type(scheduler).__name__:
                raise RuntimeError(f"layout {target.layout} needs tpu-inference's DPScheduler, the platform hook "
                                   f"gave {type(scheduler).__name__} (was TPU_MULTIPROCESS_DP off, "
                                   f"data_parallel_size passed?)")
            segments["scheduler_build"] = time.time() - step
        except BaseException:
            if scheduler is not None:
                self._shutdown_scheduler(scheduler)
            core.available_gpu_memory_for_kv_cache = saved_memory
            for name, value in saved_attrs.items():
                if value is _MISSING:  # absent before the retarget: remove what it installed
                    executor.__dict__.pop(name, None)
                else:
                    setattr(executor, name, value)
            raise
        core.available_gpu_memory_for_kv_cache = saved_memory  # the active layout stays active until the swap
        return _Layout(spec=target, config=config, scheduler=scheduler, structured_output_manager=manager,
                       kv_cache_config=kv_cache_config, request_block_hasher=hasher,
                       batch_queue_size=int(config.max_concurrent_batches), available_memory=target_memory)

    def _retarget_executor(self, config: Any) -> None:
        executor = self._executor
        executor.vllm_config = config
        for name in ("model_config", "cache_config", "lora_config", "load_config", "parallel_config",
                     "scheduler_config", "device_config", "speculative_config", "observability_config"):
            setattr(executor, name, getattr(config, name))

    @staticmethod
    def _shutdown_scheduler(scheduler: Any) -> Optional[str]:
        try:
            scheduler.shutdown()
        except BaseException as error:  # noqa: BLE001 - best effort, reported
            return f"{type(error).__name__}: {error}"
        return None

    def _install_layout(self, layout: _Layout, reset_prefix_cache: bool) -> Optional[_Layout]:
        """Swap ``layout`` into the EngineCore (and the frontend's own references) under the engine lock.
        Returns the layout that was parked under the same key and has been replaced (to shut down), if any."""
        core, driver = self.engine_core, self.engine
        guard = driver.priority_lock() if driver is not None else contextlib.nullcontext()
        with guard:
            if core.scheduler.has_requests() or core.batch_queue or self.llm_engine.has_unfinished_requests():
                raise RuntimeError(f"{self.instance_id} took work while switching layout")
            if reset_prefix_cache:  # the parked layout's KV was reallocated empty by the hosts
                if not layout.scheduler.reset_prefix_cache(reset_running_requests=False, reset_connector=False) \
                        and layout.config.cache_config.enable_prefix_caching:
                    raise RuntimeError(f"layout {layout.spec.layout}: prefix cache reset refused")
            core.vllm_config = layout.config
            core.scheduler = layout.scheduler
            core.structured_output_manager = layout.structured_output_manager
            core.request_block_hasher = layout.request_block_hasher
            core.available_gpu_memory_for_kv_cache = layout.available_memory
            core.async_scheduling = layout.config.scheduler_config.async_scheduling
            core.batch_queue_size = layout.batch_queue_size
            core.batch_queue = deque(maxlen=layout.batch_queue_size) if layout.batch_queue_size > 1 else None
            if hasattr(core, "step"):
                core.step_fn = core.step if core.batch_queue is None else core.step_with_batch_queue
            self._retarget_executor(layout.config)
            # LLMEngine keeps its own references; its logger_manager/stat loggers were built from the initial
            # config and cannot be rebuilt in place (their static metric labels keep the first layout's values).
            for name in ("vllm_config", "model_config", "observability_config"):
                if hasattr(self.llm_engine, name):
                    setattr(self.llm_engine, name,
                            layout.config if name == "vllm_config" else getattr(layout.config, name))
            self.vllm_config, self.scheduler = layout.config, layout.scheduler
            replaced = self._layouts.get(layout.spec.layout)
            self._layouts[layout.spec.layout] = layout
            self._current = layout
            if driver is not None and hasattr(driver, "_publish_snapshot"):
                driver._publish_snapshot()
            return replaced if replaced is not None and replaced is not layout else None

    def _switch_blocking(self, target: EngineSpec, built: bool) -> Dict[str, Any]:
        self._dp_env()
        segments = {name: 0.0 for name in self.LAYOUT_SEGMENTS}
        step = time.time()
        evidence = self._verify_kernels(self._handles(), expect_layout=target.layout)
        segments["verify"] = time.time() - step
        before = self._rpc.call(compile_count)
        layout = self._build_layout(target, segments) if built else self._layouts[target.layout]
        step = time.time()
        try:
            replaced = self._install_layout(layout, reset_prefix_cache=not built)
        except BaseException:
            if built:
                self._shutdown_scheduler(layout.scheduler)
            raise
        layout.visits += 0 if built else 1
        self._retarget(target)
        errors = []
        if replaced is not None:  # a rebuilt layout replaces its parked predecessor
            error = self._shutdown_scheduler(replaced.scheduler)
            errors += [error] if error else []
        segments["swap"] = time.time() - step
        after = self._rpc.call(compile_count)
        self._init_evidence["resolved_engine_config"] = self._config_evidence(self.vllm_config)
        return {"segments": segments, "compiles": max(a - b for a, b in zip(after, before)),
                "scheduler_cls": type(layout.scheduler).__module__ + "." + type(layout.scheduler).__name__,
                "num_gpu_blocks": int(getattr(layout.config.cache_config, "num_gpu_blocks", 0) or 0),
                "parked": sorted(list(k) for k in self._layouts if k != target.layout),
                "shutdown_errors": errors, "kv_capacity": self.kv_capacity(), **evidence}

    async def _apply_layout(self, target: EngineSpec, built: bool) -> Dict[str, Any]:
        return await asyncio.get_running_loop().run_in_executor(None, self._switch_blocking, target, built)

    def _config_evidence(self, vllm_config: Any) -> Dict[str, Any]:
        cache, sched, parallel = vllm_config.cache_config, vllm_config.scheduler_config, vllm_config.parallel_config
        return {"model": vllm_config.model_config.model, "dtype": str(vllm_config.model_config.dtype),
                "kv_cache_dtype": cache.cache_dtype, "block_size": int(cache.block_size),
                "num_gpu_blocks": int(getattr(cache, "num_gpu_blocks", 0) or 0),
                "max_model_len": int(vllm_config.model_config.max_model_len),
                "max_num_seqs": int(sched.max_num_seqs),
                "max_num_batched_tokens": int(sched.max_num_batched_tokens),
                "tensor_parallel_size": int(parallel.tensor_parallel_size), "dp_size": int(self.spec.dp_size),
                "scheduler_cls": type(self.scheduler).__module__ + "." + type(self.scheduler).__name__,
                "executor_class": type(self._executor).__name__, "model_build": self._model_build()}

    def _observe_failures(self) -> None:
        if self._poisoned is None and self.engine is not None:
            try:
                error = self.engine.stats().get("error")
            except Exception:  # noqa: BLE001
                return
            if error:
                self._poison(f"engine step loop failed: {error}")

    # -- serving ---------------------------------------------------------------------------
    async def generate(self, request: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        self._admit()
        if request is None:
            return await _RawEngine.generate(self, **kwargs)
        prompts = self._prompts(request)
        steps = int(request.get("steps", request.get("max_tokens", 16)))
        base = request.get("request_id")
        results = await asyncio.gather(*[
            _RawEngine.generate(self, prompt_token_ids=p, max_tokens=steps,
                                request_id=f"{base}-{i}" if base else None,
                                ignore_eos=bool(request.get("ignore_eos", True)))
            for i, p in enumerate(prompts)])
        return self._merge(list(results))

    async def start_iteration_capture(self) -> None:
        if hasattr(self.scheduler, "omp_start_capture"):
            return await _RawEngine.start_iteration_capture(self)

    async def stop_iteration_capture(self) -> Dict[str, Any]:
        if hasattr(self.scheduler, "omp_stop_capture"):
            return await _RawEngine.stop_iteration_capture(self)
        return {"rows": [], "audit": {"supported": False, "reason": "stock vLLM scheduler"}}

    def cancel_request(self, **kwargs: Any) -> Dict[str, Any]:
        if not hasattr(self.scheduler, "omp_ownership"):
            raise NotImplementedError("cancel_request needs hetero.scheduler.HeteroScheduler")
        return _RawEngine.cancel_request(self, **kwargs)

    async def drain(self, timeout_s: float = 300.0) -> Dict[str, Any]:
        result = await _RawEngine.drain(self, timeout_s)
        if self._poisoned is not None:
            self.state = "poisoned"
        return result

    async def shutdown(self) -> Dict[str, Any]:
        """Stop the step loop, shut down every scheduler (the active one and each parked layout's: DPScheduler
        children are processes) and detach the executor. The engines in the host kernels stay up: the
        controller owns them (``stop_engine``)."""
        schedulers = [layout.scheduler for layout in self._layouts.values()]
        managers = [layout.structured_output_manager for layout in self._layouts.values()]
        snapshot = await _RawEngine.shutdown(self)
        errors = [e for e in (self._shutdown_scheduler(s) for s in schedulers) if e]
        for manager in managers:  # EngineCore.shutdown would clear the active one's backend
            clear = getattr(manager, "clear_backend", None)
            if callable(clear):
                with contextlib.suppress(Exception):
                    clear()
        if schedulers:
            snapshot["schedulers_shutdown"] = len(schedulers)
        if errors:
            snapshot["scheduler_shutdown_errors"] = errors
        self._layouts.clear()
        self._current = None
        unregister_view_rpc(self._rpc_key)
        if self._rpc is not None:
            self._rpc.close()
        if self._poisoned is not None:
            snapshot["poisoned"] = self._poisoned
        return snapshot


# ================================================================================= actors


SyntheticFrontend = ray.remote(num_cpus=0, max_restarts=0)(SyntheticFrontendCore)
VllmFrontend = ray.remote(num_cpus=0, max_restarts=0)(VllmFrontendCore)
