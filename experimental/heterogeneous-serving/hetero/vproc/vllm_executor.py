"""vLLM executor for an EngineCore that addresses only its view's host kernels (plan C3).

Where the EngineCore lives
    In the controller (Ray driver) process, in-process: ``LLMEngine(vllm_config=
    engine_config(spec), executor_class=VllmViewExecutor, multiprocess_mode=False)``
    as in scripts/ici_engine.py:121. One EngineCore per engine generation, created
    and dropped by the controller; host kernels never run a scheduler. Reasons:
    the host kernel is one long-lived SPMD process that must survive engine
    switches, the scheduler output has to reach every host of the view anyway,
    and the controller already owns the placement table and the drain protocol.

How ``step`` maps onto it
    ``EngineCore.step`` = ``scheduler.schedule()`` -> ``execute_model(so)`` ->
    ``sample_tokens(grammar)`` -> ``scheduler.update_from_output``. Each executor
    call is one fan-out of ``HostKernel.engine_call("step", envelope)`` to ALL
    hosts of the view concurrently (SPMD), where ``envelope`` is
    ``{"rpc": worker_method, "args": ..., "kwargs": ...}`` (``VllmEngine.step``).
    Only ``Engine`` protocol method names are used, so the host kernel needs no
    vLLM-specific RPC. The result of rank 0 (the driver worker) is returned for
    execute_model/sample_tokens; other calls return one result per host in view
    rank order. KV-cache configs are per host (``zip``), as in hetero/ici_executor.py.

Lifecycle
    The executor never starts, builds or shuts anything down: the host kernels
    are started by the controller (``start_engine``), the engine is built there,
    and ``shutdown`` here only detaches. Warm-up can be deferred (``defer_warmup``)
    so several engines' warm-ups run concurrently (InPlaceEngine design.md).

Failure and ordering
    Every fan-out is serialized per ViewRpc (one in flight, submitted in the same order to every
    host; ``call_async`` keeps the slot until its results are gathered). A fan-out that raises or
    times out *poisons* the ViewRpc: its peers may be stuck in a collective the failed host never
    entered, so no further call is made (``ViewPoisoned`` immediately) and ``on_poison`` is told
    why. The frontend (``hetero.vproc.frontend``) turns that into ``health()["ok"] = False``; the
    controller then tears the block down. Nothing here retries.

Importable without vLLM: ``VllmViewExecutor`` is created on first attribute access
(module ``__getattr__``), which is also what ``resolve_obj_by_qualname`` uses.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence

DEFAULT_TIMEOUT_S = 900.0


def executor_additional_config(hosts: Sequence[Dict[str, str]], timeout_s: float = DEFAULT_TIMEOUT_S,
                               defer_warmup: bool = False, resolver: Optional[str] = None,
                               rpc_key: Optional[str] = None) -> Dict[str, Any]:
    """``additional_config["vproc"]`` for the controller-side VllmConfig (JSON-serializable:
    vLLM hashes additional_config). ``hosts``: one ``{"name", "namespace"}`` per host of
    the view, in view rank order (named Ray actors). ``resolver``: ``"module:function"``
    returning ``(handles, transport)`` instead, for tests and non-Ray transports.
    ``rpc_key``: use the ``ViewRpc`` registered under this key (``register_view_rpc``) instead
    of creating one, so the frontend's own calls and the executor's share one serialized,
    poisonable channel; the executor then never closes it."""
    return {"vproc": {"hosts": list(hosts), "timeout_s": timeout_s,
                      "defer_warmup": defer_warmup, "resolver": resolver, "rpc_key": rpc_key}}


class RayTransport:
    """``HostKernel`` Ray actor handles: ``engine_call`` for work, ``health`` for checks."""

    def submit(self, host: Any, envelope: Dict[str, Any]) -> Any:
        return host.engine_call.remote("step", envelope)

    def gather(self, refs: List[Any], timeout: float) -> List[Any]:
        import ray
        return ray.get(refs, timeout=timeout)

    def health(self, hosts: Sequence[Any], timeout: float) -> List[Any]:
        import ray
        return ray.get([host.health.remote() for host in hosts], timeout=timeout)


class ViewPoisoned(RuntimeError):
    """A fan-out of this view failed earlier; the view's hosts may be out of step."""


class ViewRpc:
    """Fan one worker call out to every host of a view and collect results in rank order.

    ``transport``: ``submit(host, envelope) -> ref`` (must not block, and preserve
    submission order per host), ``gather(refs, timeout) -> [result]``,
    ``health(hosts, timeout) -> [report]``. Pure Python, no vLLM, no Ray.

    One fan-out is in flight at a time (``fanouts``/``max_active`` count them); a failed or
    timed-out one poisons the object (module docstring). ``on_poison(reason)`` runs once, from
    the thread that saw the failure.
    """

    def __init__(self, hosts: Sequence[Any], transport: Any, timeout_s: float = DEFAULT_TIMEOUT_S,
                 on_poison: Optional[Callable[[str], None]] = None):
        if not hosts:
            raise ValueError("a view needs at least one host kernel")
        self.hosts = list(hosts)
        self.transport = transport
        self.timeout_s = timeout_s
        self.on_poison = on_poison
        self.poisoned: Optional[str] = None
        self.fanouts = 0
        self.max_active = 0
        self._active = 0
        self._waiter = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vproc-rpc")
        self._slot = threading.Lock()  # the one in-flight fan-out; released by whoever finishes it
        self._state = threading.Lock()

    # -- poisoning ---------------------------------------------------------
    def poison(self, reason: str) -> None:
        with self._state:
            first = self.poisoned is None
            if first:
                self.poisoned = reason
        if first and self.on_poison is not None:
            try:
                self.on_poison(reason)
            except Exception:  # noqa: BLE001 - a broken observer must not hide the failure
                pass

    def _guard(self) -> None:
        if self.poisoned is not None:
            raise ViewPoisoned(f"view is poisoned: {self.poisoned}")

    # -- one fan-out (slot held) ---------------------------------------------
    def _send(self, method: Any, args: Sequence[Any], kwargs: Optional[Dict[str, Any]],
              per_host_args: Optional[Sequence[Sequence[Any]]]) -> List[Any]:
        self._guard()
        if per_host_args is not None and len(per_host_args) != len(self.hosts):
            raise ValueError(f"{len(per_host_args)} per-host argument lists for {len(self.hosts)} hosts")
        name = method if isinstance(method, str) else getattr(method, "__name__", "<callable>")
        with self._state:
            self._active += 1
            self.fanouts += 1
            self.max_active = max(self.max_active, self._active)
        try:
            return [self.transport.submit(host, {
                "rpc": method, "args": tuple(args if per_host_args is None else per_host_args[rank]),
                "kwargs": kwargs or {}}) for rank, host in enumerate(self.hosts)]
        except BaseException as error:
            self._end()
            self.poison(f"submit of {name!r} failed on a host: {type(error).__name__}: {error}")
            raise

    def _end(self) -> None:
        with self._state:
            self._active -= 1

    def _collect(self, refs: List[Any], timeout: Optional[float], name: str) -> List[Any]:
        try:
            rows = self.transport.gather(refs, self.timeout_s if timeout is None else timeout)
            return [row["result"] for row in rows]
        except BaseException as error:
            self.poison(f"fan-out of {name!r} failed: {type(error).__name__}: {error}")
            raise
        finally:
            self._end()

    @staticmethod
    def _name(method: Any) -> str:
        return method if isinstance(method, str) else getattr(method, "__name__", "<callable>")

    def call(self, method: Any, args: Sequence[Any] = (), kwargs: Optional[Dict[str, Any]] = None,
             timeout: Optional[float] = None, per_host_args: Optional[Sequence[Sequence[Any]]] = None) -> List[Any]:
        with self._slot:
            refs = self._send(method, args, kwargs, per_host_args)
            return self._collect(refs, timeout, self._name(method))

    def call_async(self, method: Any, args: Sequence[Any] = (), kwargs: Optional[Dict[str, Any]] = None,
                   timeout: Optional[float] = None) -> "Future[List[Any]]":
        """Submit now (in call order, after any earlier fan-out finished); gather on a waiter thread.
        The slot stays taken until the results are in."""
        self._slot.acquire()
        try:
            refs = self._send(method, args, kwargs, None)
        except BaseException:
            self._slot.release()
            raise

        def finish() -> List[Any]:
            try:
                return self._collect(refs, timeout, self._name(method))
            finally:
                self._slot.release()

        try:
            return self._waiter.submit(finish)
        except BaseException:
            self._slot.release()
            raise

    def health(self) -> List[Any]:
        return self.transport.health(self.hosts, min(self.timeout_s, 30.0))

    def close(self) -> None:
        self._waiter.shutdown(wait=False)
        self.hosts = []


_SHARED: Dict[str, ViewRpc] = {}


def register_view_rpc(key: str, rpc: ViewRpc) -> None:
    """Make ``rpc`` the channel of executors whose config says ``rpc_key=key`` (same process)."""
    _SHARED[key] = rpc


def unregister_view_rpc(key: str) -> None:
    _SHARED.pop(key, None)


def _first(future: "Future[List[Any]]") -> "Future[Any]":
    out: "Future[Any]" = Future()

    def done(source):
        try:
            out.set_result(source.result()[0])
        except BaseException as error:  # noqa: BLE001 - forwarded to the waiter
            out.set_exception(error)

    future.add_done_callback(done)
    return out


def view_rpc_from_config(vllm_config: Any) -> ViewRpc:
    """Resolve the hosts named in ``additional_config["vproc"]`` into a ``ViewRpc``."""
    settings = vllm_config.additional_config.get("vproc")
    if not settings:
        raise ValueError("additional_config['vproc'] is missing (see executor_additional_config)")
    timeout = float(settings.get("timeout_s", DEFAULT_TIMEOUT_S))
    if settings.get("rpc_key"):
        try:
            return _SHARED[settings["rpc_key"]]
        except KeyError:
            raise ValueError(f"no ViewRpc registered under rpc_key {settings['rpc_key']!r}") from None
    if settings.get("resolver"):
        from .cpu_block import resolve
        hosts, transport = resolve(settings["resolver"])(settings)
    else:
        import ray
        hosts = [ray.get_actor(row["name"], namespace=row["namespace"]) for row in settings["hosts"]]
        transport = RayTransport()
    return ViewRpc(hosts, transport, timeout)


def make_executor_class() -> type:
    """Build ``VllmViewExecutor`` (needs vLLM; called lazily)."""
    from vllm.v1.executor.abstract import Executor

    class VllmViewExecutor(Executor):
        """vLLM v1 executor over the host kernels of one view (module docstring)."""

        uses_ray = False
        supports_pp = False

        def _init_executor(self) -> None:
            if self.parallel_config.pipeline_parallel_size != 1:
                raise ValueError("a view-scoped engine does not support pipeline parallelism")
            settings = self.vllm_config.additional_config["vproc"]
            self.rpc = view_rpc_from_config(self.vllm_config)
            self._owns_rpc = not settings.get("rpc_key")
            self.defer_warmup = bool(settings.get("defer_warmup"))
            self._warmup: Optional["Future[List[Any]]"] = None
            self.timings: Dict[str, float] = {}  # kv_alloc, warmup: seconds, read by the frontend

        # RPC ---------------------------------------------------------------
        def collective_rpc(self, method, timeout=None, args=(), kwargs=None, non_block=False):
            if non_block:
                return self.rpc.call_async(method, args, kwargs, timeout)
            return self.rpc.call(method, args, kwargs, timeout)

        def execute_model(self, scheduler_output, non_block=False):
            # Only the driver worker (rank 0) returns the output (tpu_worker.py:574-606).
            result = self.collective_rpc("execute_model", args=(scheduler_output,), non_block=non_block)
            return _first(result) if non_block else result[0]

        def sample_tokens(self, grammar_output, non_block=False):
            result = self.collective_rpc("sample_tokens", args=(grammar_output,), non_block=non_block)
            return _first(result) if non_block else result[0]

        def take_draft_token_ids(self):
            return self.collective_rpc("take_draft_token_ids")[0]

        # initialization ----------------------------------------------------
        def initialize_from_config(self, kv_cache_configs) -> None:
            if len(kv_cache_configs) != len(self.rpc.hosts):
                raise ValueError(f"{len(kv_cache_configs)} KV cache configs for {len(self.rpc.hosts)} hosts")
            start = time.time()
            self.rpc.call("initialize_from_config", per_host_args=[(config,) for config in kv_cache_configs])
            self.timings["kv_alloc"] = time.time() - start
            if not self.defer_warmup:
                self.start_warmup()
                self.wait_for_warmup()

        def start_warmup(self) -> None:
            self._warmup_started = time.time()
            self._warmup = self.rpc.call_async("compile_or_warm_up_model")

        def wait_for_warmup(self) -> None:
            if self._warmup is None:
                return
            times = self._warmup.result(timeout=self.rpc.timeout_s)
            self._warmup = None
            self.timings["warmup"] = time.time() - self._warmup_started
            if times:  # as Executor.initialize_from_config: max over workers
                self.vllm_config.compilation_config.compilation_time = max(t.language_model for t in times)
                self.vllm_config.compilation_config.encoder_compilation_time = max(t.encoder for t in times)

        # lifecycle ---------------------------------------------------------
        def check_health(self) -> None:
            if self.rpc.poisoned is not None:
                raise RuntimeError(f"view is poisoned: {self.rpc.poisoned}")
            bad = [report for report in self.rpc.health()
                   if isinstance(report, dict) and (report.get("failure") or report.get("ok") is False)]
            if bad:
                raise RuntimeError(f"view host kernels unhealthy: {bad}")

        def shutdown(self) -> None:
            # Host kernels and the engine in them belong to the controller (stop_engine).
            if self._owns_rpc:
                self.rpc.close()

    return VllmViewExecutor


_EXECUTOR_CLASS: Optional[type] = None


def __getattr__(name: str) -> Any:
    global _EXECUTOR_CLASS
    if name == "VllmViewExecutor":
        if _EXECUTOR_CLASS is None:
            _EXECUTOR_CLASS = make_executor_class()
            _EXECUTOR_CLASS.__module__ = __name__
            _EXECUTOR_CLASS.__qualname__ = name
        return _EXECUTOR_CLASS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
