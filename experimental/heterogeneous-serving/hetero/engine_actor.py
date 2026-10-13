"""One logical serving instance: a Ray actor owning a vLLM TPU engine.

The actor is the *driver* for its instance.  It never touches a TPU device
itself; it pins vLLM's Ray executor to this instance's owned placement group and
ships the resolved layout down to :class:`hetero.tpu_worker.HeteroTPUWorker`.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import socket
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

import ray

from .native_backend import BACKEND_REVISION, RUNTIME_REVISION

DEFAULT_MODEL = "/models/gemma-4-31b"

# Copied into vLLM's Ray worker actors in addition to its own allowlist.  The
# authoritative path is ``additional_config['omp_hetero']``; this only helps any
# code that reads the environment before ``init_device`` runs.
EXTRA_ENV_VARS_TO_COPY = (
    "TPU_TOPOLOGY",
    "TPU_PROCESS_BOUNDS",
    "TPU_CHIPS_PER_PROCESS_BOUNDS",
    "TPU_PROCESS_ADDRESSES",
    "TPU_PROCESS_PORT",
    "JAX_PLATFORMS",
    "JAX_COMPILATION_CACHE_DIR",
    "OMP_HETERO_INSTANCE_ID",
)


@dataclasses.dataclass
class EngineSettings:
    """Frozen per-instance engine configuration for one static arm."""

    model: str = DEFAULT_MODEL
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "float32"
    activation_dtype: str = "float32"
    matmul_precision: str = "highest"
    backend_revision: str = BACKEND_REVISION
    runtime_revision: str = RUNTIME_REVISION
    max_model_len: int = 8192
    max_num_seqs: int = 512
    max_num_batched_tokens: int = 2048
    gpu_memory_utilization: float = 0.85
    enable_prefix_caching: bool = False
    enforce_eager: bool = False
    seed: int = 0
    weight_snapshot: str = "off"
    weight_snapshot_verify: bool = True

    def __post_init__(self) -> None:
        for field, choices in (
            ("dtype", ("auto", "half", "float16", "bfloat16", "float", "float32")),
            ("kv_cache_dtype", ("float32",)),
            ("activation_dtype", ("float32",)),
            ("matmul_precision", ("highest",)),
            ("weight_snapshot", ("off", "write", "read")),
        ):
            value = getattr(self, field)
            if value not in choices:
                raise ValueError(f"unsupported {field} {value!r}; expected one of {choices}")
        if not isinstance(self.weight_snapshot_verify, bool):
            raise ValueError("weight_snapshot_verify must be a boolean")
        if self.backend_revision != BACKEND_REVISION:
            raise ValueError(
                f"backend_revision {self.backend_revision!r} does not match "
                f"this package's native backend {BACKEND_REVISION!r}"
            )
        if self.runtime_revision != RUNTIME_REVISION:
            raise ValueError("runtime_revision does not match this serving source package")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EngineSettings":
        data = dict(data or {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown engine settings: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@ray.remote
class HeteroEngine:
    """Driver actor for one logical serving instance."""

    def __init__(
        self,
        *,
        instance: Dict[str, Any],
        layout_id: str,
        layout_digest: str,
        engine_settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.instance = instance
        self.instance_id = instance["instance_id"]
        self.layout_id = layout_id
        self.layout_digest = layout_digest
        self.settings = EngineSettings.from_dict(engine_settings)
        self.generation = int(time.time())
        self.node_ip = ray.util.get_node_ip_address()
        self.hostname = socket.gethostname()
        self.state = "starting"
        self.engine = None
        self.llm_engine = None
        self.vllm_config = None
        self.engine_core = None
        self.scheduler = None
        self._init_evidence: Dict[str, Any] = {}
        self._inflight = 0
        self._served = 0
        self._draining = False
        self._pg_name: Optional[str] = None
        self._endpoint = None
        self._heartbeat_task = None
        self.endpoint_port: Optional[int] = None
        self._quiesced: Dict[str, str] = {}

        self._transactions: Dict[str, Dict[str, Any]] = {}
        self._migration_locks: Dict[str, asyncio.Lock] = {}
        self._cancellations: Dict[tuple, Dict[str, Any]] = {}
        self._continuations: Dict[str, Dict[str, Any]] = {}
        self._migrated: Dict[str, str] = {}
        self._cache_operation_lock = asyncio.Lock()
        self._kv_notices: Dict[str, asyncio.Queue] = {}
        self._kv_refs: Dict[str, Dict[int, Any]] = {}
        self._kv_early: Dict[str, Dict[int, list]] = {}

    # KV transport completions arrive from worker background threads.
    KV_CHUNK_TIMEOUT_S = 120.0

    async def omp_kv_notify(self, transaction_id: str, message: Dict[str, Any]) -> None:
        queue = self._kv_notices.get(transaction_id)
        if queue is not None:
            queue.put_nowait(message)

    async def omp_kv_refs(self, transaction_id: str, keys: List[int]) -> List[Any]:
        """Hand piece refs to worker threads through Ray, preserving ownership."""
        refs = self._kv_refs[transaction_id]
        return [refs[key] for key in keys]

    async def _kv_chunk(self, transaction_id: str, kind: str, workers: int, chunk: int) -> list:
        """Every worker's completion of one numbered chunk; later chunks are held back.

        With two chunks in flight a fast worker can finish chunk k+1 before a
        slow worker finishes chunk k, so completions are matched by chunk number.
        """
        queue = self._kv_notices[transaction_id]
        early = self._kv_early.setdefault(transaction_id, {})
        received: Dict[int, Dict[str, Any]] = {}
        for message in early.pop(chunk, []):
            received[message["process_index"]] = message
        deadline = time.monotonic() + self.KV_CHUNK_TIMEOUT_S
        while len(received) < workers:
            try:
                message = await asyncio.wait_for(queue.get(), max(0.0, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                evidence: Any = "unavailable"
                with contextlib.suppress(Exception):
                    async with self._cache_operation():
                        evidence = self.engine_core.model_executor.collective_rpc("omp_kv_debug")
                raise TimeoutError(
                    f"KV {kind} chunk {chunk} timed out after {self.KV_CHUNK_TIMEOUT_S}s with "
                    f"{len(received)}/{workers} completions; workers: {json.dumps(evidence)[:6000]}"
                ) from None
            if message.get("kind") != kind:
                raise RuntimeError(f"unexpected {message.get('kind')} KV completion during {kind}")
            if message.get("error"):
                raise RuntimeError(f"worker {message['process_index']} KV {kind} chunk "
                                   f"{message.get('chunk')} failed: {message['error']}")
            if message["chunk"] != chunk:
                if message["chunk"] < chunk:
                    raise RuntimeError(f"stale KV {kind} completion for chunk {message['chunk']}")
                early.setdefault(message["chunk"], []).append(message)
                continue
            if message["process_index"] in received:
                raise RuntimeError(f"duplicate worker KV {kind} completion for chunk {chunk}")
            received[message["process_index"]] = message
        return [received[index] for index in sorted(received)]

    def ping(self) -> str:
        return f"PONG {self.instance_id}@{self.hostname}"

    def identity(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "layout_id": self.layout_id,
            "layout_digest": self.layout_digest,
            "generation": self.generation,
            "node_ip": self.node_ip,
            "hostname": self.hostname,
            "tp_size": self.instance["tp_size"],
            "host_ips": list(self.instance["host_ips"]),
            "engine_settings": self.settings.to_dict(),
            "backend_revision": self.settings.backend_revision,
            "model_build": self._init_evidence.get("resolved_engine_config", {}).get("model_build"),
            "state": self.state,
        }

    def adopt_identity(
        self, layout_id: str, layout_digest: str, instance: Optional[Dict[str, Any]] = None,
        *, dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Keep serving, but report membership of another layout (partial reconfiguration).

        The engine, placement group, generation and endpoint are untouched; only
        the layout identity reported by identity()/the metrics labels changes.
        ``instance`` is the target layout's resolved placement for this instance
        and must name exactly the chips and hosts this engine already occupies.
        """
        if self.state != "ready" or self._draining:
            raise RuntimeError(f"cannot adopt a layout identity while {self.state}")
        if instance is not None:
            def placement(body: Dict[str, Any]) -> Any:
                return (body["instance_id"], body["tp_size"], list(body["host_ips"]),
                        [(h["node_ip"], [list(c) for c in h["chip_coords"]]) for h in body["hosts"]])
            if placement(instance) != placement(self.instance):
                raise RuntimeError(
                    f"{self.instance_id} does not occupy the same chips under layout {layout_id}")
        if not dry_run:
            self.layout_id = layout_id
            self.layout_digest = layout_digest
        return self.identity()

    def _model_build(self) -> str:
        return json.dumps({
            "vllm": _vllm_version(),
            "native_backend": self.settings.backend_revision,
            "runtime_revision": self.settings.runtime_revision,
            "parameter_dtype": str(self.vllm_config.model_config.dtype),
            "activation_dtype": self.settings.activation_dtype,
            "kv_cache_dtype": self.settings.kv_cache_dtype,
            "matmul_precision": self.settings.matmul_precision,
        }, sort_keys=True, separators=(",", ":"))

    def matches(self, instance: Dict[str, Any], engine_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Verifies an attached actor really implements the requested config."""
        want = EngineSettings.from_dict(engine_settings).to_dict()
        diffs = {}
        if instance["tp_size"] != self.instance["tp_size"]:
            diffs["tp_size"] = [self.instance["tp_size"], instance["tp_size"]]
        if list(instance["host_ips"]) != list(self.instance["host_ips"]):
            diffs["host_ips"] = [list(self.instance["host_ips"]), list(instance["host_ips"])]
        have = self.settings.to_dict()
        for key, value in want.items():
            if have.get(key) != value:
                diffs[f"engine.{key}"] = [have.get(key), value]
        return {"matches": not diffs, "differences": diffs, "state": self.state}

    # ------------------------------------------------------------ bring-up

    # vLLM's InputProcessor rewrites every request id to
    # f"{external}-{random_uuid():.8}" and keeps the original only on
    # EngineCoreRequest.external_req_id, which the scheduler's Request does not
    # carry (vllm/v1/engine/input_processor.py:223-240, vllm/v1/request.py:81).
    # The gateway owns globally unique ids already, so the randomisation is
    # disabled and every seam -- ownership, fencing, snapshots, block tables --
    # speaks exactly one identity.
    REQUIRED_ENV = {"VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1"}

    def prepare_environment(self) -> Dict[str, Any]:
        """Applies this instance's runtime environment to the driver process."""
        intended = dict(self.instance["intended_runtime_env"])
        intended.update(self.REQUIRED_ENV)
        existing = os.environ.get("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", "")
        merged = sorted(set(filter(None, existing.split(","))) | set(EXTRA_ENV_VARS_TO_COPY))
        intended["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] = ",".join(merged)
        for key, value in intended.items():
            os.environ[key] = value
        cache_dir = intended.get("JAX_COMPILATION_CACHE_DIR")
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        return {
            "instance_id": self.instance_id,
            "driver_node_ip": self.node_ip,
            "driver_pid": os.getpid(),
            "intended_env": intended,
            "effective_env": {k: os.environ.get(k) for k in intended},
            "physical_tpu_worker_id": os.environ.get("TPU_WORKER_ID"),
        }

    def _hetero_payload(self) -> Dict[str, Any]:
        chips_by_host = {
            host["node_ip"]: [list(c) for c in host["chip_coords"]]
            for host in self.instance["hosts"]
        }
        return {
            "instance_id": self.instance_id,
            "layout_id": self.layout_id,
            "layout_digest": self.layout_digest,
            "tp_size": self.instance["tp_size"],
            "host_ips": list(self.instance["host_ips"]),
            "runtime_env": dict(self.instance["intended_runtime_env"]),
            "chip_coords_by_host": chips_by_host,
            "chip_coords": [c for coords in chips_by_host.values() for c in coords],
            "dtype": self.settings.dtype,
            "kv_cache_dtype": self.settings.kv_cache_dtype,
            "activation_dtype": self.settings.activation_dtype,
            "matmul_precision": self.settings.matmul_precision,
            "backend_revision": self.settings.backend_revision,
            "runtime_revision": self.settings.runtime_revision,
            "weight_snapshot": self.settings.weight_snapshot,
            "weight_snapshot_verify": self.settings.weight_snapshot_verify,
        }

    async def initialize(self, *, pg_name: str, timeout_s: float = 3600.0) -> Dict[str, Any]:
        """Builds the vLLM engine bound to this instance's owned placement group."""
        if self.engine is not None:
            return {"status": "already_initialized", **self._init_evidence}

        from ray.util.placement_group import get_placement_group
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine

        from .inproc_engine import InprocAsyncEngine

        started = time.time()
        self._pg_name = pg_name
        pg = get_placement_group(pg_name)

        payload = self._hetero_payload()
        engine_args = EngineArgs(
            model=self.settings.model,
            tensor_parallel_size=self.instance["tp_size"],
            distributed_executor_backend="ray",
            dtype=self.settings.dtype,
            kv_cache_dtype=self.settings.kv_cache_dtype,
            additional_config={
                "activation_dtype": self.settings.activation_dtype,
                "matmul_precision": self.settings.matmul_precision,
                "omp_hetero": payload,
            },
            max_model_len=self.settings.max_model_len,
            max_num_seqs=self.settings.max_num_seqs,
            max_num_batched_tokens=self.settings.max_num_batched_tokens,
            gpu_memory_utilization=self.settings.gpu_memory_utilization,
            enable_prefix_caching=self.settings.enable_prefix_caching,
            enforce_eager=self.settings.enforce_eager,
            seed=self.settings.seed,
            max_logprobs=32,
            trust_remote_code=True,
            disable_log_stats=False,
            limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
        )
        vllm_config = engine_args.create_engine_config()
        vllm_config.parallel_config.placement_group = pg
        vllm_config.parallel_config.worker_cls = "hetero.tpu_worker.HeteroTPUWorker"
        vllm_config.scheduler_config.scheduler_cls = "hetero.scheduler.HeteroScheduler"
        llm_engine = LLMEngine(
            **_supported_kwargs(
                LLMEngine.__init__,
                vllm_config=vllm_config,
                executor_class=_executor_class(vllm_config),
                log_stats=True,
                multiprocess_mode=False,
            )
        )

        self.vllm_config = vllm_config
        self.llm_engine = llm_engine
        self.engine = InprocAsyncEngine(llm_engine, asyncio.get_running_loop())
        self._bind_engine_internals()
        self.engine.start()
        self.state = "ready"
        self._init_evidence = {
            "status": "initialized",
            "instance_id": self.instance_id,
            "init_s": round(time.time() - started, 2),
            "engine_settings": self.settings.to_dict(),
            "resolved_engine_config": self._config_evidence(vllm_config),
            "kv_capacity": self.kv_capacity(),
            "worker_evidence": self.worker_evidence(),
            "engine_init_started_at": started,
            "worker_init_phases": self.engine_core.model_executor.collective_rpc("omp_init_phases"),
        }
        return self._init_evidence

    def _bind_engine_internals(self) -> None:
        """Resolves the in-process EngineCore and its scheduler."""
        client = getattr(self.llm_engine, "engine_core", None)
        core = getattr(client, "engine_core", None)
        if core is None:
            raise RuntimeError(
                "LLMEngine is not running an in-process EngineCore; the fabric requires "
                "multiprocess_mode=False so the scheduler is reachable from the driver"
            )
        self.engine_core = core
        self.scheduler = core.scheduler
        self._install_iteration_hooks()

    def _config_evidence(self, vllm_config: Any) -> Dict[str, Any]:
        cache = vllm_config.cache_config
        sched = vllm_config.scheduler_config
        parallel = vllm_config.parallel_config
        additional = vllm_config.additional_config
        return {
            "model": vllm_config.model_config.model,
            "dtype": str(vllm_config.model_config.dtype),
            "kv_cache_dtype": cache.cache_dtype,
            "activation_dtype": additional["activation_dtype"],
            "matmul_precision": additional["matmul_precision"],
            "backend_revision": additional["omp_hetero"]["backend_revision"],
            "max_model_len": int(vllm_config.model_config.max_model_len),
            "block_size": int(cache.block_size),
            "gpu_memory_utilization": float(cache.gpu_memory_utilization),
            "enable_prefix_caching": bool(cache.enable_prefix_caching),
            "num_gpu_blocks": int(getattr(cache, "num_gpu_blocks", 0) or 0),
            "max_num_seqs": int(sched.max_num_seqs),
            "max_num_batched_tokens": int(sched.max_num_batched_tokens),
            "scheduler_cls": type(self.scheduler).__module__
            + "."
            + type(self.scheduler).__name__,
            "scheduler_mro": [
                f"{c.__module__}.{c.__name__}" for c in type(self.scheduler).__mro__[:4]
            ],
            "tensor_parallel_size": int(parallel.tensor_parallel_size),
            "world_size": int(parallel.world_size),
            "executor_class": type(self.engine_core.model_executor).__name__,
            "batch_queue_size": int(getattr(self.engine_core, "batch_queue_size", 0) or 0),
            "vllm_version": _vllm_version(),
            "model_build": self._model_build(),
        }

    # ---------------------------------------------------------- evidence

    def worker_evidence(self) -> List[Dict[str, Any]]:
        return self.engine_core.model_executor.collective_rpc("omp_worker_evidence")

    def worker_init_phases(self) -> List[Dict[str, Any]]:
        return self.engine_core.model_executor.collective_rpc("omp_init_phases")

    def snapshot_weights(self) -> List[Dict[str, Any]]:
        """Explicitly publish every worker's final, device-resident runner state."""
        if self.state != "ready" or self._inflight:
            raise RuntimeError("snapshot requires a ready, idle engine")
        return self.engine_core.model_executor.collective_rpc("omp_snapshot_weights")

    def worker_kv_evidence(self) -> List[Dict[str, Any]]:
        return self.engine_core.model_executor.collective_rpc("omp_kv_cache_evidence")

    def _capacity(self) -> Dict[str, Any]:
        """Lock-free capacity view.

        Published by the step thread after every iteration, so the gateway's
        capacity refresh never contends with model execution and never blocks
        this actor's event loop.
        """
        snapshot = self.engine.capacity_snapshot() if self.engine is not None else {}
        if snapshot:
            return snapshot
        with self._step_fence():
            pool = self.scheduler.kv_cache_manager.block_pool
            return {
                "total_blocks": int(pool.num_gpu_blocks),
                "free_blocks": int(pool.get_num_free_blocks()),
                "num_running": len(self.scheduler.running),
                "num_waiting": len(self.scheduler.waiting),
                "published_at": time.time(),
            }

    def _free_blocks(self) -> int:
        return int(self._capacity()["free_blocks"])

    def kv_capacity(self) -> Dict[str, Any]:
        snapshot = self._capacity()
        total = int(snapshot["total_blocks"])
        free = int(snapshot["free_blocks"])
        running = int(snapshot["num_running"])
        waiting = int(snapshot["num_waiting"])
        block_size = int(self.vllm_config.cache_config.block_size)
        return {
            "block_size": block_size,
            "total_blocks": total,
            "free_blocks": free,
            "usable_blocks": total - 1,
            "usable_tokens": (total - 1) * block_size,
            "num_running": running,
            "num_waiting": waiting,
        }

    def status(self) -> Dict[str, Any]:
        body = {
            "instance_id": self.instance_id,
            "state": "draining" if self._draining else self.state,
            "generation": self.generation,
            "inflight": self._inflight,
            "served": self._served,
            "iterations_captured": len(getattr(self.scheduler, "omp_iterations", []) or []),
        }
        if self.scheduler is not None:
            body.update(self.kv_capacity())
        return body

    # --------------------------------------------------- iteration capture

    def _install_iteration_hooks(self) -> None:
        """No-op: telemetry lives in :class:`hetero.scheduler.HeteroScheduler`."""
        if not hasattr(self.scheduler, "omp_start_capture"):
            raise RuntimeError(
                f"engine {self.instance_id} is running "
                f"{type(self.scheduler).__name__}, not the migration-aware scheduler"
            )

    async def start_iteration_capture(self) -> None:
        idle = await self.await_idle(timeout_s=30.0)
        if not idle["idle"]:
            raise TimeoutError(f"engine {self.instance_id} did not drain before capture")
        with self._step_fence():
            self.scheduler.omp_start_capture()

    async def stop_iteration_capture(self) -> Dict[str, Any]:
        # Final client delivery precedes retirement of the async queue tail.
        # Keep capturing while the normal step thread drains it; never force a
        # synchronous step or turn off native asynchronous scheduling.
        idle = await self.await_idle()
        if not idle["idle"]:
            raise TimeoutError(f"engine {self.instance_id} did not drain capture")
        with self._step_fence():
            rows = self.scheduler.omp_stop_capture()
            return {"rows": rows, "audit": self.scheduler.omp_capture_audit()}

    # ------------------------------------------------------------ serving

    async def generate(
        self,
        *,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[Sequence[int]] = None,
        max_tokens: int = 16,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        request_id: Optional[str] = None,
        ignore_eos: bool = False,
    ) -> Dict[str, Any]:
        """Runs one real request to completion and returns useful-token accounting."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        if self._draining:
            raise RuntimeError(f"{self.instance_id} is draining and refuses new work")
        if self.engine is None:
            raise RuntimeError(f"{self.instance_id} engine is not initialized")
        rid = request_id or f"{self.instance_id}-{uuid.uuid4().hex[:12]}"
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            ignore_eos=ignore_eos,
        )
        if prompt_token_ids is not None:
            inputs: Any = TokensPrompt(prompt_token_ids=list(prompt_token_ids))
        elif prompt is not None:
            inputs = prompt
        else:
            raise ValueError("either prompt or prompt_token_ids is required")

        self._inflight += 1
        started = time.time()
        try:
            final, first_token_s = await self.engine.generate(rid, inputs, params)
        finally:
            self._inflight -= 1
        migrated = self._migrated.get(rid)
        if migrated is not None:
            # The committed prefix comes from the fenced snapshot, not from the
            # last asynchronously delivered RequestOutput: the stream can lag
            # the scheduler by a delivery turn, and an undercount here would
            # silently duplicate tokens on the destination.
            snapshot = self._transactions[migrated]["snapshot"]
            committed = list(snapshot["generated_token_ids"])
            return {
                "request_id": rid,
                "instance_id": self.instance_id,
                "prompt_tokens": len(snapshot["prompt_token_ids"]),
                "output_tokens": len(committed),
                "output_token_ids": committed,
                "text": snapshot["generated_text"],
                "finish_reason": "migrated",
                "migrated_transaction": migrated,
                "ttft_s": first_token_s,
                "elapsed_s": round(time.time() - started, 4),
                "started_at_s": started,
                "finished_at_s": time.time(),
            }
        if final is None:
            raise RuntimeError(f"request {rid} produced no output")
        self._served += 1
        completion = final.outputs[0]
        elapsed = time.time() - started
        return {
            "migrated_transaction": migrated,
            "request_id": rid,
            "instance_id": self.instance_id,
            "prompt_tokens": len(final.prompt_token_ids or []),
            "output_tokens": len(completion.token_ids),
            "output_token_ids": list(completion.token_ids),
            "text": completion.text,
            "finish_reason": "migrated" if migrated else completion.finish_reason,
            "ttft_s": first_token_s,
            "elapsed_s": round(elapsed, 4),
            "started_at_s": started,
            "finished_at_s": started + elapsed,
        }

    async def generate_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Runs a real concurrent batch on this instance."""
        tasks = [asyncio.create_task(self.generate(**req)) for req in requests]
        return list(await asyncio.gather(*tasks))

    async def generate_qualification(self, *, request_id: str,
                                     prompt_token_ids: Sequence[int], max_tokens: int,
                                     checkpoint_tokens: int) -> Dict[str, Any]:
        """Real greedy source request with an exact native diagnostic checkpoint.

        Keep this RPC's ObjectRef: it resolves through normal generate delivery
        after migration release, or after abort/unquiesce resumes the source.
        """
        if self.engine is None or self._draining:
            raise RuntimeError("engine is not accepting qualification work")
        if len(prompt_token_ids) + max_tokens > self.settings.max_model_len:
            raise ValueError("qualification generation budget exceeds native model context")
        with self._step_fence():
            self.scheduler.omp_arm_qualification_checkpoint(request_id, checkpoint_tokens, max_tokens)
        try:
            return await self.generate(request_id=request_id, prompt_token_ids=prompt_token_ids,
                                       max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
        finally:
            with self._step_fence():
                self.scheduler.omp_qualification_checkpoints.pop(request_id, None)
                if self.scheduler.omp_internal_id(request_id) is None:
                    self.scheduler.omp_fenced.discard(request_id)

    async def await_qualification_checkpoint(self, *, request_id: str,
                                           checkpoint_tokens: int,
                                           timeout_s: float = 300.0) -> Dict[str, Any]:
        """Observe the exact retired boundary; never substitute an overshot history."""
        deadline = time.monotonic() + timeout_s
        while True:
            with self._step_fence():
                if self.engine._error is not None:
                    raise RuntimeError(f"native step loop failed: {self.engine._error}")
                checkpoint = self.scheduler.omp_qualification_checkpoints.get(request_id)
                state = self.scheduler.omp_fence_state(request_id)
                view = self.scheduler.omp_request_view(request_id)
                if checkpoint is not None and checkpoint[0] != checkpoint_tokens:
                    raise ValueError("requested checkpoint differs from the armed boundary")
                if view is not None:
                    if checkpoint is None:
                        raise RuntimeError("request has no armed qualification checkpoint")
                    if view["output_tokens"] > checkpoint_tokens:
                        raise RuntimeError("native checkpoint overshot; no tokens may be discarded")
                    if (view["output_tokens"] == checkpoint_tokens and state["fenced"]
                            and not state["inflight"] and view["num_output_placeholders"] == 0):
                        return {"request_id": request_id, "instance_id": self.instance_id,
                                "generation": self.generation, "checkpoint_tokens": checkpoint_tokens,
                                "checkpoint_ready": True, **view}
            if time.monotonic() >= deadline:
                raise TimeoutError("exact native qualification checkpoint did not retire")
            await asyncio.sleep(0.005)

    async def next_token_diagnostic(self, *, token_ids: Sequence[int],
                                    request_id: Optional[str] = None) -> Dict[str, Any]:
        """Real one-token matched-history probe; not a migration sampling feature."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        if self.engine is None or self._draining:
            raise RuntimeError("engine is not accepting diagnostic work")
        rid = request_id or f"diagnostic-{self.instance_id}-{uuid.uuid4().hex}"
        params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=32, ignore_eos=True)
        self._inflight += 1
        started = time.time()
        try:
            final, first_token_s = await self.engine.generate(
                rid, TokensPrompt(prompt_token_ids=list(token_ids)), params)
        finally:
            self._inflight -= 1
        completion = final.outputs[0]
        if len(completion.token_ids) != 1 or not completion.logprobs:
            raise RuntimeError("native diagnostic did not return one token and logprobs")
        self._served += 1
        return {
            "request_id": rid, "instance_id": self.instance_id, "generation": self.generation,
            "token_id": int(completion.token_ids[0]),
            "top_logprobs": {int(token): float(value.logprob)
                             for token, value in completion.logprobs[0].items()},
            "ttft_s": first_token_s, "elapsed_s": time.time() - started,
        }

    # ---------------------------------------------------------- migration

    @contextlib.contextmanager
    def _step_fence(self):
        """Serialises scheduler mutation against the engine step thread.

        The driver holds this lock for one EngineCore step, NOT the lifetime of
        every submitted batch: native async work can remain in batch_queue on
        return. Migration must additionally fence the request and await its
        identity-tracked retirements before reading/releasing KV. No awaits here.
        """
        driver = self.engine
        if driver is None:
            yield
            return
        with driver.priority_lock():
            yield

    def migration_candidates(
        self,
        *,
        min_generated_tokens: int = 64,
        min_context_tokens: int = 512,
        min_remaining_tokens: int = 64,
        limit: int = 8,
    ) -> Dict[str, Any]:
        """Reports local candidates and pressure. Selection stays at the gateway."""
        with self._step_fence():
            rows = self.scheduler.omp_candidates(
                min_generated_tokens=min_generated_tokens,
                min_context_tokens=min_context_tokens,
                min_remaining_tokens=min_remaining_tokens,
            )
            pressure = self.scheduler.omp_pressure()
        return {
            "instance_id": self.instance_id,
            "generation": self.generation,
            "pressure": pressure,
            "candidates": rows[:limit],
            "observed_at_s": time.time(),
        }

    def _transaction_lock(self, transaction_id: str) -> asyncio.Lock:
        lock = self._migration_locks.get(transaction_id)
        if lock is None:
            lock = self._migration_locks[transaction_id] = asyncio.Lock()
        return lock

    def _receipt(self, transaction_id: str, **fields: Any) -> Dict[str, Any]:
        return {"instance_id": self.instance_id, "generation": self.generation,
                "transaction_id": transaction_id, **fields}

    def _source_identity(self, request_id: str, source_generation: int,
                         source_epoch: int) -> Dict[str, Any]:
        if source_generation != self.generation:
            raise ValueError("stale source generation")
        return {"request_id": request_id, "source_instance": self.instance_id,
                "source_generation": source_generation, "source_epoch": source_epoch}

    def _check_certificate(self, transaction_id: str, ownership: Dict[str, Any],
                           expected: Dict[str, Any]) -> None:
        if (ownership.get("transaction_id") != transaction_id
                or ownership.get("state") not in ("committed", "released")
                or not ownership.get("committed_at")
                or ownership.get("destination_epoch") != ownership.get("source_epoch", -2) + 1):
            raise ValueError("a committed ownership certificate is required")
        for field in ("request_id", "source_instance", "destination_instance",
                      "source_epoch", "destination_epoch", "source_generation",
                      "destination_generation", "mechanism"):
            if field not in ownership or (field in expected and ownership[field] != expected[field]):
                raise ValueError(f"ownership certificate disagrees on {field}")

    async def quiesce(
        self, *, request_id: str, transaction_id: str, source_generation: int,
        source_epoch: int, timeout_s: float = 30.0,
    ) -> Dict[str, Any]:
        """Fence one owned epoch and snapshot after its native async work retires."""
        identity = self._source_identity(request_id, source_generation, source_epoch)
        async with self._transaction_lock(transaction_id):
            record = self._transactions.setdefault(transaction_id, {"phase": "source"})
            if record.get("source", identity) != identity:
                raise ValueError("transaction source identity changed")
            record["source"] = identity
            if record.get("resume"):
                raise RuntimeError("source transaction was already resumed")
            if record.get("snapshot") is not None:
                return record["snapshot"]
            started = time.time()
            with self._step_fence():
                owner = self.scheduler.omp_ownership.get(request_id)
                if not owner or not owner["owned"] or owner["epoch"] != source_epoch:
                    raise RuntimeError("source does not own this native epoch")
                prior = self._quiesced.get(request_id)
                if prior is not None and prior != transaction_id:
                    raise RuntimeError("request is fenced by another transaction")
                if not self.scheduler.omp_fence(request_id):
                    raise KeyError(f"request {request_id} is not resident on {self.instance_id}")
                self._quiesced[request_id] = transaction_id
            deadline = time.monotonic() + timeout_s
            while True:
                with self._step_fence():
                    state = self.scheduler.omp_fence_state(request_id)
                    view = self.scheduler.omp_request_view(request_id)
                    if not state["known"]:
                        raise RuntimeError("request completed before its fence retired")
                    if not state["inflight"] and view["num_output_placeholders"] == 0:
                        snapshot = self.scheduler.omp_snapshot(request_id)
                        frontend = self.llm_engine.output_processor.request_states[state["internal_id"]]
                        snapshot["generated_text"] = frontend.detokenizer.get_next_output_text(False, False)
                        snapshot["kv_addressing"] = self.scheduler.omp_block_ids(request_id)
                        break
                if time.monotonic() >= deadline:
                    # Fail closed. Only the gateway's acknowledged abort can
                    # resume a source whose destination may still be preparing.
                    raise TimeoutError("request did not retire its native work before quiesce timeout")
                await asyncio.sleep(0.005)
            snapshot.update(identity, transaction_id=transaction_id,
                            quiesce_s=time.time() - started, model=self.settings.model,
                            model_build=self._model_build())

            record.update(phase="quiesced", snapshot=snapshot)
            return snapshot

    async def unquiesce(self, *, request_id: str, transaction_id: str,
                        source_generation: int, source_epoch: int) -> Dict[str, Any]:
        """Resume only this precommit transaction; retain a late-quiesce tombstone."""
        from vllm.v1.request import RequestStatus

        identity = self._source_identity(request_id, source_generation, source_epoch)
        async with self._transaction_lock(transaction_id):
            record = self._transactions.setdefault(transaction_id, {"phase": "source"})
            if record.get("source", identity) != identity:
                raise ValueError("transaction source identity changed")
            record["source"] = identity
            if record.get("ownership") or record.get("release"):
                raise RuntimeError("cannot resume a source after ownership committed")
            if record.get("resume"):
                return record["resume"]
            evidence: Dict[str, Any] = {}
            async with self._cache_operation():
                owner = self.scheduler.omp_ownership.get(request_id)
                if owner and (owner["epoch"] != source_epoch or not owner["owned"]):
                    raise RuntimeError("stale source ownership epoch")
                if self._quiesced.get(request_id, transaction_id) != transaction_id:
                    raise RuntimeError("source fence belongs to another transaction")
                internal = self.scheduler.omp_internal_id(request_id)
                request = self.scheduler.requests.get(internal) if internal else None
                restore = (request is not None and internal in self.scheduler.omp_fenced
                           and request.status in (RequestStatus.RUNNING, RequestStatus.PREEMPTED))
                if restore:
                    if (self.scheduler.omp_is_inflight(request_id) or request.num_output_placeholders
                            or request.num_in_flight_tokens):
                        raise RuntimeError("source async outputs have not retired")
                    payload = {
                        "request_id": internal, "transaction_id": transaction_id,
                        "source_generation": source_generation, "source_epoch": source_epoch,
                        "prompt_token_ids": list(request.prompt_token_ids),
                        "generated_token_ids": list(request.output_token_ids),
                        "num_computed_tokens": int(request.num_computed_tokens),
                        "block_ids": self.scheduler.kv_cache_manager.get_block_ids(internal),
                    }
                    workers = self.engine_core.model_executor.collective_rpc(
                        "omp_resume_fenced_request_state", args=(payload,))
                    if not workers or not all(row.get("history_restored") for row in workers):
                        raise RuntimeError("source worker history restoration was not acknowledged")
                    evidence["worker_restore"] = workers
                self.scheduler.omp_unfence(request_id, worker_state_restored=restore)
                self._quiesced.pop(request_id, None)
                self.engine._publish_snapshot()

            receipt = self._receipt(transaction_id, request_id=request_id, resumed=True, evidence=evidence)
            record.update(phase="resumed", resume=receipt)
            return receipt

    async def release_source(self, *, request_id: str, transaction_id: str,
                             ownership: Dict[str, Any]) -> Dict[str, Any]:
        """Free source KV and detach delivery exactly once, never before commit."""
        async with self._transaction_lock(transaction_id):
            record = self._transactions[transaction_id]
            source = record["source"]
            self._check_certificate(transaction_id, ownership, source)
            if request_id != source["request_id"]:
                raise ValueError("release request does not match transaction")
            if record.get("release"):
                self._check_certificate(transaction_id, ownership, record["ownership"])
                return record["release"]
            if record.get("resume") or self._quiesced.get(request_id) != transaction_id:
                raise RuntimeError("source is not quiesced under this transaction")
            record["ownership"] = dict(ownership)
            with self._step_fence():
                internal = self.scheduler.omp_internal_id(request_id)
                result = self.scheduler.omp_release(request_id, source["source_epoch"])
                self.llm_engine.output_processor.abort_requests([internal], internal=True)
                self._migrated[internal] = transaction_id
                result["stream_detached"] = self.engine.detach(internal)
                self.engine._publish_snapshot()
            self._quiesced.pop(request_id, None)
            receipt = self._receipt(transaction_id, **result)
            record.update(phase="released", release=receipt)
            return receipt

    async def reserve_continuation(self, *, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Reserve actual native pages, including decode growth, before quiesce."""
        keys = ("transaction_id", "request_id", "destination_epoch", "destination_generation",
                "max_context_tokens", "mechanism", "source_instance", "source_generation", "source_epoch")
        identity = {key: payload[key] for key in keys}
        # Pages to hold now; defaults to the full final context. Length-blind
        # policies pass current context plus a growth allowance instead.
        identity["reserve_tokens"] = int(payload.get("reserve_tokens") or payload["max_context_tokens"])
        identity["destination_instance"] = self.instance_id
        txn = identity["transaction_id"]
        async with self._transaction_lock(txn):
            record = self._transactions.setdefault(txn, {"phase": "new"})
            if record.get("identity", identity) != identity:
                raise ValueError("transaction reservation identity changed")
            if record.get("phase") == "aborted":
                return self._receipt(txn, accepted=False, reasons=["transaction_aborted"])
            if record.get("reservation"):
                return record["reservation"]
            record["identity"] = identity
            reasons = []
            if self._draining or self.engine is None:
                reasons.append("destination_not_accepting_work")
            if identity["destination_generation"] != self.generation:
                reasons.append("stale_destination_generation")
            if identity["destination_epoch"] != identity["source_epoch"] + 1:
                reasons.append("invalid_destination_epoch")
            if identity["mechanism"] not in ("recompute", "kv_host"):
                reasons.append("unsupported_mechanism")
            if not 0 < identity["max_context_tokens"] <= self.settings.max_model_len:
                reasons.append("invalid_max_context_tokens")
            if not 0 < identity["reserve_tokens"] <= identity["max_context_tokens"]:
                reasons.append("invalid_reserve_tokens")
            count = 0
            if not reasons:
                try:
                    with self._step_fence():
                        count = self.scheduler.omp_reserve_continuation(identity)
                        self.engine._publish_snapshot()
                except (ValueError, RuntimeError) as exc:
                    reasons.append(str(exc))
            receipt = self._receipt(txn, request_id=identity["request_id"],
                                    accepted=not reasons, reasons=reasons,
                                    native_reserved_blocks=count)
            record.update(phase="reserved" if not reasons else "declined", reservation=receipt)
            return receipt

    async def prepare_continuation(self, *, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare native request, worker history and optional host KV off scheduler."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from .scheduler import _unsupported_continuation
        from types import SimpleNamespace

        txn = payload["transaction_id"]
        prior = self._transactions.get(txn)
        if prior is not None and prior["phase"] == "aborted":
            return self._receipt(txn, accepted=False, reasons=["transaction_aborted"])
        exported = None
        if payload.get("mechanism") == "kv_host":
            # The gateway passes a nested ObjectRef, never materialized arrays.
            # Await outside the transaction lock so abort can tombstone a stalled
            # export. Recheck the tombstone before any native mutation below.
            reference = payload["kv_export"]["object_ref"]
            if not isinstance(reference, ray.ObjectRef):
                raise ValueError("host KV requires a nested source-export ObjectRef")
            exported = await asyncio.wrap_future(reference.future())
        async with self._transaction_lock(txn):
            record = self._transactions[txn]
            if record["phase"] == "aborted":
                return self._receipt(txn, accepted=False, reasons=["transaction_aborted"])
            identity = record["identity"]
            for key, value in identity.items():
                if key != "destination_instance" and payload.get(key) != value:
                    raise ValueError(f"continuation identity disagrees on {key}")
            qualification_top32 = payload.get("qualification_top32", False)
            if type(qualification_top32) is not bool:
                raise ValueError("qualification_top32 must be a diagnostic boolean")
            if record.get("preparation"):
                if qualification_top32 != record["qualification_top32"]:
                    raise ValueError("transaction diagnostic mode changed")
                return record["preparation"]
            record["qualification_top32"] = qualification_top32
            if record["phase"] != "reserved" or not record["reservation"]["accepted"]:
                raise RuntimeError("native capacity was not reserved")
            started = time.monotonic()
            sampling = dict(payload["sampling"])
            reasons = list(payload.get("unsupported_continuation") or [])
            reasons.extend(_unsupported_continuation(SimpleNamespace(**sampling)))
            prompt = list(payload["prompt_token_ids"])
            generated = list(payload["generated_token_ids"])
            context = len(prompt) + len(generated)
            remaining = int(sampling["max_tokens"]) - len(generated)
            if payload.get("model") != self.settings.model or payload.get("model_build") != self._model_build():
                reasons.append("model_or_native_build_mismatch")
            if not prompt or not generated or not 0 <= int(payload["num_computed_tokens"]) < context:
                reasons.append("continuation_requires_history_and_an_uncomputed_token")
            if remaining <= 0 or remaining != payload["remaining_tokens"]:
                reasons.append("invalid_remaining_generation_budget")
            if len(prompt) + int(sampling["max_tokens"]) != identity["max_context_tokens"]:
                reasons.append("snapshot_exceeds_or_disagrees_with_reserved_growth")
            if context + 1 > identity["reserve_tokens"]:
                # Length-blind reservations hold only a growth allowance; the
                # snapshot outgrew it, so abort and let the source resume.
                reasons.append("snapshot_exceeds_reserved_pages")
            computed = int(payload["num_computed_tokens"]) if identity["mechanism"] == "kv_host" else 0
            evidence: Dict[str, Any] = {}
            if not reasons:
                try:
                    native_sampling = dict(sampling)
                    if qualification_top32:
                        # Only the destination diagnostic requests logprobs. The
                        # SOURCE contract above still rejects any migrated logprob
                        # state, including logprobs=0; nothing is fabricated/reset.
                        native_sampling["logprobs"] = 32
                    frontend, native, wave = self.engine.prepare_continuation(
                        f"{identity['request_id']}#e{identity['destination_epoch']}",
                        TokensPrompt(prompt_token_ids=prompt), SamplingParams(**native_sampling), generated,
                    )
                    native.omp_qualification_top32 = qualification_top32
                    native.skip_reading_prefix_cache = identity["mechanism"] == "recompute"
                    record.update(frontend=frontend, native=native, wave=wave,
                                  continuation_id=native.request_id)
                    addressing = None

                    def prepare_worker_state() -> None:
                        # Caller holds the drained native step lock.
                        record["worker_prepared"] = True
                        evidence["worker_prepare"] = self.engine_core.model_executor.collective_rpc(
                            "omp_prepare_continuation_state", args=({
                                "request_id": native.request_id, "transaction_id": txn,
                                "destination_generation": self.generation,
                                "prompt_token_ids": prompt, "generated_token_ids": generated,
                                "num_computed_tokens": computed, "sampling_params": native.sampling_params,
                                "block_ids": blocks.get_block_ids(),
                            },))
                        self.engine._publish_snapshot()

                    async with self._cache_operation():
                        blocks = self.scheduler.omp_prepare_continuation(txn, native)
                        native.num_computed_tokens = computed
                        evidence["native_block_ids"] = blocks.get_block_ids()
                        evidence["source_addressing"] = payload.get("kv_addressing")
                        if exported is not None:
                            if (exported.get("request_id") != identity["request_id"]
                                    or exported.get("instance_id") != identity["source_instance"]
                                    or exported.get("transaction_id") != txn
                                    or exported.get("source_generation") != identity["source_generation"]
                                    or exported["addressing"]["num_computed_tokens"] != computed):
                                raise ValueError("source KV export does not match the fenced snapshot")
                            addressing = self._destination_addressing(blocks, exported["addressing"])
                            evidence["source_addressing"] = exported["addressing"]
                            evidence["destination_addressing"] = addressing
                            evidence["export"] = {
                                key: exported.get(key) for key in (
                                    "request_id", "instance_id", "transaction_id",
                                    "source_generation", "source_epoch", "export_s", "bytes",
                                    "paused_s", "chunks", "gather_s", "d2h_s")
                            }
                            if getattr(self.scheduler, "needs_kv_cache_zeroing", False):
                                # Registered before any step can drain the new-block list.
                                imported_blocks = {block for group in addressing["groups"]
                                                   for block in group["block_ids"]}
                                self.scheduler._skip_zero_block_ids.update(imported_blocks)
                                record["imported_blocks"] = imported_blocks
                        else:
                            prepare_worker_state()
                    if exported is not None:
                        # The continuation is allocated but not admitted, so its
                        # pages are untouched while this engine keeps stepping.
                        evidence["import"] = await self._import_kv(txn, addressing, exported)
                        async with self._cache_operation():
                            prepare_worker_state()
                    record["payload"] = {key: value for key, value in payload.items() if key != "kv_export"}
                except Exception as exc:
                    reasons.append(f"native_prepare_failed: {exc}")
            receipt = self._receipt(
                txn, request_id=identity["request_id"], accepted=not reasons, reasons=reasons,
                rebuild_tokens=int(payload["num_computed_tokens"]) if identity["mechanism"] == "recompute" else 0,
                transferred_bytes=int(exported["bytes"]) if exported is not None and not reasons else 0,
                num_computed_tokens=computed, native_reserved_blocks=record["reservation"]["native_reserved_blocks"],
                prepare_s=time.monotonic() - started, evidence=evidence,
            )
            record.update(phase="prepared" if not reasons else "prepare_failed", preparation=receipt)
            return receipt

    def _destination_addressing(self, blocks: Any, source: Dict[str, Any]) -> Dict[str, Any]:
        """Translate logical source positions, not source physical IDs, to native pages."""
        groups = []
        if len(source["groups"]) != len(blocks.blocks):
            raise ValueError("source/destination native cache group counts differ")
        for gid, (group, table) in enumerate(zip(source["groups"], blocks.blocks)):
            if group["group_id"] != gid:
                raise ValueError("source cache groups are not in native order")
            ids = []
            for logical in group["logical_block_indices"]:
                if not 0 <= logical < len(table) or table[logical].is_null:
                    raise ValueError("source logical page has no native reserved destination")
                ids.append(int(table[logical].block_id))
            groups.append({**group, "block_ids": ids})
        return {"num_computed_tokens": source["num_computed_tokens"], "groups": groups}

    async def _import_kv(self, txn: str, addressing: Dict[str, Any],
                         exported: Dict[str, Any]) -> list:
        """Pipelined import: workers fetch/upload in the background while this
        engine steps; each pause only scatters a staged chunk and queues the next."""
        from .kv_transport import DEFAULT_STAGE_BYTES_PER_DEVICE, split_refs

        rpc = self.engine_core.model_executor.collective_rpc
        notifier = ray.get_runtime_context().current_actor
        self._kv_notices[txn] = asyncio.Queue()
        exported, self._kv_refs[txn] = split_refs(exported)
        rows: Dict[int, Dict[str, Any]] = {}
        cursor, outstanding, paused_s, pauses, chunks = 0, 0, 0.0, 0, 0
        total = len(exported["arrays"])

        async def advance(commit: bool) -> list:
            nonlocal cursor, outstanding, paused_s, pauses
            first = cursor == 0 and not commit
            paused = time.monotonic()
            async with self._cache_operation():
                results = rpc("omp_import_advance", args=(
                    txn, commit, cursor, DEFAULT_STAGE_BYTES_PER_DEVICE, notifier,
                    addressing if first else None, exported if first else None))
            paused_s += time.monotonic() - paused
            pauses += 1
            following = {row["next"] for row in results}
            submitted = {row["submitted"] for row in results}
            if len(following) != 1 or len(submitted) != 1:
                raise RuntimeError("workers disagree on the KV import chunk boundary")
            cursor = following.pop()
            outstanding += submitted.pop() - (1 if commit else 0)
            return results

        try:
            results = await advance(False)
            workers = len(results)
            if cursor < total:
                await advance(False)
            while outstanding:
                staged = await self._kv_chunk(txn, "import", workers, chunks)
                committed = await advance(True)
                chunks += 1
                by_process = {row["process_index"]: row["committed"] for row in committed}
                for stage in staged:
                    row = rows.setdefault(stage["process_index"], {
                        "process_index": stage["process_index"], "arrays_written": 0,
                        "bytes": 0, "fetched_bytes": 0, "head_conversions": [],
                        "fetch_s": 0.0, "stage_s": 0.0, "scatter_s": 0.0})
                    commit = by_process[stage["process_index"]]
                    row["arrays_written"] += commit["arrays_written"]
                    row["scatter_s"] += commit["scatter_s"]
                    for key in ("bytes", "fetched_bytes", "fetch_s", "stage_s"):
                        row[key] += stage[key]
                    row["head_conversions"].extend(stage["head_conversions"])
            if cursor != total:
                raise RuntimeError("KV import ended before covering every payload array")
        except BaseException:
            with contextlib.suppress(Exception):
                async with self._cache_operation():
                    rpc("omp_kv_discard", args=(txn,))
            raise
        finally:
            self._kv_notices.pop(txn, None)
            self._kv_refs.pop(txn, None)
            self._kv_early.pop(txn, None)
        return [{**row, "chunks": chunks, "pauses": pauses, "paused_s": paused_s}
                for _, row in sorted(rows.items())]

    async def commit_continuation(self, *, transaction_id: str,
                                  ownership: Dict[str, Any]) -> Dict[str, Any]:
        """Admit the prepared native request only with authoritative committed ownership."""
        async with self._transaction_lock(transaction_id):
            record = self._transactions[transaction_id]
            self._check_certificate(transaction_id, ownership, record["identity"])
            if record.get("activation"):
                return record["activation"]
            if record["phase"] not in ("prepared", "committed"):
                raise RuntimeError("no prepared continuation for this committed transaction")
            record.update(ownership=dict(ownership), phase="committed")
            # Mark commit before the first worker activation: even a lost/partial
            # acknowledgement must never allow abort to roll ownership back.
            async with self._cache_operation():
                self.scheduler.omp_commit_continuation(transaction_id)
                evidence = self.engine_core.model_executor.collective_rpc(
                    "omp_activate_continuation_state",
                    args=(record["continuation_id"], transaction_id, self.generation))
                payload = record["payload"]
                if payload["mechanism"] == "recompute":
                    self.scheduler.omp_replay_requests[payload["request_id"]] = int(payload["num_computed_tokens"])
                stream = self.engine.activate_continuation(record["frontend"], record["native"], record["wave"])
            started = time.time()
            record["started_at_s"] = started
            self._inflight += 1
            self._continuations[transaction_id] = record
            task = asyncio.create_task(self._finish_continuation(transaction_id, stream))
            record["task"] = task
            task.add_done_callback(lambda task: self._retire_continuation_task(transaction_id, task))
            receipt = self._receipt(transaction_id, continuation_id=record["continuation_id"],
                                    activated=True, replay_tokens=record["preparation"]["rebuild_tokens"],
                                    remaining_tokens=payload["remaining_tokens"],
                                    activated_at_s=started, evidence=evidence)
            record["activation"] = receipt
            return receipt

    async def _finish_continuation(self, transaction_id: str, stream: Any) -> Dict[str, Any]:
        record = self._transactions[transaction_id]
        payload = record["payload"]
        try:
            final = await stream.future
            migrated = self._migrated.get(record["continuation_id"])
            if migrated:
                # A second handoff may precede the first frontend delivery. Its
                # retired snapshot, not that possibly absent output, is authoritative.
                snapshot = self._transactions[migrated]["snapshot"]
                suffix = snapshot["generated_token_ids"][len(payload["generated_token_ids"]):]
                full_text = snapshot["generated_text"]
                text = full_text[len(payload.get("generated_text", "")):]
                finish_reason = "migrated"
            else:
                if final is None:
                    raise RuntimeError("native continuation produced no output")
                completion = final.outputs[0]
                suffix = list(completion.token_ids)
                full_text, text = stream.full_output_text, completion.text
                finish_reason = completion.finish_reason
                self._served += 1
            result = self._receipt(
                transaction_id, request_id=payload["request_id"], continuation_id=record["continuation_id"],
                replayed_tokens=record["preparation"]["rebuild_tokens"],
                new_output_tokens=len(suffix), new_output_token_ids=suffix,
                text=text, full_output_text=full_text, finish_reason=finish_reason,
                migrated_transaction=migrated, ttft_s=stream.first_token_s,
                first_token_at_s=(stream.created_at + stream.first_token_s
                                  if stream.first_token_s is not None else None),
                started_at_s=record["started_at_s"], finished_at_s=time.time(),
            )
            if record["qualification_top32"]:
                if stream.first_token_logprobs is None or stream.first_token_id is None:
                    raise RuntimeError("prepared native diagnostic produced no first-token logprobs")
                result.update(first_token_id=stream.first_token_id,
                              first_token_logprobs=stream.first_token_logprobs,
                              logprobs_origin="prepared_native_continuation")
            record["result"] = result
            return result
        except BaseException as exc:
            record["completion_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._inflight -= 1
            self._continuations.pop(transaction_id, None)
            with self._step_fence():
                owner = self.scheduler.omp_ownership[payload["request_id"]]
                if owner["native_id"] == record["continuation_id"]:
                    self.scheduler.omp_replay_requests.pop(payload["request_id"], None)
                self.scheduler.omp_continuations[transaction_id]["request"] = None
            record.pop("native", None)
            record.pop("frontend", None)

    def _retire_continuation_task(self, transaction_id: str, task: asyncio.Task) -> None:
        self._transactions[transaction_id].pop("task", None)
        if not task.cancelled():
            task.exception()  # Retrieve failures even if no client awaits the receipt.

    async def await_continuation(self, *, transaction_id: str,
                                 timeout_s: float = 3600.0) -> Dict[str, Any]:
        record = self._transactions[transaction_id]
        if "result" in record:
            return record["result"]
        if "completion_error" in record:
            raise RuntimeError(record["completion_error"])
        # A delivery timeout or cancelled waiter cannot cancel committed native work.
        return await asyncio.wait_for(asyncio.shield(record["task"]), timeout_s)

    async def abort_continuation(self, *, transaction_id: str) -> Dict[str, Any]:
        """Release precommit preparation only; a committed transaction is irreversible."""
        async with self._transaction_lock(transaction_id):
            record = self._transactions.setdefault(transaction_id, {"phase": "new"})
            if record.get("ownership") or record.get("activation"):
                raise RuntimeError("cannot roll back committed continuation")
            if record.get("abort"):
                return record["abort"]
            record["phase"] = "aborted"
            async with self._cache_operation():
                if record.get("worker_prepared"):
                    self.engine_core.model_executor.collective_rpc(
                        "omp_abort_continuation_state",
                        args=(record["continuation_id"], transaction_id, self.generation))
                if record.get("imported_blocks"):
                    self.scheduler._skip_zero_block_ids.difference_update(record["imported_blocks"])
                self.scheduler.omp_abort_continuation(transaction_id)
                self.engine._publish_snapshot()
            for key in ("native", "frontend", "payload"):
                record.pop(key, None)
            receipt = self._receipt(transaction_id, aborted=True, aborted_active=False)
            record["abort"] = receipt
            return receipt

    def continuation_status(self, *, transaction_id: str) -> Dict[str, Any]:
        record = self._transactions.get(transaction_id, {})
        return self._receipt(transaction_id, phase=record.get("phase", "unknown"),
                             **{key: record.get(key) for key in
                                ("reservation", "preparation", "activation", "release", "abort", "resume", "cancellation")})

    def cancel_request(self, *, request_id: str, epoch: int, generation: int) -> Dict[str, Any]:
        if generation != self.generation:
            raise ValueError("stale cancellation generation")
        key = (request_id, epoch)
        if key in self._cancellations:
            return self._cancellations[key]
        with self._step_fence():
            owner = self.scheduler.omp_ownership.get(request_id)
            if not owner or owner["epoch"] != epoch or not owner["owned"]:
                raise ValueError("cancellation does not own the native request epoch")
        self.engine.abort(owner["native_id"])
        owner["owned"] = False
        result = {"instance_id": self.instance_id, "request_id": request_id,
                  "epoch": epoch, "generation": generation, "cancelled": True}
        self._cancellations[key] = result
        return result

    async def cancel_continuation(self, *, transaction_id: str,
                                  ownership: Dict[str, Any]) -> Dict[str, Any]:
        async with self._transaction_lock(transaction_id):
            record = self._transactions[transaction_id]
            self._check_certificate(transaction_id, ownership, record["identity"])
            if record.get("cancellation"):
                return record["cancellation"]
            if not record.get("activation"):
                raise RuntimeError("committed cancellation must reconcile activation first")
            result = self.cancel_request(request_id=ownership["request_id"],
                                         epoch=ownership["destination_epoch"], generation=self.generation)
            result["transaction_id"] = transaction_id
            record["cancellation"] = result
            return result


    # ------------------------------------------- host-staged KV transport

    @contextlib.asynccontextmanager
    async def _cache_operation(self, timeout_s: float = 30.0):
        """Drain this engine queue before RPCs that read/donate native cache arrays.

        Only this engine briefly stops issuing; peer runtimes remain independent.
        Callers must not await while the yielded native step lock is held.
        """
        from vllm.v1.core.sched.interface import PauseState

        async with self._cache_operation_lock:
            with self._step_fence():
                previous = self.scheduler.pause_state
                self.scheduler.set_pause_state(PauseState.PAUSED_ALL)
            deadline = time.monotonic() + timeout_s
            try:
                while True:
                    with self._step_fence():
                        if not self.engine_core.batch_queue:
                            yield
                            return
                    if time.monotonic() >= deadline:
                        raise TimeoutError("native batch queue did not retire before KV RPC")
                    await asyncio.sleep(0.005)
            finally:
                with self._step_fence():
                    self.scheduler.set_pause_state(previous)
                self.engine._wake.set()

    async def kv_layout(self) -> List[Dict[str, Any]]:
        """Per-worker physical cache groups and shard indices."""
        async with self._cache_operation():
            return self.engine_core.model_executor.collective_rpc("omp_kv_layout")

    async def export_request_kv(self, *, request_id: str, transaction_id: str,
                                source_generation: int, source_epoch: int) -> Dict[str, Any]:
        """Gather only this transaction's valid retired KV, never stale physical pages.

        The engine pauses only while each chunk is gathered on device; workers
        copy it to the host and publish it on background threads while the
        engine keeps serving its other requests. The request itself stays fenced.
        """
        from .kv_transport import DEFAULT_STAGE_BYTES_PER_DEVICE, assemble_manifest

        identity = self._source_identity(request_id, source_generation, source_epoch)
        rpc = self.engine_core.model_executor.collective_rpc
        notifier = ray.get_runtime_context().current_actor
        async with self._transaction_lock(transaction_id):
            record = self._transactions[transaction_id]
            if (record.get("source") != identity or record.get("resume") or record.get("release")
                    or self._quiesced.get(request_id) != transaction_id):
                raise RuntimeError("KV export requires this source transaction's live fence")
            started = time.monotonic()
            self._kv_notices[transaction_id] = asyncio.Queue()
            addressing, layouts = None, []
            cursor, total, outstanding = 0, None, 0
            stages, gather_s, paused_s, chunks = [], 0.0, 0.0, 0
            try:
                while total is None or cursor < total or outstanding:
                    # At most two gathered chunks occupy device memory at once.
                    if (total is None or cursor < total) and outstanding < 2:
                        paused = time.monotonic()
                        async with self._cache_operation():
                            if addressing is None:
                                addressing = self.scheduler.omp_block_ids(request_id)
                            gathered = rpc("omp_export_gather", args=(
                                transaction_id, addressing, cursor,
                                DEFAULT_STAGE_BYTES_PER_DEVICE, notifier))
                        paused_s += time.monotonic() - paused
                        following = {row["next"] for row in gathered}
                        if len(following) != 1:
                            raise RuntimeError("workers disagree on the KV export chunk boundary")
                        if cursor == 0:
                            layouts = [row["layout"] for row in gathered]
                        cursor, total = following.pop(), gathered[0]["total"]
                        gather_s += max(row["gather_s"] for row in gathered)
                        outstanding += 1
                        chunks += 1
                        continue
                    stages.extend(await self._kv_chunk(transaction_id, "export", len(layouts),
                                                       chunks - outstanding))
                    outstanding -= 1
                payload = assemble_manifest(layouts, addressing, stages)
            except BaseException:
                with contextlib.suppress(Exception):
                    async with self._cache_operation():
                        rpc("omp_kv_discard", args=(transaction_id,))
                raise
            finally:
                self._kv_notices.pop(transaction_id, None)
                self._kv_early.pop(transaction_id, None)
            return {**identity, "transaction_id": transaction_id, "instance_id": self.instance_id,
                    "export_s": time.monotonic() - started, "paused_s": paused_s,
                    "chunks": chunks, "gather_s": gather_s,
                    "d2h_s": max((row["d2h_s"] for row in stages), default=0.0),
                    **payload}

    def migration_state(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "quiesced": dict(self._quiesced),
            "prepared": sorted(txn for txn, record in self._transactions.items()
                               if record["phase"] == "prepared"),
            "active_continuations": sorted(self._continuations),
            "migrated_away": dict(self._migrated),
            "pressure": self.scheduler.omp_pressure() if self.scheduler else None,
        }

    # ------------------------------------------------- reachable endpoint

    async def start_serving(
        self, *, port: int = 8000, heartbeat_s: float = 10.0
    ) -> Dict[str, Any]:
        """Binds this instance's single reachable serving endpoint."""
        from .serving_http import ServingEndpoint

        if self.state != "ready":
            raise RuntimeError(
                f"{self.instance_id} is {self.state}; refusing to advertise an endpoint"
            )
        if self._endpoint is None:
            self._endpoint = ServingEndpoint(self, host="0.0.0.0", port=port)
            info = await self._endpoint.start()
        else:
            info = {"started": False, "reason": "already_running", "port": port}
        self.endpoint_port = port
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(heartbeat_s))
        return {
            "instance_id": self.instance_id,
            "endpoint_host": self.node_ip,
            "endpoint_port": port,
            **info,
        }

    async def stop_serving(self) -> Dict[str, Any]:
        task = self._heartbeat_task
        self._heartbeat_task = None
        if task is not None:
            task.cancel()
        endpoint = self._endpoint
        self._endpoint = None
        if endpoint is not None:
            await endpoint.stop()
        return {"instance_id": self.instance_id, "endpoint_stopped": endpoint is not None}

    async def _heartbeat_loop(self, interval_s: float) -> None:
        from .registry import find_registry

        while True:
            try:
                registry = find_registry()
                if registry is not None:
                    status = self.status()
                    registry.heartbeat.remote(
                        self.instance_id,
                        self.generation,
                        state="draining" if self._draining else self.state,
                        kv_usable_blocks=int(status.get("usable_blocks", 0)),
                    )
            except Exception:  # pragma: no cover - heartbeats must never kill serving
                pass
            await asyncio.sleep(interval_s)

    # ---------------------------------------------------------- lifecycle

    async def await_idle(self, timeout_s: float = 300.0) -> Dict[str, Any]:
        """Blocks until this engine is idle, without refusing new work.

        ``drain`` is the operator lifecycle verb and latches ``_draining``;
        using it as a between-arm barrier would leave the instance permanently
        refusing admissions.  Experiment barriers use this instead.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            # Published capacity alone can be stale and says nothing about the
            # native batch queue or finished-request cleanup. Inspect all three
            # under the same lock used by the step thread and capture boundaries.
            with self._step_fence():
                error = self.engine.stats()["error"]
                if error:
                    raise RuntimeError(f"engine step loop failed: {error}")
                pool = self.scheduler.kv_cache_manager.block_pool
                pending = len(self.engine_core.batch_queue or ())
                native_work = self.scheduler.has_requests()
                idle = (
                    self._inflight == 0
                    and not self.llm_engine.has_unfinished_requests()
                    and not native_work
                    and pending == 0
                )
                snapshot = {
                    "instance_id": self.instance_id,
                    "idle": idle,
                    "inflight": self._inflight,
                    "num_running": len(self.scheduler.running),
                    "num_waiting": len(self.scheduler.waiting),
                    "free_blocks": int(pool.get_num_free_blocks()),
                    "pending_batches": pending,
                    "scheduler_has_work": native_work,
                    "draining": self._draining,
                }
            if idle or time.monotonic() >= deadline:
                return snapshot
            await asyncio.sleep(0.05)

    def resume_admission(self) -> Dict[str, Any]:
        """Clears a latched drain so the instance accepts work again."""
        was_draining = self._draining
        self._draining = False
        return {
            "instance_id": self.instance_id,
            "was_draining": was_draining,
            "draining": self._draining,
        }

    async def drain(self, timeout_s: float = 300.0) -> Dict[str, Any]:
        """Stops accepting new work and waits for in-flight requests to finish."""
        self._draining = True
        deadline = time.time() + timeout_s
        while self._inflight > 0 and time.time() < deadline:
            await asyncio.sleep(0.2)
        drained = self._inflight == 0
        self.state = "drained" if drained else "draining"
        return {"instance_id": self.instance_id, "drained": drained, "inflight": self._inflight}

    def idle_snapshot(self) -> Dict[str, Any]:
        """Evidence that the engine returned to its measured idle state."""
        capacity = self.kv_capacity()
        return {
            "instance_id": self.instance_id,
            "num_running": capacity["num_running"],
            "num_waiting": capacity["num_waiting"],
            "free_blocks": capacity["free_blocks"],
            "total_blocks": capacity["total_blocks"],
            "all_blocks_free": capacity["free_blocks"] >= capacity["total_blocks"] - 1,
            "inflight": self._inflight,
        }

    async def shutdown(self) -> Dict[str, Any]:
        snapshot = {"instance_id": self.instance_id, "served": self._served}
        driver = self.engine
        llm_engine = getattr(self, "llm_engine", None)
        self.engine = None
        self.llm_engine = None
        self.scheduler = None
        self.engine_core = None
        self.state = "stopped"
        if driver is not None:
            try:
                driver.stop()
            except Exception as exc:  # pragma: no cover - best effort
                snapshot["stop_error"] = str(exc)
        if llm_engine is not None:
            for attr in ("shutdown", "__del__"):
                fn = getattr(llm_engine, attr, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception as exc:  # pragma: no cover - best effort
                        snapshot["shutdown_error"] = str(exc)
                    break
        return snapshot



def _executor_class(vllm_config: Any) -> Any:
    from vllm.v1.executor.abstract import Executor

    return Executor.get_class(vllm_config)


def _supported_kwargs(fn: Any, **kwargs: Any) -> Dict[str, Any]:
    """Drops kwargs this build's signature does not accept.

    vLLM renamed the async-engine logging flags across the releases this fabric
    must run against; passing an unknown one is a hard TypeError at bring-up.
    """
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _vllm_version() -> str:
    try:
        import vllm

        return str(vllm.__version__)
    except Exception:  # pragma: no cover
        return "unknown"
