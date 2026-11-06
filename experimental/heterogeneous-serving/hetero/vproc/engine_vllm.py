"""Unmodified vLLM + tpu-inference as a virtual process (plan C3).

``VllmEngine(spec)`` implements ``types.Engine`` inside a host kernel whose view
(C2) is already active, so ``jax.devices()``, ``jax.process_index()`` and friends
already describe this engine's hosts only. Nothing here edits tpu-inference or
vLLM; every import of them is lazy so this module imports without either.

Where things run
- Each host of the view builds ONE ``TPUWorker`` (rank = view rank) with
  ``devices=None``: ``tpu_worker.py:410`` then picks ``jax.devices()[:tp]``, which
  is the view. Its runner builds the TP mesh exactly as in a standalone engine.
- The EngineCore (scheduler) does not live here. It lives in the controller
  process and talks to the workers through ``hetero.vproc.vllm_executor``;
  each worker RPC arrives as ``step({"rpc": method, ...})``.

Guards (the engine runs in a process whose JAX runtime is already up):
- ``jax.distributed.initialize/shutdown`` raise ``EngineGuardError`` for the
  engine's lifetime (``forbid_distributed``).
- tpu_worker's TPU env writes (``tpu_worker.py:299-304, 340-370``) are only
  reachable with ``TPU_MULTIPROCESS_DP`` or PP>1; ``check_view_config`` rejects
  both, and ``frozen_tpu_env`` verifies no TPU-backend variable changed.
- ``tpu_inference/env_override.py`` writes ``XLA_FLAGS`` and ``LIBTPU_INIT_ARGS`` at
  import. In a standalone engine that runs before the backend exists; here the
  backend is up, so the host kernel must call ``preinit_environment()`` BEFORE
  ``jax.distributed.initialize`` / first ``jax.devices()``.
- ``CompilationManager`` sets ``jax_compilation_cache_dir`` at runner init
  (``compilation_manager.py:66``). JAX ignores a changed dir once its cache is
  initialized, so ``sync_compilation_cache`` resets the cache after init_device.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import os
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .types import EngineSpec, LeafSpec

# vLLM engine args of the InPlaceEngine Gemma-4 31B experiment
# (hetero/ici_worker.py:13-23). ``spec.model`` entries override these.
DEFAULT_ENGINE_ARGS: Dict[str, Any] = {
    "model": "/models/gemma-4-31b", "dtype": "bfloat16", "kv_cache_dtype": "float32",
    "additional_config": {"activation_dtype": "float32", "matmul_precision": "highest"},
    "max_model_len": 4096, "max_num_seqs": 32, "max_num_batched_tokens": 1024,
    "gpu_memory_utilization": 0.9, "enable_prefix_caching": False, "enforce_eager": False,
    "seed": 0, "trust_remote_code": True, "disable_log_stats": True,
    "limit_mm_per_prompt": {"image": 0, "audio": 0, "video": 0},
}

# TPU-backend environment: a change after the backend is up is a bug.
TPU_ENV_PREFIXES = ("TPU_", "CLOUD_TPU_", "LIBTPU", "MEGASCALE_", "JAX_", "XLA_")
# Read by tpu_inference at call time; the host kernel env must carry these values
# (tpu_worker.py:459 topology order, utils.py:128 hbm_usage_bytes on non-addressable devices;
# TPU_MULTIPROCESS_DP=0: single-process SPMD data parallelism, else tpu_platform.py:249-263 resolves it
# by writing os.environ and tpu_worker.py:315 isolates chips per DP rank).
REQUIRED_ENV = {"TPU_MULTIHOST_BACKEND": "ray", "TPU_MULTIPROCESS_DP": "0"}


class EngineGuardError(RuntimeError):
    """Engine code touched something the host kernel owns."""


# ---------------------------------------------------------------- environment


def preinit_environment(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Apply tpu-inference's import-time env writes and the required engine env.

    The host kernel calls this before ``jax.distributed.initialize`` (backend
    creation reads ``XLA_FLAGS``/``LIBTPU_INIT_ARGS`` once). It imports
    ``tpu_inference.env_override`` (the exact writes of ``env_override.py:11-24``),
    then ``setdefault``s ``REQUIRED_ENV``. Returns the TPU-prefixed variables set.

    ``VLLM_XLA_CACHE_PATH`` defaults to ``JAX_COMPILATION_CACHE_DIR``: tpu-inference
    points JAX's cache at it (``compilation_manager.py:66``, applied by
    ``sync_compilation_cache``), and vLLM's own default is pod-local. Only one
    process per engine writes the cache (``compiler.py:802``), so a pod-local cache
    leaves every other host compiling on every build (G2 run vproc-G2-26da927:
    hosts 1 and 3 had empty caches, warm-up 158 s on every switch).
    """
    target = os.environ if env is None else env
    if env is None:
        import tpu_inference.env_override  # noqa: F401  (side effect: env writes)
    for key, value in REQUIRED_ENV.items():
        target.setdefault(key, value)
    if target.get("JAX_COMPILATION_CACHE_DIR"):
        target.setdefault("VLLM_XLA_CACHE_PATH", target["JAX_COMPILATION_CACHE_DIR"])
    return {k: v for k, v in target.items() if k.startswith(TPU_ENV_PREFIXES)}


def _tpu_env() -> Dict[str, str]:
    return {k: v for k, v in os.environ.items() if k.startswith(TPU_ENV_PREFIXES)}


@contextlib.contextmanager
def frozen_tpu_env() -> Iterator[None]:
    """Fail (and restore) if engine code changed a TPU-backend variable.

    The backend is already up, so a changed ``TPU_VISIBLE_CHIPS``/bounds/port
    could only mislead a later process spawned from this one; restoring keeps the
    host kernel's block env intact and the raise makes the write visible.
    """
    before = _tpu_env()
    try:
        yield
    finally:
        after = _tpu_env()
        changed = {k: (before.get(k), after.get(k)) for k in set(before) | set(after)
                   if before.get(k) != after.get(k)}
        if changed:
            for key, (old, _new) in changed.items():
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old
            raise EngineGuardError(f"engine code changed TPU env (restored): {changed}")


@contextlib.contextmanager
def forbid_distributed() -> Iterator[List[str]]:
    """``jax.distributed.initialize/shutdown`` raise while the context is open.

    Patches the public names and ``jax._src.distributed`` (module attributes and
    ``global_state`` methods). The yielded list records attempted calls.
    """
    import jax.distributed as public
    from jax._src import distributed as private

    attempts: List[str] = []

    def blocker(name):
        def refuse(*args, **kwargs):
            attempts.append(name)
            raise EngineGuardError(f"engine code called jax.distributed.{name}; the host kernel owns the runtime")
        return refuse

    saved: List[Tuple[Any, str, Any, bool]] = []
    targets = [(public, n) for n in ("initialize", "shutdown")] + \
              [(private, n) for n in ("initialize", "shutdown")] + \
              [(private.global_state, n) for n in ("initialize", "shutdown")]
    for owner, name in targets:
        if hasattr(owner, name):
            own = name in getattr(owner, "__dict__", {})
            saved.append((owner, name, getattr(owner, name), own))
            setattr(owner, name, blocker(name))
    try:
        yield attempts
    finally:
        for owner, name, original, own in reversed(saved):
            if own:
                setattr(owner, name, original)
            else:
                delattr(owner, name)


def sync_compilation_cache() -> bool:
    """Make JAX pick up ``jax_compilation_cache_dir`` set after its cache started.

    ``CompilationManager.__init__`` updates the dir (``compilation_manager.py:66``),
    but ``compilation_cache._initialize_cache`` ran already if the host kernel
    compiled anything. Returns True when a reset was needed.
    """
    from jax._src import compilation_cache
    if compilation_cache.is_initialized():
        compilation_cache.reset_cache()
        return True
    return False


def compile_count(worker: Any = None) -> int:
    """Backend compiles so far in this kernel process. The frontend sends it through
    ``VllmEngine.step({"rpc": compile_count})`` (a light module, so unpickling it in a kernel imports
    neither Ray actor classes nor vLLM)."""
    from .resident import _COMPILES, _install_compile_listener
    _install_compile_listener()
    return len(_COMPILES)


# ------------------------------------------------------------------- the mesh


def _env_flag(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    value = env.get(name)
    if value is None or value == "":
        return default
    return value.lower() in ("1", "true", "yes")


def mesh_device_ids(spec: EngineSpec, devices: Sequence[Any],
                    env: Optional[Mapping[str, str]] = None) -> Tuple[int, ...]:
    """Device ids of the mesh tpu-inference builds for ``spec`` (flattened, (data, model) row-major)."""
    return tp_mesh_device_ids(spec.tp_size, devices, env, spec.data_groups() if spec.dp_size > 1 else None)


def engine_mesh(spec: EngineSpec, devices: Sequence[Any], axis_names: Sequence[str] = ("data", "model")):
    """The ``(dp_size, tp_size)`` mesh of ``spec`` over ``devices`` in ``mesh_device_ids`` order, as
    ``make_optimized_mesh`` builds it (Auto axes)."""
    import jax
    import numpy as np
    devices = list(devices)
    by_id = {int(d.id): d for d in devices}
    ids = mesh_device_ids(spec, devices)
    grid = np.empty((spec.dp_size, spec.tp_size), dtype=object)
    for i, device_id in enumerate(ids):
        grid[i // spec.tp_size, i % spec.tp_size] = by_id[device_id]
    return jax.sharding.Mesh(grid, tuple(axis_names), axis_types=(jax.sharding.AxisType.Auto,) * 2)


@contextlib.contextmanager
def layout_mesh_hook(spec: EngineSpec, runner_cls: Any = None, axis_names: Optional[Sequence[str]] = None) -> Iterator[Any]:
    """While open, ``TPUModelRunner._create_2d_mesh`` builds ``engine_mesh(spec, runner.devices)`` when
    ``spec.dp_size > 1``. tpu-inference's own DP2 x TP8 mesh (``make_optimized_mesh``, ``jax.make_mesh``)
    orders each row differently from the TP8 rings (``deviceless-dp-switch-libtpu0.0.43.json``); the
    decision is one S ring per data group. TP-only layouts are not hooked: their mesh stays whatever
    tpu-inference builds, which ``tp_mesh_device_ids`` replicates and ``build`` verifies. Yields the
    patched function (or None).
    """
    if spec.dp_size == 1:
        yield None
        return
    if runner_cls is None:
        from tpu_inference.runner.tpu_runner import TPUModelRunner as runner_cls
    if axis_names is None:
        from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES_2D as axis_names
    original = runner_cls.__dict__["_create_2d_mesh"]

    def create_2d_mesh(runner):
        sharding = runner.vllm_config.sharding_config
        if (sharding.model_dp_size, sharding.tp_size) != (spec.dp_size, spec.tp_size) or sharding.device_indexes:
            raise EngineGuardError(f"runner sharding dp/tp {(sharding.model_dp_size, sharding.tp_size)} "
                                   f"is not the layout {spec.layout[::-1]} the mesh hook was installed for")
        return engine_mesh(spec, runner.devices, axis_names)

    runner_cls._create_2d_mesh = create_2d_mesh
    try:
        yield create_2d_mesh
    finally:
        runner_cls._create_2d_mesh = original


def tp_mesh_device_ids(tp_size: int, devices: Sequence[Any], env: Optional[Mapping[str, str]] = None,
                       data_groups: Optional[Sequence[Sequence[int]]] = None) -> Tuple[int, ...]:
    """Device ids in mesh order. ``data_groups`` (hosts of each data group, ``EngineSpec.data_groups()``)
    selects the single-process DP mesh: ``devices[:len(groups) * tp_size]``, row g = the TP ring of
    ``tp_size`` devices of group g's hosts (``_tp_ring_ids``, i.e. an S-block ring); the ids are
    (data, model) row-major. Without it: the TP ring of the first ``tp_size`` devices."""
    if not data_groups or len(data_groups) == 1:
        return _tp_ring_ids(tp_size, devices, env)
    env = os.environ if env is None else env
    if _env_flag(env, "NEW_MODEL_DESIGN"):
        raise NotImplementedError("data parallel meshes are replicated for the 2-D mesh only")
    devices = list(devices)[:len(data_groups) * tp_size]
    if len(devices) != len(data_groups) * tp_size:
        raise ValueError(f"DP{len(data_groups)} x TP{tp_size} needs {len(data_groups) * tp_size} devices, "
                         f"the view has {len(devices)}")
    rows: List[int] = []
    for hosts in data_groups:
        group = [d for d in devices if d.process_index in hosts]
        if len(group) != tp_size:
            raise ValueError(f"data group on hosts {tuple(hosts)} has {len(group)} devices, TP{tp_size} needs {tp_size}")
        rows.extend(_tp_ring_ids(tp_size, group, env))
    return tuple(rows)


def _tp_ring_ids(tp_size: int, devices: Sequence[Any],
                 env: Optional[Mapping[str, str]] = None) -> Tuple[int, ...]:
    """Device ids in TP-mesh order, as tpu-inference builds its mesh from ``devices``.

    Replicates, for TP-only engines (DP=EP=SP=1):
      1. ``tpu_worker.py:410``: the engine's devices are ``jax.devices()[:tp]``.
      2. ``tpu_runner.py:870``: ``NEW_MODEL_DESIGN`` unset -> ``_create_2d_mesh``
         (``tpu_runner.py:963``) -> ``utils.py:240 make_optimized_mesh`` on shape
         ``(1, tp)``: devices sorted by ``.coords``; on "TPU v6 lite" with 8/4
         devices a fixed ring order; otherwise ``jax.make_mesh`` (``create_device_mesh``
         with ``allow_split_physical_axes`` on v5e), falling back to a plain reshape.
      3. ``NEW_MODEL_DESIGN=1``: ``tpu_runner.py:894`` on the 7-axis shape, sorted by
         ``(coords[::-1], core_on_chip)`` when ``TPU_MESH_SORT_BY_COORDS`` else
         ``create_device_mesh(..., allow_split_physical_axes=True)``.
    Devices need ``.coords`` (TPU); tpu-inference itself fails on CPU devices.
    """
    import jax
    import numpy as np
    from jax.experimental import mesh_utils

    env = os.environ if env is None else env
    devices = list(devices)[:tp_size]
    if len(devices) != tp_size:
        raise ValueError(f"TP{tp_size} needs {tp_size} devices, the view has {len(devices)}")
    if not all(hasattr(d, "coords") for d in devices):
        raise ValueError("mesh_device_ids needs TPU devices with .coords (tpu-inference sorts by coords)")
    if int(env.get("NUM_SLICES") or "1") != 1:
        raise NotImplementedError("NUM_SLICES > 1 (hybrid DCN mesh) is not replicated")
    fallbacks = (AssertionError, ValueError, RuntimeError)

    if _env_flag(env, "NEW_MODEL_DESIGN"):
        shape = (1, 1, 1, 1, tp_size, 1, 1)
        if _env_flag(env, "TPU_MESH_SORT_BY_COORDS"):
            ordered = sorted(devices, key=lambda x: (x.coords[::-1], x.core_on_chip))
            return tuple(int(d.id) for d in np.array(ordered).reshape(shape).flat)
        try:
            grid = mesh_utils.create_device_mesh(shape, devices, allow_split_physical_axes=True)
        except fallbacks:
            grid = np.array(devices).reshape(shape)
        return tuple(int(d.id) for d in grid.flat)

    shape = (1, tp_size)
    ordered = sorted(devices, key=lambda x: x.coords)
    if sum(n > 1 for n in shape) == 1 and ordered[0].device_kind == "TPU v6 lite":
        ring = {8: (0, 1, 2, 3, 7, 6, 5, 4), 4: (0, 1, 3, 2)}.get(len(ordered))
        if ring is not None:
            return tuple(int(ordered[i].id) for i in ring)
    try:
        mesh = jax.make_mesh(shape, ("data", "model"), (jax.sharding.AxisType.Auto,) * 2, devices=ordered)
        return tuple(int(d.id) for d in mesh.devices.flat)
    except fallbacks:
        return tuple(int(d.id) for d in np.array(ordered).reshape(shape).flat)


# --------------------------------------------------------------------- config


def engine_config(spec: EngineSpec, executor_class: Any = None) -> Any:
    """The ``VllmConfig`` of ``spec`` (hetero/ici_worker.py:11-26): same on every host
    kernel and in the controller that runs the EngineCore."""
    from vllm.engine.arg_utils import EngineArgs
    if executor_class is None:
        from .vllm_executor import VllmViewExecutor as executor_class
    args = dict(DEFAULT_ENGINE_ARGS)
    args.update(spec.model)
    args.update(tensor_parallel_size=spec.tp_size, data_parallel_size=spec.dp_size,
                distributed_executor_backend=executor_class)
    config = EngineArgs(**args).create_engine_config()
    config.scheduler_config.async_scheduling = False
    return config


def check_view_config(config: Any, spec: EngineSpec, device_count: int) -> None:
    """Reject configurations whose engine code would leave the view or write TPU env.

    Single-process SPMD data parallelism is accepted: ``TPU_MULTIPROCESS_DP`` falsy, tpu-inference
    folded ``data_parallel_size`` into its sharding strategy (``sharding.py:308-312`` resets the vLLM
    value to 1) and ``sharding.total_devices == spec.chips`` with ``model_dp_size == spec.dp_size``.
    """
    from tpu_inference import envs
    parallel = config.parallel_config
    problems = []
    if parallel.pipeline_parallel_size != 1:
        problems.append("pipeline_parallel_size != 1 (tpu_worker.py:321-375 env writes, transfer server)")
    if parallel.data_parallel_size != 1 or envs.TPU_MULTIPROCESS_DP:
        problems.append("multi-process data parallel / TPU_MULTIPROCESS_DP (tpu_worker.py:270-308 env writes); "
                        "single-process DP needs TPU_MULTIPROCESS_DP=0 (call preinit_environment)")
    if envs.TPU_MULTIHOST_BACKEND != REQUIRED_ENV["TPU_MULTIHOST_BACKEND"]:
        problems.append("TPU_MULTIHOST_BACKEND must be 'ray' (call preinit_environment)")
    sharding = config.sharding_config
    if sharding.total_devices != spec.chips or sharding.device_indexes:
        problems.append(f"sharding must be TP{spec.tp_size} x DP{spec.dp_size} only, got {sharding}")
    if (sharding.model_dp_size, sharding.total_dp_size, sharding.tp_size) != (spec.dp_size, spec.dp_size, spec.tp_size):
        problems.append(f"sharding dp/tp {(sharding.model_dp_size, sharding.total_dp_size, sharding.tp_size)} "
                        f"!= spec (dp {spec.dp_size}, tp {spec.tp_size})")
    if config.kv_transfer_config is not None:
        problems.append("KV transfer connectors are outside the view-covered surface")
    if device_count < spec.chips:
        problems.append(f"view shows {device_count} devices, TP{spec.tp_size} x DP{spec.dp_size} needs {spec.chips}")
    if problems:
        raise EngineGuardError("; ".join(problems))


# ------------------------------------------------------------ leaf bookkeeping


def _path_keys(path: Any) -> Tuple[Any, ...]:
    # same decoding as hetero/ici_resident.py:repack_state
    return tuple(getattr(key, "key", getattr(key, "idx", getattr(key, "name", None))) for key in path)


def _tp_axis(sharding: Any) -> Optional[int]:
    for axis, entry in enumerate(getattr(sharding, "spec", ())):
        if entry == "model" or (isinstance(entry, tuple) and "model" in entry):
            return axis
    return None


def _leaf_specs(state: Any, packing: Mapping[Tuple[Any, ...], Tuple[int, ...]], tp: int) -> Tuple[LeafSpec, ...]:
    """Logical LeafSpecs of ``state`` (stored layout, ``tp`` shards); catalog.py naming."""
    import jax
    specs = []
    for path, array in jax.tree_util.tree_flatten_with_path(state)[0]:
        keys = _path_keys(path)
        module = keys[:-1] if keys and keys[-1] == "value" else keys
        fused, sizes = None, ()
        if module in packing:
            sizes = tuple(int(n) for n in packing[module])
            fused = "qkv" if "qkv_proj" in module else "gate_up" if "gate_up_proj" in module else None
            if fused is None:
                raise ValueError(f"merged projection {module} is neither qkv_proj nor gate_up_proj")
            if any(n % tp for n in sizes):
                raise ValueError(f"{module}: fused sizes {sizes} do not divide TP{tp}")
        specs.append(LeafSpec(jax.tree_util.keystr(path), tuple(array.shape), str(array.dtype),
                              _tp_axis(array.sharding), fused, sizes))
    return tuple(specs)


def _state_from_leaves(abstract: Any, leaves: Mapping[str, Any]) -> Any:
    """Rebuild ``abstract`` (an ``nnx.State``) from ``path -> array``; strict match."""
    import jax
    pairs, treedef = jax.tree_util.tree_flatten_with_path(abstract)
    wanted = {jax.tree_util.keystr(path): target for path, target in pairs}
    if set(wanted) != set(leaves):
        missing, extra = sorted(set(wanted) - set(leaves)), sorted(set(leaves) - set(wanted))
        raise ValueError(f"resident leaves differ from the engine graph: missing {missing[:3]}, extra {extra[:3]}")
    for key, target in wanted.items():
        array = leaves[key]
        if tuple(array.shape) != tuple(target.shape) or array.dtype != target.dtype:
            raise ValueError(f"resident leaf {key}: {array.shape} {array.dtype} != {target.shape} {target.dtype}")
    return jax.tree_util.tree_unflatten(
        treedef, [leaves[jax.tree_util.keystr(path)] for path, _ in pairs])


def rewrap_resident(abstract: Any, shardings: Any, leaves: Mapping[str, Any]) -> Any:
    """Resident leaves as arrays carrying the engine's own shardings, without moving any buffer.

    ``ResidentStore`` hands leaves over on 1-D ``model`` meshes (in ``mesh_device_ids`` order); the runner
    wants them on its ``(data=1, model=tp)`` (or 7-axis) mesh. ``jax.device_put`` between the two can copy
    every buffer on-chip (transient extra HBM per leaf), so the shards are rebound with
    ``ici_resident.resident_views``: it passes each original ``shard.data`` to
    ``make_array_from_single_device_arrays`` after validating shape, dtype and the exact shard index of
    every device. A device that would hold a different index raises ``ValueError``, as does a target
    device that holds no shard of the leaf. ``abstract``/``shardings``: ``nnx.State`` of the engine graph
    and ``nnx.get_named_sharding`` of it (pytrees with matching paths); ``leaves`` is keyed by ``keystr``
    of those paths (``_state_from_leaves`` enforces it). Buffer identity is asserted in the CPU test
    (tests/test_vproc_surface.py), not at run time: TPU PJRT may not report buffer addresses.
    """
    from .. import ici_resident
    state = _state_from_leaves(abstract, leaves)
    try:
        return ici_resident.resident_views([state], abstract, shardings)
    except KeyError as error:
        raise ValueError(f"resident leaf has no shard on engine device {error}: the leaves are not on the "
                         "engine's devices (mesh_device_ids order/device set mismatch)") from None


# ----------------------------------------------------------------------- engine


SWITCH_SEGMENTS = ("config", "free_kv", "reshard", "rebind", "runner_build", "attach", "prime", "kv_alloc")
# WorkerBase.__init__ (vllm/v1/worker/worker_base.py:64-75) aliases these sub-configs of the VllmConfig.
_CONFIG_ALIASES = ("model_config", "cache_config", "lora_config", "load_config", "parallel_config",
                   "scheduler_config", "device_config", "speculative_config", "observability_config",
                   "kv_transfer_config", "compilation_config")


def snapshot_state(state: Any) -> Tuple[Any, Dict[str, Any]]:
    """``(abstract, shardings)`` of a runner state: the same pytree with ``ShapeDtypeStruct`` leaves, and
    ``keystr(path) -> sharding``. Taken while the arrays are alive; a parked runner keeps only these."""
    import jax
    abstract = jax.tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=a.sharding), state)
    shardings = {jax.tree_util.keystr(path): array.sharding
                 for path, array in jax.tree_util.tree_flatten_with_path(state)[0]}
    return abstract, shardings


def conform_resident(abstract: Any, shardings: Any, leaves: Mapping[str, Any]) -> Tuple[Any, int]:
    """``leaves`` as a state of ``abstract``'s structure on ``shardings`` (the engine graph's own).

    A leaf already placed as wanted is used as is; any other is ``device_put`` to the wanted sharding
    and its source deleted (a fallback: the switch pre-computes the shardings from the source leaves'
    partition specs, which the target graph is expected to agree with). Returns ``(state, conformed)``.
    """
    import jax
    from jax.sharding import NamedSharding
    state = _state_from_leaves(abstract, leaves)
    wanted = {jax.tree_util.keystr(p): s for p, s in jax.tree_util.tree_flatten_with_path(
        shardings, is_leaf=lambda x: isinstance(x, NamedSharding))[0]}
    pairs, treedef = jax.tree_util.tree_flatten_with_path(state)
    out, conformed = [], 0
    for path, array in pairs:
        want = wanted[jax.tree_util.keystr(path)]
        if array.sharding.is_equivalent_to(want, array.ndim):
            out.append(array)
        else:
            moved = jax.device_put(array, want)
            moved.block_until_ready()
            array.delete()
            out.append(moved)
            conformed += 1
    return jax.tree_util.tree_unflatten(treedef, out), conformed


def hbm_snapshot() -> Optional[Dict[int, int]]:
    """``bytes_in_use`` per local device, None when the platform has no ``memory_stats``."""
    from .measure import device_memory_stats
    stats = device_memory_stats()
    return None if stats is None else {i: s.get("bytes_in_use", 0) for i, s in stats.items()}


@contextlib.contextmanager
def keep_jit_caches() -> Iterator[List[int]]:
    """``jax.clear_caches`` does nothing while open. ``TPUModelRunner.load_model`` ends with it
    (``tpu_runner.py:~1245``) to free XLA program reservations; with a pool of layouts that would drop
    the executables of every parked layout. Yields a list that counts suppressed calls."""
    import jax
    calls: List[int] = []
    original = jax.clear_caches
    jax.clear_caches = lambda: calls.append(1)
    try:
        yield calls
    finally:
        jax.clear_caches = original


@dataclasses.dataclass
class LayoutSlot:
    """One layout's runner. Parked: no weights, no KV; graph, jitted functions and the compilation
    manager stay inside ``runner``; ``abstract``/``shardings`` say what the weights must look like."""

    spec: EngineSpec
    config: Any
    runner: Any
    leaf_specs: Tuple[LeafSpec, ...]
    abstract: Any
    shardings: Dict[str, Any]
    parked: bool = False
    kv_cache_config: Any = None
    visits: int = 1


class VllmEngine:
    """``types.Engine`` over a TPUWorker; constructed as ``VllmEngine(spec)``.

    One TPUWorker lives as long as the engine. ``_slots`` holds one ``TPUModelRunner`` per layout
    ``(tp_size, dp_size)`` visited; ``switch_layout`` moves the weights between them.
    """

    def __init__(self, spec: EngineSpec):
        if spec.kind != "vllm":
            raise ValueError(f"VllmEngine got kind {spec.kind!r}")
        self.spec = spec
        self.worker: Any = None
        self.config: Any = None
        self._specs: Tuple[LeafSpec, ...] = ()
        self._guards = contextlib.ExitStack()
        self._precision: Any = None
        self._slots: Dict[Tuple[int, int], LayoutSlot] = {}
        self._failed: Optional[str] = None

    @staticmethod
    def mesh_device_ids(spec: EngineSpec, devices: Sequence[Any]) -> Tuple[int, ...]:
        return mesh_device_ids(spec, devices)

    # build ------------------------------------------------------------------

    def build(self, resident: Optional[Dict[str, Any]]) -> Dict[str, float]:
        import sys
        import jax
        if "tpu_inference.env_override" not in sys.modules:
            # Importing it now would write XLA_FLAGS/LIBTPU_INIT_ARGS after the backend started.
            raise EngineGuardError("the host kernel must call preinit_environment() before the backend starts")
        from vllm.config import set_current_vllm_config
        from tpu_inference.worker.tpu_worker import TPUWorker

        if self.worker is not None:
            raise RuntimeError("engine already built")
        if resident is not None and self.spec.dp_size > 1:
            raise ValueError("resident leaves are TP-only (ResidentStore); a dp_size > 1 engine is built "
                             "from the checkpoint or entered with switch_layout")
        segments: Dict[str, float] = {}

        def timed(name, call):
            start = time.time()
            try:
                return call()
            finally:
                segments[name] = segments.get(name, 0.0) + time.time() - start

        self._guards.enter_context(forbid_distributed())
        try:
            with frozen_tpu_env():
                config = timed("config", lambda: engine_config(self.spec))
                devices = jax.devices()
                check_view_config(config, self.spec, len(devices))
                self.config = config
                rank = jax.process_index()
                if jax.process_count() != len(self.spec.hosts) or not 0 <= rank < len(self.spec.hosts):
                    raise EngineGuardError(
                        f"the active view is not this engine's: process {rank}/{jax.process_count()}, "
                        f"hosts {self.spec.hosts}")
                precision = config.additional_config.get("matmul_precision")
                if precision:
                    self._precision = jax.config.jax_default_matmul_precision
                    jax.config.update("jax_default_matmul_precision", precision)
                # devices=None: tpu_worker.py:410 takes jax.devices()[:chips] (the view).
                self.worker = TPUWorker(config, 0, rank, "", is_driver_worker=rank == 0)
                with layout_mesh_hook(self.spec):
                    timed("init_device", self.worker.init_device)
                sync_compilation_cache()
                self._check_runner_mesh(self.worker.model_runner, self.spec, devices)
                with set_current_vllm_config(config):
                    if resident is None:
                        timed("load", self.worker.load_model)
                    else:
                        timed("attach_resident", lambda: self._attach(resident))
                runner = self.worker.model_runner
                self._specs = self._derive_leaf_specs(runner, self.spec)
                self._register(self.spec, config, runner, self._specs)
        except BaseException:
            self._release()
            raise
        return segments

    @staticmethod
    def _check_runner_mesh(runner: Any, spec: EngineSpec, devices: Sequence[Any]) -> None:
        expected = mesh_device_ids(spec, devices)
        actual = tuple(int(d.id) for d in runner.mesh.devices.flat)
        if actual != expected:
            raise EngineGuardError(f"runner mesh {actual} != mesh_device_ids {expected}")

    def _register(self, spec: EngineSpec, config: Any, runner: Any, leaf_specs: Tuple[LeafSpec, ...]) -> LayoutSlot:
        abstract, shardings = snapshot_state(runner.state)
        slot = LayoutSlot(spec, config, runner, leaf_specs, abstract, shardings)
        self._slots[spec.layout] = slot
        return slot

    @staticmethod
    def _graph(runner: Any) -> Any:
        """Abstract NNX graph for the runner's mesh (hetero/ici_worker.py:129-141)."""
        import jax
        from flax import nnx
        from tpu_inference.models.common import model_loader
        config = runner.vllm_config
        original_dtype = config.model_config.dtype
        config.model_config.dtype = jax.numpy.bfloat16
        config.quant_config = model_loader.get_tpu_quantization_config(config)
        cls = model_loader._get_model_architecture(config.model_config.hf_config)
        try:
            with jax.set_mesh(runner.mesh):
                return nnx.eval_shape(lambda: cls(config, jax.random.key(0), runner.mesh))
        finally:
            config.model_config.dtype = original_dtype

    def _attach(self, resident: Mapping[str, Any]) -> None:
        """Attach resident leaves with zero copies (see ``rewrap_resident``).

        After the rewrap nothing in tpu-inference copies them: ``attach_resident_model`` makes
        ``model_loader.get_flax_model`` take the graph as built (``nnx.split`` + jitted closures over the
        leaves, model_loader.py:366-466; ``runner.load_model`` only device_puts for pooling models), and
        a jit sees each leaf's own sharding, which now equals the in-sharding of the runner's mesh.
        ``ici_resident.reshard_tree`` (a ``device_put``) is NOT used.
        """
        self._attach_leaves(self.worker.model_runner, resident, rewrap=True)

    def _attach_leaves(self, runner: Any, leaves: Mapping[str, Any], rewrap: bool) -> int:
        """Load ``runner``'s model from ``leaves`` through the resident attach hook (no checkpoint read).
        ``rewrap``: leaves are on a 1-D ``model`` mesh (``ResidentStore``) and are rebound without a copy;
        else they carry shardings of the runner's mesh already and ``conform_resident`` checks them.
        Returns the number of leaves that had to be moved to the graph's sharding."""
        from flax import nnx
        from .. import ici_resident
        graph = self._graph(runner)
        abstract = nnx.state(graph)
        wanted = nnx.get_named_sharding(abstract, runner.mesh)
        conformed = 0
        if rewrap:
            state = rewrap_resident(abstract, wanted, leaves)
        else:
            state, conformed = conform_resident(abstract, wanted, leaves)
        nnx.update(graph, state)
        with keep_jit_caches():
            ici_resident.attach_resident_model(runner, graph)
        return conformed

    @staticmethod
    def _derive_leaf_specs(runner: Any, spec: EngineSpec) -> Tuple[LeafSpec, ...]:
        from .. import ici_resident
        # runner.model is the loaded (or attached) NNX module
        return _leaf_specs(runner.state, ici_resident.merged_layouts(runner.model), spec.tp_size)

    # serving ----------------------------------------------------------------

    def step(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """One worker RPC: ``{"rpc": name-or-callable, "args": (...), "kwargs": {...}}``.

        The vLLM executor sends every worker call through here (execute_model,
        sample_tokens, initialize_from_config, ...). Returns ``{"result": r,
        "tokens": [[int]], "seconds": s}``; ``tokens`` are the sampled ids when
        ``r`` is a ModelRunnerOutput, else ``[]``. Runs under the active layout's VllmConfig
        (``worker.vllm_config`` follows ``switch_layout``).
        """
        from vllm.config import set_current_vllm_config
        if self.worker is None:
            raise RuntimeError("engine not built")
        self._check_usable()
        method, args, kwargs = request["rpc"], tuple(request.get("args", ())), request.get("kwargs") or {}
        start = time.time()
        with set_current_vllm_config(self.worker.vllm_config):
            if callable(method):
                result = method(self.worker, *args, **kwargs)
            else:
                result = getattr(self.worker, method)(*args, **kwargs)
        sampled = getattr(result, "sampled_token_ids", None)
        tokens = [[int(t) for t in row] for row in sampled] if sampled is not None else []
        return {"result": result, "tokens": tokens, "seconds": time.time() - start}

    def _check_usable(self) -> None:
        if self._failed:
            raise RuntimeError(f"engine {self.spec.engine_id} is unusable after a failed layout switch "
                               f"({self._failed}); stop it and rebuild")

    # layouts ----------------------------------------------------------------

    def layouts(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """Visited layouts: ``{(tp, dp): {"active", "visits", "kv_cache_config"}}``. The KV cache config
        is what the runner got in ``initialize_from_config`` (``runner.kv_cache_config``)."""
        out = {}
        for layout, slot in self._slots.items():
            kv = slot.kv_cache_config if slot.parked else getattr(slot.runner, "kv_cache_config", None)
            out[layout] = {"active": not slot.parked, "visits": slot.visits, "kv_cache_config": kv}
        return out

    @staticmethod
    def _assert_idle(runner: Any) -> None:
        if runner.requests or runner.input_batch.num_reqs:
            raise RuntimeError("switch_layout needs an idle engine: the runner has live requests")
        for name in ("_pre_async_results", "execute_model_state"):
            if getattr(runner, name, None) is not None:
                raise RuntimeError(f"switch_layout needs an idle engine: runner.{name} is pending")

    def _update_model(self, runner: Any, state: Any) -> None:
        from flax import nnx
        nnx.update(runner.model, state)

    def switch_layout(self, target: EngineSpec) -> Dict[str, Any]:
        """In-place layout switch, see ``types.Engine.switch_layout``.

        free KV of the active runner -> reshard its parameter leaves onto the target's shardings
        (``resident.reshard_leaves``; sources are deleted as the chunks complete) -> park the runner
        (no weights, no KV) -> first visit: build a new ``TPUModelRunner`` for ``engine_config(target)``
        in the same worker and load it through the resident attach hook (``built=True``; KV init and
        warm-up are then driven through the executor); revisit: rebind the leaves into the parked runner
        and reallocate its KV with the saved ``kv_cache_config`` (``built=False``). Anything that can
        be checked is checked before the first buffer is freed; a failure after that leaves the engine
        unusable (``_failed``) and the caller rebuilds the block.
        """
        import jax
        from jax.sharding import NamedSharding
        from .resident import _COMPILES, _install_compile_listener, reshard_leaves
        if self.worker is None:
            raise RuntimeError("engine not built")
        self._check_usable()
        source = self.spec
        if target != dataclasses.replace(source, tp_size=target.tp_size, dp_size=target.dp_size):
            raise ValueError(f"switch_layout target {target} differs from {source} by more than tp_size/dp_size")
        self._assert_idle(self.worker.model_runner)  # every call, a same-layout no-op included
        if _env_flag(os.environ, "NEW_MODEL_DESIGN"):
            raise NotImplementedError("switch_layout supports the 2-D (data, model) mesh only")
        _install_compile_listener()
        started, compiles0 = time.perf_counter(), len(_COMPILES)
        seg: Dict[str, float] = dict.fromkeys(SWITCH_SEGMENTS, 0.0)

        def timed(name, call):
            t = time.perf_counter()
            try:
                return call()
            finally:
                seg[name] += time.perf_counter() - t

        hbm = {"before": hbm_snapshot()}
        report: Dict[str, Any] = {"layout": tuple(target.layout), "noop": False, "conformed": 0,
                                  "suppressed_clear_caches": 0, "reshard": {}}
        if target.layout == source.layout:
            report["noop"] = True
            built = False
        else:
            built = self._switch(source, target, timed, report)
        hbm["after"] = hbm_snapshot()
        seg["total"] = time.perf_counter() - started
        report.update(built=built, segments=seg, hbm=hbm, compiles=len(_COMPILES) - compiles0,
                      compile_s=float(sum(_COMPILES[compiles0:])))
        return report

    def _switch(self, source: EngineSpec, target: EngineSpec, timed, report: Dict[str, Any]) -> bool:
        import jax
        from jax.sharding import NamedSharding
        from .resident import reshard_leaves
        with frozen_tpu_env():
            worker = self.worker
            runner = worker.model_runner
            self._assert_idle(runner)
            src = self._slots[source.layout]
            known = self._slots.get(target.layout)
            # pre-flight: nothing is freed before all of this passed
            config = None
            if known is None:
                config = timed("config", lambda: engine_config(target))
                check_view_config(config, target, len(jax.devices()))
                mesh = engine_mesh(target, list(worker.devices))
                from .resident import _pspec
                # weights replicated over "data": "model" on the leaf's tp_axis only (resident._pspec)
                targets = {leaf.path: NamedSharding(mesh, _pspec(leaf)) for leaf in src.leaf_specs}
            else:
                if not known.parked:
                    raise RuntimeError(f"layout {target.layout} is parked but marked active")
                if getattr(known.runner, "kv_cache_config", None) is None:
                    raise ValueError(f"layout {target.layout} was entered but never got a KV cache config "
                                     "(initialize_from_config did not run); it cannot be re-entered")
                targets = known.shardings
            if set(targets) != set(src.shardings):
                raise ValueError("target layout's parameter leaves differ from the source's")
            ids = lambda shardings: {int(d.id) for s in shardings.values() for d in s.mesh.devices.flat}
            if ids(targets) != ids(src.shardings):
                raise ValueError("switch_layout needs the same chips in both layouts "
                                 f"({sorted(ids(src.shardings))} vs {sorted(ids(targets))})")
            for leaf in src.leaf_specs:
                if leaf.fused and any(n % target.tp_size for n in leaf.fused_sizes):
                    raise ValueError(f"{leaf.path}: fused sizes {leaf.fused_sizes} do not divide TP{target.tp_size}")
            try:
                self._free_kv(src, runner, timed)
                leaves = {jax.tree_util.keystr(path): array
                          for path, array in jax.tree_util.tree_flatten_with_path(runner.state)[0]}
                moved, report["reshard"] = timed("reshard", lambda: reshard_leaves(
                    leaves, src.leaf_specs, targets, source.tp_size, target.tp_size))
                del leaves
                self._park(src)
                if known is not None:
                    slot = known
                    timed("rebind", lambda: self._rebind(slot, moved))
                    timed("kv_alloc", lambda: self._realloc_kv(slot.runner))
                    slot.visits += 1
                    built = False
                else:
                    slot = self._build_slot(target, config, moved, timed, report)
                    timed("prime", lambda: self._prime_reverse(slot, src))
                    built = True
                self._activate(slot, target)
            except BaseException as error:
                self._failed = f"{type(error).__name__}: {error}"
                raise
        return built

    @staticmethod
    def _prime_reverse(slot: LayoutSlot, src: LayoutSlot) -> None:
        """Compile the helper programs (fused repack, copy, device_put) of the way back by running the
        reverse reshard once leaf-chunk by chunk on the new layout's weights and discarding the output
        (the sources are kept). Its compiles belong to this first visit, so the later switch back is
        compile-free. Only the transition ``new -> source`` is primed."""
        import jax
        from .resident import reshard_leaves
        leaves = {jax.tree_util.keystr(path): array
                  for path, array in jax.tree_util.tree_flatten_with_path(slot.runner.state)[0]}
        reshard_leaves(leaves, slot.leaf_specs, src.shardings, slot.spec.tp_size, src.spec.tp_size, discard=True)

    def _free_kv(self, slot: LayoutSlot, runner: Any, timed) -> None:
        import jax

        def free():
            slot.kv_cache_config = getattr(runner, "kv_cache_config", None)
            runner.delete_kv_cache()
            jax.effects_barrier()
        timed("free_kv", free)

    @staticmethod
    def _park(slot: LayoutSlot) -> None:
        """The runner keeps its graph, jitted functions and compilation manager; its weights were
        deleted by the reshard, so its state is the abstract snapshot until the next rebind."""
        slot.parked = True
        slot.runner.state = slot.abstract
        slot.runner.state_leaves = ()

    def _rebind(self, slot: LayoutSlot, leaves: Mapping[str, Any]) -> None:
        """Leaves (on exactly the shardings the layout's executables were compiled for) become the parked
        runner's weights: model variables, ``state`` and the flat ``state_leaves`` the jitted fns take."""
        import jax
        for path, array in leaves.items():
            if not array.sharding.is_equivalent_to(slot.shardings[path], array.ndim):
                raise ValueError(f"rebind: {path} is not on the compiled sharding")
        state = _state_from_leaves(slot.abstract, leaves)
        self._update_model(slot.runner, state)
        slot.runner.state = state
        slot.runner.state_leaves = tuple(jax.tree_util.tree_leaves(state))

    @staticmethod
    def _realloc_kv(runner: Any) -> None:
        import jax
        from vllm.config import set_current_vllm_config
        with set_current_vllm_config(runner.vllm_config):
            runner.reinitialize_kv_cache()
        jax.block_until_ready(runner.kv_caches)

    def _build_slot(self, target: EngineSpec, config: Any, leaves: Mapping[str, Any], timed,
                    report: Dict[str, Any]) -> LayoutSlot:
        """Second (third, ...) ``TPUModelRunner`` in the same worker: no ``init_device``, no checkpoint."""
        import jax
        from vllm.config import set_current_vllm_config
        from tpu_inference.runner.tpu_runner import TPUModelRunner
        worker = self.worker

        def make():
            with set_current_vllm_config(config), layout_mesh_hook(target):
                return TPUModelRunner(config, worker.devices, worker.rank, True, True)
        runner = timed("runner_build", make)
        self._check_runner_mesh(runner, target, jax.devices())

        def attach():
            with set_current_vllm_config(config), keep_jit_caches() as suppressed:
                report["conformed"] = self._attach_leaves(runner, leaves, rewrap=False)
            report["suppressed_clear_caches"] = len(suppressed)
        timed("attach", attach)
        return self._register(target, config, runner, self._derive_leaf_specs(runner, target))

    def _activate(self, slot: LayoutSlot, target: EngineSpec) -> None:
        """The worker serves ``slot``: runner and every config alias ``TPUWorker`` methods read."""
        worker = self.worker
        worker.model_runner = slot.runner
        worker.vllm_config = slot.config
        for name in _CONFIG_ALIASES:
            setattr(worker, name, getattr(slot.config, name))
        slot.parked = False
        slot.spec = target
        self.spec, self.config, self._specs = target, slot.config, slot.leaf_specs

    # hand-over --------------------------------------------------------------

    def leaf_specs(self) -> Tuple[LeafSpec, ...]:
        return self._specs

    def export(self) -> Dict[str, Any]:
        """Parameter leaves in stored layout; frees KV cache, runner and executables (of every layout)."""
        import jax
        if self.worker is None:
            raise RuntimeError("engine not built")
        self._check_usable()
        if self.spec.dp_size > 1:
            raise ValueError(f"engine {self.spec.engine_id} runs data-parallel {self.spec.layout}: "
                             "ResidentStore holds TP-only leaves; switch_layout to dp_size 1 first")
        runner = self.worker.model_runner
        # hetero/ici_worker.py:150-160 (retire): no live requests, no pending async result.
        if runner.requests or runner.input_batch.num_reqs:
            raise RuntimeError("runner has live requests at export")
        if getattr(runner, "_pre_async_results", None) is not None:
            raise RuntimeError("async sampling results remain at export")
        runner.compilation_manager._flush_compilations()
        leaves = {jax.tree_util.keystr(path): array
                  for path, array in jax.tree_util.tree_flatten_with_path(runner.state)[0]}
        jax.block_until_ready(list(leaves.values()))
        runner.delete_kv_cache()
        del runner
        self._release()
        jax.clear_caches()  # one engine per process: no other engine loses executables
        return leaves

    def shutdown(self) -> None:
        if self.worker is not None:
            try:
                self.worker.model_runner.delete_kv_cache()
            except Exception:
                pass
            self._release()
            import jax
            jax.clear_caches()
        self._guards.close()

    def _release(self) -> None:
        import jax
        if self.worker is not None:
            self.worker.model_runner = None
            self.worker = None
        self._slots.clear()
        self.config = None
        if self._precision is not None:
            jax.config.update("jax_default_matmul_precision", self._precision)
            self._precision = None
        self._guards.close()
        gc.collect()
