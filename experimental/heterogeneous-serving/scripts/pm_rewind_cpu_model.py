"""Opt-in, process-local pure-meta Gemma4 holders for terminal snapshot restore.

Call prepare_cpu_model once per TP rank, then install_prepared_models before
capture/loader entry. Seed uses the ordinary loader without a holder. These
holders contain NO initialized tensor values: materialization only allocates
storage, and terminal restore must fill every registered parameter and buffer
before inference. Never pickle holders; native restore must retain the Python
objects, custom-op registrations and closures together. No post-load or kernel
execution occurs here.
"""

from dataclasses import dataclass, field
import functools
import itertools
import sys
import threading
import types
import weakref


_LOCK = threading.RLock()
_INSTALLED = None
_REGISTRATIONS = {}


@dataclass
class PreparedCPUModel:
    model: object
    config: object
    tp_rank: int
    tp_size: int
    signature: object
    tensor_inventory: list
    device_bytes: dict
    registrations: tuple
    registry_entries: tuple
    terminal_tensor_specs: tuple
    state: str = "prepared"
    materialization: dict = field(default_factory=dict)
    terminal_restore_binding: dict = field(default_factory=dict, repr=False)
    # TorchTPU names prefix-less compiled models by a process-wide counter.
    # Each holder is built with a private counter (ordinary-process state
    # count(0) -> count(1)); consumption hands that state to the process.
    compile_counter: object = field(default=None, repr=False)
    compile_prefixes: tuple = ()

    @property
    def tensor_count(self):
        return len(self.terminal_tensor_specs)


def _check_geometry(config, tp_size):
    p = config.parallel_config
    if p.tensor_parallel_size != tp_size:
        raise RuntimeError("CPU holder TP size does not match configuration")
    for name in ("pipeline_parallel_size", "data_parallel_size",
                 "prefill_context_parallel_size", "decode_context_parallel_size"):
        if getattr(p, name, 1) != 1:
            raise RuntimeError(f"CPU holder supports TP-only geometry, not {name}")
    if getattr(p, "enable_expert_parallel", False):
        raise RuntimeError("CPU holder cannot capture expert-parallel groups")
    if config.speculative_config is not None:
        raise RuntimeError("Prepare target-only Gemma4; draft holders are not supported")


def _signature(config):
    # compute_hash deliberately ignores executor class/rank/device. Include the
    # checkpoint identity and complete HF constructor config separately.
    import json
    m = config.model_config
    return (
        config.compute_hash(),
        json.dumps(m.hf_config.to_dict(), sort_keys=True, default=str),
        tuple((name, str(getattr(m, name, None))) for name in
              ("model", "revision", "code_revision", "dtype", "quantization",
               "max_model_len", "trust_remote_code")),
        config.scheduler_config.max_num_batched_tokens,
        config.scheduler_config.max_num_seqs,
    )


def _children(obj):
    """Owned state only: never walk function globals, classes or modules."""
    if isinstance(obj, dict):
        return [(str(k), v) for k, v in obj.items()]
    if isinstance(obj, (list, tuple)):
        return [(str(i), v) for i, v in enumerate(obj)]
    if isinstance(obj, functools.partial):
        return [("args", obj.args), ("keywords", obj.keywords), ("func", obj.func)]
    if isinstance(obj, types.MethodType):
        return [("self", obj.__self__), ("func", obj.__func__)]
    if isinstance(obj, types.FunctionType):
        cells = []
        for i, cell in enumerate(obj.__closure__ or ()):
            try:
                cells.append((f"closure.{i}", cell.cell_contents))
            except ValueError:
                pass
        return cells + [("defaults", obj.__defaults__), ("kwdefaults", obj.__kwdefaults__)]
    if isinstance(obj, (type, types.ModuleType)):
        return []
    package = type(obj).__module__
    torch = sys.modules.get("torch")
    is_tensor = torch is not None and isinstance(obj, torch.Tensor)
    is_compile_wrapper = package == "torch" and type(obj).__name__ in (
        "_TorchCompileWrapper", "_TorchCompileInductorWrapper",
    )
    if (is_tensor or is_compile_wrapper
            or package.startswith(("vllm", "transformers", "torch.nn", "torch._dynamo"))):
        return list(vars(obj).items()) if hasattr(obj, "__dict__") else []
    return []


def _walk(root):
    seen = set()
    pending = [("model", root)]
    while pending:
        path, obj = pending.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        yield path, obj
        pending.extend((f"{path}.{key}", value) for key, value in _children(obj))


def _reload_infos(model):
    # initialize_model records these in a module-global WeakKeyDictionary,
    # not in the modules' __dict__. Never manufacture missing registry entries.
    from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO
    for name, module in model.named_modules():
        info = LAYERWISE_INFO.get(module)
        if info is None:
            raise RuntimeError(f"Missing constructor reload metadata: {name}")
        yield name, info


def terminal_tensors(model):
    """Match PM terminal snapshot _tensors exactly, including nonpersistent buffers."""
    out, seen = {}, set()
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if name in out or id(tensor) in seen:
            continue
        seen.add(id(tensor))
        out[name] = tensor
    return out


def terminal_tensor_signature(model):
    """Structural comparison only; never reads uninitialized tensor values."""
    return tuple(
        (name, tuple(t.shape), str(t.dtype), tuple(t.stride()),
         f"{type(t).__module__}.{type(t).__qualname__}")
        for name, t in terminal_tensors(model).items()
    )


def _state_walk(root):
    """Audit owned instance state, including arbitrary wrappers and containers.

    Unlike config rebinding this walk is read-only. Never descend into Python
    modules, classes, function globals, or weak references (not owned state).
    Unknown instance dictionaries/slots are included rather than silently
    overlooking an unregistered tensor in a user-defined container.
    """
    pending, seen = [("model", root, None, None)], set()
    while pending:
        path, obj, owner, attr = pending.pop()
        # Emit every ownership edge, even to an already-seen tensor: a host
        # scale aliased outside its lifecycle slot would stay meta after reset.
        yield path, obj, owner, attr
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if len(seen) > 100000:
            raise RuntimeError("Tensor coverage graph exceeds audited ownership bound")
        children = _children(obj)
        if isinstance(obj, dict):
            children += [(f"key.{i}", key) for i, key in enumerate(obj)]
        elif isinstance(obj, (set, frozenset)):
            children = [(str(i), value) for i, value in enumerate(obj)]
        elif isinstance(obj, types.FunctionType):
            children += list(vars(obj).items())
        elif not isinstance(obj, (
            type, types.ModuleType, types.FunctionType, types.MethodType,
            functools.partial, weakref.ReferenceType,
        )):
            if not children and hasattr(obj, "__dict__"):
                children = list(vars(obj).items())
            for cls in type(obj).__mro__:
                slots = cls.__dict__.get("__slots__", ())
                if isinstance(slots, str):
                    slots = (slots,)
                for key in slots:
                    if key.startswith("__") and not key.endswith("__"):
                        key = f"_{cls.__name__.lstrip('_')}{key}"
                    if key not in ("__dict__", "__weakref__") and hasattr(obj, key):
                        children.append((key, getattr(obj, key)))
        pending.extend((f"{path}.{key}", value, obj, key) for key, value in children)


def _postload_slots(model):
    """Only the two host-scale attributes recreated by real Attention post-load."""
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.attention.attention import (
        Attention, should_load_quant_weights,
    )
    slots = {}
    with set_current_vllm_config(model.vllm_config):
        for name, layer in model.named_modules():
            if not isinstance(layer, Attention):
                continue
            method = (
                layer.quant_config.get_quant_method(layer, prefix=layer.layer_name)
                if layer.quant_config else None
            )
            if should_load_quant_weights(method):
                raise RuntimeError(f"Host-scale post-load recreation is not guaranteed: {name}")
            for attr in ("_k_scale_cpu", "_v_scale_cpu"):
                tensor = getattr(layer, attr)
                if (not isinstance(tensor, torch.Tensor) or tensor.shape != ()
                        or tensor.dtype != torch.float32):
                    raise RuntimeError(f"Unexpected host-scale lifecycle slot: {name}.{attr}")
                if id(tensor) in slots:
                    raise RuntimeError("Host-scale lifecycle slots unexpectedly alias")
                slots[id(tensor)] = (layer, attr, f"{name}.{attr}")
    return slots


def _stage_missing_tensors(model):
    """Tensors of modality towers disabled by limit_per_prompt=0.

    vLLM's no_init_weights builds such towers under torch.device("meta") and
    hides them in an unregistered StageMissingLayer.module on the ordinary
    path too, so these stay meta: no allocation, restore or replacement.
    Only the wrapped modules' own parameters, buffers and tensor attributes
    are collected, never tensors reachable through shared config objects.
    """
    import torch
    from vllm.model_executor.models.utils import StageMissingLayer
    owned = set()
    for layer in model.modules():
        if not isinstance(layer, StageMissingLayer):
            continue
        wrapped = layer.__dict__.get("module")
        if wrapped is None:
            continue
        for module in wrapped.modules():
            for value in (*module._parameters.values(), *module._buffers.values(),
                          *vars(module).values()):
                if isinstance(value, torch.Tensor):
                    if not value.is_meta:
                        raise RuntimeError("Disabled modality tower holds a real tensor")
                    owned.add(id(value))
    return owned


def _weightless_norm_constants(model):
    """RMSNorm(has_weight=False).weight: an unregistered torch.ones constant.

    The ordinary path allocates it on the device, but every RMSNorm forward
    passes None instead of it when pass_weight/pass_weight_add are False, so
    it is left meta (any unexpected read fails) instead of being allocated
    with uninitialized values. Returns {tensor id: owning norm}.
    """
    import torch
    from vllm.model_executor.layers.layernorm import RMSNorm
    constants = {}
    for norm in model.modules():
        if not isinstance(norm, RMSNorm) or norm.has_weight:
            continue
        weight = norm.__dict__.get("weight")
        if (norm.pass_weight or norm.pass_weight_add
                or not isinstance(weight, torch.Tensor)
                or isinstance(weight, torch.nn.Parameter) or "weight" in norm._buffers):
            raise RuntimeError("Weightless RMSNorm no longer skips its constant weight")
        constants[id(weight)] = norm
    return constants


def _inventory(model):
    """Audit terminal tensors, empty placeholders, and explicit post-load slots."""
    import torch
    targets = terminal_tensors(model)
    target_names = {id(t): name for name, t in targets.items()}
    postload_slots = _postload_slots(model)
    stage_missing = _stage_missing_tensors(model)
    weightless = _weightless_norm_constants(model)
    reload_metadata = set()
    nodes = list(_state_walk(model))
    for name, info in _reload_infos(model):
        if info.loaded_weights or info.kernel_tensors or info.can_load():
            raise RuntimeError("CPU holder contains active/reloaded weight state")
        for kind, group in zip(("parameters", "buffers"), info.restore_metadata):
            for key, tensor in group.items():
                reload_metadata.add(id(tensor))
                nodes.append((f"reload_metadata.{name}.{kind}.{key}", tensor, None, None))
    if reload_metadata.intersection(target_names):
        raise RuntimeError("Reload metadata aliases live terminal-restored tensors")
    if set(postload_slots).intersection(set(target_names) | reload_metadata):
        raise RuntimeError("Post-load host-scale slots alias registered/reload tensors")
    if (stage_missing | set(weightless)).intersection(
            set(target_names) | reload_metadata | set(postload_slots)):
        raise RuntimeError("Disabled-tower/weightless-norm tensors alias live lifecycle tensors")
    nodes.extend((f"terminal.{name}", t, None, None) for name, t in targets.items())
    rows, totals, seen = [], {"cpu": 0, "meta": 0}, set()
    parameters = {id(p) for p in model.parameters()}
    empty_runtime = []
    for path, tensor, owner, attr in nodes:
        if not isinstance(tensor, torch.Tensor):
            continue
        if id(tensor) in postload_slots:
            expected_owner, expected_attr, _ = postload_slots[id(tensor)]
            if owner is not expected_owner or attr != expected_attr:
                raise RuntimeError(f"Host-scale tensor escapes its post-load slot: {path}")
        if id(tensor) in weightless and (owner is not weightless[id(tensor)] or attr != "weight"):
            raise RuntimeError(f"Weightless-norm constant escapes its owner: {path}")
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        if not tensor.is_meta:
            raise RuntimeError(f"Pure-meta holder contains a real tensor: {path}: {tensor.device}")
        if id(tensor) in reload_metadata:
            role = "reload_metadata"
        elif id(tensor) in target_names:
            role = "live"
        elif id(tensor) in postload_slots:
            role = "postload_slot"
        elif id(tensor) in stage_missing:
            role = "stage_missing"
        elif id(tensor) in weightless:
            role = "unused_constant"
        elif tensor.numel() == 0:
            role = "empty_runtime"
            empty_runtime.append(tensor)
        else:
            raise RuntimeError(f"Terminal snapshot/lifecycle does not cover tensor: {path}")
        size = tensor.numel() * tensor.element_size()
        totals["meta"] += size
        rows.append(dict(
            path=path, tensor_id=id(tensor), terminal_name=target_names.get(id(tensor)),
            device=str(tensor.device), bytes=size,
            type=f"{type(tensor).__module__}.{type(tensor).__qualname__}",
            shape=tuple(tensor.shape), stride=tuple(tensor.stride()),
            dtype=str(tensor.dtype), parameter=id(tensor) in parameters,
            role=role,
            postload_slot=(postload_slots[id(tensor)][2] if role == "postload_slot" else None),
            tp_rank=getattr(tensor, "tp_rank", None),
        ))
    return rows, totals, [*targets.values(), *empty_runtime]




def _compile_prefix_counter_cell():
    """The real counter closed over by TorchTPU's compile_prefix patch."""
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    init = TorchCompileWithNoGuardsWrapper.__init__
    if not getattr(init, "_tpu_compile_prefix_isolation_patch", False):
        raise RuntimeError("TorchTPU compile_prefix isolation patch is not active")
    cell = init.__closure__[init.__code__.co_freevars.index("counter")]
    if type(cell.cell_contents) is not itertools.count:
        raise RuntimeError("Unexpected compile_prefix counter object")
    return cell


def _construct_model(config, tp_rank, tp_size, device, register_pallas):
    if _INSTALLED is not None:
        raise RuntimeError("Cannot construct models after consumer installation")
    if (type(tp_rank) is not int or type(tp_size) is not int
            or not 0 <= tp_rank < tp_size):
        raise ValueError("Invalid TP rank")
    _check_geometry(config, tp_size)
    if config.compilation_config.static_forward_context:
        raise RuntimeError("Construction needs a fresh, model-owned configuration")
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import cpu_model_parallel_geometry
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.model_executor.layers import rotary_embedding
    from vllm_torchtpu.layers.adapter.quantization import get_tpu_quantization_config
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import set_vllm_model_wrapper_context
    config.device_config.device = torch.device(device)
    config.quant_config = get_tpu_quantization_config(config)
    dtype = torch.get_default_dtype()
    # get_rope caches modules without rank/device in the key. Isolate the real
    # per-construction cache so rank holders/reference models cannot share
    # tensors that one of them later materializes. Within-model reuse is intact.
    # Likewise give every construction the compile_prefix counter of a fresh
    # worker, so each rank model gets the ordinary single-model prefix
    # (e.g. _tpu_instance_0) instead of one shifted by earlier holders.
    # The same idempotent call TpuPlatform.check_and_update_config makes; a
    # pickled config would otherwise first activate these patches midway
    # through the first construction, after the counter must be isolated.
    from vllm_torchtpu import patch_registry
    patch_registry.apply("platform_activation", model_config=config.model_config)
    previous_rope_cache = rotary_embedding._ROPE_DICT
    counter_cell = _compile_prefix_counter_cell()
    previous_counter = counter_cell.cell_contents
    isolated_counter = itertools.count()
    try:
        counter_cell.cell_contents = isolated_counter
        rotary_embedding._ROPE_DICT = {}
        torch.set_default_dtype(config.model_config.dtype)
        with (cpu_model_parallel_geometry(tp_rank=tp_rank, tp_size=tp_size),
              torch.device(device), set_current_vllm_config(config),
              set_vllm_model_wrapper_context(mesh=None, vllm_config=config)):
            model = initialize_model(vllm_config=config, model_config=config.model_config)
            if type(model).__name__ != "Gemma4ForConditionalGeneration":
                raise RuntimeError(f"Expected real Gemma4ForConditionalGeneration, got {type(model)}")
            registrations, entries = _register_pallas(config) if register_pallas else ((), ())
    finally:
        counter_cell.cell_contents = previous_counter
        rotary_embedding._ROPE_DICT = previous_rope_cache
        torch.set_default_dtype(dtype)
    prefixes = tuple(sorted(vars(m)["_compile_prefix"] for m in model.modules()
                            if "_compile_prefix" in vars(m)))
    return model, registrations, entries, isolated_counter, prefixes


def prepare_cpu_model(config, tp_rank, tp_size):
    """Construct a real pure-meta Gemma4 holder for terminal restore only.

    config is owned by this holder; provide a separate config for each rank.
    Call only in a quiescent process. The real get_rope cache is isolated during
    each construction and restored afterward, including on exceptions.
    No CPU allocation interception, constants, weight loading or post-load.
    """
    with _LOCK:
        model, registrations, entries, counter, prefixes = _construct_model(
            config, tp_rank, tp_size, "meta", True,
        )
        rows, totals, _ = _inventory(model)
        for row in rows:
            row["preparation_rank"] = tp_rank
        return PreparedCPUModel(
            model=model, config=config, tp_rank=tp_rank, tp_size=tp_size,
            signature=_signature(config), tensor_inventory=rows,
            device_bytes=totals, registrations=registrations,
            registry_entries=entries,
            terminal_tensor_specs=terminal_tensor_signature(model),
            compile_counter=counter, compile_prefixes=prefixes,
        )


def build_cpu_reference_model(config, tp_rank, tp_size, *, register_pallas=True):
    """Ordinary CPU constructor for structural comparison, not a seed/holder.

    Runs real initialize_model with CPU default device and no factory overrides.
    This allocates the complete CPU shard, including real constructor constants.
    Per-construction RoPE cache isolation prevents sharing with meta holders.
    No post-load or weight loading is done; budget full shard-sized CPU memory.
    Compare terminal_tensor_signature and non-tensor attributes, not weight values.
    """
    with _LOCK:
        model, _, _, _, _ = _construct_model(
            config, tp_rank, tp_size, "cpu", register_pallas,
        )
        for name, tensor in terminal_tensors(model).items():
            if tensor.device.type != "cpu":
                raise RuntimeError(f"CPU reference reused non-CPU state: {name}: {tensor.device}")
        return model


def install_prepared_models(models):
    """Enable one-shot consumption. Mapping keys are real global ranks.

    TP-only geometry makes global rank == TP rank. An empty mapping still
    enables fail-closed behavior; installation cannot be silently disabled.
    Runtime consumption additionally requires the real terminal-snapshot loader;
    these holders must never enter seed/raw-checkpoint-only loading.
    """
    global _INSTALLED
    with _LOCK:
        if sys.modules.get("pm_rewind_cpu_model") is not sys.modules.get(__name__):
            raise RuntimeError("Import helper as pm_rewind_cpu_model before installing")
        if _INSTALLED is not None:
            raise RuntimeError("Prepared model consumer is already installed")
        checked = dict(models)
        configs = set()
        for rank, holder in checked.items():
            if (not isinstance(holder, PreparedCPUModel) or type(rank) is not int
                    or rank != holder.tp_rank or holder.state != "prepared"):
                raise RuntimeError("Invalid prepared rank/holder/state")
            if id(holder.config) in configs:
                raise RuntimeError("Holders must not share mutable configurations")
            configs.add(id(holder.config))
        _INSTALLED = checked


def validate_prepared_model(holder, config, tp_rank, tp_size):
    """Local, side-effect-free compatibility check; never constructs a model."""
    if (type(tp_rank) is not int or type(tp_size) is not int
            or not isinstance(holder, PreparedCPUModel)
            or holder.tp_rank != tp_rank or holder.tp_size != tp_size):
        raise RuntimeError("Requested rank/size does not match prepared holder")
    _check_geometry(config, tp_size)
    if _signature(config) != holder.signature:
        raise RuntimeError("Configuration does not match prepared holder")


def claim_prepared_cpu_model(config, *, tp_rank, tp_size):
    """Explicit local-only selector for CPU/native verification, not TPU loading.

    A locally claimed holder cannot enter TPU materialization. The production
    loader exclusively uses claim_prepared_model with real distributed rank.
    """
    with _LOCK:
        if _INSTALLED is None or tp_rank not in _INSTALLED:
            raise RuntimeError("No installed local CPU holder for requested rank")
        holder = _INSTALLED[tp_rank]
        validate_prepared_model(holder, config, tp_rank, tp_size)
        if holder.state != "prepared":
            raise RuntimeError(f"Holder cannot be claimed in state {holder.state}")
        holder.state = "claimed_cpu"
        return holder


def claim_prepared_model(config):
    """Select solely using restored real distributed state, never CLI/env rank.

    Returns None only when installation has never occurred. A failed claim or
    materialization cannot fall back to initialize_model.
    """
    with _LOCK:
        if _INSTALLED is None:
            return None
        import torch
        from vllm.distributed.parallel_state import (
            get_tp_group, get_world_group, is_cpu_model_preparation,
        )
        if is_cpu_model_preparation() or not torch.distributed.is_initialized():
            raise RuntimeError("Claim requires restored distributed runtime")
        world, tp = get_world_group(), get_tp_group()
        rank = torch.distributed.get_rank()
        holder = _INSTALLED.get(rank)
        if holder is None:
            raise RuntimeError(f"No prepared holder for real distributed rank {rank}")
        _check_geometry(config, holder.tp_size)
        if (world.rank != rank or world.world_size != holder.tp_size
                or torch.distributed.get_world_size() != holder.tp_size
                or tp.rank_in_group != holder.tp_rank
                or tp.world_size != holder.tp_size):
            raise RuntimeError("Restored group geometry does not match CPU holder")
        if holder.state != "prepared":
            raise RuntimeError(f"Holder cannot be claimed in state {holder.state}")
        validate_prepared_model(holder, config, rank, tp.world_size)
        holder.state = "claimed"
        return holder


def _bind_runtime_config(holder, fresh, target):
    """Keep layers/quant methods, bind fresh runtime state including closures."""
    from vllm.config import VllmConfig
    fresh.device_config.device = target
    old = holder.config
    old_context = old.compilation_config.static_forward_context
    new_context = fresh.compilation_config.static_forward_context
    for name, layer in old_context.items():
        if name in new_context and new_context[name] is not layer:
            raise RuntimeError(f"Fresh static_forward_context collision: {name}")
    new_context.update(old_context)
    # Quant methods and configured packed_modules_mapping belong to the
    # constructed layers, not to a second unused quant-config instance.
    fresh.quant_config = old.quant_config
    replacements = {id(old): fresh, id(old_context): new_context}
    for key, value in vars(old).items():
        if value is not None and hasattr(value, "__dict__") and hasattr(fresh, key):
            replacements[id(value)] = getattr(fresh, key)
    replacements[id(old.quant_config)] = old.quant_config
    seen = set()

    def bind(obj):
        replacement = replacements.get(id(obj), obj)
        if replacement is not obj:
            return replacement
        if id(obj) in seen:
            return obj
        seen.add(id(obj))
        if isinstance(obj, VllmConfig):
            # Nested text-model VllmConfigs carry a different HF ModelConfig.
            # Preserve that structural variant; replace runtime-owned sections.
            for key in ("parallel_config", "device_config", "compilation_config",
                        "scheduler_config", "load_config"):
                setattr(obj, key, getattr(fresh, key))
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                obj[key] = bind(value)
        elif isinstance(obj, list):
            obj[:] = [bind(value) for value in obj]
        elif isinstance(obj, tuple):
            values = tuple(bind(value) for value in obj)
            if any(a is not b for a, b in zip(values, obj)):
                replacement = type(obj)(*values) if hasattr(obj, "_fields") else values
                replacements[id(obj)] = replacement
                return replacement
        elif isinstance(obj, functools.partial):
            # partial.args is immutable. Preserve the callable and refuse
            # unrewritable config captures instead of leaving a stale backend.
            if any(bind(value) is not value for value in obj.args):
                raise RuntimeError("Immutable partial positional runtime config")
            bind(obj.keywords)
            bind(obj.func)
        elif isinstance(obj, types.FunctionType):
            for cell in obj.__closure__ or ():
                try:
                    value = cell.cell_contents
                except ValueError:
                    continue
                cell.cell_contents = bind(value)
            obj.__defaults__ = bind(obj.__defaults__)
            obj.__kwdefaults__ = bind(obj.__kwdefaults__)
        elif isinstance(obj, types.MethodType):
            if bind(obj.__self__) is not obj.__self__:
                replacement = types.MethodType(obj.__func__, bind(obj.__self__))
                replacements[id(obj)] = replacement
                return replacement
            bind(obj.__func__)
        else:
            for key, value in _children(obj):
                bound = bind(value)
                if bound is not value:
                    setattr(obj, key, bound)
        return obj

    bind(holder.model)
    for _, info in _reload_infos(holder.model):
        info.restore_device = target
        bind(info)
    for path, obj, _, _ in _state_walk(holder.model):
        if replacements.get(id(obj), obj) is not obj:
            raise RuntimeError(f"Unrebound constructor config reference: {path}")
    for name, layer in old_context.items():
        if fresh.compilation_config.static_forward_context.get(name) is not layer:
            raise RuntimeError(f"Lost CPU layer binding: {name}")
    # The deferred string executor belongs only to preparation. Runtime uses
    # the real executor class and its async-scheduling capability.
    parallel = fresh.parallel_config
    backend = parallel.distributed_executor_backend
    expected = getattr(old.parallel_config, "_pm_cpu_executor_supports_async", None)
    if expected is not None and target.type != "cpu":
        if not isinstance(backend, type) or not hasattr(backend, "supports_async_scheduling"):
            raise RuntimeError("Fresh executor class was not rebound before consumption")
        if backend.supports_async_scheduling() != expected:
            raise RuntimeError("Fresh executor async capability changed")


def materialize_prepared_model(holder, config, target_device="cpu", mesh=None):
    """Allocate terminal storage and zero-element runtime placeholders only.

    Identity-tied tensors stay tied. swap_tensors preserves the original Python
    tensor/Parameter objects, subclass and __dict__. Weakrefs, autograd aliases
    and distinct tensor objects sharing storage are rejected before allocation;
    replacing them could strand external references or change alias semantics.
    Errors poison this holder, never trigger model reconstruction.

    target_device="cpu" is the explicit local validation path: accepts an
    unclaimed or locally claimed holder, does not initialize distributed state
    or bind/execute a Pallas kernel. CPU and TPU both allocate uninitialized
    parameter AND buffer storage, with no constant copying. Terminal snapshot
    restore must fill it; raw checkpoint loading alone is not a valid consumer.
    Host-scale lifecycle slots remain the same meta objects for the original
    Attention post-load to rebuild; shape-only reload metadata also stays meta.
    A locally materialized holder is one-shot and cannot later enter TPU loading.
    """
    import torch
    with _LOCK:
        target = torch.device(target_device)
        local = target.type == "cpu"
        if holder.state not in (("prepared", "claimed_cpu") if local else ("claimed",)):
            raise RuntimeError(f"Invalid materialization state: {holder.state}")
        try:
            validate_prepared_model(holder, config, holder.tp_rank, holder.tp_size)
            if not local:
                if (target.type != "tpu" or mesh is None or _INSTALLED is None
                        or not torch.distributed.is_initialized()
                        or _INSTALLED.get(torch.distributed.get_rank()) is not holder):
                    raise RuntimeError("TPU materialization requires a real distributed claim and mesh")
                load_device = config.load_config.device
                if load_device is not None and torch.device(load_device) != target:
                    raise RuntimeError("Prepared holder cannot change the loader staging device")
            elif mesh is not None:
                raise RuntimeError("Local CPU materialization does not bind a runtime mesh")
            current_rows, _, tensors = _inventory(holder.model)
            state_specs = lambda rows: {
                r["tensor_id"]: (r["role"], r["shape"], r["stride"], r["dtype"], r["type"])
                for r in rows
            }
            if state_specs(current_rows) != state_specs(holder.tensor_inventory):
                raise RuntimeError("Tensor/lifecycle inventory changed after preparation")
            targets = terminal_tensors(holder.model)
            identities = {name: id(t) for name, t in targets.items()}
            recorded = {r["terminal_name"]: r["tensor_id"]
                        for r in holder.tensor_inventory if r["role"] == "live"}
            if (identities != recorded
                    or terminal_tensor_signature(holder.model) != holder.terminal_tensor_specs):
                raise RuntimeError("Terminal tensor identity/structure changed after preparation")
            reload_ids = tuple(
                (id(t), t.device, tuple(t.shape), tuple(t.stride()), t.dtype)
                for _, info in _reload_infos(holder.model)
                for group in info.restore_metadata for t in group.values()
            )
            postload_ids = {
                label: tensor_id
                for tensor_id, (_, _, label) in _postload_slots(holder.model).items()
            }
            if postload_ids != {
                r["postload_slot"]: r["tensor_id"]
                for r in holder.tensor_inventory if r["role"] == "postload_slot"
            }:
                raise RuntimeError("Host-scale lifecycle slot ownership changed")
            empty_specs = {
                id(t): (tuple(t.shape), tuple(t.stride()), t.dtype, type(t))
                for t in tensors if id(t) not in identities.values()
            }
            storages = {}
            for tensor in tensors:
                if getattr(tensor, "tp_rank", holder.tp_rank) != holder.tp_rank:
                    raise RuntimeError("Parameter shard rank does not match holder")
                if getattr(tensor, "tp_size", holder.tp_size) != holder.tp_size:
                    raise RuntimeError("Parameter shard size does not match holder")
                if tensor.layout != torch.strided or not tensor.is_leaf:
                    raise RuntimeError("Unsupported tensor layout/autograd state")
                if tensor._base is not None or tensor.storage_offset() != 0:
                    raise RuntimeError("Live tensor views require an alias-aware materializer")
                if weakref.getweakrefs(tensor) or tensor._use_count() != 1:
                    raise RuntimeError("Tensor weakrefs/autograd aliases prevent identity-preserving swap")
                if getattr(type(tensor), "__slots__", ()):
                    raise RuntimeError("Tensor subclass slots require an explicit materializer")
                if tensor.numel():
                    storage = (str(tensor.device), tensor.untyped_storage()._cdata)
                    other = storages.setdefault(storage, tensor)
                    if other is not tensor:
                        raise RuntimeError("Distinct storage-tied tensors require an alias-aware materializer")
            counter_cell = _compile_prefix_counter_cell()
            if (repr(counter_cell.cell_contents) != "count(0)"
                    or type(holder.compile_counter) is not itertools.count):
                raise RuntimeError("compile_prefix counter advanced before prepared-model consumption")
            _validate_pallas(holder)
            _bind_runtime_config(holder, config, target)
            if not local:
                _bind_pallas(holder, mesh)
            count = 0
            with torch.no_grad():
                for tensor in tensors:
                    data = torch.empty_strided(
                        tensor.size(), tensor.stride(), dtype=tensor.dtype,
                        device=target,
                    )
                    if (data.device.type != target.type
                            or (target.index is not None and data.device.index != target.index)):
                        raise RuntimeError("Storage factory did not allocate on the target device")
                    allocated_device = data.device
                    if type(tensor) is torch.Tensor:
                        replacement = data
                        replacement.requires_grad_(tensor.requires_grad)
                    else:
                        # Bypass subclass __init__: re-running it can overwrite
                        # loaders, pack factors, shard sizes or quant metadata.
                        replacement = torch.Tensor._make_subclass(
                            type(tensor), data, tensor.requires_grad
                        )
                    replacement.__dict__ = tensor.__dict__
                    original_type, original_dict = type(tensor), dict(tensor.__dict__)
                    torch.utils.swap_tensors(tensor, replacement)
                    if tensor.is_meta or tensor.device != allocated_device:
                        raise RuntimeError("Identity-preserving swap did not materialize target storage")
                    if type(tensor) is not original_type or any(
                        tensor.__dict__.get(k) is not v for k, v in original_dict.items()
                    ):
                        raise RuntimeError("Tensor subclass/loader metadata changed during swap")
                    count += 1
            if ({name: id(t) for name, t in terminal_tensors(holder.model).items()} != identities
                    or terminal_tensor_signature(holder.model) != holder.terminal_tensor_specs):
                raise RuntimeError("Terminal tensor identity/structure changed during materialization")
            if reload_ids != tuple(
                (id(t), t.device, tuple(t.shape), tuple(t.stride()), t.dtype)
                for _, info in _reload_infos(holder.model)
                for group in info.restore_metadata for t in group.values()
            ):
                raise RuntimeError("Shape-only reload metadata was materialized or replaced")
            if {
                label: tensor_id
                for tensor_id, (_, _, label) in _postload_slots(holder.model).items()
            } != postload_ids or any(
                not getattr(layer, attr).is_meta
                for layer, attr, _ in _postload_slots(holder.model).values()
            ):
                raise RuntimeError("Host-scale post-load slots were replaced or materialized")
            if empty_specs != {
                id(t): (tuple(t.shape), tuple(t.stride()), t.dtype, type(t))
                for t in tensors if id(t) not in identities.values()
            }:
                raise RuntimeError("Empty runtime tensor identity/structure changed")
            if any(t.numel() != 0 for t in tensors if id(t) in empty_specs):
                raise RuntimeError("Empty runtime tensor gained elements during materialization")
            # Same process-wide side effect as one ordinary construction.
            counter_cell.cell_contents = holder.compile_counter
            holder.materialization = dict(
                model_id=id(holder.model), tensors=count, device=str(target),
                rank=holder.tp_rank, parameter_identity_preserved=True,
                pallas_mesh_bound=not local, values_initialized=False,
                terminal_restore_required=True,
                terminal_tensors=len(identities), empty_runtime_tensors=len(empty_specs),
                deferred_postload_slots=len(postload_ids),
                compile_prefixes=holder.compile_prefixes,
                process_compile_counter=repr(counter_cell.cell_contents),
            )
            holder.state = "materialized_cpu" if local else "materialized"
            return holder.model
        except BaseException:
            holder.state = "failed"
            raise


def consume_prepared_model(vllm_config, *, model_config=None, prefix="", model_class=None):
    """initialize_model hook: None means opt-in was never installed."""
    if _INSTALLED is None:
        return None
    if prefix or model_class is not None or (
        model_config is not None and model_config is not vllm_config.model_config
    ):
        raise RuntimeError("Prepared holder cannot serve a different model/prefix")
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import get_vllm_model_wrapper_context
    context = get_vllm_model_wrapper_context()
    if context.vllm_config is not vllm_config:
        raise RuntimeError("Prepared model consumer is outside the real loader context")
    holder = claim_prepared_model(vllm_config)
    try:
        _require_terminal_restore(holder, vllm_config)
        return materialize_prepared_model(
            holder, vllm_config, vllm_config.device_config.device, context.mesh
        )
    except BaseException:
        holder.state = "failed"
        raise


@dataclass
class _PallasRegistration:
    name: str
    function: object
    op: object
    wrapper: object
    kwargs: dict


def _reaches(root, target):
    """Identity proof through callable closures, never through module globals."""
    pending, seen = [root], set()
    while pending:
        obj = pending.pop()
        if obj is target:
            return True
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if len(seen) > 10000:
            raise RuntimeError("Unbounded callable closure graph")
        pending.extend(value for _, value in _children(obj))
        if not isinstance(obj, (type, types.ModuleType)):
            for key in ("__wrapped__", "_fun"):
                value = getattr(obj, key, None)
                if value is not None:
                    pending.append(value)
    return False


def _validate_registration(record):
    from torch._library.custom_ops import OPDEFS
    if OPDEFS.get(record.name) is not record.op:
        raise RuntimeError(f"Pallas op registration was replaced: {record.name}")
    if record.op._init_fn is not record.wrapper:
        raise RuntimeError(f"Pallas wrapper identity changed: {record.name}")
    wrapper = record.wrapper
    if not _reaches(wrapper.jit_fn, record.function):
        raise RuntimeError(f"JAX jit does not retain the captured callable: {record.name}")
    if not any(_reaches(fn, wrapper) for fn in record.op._backend_fns.values()):
        raise RuntimeError(f"Live dispatcher does not retain JAX wrapper: {record.name}")
    if not _reaches(wrapper.exported, wrapper.jit_fn):
        raise RuntimeError(f"JAX exporter no longer retains this jit wrapper: {record.name}")
    cache_size = getattr(wrapper.jit_fn, "_cache_size", None)
    if (wrapper.output_shapes or wrapper.kernel_key_to_mlir_fingerprint
            or cache_size is None or cache_size() != 0):
        raise RuntimeError(f"Pallas callable already traced before mesh binding: {record.name}")
    if wrapper.mesh is not None or record.kwargs.get("mesh") is not None:
        raise RuntimeError("CPU preparation unexpectedly captured a native distributed mesh")


def _register_pallas(config):
    from torch_tpu._internal import pallas
    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl
    from vllm_torchtpu.layers.adapter.linear_common import (
        _get_quantized_matmul_op, _get_quantized_matmul_fp4_op,
    )
    layers = get_layers_from_vllm_config(config, AttentionLayerBase)
    original = pallas.jax_op

    def capture(name, fn=None, /, **kwargs):
        if fn is None:
            return lambda real_fn: capture(name, real_fn, **kwargs)
        # Delegate unchanged to the real API. No substitute custom op or JAX
        # function is registered, and no fake implementations are invented here.
        op = original(name, fn, **kwargs)
        wrapper = op._init_fn
        record = _PallasRegistration(name, fn, op, wrapper, dict(kwargs))
        _validate_registration(record)
        _REGISTRATIONS[id(op)] = record
        return op

    entries, registrations = [], {}
    try:
        pallas.jax_op = capture
        for name, layer in layers.items():
            impl = getattr(layer, "impl", None)
            if not isinstance(impl, PallasAttentionBackendImpl):
                raise RuntimeError(f"Unsupported attention implementation: {name}")
            if impl.kv_cache_quantized_dtype:
                raise RuntimeError("Early Pallas registration requires unquantized KV scales")
            impl.initialize_kernel(layer)
            found = False
            for key, value in impl._kernel_registry.items():
                if value is not impl.rpa_kernel:
                    continue
                found = True
                if len(key) != 16 or key[4] != id(None):
                    raise RuntimeError("Unknown attention registry mesh-key layout")
                entries.append((impl._kernel_registry, key, value))
                matches = [r for r in _REGISTRATIONS.values() if _reaches(value, r.op)]
                if len(matches) != 1:
                    raise RuntimeError("Attention callable was not captured through real pallas.jax_op")
                record = matches[0]
                if not isinstance(record.function, functools.partial) or record.function.keywords.get("mesh", object()) is not None:
                    raise RuntimeError("Attention JAX callable has no rebindable real mesh partial")
                registrations[id(record.op)] = record
            if not found:
                raise RuntimeError("Attention kernel missing from its real registry")
        for op in (_get_quantized_matmul_op(), _get_quantized_matmul_fp4_op()):
            record = _REGISTRATIONS.get(id(op))
            if record is None:
                raise RuntimeError("Pallas op predates capture; prepare in a clean process")
            registrations[id(op)] = record
    finally:
        pallas.jax_op = original
    if not entries:
        raise RuntimeError("Gemma4 preparation registered no attention kernels")
    for record in registrations.values():
        _validate_registration(record)
    # Deduplicate shared class registries without losing the actual dict/value.
    unique = {(id(registry), key): (registry, key, value)
              for registry, key, value in entries}
    return tuple(registrations.values()), tuple(unique.values())


def _validate_pallas(holder):
    for record in holder.registrations:
        _validate_registration(record)
    for registry, key, value in holder.registry_entries:
        if registry.get(key) is not value:
            raise RuntimeError("Captured attention registry entry changed before restore")


def _bind_pallas(holder, mesh):
    # Non-PCP _select_kernel_mesh returns (ctx.mesh, None, None): native
    # JaxCallable.mesh intentionally stays None, just as in the stock loader.
    # The JAX entry partial, in contrast, must receive the actual runtime mesh.
    for record in holder.registrations:
        fn = record.function
        if isinstance(fn, functools.partial) and "mesh" in fn.keywords:
            if fn.keywords["mesh"] is not None:
                raise RuntimeError("Pallas partial was already rebound")
            fn.keywords["mesh"] = mesh
            if not _reaches(record.wrapper.jit_fn, fn):
                raise RuntimeError("Rebound partial is not the live JAX callable")
    for registry, key, value in holder.registry_entries:
        runtime_key = (*key[:4], id(mesh), *key[5:])
        if runtime_key in registry and registry[runtime_key] is not value:
            raise RuntimeError("Runtime mesh registry key collides with another kernel")
        del registry[key]
        registry[runtime_key] = value


def inspect_pallas_bindings(holder):
    """Return source-backed, live callable identity evidence; executes no kernel."""
    _validate_pallas(holder)
    return [
        dict(name=r.name, op_id=id(r.op), wrapper_id=id(r.wrapper),
             callable_id=id(r.function), jit_retains_callable=True,
             dispatcher_retains_wrapper=True, exporter_retains_jit=True,
             traced=False, native_mesh=r.wrapper.mesh,
             partial_mesh=(r.function.keywords.get("mesh")
                           if isinstance(r.function, functools.partial) else None))
        for r in holder.registrations
    ]


def _snapshot_wrapper(root, name):
    """Locate the real PM snapshot closure through ordinary wrapper layers."""
    pending, seen, matches = [root], set(), []
    while pending:
        obj = pending.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if len(seen) > 10000:
            raise RuntimeError("Unbounded terminal-loader wrapper graph")
        if isinstance(obj, types.FunctionType):
            cells = {}
            for key, cell in zip(obj.__code__.co_freevars, obj.__closure__ or ()):
                try:
                    cells[key] = cell.cell_contents
                except ValueError:
                    pass
            if (obj.__qualname__.endswith(f"_patch_snapshot.<locals>.{name}")
                    and {"mode", "snap_dir"} <= cells.keys()
                    and getattr(obj, "_pm", False)):
                matches.append((obj, cells))
        pending.extend(value for _, value in _children(obj))
        wrapped = getattr(obj, "__wrapped__", None)
        if wrapped is not None:
            pending.append(wrapped)
    if len(matches) != 1:
        raise RuntimeError(f"Pure-meta holder requires the real terminal {name} wrapper")
    return matches[0]


def _require_terminal_restore(holder, config):
    """Establish the actual in-flight terminal path, not an environment promise."""
    import inspect
    from vllm.model_executor.model_loader import _LOAD_FORMAT_TO_MODEL_LOADER
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
    if _LOAD_FORMAT_TO_MODEL_LOADER.get(config.load_config.load_format) is not DefaultModelLoader:
        raise RuntimeError("Pure-meta holder requires the terminal-patched DefaultModelLoader")
    weights, wc = _snapshot_wrapper(DefaultModelLoader.load_weights, "load_weights")
    runner, rc = _snapshot_wrapper(TPUModelRunner.load_model, "load_model")
    if (wc["mode"] != "restore" or rc["mode"] != "restore"
            or wc["snap_dir"] != rc["snap_dir"]
            or weights.__globals__ is not runner.__globals__):
        raise RuntimeError("Terminal weight/runner wrappers do not describe the same restore")
    frame, in_terminal_runner = inspect.currentframe(), False
    try:
        while frame is not None:
            if (frame.f_code is runner.__code__
                    and getattr(frame.f_locals.get("self"), "vllm_config", None) is config):
                in_terminal_runner = True
                break
            frame = frame.f_back
    finally:
        del frame
    if not in_terminal_runner:
        raise RuntimeError("Pure-meta consumption is outside the active terminal runner")
    namespace = weights.__globals__
    ready, state = namespace.get("_snapshot_ready"), namespace.get("_SNAP")
    if not callable(ready) or not isinstance(state, dict) or state.get("restore_pending"):
        raise RuntimeError("Terminal restore state is absent or already armed")
    if not ready(wc["snap_dir"], holder.tp_rank):
        raise RuntimeError("Terminal snapshot is missing; raw checkpoint fallback is forbidden")
    holder.terminal_restore_binding = dict(
        weight_wrapper=weights, runner_wrapper=runner, snapshot_state=state,
        snap_dir=wc["snap_dir"], rank=holder.tp_rank,
    )


def _installed_holder_for_model(model):
    if _INSTALLED is None:
        return None
    return next((h for h in _INSTALLED.values() if h.model is model), None)


def reject_raw_checkpoint_load(model):
    """Raw loader entry guard: also closes a snapshot-readiness race."""
    holder = _installed_holder_for_model(model)
    if holder is not None:
        holder.state = "failed"
        raise RuntimeError(
            "Raw checkpoint loading cannot fill pure-meta constructor buffers; "
            "terminal snapshot restore is mandatory"
        )


def confirm_terminal_restore_pending(model):
    """Check that the real weight wrapper armed restore before original post-load."""
    holder = _installed_holder_for_model(model)
    if holder is None:
        return
    binding = holder.terminal_restore_binding
    if (holder.state != "materialized" or not binding
            or binding["snapshot_state"].get("restore_pending") is not True):
        holder.state = "failed"
        raise RuntimeError("Terminal restore was not armed by the real weight loader")
    # Not a claim that tensor values have been restored: the runner's existing
    # outer wrapper performs _restore (and its validation) after full post-load.
    holder.materialization["terminal_restore_armed"] = True
