"""TPUFlex mechanisms for vLLM-TorchTPU 0.29.0, as a vLLM general plugin (loaded in every vLLM process).

Installed into the probe venv by torchtpu_load_host.sh when TTL_PM=1. Every mechanism is off unless its
environment variable is set; with no PM_* variable, register() returns without patching anything.

- Pre-spawn gate (Decompose + Overlap), PM_GATE_FILE: each TP worker finishes its imports and constructs
  TPUWorker, then parks in WorkerWrapperBase.init_device. That is before TPUWorker.init_device, whose
  init_process_group(backend="tpu_dist") is the first call that opens the chips. The worker continues once
  the file exists. The API server and the engine core start as usual and simply wait for the workers.
  On a fresh host the gate file is the fetcher's "startup state landed" marker instead.
  Deserializing the outer AOT artifact while parked does not work: it references torch.ops.pallas.rpa_kernel_*,
  which TPUModelRunner registers only in load_model, after the chips are open.
- Terminal weight snapshot (Cache), PM_SNAP_DIR + PM_SNAP_MODE, taken at the end of TPUModelRunner.load_model,
  i.e. after process_weights_after_loading (dense weights are transposed on the device there) and the runner's
  post-load relayouts: the exact per-chip state the engine serves from.
    save     stock load_model, then each rank writes all its parameters and buffers (non-persistent ones
             included, tied tensors once) to rank<R>.bin, then rank<R>.json (commit marker).
    restore  DefaultModelLoader.load_weights becomes a no-op, the rest of load_model runs on the uninitialized
             tensors (only their shapes matter), then every tensor is overwritten from the mmap'd rank<R>.bin.
             The manifest must match the model's tensors after load_model (names, shapes, dtypes, TP size),
             else the engine fails: weights were skipped, so there is nothing to fall back to.
             If no snapshot exists at load_weights time (after waiting PM_SNAP_WAIT_S, default 0), the stock
             loader runs and nothing is restored.
  PM_SNAP_THREADS (default 1) copies with that many threads. PM_SNAP_ASYNC=1 copies the parameters in the
  background (buffers stay synchronous) and joins before the KV cache is allocated (initialize_from_config),
  or before the profile run when the KV size is not pinned. Joining later is unsafe: each host->device copy_
  stages a parameter-sized device buffer, and the KV cache leaves no HBM for it. PM_SNAP_FOREACH=1 gives each
  thread one torch._foreach_copy_ per batch of up to PM_SNAP_FOREACH_MB (default 256; 100000 = one per thread):
  same rate alone, but it holds up better next to main-thread deserialization (1.45 -> 1.64 GB/s).
- KV geometry: the host script pins --kv-cache-memory-bytes (the KV block count is part of vLLM-TorchTPU's
  compile key). This plugin logs the bytes determine_available_memory returns (PM_KV_BYTES) so the pinned
  value can be taken from a profiled boot.
- Executable bank (Cache): vLLM-TorchTPU already persists every backbone bucket as a pickled
  TpuCompilationHandle that bundles the serialized PJRT executable, and replays it in
  TpuCompilerAdaptor._replay; the outer AOT-compiled forward skips Dynamo. With the KV geometry pinned, that
  cache is a device-order-bound executable bank keyed identically across restarts. The plugin only times it:
  PM_BANK_REPLAY (per bucket: total vs. executable deserialization) and PM_AOT (outer artifact load).
  Deserializing the bank on background threads while the weights restore does not work here: the
  deserialization holds the GIL (the main thread stalled 14 s) and leaves torch.fx's global tracing flag set,
  so the main thread's next torch.compile call fails ("FX to symbolically trace a dynamo-optimized function").
  PM_AOT_PRELOAD=1 (with PM_SNAP_ASYNC) instead deserializes only the outer AOT artifact, on the main thread
  right after load_model (Pallas ops registered), while the restore threads copy; the first forward then
  takes the preloaded function. PM_AOT_SLIM=1 loads a copy of that artifact's graph without the FakeTensor
  meta of interior nodes (8.2 -> 1.3 s). PM_BANK_PRELOAD=1 then also deserializes this rank's bucket
  executables on the main thread (not threads) during the restore; each replay takes its preloaded runnable.
- Sampler bank (Cache), PM_SAMPLER_BANK=1: the helper programs vLLM-TorchTPU re-traces on every boot (sampler,
  logits, logprobs, placeholder substitution; host CPU work) are AOT-compiled with torch's aot_compile once per
  input signature and rank, saved under VLLM_CACHE_ROOT/pm_sampler_bank, and loaded on later boots; guards still
  run on every call (see _install_sampler_bank).
- Diagnostics: PM_PROFILE="module:Qual.name,..." runs each target under cProfile in whichever process calls
  it, logs the top cumulative entries (PM_PROFILE) and dumps <PM_PROFILE_DIR>/<pid>_<name>.pstats.
"""
import json
import multiprocessing.context
import os
import time

_logger = None


def _log():
    global _logger
    if _logger is None:
        from vllm.logger import init_logger
        _logger = init_logger('vllm.pm_torchtpu')
    return _logger


def register():
    import sys
    prefix = sys.modules.get('pm_rewind_ray')
    if prefix is not None and prefix.preparing_cpu():
        return  # The restored generation explicitly activates real worker/socket hooks.
    if not any(k.startswith('PM_') for k in os.environ):
        return
    _patch_worker_wrapper()
    if os.environ.get('PM_ZYGOTE_SOCK'):
        _install_zygote_client(os.environ['PM_ZYGOTE_SOCK'])
    if os.environ.get('PM_WARM') == '1':
        _install_warm_hooks()
    if os.environ.get('PM_PROFILE'):
        _install_profiles(os.environ['PM_PROFILE'], os.environ.get('PM_PROFILE_DIR', '/tmp'))
    if os.environ.get('PM_SAMPLE_DIR'):
        _start_sampler(os.environ['PM_SAMPLE_DIR'], float(os.environ.get('PM_SAMPLE_PERIOD', '0.02')))


# ---------------------------------------------------------------------------------------------- warm

def _install_warm_hooks():
    """Overlap: the API server and the engine core each load the tokenizer and the HF processor only after
    the workers are ready (scheduler / structured-output setup; chat-template detection loads AutoProcessor
    just to read its template). Both processes sit idle for ~40 s while the workers boot, so start those
    cached loads in a thread when the engine core is launched (API server) or the executor is created (engine
    core). The later calls hit the same caches."""
    import vllm.v1.engine.core_client as cc
    import vllm.v1.executor.multiproc_executor as me

    if getattr(cc.launch_core_engines, '_pm', False):
        return
    orig_launch = cc.launch_core_engines

    def launch_core_engines(vllm_config, *args, **kwargs):
        _warm_in_thread(vllm_config, 'api_server')
        return orig_launch(vllm_config, *args, **kwargs)

    launch_core_engines._pm = True
    cc.launch_core_engines = launch_core_engines
    executors = [me.MultiprocExecutor]
    if os.environ.get('TPU_MULTIHOST_BACKEND') == 'ray':  # one engine across hosts: the Ray executors
        from vllm_torchtpu.executors import ray_distributed_executor as rv1
        from vllm_torchtpu.executors import ray_distributed_executor_v2 as rv2
        executors += [rv1.RayDistributedExecutor, rv2.RayDistributedExecutorV2]
    for cls in executors:
        orig_init = cls._init_executor

        def _init_executor(self, _orig=orig_init):
            _warm_in_thread(self.vllm_config, 'engine_core')
            return _orig(self)

        _init_executor._pm = True
        cls._init_executor = _init_executor


def _warm_in_thread(vllm_config, where):
    import threading

    def run():
        t0 = time.perf_counter()
        try:
            from vllm.multimodal import MULTIMODAL_REGISTRY
            from vllm.tokenizers import cached_tokenizer_from_config
            mc = vllm_config.model_config
            tok = cached_tokenizer_from_config(mc)
            if where == 'api_server':
                from vllm.renderers.hf import resolve_chat_template_content_format
                resolve_chat_template_content_format(None, None, 'auto', tok, model_config=mc)
            MULTIMODAL_REGISTRY.supports_multimodal_inputs(mc)
            _log().info('PM_WARM %s took=%.2fs', where, time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001  (a failed warm-up only means the later call pays)
            _log().warning('PM_WARM %s failed after %.2fs: %r', where, time.perf_counter() - t0, e)

    threading.Thread(target=run, name='pm-warm', daemon=True).start()


# -------------------------------------------------------------------------------------------- zygote

class ZygoteProcess(multiprocessing.context.ForkServerProcess):
    """A forkserver process that carries its parent's environment. A forkserver child inherits the
    zygote's environment, but vLLM-TorchTPU sets each worker's chip binding (RANK, LOCAL_RANK, ...) in the
    parent's os.environ right before starting it. torch_tpu reads those lazily, so applying the parent's
    environment first thing in the child is enough."""

    def start(self):
        self._pm_env = dict(os.environ)
        super().start()

    def run(self):
        os.environ.clear()
        os.environ.update(self._pm_env)
        super().run()


class ZygoteContext(multiprocessing.context.ForkServerContext):
    Process = ZygoteProcess


def _install_zygote_client(sock):
    """Point vLLM's process creation (engine core, TP workers) at the pm_zygote.py fork server."""
    import multiprocessing.forkserver as fsrv
    from multiprocessing import resource_tracker

    import vllm.utils.system_utils as su
    import vllm.v1.engine.utils as eu
    import vllm.v1.executor.multiproc_executor as me

    srv = fsrv._forkserver
    srv._forkserver_address = sock  # children of the zygote inherit it too
    if srv._forkserver_alive_fd is None:  # any fd: children only pass it on
        srv._forkserver_alive_fd = os.open(os.devnull, os.O_RDONLY)
    srv.ensure_running = resource_tracker.ensure_running  # never start a private forkserver
    ctx = ZygoteContext()
    for mod in (su, eu, me):
        mod.get_mp_context = lambda: ctx
    _log().info('PM_ZYGOTE client pid=%d sock=%s', os.getpid(), sock)


# ---------------------------------------------------------------------------------------------- gate

def _patch_worker_wrapper():
    from vllm.v1.worker import worker_base as wb

    orig = wb.WorkerWrapperBase.init_device
    if getattr(orig, '_pm', False):
        return

    def init_device(self):
        log = _log()
        rank = getattr(self, 'global_rank', getattr(self, 'rpc_rank', '?'))
        if os.environ.get('PM_STACK_DUMP_S') or os.environ.get('PM_STACK_DUMP_SIG'):  # debug: per-rank Python stacks
            import faulthandler
            import signal
            path = os.path.join(os.environ.get('PM_STACK_DUMP_DIR', '/tmp'), f'pm_stacks_rank{rank}_pid{os.getpid()}.txt')
            fh = open(path + '.tmp', 'w')
            _SNAP['stack_fh'] = fh  # keep it open
            if os.environ.get('PM_STACK_DUMP_SIG'):  # on SIGUSR2 (host script: TTL_STACK_SIG_S); never cancelled
                faulthandler.register(signal.SIGUSR2, file=fh, all_threads=True)
            else:
                faulthandler.dump_traceback_later(float(os.environ['PM_STACK_DUMP_S']), repeat=True, file=fh)
            os.rename(path + '.tmp', path)  # the host script signals only pids whose file exists
        if os.environ.get('PM_NO_STAGE_TIMING') == '1':  # experiment: vLLM-TorchTPU's device stage timings off
            from vllm_torchtpu.runner import stage_timing as _st
            _st.RunnerStageTiming.stage = _st.RunnerStageTiming.host_stage  # CPU timings only; no TpuEvents/collector
        gate = os.environ.get('PM_GATE_FILE')
        if gate:
            log.info('PM_PARKED rank=%s pid=%d wall=%.3f', rank, os.getpid(), time.time())
            # per-host marker for the host script (Ray workers' logs do not reach the driver's log in time)
            open(f'{gate}.parked.{rank}', 'w').close()
            while not os.path.exists(gate):
                time.sleep(0.01)
            log.info('PM_GATE_OPEN rank=%s wall=%.3f', rank, time.time())
        _patch_worker_class(type(self.worker))
        if os.environ.get('PM_SAMPLER_BANK') == '1':
            _install_sampler_bank(rank, self.worker)  # before the runner exists: it binds sample_from_logits in __init__
        t0 = time.perf_counter()
        orig(self)
        log.info('PM_INIT_DEVICE rank=%s took=%.2fs wall=%.3f', rank, time.perf_counter() - t0, time.time())
        if os.environ.get('PM_BANK_PRELOAD') == '1':
            _install_bank_preload()  # before _patch_bank, whose timing wrapper then wraps it
        _patch_bank(rank)
        if os.environ.get('PM_AOT_SLIM') == '1':
            _patch_aot_slim(rank, self.worker.vllm_config)
        if os.environ.get('PM_SNAP_DIR') and os.environ.get('PM_SNAP_MODE') in ('save', 'restore'):
            _patch_snapshot()

    init_device._pm = True
    wb.WorkerWrapperBase.init_device = init_device


def _patch_worker_class(cls):
    if getattr(cls.determine_available_memory, '_pm', False):
        return
    orig_dam = cls.determine_available_memory
    orig_init = cls.initialize_from_config

    # The background restore (PM_SNAP_ASYNC) must finish before the first forward and before the KV cache is
    # allocated: every host->device copy_ stages a parameter-sized device buffer, and the KV cache leaves no HBM
    # for that (overlapping the restore with the AOT load OOMed: 110 MB requested, 37 MB free).
    def determine_available_memory(self):
        if not self.cache_config.kv_cache_memory_bytes:  # a profile run executes the model
            _join_async('determine_available_memory')
        t0 = time.perf_counter()
        n = orig_dam(self)
        _log().info('PM_KV_BYTES rank=%s bytes=%d took=%.2fs', getattr(self, 'rank', '?'), n,
                    time.perf_counter() - t0)
        return n

    def initialize_from_config(self, *args, **kwargs):
        _join_async('initialize_from_config')
        return orig_init(self, *args, **kwargs)

    for f in (determine_available_memory, initialize_from_config):
        f._pm = True
    cls.determine_available_memory = determine_available_memory
    cls.initialize_from_config = initialize_from_config


# ---------------------------------------------------------------------------------------------- bank

_BANK = {'deserialize_s': 0.0}


def _patch_bank(rank):
    """Time every bank replay and the outer AOT load."""
    from vllm_torchtpu.compilation import tpu_compiler as tc
    if getattr(tc.TpuCompilerAdaptor._replay, '_pm', False):
        return
    log = _log()
    orig_deserialize = tc._deserialize_entry

    def deserialize_entry(entry):
        t0 = time.perf_counter()
        try:
            return orig_deserialize(entry)
        finally:
            _BANK['deserialize_s'] += time.perf_counter() - t0

    tc._deserialize_entry = deserialize_entry
    orig_replay = tc.TpuCompilerAdaptor._replay

    def replay(self, path, graph, was_wrapped):
        t0 = time.perf_counter()
        d0 = _BANK['deserialize_s']
        out = orig_replay(self, path, graph, was_wrapped)
        log.info('PM_BANK_REPLAY rank=%s file=%s total=%.3fs deserialize=%.3fs', rank, os.path.basename(path),
                 time.perf_counter() - t0, _BANK['deserialize_s'] - d0)
        return out

    replay._pm = True
    tc.TpuCompilerAdaptor._replay = replay
    _patch_aot_timing(rank)


def _patch_aot_timing(rank):
    """Split the warm AOT load: outer artifact deserialization vs. the whole load (incl. backend + replays)."""
    import torch
    from vllm.compilation import decorators as dec
    log = _log()
    orig_load = torch.compiler.load_compiled_function

    def load_compiled_function(f, *args, **kwargs):
        pre = _AOT_PRE.pop(getattr(f, 'name', None), None)
        if pre is not None:
            log.info('PM_AOT rank=%s load_compiled_function preloaded', rank)
            return pre
        t0 = time.perf_counter()
        out = orig_load(f, *args, **kwargs)
        log.info('PM_AOT rank=%s load_compiled_function took=%.3fs', rank, time.perf_counter() - t0)
        return out

    torch.compiler.load_compiled_function = load_compiled_function
    orig_try = dec._try_load_aot_compiled_fn

    def try_load(model, path):
        t0 = time.perf_counter()
        out = orig_try(model, path)
        log.info('PM_AOT rank=%s try_load_aot hit=%s took=%.3fs', rank, out is not None, time.perf_counter() - t0)
        return out

    dec._try_load_aot_compiled_fn = try_load


_AOT_PRE = {}  # artifact path -> outer AOT function deserialized ahead of the first forward


def _preload_aot(model, rank):
    """PM_AOT_PRELOAD: deserialize the outer AOT artifact (FakeTensor rebuilding, ~7 s) right after
    load_model, while the background restore streams the weights, instead of on the first forward after the
    restore join. The artifact's key cannot be computed here (vLLM-TorchTPU folds the compiler hash, which
    includes the KV block count set later by initialize_from_config, into it), so this takes this rank's
    only (else newest) artifact in the cache; the first forward uses it only if it computes that same path,
    otherwise it is a plain miss and loads as usual."""
    import glob
    import torch
    from vllm import envs
    from vllm.config import set_current_vllm_config
    if not envs.VLLM_USE_AOT_COMPILE or envs.VLLM_DISABLE_COMPILE_CACHE:
        return
    mods = [m for m in model.modules()
            if not getattr(m, 'do_not_compile', True) and hasattr(m, 'was_aot_compile_fn_loaded_from_disk')]
    if len(mods) != 1:
        _log().info('PM_AOT_PRELOAD rank=%s skipped: %d compiled modules', rank, len(mods))
        return
    m, t0 = mods[0], time.perf_counter()
    pc = m.vllm_config.parallel_config
    paths = sorted(glob.glob(os.path.join(envs.VLLM_CACHE_ROOT, 'torch_compile_cache', 'torch_aot_compile', '*',
                                          f'rank_{pc.rank}_{pc.data_parallel_index}', 'model')),
                   key=os.path.getmtime)
    if not paths:
        _log().info('PM_AOT_PRELOAD rank=%s no artifact', rank)
        return
    with set_current_vllm_config(m.vllm_config), open(paths[-1], 'rb') as f:
        _AOT_PRE[paths[-1]] = torch.compiler.load_compiled_function(f, f_globals=m.forward.__globals__)
    _log().info('PM_AOT_PRELOAD rank=%s %s candidates=%d took=%.3fs', rank, paths[-1], len(paths),
                time.perf_counter() - t0)


# ------------------------------------------------------------------------------- slim AOT artifact

def _patch_aot_slim(rank, vllm_config):
    """PM_AOT_SLIM (Cache): the outer AOT artifact's graph module stores a FakeTensor in the meta of every node of
    the whole-model graph, and rebuilding them is 98% of its ~7 s load (GraphPickler -> meta_tensor). The warm path
    reads that meta only for placeholders, the output's arguments and the tensors whose .size() is decomposed
    (vllm split_graph / PiecewiseCompileInterpreter / bank signature checks), and with no splitting ops (TPU) there
    are no other subgraph boundaries. So a slim copy without tensor-valued meta on all other nodes is kept next to
    the bank, keyed by the hash of the stock bytes, and loaded instead. It is written the first time the stock
    graph is serialized (cold boot) or deserialized (warm boot) with PM_AOT_SLIM=1; the stock artifact is untouched."""
    import hashlib
    from vllm import envs
    from vllm.compilation import caching
    cls = caching.VllmSerializableFunction
    if getattr(cls.deserialize_graph_module, '_pm', False):
        return
    orig_load, orig_dump = cls.deserialize_graph_module.__func__, cls.serialize_graph_module.__func__
    slim_dir = os.path.join(envs.VLLM_CACHE_ROOT, 'pm_aot_slim')

    def slim_path(data):
        if vllm_config.compilation_config.splitting_ops:
            return None  # subgraph boundaries would need their meta too
        return os.path.join(slim_dir, hashlib.sha256(data).hexdigest()[:32] + '.gm')

    def write_slim(klass, gm, path):
        t0 = time.perf_counter()
        stripped = _strip_meta(gm)
        try:
            slim = orig_dump(klass, gm)
        finally:
            for node, meta in stripped.items():
                node.meta = meta
        os.makedirs(slim_dir, exist_ok=True)
        with open(path + '.tmp', 'wb') as f:
            f.write(slim)
        os.replace(path + '.tmp', path)
        _log().info('PM_AOT_SLIM rank=%s wrote %s: %d of %d nodes stripped, %d bytes, took=%.2fs', rank, path,
                    len(stripped), len(gm.graph.nodes), len(slim), time.perf_counter() - t0)

    def deserialize_graph_module(klass, data, fake_mode):
        path = slim_path(data)
        if path and os.path.exists(path):
            t0 = time.perf_counter()
            with open(path, 'rb') as f:
                gm = orig_load(klass, f.read(), fake_mode)
            _log().info('PM_AOT_SLIM rank=%s loaded %s took=%.2fs', rank, os.path.basename(path), time.perf_counter() - t0)
            return gm
        gm = orig_load(klass, data, fake_mode)
        if path:
            write_slim(klass, gm, path)
        return gm

    def serialize_graph_module(klass, gm):
        data = orig_dump(klass, gm)
        path = slim_path(data)
        if path and not os.path.exists(path):
            write_slim(klass, gm, path)
        return data

    deserialize_graph_module._pm = True
    cls.deserialize_graph_module = classmethod(deserialize_graph_module)
    cls.serialize_graph_module = classmethod(serialize_graph_module)


def _strip_meta(gm):
    """Drop tensor-valued meta from the nodes the warm path never reads; returns {node: original meta}."""
    import torch
    from torch.utils import _pytree as pytree
    keep = set()
    for n in gm.graph.nodes:
        if n.op in ('placeholder', 'output'):
            keep.add(n)
        if n.op == 'output':
            keep.update(a for a in pytree.tree_leaves(n.args) if isinstance(a, torch.fx.Node))
        if n.op == 'call_method' and n.target == 'size' and len(n.args) == 1 and 'dim' not in n.kwargs:
            keep.add(n.args[0])
    kinds = (torch.Tensor, torch.SymInt, torch.SymFloat, torch.SymBool)
    stripped = {}
    for n in gm.graph.nodes:
        if n in keep:
            continue
        drop = {k for k, v in n.meta.items() if any(isinstance(x, kinds) for x in pytree.tree_leaves(v))}
        if drop:
            stripped[n] = n.meta
            n.meta = {k: v for k, v in n.meta.items() if k not in drop}
    return stripped


# ------------------------------------------------------------------------------------- bank preload

_BANK_PRE = {}  # bank artifact path -> (TpuCompilationHandle, runnable), deserialized during the weight restore


def _preload_bank(vllm_config, rank):
    """PM_BANK_PRELOAD (Overlap): deserialize this rank's bank executables on the main thread right after
    load_model, while the restore copies, instead of one by one in the first forward after the KV allocation.
    torch's deserialize_bundled_cache_entry makes its own clean TracingContext when there is none (its precompile
    use case). Only if the bank has exactly one key dir for this rank; each replay still checks the handle against
    its graph (signature, output wrapping) and replays from disk if anything differs."""
    import glob
    import pickle
    from vllm import envs
    from vllm_torchtpu.compilation import tpu_compiler as tc
    pc = vllm_config.parallel_config
    paths = glob.glob(os.path.join(envs.VLLM_CACHE_ROOT, 'torch_compile_cache', '*',
                                   f'rank_{pc.rank}_{pc.data_parallel_index}', '*', 'artifact_compile_range_*'))
    keys = {p.split(os.sep)[-4] for p in paths}
    if len(keys) != 1:
        _log().info('PM_BANK_PRELOAD rank=%s skipped: %d key dirs', rank, len(keys))
        return
    t0 = time.perf_counter()
    for p in sorted(paths):
        with open(p, 'rb') as f:
            payload = pickle.load(f)
        if isinstance(payload, tc.TpuCompilationHandle) and payload.entry is not None:
            _BANK_PRE[os.path.normpath(p)] = (payload, tc._deserialize_entry(payload.entry))
    _log().info('PM_BANK_PRELOAD rank=%s key=%s artifacts=%d took=%.3fs', rank, keys.pop(), len(_BANK_PRE),
                time.perf_counter() - t0)


def _install_bank_preload():
    from vllm_torchtpu.compilation import shape_variants
    from vllm_torchtpu.compilation import tpu_compiler as tc
    orig = tc.TpuCompilerAdaptor._replay

    def replay(self, path, graph, was_wrapped):
        pre = _BANK_PRE.pop(os.path.normpath(path), None)
        if pre is not None:
            payload, fn = pre
            if payload.signature == shape_variants.graph_signature(graph) and payload.was_wrapped == was_wrapped:
                return fn
            _log().warning('PM_BANK_PRELOAD %s does not match its graph; replaying from disk', path)
        return orig(self, path, graph, was_wrapped)

    tc.TpuCompilerAdaptor._replay = replay


# ------------------------------------------------------------------------------------ sampler bank

_SB = {'window': False, 'salt': None, 'hit': 0, 'built': 0, 'failed': 0, 'stock': 0, 'load_s': 0.0, 'build_s': 0.0,
       'by_fn': {}, 'missed': set()}
_SB_FORMAT = 2  # in every key: 1 kept Dynamo's global-state guard


def _install_sampler_bank(rank, worker):
    """PM_SAMPLER_BANK (Cache): vLLM-TorchTPU's helper programs (sampler, logits, logprobs, placeholder substitution:
    the @torch.compile(backend="tpu", dynamic=False) functions of tpu_runner) are re-traced on every boot, ~7 s of
    host CPU per rank (Dynamo 34%, aot_autograd 43%, FX->MLIR 10%, XLA cache lookup 9%; TPU execution 2%). The
    backbone already avoids this with vLLM's outer AOT artifact plus the executable bank. This applies the same to
    the helpers: during the warm-up (initialize_from_config, which precompiles the placeholder substitution, and
    compile_or_warm_up_model), each call with a new input signature is compiled with torch's own aot_compile
    (Dynamo graph + guards + the bundled TpuBackend executable) and saved per rank under
    VLLM_CACHE_ROOT/pm_sampler_bank; later boots load it instead of tracing. Every call still runs the artifact's
    guards and falls back to the stock compiled function if they fail; outside the warm-up only loaded signatures
    take the artifact. The key covers the function's code, the signature, vLLM's config hash and vLLM-TorchTPU's
    compiler hash (versions, compile env, its runtime sources)."""
    import functools
    import hashlib
    from vllm_torchtpu.runner import tpu_runner as tr
    root = os.path.join(os.environ.get('VLLM_CACHE_ROOT', '/tmp'), 'pm_sampler_bank', f'rank_{rank}')

    def banked(name, cf):
        orig = cf._torchdynamo_orig_callable
        code = orig.__code__
        loaded = {}

        @functools.wraps(cf)
        def call(*args, **kwargs):
            sig = _sb_signature(args, kwargs)
            ent = loaded.get(sig)
            if ent is None and (_SB['window'] or sig not in loaded):
                # Outside the warm-up: serving signatures (e.g. the first request's sampler call, which the warm-up
                # never makes) are loaded from the bank if a priming boot saved them; PM_SAMPLER_BANK_SERVE_BUILD=1
                # (priming boot) builds and saves them. Otherwise the stock path compiles at runtime, as stock does.
                build = _SB['window'] or os.environ.get('PM_SAMPLER_BANK_SERVE_BUILD') == '1'
                ent = loaded[sig] = _sb_load_or_build(root, name, code, sig, cf, orig, args, kwargs, build=build)
            if ent and ent.guard_check(*args, **kwargs):
                return ent.fn(*args, **kwargs)
            if _SB['window']:
                _sb_count(name, 'stock')
            if (name, sig) not in _SB['missed']:  # once per signature: a stock compile (while serving: at runtime)
                _SB['missed'].add((name, sig))
                _log().warning('PM_SAMPLER_BANK rank=%s %s: stock path%s (%s) %s', rank, name,
                               '' if _SB['window'] else ' while serving',
                               _sb_guard_reason(ent, args, kwargs) if ent else 'not banked', sig[:300])
            return cf(*args, **kwargs)

        call._pm_banked = True
        return call

    n = 0
    for owner in (tr.TPUModelRunner, tr):
        for name, cf in list(vars(owner).items()):
            if (callable(cf) and hasattr(cf, 'aot_compile') and hasattr(cf, '_torchdynamo_orig_callable')
                    and not getattr(cf, '_pm_banked', False)):
                setattr(owner, name, banked(name, cf))
                n += 1
    wcls = type(worker)
    vllm_config = worker.vllm_config

    def windowed(meth, last):
        def wrapper(self, *a, **k):
            if _SB['salt'] is None:
                from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash
                _SB['salt'] = hashlib.sha256((vllm_config.compute_hash()
                                              + compute_tpu_compilation_hash(vllm_config)).encode()).hexdigest()[:16]
            _SB['window'] = True
            try:
                return meth(self, *a, **k)
            finally:
                _SB['window'] = False
                if last:
                    _log().info('PM_SAMPLER_BANK rank=%s loaded=%d (%.2fs) built=%d (%.2fs) stock=%d salt=%s %s',
                                rank, _SB['hit'], _SB['load_s'], _SB['built'], _SB['build_s'], _SB['stock'],
                                _SB['salt'], ' '.join(f'{k}={v}' for k, v in sorted(_SB['by_fn'].items())))
                    if os.environ.get('PM_SB_PRESEED') == '1':  # experiment: seed the TPU sampling generator here,
                        import torch  # in lockstep after the warm-up, not inside the first serving step
                        t0 = time.perf_counter()
                        self.model_runner._get_sampling_generator()
                        torch.accelerator.synchronize()
                        _log().info('PM_SB_PRESEED rank=%s took=%.3fs', rank, time.perf_counter() - t0)
        return wrapper

    wcls.initialize_from_config = windowed(wcls.initialize_from_config, False)
    wcls.compile_or_warm_up_model = windowed(wcls.compile_or_warm_up_model, True)
    _log().info('PM_SAMPLER_BANK rank=%s wrapped %d tpu-compiled functions of tpu_runner', rank, n)


def _sb_count(name, kind, dt=0.0):
    _SB[kind] += 1
    if kind in ('hit', 'built'):
        _SB['load_s' if kind == 'hit' else 'build_s'] += dt
    _SB['by_fn'][f'{name}:{kind}'] = _SB['by_fn'].get(f'{name}:{kind}', 0) + 1


def _sb_guard_reason(ent, args, kwargs):
    try:
        info = ent._artifacts.guard_manager.check_verbose(ent.prepare_f_locals(*args, **kwargs))
        return 'guard failed: ' + '; '.join(map(str, info.verbose_code_parts))[:600]
    except Exception as e:  # noqa: BLE001
        return f'guard failed ({e!r})'


def _sb_signature(args, kwargs):
    """Call signature: the artifact's key and its lookup. Tensors by shape, dtype, device type, strides and
    requires_grad; scalars by value; other objects by type. The default dtype stands in for Dynamo's global-state
    guard, which the artifacts drop (see _sb_load_or_build)."""
    import torch

    def one(v):
        if isinstance(v, torch.Tensor):
            return ('T', tuple(v.shape), str(v.dtype), v.device.type, tuple(v.stride()), v.requires_grad)
        if v is None or isinstance(v, (bool, int, float, str)):
            return ('V', type(v).__name__, v)
        if isinstance(v, (list, tuple)):
            return ('L', tuple(one(x) for x in v))
        return ('O', type(v).__qualname__)
    return repr((str(torch.get_default_dtype()), tuple(one(a) for a in args),
                 tuple(sorted((k, one(v)) for k, v in kwargs.items()))))


def _sb_prune_guard_state():
    """Dynamo pickles every object its guards read whole: for the helpers that is the model runner (`self`), whose
    attributes include TpuEvents and other unpicklable state. Dynamo already prunes an nn.Module's attributes to the
    ones its guards read; extend that to any guarded plain object (default pickling, a __dict__): it is rebuilt with
    only the attributes its guards read, plus immutable scalars. (Marking the other values "missing" is not enough:
    the pickler never consults reducer_override for lists and dicts.) Returns the restore function."""
    import copyreg
    import types
    import torch
    from torch._dynamo import guards
    pickler = guards.GuardsStatePickler
    orig = pickler.reducer_override
    scalars = (type(None), bool, int, float, complex, str, bytes, torch.dtype, torch.device)
    skip = (torch.Tensor, torch.nn.Module, type, types.ModuleType, types.FunctionType, types.MethodType)

    def reducer_override(self, obj):
        t = type(obj)
        d = getattr(obj, '__dict__', None)
        if (type(d) is dict and id(obj) in self.guard_tree_values and not isinstance(obj, skip)
                and t.__reduce_ex__ is object.__reduce_ex__ and t.__reduce__ is object.__reduce__
                and t.__getstate__ is object.__getstate__ and not hasattr(t, '__setstate__')):
            state = {k: v for k, v in d.items() if id(v) in self.guard_tree_values or isinstance(v, scalars)}
            if len(state) < len(d):
                return copyreg.__newobj__, (t,), state
        return orig(self, obj)

    pickler.reducer_override = reducer_override

    def restore():
        pickler.reducer_override = orig
    return restore


def _sb_load_or_build(root, name, code, sig, cf, orig, args, kwargs, build=True):
    """Load this signature's artifact, or AOT-compile and save it. The artifact keeps Dynamo's guards except the ones
    torch's AOT path already drops (globals, identity matches) and the global-state guard. That one compares the
    whole grad mode / OpenMP thread count / autocast / default dtype / deterministic-flags tuple: the warm-up runs
    with grad mode on and serving with it off, and the calling thread's OpenMP thread count differs between the
    warm-up, serving and zygote-forked workers, so with it every banked program missed after the warm-up (sgB*, slB*).
    Neither changes these programs: no input requires grad (requires_grad is in the signature for arguments and in
    the tensor guards for parameters), and the thread count only sizes the host's CPU pool. vLLM-TorchTPU uses no
    autocast; the default dtype is in the signature."""
    import hashlib
    import torch
    from torch._dynamo.guards import CheckFunctionManager
    key = hashlib.sha256(repr((_SB_FORMAT, _SB['salt'], code.co_filename, code.co_firstlineno, code.co_code,
                               sig)).encode())
    path = os.path.join(root, f'{name}-{key.hexdigest()[:20]}.aot')
    t0 = time.perf_counter()
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                ent = torch.compiler.load_compiled_function(f, f_globals=orig.__globals__)
            _sb_count(name, 'hit', time.perf_counter() - t0)
            return ent
        except Exception as e:  # noqa: BLE001
            _log().warning('PM_SAMPLER_BANK %s: load failed (%r); rebuilding', path, e)
    if not build:
        return False
    if not _SB['window']:
        _log().info('PM_SAMPLER_BANK %s: building a serving signature %s', name, sig[:200])
    drop = set(CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES) | {'GLOBAL_STATE'}
    # The compiled function's hooks, shared with its stock path: set the filter for this build only.
    hooks = dict(zip(cf.aot_compile.__code__.co_freevars, (c.cell_contents for c in cf.aot_compile.__closure__)))[
        'self']._hooks
    old_filter = hooks.guard_filter_fn
    hooks.guard_filter_fn = lambda entries: [not (g.is_global or g.guard_type in drop) for g in entries]
    restore = _sb_prune_guard_state()
    try:
        ent = cf.aot_compile((args, kwargs))
        os.makedirs(root, exist_ok=True)
        ent.save_compiled_function(path)
    except Exception as e:  # noqa: BLE001
        _log().warning('PM_SAMPLER_BANK %s %s: aot_compile/save failed (%r); stock path', name, sig[:120], e)
        _sb_count(name, 'failed')
        return False
    finally:
        restore()
        hooks.guard_filter_fn = old_filter
    _sb_count(name, 'built', time.perf_counter() - t0)
    return ent


# ------------------------------------------------------------------------------------------ snapshot

_SNAP = {}  # restore_pending: bool; async: dict(pool, futures, t0, bytes, rank)


def _patch_snapshot():
    from vllm.model_executor.model_loader import default_loader as dl
    from vllm_torchtpu.runner import tpu_runner as tr

    if getattr(tr.TPUModelRunner.load_model, '_pm', False):
        return
    mode, snap_dir = os.environ['PM_SNAP_MODE'], os.environ['PM_SNAP_DIR']
    orig_lw = dl.DefaultModelLoader.load_weights
    orig_lm = tr.TPUModelRunner.load_model

    def load_weights(self, model, model_config):
        if mode == 'restore':
            rank, world = _rank_world()
            if _snapshot_ready(snap_dir, rank):
                _SNAP['restore_pending'] = True
                _log().info('PM_SNAP rank=%s: snapshot present, skipping the checkpoint', rank)
                return None
            _log().warning('PM_SNAP rank=%s: no snapshot in %s; running the stock loader', rank, snap_dir)
        return orig_lw(self, model, model_config)

    def load_model(self, *args, **kwargs):
        out = orig_lm(self, *args, **kwargs)
        rank, world = _rank_world()
        if mode == 'save':
            _save(self.model, snap_dir, rank, world)
        elif _SNAP.pop('restore_pending', False):
            _restore(self.model, snap_dir, rank, world, background=os.environ.get('PM_SNAP_ASYNC') == '1')
            if os.environ.get('PM_AOT_PRELOAD') == '1':
                _preload_aot(self.model, rank)
            if os.environ.get('PM_BANK_PRELOAD') == '1':
                _preload_bank(self.vllm_config, rank)
        return out

    load_weights._pm = True
    load_model._pm = True
    dl.DefaultModelLoader.load_weights = load_weights
    tr.TPUModelRunner.load_model = load_model


def _rank_world():
    import torch.distributed as dist
    return dist.get_rank(), dist.get_world_size()


def _tensors(model):
    """Parameters, then buffers (persistent or not), each object once, in module order."""
    out, seen = {}, set()
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if name in out or id(t) in seen:
            continue
        seen.add(id(t))
        out[name] = t
    return out


def _paths(snap_dir, rank):
    return os.path.join(snap_dir, f'rank{rank}.bin'), os.path.join(snap_dir, f'rank{rank}.json')


def _snapshot_ready(snap_dir, rank):
    """True once rank<R>.json (written after rank<R>.bin) exists; waits up to PM_SNAP_WAIT_S for a fetch."""
    bin_path, man_path = _paths(snap_dir, rank)
    wait_s = float(os.environ.get('PM_SNAP_WAIT_S', '0'))
    t0 = time.perf_counter()
    while not (os.path.exists(man_path) and os.path.exists(bin_path)):
        if time.perf_counter() - t0 >= wait_s:
            return False
        time.sleep(0.02)
    if wait_s:
        _log().info('PM_SNAP_WAIT rank=%s waited=%.2fs', rank, time.perf_counter() - t0)
    return True


def _save(model, snap_dir, rank, world):
    import torch
    log = _log()
    t0 = time.perf_counter()
    torch.accelerator.synchronize()
    os.makedirs(snap_dir, exist_ok=True)
    bin_path, man_path = _paths(snap_dir, rank)
    entries, off = [], 0
    with open(bin_path + '.tmp', 'wb') as f:
        for name, t in _tensors(model).items():
            c = t.detach().to('cpu').contiguous()
            nbytes = c.numel() * c.element_size()
            pad = (-off) % 4096
            if pad:
                f.write(b'\0' * pad)
                off += pad
            if nbytes:
                f.write(c.reshape(-1).view(torch.uint8).numpy().data)
            entries.append({'name': name, 'dtype': str(c.dtype).removeprefix('torch.'), 'shape': list(c.shape),
                            'offset': off, 'nbytes': nbytes,
                            'kind': 'param' if isinstance(t, torch.nn.Parameter) else 'buffer'})
            off += nbytes
    os.replace(bin_path + '.tmp', bin_path)
    manifest = {'format': 2, 'stage': 'after_load_model', 'rank': rank, 'world_size': world, 'bytes': off,
                'tensors': entries}
    with open(man_path + '.tmp', 'w') as f:
        json.dump(manifest, f)
    os.replace(man_path + '.tmp', man_path)
    dt = time.perf_counter() - t0
    log.info('PM_SNAP_SAVE rank=%s tensors=%d bytes=%d took=%.2fs (%.2f GB/s)', rank, len(entries), off, dt,
             off / dt / 1e9)


def _restore(model, snap_dir, rank, world, background=False):
    import torch
    log = _log()
    bin_path, man_path = _paths(snap_dir, rank)
    t0 = time.perf_counter()
    with open(man_path) as f:
        man = json.load(f)
    tensors = _tensors(model)
    problems = []
    if man.get('format') != 2 or man.get('world_size') != world or man.get('rank') != rank:
        problems.append(f"manifest format/rank/world {man.get('format')}/{man.get('rank')}/{man.get('world_size')}"
                        f" vs 2/{rank}/{world}")
    names = [e['name'] for e in man['tensors']]
    if set(names) != set(tensors):
        problems.append(f'tensor names differ: missing={sorted(set(tensors) - set(names))[:5]} '
                        f'extra={sorted(set(names) - set(tensors))[:5]}')
    else:
        for e in man['tensors']:
            t = tensors[e['name']]
            if list(t.shape) != e['shape'] or str(t.dtype).removeprefix('torch.') != e['dtype']:
                problems.append(f"{e['name']}: {list(t.shape)} {t.dtype} vs {e['shape']} {e['dtype']}")
                break
    if os.path.getsize(bin_path) != man['bytes']:
        problems.append(f"bin size {os.path.getsize(bin_path)} != {man['bytes']}")
    if problems:
        raise RuntimeError(f"PM_SNAP rank {rank}: snapshot does not match the model after load_model: "
                           f"{'; '.join(problems)}")

    flat = torch.from_file(bin_path, shared=True, size=man['bytes'], dtype=torch.uint8)

    def copy_one(e):
        src = flat[e['offset']:e['offset'] + e['nbytes']].view(getattr(torch, e['dtype'])).reshape(e['shape'])
        with torch.no_grad():
            tensors[e['name']].data.copy_(src)

    def copy_batches(es, limit):
        """PM_SNAP_FOREACH: one torch._foreach_copy_ per batch of up to `limit` bytes, so a restore thread needs
        the GIL once per batch instead of several times per tensor. The per-tensor loop fell to 1.5-1.8 GB/s while
        the main thread deserialized (the C++ deserialization holds the GIL, so each GIL wait is a whole call)."""
        i = 0
        while i < len(es):
            j, size = i, 0
            while j < len(es) and (j == i or size + es[j]['nbytes'] <= limit):
                size += es[j]['nbytes']
                j += 1
            src = [flat[e['offset']:e['offset'] + e['nbytes']].view(getattr(torch, e['dtype'])).reshape(e['shape'])
                   for e in es[i:j]]
            with torch.no_grad():
                torch._foreach_copy_([tensors[e['name']].data for e in es[i:j]], src)
            i = j

    threads = int(os.environ.get('PM_SNAP_THREADS', '1'))
    t1 = time.perf_counter()
    if background:
        from concurrent.futures import ThreadPoolExecutor
        params = [e for e in man['tensors'] if e['kind'] == 'param']
        for e in man['tensors']:
            if e['kind'] != 'param':
                copy_one(e)
        pool = ThreadPoolExecutor(threads, thread_name_prefix='pm_snap')
        order = sorted(params, key=lambda e: -e['nbytes'])
        if os.environ.get('PM_SNAP_FOREACH') == '1':  # one task per thread, balanced by bytes, largest first
            bins, load = [[] for _ in range(threads)], [0] * threads
            for e in order:
                k = load.index(min(load))
                bins[k].append(e)
                load[k] += e['nbytes']
            limit = int(float(os.environ.get('PM_SNAP_FOREACH_MB', '256')) * (1 << 20))
            futures = [pool.submit(copy_batches, b, limit) for b in bins if b]
        else:
            futures = [pool.submit(copy_one, e) for e in order]
        _SNAP['async'] = {'pool': pool, 'futures': futures, 't0': t0, 'bytes': man['bytes'], 'rank': rank,
                          'flat': flat}
        log.info('PM_SNAP_RESTORE_ASYNC rank=%s started: %d buffers copied (check+buffers %.2fs), %d params on %d '
                 'threads', rank, len(man['tensors']) - len(params), time.perf_counter() - t0, len(params), threads)
        return
    if threads > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(copy_one, man['tensors']))
    else:
        for e in man['tensors']:
            copy_one(e)
    t2 = time.perf_counter()
    torch.accelerator.synchronize()
    t3 = time.perf_counter()
    log.info('PM_SNAP_RESTORE rank=%s tensors=%d bytes=%d check=%.2fs copy=%.2fs sync=%.2fs total=%.2fs '
             '(%.2f GB/s) threads=%d', rank, len(names), man['bytes'], t1 - t0, t2 - t1, t3 - t2, t3 - t0,
             man['bytes'] / (t3 - t0) / 1e9, threads)


def _join_async(where):
    st = _SNAP.pop('async', None)
    if not st:
        return
    import torch
    t0 = time.perf_counter()
    for f in st['futures']:
        f.result()
    torch.accelerator.synchronize()
    st['pool'].shutdown()
    t1 = time.perf_counter()
    _log().info('PM_SNAP_JOIN rank=%s at=%s waited=%.2fs restore_wall=%.2fs (%.2f GB/s)', st['rank'], where,
                t1 - t0, t1 - st['t0'], st['bytes'] / (t1 - st['t0']) / 1e9)


# ------------------------------------------------------------------------------------------- profile

def _install_profiles(spec, out_dir):
    import importlib
    for target in filter(None, (s.strip() for s in spec.split(','))):
        mod_name, qual = target.split(':')
        try:
            owner = importlib.import_module(mod_name)
            parts = qual.split('.')
            for p in parts[:-1]:
                owner = getattr(owner, p)
            orig = getattr(owner, parts[-1])
        except Exception as e:  # noqa: BLE001
            _log().warning('PM_PROFILE cannot resolve %s: %r', target, e)
            continue
        if getattr(orig, '_pm_prof', False):
            continue
        setattr(owner, parts[-1], _profiled(orig, qual, out_dir))


def _profiled(orig, qual, out_dir):
    import functools

    @functools.wraps(orig)
    def wrapped(*args, **kwargs):
        import cProfile
        import io
        import pstats
        prof = cProfile.Profile()
        t0 = time.perf_counter()
        prof.enable()
        try:
            return orig(*args, **kwargs)
        finally:
            prof.disable()
            wall = time.perf_counter() - t0
            try:
                os.makedirs(out_dir, exist_ok=True)
                prof.dump_stats(os.path.join(out_dir, f'{os.getpid()}_{qual}.pstats'))
                s = io.StringIO()
                pstats.Stats(prof, stream=s).sort_stats('cumulative').print_stats(45)
                _log().info('PM_PROFILE %s pid=%d wall=%.2fs\n%s', qual, os.getpid(), wall, s.getvalue())
            except Exception as e:  # noqa: BLE001
                _log().warning('PM_PROFILE %s dump failed: %r', qual, e)

    wrapped._pm_prof = True
    return wrapped


def _start_sampler(out_dir, period):
    """Diagnostics (py-spy is not allowed to ptrace here): a daemon thread samples the main thread's Python
    stack every `period` s and appends it to <out_dir>/<pid>.jsonl ({"t": wall, "s": [frame ids, innermost
    first]}; frames are declared once as {"f": id, "n": "file:func:line"}). Flushed every ~2 s, so a killed
    process keeps its samples. Starts at plugin registration, i.e. after vLLM's own imports."""
    import json as _json
    import sys
    import threading

    os.makedirs(out_dir, exist_ok=True)
    main_id = threading.main_thread().ident
    path = os.path.join(out_dir, f'{os.getpid()}.jsonl')
    frames = {}

    def loop():
        buf = [_json.dumps({'pid': os.getpid(), 'argv': sys.argv[:3], 't0': time.time()})]
        last = time.time()
        while True:
            time.sleep(period)
            f = sys._current_frames().get(main_id)
            ids = []
            while f is not None and len(ids) < 60:
                c = f.f_code
                key = f'{c.co_filename.rsplit("site-packages/", 1)[-1]}:{c.co_name}:{f.f_lineno}'
                fid = frames.get(key)
                if fid is None:
                    fid = frames[key] = len(frames)
                    buf.append(_json.dumps({'f': fid, 'n': key}))
                ids.append(fid)
                f = f.f_back
            buf.append(_json.dumps({'t': round(time.time(), 3), 's': ids}))
            if time.time() - last > 2:
                with open(path, 'a') as fh:
                    fh.write('\n'.join(buf) + '\n')
                buf, last = [], time.time()

    threading.Thread(target=loop, name='pm-sampler', daemon=True).start()
