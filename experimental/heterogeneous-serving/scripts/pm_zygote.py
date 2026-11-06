#!/usr/bin/env python3
"""TPUFlex zygote: a pre-imported fork server for vLLM-TorchTPU's engine core and TP workers.

Decompose + Overlap applied to process startup. Stock vLLM spawns the engine core from the API server, and
the TP workers from the engine core. Each spawned interpreter imports torch, torch_tpu, jax and vLLM again
(~10 s each), one after another. The zygote starts next to the API server with the same environment, imports
that stack once while the API server imports its own, and then serves fork requests over CPython's
multiprocessing.forkserver protocol. With PM_ZYGOTE_SOCK set, the plugin (pm_torchtpu) points vLLM's
get_mp_context() at it, so the engine core and the workers are forked from an imported interpreter.
Nothing device-specific happens here: libtpu is loaded but no PJRT client exists, and torch_tpu reads the
per-worker binding env lazily; each child applies its parent's environment first (plugin: ZygoteProcess).

Env: PM_ZYGOTE_SOCK (unix socket path, required), PM_ZYGOTE_PRELOAD (comma-separated modules, optional).
The socket listens before the imports start, so a fork request issued early simply waits.
"""
from contextlib import contextmanager
import importlib
import os
import socket
import time

PRELOAD = (
    'vllm.entrypoints.cli.main',  # the `vllm` script a child re-runs as __mp_main__
    'vllm.v1.engine.core',
    'vllm.v1.engine.utils',
    'vllm.v1.executor.multiproc_executor',
    'vllm_torchtpu.executors.tpu_multiproc_executor',
    'vllm_torchtpu.worker.tpu_worker',
    'vllm_torchtpu.runner.tpu_runner',
    'vllm_torchtpu.compilation.tpu_compiler',
    'vllm.compilation.backends',
    'vllm.compilation.decorators',
    'vllm.v1.structured_output',
    'vllm.multimodal.registry',
    'vllm.renderers.hf',
    'vllm.v1.executor.ray_utils',
    'vllm.model_executor.models.gemma4_mm',
)


def main(ready_fd=None):
    t0 = time.time()
    # The host script starts the zygote from the shell that then execs `vllm serve`, so the parent is the API
    # server. Die with it (PR_SET_PDEATHSIG): f20's engine failed, the API server exited, and the idle zygote
    # kept the session alive until the job timeout. Forked children do not inherit the setting.
    import ctypes
    import signal
    ppid = os.getppid()
    ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    if os.getppid() != ppid:
        return
    path = os.environ['PM_ZYGOTE_SOCK']
    if os.path.exists(path):
        os.unlink(path)
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(path)
    os.chmod(path, 0o600)
    listener.listen(64)
    print(f'PM_ZYGOTE listening pid={os.getpid()} sock={path}', flush=True)
    mods = [m for m in os.environ.get('PM_ZYGOTE_PRELOAD', ','.join(PRELOAD)).split(',') if m]
    failed = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001  (a missing optional module only costs its import later)
            failed.append(f'{m}: {e!r}'[:200])
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    print(f'PM_ZYGOTE_READY took={time.time() - t0:.2f}s modules={len(mods)} failed={failed}', flush=True)
    if ready_fd is not None:
        os.write(ready_fd, b'1')
        os.close(ready_fd)
    alive_r, _alive_w = os.pipe()  # the write end stays open: the zygote lives until its session is killed
    from multiprocessing import forkserver
    forkserver.main(listener.detach(), alive_r, [])


def _rewind_cpu_only():
    import torch
    if torch.tpu.is_initialized() or torch.tpu._is_in_bad_fork():
        raise RuntimeError('rewind requires an uninitialized, fork-safe TPU runtime')
    if torch.distributed.is_initialized():
        raise RuntimeError('distributed state must be created after rewind')
    if any(k in os.environ for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
                                     'RAY_ADDRESS', 'MASTER_ADDR', 'MASTER_PORT', 'TORCH_TPU_TOPOLOGY',
                                     'TORCH_TPU_SLICEBUILDER_ADDRESSES', 'TORCH_TPU_XPROF_SESSION_ID')) \
            or any(k.startswith(('VLLM_DP_', 'TORCH_TPU_DP_')) for k in os.environ):
        raise RuntimeError('rewind cannot inherit external distributed identity')


def prepare_rewind():
    """pm_rewind factory: import once, checkpoint before any serving resources exist.

    Unlike main(), this neither binds the forkserver socket nor registers general
    plugins. Both must see the fresh launch context, independently after the fork.
    The immutable environment must select local mp; Ray actors are not templates.
    Experimental: native restore is allocator-sensitive; use a qualified frozen
    runtime (the local acceptance probe uses libtcmalloc-minimal4 2.9.1).
    """
    if os.environ.get('TPU_MULTIHOST_BACKEND') != 'mp':
        raise ValueError('rewind currently requires TPU_MULTIHOST_BACKEND=mp')
    forbidden = ('PM_ZYGOTE_SOCK', 'PM_RAY_ZYGOTE_SOCK', 'RAY_ADDRESS', 'TTL_RAY_HOSTS',
                 'RANK', 'LOCAL_RANK', 'TORCH_TRACE', 'PM_SAMPLE_DIR', 'PM_GATE_FILE')
    if any(os.environ.get(k) for k in forbidden):
        raise ValueError('runtime template contains launch-specific identity or tracing state')
    for module in (*PRELOAD, 'vllm.entrypoints.cli.serve'):
        importlib.import_module(module)  # strict: never publish an incomplete template
    import vllm.plugins
    if vllm.plugins.plugins_loaded:
        raise RuntimeError('general plugins ran before the rewind binding boundary')
    import torch
    backend_name = torch._C._get_privateuse1_backend_name()
    if backend_name == 'privateuseone':
        # CPU hosts skip TorchTPU autoload's hardware-dependent registration.
        # Declare only the device NAME for CLI metadata; no client/device init,
        # no fabricated hardware count, and no general plugin registration.
        torch.utils.rename_privateuse1_backend('tpu')
    elif backend_name != 'tpu':
        raise RuntimeError(f'PrivateUse1 is already owned by {backend_name}')
    _rewind_cpu_only()
    return _resume_rewind


def _resume_rewind(argv):
    import sys
    import vllm.envs as envs
    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    if not argv or argv[0] != 'serve':
        raise ValueError('rewind entry accepts only vllm serve')
    _rewind_cpu_only()
    sys.argv = ['vllm', *argv]
    cli_env_setup()
    # vLLM's AsyncEngineArgs parser itself loads general plugins. Fork first so
    # those registrations see the fresh socket on each side, not a frozen or
    # missing socket. Only this private CPU helper exists before validation;
    # no engine, public API listener, Ray connection or TPU client is started.
    with rewind_zygote():
        parser = FlexibleArgumentParser(prog='vllm')
        serve = ServeSubcommand()
        serve.subparser_init(parser.add_subparsers(dest='subparser', required=True))
        a = parser.parse_args(argv)  # includes config, aliases and underscore normalization
        serve.validate(a)
        if (a.distributed_executor_backend not in (None, 'mp')
                or a.tensor_parallel_size not in (1, 2, 4)
                or a.pipeline_parallel_size != 1 or a.data_parallel_size != 1
                or a.prefill_context_parallel_size != 1 or a.nnodes != 1 or a.node_rank != 0
                or a.data_parallel_backend != 'mp' or a.data_parallel_size_local not in (None, 1)
                or a.data_parallel_rank is not None or a.data_parallel_start_rank is not None
                or a.data_parallel_external_lb or a.data_parallel_hybrid_lb
                or a.data_parallel_multi_port_external_lb or a.enable_elastic_ep
                or a.headless or getattr(a, 'grpc', False) or a.api_server_count not in (None, 1)
                or envs.VLLM_USE_RUST_FRONTEND or os.environ.get('TPU_MULTIHOST_BACKEND') != 'mp'):
            raise ValueError('rewind requires one local HTTP server, TP1/2/4, PP1, DP1, PCP1, mp only')
        a.distributed_executor_backend = 'mp'
        _rewind_cpu_only()
        serve.cmd(a)  # dispatch this validated namespace once; no exec or second config read


@contextmanager
def rewind_zygote():
    """Generation-owned zygote; used by serving and the real forkserver smoke."""
    import select
    import signal
    from pathlib import Path

    _rewind_cpu_only()
    sock = str(Path(os.environ['_PM_REWIND_RUN_DIR']) / 'zygote.sock')
    if len(os.fsencode(sock)) > 100:
        raise ValueError('run-dir is too long for a private Unix socket')
    os.environ['PM_ZYGOTE_SOCK'] = sock
    ready_r, ready_w = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(ready_r)
        try:
            _rewind_cpu_only()
            main(ready_w)
        except BaseException:
            import traceback
            traceback.print_exc()
            os._exit(1)
        os._exit(0)
    os.close(ready_w)
    try:
        if not select.select([ready_r], [], [], 30)[0] or os.read(ready_r, 1) != b'1':
            raise RuntimeError('restored zygote failed before ready')
        from vllm.plugins import load_general_plugins
        load_general_plugins()
        yield
    finally:
        os.close(ready_r)
        try:
            os.kill(child, signal.SIGKILL)  # bounded even if the forkserver is stopped
        except ProcessLookupError:
            pass
        os.waitpid(child, 0)
        Path(sock).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
