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


def main():
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
    alive_r, _alive_w = os.pipe()  # the write end stays open: the zygote lives until its session is killed
    from multiprocessing import forkserver
    forkserver.main(listener.detach(), alive_r, [])


if __name__ == '__main__':
    main()
