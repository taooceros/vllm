#!/usr/bin/env python3
"""TPUFlex zygote for the Ray worker processes of vLLM-TorchTPU's Ray executor (one per host).

Decompose + Overlap applied to Ray worker startup. With the Ray executor, each TP worker is a Ray actor whose
process the host's raylet starts: a fresh interpreter that imports Ray, torch, torch_tpu, jax and vLLM before the
actor even exists, four at a time per host, after the driver has asked for them. pm_zygote.py cannot serve these
processes (the raylet, not vLLM, creates them), so this zygote sits behind the interpreter the raylet runs:

  serve   (host script, at launch / T0): bind PM_RAY_ZYGOTE_SOCK, import the stack once while the API server and the
          engine core start, then fork one child per request. Nothing device-specific happens here (no PJRT client).
  exec    (the runtime env's py_executable: "<venv python> -S pm_ray_zygote.py exec"): the process the raylet
          started. It sends its argv, environment, cwd and stdio to the zygote, which forks a child that becomes the
          Ray worker (the child registers with the raylet under its own pid, so the raylet manages the real worker),
          then waits for the child's exit status and exits with it. SIGTERM/SIGINT/SIGHUP are forwarded to the
          child; the child exits when this process disappears (EOF on its copy of the connection). Without a zygote
          socket (or if connecting fails) it execs the plain interpreter: stock behaviour.

Env: PM_RAY_ZYGOTE_SOCK (unix socket path), PM_RAY_ZYGOTE_PRELOAD (comma-separated modules, optional).

Ray itself must NOT be imported here. Ray's C++ core reads its configuration from the environment once, when the
RayConfig singleton is built (core_worker.cc takes the job id from RayConfig::JOB_ID(), i.e. RAY_JOB_ID). A zygote
that has loaded ray._raylet hands every child the zygote's frozen config, and the child dies at CoreWorker creation
with `Check failed: !job_id_env.empty()` (run t8z2). So the preload stops short of Ray and every module that imports
it; the child imports Ray after it has taken the raylet's environment. If a preloaded module pulls ray._raylet in
anyway, the zygote refuses to serve (unlinks its socket) and every launcher falls back to the plain interpreter.
"""
import os
import socket
import struct
import sys

PRELOAD = (  # Ray-free: see the module docstring
    'torch', 'torch_tpu', 'jax', 'vllm.v1.worker.worker_base',
    'vllm_torchtpu.worker.tpu_worker', 'vllm_torchtpu.runner.tpu_runner', 'vllm_torchtpu.compilation.tpu_compiler',
    'vllm.compilation.backends', 'vllm.compilation.decorators', 'vllm.model_executor.models.gemma4_mm',
    'vllm.v1.structured_output',
    'pm_torchtpu', 'ttl_probe', 'ttl_ray_pin',  # plugin modules only: register() runs in the child (init_worker)
)
_HDR = struct.Struct('!I')


def _recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        part = sock.recv(n - len(buf))
        if not part:
            raise EOFError
        buf += part
    return buf


# ------------------------------------------------------------------------------------------------ exec

def exec_main(argv):
    path = os.environ.get('PM_RAY_ZYGOTE_SOCK')
    sock = None
    if path and os.path.exists(path):
        try:
            import array
            import json
            sock = socket.socket(socket.AF_UNIX)
            sock.connect(path)
            data = json.dumps({'argv': argv, 'env': dict(os.environ), 'cwd': os.getcwd()}).encode()
            sock.sendmsg([_HDR.pack(len(data)), data],
                         [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', (0, 1, 2)))])
            (pid,) = struct.unpack('!i', _recv_exact(sock, 4))
        except (OSError, EOFError) as e:
            sys.stderr.write(f'PM_RAY_ZYGOTE exec fallback ({e!r})\n')
            sock = None
    if sock is None:
        os.execv(sys.executable, [sys.executable] + argv)
    import signal

    def forward(signum, _frame):
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, forward)
    try:
        (status,) = struct.unpack('!i', _recv_exact(sock, 4))
    except EOFError:  # the zygote died
        status = 1
    os._exit(status if status >= 0 else 128 - status)


# ----------------------------------------------------------------------------------------------- serve

def _child(conn, req, fds):
    """Runs in the forked child: become the process the raylet asked for."""
    import signal
    import threading
    for s in (signal.SIGCHLD, signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, signal.SIG_DFL)
    signal.set_wakeup_fd(-1)
    os.environ.clear()
    os.environ.update(req['env'])
    os.chdir(req['cwd'])
    # torch/OpenMP sized the intra-op pool when the zygote imported torch (serve_main imports it under Ray's worker
    # value, OMP_NUM_THREADS=1). A fork cannot re-size it for every thread, so only report a mismatch here.
    nthreads = os.environ.get('OMP_NUM_THREADS')
    if nthreads and 'torch' in sys.modules and sys.modules['torch'].get_num_threads() != int(nthreads):
        print(f'PM_RAY_ZYGOTE WARNING: torch has {sys.modules["torch"].get_num_threads()} intra-op threads, the '
              f'raylet asked for OMP_NUM_THREADS={nthreads} (start the zygote with the same value)', flush=True)
    for i, fd in enumerate(fds):
        os.dup2(fd, i)
        os.close(fd)

    def watch():  # the launcher holds the other end; EOF = it is gone
        try:
            while conn.recv(64):
                pass
        except OSError:
            pass
        os.kill(os.getpid(), signal.SIGKILL)

    threading.Thread(target=watch, name='pm-ray-zygote-watch', daemon=True).start()
    argv = list(req['argv'])
    unbuffered = False
    while argv and argv[0].startswith('-') and argv[0] != '-m':  # interpreter flags (-u, -S, -s)
        unbuffered |= argv.pop(0) == '-u'
    if unbuffered:  # the existing stdio objects write to fds 0-2, which now are the raylet's
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    for p in reversed([p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]):
        if p not in sys.path:
            sys.path.insert(0, p)
    import runpy
    code = 0
    try:
        if argv and argv[0] == '-m':
            sys.argv = argv[1:]
            runpy.run_module(argv[1], run_name='__main__', alter_sys=True)
        else:
            sys.argv = argv
            sys.path.insert(0, os.path.dirname(os.path.abspath(argv[0])))
            runpy.run_path(argv[0], run_name='__main__')
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)


def _read_request(conn):
    data, anc, _flags, _addr = conn.recvmsg(1 << 16, socket.CMSG_SPACE(3 * 4))  # the fds ride on the first bytes
    fds = []
    for level, typ, cdata in anc:
        if level == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS:
            fds += list(struct.unpack(f'{len(cdata) // 4}i', cdata[:len(cdata) - len(cdata) % 4]))
    if len(data) < _HDR.size:
        data += _recv_exact(conn, _HDR.size - len(data))
    (n,) = _HDR.unpack(data[:_HDR.size])
    msg = data[_HDR.size:]
    if len(msg) < n:
        msg += _recv_exact(conn, n - len(msg))
    import json
    req = json.loads(msg)
    if len(fds) != 3:
        for fd in fds:
            os.close(fd)
        raise ValueError(f'expected 3 stdio fds, got {fds}')
    return req, fds


def serve_main():
    import ctypes
    import importlib
    import select
    import signal
    import time
    t0 = time.time()
    ppid = os.getppid()
    ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG: die with the host script
    if os.getppid() != ppid:
        return
    path = os.environ['PM_RAY_ZYGOTE_SOCK']
    if os.path.exists(path):
        os.unlink(path)
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(path)
    os.chmod(path, 0o600)
    listener.listen(64)  # requests queue up while the imports run
    print(f'PM_RAY_ZYGOTE listening pid={os.getpid()} sock={path}', flush=True)
    # Import torch with the thread count the raylet gives its workers (Ray: OMP_NUM_THREADS = the actor's num_cpus,
    # at least 1; vLLM's TP actors ask for 0). OpenMP fixes its default pool size at that import, for every thread of
    # every forked child: without this, the zygote's 56 threads per worker cut decode from 75.3 to 44.9 tok/s
    # (m8zxR1/R2/F1 vs m8zxE1/N2; every Ray-zygote run of config A too).
    os.environ.setdefault('OMP_NUM_THREADS', os.environ.get('PM_RAY_ZYGOTE_OMP_THREADS', '1'))
    mods = [m for m in os.environ.get('PM_RAY_ZYGOTE_PRELOAD', ','.join(PRELOAD)).split(',') if m]
    failed = []
    per_mod = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001  (a missing optional module only costs its import later)
            failed.append(f'{m}: {e!r}'[:200])
        per_mod.append(f'{m}:{len(os.listdir("/proc/self/task"))}')
        if 'ray._raylet' in sys.modules:  # this module dragged Ray's C++ core in: children would inherit its config
            print(f'PM_RAY_ZYGOTE_DISABLED: preloading {m} imported ray._raylet; not serving (launchers fall back '
                  f'to the plain interpreter)', flush=True)
            listener.close()
            os.unlink(path)
            return
    import threading
    tasks = os.listdir('/proc/self/task')
    print(f'PM_RAY_ZYGOTE_READY took={time.time() - t0:.2f}s modules={len(mods)} failed={failed} '
          f'os_threads={len(tasks)} threads_after_import={per_mod} py_threads={[t.name for t in threading.enumerate()]}',
          flush=True)
    if os.environ.get('PM_RAY_ZYGOTE_MAX_THREADS') and len(tasks) > int(os.environ['PM_RAY_ZYGOTE_MAX_THREADS']):
        print(f'PM_RAY_ZYGOTE_DISABLED: {len(tasks)} threads > PM_RAY_ZYGOTE_MAX_THREADS (fork-after-threads)',
              flush=True)
        listener.close()
        os.unlink(path)
        return
    rfd, wfd = os.pipe()
    os.set_blocking(wfd, False)
    signal.set_wakeup_fd(wfd)
    signal.signal(signal.SIGCHLD, lambda *_: None)
    children = {}
    while True:
        try:
            ready, _, _ = select.select([listener, rfd], [], [])
        except InterruptedError:
            continue
        if rfd in ready:
            os.read(rfd, 4096)
        while children:  # reap and report
            try:
                pid, st = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                break
            conn = children.pop(pid, None)
            code = os.waitstatus_to_exitcode(st)
            print(f'PM_RAY_ZYGOTE child pid={pid} exit={code}', flush=True)
            if conn is not None:
                try:
                    conn.sendall(struct.pack('!i', code))
                except OSError:
                    pass
                conn.close()
        if listener not in ready:
            continue
        conn, _ = listener.accept()
        try:
            req, fds = _read_request(conn)
        except (OSError, EOFError, ValueError) as e:
            print(f'PM_RAY_ZYGOTE bad request: {e!r}', flush=True)
            conn.close()
            continue
        t = time.time()
        pid = os.fork()
        if pid == 0:
            listener.close()
            os.close(rfd)
            os.close(wfd)
            for other in children.values():
                other.close()
            _child(conn, req, fds)  # never returns
        for fd in fds:
            os.close(fd)
        children[pid] = conn
        conn.sendall(struct.pack('!i', pid))
        print(f'PM_RAY_ZYGOTE fork pid={pid} argv0={req["argv"][:2]} took={time.time() - t:.3f}s', flush=True)


if __name__ == '__main__':
    if sys.argv[1:2] == ['exec']:
        exec_main(sys.argv[2:])
    elif sys.argv[1:2] == ['serve']:
        serve_main()
    else:
        sys.exit('usage: pm_ray_zygote.py serve | exec ARGV...')
