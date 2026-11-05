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
          child; the child exits when this process disappears (EOF on its copy of the connection). Non-strict
          callers retain the plain-interpreter fallback if the zygote is unavailable.

PM_RAY_ZYGOTE_STRICT=1 is fail-closed. Its CPU prefix may import Ray/vLLM but must not connect a client,
initialize a device, or retain a worker identity. A pidfd sent by the owning parent, never a saved PID,
controls proxy signaling. Fresh raylet identity is installed before worker bootstrap and plugin activation.

Env: PM_RAY_ZYGOTE_SOCK, PM_RAY_ZYGOTE_STRICT, PM_RAY_ZYGOTE_PRELOAD (customizable only in non-strict mode).
"""
import os
import socket
import struct
import sys

# Import/registration work belongs in the CPU image, not in every restored child.
# Ray connections and all per-generation binding remain after restore.
PRELOAD = ('torch', 'torch_tpu', 'jax')  # ordinary, non-snapshot zygote stays unchanged
# Ray's C++ JOB_ID is cached at import and has no Python refresh API. Worker
# bootstrap must import Ray only after the real raylet environment is installed.
# Finish SDK environment defaults before JAX's host-specific LIBTPU_INIT_ARGS.
# Both are real saved CPU outputs; late plugin activation must not extend them.
WORKER_CPU_PRELOAD = ('vllm_torchtpu.env_override', *PRELOAD, 'tokenizers')
CPU_PRELOAD = (
    'vllm_torchtpu.env_override', 'torch', 'torch_tpu', 'jax', 'ray',
    'vllm.entrypoints.cli.serve', 'vllm.v1.engine.core',
    'vllm_torchtpu.worker.tpu_worker', 'vllm_torchtpu.runner.tpu_runner',
    'vllm_torchtpu.compilation.tpu_compiler', 'vllm.renderers.hf',
)
_HDR = struct.Struct('!I')
_MAX_REQUEST = 4 * 1024 * 1024
_PID = struct.Struct('!i')
_PROXY_IDENTITY = None  # assigned only in a child from its parent's authenticated socket receipt
CPU_ENV_KEYS = (
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TORCH_DEVICE_BACKEND_AUTOLOAD',
    'PJRT_DEVICE', 'JAX_PLATFORMS', 'ENABLE_PJRT_COMPATIBILITY', 'TPU_SKIP_MDS_QUERY',
    'TPU_MULTIHOST_BACKEND', 'TPU_ACCELERATOR_TYPE', 'TPU_TOPOLOGY', 'TPU_TOPOLOGY_WRAP',
    'TPU_HOST_BOUNDS', 'TPU_CHIPS_PER_HOST_BOUNDS', 'TORCH_TPU_TOPOLOGY',
    'TPU_WORKER_HOSTNAMES', 'TTL_RAY_HOSTS', 'TTL_RAY_WORKER_UNSET',
    'PYTHONPATH', 'LD_LIBRARY_PATH', 'VLLM_PLUGINS', 'GCONV_PATH',
)
# Only lazy TPU/pin configuration may differ in the raw raylet environment.
# Paths, plugin selection, autoload and CPU threading settings must already match.
PIN_NORMALIZABLE_KEYS = (
    'PJRT_DEVICE', 'JAX_PLATFORMS', 'ENABLE_PJRT_COMPATIBILITY', 'TPU_SKIP_MDS_QUERY',
    'TPU_MULTIHOST_BACKEND', 'TPU_ACCELERATOR_TYPE', 'TPU_TOPOLOGY', 'TPU_TOPOLOGY_WRAP',
    'TPU_HOST_BOUNDS', 'TPU_CHIPS_PER_HOST_BOUNDS', 'TORCH_TPU_TOPOLOGY', 'TPU_WORKER_HOSTNAMES',
)
WORKER_IDENTITY_KEYS = (
    'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'ROLE_RANK',
    'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'RAY_JOB_ID', 'RAY_WORKER_ID',
    'RAY_NODE_ID', 'RAY_RAYLET_PID', 'TPU_WORKER_ID', 'VLLM_HOST_IP', 'TPU_VISIBLE_CHIPS',
    'TPU_PROCESS_ADDRESSES', 'TPU_PROCESS_PORT', 'CLOUD_TPU_TASK_ID', 'TPU_TASK_ID',
    'TORCH_TPU_SLICEBUILDER_ADDRESSES', 'TORCH_TPU_XPROF_SESSION_ID',
    'TTL_HOST_RANK', 'RAY_ACTOR_ID', 'CUDA_VISIBLE_DEVICES', 'TPU_VISIBLE_DEVICES',
    'JAX_PROCESS_INDEX', 'TPU_PROCESS_INDEX', 'PJRT_LOCAL_PROCESS_RANK',
)

# DMTCP 4.2.0 src/constants.h ENV_VARS_ALL: execwrappers.cpp discards envp copies
# of these fields and reconstructs them from the executing process's libc environ.
_DMTCP_CONTROL_ENV = (
    'DMTCP_COORD_HOST', 'DMTCP_COORD_PORT', 'DMTCP_CHECKPOINT_INTERVAL',
    'DMTCP_REMOTE_SHELL_CMD', 'DMTCP_ORIG_LD_PRELOAD', 'DMTCP_HIJACK_LIBS',
    'DMTCP_HIJACK_LIBS_M32', 'DMTCP_PLUGIN', 'DMTCP_CHECKPOINT_DIR', 'DMTCP_TMPDIR',
    'DMTCP_CKPT_OPEN_FILES', 'DMTCP_QUIET', 'JALIB_STDERR_PATH', 'DMTCP_GZIP',
    'DMTCP_ALLOC_PLUGIN', 'DMTCP_DL_PLUGIN', 'DMTCP_SIGCKPT', 'SCREENDIR',
    'DMTCP_VIRTUAL_PID', 'DMTCP_FSGSBASE_ENABLED', 'DMTCP_DISABLE_PAGEMAP_SCAN',
    'DMTCP_LOG_FILE',
    # Additional controls in constants.h and the path/modify-env plugins, including
    # execwrappers.cpp's separately appended NEW_PATH_PREFIX and legacy coord aliases.
    'DMTCP_RESTART_DIR', 'DMTCP_SKIP_TRUNCATE_FILE_AT_RESTART',
    'DMTCP_ALLOW_OVERWRITE_WITH_CKPTED_FILES', 'DMTCP_EXPLICIT_SRUN',
    'DMTCP_COORD_LOG_FILENAME', 'DMTCP_COORD_WRITE_KV_DATA', 'DMTCP_FORKED_CHECKPOINT',
    'DMTCP_DISABLE_PRGNAME_PREFIX', 'DMTCP_DISABLE_STRICT_CHECKING',
    'DMTCP_NEW_PATH_PREFIX', 'DMTCP_PATH_MAPPING', 'DMTCP_ENV_FILE',
    'DMTCP_HOST', 'DMTCP_PORT',
)


def install_child_environment(env):
    """Install fresh application state without losing this child's live native exec controls."""
    import ctypes
    process = ctypes.CDLL(None)
    try:
        enabled = process.dmtcp_is_enabled
    except AttributeError:
        enabled = None
    if enabled is not None:
        enabled.argtypes = []
        enabled.restype = ctypes.c_int
    if enabled is None or not enabled():
        os.environ.clear()
        os.environ.update(env)
        return

    # Python's mapping can still contain capture-era coordinator values. Resolve
    # current libc values in this child, after DMTCP's fork/restart hooks have run.
    libc = ctypes.CDLL('libc.so.6')
    getenv = libc.getenv
    getenv.argtypes = [ctypes.c_char_p]
    getenv.restype = ctypes.c_char_p
    controls = {}
    for key in _DMTCP_CONTROL_ENV:
        value = getenv(key.encode('ascii'))
        if value is not None:
            controls[key] = os.fsdecode(value)
    if not controls.get('DMTCP_HIJACK_LIBS'):
        raise RuntimeError('active DMTCP has no native DMTCP_HIJACK_LIBS for real exec')

    fresh = {key: value for key, value in env.items()
             if not key.startswith('DMTCP_') and key not in _DMTCP_CONTROL_ENV}
    # LD_PRELOAD is the fresh application's value, never a hand-built backend
    # preload. The real exec wrapper prepends HIJACK_LIBS. Its ORIG value must not
    # resurrect an old application's preload when the fresh one is absent/empty.
    controls.pop('DMTCP_ORIG_LD_PRELOAD', None)
    if fresh.get('LD_PRELOAD'):
        controls['DMTCP_ORIG_LD_PRELOAD'] = fresh['LD_PRELOAD']
    fresh.update(controls)

    # Include C-only application keys in removal, not just Python's old snapshot.
    # Snapshot names before mutation; never keep a native environ pointer afterward.
    native_names = set()
    environ = ctypes.POINTER(ctypes.c_char_p).in_dll(process, 'environ')
    for entry in environ:
        if entry is None:
            break
        key, separator, _ = entry.partition(b'=')
        if separator:
            native_names.add(os.fsdecode(key))
    del environ
    # Do not clear(): backend control keys must remain present throughout the
    # replacement, even while Python's stale mapping is being resynchronized.
    for key in set(os.environ) | native_names:
        if key not in fresh:
            if key in os.environ:
                del os.environ[key]
            else:
                os.unsetenv(key)
    os.environ.update(fresh)


def assert_real_pid():
    with open('/proc/self/stat') as f:
        actual = int(f.read().split(' ', 1)[0])
    if os.getpid() != actual:
        raise RuntimeError(f'native PID virtualization is unsupported: getpid={os.getpid()} proc={actual}')
    return actual


def _pidfd_syscall(number, *args):
    import ctypes
    if sys.platform != 'linux' or os.uname().machine != 'x86_64':
        raise RuntimeError('pidfd syscall fallback supports Linux x86_64 only')
    call = ctypes.CDLL(None, use_errno=True).syscall
    call.restype = ctypes.c_long
    ctypes.set_errno(0)
    result = call(ctypes.c_long(number), *args)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def _pidfd_int(value):
    import operator
    value = operator.index(value)
    if not -(1 << 31) <= value < (1 << 31):
        raise OverflowError('pidfd argument does not fit a C int')
    return value


def pidfd_open(pid, flags=0):
    """Linux ownership handle, including Python builds without os.pidfd_open."""
    import operator
    pid = _pidfd_int(pid)
    flags = operator.index(flags)
    if not 0 <= flags < (1 << 32):
        raise OverflowError('pidfd flags do not fit an unsigned C int')
    if hasattr(os, 'pidfd_open'):
        fd = os.pidfd_open(pid, flags)
    else:
        import ctypes
        fd = _pidfd_syscall(434, ctypes.c_int(pid), ctypes.c_uint(flags))
    try:
        os.set_inheritable(fd, False)
    except BaseException:
        os.close(fd)
        raise
    return fd


def pidfd_send_signal(fd, sig):
    """Signal only this handle. Never fall back to a numeric-PID kill."""
    import signal
    fd, sig = _pidfd_int(fd), _pidfd_int(sig)
    if hasattr(signal, 'pidfd_send_signal'):
        signal.pidfd_send_signal(fd, sig)
    else:
        import ctypes
        _pidfd_syscall(424, ctypes.c_int(fd), ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(0))


def _pidfd_pid(fd):
    if os.readlink(f'/proc/self/fd/{fd}') != 'anon_inode:[pidfd]':
        raise ValueError('received descriptor is not a Linux pidfd')
    with open(f'/proc/self/fdinfo/{fd}') as f:
        fields = dict(line.split(':', 1) for line in f if ':' in line)
    return int(fields.get('Pid', '0'))


def _pidfd_exited(fd):
    import select
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    events = poller.poll(0)
    if events and events[0][1] & (select.POLLERR | select.POLLNVAL):
        raise RuntimeError('invalid pidfd while checking process exit')
    return bool(events and events[0][1] & (select.POLLIN | select.POLLHUP))


def get_proxy_identity():
    """No environment/PID lookup: this is the authenticated receipt inherited at fork."""
    return dict(_PROXY_IDENTITY) if _PROXY_IDENTITY is not None else None


def _assert_tpu_uninitialized(*, required=False):
    tpu = sys.modules.get('torch.tpu')
    if tpu is None:
        if required:
            raise RuntimeError('TorchTPU did not register its real torch.tpu module')
        return
    # The pinned SDK registers a PEP 562 lazy public module. Its real getters
    # resolve these two read-only native-state predicates without client init.
    initialized = getattr(tpu, 'is_initialized', None)
    bad_fork = getattr(tpu, '_is_in_bad_fork', None)
    if not callable(initialized) or not callable(bad_fork):
        raise RuntimeError('torch.tpu must expose both real initialization-state predicates')
    initialized_state, bad_fork_state = initialized(), bad_fork()
    if initialized_state is not False or bad_fork_state is not False:
        raise RuntimeError('torch.tpu is initialized, fork-unsafe, or returned unknown state')


def _assert_loader_unloaded(*, required=False):
    loader = sys.modules.get('torch_tpu._loader')
    if loader is None:
        if required:
            raise RuntimeError('TorchTPU loader state is unavailable at the CPU prefix boundary')
        return
    # These flags describe SDK options/registration, NOT whether a PJRT client
    # exists. Capture must precede both; normal post-restore SDK init may set them.
    for name in ('_DEVICE_INITIALIZED', '_LOADED'):
        if loader.__dict__.get(name) is not False:
            raise RuntimeError(f'TorchTPU loader {name} is not False before the saved CPU boundary')


def assert_cpu_boundary(*, identity=True, ray_free=True, allow_identity=(), loader_unloaded=False):
    """Read real runtime-state predicates; never create a client to check for one."""
    if os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') != '0':
        raise RuntimeError('strict CPU prefix requires TORCH_DEVICE_BACKEND_AUTOLOAD=0 before Torch imports')
    assert_real_pid()
    if identity:
        keys = [k for k in os.environ if k not in allow_identity and (k in WORKER_IDENTITY_KEYS
                or k.startswith(('VLLM_DP_', 'TORCH_TPU_DP_')))]
        if keys:
            raise RuntimeError(f'inherited per-worker identity: {sorted(keys)}')
    ray = sys.modules.get('ray')
    if ray is not None and ray.__dict__.get('is_initialized', lambda: False)():
        raise RuntimeError('Ray is already initialized')
    worker_module = sys.modules.get('ray._private.worker')
    if worker_module is not None:
        worker = worker_module.__dict__.get('global_worker')
        if (worker_module.__dict__.get('_global_node') is not None
                or (worker is not None and getattr(worker, 'core_worker', None) is not None)):
            raise RuntimeError('Ray owns node/core-worker state before CPU activation')
    if ray_free and any(k == 'ray' or k.startswith('ray.') for k in sys.modules):
        raise RuntimeError('CPU prefix unexpectedly imported Ray (including ray._raylet)')
    _assert_tpu_uninitialized(required='torch_tpu' in sys.modules)
    if loader_unloaded:
        _assert_loader_unloaded(required='torch_tpu' in sys.modules)
    for name in ('torch.cuda',):
        mod = sys.modules.get(name)
        if mod is None and 'torch' in sys.modules:
            mod = sys.modules['torch'].__dict__.get(name.split('.')[1])
        if mod is not None and (mod.__dict__.get('is_initialized', lambda: False)()
                                or mod.__dict__.get('_is_in_bad_fork', lambda: False)()):
            raise RuntimeError(f'{name} is initialized or fork-unsafe')
    dist = sys.modules.get('torch.distributed')
    if dist is not None and dist.__dict__.get('is_initialized', lambda: False)():
        raise RuntimeError('torch.distributed is already initialized')
    for name in ('jax._src.xla_bridge', 'jax.lib.xla_bridge'):
        bridge = sys.modules.get(name)
        if bridge is not None and (bridge.__dict__.get('_backends')
                                   or bridge.__dict__.get('_default_backend') is not None):
            raise RuntimeError('JAX/PJRT backend is already initialized')
    if any(k == 'torch_xla' or k.startswith('torch_xla.') for k in sys.modules):
        raise RuntimeError('torch_xla client state is outside this CPU boundary')


def bind_tpu_device_name():
    """Public name metadata only, before DeviceConfig deserializes torch.device('tpu')."""
    torch = sys.modules['torch']
    backend = torch._C._get_privateuse1_backend_name()
    if backend == 'privateuseone':
        torch.utils.rename_privateuse1_backend('tpu')
    elif backend != 'tpu':
        raise RuntimeError(f'PrivateUse1 is already owned by {backend}')


def preload_cpu(role='workers'):
    """Construct role-specific CPU state, without clients, devices or workers."""
    import importlib
    if role not in ('workers', 'api'):
        raise ValueError(f'unknown CPU prefix role: {role}')
    expected = WORKER_CPU_PRELOAD if role == 'workers' else CPU_PRELOAD
    ray_free = role == 'workers'
    mods = tuple(m for m in os.environ.get('PM_RAY_ZYGOTE_PRELOAD', ','.join(expected)).split(',') if m)
    if mods != expected:
        raise ValueError(f'strict {role} preload must be exactly {expected!r}')
    before = dict(os.environ)
    assert_cpu_boundary(loader_unloaded=True)
    saved = []
    for name in mods:
        module = importlib.import_module(name)
        saved.append((name, module, id(module)))
        assert_cpu_boundary(ray_free=ray_free, loader_unloaded=True)
    _assert_tpu_uninitialized(required=True)
    _assert_loader_unloaded(required=True)
    state = {'nonce': os.urandom(16).hex(), 'modules': tuple(saved),
             'original_env': before, 'ray_free': ray_free}
    seal_preloaded(state)
    check_preloaded(state)
    return state


def _ray_import_environment(env):
    # Job/raylet identity and runtime locations are consumed by the fresh
    # bootstrap. All other Ray environment configuration must remain fixed.
    fresh = {*WORKER_IDENTITY_KEYS, 'RAY_ADDRESS', 'RAY_TMPDIR'}
    return {key: value for key, value in env.items() if key.startswith('RAY_') and key not in fresh}


def seal_preloaded(state):
    """Retain real SDK import-side environment outputs as part of saved CPU state."""
    before, after = state['original_env'], dict(os.environ)
    changed = {key: (before.get(key), after.get(key)) for key in before.keys() | after.keys()
               if before.get(key) != after.get(key)}
    for key in changed:
        if (key in WORKER_IDENTITY_KEYS or key in ('RAY_ADDRESS', 'PM_RAY_ZYGOTE_SOCK', 'PM_ZYGOTE_SOCK')
                or key.startswith(('DMTCP_', '_PM_REWIND_', 'VLLM_DP_', 'TORCH_TPU_DP_'))):
            raise RuntimeError(f'CPU preparation changed identity/native control: {key}')
    state['generated_env'] = changed
    state['env'] = {key: after.get(key) for key in CPU_ENV_KEYS}
    state['ray_env'] = _ray_import_environment(after)
    assert_cpu_boundary(ray_free=state['ray_free'], loader_unloaded=True)


def prepared_environment(state, env, *, include_missing=True):
    """Apply saved SDK outputs only to their original inputs or identical outputs."""
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise ValueError('prepared environment must be a string dictionary')
    result = dict(env)
    for key, (before, saved) in state['generated_env'].items():
        if not include_missing and key not in result:
            continue
        if result.get(key) not in (before, saved):
            raise RuntimeError(f'fresh environment conflicts with CPU import output: {key}')
        if saved is None:
            result.pop(key, None)
        else:
            result[key] = saved
    return result


def apply_prepared_environment(state):
    """Replay SDK defaults without replacing fresh identity or native exec controls."""
    updated = prepared_environment(state, dict(os.environ))
    for key in state['generated_env']:
        if key in updated:
            os.environ[key] = updated[key]
        else:
            os.environ.pop(key, None)
    if os.environ.get('TTL_RAY_WORKER_ENV'):
        import json
        original = json.loads(os.environ['TTL_RAY_WORKER_ENV'])
        # A later pin callback must not overwrite (e.g.) cv2's captured
        # LD_LIBRARY_PATH with the import-time input from the launch config.
        normalized = prepared_environment(state, original, include_missing=False)
        if original != normalized:
            os.environ['TTL_RAY_WORKER_ENV'] = json.dumps(normalized, separators=(',', ':'), sort_keys=True)


def check_preloaded(state, *, identity=True, allow_identity=(), effective_pin_env=False):
    assert_cpu_boundary(identity=identity, ray_free=state['ray_free'],
                        allow_identity=allow_identity, loader_unloaded=True)
    # Only a prefix that imported Ray has Ray import-time inputs to preserve.
    # Ray-free workers install the real raylet environment before their first import.
    if not state['ray_free']:
        ray_env = _ray_import_environment(os.environ)
        if ray_env != state['ray_env']:
            changed = sorted(key for key in ray_env.keys() | state['ray_env'].keys()
                             if ray_env.get(key) != state['ray_env'].get(key))
            raise RuntimeError(f'Ray import-time configuration changed after CPU preparation: {changed}')
    for key, (_, saved) in state['generated_env'].items():
        if os.environ.get(key) != saved:
            raise RuntimeError(f'saved SDK CPU environment output changed: {key}')
    for name, module, address in state['modules']:
        if sys.modules.get(name) is not module or id(module) != address:
            raise RuntimeError(f'saved CPU module was replaced: {name}')
    # Raylet can overwrite geometry with per-actor values. Validate the effective
    # environment that ttl_ray_pin will install (UNSET, then ENV), but do NOT
    # mutate the fresh raylet environment before Ray bootstrap.
    import json
    overrides = json.loads(os.environ.get('TTL_RAY_WORKER_ENV') or '{}')
    if not isinstance(overrides, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                for k, v in overrides.items()):
        raise ValueError('TTL_RAY_WORKER_ENV must be a string environment dictionary')
    unsets = tuple(filter(None, os.environ.get('TTL_RAY_WORKER_UNSET', '').split(',')))
    forbidden = (*WORKER_IDENTITY_KEYS, 'RAY_ADDRESS', 'TTL_RAY_RUNTIME_ENV',
                 'TTL_RAY_JOBID_FILE', 'TTL_RAY_WORKER_ENV', 'TTL_RAY_WORKER_UNSET', 'PM_ZYGOTE_SOCK')
    for key in (*unsets, *overrides):
        if key in forbidden or key.startswith(('VLLM_DP_', 'TORCH_TPU_DP_')):
            raise ValueError(f'pin plugin cannot replace/remove fresh identity or control state: {key}')
    for key in ('TTL_RUN', 'PM_RAY_ZYGOTE_SOCK'):
        if key in unsets or (key in overrides and overrides[key] != os.environ.get(key)):
            raise ValueError(f'pin plugin generation mismatch: {key}')
    effective = dict(os.environ)
    for key in unsets:
        effective.pop(key, None)
    effective.update(overrides)
    normalized = []
    for key, value in state['env'].items():
        if effective.get(key) != value:
            raise RuntimeError(f'effective pin environment differs from frozen CPU state: {key}')
        if os.environ.get(key) != value:
            if not effective_pin_env or key not in PIN_NORMALIZABLE_KEYS:
                raise RuntimeError(f'import-sensitive environment changed: {key}')
            normalized.append(key)
    return sorted(normalized)


def get_prepared_state():
    """Expose the factory's existing witness without creating a fresh-import substitute."""
    factory = sys.modules.get('pm_rewind_ray')
    if factory is None:
        raise RuntimeError('no saved Ray factory state in this interpreter')
    return factory.get_prepared_state()


def private_socket_path(key):
    import stat
    path = os.environ[key]
    parent = os.path.dirname(path)
    if not os.path.isabs(path) or len(os.fsencode(path)) > 100 or os.path.realpath(parent) != parent:
        raise ValueError(f'{key} must be a short absolute path in a real private directory')
    st = os.stat(parent)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) != 0o700:
        raise ValueError(f'{key} parent must be owned by this uid and mode 0700')
    if os.path.lexists(path):
        raise ValueError(f'{key} must be a new per-generation path: {path}')
    return path


def _close_fds(fds):
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass


def _recv_fds(sock, size, count):
    import array
    data, ancillary, flags, _ = sock.recvmsg(
        size, socket.CMSG_SPACE(count * array.array('i').itemsize), socket.MSG_CMSG_CLOEXEC)
    fds = []
    bad = bool(flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC))
    for level, kind, value in ancillary:
        if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
            bad = True
            continue
        entries = array.array('i')
        end = len(value) - len(value) % entries.itemsize
        entries.frombytes(value[:end])
        fds.extend(entries)
        bad |= end != len(value)
    if bad or len(fds) > count:
        _close_fds(fds)
        raise ValueError('truncated or unexpected ancillary descriptors')
    return data, fds


def _send_pid(conn, pid, pidfd=None):
    import array
    data = _PID.pack(pid)
    ancillary = [] if pidfd is None else [
        (socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [pidfd]))]
    n = conn.sendmsg([data], ancillary)
    if n == 0:
        raise EOFError('closed PID handshake')
    conn.sendall(data[n:])


def _receive_pid(sock, strict):
    data, fds = _recv_fds(sock, _PID.size, 1)
    try:
        data += _recv_exact(sock, _PID.size - len(data))
        (pid,) = _PID.unpack(data)
        if pid <= 0 or len(fds) != (1 if strict else 0):
            raise ValueError('invalid PID/pidfd handshake')
        if strict:
            # A child can exit before the handshake is consumed. A dead pidfd remains safe.
            if _pidfd_pid(fds[0]) not in (pid, -1):
                raise ValueError('pidfd does not refer to the advertised worker')
        return fds.pop() if fds else None
    finally:
        _close_fds(fds)


def _recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        part = sock.recv(n - len(buf))
        if not part:
            raise EOFError
        buf += part
    return buf


# ------------------------------------------------------------------------------------------------ exec

def _connect_socket(path, strict):
    import errno
    import time
    deadline = time.monotonic() + 30
    while True:
        sock = socket.socket(socket.AF_UNIX)
        try:
            if strict:
                sock.settimeout(max(0.001, deadline - time.monotonic()))
            sock.connect(path)
            return sock
        except OSError as e:
            sock.close()
            if not strict or e.errno not in (errno.ENOENT, errno.ECONNREFUSED):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('strict generation socket did not become available within 30s') from e
            time.sleep(min(0.05, remaining))


def exec_main(argv):
    import array
    import json
    import signal
    strict = os.environ.get('PM_RAY_ZYGOTE_STRICT') == '1'
    sock = None
    pidfd = None
    self_pidfd = None
    try:
        if strict:
            self_pidfd = pidfd_open(assert_real_pid())
            pidfd_send_signal(self_pidfd, 0)
        path = os.environ.get('PM_RAY_ZYGOTE_SOCK')
        if not path:
            raise ValueError('PM_RAY_ZYGOTE_SOCK is unset')
        sock = _connect_socket(path, strict)
        data = json.dumps({'argv': argv, 'env': dict(os.environ), 'cwd': os.getcwd(),
                           'protocol': 3 if strict else 2, 'strict': strict}).encode()
        if len(data) > _MAX_REQUEST:
            raise ValueError('worker request is too large')
        packet = _HDR.pack(len(data)) + data
        rights = (0, 1, 2, self_pidfd) if strict else (0, 1, 2)
        n = sock.sendmsg([packet],
                         [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', rights))])
        if n == 0:
            raise EOFError('closed request')
        sock.sendall(packet[n:])
        if self_pidfd is not None:
            os.close(self_pidfd)
            self_pidfd = None
        pidfd = _receive_pid(sock, strict)
        sock.settimeout(None)
    except (OSError, EOFError, ValueError, RuntimeError) as e:
        if sock is not None:
            sock.close()  # EOF also kills any worker forked before a bad handshake
        if pidfd is not None:
            os.close(pidfd)
        if self_pidfd is not None:
            os.close(self_pidfd)
        if strict:
            sys.stderr.write(f'PM_RAY_ZYGOTE strict exec failed: {e!r}\n')
            raise SystemExit(1) from e
        sys.stderr.write(f'PM_RAY_ZYGOTE exec fallback ({e!r})\n')
        os.execv(sys.executable, [sys.executable] + argv)

    def forward(signum, _frame):
        try:
            if pidfd is not None:
                pidfd_send_signal(pidfd, signum)
            else:
                # Non-strict protocol v2: the child signals itself. Never kill a foreign numeric PID.
                sock.sendall(_PID.pack(signum))
        except OSError:
            pass

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, forward)
    try:
        (status,) = _PID.unpack(_recv_exact(sock, _PID.size))
        if not -signal.NSIG < status <= 255:
            raise ValueError('invalid worker exit status')
    except (OSError, EOFError, ValueError):
        status = 1
    finally:
        sock.close()
        if pidfd is not None:
            os.close(pidfd)
    os._exit(status if status >= 0 else 128 - status)


# ----------------------------------------------------------------------------------------------- serve

def _child(conn, req, fds, preloaded=None, proxy=None):
    """Runs in the forked child; Ray has not seen this child's environment yet."""
    import signal
    import threading
    global _PROXY_IDENTITY
    _PROXY_IDENTITY = dict(proxy) if proxy is not None else None
    code = 1
    try:
        for s in (signal.SIGCHLD, signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, signal.SIG_DFL)
        signal.set_wakeup_fd(-1)
        install_child_environment(req['env'])
        if preloaded is not None:
            apply_prepared_environment(preloaded)
        os.chdir(req['cwd'])
        for i, fd in enumerate(fds):
            os.dup2(fd, i)
        _close_fds([fd for fd in fds if fd > 2])
        fds = []

        def watch():
            try:
                while True:
                    (signum,) = _PID.unpack(_recv_exact(conn, _PID.size))
                    if signum not in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                        break
                    os.kill(os.getpid(), signum)
            except (OSError, EOFError, ValueError):
                pass
            os.kill(os.getpid(), signal.SIGKILL)

        threading.Thread(target=watch, name='pm-ray-zygote-watch', daemon=True).start()
        argv = list(req['argv'])
        unbuffered = False
        while argv and argv[0].startswith('-') and argv[0] != '-m':
            flag = argv.pop(0)
            if preloaded is not None and flag not in ('-u', '-S', '-s'):
                raise ValueError(f'unsupported strict interpreter flag: {flag}')
            unbuffered |= flag == '-u'
        if not argv or (argv[0] == '-m' and len(argv) < 2):
            raise ValueError('missing Python worker entrypoint')
        if unbuffered:
            sys.stdout.reconfigure(line_buffering=True, write_through=True)
            sys.stderr.reconfigure(line_buffering=True, write_through=True)
        for p in reversed([p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]):
            if p not in sys.path:
                sys.path.insert(0, p)
        if preloaded is not None:
            import json
            normalized = check_preloaded(preloaded, identity=False, effective_pin_env=True)
            proof = {
                'event': 'PM_RAY_ZYGOTE_PROOF',
                'nonce': preloaded['nonce'], 'pid': assert_real_pid(),
                'modules': {name: hex(address) for name, _, address in preloaded['modules']},
                'tpu_initialized': False, 'ray_imported': 'ray' in sys.modules,
                'ray_initialized': bool(sys.modules.get('ray') and sys.modules['ray'].is_initialized()),
                'pin_normalized_keys': normalized,
                'proxy': get_proxy_identity(),
            }
            payload = json.dumps(proof, separators=(',', ':'))
            print('PM_RAY_ZYGOTE_PROOF ' + payload, flush=True)
            # Before Ray takes over stdio, retain the same boundary witness in
            # this authenticated, private generation (never in the CPU image).
            import stat
            proof_path = os.path.join(os.path.dirname(os.environ['PM_RAY_ZYGOTE_SOCK']),
                                      'worker-prefix.jsonl')
            proof_fd = os.open(proof_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT
                               | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, 0o600)
            try:
                info = os.fstat(proof_fd)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                    raise RuntimeError('worker prefix evidence must be an owned private regular file')
                data = (payload + '\n').encode()
                if os.write(proof_fd, data) != len(data):
                    raise RuntimeError('short worker prefix evidence append')
            finally:
                os.close(proof_fd)
            factory = sys.modules.get('pm_rewind_ray')
            if factory is not None:
                factory.activate_generation()
        import runpy
        code = 0
        if argv[0] == '-m':
            sys.argv = argv[1:]
            runpy.run_module(argv[1], run_name='__main__', alter_sys=True)
        else:
            sys.argv = argv
            sys.path.insert(0, os.path.dirname(os.path.abspath(argv[0])))
            runpy.run_path(argv[0], run_name='__main__')
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        _close_fds(fds)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(code)


def _read_request(conn):
    import json
    data, fds = _recv_fds(conn, _HDR.size, 4)
    try:
        data += _recv_exact(conn, _HDR.size - len(data))
        (n,) = _HDR.unpack(data)
        if not 0 < n <= _MAX_REQUEST:
            raise ValueError('invalid worker request length')
        req = json.loads(_recv_exact(conn, n))
        if (not isinstance(req, dict)
                or not isinstance(req.get('argv'), list) or not req['argv']
                or not all(isinstance(a, str) and '\0' not in a for a in req['argv'])
                or not isinstance(req.get('env'), dict)
                or not all(isinstance(k, str) and k and '=' not in k and '\0' not in k
                           and isinstance(v, str) and '\0' not in v for k, v in req['env'].items())
                or not isinstance(req.get('cwd'), str) or not os.path.isabs(req['cwd'])
                or '\0' in req['cwd'] or type(req.get('strict', False)) is not bool
                or type(req.get('protocol', 1)) is not int
                or req.get('protocol', 1) not in (1, 2, 3)):
            raise ValueError('malformed worker request')
        if len(fds) != (4 if req.get('strict') else 3):
            raise ValueError('expected stdio plus a self-pidfd for strict requests')
        if req.get('strict') and req.get('protocol') != 3:
            raise ValueError('strict requests require protocol v3 with an owned proxy pidfd')
        return req, fds
    except BaseException:
        _close_fds(fds)
        raise


def _proxy_receipt(conn, fd, preloaded, path):
    peer_pid, uid, _gid = struct.unpack('3i', conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    if peer_pid <= 0 or uid != os.getuid() or _pidfd_pid(fd) != peer_pid or _pidfd_exited(fd):
        raise ValueError('proxy pidfd does not identify the live same-uid Unix peer')
    with open(f'/proc/{peer_pid}/stat') as source:
        raw = source.read()
    if int(raw.split(' ', 1)[0]) != peer_pid:
        raise ValueError('proxy proc identity mismatch')
    start_ticks = int(raw.rsplit(')', 1)[1].split()[19])
    # pidfd readiness is permanent after exit. This second check excludes a
    # procfs read of a reused PID; SO_PEERCRED alone would not be enough.
    if start_ticks <= 0 or _pidfd_pid(fd) != peer_pid or _pidfd_exited(fd):
        raise ValueError('proxy exited while authenticating its start time')
    run_dir = os.environ.get('_PM_REWIND_RUN_DIR') or os.path.dirname(path)
    return {
        'run_id': os.environ.get('TTL_RUN') or os.path.basename(run_dir),
        'proxy': {'pid': peer_pid, 'start_ticks': start_ticks},
        'nonce': preloaded['nonce'], 'server_pid': assert_real_pid(),
        'worker_pid': None, 'socket': path,
    }


def _proxy_event(event, proof, **fields):
    import json
    print(event + ' ' + json.dumps({**proof, **fields}, separators=(',', ':')), flush=True)


def _retire_proxies(proxies):
    for fd, proof in list(proxies.items()):
        if _pidfd_exited(fd):
            del proxies[fd]
            try:
                # Exit is not proof of Raylet reaping: the external helper must
                # still audit disappearance of this exact (pid, start_ticks).
                _proxy_event('PM_RAY_ZYGOTE_PROXY_EXIT', proof, exited=True)
            finally:
                os.close(fd)


def _stop_proxies(proxies):
    import select
    import signal
    import time

    def send(sig):
        for fd, proof in list(proxies.items()):
            try:
                pidfd_send_signal(fd, sig)
            except ProcessLookupError:
                pass
            except OSError as exc:
                _proxy_event('PM_RAY_ZYGOTE_PROXY_CLEANUP_ERROR', proof,
                             signal=sig, error=repr(exc))

    def drain(seconds):
        deadline = time.monotonic() + seconds
        while proxies:
            _retire_proxies(proxies)
            remaining = deadline - time.monotonic()
            if not proxies or remaining <= 0:
                break
            poller = select.poll()
            for fd in proxies:
                poller.register(fd, select.POLLIN)
            poller.poll(max(1, int(remaining * 1000)))

    try:
        _retire_proxies(proxies)
        send(signal.SIGTERM)
        send(signal.SIGCONT)  # a stopped uncheckpointed proxy must also retire
        drain(0.5)
        if proxies:
            send(signal.SIGKILL)
            drain(2.0)
        for proof in proxies.values():
            _proxy_event('PM_RAY_ZYGOTE_PROXY_PENDING', proof, exited=False)
    finally:
        _close_fds(proxies)
        proxies.clear()


def serve_main(*, preloaded=None):
    import importlib
    import select
    import signal
    import time
    t0 = time.time()
    strict = os.environ.get('PM_RAY_ZYGOTE_STRICT') == '1'
    if preloaded is not None and not strict:
        raise ValueError('saved CPU state requires PM_RAY_ZYGOTE_STRICT=1')
    if strict:
        capability = pidfd_open(assert_real_pid())
        try:
            pidfd_send_signal(capability, 0)
        finally:
            os.close(capability)
        # Restore takes this branch without ANY preload import loop.
        if preloaded is None:
            preloaded = preload_cpu()
            bind_tpu_device_name()
            check_preloaded(preloaded)
        else:
            check_preloaded(preloaded)
        path = private_socket_path('PM_RAY_ZYGOTE_SOCK')
    else:
        import ctypes
        ppid = os.getppid()
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)
        if os.getppid() != ppid:
            return
        path = os.environ['PM_RAY_ZYGOTE_SOCK']
        if os.path.exists(path):
            os.unlink(path)
    listener = socket.socket(socket.AF_UNIX)
    children, proxies = {}, {}
    rfd = wfd = old_wakeup = None
    old_handlers = {}
    stopping = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)

    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    try:
        if strict:
            for sig in stopping:
                old_handlers[sig] = signal.signal(sig, stop)
        listener.bind(path)
        os.chmod(path, 0o600)
        listener.listen(64)
        print(f'PM_RAY_ZYGOTE listening pid={os.getpid()} sock={path}', flush=True)
        if not strict:
            mods = [m for m in os.environ.get('PM_RAY_ZYGOTE_PRELOAD', ','.join(PRELOAD)).split(',') if m]
            failed = []
            for m in mods:
                try:
                    importlib.import_module(m)
                except Exception as e:
                    failed.append(f'{m}: {e!r}'[:200])
            if 'ray._raylet' in sys.modules:
                os.unlink(path)
                raise RuntimeError('unexpected Ray preload; refusing frozen RayConfig')
        else:
            mods, failed = WORKER_CPU_PRELOAD, []
        print(f'PM_RAY_ZYGOTE_READY took={time.time() - t0:.2f}s modules={len(mods)} failed={failed}', flush=True)
        rfd, wfd = os.pipe()
        os.set_blocking(wfd, False)
        old_wakeup = signal.set_wakeup_fd(wfd)
        old_handlers[signal.SIGCHLD] = signal.signal(signal.SIGCHLD, lambda *_: None)
        # The generic reaper owns the server/descendants. Raylet-owned proxies
        # are separately held by authenticated pidfds until their exit.
        while True:
            poller = select.poll()
            for fd in (listener.fileno(), rfd, *proxies):
                poller.register(fd, select.POLLIN)
            try:
                ready = {fd for fd, _ in poller.poll()}
            except InterruptedError:
                continue
            if rfd in ready:
                os.read(rfd, 4096)
            for pid, conn in list(children.items()):
                done, st = os.waitpid(pid, os.WNOHANG)
                if not done:
                    continue
                del children[pid]
                code = os.waitstatus_to_exitcode(st)
                for proof in proxies.values():
                    if proof['worker_pid'] == pid:
                        proof['worker_returncode'] = code
                print(f'PM_RAY_ZYGOTE child pid={pid} exit={code}', flush=True)
                try:
                    conn.sendall(_PID.pack(code))
                except OSError:
                    pass
                conn.close()
            _retire_proxies(proxies)
            if listener.fileno() not in ready:
                continue
            conn, _ = listener.accept()
            fds = []
            child_pidfd = pid = proof = None
            try:
                conn.settimeout(30)
                req, fds = _read_request(conn)
                if strict != bool(req.get('strict')):
                    raise ValueError('strict client/server mode mismatch')
                if strict:
                    check_preloaded(preloaded)
                    if (req['env'].get('PM_RAY_ZYGOTE_STRICT') != '1'
                            or req['env'].get('PM_RAY_ZYGOTE_SOCK') != path
                            or req['env'].get('TTL_RUN') != os.environ.get('TTL_RUN')):
                        raise ValueError('request environment is not bound to this strict generation')
                    previous = signal.pthread_sigmask(signal.SIG_BLOCK, stopping)
                    try:
                        proof = _proxy_receipt(conn, fds[-1], preloaded, path)
                        proxies[fds[-1]] = proof
                        fds.pop()  # ownership transfer cannot be interrupted by normal shutdown
                        _proxy_event('PM_RAY_ZYGOTE_PROXY_IDENTITY', proof)  # flushed BEFORE fork
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
                conn.settimeout(None)
                t = time.time()
                previous = signal.pthread_sigmask(signal.SIG_BLOCK, stopping)
                try:
                    pid = os.fork()
                    if pid == 0:
                        # Never let a child run the parent's proxy-teardown finally.
                        try:
                            for sig in (*stopping, signal.SIGCHLD):
                                signal.signal(sig, signal.SIG_DFL)
                            signal.set_wakeup_fd(-1)
                            _close_fds(proxies)
                            proxies.clear()
                            listener.close()
                            os.close(rfd)
                            os.close(wfd)
                            for other in children.values():
                                other.close()
                            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
                            _child(conn, req, fds, preloaded,
                                   proof['proxy'] if proof is not None else None)
                        finally:
                            os._exit(1)
                    children[pid] = conn
                    if strict:
                        proof['worker_pid'] = pid
                        _proxy_event('PM_RAY_ZYGOTE_PROXY_IDENTITY', proof)
                        # Still our unreaped child: PID cannot be reused before this open.
                        child_pidfd = pidfd_open(pid)
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous)
                _send_pid(conn, pid, child_pidfd)
                print(f'PM_RAY_ZYGOTE fork pid={pid} argv0={req["argv"][:2]} took={time.time() - t:.3f}s', flush=True)
            except (OSError, EOFError, ValueError, RuntimeError) as e:
                print(f'PM_RAY_ZYGOTE bad request: {e!r}', flush=True)
                if pid:
                    # This remains our unreaped child, not an external PID receipt.
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            finally:
                _close_fds(fds)
                if child_pidfd is not None:
                    os.close(child_pidfd)
                if pid is None:
                    conn.close()
    finally:
        if strict:
            for sig in stopping:
                signal.signal(sig, signal.SIG_IGN)
        listener.close()
        try:
            for conn in children.values():
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                conn.close()
            _stop_proxies(proxies)
        finally:
            if old_wakeup is not None:
                signal.set_wakeup_fd(old_wakeup)
            _close_fds(fd for fd in (rfd, wfd) if fd is not None)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        # Direct SIGKILL cannot run this finally. Missing EXIT evidence is never
        # a cleanup claim; the host helper must audit exact proxy identities.


if __name__ == '__main__':
    if sys.argv[1:2] == ['exec']:
        exec_main(sys.argv[2:])
    elif sys.argv[1:2] == ['serve']:
        serve_main()
    else:
        sys.exit('usage: pm_ray_zygote.py serve | exec ARGV...')
