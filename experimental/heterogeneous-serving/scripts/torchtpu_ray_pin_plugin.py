"""Ray placement for vLLM-TorchTPU on a host subset of a shared Ray cluster (vLLM general plugin, installed by
torchtpu_load_host.sh when TTL_RAY_HOSTS is set; inert otherwise).

vLLM-TorchTPU 0.29.0's Ray executors assume the engine owns every TPU node of the Ray cluster:
- RayDistributedExecutorV2 (the default, VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1) builds TORCH_TPU_SLICEBUILDER_ADDRESSES,
  NNODES and the node ranks from ray.nodes(), i.e. all 8 hosts / 32 chips of this cluster, and lets its PACK
  placement group land the non-driver bundles on whichever host has free chips;
- RayDistributedExecutor (V1) asks for one bundle per TPU chip of every TPU node.
On a shared 4x8 slice that runs other TP8 instances next to this one, both are wrong. This plugin changes placement
only, the way hetero's engine actor hands vLLM-JAX its own placement group:
1. _initialize_ray_cluster: ray.init(address="auto") with the runtime env in TTL_RAY_RUNTIME_ENV (the worker
   interpreter and environment), then one {TPU: 1} bundle per worker pinned with node:<ip> resources to the hosts in
   TTL_RAY_HOSTS (host-rank order, world_size / len(hosts) per host), set as parallel_config.placement_group.
2. V2 _slice_host_layout: the slice is exactly those hosts, in that order.
3. In every process (Ray workers included, at init_worker before init_device opens the chips): TTL_RAY_WORKER_ENV
   (JSON) is applied over the inherited environment, TTL_RAY_WORKER_UNSET names are removed, and TPU_WORKER_ID is
   this host's index in TTL_RAY_HOSTS. The raylet's own pod environment describes the whole 4x8 slice.
4. Chip binding (TTL_CHIP_BIND, Ray workers): stock vLLM-TorchTPU gives local rank k the chip at this VM's local
   enumeration index k (Ray's TPU_VISIBLE_CHIPS; torch_tpu copies LOCAL_RANK into it only when unset). Which physical
   chip that index is differs between VMs, so TP rank r lands on a different slice device_id per VM pair:
   small_d/small_a (1,0,2,3,7,5,4,6), small_b (1,0,2,3,7,6,4,5). TTL_CHIP_BIND=canonical pins local rank k to
   canonical[k] of the pair's chipmap (TTL_CHIPMAP, torchtpu_chipmap.py: the local chips sorted by slice coords),
   right after tpu_worker sets up the process env and before torch_tpu opens the chip, so rank r gets device_id r
   and the same coords on every 2x4 slice. TTL_CHIP_BIND=perm:a,b,c,d pins local rank k to index perm[k] (tests).
   Every worker logs TTL_RANK_DEVICE (rank, chip index, device_id, coords) after init_device.
"""
import json
import os

_done = False
_BIND = {}


def _log():
    from vllm.logger import init_logger
    return init_logger('vllm.ttl_ray_pin')


def _local_ip():
    from vllm.utils.network_utils import get_ip
    return get_ip()


def register():
    global _done
    hosts = [h for h in os.environ.get('TTL_RAY_HOSTS', '').split(',') if h]
    if not hosts or _done:
        return
    _done = True
    log = _log()
    for k in filter(None, os.environ.get('TTL_RAY_WORKER_UNSET', '').split(',')):
        os.environ.pop(k, None)
    os.environ.update(json.loads(os.environ.get('TTL_RAY_WORKER_ENV') or '{}'))
    ip = _local_ip()
    if ip in hosts:
        os.environ['TPU_WORKER_ID'] = str(hosts.index(ip))
    log.info('TTL_RAY_PIN pid=%d ip=%s TPU_WORKER_ID=%s hosts=%s', os.getpid(), ip,
             os.environ.get('TPU_WORKER_ID'), hosts)
    _patch_executors(hosts)
    if os.environ.get('TTL_RAY_WORKER') == '1':
        _install_chip_bind(hosts)


def _chip_for(mode, local_rank, hosts):
    """Local chip index for local rank `local_rank` under TTL_CHIP_BIND=mode, plus the chipmap's expectation."""
    if mode.startswith('perm:'):
        perm = [int(x) for x in mode[5:].split(',')]
        if sorted(perm) != list(range(len(perm))):
            raise ValueError(f'TTL_CHIP_BIND={mode}: not a permutation')
        return perm[local_rank], None
    if mode != 'canonical':
        raise ValueError(f'TTL_CHIP_BIND={mode!r}: expected canonical or perm:a,b,c,d')
    path = os.environ.get('TTL_CHIPMAP', '')
    with open(path) as f:
        cm = json.load(f)
    ctx = cm.get('context') or {}
    hr = int(os.environ.get('TPU_WORKER_ID', '-1'))
    if ctx.get('hosts') != list(hosts) or ctx.get('host_rank') != hr or not cm.get('ok'):
        raise RuntimeError(f'TTL_CHIP_BIND=canonical: chipmap {path} was probed for {ctx} (ok={cm.get("ok")}), '
                           f'this worker is host rank {hr} of {hosts}; run the chipmap phase on this pair')
    chip = cm['canonical'][local_rank]
    return chip, next(c for c in cm['chips'] if c['index'] == chip)


def _install_chip_bind(hosts):
    from vllm_torchtpu.worker import tpu_worker as tw
    if getattr(tw._configure_tpu_process_env, '_ttl', False):
        return
    orig_env = tw._configure_tpu_process_env
    mode = os.environ.get('TTL_CHIP_BIND', '')

    def configure(rank, local_rank, world_size, local_world_size):
        orig_env(rank, local_rank, world_size, local_world_size)
        ray_chip = os.environ.get('TPU_VISIBLE_CHIPS')
        _BIND.update(rank=rank, local_rank=local_rank, mode=mode or 'stock', ray_chip=ray_chip)
        if mode:
            chip, exp = _chip_for(mode, int(local_rank), hosts)
            # a preset TPU_VISIBLE_CHIPS picks the chip (chipmap round "visible"); tpu_worker keeps both in agreement
            os.environ['TPU_VISIBLE_CHIPS'] = os.environ['TPU_VISIBLE_DEVICES'] = str(chip)
            _BIND.update(chip=chip, expect=exp)
        _log().info('TTL_CHIP_BIND rank=%s local_rank=%s mode=%s ray TPU_VISIBLE_CHIPS=%s -> chip %s expect %s',
                    rank, local_rank, _BIND['mode'], ray_chip, os.environ.get('TPU_VISIBLE_CHIPS'),
                    _BIND.get('expect'))

    configure._ttl = True
    tw._configure_tpu_process_env = configure
    orig_init = tw.TPUWorker.init_device

    def init_device(self):
        orig_init(self)
        try:
            import jax
            d = jax.local_devices()[0]
            dev, coords = int(d.id), [int(c) for c in d.coords]
        except Exception as e:  # noqa: BLE001
            dev, coords = None, repr(e)
        _BIND.update(device_id=dev, coords=coords)
        _log().info('TTL_RANK_DEVICE rank=%s local_rank=%s chip=%s device_id=%s coords=%s mode=%s', self.rank,
                    self.local_rank, os.environ.get('TPU_VISIBLE_CHIPS'), dev, coords, _BIND.get('mode', 'stock'))
        exp = _BIND.get('expect')
        if exp and (exp['device_id'] != dev or exp['coords'] != coords):
            _log().error('TTL_CHIP_BIND rank=%s: bound chip %s gave device_id=%s coords=%s, the chipmap expects %s',
                         self.rank, _BIND.get('chip'), dev, coords, exp)
        try:  # host-CPU state of the worker (a zygote-forked worker inherits the zygote's, not the raylet's)
            import torch
            _log().info('TTL_WORKER_CPU rank=%s torch_threads=%s interop=%s OMP_NUM_THREADS=%s os_threads=%d '
                        'affinity=%d', self.rank, torch.get_num_threads(), torch.get_num_interop_threads(),
                        os.environ.get('OMP_NUM_THREADS'), len(os.listdir('/proc/self/task')),
                        len(os.sched_getaffinity(0)))
        except Exception as e:  # noqa: BLE001
            _log().info('TTL_WORKER_CPU rank=%s unavailable: %r', self.rank, e)

    tw.TPUWorker.init_device = init_device


def _pinned_pg(self, hosts, orig):
    import ray
    from vllm.v1.executor.ray_utils import _wait_until_pg_ready
    pc = self.parallel_config
    if pc.placement_group is not None:
        return orig(self)
    if not ray.is_initialized():
        rt = json.loads(os.environ.get('TTL_RAY_RUNTIME_ENV') or '{}') or None
        # log_to_driver off: on this shared cluster the driver would also receive every unattributed log line of the
        # other tenants' hosts (libtpu compiler logs), late; the host scripts collect this job's worker logs instead.
        ray.init(address=os.environ.get('RAY_ADDRESS') or 'auto', runtime_env=rt, ignore_reinit_error=True,
                 log_to_driver=os.environ.get('TTL_RAY_LOG_TO_DRIVER') == '1')
        job_id = ray.get_runtime_context().get_job_id()
        _log().info('TTL_RAY_PIN ray job_id=%s', job_id)
        if os.environ.get('TTL_RAY_JOBID_FILE'):
            with open(os.environ['TTL_RAY_JOBID_FILE'], 'w') as f:
                f.write(job_id + '\n')
    world = pc.world_size
    if world % len(hosts):
        raise ValueError(f'world_size {world} is not a multiple of the {len(hosts)} pinned hosts {hosts}')
    per_host = world // len(hosts)
    specs = [{'TPU': 1.0, f'node:{ip}': 0.001} for ip in hosts for _ in range(per_host)]
    pg = ray.util.placement_group(specs, strategy='PACK')
    _wait_until_pg_ready(pg)
    pc.placement_group = pg
    _log().info('TTL_RAY_PIN placement_group=%s bundles=%s', pg.id.hex(), specs)


def _patch_executors(hosts):
    try:
        from vllm_torchtpu.executors import ray_distributed_executor as v1
        from vllm_torchtpu.executors import ray_distributed_executor_v2 as v2
    except Exception as e:  # noqa: BLE001  (no Ray in this venv: nothing to pin)
        _log().warning('TTL_RAY_PIN executors not importable: %r', e)
        return
    for cls in (v1.RayDistributedExecutor, v2.RayDistributedExecutorV2):
        orig = cls._initialize_ray_cluster
        if getattr(orig, '_ttl', False):
            continue

        def init(self, _orig=orig):
            return _pinned_pg(self, hosts, _orig)

        init._ttl = True
        cls._initialize_ray_cluster = init

    def _slice_host_layout(self, device_str):
        per_host = self.parallel_config.world_size // len(hosts)
        _log().info('TTL_RAY_PIN slice hosts %s, %d chips each', hosts, per_host)
        return list(hosts), {ip: per_host for ip in hosts}

    v2.RayDistributedExecutorV2._slice_host_layout = _slice_host_layout
