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
"""
import json
import os

_done = False


def _log():
    from vllm.logger import init_logger
    return init_logger('vllm.ttl_ray_pin')


def _local_ip():
    from vllm.utils.network_utils import get_ip
    return get_ip()


def register():
    global _done
    import sys
    prefix = sys.modules.get('pm_rewind_ray')
    if prefix is not None and prefix.preparing_cpu():
        return  # Never capture this host/job's IP, TPU_WORKER_ID or runtime_env.
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
