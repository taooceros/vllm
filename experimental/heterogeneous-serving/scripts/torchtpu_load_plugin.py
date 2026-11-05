"""Measurement-only vLLM general plugin (installed into the isolated probe venv by torchtpu_load_host.sh).

Wraps DefaultModelLoader.load_weights to cross-check vLLM's own 'Loading weights took X seconds':
logs, per worker process, the wall time of the call, an extra torch.accelerator.synchronize() after it
(to catch asynchronous host->TPU transfers still in flight), and device HBM used before/after.
It never changes what is loaded.
"""
import time


def register():
    import torch
    from vllm.logger import init_logger
    from vllm.model_executor.model_loader import default_loader as dl
    from vllm.model_executor.model_loader import runai_streamer_loader as rsl

    logger = init_logger('vllm.ttl_probe')

    def hbm():
        try:
            free, total = torch.accelerator.get_memory_info()
            return total - free, total
        except Exception as e:  # noqa: BLE001
            return None, repr(e)

    def wrap_load_weights(cls):
        orig = cls.load_weights
        if getattr(orig, '_ttl_probe', False):
            return

        def wrapped(self, model, model_config):
            u0, total = hbm()
            t0 = time.perf_counter()
            w0 = time.time()
            try:
                return orig(self, model, model_config)
            finally:
                t1 = time.perf_counter()
                u1, _ = hbm()
                try:
                    torch.accelerator.synchronize()
                except Exception as e:  # noqa: BLE001
                    logger.warning('TTL_PROBE synchronize failed: %r', e)
                t2 = time.perf_counter()
                u2, _ = hbm()
                gib = 2.0 ** 30
                fmt = lambda v: 'n/a' if v is None else f'{v / gib:.3f}GiB'  # noqa: E731
                logger.info(
                    'TTL_PROBE load_weights loader=%s wall_start=%.3f call=%.2fs sync_extra=%.2fs call+sync=%.2fs '
                    'hbm_used_before=%s after_call=%s after_sync=%s delta=%s hbm_total=%s',
                    cls.__name__, w0, t1 - t0, t2 - t1, t2 - t0, fmt(u0), fmt(u1), fmt(u2),
                    fmt(None if u0 is None or u2 is None else u2 - u0),
                    fmt(total if isinstance(total, int) else None))

        wrapped._ttl_probe = True
        cls.load_weights = wrapped

    wrap_load_weights(dl.DefaultModelLoader)
    wrap_load_weights(rsl.RunaiModelStreamerLoader)

    # Whole load_model(): initialize_model on the device + load_weights + process_weights_after_loading.
    from vllm.model_executor.model_loader import base_loader as bl
    orig_lm = bl.BaseModelLoader.load_model

    def wrapped_lm(self, *a, **kw):
        u0, total = hbm()
        t0 = time.perf_counter()
        try:
            return orig_lm(self, *a, **kw)
        finally:
            t1 = time.perf_counter()
            try:
                torch.accelerator.synchronize()
            except Exception as e:  # noqa: BLE001
                logger.warning('TTL_PROBE synchronize failed: %r', e)
            t2 = time.perf_counter()
            u2, _ = hbm()
            gib = 2.0 ** 30
            fmt = lambda v: 'n/a' if v is None else f'{v / gib:.3f}GiB'  # noqa: E731
            logger.info(
                'TTL_PROBE load_model call=%.2fs sync_extra=%.2fs call+sync=%.2fs hbm_used_before=%s after_sync=%s delta=%s',
                t1 - t0, t2 - t1, t2 - t0, fmt(u0), fmt(u2), fmt(None if u0 is None or u2 is None else u2 - u0))

    if not getattr(orig_lm, '_ttl_probe', False):
        wrapped_lm._ttl_probe = True
        bl.BaseModelLoader.load_model = wrapped_lm
