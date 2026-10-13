"""Host-KV transport on emulated CPU meshes: TP8 source to TP16 destination."""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=16")

import queue
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
except ImportError:  # pragma: no cover - optional local dependency
    jax = None

from hetero import kv_transport

PAGE, HEAD_DIM, ORIGINAL, RAW_HEADS, BLOCKS = 8, 4, 4, 16, 12


class FullAttentionSpec:
    def __init__(self, heads):
        self.block_size = self.storage_block_size = PAGE
        self.head_size = HEAD_DIM
        self.num_kv_heads = heads


def semantics(runner, name, spec):
    if name != "layer.0":
        return None
    return {"order": "gemma4-rpa-consecutive-repeat", "original_num_kv_heads": ORIGINAL,
            "effective_num_kv_heads": spec.num_kv_heads, "original_head_dim": HEAD_DIM,
            "attention_tp_size": runner.tp}


def runner(tp, arrays):
    mesh = Mesh(np.array(jax.devices()[:tp]), ("model",))
    sharding = NamedSharding(mesh, PartitionSpec(None, None, "model", None, None))
    specs = {"layer.0": FullAttentionSpec(max(ORIGINAL, tp)),
             "layer.1": FullAttentionSpec(RAW_HEADS)}
    group = SimpleNamespace(layer_names=["layer.0", "layer.1"],
                            kv_cache_spec=SimpleNamespace(kv_cache_specs=specs))
    return SimpleNamespace(
        tp=tp, kv_caches=[jax.device_put(a, sharding) for a in arrays],
        layer_name_to_kvcache_index={"layer.0": 0, "layer.1": 1},
        kv_cache_config=SimpleNamespace(kv_cache_groups=[group]))


def addressing(block_ids, computed):
    logical = list(range(len(block_ids)))
    ranges = [[p * PAGE, min((p + 1) * PAGE, computed)] for p in logical]
    return {"num_computed_tokens": computed,
            "groups": [{"group_id": 0, "block_ids": block_ids,
                        "logical_block_indices": logical, "valid_token_ranges": ranges}]}


@unittest.skipIf(jax is None or len(jax.devices()) < 16, "needs 16 emulated JAX devices")
class HostKVTransport(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.canonical = rng.standard_normal((BLOCKS, PAGE, ORIGINAL, 2, HEAD_DIM), np.float32)
        self.raw = rng.standard_normal((BLOCKS, PAGE, RAW_HEADS, 2, HEAD_DIM), np.float32)
        self.source = runner(8, [np.repeat(self.canonical, 2, axis=2), self.raw])
        self.destination_before = [
            rng.standard_normal((BLOCKS, PAGE, 16, 2, HEAD_DIM), np.float32),
            rng.standard_normal((BLOCKS, PAGE, RAW_HEADS, 2, HEAD_DIM), np.float32)]
        self.destination = runner(16, self.destination_before)
        self.store = {}
        self.messages = queue.Queue()
        patches = [
            mock.patch.object(kv_transport, "_native_head_semantics", semantics),
            mock.patch.object(kv_transport, "_put", self.put),
            mock.patch.object(kv_transport, "_fetch",
                              lambda notifier, txn, keys: [self.store[self.refs[k]] for k in keys]),
            mock.patch.object(kv_transport, "_notify",
                              lambda notifier, txn, message: self.messages.put(message)),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def put(self, value):
        key = f"ref-{len(self.store)}"
        self.store[key] = value
        return key

    def notices(self, count):
        return [self.messages.get(timeout=30) for _ in range(count)]

    def export(self, source_addressing, budget):
        staging, cursor, chunks, layouts, stages = {}, 0, 0, [], []
        while True:
            gathered = kv_transport.export_gather(self.source, staging, "t", source_addressing,
                                                  cursor, budget, notifier=None)
            if cursor == 0:
                layouts = [gathered["layout"]]
            chunks += 1
            cursor = gathered["next"]
            if cursor >= gathered["total"]:
                break
        for message in self.notices(chunks):
            self.assertNotIn("error", message)
            stages.append(message)
        self.assertEqual([key for key in staging if key != "_executor"], [])
        return kv_transport.assemble_manifest(layouts, source_addressing, stages), chunks

    def import_(self, destination_addressing, payload, budget):
        payload, self.refs = kv_transport.split_refs(payload)
        staging, written = {}, 0
        result = kv_transport.import_advance(
            self.destination, staging, "t", commit=False, start=0, budget_bytes=budget,
            notifier=None, addressing=destination_addressing, payload=payload)
        cursor, outstanding = result["next"], 1
        if cursor < result["total"]:  # the actor keeps two chunks staging
            result = kv_transport.import_advance(
                self.destination, staging, "t", commit=False, start=cursor,
                budget_bytes=budget, notifier=None)
            cursor, outstanding = result["next"], outstanding + result["submitted"]
        while outstanding:
            (message,) = self.notices(1)
            self.assertNotIn("error", message)
            result = kv_transport.import_advance(
                self.destination, staging, "t", commit=True, start=cursor,
                budget_bytes=budget, notifier=None)
            written += result["committed"]["arrays_written"]
            cursor = result["next"]
            outstanding += result["submitted"] - 1
        self.assertEqual([key for key in staging if key != "_executor"], [])
        return written

    def test_tp8_to_tp16_moves_one_canonical_copy_in_chunks(self):
        computed = 2 * PAGE + 5
        source_blocks, destination_blocks = [3, 5, 6], [7, 2, 9]
        payload, chunks = self.export(addressing(source_blocks, computed), budget=1)
        self.assertEqual(chunks, 2)
        valid = np.arange(3 * PAGE) < computed
        expected_heads = self.canonical[source_blocks] * valid.reshape(3, PAGE, 1, 1, 1)
        expected_raw = self.raw[source_blocks] * valid.reshape(3, PAGE, 1, 1, 1)
        # Only one replica of every original head crosses the host.
        self.assertEqual(payload["bytes"], expected_heads.nbytes + expected_raw.nbytes)

        written = self.import_(addressing(destination_blocks, computed), payload, budget=1)
        self.assertEqual(written, 2)
        heads, raw = (np.asarray(c) for c in self.destination.kv_caches)
        np.testing.assert_array_equal(heads[destination_blocks],
                                      np.repeat(expected_heads, 4, axis=2))
        np.testing.assert_array_equal(raw[destination_blocks], expected_raw)
        untouched = [b for b in range(BLOCKS) if b not in destination_blocks]
        np.testing.assert_array_equal(heads[untouched], self.destination_before[0][untouched])
        np.testing.assert_array_equal(raw[untouched], self.destination_before[1][untouched])

    def test_disagreeing_source_replica_is_refused(self):
        corrupted = np.repeat(self.canonical, 2, axis=2)
        corrupted[3, 0, 1, 0, 0] += 1.0  # second replica of original head 0
        self.source = runner(8, [corrupted, self.raw])
        kv_transport.export_gather(self.source, {}, "t", addressing([3], PAGE), 0, 1 << 30,
                                   notifier=None)
        (message,) = self.notices(1)
        self.assertRegex(message["error"], "repeated KV heads disagree")

    def test_changed_computed_prefix_is_refused_before_upload(self):
        payload, _ = self.export(addressing([3, 5], 2 * PAGE), budget=1 << 30)
        with self.assertRaisesRegex(ValueError, "computed prefix differs"):
            self.import_(addressing([7, 2], 2 * PAGE - 1), payload, budget=1 << 30)
        np.testing.assert_array_equal(np.asarray(self.destination.kv_caches[0]),
                                      self.destination_before[0])


if __name__ == "__main__":
    unittest.main()


class ChunkCompletionOrder(unittest.IsolatedAsyncioTestCase):
    """Two chunks in flight: a fast worker may finish chunk 1 before a slow one finishes 0."""

    async def asyncSetUp(self):
        import asyncio
        from hetero.engine_actor import HeteroEngine

        cls = getattr(HeteroEngine, "__ray_actor_class__", HeteroEngine)
        self.engine = object.__new__(cls)
        self.engine._kv_notices = {"t": asyncio.Queue()}
        self.engine._kv_early = {}

    async def notify(self, process_index, chunk, **extra):
        await self.engine.omp_kv_notify("t", {"kind": "import", "process_index": process_index,
                                              "chunk": chunk, **extra})

    async def test_later_chunk_is_held_until_requested(self):
        await self.notify(0, 0)
        await self.notify(0, 1)
        await self.notify(1, 0)
        await self.notify(1, 1)
        first = await self.engine._kv_chunk("t", "import", 2, 0)
        second = await self.engine._kv_chunk("t", "import", 2, 1)
        self.assertEqual([(m["process_index"], m["chunk"]) for m in first], [(0, 0), (1, 0)])
        self.assertEqual([(m["process_index"], m["chunk"]) for m in second], [(0, 1), (1, 1)])

    async def test_duplicate_and_error_completions_fail_closed(self):
        await self.notify(0, 0)
        await self.notify(0, 0)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            await self.engine._kv_chunk("t", "import", 2, 0)
        await self.notify(1, 1, error="ValueError: boom")
        with self.assertRaisesRegex(RuntimeError, "boom"):
            await self.engine._kv_chunk("t", "import", 2, 1)
