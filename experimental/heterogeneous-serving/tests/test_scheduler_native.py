"""HeteroScheduler against the real vLLM Scheduler/AsyncScheduler (skipped without vllm).

Covers the private scheduler surface hetero relies on: the waiting queues the fence hides and
restores, ``_preempt_request`` (called with ``drop_stale_output``), the zero-skip contract through
``_get_new_block_ids_to_zero``, and fence -> unfence resume. Pure CPU, config-only tiny Llama.
"""
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import importlib.util

HAVE_VLLM = importlib.util.find_spec("vllm") is not None
if HAVE_VLLM:  # real symbols: an API-drift ImportError must fail, not skip
    import torch
    from vllm.config import (CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig)
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager
    from vllm.utils.hashing import sha256
    from hetero.scheduler import HeteroScheduler

BLOCK = 16
TINY_LLAMA = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 64,
              "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
              "num_hidden_layers": 2, "vocab_size": 128, "max_position_embeddings": 512,
              "rms_norm_eps": 1e-5, "torch_dtype": "float16"}


def make_scheduler(directory: str, num_blocks: int = 64, max_num_seqs: int = 8):
    Path(directory, "config.json").write_text(json.dumps(TINY_LLAMA))
    model = ModelConfig(model=directory, skip_tokenizer_init=True, dtype="float16", seed=1,
                        max_model_len=512)
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs, max_num_batched_tokens=512, max_model_len=512,
        enable_chunked_prefill=True, async_scheduling=True,
        is_encoder_decoder=model.is_encoder_decoder, watermark=0.0)
    cache = CacheConfig(block_size=BLOCK, gpu_memory_utilization=0.9, cache_dtype="auto",
                        enable_prefix_caching=False)
    config = VllmConfig(scheduler_config=scheduler_config, model_config=model, cache_config=cache,
                        parallel_config=ParallelConfig())
    spec = FullAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32)
    kv = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[],
                       kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)])
    cache.num_gpu_blocks = num_blocks
    return HeteroScheduler(vllm_config=config, kv_cache_config=kv, block_size=BLOCK, log_stats=True,
                           structured_output_manager=StructuredOutputManager(config))


def make_request(request_id: str, prompt_len: int = 20, max_tokens: int = 40):
    init_none_hash(sha256)  # idempotent
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    params.update_from_generation_config({}, 100)
    return Request(request_id=request_id, prompt_token_ids=[1] * prompt_len, sampling_params=params,
                   pooling_params=None, block_hasher=get_request_block_hasher(BLOCK, sha256))


def step(scheduler, token: int = 7):
    """One schedule + retire cycle with every scheduled request sampling ``token``."""
    output = scheduler.schedule()
    ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(output, ModelRunnerOutput(
        req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
        sampled_token_ids=[[token] for _ in ids], logprobs=None, prompt_logprobs_dict={},
        pooler_output=[]))
    return output


@unittest.skipUnless(HAVE_VLLM, "vllm not installed")
class HeteroSchedulerNativeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def scheduler(self, **kwargs):
        return make_scheduler(self._tmp.name, **kwargs)

    def test_overridden_signatures_accept_native_calls(self):
        native = inspect.signature(AsyncScheduler._preempt_request)
        self.assertIn("drop_stale_output", native.parameters)
        ours = inspect.signature(HeteroScheduler._preempt_request)
        self.assertEqual(list(ours.parameters), list(native.parameters))
        for name in ("waiting", "kv_holding_waiting"):
            self.assertTrue(hasattr(self.scheduler(), name), name)

    def test_fence_hides_request_and_unfence_resumes_it(self):
        scheduler = self.scheduler()
        scheduler.add_request(make_request("a"))
        scheduler.add_request(make_request("b"))
        step(scheduler)
        step(scheduler)
        scheduler.omp_fence("a")
        output = step(scheduler)
        self.assertNotIn("a", output.num_scheduled_tokens)
        self.assertIn("b", output.num_scheduled_tokens)
        self.assertIn("a", {r.request_id for r in scheduler.running})  # residency restored
        self.assertTrue(scheduler.omp_unfence("a", worker_state_restored=True))
        resumed = step(scheduler)
        self.assertIn("a", resumed.num_scheduled_tokens)

    def test_fence_of_waiting_request_restores_native_queue(self):
        scheduler = self.scheduler(max_num_seqs=1)
        scheduler.add_request(make_request("a"))
        scheduler.add_request(make_request("w"))
        step(scheduler)
        self.assertEqual([r.request_id for r in scheduler.waiting], ["w"])
        scheduler.omp_fence("w")
        output = step(scheduler)
        self.assertNotIn("w", output.num_scheduled_tokens)
        self.assertEqual([r.request_id for r in scheduler.waiting], ["w"])

    def test_native_preemption_passes_drop_stale_output(self):
        scheduler = self.scheduler()
        scheduler.add_request(make_request("a"))
        step(scheduler)
        step(scheduler)
        request = scheduler.requests["a"]
        self.assertEqual(request.status, RequestStatus.RUNNING)
        scheduler.running.remove(request)
        scheduler._preempt_request(request, 0.0, drop_stale_output=True)
        self.assertEqual(request.status, RequestStatus.PREEMPTED)
        self.assertIn("a", scheduler.omp_native_replay_requests)

    def test_skip_zero_contract(self):
        scheduler = self.scheduler()
        scheduler.needs_kv_cache_zeroing = True
        pending = [3, 4, 5]
        scheduler.kv_cache_manager.take_new_block_ids = lambda: list(pending)
        self.assertEqual(scheduler._get_new_block_ids_to_zero(), [3, 4, 5])
        # registered ids (KV imported into them) are not zeroed by the next drain only
        scheduler._skip_zero_block_ids.update({4, 9})
        self.assertEqual(scheduler._get_new_block_ids_to_zero(), [3, 5])
        self.assertEqual(scheduler._skip_zero_block_ids, set())
        self.assertEqual(scheduler._get_new_block_ids_to_zero(), [3, 4, 5])
        scheduler._skip_zero_block_ids.update({3, 4, 5})
        self.assertIsNone(scheduler._get_new_block_ids_to_zero())
        scheduler.needs_kv_cache_zeroing = False
        scheduler._skip_zero_block_ids.add(3)
        self.assertIsNone(scheduler._get_new_block_ids_to_zero())


if __name__ == "__main__":
    unittest.main()
