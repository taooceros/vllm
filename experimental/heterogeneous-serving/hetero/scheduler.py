"""Migration-aware scheduler for the heterogeneous fabric.

Subclasses the installed asynchronous scheduler rather than replacing it: native
per-iteration batching, sampling and physical allocation stay authoritative.
Three migration capabilities are added.

**Request fencing.** ``omp_fence`` stops *this request only* from being
scheduled again while every other request keeps flowing through the native
asynchronous batch queue.  The fence does not free KV: the source keeps its
blocks until the gateway commits ownership and authorises release.

**Continuation snapshots.** ``omp_snapshot`` reads the native request state
(raw token ids, the prompt/output boundary, computed positions, remaining
generation budget and stop state) so the gateway can hand a *continuation* to a
destination instead of a new user prompt.

**Prepared continuations.** BlockPool pages are held before source quiescence,
then atomically converted to KVCacheManager-owned full-growth tables. A prepared
Request stays off the scheduler until ownership commits, and resumes as a native
PREEMPTED request with its original prompt boundary and full generated history.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import RequestStatus

UNSUPPORTED_CONTINUATION_FIELDS = (
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
)


class HeteroScheduler(AsyncScheduler):
    """Native scheduler plus gateway-authorised fencing and snapshots."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.omp_fenced: Set[str] = set()
        self.omp_inflight_batches: Dict[int, Any] = {}
        self.omp_iterations: List[Dict[str, Any]] = []
        self.omp_capture = False
        self.omp_iteration_seq = 0
        self.omp_released: Dict[str, float] = {}
        # Async scheduling runs with batch_queue_size > 1, so batch N retires
        # after batch N+1 has been scheduled.  Rows are therefore matched to
        # their own SchedulerOutput by identity, never to "the last row".
        self.omp_rows_by_output: Dict[int, tuple[Dict[str, Any], float]] = {}
        self.omp_unmatched_retirements = 0
        self.omp_replay_requests: Dict[str, int] = {}
        self.omp_native_replay_requests: Dict[str, int] = {}
        self.omp_continuations: Dict[str, Dict[str, Any]] = {}
        self.omp_ownership: Dict[str, Dict[str, Any]] = {}
        self.omp_qualification_checkpoints: Dict[str, tuple[int, int]] = {}

    def add_request(self, request: Any) -> None:
        checkpoint = self.omp_qualification_checkpoints.get(request.request_id)
        if checkpoint is not None and (request.max_tokens != checkpoint[1] or request.num_output_tokens):
            raise RuntimeError("qualification must preserve the original generation limit and empty output")
        transaction_id = getattr(request, "omp_transaction_id", None)
        if transaction_id is not None:
            record = self.omp_continuations[transaction_id]
            if record["phase"] != "committed" or record["request"] is not request:
                raise RuntimeError("continuation has no committed native ownership")
        else:
            rid = self.omp_external_id(request)
            if rid in self.omp_ownership:
                raise RuntimeError("request identity already has native ownership history")
            self.omp_ownership[rid] = {"epoch": 0, "owned": True,
                                       "native_id": request.request_id}
        super().add_request(request)

    def omp_reserve_continuation(self, payload: Dict[str, Any]) -> int:
        """Hold real BlockPool pages; caller holds the native step lock."""
        txn, rid = payload["transaction_id"], payload["request_id"]
        prior = self.omp_continuations.get(txn)
        if prior is not None:
            if prior["identity"] != payload or prior["phase"] == "aborted":
                raise RuntimeError("conflicting or aborted continuation reservation")
            return prior["reserved_count"]
        epoch = int(payload["destination_epoch"])
        owner = self.omp_ownership.get(rid)
        if owner and (owner["owned"] or epoch <= owner["epoch"]):
            raise RuntimeError("stale epoch or request still owned by destination")
        if any(r["identity"]["request_id"] == rid
               and r["phase"] in ("reserved", "prepared")
               for r in self.omp_continuations.values()):
            raise RuntimeError("request already reserved under another transaction")
        manager = self.kv_cache_manager
        native_id = f"{rid}#e{epoch}"
        maximum = int(payload.get("reserve_tokens") or payload["max_context_tokens"])
        count = manager.coordinator.get_num_blocks_to_allocate(
            request_id=native_id, num_tokens=maximum,
            new_computed_blocks=manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0, total_computed_tokens=0,
            num_local_computed_tokens=0, num_tokens_main_model=maximum,
        )
        blocks = manager.block_pool.get_new_blocks(count)
        self.omp_continuations[txn] = {
            "identity": dict(payload), "native_id": native_id,
            "phase": "reserved", "blocks": blocks,
            "reserved_count": count, "request": None,
        }
        return count

    def omp_prepare_continuation(self, transaction_id: str, request: Any) -> Any:
        """Atomically convert held pages to a native full-growth block table."""
        record = self.omp_continuations[transaction_id]
        if record["phase"] != "reserved" or request.request_id != record["native_id"]:
            raise RuntimeError("continuation does not match native reservation")
        manager = self.kv_cache_manager
        # No step may run between returning the hold and native allocation. No
        # parallel block table or allocator is introduced: KVCacheManager owns it.
        manager.block_pool.free_blocks(record["blocks"])
        record["blocks"] = []
        try:
            allocated = manager.allocate_slots(
                request, num_new_tokens=int(record["identity"].get("reserve_tokens")
                                            or record["identity"]["max_context_tokens"]),
                delay_cache_blocks=True, has_scheduled_reqs=False,
            )
            if allocated is None:
                raise RuntimeError("native reservation conversion failed")
        except BaseException:
            manager.free(request)
            record["phase"] = "aborted"
            raise
        request.omp_transaction_id = transaction_id
        request.external_req_id = record["identity"]["request_id"]
        record.update(phase="prepared", request=request)
        return manager.get_blocks(request.request_id)

    def omp_commit_continuation(self, transaction_id: str) -> None:
        record = self.omp_continuations[transaction_id]
        if record["phase"] == "committed":
            return
        if record["phase"] != "prepared":
            raise RuntimeError("native continuation is not prepared")
        identity = record["identity"]
        self.omp_ownership[identity["request_id"]] = {
            "epoch": identity["destination_epoch"], "owned": True,
            "native_id": record["native_id"], "transaction_id": transaction_id,
        }
        record["phase"] = "committed"

    def omp_abort_continuation(self, transaction_id: str) -> None:
        record = self.omp_continuations.get(transaction_id)
        if record is None or record["phase"] == "aborted":
            return
        if record["phase"] == "committed":
            raise RuntimeError("cannot roll back committed native ownership")
        if record["request"] is not None:
            self.kv_cache_manager.free(record["request"])
            record["request"] = None
        else:
            self.kv_cache_manager.block_pool.free_blocks(record["blocks"])
        record["blocks"] = []
        record["phase"] = "aborted"

    # ------------------------------------------------------------ identity

    def omp_internal_id(self, request_id: str) -> Optional[str]:
        """Resolve the logical gateway ID to its native ownership-epoch request.

        Initial requests use the gateway ID directly. Continuations retain it in
        external_req_id while their native IDs distinguish successive epochs.
        No caller parses native IDs or reconstructs a suffix.
        """
        if request_id in self.requests:
            return request_id
        for internal, request in self.requests.items():
            if getattr(request, "external_req_id", None) == request_id:
                return internal
        return None

    @staticmethod
    def omp_external_id(request: Any) -> str:
        return getattr(request, "external_req_id", None) or request.request_id

    def omp_arm_qualification_checkpoint(self, request_id: str, checkpoint_tokens: int,
                                         max_tokens: int) -> None:
        """Stop issuing at an exact greedy output count, without changing stop budget."""
        if not 0 < checkpoint_tokens < max_tokens:
            raise ValueError("checkpoint must precede the original generation limit")
        if self.num_spec_tokens != 0 or self.num_sampled_tokens_per_step != 1:
            raise ValueError("exact checkpoints require one-token non-speculative native sampling")
        if request_id in self.omp_ownership or request_id in self.omp_qualification_checkpoints:
            raise ValueError("qualification request identity is already used")
        self.omp_qualification_checkpoints[request_id] = (checkpoint_tokens, max_tokens)

    # ------------------------------------------------------------- fencing

    def omp_fence(self, request_id: str) -> bool:
        internal = self.omp_internal_id(request_id)
        if internal is None:
            return False
        self.omp_fenced.add(internal)
        return True

    def omp_unfence(self, request_id: str, *, worker_state_restored: bool = False) -> bool:
        """Resume resident KV through native cached-request re-admission.

        Fencing can evict a TPU request from input_batch before its last async
        sampled token replaces the worker placeholder. Merely making RUNNING
        schedulable again cannot restore that token or a stale device batch row.
        """
        internal = self.omp_internal_id(request_id) or request_id
        existed = internal in self.omp_fenced
        request = self.requests.get(internal)
        if existed and request is not None:
            if (self.omp_is_inflight(request_id) or request.num_output_placeholders
                    or request.num_in_flight_tokens):
                raise RuntimeError("cannot resume source with issued native work pending")
            if request.status in (RequestStatus.RUNNING, RequestStatus.PREEMPTED):
                if not worker_state_restored:
                    raise RuntimeError("fenced source needs authoritative worker history before resume")
                if request.status == RequestStatus.RUNNING:
                    self.running.remove(request)
                    # Cached-state re-admission, NOT memory preemption: retain
                    # Request, KV, computed prefix, budget, counters and epoch.
                    request.status = RequestStatus.PREEMPTED
                    self.waiting.prepend_request(request)
                self.prev_step_scheduled_req_ids.discard(internal)
            elif request.num_computed_tokens or request.num_output_tokens:
                raise RuntimeError("fenced source has no resumable native request state")
            # An untouched waiting request has no worker state to restore.
            # Preserve its native waiting/grammar status rather than promote it.
        self.omp_fenced.discard(internal)
        self.omp_qualification_checkpoints.pop(internal, None)
        return existed

    def omp_is_inflight(self, request_id: str) -> bool:
        internal = self.omp_internal_id(request_id) or request_id
        return any(
            internal in output.num_scheduled_tokens
            for output in self.omp_inflight_batches.values()
        )

    def omp_fence_state(self, request_id: str) -> Dict[str, Any]:
        internal = self.omp_internal_id(request_id)
        return {
            "request_id": request_id,
            "internal_id": internal,
            "fenced": internal in self.omp_fenced if internal else False,
            "inflight": self.omp_is_inflight(request_id),
            "known": internal is not None,
            "inflight_batches": len(self.omp_inflight_batches),
        }

    # ------------------------------------------------------------ stepping

    def schedule(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter() if self.omp_capture else 0.0
        for request_id, (target, _) in self.omp_qualification_checkpoints.items():
            request = self.requests.get(request_id)
            if request is None:
                continue
            issued_outputs = request.num_output_tokens + request.num_output_placeholders
            if issued_outputs > target:
                raise RuntimeError("native checkpoint overshot; refusing to truncate sampled history")
            if issued_outputs == target:
                # Native async batches already issued must retire normally. Hide
                # the request from BOTH running and waiting admission before any
                # additional sample is scheduled; never drop or truncate outputs.
                self.omp_fenced.add(request_id)
        original = list(self.running) if self.omp_fenced else None
        waiting = None
        if original is not None:
            self.running = [r for r in original if r.request_id not in self.omp_fenced]
            waiting = {}
            for name in ("waiting", "skipped_waiting"):
                queue = getattr(self, name)
                waiting[name] = list(queue)
                queue.remove_requests(r for r in waiting[name] if r.request_id in self.omp_fenced)
        try:
            output = super().schedule(*args, **kwargs)
        finally:
            if original is not None:
                from vllm.v1.core.sched.request_queue import create_request_queue

                self.running = _restore_residency(original, self.running, self.omp_fenced)
                for name, before in waiting.items():
                    merged = _restore_residency(before, list(getattr(self, name)), self.omp_fenced)
                    queue = create_request_queue(self.policy)
                    for request in merged:
                        queue.add_request(request)
                    setattr(self, name, queue)
        # Retain the output itself: its identity cannot be reused while pending.
        # Fencing and telemetry both retire this exact batch, not a FIFO guess.
        self.omp_inflight_batches[id(output)] = output
        if self.omp_capture:
            scheduled = time.perf_counter()
            row = self._omp_record_schedule(output)
            row["schedule_duration_s"] = round(scheduled - started, 6)
            self.omp_rows_by_output[id(output)] = (row, scheduled)
        return output

    def _preempt_request(self, request: Any, timestamp: float) -> None:
        retired = max(0, request.num_computed_tokens - request.num_in_flight_tokens)
        self.omp_native_replay_requests[request.request_id] = max(
            retired, self.omp_native_replay_requests.get(request.request_id, 0))
        super()._preempt_request(request, timestamp)

    def update_from_output(self, scheduler_output: Any, model_runner_output: Any) -> Any:
        result = super().update_from_output(scheduler_output, model_runner_output)
        self.omp_inflight_batches.pop(id(scheduler_output), None)
        for request_id in scheduler_output.num_scheduled_tokens:
            if request_id not in self.requests:
                self.omp_native_replay_requests.pop(request_id, None)
        if self.omp_capture:
            self._omp_record_update(scheduler_output, model_runner_output, result)
        return result

    # ----------------------------------------------------------- telemetry

    def omp_start_capture(self) -> None:
        """Starts at a drained boundary; caller holds the engine step lock."""
        if self.omp_capture:
            raise RuntimeError("iteration capture is already active")
        if self.omp_inflight_batches:
            raise RuntimeError("cannot start capture with native batches in flight")
        self.omp_iterations = []
        self.omp_iteration_seq = 0
        self.omp_rows_by_output = {}
        self.omp_unmatched_retirements = 0
        self.omp_capture = True

    def omp_stop_capture(self) -> List[Dict[str, Any]]:
        """Stops only after every issued native batch has retired."""
        if self.omp_inflight_batches:
            raise RuntimeError("cannot stop capture with native batches in flight")
        self.omp_capture = False
        return self.omp_iterations

    def omp_capture_audit(self) -> Dict[str, Any]:
        """One-to-one schedule/retirement accounting for the captured series."""
        rows = self.omp_iterations
        return {
            "rows": len(rows),
            "retired_rows": sum(1 for r in rows if r.get("completed_at_s") is not None),
            "open_rows": len(self.omp_rows_by_output),
            "unmatched_retirements": self.omp_unmatched_retirements,
            "inflight_batches": len(self.omp_inflight_batches),
            "latency_clock": "perf_counter",
            "latency_scope": "host_schedule_return_to_retirement_including_queue",
        }

    def _omp_record_schedule(self, output: Any) -> Dict[str, Any]:
        self.omp_iteration_seq += 1
        scheduled = output.num_scheduled_tokens
        pool = self.kv_cache_manager.block_pool
        prefill_tokens = decode_tokens = replayed_tokens = native_replayed_tokens = 0
        prefill_requests = decode_requests = 0
        for request_id, count in scheduled.items():
            request = self.requests[request_id]
            # Native async scheduling has already advanced this request. Historical
            # output replay is teacher-forced prefill, not new useful decode. The
            # final sampled-but-uncomputed token is the first genuine decode slot.
            start = request.num_computed_tokens - count
            prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
            prefill = min(count, max(0, prefill_end - start))
            decode = count - prefill
            prefill_tokens += prefill
            decode_tokens += decode
            prefill_requests += int(prefill > 0)
            decode_requests += int(decode > 0)
            replay_prefix = self.omp_replay_requests.get(self.omp_external_id(request), 0)
            native_prefix = self.omp_native_replay_requests.get(request_id, 0)
            native_replayed_tokens += min(count, max(0, native_prefix - start))
            # Work redone because destination preempted is not migration replay a
            # second time. The as-yet-unreplayed migration suffix still is.
            replayed_tokens += max(0, min(start + count, replay_prefix) - max(start, native_prefix))
        phase = (
            "mixed" if prefill_tokens and decode_tokens else
            "prefill" if prefill_tokens else
            "decode" if decode_tokens else "control"
        )
        row = {
            "iteration": self.omp_iteration_seq,
            "scheduled_at_s": time.time(),
            "scheduled_tokens": int(sum(scheduled.values())),
            "scheduled_requests": len(scheduled),
            "phase": phase,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "prefill_requests": prefill_requests,
            "decode_requests": decode_requests,
            "replayed_tokens": replayed_tokens,
            "native_preemption_replayed_tokens": native_replayed_tokens,
            "new_requests": len(output.scheduled_new_reqs),
            "running_requests": len(self.running),
            "waiting_requests": len(self.waiting),
            "fenced_requests": len(self.omp_fenced),
            "free_kv_blocks_after_schedule": int(pool.get_num_free_blocks()),
            "preempted_request_ids": sorted(
                getattr(output, "preempted_req_ids", set()) or []
            ),
            "context_tokens": int(
                sum(getattr(r, "num_computed_tokens", 0) for r in self.running)
            ),
        }
        self.omp_iterations.append(row)
        return row

    def _omp_record_update(
        self, scheduler_output: Any, model_runner_output: Any, result: Any
    ) -> None:
        entry = self.omp_rows_by_output.pop(id(scheduler_output), None)
        if entry is None:
            # Never attribute a retirement to an unrelated row, including a
            # zero-token cleanup batch. Every native schedule must be retired.
            self.omp_unmatched_retirements += 1
            return
        row, scheduled = entry
        row["completed_at_s"] = time.time()
        row["scheduled_to_output_s"] = round(time.perf_counter() - scheduled, 6)
        sampled = getattr(model_runner_output, "sampled_token_ids", None) or []
        row["sampled_tokens"] = sum(len(x or []) for x in sampled)
        emitted = 0
        finished: List[str] = []
        groups = result.values() if isinstance(result, dict) else [result]
        for group in groups:
            for item in getattr(group, "outputs", None) or []:
                emitted += len(getattr(item, "new_token_ids", None) or [])
                if getattr(item, "finish_reason", None) is not None:
                    finished.append(item.request_id)
        row["emitted_output_tokens"] = emitted
        row["finished_request_ids"] = finished

    # ---------------------------------------------------------- snapshots

    def omp_request_view(self, request_id: str) -> Optional[Dict[str, Any]]:
        internal = self.omp_internal_id(request_id)
        request = self.requests.get(internal) if internal else None
        if request is None:
            return None
        params = request.sampling_params
        prompt = list(request.prompt_token_ids or [])
        output = list(request.output_token_ids or [])
        return {
            "request_id": request_id,
            "status": str(request.status),
            "prompt_tokens": len(prompt),
            "output_tokens": len(output),
            "num_computed_tokens": int(request.num_computed_tokens),
            "num_tokens": int(request.num_tokens),
            "num_output_placeholders": int(getattr(request, "num_output_placeholders", 0)),
            "max_tokens": int(getattr(params, "max_tokens", 0) or 0),
            "remaining_tokens": max(
                0, int(getattr(params, "max_tokens", 0) or 0) - len(output)
            ),
            "allocated_blocks": self._omp_allocated_blocks(internal),
        }

    def _omp_allocated_blocks(self, request_id: str) -> int:
        try:
            blocks = self.kv_cache_manager.coordinator.get_blocks(request_id)
        except Exception:
            return 0
        total = 0
        for group in blocks or []:
            try:
                total += len(group)
            except TypeError:
                total += 1
        return total

    def omp_snapshot(self, request_id: str) -> Dict[str, Any]:
        """Complete continuation state for one fenced request."""
        internal = self.omp_internal_id(request_id)
        request = self.requests.get(internal) if internal else None
        if request is None:
            raise KeyError(f"request {request_id} is not resident on this engine")
        if internal not in self.omp_fenced:
            raise RuntimeError(f"request {request_id} must be fenced before snapshotting")
        if self.omp_is_inflight(request_id):
            raise RuntimeError(f"request {request_id} still has issued native work in flight")
        params = request.sampling_params
        placeholders = int(getattr(request, "num_output_placeholders", 0))
        if placeholders:
            raise RuntimeError(
                f"request {request_id} has {placeholders} generated-but-uncomputed tokens; "
                "retire them before snapshotting"
            )
        unsupported = _unsupported_continuation(params)
        for field in ("mm_features", "lora_request", "prompt_embeds", "pooling_params"):
            if getattr(request, field, None) is not None and (
                field != "mm_features" or request.mm_features
            ):
                unsupported.append(f"{field}_continuation_not_supported")
        prompt = list(request.prompt_token_ids or [])
        output = list(request.output_token_ids or [])
        return {
            "request_id": request_id,
            "prompt_token_ids": prompt,
            "generated_token_ids": output,
            "prompt_len": len(prompt),
            "generated_len": len(output),
            "num_computed_tokens": int(request.num_computed_tokens),
            "sampling": {
                field: getattr(params, field)
                for field in (
                    "max_tokens", "min_tokens", "temperature", "top_p", "top_k", "min_p",
                    "seed", "ignore_eos", "stop", "stop_token_ids", "presence_penalty",
                    "frequency_penalty", "repetition_penalty", "n", "detokenize",
                    "skip_special_tokens", "spaces_between_special_tokens",
                    "include_stop_str_in_output", "logit_bias", "allowed_token_ids",
                    "bad_words", "skip_reading_prefix_cache",
                )
                if hasattr(params, field)
            },
            "remaining_tokens": max(
                0, int(getattr(params, "max_tokens", 0) or 0) - len(output)
            ),
            "allocated_blocks": self._omp_allocated_blocks(internal),
            "unsupported_continuation": unsupported,
            "snapshot_at_s": time.time(),
        }

    def omp_block_ids(self, request_id: str) -> Dict[str, Any]:
        """Physical KV block ids backing a request's computed prefix."""
        internal = self.omp_internal_id(request_id)
        request = self.requests.get(internal) if internal else None
        if request is None:
            raise KeyError(f"request {request_id} is not resident on this engine")
        if internal not in self.omp_fenced or self.omp_is_inflight(request_id):
            raise RuntimeError("KV addressing requires a fenced, retired request")
        manager = self.kv_cache_manager
        blocks_by_group = manager.get_blocks(internal).blocks
        specs = manager.kv_cache_config.kv_cache_groups
        if len(blocks_by_group) != len(specs):
            raise RuntimeError("native cache groups and block tables disagree")
        computed = int(request.num_computed_tokens)
        groups = []
        for group_id, (spec, blocks) in enumerate(zip(specs, blocks_by_group)):
            block_size = int(spec.kv_cache_spec.block_size)
            group = {"group_id": group_id, "block_ids": [],
                     "logical_block_indices": [], "valid_token_ranges": []}
            for logical, block in enumerate(blocks):
                start = logical * block_size
                if start >= computed:
                    break
                if block.is_null:
                    continue
                group["block_ids"].append(int(block.block_id))
                group["logical_block_indices"].append(logical)
                group["valid_token_ranges"].append([start, min(start + block_size, computed)])
            groups.append(group)
        return {"request_id": request_id, "internal_id": internal,
                "num_computed_tokens": computed, "groups": groups}

    def omp_release(self, request_id: str, source_epoch: int) -> Dict[str, Any]:
        """Release only the fenced, retired request at its native ownership epoch."""
        internal = self.omp_internal_id(request_id)
        request = self.requests.get(internal) if internal else None
        owner = self.omp_ownership.get(request_id)
        if request is None or not owner or not owner["owned"] or owner["epoch"] != source_epoch:
            raise RuntimeError("source no longer owns this request epoch")
        if internal not in self.omp_fenced or self.omp_is_inflight(request_id):
            raise RuntimeError("source release requires a fenced, retired request")
        before = self.kv_cache_manager.block_pool.get_num_free_blocks()
        self.finish_requests([internal], RequestStatus.FINISHED_ABORTED)
        self.omp_fenced.discard(internal)
        owner["owned"] = False
        after = self.kv_cache_manager.block_pool.get_num_free_blocks()
        self.omp_released[request_id] = time.time()
        return {
            "request_id": request_id, "released": True,
            "freed_blocks": int(after - before), "free_blocks_after": int(after),
        }

    def omp_candidates(
        self,
        *,
        min_generated_tokens: int,
        min_context_tokens: int,
        min_remaining_tokens: int,
    ) -> List[Dict[str, Any]]:
        """Local migration candidates from *observed* state only.

        Nothing here reads a realized future output length; only public request
        parameters and progress already made.
        """
        rows = []
        for request in self.running:
            internal = request.request_id
            if internal in self.omp_fenced:
                continue
            rid = self.omp_external_id(request)
            params = request.sampling_params
            generated = len(request.output_token_ids or [])
            context = int(request.num_computed_tokens)
            remaining = max(0, int(getattr(params, "max_tokens", 0) or 0) - generated)
            if generated < min_generated_tokens:
                continue
            if context < min_context_tokens:
                continue
            if remaining < min_remaining_tokens:
                continue
            rows.append(
                {
                    "request_id": rid,
                    "prompt_tokens": len(request.prompt_token_ids or []),
                    "generated_tokens": generated,
                    "context_tokens": context,
                    "remaining_tokens": remaining,
                    "allocated_blocks": self._omp_allocated_blocks(internal),
                    "unsupported_continuation": _unsupported_continuation(params),
                }
            )
        rows.sort(key=lambda r: (-r["context_tokens"], r["request_id"]))
        return rows

    def omp_pressure(self) -> Dict[str, Any]:
        pool = self.kv_cache_manager.block_pool
        return {
            "running": len(self.running),
            "waiting": len(self.waiting),
            "fenced": len(self.omp_fenced),
            "free_blocks": int(pool.get_num_free_blocks()),
            "total_blocks": int(pool.num_gpu_blocks),
            "inflight_batches": len(self.omp_inflight_batches),
        }


def _restore_residency(
    original: List[Any], after: List[Any], fenced: Set[str]
) -> List[Any]:
    """Reinstates fenced requests in their original residency position.

    Requests the native scheduler removed while the fence was applied (finished
    or preempted) are not resurrected, and requests it admitted from the waiting
    queue are appended.
    """
    after_ids = {r.request_id for r in after}
    merged: List[Any] = []
    for request in original:
        if request.request_id in fenced or request.request_id in after_ids:
            merged.append(request)
    seen = {r.request_id for r in merged}
    merged.extend(r for r in after if r.request_id not in seen)
    return merged


def _unsupported_continuation(params: Any) -> List[str]:
    """Continuation features this implementation refuses to reset or fake."""
    reasons: List[str] = []
    if float(getattr(params, "temperature", 0.0)) > 0.0:
        reasons.append("stochastic_sampling_rng_state_not_transferable")
    for field in UNSUPPORTED_CONTINUATION_FIELDS:
        default = 1.0 if field == "repetition_penalty" else 0.0
        if float(getattr(params, field, default)) != default:
            reasons.append(f"{field}_history_not_transferable")
    if getattr(params, "stop", None):
        reasons.append("string_stop_state_not_transferable")
    if getattr(params, "structured_outputs", None) is not None:
        reasons.append("structured_output_state_not_transferable")
    if int(getattr(params, "n", 1)) != 1:
        reasons.append("multi_sample_requests_not_supported")
    if any(getattr(params, field, None) is not None
           for field in ("logprobs", "prompt_logprobs", "logprob_token_ids")):
        reasons.append("logprob_accounting_not_transferable")
    # Snapshot sampling omits output_kind and reconstructs the native cumulative default.
    if hasattr(params, "output_kind") and getattr(params.output_kind, "name", None) != "CUMULATIVE":
        reasons.append("noncumulative_frontend_output_not_supported")
    for field in ("extra_args", "thinking_token_budget", "repetition_detection"):
        if getattr(params, field, None) is not None:
            reasons.append(f"{field}_state_not_transferable")
    return reasons
