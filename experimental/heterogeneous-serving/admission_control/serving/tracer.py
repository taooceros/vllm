"""
Admission and Preemption Scheduling Decision Tracer for vLLM v1 Engine.

Provides TracedScheduler (subclass of vllm.v1.core.sched.scheduler.Scheduler)
to capture 100% empirical hardware scheduling decisions:
1. Exact admission events (from waiting queue into running)
2. Preemption events (eviction due to KV cache exhaustion)
3. Step-by-step running batch size and queue depth
4. KV cache block utilization over time
5. Request completions and block releases
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

try:
    from vllm.v1.core.sched.scheduler import Scheduler
except ImportError:
    Scheduler = object

DEFAULT_TRACE_FILE = os.environ.get("VLLM_SCHEDULING_TRACE_FILE", "/tmp/vllm_scheduling_decision_trace.jsonl")


class TracedScheduler(Scheduler):
    """
    100% empirical scheduling decision tracer subclassing vLLM v1 Scheduler.
    Loaded dynamically by vLLM via scheduler_cls argument.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_file = DEFAULT_TRACE_FILE
        os.makedirs(os.path.dirname(os.path.abspath(self.trace_file)), exist_ok=True)
        try:
            with open(self.trace_file, "w", encoding="utf-8") as f:
                f.write("")
            print(f"[TracedScheduler] Initialized! Logging scheduling trace to: {self.trace_file}", flush=True)
        except Exception as e:
            print(f"[TracedScheduler Warning] Failed to reset {self.trace_file}: {e}", flush=True)

    def _preempt_request(self, request, timestamp: float):
        prompt_tokens = int(getattr(request, "num_prompt_tokens", 0))
        computed_tokens = int(getattr(request, "num_computed_tokens", 0))
        discarded = max(0, computed_tokens - prompt_tokens)
        req_id = str(getattr(request, "request_id", "unknown"))
        preempt_count = int(getattr(request, "num_preemptions", 0)) + 1
        step = getattr(self, "current_step", 0)

        event = {
            "event_type": "PREEMPTION",
            "step": step,
            "timestamp": round(float(timestamp), 4),
            "req_id": req_id,
            "prompt_tokens": prompt_tokens,
            "computed_tokens": computed_tokens,
            "discarded_tokens": discarded,
            "preemption_index": preempt_count,
            "running_before_evict": len(self.running),
            "waiting_before_prepend": len(self.waiting),
        }

        try:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
            print(
                f"[TracedScheduler PREEMPTION] Step={step} Req={req_id} Prompt={prompt_tokens} "
                f"Discarded={discarded} RunningAfter={len(self.running)}",
                flush=True,
            )
        except Exception as e:
            print(f"[TracedScheduler Error] Failed to write preemption: {e}", flush=True)

        return super()._preempt_request(request, timestamp)

    def schedule(self, throttle_prefills: bool = False):
        output = super().schedule(throttle_prefills=throttle_prefills)

        step = self.current_step
        ts = time.monotonic()
        running_after = len(self.running)
        waiting_after = len(self.waiting)

        free_blocks = getattr(self.kv_cache_manager, "num_free_blocks", 0)
        total_blocks = getattr(self.kv_cache_manager, "num_total_blocks", 1)
        util_pct = round((1.0 - (free_blocks / max(1, total_blocks))) * 100.0, 2)

        new_reqs = []
        if hasattr(output, "scheduled_new_reqs") and output.scheduled_new_reqs:
            for nr in output.scheduled_new_reqs:
                rid = getattr(nr, "req_id", None) or getattr(nr, "request_id", "unknown")
                new_reqs.append(str(rid))

        preempted_ids = []
        if hasattr(output, "preempted_req_ids") and output.preempted_req_ids:
            preempted_ids = [str(pid) for pid in output.preempted_req_ids]

        finished_ids = []
        if hasattr(output, "finished_req_ids") and output.finished_req_ids:
            finished_ids = [str(fid) for fid in output.finished_req_ids]

        step_record = {
            "event_type": "STEP",
            "step": step,
            "timestamp": round(ts, 4),
            "running_bs": running_after,
            "waiting_depth": waiting_after,
            "scheduled_reqs": len(output.num_scheduled_tokens),
            "total_scheduled_tokens": output.total_num_scheduled_tokens,
            "admitted_req_ids": new_reqs,
            "preempted_req_ids": preempted_ids,
            "finished_req_ids": finished_ids,
            "free_kv_blocks": free_blocks,
            "kv_cache_util_pct": util_pct,
        }

        try:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(step_record) + "\n")
            if step == 1 or step % 50 == 0 or new_reqs or preempted_ids:
                print(
                    f"[TracedScheduler Step {step:04d}] Running={running_after:2d} | Waiting={waiting_after:3d} | "
                    f"Admitted={len(new_reqs)} | Preempted={len(preempted_ids)} | KVUtil={util_pct:.1f}% | "
                    f"Toks={output.total_num_scheduled_tokens}",
                    flush=True,
                )
        except Exception as e:
            print(f"[TracedScheduler Error] Failed to write step {step}: {e}", flush=True)

        return output


class SchedulingDecisionTracer:
    """Helper to parse and format the trace generated by TracedScheduler."""

    _trace_path = DEFAULT_TRACE_FILE

    @classmethod
    def install(cls, trace_path: Optional[str] = None):
        """Sets trace path and ensures directory exists."""
        if trace_path:
            cls._trace_path = trace_path
        os.makedirs(os.path.dirname(os.path.abspath(cls._trace_path)), exist_ok=True)
        print(f"[Tracer] Configured TracedScheduler trace file: {cls._trace_path}", flush=True)

    @classmethod
    def parse_trace(cls, trace_path: Optional[str] = None) -> Dict[str, Any]:
        """Parses the JSONL trace file into structured event metrics and summaries."""
        path = trace_path or cls._trace_path
        if not os.path.exists(path):
            return {"error": f"Trace file '{path}' not found."}

        step_events = []
        preemption_events = []
        admission_events = []
        completion_events = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    etype = rec.get("event_type")
                    if etype == "STEP":
                        step_events.append(rec)
                        if rec.get("admitted_req_ids"):
                            for rid in rec["admitted_req_ids"]:
                                admission_events.append({
                                    "step": rec["step"],
                                    "timestamp": rec["timestamp"],
                                    "req_id": rid,
                                    "running_after": rec["running_bs"],
                                    "kv_cache_util_pct": rec["kv_cache_util_pct"],
                                })
                        if rec.get("finished_req_ids"):
                            for rid in rec["finished_req_ids"]:
                                completion_events.append({
                                    "step": rec["step"],
                                    "timestamp": rec["timestamp"],
                                    "req_id": rid,
                                    "running_after": rec["running_bs"],
                                })
                    elif etype == "PREEMPTION":
                        preemption_events.append(rec)
                except Exception:
                    continue

        if not step_events:
            return {"error": "No step events captured in trace."}

        running_sizes = [s["running_bs"] for s in step_events]
        waiting_depths = [s["waiting_depth"] for s in step_events]
        kv_utils = [s["kv_cache_util_pct"] for s in step_events]

        sorted_bs = sorted(running_sizes)
        n = len(sorted_bs)

        summary = {
            "total_steps": len(step_events),
            "total_admissions": len(admission_events),
            "total_preemptions": len(preemption_events),
            "total_completions": len(completion_events),
            "running_batch_size": {
                "mean": round(sum(running_sizes) / max(1, len(running_sizes)), 2),
                "min": min(running_sizes) if running_sizes else 0,
                "max": max(running_sizes) if running_sizes else 0,
                "p50": sorted_bs[int(0.50 * n)] if n else 0,
                "p90": sorted_bs[int(0.90 * n)] if n else 0,
                "p99": sorted_bs[int(0.99 * n)] if n else 0,
            },
            "waiting_queue_depth": {
                "initial": waiting_depths[0] if waiting_depths else 0,
                "max": max(waiting_depths) if waiting_depths else 0,
                "final": waiting_depths[-1] if waiting_depths else 0,
            },
            "kv_cache_utilization_pct": {
                "mean": round(sum(kv_utils) / max(1, len(kv_utils)), 2),
                "peak": max(kv_utils) if kv_utils else 0.0,
                "final": kv_utils[-1] if kv_utils else 0.0,
            },
            "admissions_sample": admission_events[:20],
            "preemptions_log": preemption_events,
            "completions_sample": completion_events[:20],
            "step_timeline_sample": step_events[::max(1, len(step_events) // 50)],
        }
        return summary
