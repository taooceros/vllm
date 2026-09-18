"""
Data Models for Admission Control, Scheduling, and Hardware Profiles.
====================================================================
Provides typed, validated representations of requests, physical hardware limits,
and deterministic execution schedule plans.
"""

from dataclasses import dataclass, field, asdict
import json
import os
from typing import List, Dict, Tuple, Any, Optional, Sequence

from admission_control.core.constants import (
    TP8_DEFAULT_KV_CAPACITY_TOKENS,
    DEFAULT_XLA_BUCKETS,
    EMPIRICAL_STEP_LATENCIES_S,
    DEFAULT_ALPHA_S,
    DEFAULT_BETA_S,
    DEFAULT_STEP_TOKEN_BUDGET,
    DEFAULT_CHUNK_SIZE_CAP,
    DEFAULT_BLOCK_SIZE,
    BATCH_COHORT_SIZE,
)


@dataclass(frozen=True)
class AdmissionDecision:
    """Evaluation result of an admission check with diagnostic reason and metadata."""
    admitted: bool
    guard_name: str = ""
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.admitted

    def to_dict(self) -> Dict[str, Any]:
        return {
            "admitted": self.admitted,
            "guard_name": self.guard_name,
            "reason": self.reason,
            "details": self.details,
        }


@dataclass
class Request:
    """Represents a conversational inference request with known or estimated token demands."""
    req_id: int
    prompt_len: int
    decode_len: int
    arrival_time: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_len + self.decode_len

    def to_dict(self) -> Dict[str, Any]:
        return {
            "req_id": self.req_id,
            "prompt_len": self.prompt_len,
            "decode_len": self.decode_len,
            "arrival_time": self.arrival_time,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Request":
        return cls(
            req_id=int(data.get("req_id", 0)),
            prompt_len=int(data.get("prompt_len", 0)),
            decode_len=int(data.get("decode_len", 0)),
            arrival_time=int(data.get("arrival_time", 0)),
            metadata=data.get("metadata", {}),
        )

@dataclass(frozen=True)
class BatchCohort:
    """Immutable cohort of requests processed together under cohort-isolated scheduling."""
    batch_id: int
    requests: Tuple[Request, ...]
    arrival_step: int = 0

    def __post_init__(self) -> None:
        if len(self.requests) > BATCH_COHORT_SIZE:
            raise ValueError(
                f"Cohort size {len(self.requests)} exceeds maximum cohort invariant "
                f"BATCH_COHORT_SIZE={BATCH_COHORT_SIZE}."
            )
        for r in self.requests:
            r_bid = getattr(r, "batch_id", None)
            if r_bid is not None and r_bid != self.batch_id:
                raise ValueError(
                    f"Cross-cohort interleaving rejected: request {getattr(r, 'req_id', r)} "
                    f"has batch_id={r_bid}, expected cohort batch_id={self.batch_id}."
                )
    def __len__(self) -> int:
        return len(self.requests)

    def __iter__(self):
        return iter(self.requests)

    def __getitem__(self, idx):
        return self.requests[idx]

    def to_list(self) -> List[Request]:
        return list(self.requests)

    def reorder(self, ordered_requests: Sequence[Request]) -> "BatchCohort":
        if len(ordered_requests) != len(self.requests):
            raise ValueError(
                f"Cross-cohort interleaving rejected: reordered count {len(ordered_requests)} "
                f"does not match cohort size {len(self.requests)}."
            )
        orig_ids = {getattr(r, "req_id", id(r)) for r in self.requests}
        new_ids = {getattr(r, "req_id", id(r)) for r in ordered_requests}
        if orig_ids != new_ids:
            raise ValueError(
                f"Cross-cohort interleaving rejected: reordered requests do not match "
                f"cohort {self.batch_id} requests."
            )
        for r in ordered_requests:
            r_bid = getattr(r, "batch_id", None)
            if r_bid is not None and r_bid != self.batch_id:
                raise ValueError(
                    f"Cross-cohort interleaving rejected: request {getattr(r, 'req_id', r)} "
                    f"has batch_id={r_bid}, expected cohort batch_id={self.batch_id}."
                )
        return BatchCohort(batch_id=self.batch_id, requests=tuple(ordered_requests), arrival_step=self.arrival_step)

    @classmethod
    def from_requests(cls, requests: Sequence[Request], batch_id: int = 0, arrival_step: int = 0) -> "BatchCohort":
        return cls(batch_id=batch_id, requests=tuple(requests), arrival_step=arrival_step)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "requests": [r.to_dict() for r in self.requests],
            "arrival_step": self.arrival_step,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchCohort":
        reqs = [Request.from_dict(r) for r in data.get("requests", [])]
        return cls(
            batch_id=int(data.get("batch_id", 0)),
            requests=tuple(reqs),
            arrival_step=int(data.get("arrival_step", 0)),
        )

@dataclass
class ActiveRequest:
    """Dynamic state of a currently active (running or paused) request in the scheduler."""
    req_id: Any
    prompt_len: int
    decode_len: int
    tokens_decoded: int = 0
    batch_id: int = 0
    arrival_step: int = 0
    admit_step: Optional[int] = None
    finish_step: Optional[int] = None
    state: str = "RUNNING"  # RUNNING, PAUSED, FINISHED, PREEMPTED
    preemption_count: int = 0
    discarded_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def current_tokens(self) -> int:
        return self.prompt_len + self.tokens_decoded

    @property
    def remaining_decode_tokens(self) -> int:
        return max(0, self.decode_len - self.tokens_decoded)

    @property
    def is_finished(self) -> bool:
        return self.tokens_decoded >= self.decode_len

    def to_dict(self) -> Dict[str, Any]:
        return {
            "req_id": self.req_id,
            "prompt_len": self.prompt_len,
            "decode_len": self.decode_len,
            "tokens_decoded": self.tokens_decoded,
            "batch_id": self.batch_id,
            "arrival_step": self.arrival_step,
            "admit_step": self.admit_step,
            "finish_step": self.finish_step,
            "state": self.state,
            "preemption_count": self.preemption_count,
            "discarded_tokens": self.discarded_tokens,
            "metadata": self.metadata,
        }


@dataclass
class SystemState:
    """Snapshot of execution engine state at a given scheduling step."""
    step: int = 0
    current_time_s: float = 0.0
    capacity_tokens: int = TP8_DEFAULT_KV_CAPACITY_TOKENS
    used_tokens: int = 0
    max_batch_size: int = 128
    block_size: int = DEFAULT_BLOCK_SIZE
    current_batch_id: int = 0
    total_preemptions: int = 0
    total_discarded_tokens: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def free_tokens(self) -> int:
        return max(0, self.capacity_tokens - self.used_tokens)

    @property
    def memory_utilization_pct(self) -> float:
        return (self.used_tokens / self.capacity_tokens * 100.0) if self.capacity_tokens > 0 else 0.0


@dataclass
class HardwareSpec:
    """
    Physical hardware specifications and execution constraints on Cloud TPU.
    Incorporates static XLA compilation buckets and empirical decode step latencies.
    """
    capacity_tokens: int = TP8_DEFAULT_KV_CAPACITY_TOKENS
    max_batch_size: int = 128
    xla_buckets: Tuple[int, ...] = DEFAULT_XLA_BUCKETS
    alpha_s: float = DEFAULT_ALPHA_S
    beta_s: float = DEFAULT_BETA_S
    recompute_penalty: float = 1.0
    empirical_latencies_s: Optional[Dict[int, float]] = None
    profile_source: Optional[str] = None
    step_token_budget: int = DEFAULT_STEP_TOKEN_BUDGET
    chunk_size_cap: int = DEFAULT_CHUNK_SIZE_CAP
    block_size: int = DEFAULT_BLOCK_SIZE

    def __post_init__(self):
        if self.empirical_latencies_s is None:
            self.empirical_latencies_s = dict(EMPIRICAL_STEP_LATENCIES_S)

    @classmethod
    def from_profile_json(
        cls,
        profile_path: str,
        capacity_tokens: int = TP8_DEFAULT_KV_CAPACITY_TOKENS,
        max_batch_size: int = 128,
        xla_buckets: Optional[Tuple[int, ...]] = None,
        recompute_penalty: float = 1.0,
    ) -> "HardwareSpec":
        """Constructs a HardwareSpec driven directly by real empirical hardware benchmark JSON."""
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        empirical: Dict[int, float] = {}
        if "topology_2x4_tp8" in data and "decode_batch_sweep_s512" in data["topology_2x4_tp8"]:
            sweep = data["topology_2x4_tp8"]["decode_batch_sweep_s512"]
            for item in sweep:
                bs = int(item["batch_size"])
                lat_s = float(item["step_latency_ms"]) / 1000.0
                empirical[bs] = lat_s
        elif "decode_batch_sweep" in data:
            for item in data["decode_batch_sweep"]:
                bs = int(item["batch_size"])
                lat_s = float(item["step_latency_ms"]) / 1000.0
                empirical[bs] = lat_s

        buckets = xla_buckets
        if buckets is None:
            buckets = tuple(sorted(b for b in [16, 32, 64, 128, 256] if b <= max_batch_size))
            if not buckets:
                buckets = (max_batch_size,)

        return cls(
            capacity_tokens=capacity_tokens,
            max_batch_size=max_batch_size,
            xla_buckets=buckets,
            empirical_latencies_s=empirical if empirical else None,
            profile_source=os.path.basename(profile_path),
            recompute_penalty=recompute_penalty,
        )

    def get_bucket(self, active_count: int) -> int:
        """Finds the smallest compiled XLA static shape bucket >= active_count."""
        if active_count <= 0:
            return 0
        for b in sorted(self.xla_buckets):
            if b >= active_count:
                return b
        return max(self.xla_buckets)

    def step_latency(self, active_count: int) -> float:
        """Calculates exact physical execution latency for a step with active_count streams."""
        if active_count <= 0:
            return 0.0
        b = self.get_bucket(active_count)
        if self.empirical_latencies_s and b in self.empirical_latencies_s:
            return self.empirical_latencies_s[b]
        return self.alpha_s + self.beta_s * b

    def padding_slots(self, active_count: int) -> int:
        """Calculates padded bubble slots for a step with active_count streams."""
        if active_count <= 0:
            return 0
        return self.get_bucket(active_count) - active_count


@dataclass
class SchedulePlan:
    """
    Deterministic execution schedule plan produced by an admission control or oracle solver.
    Can be hot-swapped in memory on live benchmark workers without reloading model weights.
    """
    start_times: Dict[int, int]
    chunk_sizes: Dict[int, List[int]] = field(default_factory=dict)
    pause_intervals: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    policy_name: str = "Oracle Profile Schedule"
    makespan_steps: int = 0
    total_time_s: float = 0.0
    goodput_tok_s: float = 0.0
    peak_mem_tokens: int = 0
    peak_mem_pct: float = 0.0
    padding_waste_pct: float = 0.0
    preemptions: int = 0
    discarded_tokens: int = 0
    step_active_profile: List[int] = field(default_factory=list)
    step_memory_profile: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_admissions_at_step(self, step: int) -> List[int]:
        """Returns list of req_ids scheduled to start at the given step."""
        return [req_id for req_id, s in self.start_times.items() if s == step]

    def get_pauses_at_step(self, step: int) -> List[int]:
        """Returns list of req_ids scheduled to pause at the given step."""
        pauses = []
        for req_id, intervals in self.pause_intervals.items():
            for t_pause, _ in intervals:
                if t_pause == step:
                    pauses.append(req_id)
        return pauses

    def get_resumptions_at_step(self, step: int) -> List[int]:
        """Returns list of req_ids scheduled to resume at the given step."""
        resumptions = []
        for req_id, intervals in self.pause_intervals.items():
            for _, t_resume in intervals:
                if t_resume == step:
                    resumptions.append(req_id)
        return resumptions

    def is_paused_at_step(self, req_id: int, step: int) -> bool:
        """Returns True if req_id is in-flight but frozen in HBM at step."""
        for t_pause, t_resume in self.pause_intervals.get(req_id, []):
            if t_pause <= step < t_resume:
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "makespan_steps": self.makespan_steps,
            "total_time_s": self.total_time_s,
            "goodput_tok_s": self.goodput_tok_s,
            "peak_mem_tokens": self.peak_mem_tokens,
            "peak_mem_pct": self.peak_mem_pct,
            "padding_waste_pct": self.padding_waste_pct,
            "preemptions": self.preemptions,
            "discarded_tokens": self.discarded_tokens,
            "start_times": self.start_times,
            "chunk_sizes": self.chunk_sizes,
            "pause_intervals": self.pause_intervals,
            "metadata": self.metadata,
        }

    def save(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchedulePlan":
        start_times = {int(k): int(v) for k, v in data.get("start_times", {}).items()}
        chunk_sizes = {int(k): v for k, v in data.get("chunk_sizes", {}).items()}
        pause_intervals = {
            int(k): [tuple(item) for item in v]
            for k, v in data.get("pause_intervals", {}).items()
        }
        return cls(
            start_times=start_times,
            chunk_sizes=chunk_sizes,
            pause_intervals=pause_intervals,
            policy_name=data.get("policy_name", "Oracle Profile Schedule"),
            makespan_steps=data.get("makespan_steps", 0),
            total_time_s=data.get("total_time_s", 0.0),
            goodput_tok_s=data.get("goodput_tok_s", 0.0),
            peak_mem_tokens=data.get("peak_mem_tokens", 0),
            peak_mem_pct=data.get("peak_mem_pct", 0.0),
            padding_waste_pct=data.get("padding_waste_pct", 0.0),
            preemptions=data.get("preemptions", 0),
            discarded_tokens=data.get("discarded_tokens", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TokenProfile:
    """Summary of token lengths across a batch or workload trace."""
    num_requests: int
    total_prompt_tokens: int
    total_decode_tokens: int
    total_demand_tokens: int
    min_prompt: int
    max_prompt: int
    avg_prompt: float
    min_decode: int
    max_decode: int
    avg_decode: float

    @classmethod
    def from_requests(cls, requests: List[Request]) -> "TokenProfile":
        if not requests:
            return cls(0, 0, 0, 0, 0, 0, 0.0, 0, 0, 0.0)
        prompts = [r.prompt_len for r in requests]
        decodes = [r.decode_len for r in requests]
        tot_p = sum(prompts)
        tot_d = sum(decodes)
        return cls(
            num_requests=len(requests),
            total_prompt_tokens=tot_p,
            total_decode_tokens=tot_d,
            total_demand_tokens=tot_p + tot_d,
            min_prompt=min(prompts),
            max_prompt=max(prompts),
            avg_prompt=tot_p / len(requests),
            min_decode=min(decodes),
            max_decode=max(decodes),
            avg_decode=tot_d / len(requests),
        )
