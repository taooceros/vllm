"""Ray Job Submission Client & Typed TPU Serving Interface."""

import json, re, time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence
import ray
from ray.job_submission import JobSubmissionClient

from admission_control.core.constants import BATCH_COHORT_SIZE
from admission_control.core.models import BatchCohort, Request

PERSISTENT_ACTOR_NAME = "persistent_tpu_benchmark_worker"
PERSISTENT_NAMESPACE = "omp_benchmark"
_EVENT_REGEX = re.compile(r"\[OMP_EVENT:\s*([A-Za-z0-9_]+)(.*)\]\s*$")
_KEY_VAL_REGEX = re.compile(r'([A-Za-z0-9_]+)=(?:(\[[^\]]*\]|\{[^\}]*\})|"([^"]*)"|(\S+))')


def emit_omp_event(event_name: str, **fields: Any) -> str:
    """Formats and emits a standardized OMP structured lifecycle event to stdout."""
    parts = [f'{k}="{v}"' if isinstance(v, str) and (" " in v or '"' in v) else f"{k}={json.dumps(v) if isinstance(v, (dict, list)) else v}" for k, v in fields.items()]
    line = f"[OMP_EVENT: {event_name}{(' ' + ' '.join(parts)) if parts else ''}]"
    print(line, flush=True)
    return line


@dataclass
class StructuredEvent:
    """Parsed record of an in-band structured OMP event."""
    name: str
    fields: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""

    @classmethod
    def parse(cls, line: str) -> Optional["StructuredEvent"]:
        if not (m := _EVENT_REGEX.search(line)):
            return None
        fields = {}
        for match in _KEY_VAL_REGEX.finditer(m.group(2)):
            val = match.group(3) if match.group(3) is not None else (match.group(2) or match.group(4))
            try: fields[match.group(1)] = json.loads(val)
            except Exception: fields[match.group(1)] = val
        return cls(name=m.group(1).strip(), fields=fields, raw=line.strip())


class TPUJobSubmitter:
    """Direct Ray JobSubmissionClient wrapper for TPU benchmark jobs."""
    def __init__(self, ray_dashboard_url: str = "http://127.0.0.1:8265"):
        self.dashboard_url = ray_dashboard_url
        self.client = JobSubmissionClient(ray_dashboard_url)

    def submit_and_stream(self, entrypoint: str, working_dir: str, runtime_env: Optional[Dict[str, Any]] = None, job_timeout_s: float = 600.0, on_event: Optional[Callable[[StructuredEvent], None]] = None) -> str:
        job_id = self.client.submit_job(entrypoint=entrypoint, runtime_env=runtime_env or {"working_dir": working_dir})
        emit_omp_event("JOB_SUBMITTED", job_id=job_id, dashboard=self.dashboard_url)
        return job_id


class TPUServingClient:
    """High-level typed client for direct interaction with persistent Ray TPU workers."""
    def __init__(self, ray_address: Optional[str] = None, actor_name: str = PERSISTENT_ACTOR_NAME, namespace: str = PERSISTENT_NAMESPACE, dashboard_url: str = "http://127.0.0.1:8265"):
        self.ray_address, self.actor_name, self.namespace = ray_address, actor_name, namespace
        self.dashboard_url, self._actor, self._submitter = dashboard_url, None, None

    @property
    def submitter(self) -> TPUJobSubmitter:
        if self._submitter is None:
            self._submitter = TPUJobSubmitter(self.dashboard_url)
        return self._submitter

    def _get_worker_actor(self):
        if not ray.is_initialized():
            ray.init(address=self.ray_address or "auto", ignore_reinit_error=True)
        if self._actor is None:
            self._actor = ray.get_actor(self.actor_name, namespace=self.namespace)
        return self._actor

    def ping(self) -> str:
        return ray.get(self._get_worker_actor().ping.remote())

    def get_status(self) -> Dict[str, Any]:
        return ray.get(self._get_worker_actor().get_status.remote())

    def reset_kv_cache(self) -> str:
        res = ray.get(self._get_worker_actor().reset_kv_cache.remote())
        emit_omp_event("KV_CACHE_RESET", status="SUCCESS")
        return res

    def set_policy(self, policy: Any, **kwargs: Any) -> Dict[str, Any]:
        t0 = time.perf_counter()
        res = ray.get(self._get_worker_actor().set_policy.remote(policy, **kwargs))
        res["client_roundtrip_overhead_s"] = time.perf_counter() - t0
        return res

    def submit_cohort(self, cohort: BatchCohort, policy: Optional[Any] = None, target_concurrency: int = 96, capacity_tokens: int = 120_000, **policy_kwargs: Any) -> Dict[str, Any]:
        if len(cohort) > BATCH_COHORT_SIZE:
            raise ValueError(f"Cohort size {len(cohort)} exceeds BATCH_COHORT_SIZE={BATCH_COHORT_SIZE}.")
        if policy is not None:
            self.set_policy(policy, **policy_kwargs)
        emit_omp_event("COHORT_SUBMITTED", cohort_id=cohort.batch_id, requests=len(cohort), arrival_step=cohort.arrival_step)
        return ray.get(self._get_worker_actor().execute_policy.remote(
            policy="active", batch_stream=[r.to_dict() for r in cohort.requests],
            batch_size=len(cohort), target_concurrency=target_concurrency,
            capacity_tokens=capacity_tokens, **policy_kwargs,
        ))

    def submit_requests(self, requests: Sequence[Request], batch_id: int = 0, arrival_step: int = 0, policy: Optional[Any] = None, target_concurrency: int = 96, capacity_tokens: int = 120_000, **policy_kwargs: Any) -> Dict[str, Any]:
        return self.submit_cohort(BatchCohort.from_requests(requests=requests, batch_id=batch_id, arrival_step=arrival_step), policy=policy, target_concurrency=target_concurrency, capacity_tokens=capacity_tokens, **policy_kwargs)

    def submit_cohort_stream(self, cohorts: Sequence[BatchCohort], policy: Optional[Any] = None, target_concurrency: int = 96, capacity_tokens: int = 120_000, **policy_kwargs: Any) -> List[Dict[str, Any]]:
        if policy is not None:
            self.set_policy(policy, **policy_kwargs)
        return [self.submit_cohort(c, target_concurrency=target_concurrency, capacity_tokens=capacity_tokens, **policy_kwargs) for c in cohorts]
