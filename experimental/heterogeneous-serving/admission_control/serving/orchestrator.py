"""TPU Cluster Benchmark Orchestrator & Policy Execution Engine."""

import argparse
import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional

import ray
from ray.job_submission import JobSubmissionClient

from admission_control.algorithms.solver import solve_schedule
from admission_control.core.constants import BATCH_COHORT_SIZE
from admission_control.core.models import BatchCohort, HardwareSpec, Request
from admission_control.serving.client import emit_omp_event
from admission_control.serving.worker import TPUPreemptionBenchmarkWorker
from admission_control.tpu.placement import create_node_pinned_placement_group, validate_tp8_v5e_topology
from admission_control.tpu.topology import TPUTopologyDiscovery

MODEL_PATH, TP_SIZE = "/models/gemma-4-31b", 8
PERSISTENT_ACTOR_NAME, PERSISTENT_NAMESPACE = "persistent_tpu_benchmark_worker", "omp_benchmark"


def get_or_create_persistent_worker(
    target_ips: List[str], force_reinit: bool = False, model_path: str = MODEL_PATH, tp_size: int = TP_SIZE, max_num_seqs: int = 256,
) -> Any:
    """Acquires existing hot detached worker actor or provisions a fresh one."""
    if not force_reinit:
        try:
            worker = ray.get_actor(PERSISTENT_ACTOR_NAME, namespace=PERSISTENT_NAMESPACE)
            probe_resp = ray.get(worker.ping.remote())
            print(f"[Orchestrator] Attached to existing hot persistent worker: {probe_resp}", flush=True)
            emit_omp_event("PERSISTENT_ACTOR_ATTACHED", actor_name=PERSISTENT_ACTOR_NAME, probe=probe_resp)
            return worker
        except Exception:
            pass

    print("[Orchestrator] Provisioning node-pinned placement group for persistent worker...", flush=True)
    validate_tp8_v5e_topology(target_ips, tpu_per_node=4.0, expected_chips=8)
    pg = create_node_pinned_placement_group(target_ips, strategy="STRICT_SPREAD")
    worker = TPUPreemptionBenchmarkWorker.options(
        name=PERSISTENT_ACTOR_NAME, namespace=PERSISTENT_NAMESPACE, lifetime="detached",
        placement_group=pg, placement_group_bundle_index=0,
    ).remote(model_path=model_path, tp_size=tp_size, target_ips=target_ips)

    ray.get(worker.initialize_engine.remote(max_num_seqs=max_num_seqs, watermark=0.0))
    emit_omp_event("PERSISTENT_ACTOR_PROVISIONED", actor_name=PERSISTENT_ACTOR_NAME, namespace=PERSISTENT_NAMESPACE)
    return worker


def submit_orchestrator_job(
    entrypoint_args: Optional[List[str]] = None, dashboard_url: str = "http://127.0.0.1:8265",
    working_dir: Optional[str] = None, timeout_s: float = 600.0,
) -> str:
    """Submits orchestrator job directly to Ray cluster using Python JobSubmissionClient SDK."""
    client = JobSubmissionClient(dashboard_url)
    root_dir = working_dir or str(pathlib.Path(__file__).resolve().parent.parent.parent)
    entrypoint = f"python3 -m admission_control.serving.orchestrator {' '.join(entrypoint_args or [])}".strip()
    print(f"[Orchestrator SDK] Submitting job via Ray JobSubmissionClient: {entrypoint}", flush=True)
    job_id = client.submit_job(entrypoint=entrypoint, runtime_env={"working_dir": root_dir})
    emit_omp_event("JOB_SUBMITTED", job_id=job_id, dashboard=dashboard_url)
    return job_id


def run_benchmark_orchestrator(
    workload_json: Optional[str] = None, policy: str = "both", num_requests: int = 64,
    prompt_len: int = 256, output_len: int = 2048, force_reinit: bool = False,
    results_out_dir: str = "admission_control/experiments/results",
    custom_requests: Optional[List[Dict[str, Any]]] = None, cohort: Optional[BatchCohort] = None,
    target_concurrency: int = 64, ray_address: Optional[str] = None,
) -> Dict[str, Any]:
    """Coordinates comparative policy evaluation on Cloud TPU v5e cluster."""
    emit_omp_event("ORCHESTRATION_STARTED", policy=policy, num_requests=num_requests)
    if not ray.is_initialized():
        ray.init(address=ray_address or "auto", ignore_reinit_error=True)

    pair = TPUTopologyDiscovery().select_connected_2host_submesh(validate_tp8=True)
    target_ips = [pair.host_a.node_ip, pair.host_b.node_ip]
    print(f"[Orchestrator] Discovered connected TPU host pair: {target_ips} (Mesh: {pair.mesh_dimension})", flush=True)
    emit_omp_event("TOPOLOGY_DISCOVERED", hosts=target_ips, mesh=pair.mesh_dimension, total_tpus=pair.total_tpus)

    worker = get_or_create_persistent_worker(target_ips=target_ips, force_reinit=force_reinit)

    if cohort is not None:
        active_requests = [r.to_dict() for r in cohort.requests]
    elif custom_requests is not None:
        active_requests = list(custom_requests)
    elif workload_json and os.path.exists(workload_json):
        with open(workload_json, "r", encoding="utf-8") as f:
            active_requests = json.load(f)
    else:
        active_requests = [{"req_id": i, "prompt_len": prompt_len, "decode_len": output_len} for i in range(num_requests)]
    concurrency = len(active_requests)

    results: Dict[str, Any] = {}
    overheads: Dict[str, float] = {}

    # Policy A: Greedy Reactive
    if policy in ("all", "both", "a", "greedy"):
        ray.get(worker.reset_kv_cache.remote())
        t_sw = time.perf_counter()
        ray.get(worker.set_policy.remote("greedy"))
        overheads["switch_to_greedy_overhead_s"] = time.perf_counter() - t_sw
        emit_omp_event("POLICY_EVALUATION_STARTED", policy="greedy", concurrency=concurrency)
        t0 = time.perf_counter()
        res_a = ray.get(worker.run_policy_a_greedy.remote(
            num_requests=concurrency, prompt_len=prompt_len, max_tokens=output_len, custom_requests=active_requests,
        ))
        dur_a = time.perf_counter() - t0
        print(f"[Policy A] Completed in {dur_a:.2f}s | Results: {res_a}", flush=True)
        emit_omp_event("POLICY_EVALUATION_COMPLETED", policy="greedy", total_time_s=round(dur_a, 2))
        results["policy_a_greedy"] = res_a

    # Policy Oracle: Optimal Clairvoyant Planning
    if policy in ("all", "both", "oracle", "oracle_optimal"):
        ray.get(worker.reset_kv_cache.remote())
        hw = HardwareSpec(capacity_tokens=120_000, max_batch_size=128, xla_buckets=(16, 32, 64, 128, 256))
        req_objects = [
            Request(req_id=item.get("req_id", i), prompt_len=item.get("prompt_len", prompt_len), decode_len=item.get("decode_len", output_len))
            for i, item in enumerate(active_requests)
        ]
        plan = solve_schedule(req_objects, hw, method="auto", priority="lpt")
        t_sw = time.perf_counter()
        sw_res = ray.get(worker.set_policy.remote(plan))
        overheads["switch_to_oracle_overhead_s"] = time.perf_counter() - t_sw

        total_demand = sum(r.prompt_len + r.decode_len for r in req_objects)
        avg_demand = (total_demand / len(req_objects)) if req_objects else (prompt_len + output_len)
        safe_cap = int(hw.capacity_tokens // avg_demand)
        oracle_cap = min(concurrency, hw.max_batch_size) if total_demand <= hw.capacity_tokens else max([b for b in hw.xla_buckets if b <= safe_cap] or [16])

        def get_priority_key(item):
            r_id = item.get("req_id", 0)
            st = plan.start_times.get(str(r_id), plan.start_times.get(r_id, 0))
            return (st, -item.get("decode_len", output_len) + 0.006 * item.get("prompt_len", prompt_len), -item.get("decode_len", output_len), item.get("prompt_len", prompt_len))

        oracle_requests = sorted(active_requests, key=get_priority_key)
        emit_omp_event("POLICY_EVALUATION_STARTED", policy="oracle", concurrency=concurrency, cap=oracle_cap)
        t0 = time.perf_counter()
        res_oracle = ray.get(worker.run_policy_f_continuous_slotted_pipeline.remote(
            num_requests=concurrency, prompt_len=prompt_len, max_tokens=output_len, target_concurrency=oracle_cap, custom_requests=oracle_requests,
        ))
        dur_oracle = time.perf_counter() - t0
        res_oracle["policy"] = f"Oracle Continuous Slotted Pipeline (Cap={oracle_cap})"
        print(f"[Policy Oracle] Completed in {dur_oracle:.2f}s | Results: {res_oracle}", flush=True)
        emit_omp_event("POLICY_EVALUATION_COMPLETED", policy="oracle", total_time_s=round(dur_oracle, 2))
        results["policy_oracle_optimal"] = res_oracle

    # Custom or named policy execution (DynamicKnapsack, Predictive, etc.)
    if policy not in ("all", "both", "a", "greedy", "oracle", "oracle_optimal"):
        ray.get(worker.reset_kv_cache.remote())
        t_sw = time.perf_counter()
        sw_res = ray.get(worker.set_policy.remote(policy))
        overheads[f"switch_to_{policy}_overhead_s"] = time.perf_counter() - t_sw
        emit_omp_event("POLICY_EVALUATION_STARTED", policy=policy, concurrency=concurrency)
        t0 = time.perf_counter()
        res_custom = ray.get(worker.execute_policy.remote(
            policy="active", batch_stream=active_requests, batch_size=min(concurrency, BATCH_COHORT_SIZE), target_concurrency=target_concurrency,
        ))
        dur_custom = time.perf_counter() - t0
        emit_omp_event("POLICY_EVALUATION_COMPLETED", policy=policy, total_time_s=round(dur_custom, 2))
        results[f"policy_{policy}"] = res_custom

    out_data = {
        "timestamp": time.time(), "workload": os.path.basename(workload_json) if workload_json else "synthetic_stress",
        "concurrency": concurrency, "results": results, "policy_switch_overheads": overheads,
    }
    os.makedirs(results_out_dir, exist_ok=True)
    out_file = os.path.join(results_out_dir, "tp8_preemption_benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"[Orchestrator] Results written to {out_file}", flush=True)
    emit_omp_event("ORCHESTRATION_COMPLETED", output_file=out_file)
    return out_data


def main():
    parser = argparse.ArgumentParser(description="Cloud TPU v5e Benchmark Orchestrator")
    parser.add_argument("--workload", type=str, default=None, help="Path to workload JSON")
    parser.add_argument("--policy", type=str, default="both", help="Policy to evaluate")
    parser.add_argument("--num-requests", type=int, default=64, help="Number of requests")
    parser.add_argument("--prompt-len", type=int, default=256, help="Prompt token length")
    parser.add_argument("--output-len", type=int, default=2048, help="Output token length")
    parser.add_argument("--force-reinit", action="store_true", help="Force persistent actor reinit")
    parser.add_argument("--results-dir", type=str, default="admission_control/experiments/results")
    parser.add_argument("--remote-submit", action="store_true", help="Submit via Ray JobSubmissionClient")
    parser.add_argument("--dashboard-url", type=str, default="http://127.0.0.1:8265")
    args = parser.parse_args()

    if args.remote_submit:
        forward_args = [
            f"--policy={args.policy}", f"--num-requests={args.num_requests}",
            f"--prompt-len={args.prompt_len}", f"--output-len={args.output_len}",
            f"--results-dir={args.results_dir}",
        ]
        if args.workload: forward_args.append(f"--workload={args.workload}")
        if args.force_reinit: forward_args.append("--force-reinit")
        submit_orchestrator_job(entrypoint_args=forward_args, dashboard_url=args.dashboard_url)
    else:
        run_benchmark_orchestrator(
            workload_json=args.workload, policy=args.policy, num_requests=args.num_requests,
            prompt_len=args.prompt_len, output_len=args.output_len, force_reinit=args.force_reinit,
            results_out_dir=args.results_dir,
        )


if __name__ == "__main__":
    main()
