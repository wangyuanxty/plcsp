"""Batch runner for the two-stage MIP-AGV baseline.

Solves one shard of the 100 test instances and writes a CSV incrementally.
Run several shards in parallel; each shard uses ``--workers`` CP-SAT threads.

Example::

    PYTHONPATH=/root /root/venv/bin/python -m plcsp_v5.baselines.mip_agv.run_batch \
        --shard 0 --n_shards 4 --workers 8 --out results/mip_agv_100/shard0.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import time

from plcsp_v5.baselines.mip_agv.agv_routing import build_legs, solve_routing
from plcsp_v5.baselines.mip_agv.agv_sim import simulate
from plcsp_v5.baselines.mip_agv.model_noagv import load_instance, solve
from plcsp_v5.baselines.mip_agv.validate_env import task_order_from_starts

FIELDS = ["idx", "stage_a_obj", "rule_obj", "milp_obj", "best_obj",
          "best_source", "makespan", "pec_mean", "idle", "trans",
          "stage_a_status", "seconds"]


def run_one(inst_path: str, idx: int, args) -> dict:
    started = time.time()
    inst = load_instance(inst_path, idx)
    sched = solve(inst, time_limit=args.stage_a_time, workers=args.workers)
    if sched is None:
        return {"idx": idx, "stage_a_status": "no_solution",
                "seconds": round(time.time() - started, 1)}

    order = task_order_from_starts(sched.start, inst.n_job, inst.n_op)
    rule = simulate(inst, order, sched.machine, n_agv=args.n_agv)
    best, best_source = rule, "rule"
    starts = rule.starts
    for it in range(args.iters):
        legs = build_legs(inst, order, sched.machine, starts)
        if not legs:
            break
        plan, _ = solve_routing(inst, legs, args.n_agv,
                                time_limit=args.routing_time,
                                workers=args.workers)
        sim = simulate(
            inst, order, sched.machine, n_agv=args.n_agv,
            pick=lambda step, ctx: plan.get(
                ctx["tau"], (ctx["src_pool"][0],
                             ctx["dst_pool"][0] if ctx["dst_pool"] else -1)))
        if sim.objective < best.objective:
            best, best_source = sim, f"milp{it}"
        starts = sim.starts

    return {
        "idx": idx,
        "stage_a_obj": round(sched.objective, 4),
        "rule_obj": round(rule.objective, 4),
        "milp_obj": round(best.objective, 4),
        "best_obj": round(best.objective, 4),
        "best_source": best_source,
        "makespan": round(best.makespan, 4),
        "pec_mean": round(best.pec_mean, 4),
        "idle": round(best.idle, 4),
        "trans": round(best.trans, 4),
        "stage_a_status": sched.status,
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default=(
        "plcsp_v5/instance/test_Instance_J6M6E2.pkl"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=4)
    ap.add_argument("--n_instances", type=int, default=100)
    ap.add_argument("--n_agv", type=int, default=6)
    ap.add_argument("--stage_a_time", type=float, default=10.0)
    ap.add_argument("--routing_time", type=float, default=8.0)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        fh.flush()
        for idx in range(args.shard, args.n_instances, args.n_shards):
            try:
                row = run_one(args.instance, idx, args)
            except Exception as exc:  # keep the shard alive
                row = {"idx": idx, "stage_a_status": f"error:{exc}"[:120]}
            row = {k: row.get(k, "") for k in FIELDS}
            writer.writerow(row)
            fh.flush()
            print(f"shard {args.shard}: {row}", flush=True)


if __name__ == "__main__":
    main()
