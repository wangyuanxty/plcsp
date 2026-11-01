"""Batch runner for the fixed-machine joint model.

For each instance: solve the no-AGV stage-A model to get a machine assignment,
then solve the joint model with that assignment fixed.  The joint step decides
the operation starts, the transport legs and the AGV routes together, and it
reports the CP-SAT bound.

Usage::

    D:/anaconda/envs/py312/python.exe -m plcsp_v5.baselines.mip_agv.run_joint_batch \
        --instance plcsp_v5/instance/test_Instance_J6M6E2_lu.pkl \
        --stage_a_time 10 --joint_time 30 --shard 0 --shards 4 \
        --out results/joint_6x6_shard0.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from plcsp_v5.baselines.mip_agv.model_joint import solve_joint
from plcsp_v5.baselines.mip_agv.model_noagv import load_instance, solve

FIELDS = ["idx", "stage_a_obj", "stage_a_status", "joint_obj", "joint_bound",
          "joint_status", "gap_pct", "mk", "pec_mean", "idle", "loaded",
          "empty", "seconds"]


def run_one(path: str, idx: int, args) -> dict:
    started = time.time()
    inst = load_instance(path, idx)
    sched = solve(inst, time_limit=args.stage_a_time, workers=args.workers,
                  objective="full")
    if sched is None:
        return {"idx": idx, "stage_a_status": "no_solution",
                "joint_status": "skipped",
                "seconds": round(time.time() - started, 1)}
    joint = solve_joint(inst, args.n_agv, args.joint_time, args.workers,
                        fix_machines=sched.machine, score=False)
    row = {
        "idx": idx,
        "stage_a_obj": round(sched.objective, 4),
        "stage_a_status": sched.status,
        "joint_obj": round(joint.get("objective", 0.0), 4),
        "joint_bound": round(joint.get("bound", 0.0), 4),
        "joint_status": joint.get("status", "unknown"),
        "mk": round(joint.get("mk", 0.0), 4),
        "pec_mean": round(joint.get("pec_mean", 0.0), 4),
        "idle": round(joint.get("idle", 0.0), 4),
        "loaded": round(joint.get("loaded", 0.0), 4),
        "empty": round(joint.get("empty", 0.0), 4),
        "seconds": round(time.time() - started, 1),
    }
    if joint.get("feasible") and joint["bound"] > 0:
        row["gap_pct"] = round(
            100.0 * (joint["objective"] - joint["bound"]) / joint["bound"], 3)
    else:
        row["gap_pct"] = ""
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n_agv", type=int, default=6)
    ap.add_argument("--stage_a_time", type=float, default=10.0)
    ap.add_argument("--joint_time", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for row in csv.DictReader(open(out, encoding="utf-8")):
            done.add(int(row["idx"]))
    with open(out, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if out.stat().st_size == 0:
            writer.writeheader()
        idx = 0
        while args.limit == 0 or idx < args.limit:
            if idx % args.shards != args.shard:
                idx += 1
                continue
            if idx in done:
                idx += 1
                continue
            try:
                row = run_one(args.instance, idx, args)
            except Exception as exc:              # keep the shard alive
                row = {"idx": idx, "joint_status": f"error: {exc}",
                       "seconds": 0}
            writer.writerow({k: row.get(k, "") for k in FIELDS})
            fh.flush()
            print(idx, row.get("joint_status"), row.get("joint_obj"),
                  flush=True)
            idx += 1


if __name__ == "__main__":
    main()
