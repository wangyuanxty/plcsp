# -*- coding: utf-8 -*-
"""Independent feasibility check for one RGA best schedule.

Checks machine eligibility, job precedence (+ transport), machine
non-overlap, and recomputes idle time from the final starts.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plcsp_v5"))

from baselines.agv_baselines import decode, rga_instance  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances",
                    default="plcsp_v5/instance/test_Instance_J6M6E2.pkl")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--gen", type=int, default=100)
    args = ap.parse_args()

    data = pickle.load(open(ROOT / args.instances, "rb"))
    t, p, tt, edge = [x[args.instance] for x in data]
    n_task, n_machine = t.shape
    n_job = n_task // n_machine
    n_ops = n_machine

    r = rga_instance(t, p, tt, edge, seed=args.seed, pop=args.pop,
                     gen=args.gen, agv=False, return_schedule=True)
    os_seq, ms = r["os_seq"], r["ms"]
    out = decode(t, p, tt, edge, os_seq, ms, agv=False)
    start, finish = out["start"], out["finish"]
    violations = []

    # 1) every task once, and every job has n_ops occurrences
    counts = [0] * n_task
    op_count = [0] * n_job
    for j in os_seq:
        j = int(j)
        oi = op_count[j]
        task = j * n_ops + oi
        counts[task] += 1
        op_count[j] = oi + 1
    if any(c != 1 for c in counts) or any(c != n_ops for c in op_count):
        violations.append("task count != 1")

    # 2) eligibility
    for task in range(n_task):
        m = int(ms[task])
        if t[task][m] < 0:
            violations.append(f"task {task} ineligible machine {m}")

    # 3) job precedence + transport
    for j in range(n_job):
        for oi in range(1, n_ops):
            task = j * n_ops + oi
            prev = task - 1
            pm, cm = int(ms[prev]), int(ms[task])
            tr = 0.0 if pm == cm else float(tt[pm][cm])
            if start[task] + 1e-6 < finish[prev] + tr:
                violations.append(
                    f"precedence task {task}: start {start[task]} < "
                    f"{finish[prev]}+{tr}")

    # 4) machine non-overlap
    for m in range(n_machine):
        tasks = sorted(
            [k for k in range(n_task) if int(ms[k]) == m],
            key=lambda k: start[k])
        for a, b in zip(tasks, tasks[1:]):
            if start[b] + 1e-6 < finish[a]:
                violations.append(
                    f"overlap machine {m}: task {a} -> {b}")

    # 5) idle from final starts, independent of decode internals
    idle = 0.0
    for m in range(n_machine):
        tasks = [k for k in range(n_task) if int(ms[k]) == m]
        if not tasks:
            continue
        idle += max(finish[k] for k in tasks) - sum(
            finish[k] - start[k] for k in tasks)

    print("instance", args.instance)
    print("obj", round(out["obj"], 4), "idle_dec", round(out["idleT"], 4),
          "idle_independent", round(idle, 4))
    print("violations", len(violations))
    for v in violations[:20]:
        print(" ", v)


if __name__ == "__main__":
    main()
