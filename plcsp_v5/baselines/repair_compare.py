"""PDR and RGA when one machine needs a repair mid-schedule.

The rule matches the controller's evaluation: at step ``--after`` the machine
that step used needs ``--repair`` time, and every operation placed on it later
pays that repair.  The plan is unchanged; only the durations move.

PDR is deterministic and is run once per instance.  RGA is a random-key
genetic algorithm and is run with ``--seeds`` seeds (default 3); the printed
summary is the mean over all instance-seed runs.  The PDR rule pair is
parameterised so that the repair run can use the pair reported in the main
table, for example ``--job-rule mor --machine-rule seyc`` at 10x6.

Usage::

    python -m plcsp_v5.baselines.repair_compare --size 6x6 \
        --instances 0 1 2 --out res.json

    python -m plcsp_v5.baselines.repair_compare --size 10x6 \
        --instances $(seq 0 99) --job-rule mor --machine-rule seyc \
        --workers 8 --out res.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from plcsp_v5.baselines.agv_baselines import pdr_schedule, rga_instance
from plcsp_v5.baselines.mip_agv.model_noagv import load_instance
from plcsp_v5.baselines.mip_agv.agv_sim import simulate

POOLS = {"6x6": "plcsp_v5/instance/test_Instance_J6M6E2_lu.pkl",
         "10x6": "plcsp_v5/instance/test_Instance_J10M6E2_lu.pkl",
         "10x10": "plcsp_v5/instance/test_Instance_J10M10E2_lu.pkl"}


def to_order(os_seq, n_op):
    cnt, order = {}, []
    for j in os_seq:
        o = cnt.get(j, 0)
        cnt[j] = o + 1
        order.append(j * n_op + o)
    return order


def inflate(inst, order, machines, after, repair):
    """Raise the time of the operations placed on the failed machine later."""
    m = int(machines[order[after - 1]])
    late = [int(t) for t in order[after:] if int(machines[t]) == m]
    if not late:
        return inst, m, 0
    t = inst.t.copy()
    t[late, m] += repair
    return dataclasses.replace(inst, t=t), m, len(late)


def _run_one(task):
    """Run PDR or one RGA seed on one instance; return its two rows."""
    (size, instance, method, seed, job_rule, machine_rule,
     after, repair) = task
    inst = load_instance(POOLS[size], instance)
    if method == "PDR":
        os_seq, ms = pdr_schedule(inst.t, inst.power, job_rule, machine_rule)
    else:
        r = rga_instance(inst.t, inst.power, inst.travel, inst.edges,
                         seed=seed, pop=50, gen=100, agv=True,
                         lu_machine=inst.lu_machine,
                         lu_edge=inst.lu_edge, idle_mode="trailing",
                         eval_mode="sim", n_agv=6,
                         return_schedule=True)
        os_seq, ms = r["os_seq"], r["ms"]
    order = to_order(os_seq, inst.n_op)

    base = simulate(inst, order, list(ms), n_agv=6, lu_station=1)
    bad, m, n_late = inflate(inst, order, ms, after, repair)
    hit = simulate(bad, order, list(ms), n_agv=6, lu_station=1)
    return [{"instance": instance, "method": method, "seed": seed,
             "machine": m, "late": n_late, "obj": base.objective,
             "obj_hit": hit.objective, "mk": base.makespan,
             "mk_hit": hit.makespan}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="6x6",
                    choices=["6x6", "10x6", "10x10"])
    ap.add_argument("--instances", type=int, nargs="+", required=True)
    ap.add_argument("--job-rule", default="mor",
                    help="PDR job rule, e.g. mor, fifo, mwkr_t")
    ap.add_argument("--machine-rule", default="spt",
                    help="PDR machine rule, e.g. spt, seyc")
    ap.add_argument("--seeds", type=int, default=3,
                    help="number of RGA seeds; PDR is always one run")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes")
    ap.add_argument("--after", type=int, default=14)
    ap.add_argument("--repair", type=float, default=100.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tasks = []
    for i in a.instances:
        tasks.append((a.size, i, "PDR", None, a.job_rule, a.machine_rule,
                      a.after, a.repair))
        for seed in range(1, a.seeds + 1):
            tasks.append((a.size, i, "RGA", seed, a.job_rule,
                          a.machine_rule, a.after, a.repair))

    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            chunks = list(ex.map(_run_one, tasks))
    else:
        chunks = [_run_one(task) for task in tasks]
    rows = [row for chunk in chunks for row in chunk]

    payload = {"size": a.size, "job_rule": a.job_rule,
               "machine_rule": a.machine_rule, "seeds": a.seeds,
               "after": a.after, "repair": a.repair, "rows": rows}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    for name in ("PDR", "RGA"):
        sel = [r for r in rows if r["method"] == name]
        b = float(np.mean([r["obj"] for r in sel]))
        h = float(np.mean([r["obj_hit"] for r in sel]))
        print(f"{name}: {b:.2f} -> {h:.2f}  ({100 * (h - b) / b:+.1f}%)")
        if name == "RGA" and a.seeds > 1:
            for seed in range(1, a.seeds + 1):
                srows = [r for r in sel if r["seed"] == seed]
                sb = float(np.mean([r["obj"] for r in srows]))
                sh = float(np.mean([r["obj_hit"] for r in srows]))
                print(f"  seed {seed}: {sb:.2f} -> {sh:.2f}  "
                      f"({100 * (sh - sb) / sb:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())