"""PDR and RGA when one machine needs a repair mid-schedule.

The rule matches the controller's evaluation: at step ``--after`` the machine
that step used needs ``--repair`` time, and every operation placed on it later
pays that repair.  The plan is unchanged; only the durations move.

Usage::

    python -m baselines.repair_compare --instances 0 1 2 --out res.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys

import numpy as np

REPO = __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__)))
sys.path.insert(0, REPO)

from plcsp_v5.baselines.agv_baselines import pdr_schedule, rga_instance
from plcsp_v5.baselines.mip_agv.model_noagv import load_instance
from plcsp_v5.baselines.mip_agv.agv_sim import simulate

POOL = "plcsp_v5/instance/test_Instance_J6M6E2_lu.pkl"
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="6x6",
                    choices=["6x6", "10x6", "10x10"])
    ap.add_argument("--instances", type=int, nargs="+", required=True)
    ap.add_argument("--after", type=int, default=14)
    ap.add_argument("--repair", type=float, default=100.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pool = POOLS[a.size]
    rows = []
    for i in a.instances:
        inst = load_instance(pool, i)
        for name in ("PDR", "RGA"):
            if name == "PDR":
                os_seq, ms = pdr_schedule(inst.t, inst.power, "mor", "spt")
            else:
                r = rga_instance(inst.t, inst.power, inst.travel, inst.edges,
                                 seed=1, pop=50, gen=100, agv=True,
                                 lu_machine=inst.lu_machine,
                                 lu_edge=inst.lu_edge, idle_mode="trailing",
                                 eval_mode="sim", n_agv=6,
                                 return_schedule=True)
                os_seq, ms = r["os_seq"], r["ms"]
            order = to_order(os_seq, inst.n_op)

            base = simulate(inst, order, list(ms), n_agv=6, lu_station=1)
            bad, m, n_late = inflate(inst, order, ms, a.after, a.repair)
            hit = simulate(bad, order, list(ms), n_agv=6, lu_station=1)
            rows.append({"instance": i, "method": name, "machine": m,
                         "late": n_late, "obj": base.objective,
                         "obj_hit": hit.objective, "mk": base.makespan,
                         "mk_hit": hit.makespan})
        print(f"instance {i} done", flush=True)

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh)
    for name in ("PDR", "RGA"):
        sel = [r for r in rows if r["method"] == name]
        b = np.mean([r["obj"] for r in sel])
        h = np.mean([r["obj_hit"] for r in sel])
        print(f"{name}: {b:.2f} -> {h:.2f}  ({100 * (h - b) / b:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())