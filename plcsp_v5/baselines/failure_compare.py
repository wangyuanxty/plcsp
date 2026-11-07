"""PDR and RGA objectives with and without the exposure failure model.

The controller is evaluated in its own environment (``eval_st_seed3.py`` with
the ``--machine_failure`` flags).  This script produces the same two readings
for the two baselines, so the three methods can be compared on one table.

Usage::

    python -m baselines.failure_compare --instances 0 1 2 --out res.json
"""
from __future__ import annotations

import argparse
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
FAIL = {"mtbf": 300.0, "mttr": 100.0, "seed": 7}


def to_order(os_seq, n_op):
    cnt, order = {}, []
    for j in os_seq:
        o = cnt.get(j, 0)
        cnt[j] = o + 1
        order.append(j * n_op + o)
    return order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for i in a.instances:
        inst = load_instance(POOL, i)

        os_seq, ms = pdr_schedule(inst.t, inst.power, "mor", "spt")
        order = to_order(os_seq, inst.n_op)
        base = simulate(inst, order, list(ms), n_agv=6, lu_station=1)
        hit = simulate(inst, order, list(ms), n_agv=6, lu_station=1,
                       failure=FAIL)
        rows.append({"instance": i, "method": "PDR",
                     "obj": base.objective, "obj_hit": hit.objective,
                     "mk": base.makespan, "mk_hit": hit.makespan})

        clean = rga_instance(inst.t, inst.power, inst.travel, inst.edges,
                             seed=1, pop=50, gen=100, agv=True,
                             lu_machine=inst.lu_machine,
                             lu_edge=inst.lu_edge, idle_mode="trailing",
                             eval_mode="sim", n_agv=6)
        hard = rga_instance(inst.t, inst.power, inst.travel, inst.edges,
                            seed=1, pop=50, gen=100, agv=True,
                            lu_machine=inst.lu_machine,
                            lu_edge=inst.lu_edge, idle_mode="trailing",
                            eval_mode="sim", n_agv=6, failure=FAIL)
        rows.append({"instance": i, "method": "RGA",
                     "obj": clean["obj"], "obj_hit": hard["obj"],
                     "mk": clean["mk"], "mk_hit": hard["mk"]})
        print(f"instance {i} done", flush=True)

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh)
    for name in ("PDR", "RGA"):
        sel = [r for r in rows if r["method"] == name]
        d = np.mean([r["obj_hit"] - r["obj"] for r in sel])
        b = np.mean([r["obj"] for r in sel])
        print(f"{name}: {b:.2f} -> {b + d:.2f}  ({100 * d / b:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())