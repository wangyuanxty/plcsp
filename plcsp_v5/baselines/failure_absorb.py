"""Compare how a fixed plan absorbs a machine or AGV failure.

For every method the plan itself is unchanged: this experiment does not
re-plan.  Each plan is replayed through ``agv_sim.simulate`` with downtime
windows, so an operation on a failed machine finishes later and an AGV trip
that meets a failed vehicle waits for the repair.

The failures are placed at fixed fractions of each method's own no-failure
makespan, with the resource drawn among those busy at that instant, so every
method meets four disruptions that matter to it.

Usage::

    python -m baselines.failure_absorb --instances 0 1 2 --plans plans.json \
        --out absorb.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys

import numpy as np

REPO = __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__)))
sys.path.insert(0, REPO)

from plcsp_v5.baselines.agv_baselines import pdr_schedule, rga_instance
from plcsp_v5.baselines.mip_agv.model_noagv import load_instance
from plcsp_v5.baselines.mip_agv.agv_sim import simulate

POOL = (REPO + "/instance/test_Instance_J6M6E2_lu.pkl"
        if REPO.endswith("plcsp_v5")
        else "plcsp_v5/instance/test_Instance_J6M6E2_lu.pkl")

N_MACHINE, N_AGV = 6, 6
N_FAIL = 2
REPAIR = (0.5, 1.5)
ONSET = (0.15, 0.85)


def to_order(os_seq, n_op):
    cnt, order = {}, []
    for j in os_seq:
        o = cnt.get(j, 0)
        cnt[j] = o + 1
        order.append(j * n_op + o)
    return order


def make_failures(base, mean_t, rng):
    """Two machine and two AGV failures, drawn without looking at the plan.

    The resource is uniform over the fleet, so a failure can land on an idle
    machine; it still delays whatever is scheduled on that machine before the
    repair ends.
    """
    lo, hi = REPAIR
    span = base.makespan
    down = {"machine": {}, "agv": {}}
    for _ in range(N_FAIL):
        m = rng.randrange(N_MACHINE)
        t = rng.uniform(*ONSET) * span
        r = rng.uniform(lo, hi) * mean_t
        down["machine"].setdefault(m, []).append((t, t + r))
    for _ in range(N_FAIL):
        a = rng.randrange(N_AGV)
        t = rng.uniform(*ONSET) * span
        r = rng.uniform(lo, hi) * mean_t
        down["agv"].setdefault(a, []).append((t, t + r))
    return down


def plans_for(inst, ours):
    """(order, machines, pick) for the controller and the two baselines."""
    n_op = inst.n_op
    out = {}

    os_seq, ms = pdr_schedule(inst.t, inst.power, "mor", "spt")
    out["PDR"] = (to_order(os_seq, n_op), list(ms), None)

    rga = rga_instance(inst.t, inst.power, inst.travel, inst.edges, seed=1,
                       pop=50, gen=100, agv=True, return_schedule=True,
                       lu_machine=inst.lu_machine, lu_edge=inst.lu_edge,
                       idle_mode="trailing", eval_mode="sim", n_agv=6)
    out["RGA"] = (to_order(rga["os_seq"], n_op), list(rga["ms"]), None)

    order = [s["task"] for s in ours["steps"]]
    ms = [0] * len(order)
    for s in ours["steps"]:
        ms[s["task"]] = s["machine"]
    picks = [tuple(t) for t in ours.get("transports", [])]
    out["Ours"] = (order, ms, picks)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, nargs="+", required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.plans, encoding="utf-8") as fh:
        dump = json.load(fh)
    by_idx = {row["instance"]: row for row in dump["rows"]}

    rows = []
    for i in a.instances:
        inst = load_instance(POOL, i)
        mean_t = float(np.mean(inst.t[inst.t >= 0]))
        plans = plans_for(inst, by_idx[i])
        rng = random.Random(1000 + i)
        for name, (order, ms, picks) in plans.items():
            def picker(picks=picks):
                if picks is None:
                    return None
                it = iter(picks)
                return lambda step, ctx: next(it, (ctx["src_pool"][0], -1))

            base = simulate(inst, order, ms, n_agv=6, lu_station=1,
                            pick=picker())
            down = make_failures(base, mean_t, random.Random(1000 + i))
            hit = simulate(inst, order, ms, n_agv=6, lu_station=1,
                           pick=picker(), failures=down)
            shifted = int(sum(1 for s0, s1 in zip(base.starts, hit.starts)
                              if abs(s0 - s1) > 1e-9))
            rows.append({
                "instance": i, "method": name,
                "obj": base.objective, "obj_hit": hit.objective,
                "mk": base.makespan, "mk_hit": hit.makespan,
                "shifted": shifted,
                "failures": {k: {str(r): v for r, v in d.items()}
                             for k, d in down.items()},
            })
        print(f"instance {i} done", flush=True)

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh)
    for name in ("Ours", "PDR", "RGA"):
        sel = [r for r in rows if r["method"] == name]
        if not sel:
            continue
        d_obj = np.mean([r["obj_hit"] - r["obj"] for r in sel])
        d_mk = np.mean([r["mk_hit"] - r["mk"] for r in sel])
        print(f"{name:5s} obj +{d_obj:8.2f}   mk +{d_mk:8.2f}   "
              f"({len(sel)} instances)")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())