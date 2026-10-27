"""Validate the CP-SAT cost accounting against the v5 simulator.

Take a schedule, run it through ``Parallel_env``, then recompute the four
cost components from the environment's own start times and machine
assignment with ``model_noagv._cost``.  If the numbers match, the model's
objective is the simulator's objective.

Run from the repo root with the OR-Tools interpreter::

    D:/anaconda/envs/py312/python.exe -m plcsp_v5.baselines.mip_agv.validate_env --index 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plcsp_v5"))
sys.path.insert(0, str(ROOT / "plcsp_v5" / "graph-jsp-env" / "src"))

from baselines.check_agv_env import run_env  # noqa: E402
from baselines.agv_baselines import pdr_schedule  # noqa: E402
from plcsp_v5.baselines.mip_agv.model_noagv import (  # noqa: E402
    _cost, load_instance, solve)


def sequence_from_starts(starts, machines, n_job: int, n_op: int):
    """Operation sequence (job ids) that follows the model's start order."""
    starts = list(starts)
    nxt = {j: 0 for j in range(n_job)}
    os_seq = []
    for _ in range(n_job * n_op):
        best = None
        for j in range(n_job):
            o = nxt[j]
            if o >= n_op:
                continue
            tau = j * n_op + o
            key = (float(starts[tau]), tau)
            if best is None or key < best[0]:
                best = (key, j)
        _, j = best
        os_seq.append(j)
        nxt[j] += 1
    return os_seq


def task_order_from_starts(starts, n_job: int, n_op: int):
    """Operation sequence as task indices, for ``agv_sim.simulate``."""
    starts = list(starts)
    nxt = {j: 0 for j in range(n_job)}
    order = []
    for _ in range(n_job * n_op):
        best = None
        for j in range(n_job):
            o = nxt[j]
            if o >= n_op:
                continue
            tau = j * n_op + o
            key = (float(starts[tau]), tau)
            if best is None or key < best[0]:
                best = (key, j, tau)
        _, j, tau = best
        order.append(tau)
        nxt[j] += 1
    return order


def compare(inst, starts, machines, tt, edge, use_agv: int, n_agv: int):
    os_seq = sequence_from_starts(starts, machines, inst.n_job, inst.n_op)
    env = run_env(inst.t, inst.power, tt, edge, os_seq, list(machines),
                  use_agv, n_agv)
    mine = _cost(inst, env["starts"], list(machines))
    return env, mine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance",
                    default="plcsp_v5/instance/test_Instance_J6M6E2.pkl")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--use_agv", type=int, default=0)
    ap.add_argument("--n_agv", type=int, default=6)
    ap.add_argument("--time_limit", type=float, default=20.0)
    ap.add_argument("--source", choices=("pdr", "cpsat"), default="cpsat")
    args = ap.parse_args()

    import pickle
    raw = pickle.load(open(args.instance, "rb"))
    tt, edge = raw[2][args.index], raw[3][args.index]
    inst = load_instance(args.instance, args.index)

    if args.source == "pdr":
        os_seq, ms = pdr_schedule(inst.t, inst.power, "mor", "spt")
        starts = [0.0] * inst.n_task
        # Only the assignment matters for the PDR path; the sequence is
        # already ordered, so re-derive starts from the env below.
        env = run_env(inst.t, inst.power, tt, edge, os_seq, ms,
                      args.use_agv, args.n_agv)
        mine = _cost(inst, env["starts"], list(ms))
    else:
        sched = solve(inst, time_limit=args.time_limit, workers=8)
        if sched is None:
            raise SystemExit("CP-SAT found no solution")
        env, mine = compare(inst, sched.start, sched.machine, tt, edge,
                            args.use_agv, args.n_agv)

    print("component        env        mine      diff")
    for key_env, key_mine in (("mk", "makespan"), ("pt", "pec_mean"),
                              ("idleT", "idle"), ("transT", "trans")):
        a = float(env[key_env])
        b = float(getattr(mine, key_mine)) * (inst.n_task
                                               if key_env == "pt" else 1.0)
        print(f"{key_mine:12s} {a:10.4f} {b:10.4f} {a - b:9.4f}")
    print(f"{'objective':12s} {env['obj']:10.4f} "
          f"{mine.objective:10.4f} {env['obj'] - mine.objective:9.4f}")


if __name__ == "__main__":
    main()
