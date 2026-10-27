"""CP-SAT model for the v5 MT-FJSP, no-AGV baseline.

Objective (same as the v5 simulator and the shipped MIP CSV)::

    0.4 * makespan + 0.4 * (mean PEC + idleT) + 0.2 * transT

``mean PEC`` is the total processing energy divided by ``n_job * n_op``.
``idleT`` is the sum over machines of ``last_finish - processing_sum``,
which equals the initial idle plus every gap.

Times come from the instance as floats.  CP-SAT needs integers, so all
times are scaled by ``SCALE`` and rounded.

Run with the OR-Tools interpreter::

    D:/anaconda/envs/py312/python.exe -m plcsp_v5.baselines.mip_agv.model_noagv
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from ortools.sat.python import cp_model

SCALE = 1000
# Weights are 0.4 / 0.4 / 0.2.  Multiply by 5 to clear the fractions.
W_MK, W_EC, W_TT = 2, 2, 1


@dataclass(frozen=True)
class Instance:
    n_job: int
    n_op: int
    n_machine: int
    t: np.ndarray          # (n_task, n_machine) processing time, <=0 means ineligible
    power: np.ndarray      # (n_task, n_machine) power, sign ignored
    travel: np.ndarray     # (n_machine, n_machine) travel time
    edges: np.ndarray      # (n_edge, n_machine_per_edge) machine ids
    task: int

    @property
    def n_task(self) -> int:
        return self.n_job * self.n_op

    def eligible(self, tau: int) -> List[int]:
        return [m for m in range(self.n_machine) if self.t[tau, m] > 0]

    def proc(self, tau: int, m: int) -> int:
        return int(round(float(self.t[tau, m]) * SCALE))

    def energy(self, tau: int, m: int) -> int:
        return int(round(float(self.t[tau, m]) * abs(float(self.power[tau, m]))
                         * SCALE))

    def travel_int(self, m1: int, m2: int) -> int:
        return int(round(float(self.travel[m1, m2]) * SCALE))


def load_instance(path: str, idx: int) -> Instance:
    with open(path, "rb") as fh:
        raw = pickle.load(fh)
    t, power, travel, edges = raw
    t_i = np.asarray(t[idx], dtype=np.float64)
    p_i = np.asarray(power[idx], dtype=np.float64)
    tt_i = np.asarray(travel[idx], dtype=np.float64)
    ed_i = np.asarray(edges[idx], dtype=np.int64)
    n_task, n_machine = t_i.shape
    # The task matrix is (n_job * n_op, n_machine).  In this instance family
    # n_op == n_machine; the edge table is (n_edge, machines_per_edge) and
    # must NOT be read as the job count.
    n_op = n_machine
    if n_task % n_op != 0:
        raise ValueError(f"task count {n_task} not divisible by ops {n_op}")
    n_job = n_task // n_op
    return Instance(n_job=n_job, n_op=n_op, n_machine=n_machine,
                    t=t_i, power=p_i, travel=tt_i, edges=ed_i, task=idx)


@dataclass
class Schedule:
    start: List[float]
    machine: List[int]
    makespan: float
    pec_mean: float
    idle: float
    trans: float
    objective: float
    status: str = ""


def _horizon(inst: Instance) -> int:
    total = 0
    for tau in range(inst.n_task):
        total += max(inst.proc(tau, m) for m in inst.eligible(tau))
    total += inst.n_task * int(round(float(inst.travel.max()) * SCALE))
    return int(total) + 1


def solve(inst: Instance, time_limit: float = 60.0,
          workers: int = 8, objective: str = "full") -> Optional[Schedule]:
    model = cp_model.CpModel()
    n, m_count = inst.n_task, inst.n_machine
    horizon = _horizon(inst)

    x: Dict[Tuple[int, int], cp_model.IntVar] = {}
    start: Dict[int, cp_model.IntVar] = {}
    end: Dict[int, cp_model.IntVar] = {}
    intervals: Dict[int, List[cp_model.IntervalVar]] = {
        m: [] for m in range(m_count)}

    for tau in range(n):
        elig = inst.eligible(tau)
        if not elig:
            raise ValueError(f"task {tau} has no eligible machine")
        start[tau] = model.NewIntVar(0, horizon, f"s_{tau}")
        end[tau] = model.NewIntVar(0, horizon, f"e_{tau}")
        for m in elig:
            x[tau, m] = model.NewBoolVar(f"x_{tau}_{m}")
        model.Add(end[tau] == start[tau]
                  + sum(inst.proc(tau, m) * x[tau, m] for m in elig))
        model.AddExactlyOne(x[tau, m] for m in elig)
        for m in elig:
            iv = model.NewOptionalIntervalVar(
                start[tau], inst.proc(tau, m), end[tau], x[tau, m],
                f"iv_{tau}_{m}")
            intervals[m].append(iv)

    # Precedence inside each job: consecutive tasks are tau and tau + 1.
    # y[a, m1, m2] = x[a, m1] * x[b, m2] is encoded with a transportation
    # polytope.  With x integral this is exact and much tighter than
    # AddMultiplicationEquality.
    arc: Dict[Tuple[int, int, int], cp_model.IntVar] = {}
    for j in range(inst.n_job):
        for o in range(inst.n_op - 1):
            a = j * inst.n_op + o
            b = a + 1
            elig_a, elig_b = inst.eligible(a), inst.eligible(b)
            for m1 in elig_a:
                for m2 in elig_b:
                    arc[a, m1, m2] = model.NewBoolVar(f"y_{a}_{m1}_{m2}")
            for m1 in elig_a:
                model.Add(sum(arc[a, m1, m2] for m2 in elig_b)
                          == x[a, m1])
            for m2 in elig_b:
                model.Add(sum(arc[a, m1, m2] for m1 in elig_a)
                          == x[b, m2])
            transport_terms = [inst.travel_int(m1, m2) * y
                               for (aa, m1, m2), y in arc.items() if aa == a]
            model.Add(start[b] >= end[a] + sum(transport_terms))

    for m in range(m_count):
        if len(intervals[m]) > 1:
            model.AddNoOverlap(intervals[m])

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, list(end.values()))

    total_energy = sum(inst.energy(tau, m) * x[tau, m]
                       for tau in range(n) for m in inst.eligible(tau))

    idle_terms = []
    for m in range(m_count):
        ends_on_m = []
        for tau in range(n):
            if m not in inst.eligible(tau):
                continue
            e_tm = model.NewIntVar(0, horizon, f"em_{tau}_{m}")
            model.Add(e_tm == end[tau]).OnlyEnforceIf(x[tau, m])
            model.Add(e_tm == 0).OnlyEnforceIf(x[tau, m].Not())
            ends_on_m.append(e_tm)
        if not ends_on_m:
            continue
        last = model.NewIntVar(0, horizon, f"last_{m}")
        model.AddMaxEquality(last, ends_on_m)
        proc_sum = sum(inst.proc(tau, m) * x[tau, m]
                       for tau in range(n) if m in inst.eligible(tau))
        idle_terms.append(last - proc_sum)

    idle = model.NewIntVar(0, horizon * m_count, "idle")
    model.Add(idle == sum(idle_terms))

    trans = model.NewIntVar(0, horizon * n, "trans")
    model.Add(trans == sum(
        inst.travel_int(m1, m2) * y
        for (a, m1, m2), y in arc.items()))

    # Clear the 0.4/0.4/0.2 fractions and the PEC mean denominator.
    denom = inst.n_task
    total = model.NewIntVar(0, 1 << 60, "total")
    model.Add(total == W_MK * denom * makespan
              + W_EC * (total_energy + denom * idle)
              + W_TT * denom * trans)
    if objective == "makespan":
        model.Minimize(makespan)
    else:
        model.Minimize(total)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    starts = [solver.Value(start[t]) / SCALE for t in range(n)]
    machines = []
    for tau in range(n):
        chosen = [m for m in inst.eligible(tau) if solver.Value(x[tau, m])]
        machines.append(int(chosen[0]))

    sched = _cost(inst, starts, machines)
    sched.status = solver.StatusName(status)
    return sched

def _cost(inst: Instance, starts: Sequence[float],
          machines: Sequence[int]) -> Schedule:
    n, m_count = inst.n_task, inst.n_machine
    end = [starts[t] + float(inst.t[t, machines[t]]) for t in range(n)]
    makespan = max(end)

    total_energy = sum(float(inst.t[t, machines[t]])
                       * abs(float(inst.power[t, machines[t]]))
                       for t in range(n))
    pec_mean = total_energy / n

    idle = 0.0
    for m in range(m_count):
        tau_m = [t for t in range(n) if machines[t] == m]
        if not tau_m:
            continue
        idle += max(end[t] for t in tau_m) - sum(
            float(inst.t[t, m]) for t in tau_m)

    trans = 0.0
    for j in range(inst.n_job):
        for o in range(inst.n_op - 1):
            a, b = j * inst.n_op + o, j * inst.n_op + o + 1
            trans += float(inst.travel[machines[a], machines[b]])

    objective = (0.4 * makespan + 0.4 * (pec_mean + idle) + 0.2 * trans)
    return Schedule(start=list(starts), machine=list(machines),
                    makespan=makespan, pec_mean=pec_mean, idle=idle,
                    trans=trans, objective=objective)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instance", default=(
        "plcsp_v5/instance/test_Instance_J6M6E2.pkl"))
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--time_limit", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--objective", choices=("full", "makespan"),
                    default="full")
    args = ap.parse_args()

    inst = load_instance(args.instance, args.index)
    sched = solve(inst, time_limit=args.time_limit, workers=args.workers,
                  objective=args.objective)
    if sched is None:
        print(json.dumps({"status": "no_solution"}, ensure_ascii=False))
        return
    print(json.dumps({
        "instance": inst.task,
        "status": sched.status,
        "makespan": round(sched.makespan, 4),
        "pec_mean": round(sched.pec_mean, 4),
        "idle": round(sched.idle, 4),
        "trans": round(sched.trans, 4),
        "objective": round(sched.objective, 4),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
