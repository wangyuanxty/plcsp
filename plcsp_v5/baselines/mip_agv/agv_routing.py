"""AGV routing MILP for a fixed job/machine schedule (Stage B).

Given an operation order and machine assignment, each consecutive pair of
operations on different machines needs a transport.  A transport is one
leg (same workshop) or two legs with a fixed delay (cross workshop).

The model assigns legs to AGVs (home-workshop fleets) and sequences each
AGV with a circuit.  The sequence-dependent setup is the deadhead travel
from the previous drop-off to the next pick-up.  The objective minimises
the sum of the final-leg completion times.

This is the second stage of a two-stage MIP-AGV baseline: Stage A fixes
the operation schedule, Stage B routes the AGVs.  The result is evaluated
with ``agv_sim.simulate`` so left-shift and waiting are handled exactly.

Run::

    D:/anaconda/envs/py312/python.exe -m plcsp_v5.baselines.mip_agv.agv_routing --index 0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ortools.sat.python import cp_model

from plcsp_v5.agv.state import split_cross_edge
from plcsp_v5.baselines.mip_agv.agv_sim import simulate
from plcsp_v5.baselines.mip_agv.model_noagv import (
    SCALE, Instance, Schedule, load_instance, solve)
from plcsp_v5.baselines.mip_agv.validate_env import task_order_from_starts

DEPOT = 0


@dataclass(frozen=True)
class Leg:
    tau: int           # transport task (the operation that is moved)
    kind: str          # "single" | "src" | "dst"
    workshop: int
    pickup: int
    dropoff: int
    duration: int      # scaled
    release: int       # scaled
    final: bool
    delay: int = 0     # fixed cross-workshop delay before a "dst" leg


def _edge_of(edges, machine: int) -> int:
    for e, members in enumerate(edges):
        if machine in [int(x) for x in members]:
            return e
    raise ValueError(f"machine {machine} is in no workshop")


def build_legs(inst: Instance, order: Sequence[int],
               machines: Sequence[int], starts: Sequence[float],
               cross_delay: float = 5.0,
               leg_ratio: float = 0.5) -> List[Leg]:
    """Build transport legs from an operation schedule."""
    legs: List[Leg] = []
    finishes = [starts[t] + float(inst.t[t, machines[t]])
                for t in range(inst.n_task)]
    for tau in order:
        tau = int(tau)
        o = tau % inst.n_op
        if o == 0:
            continue
        prev_m = int(machines[tau - 1])
        m = int(machines[tau])
        if prev_m == m:
            continue
        ready = int(round(finishes[tau - 1] * SCALE))
        total = float(inst.travel[prev_m][m])
        w1 = _edge_of(inst.edges, prev_m)
        w2 = _edge_of(inst.edges, m)
        if w1 == w2:
            legs.append(Leg(tau=tau, kind="single", workshop=w1,
                            pickup=prev_m, dropoff=m,
                            duration=int(round(total * SCALE)),
                            release=ready, final=True))
        else:
            t_src, d_eff, t_dst = split_cross_edge(total, cross_delay,
                                                   leg_ratio)
            legs.append(Leg(tau=tau, kind="src", workshop=w1,
                            pickup=prev_m, dropoff=int(inst.edges[w2][0]),
                            duration=int(round(t_src * SCALE)),
                            release=ready, final=False))
            legs.append(Leg(tau=tau, kind="dst", workshop=w2,
                            pickup=int(inst.edges[w2][0]), dropoff=m,
                            duration=int(round(t_dst * SCALE)),
                            release=0, final=True,
                            delay=int(round(d_eff * SCALE))))
    return legs


def solve_routing(inst: Instance, legs: Sequence[Leg], n_agv: int,
                  time_limit: float = 60.0, workers: int = 8
                  ) -> Tuple[Dict[int, Tuple[int, int]], float]:
    """Assign and sequence legs.  Returns (plan, objective)."""
    n_edge = len(inst.edges)
    horizon = max([leg.release for leg in legs] or [0])
    horizon += sum(leg.duration for leg in legs)
    horizon += int(round(float(inst.travel.max()) * SCALE)) * len(legs)
    horizon += 1

    model = cp_model.CpModel()
    start = [model.NewIntVar(0, horizon, f"s_{k}")
             for k in range(len(legs))]
    end = [model.NewIntVar(0, horizon, f"e_{k}")
           for k in range(len(legs))]
    for k, leg in enumerate(legs):
        model.Add(end[k] == start[k] + leg.duration)
        model.Add(start[k] >= leg.release)

    # Cross-workshop coupling: target leg starts after source end + delay.
    src_index = {(leg.tau): k for k, leg in enumerate(legs)
                 if leg.kind == "src"}
    for k, leg in enumerate(legs):
        if leg.kind == "dst":
            model.Add(start[k] >= end[src_index[leg.tau]] + leg.delay)

    # Per-AGV circuits.  All legs of a workshop are candidate nodes for
    # every AGV in that workshop; presence literals pick one AGV per leg.
    present: Dict[Tuple[int, int], cp_model.IntVar] = {}
    for a in range(n_agv):
        w = a % n_edge
        members = [k for k, leg in enumerate(legs) if leg.workshop == w]
        if not members:
            continue
        local = {k: i + 1 for i, k in enumerate(members)}
        nodes = [DEPOT] + list(local.values())
        arcs = []
        self_loop: Dict[int, cp_model.IntVar] = {}
        for node in nodes:
            if node == DEPOT:
                continue
            k = members[node - 1]
            present[a, k] = model.NewBoolVar(f"p_{a}_{k}")
            self_loop[node] = model.NewBoolVar(f"loop_{a}_{node}")
            model.Add(self_loop[node] == present[a, k].Not())
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                lit = model.NewBoolVar(f"arc_{a}_{i}_{j}")
                arcs.append((i, j, lit))
                if i == DEPOT:
                    k = members[j - 1]
                    dead = int(round(float(
                        inst.travel[int(inst.edges[w][0])][legs[k].pickup])
                        * SCALE))
                    model.Add(start[k] >= dead).OnlyEnforceIf(lit)
                else:
                    k_i = members[i - 1]
                    if j != DEPOT:
                        k_j = members[j - 1]
                        dead = int(round(float(
                            inst.travel[legs[k_i].dropoff][legs[k_j].pickup])
                            * SCALE))
                        model.Add(start[k_j] >= end[k_i] + dead
                                  ).OnlyEnforceIf(lit)
        all_arcs = list(arcs) + [(node, node, loop)
                                 for node, loop in self_loop.items()]
        # An AGV with no leg must still form a circuit.  A self-loop at the
        # depot represents an idle AGV; without it the model is infeasible
        # whenever one workshop receives no transport.
        depot_loop = model.NewBoolVar(f"depot_loop_{a}")
        present_sum = sum(present[a, k] for k in members)
        model.Add(present_sum == 0).OnlyEnforceIf(depot_loop)
        model.Add(present_sum >= 1).OnlyEnforceIf(depot_loop.Not())
        all_arcs.append((DEPOT, DEPOT, depot_loop))
        for node in nodes:
            model.AddExactlyOne(
                lit for (i, j, lit) in all_arcs if i == node)
            model.AddExactlyOne(
                lit for (i, j, lit) in all_arcs if j == node)
        model.AddCircuit(all_arcs)

    for k in range(len(legs)):
        model.Add(sum(present[a, k] for a in range(n_agv)
                      if (a, k) in present) == 1)

    obj = sum(end[k] for k, leg in enumerate(legs) if leg.final)
    model.Minimize(obj)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("AGV routing found no solution")

    plan: Dict[int, Tuple[int, int]] = {}
    for k, leg in enumerate(legs):
        chosen = [a for a in range(n_agv) if (a, k) in present
                  and solver.Value(present[a, k])]
        if len(chosen) != 1:
            raise RuntimeError(f"leg {k} assigned to {len(chosen)} AGVs")
        a = chosen[0]
        if leg.kind == "dst":
            src = plan.setdefault(leg.tau, [-1, -1])
            src[1] = a
        else:
            src = plan.setdefault(leg.tau, [-1, -1])
            src[0] = a
    return {tau: (v[0], v[1]) for tau, v in plan.items()}, \
        float(solver.ObjectiveValue())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance",
                    default="plcsp_v5/instance/test_Instance_J6M6E2.pkl")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--n_agv", type=int, default=6)
    ap.add_argument("--stage_a_time", type=float, default=20.0)
    ap.add_argument("--routing_time", type=float, default=30.0)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    inst = load_instance(args.instance, args.index)
    sched = solve(inst, time_limit=args.stage_a_time, workers=8)
    if sched is None:
        raise SystemExit("Stage A found no solution")
    order = task_order_from_starts(sched.start, inst.n_job, inst.n_op)

    rule = simulate(inst, order, sched.machine, n_agv=args.n_agv)
    best = rule
    best_name = "rule"
    starts = rule.starts
    legs: List[Leg] = []
    for it in range(args.iters):
        legs = build_legs(inst, order, sched.machine, starts)
        if not legs:
            break
        plan, _ = solve_routing(inst, legs, args.n_agv,
                                time_limit=args.routing_time)
        sim = simulate(inst, order, sched.machine, n_agv=args.n_agv,
                       pick=lambda step, ctx: plan.get(
                           ctx["tau"], (ctx["src_pool"][0],
                                        ctx["dst_pool"][0]
                                        if ctx["dst_pool"] else -1)))
        if sim.objective < best.objective:
            best, best_name = sim, f"milp{it}"
        starts = sim.starts

    print(f"legs={len(legs)}  iters={args.iters}")
    print("case            mk      pec     idle    trans    obj")
    for name, s in (("rule", rule), (best_name, best)):
        print(f"{name:12s} {s.makespan:8.2f} {s.pec_mean:8.2f} "
              f"{s.idle:8.2f} {s.trans:8.2f} {s.objective:8.2f}")


if __name__ == "__main__":
    main()
