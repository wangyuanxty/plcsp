"""AGV routing MILP for a fixed job/machine schedule (Stage B).

Given an operation order and machine assignment, each consecutive pair of
operations on different machines needs a transport.  A transport is one
leg (same workshop) or two legs with a fixed delay (cross workshop).

With ``lu_station=1`` the M2 model applies: every workshop has a
load/unload station, the first operation of a job also needs a leg
(station -> machine), and a cross-workshop transport splits at the
stations (machine -> source station, no-AGV transfer, target station ->
machine).

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

import numpy as np
from ortools.sat.python import cp_model

from plcsp_v5.agv.state import (
    m2_node_dist, split_cross_edge, station_node)
from plcsp_v5.baselines.mip_agv.agv_sim import simulate
from plcsp_v5.baselines.mip_agv.model_noagv import (
    SCALE, Instance, load_instance, solve)
from plcsp_v5.baselines.mip_agv.validate_env import task_order_from_starts

DEPOT = 0


@dataclass(frozen=True)
class Leg:
    tau: int           # transport task (the operation that is moved)
    kind: str          # "single" | "src" | "dst"
    workshop: int
    pickup: int        # machine id, or station node n_machine + workshop
    dropoff: int       # machine id, or station node n_machine + workshop
    duration: int      # scaled
    release: int       # scaled
    final: bool
    delay: int = 0     # fixed cross-workshop delay before a "dst" leg


def _edge_of(edges, machine: int) -> int:
    for e, members in enumerate(edges):
        if machine in [int(x) for x in members]:
            return e
    raise ValueError(f"machine {machine} is in no workshop")


def _machine_edge(inst: Instance) -> np.ndarray:
    """Workshop index of every machine."""
    out = np.zeros(inst.n_machine, dtype=int)
    for w, members in enumerate(inst.edges):
        for m in members:
            out[int(m)] = w
    return out


def _check_m2(inst: Instance, lu_station: int) -> None:
    if lu_station and (inst.lu_machine is None or inst.lu_edge is None):
        raise ValueError(
            "lu_station=1 needs an instance with lu_machine / lu_edge; "
            "use a *_lu.pkl pool")


def build_legs(inst: Instance, order: Sequence[int],
               machines: Sequence[int], starts: Sequence[float],
               cross_delay: float = 5.0,
               leg_ratio: float = 0.5,
               lu_station: int = 0) -> List[Leg]:
    """Build transport legs from an operation schedule."""
    _check_m2(inst, lu_station)
    legs: List[Leg] = []
    finishes = [starts[t] + float(inst.t[t, machines[t]])
                for t in range(inst.n_task)]
    machine_edge = _machine_edge(inst) if lu_station else None
    for tau in order:
        tau = int(tau)
        o = tau % inst.n_op
        if lu_station:
            m = int(machines[tau])
            de = int(machine_edge[m])
            if o == 0:
                # Entry leg: the job starts at the station of its first
                # operation's workshop.
                legs.append(Leg(
                    tau=tau, kind="single", workshop=de,
                    pickup=station_node(de, inst.n_machine), dropoff=m,
                    duration=int(round(float(inst.lu_machine[m]) * SCALE)),
                    release=0, final=True))
                continue
            prev_m = int(machines[tau - 1])
            if prev_m == m:
                continue
            ready = int(round(finishes[tau - 1] * SCALE))
            pe = int(machine_edge[prev_m])
            if pe == de:
                legs.append(Leg(
                    tau=tau, kind="single", workshop=pe,
                    pickup=prev_m, dropoff=m,
                    duration=int(round(float(inst.travel[prev_m][m]) * SCALE)),
                    release=ready, final=True))
            else:
                legs.append(Leg(
                    tau=tau, kind="src", workshop=pe,
                    pickup=prev_m,
                    dropoff=station_node(pe, inst.n_machine),
                    duration=int(round(float(
                        inst.lu_machine[prev_m]) * SCALE)),
                    release=ready, final=False))
                legs.append(Leg(
                    tau=tau, kind="dst", workshop=de,
                    pickup=station_node(de, inst.n_machine), dropoff=m,
                    duration=int(round(float(inst.lu_machine[m]) * SCALE)),
                    release=0, final=True,
                    delay=int(round(float(inst.lu_edge[pe][de]) * SCALE))))
            continue
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
                  time_limit: float = 60.0, workers: int = 8,
                  lu_station: int = 0
                  ) -> Tuple[Dict[int, Tuple[int, int]], float]:
    """Assign and sequence legs.  Returns (plan, objective).

    ``lu_station=1`` measures every deadhead in the M2 node space, where
    station nodes are ``n_machine + workshop``.
    """
    _check_m2(inst, lu_station)
    n_edge = len(inst.edges)
    machine_edge = _machine_edge(inst) if lu_station else None

    def _travel(a: int, b: int) -> float:
        return float(inst.travel[int(a)][int(b)])

    def dist(a: int, b: int) -> float:
        if lu_station:
            return float(m2_node_dist(
                a, b, _travel, inst.n_machine, machine_edge,
                inst.lu_machine, inst.lu_edge))
        return float(inst.travel[int(a)][int(b)])

    if lu_station:
        far = max(float(inst.travel.max()), float(inst.lu_machine.max()),
                  float(inst.lu_edge.max()))
    else:
        far = float(inst.travel.max())
    horizon = max([leg.release for leg in legs] or [0])
    horizon += sum(leg.duration for leg in legs)
    horizon += int(round(far * SCALE)) * len(legs)
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
        if lu_station:
            depot_loc = station_node(w, inst.n_machine)
        else:
            depot_loc = int(inst.edges[w][0])
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
                    dead = int(round(dist(depot_loc, legs[k].pickup) * SCALE))
                    model.Add(start[k] >= dead).OnlyEnforceIf(lit)
                else:
                    k_i = members[i - 1]
                    if j != DEPOT:
                        k_j = members[j - 1]
                        dead = int(round(dist(
                            legs[k_i].dropoff, legs[k_j].pickup) * SCALE))
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
    ap.add_argument("--instance", default=None,
                    help="default: test_Instance_J6M6E2(_lu).pkl")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--n_agv", type=int, default=6)
    ap.add_argument("--stage_a_time", type=float, default=20.0)
    ap.add_argument("--routing_time", type=float, default=30.0)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--lu_station", type=int, default=0,
                    help="1 = M2 load/unload-station transport model")
    args = ap.parse_args()
    if args.instance is None:
        suffix = "_lu" if args.lu_station else ""
        args.instance = f"plcsp_v5/instance/test_Instance_J6M6E2{suffix}.pkl"

    inst = load_instance(args.instance, args.index)
    sched = solve(inst, time_limit=args.stage_a_time, workers=8)
    if sched is None:
        raise SystemExit("Stage A found no solution")
    order = task_order_from_starts(sched.start, inst.n_job, inst.n_op)

    rule = simulate(inst, order, sched.machine, n_agv=args.n_agv,
                    lu_station=args.lu_station)
    best = rule
    best_name = "rule"
    starts = rule.starts
    legs: List[Leg] = []
    for it in range(args.iters):
        legs = build_legs(inst, order, sched.machine, starts,
                          lu_station=args.lu_station)
        if not legs:
            break
        plan, _ = solve_routing(inst, legs, args.n_agv,
                                time_limit=args.routing_time,
                                lu_station=args.lu_station)
        sim = simulate(inst, order, sched.machine, n_agv=args.n_agv,
                       pick=lambda step, ctx: plan.get(
                           ctx["tau"], (ctx["src_pool"][0],
                                        ctx["dst_pool"][0]
                                        if ctx["dst_pool"] else -1)),
                       lu_station=args.lu_station)
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
