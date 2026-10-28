"""Verification helpers for the M2 MIP baseline.

Two independent replays of a schedule:

``verify_m2``
    reconstructs the AGV chains from the model's arcs (or from any list
    of chains) and re-times them with the distances of
    ``plcsp_v5.agv.state.m2_node_dist``.

``verify_reference``
    re-simulates the same schedule through the authoritative
    ``plcsp_v5.agv.state.transport_timing_m2`` with real ``AgvState``
    objects.  This is the check that ties the MIP to the rest of the
    pipeline; it shares no timing code with the model builder.

``accept_check``
    the acceptance test: fix the model to a ``decode_m2`` schedule and
    require the model to accept it with the same objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ..agv_baselines import decode_m2
from ...agv.state import (RHO_EMPTY, AgvState, m2_node_dist, station_node,
                          transport_timing_m2)
from .model_m2 import (OBJ_SCALE, W_EC, W_MK, W_TT, InstanceM2, Leg,
                       _dist, build_legs)


# ---------------------------------------------------------------------------
# chain replay with m2_node_dist
# ---------------------------------------------------------------------------

def _release_of(legs: Sequence[Leg], idx: int, finish: Sequence[float],
                arrival: Dict[int, float], transfer_of_src: Dict[int, float]
                ) -> float:
    leg = legs[idx]
    if leg.kind == "dst":
        return arrival[leg.src_leg] + transfer_of_src[idx]
    if leg.kind == "entry":
        return 0.0
    return finish[leg.task - 1]


def verify_m2(inst: InstanceM2, machines: Sequence[int],
              starts: Sequence[float], chains: Sequence[Sequence[int]]
              ) -> Dict[str, object]:
    """Recompute the v5 objective from a schedule and its AGV chains.

    The chains are simulated with the ``decode_m2`` timing rules and
    iterated to a fixed point, because a target leg's release comes from
    its source leg on another AGV.  ``feasible`` means every operation
    starts no earlier than its transport allows, no machine overlaps,
    the job precedence holds, and every chain stays in one workshop.
    """
    me = inst.machine_edge_array()
    legs = build_legs(inst, machines)
    n = inst.n_task
    dur = [float(inst.t[tau, int(machines[tau])]) for tau in range(n)]
    finish = [float(starts[tau]) + dur[tau] for tau in range(n)]

    chain_edges_ok = True
    for chain in chains:
        edges = {legs[i].edge for i in chain}
        if len(edges) > 1:
            chain_edges_ok = False

    arrival = {idx: 0.0 for idx in range(len(legs))}
    for _ in range(len(legs) + 1):
        changed = False
        for chain in chains:
            free = 0.0
            loc = station_node(legs[chain[0]].edge, inst.n_machine)
            for idx in chain:
                leg = legs[idx]
                dead = _dist(inst, me, loc, leg.pickup)
                rel = _release_of(legs, idx, finish, arrival, _transfers(legs))
                arrive = max(free, rel) + dead + leg.loaded
                if arrive > arrival[idx] + 1e-9:
                    arrival[idx] = arrive
                    changed = True
                free = arrival[idx]
                loc = leg.dropoff
        if not changed:
            break

    covered = [idx for chain in chains for idx in chain]
    feasible = (len(covered) == len(legs)
                and len(set(covered)) == len(legs) and chain_edges_ok)
    empty = 0.0
    for chain in chains:
        loc = station_node(legs[chain[0]].edge, inst.n_machine)
        for idx in chain:
            empty += _dist(inst, me, loc, legs[idx].pickup)
            loc = legs[idx].dropoff
    for idx, leg in enumerate(legs):
        if idx in covered and float(starts[leg.task]) + 1e-6 < arrival[idx]:
            feasible = False

    for tau in range(n):
        if tau % inst.n_op and int(machines[tau]) == int(machines[tau - 1]):
            if float(starts[tau]) + 1e-6 < finish[tau - 1]:
                feasible = False

    by_machine: Dict[int, List[int]] = {}
    for tau in range(n):
        by_machine.setdefault(int(machines[tau]), []).append(tau)
    for tasks in by_machine.values():
        ordered = sorted(tasks, key=lambda t: float(starts[t]))
        for a, b in zip(ordered, ordered[1:]):
            if float(starts[b]) + 1e-6 < float(starts[a]) + dur[a]:
                feasible = False

    mk = max(finish)
    pec = sum(float(inst.t[tau, int(machines[tau])])
              * float(inst.power[tau, int(machines[tau])])
              for tau in range(n))
    idle = 0.0
    for tasks in by_machine.values():
        idle += max(finish[t] for t in tasks) - sum(dur[t] for t in tasks)
    loaded = sum(leg.loaded for leg in legs)
    trans = loaded + RHO_EMPTY * empty
    obj = W_MK * mk + W_EC * (pec / n + idle) + W_TT * trans
    return {"feasible": bool(feasible), "obj": obj, "mk": mk, "pec": pec,
            "idle": idle, "loaded": loaded, "empty": empty,
            "trans": trans, "n_legs": len(legs), "n_chains": len(chains)}


def _transfers(legs: Sequence[Leg]) -> Dict[int, float]:
    return {i: leg.transfer for i, leg in enumerate(legs) if leg.kind == "dst"}


# ---------------------------------------------------------------------------
# chain replay with transport_timing_m2 (authoritative state module)
# ---------------------------------------------------------------------------

def verify_reference(inst: InstanceM2, machines: Sequence[int],
                     starts: Sequence[float], chains: Sequence[Sequence[int]],
                     n_agv: int = 6, tol: float = 1e-6) -> Dict[str, object]:
    """Re-simulate a schedule with ``transport_timing_m2``.

    Every chain is replayed on a fresh ``AgvState`` from the workshop
    station.  A target leg's release is the source leg's arrival plus
    the station transfer, and the source leg sits on another AGV, so the
    replay is iterated to the least fixed point.  Each leg uses the
    same-workshop branch of ``transport_timing_m2``: with the drop-off
    set to the station, a source leg's ``src_free`` is its arrival at
    that station, and a target leg starts from the boundary.
    """
    me = inst.machine_edge_array()
    legs = build_legs(inst, machines)
    n = inst.n_task
    dur = [float(inst.t[tau, int(machines[tau])]) for tau in range(n)]
    finish = [float(starts[tau]) + dur[tau] for tau in range(n)]

    def travel(a: int, b: int) -> float:
        if a < 0 or b < 0:
            return 0.0
        return float(inst.travel[int(a)][int(b)])

    edges_ok = True
    for chain in chains:
        if len({legs[i].edge for i in chain}) > 1:
            edges_ok = False
    seen = [False] * len(legs)
    for chain in chains:
        for i in chain:
            seen[i] = True
    feasible = edges_ok and all(seen)

    arrival = {idx: 0.0 for idx in range(len(legs))}
    loaded = 0.0
    empty = 0.0
    for _ in range(len(legs) + 1):
        changed = False
        loaded = 0.0
        empty = 0.0
        for chain in chains:
            edge = legs[chain[0]].edge
            state = AgvState(idx=0, edge=edge,
                             location=station_node(edge, inst.n_machine))
            for idx in chain:
                leg = legs[idx]
                if leg.kind == "dst":
                    ready = arrival[leg.src_leg] + leg.transfer
                elif leg.kind == "entry":
                    ready = 0.0
                else:
                    ready = finish[leg.task - 1]
                timing = transport_timing_m2(
                    state, leg.pickup, leg.dropoff, ready, travel,
                    inst.n_machine, me, inst.lu_machine, inst.lu_edge)
                loaded += float(timing["loaded"])
                empty += float(timing["empty"])
                arrive = float(timing["src_free"])
                if arrive > arrival[idx] + 1e-9:
                    arrival[idx] = arrive
                    changed = True
                state.free_time = arrival[idx]
                state.location = int(timing["src_location"])
        if not changed:
            break

    for idx, leg in enumerate(legs):
        if float(starts[leg.task]) + tol < arrival[idx]:
            feasible = False
    for tau in range(n):
        if tau % inst.n_op and int(machines[tau]) == int(machines[tau - 1]):
            if float(starts[tau]) + tol < finish[tau - 1]:
                feasible = False
    by_machine: Dict[int, List[int]] = {}
    for tau in range(n):
        by_machine.setdefault(int(machines[tau]), []).append(tau)
    for tasks in by_machine.values():
        ordered = sorted(tasks, key=lambda t: float(starts[t]))
        for a, b in zip(ordered, ordered[1:]):
            if float(starts[b]) + tol < float(starts[a]) + dur[a]:
                feasible = False

    mk = max(finish)
    pec = sum(float(inst.t[tau, int(machines[tau])])
              * float(inst.power[tau, int(machines[tau])])
              for tau in range(n))
    idle = 0.0
    for tasks in by_machine.values():
        idle += max(finish[t] for t in tasks) - sum(dur[t] for t in tasks)
    trans = loaded + RHO_EMPTY * empty
    obj = W_MK * mk + W_EC * (pec / n + idle) + W_TT * trans
    return {"feasible": bool(feasible), "obj": obj, "mk": mk, "pec": pec,
            "idle": idle, "loaded": loaded, "empty": empty,
            "trans": trans, "n_legs": len(legs), "n_chains": len(chains)}


# ---------------------------------------------------------------------------
# acceptance check: the MIP must accept a decoder schedule
# ---------------------------------------------------------------------------

@dataclass
class AcceptResult:
    """Outcome of the fixed-schedule acceptance check."""

    feasible: bool
    model_obj: float
    decoded_obj: float
    variance: float
    n_vars: int
    n_constrs: int
    detail: Dict[str, float] = field(default_factory=dict)


def _decode_trace(inst: InstanceM2, machines: Sequence[int],
                  os_seq: Sequence[int]) -> Dict[str, object]:
    """Replay ``decode_m2`` and keep the AGV chains it builds."""
    n_task, n_machine = inst.t.shape
    n_job = n_task // n_machine
    n_ops = n_machine
    n_edge = inst.n_edge
    me = inst.machine_edge_array()

    def travel(a: int, b: int) -> float:
        if a < 0 or b < 0:
            return 0.0
        return float(inst.travel[int(a)][int(b)])

    def dist(a: int, b: int) -> float:
        return float(m2_node_dist(a, b, travel, n_machine, me,
                                  inst.lu_machine, inst.lu_edge))

    agvs = [[k % n_edge, station_node(k % n_edge, n_machine), 0.0, []]
            for k in range(6)]
    station = [station_node(w, n_machine) for w in range(n_edge)]
    op_count = [0] * n_job
    finish = [0.0] * n_task
    start = [0.0] * n_task
    routes: Dict[int, List[tuple]] = {m: [] for m in range(n_machine)}
    leg_task: List[tuple] = []
    chain_of: Dict[int, int] = {}

    for j in os_seq:
        j = int(j)
        oi = op_count[j]
        task = j * n_ops + oi
        m = int(machines[task])
        dur = float(inst.t[task][m])
        op_count[j] += 1
        de = int(me[m])
        if oi == 0:
            prev_ft, prev_m, pickup, pe = 0.0, -1, station[de], de
        else:
            prev_m = int(machines[task - 1])
            prev_ft = finish[task - 1]
            pickup, pe = prev_m, int(me[prev_m])

        if oi > 0 and prev_m == m:
            ready = prev_ft
        elif pe == de:
            src = min((k for k, a in enumerate(agvs) if a[0] == pe),
                      key=lambda k: (agvs[k][2], k))
            start_src = max(prev_ft, agvs[src][2])
            dead = dist(agvs[src][1], pickup)
            loaded = dist(pickup, m)
            finish_src = start_src + dead + loaded
            agvs[src][2] = finish_src
            agvs[src][1] = m
            idx = len(leg_task)
            leg_task.append((task, "entry" if oi == 0 else "single"))
            chain_of[idx] = src
            agvs[src][3].append(idx)
            ready = finish_src
        else:
            src = min((k for k, a in enumerate(agvs) if a[0] == pe),
                      key=lambda k: (agvs[k][2], k))
            dst = min((k for k, a in enumerate(agvs) if a[0] == de),
                      key=lambda k: (agvs[k][2], k))
            start_src = max(prev_ft, agvs[src][2])
            dead_src = dist(agvs[src][1], pickup)
            t_src = float(inst.lu_machine[pickup])
            src_arrival = start_src + dead_src + t_src
            boundary = src_arrival + float(inst.lu_edge[pe][de])
            start_dst = max(boundary, agvs[dst][2])
            dead_dst = dist(agvs[dst][1], station[de])
            t_dst = float(inst.lu_machine[m])
            finish_dst = start_dst + dead_dst + t_dst
            agvs[src][2] = src_arrival
            agvs[src][1] = station[pe]
            agvs[dst][2] = finish_dst
            agvs[dst][1] = m
            i_src = len(leg_task)
            leg_task.append((task, "src"))
            chain_of[i_src] = src
            agvs[src][3].append(i_src)
            i_dst = len(leg_task)
            leg_task.append((task, "dst"))
            chain_of[i_dst] = dst
            agvs[dst][3].append(i_dst)
            ready = finish_dst

        route = routes[m]
        pos = len(route)
        for k in range(len(route) + 1):
            prev_fin = route[k - 1][1] if k > 0 else 0.0
            next_st = route[k][0] if k < len(route) else float("inf")
            st = max(ready, prev_fin)
            if st + dur <= next_st:
                pos = k
                break
        route.insert(pos, (st, st + dur, task))
        start[task] = st
        finish[task] = st + dur

    chains = [list(a[3]) for a in agvs if a[3]]
    return {"start": start, "finish": finish, "chains": chains,
            "leg_task": leg_task}


def _agv_free_times(inst: InstanceM2, machines: Sequence[int],
                    trace: Dict[str, object]) -> Dict[int, float]:
    """Departure (AGV free) time of every leg in a decoded trace."""
    legs = build_legs(inst, machines)
    key_to_idx = {(leg.task, leg.kind): i for i, leg in enumerate(legs)}
    me = inst.machine_edge_array()
    finish = trace["finish"]
    dep_out: Dict[int, float] = {i: 0.0 for i in range(len(legs))}
    arrive_out: Dict[int, float] = {i: 0.0 for i in range(len(legs))}
    for _ in range(len(legs) + 1):
        changed = False
        for chain in trace["chains"]:
            free = 0.0
            first_task, first_kind = trace["leg_task"][chain[0]]
            loc = station_node(legs[key_to_idx[(first_task, first_kind)]].edge,
                               inst.n_machine)
            for i in chain:
                task, kind = trace["leg_task"][i]
                idx = key_to_idx[(task, kind)]
                leg = legs[idx]
                if kind == "dst":
                    src_idx = key_to_idx[(task, "src")]
                    rel = arrive_out[src_idx] + leg.transfer
                elif kind == "entry":
                    rel = 0.0
                else:
                    rel = float(finish[task - 1])
                dep = max(rel, free)
                arrive = dep + _dist(inst, me, loc, leg.pickup) + leg.loaded
                if (abs(dep - dep_out[idx]) > 1e-9
                        or abs(arrive - arrive_out[idx]) > 1e-9):
                    dep_out[idx], arrive_out[idx] = dep, arrive
                    changed = True
                free = arrive_out[idx]
                loc = leg.dropoff
        if not changed:
            break
    return dep_out


def accept_check(inst: InstanceM2, machines: Sequence[int],
                 os_seq: Sequence[int], time_limit: float = 30.0,
                 tol: float = 1e-6) -> AcceptResult:
    """Fix the MIP to a ``decode_m2`` schedule and re-solve it.

    The model must accept the schedule (feasible) and its objective must
    equal the decoder's objective to ``tol``.
    """
    from .model_m2 import _model_size, build_m2

    model, var = build_m2(inst, machines)
    n_vars, n_constrs = _model_size(model)
    legs: List[Leg] = var["legs"]
    trace = _decode_trace(inst, machines, os_seq)
    decoded = decode_m2(inst.t, inst.power, inst.travel, inst.edges,
                        inst.lu_machine, inst.lu_edge, os_seq, machines)
    for tau in range(inst.n_task):
        if abs(float(trace["start"][tau])
               - float(decoded["start"][tau])) > 1e-6:
            raise AssertionError(
                f"decode trace diverges from decode_m2 at task {tau}")

    key_to_idx = {(leg.task, leg.kind): i for i, leg in enumerate(legs)}
    new_to_model = {i: key_to_idx[(task, kind)]
                    for i, (task, kind) in enumerate(trace["leg_task"])}
    if len(new_to_model) != len(legs):
        raise AssertionError(
            f"decoder produced {len(new_to_model)} legs, model has "
            f"{len(legs)}")

    for tau in range(inst.n_task):
        var["start"][tau].lb = var["start"][tau].ub = float(
            trace["start"][tau])

    free_of = _agv_free_times(inst, machines, trace)
    for idx in range(len(legs)):
        var["dep"][idx].lb = var["dep"][idx].ub = free_of.get(idx, 0.0)

    used_arcs = set()
    first_legs = set()
    for chain in trace["chains"]:
        mapped = [new_to_model[i] for i in chain]
        first_legs.add(mapped[0])
        for a, b in zip(mapped, mapped[1:]):
            used_arcs.add((a, b))
    for idx in range(len(legs)):
        var["first"][idx].lb = var["first"][idx].ub = (
            1.0 if idx in first_legs else 0.0)
    for (a, b), v in var["arcs"].items():
        v.lb = v.ub = 1.0 if (a, b) in used_arcs else 0.0
    for v in var["order"].values():
        v.lb, v.ub = 0.0, 1.0

    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(1.0, float(time_limit))
    model.optimize()
    if model.SolCount == 0:
        return AcceptResult(False, float("nan"), float(decoded["obj"]),
                            float("nan"), n_vars, n_constrs)
    model_obj = float(model.ObjVal) / OBJ_SCALE
    dec_obj = float(decoded["obj"])
    return AcceptResult(abs(model_obj - dec_obj) <= max(tol, tol * dec_obj),
                        model_obj, dec_obj, abs(model_obj - dec_obj),
                        n_vars, n_constrs,
                        detail={"mk": float(decoded["mk"]),
                                "idle": float(decoded["idleT"]),
                                "trans": float(decoded["transT"])})
