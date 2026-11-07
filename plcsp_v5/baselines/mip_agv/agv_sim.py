"""AGV Stage-0 schedule simulator.

Replicates ``Parallel_env`` for a fixed operation order and machine
assignment: source AGV, cross-workshop split, fixed delay, target AGV,
deadhead, and the four cost components.

Two transport models are supported.  ``lu_station=0`` is the original
machine-to-machine model.  ``lu_station=1`` is M2: every workshop has one
load/unload station, the first operation of a job also needs a transport,
and a cross-workshop leg splits at the stations.  The M2 transport term
counts loaded and empty AGV travel.

The AGV choice is either the earliest-free rule (the environment's
fallback) or an explicit per-step plan from the routing model.

Validate against the simulator::

    D:/anaconda/envs/py312/python.exe -m plcsp_v5.baselines.mip_agv.agv_sim --index 0
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from plcsp_v5.agv.state import (
    RHO_EMPTY, AgvState, idle_cost, station_node, transport_timing,
    transport_timing_m2)
from plcsp_v5.baselines.mip_agv.model_noagv import Instance, load_instance


@dataclass
class AgvChoice:
    used: bool
    src: int = -1
    dst: int = -1
    loaded: float = 0.0
    wait: float = 0.0
    cross: bool = False
    empty: float = 0.0


@dataclass
class SimResult:
    starts: List[float]
    finishes: List[float]
    machines: List[int]
    makespan: float
    pec_mean: float
    idle: float
    trans: float
    objective: float
    loaded: float = 0.0
    empty: float = 0.0
    choices: Dict[int, AgvChoice] = field(default_factory=dict)


def _edge_of(edges: np.ndarray, machine: int) -> int:
    for e, members in enumerate(edges):
        if machine in [int(x) for x in members]:
            return e
    raise ValueError(f"machine {machine} is in no workshop")


def _machine_edge(edges: np.ndarray, n_machine: int) -> np.ndarray:
    """Workshop index of every machine."""
    out = np.zeros(n_machine, dtype=int)
    for w, members in enumerate(edges):
        for m in members:
            out[int(m)] = w
    return out


def _new_agvs(n_agv: int, edges: np.ndarray,
              n_machine: Optional[int] = None,
              lu_station: int = 0) -> List[AgvState]:
    n_edge = len(edges)
    out = []
    for k in range(n_agv):
        edge = k % n_edge
        if lu_station:
            if n_machine is None:
                raise ValueError("lu_station=1 needs n_machine")
            location = station_node(edge, n_machine)
        else:
            location = int(edges[edge][0])
        out.append(AgvState(idx=k, edge=edge, location=location))
    return out


Downtime = Sequence[Tuple[float, float]]


def _up_at(t: float, downs: Optional[Downtime]) -> float:
    """Earliest instant at or after ``t`` at which the resource is up."""
    if not downs:
        return t
    moved = True
    while moved:
        moved = False
        for a, b in downs:
            if a <= t < b:
                t = b
                moved = True
    return t


def _place(start: float, duration: float,
           downs: Optional[Downtime]) -> Tuple[float, float]:
    """Start and finish when the resource is down for part of the interval.

    An operation that is running when the resource fails is suspended and
    resumes when it is repaired; it is not restarted.  The downtime windows
    are scanned once in time order, so an interval is charged once.
    """
    if not downs:
        return start, start + duration
    t = _up_at(start, downs)
    finish = t + duration
    for a, b in sorted(downs):
        if b <= t:
            continue
        if a >= finish:
            break
        finish += b - max(a, t)
    return t, finish


def simulate(inst: Instance, order: Sequence[int],
             machines: Sequence[int], n_agv: int = 6,
             cross_delay: float = 5.0, leg_ratio: float = 0.5,
             pick: Optional[Callable[[int, dict], Tuple[int, int]]] = None,
             lu_station: int = 0,
             fixed_starts: Optional[Sequence[float]] = None,
             failures: Optional[dict] = None,
             failure: Optional[dict] = None) -> SimResult:
    """Run one fixed (order, machines) schedule.

    ``pick(step_index, ctx)`` returns ``(src_agv, dst_agv)``; ``dst_agv``
    is ignored for same-workshop trips.  ``None`` uses the earliest-free
    rule, which is the environment's fallback.

    ``lu_station=1`` selects the M2 transport model and requires an
    instance with ``lu_machine`` / ``lu_edge``.

    ``failure`` turns on the same exposure model the environment uses:
    ``{"mtbf": 300, "mttr": 10, "seed": 1}``.  Each operation samples one
    repair from ``constraints.sample_repair``, and the repair is added to its
    processing time, exactly as the disjunctive-graph environment does.

    ``failures`` optionally gives downtime windows,
    ``{"machine": {m: [(a, b), ...]}, "agv": {k: [(a, b), ...]}}``, one list
    per resource.  ``None`` reproduces the no-failure schedule exactly.
    """
    n, m_count = inst.n_task, inst.n_machine
    travel = inst.travel
    edges = inst.edges
    if lu_station and (inst.lu_machine is None or inst.lu_edge is None):
        raise ValueError(
            "lu_station=1 needs an instance with lu_machine / lu_edge; "
            "use a *_lu.pkl pool")
    if failures is not None and not lu_station:
        raise ValueError("failures are implemented for lu_station=1 only")
    fail_base = None
    if failure is not None:
        from plcsp_v5.constraints import sample_repair
        fail_base = int(failure.get("seed", 1))
    agvs = _new_agvs(n_agv, edges, n_machine=m_count,
                     lu_station=lu_station)
    machine_edge = (_machine_edge(edges, m_count) if lu_station else None)
    starts = [0.0] * n
    finishes = [0.0] * n
    routes: List[List[int]] = [[] for _ in range(m_count)]
    choices: Dict[int, AgvChoice] = {}
    loaded_total = 0.0
    empty_total = 0.0

    def _travel(a: int, b: int) -> float:
        if a < 0 or b < 0:
            return 0.0
        return float(travel[int(a)][int(b)])

    for step, tau in enumerate(order):
        tau = int(tau)
        m = int(machines[tau])
        o = tau % inst.n_op
        duration = float(inst.t[tau, m])
        if fail_base is not None:
            # The repair follows the operation, not its position in the order:
            # the generator is keyed by (seed, task, machine), so a search
            # cannot place long operations where the random draws are small.
            rng_tm = np.random.default_rng([fail_base, tau, m])
            duration += sample_repair(rng_tm, duration,
                                      float(failure["mtbf"]),
                                      float(failure["mttr"]))
        start_job = 0.0

        if lu_station:
            prev_m = -1
            if o == 0:
                ready = 0.0
                pickup_edge = int(machine_edge[m])
                dropoff_edge = pickup_edge
                pickup = station_node(pickup_edge, m_count)
                cross = False
            else:
                prev = tau - 1
                prev_m = int(machines[prev])
                ready = finishes[prev]
                pickup = prev_m
                pickup_edge = int(machine_edge[prev_m])
                dropoff_edge = int(machine_edge[m])
                cross = pickup_edge != dropoff_edge

            if o > 0 and prev_m == m:
                start_job = finishes[tau - 1]
            else:
                src_pool = [a for a in agvs if a.edge == pickup_edge]
                dst_pool = [a for a in agvs if a.edge == dropoff_edge]
                if not src_pool:
                    raise ValueError(
                        f"workshop edge {pickup_edge} has no AGV")
                if cross and not dst_pool:
                    raise ValueError(
                        f"workshop edge {dropoff_edge} has no AGV")

                if pick is None:
                    src_idx = int(min(
                        (a.idx for a in src_pool),
                        key=lambda k: agvs[k].free_time))
                    dst_idx = int(min(
                        (a.idx for a in dst_pool),
                        key=lambda k: agvs[k].free_time)) if cross else -1
                else:
                    ctx = {"tau": tau, "prev_m": prev_m, "m": m,
                           "ready": ready, "cross": cross,
                           "pickup_edge": pickup_edge,
                           "dropoff_edge": dropoff_edge,
                           "src_pool": [a.idx for a in src_pool],
                           "dst_pool": [a.idx for a in dst_pool]}
                    src_idx, dst_idx = pick(step, ctx)

                src = agvs[src_idx]
                dst = agvs[dst_idx] if cross else None
                timing = transport_timing_m2(
                    src, pickup, m, ready, _travel,
                    m_count, machine_edge, inst.lu_machine, inst.lu_edge,
                    dst=dst)
                if fail_base is not None:
                    rng_ag = np.random.default_rng([fail_base, 7919, tau])
                    rep = sample_repair(rng_ag, float(timing["loaded"]),
                                        float(failure.get("agv_mtbf",
                                                          failure["mtbf"])),
                                        float(failure.get("agv_mttr",
                                                          failure["mttr"])))
                    timing["src_free"] += rep
                    if dst is not None:
                        timing["dst_free"] += rep
                if failures is not None:
                    downs = failures["agv"]
                    d_src = downs.get(src_idx)
                    if d_src:
                        s0 = max(float(ready), float(src.free_time))
                        _, f1 = _place(s0, timing["src_free"] - s0, d_src)
                        timing["src_free"] = f1
                    if dst is not None:
                        d_dst = downs.get(dst_idx)
                        if d_dst:
                            s0 = max(timing["src_free"] + timing["transfer"],
                                     float(dst.free_time))
                            _, f2 = _place(s0, timing["dst_free"] - s0, d_dst)
                            timing["dst_free"] = f2
                if dst is not None:
                    dst.free_time = timing["dst_free"]
                    dst.location = timing["dst_location"]
                    dst.travelled += timing["dst_travelled"]
                src.free_time = timing["src_free"]
                src.location = timing["src_location"]
                src.travelled += timing["src_travelled"]
                loaded_total += float(timing["loaded"])
                empty_total += float(timing["empty"])
                choices[tau] = AgvChoice(
                    used=True, src=src_idx, dst=dst_idx,
                    loaded=timing["loaded"], wait=timing["wait"],
                    cross=cross, empty=timing["empty"])
                start_job = (timing["dst_free"] if dst is not None
                             else timing["src_free"])
        elif o > 0:
            prev = tau - 1
            prev_m = int(machines[prev])
            if prev_m == m:
                start_job = finishes[prev]
            else:
                ready = finishes[prev]
                pickup_edge = _edge_of(edges, prev_m)
                dropoff_edge = _edge_of(edges, m)
                total = float(travel[prev_m][m])
                cross = pickup_edge != dropoff_edge
                src_pool = [a for a in agvs if a.edge == pickup_edge]
                dst_pool = [a for a in agvs if a.edge == dropoff_edge]

                if pick is None:
                    src_idx = int(min(
                        (a.idx for a in src_pool),
                        key=lambda k: agvs[k].free_time))
                    dst_idx = int(min(
                        (a.idx for a in dst_pool),
                        key=lambda k: agvs[k].free_time)) if cross else -1
                else:
                    ctx = {"tau": tau, "prev_m": prev_m, "m": m,
                           "ready": ready, "cross": cross,
                           "pickup_edge": pickup_edge,
                           "dropoff_edge": dropoff_edge,
                           "src_pool": [a.idx for a in src_pool],
                           "dst_pool": [a.idx for a in dst_pool]}
                    src_idx, dst_idx = pick(step, ctx)

                src = agvs[src_idx]
                if cross:
                    dst = agvs[dst_idx]
                    boundary = int(edges[dropoff_edge][0])
                    timing = transport_timing(
                        src, prev_m, m, ready, total, _travel, dst=dst,
                        cross_delay=cross_delay, leg_ratio=leg_ratio,
                        boundary_machine=boundary)
                    dst.free_time = timing["dst_free"]
                    dst.location = timing["dst_location"]
                    dst.travelled += timing["dst_travelled"]
                    choices[tau] = AgvChoice(
                        used=True, src=src_idx, dst=dst_idx,
                        loaded=timing["loaded"], wait=timing["wait"],
                        cross=True)
                else:
                    timing = transport_timing(
                        src, prev_m, m, ready, total, _travel)
                    choices[tau] = AgvChoice(
                        used=True, src=src_idx, loaded=timing["loaded"],
                        wait=timing["wait"], cross=False)

                src.free_time = timing["src_free"]
                src.location = timing["src_location"]
                src.travelled += timing["src_travelled"]
                loaded_total += float(timing["loaded"])
                start_job = ready + timing["wait"] + timing["loaded"]

        # Left-shift placement, matching _schedule_task in the environment.
        down_m = (failures["machine"].get(m) if failures else None)
        route = routes[m]

        def fit(candidate: float) -> Tuple[float, float]:
            return _place(candidate, duration, down_m)

        if fixed_starts is not None:
            start, finish = fit(float(fixed_starts[tau]))
            route.append(tau)
            starts[tau] = start
            finishes[tau] = finish
            continue
        if not route:
            start, finish = fit(start_job)
            route.append(tau)
        elif fit(start_job)[1] <= starts[route[0]]:
            start, finish = fit(start_job)
            route.insert(0, tau)
        elif len(route) == 1:
            start, finish = fit(max(start_job, finishes[route[-1]]))
            route.append(tau)
        else:
            placed = False
            for i in range(len(route) - 1):
                prev_t, next_t = route[i], route[i + 1]
                s, f = fit(max(start_job, finishes[prev_t]))
                if f > starts[next_t]:
                    continue
                start, finish = s, f
                route.insert(i + 1, tau)
                placed = True
                break
            if not placed:
                start, finish = fit(max(start_job, finishes[route[-1]]))
                route.append(tau)

        starts[tau] = start
        finishes[tau] = finish

    makespan = max(finishes)
    total_energy = sum(float(inst.t[t, machines[t]])
                       * abs(float(inst.power[t, machines[t]]))
                       for t in range(n))
    pec_mean = total_energy / n
    # Match the environment: idle is summed along the machine route in
    # scheduling order, not along start-time order.  Left-shift can make
    # those orders differ.
    gap_idle = 0.0
    for m in range(m_count):
        route = routes[m]
        if not route:
            continue
        gap_idle += starts[route[0]]
        for i in range(len(route) - 1):
            gap_idle += starts[route[i + 1]] - finishes[route[i]]
    # v5 idle term.  "trailing" charges every machine from 0 to the makespan
    # minus its processing time; "gap" keeps the old leading-plus-gap reading.
    proc_by_machine = [0.0] * m_count
    for task in range(len(machines)):
        proc_by_machine[int(machines[task])] += float(
            inst.t[task, int(machines[task])])
    idle = idle_cost(None, gap_idle, proc_by_machine, makespan)
    trans_total = loaded_total + RHO_EMPTY * empty_total
    objective = 0.4 * makespan + 0.4 * (pec_mean + idle) + 0.2 * trans_total
    return SimResult(starts=starts, finishes=finishes,
                     machines=list(machines), makespan=makespan,
                     pec_mean=pec_mean, idle=idle, trans=trans_total,
                     objective=objective, loaded=loaded_total,
                     empty=empty_total, choices=choices)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default=None,
                    help="default: test_Instance_J6M6E2(_lu).pkl")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--n_agv", type=int, default=6)
    ap.add_argument("--lu_station", type=int, default=0,
                    help="1 = M2 load/unload-station transport model")
    args = ap.parse_args()
    if args.instance is None:
        suffix = "_lu" if args.lu_station else ""
        args.instance = f"plcsp_v5/instance/test_Instance_J6M6E2{suffix}.pkl"

    from plcsp_v5.baselines.mip_agv.validate_env import run_env
    from baselines.agv_baselines import pdr_schedule

    inst = load_instance(args.instance, args.index)
    raw = pickle.load(open(args.instance, "rb"))
    if args.lu_station and len(raw) < 6:
        raise SystemExit(
            "--lu_station 1 needs a *_lu.pkl pool with lu_machine/lu_edge")
    tt, edge = raw[2][args.index], raw[3][args.index]
    os_seq, ms = pdr_schedule(inst.t, inst.power, "mor", "spt")
    if args.lu_station:
        from plcsp_v5.baselines.check_agv_env import run_env_m2
        lu_m, lu_e = raw[4][args.index], raw[5][args.index]
        env = run_env_m2(inst.t, inst.power, tt, edge, lu_m, lu_e,
                         os_seq, ms, args.n_agv)
    else:
        env = run_env(inst.t, inst.power, tt, edge, os_seq, ms, 1,
                      args.n_agv)

    order = []
    counter = {j: 0 for j in range(inst.n_job)}
    for j in os_seq:
        j = int(j)
        order.append(j * inst.n_op + counter[j])
        counter[j] += 1
    sim = simulate(inst, order, ms, n_agv=args.n_agv,
                   lu_station=args.lu_station)

    print("component        env        sim       diff")
    for key_env, key_sim in (("mk", "makespan"), ("idleT", "idle"),
                              ("transT", "trans")):
        a, b = float(env[key_env]), float(getattr(sim, key_sim))
        print(f"{key_sim:12s} {a:10.4f} {b:10.4f} {a - b:9.4f}")
    a = float(env["pt"]) / inst.n_task
    print(f"{'pec_mean':12s} {a:10.4f} {sim.pec_mean:10.4f} "
          f"{a - sim.pec_mean:9.4f}")
    print(f"{'objective':12s} {env['obj']:10.4f} {sim.objective:10.4f} "
          f"{env['obj'] - sim.objective:9.4f}")


if __name__ == "__main__":
    main()
