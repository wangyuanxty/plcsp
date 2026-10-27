"""AGV Stage-0 schedule simulator.

Replicates ``Parallel_env`` for a fixed operation order and machine
assignment: source AGV, cross-workshop split, fixed delay, target AGV,
deadhead, and the four cost components.

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

from plcsp_v5.agv.state import AgvState, transport_timing
from plcsp_v5.baselines.mip_agv.model_noagv import Instance, load_instance


@dataclass
class AgvChoice:
    used: bool
    src: int = -1
    dst: int = -1
    loaded: float = 0.0
    wait: float = 0.0
    cross: bool = False


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
    choices: Dict[int, AgvChoice] = field(default_factory=dict)


def _edge_of(edges: np.ndarray, machine: int) -> int:
    for e, members in enumerate(edges):
        if machine in [int(x) for x in members]:
            return e
    raise ValueError(f"machine {machine} is in no workshop")


def _new_agvs(n_agv: int, edges: np.ndarray) -> List[AgvState]:
    n_edge = len(edges)
    out = []
    for k in range(n_agv):
        edge = k % n_edge
        out.append(AgvState(idx=k, edge=edge,
                            location=int(edges[edge][0])))
    return out


def simulate(inst: Instance, order: Sequence[int],
             machines: Sequence[int], n_agv: int = 6,
             cross_delay: float = 5.0, leg_ratio: float = 0.5,
             pick: Optional[Callable[[int, dict], Tuple[int, int]]] = None
             ) -> SimResult:
    """Run one fixed (order, machines) schedule.

    ``pick(step_index, ctx)`` returns ``(src_agv, dst_agv)``; ``dst_agv``
    is ignored for same-workshop trips.  ``None`` uses the earliest-free
    rule, which is the environment's fallback.
    """
    n, m_count = inst.n_task, inst.n_machine
    travel = inst.travel
    edges = inst.edges
    agvs = _new_agvs(n_agv, edges)
    starts = [0.0] * n
    finishes = [0.0] * n
    routes: List[List[int]] = [[] for _ in range(m_count)]
    choices: Dict[int, AgvChoice] = {}
    trans_total = 0.0

    def _travel(a: int, b: int) -> float:
        if a < 0 or b < 0:
            return 0.0
        return float(travel[int(a)][int(b)])

    for step, tau in enumerate(order):
        tau = int(tau)
        m = int(machines[tau])
        o = tau % inst.n_op
        duration = float(inst.t[tau, m])
        start_job = 0.0

        if o > 0:
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
                trans_total += timing["loaded"]
                start_job = ready + timing["wait"] + timing["loaded"]

        # Left-shift placement, matching _schedule_task in the environment.
        route = routes[m]
        if not route:
            start = start_job
            route.append(tau)
        elif start_job + duration <= starts[route[0]]:
            start = start_job
            route.insert(0, tau)
        elif len(route) == 1:
            start = max(start_job, finishes[route[-1]])
            route.append(tau)
        else:
            placed = False
            for i in range(len(route) - 1):
                prev_t, next_t = route[i], route[i + 1]
                if start_job + duration > starts[next_t]:
                    continue
                if starts[next_t] - finishes[prev_t] < duration:
                    continue
                start = max(start_job, finishes[prev_t])
                route.insert(i + 1, tau)
                placed = True
                break
            if not placed:
                start = max(start_job, finishes[route[-1]])
                route.append(tau)

        starts[tau] = start
        finishes[tau] = start + duration

    makespan = max(finishes)
    total_energy = sum(float(inst.t[t, machines[t]])
                       * abs(float(inst.power[t, machines[t]]))
                       for t in range(n))
    pec_mean = total_energy / n
    # Match the environment: idle is summed along the machine route in
    # scheduling order, not along start-time order.  Left-shift can make
    # those orders differ.
    idle = 0.0
    for m in range(m_count):
        route = routes[m]
        if not route:
            continue
        idle += starts[route[0]]
        for i in range(len(route) - 1):
            idle += starts[route[i + 1]] - finishes[route[i]]
    objective = 0.4 * makespan + 0.4 * (pec_mean + idle) + 0.2 * trans_total
    return SimResult(starts=starts, finishes=finishes,
                     machines=list(machines), makespan=makespan,
                     pec_mean=pec_mean, idle=idle, trans=trans_total,
                     objective=objective, choices=choices)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance",
                    default="plcsp_v5/instance/test_Instance_J6M6E2.pkl")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--n_agv", type=int, default=6)
    args = ap.parse_args()

    from plcsp_v5.baselines.mip_agv.validate_env import run_env
    from baselines.agv_baselines import pdr_schedule

    inst = load_instance(args.instance, args.index)
    raw = pickle.load(open(args.instance, "rb"))
    tt, edge = raw[2][args.index], raw[3][args.index]
    os_seq, ms = pdr_schedule(inst.t, inst.power, "mor", "spt")
    env = run_env(inst.t, inst.power, tt, edge, os_seq, ms, 1, args.n_agv)

    order = []
    counter = {j: 0 for j in range(inst.n_job)}
    for j in os_seq:
        j = int(j)
        order.append(j * inst.n_op + counter[j])
        counter[j] += 1
    sim = simulate(inst, order, ms, n_agv=args.n_agv)

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
