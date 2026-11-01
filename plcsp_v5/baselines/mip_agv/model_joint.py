"""Joint CP-SAT model for the v5 MT-FJSP with an AGV fleet and LU stations.

Unlike the two-stage baseline in ``run_batch.py``, this model decides the
machine assignment, the operation starts, the transport legs and the AGV
routes in one model, and reports the CP-SAT bound so the gap to the joint
optimum is measurable.

Transport semantics follow the simulator (``plcsp_v5/agv/state.py``):

* node ids ``0 .. M-1`` are machines, node ``M + e`` is the station of
  workshop ``e``;
* the first operation of a job needs an entry leg
  ``station(edge(m)) -> m``;
* two consecutive operations on the same machine need no transport;
* two consecutive operations on different machines in the same workshop need
  one leg ``m_prev -> m``;
* a cross-workshop move is two legs, ``m_prev -> station(edge(m_prev))`` and
  ``station(edge(m)) -> m``, separated by ``CROSS_DELAY`` that consumes no
  AGV;
* an AGV never leaves its home workshop.

Objective, as in the simulator::

    0.4 * makespan + 0.4 * (PEC / n_task + idle)
  + 0.2 * (loaded + 0.5 * empty)

with ``loaded`` the sum of the loaded leg durations and ``empty`` the sum of
the deadhead distances.

Run with the OR-Tools interpreter::

    D:/anaconda/envs/py312/python.exe -m plcsp_v5.baselines.mip_agv.model_joint \
        --instance plcsp_v5/instance/test_Instance_J6M6E2_lu.pkl --index 0
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from plcsp_v5.baselines.mip_agv.model_noagv import (SCALE, Instance,
                                                    _horizon, load_instance)

CROSS_DELAY = 5.0
RHO_EMPTY_NUM, RHO_EMPTY_DEN = 1, 2   # 0.5, kept integral


def _machine_edge(inst: Instance) -> List[int]:
    edge = [0] * inst.n_machine
    for e, machines in enumerate(inst.edges):
        for m in machines:
            edge[int(m)] = int(e)
    return edge


def _node_distance(inst: Instance, edge: List[int]) -> List[int]:
    """Flat ``n_node * n_node`` integer distance table with M2 semantics."""
    m_count, n_ws = inst.n_machine, len(inst.edges)
    n_node = m_count + n_ws

    def one(a: int, b: int) -> int:
        if a < 0 or b < 0:
            return 0
        a_st, b_st = a >= m_count, b >= m_count
        if not a_st and not b_st:
            return int(round(float(inst.travel[a][b]) * SCALE))
        if a_st and b_st:
            return int(round(float(inst.lu_edge[a - m_count][b - m_count])
                             * SCALE))
        if not a_st and b_st:
            w = b - m_count
            if edge[a] == w:
                return int(round(float(inst.lu_machine[a]) * SCALE))
            return int(round((float(inst.lu_machine[a])
                              + float(inst.lu_edge[edge[a]][w])) * SCALE))
        w = a - m_count
        if edge[b] == w:
            return int(round(float(inst.lu_machine[b]) * SCALE))
        return int(round((float(inst.lu_edge[w][edge[b]])
                          + float(inst.lu_machine[b])) * SCALE))

    return [one(a, b) for a in range(n_node) for b in range(n_node)]


def solve_joint(inst: Instance, n_agv: int = 6, time_limit: float = 600.0,
                workers: int = 8,
                fix_machines: Optional[List[int]] = None,
                hint_machines: Optional[List[int]] = None,
                score: bool = True,
                partition: bool = True) -> Optional[dict]:
    started = time.time()
    model = cp_model.CpModel()
    n, m_count = inst.n_task, inst.n_machine
    n_ws = len(inst.edges)
    n_node = m_count + n_ws
    edge = _machine_edge(inst)
    horizon = _horizon(inst)
    dist_flat = _node_distance(inst, edge)
    big = 1 << 30

    # ------------------------------------------------------------- machines
    x: Dict[Tuple[int, int], cp_model.IntVar] = {}
    start: Dict[int, cp_model.IntVar] = {}
    end: Dict[int, cp_model.IntVar] = {}
    midx: Dict[int, cp_model.IntVar] = {}
    eidx: Dict[int, cp_model.IntVar] = {}
    per_machine: Dict[int, List] = {m: [] for m in range(m_count)}

    for tau in range(n):
        elig = inst.eligible(tau)
        start[tau] = model.NewIntVar(0, horizon, f"s{tau}")
        end[tau] = model.NewIntVar(0, horizon, f"e{tau}")
        midx[tau] = model.NewIntVar(0, m_count - 1, f"mx{tau}")
        eidx[tau] = model.NewIntVar(0, n_ws - 1, f"ex{tau}")
        for m in elig:
            x[tau, m] = model.NewBoolVar(f"x{tau}_{m}")
        model.AddExactlyOne(x[tau, m] for m in elig)
        model.Add(midx[tau] == sum(m * x[tau, m] for m in elig))
        model.AddElement(midx[tau], edge, eidx[tau])
        pvar = model.NewIntVar(0, horizon, f"p{tau}")
        model.AddElement(midx[tau],
                         [inst.proc(tau, m) if inst.t[tau, m] > 0 else 0
                          for m in range(m_count)], pvar)
        model.Add(end[tau] == start[tau] + pvar)
        for m in elig:
            per_machine[m].append(model.NewOptionalIntervalVar(
                start[tau], inst.proc(tau, m), end[tau], x[tau, m],
                f"iv{tau}_{m}"))
    for m in range(m_count):
        model.AddNoOverlap(per_machine[m])
    if fix_machines is not None:
        for tau in range(n):
            model.Add(midx[tau] == int(fix_machines[tau]))
    if hint_machines is not None:
        for tau in range(n):
            model.AddHint(midx[tau], int(hint_machines[tau]))

    # ------------------------------------------------------------------ legs
    lu_scaled = [int(round(float(v) * SCALE)) for v in inst.lu_machine]
    travel_flat = [int(round(float(inst.travel[a][b]) * SCALE))
                   for a in range(m_count) for b in range(m_count)]
    # Cross-workshop legs follow ``split_cross_edge``: the machine-to-machine
    # travel is cut into a fixed delay that consumes no AGV plus a source leg
    # and a destination leg.  The simulator charges those two legs, not the
    # machine-to-station distance.
    src_flat, dst_flat = [], []
    for a in range(m_count):
        for b in range(m_count):
            total = float(inst.travel[a][b])
            delay = min(max(0.0, CROSS_DELAY), total)
            rest = total - delay
            src_flat.append(int(round(rest * 0.5 * SCALE)))
            dst_flat.append(int(round((rest - rest * 0.5) * SCALE)))

    def new_leg(tau: int, slot: str) -> dict:
        leg = {
            "tau": tau, "slot": slot,
            "ws": model.NewIntVar(0, n_ws - 1, f"ws{tau}{slot}"),
            "pick": model.NewIntVar(0, n_node - 1, f"pk{tau}{slot}"),
            "drop": model.NewIntVar(0, n_node - 1, f"dp{tau}{slot}"),
            "dur": model.NewIntVar(0, horizon, f"du{tau}{slot}"),
            "active": model.NewBoolVar(f"ac{tau}{slot}"),
            "start": model.NewIntVar(0, horizon, f"ls{tau}{slot}"),
        }
        leg["end"] = model.NewIntVar(0, horizon, f"le{tau}{slot}")
        model.Add(leg["end"] == leg["start"] + leg["dur"])
        model.Add(leg["dur"] == 0).OnlyEnforceIf(leg["active"].Not())
        model.Add(leg["start"] == 0).OnlyEnforceIf(leg["active"].Not())
        return leg

    legs: List[dict] = []
    for j in range(inst.n_job):
        for o in range(inst.n_op):
            tau = j * inst.n_op + o
            la, lb = new_leg(tau, "a"), new_leg(tau, "b")
            legs.extend([la, lb])
            if o == 0:
                model.Add(la["active"] == 1)
                model.Add(la["ws"] == eidx[tau])
                model.Add(la["pick"] == m_count + la["ws"])
                model.Add(la["drop"] == midx[tau])
                model.AddElement(midx[tau], lu_scaled, la["dur"])
                # the entry leg is released at 0 but the AGV may be busy, so
                # its start is left to the route timing
                model.Add(start[tau] >= la["end"])
                model.Add(lb["active"] == 0)
                continue

            prev = tau - 1
            model.Add(start[tau] >= end[prev])   # job precedence
            same_m = model.NewBoolVar(f"sm{tau}")
            model.Add(midx[prev] == midx[tau]).OnlyEnforceIf(same_m)
            model.Add(midx[prev] != midx[tau]).OnlyEnforceIf(same_m.Not())
            same_e = model.NewBoolVar(f"se{tau}")
            model.Add(eidx[prev] == eidx[tau]).OnlyEnforceIf(same_e)
            model.Add(eidx[prev] != eidx[tau]).OnlyEnforceIf(same_e.Not())

            # slot a: the same-workshop leg, or the source leg of a cross move
            model.Add(la["active"] == same_m.Not())
            model.Add(la["pick"] == midx[prev])
            model.Add(la["ws"] == eidx[prev])
            model.Add(la["start"] >= end[prev]).OnlyEnforceIf(la["active"])
            pair_a = model.NewIntVar(0, m_count * m_count - 1, f"ia{tau}")
            model.Add(pair_a == m_count * midx[prev] + midx[tau])
            model.Add(la["drop"] == midx[tau]).OnlyEnforceIf(same_e)
            model.AddElement(pair_a, travel_flat, la["dur"]) \
                .OnlyEnforceIf(same_e)
            model.Add(la["drop"] == m_count + eidx[prev]) \
                .OnlyEnforceIf(same_e.Not())
            model.AddElement(pair_a, src_flat, la["dur"]) \
                .OnlyEnforceIf(same_e.Not())

            # slot b: the destination leg of a cross move
            cross = model.NewBoolVar(f"cx{tau}")
            model.AddBoolAnd([same_m.Not(), same_e.Not()]).OnlyEnforceIf(cross)
            model.AddBoolOr([same_m, same_e]).OnlyEnforceIf(cross.Not())
            model.Add(lb["active"] == cross)
            model.Add(lb["ws"] == eidx[tau]).OnlyEnforceIf(cross)
            model.Add(lb["pick"] == m_count + eidx[tau]).OnlyEnforceIf(cross)
            model.Add(lb["drop"] == midx[tau]).OnlyEnforceIf(cross)
            model.AddElement(pair_a, dst_flat, lb["dur"]).OnlyEnforceIf(cross)
            model.Add(lb["start"] >= la["end"]
                      + int(round(CROSS_DELAY * SCALE))).OnlyEnforceIf(cross)
            model.Add(start[tau] >= lb["end"]).OnlyEnforceIf(cross)
            model.Add(start[tau] >= la["end"]).OnlyEnforceIf(cross.Not())

    # ------------------------------------------------------------ AGV routes
    per_ws = max(1, n_agv // n_ws)
    agv_ws = [min(r // per_ws, n_ws - 1) for r in range(n_agv)]
    n_leg = len(legs)

    # An AGV never leaves its own workshop.  When the machine assignment is
    # fixed, every leg belongs to exactly one workshop, so the circuit of an
    # AGV needs its own legs only.  Without this every AGV carries every leg,
    # which is quadratic in the fleet size and made the 10x10 model take
    # minutes before the solver started.
    leg_ws: Optional[List[int]] = None
    if fix_machines is not None and partition:
        ws_op = [int(edge[int(fix_machines[t])]) for t in range(n)]
        leg_ws = []
        for j in range(inst.n_job):
            for o in range(inst.n_op):
                tau = j * inst.n_op + o
                if o == 0:
                    leg_ws.append(ws_op[tau])
                    leg_ws.append(ws_op[tau])
                else:
                    leg_ws.append(ws_op[tau - 1])
                    leg_ws.append(ws_op[tau])

    assign: Dict[Tuple[int, int], cp_model.IntVar] = {}
    for li in range(n_leg):
        for r in range(n_agv):
            assign[li, r] = model.NewBoolVar(f"a{li}_{r}")
    for li, leg in enumerate(legs):
        model.Add(sum(assign[li, r] for r in range(n_agv)) == leg["active"])
        for r in range(n_agv):
            ok = model.NewBoolVar(f"ok{li}_{r}")
            model.Add(leg["ws"] == agv_ws[r]).OnlyEnforceIf(ok)
            model.Add(leg["ws"] != agv_ws[r]).OnlyEnforceIf(ok.Not())
            model.AddImplication(assign[li, r], ok)

    def _dist_var(u_drop, v_pick, name):
        idx = model.NewIntVar(0, n_node * n_node - 1, name + "i")
        model.Add(idx == n_node * u_drop + v_pick)
        d = model.NewIntVar(0, max(dist_flat), name)
        model.AddElement(idx, dist_flat, d)
        return d

    # The deadhead distance between two legs depends on the legs, not on the
    # AGV, so it is built once and reused.
    pair_dist: Dict[Tuple[int, int], cp_model.IntVar] = {}
    for li in range(n_leg):
        for lj in range(n_leg):
            if lj == li:
                continue
            if leg_ws is not None and leg_ws[li] != leg_ws[lj]:
                continue
            pair_dist[li, lj] = _dist_var(legs[li]["drop"], legs[lj]["pick"],
                                          f"qe{li}_{lj}")

    empty_terms: List[cp_model.IntVar] = []
    for r in range(n_agv):
        my = [li for li in range(n_leg)
              if leg_ws is None or leg_ws[li] == agv_ws[r]]
        node = {li: k + 1 for k, li in enumerate(my)}
        n_cnode = len(my) + 1               # node 0 is the depot
        arc: Dict[Tuple[int, int], cp_model.IntVar] = {}
        circuit = []
        for u in range(n_cnode):
            for v in range(n_cnode):
                lit = model.NewBoolVar(f"c{r}_{u}_{v}")
                arc[u, v] = lit
                circuit.append((u, v, lit))
        model.AddCircuit(circuit)
        # the depot self-loop means "this AGV serves no leg"
        model.Add(sum(assign[li, r] for li in my) == 0) \
            .OnlyEnforceIf(arc[0, 0])
        for li in my:
            model.Add(arc[node[li], node[li]] == 1 - assign[li, r])

        for li in my:
            u = node[li]
            d = _dist_var(m_count + agv_ws[r], legs[li]["pick"],
                          f"qd{r}_{li}")
            model.Add(legs[li]["start"] >= d).OnlyEnforceIf(arc[0, u])
            dd = model.NewIntVar(0, max(dist_flat), f"zd{r}_{li}")
            model.Add(dd == d).OnlyEnforceIf(arc[0, u])
            model.Add(dd == 0).OnlyEnforceIf(arc[0, u].Not())
            empty_terms.append(dd)
            for lj in my:
                if lj == li:
                    continue
                v = node[lj]
                d2 = pair_dist[li, lj]
                model.Add(legs[lj]["start"] >= legs[li]["end"] + d2)                     .OnlyEnforceIf(arc[u, v])
                dd2 = model.NewIntVar(0, max(dist_flat), f"ze{r}_{li}_{lj}")
                model.Add(dd2 == d2).OnlyEnforceIf(arc[u, v])
                model.Add(dd2 == 0).OnlyEnforceIf(arc[u, v].Not())
                empty_terms.append(dd2)

    # -------------------------------------------------------------- objective
    makespan = model.NewIntVar(0, horizon, "cmax")
    model.AddMaxEquality(makespan, [end[t] for t in range(n)])
    energy = model.NewIntVar(0, horizon * n, "energy")
    model.Add(energy == sum(inst.energy(t, m) * x[t, m]
                            for t in range(n) for m in inst.eligible(t)))
    # Trailing idle: sum over machines of (makespan - processing time on m),
    # which equals n_machine * makespan - total processing time.
    proc_total = sum(inst.proc(t, m) * x[t, m]
                     for t in range(n) for m in inst.eligible(t))
    idle = model.NewIntVar(0, horizon * m_count, "idle")
    model.Add(idle == m_count * makespan - proc_total)
    loaded = model.NewIntVar(0, horizon * n, "loaded")
    model.Add(loaded == sum(leg["dur"] for leg in legs))
    empty = model.NewIntVar(0, big, "empty")
    model.Add(empty == sum(empty_terms) if empty_terms else 0)

    denom = n
    total = model.NewIntVar(0, 1 << 60, "total")
    # F = 0.4*C + 0.4*(E/N + idle) + 0.2*(L + 0.5*Em); multiply by 10*N
    model.Add(total == 4 * denom * makespan
              + 4 * (energy + denom * idle)
              + 2 * denom * loaded
              + denom * empty)
    model.Minimize(total)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": solver.StatusName(status), "feasible": False,
                "seconds": round(time.time() - started, 1)}
    scale = 10 * denom * SCALE
    out = {
        "status": solver.StatusName(status),
        "feasible": True,
        "objective": float(solver.ObjectiveValue()) / scale,
        "bound": float(solver.BestObjectiveBound()) / scale,
        "seconds": round(time.time() - started, 1),
        "mk": solver.Value(makespan) / SCALE,
        "pec_mean": solver.Value(energy) / (denom * SCALE),
        "idle": solver.Value(idle) / SCALE,
        "loaded": solver.Value(loaded) / SCALE,
        "empty": solver.Value(empty) / SCALE,
    }
    if score:
        out.update(_score_solution(solver, inst, start, midx, assign, legs,
                                   n_agv, n))
    return out



def _score_solution(solver, inst, start, midx, assign, legs, n_agv, n):
    """Rescore the joint solution with the reference simulator."""
    from plcsp_v5.baselines.mip_agv.agv_sim import simulate
    from plcsp_v5.baselines.mip_agv.validate_env import task_order_from_starts

    machines = [int(solver.Value(midx[t])) for t in range(n)]
    starts = [solver.Value(start[t]) / SCALE for t in range(n)]
    order = task_order_from_starts(starts, inst.n_job, inst.n_op)

    plan = {}
    for tau in range(n):
        src = None
        for r in range(n_agv):
            if solver.Value(assign[2 * tau, r]):
                src = r
                break
        dst = -1
        for r in range(n_agv):
            if solver.Value(assign[2 * tau + 1, r]):
                dst = r
                break
        plan[tau] = (src if src is not None else 0, dst)

    def pick(step, ctx):
        s, d = plan.get(ctx["tau"], (0, -1))
        if d is None or d < 0 or not ctx["dst_pool"]:
            d = ctx["dst_pool"][0] if ctx["dst_pool"] else -1
        return (s if s in ctx["src_pool"] else ctx["src_pool"][0],
                d)

    sim = simulate(inst, order, machines, n_agv=n_agv, pick=pick,
                   lu_station=1, fixed_starts=starts)
    return {"sim_objective": float(sim.objective),
            "sim_makespan": float(sim.makespan),
            "sim_trans": float(sim.trans),
            "sim_idle": float(sim.idle)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--n_agv", type=int, default=6)
    ap.add_argument("--time_limit", type=float, default=600.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    inst = load_instance(args.instance, args.index)
    print(json.dumps(solve_joint(inst, args.n_agv, args.time_limit,
                                 args.workers), indent=1))
