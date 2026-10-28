"""MIP baseline for the v5 MT-FJSP under the M2 transport model.

The no-AGV reproduction of Wang et al. (IEEE IoT-J 2025) stays in
``model.py``.  This module adds the AGV layer of the v5 problem:

* per-workshop AGV fleets, ``n_agv = 6``, single load, no charging;
* an entry leg (station -> first machine) for every job;
* same-workshop transport = deadhead + loaded leg;
* cross-workshop transport = source leg (machine -> source station) + a
  station transfer that uses no AGV + target leg (target station ->
  machine); after a source leg the AGV sits at the source station;
* sequencing per AGV, with the deadhead set by the previous leg's
  drop-off node.

Objective (the v5 reporting objective, as in
``plcsp_v5.baselines.agv_baselines.decode_m2``)::

    0.4 * Cmax + 0.4 * (PEC / n_task + idle) + 0.2 * (loaded + 0.5 * empty)

``idle`` is the leading gap plus the internal machine gaps, which equals
``sum_k (last_finish_k - processing_on_k)``.

Formulation
-----------

The AGV layer is a chain cover of the leg set, one cover per workshop.
A leg is served by an AGV of its own workshop, so the legs of a workshop
form disjoint chains and one chain is one AGV route.  The AGVs of a
workshop are identical and start at the same station, so a chain cover
with at most ``n_agv_per_workshop`` chains is exactly an assignment of
legs to AGVs plus a sequence for each AGV.

For leg ``l`` let ``D_l`` be the time the AGV is free to leave its
current location and ``C_l`` its arrival at the drop-off::

    C_l = D_l + loaded_l + dead_l
    dead_l = sum_{l' -> l} a[l', l] * dist(drop[l'], pick[l])
             + f_l * dist(station(edge_l), pick[l])

``f_l = 1`` marks the first leg of a chain.  Each leg has exactly one
predecessor (``f_l + sum_l' a[l', l] = 1``) and at most one successor.
The timing rows ``D_l2 >= C_l1 + dist(drop[l1], pick[l2]) - M(1-a)``
sequence a chain.  Every leg duration is positive, so the timing rows
also rule out subtours.

Model size
----------

At 6x6 the integrated model with a free machine assignment does not fit
the Gurobi size-limited licence (2000 variables, 2000 constraints).  The
machine layer alone costs 1000-2000 rows when the machine choice is
free (measured with ``model.model_size``), and the AGV leg set is only
defined once the machines are fixed, because a leg's pick-up node is the
previous operation's machine.  This module therefore fixes the machine
assignment to a heuristic one (:func:`choose_assignment`) and optimises
everything else exactly: start times, machine sequence and AGV routes.
The fixed-assignment model is a restriction of the integrated model, so
its objective is an upper bound on the integrated optimum.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - import guard
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "gurobipy is required; use D:/anaconda/envs/py312/python.exe") from exc

from ..agv_baselines import decode_m2, pdr_schedule, rga_instance
from ...agv.state import RHO_EMPTY, m2_node_dist, station_node

# v5 reporting weights.
W_MK = 0.4
W_EC = 0.4
W_TT = 0.2
# The model minimises 10 * objective so that every coefficient is an
# integer; reported values divide by this factor.
OBJ_SCALE = 10.0

# Gurobi size-limited licence.
LIMIT_VARS = 2000
LIMIT_CONSTR = 2000

_STATUS_NAMES = {
    1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
    5: "UNBOUNDED", 6: "CUTOFF", 7: "ITERATION_LIMIT", 8: "NODE_LIMIT",
    9: "TIME_LIMIT", 10: "SOLUTION_LIMIT", 11: "INTERRUPTED",
    12: "NUMERIC", 13: "SUBOPTIMAL", 14: "INPROGRESS",
    15: "USER_OBJ_LIMIT", 16: "WORK_LIMIT", 17: "MEM_LIMIT",
}


# ---------------------------------------------------------------------------
# instance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstanceM2:
    """One MT-FJSP instance with the M2 station data."""

    n_job: int
    n_op: int
    n_machine: int
    t: np.ndarray            # (n_task, n_machine), <=0 ineligible
    power: np.ndarray        # (n_task, n_machine)
    travel: np.ndarray       # (n_machine, n_machine)
    edges: np.ndarray        # (n_edge, machines per edge)
    lu_machine: np.ndarray   # (n_machine,) machine -> own station
    lu_edge: np.ndarray      # (n_edge, n_edge) station -> station
    index: int = 0

    @property
    def n_task(self) -> int:
        return self.n_job * self.n_op

    @property
    def n_edge(self) -> int:
        return int(self.edges.shape[0])

    def eligible(self, tau: int) -> List[int]:
        return [m for m in range(self.n_machine) if self.t[tau, m] > 0]

    def machine_edge_array(self) -> np.ndarray:
        out = np.zeros(self.n_machine, dtype=int)
        for w in range(self.n_edge):
            for m in self.edges[w]:
                out[int(m)] = w
        return out


def load_instance_m2(path: str, idx: int) -> InstanceM2:
    """Load one sample from a six-array ``*_lu.pkl`` pool."""
    with open(path, "rb") as fh:
        raw = pickle.load(fh)
    if len(raw) < 6:
        raise ValueError(
            f"{path} has {len(raw)} arrays; the M2 model needs the six-array "
            "_lu pool (t, p, tt, edge_info, lu_machine, lu_edge)")
    t, power, travel, edges = raw[:4]
    t_i = np.asarray(t[idx], dtype=np.float64)
    p_i = np.asarray(power[idx], dtype=np.float64)
    tt_i = np.asarray(travel[idx], dtype=np.float64)
    ed_i = np.asarray(edges[idx], dtype=np.int64)
    lu_m = np.asarray(raw[4][idx], dtype=np.float64)
    lu_e = np.asarray(raw[5][idx], dtype=np.float64)
    n_task, n_machine = t_i.shape
    n_op = n_machine
    if n_task % n_op != 0:
        raise ValueError(f"task count {n_task} not divisible by ops {n_op}")
    return InstanceM2(n_job=n_task // n_op, n_op=n_op, n_machine=n_machine,
                      t=t_i, power=p_i, travel=tt_i, edges=ed_i,
                      lu_machine=lu_m, lu_edge=lu_e, index=idx)


def agv_per_workshop(n_edge: int, n_agv: int) -> List[int]:
    """Fleet size per workshop, matching ``decode_m2``'s initial layout."""
    counts = [0] * n_edge
    for k in range(n_agv):
        counts[k % n_edge] += 1
    return counts


# ---------------------------------------------------------------------------
# legs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Leg:
    """One AGV leg of the M2 transport model."""

    task: int          # operation whose arrival this leg serves
    kind: str          # "entry" | "single" | "src" | "dst"
    edge: int          # workshop of the AGV that serves it
    pickup: int        # node id (machine, or n_machine + workshop)
    dropoff: int       # node id
    loaded: float
    src_leg: int = -1          # for a "dst" leg: the paired source leg
    transfer: float = 0.0      # station transfer time (dst legs)


def _dist(inst: InstanceM2, machine_edge: np.ndarray, a: int, b: int) -> float:
    def travel(x: int, y: int) -> float:
        if x < 0 or y < 0:
            return 0.0
        return float(inst.travel[int(x)][int(y)])

    return float(m2_node_dist(a, b, travel, inst.n_machine, machine_edge,
                              inst.lu_machine, inst.lu_edge))


def build_legs(inst: InstanceM2, machines: Sequence[int]) -> List[Leg]:
    """Transport legs induced by one machine assignment.

    A consecutive pair of operations on the same machine needs no
    transport, exactly as in ``decode_m2``.
    """
    me = inst.machine_edge_array()
    legs: List[Leg] = []
    for i in range(inst.n_job):
        for o in range(inst.n_op):
            tau = i * inst.n_op + o
            m = int(machines[tau])
            de = int(me[m])
            if o == 0:
                legs.append(Leg(
                    task=tau, kind="entry", edge=de,
                    pickup=station_node(de, inst.n_machine), dropoff=m,
                    loaded=_dist(inst, me, station_node(de, inst.n_machine),
                                 m)))
                continue
            pm = int(machines[tau - 1])
            if pm == m:
                continue
            pe = int(me[pm])
            if pe == de:
                legs.append(Leg(task=tau, kind="single", edge=pe,
                                pickup=pm, dropoff=m,
                                loaded=_dist(inst, me, pm, m)))
            else:
                src_idx = len(legs)
                legs.append(Leg(
                    task=tau, kind="src", edge=pe, pickup=pm,
                    dropoff=station_node(pe, inst.n_machine),
                    loaded=_dist(inst, me, pm,
                                 station_node(pe, inst.n_machine))))
                legs.append(Leg(
                    task=tau, kind="dst", edge=de,
                    pickup=station_node(de, inst.n_machine), dropoff=m,
                    loaded=_dist(inst, me, station_node(de, inst.n_machine),
                                 m),
                    src_leg=src_idx, transfer=float(inst.lu_edge[pe][de])))
    return legs


def leg_setup(inst: InstanceM2, legs: Sequence[Leg], a: int, b: int) -> float:
    """Deadhead from leg ``a``'s drop-off to leg ``b``'s pick-up."""
    me = inst.machine_edge_array()
    return _dist(inst, me, legs[a].dropoff, legs[b].pickup)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

@dataclass
class SolutionM2:
    """One MIP solution, in the v5 cost terms."""

    status: str
    ok: bool
    objective: float
    mk: float
    pec: float
    idle: float
    loaded: float
    empty: float
    trans: float
    machines: List[int]
    starts: List[float]
    chains: List[List[int]]
    n_vars: int
    n_constrs: int
    runtime: float
    gap: float
    message: str = ""
    lower_bound: float = float("nan")
    decoded: Dict[str, float] = field(default_factory=dict)


def _machine_pairs(inst: InstanceM2, machines: Sequence[int]
                   ) -> List[Tuple[int, int]]:
    """Operation pairs that share a machine."""
    by_machine: Dict[int, List[int]] = {}
    for tau in range(inst.n_task):
        by_machine.setdefault(int(machines[tau]), []).append(tau)
    out: List[Tuple[int, int]] = []
    for k in sorted(by_machine):
        tasks = by_machine[k]
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                out.append((tasks[i], tasks[j]))
    return out


def _upper_bound(inst: InstanceM2, machines: Sequence[int]) -> float:
    """Feasible makespan of the fixed assignment, as a big-M bound.

    ``decode_m2`` on the PDR job order is a feasible schedule for these
    machines, so its makespan bounds every start time in the model.
    """
    os_seq, _ = pdr_schedule(inst.t, inst.power, "mor", "spt")
    out = decode_m2(inst.t, inst.power, inst.travel, inst.edges,
                    inst.lu_machine, inst.lu_edge, os_seq, machines)
    return float(np.ceil(out["mk"])) + 1.0


def build_m2(inst: InstanceM2, machines: Sequence[int], n_agv: int = 6,
             big_m: Optional[float] = None
             ) -> Tuple["gp.Model", Dict[str, object]]:
    """Build the fixed-assignment M2 MIP.

    Every leg is a decision variable and every within-workshop arc is a
    sequencing row; nothing is dropped.  ``solve_m2`` refuses to solve a
    model that exceeds the licence rather than relaxing it.
    """
    n = inst.n_task
    me = inst.machine_edge_array()
    legs = build_legs(inst, machines)
    n_leg = len(legs)
    for leg in legs:
        if leg.kind == "single" and leg.loaded <= 0.0:
            raise ValueError(
                "zero loaded time on a same-workshop leg; the chain "
                "formulation needs positive leg durations")

    ub = _upper_bound(inst, machines)
    if big_m is None:
        big_m = ub + 2.0 * float(inst.t.max())

    model = gp.Model(f"m2_mip_{inst.index}")
    model.Params.OutputFlag = 0

    start = model.addVars(n, lb=0.0, ub=big_m, vtype=GRB.CONTINUOUS, name="S")
    dur = [float(inst.t[tau, int(machines[tau])]) for tau in range(n)]

    dep = model.addVars(n_leg, lb=0.0, ub=big_m, vtype=GRB.CONTINUOUS,
                        name="D")
    first = model.addVars(n_leg, vtype=GRB.BINARY, name="f")

    setup_first: List[float] = []
    by_edge: Dict[int, List[int]] = {}
    for idx, leg in enumerate(legs):
        by_edge.setdefault(leg.edge, []).append(idx)
        setup_first.append(_dist(inst, me,
                                 station_node(leg.edge, inst.n_machine),
                                 leg.pickup))

    arcs: Dict[Tuple[int, int], "gp.Var"] = {}
    for edge in sorted(by_edge):
        members = by_edge[edge]
        for a in members:
            for b in members:
                if a == b:
                    continue
                arcs[a, b] = model.addVar(vtype=GRB.BINARY,
                                          name=f"a_{a}_{b}")

    preds: Dict[int, List["gp.Var"]] = {i: [] for i in range(n_leg)}
    succs: Dict[int, List["gp.Var"]] = {i: [] for i in range(n_leg)}
    for (a, b), var in arcs.items():
        preds[b].append(var)
        succs[a].append(var)
    for idx in range(n_leg):
        model.addConstr(first[idx] + gp.quicksum(preds[idx]) == 1,
                        name=f"pred_{idx}")
        if succs[idx]:
            model.addConstr(gp.quicksum(succs[idx]) <= 1, name=f"succ_{idx}")

    fleet = agv_per_workshop(inst.n_edge, n_agv)
    for edge in sorted(by_edge):
        model.addConstr(
            gp.quicksum(first[i] for i in by_edge[edge]) <= fleet[edge],
            name=f"fleet_{edge}")

    # C_l = D_l + loaded_l + dead_l.
    arrival: Dict[int, "gp.LinExpr"] = {}
    for idx, leg in enumerate(legs):
        dead: "gp.LinExpr" = first[idx] * setup_first[idx]
        for key in arcs:
            if key[1] == idx:
                dead = dead + arcs[key] * leg_setup(inst, legs, key[0], idx)
        arrival[idx] = dep[idx] + leg.loaded + dead

    for idx, leg in enumerate(legs):
        if leg.kind == "dst":
            model.addConstr(dep[idx] >= arrival[leg.src_leg] + leg.transfer,
                            name=f"xfer_{idx}")
        elif leg.kind in ("single", "src"):
            prev = leg.task - 1
            model.addConstr(dep[idx] >= start[prev] + dur[prev],
                            name=f"rel_{idx}")

    # The AGV is free at ``arrival[a]``; the deadhead of leg ``b`` is
    # charged inside ``arrival[b]`` through the chosen arc.
    for (a, b), var in arcs.items():
        model.addConstr(
            dep[b] >= arrival[a] - big_m * (1 - var), name=f"seq_{a}_{b}")

    # Job precedence: an operation starts when its incoming leg arrives.
    leg_of: Dict[int, int] = {}
    for idx, leg in enumerate(legs):
        if leg.kind in ("entry", "single", "dst"):
            leg_of[leg.task] = idx
    for tau in range(n):
        if tau in leg_of:
            model.addConstr(start[tau] >= arrival[leg_of[tau]],
                            name=f"job_{tau}")
        else:  # same machine: no transport
            model.addConstr(start[tau] >= start[tau - 1] + dur[tau - 1],
                            name=f"job_{tau}")

    order: Dict[Tuple[int, int], "gp.Var"] = {}
    for a, b in _machine_pairs(inst, machines):
        order[a, b] = model.addVar(vtype=GRB.BINARY, name=f"y_{a}_{b}")
        model.addConstr(
            start[b] >= start[a] + dur[a] - big_m * (1 - order[a, b]),
            name=f"mo1_{a}_{b}")
        model.addConstr(
            start[a] >= start[b] + dur[b] - big_m * order[a, b],
            name=f"mo2_{a}_{b}")

    last = model.addVars(inst.n_machine, lb=0.0, ub=big_m,
                         vtype=GRB.CONTINUOUS, name="last")
    by_machine: Dict[int, List[int]] = {}
    for tau in range(n):
        by_machine.setdefault(int(machines[tau]), []).append(tau)
    for k, tasks in by_machine.items():
        for tau in tasks:
            model.addConstr(last[k] >= start[tau] + dur[tau],
                            name=f"last_{k}_{tau}")
    mkt = model.addVar(lb=0.0, ub=big_m, vtype=GRB.CONTINUOUS, name="MKT")
    for i in range(inst.n_job):
        tau = i * inst.n_op + inst.n_op - 1
        model.addConstr(mkt >= start[tau] + dur[tau], name=f"mk_{i}")

    pec = sum(float(inst.t[tau, int(machines[tau])])
              * float(inst.power[tau, int(machines[tau])])
              for tau in range(n))
    proc_total = sum(dur)
    idle = gp.quicksum(last[k] for k in range(inst.n_machine)) - proc_total
    loaded = sum(leg.loaded for leg in legs)
    empty = (gp.quicksum(first[idx] * setup_first[idx] for idx in range(n_leg))
             + gp.quicksum(arcs[key] * leg_setup(inst, legs, key[0], key[1])
                           for key in arcs))

    model.setObjective(
        OBJ_SCALE * (W_MK * mkt + W_EC * (pec / n + idle)
                     + W_TT * (loaded + RHO_EMPTY * empty)),
        GRB.MINIMIZE)
    model.update()
    var = {"start": start, "dep": dep, "first": first, "arcs": arcs,
           "last": last, "mkt": mkt, "order": order, "legs": legs,
           "arrival": arrival, "dur": dur, "big_m": big_m, "ub": ub}
    return model, var


def model_size_m2(inst: InstanceM2, machines: Sequence[int],
                  n_agv: int = 6) -> Tuple[int, int]:
    """(variables, constraints) of the full fixed-assignment model."""
    model, _ = build_m2(inst, machines, n_agv)
    return _model_size(model)


def _model_size(model: "gp.Model") -> Tuple[int, int]:
    return (int(model.NumVars),
            int(model.NumConstrs) + int(model.NumGenConstrs)
            + int(model.NumQConstrs))


# ---------------------------------------------------------------------------
# assignment heuristic
# ---------------------------------------------------------------------------

def score_assignment(inst: InstanceM2, machines: Sequence[int],
                     os_seq: Sequence[int]) -> float:
    """``decode_m2`` objective of one assignment and one job order."""
    return float(decode_m2(inst.t, inst.power, inst.travel, inst.edges,
                           inst.lu_machine, inst.lu_edge,
                           os_seq, machines)["obj"])


def choose_assignment(inst: InstanceM2, rga_seeds: int = 1,
                      rga_pop: int = 30, rga_gen: int = 60,
                      seed: int = 1) -> Tuple[np.ndarray, float, str]:
    """Best machine assignment over the PDR rules and a short RGA run.

    The M2 MIP fixes the machine assignment (see the module docstring),
    so this choice matters.  Every candidate is scored with
    ``decode_m2`` on its own job order, which is the same objective the
    PDR and RGA baselines report.
    """
    best: Optional[np.ndarray] = None
    best_obj = float("inf")
    best_src = ""
    for jr in ("fifo", "mor", "lwkr_t", "lwkr_pt", "mwkr_t", "mwkr_pt"):
        for mr in ("spt", "seyc"):
            os_seq, ms = pdr_schedule(inst.t, inst.power, jr, mr)
            obj = score_assignment(inst, ms, os_seq)
            if obj < best_obj:
                best, best_obj, best_src = ms.copy(), obj, f"pdr:{jr}+{mr}"
    for s in range(rga_seeds):
        out = rga_instance(inst.t, inst.power, inst.travel, inst.edges,
                           seed=seed + s, pop=rga_pop, gen=rga_gen,
                           return_schedule=True,
                           lu_machine=inst.lu_machine,
                           lu_edge=inst.lu_edge)
        if out["obj"] < best_obj:
            best = np.asarray(out["ms"], dtype=int)
            best_obj, best_src = float(out["obj"]), f"rga:{seed + s}"
    if best is None:  # pragma: no cover - PDR always yields one
        raise RuntimeError("no assignment found")
    return np.asarray(best, dtype=int), best_obj, best_src


# ---------------------------------------------------------------------------
# solve, extract, verify
# ---------------------------------------------------------------------------

def _status_name(model: "gp.Model") -> str:
    return _STATUS_NAMES.get(int(model.Status), f"STATUS_{int(model.Status)}")


def chains_from_solution(legs: Sequence[Leg], arcs: Dict[Tuple[int, int],
                                                        "gp.Var"],
                         first) -> List[List[int]]:
    """Ordered leg chains implied by the arc variables."""
    first_ids = [i for i in range(len(legs)) if first[i].X > 0.5]
    succ: Dict[int, int] = {a: b for (a, b), var in arcs.items()
                            if var.X > 0.5}
    chains: List[List[int]] = []
    for f in sorted(first_ids):
        chain = [f]
        while chain[-1] in succ:
            chain.append(succ[chain[-1]])
        chains.append(chain)
    return chains


def _extract(inst: InstanceM2, model: "gp.Model", var: Dict[str, object],
             machines: Sequence[int], runtime: float, n_vars: int,
             n_constrs: int, status: str, gap: float, bound: float
             ) -> SolutionM2:
    from .verify_m2 import verify_m2

    chains = chains_from_solution(var["legs"], var["arcs"], var["first"])
    starts = [float(var["start"][tau].X) for tau in range(inst.n_task)]
    chk = verify_m2(inst, machines, starts, chains)
    return SolutionM2(
        status=status, ok=True, objective=float(chk["obj"]),
        mk=float(chk["mk"]), pec=float(chk["pec"]), idle=float(chk["idle"]),
        loaded=float(chk["loaded"]), empty=float(chk["empty"]),
        trans=float(chk["trans"]), machines=[int(m) for m in machines],
        starts=starts, chains=chains, n_vars=n_vars, n_constrs=n_constrs,
        runtime=runtime, gap=gap, lower_bound=bound,
        message="" if chk["feasible"] else "schedule failed verification")


def _polish(model: "gp.Model", var: Dict[str, object]) -> bool:
    """Round the binaries, fix them, and re-solve the continuous part.

    A big-M row turns the integer tolerance into a constraint
    violation: a binary at ``1e-8`` shifts a row by ``big_m * 1e-8``.
    Rounding and re-solving removes that, so the extracted schedule is
    feasible for the rounded (integral) decisions.
    """
    for v in var["order"].values():
        v.lb = v.ub = float(round(v.X))
    for v in var["arcs"].values():
        v.lb = v.ub = float(round(v.X))
    for i in range(len(var["legs"])):
        f = var["first"][i]
        f.lb = f.ub = float(round(f.X))
    model.Params.MIPGap = 0.0
    model.optimize()
    return model.SolCount > 0


def solve_m2(inst: InstanceM2, machines: Sequence[int], n_agv: int = 6,
             time_limit: float = 60.0, mip_gap: float = 1e-4,
             threads: int = 0, verbose: bool = False) -> SolutionM2:
    """Build and solve the fixed-assignment M2 MIP."""
    t0 = time.time()
    model, var = build_m2(inst, machines, n_agv)
    if verbose:
        model.Params.OutputFlag = 1
    model.Params.TimeLimit = max(1.0, float(time_limit))
    model.Params.MIPGap = float(mip_gap)
    model.Params.IntFeasTol = 1e-9
    model.Params.Seed = 0
    if threads > 0:
        model.Params.Threads = int(threads)
    n_vars, n_constrs = _model_size(model)
    if n_vars > LIMIT_VARS or n_constrs > LIMIT_CONSTR:
        return SolutionM2(
            status="LICENSE_LIMIT", ok=False, objective=float("nan"),
            mk=float("nan"), pec=float("nan"), idle=float("nan"),
            loaded=float("nan"), empty=float("nan"), trans=float("nan"),
            machines=[int(m) for m in machines], starts=[], chains=[],
            n_vars=n_vars, n_constrs=n_constrs,
            runtime=time.time() - t0, gap=float("nan"),
            message=(f"model needs {n_vars} variables and {n_constrs} "
                     f"constraints; licence allows {LIMIT_VARS}/"
                     f"{LIMIT_CONSTR}"))
    try:
        model.optimize()
    except gp.GurobiError as exc:  # pragma: no cover - licence errors
        return SolutionM2(
            status="GUROBI_ERROR", ok=False, objective=float("nan"),
            mk=float("nan"), pec=float("nan"), idle=float("nan"),
            loaded=float("nan"), empty=float("nan"), trans=float("nan"),
            machines=[int(m) for m in machines], starts=[], chains=[],
            n_vars=n_vars, n_constrs=n_constrs,
            runtime=time.time() - t0, gap=float("nan"), message=str(exc))
    runtime = time.time() - t0
    if model.SolCount == 0:
        return SolutionM2(
            status=_status_name(model), ok=False, objective=float("nan"),
            mk=float("nan"), pec=float("nan"), idle=float("nan"),
            loaded=float("nan"), empty=float("nan"), trans=float("nan"),
            machines=[int(m) for m in machines], starts=[], chains=[],
            n_vars=n_vars, n_constrs=n_constrs, runtime=runtime,
            gap=float("nan"), message="no incumbent")
    status = _status_name(model)
    gap = float(model.MIPGap)
    bound = float(model.ObjBound) / OBJ_SCALE
    if not _polish(model, var):
        return SolutionM2(
            status=status, ok=False, objective=float("nan"),
            mk=float("nan"), pec=float("nan"), idle=float("nan"),
            loaded=float("nan"), empty=float("nan"), trans=float("nan"),
            machines=[int(m) for m in machines], starts=[], chains=[],
            n_vars=n_vars, n_constrs=n_constrs, runtime=runtime,
            gap=gap, message="rounding the incumbent made it infeasible")
    return _extract(inst, model, var, machines, runtime, n_vars, n_constrs,
                    status, gap, bound)
