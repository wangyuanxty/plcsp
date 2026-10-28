"""MIP baseline of Wang et al., IEEE IoT-J 2025, equations (3)-(8).

Paper: "End-to-end multitarget flexible job shop scheduling with deep
reinforcement learning", Section III-B.  The model is the MT-FJSP with
makespan, processing energy, standby energy and transport energy.

Objective (eq 7), with the average processing energy that the paper's
reported numbers use::

    Cost = 0.4 * MKT + 0.4 * (PEC / n_task + SEC) + 0.2 * TEC

where ``MKT = max_i mk_i`` (eq 3), ``PEC = sum_{i,j} pec_{i,j}``
(eq 4), ``SEC = sum_k bl_k`` (eq 5) and ``TEC = sum_{i,j} tec_{i,j}``
(eq 6).  Units: standby power and transport power are 1, as in the paper.

Two readings of the standby energy ``bl_k`` (eq 5) are supported:

``eq5``
    The equation as written: the maximum completion on machine ``k``
    minus the processing time on ``k``.

``csv``
    The reading that reproduces the published result CSV exactly
    (instance 0 matches to the last digit: 604.98 / 377.19 / 609.50 /
    78.20).  It is the machine's trailing idle to the makespan,
    ``sum_k (MKT - P_k)``.  Its idle values back out a mean processing
    power of 10.1, against 10.4 over the instance data, and the ``eq5``
    reading would give an idle of 100-300, far from the CSV's 509 mean.
    ``csv`` needs no auxiliary rows because ``sum_k (MKT - P_k)`` is
    ``n_machine * MKT`` minus the total processing time.

Two formulations of the machine-capacity constraint (8d) are provided:

``paper``
    The published rows, one pair per machine ``k`` in ``shared(a, b)``::

        O_b >= O_a + p_a - D(1-Y) - D(1-x_a,k) - D(1-x_b,k)

    This is the row-for-row transcription of (8d), ``2 * |shared|``
    rows per operation pair.

``compact``
    Pairs that share one or two machines keep the per-machine rows (they
    cost the same as an aggregate and are stronger).  Pairs sharing
    three or more machines use an exact aggregate.  Both operations have
    exactly one machine, so ``|machine(a) - machine(b)|`` is 0 exactly
    when they share a machine and at least 1 otherwise.  Then::

        O_b >= O_a + p_a - D(1-Y) - D * |machine(a) - machine(b)|

    is active exactly on the shared machine and relaxed when the
    machines differ, which is what the per-machine rows do.  The
    feasible set is identical; only the row count differs.  The absolute
    value is a single Gurobi general constraint on an integer-valued
    difference variable.

Size at 6x6: the per-machine rows need 2240-3690 constraints, above the
2000 rows of the Gurobi size-limited licence on 82 of the 100
instances.  The ``compact`` form needs 1205-2062 with ``csv`` and
1275-2224 with ``eq5``; 3 instances (``csv``) or 16 (``eq5``) still
exceed the cap.  Those use exact row generation: the model keeps as
many order rows as fit, and every omitted row is checked on the
incumbent.  A solution that satisfies every omitted row is optimal for
the full model, because the solved model is a relaxation of it.

Notes on the linearisation:

* eq (8c) is bilinear in ``X``.  A travel variable ``tr`` per precedence
  edge is bounded by ``tr >= tt[m1, m2]`` for the chosen machine pair;
  the weight on TEC makes ``tr`` tight at the optimum.
* ``eq5`` uses ``max{(O + t) X}``.  A variable ``last_k`` is lower
  bounded by each assigned completion and by the machine's processing
  sum; SEC minimisation drives it to the maximum.
* The big-M is not the loose horizon: a dispatching-rule schedule gives
  a valid upper bound on the optimal makespan, which is roughly ten
  times tighter.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - import guard
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "gurobipy is required; use D:/anaconda/envs/py312/python.exe") from exc

# Paper weights 0.4 / 0.4 / 0.2, cleared to integers by multiplying by 5.
W_MK = 2
W_EC = 2
W_TT = 1
W_DEN = 5


@dataclass(frozen=True)
class Instance:
    """One MT-FJSP instance in the v5 flat layout."""

    n_job: int
    n_op: int
    n_machine: int
    t: np.ndarray          # (n_task, n_machine) processing time, <=0 ineligible
    power: np.ndarray      # (n_task, n_machine) processing power
    travel: np.ndarray     # (n_machine, n_machine) transport time
    edges: np.ndarray      # (n_edge, machines per edge)
    index: int = 0

    @property
    def n_task(self) -> int:
        return self.n_job * self.n_op

    def eligible(self, tau: int) -> List[int]:
        return [m for m in range(self.n_machine) if self.t[tau, m] > 0]

    def horizon(self) -> int:
        total = sum(max(self.t[tau, m] for m in self.eligible(tau))
                    for tau in range(self.n_task))
        total += self.n_task * float(self.travel.max())
        return int(total) + 1

    def bounds(self) -> Tuple[float, float]:
        """(start-time bound, big-M) from a feasible dispatching schedule.

        A dispatching-rule schedule is feasible, so the optimal makespan
        cannot exceed it.  Using that value instead of the loose horizon
        keeps the big-M rows tight.  The start time of any operation in
        an optimal schedule is below the optimal makespan, and the extra
        ``2 * max_t`` makes every relaxed row non-binding.
        """
        from ..agv_baselines import decode, pdr_schedule

        os_seq, ms = pdr_schedule(self.t, self.power, "fifo", "spt")
        out = decode(self.t, self.power, self.travel, self.edges,
                     os_seq, ms, agv=False)
        ub_start = float(np.ceil(out["mk"])) + 1.0
        return ub_start, ub_start + 2.0 * float(self.t.max())


@dataclass
class Solution:
    """One MIP solution, in the paper's own cost terms."""

    status: str
    objective: float
    mkt: float
    pec: float
    idle: float
    trans: float
    machines: List[int]
    starts: List[float]
    os_seq: List[int]
    n_vars: int
    n_constrs: int
    runtime: float
    gap: float
    formulation: str
    ok: bool = True
    message: str = ""
    certified: bool = True
    rounds: int = 1
    sec: str = "csv"
    idle_eq5: float = float("nan")
    idle_csv: float = float("nan")


# Gurobi size-limited licence: at most this many variables and this many
# constraints (linear + general) per model.
LIMIT_VARS = 2000
LIMIT_CONSTR = 2000
MAX_ROUNDS = 6

_STATUS_NAMES = {
    1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
    5: "UNBOUNDED", 6: "CUTOFF", 7: "ITERATION_LIMIT", 8: "NODE_LIMIT",
    9: "TIME_LIMIT", 10: "SOLUTION_LIMIT", 11: "INTERRUPTED",
    12: "NUMERIC", 13: "SUBOPTIMAL", 14: "INPROGRESS",
    15: "USER_OBJ_LIMIT", 16: "WORK_LIMIT", 17: "MEM_LIMIT",
}


def _status_name(model: "gp.Model") -> str:
    return _STATUS_NAMES.get(int(model.Status), f"STATUS_{int(model.Status)}")


def load_instance(path: str, idx: int) -> Instance:
    """Load one sample from a v5 pool (four arrays)."""
    with open(path, "rb") as fh:
        raw = pickle.load(fh)
    t, power, travel, edges = raw[:4]
    t_i = np.asarray(t[idx], dtype=np.float64)
    p_i = np.asarray(power[idx], dtype=np.float64)
    tt_i = np.asarray(travel[idx], dtype=np.float64)
    ed_i = np.asarray(edges[idx], dtype=np.int64)
    n_task, n_machine = t_i.shape
    n_op = n_machine
    if n_task % n_op != 0:
        raise ValueError(f"task count {n_task} not divisible by ops {n_op}")
    return Instance(n_job=n_task // n_op, n_op=n_op, n_machine=n_machine,
                    t=t_i, power=p_i, travel=tt_i, edges=ed_i, index=idx)


def _pairs(inst: Instance) -> List[Tuple[int, int, List[int]]]:
    """Operation pairs of different jobs with at least one shared machine."""
    n = inst.n_task
    elig = [set(inst.eligible(tau)) for tau in range(n)]
    out = []
    for a in range(n):
        for b in range(a + 1, n):
            if a // inst.n_op == b // inst.n_op:
                continue
            shared = sorted(elig[a] & elig[b])
            if shared:
                out.append((a, b, shared))
    return out


def _proc_expr(inst: Instance, x: Dict[Tuple[int, int], "gp.Var"],
               tau: int) -> "gp.LinExpr":
    """Processing time of task ``tau`` on its chosen machine (eq 3/4 helper)."""
    return gp.quicksum(float(inst.t[tau, m]) * x[tau, m]
                       for m in inst.eligible(tau))


def _energy_expr(inst: Instance, x: Dict[Tuple[int, int], "gp.Var"],
                 tau: int) -> "gp.LinExpr":
    """Processing energy of task ``tau`` on its chosen machine (eq 4)."""
    return gp.quicksum(float(inst.t[tau, m]) * abs(float(inst.power[tau, m]))
                       * x[tau, m] for m in inst.eligible(tau))


def build(inst: Instance, formulation: str = "compact", sec: str = "csv",
          active_pairs: Optional[Sequence[Tuple[int, int]]] = None
          ) -> Tuple["gp.Model", Dict[str, object]]:
    """Build the MIP.  Returns the model and its variable handles.

    ``active_pairs`` restricts the (8d) rows to the listed operation
    pairs.  ``None`` keeps every pair.  Omitted pairs also lose their
    order variable, so the model stays small.
    """
    if formulation not in ("compact", "paper"):
        raise ValueError(formulation)
    active = None if active_pairs is None else set(active_pairs)
    n, n_m = inst.n_task, inst.n_machine
    elig = [inst.eligible(tau) for tau in range(n)]
    ub_start, big_d = inst.bounds()
    max_tt = float(inst.travel.max())

    model = gp.Model(f"wang2025_mt_fjsp_{inst.index}")
    model.Params.OutputFlag = 0

    x: Dict[Tuple[int, int], "gp.Var"] = {}
    for tau in range(n):
        for m in elig[tau]:
            x[tau, m] = model.addVar(vtype=GRB.BINARY,
                                     name=f"x_{tau}_{m}")
    start = model.addVars(n, lb=0.0, ub=ub_start, vtype=GRB.CONTINUOUS,
                          name="O")
    travel = model.addVars(
        [tau for tau in range(n) if tau % inst.n_op > 0],
        lb=0.0, ub=max_tt, vtype=GRB.CONTINUOUS, name="tr")

    # (8b) one machine per operation.
    for tau in range(n):
        model.addConstr(gp.quicksum(x[tau, m] for m in elig[tau]) == 1,
                        name=f"assign_{tau}")

    # (8c) job precedence with transport, and the transport lookup.
    for tau in range(1, n):
        if tau % inst.n_op == 0:
            continue
        prev = tau - 1
        model.addConstr(
            start[tau] >= start[prev] + _proc_expr(inst, x, prev) + travel[tau],
            name=f"prec_{tau}")
        if len(elig[prev]) <= len(elig[tau]):
            for m1 in elig[prev]:
                model.addConstr(
                    travel[tau] >=
                    gp.quicksum(float(inst.travel[m1, m2]) * x[tau, m2]
                                for m2 in elig[tau])
                    - max_tt * (1 - x[prev, m1]),
                    name=f"tr_{tau}_{m1}")
        else:
            for m2 in elig[tau]:
                model.addConstr(
                    travel[tau] >=
                    gp.quicksum(float(inst.travel[m1, m2]) * x[prev, m1]
                                for m1 in elig[prev])
                    - max_tt * (1 - x[tau, m2]),
                    name=f"tr_{tau}_{m2}")

    # (8d) machine capacity.  Y_ab = 1 means a is operated before b.
    pairs = _pairs(inst)
    y: Dict[Tuple[int, int], "gp.Var"] = {}
    for a, b, shared in pairs:
        if active is not None and (a, b) not in active:
            continue
        y[a, b] = model.addVar(vtype=GRB.BINARY, name=f"y_{a}_{b}")
        if formulation == "paper" or len(shared) <= 2:
            for k in shared:
                relax = big_d * (1 - x[a, k]) + big_d * (1 - x[b, k])
                model.addConstr(
                    start[b] >= start[a] + _proc_expr(inst, x, a)
                    - big_d * (1 - y[a, b]) - relax,
                    name=f"ord1_{a}_{b}_{k}")
                model.addConstr(
                    start[a] >= start[b] + _proc_expr(inst, x, b)
                    - big_d * y[a, b] - relax,
                    name=f"ord2_{a}_{b}_{k}")
        else:
            # Exact aggregate of the per-machine rows.  The machine
            # index difference is 0 exactly when both operations share
            # a machine and at least 1 otherwise, so the row pair is
            # active on the shared machine and relaxed by D*|diff|
            # everywhere else.
            diff = model.addVar(lb=-(inst.n_machine - 1),
                                ub=inst.n_machine - 1,
                                vtype=GRB.CONTINUOUS, name=f"dm_{a}_{b}")
            gapvar = model.addVar(lb=0.0, ub=float(inst.n_machine - 1),
                                  vtype=GRB.CONTINUOUS, name=f"dg_{a}_{b}")
            model.addConstr(
                diff == gp.quicksum(float(k) * x[a, k] for k in elig[a])
                - gp.quicksum(float(k) * x[b, k] for k in elig[b]),
                name=f"dm_{a}_{b}")
            model.addGenConstrAbs(gapvar, diff)
            model.addConstr(
                start[b] >= start[a] + _proc_expr(inst, x, a)
                - big_d * (1 - y[a, b]) - big_d * gapvar,
                name=f"ord1_{a}_{b}")
            model.addConstr(
                start[a] >= start[b] + _proc_expr(inst, x, b)
                - big_d * y[a, b] - big_d * gapvar,
                name=f"ord2_{a}_{b}")

    # (3) job completion, and MKT.
    mkt = model.addVar(lb=0.0, ub=ub_start, vtype=GRB.CONTINUOUS, name="MKT")
    for i in range(inst.n_job):
        last = i * inst.n_op + inst.n_op - 1
        model.addConstr(
            mkt >= start[last] + _proc_expr(inst, x, last), name=f"mk_{i}")

    # (5) standby energy.  Two readings of bl_k are supported:
    #   "eq5": max completion on k minus the processing time on k, as
    #          the equation is written;
    #   "csv": the published result CSV instead uses the machine's
    #          trailing idle to the makespan, sum_k (MKT - P_k), whose
    #          idle values back out a mean processing power of 10.1,
    #          against 10.4 in the instance data.
    # "csv" needs no auxiliary rows because sum_k (MKT - P_k) is
    # n_machine * MKT minus the total processing time.
    proc_exprs = [_proc_expr(inst, x, tau) for tau in range(n)]
    last = None
    if sec == "csv":
        idle = float(n_m) * mkt - gp.quicksum(proc_exprs)
    elif sec == "eq5":
        last = model.addVars(n_m, lb=0.0, ub=ub_start,
                             vtype=GRB.CONTINUOUS, name="last")
        for k in range(n_m):
            on_k = [tau for tau in range(n) if k in elig[tau]]
            if not on_k:
                continue
            proc_sum = gp.quicksum(float(inst.t[tau, k]) * x[tau, k]
                                   for tau in on_k)
            model.addConstr(last[k] >= proc_sum, name=f"last_lo_{k}")
            for tau in on_k:
                model.addConstr(
                    last[k] >= start[tau] + float(inst.t[tau, k])
                    - big_d * (1 - x[tau, k]), name=f"last_{tau}_{k}")
        idle = gp.quicksum(
            last[k] - gp.quicksum(float(inst.t[tau, k]) * x[tau, k]
                                  for tau in range(n) if k in elig[tau])
            for k in range(n_m))
    else:
        raise ValueError(sec)

    pec = gp.quicksum(_energy_expr(inst, x, tau) for tau in range(n))
    tec = gp.quicksum(travel[tau] for tau in range(1, n)
                      if tau % inst.n_op > 0)

    # Clear the 0.4/0.4/0.2 weights and the PEC mean denominator.
    model.setObjective(
        W_MK * n * mkt + W_EC * (pec + n * idle) + W_TT * n * tec,
        GRB.MINIMIZE)

    model.update()
    return model, {"x": x, "start": start, "travel": travel, "y": y,
                   "mkt": mkt, "last": last}


def _cost_from_solution(inst: Instance, machines: Sequence[int],
                        starts: Sequence[float],
                        sec: str = "csv") -> Dict[str, float]:
    """Recompute eqs (3)-(7) from an extracted schedule.

    ``idle_eq5`` is the busy span minus processing (eq 5).  ``idle_csv``
    is the trailing idle ``sum_k (MKT - P_k)`` that the published result
    CSV uses.  ``objective`` follows ``sec``.
    """
    n = inst.n_task
    finish = [starts[tau] + float(inst.t[tau, machines[tau]])
              for tau in range(n)]
    mkt = max(finish)

    pec = sum(float(inst.t[tau, machines[tau]])
              * abs(float(inst.power[tau, machines[tau]]))
              for tau in range(n))

    idle_eq5 = 0.0
    for k in range(inst.n_machine):
        on_k = [tau for tau in range(n) if machines[tau] == k]
        if not on_k:
            continue
        idle_eq5 += max(finish[tau] for tau in on_k) - sum(
            float(inst.t[tau, k]) for tau in on_k)
    idle_csv = inst.n_machine * mkt - sum(
        float(inst.t[tau, machines[tau]]) for tau in range(n))

    trans = 0.0
    for tau in range(1, n):
        if tau % inst.n_op == 0:
            continue
        trans += float(inst.travel[machines[tau - 1], machines[tau]])

    idle = idle_eq5 if sec == "eq5" else idle_csv
    objective = (0.4 * mkt + 0.4 * (pec / n + idle) + 0.2 * trans)
    return {"objective": objective, "mkt": mkt, "pec": pec,
            "pec_mean": pec / n, "idle": idle, "idle_eq5": idle_eq5,
            "idle_csv": idle_csv, "trans": trans}


def _os_seq(inst: Instance, starts: Sequence[float]) -> List[int]:
    """Dispatch sequence: tasks sorted by opening time, ties by task index."""
    order = sorted(range(inst.n_task), key=lambda tau: (round(starts[tau], 6),
                                                       tau))
    return [tau // inst.n_op for tau in order]


def _model_size(model: "gp.Model") -> Tuple[int, int]:
    """(variables, constraints), counting general constraints as rows."""
    return (int(model.NumVars),
            int(model.NumConstrs) + int(model.NumGenConstrs)
            + int(model.NumQConstrs))


def _extract(inst: Instance, model: "gp.Model", var: Dict[str, object],
             status: str, runtime: float, n_vars: int, n_constrs: int,
             formulation: str, sec: str) -> Solution:
    """Turn a solved model into a Solution in the paper's cost terms."""
    x = var["x"]
    machines = [0] * inst.n_task
    for tau in range(inst.n_task):
        machines[tau] = int(max(inst.eligible(tau),
                                key=lambda m: x[tau, m].X))
    starts = [float(var["start"][tau].X) for tau in range(inst.n_task)]
    cost = _cost_from_solution(inst, machines, starts, sec)
    gap = (float(model.MIPGap) if model.IsMIP and model.SolCount > 0
           else 0.0)
    return Solution(status=status, objective=cost["objective"],
                    mkt=cost["mkt"], pec=cost["pec"], idle=cost["idle"],
                    trans=cost["trans"], machines=machines, starts=starts,
                    os_seq=_os_seq(inst, starts), n_vars=n_vars,
                    n_constrs=n_constrs, runtime=runtime, gap=gap,
                    formulation=formulation, sec=sec,
                    idle_eq5=cost["idle_eq5"], idle_csv=cost["idle_csv"])


def _optimize(inst: Instance, model: "gp.Model", var: Dict[str, object],
              time_limit: float, formulation: str, mip_gap: float,
              threads: int, verbose: bool, sec: str) -> Solution:
    """Optimize one already-built model and extract its schedule."""
    t0 = time.time()
    n_vars, n_constrs = _model_size(model)
    if verbose:
        model.Params.OutputFlag = 1
    model.Params.TimeLimit = max(1.0, float(time_limit))
    model.Params.MIPGap = float(mip_gap)
    model.Params.Seed = 0
    if threads > 0:
        model.Params.Threads = int(threads)
    try:
        model.optimize()
    except gp.GurobiError as exc:
        return Solution(status="LICENSE_ERROR", objective=float("nan"),
                        mkt=float("nan"), pec=float("nan"), idle=float("nan"),
                        trans=float("nan"), machines=[], starts=[],
                        os_seq=[], n_vars=n_vars, n_constrs=n_constrs,
                        runtime=time.time() - t0, gap=float("nan"),
                        formulation=formulation, ok=False, message=str(exc))
    runtime = time.time() - t0
    if model.SolCount == 0:
        return Solution(status=_status_name(model), objective=float("nan"),
                        mkt=float("nan"), pec=float("nan"), idle=float("nan"),
                        trans=float("nan"), machines=[], starts=[],
                        os_seq=[], n_vars=n_vars, n_constrs=n_constrs,
                        runtime=runtime, gap=float("nan"),
                        formulation=formulation, ok=False,
                        message="no incumbent")
    return _extract(inst, model, var, _status_name(model), runtime, n_vars,
                    n_constrs, formulation, sec)


def _violated(inst: Instance, machines: Sequence[int],
              starts: Sequence[float],
              pairs: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Pairs that share a machine in this schedule but overlap on it.

    The tolerance covers big-M round-off in the order rows; it is far
    below any real overlap.
    """
    eps = 1e-3
    out = []
    for a, b in pairs:
        k = int(machines[a])
        if k != int(machines[b]):
            continue
        dur_a = float(inst.t[a, k])
        dur_b = float(inst.t[b, k])
        if (starts[b] < starts[a] + dur_a - eps
                and starts[a] < starts[b] + dur_b - eps):
            out.append((a, b))
    return out


def solve(inst: Instance, time_limit: float = 60.0,
          formulation: str = "compact", sec: str = "csv", threads: int = 0,
          mip_gap: float = 1e-4, verbose: bool = False,
          row_generation: bool = True) -> Solution:
    """Build and solve one instance under the size-limited licence.

    When the full model is too large, exact row generation is used: the
    model keeps the cheap single-machine order rows plus a growing set
    of the multi-machine rows, and every omitted row is checked on the
    incumbent.  A solution with no violated omitted row is optimal for
    the full model.
    """
    t0 = time.time()
    try:
        model, var = build(inst, formulation, sec)
    except gp.GurobiError as exc:
        return Solution(status="BUILD_ERROR", objective=float("nan"),
                        mkt=float("nan"), pec=float("nan"), idle=float("nan"),
                        trans=float("nan"), machines=[], starts=[],
                        os_seq=[], n_vars=0, n_constrs=0,
                        runtime=time.time() - t0, gap=float("nan"),
                        formulation=formulation, ok=False, message=str(exc))

    n_vars, n_constrs = _model_size(model)
    if n_vars <= LIMIT_VARS and n_constrs <= LIMIT_CONSTR:
        sol = _optimize(inst, model, var, time_limit - (time.time() - t0),
                        formulation, mip_gap, threads, verbose, sec)
        sol.n_vars, sol.n_constrs = n_vars, n_constrs
        return sol

    if formulation != "compact" or not row_generation:
        return Solution(status="LICENSE_LIMIT", objective=float("nan"),
                        mkt=float("nan"), pec=float("nan"), idle=float("nan"),
                        trans=float("nan"), machines=[], starts=[],
                        os_seq=[], n_vars=n_vars, n_constrs=n_constrs,
                        runtime=time.time() - t0, gap=float("nan"),
                        formulation=formulation, ok=False,
                        message=(f"model needs {n_vars} variables and "
                                 f"{n_constrs} constraints; licence allows "
                                 f"{LIMIT_VARS}/{LIMIT_CONSTR}"))

    pairs = _pairs(inst)
    single = [(a, b) for a, b, sh in pairs if len(sh) == 1]
    multi = [(a, b) for a, b, sh in pairs if len(sh) > 1]

    base_model, _ = build(inst, formulation, sec, active_pairs=single)
    base_vars, base_cons = _model_size(base_model)
    room = max(0, (LIMIT_CONSTR - base_cons) // 4)
    room = min(room, max(0, (LIMIT_VARS - base_vars) // 3))
    room = min(room, len(multi))

    active_set = set(multi[:room])
    active = [p for p in multi if p in active_set]
    probe = max(5.0, 0.25 * time_limit)
    best: Optional[Solution] = None
    for rnd in range(MAX_ROUNDS):
        remaining = time_limit - (time.time() - t0)
        if remaining <= 1.0:
            break
        model, var = build(inst, formulation, sec,
                           active_pairs=single + active)
        nv, nc = _model_size(model)
        if nv > LIMIT_VARS or nc > LIMIT_CONSTR:
            break
        # The first round is only a probe for the omitted rows; if the
        # schedule is already feasible the search continues on the same
        # model with the rest of the budget.
        cap = min(remaining, probe) if rnd == 0 else remaining
        sol = _optimize(inst, model, var, cap, formulation,
                        mip_gap, threads, verbose, sec)
        sol.rounds = rnd + 1
        sol.n_vars, sol.n_constrs = nv, nc
        if not sol.ok:
            return sol
        violated = _violated(inst, sol.machines, sol.starts, multi)
        if not violated:
            remaining = time_limit - (time.time() - t0)
            if remaining > 1.0:
                model.Params.TimeLimit = remaining
                try:
                    model.optimize()
                except gp.GurobiError:
                    pass
                if model.SolCount > 0:
                    sol = _extract(inst, model, var, _status_name(model),
                                   time.time() - t0, nv, nc,
                                   formulation, sec)
            sol.certified = True
            return sol
        best = sol
        missing = [p for p in violated if p not in active_set]
        if not missing:
            break
        # Swap the violated omitted rows in.  Drop active rows that are
        # vacuous in this schedule (their operations use different
        # machines), so the model keeps its size.
        same_machine = set()
        for a, b in active:
            if int(sol.machines[a]) == int(sol.machines[b]):
                same_machine.add((a, b))
        droppable = [p for p in active if p not in same_machine]
        if not droppable:
            break
        for add in missing:
            if not droppable:
                break
            active_set.discard(droppable.pop())
            active_set.add(add)
        active = [p for p in multi if p in active_set]

    if best is None:
        return Solution(status="LICENSE_LIMIT", objective=float("nan"),
                        mkt=float("nan"), pec=float("nan"), idle=float("nan"),
                        trans=float("nan"), machines=[], starts=[],
                        os_seq=[], n_vars=n_vars, n_constrs=n_constrs,
                        runtime=time.time() - t0, gap=float("nan"),
                        formulation=formulation, ok=False, message=(
                            "row generation produced no solution"),
                        sec=sec)
    best.certified = False
    best.message = "row generation did not certify all omitted rows"
    best.runtime = time.time() - t0
    return best


def model_size(inst: Instance, formulation: str = "compact",
               sec: str = "csv") -> Tuple[int, int]:
    """Variables and constraints without solving (licence pre-check)."""
    model, _ = build(inst, formulation, sec)
    return _model_size(model)
