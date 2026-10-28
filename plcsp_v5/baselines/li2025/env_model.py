# -*- coding: utf-8 -*-
"""Partial-schedule state and cost accounting for the adapted Li et al. baseline.

The time accounting mirrors ``plcsp_v5/baselines/agv_baselines.decode`` exactly
when the AGV choice is left to the earliest-free rule.  The policy layer passes
explicit AGV choices instead; the rest of the decoder (left-shift insertion,
per-workshop fleets, cross-workshop split, loaded-time accounting, machine idle
energy) is unchanged.

The v5 objective is::

    0.4 * makespan
    + 0.4 * (processing_energy / n_task + machine_idle_energy)
    + 0.2 * loaded_transport_time

``processing_energy`` is the sum of ``t * p`` (negative values mean energy
recovery, as in the v5 data).  ``machine_idle_energy`` is the sum of idle gaps
before the last operation of every used machine, as in the v5 decoder.

Two transport models are supported.  The default (M1) is machine-to-machine
travel with a fixed cross-workshop delay; only loaded travel enters the
objective.  With ``Instance.lu_station`` the M2 model applies: every workshop
has one load/unload station, every operation including the first needs
transport, and the objective counts loaded plus empty AGV travel at unit
power.  M2 timing comes from ``agv.state.transport_timing_m2``, so it matches
the v5 environment and ``agv_baselines.decode_m2``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:  # scripts add ``plcsp_v5/`` to ``sys.path``
    from agv.state import (DEFAULT_IDLE_MODE, RHO_EMPTY, idle_cost,
                           m2_node_dist, resolve_idle_mode, station_node,
                           transport_timing_m2)
except ImportError:  # repository root only
    from plcsp_v5.agv.state import (DEFAULT_IDLE_MODE, RHO_EMPTY,
                                    idle_cost, m2_node_dist,
                                    resolve_idle_mode, station_node,
                                    transport_timing_m2)

CROSS_DELAY = 5.0
LEG_RATIO = 0.5
N_AGV = 6
W_MK = 0.4
W_EC = 0.4
W_TT = 0.2


def edge_of(edge_info: tuple[tuple[int, ...], ...], m: int) -> int:
    """Return the workshop that owns machine ``m``."""
    for e, members in enumerate(edge_info):
        if int(m) in [int(x) for x in members]:
            return e
    return 0


def travel(tt: np.ndarray, a: int, b: int) -> float:
    """Transport time lookup; negative machines mean "no location"."""
    if a < 0 or b < 0:
        return 0.0
    return float(tt[int(a)][int(b)])


def split_cross(total: float, delay: float = CROSS_DELAY,
                ratio: float = LEG_RATIO) -> tuple[float, float, float]:
    """Split one cross-workshop trip into (source leg, delay, target leg)."""
    total = max(0.0, float(total))
    d = min(max(0.0, float(delay)), total)
    rem = total - d
    src = rem * min(max(float(ratio), 0.0), 1.0)
    return src, d, rem - src


def norm01(values: np.ndarray) -> np.ndarray:
    """Min-max normalisation; a constant vector maps to zeros."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo <= 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def moments(values: np.ndarray) -> tuple[float, float]:
    """Return (mean, population std); an empty vector maps to (0, 0)."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return 0.0, 0.0
    return float(x.mean()), float(x.std(ddof=0))


def minmax4(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean, std, min, max); an empty vector maps to zeros."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(x.mean()), float(x.std(ddof=0)), float(x.min()), float(x.max())


@dataclass(frozen=True)
class Instance:
    """One v5 instance: processing times, power, travel matrix, workshops.

    ``lu_machine[m]`` is the distance from machine ``m`` to the station of its
    workshop and ``lu_edge[a][b]`` the distance between the stations of
    workshops ``a`` and ``b``.  Old four-array pools leave both ``None``.
    ``lu_station`` selects the M2 transport model.
    """

    t: np.ndarray
    p: np.ndarray
    tt: np.ndarray
    edge: tuple[tuple[int, ...], ...]
    n_job: int
    n_machine: int
    n_ops: int
    lu_machine: np.ndarray | None = None
    lu_edge: np.ndarray | None = None
    lu_station: bool = False

    @classmethod
    def from_arrays(cls, t: np.ndarray, p: np.ndarray, tt: np.ndarray,
                    edge: object, lu_machine: object = None,
                    lu_edge: object = None,
                    lu_station: bool = False) -> "Instance":
        t = np.asarray(t, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        tt = np.asarray(tt, dtype=np.float64)
        if t.ndim != 2 or t.shape != p.shape or tt.shape != (t.shape[1],) * 2:
            raise ValueError(f"bad instance shapes: t={t.shape}, "
                             f"p={p.shape}, tt={tt.shape}")
        n_task, n_machine = t.shape
        if n_task % n_machine:
            raise ValueError(f"n_task={n_task} is not a multiple of "
                             f"n_machine={n_machine}")
        n_job = n_task // n_machine
        edge_t = tuple(tuple(int(x) for x in row) for row in edge)
        lu_m = None
        lu_e = None
        if lu_machine is not None:
            lu_m = np.asarray(lu_machine, dtype=np.float64)
            if lu_m.shape != (n_machine,):
                raise ValueError(f"bad lu_machine shape: {lu_m.shape}")
        if lu_edge is not None:
            lu_e = np.asarray(lu_edge, dtype=np.float64)
            if lu_e.shape != (len(edge_t), len(edge_t)):
                raise ValueError(f"bad lu_edge shape: {lu_e.shape}")
        if lu_station and (lu_m is None or lu_e is None):
            raise ValueError("lu_station=1 needs lu_machine and lu_edge")
        return cls(t, p, tt, edge_t, n_job, n_machine, n_machine,
                   lu_m, lu_e, bool(lu_station))

    @property
    def n_task(self) -> int:
        return self.n_job * self.n_ops

    @property
    def n_edge(self) -> int:
        return len(self.edge)

    def machines(self, task: int) -> np.ndarray:
        """Available machines of ``task`` (negative t entries are masked)."""
        return np.flatnonzero(self.t[int(task)] >= 0)

    def edge_of(self, m: int) -> int:
        return edge_of(self.edge, m)

    def travel(self, a: int, b: int) -> float:
        return travel(self.tt, a, b)


@dataclass
class Agv:
    """Minimal AGV state used by the v5 Stage-0 transport model."""

    idx: int
    edge: int
    location: int
    free_time: float = 0.0
    travelled: float = 0.0


@dataclass
class Route:
    """One operation placed on a machine."""

    start: float
    finish: float
    task: int


@dataclass
class ScheduleSession:
    """Incremental serial schedule-generation scheme over one instance.

    The session exposes the partial schedule, the AGV states, and the rule
    inputs (job/machine/AGV attributes) at every decision point.  ``apply``
    commits one operation; the caller chooses the job, the machine, and the
    source AGV.
    """

    inst: Instance
    n_agv: int = N_AGV
    idle_mode: str = DEFAULT_IDLE_MODE
    routes: list[list[Route]] = field(init=False, default_factory=list)
    agvs: list[Agv] = field(init=False, default_factory=list)
    start: np.ndarray = field(init=False)
    finish: np.ndarray = field(init=False)
    machine_of: np.ndarray = field(init=False)
    op_count: np.ndarray = field(init=False)
    scheduled: np.ndarray = field(init=False)
    proc_energy: float = field(init=False, default=0.0)
    trans_total: float = field(init=False, default=0.0)
    loaded_total: float = field(init=False, default=0.0)
    empty_total: float = field(init=False, default=0.0)
    _mean_tt: float = field(init=False, default=0.0)
    machine_edge: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n_task = self.inst.n_task
        self.idle_mode = resolve_idle_mode(self.idle_mode)
        self.routes = [[] for _ in range(self.inst.n_machine)]
        self.start = np.full(n_task, np.nan, dtype=np.float64)
        self.finish = np.full(n_task, np.nan, dtype=np.float64)
        self.machine_of = np.full(n_task, -1, dtype=int)
        self.op_count = np.zeros(self.inst.n_job, dtype=int)
        self.scheduled = np.zeros(n_task, dtype=bool)
        self.machine_edge = np.zeros(self.inst.n_machine, dtype=int)
        for w, members in enumerate(self.inst.edge):
            for m in members:
                self.machine_edge[int(m)] = w
        for k in range(int(self.n_agv)):
            e = k % self.inst.n_edge
            if self.inst.lu_station:
                # M2: every AGV waits at the station of its workshop.
                loc = station_node(e, self.inst.n_machine)
            else:
                loc = int(self.inst.edge[e][0])
            self.agvs.append(Agv(idx=k, edge=e, location=loc))
        off = ~np.eye(self.inst.n_machine, dtype=bool)
        self._mean_tt = (float(self.inst.tt[off].mean())
                         if bool(off.any()) else 0.0)

    # ------------------------------------------------------------------
    # Job and machine state
    # ------------------------------------------------------------------
    def pending_jobs(self) -> list[int]:
        """Jobs with at least one unscheduled operation."""
        return [j for j in range(self.inst.n_job)
                if self.op_count[j] < self.inst.n_ops]

    def next_task(self, j: int) -> int:
        """Task index of the next operation of job ``j``."""
        return int(j) * self.inst.n_ops + int(self.op_count[j])

    def job_ready(self, j: int) -> float:
        """Earliest release time of the next operation of job ``j``."""
        c = int(self.op_count[j])
        if c == 0:
            return 0.0
        return float(self.finish[int(j) * self.inst.n_ops + c - 1])

    def machine_free(self, m: int) -> float:
        route = self.routes[int(m)]
        return float(route[-1].finish) if route else 0.0

    def machine_busy(self, m: int) -> float:
        route = self.routes[int(m)]
        return float(sum(r.finish - r.start for r in route))

    def machine_util(self, m: int) -> float:
        """Fraction of the machine horizon spent processing, in [0, 1]."""
        free = self.machine_free(m)
        if free <= 1e-9:
            return 0.0
        return float(np.clip(self.machine_busy(m) / free, 0.0, 1.0))

    def agv_util(self, k: int) -> float:
        """Fraction of the AGV horizon spent travelling, in [0, 1]."""
        agv = self.agvs[int(k)]
        if agv.free_time <= 1e-9:
            return 0.0
        return float(np.clip(agv.travelled / agv.free_time, 0.0, 1.0))

    def world_time(self) -> float:
        """Current schedule horizon: maximum finish time so far."""
        if not bool(self.scheduled.any()):
            return 0.0
        return float(np.max(self.finish[self.scheduled]))

    def idle_energy(self, makespan: float | None = None) -> float:
        """Machine-idle term under ``self.idle_mode``.

        ``gap`` is the leading gap plus the internal machine gaps.
        ``trailing`` is ``sum_k (MKT - P_k)``; ``makespan`` defaults to
        the current schedule horizon.
        """
        gap = 0.0
        for route in self.routes:
            if not route:
                continue
            gap += route[0].start
            for k in range(len(route) - 1):
                gap += route[k + 1].start - route[k].finish
        if self.idle_mode == "gap":
            return float(gap)
        mkt = self.world_time() if makespan is None else float(makespan)
        proc = [sum(float(op.finish) - float(op.start) for op in route)
                for route in self.routes]
        return idle_cost(self.idle_mode, gap, proc, mkt)

    def job_remaining(self, j: int) -> tuple[float, float, float]:
        """Return (RTLP, remaining energy, remaining transport).

        RTLP is the sum of the minimum processing time of every remaining
        operation plus a mean-travel estimate for every remaining transition.
        The estimate is used by the rule layer and by the reward potential;
        it never enters the reported objective.
        """
        c = int(self.op_count[j])
        rem_t = 0.0
        rem_e = 0.0
        for o in range(c, self.inst.n_ops):
            task = int(j) * self.inst.n_ops + o
            avail = self.inst.machines(task)
            if avail.size == 0:
                continue
            rem_t += float(np.min(self.inst.t[task][avail]))
            rem_e += float(np.min(self.inst.t[task][avail]
                                  * self.inst.p[task][avail]))
        n_transport = self.inst.n_ops - max(c, 1)
        rem_tr = float(n_transport) * self._mean_tt
        return rem_t + rem_tr, rem_e, rem_tr

    def cost_estimate(self) -> float:
        """Potential C_t.  At the terminal state it equals the objective.

        Unscheduled work is estimated with optimistic (minimum) processing
        times and a mean transport time.  Idle energy uses the idle already
        realised; future idle is not extrapolated.
        """
        est_mk = self.world_time()
        rem_e = 0.0
        rem_tr = 0.0
        for j in self.pending_jobs():
            rtlp, e_j, tr_j = self.job_remaining(j)
            rem_e += e_j
            rem_tr += tr_j
            est_mk = max(est_mk, self.job_ready(j) + rtlp)
        est_proc = self.proc_energy + rem_e
        est_idle = self.idle_energy()
        est_trans = self.trans_total + rem_tr
        return float(W_MK * est_mk
                     + W_EC * (est_proc / self.inst.n_task + est_idle)
                     + W_TT * est_trans)

    # ------------------------------------------------------------------
    # Transport planning
    # ------------------------------------------------------------------
    def source_candidates(self, prev_m: int) -> list[int]:
        """AGVs of the workshop that owns ``prev_m``."""
        e = self.inst.edge_of(prev_m)
        return [a.idx for a in self.agvs if a.edge == e]

    def destination_candidates(self, m: int) -> list[int]:
        """AGVs of the workshop that owns the drop-off machine."""
        e = self.inst.edge_of(m)
        return [a.idx for a in self.agvs if a.edge == e]

    def station(self, edge: int) -> int:
        """Node id of the load/unload station of workshop ``edge``."""
        return station_node(edge, self.inst.n_machine)

    def node_dist(self, a: int, b: int) -> float:
        """Distance between two M2 nodes; M1 uses machine-to-machine travel."""
        if not self.inst.lu_station:
            return self.inst.travel(a, b)
        return m2_node_dist(a, b, self.inst.travel, self.inst.n_machine,
                            self.machine_edge, self.inst.lu_machine,
                            self.inst.lu_edge)

    def plan_transport(self, prev_m: int, m: int, ready: float,
                       src_idx: int | None = None,
                       dst_idx: int | None = None) -> dict:
        """Compute transport timing without committing it.

        With ``src_idx=None`` the source AGV is the earliest-free one, and
        with ``dst_idx=None`` the destination AGV is the earliest-free one at
        dispatch time.  Both defaults mirror ``agv_baselines.decode`` (M1) and
        ``agv_baselines.decode_m2`` (M2).

        ``prev_m < 0`` marks the first operation of a job.  Under M1 it needs
        no transport; under M2 the job starts at the station of the workshop
        of its first machine.
        """
        if not self.inst.lu_station:
            return self._plan_transport_m1(prev_m, m, ready, src_idx,
                                           dst_idx)
        return self._plan_transport_m2(prev_m, m, ready, src_idx, dst_idx)

    def _plan_transport_m1(self, prev_m: int, m: int, ready: float,
                           src_idx: int | None,
                           dst_idx: int | None) -> dict:
        """M1 transport plan: machine-to-machine, loaded time only."""
        pickup_edge = self.inst.edge_of(prev_m)
        dropoff_edge = self.inst.edge_of(m)
        src_c = self.source_candidates(prev_m)
        if not src_c:
            raise ValueError(f"workshop {pickup_edge} has no AGV")
        if src_idx is None:
            src_idx = min(src_c, key=lambda k: (self.agvs[k].free_time, k))
        elif int(src_idx) not in src_c:
            raise ValueError(f"AGV {src_idx} is not in workshop {pickup_edge}")
        src = self.agvs[int(src_idx)]
        total = self.inst.travel(prev_m, m)
        dead_src = self.inst.travel(src.location, prev_m)
        start_src = max(float(ready), float(src.free_time))
        base = {
            "prev_m": int(prev_m), "m": int(m), "ready_in": float(ready),
            "pickup_edge": pickup_edge, "dropoff_edge": dropoff_edge,
            "src_idx": int(src_idx), "dead_src": dead_src,
        }
        if pickup_edge == dropoff_edge:
            loaded = total
            finish = start_src + dead_src + loaded
            base.update({
                "kind": "same", "dst_idx": None,
                "src_free": finish, "src_loc": int(m),
                "src_travelled": dead_src + total,
                "dst_free": None, "dst_loc": None, "dst_travelled": 0.0,
                "loaded": loaded, "trans": loaded, "ready": finish,
                "boundary": None, "boundary_machine": None,
                "dead_dst": 0.0,
            })
            return base
        dst_c = self.destination_candidates(m)
        if not dst_c:
            raise ValueError(f"workshop {dropoff_edge} has no AGV")
        if dst_idx is None:
            dst_idx = min(dst_c, key=lambda k: (self.agvs[k].free_time, k))
        elif int(dst_idx) not in dst_c:
            raise ValueError(f"AGV {dst_idx} is not in workshop {dropoff_edge}")
        dst = self.agvs[int(dst_idx)]
        t_src, delay, t_dst = split_cross(total)
        src_arrival = start_src + dead_src + t_src
        boundary = src_arrival + delay
        boundary_machine = int(self.inst.edge[dropoff_edge][0])
        dead_dst = self.inst.travel(dst.location, boundary_machine)
        start_dst = max(boundary, float(dst.free_time))
        dst_free = start_dst + dead_dst + t_dst
        base.update({
            "kind": "cross", "dst_idx": int(dst_idx),
            "src_free": src_arrival, "src_loc": -1,
            "src_travelled": dead_src + t_src,
            "dst_free": dst_free, "dst_loc": int(m),
            "dst_travelled": dead_dst + t_dst,
            "loaded": t_src + t_dst, "trans": t_src + t_dst,
            "ready": dst_free,
            "boundary": boundary, "boundary_machine": boundary_machine,
            "dead_dst": dead_dst,
        })
        return base

    def _plan_transport_m2(self, prev_m: int, m: int, ready: float,
                           src_idx: int | None,
                           dst_idx: int | None) -> dict:
        """M2 transport plan: stations, entry legs, loaded and empty time."""
        first = int(prev_m) < 0
        dropoff_edge = self.inst.edge_of(m)
        if first:
            pickup = self.station(dropoff_edge)
            pickup_edge = dropoff_edge
        else:
            pickup = int(prev_m)
            pickup_edge = self.inst.edge_of(pickup)
        src_c = [a.idx for a in self.agvs if a.edge == pickup_edge]
        if not src_c:
            raise ValueError(f"workshop {pickup_edge} has no AGV")
        if src_idx is None:
            src_idx = min(src_c, key=lambda k: (self.agvs[k].free_time, k))
        elif int(src_idx) not in src_c:
            raise ValueError(f"AGV {src_idx} is not in workshop {pickup_edge}")
        src = self.agvs[int(src_idx)]
        dst = None
        if pickup_edge != dropoff_edge:
            dst_c = self.destination_candidates(m)
            if not dst_c:
                raise ValueError(f"workshop {dropoff_edge} has no AGV")
            if dst_idx is None:
                dst_idx = min(dst_c, key=lambda k: (self.agvs[k].free_time, k))
            elif int(dst_idx) not in dst_c:
                raise ValueError(
                    f"AGV {dst_idx} is not in workshop {dropoff_edge}")
            dst = self.agvs[int(dst_idx)]
        timing = transport_timing_m2(
            src, pickup, int(m), float(ready), self.inst.travel,
            self.inst.n_machine, self.machine_edge, self.inst.lu_machine,
            self.inst.lu_edge, dst=dst)
        if dst is None:
            ready_out = float(timing["src_free"])
            boundary = None
            boundary_node = None
        else:
            ready_out = float(timing["dst_free"])
            boundary = float(timing["src_free"] + timing["transfer"])
            boundary_node = self.station(dropoff_edge)
        return {
            "prev_m": int(prev_m), "m": int(m), "ready_in": float(ready),
            "pickup": int(pickup), "pickup_edge": pickup_edge,
            "dropoff_edge": dropoff_edge, "src_idx": int(src_idx),
            "dead_src": float(timing["dead_src"]),
            "kind": "same" if dst is None else "cross",
            "dst_idx": None if dst is None else int(dst.idx),
            "src_free": float(timing["src_free"]),
            "src_loc": int(timing["src_location"]),
            "src_travelled": float(timing["src_travelled"]),
            "dst_free": timing["dst_free"],
            "dst_loc": timing["dst_location"],
            "dst_travelled": float(timing["dst_travelled"]),
            "loaded": float(timing["loaded"]),
            "empty": float(timing["empty"]),
            "trans": float(timing["loaded"] + RHO_EMPTY * timing["empty"]),
            "ready": ready_out,
            "wait": float(timing["wait"]),
            "boundary": boundary,
            "boundary_machine": boundary_node,
            "dead_dst": float(timing["dead_dst"]),
            "transfer": float(timing["transfer"]),
        }

    def best_destination(self, plan: dict) -> dict:
        """Re-plan a cross-workshop transport with the best destination AGV.

        The destination AGV is chosen by SCT at the boundary arrival time:
        ``max(boundary, free_time) + deadhead to the boundary node``.  This
        is a deterministic added rule; the learned AGV agent only chooses the
        source AGV.  Under M2 the boundary node is the destination station.
        """
        if plan["kind"] != "cross":
            return plan
        boundary = float(plan["boundary"])
        bm = int(plan["boundary_machine"])
        dst_c = self.destination_candidates(plan["m"])
        if self.inst.lu_station:
            best = min(dst_c, key=lambda k: (
                max(boundary, self.agvs[k].free_time)
                + self.node_dist(self.agvs[k].location, bm), k))
        else:
            best = min(dst_c, key=lambda k: (
                max(boundary, self.agvs[k].free_time)
                + self.inst.travel(self.agvs[k].location, bm), k))
        return self.plan_transport(plan["prev_m"], plan["m"],
                                   plan["ready_in"],
                                   src_idx=plan["src_idx"], dst_idx=best)

    def commit_plan(self, plan: dict) -> None:
        """Apply one transport plan to the AGV states."""
        src = self.agvs[int(plan["src_idx"])]
        src.free_time = float(plan["src_free"])
        src.location = int(plan["src_loc"])
        src.travelled += float(plan["src_travelled"])
        if plan["kind"] == "cross":
            dst = self.agvs[int(plan["dst_idx"])]
            dst.free_time = float(plan["dst_free"])
            dst.location = int(plan["dst_loc"])
            dst.travelled += float(plan["dst_travelled"])
        self.trans_total += float(plan["trans"])
        self.loaded_total += float(plan.get("loaded", plan["trans"]))
        self.empty_total += float(plan.get("empty", 0.0))

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------
    def apply(self, j: int, m: int, plan: dict | None = None) -> dict:
        """Schedule the next operation of job ``j`` on machine ``m``.

        ``plan`` is the transport plan for this operation.  It is required
        when the previous operation is on another machine; it is ignored for
        the first operation and for consecutive operations on one machine
        under M1.  Under M2 the first operation also needs transport; if no
        plan is given, the earliest-free source AGV is planned here.
        """
        task = self.next_task(j)
        if m not in self.inst.machines(task):
            raise ValueError(f"task {task} cannot use machine {m}")
        c = int(self.op_count[j])
        prev_m = -1
        if c == 0:
            if self.inst.lu_station:
                if plan is None:
                    plan = self.plan_transport(-1, m, 0.0)
                self.commit_plan(plan)
                ready = float(plan["ready"])
            else:
                ready = 0.0
        else:
            prev_task = task - 1
            prev_m = int(self.machine_of[prev_task])
            if prev_m == int(m):
                if plan is not None:
                    raise ValueError("transport plan given for a same-machine "
                                     "operation")
                ready = float(self.finish[prev_task])
            else:
                if plan is None:
                    plan = self.plan_transport(
                        prev_m, m, float(self.finish[prev_task]))
                self.commit_plan(plan)
                ready = float(plan["ready"])

        dur = float(self.inst.t[task][m])
        route = self.routes[int(m)]
        pos = len(route)
        # ``decode_m2`` accepts an insertion without an epsilon; the M1
        # reference decoder uses 1e-9.
        tol = 0.0 if self.inst.lu_station else 1e-9
        for k in range(len(route) + 1):
            prev_ft = route[k - 1].finish if k > 0 else 0.0
            next_st = route[k].start if k < len(route) else math.inf
            st = max(ready, prev_ft)
            if st + dur <= next_st + tol:
                pos = k
                break
        st = max(ready, route[pos - 1].finish if pos > 0 else 0.0)
        route.insert(pos, Route(start=st, finish=st + dur, task=task))
        self.start[task] = st
        self.finish[task] = st + dur
        self.machine_of[task] = int(m)
        self.scheduled[task] = True
        self.op_count[j] += 1
        self.proc_energy += dur * float(self.inst.p[task][m])
        return {
            "task": task, "job": int(j), "machine": int(m),
            "start": st, "finish": st + dur, "ready": ready,
            "prev_machine": prev_m, "transport": plan,
        }

    def is_done(self) -> bool:
        return bool(np.all(self.op_count >= self.inst.n_ops))

    def n_scheduled(self) -> int:
        return int(np.sum(self.op_count))

    def finalize(self) -> dict:
        """Return the four v5 cost components and the weighted objective."""
        if not self.is_done():
            raise RuntimeError("cannot finalize an unfinished schedule")
        mk = float(np.max(self.finish))
        idle = self.idle_energy(mk)
        obj = (W_MK * mk
               + W_EC * (self.proc_energy / self.inst.n_task + idle)
               + W_TT * self.trans_total)
        out = {"mk": mk, "pt": float(self.proc_energy),
               "transT": float(self.trans_total), "idleT": float(idle),
               "obj": float(obj)}
        if self.inst.lu_station:
            out["loaded"] = float(self.loaded_total)
            out["empty"] = float(self.empty_total)
        return out
