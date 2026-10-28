# -*- coding: utf-8 -*-
"""Stepwise v5 simulator and the NHDRL dispatching-rule layer.

This module is part of an **adapted** baseline for:

    J. Shi, F. Qiao, J. Liu, Y. Ma, D. Wang, C. Ding,
    "Production-logistics collaborative scheduling in dynamic flexible
    job shops using nested-hierarchical deep reinforcement learning",
    Advanced Engineering Informatics 65 (2025) 103195.
    doi:10.1016/j.aei.2025.103195

It is not a reproduction of the paper.  The paper's shop, data, and
disturbance model differ from the v5 shop.  The deviations are listed in
``plcsp_v5/baselines/nhdrl/__init__.py``.  The two points that concern this
module are:

* v5 has two workshops and per-workshop AGV fleets.  Cross-workshop transport
  is a source AGV leg, a fixed delay, and a destination AGV leg.  The paper's
  AGVs can serve the whole shop.
* The default v5 model (M1) has no load/unload station and no return trip.
  The first operation of a job and a move to the same machine need no
  transport.  The paper gives every operation a logistics task.  The optional
  M2 model (``lu_station=True``) adds one load/unload station per workshop,
  an entry transport for every first operation, and loaded plus empty travel
  in the objective.  Its timing comes from ``agv.state.transport_timing_m2``.

The simulator mirrors ``plcsp_v5.baselines.agv_baselines.decode`` exactly for
the earliest-free AGV rule, and ``decode_m2`` under M2.  ``--check`` in the
CLI validates this.  The three NHDRL AGV rules are applied on top of the same
timing formulas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:  # scripts add ``plcsp_v5/`` to ``sys.path``
    from agv.state import (RHO_EMPTY, m2_node_dist, station_node,
                           transport_timing_m2)
except ImportError:  # repository root only
    from plcsp_v5.agv.state import (RHO_EMPTY,
                                   m2_node_dist, station_node,
                                    transport_timing_m2)

N_AGV_DEFAULT = 6
CROSS_DELAY = 5.0
LEG_RATIO = 0.5

W_MK = 0.4
W_EC = 0.4
W_TT = 0.2

# Unit logistics cost used only by the LA state features (the paper uses 0.5).
UC = 0.5

JOB_RULES = ("spt", "eaj", "srpt")
MACHINE_RULES = ("smpt", "minu", "eam")
AGV_RULES = ("cost", "util", "noload")
EARLIEST = "earliest"  # validation-only rule, not part of the action space

N_OA = 31
N_PA = 24
N_LA = 30
N_OA_ACTIONS = 2
N_PA_ACTIONS = 9
N_LA_ACTIONS = 3

FEATURE_NAMES = {
    "oa": [
        "n_machine", "m_util_mean", "m_util_min", "m_util_max",
        "m_ct_mean", "m_ct_min", "m_ct_max",
        "n_agv", "n_agv_used",
        "a_util_mean", "a_util_min", "a_util_max",
        "a_ct_mean", "a_ct_min", "a_ct_max",
        "n_task", "cum_cost", "n_remaining", "n_unfinished_jobs",
        "completion_rate", "job_rate_mean", "job_rate_min", "job_rate_max",
        "apt_mean", "apt_min", "apt_max",
        "rtj_mean", "rtj_min", "rtj_max",
        "pref_ms", "pref_tc",
    ],
    "pa": [
        "n_machine", "m_util_mean", "m_util_min", "m_util_max",
        "m_ct_mean", "m_ct_min", "m_ct_max",
        "n_task", "cum_cost", "n_remaining", "n_unfinished_jobs",
        "completion_rate", "job_rate_mean", "job_rate_min", "job_rate_max",
        "apt_mean", "apt_min", "apt_max",
        "rtj_mean", "rtj_min", "rtj_max",
        "pref_ms", "pref_tc", "oa_action",
    ],
    "la": [
        "n_agv", "n_agv_used",
        "a_util_mean", "a_util_min", "a_util_max",
        "a_ct_mean", "a_ct_min", "a_ct_max",
        "n_task", "cum_cost", "n_remaining",
        "lt_ready", "lt_dropoff", "lt_pickup", "lt_machine_ct",
        "lt_all_mean", "lt_all_min", "lt_all_max",
        "lt_all_cost_mean", "lt_all_cost_min", "lt_all_cost_max",
        "lt_noload_mean", "lt_noload_min", "lt_noload_max",
        "lt_noload_cost_mean", "lt_noload_cost_min", "lt_noload_cost_max",
        "pref_ms", "pref_tc", "oa_action",
    ],
}


@dataclass
class Agv:
    """One v5 AGV.  The home edge never changes."""

    idx: int
    edge: int
    location: int
    free_time: float = 0.0
    travelled: float = 0.0
    loaded: float = 0.0

    def utilization(self) -> float:
        if self.free_time > 1e-9:
            return float(self.travelled / self.free_time)
        return 0.0


@dataclass
class StepOutcome:
    """Cost movement produced by one operation assignment."""

    task: int
    machine: int
    job: int
    ms_before: float
    ms_after: float
    cost_before: float
    cost_after: float
    transport: bool

    @property
    def ms_gain(self) -> float:
        """Makespan gain: before minus after.  It is <= 0."""
        return float(self.ms_before - self.ms_after)

    @property
    def cost_delta(self) -> float:
        """Change of the non-makespan cost: after minus before."""
        return float(self.cost_after - self.cost_before)


def split_cross(total: float, delay: float = CROSS_DELAY,
                ratio: float = LEG_RATIO) -> tuple[float, float, float]:
    """Split one cross-workshop trip into (source leg, delay, target leg)."""
    total = max(0.0, float(total))
    d = min(max(0.0, float(delay)), total)
    rem = total - d
    src = rem * min(max(float(ratio), 0.0), 1.0)
    return src, d, rem - src


class NestedSim:
    """Replay one v5 instance under sequential operation assignment.

    Parameters
    ----------
    t, p : (n_task, n_machine) arrays
        Processing time and power.  Negative entries mark incompatible pairs.
    tt : (n_machine, n_machine) array
        Machine-to-machine travel time.
    edge_info : sequence of sequences
        Workshop membership.  ``edge_info[e]`` lists the machines of edge e.
    n_agv : int
        Number of AGVs.  AGV k starts in edge ``k % n_edge``.
    lu_station : bool
        Select the M2 transport model.  ``lu_machine`` and ``lu_edge`` are
        then required: ``lu_machine[m]`` is the distance from machine ``m`` to
        the station of its workshop, and ``lu_edge[a][b]`` the distance
        between the stations of workshops ``a`` and ``b``.
    """

    def __init__(self, t: np.ndarray, p: np.ndarray, tt: np.ndarray,
                 edge_info: Sequence[Sequence[int]],
                 n_agv: int = N_AGV_DEFAULT,
                 lu_station: bool = False,
                 lu_machine: Optional[np.ndarray] = None,
                 lu_edge: Optional[np.ndarray] = None):
        self.t = np.asarray(t, dtype=np.float64)
        self.p = np.asarray(p, dtype=np.float64)
        self.tt = np.asarray(tt, dtype=np.float64)
        self.edge_info = [[int(x) for x in members]
                          for members in edge_info]
        self.n_task, self.n_machine = self.t.shape
        self.n_job = self.n_task // self.n_machine
        self.n_ops = self.n_machine
        self.n_edge = len(self.edge_info)
        self.n_agv = int(n_agv)
        if self.n_agv < self.n_edge:
            raise ValueError("n_agv must be at least n_edge")

        self.lu_station = bool(lu_station)
        self.lu_machine: Optional[np.ndarray] = None
        self.lu_edge: Optional[np.ndarray] = None
        if self.lu_station:
            if lu_machine is None or lu_edge is None:
                raise ValueError("lu_station=1 needs lu_machine and lu_edge")
            self.lu_machine = np.asarray(lu_machine, dtype=np.float64)
            self.lu_edge = np.asarray(lu_edge, dtype=np.float64)
            if self.lu_machine.shape != (self.n_machine,):
                raise ValueError(f"bad lu_machine shape: "
                                 f"{self.lu_machine.shape}")
            if self.lu_edge.shape != (self.n_edge, self.n_edge):
                raise ValueError(f"bad lu_edge shape: {self.lu_edge.shape}")
        self.machine_edge = np.zeros(self.n_machine, dtype=int)
        for e, members in enumerate(self.edge_info):
            for m in members:
                self.machine_edge[int(m)] = e
        # ``decode_m2`` accepts an insertion without an epsilon; the M1
        # reference decoder uses 1e-9.
        self.insert_tol = 0.0 if self.lu_station else 1e-9

        self.agvs = [
            Agv(idx=k, edge=k % self.n_edge,
                location=self.station_node(k % self.n_edge)
                if self.lu_station else int(self.edge_info[k % self.n_edge][0]))
            for k in range(self.n_agv)
        ]

        self.op_count = [0] * self.n_job
        self.start = np.zeros(self.n_task, dtype=np.float64)
        self.finish = np.zeros(self.n_task, dtype=np.float64)
        self.ms = np.full(self.n_task, -1, dtype=int)
        self.routes: list[list[tuple[float, float, int]]] = [
            [] for _ in range(self.n_machine)
        ]
        self.machine_proc = [0.0] * self.n_machine
        self.proc_energy = 0.0
        self.loaded_total = 0.0
        self.empty_total = 0.0
        self.travel_total = 0.0
        self.scheduled = 0
        self.os_seq: list[int] = []

        positive = self.t[self.t >= 0]
        self.t_scale = float(np.mean(positive)) if positive.size else 1.0
        self.t_scale = max(self.t_scale, 1.0)

        # APT_ij: mean compatible processing time.  Used by SPT, SRPT, and
        # the paper's f24-f29.
        self.apt = np.zeros(self.n_task, dtype=np.float64)
        for task in range(self.n_task):
            avail = self.t[task] >= 0
            self.apt[task] = (float(np.mean(self.t[task][avail]))
                              if avail.any() else 0.0)
        # Remaining processing time of job j from operation o onward.
        self.rtj = np.zeros((self.n_job, self.n_ops + 1), dtype=np.float64)
        for j in range(self.n_job):
            for o in range(self.n_ops - 1, -1, -1):
                task = j * self.n_ops + o
                self.rtj[j, o] = self.rtj[j, o + 1] + self.apt[task]

    # ------------------------------------------------------------------
    # Basic geometry
    # ------------------------------------------------------------------
    def travel(self, a: int, b: int) -> float:
        if a < 0 or b < 0:
            return 0.0
        return float(self.tt[int(a)][int(b)])

    def station_node(self, edge: int) -> int:
        """Node id of the load/unload station of workshop ``edge``."""
        return station_node(edge, self.n_machine)

    def m2_dist(self, a: int, b: int) -> float:
        """Distance between two M2 nodes; M1 uses machine-to-machine travel."""
        if not self.lu_station:
            return self.travel(a, b)
        return m2_node_dist(a, b, self.travel, self.n_machine,
                            self.machine_edge, self.lu_machine, self.lu_edge)

    def transport_estimate(self, prev_m: int, machine: int, oi: int) -> float:
        """Direct travel estimate used by the PA rules.

        M1 ignores the entry transport and same-machine moves.  M2 charges
        ``station -> machine`` for a first operation.
        """
        if not self.lu_station:
            return (self.travel(prev_m, machine)
                    if (oi > 0 and prev_m != machine) else 0.0)
        if oi == 0:
            return self.m2_dist(
                self.station_node(self.edge_of(machine)), machine)
        if prev_m != machine:
            return self.m2_dist(prev_m, machine)
        return 0.0

    def edge_of(self, machine: int) -> int:
        for e, members in enumerate(self.edge_info):
            if int(machine) in members:
                return e
        return 0

    def compatible(self, task: int) -> np.ndarray:
        return np.where(self.t[task] >= 0)[0]

    def next_task(self, job: int) -> int:
        return job * self.n_ops + self.op_count[job]

    @property
    def done(self) -> bool:
        return self.scheduled >= self.n_task

    # ------------------------------------------------------------------
    # Cost accounting
    # ------------------------------------------------------------------
    def partial_ms(self) -> float:
        if self.scheduled == 0:
            return 0.0
        return float(np.max(self.finish[self.finish > 0])) if np.any(
            self.finish > 0) else 0.0

    def partial_idle(self) -> float:
        idle = 0.0
        for route in self.routes:
            if not route:
                continue
            idle += route[0][0]
            for k in range(len(route) - 1):
                idle += route[k + 1][0] - route[k][1]
        return float(idle)

    def cost_non_ms(self) -> float:
        """Non-makespan part of the v5 objective."""
        return (W_EC * (self.proc_energy / self.n_task
                        + self.partial_idle())
                + W_TT * self.loaded_total)

    def objective(self) -> float:
        return W_MK * self.partial_ms() + self.cost_non_ms()

    def machine_util(self, machine: int) -> float:
        route = self.routes[machine]
        if not route:
            return 0.0
        mct = route[-1][1]
        if mct <= 1e-9:
            return 0.0
        return float(self.machine_proc[machine] / mct)

    def machine_ct(self, machine: int) -> float:
        route = self.routes[machine]
        return float(route[-1][1]) if route else 0.0

    def earliest_start(self, machine: int, ready: float,
                       duration: float) -> float:
        """Earliest insertion start of a task on a machine."""
        route = self.routes[machine]
        for k in range(len(route) + 1):
            prev_end = route[k - 1][1] if k > 0 else 0.0
            next_start = route[k][0] if k < len(route) else math.inf
            start = max(float(ready), float(prev_end))
            if start + float(duration) <= next_start + self.insert_tol:
                return float(start)
        return float(ready)

    # ------------------------------------------------------------------
    # Transport need and AGV rules
    # ------------------------------------------------------------------
    def transport_info(self, task: int, machine: int
                       ) -> Optional[tuple[int, int, float]]:
        """Return (prev_task, prev_machine, prev_finish) or None.

        Under M2 a first operation returns ``prev_machine == -1``: the job
        enters from the station of its first machine's workshop.
        """
        oi = task % self.n_ops
        if oi == 0:
            if self.lu_station:
                return task, -1, 0.0
            return None
        prev = task - 1
        prev_m = int(self.ms[prev])
        if prev_m < 0 or prev_m == int(machine):
            return None
        return prev, prev_m, float(self.finish[prev])

    def _candidates(self, edge: int) -> list[Agv]:
        return [a for a in self.agvs if a.edge == edge]

    def pick_agv(self, rule: str, edge: int, pickup: int, dropoff: int,
                 ready: float, loaded: float) -> Agv:
        """Pick an AGV of one workshop with an NHDRL rule.

        ``rule`` is one of ``cost``, ``util``, ``noload``.  ``earliest`` is
        accepted for the decode cross-check only.  Deadhead uses M2 node
        distances when ``lu_station`` is on.
        """
        cand = self._candidates(edge)
        if not cand:
            raise ValueError(f"workshop {edge} has no AGV")
        dist = self.m2_dist if self.lu_station else self.travel
        if rule == EARLIEST:
            return min(cand, key=lambda a: (a.free_time, a.idx))
        if rule == "cost":
            def cost(a: Agv) -> tuple[float, int]:
                dead = dist(a.location, pickup)
                return dead + float(loaded), a.idx
            return min(cand, key=cost)
        if rule == "util":
            return min(cand, key=lambda a: (a.utilization(), a.idx))
        if rule == "noload":
            def noload(a: Agv) -> tuple[float, int]:
                return a.free_time + dist(a.location, pickup), a.idx

            eligible = [a for a in cand
                        if noload(a)[0] <= float(ready) + 1e-9]
            if eligible:
                latest = max(noload(a)[0] for a in eligible)
                near = [a for a in eligible
                        if abs(noload(a)[0] - latest) <= 1e-9]
                return min(near, key=lambda a: a.idx)
            return min(cand, key=noload)
        raise ValueError(f"unknown AGV rule: {rule}")

    def _apply_transport(self, task: int, machine: int,
                         rule: str) -> float:
        """Run one transport and return the drop-off completion time."""
        _prev, prev_m, ready = self.transport_info(task, machine)  # type: ignore
        if self.lu_station:
            return self._apply_transport_m2(task, int(machine), rule,
                                            int(prev_m), float(ready))
        total = self.travel(prev_m, machine)
        pe = self.edge_of(prev_m)
        de = self.edge_of(machine)
        src = self.pick_agv(rule, pe, prev_m, machine, ready, total)
        dead_src = self.travel(src.location, prev_m)
        start_src = max(float(ready), float(src.free_time))

        if pe == de:
            loaded = total
            src.free_time = start_src + dead_src + loaded
            src.location = int(machine)
            src.travelled += dead_src + loaded
            src.loaded += loaded
            self.loaded_total += loaded
            self.travel_total += dead_src + loaded
            return float(src.free_time)

        t_src, delay, t_dst = split_cross(total)
        src_arrival = start_src + dead_src + t_src
        boundary = src_arrival + delay
        boundary_machine = int(self.edge_info[de][0])
        dst = self.pick_agv(rule, de, boundary_machine, machine,
                            boundary, t_dst)
        dead_dst = self.travel(dst.location, boundary_machine)
        start_dst = max(boundary, float(dst.free_time))
        dst.free_time = start_dst + dead_dst + t_dst
        dst.location = int(machine)
        src.free_time = src_arrival
        src.location = -1
        src.travelled += dead_src + t_src
        src.loaded += t_src
        dst.travelled += dead_dst + t_dst
        dst.loaded += t_dst
        self.loaded_total += t_src + t_dst
        self.travel_total += dead_src + t_src + dead_dst + t_dst
        return float(dst.free_time)

    def _apply_transport_m2(self, task: int, machine: int, rule: str,
                            prev_m: int, ready: float) -> float:
        """M2 transport via ``transport_timing_m2``.

        The source AGV picks the job up at the previous machine, or at the
        station of the target workshop for a first operation.  Cross-workshop
        trips add a station-to-station transfer with no AGV and a destination
        AGV.  ``loaded`` and ``empty`` are both counted by the objective.
        """
        de = self.edge_of(machine)
        if prev_m < 0:
            pickup = self.station_node(de)
            pe = de
        else:
            pickup = prev_m
            pe = self.edge_of(prev_m)
        if pe == de:
            src_loaded = self.m2_dist(pickup, machine)
        else:
            src_loaded = self.m2_dist(pickup, self.station_node(pe))
        src = self.pick_agv(rule, pe, pickup, machine, ready, src_loaded)

        dst: Optional[Agv] = None
        if pe != de:
            station_dst = self.station_node(de)
            start_src = max(float(ready), float(src.free_time))
            dead_src = self.m2_dist(src.location, pickup)
            t_src = self.m2_dist(pickup, self.station_node(pe))
            boundary = (start_src + dead_src + t_src
                        + float(self.lu_edge[pe][de]))
            dst = self.pick_agv(rule, de, station_dst, machine, boundary,
                                self.m2_dist(station_dst, machine))

        timing = transport_timing_m2(
            src, pickup, machine, ready, self.travel, self.n_machine,
            self.machine_edge, self.lu_machine, self.lu_edge, dst=dst)
        src.free_time = float(timing["src_free"])
        src.location = int(timing["src_location"])
        src.travelled += float(timing["src_travelled"])
        src.loaded += float(timing["t_src"])
        self.loaded_total += float(timing["loaded"])
        self.empty_total += float(timing["empty"])
        self.travel_total += float(timing["src_travelled"])
        if dst is not None:
            dst.free_time = float(timing["dst_free"])
            dst.location = int(timing["dst_location"])
            dst.travelled += float(timing["dst_travelled"])
            dst.loaded += float(timing["t_dst"])
            self.travel_total += float(timing["dst_travelled"])
            return float(timing["dst_free"])
        return float(timing["src_free"])

    # ------------------------------------------------------------------
    # PA rules
    # ------------------------------------------------------------------
    def job_earliest_start(self, job: int) -> float:
        """Estimate the earliest start of a job's next operation.

        The estimate uses a direct travel time: machine-to-machine under M1,
        M2 node distance (including the entry leg) under M2.  It ignores AGV
        deadhead and waiting because PA acts before LA.
        """
        oi = self.op_count[job]
        task = job * self.n_ops + oi
        prev_ready = 0.0
        prev_m = -1
        if oi > 0:
            prev = task - 1
            prev_ready = float(self.finish[prev])
            prev_m = int(self.ms[prev])
        best = math.inf
        for m in self.compatible(task):
            m = int(m)
            trans = self.transport_estimate(prev_m, m, oi)
            start = self.earliest_start(m, prev_ready + trans,
                                        float(self.t[task][m]))
            if start < best:
                best = start
        return float(best)

    def _job_score(self, job: int, rule: str) -> float:
        task = self.next_task(job)
        if rule == "spt":
            return float(self.apt[task])
        if rule == "eaj":
            return self.job_earliest_start(job)
        if rule == "srpt":
            return float(self.rtj[job, self.op_count[job]])
        raise ValueError(f"unknown job rule: {rule}")

    def _machine_score(self, task: int, machine: int, rule: str,
                       ready: float, duration: float) -> float:
        if rule == "smpt":
            return float(self.t[task][machine])
        if rule == "minu":
            return self.machine_util(machine)
        if rule == "eam":
            return self.earliest_start(machine, ready, duration)
        raise ValueError(f"unknown machine rule: {rule}")

    def choose_pa(self, action: int) -> tuple[int, int, int]:
        """Apply a PA composite rule.  Return (job, task, machine)."""
        if not 0 <= int(action) < N_PA_ACTIONS:
            raise ValueError(f"PA action out of range: {action}")
        job_rule = JOB_RULES[int(action) // 3]
        machine_rule = MACHINE_RULES[int(action) % 3]

        jobs = [j for j in range(self.n_job)
                if self.op_count[j] < self.n_ops]
        if not jobs:
            raise RuntimeError("no unfinished job")
        best_job = min(jobs, key=lambda j: (self._job_score(j, job_rule), j))
        task = self.next_task(best_job)
        oi = self.op_count[best_job]

        prev_ready = 0.0
        prev_m = -1
        if oi > 0:
            prev = task - 1
            prev_ready = float(self.finish[prev])
            prev_m = int(self.ms[prev])

        def m_score(m: int) -> float:
            duration = float(self.t[task][m])
            trans = self.transport_estimate(prev_m, m, oi)
            return self._machine_score(task, m, machine_rule,
                                       prev_ready + trans, duration)

        machines = [int(m) for m in self.compatible(task)]
        best_m = min(machines, key=lambda m: (m_score(m), m))
        return best_job, task, best_m

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------
    def schedule(self, task: int, machine: int,
                 agv_rule: Optional[str]) -> StepOutcome:
        """Schedule one operation and return the cost movement."""
        task = int(task)
        machine = int(machine)
        job = task // self.n_ops
        oi = task % self.n_ops
        if oi != self.op_count[job]:
            raise ValueError("operations of a job must be scheduled in order")
        if self.t[task][machine] < 0:
            raise ValueError(f"task {task} cannot use machine {machine}")

        ms_before = self.partial_ms()
        cost_before = self.cost_non_ms()

        need = self.transport_info(task, machine) is not None
        if not need:
            ready = 0.0 if oi == 0 else float(self.finish[task - 1])
        else:
            if agv_rule is None:
                raise ValueError("transport is needed but no AGV rule given")
            ready = self._apply_transport(task, machine, agv_rule)

        duration = float(self.t[task][machine])
        start = self.earliest_start(machine, ready, duration)
        end = start + duration
        route = self.routes[machine]
        pos = len(route)
        for k in range(len(route) + 1):
            prev_end = route[k - 1][1] if k > 0 else 0.0
            next_start = route[k][0] if k < len(route) else math.inf
            if (max(start, prev_end) + duration
                    <= next_start + self.insert_tol):
                pos = k
                break
        route.insert(pos, (start, end, task))

        self.start[task] = start
        self.finish[task] = end
        self.ms[task] = machine
        self.machine_proc[machine] += duration
        self.proc_energy += duration * float(self.p[task][machine])
        self.op_count[job] += 1
        self.scheduled += 1
        self.os_seq.append(job)

        return StepOutcome(
            task=task, machine=machine, job=job,
            ms_before=ms_before, ms_after=self.partial_ms(),
            cost_before=cost_before, cost_after=self.cost_non_ms(),
            transport=need,
        )

    # ------------------------------------------------------------------
    # State features (31 / 24 / 30, adapted from the paper)
    # ------------------------------------------------------------------
    def _oa_vector(self, pref_ms: float, pref_tc: float,
                   maxp: float) -> np.ndarray:
        m_util = [self.machine_util(m) for m in range(self.n_machine)]
        m_ct = [self.machine_ct(m) for m in range(self.n_machine)]
        a_util = [a.utilization() for a in self.agvs]
        a_ct = [a.free_time for a in self.agvs]
        used = sum(1 for a in self.agvs if a.travelled > 1e-9)

        unfinished = [j for j in range(self.n_job)
                      if self.op_count[j] < self.n_ops]
        rates = [self.op_count[j] / self.n_ops for j in range(self.n_job)]
        apt_next = [self.apt[self.next_task(j)] for j in unfinished]
        rtj = [self.rtj[j, self.op_count[j]] for j in unfinished]

        def stat(values: list[float], fn) -> float:
            return float(fn(values)) if values else 0.0

        f = np.zeros(N_OA, dtype=np.float32)
        f[0] = self.n_machine / 10.0
        f[1] = stat(m_util, np.mean)
        f[2] = stat(m_util, np.min)
        f[3] = stat(m_util, np.max)
        f[4] = stat(m_ct, np.mean) / self.t_scale
        f[5] = stat(m_ct, np.min) / self.t_scale
        f[6] = stat(m_ct, np.max) / self.t_scale
        f[7] = self.n_agv / 10.0
        f[8] = used / 10.0
        f[9] = stat(a_util, np.mean)
        f[10] = stat(a_util, np.min)
        f[11] = stat(a_util, np.max)
        f[12] = stat(a_ct, np.mean) / self.t_scale
        f[13] = stat(a_ct, np.min) / self.t_scale
        f[14] = stat(a_ct, np.max) / self.t_scale
        f[15] = self.n_task / 100.0
        f[16] = self.cost_non_ms() / self.t_scale
        f[17] = (self.n_task - self.scheduled) / 100.0
        f[18] = len(unfinished) / self.n_job
        f[19] = self.scheduled / self.n_task
        f[20] = stat(rates, np.mean)
        f[21] = stat(rates, np.min)
        f[22] = stat(rates, np.max)
        f[23] = stat(apt_next, np.mean) / self.t_scale
        f[24] = stat(apt_next, np.min) / self.t_scale
        f[25] = stat(apt_next, np.max) / self.t_scale
        f[26] = stat(rtj, np.mean) / self.t_scale
        f[27] = stat(rtj, np.min) / self.t_scale
        f[28] = stat(rtj, np.max) / self.t_scale
        f[29] = pref_ms / maxp
        f[30] = pref_tc / maxp
        return f

    def oa_features(self, pref_ms: float = 5.0, pref_tc: float = 5.0,
                    maxp: float = 10.0) -> np.ndarray:
        return self._oa_vector(pref_ms, pref_tc, maxp)

    def pa_features(self, oa_action: int,
                    oa_vec: Optional[np.ndarray] = None) -> np.ndarray:
        if oa_vec is None:
            oa_vec = self.oa_features()
        f = np.zeros(N_PA, dtype=np.float32)
        f[0:7] = oa_vec[0:7]
        f[7:21] = oa_vec[15:29]
        f[21] = oa_vec[29]
        f[22] = oa_vec[30]
        f[23] = float(oa_action)
        return f

    def la_features(self, task: int, machine: int, oa_action: int,
                    oa_vec: Optional[np.ndarray] = None) -> np.ndarray:
        if oa_vec is None:
            oa_vec = self.oa_features()
        f = np.zeros(N_LA, dtype=np.float32)
        f[0:8] = oa_vec[7:15]
        f[8:11] = oa_vec[15:18]

        info = self.transport_info(task, machine)
        if info is not None:
            _prev, prev_m, prev_ready = info
            if self.lu_station:
                de = self.edge_of(machine)
                src_edge = de if prev_m < 0 else self.edge_of(prev_m)
                pickup = (self.station_node(de) if prev_m < 0
                          else int(prev_m))
                cand = self._candidates(src_edge)
                total = (self.m2_dist(pickup, machine)
                         if src_edge == de
                         else self.m2_dist(pickup,
                                           self.station_node(src_edge)))
                all_stage = [self.m2_dist(a.location, pickup) + total
                             for a in cand]
                noload = [self.m2_dist(a.location, pickup) for a in cand]
                f[13] = pickup / max(1, self.n_machine)
            else:
                src_edge = self.edge_of(prev_m)
                cand = self._candidates(src_edge)
                total = self.travel(prev_m, machine)
                all_stage = [self.travel(a.location, prev_m) + total
                             for a in cand]
                noload = [self.travel(a.location, prev_m) for a in cand]
                f[13] = prev_m / max(1, self.n_machine)
            f[11] = prev_ready / self.t_scale
            f[12] = machine / max(1, self.n_machine)
            f[14] = self.machine_ct(machine) / self.t_scale
            f[15] = float(np.mean(all_stage)) / self.t_scale
            f[16] = float(np.min(all_stage)) / self.t_scale
            f[17] = float(np.max(all_stage)) / self.t_scale
            f[18] = UC * f[15]
            f[19] = UC * f[16]
            f[20] = UC * f[17]
            f[21] = float(np.mean(noload)) / self.t_scale
            f[22] = float(np.min(noload)) / self.t_scale
            f[23] = float(np.max(noload)) / self.t_scale
            f[24] = UC * f[21]
            f[25] = UC * f[22]
            f[26] = UC * f[23]

        f[27] = oa_vec[29]
        f[28] = oa_vec[30]
        f[29] = float(oa_action)
        return f

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------
    def result(self) -> dict[str, float]:
        if not self.done:
            raise RuntimeError("the schedule is not complete")
        mk = float(np.max(self.finish))
        idle = self.partial_idle()
        pt = float(self.proc_energy)
        # M1 counts loaded travel only.  M2 counts loaded plus empty travel,
        # at unit power (``empty_total`` stays zero under M1).
        trans = float(self.loaded_total + RHO_EMPTY * self.empty_total)
        obj = W_MK * mk + W_EC * (pt / self.n_task + idle) + W_TT * trans
        out = {
            "mk": mk,
            "pt": pt,
            "transT": trans,
            "idleT": idle,
            "obj": float(obj),
            "cost_non_ms": float(W_EC * (pt / self.n_task + idle)
                                 + W_TT * trans),
        }
        if self.lu_station:
            out["loaded"] = float(self.loaded_total)
            out["empty"] = float(self.empty_total)
        return out
