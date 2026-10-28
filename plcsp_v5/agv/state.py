"""Stage-0.1 AGV resource model.

Assumptions:
  * per-workshop AGV fleets; an AGV never changes its home edge
  * single-load AGVs
  * no charging
  * no L/U capacity limit (load / unload time = 0)
  * cross-workshop transport = source-edge AGV leg
    + fixed cross-workshop delay (no AGV)
    + target-edge AGV leg
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable

import numpy as np

# Empty-travel power as a fraction of loaded-travel power.
RHO_EMPTY = 0.5

# Machine-idle accounting.  ``trailing`` is the published definition
# ``sum_k (MKT - P_k)``: every machine is charged until the makespan, so
# the trailing interval after its last operation counts.  ``gap`` is the
# old reading: leading idle plus the gaps between operations only.
IDLE_MODES = ("trailing", "gap")
DEFAULT_IDLE_MODE = "trailing"


def resolve_idle_mode(mode: str | None = None) -> str:
    """Validate ``idle_mode`` and fill the default for ``None``/empty."""
    if mode is None or str(mode).strip() == "":
        return DEFAULT_IDLE_MODE
    key = str(mode).strip().lower()
    if key not in IDLE_MODES:
        raise ValueError(
            f"idle_mode must be one of {IDLE_MODES}, got {mode!r}")
    return key


def trailing_idle(processing_by_machine, makespan: float,
                  idle_power=None) -> float:
    """``sum_k w_k * (MKT - P_k)`` over every machine.

    ``processing_by_machine`` holds ``P_k``; an unused machine has
    ``P_k = 0`` and is charged the whole makespan.  ``idle_power`` is the
    optional per-machine idle-power weight (v5 uses 1.0).
    """
    mkt = float(makespan)
    total = 0.0
    for k, proc in enumerate(processing_by_machine):
        weight = 1.0 if idle_power is None else float(idle_power[k])
        total += weight * (mkt - float(proc))
    return float(total)


def idle_cost(mode: str | None, gap_idle: float, processing_by_machine,
              makespan: float, idle_power=None) -> float:
    """Return the machine-idle term under ``mode``.

    ``gap_idle`` is the already-computed old reading, so callers that
    collect the machine routes once can serve both modes.
    """
    if resolve_idle_mode(mode) == "gap":
        return float(gap_idle)
    return trailing_idle(processing_by_machine, makespan, idle_power)


class AgvStatus(IntEnum):
    IDLE = 0
    TO_PICKUP = 1
    LOADED = 2
    TO_DROPOFF = 3


@dataclass
class AgvState:
    idx: int
    edge: int              # home workshop; never changes
    location: int          # current machine, or -1 at a workshop boundary
    speed: float = 1.0
    status: int = int(AgvStatus.IDLE)
    free_time: float = 0.0
    load: int = 0
    travelled: float = 0.0

    def assign(self, now: float, pickup: int, dropoff: int,
               travel: Callable[[int, int], float],
               load_time: float = 0.0, unload_time: float = 0.0) -> dict:
        """Assign one transport task; the AGV is free after unloading."""
        depart_free = max(float(now), self.free_time)
        arrive_pickup = depart_free + float(travel(self.location, pickup))
        depart_pickup = arrive_pickup + float(load_time)
        arrive_dropoff = depart_pickup + float(travel(pickup, dropoff))

        self.travelled += float(travel(self.location, pickup)) + float(travel(pickup, dropoff))
        self.location = dropoff
        self.status = int(AgvStatus.IDLE)
        self.load = 0
        self.free_time = arrive_dropoff + float(unload_time)
        return {
            "arrive_pickup": arrive_pickup,
            "depart_pickup": depart_pickup,
            "arrive_dropoff": arrive_dropoff,
            "free_time": self.free_time,
        }


def agv_features(agvs: list[AgvState], pickup: int, dropoff: int,
                 travel: Callable[[int, int], float], now: float) -> np.ndarray:
    """Per-AGV feature matrix of shape (n_agv, 11)."""
    feats = []
    for a in agvs:
        # Stage 0 tracks only free_time. A busy AGV is reported as LOADED;
        # the pickup / drop-off phases are not modeled.
        busy = 1.0 if a.free_time > now + 1e-9 else 0.0
        feats.append([
            1.0 - busy,
            0.0,
            busy,
            float(a.travelled),  # cumulative travel = busy time
            float(a.load),
            float(a.edge),
            float(a.location),
            float(travel(a.location, pickup)),
            float(travel(pickup, dropoff)),
            float(max(0.0, a.free_time - now)),
            float(a.free_time),
        ])
    return np.asarray(feats, dtype=np.float32)


def context_features(pickup: int, dropoff: int, pickup_edge: int,
                     dropoff_edge: int, travel: Callable[[int, int], float],
                     now: float, n_available: int, n_agv: int) -> np.ndarray:
    """Decision context of shape (6,)."""
    return np.asarray([
        float(pickup),
        float(dropoff),
        1.0 if pickup_edge != dropoff_edge else 0.0,
        float(travel(pickup, dropoff)),
        float(now),
        float(n_available) / max(1, n_agv),
    ], dtype=np.float32)


def split_cross_edge(total: float, cross_delay: float,
                     leg_ratio: float = 0.5) -> tuple[float, float, float]:
    """Split one cross-workshop trip into (source leg, fixed delay, target leg).

    ``total`` is the machine-to-machine time. The fixed delay consumes no AGV.
    """
    total = max(0.0, float(total))
    delay = min(max(0.0, float(cross_delay)), total)
    remaining = total - delay
    src = remaining * min(max(float(leg_ratio), 0.0), 1.0)
    dst = remaining - src
    return src, delay, dst


def transport_timing(
        src: AgvState, pickup: int, dropoff: int, ready: float,
        total: float, travel: Callable[[int, int], float],
        dst: AgvState | None = None,
        cross_delay: float = 0.0, leg_ratio: float = 0.5,
        boundary_machine: int | None = None,
        src_repair: float = 0.0) -> dict:
    """Stage-0 AGV timing, including empty deadhead travel.

    Same workshop (``dst is None``):
        deadhead(src.location -> pickup) + loaded(pickup -> dropoff).
    Cross workshop (``dst`` given):
        source deadhead + source leg; fixed delay; target deadhead to
        ``boundary_machine`` + target leg.

    ``src_repair`` is an extra delay on the source loaded leg (AGV failure).
    ``wait`` is the non-loaded delay used by the environment override;
    ``loaded`` is the loaded travel time used by TEC / edge weights.
    """
    total = max(0.0, float(total))
    dead_src = float(travel(src.location, pickup))
    start_src = max(float(ready), float(src.free_time))
    wait_src = start_src - float(ready)

    if dst is None:
        loaded = total + float(src_repair)
        return {
            "wait": wait_src + dead_src,
            "loaded": loaded,
            "src_free": start_src + dead_src + loaded,
            "src_location": int(dropoff),
            "src_travelled": dead_src + total,
            "dst_free": None,
            "dst_location": None,
            "dst_travelled": 0.0,
            "t_src": loaded,
            "t_dst": 0.0,
            "dead_src": dead_src,
            "dead_dst": 0.0,
        }

    if boundary_machine is None:
        raise ValueError("cross-workshop transport needs boundary_machine")
    t_src, d_eff, t_dst = split_cross_edge(total, cross_delay, leg_ratio)
    t_src_eff = t_src + float(src_repair)
    src_arrival = start_src + dead_src + t_src_eff
    boundary = src_arrival + d_eff
    dead_dst = float(travel(dst.location, boundary_machine))
    start_dst = max(boundary, float(dst.free_time))
    wait_dst = start_dst - boundary
    return {
        "wait": wait_src + dead_src + d_eff + wait_dst + dead_dst,
        "loaded": t_src_eff + t_dst,
        "src_free": src_arrival,
        "src_location": -1,
        "src_travelled": dead_src + t_src,
        "dst_free": start_dst + dead_dst + t_dst,
        "dst_location": int(dropoff),
        "dst_travelled": dead_dst + t_dst,
        "t_src": t_src_eff,
        "t_dst": t_dst,
        "dead_src": dead_src,
        "dead_dst": dead_dst,
    }


def station_node(edge: int, n_machine: int) -> int:
    """Node id of the load/unload station of workshop ``edge``."""
    return int(n_machine) + int(edge)


def m2_node_dist(a: int, b: int, travel: Callable[[int, int], float],
                 n_machine: int, machine_edge, lu_machine,
                 lu_edge) -> float:
    """Distance between two nodes in the M2 model.

    Nodes 0..n_machine-1 are machines; nodes n_machine+w are stations.
    """
    if a < 0 or b < 0:
        return 0.0
    a_st = a >= n_machine
    b_st = b >= n_machine
    if not a_st and not b_st:
        return float(travel(int(a), int(b)))
    if a_st and b_st:
        return float(lu_edge[int(a) - n_machine][int(b) - n_machine])
    if not a_st and b_st:
        w = int(b) - n_machine
        w_a = int(machine_edge[int(a)])
        if w_a == w:
            return float(lu_machine[int(a)])
        # Machine -> station of another workshop: the machine's own station
        # plus the station transfer.  Used by the AGV feature encoder, which
        # queries every AGV, not only the ones that could serve the job.
        return float(lu_machine[int(a)]) + float(lu_edge[w_a][w])
    w = int(a) - n_machine
    w_b = int(machine_edge[int(b)])
    if w_b == w:
        return float(lu_machine[int(b)])
    return float(lu_edge[w][w_b]) + float(lu_machine[int(b)])


def transport_timing_m2(
        src: AgvState, pickup: int, dropoff: int, ready: float,
        travel: Callable[[int, int], float],
        n_machine: int, machine_edge, lu_machine, lu_edge,
        dst: AgvState | None = None,
        src_repair: float = 0.0) -> dict:
    """M2 transport timing with load/unload stations.

    Same workshop (``dst is None``):
        deadhead(src.location -> pickup) + loaded(pickup -> dropoff).
        ``pickup`` may be a station node, which is how a job enters the shop.

    Cross workshop (``dst`` given):
        source deadhead + loaded(pickup -> station of the source workshop);
        station-to-station transfer (no AGV);
        target deadhead (station -> station) + loaded(station -> dropoff).

    ``loaded`` and ``empty`` are the AGV travel times used by the objective.
    """
    def dist(a, b):
        return m2_node_dist(a, b, travel, n_machine, machine_edge,
                            lu_machine, lu_edge)

    dead_src = dist(src.location, pickup)
    start_src = max(float(ready), float(src.free_time))
    wait_src = start_src - float(ready)

    if dst is None:
        loaded = dist(pickup, dropoff) + float(src_repair)
        return {
            "wait": wait_src + dead_src,
            "loaded": loaded,
            "empty": dead_src,
            "src_free": start_src + dead_src + loaded,
            "src_location": int(dropoff),
            "src_travelled": dead_src + loaded,
            "dst_free": None,
            "dst_location": None,
            "dst_travelled": 0.0,
            "t_src": loaded,
            "t_dst": 0.0,
            "dead_src": dead_src,
            "dead_dst": 0.0,
            "transfer": 0.0,
        }

    src_edge = int(machine_edge[int(pickup)])
    dst_edge = int(machine_edge[int(dropoff)])
    station_src = station_node(src_edge, n_machine)
    station_dst = station_node(dst_edge, n_machine)
    t_src = dist(pickup, station_src) + float(src_repair)
    transfer = float(lu_edge[src_edge][dst_edge])
    src_arrival = start_src + dead_src + t_src
    boundary = src_arrival + transfer
    dead_dst = dist(dst.location, station_dst)
    start_dst = max(boundary, float(dst.free_time))
    wait_dst = start_dst - boundary
    t_dst = dist(station_dst, dropoff)
    return {
        "wait": wait_src + dead_src + transfer + wait_dst + dead_dst,
        "loaded": t_src + t_dst,
        "empty": dead_src + dead_dst,
        "src_free": src_arrival,
        "src_location": station_src,
        "src_travelled": dead_src + t_src,
        "dst_free": start_dst + dead_dst + t_dst,
        "dst_location": int(dropoff),
        "dst_travelled": dead_dst + t_dst,
        "t_src": t_src,
        "t_dst": t_dst,
        "dead_src": dead_src,
        "dead_dst": dead_dst,
        "transfer": transfer,
    }


def available_mask(agvs: list[AgvState], now: float | None = None,
                   edge: int | None = None) -> np.ndarray:
    """Selectable AGVs by home workshop.

    Busy AGVs stay selectable: the caller may choose one and pay the wait.
    ``now`` is kept for API compatibility; it no longer filters the mask.
    """
    return np.asarray([
        edge is None or a.edge == edge
        for a in agvs
    ], dtype=bool)


def earliest_free_agv(agvs: list[AgvState], mask=None) -> int:
    """Rule dispatcher: earliest-free AGV, restricted by ``mask``."""
    candidates = [k for k, _a in enumerate(agvs)
                  if mask is None or bool(mask[k])]
    if not candidates:
        return 0
    return min(candidates, key=lambda k: (agvs[k].free_time, k))
