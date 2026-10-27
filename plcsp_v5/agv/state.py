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
            0.0,
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
