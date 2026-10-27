"""Constraint event models for v5 (Stage-0).

Machine failure and AGV failure are modeled as random repair delays:

* failure probability per active interval = 1 - exp(-active / MTBF)
* repair time ~ Exponential(MTTR)

The base environment is unchanged when the flags are off.
"""

from __future__ import annotations

import math

import numpy as np


def failure_probability(active_time: float, mtbf: float) -> float:
    if mtbf <= 0.0 or active_time <= 0.0:
        return 0.0
    return 1.0 - math.exp(-float(active_time) / float(mtbf))


def sample_repair(rng: np.random.Generator, active_time: float,
                  mtbf: float, mttr: float) -> float:
    """Return 0.0 if no failure; otherwise an exponential repair time."""
    if mtbf <= 0.0 or mttr <= 0.0 or active_time <= 0.0:
        return 0.0
    if rng.random() >= failure_probability(active_time, mtbf):
        return 0.0
    return float(rng.exponential(float(mttr)))


def buffer_mask(routes, start_time, ready: float, capacity: int,
                n_machine: int) -> np.ndarray:
    """Stage-0 finite-buffer mask: True = machine input buffer is full.

    A scheduled task with ``start_time > ready`` is counted as waiting.
    """
    mask = np.zeros(int(n_machine), dtype=bool)
    if capacity <= 0:
        return mask
    for m, route in routes.items():
        waiting = sum(
            1 for t in route
            if float(start_time.get(int(t), 0.0)) > float(ready))
        if int(m) < int(n_machine):
            mask[int(m)] = waiting >= int(capacity)
    return mask


def buffer_mask_for_env(env, task_idx: int, capacity: int,
                        n_machine: int) -> np.ndarray:
    """Same as :func:`buffer_mask`, using a single graph env."""
    if capacity <= 0:
        return np.zeros(int(n_machine), dtype=bool)
    if int(task_idx) % int(n_machine) == 0:
        ready = 0.0
    else:
        # task_idx is 0-based; graph node ``task_idx`` is the predecessor.
        ready = float(env.G.nodes[int(task_idx)].get("finish_time") or 0.0)
    start_time = {}
    for m, route in env.machine_routes.items():
        for t in route:
            start_time[int(t)] = float(
                env.G.nodes[int(t)].get("start_time") or 0.0)
    return buffer_mask(env.machine_routes, start_time, ready,
                       capacity, n_machine)
