# -*- coding: utf-8 -*-
"""Rule features and priority vectors for the adapted Li et al. baseline.

Li et al. (AEI 65 (2025) 103216) use three agents with tiny rule-weight action
spaces.  The real choice is made by min-max normalised priority vectors (PDRs):

* job filter: ``Norm(Sigmoid(a_JF) . [PV_BETC, PV_LWKR, PV_CR, PV_EDD])
  + 50 * PV_SEAT``;
* job selection: ``Norm(||a_JS - attribute||_2) + 50 * PV_SEAT``;
* AGV selection: ``Norm(a_AS . [PV_SCT, PV_SPUT]) + 5 * PV_SCPT``.

This module ports that mechanism to the v5 shop.  Two rules need due dates
(CR and EDD) and job weights, which the v5 instances do not contain.  They are
replaced by objective-aligned rules (MEC: remaining processing energy, MTT:
remaining transport).  No due date is synthesised.  A fourth, added machine
selection agent uses SPT, SEYEC, and EFT.  See ``eval_li2025.py`` for the full
adaptation list.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from li2025.env_model import (W_EC, W_MK, W_TT, Instance, ScheduleSession,
                              minmax4, moments, norm01)

FILTER_SIZE = 30
OMEGA_SEAT = 50.0
OMEGA_SCPT = 5.0

JOB_STATE_DIM = 26
MACHINE_STATE_DIM = 21


@dataclass(frozen=True)
class JobContext:
    """Per-job decision input at one macro-step."""

    job: int
    task: int
    eat: float
    rtlp: float
    rem_energy: float
    rem_transport: float
    next_time: float
    etc: float
    seat: float
    mean_machine_util: float
    mean_machine_ctm: float


def build_job_contexts(session: ScheduleSession) -> list[JobContext]:
    """Build the decision context of every pending job."""
    inst = session.inst
    now = session.world_time()
    out: list[JobContext] = []
    for j in session.pending_jobs():
        task = session.next_task(j)
        eat = session.job_ready(j)
        rtlp, rem_e, rem_tr = session.job_remaining(j)
        avail = inst.machines(task)
        next_t = float(np.min(inst.t[task][avail]))
        util = float(np.mean([session.machine_util(int(m)) for m in avail]))
        ctm = float(np.mean([session.machine_free(int(m)) for m in avail]))
        # Objective-aligned surrogate for the paper's estimated tardiness
        # cost.  It uses the same three v5 objective weights.
        etc = (W_MK * (eat + rtlp)
               + W_EC * (rem_e / inst.n_task)
               + W_TT * rem_tr)
        seat = 1.0 if eat <= now + 1e-9 else 0.0
        out.append(JobContext(
            job=int(j), task=int(task), eat=float(eat), rtlp=float(rtlp),
            rem_energy=float(rem_e), rem_transport=float(rem_tr),
            next_time=next_t, etc=float(etc), seat=seat,
            mean_machine_util=util, mean_machine_ctm=ctm,
        ))
    return out


def job_rule_pvs(ctxs: list[JobContext]) -> np.ndarray:
    """Return a (5, n) matrix: BETC, LWKR, MEC, MTT, SEAT priority vectors."""
    etc = np.asarray([c.etc for c in ctxs], dtype=np.float64)
    rtlp = np.asarray([c.rtlp for c in ctxs], dtype=np.float64)
    energy = np.asarray([c.rem_energy for c in ctxs], dtype=np.float64)
    trans = np.asarray([c.rem_transport for c in ctxs], dtype=np.float64)
    seat = np.asarray([c.seat for c in ctxs], dtype=np.float64)
    return np.vstack([
        norm01(etc),          # BETC: biggest estimated cost first
        1.0 - norm01(rtlp),   # LWKR: least remaining work first
        norm01(energy),       # MEC: most remaining energy first
        norm01(trans),        # MTT: most remaining transport first
        seat,                 # SEAT: ready jobs first
    ])


def job_scores(weights: np.ndarray, pvs: np.ndarray) -> np.ndarray:
    """FPJF with four rule weights already in [0, 1] (sigmoid applied)."""
    z = np.asarray(weights, dtype=np.float64) @ pvs[:4]
    return norm01(z) + OMEGA_SEAT * pvs[4]


def job_attr_matrix(ctxs: list[JobContext]) -> np.ndarray:
    """Candidate attribute vectors scaled to [-1, 1] (paper Eq. 13)."""
    raw = np.asarray(
        [[c.rtlp, c.etc, c.rem_energy, c.rem_transport] for c in ctxs],
        dtype=np.float64,
    )
    if raw.shape[0] == 1:
        return np.zeros((1, 4), dtype=np.float64)
    out = np.empty_like(raw)
    for col in range(raw.shape[1]):
        out[:, col] = 2.0 * norm01(raw[:, col]) - 1.0
    return out


def job_proto_scores(proto: np.ndarray, attrs: np.ndarray,
                     seat: np.ndarray) -> np.ndarray:
    """FPJS.  The paper writes ``Norm(||a - attr||)`` and then "highest
    priority"; a distance is smaller-is-better, so we negate it.  The sign is
    otherwise ambiguous in the paper.  ``seat`` is added with weight 50."""
    dist = np.linalg.norm(attrs - np.asarray(proto)[None, :], axis=1)
    return norm01(-dist) + OMEGA_SEAT * np.asarray(seat, dtype=np.float64)


def machine_rule_pvs(session: ScheduleSession, task: int, prev_m: int,
                     candidates: np.ndarray) -> np.ndarray:
    """Return a (3, k) matrix: SPT, SEYEC, EFT priority vectors."""
    inst: Instance = session.inst
    cand = np.asarray(candidates, dtype=int)
    times = inst.t[int(task)][cand]
    energy = times * inst.p[int(task)][cand]
    free = np.asarray([session.machine_free(int(m)) for m in cand])
    dead = np.asarray([
        session.node_dist(prev_m, int(m)) if prev_m >= 0 else 0.0
        for m in cand
    ])
    completion = free + dead + times
    return np.vstack([
        1.0 - norm01(times),       # SPT
        1.0 - norm01(energy),      # SEYEC
        1.0 - norm01(completion),  # EFT
    ])


def machine_scores(weights: np.ndarray, pvs: np.ndarray) -> np.ndarray:
    """FPMS, the added machine-selection composite."""
    return norm01(np.asarray(weights, dtype=np.float64) @ pvs)


def agv_rule_pvs(session: ScheduleSession, prev_m: int, m: int,
                 ready: float) -> tuple[np.ndarray, list[int]]:
    """Return ((3, k) SCT/SPUT/SCPT vectors, source AGV ids).

    ``CTA`` is the earliest start time of the transport, i.e.
    ``max(AGV free time, job ready time)``.  This differs slightly from the
    paper, where CTA is the AGV completion time only; including readiness
    avoids selecting a vehicle that arrives before the load is ready.
    """
    src = session.source_candidates(prev_m)
    cta = np.asarray([
        max(float(session.agvs[k].free_time), float(ready)) for k in src
    ])
    # ``node_dist`` handles station nodes under M2 and is machine-to-machine
    # travel under M1.
    put = np.asarray([
        session.node_dist(session.agvs[k].location, prev_m) for k in src
    ])
    scpt = cta + put
    return np.vstack([
        1.0 - norm01(cta),
        1.0 - norm01(put),
        1.0 - norm01(scpt),
    ]), src


def agv_scores(weights: np.ndarray, pvs: np.ndarray) -> np.ndarray:
    """FPAS with raw weights (the paper does not sigmoid the AGV weights)."""
    z = np.asarray(weights, dtype=np.float64) @ pvs[:2]
    return norm01(z) + OMEGA_SCPT * pvs[2]


# ----------------------------------------------------------------------
# State builders (adapted S1-S6)
# ----------------------------------------------------------------------
def _job_stats(session: ScheduleSession) -> np.ndarray:
    """S1: order completion rate and mean/std of job completion rates."""
    inst = session.inst
    rates = np.asarray([session.op_count[j] / inst.n_ops
                        for j in range(inst.n_job)], dtype=np.float64)
    mean, std = moments(rates)
    return np.asarray([session.n_scheduled() / inst.n_task, mean, std])


def _machine_stats(session: ScheduleSession) -> np.ndarray:
    """S2: mean/std of machine utilisation and CTM (machine free time)."""
    util = np.asarray([session.machine_util(m)
                       for m in range(session.inst.n_machine)])
    ctm = np.asarray([session.machine_free(m)
                      for m in range(session.inst.n_machine)])
    u_mean, u_std = moments(util)
    c_mean, c_std = moments(ctm)
    return np.asarray([u_mean, u_std, c_mean, c_std])


def _agv_stats(session: ScheduleSession) -> np.ndarray:
    """S3: mean/std of AGV utilisation and CTA (AGV free time)."""
    util = np.asarray([session.agv_util(k) for k in range(len(session.agvs))])
    cta = np.asarray([session.agvs[k].free_time
                      for k in range(len(session.agvs))])
    u_mean, u_std = moments(util)
    c_mean, c_std = moments(cta)
    return np.asarray([u_mean, u_std, c_mean, c_std])


def _pool_stats(ctxs: list[JobContext]) -> np.ndarray:
    """S4/S5: statistics of a job pool (15 features, as in Table 4)."""
    if not ctxs:
        return np.zeros(15, dtype=np.float64)
    next_t = np.asarray([c.next_time for c in ctxs])
    rtlp = np.asarray([c.rtlp for c in ctxs])
    eat = np.asarray([c.eat for c in ctxs])
    etc = np.asarray([c.etc for c in ctxs])
    n_mean, n_std = moments(next_t)
    r_mean, r_std = moments(rtlp)
    e_mean, e_std, e_min, e_max = minmax4(eat)
    c_mean, c_std, c_min, c_max = minmax4(etc)
    util = float(np.mean([c.mean_machine_util for c in ctxs]))
    ctm = float(np.mean([c.mean_machine_ctm for c in ctxs]))
    return np.asarray([
        float(len(ctxs)), n_mean, n_std, r_mean, r_std,
        e_mean, e_std, e_min, e_max,
        c_mean, c_std, c_min, c_max, util, ctm,
    ])


def job_filter_state(session: ScheduleSession,
                     ctxs: list[JobContext]) -> np.ndarray:
    """Observation of the job filter agent: {S1, S2, S3, S4}."""
    return np.concatenate([
        _job_stats(session), _machine_stats(session),
        _agv_stats(session), _pool_stats(ctxs),
    ])


def job_selection_state(session: ScheduleSession,
                        ctxs: list[JobContext]) -> np.ndarray:
    """Observation of the job selection agent: {S1, S2, S3, S5}."""
    return np.concatenate([
        _job_stats(session), _machine_stats(session),
        _agv_stats(session), _pool_stats(ctxs),
    ])


def machine_state(session: ScheduleSession, task: int, prev_m: int,
                  candidates: np.ndarray) -> np.ndarray:
    """Observation of the added machine agent: {S2} + candidate statistics."""
    inst = session.inst
    cand = np.asarray(candidates, dtype=int)
    times = inst.t[int(task)][cand]
    energy = times * inst.p[int(task)][cand]
    free = np.asarray([session.machine_free(int(m)) for m in cand])
    dead = np.asarray([
        session.node_dist(prev_m, int(m)) if prev_m >= 0 else 0.0
        for m in cand
    ])
    stats: list[float] = []
    for col in (times, energy, free, dead):
        stats.extend(minmax4(np.asarray(col, dtype=np.float64)))
    return np.concatenate([
        _machine_stats(session), np.asarray(stats + [float(len(cand))]),
    ])


def agv_state_dim(n_agv: int) -> int:
    """S2 (4) + S3 (4) + S6 (2 * n_agv + 3)."""
    return 11 + 2 * int(n_agv)


def agv_state(session: ScheduleSession, ctx: JobContext, m: int,
              prev_m: int) -> np.ndarray:
    """Observation of the AGV agent: {S2, S3, S6}."""
    ready = float(ctx.eat)
    cta = np.asarray([max(float(a.free_time), ready) for a in session.agvs])
    # An AGV may wait at a station node under M2; ``node_dist`` supports it.
    put = np.asarray([session.node_dist(a.location, prev_m)
                      for a in session.agvs])
    return np.concatenate([
        _machine_stats(session), _agv_stats(session), cta, put,
        np.asarray([ctx.eat, ctx.etc, session.machine_free(int(m))]),
    ])


def state_dims(n_agv: int) -> dict[str, int]:
    """Observation dimensions of the four actors."""
    return {
        "jf": JOB_STATE_DIM,
        "js": JOB_STATE_DIM,
        "ms": MACHINE_STATE_DIM,
        "as": agv_state_dim(n_agv),
    }
