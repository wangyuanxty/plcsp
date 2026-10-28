# -*- coding: utf-8 -*-
"""One episode of the adapted Li et al. scheduler.

Decision order at every macro-step (one scheduled operation):

1. job filter agent: rule weights over BETC/LWKR/MEC/MTT plus SEAT; keeps
   up to 30 jobs (no-op for the 6- and 10-job v5 instances);
2. job selection agent: a 4-dimensional proto-action, decoded by Euclidean
   distance to the candidate attributes plus SEAT;
3. added machine agent: rule weights over SPT/SEYEC/EFT (not in the paper;
   FJSP needs a machine choice);
4. AGV selection agent: rule weights over SCT/SPUT plus SCPT; the source
   workshop AGV is learned, the destination AGV is deterministic.

The shared reward is the drop in the objective potential between consecutive
macro-steps.  Summed over an episode it equals ``C_0 - objective``, the v5
analogue of the paper's Eq. (18).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from li2025.env_model import DEFAULT_IDLE_MODE, Instance, ScheduleSession
from li2025.features import (FILTER_SIZE, JobContext, agv_rule_pvs,
                             agv_scores, agv_state, build_job_contexts,
                             job_attr_matrix, job_filter_state,
                             job_proto_scores, job_rule_pvs, job_scores,
                             job_selection_state, machine_rule_pvs,
                             machine_scores, machine_state)
from li2025.policy import (AGENTS, PolicyBundle, PPOConfig, Step, Transition,
                           compute_gae)


@dataclass
class EpisodeResult:
    """Cost components and per-agent transitions of one episode."""

    cost: dict[str, float]
    transitions: dict[str, list[Transition]] = field(default_factory=dict)
    steps: int = 0
    n_transport: int = 0
    return_sum: float = 0.0
    wall_s: float = 0.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _transition(step: Step, reward: float,
                global_state: np.ndarray) -> Transition:
    return Transition(obs=step.obs, action=step.action,
                      log_prob=step.log_prob, value=step.value,
                      reward=float(reward), done=False,
                      global_state=np.asarray(global_state,
                                              dtype=np.float32))


def run_episode(inst: Instance, bundle: PolicyBundle, cfg: PPOConfig,
                rng: np.random.Generator, training: bool,
                n_agv: int = 6,
                idle_mode: str = DEFAULT_IDLE_MODE) -> EpisodeResult:
    """Run one instance and return its cost and PPO transitions."""
    t0 = time.perf_counter()
    session = ScheduleSession(inst, n_agv=n_agv, idle_mode=idle_mode)
    noise = rng.normal(0.0, cfg.noise_sigma, cfg.noise_dim).astype(np.float32)
    c_prev = session.cost_estimate()
    scale = max(1.0, abs(c_prev))
    episode = {agent: [] for agent in AGENTS}
    pending = {agent: 0.0 for agent in AGENTS}
    steps = 0
    n_transport = 0
    return_sum = 0.0

    while not session.is_done():
        acted: dict[str, Step] = {}
        ctxs = build_job_contexts(session)
        raw_jf = job_filter_state(session, ctxs)
        norm_jf = bundle.normalize("jf", raw_jf, update=training)
        global_state = np.concatenate([norm_jf, noise])

        # 1) Job filter agent (inactive when the pool has <= 30 jobs).
        if len(ctxs) > FILTER_SIZE:
            step_jf = bundle.act("jf", norm_jf, global_state,
                                 deterministic=not training)
            weights = _sigmoid(step_jf.action)
            pvs_jf = job_rule_pvs(ctxs)
            order = np.argsort(-job_scores(weights, pvs_jf), kind="stable")
            filtered = [ctxs[int(i)] for i in order[:FILTER_SIZE]]
            acted["jf"] = step_jf
        else:
            filtered = ctxs

        # 2) Job selection agent.
        raw_js = job_selection_state(session, filtered)
        obs_js = bundle.normalize("js", raw_js, update=training)
        step_js = bundle.act("js", obs_js, global_state,
                             deterministic=not training)
        proto = np.tanh(step_js.action)
        attrs = job_attr_matrix(filtered)
        seat = np.asarray([c.seat for c in filtered], dtype=np.float64)
        chosen: JobContext = filtered[int(np.argmax(
            job_proto_scores(proto, attrs, seat)))]
        acted["js"] = step_js

        j = chosen.job
        task = chosen.task
        prev_m = (-1 if session.op_count[j] == 0
                  else int(session.machine_of[task - 1]))
        candidates = inst.machines(task)

        # 3) Added machine selection agent (skip when only one machine works).
        if len(candidates) > 1:
            raw_ms = machine_state(session, task, prev_m, candidates)
            obs_ms = bundle.normalize("ms", raw_ms, update=training)
            step_ms = bundle.act("ms", obs_ms, global_state,
                                 deterministic=not training)
            pvs_ms = machine_rule_pvs(session, task, prev_m, candidates)
            scores_ms = machine_scores(_sigmoid(step_ms.action), pvs_ms)
            m = int(candidates[int(np.argmax(scores_ms))])
            acted["ms"] = step_ms
        else:
            m = int(candidates[0])

        # 4) AGV selection agent (only when a loaded leg is needed).
        plan = None
        if prev_m >= 0 and prev_m != m:
            ready = session.job_ready(j)
            src_c = session.source_candidates(prev_m)
            if len(src_c) > 1:
                raw_as = agv_state(session, chosen, m, prev_m)
                obs_as = bundle.normalize("as", raw_as, update=training)
                step_as = bundle.act("as", obs_as, global_state,
                                     deterministic=not training)
                pvs_as, src_c = agv_rule_pvs(session, prev_m, m, ready)
                scores_as = agv_scores(step_as.action, pvs_as)
                src_idx = int(src_c[int(np.argmax(scores_as))])
                acted["as"] = step_as
            else:
                src_idx = int(src_c[0])
            plan = session.plan_transport(prev_m, m, ready, src_idx=src_idx)
            plan = session.best_destination(plan)
            n_transport += 1
        session.apply(j, m, plan=plan)

        # Shared potential-based reward, as in the paper's Eq. (17).
        c_new = session.cost_estimate()
        reward = (c_prev - c_new) / scale
        c_prev = c_new
        return_sum += reward
        for agent in AGENTS:
            if agent in acted:
                tr = _transition(acted[agent], pending[agent] + reward,
                                 global_state)
                pending[agent] = 0.0
                episode[agent].append(tr)
            else:
                pending[agent] += reward
        steps += 1

    cost = session.finalize()
    for agent in AGENTS:
        if episode[agent]:
            episode[agent][-1].done = True
            episode[agent][-1].reward += pending[agent]
        compute_gae(episode[agent], cfg.gamma, cfg.lam)
    return EpisodeResult(cost=cost, transitions=episode, steps=steps,
                         n_transport=n_transport,
                         return_sum=float(return_sum),
                         wall_s=time.perf_counter() - t0)
