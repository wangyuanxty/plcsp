"""Rule definitions and logit-bias helpers for the three top-level heads.

The top level selects one job rule, one machine rule, and (when AGV is
enabled) one AGV rule.  The heads are sampled independently; each low-level
actor receives the bias of its own head.

Task feature layout (see ``disjunctive_graph_jsp_env_singlestep._state_array``):
    0 estimated start time, 1 estimated finish time, 2 estimated processing
    time, 3 scheduled flag, 4 in-degree, 5 assigned machine id, 6 assigned
    processing time, 7 assigned power, 8 job id, 9..11 random weights.

Machine candidate feature layout (see
``Parallel_env.cal_cur_task_machine_feature``):
    0 processing time, 1 processing energy (p*t), 2 transport time,
    3 selectable flag, 4 power, 5 edge id, 6..9 AGV features.

AGV feature layout (see ``agv.state.agv_features``):
    0 idle, 2 busy, 4 load, 5 edge, 6 location, 7 deadhead,
    8 loaded travel, 9 wait, 10 free time.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch


# Three top-level heads: job, machine, AGV.
JOB_RULES: Tuple[str, ...] = ("est", "eft", "spt", "mwko")
MACHINE_RULES: Tuple[str, ...] = ("min_t", "min_pt", "min_trans")
AGV_RULES: Tuple[str, ...] = ("earliest_free", "nearest", "random")
N_JOB_RULES: int = len(JOB_RULES)
N_MACHINE_RULES: int = len(MACHINE_RULES)
N_AGV_RULES: int = len(AGV_RULES)
# Kept for old call sites that still reason about the pair count.
N_RULE_PAIRS: int = N_JOB_RULES * N_MACHINE_RULES

# Rule indices that maximise a feature instead of minimising it.
_MAXIMISE_JOB_RULES = frozenset({3})


def _as_tensor(value, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _as_rule_ids(rule_ids, batch: int, device: torch.device) -> torch.Tensor:
    rid = _as_tensor(rule_ids, device, torch.long).reshape(-1)
    if rid.numel() == 1 and batch > 1:
        rid = rid.expand(batch)
    if rid.numel() != batch:
        raise ValueError(f"rule_ids has {rid.numel()} entries, expected {batch}")
    return rid


def _as_task_tensor(task_fea, batch: int, device: torch.device) -> torch.Tensor:
    x = _as_tensor(task_fea, device, torch.float32)
    if x.dim() == 2:
        if x.shape[0] % batch != 0:
            raise ValueError(
                f"task features {tuple(x.shape)} cannot split into {batch} envs")
        x = x.reshape(batch, x.shape[0] // batch, x.shape[-1])
    if x.dim() != 3 or x.shape[0] != batch:
        raise ValueError(f"bad task feature shape {tuple(x.shape)}")
    return x


def _as_candidate(candidate, device: torch.device) -> torch.Tensor:
    cand = _as_tensor(candidate, device, torch.long)
    if cand.dim() == 1:
        cand = cand.unsqueeze(0)
    return cand


def _as_mask(mask, shape: Sequence[int], device: torch.device) -> torch.Tensor:
    m = _as_tensor(mask, device, torch.bool)
    if m.dim() == 1:
        m = m.unsqueeze(0)
    if tuple(m.shape) != tuple(shape):
        m = m.reshape(*shape)
    return m


def _gather_candidate_features(
        task_fea: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    idx = candidate.unsqueeze(-1).expand(-1, -1, task_fea.size(-1))
    return torch.gather(task_fea, 1, idx)


def _preference(x: torch.Tensor, valid: torch.Tensor,
                maximise: bool) -> torch.Tensor:
    """Per-env min-max preference in [-1, 1], zero for invalid entries."""
    pos_inf = torch.finfo(x.dtype).max
    neg_inf = torch.finfo(x.dtype).min
    has_valid = valid.any(dim=1, keepdim=True)
    x_min = torch.where(valid, x, pos_inf).min(dim=1, keepdim=True).values
    x_max = torch.where(valid, x, neg_inf).max(dim=1, keepdim=True).values
    x_min = torch.where(has_valid, x_min, torch.zeros_like(x_min))
    x_max = torch.where(has_valid, x_max, torch.zeros_like(x_max))
    span = (x_max - x_min).clamp_min(1e-6)
    norm = ((x - x_min) / span).clamp(0.0, 1.0)
    if not maximise:
        norm = 1.0 - norm
    norm = torch.where(valid, norm, torch.zeros_like(norm))
    count = valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = norm.sum(dim=1, keepdim=True) / count
    return torch.where(valid, norm - mean, torch.zeros_like(norm))


def _masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    count = valid.float().sum(dim=1).clamp_min(1.0)
    total = torch.where(valid, x, torch.zeros_like(x)).sum(dim=1)
    return total / count


def _masked_min(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    has_valid = valid.any(dim=1)
    value = torch.where(valid, x, torch.finfo(x.dtype).max).min(dim=1).values
    return torch.where(has_valid, value, torch.zeros_like(value))


def job_rule_bias(task_fea, candidate, mask_operation, rule_ids,
                  n_machine: int, alpha: float = 1.0,
                  device: Optional[torch.device] = None) -> torch.Tensor:
    """Return a (B, n_job) zero-centred bias for the selected job rule."""
    cand = _as_candidate(candidate, device or torch.device("cpu"))
    batch, n_job = cand.shape
    dev = cand.device
    rid = _as_rule_ids(rule_ids, batch, dev)
    mask = _as_mask(mask_operation, (batch, n_job), dev)
    x_all = _as_task_tensor(task_fea, batch, dev)
    feat = _gather_candidate_features(x_all, cand)        # (B, n_job, F)

    job_rule = rid.clamp(0, N_JOB_RULES - 1)
    bias = torch.zeros((batch, n_job), device=dev, dtype=torch.float32)
    valid = ~mask
    for rule in range(len(JOB_RULES)):
        rows = job_rule == rule
        if not bool(rows.any()):
            continue
        if JOB_RULES[rule] == "mwko":
            # Remaining operations of the candidate job (including this one).
            x = (n_machine - (cand % n_machine)).float()
        else:
            index = {"est": 0, "eft": 1, "spt": 2}[JOB_RULES[rule]]
            x = feat[:, :, index]
        pref = _preference(x, valid, maximise=rule in _MAXIMISE_JOB_RULES)
        bias[rows] = alpha * pref[rows]
    return bias


def machine_rule_bias(machine_fea1, machine_mask, rule_ids,
                      alpha: float = 1.0,
                      device: Optional[torch.device] = None) -> torch.Tensor:
    """Return a (B, n_machine) zero-centred bias for the selected machine rule."""
    dev = device or (machine_fea1.device if torch.is_tensor(machine_fea1)
                     else torch.device("cpu"))
    m1 = _as_tensor(machine_fea1, dev, torch.float32)
    if m1.dim() != 3:
        raise ValueError(f"bad machine feature shape {tuple(m1.shape)}")
    batch, n_machine, _ = m1.shape
    rid = _as_rule_ids(rule_ids, batch, dev)
    mask = _as_mask(machine_mask, (batch, 1, n_machine), dev).squeeze(1)
    valid = ~mask

    machine_rule = rid.clamp(0, N_MACHINE_RULES - 1)
    bias = torch.zeros((batch, n_machine), device=dev, dtype=torch.float32)
    for rule in range(len(MACHINE_RULES)):
        rows = machine_rule == rule
        if not bool(rows.any()):
            continue
        x = m1[:, :, rule]
        pref = _preference(x, valid, maximise=False)
        bias[rows] = alpha * pref[rows]
    return bias


def agv_rule_bias(agv_feat, avail_mask, rule_ids,
                  alpha: float = 1.0,
                  device: Optional[torch.device] = None) -> torch.Tensor:
    """Return a (B, n_agv) zero-centred bias for the selected AGV rule.

    ``rule_ids`` are indices into :data:`AGV_RULES`:
    0 earliest-free, 1 nearest (minimum deadhead), 2 random (zero bias).
    """
    dev = device or (agv_feat.device if torch.is_tensor(agv_feat)
                     else torch.device("cpu"))
    feat = _as_tensor(agv_feat, dev, torch.float32)
    if feat.dim() != 3:
        raise ValueError(f"bad AGV feature shape {tuple(feat.shape)}")
    batch, n_agv, _ = feat.shape
    rid = _as_rule_ids(rule_ids, batch, dev)
    valid = _as_mask(avail_mask, (batch, n_agv), dev)
    bias = torch.zeros((batch, n_agv), device=dev, dtype=torch.float32)
    for rule, name in enumerate(AGV_RULES):
        rows = rid == rule
        if not bool(rows.any()):
            continue
        if name == "earliest_free":
            pref = _preference(feat[:, :, 10], valid, maximise=False)
        elif name == "nearest":
            pref = _preference(feat[:, :, 7], valid, maximise=False)
        else:  # random: no prior
            pref = torch.zeros_like(bias)
        bias[rows] = alpha * pref[rows]
    return bias


@torch.no_grad()
def select_agv_by_rule(agv_feat, avail_mask, rule_ids,
                       generator: Optional[torch.Generator] = None
                       ) -> torch.Tensor:
    """Pick one AGV per env by the selected rule.

    Returns a (B,) int64 tensor.  ``random`` samples uniformly among the
    selectable AGVs; the other rules pick the minimum score.
    """
    dev = agv_feat.device if torch.is_tensor(agv_feat) else torch.device("cpu")
    feat = _as_tensor(agv_feat, dev, torch.float32)
    if feat.dim() != 3:
        raise ValueError(f"bad AGV feature shape {tuple(feat.shape)}")
    batch, n_agv, _ = feat.shape
    rid = _as_rule_ids(rule_ids, batch, dev)
    valid = _as_mask(avail_mask, (batch, n_agv), dev)
    scores = torch.zeros((batch, n_agv), device=dev, dtype=torch.float32)
    for rule, name in enumerate(AGV_RULES):
        rows = rid == rule
        if not bool(rows.any()):
            continue
        if name == "earliest_free":
            scores[rows] = -feat[rows, :, 10]
        elif name == "nearest":
            scores[rows] = -feat[rows, :, 7]
        else:
            scores[rows] = torch.rand(
                (int(rows.sum()), n_agv), device=dev, generator=generator)
    scores = scores.masked_fill(~valid, float("-inf"))
    action = scores.argmax(dim=-1)
    no_valid = ~valid.any(dim=-1)
    return torch.where(no_valid, torch.zeros_like(action), action)


def top_state_features(task_fea, candidate, mask_operation,
                       machine_fea2, step: int,
                       total_steps: int, n_machine: int, n_job: int,
                       weights=None, agv_stats=None, cost_trend=None,
                       device: Optional[torch.device] = None) -> torch.Tensor:
    """Compact per-env state summary for the top-level PPO actor (B, 18).

    Fifteen state summaries plus the three active preference weights.

    Summaries: progress, available fraction, min/mean EST, mean PT, mean
    remaining, four machine stats, machine load imbalance, WIP, AGV busy
    fraction, AGV mean wait, and the last segment cost trend.
    """
    cand = _as_candidate(candidate, device or torch.device("cpu"))
    batch, n_job = cand.shape
    dev = cand.device
    mask = _as_mask(mask_operation, (batch, n_job), dev)
    valid = ~mask
    x_all = _as_task_tensor(task_fea, batch, dev)
    feat = _gather_candidate_features(x_all, cand)

    est = feat[:, :, 0]
    eft = feat[:, :, 1]
    pt = feat[:, :, 2]
    scale = torch.where(valid, eft, torch.zeros_like(eft)).max(
        dim=1).values.clamp_min(1.0)

    remaining = (n_machine - (cand % n_machine)).float()
    progress = torch.full((batch,), float(step) / max(1, total_steps),
                          device=dev)
    n_available = valid.float().mean(dim=1)
    min_est = _masked_min(est, valid) / scale
    mean_est = _masked_mean(est, valid) / scale
    mean_pt = _masked_mean(pt, valid) / scale
    mean_remaining = _masked_mean(remaining, valid) / max(1, n_machine)
    # WIP proxy: fraction of tasks not yet scheduled (feature 3).
    wip = 1.0 - x_all[:, :, 3].mean(dim=1)

    # Scheduled-machine state: last finish, cumulative PT / transport / idle.
    if machine_fea2 is None:
        m2_stats = torch.zeros((batch, 4), device=dev)
        load_imb = torch.zeros((batch,), device=dev)
    else:
        m2 = _as_tensor(machine_fea2, dev, torch.float32)
        m2_stats = torch.stack(
            [m2[:, :, 0].mean(dim=1) / scale,
             m2[:, :, 1].mean(dim=1) / scale,
             m2[:, :, 2].mean(dim=1) / scale,
             m2[:, :, 3].mean(dim=1) / scale], dim=-1)
        # Coefficient of variation of cumulative processing load.
        load = m2[:, :, 1]
        load_imb = (load.std(dim=1, unbiased=False)
                    / load.mean(dim=1).clamp_min(1.0))

    if agv_stats is None:
        agv_busy = torch.zeros((batch,), device=dev)
        agv_wait = torch.zeros((batch,), device=dev)
    else:
        a = _as_tensor(agv_stats, dev, torch.float32)
        if a.dim() == 1:
            a = a.unsqueeze(0).expand(batch, -1)
        agv_busy = a[:, 0]
        agv_wait = a[:, 1]
    if cost_trend is None:
        trend = torch.zeros((batch,), device=dev)
    else:
        trend = _as_tensor(cost_trend, dev, torch.float32).reshape(-1)
        if trend.numel() == 1 and batch > 1:
            trend = trend.expand(batch)

    if weights is None:
        w = torch.tensor([0.4, 0.4, 0.2], device=dev,
                         dtype=torch.float32).unsqueeze(0).expand(batch, -1)
    else:
        w = _as_tensor(weights, dev, torch.float32)
        if w.dim() == 1:
            w = w.unsqueeze(0).expand(batch, -1)
        if tuple(w.shape) != (batch, 3):
            raise ValueError(
                f"top-level weights {tuple(w.shape)} do not match "
                f"batch {batch}")

    base = torch.stack(
        [progress, n_available, min_est, mean_est, mean_pt,
         mean_remaining, m2_stats[:, 0], m2_stats[:, 1],
         m2_stats[:, 2], m2_stats[:, 3], load_imb, wip,
         agv_busy, agv_wait, trend], dim=-1)
    return torch.cat([base, w], dim=-1).clamp(-1e6, 1e6)
