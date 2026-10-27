# -*- coding: utf-8 -*-
"""**top-K 候选构造**（v3 新架构）—— 规则只负责筛，打分头负责排。

设计见 `docs/v3/topk-rerank-spec.md`。三条口径：

1. **严格 K**：`K_eff = min(K, 合法候选数)`，永不超过 K。
2. **边界平局随机截断**：截断线落在平局组内时，从该组**均匀无放回抽**。
   不按工件号 / 机台号 / 车号 —— 那是任意标签偏置（同 `LAView.tie_rng` 的教训，§4.28）。
3. **每次决策重建 rng**：`default_rng([seed_chain, TOPK_SEED, kind, decision_index])`。
   候选集只由这三个数决定，与采样路径、回放路径无关。

⚠️ 本模块**不改** `rules.py` 的规则函数。`λ` 参数仍留给 CDR 基线；本模块一律传 0。
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..env.des import PM_CAND_DEFER, PM_CAND_NOW
from ..env.rule_views import PAView
from ..env.snapshot import Snapshot
from ..nn.features import NormContext
from .rules import (LA_RULE_NAMES, PA_JOB_RULES, PA_MACHINE_RULES,
                    PA_RULE_NAMES, PM_RULE_NAMES, la_score, la_terms,
                    pa_job_score, pa_job_terms, pa_machine_score,
                    pa_machine_terms)

K_PA = 16
"""PA 的严格 K。覆盖率探针：K=16 在 40 个采样决策上覆盖 100%。"""

K_LA = 6
"""LA 的严格 K。覆盖率探针：K=6 覆盖 100%（车队通常 <=6 台）。"""

TOPK_SEED = 6000
"""候选边界平局随机源的**独立子种子**。

⚠️ 不得借 `PA_TIE_SEED` / `LA_TIE_SEED` —— 借它会把规则平局的抽签流整体移位。
本种子只服务候选构造与 greedy 分数并列，**不进环境动力学**。
"""

KIND_CODE: dict[str, int] = {"PA": 0, "LA": 1, "RA": 2}
"""逐决策 rng 的 kind 分量（避免 PA/LA 的第 0 次决策抽到同一串数）。"""

PA_CAND_DIM = 9
LA_CAND_DIM = 4
RA_FEAT_DIM = 5
"""打分头的候选特征宽度（定义见 `_pa_feats` / `_la_feats` / `ra_features`）。"""


class PACand(NamedTuple):
    """PA 的一个候选：`(池下标, 工件, 工序, 机台)` + 9 维特征。"""

    pi: int
    job: int
    oi: int
    mach: int
    feats: np.ndarray


class LACand(NamedTuple):
    """LA 的一个候选：车号 + 4 维特征。"""

    agv: int
    feats: np.ndarray


class _PARow(NamedTuple):
    js: float
    ms: float
    pi: int
    mi: int
    job: int
    oi: int
    mach: int
    feats: np.ndarray


class _LARow(NamedTuple):
    score: tuple
    agv: int
    feats: np.ndarray


def topk_rng(seed_chain: int, kind: str, index: int) -> np.random.Generator:
    """本决策的候选构造随机源（**每次重建**，不跨决策推进）。"""
    if kind not in KIND_CODE:
        raise ValueError(f"未知的决策种类 {kind!r}；可选={tuple(KIND_CODE)}")
    return np.random.default_rng([int(seed_chain), TOPK_SEED,
                                 KIND_CODE[kind], int(index)])


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), lo), hi))


def _pa_idx(rule: int | str) -> int:
    if isinstance(rule, str):
        if rule not in PA_RULE_NAMES:
            raise ValueError(f"未知的 PA 规则名 {rule!r}；可选={PA_RULE_NAMES}")
        return PA_RULE_NAMES.index(rule)
    idx = int(rule)
    if not (0 <= idx < len(PA_RULE_NAMES)):
        raise ValueError(f"PA 规则下标 {idx} 越界（合法 0..{len(PA_RULE_NAMES) - 1}）")
    return idx


def _la_idx(rule: int | str) -> int:
    if isinstance(rule, str):
        if rule not in LA_RULE_NAMES:
            raise ValueError(f"未知的 LA 规则名 {rule!r}；可选={LA_RULE_NAMES}")
        return LA_RULE_NAMES.index(rule)
    idx = int(rule)
    if not (0 <= idx < len(LA_RULE_NAMES)):
        raise ValueError(f"LA 规则下标 {idx} 越界（合法 0..{len(LA_RULE_NAMES) - 1}）")
    return idx


def _pm_name(rule: int | str) -> str:
    if isinstance(rule, str):
        if rule not in PM_RULE_NAMES:
            raise ValueError(f"未知的保养规则名 {rule!r}；可选={PM_RULE_NAMES}")
        return rule
    idx = int(rule)
    if not (0 <= idx < len(PM_RULE_NAMES)):
        raise ValueError(f"保养规则下标 {idx} 越界（合法 0..{len(PM_RULE_NAMES) - 1}）")
    return PM_RULE_NAMES[idx]


def _cut_with_tie_groups(rows: list, k: int, score_key, order_key,
                         rng: np.random.Generator | None) -> list:
    """严格取前 `k` 行；截断线在平局组内时**均匀抽**。

    `rows` 已按 `(score_key, order_key)` 排好。`score_key` 相同 = 同一平局组。
    `rng is None` ⟹ 确定性取组内前几个（只给单测钉具体值；生产路径恒给 rng）。
    """
    if k <= 0:
        raise ValueError(f"K 必须为正，收到 {k}")
    if len(rows) <= k:
        return list(rows)
    cut = score_key(rows[k - 1])
    first = k - 1
    while first > 0 and score_key(rows[first - 1]) == cut:
        first -= 1
    last = k - 1
    while last + 1 < len(rows) and score_key(rows[last + 1]) == cut:
        last += 1
    group = rows[first:last + 1]
    need = k - first
    if need >= len(group):                       # 整组都在 K 内（防御性）
        return list(rows[:k])
    if rng is None:
        chosen = list(group[:need])
    else:
        perm = rng.permutation(len(group))
        chosen = [group[int(i)] for i in perm[:need]]
    chosen.sort(key=order_key)
    return list(rows[:first]) + chosen


def _pa_feats(jt: dict[str, float], mt: dict[str, float],
              m_ref: float) -> np.ndarray:
    """PA 候选的 9 维特征。定义与裁边见 spec §3.2。"""
    m = max(float(m_ref), 1e-9)
    return np.array([
        _clip(jt["apt"] / m, 0.0, 4.0),
        _clip(jt["rtj"] / m, 0.0, 4.0),
        _clip(jt["wait"] / m, 0.0, 4.0),
        _clip(jt["due"] / m, -4.0, 4.0),
        _clip(jt["ready_delay"] / m, 0.0, 4.0),
        _clip(mt["pt"] / m, 0.0, 4.0),
        _clip(mt["avail_delay"] / m, 0.0, 4.0),
        _clip(mt["util"], 0.0, 4.0),
        _clip(mt["pt_over_den"], 0.0, 4.0),
    ], dtype=np.float32)


def _la_feats(lt: dict[str, float], m_ref: float) -> np.ndarray:
    """LA 候选的 4 维特征。定义与裁边见 spec §3.2。"""
    m = max(float(m_ref), 1e-9)
    return np.array([
        _clip(lt["cost"] / m, 0.0, 4.0),
        _clip(lt["util"], 0.0, 4.0),
        _clip(lt["cost_over_den"], 0.0, 4.0),
        _clip(lt["arrive_delay"] / m, 0.0, 4.0),
    ], dtype=np.float32)


def pa_candidates(view: PAView, pool, rule: int | str, *,
                  k: int = K_PA, m_ref: float = 1.0,
                  rng: np.random.Generator | None = None) -> list[PACand]:
    """复合 PA 规则 → 按 `(工件分, 机台分)` 升序的前 K 个 `(j, oi, m)`。

    ⚠️ **先工件、后机台**的字典序：工件分不同的两行，永远按工件分排；
    工件分相同的组内才按机台分排。这与 `apply_pa_rule` 的"先选工件、再选机台"一致。
    """
    idx = _pa_idx(rule)
    n_mr = len(PA_MACHINE_RULES)
    jr = PA_JOB_RULES[idx // n_mr]
    mr = PA_MACHINE_RULES[idx % n_mr]
    rows: list[_PARow] = []
    for pi, (j, oi) in enumerate(pool):
        j, oi = int(j), int(oi)
        jv = view.jobs[j]
        if not jv.alts:
            continue
        js = float(pa_job_score(jr, view, j, (0.0, 0.0)))
        jt = pa_job_terms(view, j)
        for mi, (m, t) in enumerate(jv.alts):
            m, t = int(m), float(t)
            ms = float(pa_machine_score(mr, view, (m, t), (0.0,)))
            mt = pa_machine_terms(view, (m, t))
            rows.append(_PARow(js, ms, pi, mi, j, oi, m,
                               _pa_feats(jt, mt, m_ref)))
    rows.sort(key=lambda r: (r.js, r.ms, r.pi, r.mi))
    kept = _cut_with_tie_groups(rows, k, lambda r: (r.js, r.ms),
                                lambda r: (r.pi, r.mi), rng)
    return [PACand(pi=r.pi, job=r.job, oi=r.oi, mach=r.mach, feats=r.feats)
            for r in kept]


def la_candidates(view, rule: int | str, cand_v, *,
                  k: int = K_LA, m_ref: float = 1.0,
                  rng: np.random.Generator | None = None) -> list[LACand]:
    """LA 规则 → 按规则分升序的前 K 台车。"""
    name = LA_RULE_NAMES[_la_idx(rule)]
    rows: list[_LARow] = []
    for a in cand_v:
        a = int(a)
        s = tuple(la_score(name, view, a, (0.0,)))
        rows.append(_LARow(s, a, _la_feats(la_terms(view, a), m_ref)))
    rows.sort(key=lambda r: (r.score, r.agv))
    kept = _cut_with_tie_groups(rows, k, lambda r: r.score,
                                lambda r: r.agv, rng)
    return [LACand(agv=r.agv, feats=r.feats) for r in kept]


def ra_candidates(snap: Snapshot, rule: int | str, mach: int, cands) -> tuple[list[int], np.ndarray]:
    """RA 规则 → `(允许的动作码列表, 长度 2 的 mask)`。

    ⚠️ 阈值 `A` **不存在**：规则只做候选过滤，保养时机由打分头自己学。
    """
    name = _pm_name(rule)
    ms = snap.machines[int(mach)]
    if name == "AGE":
        allowed = [PM_CAND_NOW, PM_CAND_DEFER]
    elif name == "QUEUE_EMPTY":
        allowed = ([PM_CAND_NOW, PM_CAND_DEFER] if int(ms.in_q_len) == 0
                   else [PM_CAND_DEFER])
    elif name == "OLDEST":
        oldest = all(float(ms.pm_used_min) >= float(o.pm_used_min)
                     for o in snap.machines)
        allowed = ([PM_CAND_NOW, PM_CAND_DEFER] if oldest
                   else [PM_CAND_DEFER])
    else:                                        # pragma: no cover（_pm_name 已拦）
        raise ValueError(f"未知的保养规则名 {name!r}")
    present = {int(a) for a in cands}
    allowed = [a for a in allowed if a in present]
    mask = np.array([(PM_CAND_NOW in allowed), (PM_CAND_DEFER in allowed)],
                    dtype=bool)
    return allowed, mask


def ra_features(snap: Snapshot, mach: int, ctx: NormContext) -> np.ndarray:
    """RA 的 5 维特征（定义见 spec §3.2）。"""
    ms = snap.machines[int(mach)]
    pm = max(float(ctx.pm_interval), 1e-9)
    work = max(float(ctx.total_work_min), 1e-9)
    oldest = 1.0 if all(float(ms.pm_used_min) >= float(o.pm_used_min)
                        for o in snap.machines) else 0.0
    return np.array([
        _clip(float(ms.pm_used_min) / pm, 0.0, 1.5),
        _clip(int(ms.in_q_len) / max(int(ctx.max_queued), 1), 0.0, 1.0),
        oldest,
        _clip(float(ms.remaining_min) / work, 0.0, 1.0),
        _clip(float(ms.backlog_min) / work, 0.0, 1.0),
    ], dtype=np.float32)
