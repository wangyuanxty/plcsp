# -*- coding: utf-8 -*-
"""top-K 候选构造（`hrl/candidates.py`）的验收。

钉住五条：
1. **严格 K**：候选数永不超过 K；合法候选 <= K 时全部出现且不重复。
2. **边界平局随机截断**：截断线在平局组内时，每个成员被选中的频率相近，
   **与工件号 / 机台号 / 车号无关**。
3. **PA 首候选 = 规则的最小 `(工件分, 机台分)`**。
4. **LA**：`n_agv <= K` 时等于全部车。
5. **RA**：规则只做候选过滤（`QUEUE_EMPTY` / `OLDEST`）。
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from plcsp_v3.env import layout as layout_mod
from plcsp_v3.env.corridors import build_corridor_graph, dock_distance_matrix
from plcsp_v3.env.des import PM_CAND_DEFER, PM_CAND_NOW, SimConfig, SimWorld
from plcsp_v3.env.instances import load_mk
from plcsp_v3.env.rule_views import LAVehicle, LAView, build_pa_view
from plcsp_v3.hrl.candidates import (K_LA, K_PA, _cut_with_tie_groups,
                                     la_candidates, pa_candidates,
                                     ra_candidates, ra_features, topk_rng)
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, PA_JOB_RULES, PA_MACHINE_RULES,
                                PA_RULE_NAMES, pa_job_score, pa_machine_score)
from plcsp_v3.nn.features import norm_context


def _fixture():
    inst = load_mk("mk01")
    cfg = SimConfig()
    lay = layout_mod.sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    dm = dock_distance_matrix(g)
    w = SimWorld(inst, lay, dm, cfg, graph=g)
    ctx = norm_context(inst, lay, 75.87, cfg)
    snap = w.snapshot()
    plans = {j: [None] * len(ops) for j, ops in enumerate(inst.jobs)}
    pa_mct = [0.0] * inst.n_machines
    pa_assigned = [0.0] * inst.n_machines
    view = build_pa_view(snap, inst, plans, pa_mct, pa_assigned, lay, w.leg_min)
    pool = [(j, 0) for j in range(inst.n_jobs)]
    return inst, lay, w, ctx, snap, view, pool


@pytest.mark.unit
def test_k_cut_is_strict_and_unbiased_at_the_boundary():
    """边界平局组内均匀抽：10 行同分、K=5 ⟹ 每行被选中约一半，且与行号无关。"""
    rows = [(0.0, i) for i in range(10)] + [(1.0, i) for i in range(10, 15)]
    counts = np.zeros(15)
    draws = 400
    for seed in range(draws):
        rng = np.random.default_rng([seed, 0])
        kept = _cut_with_tie_groups(rows, 5, lambda r: r[0], lambda r: r[1], rng)
        assert len(kept) == 5
        for r in kept:
            assert r[0] == 0.0, "K=5 时不该越过整个同分组"
            counts[r[1]] += 1
    freq = counts[:10] / draws
    assert np.all(freq > 0.30) and np.all(freq < 0.70), freq
    assert counts[10:].sum() == 0, "同分组装满 K 后不该再取更差的行"


@pytest.mark.unit
def test_k_cut_keeps_distinct_scores_in_order():
    rows = [(float(i), i) for i in range(10)]
    rng = np.random.default_rng([0, 0])
    kept = _cut_with_tie_groups(rows, 4, lambda r: r[0], lambda r: r[1], rng)
    assert [r[0] for r in kept] == [0.0, 1.0, 2.0, 3.0]


@pytest.mark.unit
def test_pa_candidates_are_strict_k_and_first_is_the_rule_minimum():
    inst, _lay, _w, ctx, _snap, view, pool = _fixture()
    for rule in PA_RULE_NAMES:
        rng = topk_rng(0, "PA", 0)
        cands = pa_candidates(view, pool, rule, k=K_PA, m_ref=ctx.m_ref, rng=rng)
        assert 1 <= len(cands) <= K_PA, rule
        idx = PA_RULE_NAMES.index(rule)
        jr = PA_JOB_RULES[idx // len(PA_MACHINE_RULES)]
        mr = PA_MACHINE_RULES[idx % len(PA_MACHINE_RULES)]
        best = min(
            (float(pa_job_score(jr, view, j)), float(pa_machine_score(
                mr, view, (int(m), float(t)), (0.0,))))
            for j, _oi in pool for m, t in view.jobs[j].alts)
        first = cands[0]
        got = (float(pa_job_score(jr, view, first.job)),
               float(pa_machine_score(mr, view,
                                      (int(first.mach), _t_of(view, first)),
                                      (0.0,))))
        assert got == best, (rule, got, best)


def _t_of(view, cand) -> float:
    for m, t in view.jobs[cand.job].alts:
        if int(m) == int(cand.mach):
            return float(t)
    raise AssertionError("候选机台不在工件 alts 里")


@pytest.mark.unit
def test_pa_candidates_include_everything_when_legal_pairs_fit():
    """合法 `(工件, 机台)` 对 <= K 时，候选集 = 全部对（不重不漏）。"""
    inst, _lay, _w, ctx, _snap, view, pool = _fixture()
    n_pairs = sum(len(view.jobs[j].alts) for j, _oi in pool)
    if n_pairs > K_PA:
        pytest.skip(f"本夹具合法对数 {n_pairs} > K")
    for rule in PA_RULE_NAMES:
        cands = pa_candidates(view, pool, rule, k=K_PA, m_ref=ctx.m_ref,
                              rng=topk_rng(0, "PA", 0))
        got = {(c.job, c.oi, c.mach) for c in cands}
        want = {(j, oi, int(m)) for j, oi in pool for m, _t in view.jobs[j].alts}
        assert got == want, rule


@pytest.mark.unit
def test_la_candidates_cover_all_agvs_when_k_is_large_enough():
    _inst, lay, w, ctx, snap, _view, _pool = _fixture()
    cand_v = list(range(len(snap.vehicles)))
    if len(cand_v) > K_LA:
        pytest.skip("本夹具车数 > K")
    lu = int(lay.grid.lu_dock_node)
    vehicles = tuple(LAVehicle(node=int(v.node), busy_min=0.0, free_at=0.0)
                     for v in snap.vehicles)
    view = LAView(now=0.0, job=0, oi=0,
                  frm_node=lu, to_node=int(lay.machines[0].dock_node),
                  ready_at=0.0,
                  vehicles=vehicles, leg_min=w.leg_min)
    for rule in LA_RULE_NAMES:
        cands = la_candidates(view, rule, cand_v, k=K_LA, m_ref=ctx.m_ref,
                              rng=topk_rng(0, "LA", 0))
        assert {c.agv for c in cands} == set(cand_v), rule


@pytest.mark.unit
def test_ra_rules_filter_the_action_set():
    """`AGE` 恒两动作；`QUEUE_EMPTY` 队列非空只留 DEFER；`OLDEST` 非最旧只留 DEFER。"""
    _inst, _lay, _w, _ctx, snap, _view, _pool = _fixture()
    m0 = dataclasses.replace(snap.machines[0], in_q_len=1, pm_used_min=0.0)
    m1 = dataclasses.replace(snap.machines[1], in_q_len=0, pm_used_min=5.0)
    snap2 = dataclasses.replace(snap, machines=(m0, m1) + tuple(snap.machines[2:]))
    # AGE：永远两动作
    allowed, mask = ra_candidates(snap2, "AGE", 0, (PM_CAND_NOW, PM_CAND_DEFER))
    assert allowed == [PM_CAND_NOW, PM_CAND_DEFER] and mask.all()
    # QUEUE_EMPTY：m0 队列非空 ⟹ 只有 DEFER
    allowed, mask = ra_candidates(snap2, "QUEUE_EMPTY", 0, (PM_CAND_NOW, PM_CAND_DEFER))
    assert allowed == [PM_CAND_DEFER] and (not mask[0] and mask[1])
    # OLDEST：m0 不是最旧 ⟹ 只有 DEFER；m1 是最旧 ⟹ 两动作
    allowed, _ = ra_candidates(snap2, "OLDEST", 0, (PM_CAND_NOW, PM_CAND_DEFER))
    assert allowed == [PM_CAND_DEFER]
    allowed, _ = ra_candidates(snap2, "OLDEST", 1, (PM_CAND_NOW, PM_CAND_DEFER))
    assert allowed == [PM_CAND_NOW, PM_CAND_DEFER]


@pytest.mark.unit
def test_ra_features_shape_and_finiteness():
    _inst, _lay, _w, ctx, snap, _view, _pool = _fixture()
    f = ra_features(snap, 0, ctx)
    assert f.shape == (5,)
    assert np.all(np.isfinite(f))


@pytest.mark.unit
def test_topk_rng_is_deterministic_and_kind_separated():
    a = topk_rng(7, "PA", 3).random()
    b = topk_rng(7, "PA", 3).random()
    c = topk_rng(7, "LA", 3).random()
    assert a == b
    assert a != c
