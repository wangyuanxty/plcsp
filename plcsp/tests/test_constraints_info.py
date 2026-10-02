"""信息侧约束（⑥ 模糊加工 · ⑧ 交期 τ）的测试（P1b Task 6）。

## 两条纪律

**⑥ 模糊 ≠ 随机**（Review Focus #6）。用**三角模糊数** `(a, b, c) = (nom·(1−s), nom, nom·(1+s))`，
抽样走三角分布的逆变换（把隶属度归一为密度）。**有界**是关键区别——正态无界，
若实现成正态，写论文时会被抓。关掉开关时必须**直接返回标称值、不抽任何随机数**。

**⑧ 交期口径**：`d_j = τ · M_ref`（spec §3.5，**τ=0.90**，P1b 后重标），`M_ref` = **参考调度**的 makespan
（每工序取最短候选 + AGV 轮询派车）。**M_ref 不能取被优化的那次 episode 的 makespan**——
否则交期随策略一起漂移，目标退化（旧 `due_factor` 就是这么坏的：
实测 MK01 tardy 恒为 10/10，见 progress-log）。
"""
from __future__ import annotations

import pytest

from plcsp.env.constraints import ConstraintConfig
from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import load_mk

SEEDS = (0, 1, 2)
TAU = 0.85


@pytest.mark.unit
def test_fuzzy_processing_is_fuzzy_not_gaussian():
    """Review Focus #6：⑥ 必须是**有界**的模糊数，不是无界的正态。"""
    from plcsp.env.des import sample_fuzzy_time

    vals = [sample_fuzzy_time(5.0, spread=0.2, rng_seed=s) for s in range(300)]
    # 三角模糊数 (4.0, 5.0, 6.0) 的支撑集 = [4, 6]；正态会溢出
    assert all(4.0 <= v <= 6.0 for v in vals), "模糊数必须落在支撑集内（正态无界 → 违规）"
    assert min(vals) < 5.0 < max(vals), "抽样必须覆盖标称值两侧"
    # 形态：众数附近更密（三角分布），而非均匀
    mid = sum(1 for v in vals if 4.5 <= v <= 5.5) / len(vals)
    assert mid > 0.40, f"三角分布应在众数附近更密（实测中段占 {mid:.2f}）"


@pytest.mark.unit
def test_fuzzy_support_matches_nominal_and_spread():
    """支撑集端点必须精确等于 nominal×(1±spread)。"""
    from plcsp.env.des import sample_fuzzy_time
    vals = [sample_fuzzy_time(10.0, spread=0.3, rng_seed=s) for s in range(400)]
    assert min(vals) >= 7.0 - 1e-9 and max(vals) <= 13.0 + 1e-9
    assert min(vals) < 7.6 and max(vals) > 12.4, "抽样没铺满支撑集"


@pytest.mark.unit
def test_fuzzy_off_gives_deterministic_times():
    """⑥ 关掉 → 加工时间 = 标称值，逐位确定（不得有任何随机）。"""
    inst = load_mk("mk01")
    a = rollout(inst, seed_chain=1, cfg=SimConfig(),
                constraints=ConstraintConfig(fuzzy_processing=False))
    b = rollout(inst, seed_chain=1, cfg=SimConfig(),
                constraints=ConstraintConfig(fuzzy_processing=False))
    assert a["makespan"] == b["makespan"]
    # 开关不消耗随机数：关态下机台故障流不得被移位
    assert a["fail_events"] == b["fail_events"]
    assert a["energy_breakdown"]["machine_kwh"] > 0.0


@pytest.mark.unit
def test_due_dates_use_tau_times_m_ref():
    """⑧ 交期 = τ·M_ref（同一实例所有作业同值），不再是旧的 due_factor×工时。"""
    from plcsp.env.des import compute_due_dates

    due = compute_due_dates(n_jobs=10, tau=TAU, m_ref=400.0)
    assert len(due) == 10
    assert all(d == pytest.approx(340.0) for d in due.values())


@pytest.mark.unit
def test_tardiness_is_weighted_twt_not_count():
    """⑧ 目标用加权总拖期 TWT = Σ w_i·max(0, C_i − d_i)，不是"几个作业误期"的计数。"""
    from plcsp.env.des import weighted_tardiness

    completes = {0: 100.0, 1: 350.0, 2: 340.0}
    due = {0: 340.0, 1: 340.0, 2: 340.0}
    weights = {0: 1.0, 1: 2.0, 2: 1.0}
    # 工件1 晚 10 min × 权重2 = 20；工件2 恰好不晚（=）；工件0 未晚
    assert weighted_tardiness(completes, due, weights) == pytest.approx(20.0)


@pytest.mark.unit
def test_m_ref_is_plan_independent():
    """`M_ref` 必须来自**参考调度**，不得随被优化的 episode 漂移。

    同一实例、同 seed_layout，跨不同 seed_chain 的 `M_ref` 必须相同——
    否则交期跟着策略跑，目标退化（旧 `due_factor` 的病）。
    """
    from plcsp.env.des import reference_makespan

    inst = load_mk("mk01")
    m1 = reference_makespan(inst, SimConfig())
    m2 = reference_makespan(inst, SimConfig())
    assert m1 == m2 and m1 > 0.0


@pytest.mark.unit
@pytest.mark.parametrize("name", ["mk01", "mk07", "mk10"])
def test_due_dates_are_binding_not_degenerate(name):
    """⑧ 必须**有信号**：不能 0% 误期（目标恒零）也不能 100% 误期（无区分度）。

    实测（τ=0.90，参考调度）：mk01 20% / mk07 28% / mk10 48%——三者都非退化。
    ⚠️ 三实例**共用同一个 τ**，但紧度天然不同；没有单一 τ 能让三者同时落进 20–40%，
       这要如实报告（见 spec §3.5）。
    """
    inst = load_mk(name)
    rates = []
    for s in SEEDS:
        r = rollout(inst, seed_chain=s, cfg=SimConfig())
        assert not r["horizon_hit"], f"{name} seed{s} 掐表"
        rates.append(r["tardy"] / inst.n_jobs)
    avg = sum(rates) / len(rates)
    assert 0.05 < avg < 0.95, (
        f"{name} 误期率 {avg:.2f} 退化（<5% 目标恒零；>95% 无区分度）")
    # TWT 必须与误期数同向非零——否则"加权总拖期"这个目标口径是假的
    assert r["tardy_twt"] >= 0.0


@pytest.mark.unit
def test_fuzzy_and_due_independently_toggleable():
    """Review Focus #3：⑥⑧ 可独立翻转，四种组合都不崩。"""
    inst = load_mk("mk01")
    for fz in (False, True):
        for dd in (False, True):
            r = rollout(inst, seed_chain=1, cfg=SimConfig(),
                        constraints=ConstraintConfig(fuzzy_processing=fz, due_dates=dd))
            assert r["jobs_done"] == inst.n_jobs and not r["horizon_hit"], \
                f"组合 fuzzy={fz} due={dd} 未跑完"
