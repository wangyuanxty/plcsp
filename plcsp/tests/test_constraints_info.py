"""信息侧约束（⑧ 交期 τ）的测试（P1b Task 6）。

> **⑥ 模糊加工已砍**（2026-10-03，用户指令）：模糊加工在本平台上实测**两档都不 binding**
> （关态 +1.83%/+0.98%，极端档 +0.51%/+3.32%，均不过噪声），且与 ① 拥堵同属"重复表达不确定性"
> 一类。实现（三角模糊数）已整体移除，不再有死开关。原 binding 读数保留在 `spec §3.3.3`。

**⑧ 交期口径**：`d_j = τ · M_ref`（spec §3.5，**τ=0.90**，P1b 后重标），`M_ref` = **参考调度**的
makespan（每工序取最短候选 + AGV 轮询派车）。**M_ref 不能取被优化的那次 episode 的 makespan**——
否则交期随策略一起漂移，目标退化（旧 `due_factor` 就是这么坏的：
实测 MK01 tardy 恒为 10/10，见 progress-log）。
"""
from __future__ import annotations

import pytest

from plcsp.env.constraints import ConstraintConfig
from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import load_mk

SEEDS = (0, 1, 2)
TAU = 0.90


@pytest.mark.unit
def test_due_dates_use_tau_times_m_ref():
    """⑧ 交期 = τ·M_ref（同一实例所有作业同值），不再是旧的 due_factor×工时。"""
    from plcsp.env.des import compute_due_dates

    due = compute_due_dates(n_jobs=10, tau=TAU, m_ref=400.0)
    assert len(due) == 10
    assert all(d == pytest.approx(360.0) for d in due.values())


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

    同实例同 seed_layout 多次调用必须同值——否则交期跟着策略跑，目标退化（旧 `due_factor` 的病）。
    """
    from plcsp.env.des import reference_makespan

    inst = load_mk("mk01")
    m1 = reference_makespan(inst, SimConfig())
    m2 = reference_makespan(inst, SimConfig())
    assert m1 == m2 and m1 > 0.0


@pytest.mark.unit
def test_reference_cache_key_covers_dynamics_cfg():
    """⚠️ 缓存键必须含**影响参考运行的 cfg 字段**：同实例换车队规模不得命中同一份 M_ref。

    `_instance_key` 只看实例内容，而 `n_agv` 等 cfg 一变参考调度结果就变（实测 mk01：
    n_agv=3 → 103.42，n_agv=1 → 109.95）——漏进键会**静默**拿到错的 M_ref 与 f^ref，
    奖励权重随之算错（错得很安静）。
    `tau` 相反：只进 `compute_due_dates`（交期是 metric、参考运行内被短路），实测换 τ
    参考 makespan **逐位不变**——故不进键，免得 τ 扫描反复重跑同一份参考运行。
    """
    from plcsp.env.des import reference_makespan

    inst = load_mk("mk01")
    m3 = reference_makespan(inst, SimConfig(n_agv=3))
    m1 = reference_makespan(inst, SimConfig(n_agv=1))
    assert m1 != m3, "换 cfg 命中了同一份 M_ref——缓存键漏了 cfg 字段"
    assert reference_makespan(inst, SimConfig(n_agv=3)) == m3    # 同 cfg 仍命中缓存
    assert reference_makespan(inst, SimConfig(n_agv=1)) == m1
    assert reference_makespan(inst, SimConfig(tau=0.50)) == m3   # τ 与参考运行无关


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
    assert r["tardy_twt"] >= 0.0, "TWT 口径是假的"


@pytest.mark.unit
def test_due_dates_switch_is_not_a_dead_flag():
    """Review Focus #2：⑧ 关掉必须**真的生效**——目标里的拖期项要么消失要么变。

    ⑧ 不影响仿真（不改变任何事件时序），只影响**目标读数**，故判据是 TWT 归零。
    """
    inst = load_mk("mk01")
    on = rollout(inst, seed_chain=1, cfg=SimConfig(), constraints=ConstraintConfig())
    off = rollout(inst, seed_chain=1, cfg=SimConfig(),
                  constraints=ConstraintConfig(due_dates=False))
    assert on["tardy_twt"] > 0.0, "开启时应有非零拖期，否则测试空过"
    assert off["tardy_twt"] == 0.0 and off["tardy"] == 0
    assert off["makespan"] == on["makespan"], "⑧ 不该影响仿真时序（只影响目标）"
