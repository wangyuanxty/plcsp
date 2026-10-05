"""信息侧约束（⑧ 交期 TF/RDD）的测试。

> **⑥ 模糊加工已砍**（2026-10-03，用户指令）：模糊加工在本平台上实测**关态与极端参数下都不 binding**
> （关态 +1.83%/+0.98%，极端参数 +0.51%/+3.32%，均不过噪声），且与 ① 拥堵同属"重复表达不确定性"
> 一类。实现（三角模糊数）已整体移除，不再有死开关。原 binding 读数保留在 `spec §3.3.3`。

**⑧ 交期口径**（2026-10-03 重设计，见 `plcsp/env/due_dates.py`）：**逐作业**、**外生**，
`d_j = LB·τ·(1 + R·(2ρ_j − 1))`（TWK 的作业工时 + TF/RDD 参数）。旧口径 `d_j = τ·M_ref`
（共同交期、锚在参考调度上）已废：它内生（换车队规模就换交期），且 τ=0.90 只对**参考策略**
标定，训练后策略改进 12–14% 就把 TWT 清零（旧 `due_factor` 更是实测 MK01 tardy 恒为 10/10）。
"""
from __future__ import annotations

import pytest

from plcsp.env.constraints import ConstraintConfig
from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import Instance, load_mk

SEEDS = (0, 1, 2)


@pytest.mark.unit
def test_due_dates_use_tf_rdd_formula():
    """⑧ 交期 = `LB·τ·(1 + R·(2ρ_j − 1))` 的**逐作业具体值**（不再收 n_jobs/m_ref）。

    手搓实例使 W_j / LB / ρ 都可手算：
    各作业最短工时 = [1, 2, 3, 4] ⟹ ΣW = 10、m = 2 ⟹ **LB = max(10/2, 4) = 5**；
    升序排名 ρ = [0, 1/3, 2/3, 1]。取 τ=2.0、R=0.5 ⟹
    `d_j = 5·2·(1 + 0.5·(2ρ_j − 1))` = **[5, 25/3, 35/3, 15]**——四值**互不相同**。
    """
    from plcsp.env.des import compute_due_dates

    inst = Instance(n_jobs=4, n_machines=2,
                    jobs=[[[(0, 1.0)]], [[(0, 2.0)]], [[(0, 3.0)]], [[(0, 4.0)]]],
                    source="handmade")
    due = compute_due_dates(inst, tau=2.0, due_range=0.5)
    assert due == pytest.approx({0: 5.0, 1: 25.0 / 3, 2: 35.0 / 3, 3: 15.0})
    assert len(set(round(v, 6) for v in due.values())) == 4, "退化成共同交期了"


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
    # ⑧ 重设计后交期多了一个覆盖字段：两者都只进交期、都不进参考运行的缓存键（Review Focus #2）
    assert reference_makespan(inst, SimConfig(due_range=0.50)) == m3


@pytest.mark.unit
@pytest.mark.parametrize("name", ["mk01", "mk07", "mk10"])
def test_due_dates_are_binding_not_degenerate(name):
    """⑧ 必须**有信号**：不能 0% 误期（目标恒零）也不能 100% 误期（无区分度）。

    实测（⑧ 重设计后的 TF/RDD 口径，种子 0/1/2）：mk01 50%/50%/60%，mk07 45%/50%/45%，
    mk10 45%/60%/60%——三者都非退化，且**逐实例标定**后不再有"单一 τ 迁就所有实例"的问题
    （旧口径 τ=0.90 的 20%/28%/48% 是拿参考 makespan 当锚测的，训练后即失效）。
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
