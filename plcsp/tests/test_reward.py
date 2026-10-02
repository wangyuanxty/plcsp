"""三目标奖励的测试（P2 Task 6，spec §4.2）。"""
from __future__ import annotations

import pytest

from plcsp.env.reward import (ReferenceObjectives, objective_vector, reward_weights,
                              scalar_reward)
from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import load_mk


@pytest.mark.unit
def test_objective_vector_reads_three_objectives():
    r = {"makespan": 100.0, "energy": 8.0, "tardy_twt": 12.5}
    assert objective_vector(r) == (100.0, 8.0, 12.5)


@pytest.mark.unit
def test_weights_are_inverse_reference_and_sum_to_one():
    w = reward_weights((100.0, 8.0, 20.0))
    assert sum(w) == pytest.approx(1.0)
    # 参考值越小 → 权重越大（"相对参考各改进一个单位，贡献相同"）
    assert w[1] > w[2] > w[0]


@pytest.mark.unit
def test_weights_reject_zero_reference():
    """参考值为 0（如 TWT=0 的实例）不得产生 inf 权重——必须显式报错或兜底。"""
    with pytest.raises(ValueError):
        reward_weights((100.0, 8.0, 0.0))


@pytest.mark.unit
def test_scalar_reward_is_monotone_in_each_objective():
    """三个目标各自变小（更好）时，奖励必须上升——这是"多目标"最容易被写反的地方。"""
    w = reward_weights((100.0, 8.0, 20.0))
    base = scalar_reward((100.0, 8.0, 20.0), w)
    assert scalar_reward((90.0, 8.0, 20.0), w) > base      # makespan ↓
    assert scalar_reward((100.0, 7.0, 20.0), w) > base     # energy ↓
    assert scalar_reward((100.0, 8.0, 10.0), w) > base     # TWT ↓


@pytest.mark.unit
def test_reference_objectives_cached_and_consistent_with_m_ref():
    """f^ref 与 M_ref 必须取自**同一次**参考运行（口径一致）。"""
    from plcsp.env.des import reference_makespan
    inst = load_mk("mk01")
    ref = ReferenceObjectives.of(inst, SimConfig())
    assert ref.makespan == pytest.approx(reference_makespan(inst, SimConfig()), rel=1e-9)
    # ⚠️ 三项都须为正的**实测值**——参考运行内交期被短路，TWT 若直接取 dict 恒为 0，
    #    这里会红（0 不是"实测值"，且会让 `reward_weights` 显式报错）。
    assert ref.energy > 0.0 and ref.twt > 0.0
    # 实例指纹：奖励绝对值只在同一实例内有意义，跨实例混用须能被挡住
    assert ref.matches(inst, SimConfig()) and not ref.matches(load_mk("mk07"), SimConfig())


@pytest.mark.unit
def test_reference_objectives_fingerprint_covers_cfg():
    """⚠️ 评审 I-1：f^ref（进而 w）随 **cfg** 变——同实例不同 cfg 的参考值不得互相 `matches`。

    失败场景：P4 扫 `n_agv`（1/3/5）时沿用默认 cfg 算出的 w；指纹若只挡实例，M_ref 已变
    （实测 mk01：n_agv=1/3/5 → 109.95/103.42/97.24；车速 0.5/1.0 → 103.42/106.38）而 w 未变
    ——"按参考调度归一化"在这条扫描轴上**静默**不成立、零报错。τ 同理：TWT 分量按
    `d_j = τ·M_ref` 事后算，τ 一变 TWT（进而 w）跟着变。
    """
    inst = load_mk("mk01")
    ref3 = ReferenceObjectives.of(inst, SimConfig(n_agv=3))
    ref5 = ReferenceObjectives.of(inst, SimConfig(n_agv=5))
    assert ref3.as_tuple() != ref5.as_tuple(), "两 cfg 的 f^ref 实测竟然相同——本判据失去意义"
    assert ref3.matches(inst, SimConfig(n_agv=3)) and ref5.matches(inst, SimConfig(n_agv=5))
    assert not ref3.matches(inst, SimConfig(n_agv=5)), "换了车队规模旧参考值仍 matches——指纹漏 cfg"
    assert not ref5.matches(inst, SimConfig(n_agv=3))
    assert not ref3.matches(inst, SimConfig(n_agv=3, tau=0.50)), "τ 变了仍 matches——指纹漏 τ"
    # `None` = 默认 cfg（与 `of` / `reference_run` 的 None 语义一致），**不是**"跳过 cfg 检查"
    assert ref3.matches(inst, SimConfig()) and not ref5.matches(inst, None)


@pytest.mark.unit
def test_end_to_end_reward_on_real_rollout():
    inst = load_mk("mk01")
    ref = ReferenceObjectives.of(inst, SimConfig())
    w = reward_weights(ref.as_tuple())
    r = rollout(inst, seed_chain=1, cfg=SimConfig())
    got = scalar_reward(objective_vector(r), w)
    assert got < 0.0, "奖励应为负（三项都是'越小越好'）"
