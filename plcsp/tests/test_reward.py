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
    assert ref.matches(inst) and not ref.matches(load_mk("mk07"))


@pytest.mark.unit
def test_end_to_end_reward_on_real_rollout():
    inst = load_mk("mk01")
    ref = ReferenceObjectives.of(inst, SimConfig())
    w = reward_weights(ref.as_tuple())
    r = rollout(inst, seed_chain=1, cfg=SimConfig())
    got = scalar_reward(objective_vector(r), w)
    assert got < 0.0, "奖励应为负（三项都是'越小越好'）"
