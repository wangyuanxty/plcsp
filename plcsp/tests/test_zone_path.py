"""路径级逐段 zone 申请的测试（P1a Task 3）。

对应 spec §3.2：zone = 通道节点；AGV 沿**真实最短路**逐段推进
（持当前 → 申请下一 → 成功后释放上一），取代旧的 `frm % n / to % n` 伪造分配。
"""
from __future__ import annotations

import pytest

from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import load_mk


@pytest.mark.unit
def test_episode_completes_on_grid():
    """网格上跑通：全部作业完成、无掐表。"""
    inst = load_mk("mk01")
    r = rollout(inst, seed_chain=1, cfg=SimConfig(n_agv=4))
    assert r["jobs_done"] == inst.n_jobs
    assert r["horizon_hit"] is False


@pytest.mark.unit
def test_no_zone_leak_after_episode():
    """Review Focus #3：episode 结束后不得有 zone 仍被持有。"""
    inst = load_mk("mk01")
    r = rollout(inst, seed_chain=1, cfg=SimConfig(n_agv=4))
    assert r["zone_holders_free"] is True, "有 zone 未释放 → 下一轮 rollout 会卡死"


@pytest.mark.unit
def test_zone_wait_is_reported_and_nonzero_under_contention():
    """Review Focus #2：多车时应有真实的区段等待（路径级争用可见）。"""
    inst = load_mk("mk01")
    r = rollout(inst, seed_chain=1, cfg=SimConfig(n_agv=4))
    zw = r["zone_wait"]
    assert zw["n"] > 0, "路径级 zone 申请应产生等待记录"
    assert zw["total"] > 0.0


@pytest.mark.unit
def test_single_agv_has_no_zone_contention():
    """单车不可能与自己争用 → 等待次数应为 0（交叉口一次只被一车申请）。"""
    inst = load_mk("mk01")
    r = rollout(inst, seed_chain=1, cfg=SimConfig(n_agv=1))
    assert r["zone_wait"]["n"] == 0


@pytest.mark.unit
def test_route_phi_is_a_decision_variable():
    """**路线是决策变量**（spec §1 边界声明 / §5.2）：传入 route_phi 全选第 2 条候选，
    总行驶时间不得短于全选最短路（即路线确实被消费了，不是摆设）。"""
    # Arrange
    inst = load_mk("mk07")
    cfg = SimConfig(n_agv=4)
    r_short = rollout(inst, seed_chain=1, cfg=cfg)          # 缺省 = 最短路
    n_tasks = int(r_short["dbg"].get("trans_evt", 0))

    # Act
    r_long = rollout(inst, seed_chain=1, cfg=cfg, route_phi=[1] * n_tasks)

    # Assert
    assert r_short["jobs_done"] == inst.n_jobs and r_long["jobs_done"] == inst.n_jobs
    assert r_long["travel_time_total"] >= r_short["travel_time_total"], \
        "选了更长候选路径却行驶更少 → route_phi 没被消费"
    assert r_short["travel_time_total"] > 0.0
