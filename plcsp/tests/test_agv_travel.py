"""AGV 行驶的测试（P1a Task 3，2026-10-02 改版）。

背景：区段管制（ZoneManager）与路线决策已整条砍除——实测通道争用 ≤0.07%（见 progress-log §十五）。
AGV 现在按**格点距离矩阵**一次行驶到底，物流的结构性瓶颈是**车辆数量**而非通道容量。
"""
from __future__ import annotations

import pytest

from plcsp.env.corridors import build_corridor_graph, dock_distance_matrix
from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import load_mk
from plcsp.env.layout import sample_layout


@pytest.mark.unit
def test_episode_completes_on_grid():
    """网格上跑通：全部作业完成、无掐表。"""
    inst = load_mk("mk01")
    r = rollout(inst, seed_chain=1, cfg=SimConfig(n_agv=4))
    assert r["jobs_done"] == inst.n_jobs
    assert r["horizon_hit"] is False


@pytest.mark.unit
def test_travel_time_is_small_fraction_of_makespan():
    """运输应占 makespan 的**小头**（bug#13 量纲对齐后应成立；若运输占大头说明量纲又歪了）。"""
    inst = load_mk("mk01")
    r = rollout(inst, seed_chain=1, cfg=SimConfig(n_agv=4))
    frac = r["travel_time_total"] / r["makespan"]
    assert 0.0 < frac < 0.20, f"运输/makespan = {frac:.3f}，超出预期区间 (0, 0.20)"


@pytest.mark.unit
def test_travel_time_matches_distance_matrix():
    """单 AGV、单车任务时，行驶时长应等于格点最短路 ÷ 车速（单位换算见 bug#13）。"""
    # Arrange
    inst = load_mk("mk01")
    cfg = SimConfig(n_agv=1)
    lay = sample_layout(inst.n_machines, seed=0)
    dm = dock_distance_matrix(build_corridor_graph(lay))

    # Act
    r = rollout(inst, seed_chain=1, cfg=cfg)
    n_transports = int(r["dbg"].get("trans_evt", 0))

    # Assert：总行驶时间不应超过"每条任务走全网格最远两点"的上界
    far = float(dm[dm < float("inf")].max()) / cfg.eff_speed / cfg.agv_speed_mps / 60.0
    assert r["travel_time_total"] <= far * n_transports + 1e-9


@pytest.mark.unit
def test_more_agvs_do_not_increase_makespan_beyond_noise():
    """车队数不应让 makespan 大幅变差（无区段争用后，"加车更慢"的机制已不成立）。"""
    inst = load_mk("mk01")
    ms = {}
    for na in (2, 4, 8):
        vals = [rollout(inst, seed_chain=s, cfg=SimConfig(n_agv=na))["makespan"]
                for s in range(5)]
        ms[na] = sum(vals) / len(vals)
    worst = max(ms.values())
    best = min(ms.values())
    assert (worst - best) / best < 0.10, f"车队数对 makespan 影响过大：{ms}"
