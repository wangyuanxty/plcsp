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
def test_travel_time_equals_distance_matrix_exactly():
    """空载段 + 负载段必须**逐位等于** Σ 距离矩阵最短路，且**分账正确**（Review Focus ④）。

    这是防"索引用了机台号而非通道节点号"的唯一有效断言——旧版只断言上界，
    实测余量 3.7 倍，把 `dock_node` 换成机台号**照样通过**（P1a 评审 I3）。

    P1b Task 2 加了空载段：n_agv=1 时任务按 `task_flow` 生成序 FIFO 执行，
    故第 k 条任务的空载段 = 第 k−1 条任务的卸货点到第 k 条的取货点（首条无空载段）。
    这里同时校验**总额**与**空载/负载分账**——只对总额会让"两段记反"蒙混过关。
    """
    # Arrange
    inst = load_mk("mk01")
    cfg = SimConfig(n_agv=1)
    lay = sample_layout(inst.n_machines, seed=0)          # 与 rollout 默认 seed_layout=0 一致
    dm = dock_distance_matrix(build_corridor_graph(lay))

    # Act
    r = rollout(inst, seed_chain=1, cfg=cfg)

    # Assert：对每条搬运任务逐条重算两段，求和后必须逐位相等
    def _minutes(u: int, v: int) -> float:
        return float(dm[u, v]) / cfg.eff_speed / cfg.agv_speed_mps / 60.0

    loaded, empty, prev_dock = 0.0, 0.0, None
    for _job, _oi, frm_m, to_m in r["task_flow"]:
        a = lay.machines[frm_m].dock_node
        b = lay.machines[to_m].dock_node
        loaded += _minutes(a, b)
        if prev_dock is not None:
            empty += _minutes(prev_dock, a)
        prev_dock = b
    assert len(r["task_flow"]) > 0, "本 episode 应有搬运任务"
    assert r["travel_time_total"] == pytest.approx(loaded + empty, rel=1e-9)
    assert r["energy_breakdown"]["agv_states_min"]["loaded"] == pytest.approx(loaded, rel=1e-9)
    assert r["energy_breakdown"]["agv_states_min"]["empty"] == pytest.approx(empty, rel=1e-9)
    assert empty > 0.0, "空载段未接入（v0 的 AGV 瞬移取货问题复发）"


@pytest.mark.unit
def test_narrow_aisle_slows_travel_down():
    """窄通道必须真的降速（`eff_speed` 读 `cfg.aisle_width`）——否则窄道敏感性实验静默失效。

    净效果 = 几何缩短（段长 cell_w+aisle_w 变小）× 速度倍率（0.4–1.0），故约 2.0×。
    """
    inst = load_mk("mk01")
    wide = rollout(inst, seed_chain=1, aisle_width=1.5)   # 倍率 1.0
    narrow = rollout(inst, seed_chain=1, aisle_width=0.6)  # 倍率 0.4（下限）
    assert narrow["travel_time_total"] > wide["travel_time_total"] * 1.8, \
        "窄道未降速 → aisle_width 没进 SimConfig"


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
