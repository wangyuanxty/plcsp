"""AGV 行驶的测试（P1a Task 3，2026-10-02 改版）。

背景：区段管制（ZoneManager）与路线决策已整条砍除——实测通道争用 ≤0.07%（见 progress-log §十五）。
AGV 现在按**格点距离矩阵**一次行驶到底，物流的结构性瓶颈是**车辆数量**而非通道容量。
"""
from __future__ import annotations

import pytest

from plcsp.env.constraints import ConstraintConfig
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
def test_travel_time_is_a_bounded_fraction_of_makespan():
    """运输占 makespan 的比例必须有界（量纲 sanity）。**下界与上界都是本批实测定的**。

    ⚠️ **上界从 0.20 放宽到 0.50（P4-B Task 2b，实测记录）**：作业改为在**装卸站**入场、
    完工**回站**后，每个作业多两条 LU 负载段，运输不再只是"机台间的小头"。
    实测 mk01（n_agv=4，几何口径）：**travel/makespan = 0.314**（改造前 ≈ 0.09）——
    旧上界 0.20 被**真实结构变化**作废，不是被调参作废。放宽后的界仍能挡住量纲错误
    （若行程再被除以 60，frac 会掉到 0.006 以下；若单位反过来则远超 1）。
    下界 0.05 挡的是"运输根本没接进仿真"。⚠️ 论文里"运输不在关键路径"这句（spec §3.3.3）
    必须按**矩阵口径**重测后再写——本批不得沿用旧结论。
    """
    inst = load_mk("mk01")
    r = rollout(inst, seed_chain=1, cfg=SimConfig(n_agv=4))
    frac = r["travel_time_total"] / r["makespan"]
    assert 0.05 < frac < 0.50, f"运输/makespan = {frac:.3f}，超出实测区间 (0.05, 0.50)"


@pytest.mark.unit
def test_travel_time_equals_distance_matrix_exactly():
    """空载段 + 负载段必须**逐位等于** Σ 距离矩阵最短路，且**分账正确**（Review Focus ④）。

    这是防"索引用了机台号而非通道节点号"的唯一有效断言——旧版只断言上界，
    实测余量 3.7 倍，把 `dock_node` 换成机台号**照样通过**（P1a 评审 I3）。

    P1b Task 2 加了空载段：n_agv=1 时任务按 `task_flow` 生成序 FIFO 执行，
    故第 k 条任务的空载段 = 第 k−1 条任务的卸货点到第 k 条的取货点（首条无空载段）。
    这里同时校验**总额**与**空载/负载分账**——只对总额会让"两段记反"蒙混过关。

    ⚠️ P4-B Task 2b：任务端点多了**装卸站**（号 = `inst.n_machines`）——首/末条任务的两端
    就是站。故端点→通道节点要走 `_dock_node`（站取 `lay.lu.node`，其位置由连接段接入）。
    """
    # Arrange
    # ⚠️ 必须在**单载**下测：⑩ 多载量开启时一趟送多件，`travel_time_total` 会**小于**
    #    Σ 每件的负载段（实测少 ~2.4 min），精确等式不成立。这条断言守的是"节点索引没错"，
    #    与拼车无关，故把异构车队关掉（载量退化为 1）后再对拍。
    inst = load_mk("mk01")
    lu_idx = inst.n_machines
    cfg = SimConfig(n_agv=1)
    lay = sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)   # 同 rollout 默认 seed_layout
    dm = dock_distance_matrix(build_corridor_graph(lay))

    # Act
    r = rollout(inst, seed_chain=1, cfg=cfg,
                constraints=ConstraintConfig(heterogeneous_fleet=False))

    # Assert：对每条搬运任务逐条重算两段，求和后必须逐位相等
    def _minutes(u: int, v: int) -> float:
        return float(dm[u, v]) / cfg.eff_speed / cfg.agv_speed_mps / 60.0

    def _dock(idx: int) -> int:
        return lay.lu.node if idx == lu_idx else lay.machines[idx].dock_node

    loaded, empty, prev_dock = 0.0, 0.0, None
    for _job, _oi, frm_m, to_m in r["task_flow"]:
        a, b = _dock(frm_m), _dock(to_m)
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
