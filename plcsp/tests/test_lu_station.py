"""装卸站（LU）接入的测试（P4-B Task 2b）。

形态依据：*Real-time scheduling for production-logistics collaborative environment using
multi-agent deep reinforcement learning*, Advanced Engineering Informatics 65:103216, 2025,
**Fig. 2**（该文自陈采用其 [5] 的车间布局 = **Cai et al., International Journal of Production
Research 61(4):1373-1393, 2023**）。Figure 形态：单一装卸单元在**机位网格外左侧**、约在第 1/2 行
分界的高度，由**一条短连接段**接到左边缘交叉口；**它不是格点交叉口**。

⚠️ 本任务的裁定：**口径差只允许体现在"行程时间从哪来"，不得改变问题结构**——故几何口径
（原始 MK）也有站→首工序机台与末工序机台→站两条腿。
"""
from __future__ import annotations

import numpy as np
import pytest

from plcsp.env.constraints import ConstraintConfig
from plcsp.env.corridors import build_corridor_graph, dock_distance_matrix
from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import Instance, load_mk
from plcsp.env.layout import sample_layout
from plcsp.env.mkt import load_mkt
from plcsp.env.transport import TransportCaliber

# 十约束全关：本文件要钉的是**结构与时长**，不是约束交互（约束交互有各自的测试文件）
_ALL_OFF = ConstraintConfig().with_off(
    "congestion", "finite_buffer", "machine_failure", "rework", "setup_time", "due_dates",
    "agv_failure", "heterogeneous_fleet", "charging", "maintenance")


def _gmin(dm, u: int, v: int, cfg: SimConfig) -> float:
    """几何口径的节点 u→v 分钟数（与 `AgvSim._seg_min` 同式）。"""
    return float(dm[u, v]) / cfg.eff_speed / cfg.agv_speed_mps / 60.0


# ── 布局：站是格点外的节点 + 一条连接段 ──

@pytest.mark.unit
def test_station_sits_outside_the_lattice_on_the_left_with_a_connector():
    """⚠️ 装卸站是**格点外**的新节点 + 一条连接段（Fig. 2 形态）——**不得**复用边界交叉口。

    复用交叉口会静默短掉一整段连接段，并把装卸站塞进通道网当普通路口（AGV 会在那里与别的
    车争用同一个区段），两者都不报错、只是数字变小、拥堵数据失真。
    """
    lay = sample_layout(6, seed=0)
    s, lu = lay.grid, lay.lu
    assert lu is not None, "布局必须带装卸站"
    assert lu.node >= s.n_nodes, "装卸站节点落进了格点编号区间——它就是被复用的交叉口"
    assert lu.dock_node == s.node_id(1, 0), "连接段应接在左边缘、第 1/2 行分界处"
    assert lu.x < min(m.x for m in lay.machines), "装卸站必须在网格外（左侧）"
    assert lu.y == pytest.approx(s.node_xy(1, 0)[1]), "高度应取第 1/2 行分界"
    assert lu.connector_m > 0.0


@pytest.mark.unit
def test_station_has_a_single_role_no_second_half_used_concept():
    """⚠️ `BufferPad`/`buffers` 是死数据（全仓零读者）——它的角色由装卸站承担，不得两个并存。"""
    lay = sample_layout(6, seed=0)
    assert not hasattr(lay, "buffers"), "旧 buffers 还在——同一角色有两个概念"
    assert lay.lu is not None


@pytest.mark.unit
def test_graph_has_the_station_joined_by_exactly_one_connector():
    lay = sample_layout(6, seed=0)
    g = build_corridor_graph(lay)
    assert g.number_of_nodes() == lay.grid.n_nodes + 1, "装卸站没进图（或多进了节点）"
    assert g.degree(lay.lu.node) == 1, "装卸站应只由一条连接段接入"
    assert g[lay.lu.node][lay.lu.dock_node]["weight"] == pytest.approx(lay.lu.connector_m)


@pytest.mark.unit
def test_every_machine_dock_can_reach_the_station_through_the_connector():
    """距离必须**含连接段**：站的行程 = 到左边缘交叉口的最短路 + 连接段长。"""
    lay = sample_layout(6, seed=0)
    dm = dock_distance_matrix(build_corridor_graph(lay))
    for m in lay.machines:
        d = float(dm[m.dock_node, lay.lu.node])
        assert np.isfinite(d), f"dock {m.dock_node} 到装卸站不可达"
        assert d == pytest.approx(float(dm[m.dock_node, lay.lu.dock_node]) + lay.lu.connector_m)


@pytest.mark.unit
@pytest.mark.parametrize("granularity", ["node", "row", "col", "all"])
def test_zone_map_covers_the_station_node(granularity: str):
    """⚠️ `_drive` 逐段查 `zone_of`；装卸站节点漏了就是 KeyError（或更糟：静默错段）。

    站是格点外节点，`row`/`col` 粒度没有它的行列号——必须给它一个**独占**的区段号。
    """
    from plcsp.env.des import build_zone_map

    lay = sample_layout(6, seed=0)
    zof, nz = build_zone_map(lay, granularity)
    assert lay.lu.node in zof, f"{granularity} 粒度的区段表没有装卸站节点"
    assert 0 <= zof[lay.lu.node] < nz


# ── 口径：装卸站 ↔ 矩阵下标 0 ──

@pytest.mark.unit
def test_matrix_caliber_maps_the_station_to_index_zero():
    """⭐ 本任务的正题：矩阵的**第 0 行/列**（装卸站）终于有消费者。

    同时复核"机台 i ↔ 下标 i+1"在接入装卸站后**没有漂**。
    """
    mkt = load_mkt("mk01")
    full = mkt.base.trans_time_full
    lay = sample_layout(6, seed=0, n_agv=3)
    c = TransportCaliber.for_instance(mkt.base, lay)
    assert c.slot_of(lay.lu.node) == 0, "装卸站没有映射到矩阵下标 0"
    for i, mp in enumerate(lay.machines):
        assert c.slot_of(mp.dock_node) == i + 1
    assert c.minutes(lay.lu.node, lay.machines[2].dock_node) == pytest.approx(float(full[0, 3]))
    assert c.minutes(lay.machines[2].dock_node, lay.lu.node) == pytest.approx(float(full[3, 0]))


@pytest.mark.unit
def test_lu_legs_are_order_preserving_too():
    """⚠️ LU 行/列**也非对称**（实测 mk01：站→M3 ≠ M3→站）——查表不得转置。"""
    mkt = load_mkt("mk01")
    full = mkt.base.trans_time_full
    assert not np.allclose(full[0, 1:], full[1:, 0]), "本测试的前提：mk01 的 LU 行/列确实非对称"
    lay = sample_layout(6, seed=0, n_agv=3)
    c = TransportCaliber.for_instance(mkt.base, lay)
    a = lay.lu.node
    b = lay.machines[0].dock_node
    assert c.minutes(a, b) == pytest.approx(float(full[0, 1]))
    assert c.minutes(b, a) == pytest.approx(float(full[1, 0]))
    assert c.minutes(a, b) != c.minutes(b, a)


# ── 作业结构：在站入场、完工回站 ──

@pytest.mark.unit
def test_every_job_starts_at_the_station_and_returns_to_it():
    """⭐ 结构断言：每件作业恰好一条"站→首工序机台"与一条"末工序机台→站"的任务。

    端点号约定：机台 0..m-1，**装卸站 = m**。`task_flow` 的 oi 是"这条搬运服务于第几道工序"
    （投放 = 0、回站 = 末工序号 + 1）。
    """
    inst = load_mk("mk01")
    lu = inst.n_machines
    r = rollout(inst, seed_chain=1, cfg=SimConfig(n_agv=3))
    flow = r["task_flow"]
    assert r["jobs_done"] == inst.n_jobs
    for j, job in enumerate(inst.jobs):
        tasks = [t for t in flow if t[0] == j]
        assert tasks, f"作业 {j} 没有任何搬运任务"
        assert tasks[0][2] == lu and tasks[0][1] == 0, \
            f"作业 {j} 的首工序不是从装卸站出发：{tasks[0]}"
        assert tasks[-1][3] == lu, f"作业 {j} 没有回站任务：{tasks[-1]}"
        assert tasks[-1][1] == len(job), "回站任务应挂在末工序之后"


@pytest.mark.unit
def test_makespan_ends_at_the_station_arrival_not_the_last_machine_op():
    """⭐⭐ 最强的一条：**单车、单作业、单工序**下 makespan 逐位等于
    `站→机台 + 加工 + 机台→站`。

    这条挡的是"忘了把回站腿算进 makespan"——那种实现下数字只小一点点、全程不报错，
    正是本批最危险的静默形态。用合成实例（不依赖任何已发表数字）把等式钉死。
    """
    inst = Instance(n_jobs=1, n_machines=2, jobs=[[[(0, 10.0)]]], source="lu-probe")
    cfg = SimConfig(n_agv=1)
    lay = sample_layout(inst.n_machines, seed=0, n_agv=1)
    dm = dock_distance_matrix(build_corridor_graph(lay))
    out = _gmin(dm, lay.lu.node, lay.machines[0].dock_node, cfg)
    back = _gmin(dm, lay.machines[0].dock_node, lay.lu.node, cfg)

    r = rollout(inst, seed_chain=1, cfg=cfg, constraints=_ALL_OFF)
    assert r["travel_time_total"] == pytest.approx(out + back, rel=1e-9), \
        "行程不等于两条 LU 腿之和——回站腿没跑或没记账"
    assert r["makespan"] == pytest.approx(out + 10.0 + back, rel=1e-9), \
        "makespan 不是到站时刻——完工时刻仍记在机台上（回站腿没进 makespan）"


@pytest.mark.unit
def test_matrix_travel_includes_the_lu_legs_looked_up_at_index_zero():
    """⚠️ 矩阵口径下两条 LU 腿必须**查表**（LU = 下标 0），不得退回几何。

    与 `test_transport_wiring.py` 的同型断言相比，这一条**把 LU 段也算进去**：端点映射为
    机台 i → i+1、装卸站 → 0。
    """
    mkt = load_mkt("mk01")
    full = mkt.base.trans_time_full
    cons = ConstraintConfig().with_off("congestion", "due_dates", "heterogeneous_fleet", "charging")
    r = rollout(mkt.base, seed_chain=1, cfg=SimConfig(n_agv=1), constraints=cons)

    def slot(idx: int) -> int:
        return 0 if idx == mkt.base.n_machines else idx + 1

    loaded, empty, prev = 0.0, 0.0, None
    for _j, _oi, frm, to in r["task_flow"]:
        loaded += float(full[slot(frm), slot(to)])
        if prev is not None:
            empty += float(full[slot(prev), slot(frm)])
        prev = to
    assert len(r["task_flow"]) == 2 * mkt.base.n_jobs + sum(
        len(job) - 1 for job in mkt.base.jobs), "任务数应是 投放n + 回站n + 工序流转数"
    assert r["travel_time_total"] == pytest.approx(loaded + empty, rel=1e-9), (
        f"矩阵口径行程 {r['travel_time_total']:.2f} ≠ 查表值 {loaded + empty:.2f}"
        "——LU 段疑似走了几何、或矩阵下标映射漏了 0 号位")


@pytest.mark.unit
def test_station_legs_count_as_loaded_and_leave_no_empty_leg():
    """⚠️ 两条 LU 段载着工件 ⟹ 计入**负载**态；单车单作业下空载态恒为 0。

    载错态不报错，但三态能耗（负载/空载功率不同）与"运输负荷"读数会一起错。
    """
    inst = Instance(n_jobs=1, n_machines=2, jobs=[[[(0, 10.0)]]], source="lu-probe")
    cfg = SimConfig(n_agv=1)
    lay = sample_layout(inst.n_machines, seed=0, n_agv=1)
    dm = dock_distance_matrix(build_corridor_graph(lay))
    out = _gmin(dm, lay.lu.node, lay.machines[0].dock_node, cfg)
    back = _gmin(dm, lay.machines[0].dock_node, lay.lu.node, cfg)

    r = rollout(inst, seed_chain=1, cfg=cfg, constraints=_ALL_OFF)
    st = r["energy_breakdown"]["agv_states_min"]
    assert st["loaded"] == pytest.approx(out + back, rel=1e-9), "LU 两段没进负载态"
    assert st["empty"] == pytest.approx(0.0, abs=1e-12), "本 episode 不该有空载段"


# ── 特征层：装卸站坐标必须在 node_xy 里 ──

@pytest.mark.unit
def test_norm_context_node_xy_covers_the_station():
    """⚠️ `nn/state_emb` 按 `pos_node` 取 `ctx.node_xy[·]`；装卸站不在里面就是运行期 IndexError。

    `GridSpec.n_nodes` **保持为格点数**（`nn` 的逐节点特征按它建），装卸站坐标**追加在末位**，
    正好对上"站节点号 = n_nodes"的编号约定。
    """
    from plcsp.algo.setup import build_ctx_for_unit_test

    inst = load_mk("mk01")
    lay = sample_layout(inst.n_machines, seed=0, n_agv=1)
    ctx = build_ctx_for_unit_test(inst, lay)
    assert len(ctx.node_xy) == lay.grid.n_nodes + 1
    assert ctx.node_xy[lay.lu.node] == pytest.approx((lay.lu.x, lay.lu.y))
