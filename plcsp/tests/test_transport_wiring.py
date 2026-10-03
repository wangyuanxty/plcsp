"""矩阵口径接进仿真的测试（P4-B Task 2）。

⚠️ 本文件的核心是**直接钉住算术**（逐位对拍 Σ 矩阵查表值），不靠端到端间接推断——
本项目的教训是"测试名声称的属性 > 实际验证的内容"已发生 6 次。
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import simpy

from plcsp.env.constraints import ConstraintConfig
from plcsp.env.corridors import build_corridor_graph, dock_distance_matrix
from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import load_mk
from plcsp.env.layout import AgvSpec, sample_layout
from plcsp.env.mkt import load_mkt
from plcsp.env.transport import TransportCaliber


def _agv(inst, *, speed_factor: float, hetero: bool, unmapped: str = "raise", n_agv: int = 2):
    """造一台**不跑仿真**的 AgvSim，只用来直接调 `_leg_min`（算术判据不该绕道端到端）。"""
    from plcsp.env.des import AgvSim, SimTrack, ZoneManager, build_zone_map

    cfg = SimConfig(n_agv=n_agv, transport_unmapped=unmapped)
    lay = sample_layout(inst.n_machines, seed=0, n_agv=n_agv)
    g = build_corridor_graph(lay)
    zof, nz = build_zone_map(lay, cfg.zone_granularity)
    env = simpy.Environment()
    con = ConstraintConfig() if hetero else ConstraintConfig(heterogeneous_fleet=False)
    spec = AgvSpec(id=0, speed_factor=speed_factor, capacity=1, battery_kwh=3.0)
    cal = TransportCaliber.for_instance(inst, lay, unmapped=unmapped)
    agv = AgvSim(env, 0, dock_distance_matrix(g), cal, cfg, {}, simpy.Store(env), [], g,
                 ZoneManager(env, zof, nz, cfg.zone_wait_limit), con, spec,
                 np.random.default_rng(0), SimTrack(1, inst.n_machines, n_agv),
                 chargers=lay.chargers)
    return agv, lay


def _exact_caliber_constraints():
    """精确对拍需要的四个前提：① 关（不拆区段）、⑧ 关、⑩ 关（载量 1、倍率 1）、⑪ 关。

    ⚠️ **⑪ 必须关**：矩阵口径下单车一趟 episode 会耗掉 ~3.5 kWh（0.5 kW × ~342 min 负载 +
    0.2 kW × 空载），与电池容量 2–4 kWh 同量级 ⟹ 车**会**去充电桩，而充电桩没有矩阵项
    （默认策略 `raise` 会直接报错、`geometry` 会往总额里混入几何腿）。既有几何口径测试没这个问题
    （整段 episode 只耗 0.27 kWh，见 §5.7），**不要**照抄它的约束配置。

    ⚠️ **⑧ 也必须关**（本批新增的判断）：矩阵口径的交期标定表 `TF_RDD_MATRIX` 要到
    Task 3 才落盘，在那之前 `due_dates_for` 对 `matrix` 口径**按设计显式报错**（不得回退几何表）。
    Task 2 的这组测试只问"行程时间对不对"，交期在这条判据里是无关量——照 `m16_due_calib.
    reference_completes` 的既有先例关掉 ⑧，而不是把 MKT 交期表提前造出来（那会绕过 Task 3
    的可复现标定）。Task 4 的表则在 ⑧ 开着的情况下跑（那时表已落盘）。
    """
    return ConstraintConfig().with_off("congestion", "due_dates", "heterogeneous_fleet", "charging")


# ── 主线：逐位对拍 ──

@pytest.mark.unit
def test_matrix_travel_time_equals_the_lookup_sum_exactly():
    """⭐ Review Focus #1：矩阵口径下 `travel_time_total` 必须**逐位等于** Σ 矩阵查表值。

    这条同时挡住三类错误实现：① 又除了一次 `60·车速`（数会小 ~60 倍）；② 用了丢过 LU 的错行
    矩阵；③ 几何口径的 `m_dm` 还在被读。口径：① 关、⑩ 关、⑪ 关、单车
    （空载段 = 上一卸货点 → 本次取货点）。
    """
    mkt = load_mkt("mk01")
    t = mkt.base.trans_time_full                        # (m+1)×(m+1)：**含装卸站**
    r = rollout(mkt.base, seed_chain=1, cfg=SimConfig(n_agv=1), constraints=_exact_caliber_constraints())

    def slot(idx: int) -> int:                          # 机台 i → i+1；装卸站 → 0
        return 0 if idx == mkt.base.n_machines else idx + 1

    loaded, empty, prev = 0.0, 0.0, None
    for _j, _oi, frm, to in r["task_flow"]:
        loaded += float(t[slot(frm)][slot(to)])
        if prev is not None:
            empty += float(t[slot(prev)][slot(frm)])
        prev = to
    assert len(r["task_flow"]) > 0, "本 episode 应有搬运任务"
    assert r["travel_time_total"] == pytest.approx(loaded + empty, rel=1e-9), (
        f"矩阵口径的行程 {r['travel_time_total']:.2f} ≠ 查表值 {loaded + empty:.2f}"
        "——疑似又套了一次几何换算（或读的不是这张矩阵）")


@pytest.mark.unit
def test_zone_split_preserves_the_matrix_total():
    """⚠️ Review Focus #7：① **开着**时行程被拆成逐节点区段，但各段之和仍须等于矩阵查表值。

    矩阵没有"中间走廊节点"的概念；整段矩阵时长按**几何占比**摊到各段（占比和恒为 1）。
    单车 ⟹ 无区段争用 ⟹ 不会中途失败，故等式应**逐位**成立。⑪ 关（理由同上一条）；
    ⑧ 关（矩阵交期表 Task 3 才落盘，见 `_exact_caliber_constraints`）。
    """
    mkt = load_mkt("mk01")
    t = mkt.base.trans_time_full                        # 含装卸站（P4-B Task 2b）
    cons = ConstraintConfig().with_off("due_dates", "heterogeneous_fleet", "charging")   # ① 保持默认开
    r = rollout(mkt.base, seed_chain=1, cfg=SimConfig(n_agv=1), constraints=cons)

    def slot(idx: int) -> int:
        return 0 if idx == mkt.base.n_machines else idx + 1

    loaded, empty, prev = 0.0, 0.0, None
    for _j, _oi, frm, to in r["task_flow"]:
        loaded += float(t[slot(frm)][slot(to)])
        if prev is not None:
            empty += float(t[slot(prev)][slot(frm)])
        prev = to
    assert r["zone_wait"]["n"] >= 0                            # 若 ① 根本没生效，下面的等式无意义
    assert r["travel_time_total"] == pytest.approx(loaded + empty, rel=1e-9)


@pytest.mark.unit
def test_matrix_minutes_are_not_rescaled_by_speed_or_aisle_width():
    """⚠️ Review Focus #1：矩阵**已是分钟**——改绝对车速/通道宽**不得**改变矩阵口径的行程。

    几何口径下两者**必须**改变行程（见 `test_agv_travel.py`），这里钉的是相反的属性。
    """
    mkt = load_mkt("mk01")
    cons = _exact_caliber_constraints()
    fast = rollout(mkt.base, seed_chain=1,
                   cfg=SimConfig(n_agv=1, agv_speed_mps=0.5, aisle_width=1.5), constraints=cons)
    slow = rollout(mkt.base, seed_chain=1,
                   cfg=SimConfig(n_agv=1, agv_speed_mps=0.05, aisle_width=0.6), constraints=cons)
    assert fast["travel_time_total"] == slow["travel_time_total"], "矩阵口径被车速/通道宽重标了"
    assert fast["makespan"] == slow["makespan"]


# ── ⑩ 的相对倍率必须仍然生效（否则机制变常数）──

@pytest.mark.unit
def test_relative_speed_factor_still_applies_under_the_matrix():
    """⚠️ Review Focus #5：⑩ 的**相对**速度倍率在矩阵口径下必须生效。

    矩阵是"标准车速"下的分钟数；车队内部的快慢差（±20%）仍要表达出来，否则"异构车队"
    在 MKT 口径里静默退化成同构车队（机制变常数、消融表照绿）。
    """
    mkt = load_mkt("mk01")
    t = mkt.trans_time
    agv1, lay = _agv(mkt.base, speed_factor=1.0, hetero=True)
    agv2, _ = _agv(mkt.base, speed_factor=2.0, hetero=True)     # 同一 seed ⟹ 布局与 dock 节点相同
    u, v = lay.machines[0].dock_node, lay.machines[1].dock_node
    assert agv1._leg_min(u, v) == pytest.approx(float(t[0][1]))
    assert agv2._leg_min(u, v) == pytest.approx(float(t[0][1]) / 2.0)


@pytest.mark.unit
def test_speed_factor_is_inert_when_heterogeneous_fleet_is_off():
    """⑩ 关 ⟹ 倍率恒 1（与"该约束从未存在"逐位相同）——反方向钉一次，防"开关关不掉"。"""
    mkt = load_mkt("mk01")
    t = mkt.trans_time
    agv, lay = _agv(mkt.base, speed_factor=2.0, hetero=False)
    u, v = lay.machines[0].dock_node, lay.machines[1].dock_node
    assert agv._leg_min(u, v) == pytest.approx(float(t[0][1]))


# ── 未映射端点（充电桩）的两条路径 ──

@pytest.mark.unit
def test_unmapped_endpoint_raises_by_default():
    """⚠️ Review Focus #5：矩阵没有对应项的端点**默认显式报错**，不得静默猜一个值。"""
    mkt = load_mkt("mk01")
    agv, lay = _agv(mkt.base, speed_factor=1.0, hetero=True, unmapped="raise")
    with pytest.raises(ValueError, match="没有对应项"):
        agv._leg_min(lay.machines[0].dock_node, lay.chargers[0].node)


@pytest.mark.unit
def test_unmapped_endpoint_falls_back_to_geometry_and_is_counted():
    """⚠️ Review Focus #5：显式选 `geometry` 策略时**降级要留痕**（计数 + 时长都进 metrics）。

    口径：这一档是"矩阵覆盖不到的路段用几何补"，必须在表里声明——**不得**无声降级。
    """
    mkt = load_mkt("mk01")
    agv, lay = _agv(mkt.base, speed_factor=1.0, hetero=True, unmapped="geometry")
    u = lay.machines[0].dock_node
    got = agv._leg_min(u, lay.chargers[0].node)
    assert got > 0.0
    assert agv.stats["unmapped_legs"] == 1 and agv.stats["unmapped_min"] == pytest.approx(got)


# ── 口径跟随实例（不是全局开关）──

@pytest.mark.unit
def test_no_global_switch_can_turn_the_matrix_on():
    """⚠️ Review Focus #3：`SimConfig` 里**不得**出现任何口径/矩阵开关（用户裁定）。

    若日后有人加 `SimConfig.use_mkt_matrix`，这条会红——那正是"同一实例跑出两套行程时间"的入口。
    """
    names = {f.name for f in dataclasses.fields(SimConfig)}
    bad = sorted(n for n in names if "matrix" in n or "mkt" in n.lower())
    assert not bad, f"SimConfig 里出现了口径开关：{bad}（口径只能由实例携带）"


@pytest.mark.unit
def test_matrix_label_without_matrix_data_raises():
    """⚠️ Review Focus #3：标了 `matrix` 却没有矩阵 ⟹ 显式报错，**不得**静默退回几何。"""
    inst = load_mk("mk01")
    inst.transport = "matrix"                     # 手工制造"标签与数据不符"
    with pytest.raises(ValueError, match="trans_time_full"):
        TransportCaliber.for_instance(inst, sample_layout(6, seed=0, n_agv=3))


@pytest.mark.unit
def test_geometry_instance_never_gets_a_matrix_caliber():
    """⚠️ Review Focus #3：几何实例 + 任何布局 ⟹ 几何口径（矩阵进不来）。"""
    lay = sample_layout(6, seed=0, n_agv=3)
    c = TransportCaliber.for_instance(load_mk("mk01"), lay)
    assert c.mode == "geometry" and c.matrix is None


@pytest.mark.unit
def test_rollout_declares_its_transport_caliber_in_the_metrics():
    """口径必须**随结果自报**——否则两档口径的数并排放时无从分辨（P4-A Review Focus #1 同型）。"""
    mk = rollout(load_mk("mk01"), seed_chain=1, cfg=SimConfig(n_agv=3))["transport"]
    # ⚠️ 矩阵 + ⑪ 开时充电桩没有矩阵项 ⟹ 必须显式选降级策略（本测试只看口径标签，用 ⑧ 关
    #    把交期标定的依赖摘掉——矩阵交期表要到 Task 3 才落盘）。
    mkt = rollout(load_mkt("mk01").base, seed_chain=1,
                  cfg=SimConfig(n_agv=6, transport_unmapped="geometry"),
                  constraints=ConstraintConfig().with_off("due_dates"))["transport"]
    assert mk == "geometry" and mkt == "matrix"


@pytest.mark.unit
def test_reference_runs_of_the_two_calibers_do_not_share_the_cache():
    """⚠️ Review Focus #4：同名 MK 与 MKT 实例的参考运行**不得共用缓存**。

    `_instance_key` 若不含口径与矩阵，`reference_run` 会把**几何口径**的参考运行静默喂给
    矩阵口径的调用方（M_ref、奖励权重、特征归一化一起错，且零报错）。
    """
    from plcsp.env.des import reference_run

    # ⚠️ 矩阵口径的参考运行必须显式声明"充电桩用几何补"——矩阵口径下 ⑪ 真的会 binding
    #    （单车一趟耗电与电池同量级），默认 `raise` 会在充电腿处显式报错（这正是设计要的）。
    cfg = SimConfig(n_agv=6, transport_unmapped="geometry")
    a = reference_run(load_mk("mk01"), cfg=cfg, seed_layout=0)["makespan"]
    b = reference_run(load_mkt("mk01").base, cfg=cfg, seed_layout=0)["makespan"]
    assert a != b, "两个口径的参考运行跑出同一 makespan——缓存串味（_instance_key 漏了口径/矩阵）"
    assert b > a, "矩阵是分钟量级，矩阵口径的参考 makespan 应显著更大"


# ── 交期标定表：MKT 不得命中 MK 的条目 ──

@pytest.mark.unit
def test_due_date_table_is_selected_by_caliber():
    """⚠️ Review Focus #6：交期标定表必须**按口径选**——MKT 不得读到原始 MK 那本尺。

    `load_mkt(name).base.source` 与 `load_mk(name).source` 是同一个文件名（加工数据一字不改），
    故**文件名主干相同**；分口径靠的是"读哪张表"，不是键长什么样。
    """
    from plcsp.env.due_dates import _instance_name, due_dates_for

    mk, mkt = load_mk("mk01"), load_mkt("mk01").base
    assert _instance_name(mk) == _instance_name(mkt) == "mk01"     # 主干确实相同（这正是危险所在）
    assert due_dates_for(mk)                                        # 几何表有 mk01

    inst = load_mk("mk01")
    inst.transport = "mkt-ish"                                      # 未知口径
    with pytest.raises(ValueError, match="口径"):
        due_dates_for(inst)


@pytest.mark.unit
def test_mkt_never_silently_reuses_the_mk_calibration():
    """⚠️ Review Focus #6：MKT 实例要么**显式报错**（未标定），要么给出与 MK **不同**的交期。

    唯一不允许的是"静默等于 MK 的值"——那是两个不同问题共用一把尺。
    （Task 3 落 `TF_RDD_MATRIX` 后本测试走"值不同"那一支；本测试**不需要改写**。）
    """
    from plcsp.env.due_dates import due_dates_for

    mk, mkt = load_mk("mk01"), load_mkt("mk01").base
    try:
        d_mkt = due_dates_for(mkt)
    except ValueError as e:
        assert "matrix" in str(e), f"报错必须说清是哪个口径下未标定，实得：{e}"
        d_mkt = None                       # 未标定：显式报错是正确行为
    if d_mkt is not None:
        assert d_mkt != due_dates_for(mk), "MKT 交期与 MK 逐位相同——标定表被静默复用了"


@pytest.mark.unit
def test_mkt_instance_still_presents_the_dropped_view():
    """`MktInstance.trans_time` 改成属性后，P4-A 交付的语义**不得变**：仍是丢 LU 的 m×m。"""
    mkt = load_mkt("mk01")
    from plcsp.env.mkt import load_mkt_layout
    assert mkt.trans_time.shape == (6, 6)
    assert np.array_equal(mkt.trans_time, load_mkt_layout(6))
    assert mkt.base.trans_time_full.shape == (7, 7), "全矩阵（含 LU）必须保在实例上"
