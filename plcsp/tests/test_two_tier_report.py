"""两个报告档的测试（P4-B Task 4）。"""
from __future__ import annotations

import dataclasses

import pytest

from plcsp.env.constraints import ABLATION_GROUPS, ConstraintConfig, REPORT_TIERS


@pytest.mark.unit
def test_tier_a_is_exactly_the_existing_none_group():
    """★ 用户裁定的落点：报告档 A（MKT 口径）**就是** `ABLATION_GROUPS["None"]`——不新造配置。

    十个机制（①拥堵 ②有限缓冲 ③机台故障 ④返工 ⑤换型 ⑧交期 ⑨AGV故障 ⑩异构车队 ⑪充电 ⑫维护）
    MKT 一个都没有 ⟹ "MKT 没有对应项的机制全关" == "全关"。
    """
    assert REPORT_TIERS["A-MKT"] is ABLATION_GROUPS["None"], "报告档 A 不是复用而是新造了一个配置"
    assert REPORT_TIERS["B-Full"] is ABLATION_GROUPS["Full"]


@pytest.mark.unit
def test_tier_a_turns_off_every_dataclass_field():
    """⚠️ **按 dataclass 字段**逐个查（不是查 `test_constraints.py` 里硬编码的 10 个名字）。

    将来加第 11 个机制时，若它 MKT 也没有对应项而 `ABLATION_GROUPS["None"]` 忘了关，
    报告档 A 就不再是"MKT 口径"了——这条会红。
    """
    on = [f.name for f in dataclasses.fields(ConstraintConfig) if getattr(REPORT_TIERS["A-MKT"], f.name)]
    assert not on, f"报告档 A 里还开着的机制：{on}——若该机制 MKT 确实有对应项，请补进文档并说明理由"


@pytest.mark.unit
def test_both_tiers_run_on_the_matrix_and_differ():
    """两个报告档都必须**跑得出且不同**——相同则说明机制没接进仿真（消融表的老毛病）。

    ⚠️ 行里的 makespan 列叫 **`ours`**（与已发表数字并列表的列名一致；本表逐行是"我们的
    参考调度"）。计划原文此处的键写作 `makespan`，与它自己给的 `reference_table` 行结构
    不一致——按行的真实结构取 `ours`（`seeds=1` 时两者是同一个数，断言强度不变）。
    """
    from plcsp.m14_mkt_reference import reference_table

    rows = {r["tier"]: r for r in reference_table(("mk01",), n_agv=6, seeds=1)}
    a, b = rows["A-MKT"], rows["B-Full"]
    assert a["transport"] == b["transport"] == "matrix"
    assert a["ours"] != b["ours"], "两个报告档跑出同一 Cmax——机制没接上"
    assert a["energy"] != b["energy"], "两个报告档 energy 相同——运输/约束没进能耗"


@pytest.mark.unit
def test_tier_a_has_no_unmapped_legs_and_tier_b_declares_its_own():
    """⚠️ Review Focus #5：报告档 A **不该**有未映射路段（⑪ 关 ⟹ 车不会开去充电桩）。

    报告档 B 开着 ⑪，可能产生未映射路段——**计数与分钟数必须随行带出**，供论文如实声明
    "本报告档里有 x% 的行程是几何口径补的"。
    """
    from plcsp.m14_mkt_reference import reference_table

    rows = {r["tier"]: r for r in reference_table(("mk01",), n_agv=6, seeds=1)}
    assert rows["A-MKT"]["unmapped_legs"] == 0, "报告档 A 出现未映射路段——有 MKT 没有的机制在驱使 AGV"
    for tier, r in rows.items():
        assert "unmapped_legs" in r and "unmapped_min" in r, f"{tier} 没带未映射计数"
        assert r["unmapped_min"] >= 0.0


@pytest.mark.unit
def test_forced_low_battery_under_matrix_raises_by_default():
    """⚠️ Review Focus #5 的**不happy 路径**：矩阵口径 + ⑪ 开 + 默认策略 ⟹ 充电段必须**显式报错**。

    用 `battery_low=0.99 / battery_high=1.0` **力迫**第一趟之后就充电（默认参数下矩阵口径的
    耗电也确实会到这一水平，见 §5.7 的对比），把"静默猜一个值"这条路堵死。
    """
    from plcsp.env.des import SimConfig, rollout
    from plcsp.env.mkt import load_mkt

    mkt = load_mkt("mk01")
    with pytest.raises(ValueError, match="没有对应项"):
        rollout(mkt.base, seed_chain=1,
                cfg=SimConfig(n_agv=6, battery_low=0.99, battery_high=1.0,
                              transport_unmapped="raise"),
                constraints=ConstraintConfig())


@pytest.mark.unit
def test_forced_low_battery_under_matrix_counts_the_geometry_fallback():
    """⚠️ Review Focus #5：显式选 `geometry` 时充电段**跑得通且留痕**（段数 + 分钟都 > 0）。

    这就是**报告档 B 的真实形态**（⑪ 开、矩阵无充电桩项）——论文里必须如实声明这一部分的占比。
    """
    from plcsp.env.des import SimConfig, rollout
    from plcsp.env.mkt import load_mkt

    mkt = load_mkt("mk01")
    r = rollout(mkt.base, seed_chain=1,
                cfg=SimConfig(n_agv=6, battery_low=0.99, battery_high=1.0,
                              transport_unmapped="geometry"),
                constraints=ConstraintConfig())
    assert r["charge_events"] > 0, "力迫低电却没充过电——本测试的前提不成立"
    assert r["unmapped_legs"] > 0 and r["unmapped_min"] > 0.0


@pytest.mark.unit
def test_rows_carry_tier_and_caliber_labels():
    """⚠️ Review Focus：两个报告档的数并排放时必须**逐行自报报告档与口径**（P4-A Review Focus #3 同型）。"""
    from plcsp.m14_mkt_reference import reference_table

    for r in reference_table(("mk01",), n_agv=6, seeds=1):
        assert r["tier"] in ("A-MKT", "B-Full")
        assert r["ours_benchmark"] == "MKT"
        assert r["benchmark"] == "MKT"          # 对照列的口径（未变）


@pytest.mark.unit
def test_no_horizon_truncation_in_the_tested_configs():
    """⚠️ Review Focus #8：矩阵口径下 makespan 大数倍，固定护栏可能**掐表**。

    掐表的结果只是"看起来更慢"，会被当成正常读数——故必须逐行检查。
    """
    from plcsp.m14_mkt_reference import reference_table

    for r in reference_table(("mk01",), n_agv=6, seeds=1):
        assert r["horizon_hit"] is False, f"{r['tier']} 掐表了——先按声明的方式抬护栏（它是护栏不是物理量）"
