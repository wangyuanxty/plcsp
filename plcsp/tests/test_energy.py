"""M2 能耗模型的单元测试（P1b Task 2）。参数全部引证 GFJSPT-MMRS（SWEVO 99:102181）。

单位纪律（bug#13 的能耗版）：**功率 [kW] × 时长 [min] ÷ 60 = 能耗 [kWh]**。
漏掉 /60 是 60 倍错误——本文件的首要用例就是钉住这一点。
"""
from __future__ import annotations

import pytest

from plcsp.energy import (AGV_EMPTY_KW, AGV_IDLE_KW, AGV_LOADED_KW, SHOP_FIXED_KW,
                          agv_energy_kwh, machine_energy_kwh, machine_params_for,
                          total_energy_kwh)


@pytest.mark.unit
def test_agv_params_match_cited_source():
    """AGV 三态功率必须是文献值（spec §3.4：待机 0.1 / 空载 0.2 / 负载 0.5 kW）。"""
    assert (AGV_IDLE_KW, AGV_EMPTY_KW, AGV_LOADED_KW) == (0.1, 0.2, 0.5)
    assert SHOP_FIXED_KW == 0.4


@pytest.mark.unit
def test_agv_energy_is_kwh_not_kw_times_min():
    """单位核验：0.5 kW 跑 60 min = 0.5 kWh（不是 30）。这是本任务最易错处。"""
    e = agv_energy_kwh(idle_min=0.0, empty_min=0.0, loaded_min=60.0)
    assert e == pytest.approx(0.5, rel=1e-9)


@pytest.mark.unit
def test_machine_energy_uses_three_states():
    """机床三态各自贡献：待机/加工/换型 功率不同，缺一不可。"""
    p = dict(idle_kw=0.74, proc_kw=0.951, setup_kw=0.74)     # GFJSPT-MMRS M1–M4 低速档
    only_idle = machine_energy_kwh(0.0, 60.0, 0.0, **p)
    only_proc = machine_energy_kwh(60.0, 0.0, 0.0, **p)
    only_setup = machine_energy_kwh(0.0, 0.0, 60.0, **p)
    assert only_idle == pytest.approx(0.74, rel=1e-9)
    assert only_proc == pytest.approx(0.951, rel=1e-9)
    assert only_setup == pytest.approx(0.74, rel=1e-9)
    assert only_proc > only_idle, "加工功率应高于空载"


@pytest.mark.unit
def test_machine_tier_table_matches_cited_source():
    """三档转速的功率必须逐字对上 spec §3.4 的表（防手抄漂移）。"""
    from plcsp.energy import MACHINE_TIERS
    assert MACHINE_TIERS["M1_M4"]["proc"] == (0.951, 1.17)
    assert MACHINE_TIERS["M1_M4"]["idle"] == (0.74, 0.90)
    assert MACHINE_TIERS["M5_M6"]["proc"] == (0.30, 0.60)
    assert MACHINE_TIERS["M5_M6"]["idle"] == (0.24, 0.50)
    assert MACHINE_TIERS["M7_M10"]["proc"] == (0.20, 0.56)
    assert MACHINE_TIERS["M7_M10"]["idle"] == (0.16, 0.36)


@pytest.mark.unit
def test_machine_params_follow_paper_machine_grouping():
    """机位 → 档位的映射按原文献分组：第 1–4 台 M1_M4、5–6 台 M5_M6、7 台起 M7_M10。"""
    assert machine_params_for(0, 10)["tier"] == "M1_M4"
    assert machine_params_for(3, 10)["tier"] == "M1_M4"
    assert machine_params_for(4, 10)["tier"] == "M5_M6"
    assert machine_params_for(5, 10)["tier"] == "M5_M6"
    assert machine_params_for(6, 10)["tier"] == "M7_M10"
    # 取档内**低速**值（保守下界），不是均值
    assert machine_params_for(0, 10)["proc_kw"] == 0.951
    assert machine_params_for(6, 10)["idle_kw"] == 0.16


@pytest.mark.unit
def test_machine_params_extrapolate_beyond_paper_table_by_cycling():
    """原文献只有 10 台机位表，**MK10 有 15 台** → 第 11 台起按三档循环（assumed，见 spec §9）。"""
    from plcsp.energy import MAX_MACHINES
    assert MAX_MACHINES >= 15, "外推范围必须覆盖 MK10（15 台）"
    assert machine_params_for(10, 15)["tier"] == "M1_M4"      # 第 11 台
    assert machine_params_for(11, 15)["tier"] == "M5_M6"      # 第 12 台
    assert machine_params_for(12, 15)["tier"] == "M7_M10"     # 第 13 台
    assert machine_params_for(13, 15)["tier"] == "M1_M4"      # 循环


@pytest.mark.unit
def test_machine_params_rejects_out_of_range():
    """越界必须显式报错，不得静默兜底。"""
    from plcsp.energy import MAX_MACHINES
    with pytest.raises(ValueError):
        machine_params_for(0, MAX_MACHINES + 1)     # 机位数超出覆盖范围
    with pytest.raises(ValueError):
        machine_params_for(9, 6)                    # 机位号越界


@pytest.mark.unit
def test_total_energy_matches_literature_order_of_magnitude():
    """Review Focus #1：量级必须落在现实锚附近。

    现实锚（GFJSPT-MMRS）：makespan 516 min 的同规模实例 → 42.4 kWh。
    本测试用一组典型时长验证**量级**在 10^0–10^2 kWh，防止单位错（kW×min 会得 10^3）。
    """
    e = total_energy_kwh(machine_kwh=20.0, agv_kwh=10.0, makespan_min=500.0)
    assert 10.0 < e < 100.0, f"总能耗 {e:.1f} kWh 偏离现实锚量级（应 ~40 kWh）"


@pytest.mark.unit
@pytest.mark.parametrize("name", ["mk01", "mk07", "mk10"])
def test_rollout_energy_matches_cited_anchor_ratio(name):
    """端到端量级核验（Review Focus #1）——**用现实锚的比值，不用绝对区间**。

    现实锚（GFJSPT-MMRS）：516 min → 42.4 kWh 与 619 min → 39 kWh，
    即 **0.063–0.082 kWh/min**。我们的实例量级差得远（MK 才几十到几百分钟），
    绝对值不可比，**比值可比**——故判据取每单位 makespan 的能耗。

    本测试同时钉死两种失败：
    - 漏 `/60`（kW×min 当 kWh）→ 比值 ×60 ≈ 4–8，远超上界；
    - 旧式"时间冒充能量"（`(process+travel)×1.0`）→ 比值 ≈ 1；
    - 实测：mk01 0.077 / mk07 0.073 / mk10 0.125（MK10 偏高因其 15 台机位空闲功率占比大）。
    """
    from plcsp.env.des import SimConfig, rollout
    from plcsp.env.instances import load_mk

    inst = load_mk(name)
    r = rollout(inst, seed_chain=1, cfg=SimConfig())
    assert not r["horizon_hit"] and r["jobs_done"] == inst.n_jobs
    ratio = r["energy"] / r["makespan"]
    assert 0.02 < ratio < 0.30, (
        f"{name} 能耗比 {ratio:.4f} kWh/min 偏离现实锚（0.063–0.082）——量级错了")
    bd = r["energy_breakdown"]
    assert bd["machine_kwh"] > 0.0 and bd["shop_kwh"] > 0.0
    # 三态时长必须闭合到在场时长（否则状态统计漏了一段，能耗被低估）
    states = bd["agv_states_min"]
    assert sum(states.values()) == pytest.approx(r["makespan"] * SimConfig().n_agv, rel=1e-9), \
        "AGV 三态时长之和不等于 makespan×车数——空闲态残差算错了"
