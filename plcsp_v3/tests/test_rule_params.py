# -*- coding: utf-8 -*-
"""v3 的规则参数化：**PA 机台侧**（每条 1 个 λ）· **LA**（每条 1 个 λ）。

设计依据：`docs/v3/design.md` §4.3 / §4.4。
⚠️ 作业侧在 `test_pa_rules.py` §9（那是步 1 做的）。
"""
from __future__ import annotations

import pytest

from plcsp_v3.env.rule_views import (LAVehicle, LAView, PAMachine, PAJob,
                                     PAView)
from plcsp_v3.hrl.rules import (LA_MIX, LA_RULE_NAMES, PA_MACH_MIX,
                                PA_MACHINE_RULES, la_score, la_terms,
                                pa_machine_score, pa_machine_terms)


def _paview() -> PAView:
    """候选机台 **0 与 1 故意冲突**：0 **快但要等**、1 **慢但已空闲**。

    ⚠️ **光"量不同"不够 —— 得冲突**：若"快的机台也更快空闲"，
    那 `λ ≥ 0` 时 `SMPT` 的选择**永不翻转**（初版夹具就是这么挂的）。
    """
    return PAView(now=10.0, jobs=(PAJob(oi=0, done_oi=0, alts=((0, 5.0), (1, 9.0)),
                                        apt=5.0, rem_apt=5.0, ready_at=10.0, due=50.0),),
                  machines=(PAMachine(avail_at=30.0, assigned_min=30.0),   # 快，但要等 20
                            PAMachine(avail_at=10.0, assigned_min=0.0),    # 慢，但已空闲
                            PAMachine(avail_at=10.0, assigned_min=100.0)))


def _laview() -> LAView:
    """三台车，**故意让三个量互相冲突**（口径照 `test_pa_rules.py` 的 `_la_view`）。"""
    now = 10.0
    return LAView(now=now, job=0, oi=1, frm_node=100, to_node=200, ready_at=25.0,
                  vehicles=(LAVehicle(node=100, busy_min=10.0, free_at=now),   # 已在取货点
                            LAVehicle(node=101, busy_min=1.0, free_at=now),    # 最近 + 最闲
                            LAVehicle(node=105, busy_min=50.0, free_at=now)),
                  leg_min=lambda u, v: float(abs(u - v)))


# ══════════════════ PA · 机台侧 ══════════════════

@pytest.mark.unit
def test_machine_lambda_zero_is_bit_identical():
    """**设计 §13.2 断言 1（机台侧）**：`λ = 0` 与原规则**同值**（不是同序 —— 同值）。"""
    v = _paview()
    for rule in PA_MACHINE_RULES:
        for alt in v.jobs[0].alts:
            a = pa_machine_score(rule, v, alt, (0.0,))
            if rule == "SMPT":
                b = float(alt[1])
            elif rule == "EAM":
                b = v.machines[alt[0]].avail_at
            else:
                b = v.machines[alt[0]].assigned_min / max(v.machines[alt[0]].avail_at, 1.0)
            assert a == b, f"{rule} @ {alt}: {a} != {b}"


@pytest.mark.unit
def test_machine_lambda_out_of_range_is_rejected():
    v = _paview()
    with pytest.raises(ValueError, match="越界"):
        pa_machine_score("SMPT", v, v.jobs[0].alts[0], (1.5,))


@pytest.mark.unit
def test_machine_lambda_can_change_the_pick():
    """**参数化必须真的改变选择** —— 否则是装饰（设计 §4.0 ①）。"""
    v = _paview()
    alts = v.jobs[0].alts
    base = min(alts, key=lambda a: pa_machine_score("SMPT", v, a))
    moved = any(min(alts, key=lambda a: pa_machine_score("SMPT", v, a, (lam,))) != base
                for lam in (0.5, 1.0))
    assert moved, "λ 变了都没改变机台选择 —— 参数化是装饰"


@pytest.mark.unit
def test_machine_mix_units_are_consistent():
    """**主量与辅量必须同单位**（设计 §4.3）—— 否则"相加"无意义。

    `pt` / `avail_delay` 是分钟；`util` / `pt_over_den` 是无量纲。
    """
    v = _paview()
    t = pa_machine_terms(v, v.jobs[0].alts[0])
    assert set(t) == {"pt", "avail_delay", "util", "pt_over_den"}
    m, _pt = v.jobs[0].alts[0]
    den = max(v.machines[m].avail_at, 1.0)
    assert t["util"] == pytest.approx(v.machines[m].assigned_min / den)
    assert t["pt_over_den"] == pytest.approx(v.jobs[0].alts[0][1] / den)
    # 三对 (主, 辅) 各自的单位一致
    for rule, (main, aux) in PA_MACH_MIX.items():
        mm = {"pt", "avail_delay"}
        both_min = main in mm and aux in mm
        both_dimless = main not in mm and aux not in mm
        assert both_min or both_dimless, f"{rule}: 主辅不同单位（{main} / {aux}）"


# ══════════════════ LA ══════════════════

@pytest.mark.unit
def test_la_lambda_zero_is_bit_identical():
    """**设计 §13.2 断言 1（LA 侧）**：`λ = 0` 逐位等于原打分（**含元组**）。"""
    v = _laview()
    for aid in range(len(v.vehicles)):
        av = v.vehicles[aid]
        assert la_score("MINCOST", v, aid, (0.0,)) == \
            (v.empty_leg(aid) + v.loaded_leg(aid),)
        assert la_score("MINU", v, aid, (0.0,)) == \
            (av.busy_min / max(av.free_at, 1.0),)
        arrive = max(av.free_at, v.now) + v.empty_leg(aid)
        want = (0, -arrive) if arrive <= v.ready_at else (1, arrive)
        assert la_score("LATEST_IN_TIME", v, aid, (0.0,)) == want


@pytest.mark.unit
def test_la_lambda_out_of_range_is_rejected():
    v = _laview()
    with pytest.raises(ValueError, match="越界"):
        la_score("MINCOST", v, 0, (1.5,))


@pytest.mark.unit
def test_la_latest_in_time_keeps_its_two_level_structure():
    """**`LATEST_IN_TIME` 的第一级（赶得上 / 赶不上）不被参数动**（设计 §4.4）。

    参数只在**第二级**按 `cost` 加权 —— 否则"赶得上"的语义会随参数漂。
    """
    v = _laview()
    for aid in range(len(v.vehicles)):
        for lam in (0.0, 0.5, 1.0):
            s = la_score("LATEST_IN_TIME", v, aid, (lam,))
            assert s[0] in (0, 1), f"第一级不是硬分类：{s}"


@pytest.mark.unit
def test_la_mix_units_are_consistent():
    """LA 的**主量与辅量必须同单位**（设计 §4.4）。"""
    v = _laview()
    t = la_terms(v, 0)
    assert set(t) == {"cost", "util", "cost_over_den", "arrive", "arrive_delay"}
    assert set(LA_MIX) == set(LA_RULE_NAMES), "LA_MIX 没覆盖全部 LA 规则"
    minutes = {"cost", "arrive", "arrive_delay"}
    for rule, aux in LA_MIX.items():
        if rule == "LATEST_IN_TIME":
            continue                       # 两级结构，另测
        main = "cost" if rule == "MINCOST" else "util"
        assert (main in minutes) == (aux in minutes), \
            f"{rule}: 主辅不同单位（{main} / {aux}）"
