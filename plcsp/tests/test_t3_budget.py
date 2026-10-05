# -*- coding: utf-8 -*-
"""T3 的**冻结预算表**与纯函数层（`plcsp/env/t3_budget.py`）——设计 `docs/t3-design.md`。

本文件只管"表是标定脚本的可复现产物"（同 ⑧ 交期的 `test_due_calib.py`）与三件纯逻辑：
归一化、对偶上升、可控性守卫。**机制**（罚项进优势、λ 跨步、②⑫ 的计数、动作退化守卫）
在 `test_t3_lagrangian.py`。
"""
from __future__ import annotations

import numpy as np
import pytest

from plcsp.env.instances import gen_random, load_mk
from plcsp.env.t3_budget import (ACTIVATION_SOURCES, A_REF, A_REF_MATRIX, BUDGET_RATIO,
                                 GEOMETRY, MATRIX, T3_CONSTRAINTS, T3_ETA, T3_LAMBDA_MAX,
                                 T3Budget, T3Lagrangian, activation_from_metrics,
                                 calibration_cfg, check_influenceable, dual_ascent,
                                 instance_key, normalized_activation, reference_activation,
                                 t3_ref_vector, usage_is_degenerate)

MK10 = [f"mk{i:02d}" for i in range(1, 11)]


# ────────────────────────── 1. 冻结表是可复现产物 ──────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("name", MK10)
def test_frozen_table_is_reproduced_by_the_calibration_script(name: str):
    """⚠️ 表必须是 `m17_t3_calib` 的**可复现产物**，不是手抄魔数。

    判据：冻结的表项 == 标定环境下的参考调度重算值。⚠️ **容差 `rel=1e-4` 不是放水**：
    表里存的是标定脚本**打印的 4 位小数**（脚本的格式化精度），而 ① 的激活量是浮点求和
    （实测 mk01 `zone_wait` = 1.7504906788762122 → 表里 1.7505）；其余五条是整数计数，逐位相等。
    改标定环境（cfg/布局/种子）而没重跑脚本，这条必然红。
    ⚠️ 这条同时把**标定环境**钉进测试：`calibration_cfg` / `calibration_layout` 一动，
    10 行表全部重算，红在这里而不是在训练曲线上。
    """
    inst = load_mk(name)
    got = reference_activation(inst)
    want = dict(zip(T3_CONSTRAINTS, A_REF[instance_key(inst)]))
    for c in T3_CONSTRAINTS:
        assert got[c] == pytest.approx(want[c], rel=1e-4), \
            f"{name} 的 {c} 参考激活量与冻结表不符：{got[c]} != {want[c]}"


@pytest.mark.unit
@pytest.mark.parametrize("name", MK10)
def test_every_constraint_is_alive_in_the_calibration_environment(name: str):
    """六条约束的 `a^ref` 必须**全部 > 0**——为 0 就无法归一化（`â = a/a^ref` 无定义）。

    这条是"标定环境选得对"的判据：默认档下 ⑪（默认电池放不空）与 ⑫（mk01/mk02 的
    `pm_interval=120` 从不逾期）的参考激活量恒 0（实测），故标定环境显式取小电池 + 短间隔。
    """
    row = t3_ref_vector(load_mk(name))
    assert np.all(row > 0.0), f"{name} 的 a^ref 有 0：{row}"


@pytest.mark.unit
def test_uncalibrated_instance_raises():
    """未标定实例**显式报错**（同 ⑧ 交期的 Review Focus #3），不得静默取默认值。"""
    with pytest.raises(ValueError, match="未标定"):
        t3_ref_vector(gen_random(4, 3, seed=0))


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(A_REF_MATRIX))
def test_matrix_table_is_reproduced_by_the_calibration_script(name: str):
    """矩阵口径（MKT）的表同样是标定脚本的可复现产物（只收六条全活的实例）。

    ⚠️ 矩阵表**只有 4 行**是**实测结论**，不是漏标：另 6 个实例的 ② 有限缓冲在参考调度上
    恒 0（矩阵行程是分钟级 ⟹ AGV 慢，机台把缓冲排空了才等到下一趟），mk02/mk04 的 ③ 也为 0
    ——参考水平为 0 ⟹ 无法归一化（`â = a/0`）⟹ **不进表、查表显式报错**。
    """
    from plcsp.env.mkt import load_mkt

    inst = load_mkt(name).base
    got = reference_activation(inst, caliber=MATRIX)
    want = dict(zip(T3_CONSTRAINTS, A_REF_MATRIX[name]))
    for c in T3_CONSTRAINTS:
        assert got[c] == pytest.approx(want[c], rel=1e-4), \
            f"{name}(matrix) 的 {c} 参考激活量与冻结表不符：{got[c]} != {want[c]}"


@pytest.mark.unit
def test_uncalibrated_caliber_raises_not_silently_falls_back():
    """两个"查不到"的分支都**显式报错**（不得静默回退几何表或取默认值）：

    - **未知口径**（写错标签）；
    - **矩阵口径里没进表的实例**（参考激活量为 0 ⟹ 无法归一化，见矩阵表的说明）。
    """
    from dataclasses import replace

    inst = load_mk("mk01")
    with pytest.raises(ValueError, match="口径"):
        calibration_cfg(inst, "teleport")
    with pytest.raises(ValueError, match="口径"):
        t3_ref_vector(inst, "teleport")
    mk02_matrix = replace(load_mk("mk02"), transport=MATRIX,
                          trans_time_full=np.zeros((2, 2)))
    assert instance_key(mk02_matrix) not in A_REF_MATRIX, "前提：mk02 不在矩阵表里（②/③ 恒 0）"
    with pytest.raises(ValueError, match="未标定"):
        t3_ref_vector(mk02_matrix, MATRIX)


# ────────────────────────── 2. 激活量的口径 ──────────────────────────

@pytest.mark.unit
def test_activation_sources_are_the_forced_quantities():
    """⚠️ 设计 §2 的关键定义：**激活量只算"被迫发作"**。

    - ⑪ 用 `agv_dry_events`（耗尽停机），**不是** `charge_events`（那是策略主动充的次数）；
    - ⑫ 用 `pm_events_forced`（被阈值强制触发），**不是** `pm_events`（含策略主动保养）。

    罚"策略主动做的动作"= 惩罚它刚拿到的动作，自相矛盾（设计原文）。
    """
    assert "agv_dry_events" in ACTIVATION_SOURCES["charging"]
    assert "charge_events" not in ACTIVATION_SOURCES["charging"]
    assert "pm_events_forced" in ACTIVATION_SOURCES["maintenance"]
    assert ACTIVATION_SOURCES["maintenance"] != 'metrics["pm_events"]'
    assert "buffer_block_min" in ACTIVATION_SOURCES["finite_buffer"]
    assert "zone_wait" in ACTIVATION_SOURCES["congestion"]


@pytest.mark.unit
def test_missing_metric_key_raises_instead_of_reading_zero():
    """metrics 里取不到键 ⟹ **显式报错**（不静默取 0）。

    静默取 0 会让"读错指标名"变成"该约束从不激活"——两者在表里长得一模一样。
    """
    with pytest.raises(ValueError, match="① 拥堵"):
        activation_from_metrics({})
    met = {"zone_wait": {"total": 1.0}, "buffer_block_min": 2.0, "fail_events": 1,
           "setup_minutes_total": 3.0, "agv_dry_events": 0, "pm_events_forced": 4}
    got = activation_from_metrics(met)
    assert got["congestion"] == 1.0 and got["maintenance"] == 4.0 and got["finite_buffer"] == 2.0


# ────────────────────────── 3. 归一化 ──────────────────────────

@pytest.mark.unit
def test_normalization_is_a_ratio_to_the_reference_level():
    """`âᵢ = aᵢ / aᵢ^ref`——量纲从 1.75（分钟级）到 2503（分钟）拉平到"占参考水平的比例"。"""
    a_ref = np.array([2.0, 100.0])
    met = {"zone_wait": {"total": 1.0}, "buffer_block_min": 50.0, "fail_events": 0,
           "setup_minutes_total": 0, "agv_dry_events": 0, "pm_events_forced": 0}
    hat = normalized_activation(met, a_ref, ("congestion", "finite_buffer"))
    assert hat == pytest.approx([0.5, 0.5])


# ────────────────────────── 4. 对偶上升（验收判据 3） ──────────────────────────

@pytest.mark.unit
def test_dual_ascent_direction_and_floor():
    """判据 3：`â > b ⟹ λ 升`；`â < b ⟹ λ 降`；**λ 恒非负**（且不超上界）。"""
    lam = np.array([1.0, 1.0, 0.02])
    ahat = np.array([0.9, 0.1, 0.1])
    b = np.array([0.5, 0.5, 0.5])
    out = dual_ascent(lam, ahat, b, eta=0.1)
    assert out[0] > lam[0], "â > b 时 λ 未上升"
    assert out[1] < lam[1], "â < b 时 λ 未下降"
    assert out[2] == pytest.approx(0.0), "λ 触底后必须停在 0（不得为负）"
    # 上界：η 极大时也不得超 λ_max（防发散，设计 §3.4）
    assert np.all(dual_ascent(np.array([9.0]), np.array([100.0]), np.array([0.0]),
                              eta=1000.0) <= T3_LAMBDA_MAX)
    # 恰好等于预算 ⟹ 不动
    assert dual_ascent(np.array([1.0]), np.array([0.5]), np.array([0.5]), 0.1) == pytest.approx([1.0])


@pytest.mark.unit
def test_zero_lambda_and_zero_eta_are_rejected_as_silent_idle():
    """T3 开着而 λ 不动（η ≤ 0）或预算比例越界（0/1）⟹ **显式报错**（静默空转是缺陷形态）。"""
    inst = load_mk("mk01")
    with pytest.raises(ValueError, match="eta"):
        T3Lagrangian(T3Budget(inst), eta=0.0)
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="ratio"):
            T3Budget(inst, ratio=bad)


@pytest.mark.unit
def test_unknown_constraint_name_raises():
    """约束名写错 ⟹ 报错（不得静默忽略——那会让"T3 到底罚了什么"与配置不符）。"""
    with pytest.raises(ValueError, match="未知 T3 约束"):
        T3Budget(load_mk("mk01"), keep=("congestion", "traffic"))


# ────────────────────────── 5. 可控性守卫（③ 纳入的理由） ──────────────────────────

@pytest.mark.unit
def test_influenceable_guard_requires_the_matching_action_head():
    """不可控的约束不纳入 T3（同 ④⑨ 不纳入的理由）——三条各自的"作用手段"缺一即报错。

    - ⑫ 维护：提前保养 = M 头的动作 ⟹ 要 `pm_head`；
    - ③ 机器故障：役龄故障率 + 提前保养把它归零 ⟹ 要 `pm_head` **且** `machine_age_failure`；
    - ⑪ 充电：早充 = C 头的动作 ⟹ 要 `charge_head`。

    ⚠️ ③ 那条是本机制与 §44（役龄模型）的**耦合点**：没有役龄，故障率与策略动作无关，
    罚它只是加噪（设计 §1.1）。
    """
    check_influenceable(("congestion", "finite_buffer", "setup_time"), False, False, False)
    with pytest.raises(ValueError, match="⑫"):
        check_influenceable(("maintenance",), False, True, True)
    with pytest.raises(ValueError, match="③"):
        check_influenceable(("machine_failure",), True, True, False)   # pm_head 开、役龄关
    with pytest.raises(ValueError, match="③"):
        check_influenceable(("machine_failure",), False, True, True)   # 役龄开、pm_head 关
    with pytest.raises(ValueError, match="⑪"):
        check_influenceable(("charging",), True, False, True)
    # 全开档：不报错
    check_influenceable(T3_CONSTRAINTS, True, True, True)


@pytest.mark.unit
def test_budget_targets_are_the_ratio_of_the_reference_level():
    """`b` = 归一化后的目标水平（`BUDGET_RATIO`），与 `a^ref` 一样逐约束同序。"""
    b = T3Budget(load_mk("mk01"), ratio=0.4)
    assert len(b) == len(T3_CONSTRAINTS) and b.keep == T3_CONSTRAINTS
    assert np.allclose(b.b, 0.4)
    sub = T3Budget(load_mk("mk01"), keep=("congestion", "charging"))
    assert sub.keep == ("congestion", "charging")
    assert np.allclose(sub.a_ref, np.array([A_REF["mk01"][0], A_REF["mk01"][4]]))
    assert np.allclose(sub.b, BUDGET_RATIO)


# ────────────────────────── 6. λ 的持有者（验收判据 2 的接线） ──────────────────────────

@pytest.mark.unit
def test_lagrangian_starts_at_zero_and_advances_from_diag():
    """λ 初值 0、只从 `diag["t3_lambda"]` 前进；diag 缺键 **报错**（不得静默不变）。"""
    t3 = T3Lagrangian(T3Budget(load_mk("mk01"), keep=("congestion",)),
                      eta=T3_ETA, lam0=None)
    assert np.all(t3.lam == 0.0), "λ 初值必须为 0（非零初值会污染第一组的优势）"
    kw = t3.step_kwargs()
    assert set(kw) == {"t3_lambda", "t3_budget", "t3_eta"} and kw["t3_eta"] == T3_ETA
    t3.advance({"t3_lambda": (0.7,)})
    assert np.allclose(t3.lam, [0.7])
    with pytest.raises(KeyError, match="t3_lambda"):
        t3.advance({})


@pytest.mark.unit
def test_usage_guard_thresholds():
    """上界退化守卫的判据：使用率落在 [LO, HI] 之外 = 退化成单一取值（~0 或 ~1）。"""
    assert usage_is_degenerate(0.0) and usage_is_degenerate(1.0)
    assert usage_is_degenerate(0.01) and usage_is_degenerate(0.999)
    assert not usage_is_degenerate(0.5) and not usage_is_degenerate(0.2)


def test_calibration_cfg_is_per_caliber():
    """标定环境的 cfg **按口径**取（矩阵口径换车队规模与降级策略，同 `m16_due_calib`）。"""
    from dataclasses import replace
    from plcsp.env.des import SimConfig

    inst = load_mk("mk01")
    g = calibration_cfg(inst, GEOMETRY)
    m = calibration_cfg(replace(inst, transport=MATRIX), MATRIX)
    assert g.n_agv == SimConfig().n_agv, "几何口径的标定 cfg 不该动车队规模（默认档）"
    assert g.pm_interval > 0.0 and g.battery_low == 0.0, "标定环境：短间隔 + 到 0 才补电"
    assert m.n_agv == inst.n_machines and m.transport_unmapped == GEOMETRY
    with pytest.raises(ValueError, match="口径"):
        calibration_cfg(inst, "teleport")
