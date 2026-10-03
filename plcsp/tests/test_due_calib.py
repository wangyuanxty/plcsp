"""交期标定的测试（⑧ 重设计 Task 2）。"""
from __future__ import annotations

import pytest

from plcsp.env.instances import load_mk

MK10 = [f"mk{i:02d}" for i in range(1, 11)]
IMPROVE = 0.88


@pytest.mark.unit
@pytest.mark.parametrize("name", MK10)
def test_calibrated_due_dates_stay_live_after_improvement(name: str):
    """⭐⭐ **本计划的判据测试**：交期在策略改进 12% 后**不得退化**。

    旧口径 `d_j = τ·M_ref` 正是在这里死的：训练后 makespan 89.5 < d_j=93.08 ⟹ TWT ≡ 0。
    新口径下，参考策略与"改进 12% 的策略"的误期率都必须落在 (0.15, 0.85)。
    """
    from plcsp.env.des import SimConfig, rollout
    from plcsp.env.due_dates import due_dates_for

    inst = load_mk(name)
    d = due_dates_for(inst)
    C = rollout(inst, seed_chain=0, cfg=SimConfig())["completes"]
    n = inst.n_jobs
    ref = sum(1 for j in C if C[j] > d[j]) / n
    imp = sum(1 for j in C if C[j] * IMPROVE > d[j]) / n
    assert 0.15 < ref < 0.85, f"{name} 参考策略误期率 {ref:.2f} 退化"
    assert 0.15 < imp < 0.85, f"{name} 改进 12% 后误期率 {imp:.2f} 退化（目标失效）"


@pytest.mark.unit
@pytest.mark.parametrize("name", MK10)
def test_twt_is_positive_after_improvement(name: str):
    """TWT 在改进后必须**非零**——这是"目标还活着"的直接读数。"""
    from plcsp.env.des import SimConfig, rollout
    from plcsp.env.due_dates import due_dates_for

    inst = load_mk(name)
    d = due_dates_for(inst)
    C = rollout(inst, seed_chain=0, cfg=SimConfig())["completes"]
    twt = sum(max(0.0, C[j] * IMPROVE - d[j]) for j in C)
    assert twt > 0.0, f"{name}: 改进 12% 后 TWT 恒 0——目标又死了"


@pytest.mark.unit
def test_uncalibrated_instance_raises():
    """⚠️ Review Focus #3：没标定的实例必须**显式报错**，不得静默取默认值。

    ⚠️ 只收 **`ValueError`**（原写作 `(KeyError, ValueError)`）：`KeyError` 会让本测试在
    "显式报错退化成深处的字典 KeyError"时**照样绿**——而那正是 Review Focus #3 明令禁止的形态。
    收窄后，任何把显式报错改回裸 `TF_RDD[key]` 的改动都会在这里变红。
    """
    from plcsp.env.due_dates import due_dates_for
    from plcsp.env.instances import gen_random

    with pytest.raises(ValueError, match="未标定"):
        due_dates_for(gen_random(4, 3, seed=0))


@pytest.mark.unit
@pytest.mark.parametrize("name", MK10)
def test_frozen_table_is_reproduced_by_the_calibration_script(name: str):
    """⚠️ Review Focus #6：`TF_RDD` 必须是标定脚本的**可复现产物**，不是手抄魔数。

    判据：冻结的表项 == 脚本在 (τ,R) 网格上按标定规则重算的 argmin。改了规则或网格
    而没重跑脚本，这条会红。
    """
    from plcsp.env.due_dates import TF_RDD
    from plcsp.m16_due_calib import calibrate

    tau, due_range = calibrate(load_mk(name))
    tab_tau, tab_r = TF_RDD[name]
    assert tab_tau == pytest.approx(tau, abs=1e-9), f"{name}: 表 τ={tab_tau} ≠ 脚本 {tau}"
    assert tab_r == pytest.approx(due_range, abs=1e-9), f"{name}: 表 R={tab_r} ≠ 脚本 {due_range}"


@pytest.mark.unit
def test_table_covers_exactly_the_mk_series():
    """表必须**恰好**覆盖 MK01–MK10：少一个实例仿真会报错，多一个则是手抄残留。"""
    from plcsp.env.due_dates import TF_RDD

    assert sorted(TF_RDD) == MK10
