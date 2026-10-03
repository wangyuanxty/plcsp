"""MKT（矩阵）口径的交期标定测试（P4-B Task 3）。

为什么要有这张表：矩阵口径下参考 makespan 大数倍 ⟹ 交期若沿用几何口径的 (τ,R)，
误期率会整体塌到 0（TWT 变死目标，`progress-log §20.4-2` 的老病）。
"""
from __future__ import annotations

import pytest

from plcsp.env.mkt import load_mkt

MK10 = [f"mk{i:02d}" for i in range(1, 11)]


@pytest.mark.unit
def test_matrix_table_exists_and_is_not_the_geometry_table():
    """⚠️ Review Focus #6：两张表**必须都在**，且值不得相同——口径不同 ⟹ 尺不同。"""
    from plcsp.env.due_dates import TF_RDD, TF_RDD_MATRIX

    assert sorted(TF_RDD) == MK10
    assert sorted(TF_RDD_MATRIX) == MK10, "矩阵口径的表没落盘（或缺项）"
    same = [n for n in MK10 if TF_RDD[n] == TF_RDD_MATRIX[n]]
    assert not same, f"这些实例两个口径的 (τ,R) 完全相同（疑似照抄）：{same}"


@pytest.mark.unit
def test_mkt_due_dates_actually_read_the_matrix_table():
    """⚠️ Review Focus #6 的**直接判据**：MKT 实例算出的交期必须等于"用矩阵表的 (τ,R)"那一组。

    只断言"两表不同值"还不够——若 `due_dates_for` 读了另一张表，值也会不同且不报错。
    这里把"用的是哪组参数"钉死：等于矩阵表的那组，且**不等于**几何表的那组。
    """
    from plcsp.env.due_dates import TF_RDD, TF_RDD_MATRIX, due_dates_for, tf_rdd_due_dates

    mkt = load_mkt("mk01").base
    tau_m, r_m = TF_RDD_MATRIX["mk01"]
    tau_g, r_g = TF_RDD["mk01"]
    assert due_dates_for(mkt) == tf_rdd_due_dates(mkt, tau_m, r_m)
    assert due_dates_for(mkt) != tf_rdd_due_dates(mkt, tau_g, r_g)


@pytest.mark.unit
@pytest.mark.parametrize("name", MK10)
def test_matrix_table_is_reproduced_by_the_script(name: str):
    """⚠️ Review Focus：`TF_RDD_MATRIX` 必须是标定脚本的**可复现产物**，不是手抄魔数。"""
    from plcsp.env.des import SimConfig
    from plcsp.env.due_dates import TF_RDD_MATRIX
    from plcsp.m16_due_calib import calibrate

    mkt = load_mkt(name)
    cfg = SimConfig(n_agv=mkt.layout_m, transport_unmapped="geometry")
    tau, r = calibrate(mkt.base, cfg=cfg)
    tab_tau, tab_r = TF_RDD_MATRIX[name]
    assert tab_tau == pytest.approx(tau, abs=1e-9), f"{name}: 表 τ={tab_tau} ≠ 脚本 {tau}"
    assert tab_r == pytest.approx(r, abs=1e-9), f"{name}: 表 R={tab_r} ≠ 脚本 {r}"


@pytest.mark.unit
@pytest.mark.parametrize("name", ["mk01", "mk07", "mk10"])
def test_mkt_due_dates_make_twt_alive_at_the_reference_point(name: str):
    """判据：矩阵口径下，**参考策略**的误期率必须落在标定带内（否则目标又死了）。

    这条与几何口径的 `test_calibrated_due_dates_stay_live_after_improvement` 同型，
    只是换了口径——**不得**复用几何表来通过它。
    """
    from plcsp.env.des import SimConfig, rollout
    from plcsp.env.due_dates import due_dates_for

    mkt = load_mkt(name)
    d = due_dates_for(mkt.base)
    cfg = SimConfig(n_agv=mkt.layout_m, transport_unmapped="geometry")
    C = rollout(mkt.base, seed_chain=0, cfg=cfg)["completes"]
    rate = sum(1 for j in C if C[j] > d[j]) / mkt.base.n_jobs
    assert 0.15 < rate < 0.85, f"{name}: 矩阵口径参考误期率 {rate:.2f}——交期又成死目标"
