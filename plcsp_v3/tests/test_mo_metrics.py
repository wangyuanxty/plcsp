# -*- coding: utf-8 -*-
"""`hrl/mo_metrics` 的规格测试 —— **手算可验的算例**，不靠实现自证。

用途：按 Shi et al., AEI 65 (2025) 103195 的协议（每实例跑 20 次 → Pareto 解集 → HV/SP）
复现其 Table 8–12。
"""
from __future__ import annotations

import pytest

from plcsp_v3.hrl.mo_metrics import hypervolume_2d, pareto_front, spacing


@pytest.mark.unit
def test_pareto_front_drops_dominated():
    """被支配的点必须被剔除；互不支配的必须全留（两目标、都越小越好）。"""
    pts = [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0), (2.5, 2.5)]
    F = pareto_front(pts)
    got = {tuple(p) for p in F}
    assert got == {(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)}, got
    assert (2.5, 2.5) not in got, "被 (2,2) 支配的点没被剔除"


@pytest.mark.unit
def test_pareto_front_keeps_ties():
    """**完全相同的两个点互不支配**（谁都不"至少一个更小"）⟹ 都留。

    ⚠️ 这条容易写错：若判据写成 `<=` 而不是「≤ 且至少一个 <」，重复点会互相剔除。
    """
    F = pareto_front([(1.0, 1.0), (1.0, 1.0), (2.0, 2.0)])
    assert len(F) == 2, F.tolist()


@pytest.mark.unit
def test_spacing_of_even_front_is_zero():
    """均匀前沿 SP = 0（spacing 的定义：各点到最近邻距离的**标准差**）。"""
    even = [(0.0, 10.0), (5.0, 5.0), (10.0, 0.0)]
    assert spacing(even) == pytest.approx(0.0, abs=1e-12)
    # 单点 / 空
    assert spacing([(1.0, 1.0)]) == 0.0
    assert spacing([]) == 0.0


@pytest.mark.unit
def test_spacing_positive_when_uneven():
    """一个挤在一起、一个孤立 ⟹ SP > 0。"""
    assert spacing([(0.0, 0.0), (0.1, 0.0), (10.0, 10.0)]) > 0.5


@pytest.mark.unit
def test_hypervolume_hand_computed():
    """`{(1,0), (0,1)}` 相对参考点 `(2,2)` 的支配面积 **手算 = 3**。

    逐个矩形：`(1,0)` 支配 `[1,2]×[0,2]` = 2；`(0,1)` 支配 `[0,2]×[1,2]` = 2；
    重叠 `[1,2]×[1,2]` = 1 ⟹ 并集 = 2 + 2 − 1 = **3**。
    """
    assert hypervolume_2d([(1.0, 0.0), (0.0, 1.0)], (2.0, 2.0)) == pytest.approx(3.0)


@pytest.mark.unit
def test_hypervolume_single_point():
    """单点的 HV = 它到参考点的矩形面积。"""
    assert hypervolume_2d([(1.0, 1.0)], (3.0, 4.0)) == pytest.approx(2.0 * 3.0)


@pytest.mark.unit
def test_hypervolume_rejects_bad_reference_point():
    """`ref` 不比前沿差 ⟹ **显式报错**（静默算会把该点当 0 贡献，是假读数）。"""
    with pytest.raises(ValueError, match="不比前沿上所有点都差"):
        hypervolume_2d([(1.0, 0.0), (0.0, 1.0)], (0.5, 2.0))
