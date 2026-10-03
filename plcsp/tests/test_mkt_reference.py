"""MKT 上跑我们参考调度的测试（P4-A Task 3）。

⚠️ **本批不把 `trans_time` 接进仿真**（那是 P4-B 的设计决定）——故 `ours` 跑在**原始 MK 几何**下，
只在**量级**上与已发表数字可比。本文件的断言全部据此设计。
"""
from __future__ import annotations

import pytest

from plcsp.env.mkt import MKT_PUBLISHED, load_mkt


@pytest.mark.unit
def test_reference_table_has_fleet_column():
    """⚠️ Review Focus #1：产出的表**必须带"车辆数"列**——否则不同车数的数字会混在一起比。"""
    from plcsp.m14_mkt_reference import reference_table

    rows = reference_table(("mk01",), n_agv=2)
    assert rows, "表为空"
    assert "n_agv" in rows[0], f"表缺车辆数列：{list(rows[0])}"
    assert rows[0]["n_agv"] == 2


@pytest.mark.unit
def test_reference_table_labels_the_benchmark():
    """⚠️ Review Focus #3：表必须自报对照列用的是 MKT（不是原始 MK），防混表。"""
    from plcsp.m14_mkt_reference import reference_table

    rows = reference_table(("mk01",), n_agv=6)
    assert rows[0]["benchmark"] == "MKT"


@pytest.mark.unit
def test_our_column_declares_its_own_basis_and_is_not_mkt_yet():
    """⚠️ Review Focus #3 的**绊线**：`ours` 那一列的口径必须**单独**声明，且本批**不是** MKT。

    本批 `ours` 跑在**原始 MK 几何**上（`trans_time` 还没接进 `AgvSim`），而对照列是 MKT——
    **两者不同口径**，只判量级。若把 `ours` 也标成 MKT，读者会以为这是可直接比的同口径数字。

    ⭐ **这个断言是故意会红的**：P4-B 把 `trans_time` 接进仿真后，`ours_benchmark` 必须改成
    `"MKT"`，那时**本测试会失败**，逼着改的人同时更新表与文档——而不是让标签悄悄留在旧值上。
    """
    from plcsp.m14_mkt_reference import reference_table

    row = reference_table(("mk01",), n_agv=6)[0]
    assert "ours_benchmark" in row, "表没声明 ours 那一列的口径"
    assert row["ours_benchmark"] != "MKT", (
        "ours 已标成 MKT——若 P4-B 真的接上了 trans_time，请把本断言一并改为 == 'MKT'；"
        "若没接上，这个标签就是错的（Review Focus #3）")


@pytest.mark.unit
def test_our_reference_is_in_the_published_ballpark():
    """量级核验：我们的参考调度（最短候选 + 轮询派车）应落在已发表数字的**同一量级**。

    这一步**只判量级**（训练后的 DRL 当然比未训练的参考调度好）：
    若我们跑出 40 上下 ⟹ 行程时间根本没接上；若跑出几千 ⟹ 单位错了。
    """
    from plcsp.m14_mkt_reference import reference_table

    r = reference_table(("mk01",), n_agv=6)[0]
    pub = MKT_PUBLISHED["mk01"]["HGS_JMS2024"]
    assert 0.3 * pub < r["ours"] < 5.0 * pub, (
        f"MK01(MKT) 参考调度 Cmax={r['ours']:.1f}，已发表 HGS={pub}——量级不符")


@pytest.mark.unit
def test_fleet_size_changes_the_reference_result():
    """⚠️ Review Focus #1：换车数**必须真的改变**参考调度的读数——否则车辆数列是装饰。

    （若不同 v 跑出同一 Cmax，说明 `n_agv` 没进 `SimConfig`／没进布局，两档车数会被当成可比。）
    """
    from plcsp.m14_mkt_reference import reference_table

    a = reference_table(("mk01",), n_agv=1, seeds=1)[0]
    b = reference_table(("mk01",), n_agv=6, seeds=1)[0]
    assert a["n_agv"] == 1 and b["n_agv"] == 6
    assert a["ours"] != b["ours"], "1 台车与 6 台车跑出同一 Cmax——n_agv 没生效"


@pytest.mark.unit
def test_benchmark_instances_do_not_get_mixed():
    """⚠️ Review Focus #3：**MKT 的数字不得与原始 MK 的数字混进同一张表**。

    `MKT_PUBLISHED` 是 MKT 口径；原始 MK 的 BKS 是另一套（MK01=40 vs MKT01 的 153）。
    本测试挡住"某天有人把 `MK_OPTIMAL` 的列加进这张表"。
    """
    from plcsp.env.instances import MK_OPTIMAL
    from plcsp.m14_mkt_reference import reference_table

    row = reference_table(("mk01",), n_agv=6)[0]
    assert "MK_BKS" not in row and "bks" not in row, (
        "表里出现了原始 MK 的 BKS 列——两套口径不得同表（Review Focus #3）")
    # 对照列全是 MKT 量级：都远离原始 MK01 的 BKS=40
    for col in ("HGS_JMS2024", "HA_DQN_CIS2025", "HF2021_LAHC"):
        assert row[col] > 2.0 * MK_OPTIMAL["mk01"], f"{col} 疑似混入了原始 MK 的数"
