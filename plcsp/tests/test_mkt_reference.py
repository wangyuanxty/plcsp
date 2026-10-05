"""MKT 上跑我们参考调度的测试（P4-A Task 3；P4-B Task 4 起**两个报告档、矩阵口径**）。

⚠️ P4-A 时本文件的设计前提是「`trans_time` 未接进仿真 ⟹ `ours` 跑在原始 MK 几何上」；
P4-B 已把矩阵接进仿真 ⟹ 该前提**作废**，对应断言按下述方式更新：

- 绊线 `test_our_column_declares_its_own_basis_and_is_not_mkt_yet` **翻转**为
  `test_our_column_declares_the_mkt_basis`（**改写非删除**，计数不减）；
- `test_our_reference_is_in_the_published_ballpark` 仍留（量级判据仍有效），
  但量级基准从"几何口径的 ~107"变成"矩阵口径的 ~205（报告档 A）/ ~678（报告档 B）"。
"""
from __future__ import annotations

import pytest

from plcsp.env.mkt import MKT_PUBLISHED


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
def test_our_column_declares_the_mkt_basis():
    """⚠️ Review Focus #3 的**绊线翻转**：`ours` 那一列的口径必须**单独**声明，且本批**就是** MKT。

    P4-A 埋的原绊线（`…_and_is_not_mkt_yet`）要求"接上 `trans_time` 后把它改成 MKT"——
    P4-B Task 2 已把矩阵接进 `AgvSim`，故本批按它写明的做法**翻转**：断言改为 `== "MKT"`
    （**改写不删除**，计数不减）。两列的**行程来源**同为矩阵（⚠️ 不是"整表可比"——运输边界与
    仿真器仍不同，只判量级；见 `progress-log.md` §52.6/§52.7）。
    """
    from plcsp.m14_mkt_reference import reference_table

    row = reference_table(("mk01",), n_agv=6)[0]
    assert "ours_benchmark" in row, "表没声明 ours 那一列的口径"
    assert row["ours_benchmark"] == "MKT", (
        "ours 那一列的标签不是 MKT——若矩阵口径被回退成几何口径，这才是正确取值；"
        "若矩阵仍接着，本标签与 `test_rows_carry_tier_and_caliber_labels` 都会红")
    assert row["transport"] == "matrix", "行里自报的行程时间口径不是矩阵——接线被回退了"


@pytest.mark.unit
def test_our_reference_is_in_the_published_ballpark():
    """量级核验：我们的参考调度（最短候选 + 轮询派车）应落在已发表数字的**同一量级**。

    ⚠️ P4-B 起 `ours` 走**矩阵口径**，量级比 P4-A 的几何口径（~107）大数倍：
    报告档 A（机制全关）实测 **205.0**、报告档 B（全开）**711.5**（§22.2 同参数），对照 HGS=153 ⟹ 带 [0.3×, 5.0×] 仍覆盖。
    这一步**只判量级**：若掉回 40 上下 ⟹ 矩阵口径被回退成几何（旧断言想抓的缺陷）；
    若跑出几千 ⟹ 单位错了（矩阵被当成秒或又除了一次换算）。
    """
    from plcsp.m14_mkt_reference import reference_table

    r = reference_table(("mk01",), n_agv=6)[0]
    pub = MKT_PUBLISHED["mk01"]["HGS_JMS2024"]
    assert 0.3 * pub < r["ours"] < 5.0 * pub, (
        f"MK01(MKT) 参考调度 Cmax={r['ours']:.1f}，已发表 HGS={pub}——量级不符")


@pytest.mark.unit
def test_fleet_size_changes_the_reference_result():
    """⚠️ Review Focus #1：换车数**必须真的改变**参考调度的读数——否则车辆数列是装饰。

    （若不同 v 跑出同一 Cmax，说明 `n_agv` 没进 `SimConfig`／没进布局，两个报告档的车数会被当成可比。）
    """
    from plcsp.m14_mkt_reference import reference_table

    a = reference_table(("mk01",), n_agv=1, seeds=1)[0]
    b = reference_table(("mk01",), n_agv=6, seeds=1)[0]
    assert a["n_agv"] == 1 and b["n_agv"] == 6
    assert a["ours"] != b["ours"], "1 台车与 6 台车跑出同一 Cmax——n_agv 没生效"


@pytest.mark.unit
def test_fleet_comparability_is_computed_not_hardcoded():
    """⚠️ Review Focus #1：**哪一列与我们的车数同口径**必须是**算出来的**，不能是写死的脚注。

    `HF2021` 用 2 台、`HGS` 与 `HA-DQN` 用 v=m。摘要表固定写一句"HF2021 与其余列不同口径"，
    在 `--n-agv 2`（HF2021 口径）那张敏感性表上就**恰好说反**：那时 `ours` 也是 2 台，
    而与 `ours` 不同口径的反倒是 HGS/HA-DQN。读者会以为 HGS/HA-DQN 与我们的数可直接比。

    故每行必须带**逐列**的可比性判定，且随 `n_agv` 变。
    """
    from plcsp.m14_mkt_reference import reference_table

    r_m = reference_table(("mk01",), n_agv=6, seeds=1)[0]      # mk01 是 6 机 → v=m 即 6 台
    assert r_m["comparable_fleet"] == {
        "HGS_JMS2024": True, "HA_DQN_CIS2025": True, "HF2021_LAHC": False}, \
        "v=m 这一车数：HGS/HA-DQN 同车数，HF2021(2 台) 不同"

    r_2 = reference_table(("mk01",), n_agv=2, seeds=1)[0]      # 与 HF2021 同口径
    assert r_2["comparable_fleet"] == {
        "HGS_JMS2024": False, "HA_DQN_CIS2025": False, "HF2021_LAHC": True}, \
        "v=2 这一车数：只有 HF2021 同车数——脚注若写死就会说反"


@pytest.mark.unit
def test_published_fleet_specs_are_recorded_machine_readably():
    """⚠️ Review Focus #1：每个方法的车数设定必须**可机读**——否则下游画表/画图又会丢掉它。"""
    from plcsp.env.mkt import MKT_PUBLISHED_AGV

    assert MKT_PUBLISHED_AGV == {"HGS_JMS2024": "m", "HA_DQN_CIS2025": "m", "HF2021_LAHC": 2}


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
