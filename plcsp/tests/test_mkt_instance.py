"""MKT 实例适配器的测试（P4-A Task 2）。

口径：**MKT = 原始 MK 的加工数据（一字不改）+ 行程时间矩阵 + 车辆数**（`progress-log §19.7d`）。
"""
from __future__ import annotations

import numpy as np
import pytest

from plcsp.env.instances import load_mk
from plcsp.env.mkt import MKT_PUBLISHED, load_mkt

# ⚠️ 实测的「MK 实例 → 机台数」映射（本机 `load_mk` 读官方 .fjs 得到）。
# 注意 **mk06 也是 15 机**（10 工件 × 15 机台），与 mk10 共用同一份 15 机布局——
# 布局文件是按**机台数**命名的，不是按实例。故这里的 m 是「该实例需要哪份布局」。
MK_MACHINES = {"mk01": 6, "mk02": 6, "mk03": 8, "mk04": 8, "mk05": 4,
               "mk06": 15, "mk07": 5, "mk08": 10, "mk09": 10, "mk10": 15}


@pytest.mark.unit
@pytest.mark.parametrize("name,m", [("mk01", 6), ("mk07", 5), ("mk10", 15)])
def test_processing_data_is_identical_to_raw_mk(name: str, m: int):
    """⚠️ 核心断言：**MKT 的加工数据与原始 MK 逐位相同**（§19.7d 已核实的事实）。

    MKT 相对 MK 的增量**只有**行程时间矩阵与车辆数——若这里不等，说明适配器动了加工数据。
    """
    inst = load_mkt(name)
    base = load_mk(name)
    assert inst.base.n_jobs == base.n_jobs
    assert inst.base.n_machines == base.n_machines
    for j, (a, b) in enumerate(zip(inst.base.jobs, base.jobs)):
        assert a == b, f"{name} 作业 {j} 的工序数据被改动了"


@pytest.mark.unit
@pytest.mark.parametrize("name,m", [("mk01", 6), ("mk07", 5), ("mk10", 15)])
def test_transport_matrix_matches_machine_count(name: str, m: int):
    """行程时间矩阵维度必须等于**该实例实际的机台数**（Review Focus #6）。"""
    inst = load_mkt(name)
    assert inst.trans_time.shape == (m, m)
    assert inst.layout_m == m


@pytest.mark.unit
@pytest.mark.parametrize("name,m", sorted(MK_MACHINES.items()))
def test_every_published_mk_instance_resolves_its_layout(name: str, m: int):
    """⚠️ Review Focus #6：**10 个 MK 实例全部**要能按自己的机台数取到布局——不能只测 3 个。

    布局是按机台数命名的，10 个实例只用到 {4,5,6,8,10,15} 六种；其中 **mk06 与 mk10 都是 15 机**，
    共用同一份 `15_machine_layout.txt`。若哪天 `load_mkt` 改成按实例名找文件，这个测试必须红。
    """
    inst = load_mkt(name)
    assert inst.layout_m == m
    assert inst.trans_time.shape == (m, m)
    assert inst.base.n_machines == m, "适配器选的布局机台数与实例机台数不符"
    # 这六种机台数（4/5/6/8/10/15）实测对角全 0——17 是唯一已知例外，此处用不到
    assert np.allclose(np.diag(inst.trans_time), 0.0), f"{name} 的行程矩阵对角非 0"


@pytest.mark.unit
def test_default_fleet_is_v_equals_m():
    """⚠️ Review Focus #1：默认车辆数 = **m**（HGS 与 HA-DQN 的设定，由 HA-DQN 表 11 的
    J-M-A 列读出：10-6-6 / 20-5-5 / 20-15-15）。"""
    assert load_mkt("mk01").n_agv == 6
    assert load_mkt("mk07").n_agv == 5
    assert load_mkt("mk10").n_agv == 15


@pytest.mark.unit
def test_fleet_size_is_explicit_in_the_result():
    """⚠️ Review Focus #1：车辆数**只能显式传**，且结果里带着它——不同车数下 Cmax 不可比。"""
    a = load_mkt("mk01", n_agv=2)
    assert a.n_agv == 2
    assert a.n_agv != load_mkt("mk01").n_agv


@pytest.mark.unit
def test_instance_name_carries_fleet_and_layout():
    """⚠️ Review Focus #1：结果自带的 `name` 必须**把车辆数与机台数写出来**——
    否则两档车数的数混在一起时无从分辨（HF2021 的 2 台 vs HGS 的 v=m）。"""
    s = load_mkt("mk01", n_agv=2).name
    assert "v=2" in s and "m=6" in s


@pytest.mark.unit
def test_published_numbers_are_labelled_by_benchmark():
    """⚠️ Review Focus #3：已发表数字必须**标注基准**，防与原始 MK 混表。

    原始 MK01 的 BKS 是 40，而 MKT01 的已发表数字是 97–187——两套口径。
    """
    ref = MKT_PUBLISHED["mk01"]
    assert set(ref) >= {"HGS_JMS2024", "HA_DQN_CIS2025", "HF2021_LAHC"}
    assert 90.0 < ref["HGS_JMS2024"] < 200.0, "HGS 在 MKT01 上报 153，不该落在原始 MK 的量级"
    assert ref["HGS_JMS2024"] > 40.0, "MKT01 的数字不得与原始 MK01 的 BKS(40) 混淆"


@pytest.mark.unit
def test_published_table_covers_all_ten_instances_and_is_mkt_scale():
    """⚠️ Review Focus #3：**10 个实例都要有**，且**每一个**都必须远高于原始 MK 的 BKS。

    原始 MK 的 BKS 是 27–523；MKT 因加了运输普遍更高。**逐实例**核对低位：
    若哪一行漏了运输、混进了原始 MK 的数，它多半会掉到 BKS 附近。
    """
    from plcsp.env.instances import MK_OPTIMAL

    assert set(MKT_PUBLISHED) == {f"mk{i:02d}" for i in range(1, 11)}
    for name, row in MKT_PUBLISHED.items():
        assert set(row) == {"HGS_JMS2024", "HA_DQN_CIS2025", "HF2021_LAHC"}, f"{name} 列不全"
        # HF2021 的 LAHC 是元启发式参照，走得最远（值最小）；仍必须高于原始 BKS
        assert row["HF2021_LAHC"] > MK_OPTIMAL[name], (
            f"{name}: HF2021={row['HF2021_LAHC']} 不高于原始 MK 的 BKS={MK_OPTIMAL[name]}"
            "——疑似把原始 MK 的数字混了进来")
        assert row["HGS_JMS2024"] > row["HA_DQN_CIS2025"], \
            f"{name}: 表 IV(HGS) 与表 8(HA-DQN) 的相对关系反了——数字可能串行"
