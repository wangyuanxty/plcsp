"""行程时间口径对象的测试（P4-B Task 1）。

口径的两条硬约束来自数据本身（`plcsp/data/mkt/README.md`）：矩阵是 `(m+1)×(m+1)`、
**第 0 行/列是装卸站（LU）**、机台 i ↔ 下标 i+1；矩阵**已是分钟**、且**非对称**
（`m=4` 是唯一例外）。本文件只测**对象本身**，接进仿真的测试在 `test_transport_wiring.py`。
"""
from __future__ import annotations

import pytest

from plcsp.env.layout import sample_layout
from plcsp.env.mkt import load_mkt_layout
from plcsp.env.transport import TransportCaliber


def _layout(m: int, seed: int = 0):
    return sample_layout(m, seed=seed, n_agv=m)


@pytest.mark.unit
def test_geometry_caliber_has_no_matrix():
    """几何口径是**缺省**：原始 MK 实例走它，不得带上任何矩阵。"""
    c = TransportCaliber.geometry()
    assert c.mode == "geometry" and c.matrix is None and not c.node_slot


@pytest.mark.unit
def test_dropped_lu_matrix_is_rejected_loudly():
    """⚠️ Review Focus #2：喂进**已丢 LU** 的 m×m 矩阵必须**显式报错**。

    `load_mkt_layout(m)` 的默认返回是丢过 LU 的 `m×m`；若它被当全矩阵用，机台 i 会整体错行
    一行——数字看着合理、全程不报错，正是 P4-A 同类 off-by-one 的复发形态。
    """
    lay = _layout(6)
    with pytest.raises(ValueError, match="LU"):
        TransportCaliber.from_matrix(load_mkt_layout(6), lay)      # 丢 LU 的 6×6


@pytest.mark.unit
def test_non_square_matrix_is_rejected():
    """裁错方向（`(m+1)×m` 之类）也要在**构造时**就报错，不得等到查表才崩。"""
    lay = _layout(6)
    bad = load_mkt_layout(6, drop_lu=False)[:, 1:]                  # 7×6
    with pytest.raises(ValueError, match="方阵"):
        TransportCaliber.from_matrix(bad, lay)


@pytest.mark.unit
def test_slot_zero_is_the_load_unload_station_and_nothing_else():
    """⚠️ 下标 0 是装卸站（LU）——**恰好一个**节点映射到它，且那是装卸站节点。

    ⚠️ **绊线翻转**（P4-B Task 2b）：Task 1 时本项目还没有 LU 运输，本测试当时钉的是
    "**没有**任何节点映射到下标 0"；用户裁定本批就加装卸站后，这一位**必须有**消费者。
    本测试当时就写明"加了 LU 运输它会红，逼人同时改口径与文档"——现在按它的要求翻转：
    下标 0 只能属于装卸站，机台仍从下标 1 起（少加一会让这条红）。
    """
    lay = _layout(6)
    c = TransportCaliber.from_matrix(load_mkt_layout(6, drop_lu=False), lay)
    slots = set(c.node_slot.values())
    assert slots == set(range(7)), "机台必须占满 1..6（+ 装卸站的 0）"
    assert c.slot_of(lay.lu.node) == 0, "装卸站没有映射到下标 0"
    assert list(c.node_slot.values()).count(0) == 1
    for i, mp in enumerate(lay.machines):
        assert c.slot_of(mp.dock_node) == i + 1


@pytest.mark.unit
@pytest.mark.parametrize("m", [5, 6, 15])
def test_machine_i_maps_to_matrix_row_i_plus_one(m: int):
    """⚠️ Review Focus #2：机台 i ↔ 矩阵下标 **i+1**（+1 是 LU 占的第 0 位）。"""
    lay = _layout(m)
    c = TransportCaliber.from_matrix(load_mkt_layout(m, drop_lu=False), lay)
    for i, mp in enumerate(lay.machines):
        assert c.slot_of(mp.dock_node) == i + 1, f"机台 {i} 的矩阵下标不是 {i + 1}"


@pytest.mark.unit
def test_matrix_values_are_used_verbatim():
    """⚠️ Review Focus #1 的对象侧：查表值必须**逐位等于**文件里的数（分钟），不得有任何换算。"""
    lay = _layout(6)
    raw = load_mkt_layout(6, drop_lu=False)
    c = TransportCaliber.from_matrix(raw, lay)
    for i in range(6):
        for j in range(6):
            assert c.minutes(lay.machines[i].dock_node, lay.machines[j].dock_node) == float(raw[i + 1, j + 1])


@pytest.mark.unit
def test_asymmetric_entry_survives_the_node_mapping():
    """⚠️ Review Focus #2：**保序查表**——用一条实测非对称项钉住（转置/错行都会红）。

    10 机布局：1 基机台号 M[1][2] = 3 而 M[2][1] = 12（`progress-log §19.7d`、P4-A F3 已核）。
    本测试走**节点→下标**这条路（本批的新增映射层），比直接读文件多挡一层。
    """
    lay = _layout(10)
    c = TransportCaliber.from_matrix(load_mkt_layout(10, drop_lu=False), lay)
    a, b = lay.machines[0].dock_node, lay.machines[1].dock_node
    assert c.minutes(a, b) == pytest.approx(3.0)
    assert c.minutes(b, a) == pytest.approx(12.0)
    assert c.minutes(a, b) != c.minutes(b, a), "矩阵被对称化了"


@pytest.mark.unit
def test_unmapped_node_is_reported_not_guessed():
    """⚠️ Review Focus #5：充电桩在矩阵里**没有对应项** ⟹ 返回 None，**不得**猜最近机台或取 0。"""
    lay = _layout(6)
    c = TransportCaliber.from_matrix(load_mkt_layout(6, drop_lu=False), lay)
    assert lay.chargers, "该布局应有充电桩（本测试的前提）"
    charger_node = lay.chargers[0].node
    assert c.slot_of(charger_node) is None
    assert c.minutes(lay.machines[0].dock_node, charger_node) is None
    assert c.minutes(charger_node, lay.machines[0].dock_node) is None


@pytest.mark.unit
def test_unknown_unmapped_policy_is_rejected():
    """策略字符串写错必须当场报错——否则运行期的"降级"会变成静默行为。"""
    lay = _layout(6)
    with pytest.raises(ValueError, match="未映射"):
        TransportCaliber.from_matrix(load_mkt_layout(6, drop_lu=False), lay, unmapped="fallback")


@pytest.mark.unit
def test_geometry_caliber_has_no_matrix_lookup():
    """几何口径下查表是**编程错误**（调用方必须先判 mode）——不得静默返回几何值。"""
    with pytest.raises(ValueError, match="矩阵"):
        TransportCaliber.geometry().minutes(0, 1)


@pytest.mark.unit
def test_duplicate_dock_nodes_are_rejected():
    """机台共用 dock 节点时"节点→机台"不是函数，必须报错（否则整张表错行且难查）。"""

    class _FakeM:
        def __init__(self, node):
            self.dock_node = node

    class _FakeLayout:
        machines = [_FakeM(3), _FakeM(3)]

    with pytest.raises(ValueError, match="重复"):
        TransportCaliber.from_matrix(load_mkt_layout(4, drop_lu=False), _FakeLayout())
