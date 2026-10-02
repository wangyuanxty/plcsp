"""网格布局生成器的单元测试（P1a Task 1）。"""
from __future__ import annotations

import pytest

from plcsp.env.layout import GridSpec, sample_layout


@pytest.mark.unit
def test_gridspec_node_count_and_ids():
    spec = GridSpec(n_rows=2, n_cols=3)
    assert spec.n_nodes == (2 + 1) * (3 + 1) == 12
    assert spec.node_id(0, 0) == 0
    assert spec.node_id(0, 3) == 3
    assert spec.node_id(1, 0) == 4
    assert spec.node_rc(spec.node_id(2, 3)) == (2, 3)


@pytest.mark.unit
@pytest.mark.parametrize("n_machines", [5, 6, 10, 14, 20])
def test_layout_places_all_machines_on_distinct_nodes(n_machines: int):
    lay = sample_layout(n_machines, seed=0)
    assert lay.n_machines == n_machines
    nodes = [m.dock_node for m in lay.machines]
    assert len(set(nodes)) == n_machines, "机台 dock 节点必须互不重合"
    assert all(0 <= n < lay.grid.n_nodes for n in nodes)


@pytest.mark.unit
def test_layout_has_at_least_two_distinct_chargers():
    lay = sample_layout(6, seed=0, n_chargers=2)
    assert len(lay.chargers) >= 2
    nodes = [c.node for c in lay.chargers]
    assert len(set(nodes)) == len(nodes), "充电桩不得落在同一节点"
    machine_nodes = {m.dock_node for m in lay.machines}
    assert not (set(nodes) & machine_nodes), "充电桩不得与机台 dock 重合"


@pytest.mark.unit
def test_layout_is_deterministic():
    a, b = sample_layout(10, seed=7), sample_layout(10, seed=7)
    assert [m.dock_node for m in a.machines] == [m.dock_node for m in b.machines]
    assert [c.node for c in a.chargers] == [c.node for c in b.chargers]
