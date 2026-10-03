"""格点走廊图与距离矩阵的单元测试（P1a Task 2）。"""
from __future__ import annotations

import networkx as nx
import pytest

from plcsp.env.corridors import build_corridor_graph, dock_distance_matrix
from plcsp.env.layout import sample_layout


@pytest.mark.unit
def test_lattice_is_connected_and_sized():
    lay = sample_layout(6, seed=0)
    g = build_corridor_graph(lay)
    # ⚠️ P4-B Task 2b：图里还有**装卸站**——格点外的第 `n_nodes` 号节点（+ 一条连接段边）。
    #    `grid.n_nodes` 仍是**格点交叉口**数（`nn` 的逐节点特征按它建），故这里是 +1。
    assert g.number_of_nodes() == lay.grid.n_nodes + 1
    assert g.degree(lay.lu.node) == 1, "装卸站应只有一条连接段"
    assert nx.is_connected(g), "格点图必须连通"


@pytest.mark.unit
@pytest.mark.parametrize("n_machines", [5, 6, 10, 14, 20])
def test_all_machine_docks_mutually_reachable(n_machines: int):
    """Review Focus #1：任意两机台 dock 必须可达（不得抛 NetworkXNoPath）。"""
    lay = sample_layout(n_machines, seed=0)
    g = build_corridor_graph(lay)
    dm = dock_distance_matrix(g)
    nodes = [m.dock_node for m in lay.machines]
    for a in nodes:
        for b in nodes:
            assert dm[a, b] < float("inf"), f"dock {a} → {b} 不可达"


@pytest.mark.unit
def test_lattice_distance_matches_span_on_full_grid():
    """相邻节点的距离应等于段长（机位 + 通道）。"""
    lay = sample_layout(6, seed=0)
    g = build_corridor_graph(lay)
    s = lay.grid
    dm = dock_distance_matrix(g)
    assert dm[s.node_id(0, 0), s.node_id(0, 1)] == pytest.approx(s.cell_w + s.aisle_w)
    assert dm[s.node_id(0, 0), s.node_id(1, 0)] == pytest.approx(s.cell_h + s.aisle_w)
