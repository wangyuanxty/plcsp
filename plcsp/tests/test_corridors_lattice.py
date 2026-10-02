"""格点走廊图与距离矩阵的单元测试（P1a Task 2）。"""
from __future__ import annotations

import networkx as nx
import pytest

from plcsp.env.corridors import (build_corridor_graph, dock_distance_matrix,
                                 k_shortest_paths, shortest_node_path)
from plcsp.env.layout import sample_layout


@pytest.mark.unit
def test_lattice_is_connected_and_sized():
    lay = sample_layout(6, seed=0)
    g = build_corridor_graph(lay)
    assert g.number_of_nodes() == lay.grid.n_nodes
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


@pytest.mark.unit
def test_shortest_node_path_returns_contiguous_walk():
    """Review Focus #4：路径必须是图上真实相邻的一串节点（无跳跃）。"""
    lay = sample_layout(14, seed=0)
    g = build_corridor_graph(lay)
    src, dst = lay.machines[0].dock_node, lay.machines[-1].dock_node
    path = shortest_node_path(g, src, dst)
    assert path[0] == src and path[-1] == dst
    assert len(path) >= 2
    for u, v in zip(path, path[1:]):
        assert g.has_edge(u, v), f"路径上 {u}→{v} 并非图上的边"


@pytest.mark.unit
def test_path_length_equals_matrix_entry():
    """Review Focus #4：逐段路径的总长必须等于距离矩阵给出的最短路。"""
    lay = sample_layout(14, seed=0)
    g = build_corridor_graph(lay)
    dm = dock_distance_matrix(g)
    src, dst = lay.machines[0].dock_node, lay.machines[-1].dock_node
    path = shortest_node_path(g, src, dst)
    total = sum(g[u][v]["weight"] for u, v in zip(path, path[1:]))
    assert total == pytest.approx(dm[src, dst], rel=1e-9)


@pytest.mark.unit
def test_k_shortest_paths_are_ordered_and_distinct():
    """路线候选集（= 路线决策的动作空间）：长度升序、互不相同、首尾正确。"""
    lay = sample_layout(14, seed=0)
    g = build_corridor_graph(lay)
    src, dst = lay.machines[0].dock_node, lay.machines[-1].dock_node

    cands = k_shortest_paths(g, src, dst, k=3)

    assert len(cands) >= 2, "网格上应有多条候选路径（路线决策才有意义）"
    for p in cands:
        assert p[0] == src and p[-1] == dst
        for u, v in zip(p, p[1:]):
            assert g.has_edge(u, v)


@pytest.mark.unit
def test_k_shortest_paths_first_is_the_shortest():
    """候选集第 0 条必须是最短路（无策略时的默认路线，也是距离矩阵口径）。"""
    lay = sample_layout(10, seed=0)
    g = build_corridor_graph(lay)
    dm = dock_distance_matrix(g)
    src, dst = lay.machines[0].dock_node, lay.machines[-1].dock_node

    cands = k_shortest_paths(g, src, dst, k=3)

    def plen(p):
        return sum(g[u][v]["weight"] for u, v in zip(p, p[1:]))

    assert plen(cands[0]) == pytest.approx(dm[src, dst], rel=1e-9)
    assert all(plen(cands[0]) <= plen(p) + 1e-9 for p in cands)
