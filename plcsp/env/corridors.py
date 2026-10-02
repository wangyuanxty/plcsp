"""格点走廊图与距离矩阵（spec §3.2）。

节点 = 通道交叉口（编号 0..n_nodes-1）；边 = 横/竖相邻节点，权重 = 段长 [m]。
机台/充电桩/缓冲都**挂在**某个通道节点上（`dock_node` / `node`），故距离矩阵按节点算。

与上一代的区别：旧版是"相邻 dock + 首尾闭环"的**单环**，且连线可能穿过机台；
新版通道在格子**之间**、机台在格子**之内**，两问题同时消失（无需栅格 A*）。

2026-10-02：区段管制**已按参数化粒度恢复**（见 `des.SimConfig.zone_granularity`）——
`shortest_node_path` 用于取 AGV 实际经过的节点序列。路线决策（选哪条路）仍未恢复。
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .layout import Layout


def build_corridor_graph(layout: Layout) -> nx.Graph:
    """格点图：节点带 pos 属性；横/竖相邻边，权重 = 欧氏段长 [m]。"""
    spec = layout.grid
    g = nx.Graph()
    for r in range(spec.n_rows + 1):
        for c in range(spec.n_cols + 1):
            g.add_node(spec.node_id(r, c), pos=spec.node_xy(r, c))
    for r in range(spec.n_rows + 1):
        for c in range(spec.n_cols + 1):
            u = spec.node_id(r, c)
            if c + 1 <= spec.n_cols:                     # 横向
                v = spec.node_id(r, c + 1)
                g.add_edge(u, v, weight=abs(spec.node_xy(r, c + 1)[0] - spec.node_xy(r, c)[0]))
            if r + 1 <= spec.n_rows:                     # 纵向
                v = spec.node_id(r + 1, c)
                g.add_edge(u, v, weight=abs(spec.node_xy(r + 1, c)[1] - spec.node_xy(r, c)[1]))
    return g


def dock_distance_matrix(g: nx.Graph) -> np.ndarray:
    """节点两两最短路距离矩阵 (n_nodes, n_nodes) [m]。不可达处为 inf。"""
    nodes = sorted(g.nodes)
    n = len(nodes)
    dm = np.full((n, n), np.inf, dtype=float)
    idx = {node: i for i, node in enumerate(nodes)}
    for src in nodes:
        lengths = nx.single_source_dijkstra_path_length(g, src, weight="weight")
        for dst, dist in lengths.items():
            dm[idx[src], idx[dst]] = float(dist)
    return dm


def shortest_node_path(g: nx.Graph, src: int, dst: int) -> list[int]:
    """两节点间最短路的节点序列（含首尾）——AGV 逐段申请区段的依据。"""
    return list(nx.dijkstra_path(g, src, dst, weight="weight"))
