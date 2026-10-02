"""距离矩阵构建：机台装卸点（dock）连接图 + 最短路距离（《方法设计文档》§2.1 nav 段）。

⚠️ 建模边界（v0 简化，论文须如实声明）：本模块构建的**不是物理走廊、不是避障路径规划**，
而是**机台装卸点之间的连接图**：节点 = 各机台 dock，边 = 相邻 dock 的直线连接（权重 = 欧氏距离）
+ 首尾闭环。因此：
- 链上连线在几何上可能**穿过机台矩形**（无净空判定）——AGV 实际不可通行；
- 运输距离 = 沿该链的最短路（非栅格 A* 避障路径）→ 运输时间可能被低估；
- 相对结论不受影响（所有方法共用同一距离矩阵，一视同仁）。
v1 升级方向：栅格化 + 机台为障碍的 A*（或曼哈顿距离折中）。
拓扑差异化：不同 layout_type 的链结构不同 → 距离矩阵不同（零样本拓扑轴的物理来源）。
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from .layout import Layout


def build_corridor_graph(layout: Layout) -> nx.Graph:
    """走廊链图：节点=机台 dock（d0..dN-1）；边=链上相邻（权重=欧氏距离）+ 首尾闭环。"""
    g = nx.Graph()
    pts = [m.dock for m in layout.machines]
    for i in range(len(pts) - 1):
        g.add_node(f"d{i}", pos=pts[i])
        g.add_edge(f"d{i}", f"d{i + 1}", weight=float(np.hypot(pts[i][0] - pts[i + 1][0],
                                                               pts[i][1] - pts[i + 1][1])))
    if pts:
        g.add_node(f"d{len(pts) - 1}", pos=pts[-1])
    if len(pts) >= 2:
        g.add_edge(f"d{len(pts) - 1}", "d0", weight=float(np.hypot(pts[-1][0] - pts[0][0],
                                                                    pts[-1][1] - pts[0][1])))
    return g


def dock_distance_matrix(g: nx.Graph, extra_nodes: dict[str, tuple[float, float]] | None = None) -> np.ndarray:
    """机台装卸点两两 A* 最短路矩阵（加锁 head/tail 节点供 AGV 起终点）。"""
    dnodes = [n for n in g.nodes if n.startswith("d")]
    nodes = dnodes + (list(extra_nodes or {}))
    dm = np.zeros((len(nodes), len(nodes)))
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if i != j and g.has_node(a) and g.has_node(b):
                dm[i, j] = nx.astar_path_length(g, a, b, heuristic=lambda u, v: 0.0, weight="weight")
    return dm
