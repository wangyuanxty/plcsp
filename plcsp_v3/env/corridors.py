"""格点走廊图与距离矩阵（spec §3.2）。

节点 = 通道交叉口（编号 0..n_nodes-1）；边 = 横/竖相邻节点，权重 = 段长 [m]。
机台/充电桩/缓冲都**挂在**某个通道节点上（`dock_node` / `node`），故距离矩阵按节点算。

与上一代的区别：旧版是"相邻 dock + 首尾闭环"的**单环**，且连线可能穿过机台；
新版通道在格子**之间**、机台在格子**之内**，两问题同时消失（无需栅格 A*）。

2026-10-02：区段管制**已按参数化粒度恢复**（见 `des.SimConfig.zone_granularity`）——
`shortest_node_path` 用于取 AGV 实际经过的节点序列。路线决策（选哪条路）仍未恢复。

2026-10-04：**路线决策（R）恢复**（见 `docs/progress-log.md` §27.3/§28）。历史必须记清：
`route_logits` 当初是**随 ① 拥堵一并取消**的，理由是"无拥堵时选远路严格更差"（spec §5.3）；
但 ① 后来在 `eb1d1da` 恢复，**路线头没跟着恢复**、落下了——本函数即那次遗漏的补建。
候选集是 R 头的**动作空间**：`k_shortest_paths` 给 AGV 的 k 条候选，策略按区段争用挑一条。
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .layout import Layout


def build_corridor_graph(layout: Layout, directed: bool = False) -> nx.Graph:
    """格点图：节点带 pos 属性；横/竖相邻边，权重 = 欧氏段长 [m]。

    ⚠️ 另加**装卸站**：格点外的一个节点 + **一条**连接段边（接到 `GridSpec.lu_dock_node`）。
    装卸站**不是**格点交叉口——AGV 不在那里与别的车争用路口区段（见 `layout.LuPad`）。

    ══ ⚠️ **`directed`（2026-10-17 加）—— 照原文 Fig. 4 的箭头** ══

    **Fig. 4 的走廊不是全双向的**（第 14 页高倍渲染逐条读）：

    ```
    横向走廊：每条边**两端都有箭头**（◄ ►）   ⟹ **双向**
    纵向走廊：**一整条只有一个方向**，且**逐列交替**
              竖线 c = 0(最左) ↑ · 1 ↓ · 2 ↑ · 3 ↓ · 4 ↑
    机台接入：机台的 dock 就在走廊节点上，无独立接入段
    ```

    `directed=False`（默认）⟹ **无向图，逐位等于旧行为**。
    `directed=True` ⟹ `nx.DiGraph`：横向**双向**；纵向 **`c` 偶 ⟹ 向上（r+1 → r）· `c` 奇 ⟹ 向下（r → r+1）**。

    **⟹ 距离矩阵变成非对称（`d(A→B) ≠ d(B→A)`），AGV 的「位置」带上方向性。**
    **⚠️ 强连通性**：横向全双向 ⟹ 任何节点仍可达任何节点（先横到方向对的那列、再纵、再横回来）。
    """
    spec = layout.grid
    g = nx.DiGraph() if directed else nx.Graph()
    for r in range(spec.n_rows + 1):
        for c in range(spec.n_cols + 1):
            g.add_node(spec.node_id(r, c), pos=spec.node_xy(r, c))
    for r in range(spec.n_rows + 1):
        for c in range(spec.n_cols + 1):
            u = spec.node_id(r, c)
            if c + 1 <= spec.n_cols:                     # 横向：**双向**
                v = spec.node_id(r, c + 1)
                w = abs(spec.node_xy(r, c + 1)[0] - spec.node_xy(r, c)[0])
                g.add_edge(u, v, weight=w)
                if directed:
                    g.add_edge(v, u, weight=w)
            if r + 1 <= spec.n_rows:                     # 纵向：**单向，逐列交替**
                v = spec.node_id(r + 1, c)
                w = abs(spec.node_xy(r + 1, c)[1] - spec.node_xy(r, c)[1])
                if not directed:
                    g.add_edge(u, v, weight=w)
                elif c % 2 == 0:                         # 偶数列 ↑（下 → 上）
                    g.add_edge(v, u, weight=w)
                else:                                    # 奇数列 ↓（上 → 下）
                    g.add_edge(u, v, weight=w)
    if layout.lu is not None:
        g.add_node(layout.lu.node, pos=(layout.lu.x, layout.lu.y))
        w = float(layout.lu.connector_m)
        g.add_edge(layout.lu.node, layout.lu.dock_node, weight=w)
        if directed:
            g.add_edge(layout.lu.dock_node, layout.lu.node, weight=w)
    # ⚠️ **机台接入点 = 横走廊中点**（`mid_edge_dock`，2026-10-17）——
    #    每台机一个**新节点**，**双向**接到它横走廊上的两个路口（各半格）。
    #    ⟹ 车到了机台，必须**先横着走到路口**，才能转进竖向走廊（照 Fig. 4）。
    for mp in layout.machines:
        if mp.dock_neighbors is None:
            continue
        a, b = int(mp.dock_neighbors[0]), int(mp.dock_neighbors[1])
        xa, ya = g.nodes[a]["pos"]
        xb, yb = g.nodes[b]["pos"]
        g.add_node(mp.dock_node, pos=(0.5 * (xa + xb), 0.5 * (ya + yb)))
        hw = 0.5 * abs(xb - xa)
        g.add_edge(mp.dock_node, a, weight=hw)
        g.add_edge(a, mp.dock_node, weight=hw)
        g.add_edge(mp.dock_node, b, weight=hw)
        g.add_edge(b, mp.dock_node, weight=hw)
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


def k_shortest_paths(g: nx.Graph, src: int, dst: int, k: int) -> list[list[int]]:
    """**k 条最短简单路径**（按长度升序，最短在前）——R 头（路线决策）的动作空间。

    恢复的机制（不是新发明）：`route_logits` 随 ① 拥堵被砍、"无拥堵时选远路严格更差"
    （spec §5.3）；① 后来恢复而路线头漏恢复（`docs/progress-log.md` §27.3、§28）。
    故这里给的是**候选集**，选哪条由策略按区段争用决定——正是"把被规则拿走的决策权交还策略"。

    ⚠️ **候选数可能 < k**（相邻节点、或装卸站那条**桥**边）：此时返回实际条数，
    **不补齐、不报错**。调用方按 `len()` 判断——只剩一条时"选路"不构成决策（`_drive` 直接走它）。
    ⚠️ 代价：`nx.shortest_simple_paths` 是 Yen 式**生成器**（每次调用重新枚举）。
    本函数**自带零缓存**——缓存由调用方按 (src, dst) 做（见 `des.SimWorld` 的路线缓存）：
    图对象每次 `roll_chain` 重建，按图对象做模块级缓存会跨世界串味（静默错误）。
    """
    out: list[list[int]] = []
    for p in nx.shortest_simple_paths(g, src, dst, weight="weight"):
        out.append(list(p))
        if len(out) >= k:
            break
    return out
