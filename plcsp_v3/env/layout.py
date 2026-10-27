"""网格布局生成器（spec §3.2）：机位网格 + 通道格点 + 充电桩。

形态来源：AEI 103216 Fig. 2（该文借用 Cai et al., IJPR 61(4):1373-1393, 2023）。
- 机台占**格子**，通道是**格子之间的格线** → 通道天然不穿过机台（v0 缺陷自动消失）
- 通道图节点 = (n_rows+1) × (n_cols+1) 个交叉口
- zone = 通道节点（一节点一车互斥，见 des.ZoneManager）

单位：坐标 = 米（bug#13 已定单位约定）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """网格规格。机位 n_rows × n_cols；通道节点 (n_rows+1) × (n_cols+1)。

    ⚠️ **装卸站是格点外的第 n_nodes 号节点**（不是某个交叉口被挪用，见 `lu_node`）——
    形态来自 AEI 103216 Fig. 2：装卸单元在网格外左侧、经一条**连接段**接到边界交叉口。
    """
    n_rows: int
    n_cols: int
    cell_w: float = 3.0      # 机位宽 [m]
    cell_h: float = 2.4      # 机位高 [m]
    aisle_w: float = 1.5     # 通道宽 [m]（< 1.5 触发窄道降速，见 des.eff_speed）
    lu_connector_m: float = 3.0   # 装卸站连接段长 [m]（⚠️ assumed，无出处，见 spec §9.2）
    # ⚠️ **簇间距**（2026-10-16 加）：在**列 `cluster_split` 及其右**的每个节点上，
    #    给 x 加一个额外偏移 `cluster_gap` [m] ⟹ 车间被一条**宽通道**分成左右两簇。
    #
    # **动机（用户提出）**：均匀网格里 18 台机两两差不多远（实测中位 4.0 / max 9.3 min），
    # **「选哪台机」对 TC 的影响被压平了** ⟹ 27 条规则在 TC 上拉不开 ⟹ 散度全铺在
    # 「两个都差 ↔ 两个都好」的对角线上。**簇间距让「距离」第一次成为一件有后果的事。**
    #
    # ⚠️ **原文没有依据** —— Fig. 4 只给拓扑，**尺寸一个字都没有**（本文件顶部的注释）。
    #    `shop_len` 只钉住了**总运输量**（TC 靶），钉不住**距离的分布**。
    # ⚠️ **它**不是**置换**（`RANDOM_SCHEME` 那条免责说的是置换 ⟹ 统计量不变；
    #    改间距改的是**距离矩阵本身**，不是编号）。
    # ⚠️ **默认 `0.0` = 均匀 ⟹ v3 主线与既有读数逐位不变。**
    cluster_gap: float = 0.0
    cluster_split: int = -1       # 偏移打在**列号 ≥ 本值**的节点上；`-1` = 不启用

    @property
    def n_nodes(self) -> int:
        """**格点交叉口**数（不含装卸站——装卸站是格点外挂节点，编号紧接其后）。"""
        return (self.n_rows + 1) * (self.n_cols + 1)

    @property
    def lu_node(self) -> int:
        """装卸站的节点号 = 格点末位的下一个（格点外）。"""
        return self.n_nodes

    @property
    def lu_dock_node(self) -> int:
        """连接段接入的格点交叉口：左边缘、第 0/1 行机位分界处（Fig. 2 的位置）。"""
        return self.node_id(1, 0)

    @property
    def lu_xy(self) -> tuple[float, float]:
        """装卸站坐标：在被连交叉口的左侧 `lu_connector_m` 处。"""
        x, y = self.node_xy(1, 0)
        return (x - self.lu_connector_m, y)

    def node_id(self, r: int, c: int) -> int:
        return r * (self.n_cols + 1) + c

    def node_rc(self, node: int) -> tuple[int, int]:
        return divmod(node, self.n_cols + 1)

    def node_xy(self, r: int, c: int) -> tuple[float, float]:
        """节点 (r,c) 的坐标：节点在格子**左上角**，间距 = 机位 + 通道。

        ⚠️ **簇间距**（`cluster_gap` / `cluster_split`）：列号 `≥ cluster_split` 的节点整体右移
        `cluster_gap` 米 ⟹ 在那一列与前一列之间造出一条**宽通道**，车间分成左右两簇。
        **`cluster_gap = 0.0`（默认）⟹ 逐位等于均匀网格。**
        """
        gap = self.cluster_gap if (self.cluster_split >= 0 and c >= self.cluster_split) else 0.0
        return (c * (self.cell_w + self.aisle_w) + gap,
                r * (self.cell_h + self.aisle_w))


RANDOM_SCHEME = "random"
"""**机位随机撒在网格里**（2026-10-15 加）—— 复现层的默认，**不用 Fig. 4 的具体排布**。

**为什么**：`α_ijk`（哪道工序能用哪几台）在 `make_shi_routings` 里**独立均匀随机**抽，
**不看机台在哪** ⟹ 「哪台机在哪个格子」与「哪道工序用哪台机」**统计独立**
⟹ **换一个排布 ≡ 给机台重新编号 ⟹ 所有统计量不变**（用户裁定，2026-10-15）。

**⟹ 既然排布形状不重要，就不要**依赖读图**（`FIG4_PLACEMENT` 曾因读错而错两处，见其注释）。

**⟹ 随机撒而不是顺序填**：顺序填（`"grid"`）意味着"格子按行填、最后一行不满"，
即**空的那 2 格恒在右下角** ⟹ 有系统偏置。**随机撒则把「哪几格空着」也平均掉。**

⚠️ **约束（用户给）**：**L/U 仍在左侧**（`lu_dock_node = node_id(1, 0)` 不动）·
**行列数 = 4 × 5**（由 `grid_shape(18)` 定，与 Fig. 4 同规格）。
"""

RANDOM_PLACE_SEED = 4300
"""`scheme="random"` 的**机位选取子流**：`default_rng([seed, 本值])`。

⚠️ **必须独立**：并进主 `rng` 会把 `MachinePad.fail_rate` / `in_cap` 的抽样**整体移位**
（同 `MACHINE_CAP_SEED` / `PA_TIE_SEED` 那条纪律）。
"""

FIG4_SCHEME = "fig4"
"""**Shi et al. 2025（AEI 65:103195）Fig. 4 的机位排布**（2026-10-14 加）。

⚠️ **本仓原来的 `layout.py` 照的是**同卷相邻的另一篇**：AEI 65:**103216**（Li, Li & Gao）
的 Fig. 2（见本文件开头与 `LuPad` 的 docstring）。**那是 v3 主线的基线论文，不是 Shi 2025。**

⚠️ **`sample_layout(scheme=...)` 默认 `"grid"` ⟹ v3 主线逐位不变。**
**只有 `repro/` 的复现脚本显式传 `scheme=FIG4_SCHEME`。**

Fig. 4 逐格读出来的排布（**⚠️ 2026-10-15 修正，此前那张图是错的，见 `FIG4_PLACEMENT` 的注释**）：

```
     列0    列1    列2    列3   ║ 列4（主干道外）
行0    —     M7     M8     M15   ║  M16
行1   M1     M6     M9     M14   ║  M17
行2   M2     M5     M10    M13   ║  M18
行3   M3     M4     M11    M12   ║
      3 台   4 台   4 台   4 台     3 台   = 18 ✓
```

⚠️ **尺寸图里没有** —— 只给拓扑。尺寸仍由 `cell_w/cell_h/aisle_w` 定。
"""

FIG4_PLACEMENT: tuple[tuple[int, int], ...] = (
    # ⚠️ **元素是 `(列, 行)`**；**下标 = 机台 id**（0 起，= Fig. 4 的 M1..M18）
    #
    # ⚠️⚠️ **2026-10-15 修正**：此前这张表把 **M1–M4 全放在列 0、M5–M7 全放在列 1**
    #    （每列 [4,3,4,4,3]）。**那是错的** —— 按 Fig. 4 的高分渲染逐格读：
    #
    #    · **列 0 只有 3 台**（M1/M2/M3），且**行 0 是空的**（M1 在一列里靠**中上**，不是顶上）
    #    · **列 1 有 4 台**：M7(行0) · M6(行1) · **M5(行2) · M4(行3)** ⟹ **M4 在 M5 下面**
    #    · **每列台数 = [3, 4, 4, 4, 3]**（此前记的 [4,3,4,4,3] 是错的）
    #
    #    ```
    #        列0    列1    列2    列3   ║ 列4（主干道外）
    #    行0   —     M7     M8     M15   ║  M16
    #    行1  M1     M6     M9     M14   ║  M17
    #    行2  M2     M5     M10    M13   ║  M18
    #    行3  M3     M4     M11    M12   ║
    #         3 台   4 台   4 台   4 台     3 台   = 18 ✓
    #    ```
    # ⚠️ **行的顺序 = 机台 id 的顺序**（下标 i ↔ 第 i+1 台机），**不是图上从上到下的顺序**。
    (0, 1), (0, 2), (0, 3),              # M1  M2  M3
    (1, 3), (1, 2), (1, 1), (1, 0),      # M4  M5  M6  M7
    (2, 0), (2, 1), (2, 2), (2, 3),      # M8  M9  M10 M11
    (3, 3), (3, 2), (3, 1), (3, 0),      # M12 M13 M14 M15
    (4, 0), (4, 1), (4, 2),              # M16 M17 M18（最右、主干道外）
)

FIG4_SHAPE = (4, 5)
"""Fig. 4 的 (行数, 列数) —— 每列最多 **4** 台 ⟹ 4 行；共 5 列（含最右那列）。"""

MACHINE_CAP_SEED = 4000
"""② 有限缓冲容量的**独立子流**：`default_rng([layout_seed, MACHINE_CAP_SEED])`。

⚠️ **不得并进主 `rng`**：主流的取数次序一变，**后面所有抽样（`AgvSpec.battery_kwh` 等）整体移位**
⟹ 既有布局全变、读数全废。**实测踩过**（同 `des.PA_TIE_SEED` 那条纪律）。
"""


@dataclass
class MachinePad:
    id: int
    x: float
    y: float
    w: float
    h: float
    dock_node: int            # 该机台装卸点所在的通道节点
    # ⚠️ **机台接入点 = 横走廊上的一点**（2026-10-17 加，照原文 Fig. 4）。
    #
    # **Fig. 4 高倍渲染**：每台机**只引出一条接入段**，接到它**上方那条横向走廊**
    # （M1/M2 向右、M16/M17/M18 向左、其余向上 —— 三条边都落在横走廊上）。
    # **⟹ 机台不在路口上**，车到了机台必须**先横着走到路口**，才能转进竖向走廊。
    #
    # `None`（默认）⟹ `dock_node` 就是**路口** `node_id(r, c)`（旧行为，逐位不变）。
    # 非 `None` ⟹ `dock_node` 是一个**新增节点**（编号 `GridSpec.lu_node + 1 + id`），
    # `dock_neighbors = (左路口, 右路口)` 是它在横走廊上的两个端点。
    dock_neighbors: tuple[int, int] | None = None
    fail_rate: float = 0.001  # ⚠️ assumed（无文献出处），见 spec §9
    # ② **有限缓冲**（2026-10-10 加回；v2 在 2026-10-08 删过）。**⚠️ `assumed`**（v1 的取值）。
    # ⚠️ **默认关**（`ConstraintConfig.finite_buffer=False`）⟹ 两个队列仍无界 ⟹ **逐位不变**。
    # 开时：`cap = in_cap/out_cap` ⟹ 投递满则让步重试（见 `AgvSim` 的那两处循环）。
    in_cap: int = 2
    out_cap: int = 2


@dataclass
class LuPad:
    """装卸站（Load/Unload unit）——**网格外一侧**，由一条**连接段**接到格点边界交叉口。

    形态来源：**AEI 103216 Fig. 2**（*Real-time scheduling for production-logistics
    collaborative environment using multi-agent deep reinforcement learning*, Advanced
    Engineering Informatics 65:103216, 2025）；该文自陈采用其 [5] 的车间布局，即
    **Cai et al., International Journal of Production Research 61(4):1373-1393, 2023**。
    Fig. 2 形态：单一装卸单元在机位网格**左侧**、约在第 1/2 行机位分界的高度，
    经一条**短连接段**接入左边缘交叉口；图上的 AGV 全部落在格点交叉口上，装卸站不在其中。

    ⚠️ 故 `node`（装卸站自己的节点号）**不是**任何交叉口——`dock_node` 才是它接入的那个。
    ❌ 不得用某个边界交叉口冒充当装卸站：那会静默短掉一整段连接段，并把站塞进通道网当普通路口。
    ❌ 不得引 JMS 82（MMSLS）作为布局来源（spec §3.2：该文不是物理平面图）。
    """
    node: int          # 装卸站自己的节点号（格点外，= GridSpec.lu_node）
    dock_node: int     # 连接段接入的格点交叉口（= GridSpec.lu_dock_node）
    x: float
    y: float
    connector_m: float


@dataclass(frozen=True)
class AgvSpec:
    """单台 AGV 的规格。

    ⚠️ **2026-10-10（⑪ 充电整条删除）后，这里只剩 `id`** —— 原 `battery_kwh` / `soc0`
    随充电机制一并删；`speed_factor` / `capacity` 早在 **v2（⑩ 去后）** 就删了。
    **车队现在完全同构**：绝对车速取 `SimConfig.agv_speed_mps`，载量恒 1。
    """

    id: int


@dataclass
class Layout:
    grid: GridSpec
    machines: list[MachinePad]
    lu: LuPad
    layout_seed: int
    agvs: list[AgvSpec] = None      # 车队规格（默认 None → 由 sample_layout 填）

    @property
    def n_machines(self) -> int:
        return len(self.machines)

    @property
    def n_agv(self) -> int:
        return len(self.agvs or [])


def grid_shape(n_machines: int, n_cols: int | None = None,
               n_rows: int | None = None) -> tuple[int, int]:
    """机位数 → (n_rows, n_cols)。`n_cols=None`（默认）⟹ 尽量方，行数 ≤ 列数。

    ⚠️ **`n_cols` 覆盖**（2026-10-14 加）：原文 **Fig. 4** 的布局是
    `L/U` 在左中 + 18 台机排成 **4 列**，而**图没标尺寸** ⟹ **形状本身是自由的**。
    改形状会改 `tt` 矩阵的**结构**（不只是尺度）⟹ 改 MS/TC 的权衡方向。
    **⟹ 这是「只改尺度」（`cell_w/cell_h/aisle_w`）之外的一条独立自由度。**

    ⚠️ **`n_rows` 覆盖**（2026-10-17 加，用户提出）：默认 `n_rows = ceil(n_machines/n_cols)`
    **⟹ 5 列必然给 4 行**（20 格）。要 **5×5 = 25 格**就得显式给行数。
    **`n_rows=None`（默认）⟹ 逐位不变。**
    """
    if n_cols is None:
        n_cols = max(1, math.ceil(math.sqrt(n_machines)))
    n_cols = max(1, int(n_cols))
    n_rows = (max(1, math.ceil(n_machines / n_cols)) if n_rows is None
              else max(1, int(n_rows)))
    if n_rows * n_cols < n_machines:
        raise ValueError(
            f"格子数不够：{n_rows}×{n_cols} = {n_rows * n_cols} < {n_machines} 台机")
    return n_rows, n_cols


def sample_layout(n_machines: int, seed: int = 0, *, cell_w: float = 3.0,
                  cell_h: float = 2.4, aisle_w: float = 1.5,
                  lu_connector_m: float = 3.0, n_cols: int | None = None,
                  cap_fixed: int | None = None, n_agv: int = 3,
                  scheme: str = "grid", n_rows: int | None = None,
                  mid_edge_dock: bool = False,
                  cluster_gap: float = 0.0, cluster_split: int = -1) -> Layout:
    """采样一个网格布局 + 车队规格。同 (n_machines, seed, 尺寸, 车队参数) → 同结果（确定性）。

    机台按格子顺序（逐行）占用前 n_machines 个格子；dock 落在该格左上角节点。
    充电桩从**未被机台占用**的节点里确定性选取（在自由节点中等间隔取，避免全挤一角）。

    ⚠️ **`n_agv` 必须与 `SimConfig.n_agv` 一致**——否则布局的车队与仿真的车队数量不符，
    是**静默错误**（`rollout` 已负责传参；直接调 `SimWorld` 的调用点须自己保证）。
    """
    rng = np.random.default_rng(seed)
    if scheme == FIG4_SCHEME:
        # ⚠️ **Fig. 4 的拓扑**（2026-10-14）：**按列填**，不是按行填。
        #    默认 `"grid"` ⟹ 这行不进；v3 主线逐位不变。
        if n_machines != len(FIG4_PLACEMENT):
            raise ValueError(
                f"scheme='fig4' 只编了 {len(FIG4_PLACEMENT)} 台机（Fig. 4 的 M1..M18），"
                f"收到 n_machines={n_machines}")
        n_rows, n_cols = FIG4_SHAPE
        cells = list(FIG4_PLACEMENT)          # (列, 行)
    elif scheme == RANDOM_SCHEME:
        # ⚠️ **机位随机撒**（2026-10-15）：行列数仍由 `grid_shape` 定（18 台 ⟹ 4×5），
        #    但**哪 18 个格子被占**是随机的 ⟹ 把「哪几格空着」也平均掉。
        #    ⚠️ **独立子流**（`RANDOM_PLACE_SEED`）—— **不消耗主 `rng`**（同 `MACHINE_CAP_SEED`）。
        n_rows, n_cols = grid_shape(n_machines, n_cols, n_rows)
        prng = np.random.default_rng([seed, RANDOM_PLACE_SEED])
        idx = sorted(int(i) for i in
                     prng.choice(n_rows * n_cols, size=n_machines, replace=False))
        cells = [divmod(i, n_cols) for i in idx]     # (行, 列)
    else:
        n_rows, n_cols = grid_shape(n_machines, n_cols, n_rows)
        cells = None
    spec = GridSpec(n_rows=n_rows, n_cols=n_cols, cell_w=cell_w, cell_h=cell_h,
                    aisle_w=aisle_w, lu_connector_m=lu_connector_m,
                    cluster_gap=float(cluster_gap), cluster_split=int(cluster_split))

    machines: list[MachinePad] = []
    # ② 的容量抽样走**独立子流**（见下方 `MachinePad` 的注释）：不动主 `rng` 的取数次序。
    cap_rng = np.random.default_rng([seed, MACHINE_CAP_SEED])
    for i in range(n_machines):
        # ⚠️ **三条分支的元组顺序不同，必须分开**：
        #    · `fig4`   —— `FIG4_PLACEMENT` 存 **(列, 行)**
        #    · `random` —— `divmod` 给 **(行, 列)**
        #    · `grid`   —— `divmod` 给 **(行, 列)**（按行填）
        if cells is not None:
            r, c = cells[i]
            if scheme == FIG4_SCHEME:
                c, r = r, c                   # fig4 存的是 (列, 行)
        else:
            r, c = divmod(i, n_cols)
        node = spec.node_id(r, c)
        x, y = spec.node_xy(r, c)
        # ⚠️ **机台接入点 = 横走廊中点**（`mid_edge_dock`，2026-10-17）—— 见 `MachinePad`。
        #    新节点编号 = `lu_node + 1 + i`（**接在装卸站之后** ⟹ `lu_node` 不变）。
        neigh = None
        if mid_edge_dock and c + 1 <= spec.n_cols:
            neigh = (node, spec.node_id(r, c + 1))
            node = spec.lu_node + 1 + i
        machines.append(MachinePad(
            id=i, x=x, y=y, w=cell_w, h=cell_h, dock_node=node,
            dock_neighbors=neigh,
            fail_rate=float(rng.uniform(0.0, 0.003)),      # assumed，见 spec §9
            # ② 有限缓冲（2026-10-10 加回）：照 v1 的采样口径 `integers(1, 4)` ⟹ 1–3。
            # ⚠️ **必须走独立子流**：主 `rng` 后面还要抽 `AgvSpec.battery_kwh`，
            # 在这里多抽两次会把它**整体移位** ⟹ 既有布局全变（实测踩过：
            # AGV0 电池 3.2133 → 3.7148）。同 `PA_TIE_SEED` 那条纪律。
            # ⚠️ **`cap_fixed`**（2026-10-14）：把缓冲容量钉死成一个值。
            #    原文的约束表里**完全没有缓冲上限** ⟹ 这是本仓 assumed 的（v1 遗留）。
            in_cap=(int(cap_fixed) if cap_fixed else int(cap_rng.integers(1, 4))),
            out_cap=(int(cap_fixed) if cap_fixed else int(cap_rng.integers(1, 4))),
        ))

    # ⚠️ **2026-10-10：充电机制整条删除**（⑪ 去掉，用户裁定）—— `ChargerPad` 与
    # 桩的选取（原为"自由节点里等间隔取"）一并删。**删它不移位随机流**：
    # 桩的选取**不耗 `rng`**，而 `AgvSpec` 的抽样本来就在**最后**。
    #
    # 装卸站（Fig. 2）：单一，在网格**外左侧**、经一条连接段接入左边缘交叉口。
    lu = LuPad(node=spec.lu_node, dock_node=spec.lu_dock_node,
               x=spec.lu_xy[0], y=spec.lu_xy[1], connector_m=spec.lu_connector_m)

    # 车队规格。⚠️ 抽样放在**最后**——机台的抽样序列才不会被扰动。
    # ⚠️ **2026-10-10（⑪ 去后）**：车队**同构**，规格里已无逐车不同的量；
    # `AgvSpec` 只剩 `id`。（速度取 `SimConfig.agv_speed_mps`、载量恒 1、**无电池**。）
    agvs = [AgvSpec(id=a) for a in range(n_agv)]

    return Layout(grid=spec, machines=machines, lu=lu,
                  layout_seed=seed, agvs=agvs)
