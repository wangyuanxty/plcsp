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
        """节点 (r,c) 的坐标：节点在格子**左上角**，间距 = 机位 + 通道。"""
        return (c * (self.cell_w + self.aisle_w), r * (self.cell_h + self.aisle_w))


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


def grid_shape(n_machines: int) -> tuple[int, int]:
    """机位数 → (n_rows, n_cols)：尽量方，行数 ≤ 列数。"""
    n_cols = max(1, math.ceil(math.sqrt(n_machines)))
    n_rows = max(1, math.ceil(n_machines / n_cols))
    return n_rows, n_cols


def sample_layout(n_machines: int, seed: int = 0, *, cell_w: float = 3.0,
                  cell_h: float = 2.4, aisle_w: float = 1.5,
                  n_agv: int = 3) -> Layout:
    """采样一个网格布局 + 车队规格。同 (n_machines, seed, 尺寸, 车队参数) → 同结果（确定性）。

    机台按格子顺序（逐行）占用前 n_machines 个格子；dock 落在该格左上角节点。
    充电桩从**未被机台占用**的节点里确定性选取（在自由节点中等间隔取，避免全挤一角）。

    ⚠️ **`n_agv` 必须与 `SimConfig.n_agv` 一致**——否则布局的车队与仿真的车队数量不符，
    是**静默错误**（`rollout` 已负责传参；直接调 `SimWorld` 的调用点须自己保证）。
    """
    rng = np.random.default_rng(seed)
    n_rows, n_cols = grid_shape(n_machines)
    spec = GridSpec(n_rows=n_rows, n_cols=n_cols, cell_w=cell_w, cell_h=cell_h, aisle_w=aisle_w)

    machines: list[MachinePad] = []
    # ② 的容量抽样走**独立子流**（见下方 `MachinePad` 的注释）：不动主 `rng` 的取数次序。
    cap_rng = np.random.default_rng([seed, MACHINE_CAP_SEED])
    for i in range(n_machines):
        r, c = divmod(i, n_cols)
        node = spec.node_id(r, c)
        x, y = spec.node_xy(r, c)
        machines.append(MachinePad(
            id=i, x=x, y=y, w=cell_w, h=cell_h, dock_node=node,
            fail_rate=float(rng.uniform(0.0, 0.003)),      # assumed，见 spec §9
            # ② 有限缓冲（2026-10-10 加回）：照 v1 的采样口径 `integers(1, 4)` ⟹ 1–3。
            # ⚠️ **必须走独立子流**：主 `rng` 后面还要抽 `AgvSpec.battery_kwh`，
            # 在这里多抽两次会把它**整体移位** ⟹ 既有布局全变（实测踩过：
            # AGV0 电池 3.2133 → 3.7148）。同 `PA_TIE_SEED` 那条纪律。
            in_cap=int(cap_rng.integers(1, 4)),            # ② assumed
            out_cap=int(cap_rng.integers(1, 4)),           # ② assumed
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
