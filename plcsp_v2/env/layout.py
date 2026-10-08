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


@dataclass
class MachinePad:
    id: int
    x: float
    y: float
    w: float
    h: float
    dock_node: int            # 该机台装卸点所在的通道节点
    fail_rate: float = 0.001  # ⚠️ assumed（无文献出处），见 spec §9
    in_cap: int = 2           # ⚠️ assumed
    out_cap: int = 2          # ⚠️ assumed


@dataclass
class ChargerPad:
    id: int
    node: int


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
    """单台 AGV 的规格（⑩ 异构车队的载体）。**参数全部 assumed**，见 spec §9。

    `speed_factor` 是**相对倍率**（不是绝对 m/s）——绝对车速仍是 `SimConfig.agv_speed_mps`
    这条扫描轴，此处只表达"车队内部有快慢差"。异构关闭时倍率一律取 1.0。
    """
    id: int
    speed_factor: float       # × cfg.agv_speed_mps，∈ [0.8, 1.2]
    capacity: int             # 同向拼车的件数上限
    battery_kwh: float        # 电池容量 [kWh]


@dataclass
class Layout:
    grid: GridSpec
    machines: list[MachinePad]
    chargers: list[ChargerPad]
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
                  n_chargers: int = 2, n_agv: int = 3,
                  max_agv_capacity: int = 3) -> Layout:
    """采样一个网格布局 + 车队规格。同 (n_machines, seed, 尺寸, 车队参数) → 同结果（确定性）。

    机台按格子顺序（逐行）占用前 n_machines 个格子；dock 落在该格左上角节点。
    充电桩从**未被机台占用**的节点里确定性选取（在自由节点中等间隔取，避免全挤一角）。

    ⚠️ **`n_agv` 必须与 `SimConfig.n_agv` 一致**——否则布局的车队与仿真的车队数量不符，
    是**静默错误**（`rollout` 已负责传参；直接调 `SimWorld` 的调用点须自己保证）。
    """
    if max_agv_capacity < 1:
        raise ValueError(f"载量上限必须 ≥ 1，收到 {max_agv_capacity}")
    rng = np.random.default_rng(seed)
    n_rows, n_cols = grid_shape(n_machines)
    spec = GridSpec(n_rows=n_rows, n_cols=n_cols, cell_w=cell_w, cell_h=cell_h, aisle_w=aisle_w)

    machines: list[MachinePad] = []
    for i in range(n_machines):
        r, c = divmod(i, n_cols)
        node = spec.node_id(r, c)
        x, y = spec.node_xy(r, c)
        machines.append(MachinePad(
            id=i, x=x, y=y, w=cell_w, h=cell_h, dock_node=node,
            fail_rate=float(rng.uniform(0.0, 0.003)),      # assumed，见 spec §9
            in_cap=int(rng.integers(1, 4)),                # assumed
            out_cap=int(rng.integers(1, 4)),               # assumed
        ))

    used = {m.dock_node for m in machines}
    free = [n for n in range(spec.n_nodes) if n not in used]
    if len(free) < n_chargers:
        raise ValueError(f"自由节点不足：需 {n_chargers} 个充电桩，仅 {len(free)} 个可用")
    # 确定性分散：在自由节点里等间隔取
    if n_chargers > 1:
        idx = [round(k * (len(free) - 1) / (n_chargers - 1)) for k in range(n_chargers)]
    else:
        idx = [0]
    charger_nodes = [free[i] for i in idx]
    chargers = [ChargerPad(id=j, node=n) for j, n in enumerate(charger_nodes)]

    # 装卸站（Fig. 2）：单一，在网格**外左侧**、经一条连接段接入左边缘交叉口。
    # ⚠️ 它**不占自由节点**（是格点外的新节点），故与充电桩的选取互不干扰。
    lu = LuPad(node=spec.lu_node, dock_node=spec.lu_dock_node,
               x=spec.lu_xy[0], y=spec.lu_xy[1], connector_m=spec.lu_connector_m)

    # 车队规格（⑩ 异构）。⚠️ 抽样放在**最后**——机台/充电桩的抽样序列才不会被扰动。
    # 这些数只被 `AgvSim` 在 `heterogeneous_fleet=True` 时读；关掉时行为与"从未存在"一致。
    agvs = [AgvSpec(id=a,
                    speed_factor=float(rng.uniform(0.8, 1.2)),       # assumed：±20% 速度差
                    capacity=int(rng.integers(1, max_agv_capacity + 1)),  # assumed
                    battery_kwh=float(rng.uniform(2.0, 4.0)))        # assumed
            for a in range(n_agv)]

    return Layout(grid=spec, machines=machines, chargers=chargers,
                  lu=lu, layout_seed=seed, agvs=agvs)
