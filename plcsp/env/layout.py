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
    """网格规格。机位 n_rows × n_cols；通道节点 (n_rows+1) × (n_cols+1)。"""
    n_rows: int
    n_cols: int
    cell_w: float = 3.0      # 机位宽 [m]
    cell_h: float = 2.4      # 机位高 [m]
    aisle_w: float = 1.5     # 通道宽 [m]（< 1.5 触发窄道降速，见 des.eff_speed）

    @property
    def n_nodes(self) -> int:
        return (self.n_rows + 1) * (self.n_cols + 1)

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
class BufferPad:
    id: int
    node: int


@dataclass
class Layout:
    grid: GridSpec
    machines: list[MachinePad]
    chargers: list[ChargerPad]
    buffers: list[BufferPad]
    layout_seed: int

    @property
    def n_machines(self) -> int:
        return len(self.machines)


def grid_shape(n_machines: int) -> tuple[int, int]:
    """机位数 → (n_rows, n_cols)：尽量方，行数 ≤ 列数。"""
    n_cols = max(1, math.ceil(math.sqrt(n_machines)))
    n_rows = max(1, math.ceil(n_machines / n_cols))
    return n_rows, n_cols


def sample_layout(n_machines: int, seed: int = 0, *, cell_w: float = 3.0,
                  cell_h: float = 2.4, aisle_w: float = 1.5,
                  n_chargers: int = 2) -> Layout:
    """采样一个网格布局。同 (n_machines, seed, 尺寸) → 同结果（确定性）。

    机台按格子顺序（逐行）占用前 n_machines 个格子；dock 落在该格左上角节点。
    充电桩从**未被机台占用**的节点里确定性选取（在自由节点中等间隔取，避免全挤一角）。
    """
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

    # 缓冲/装卸：取最后两个自由节点（网格外侧），避开充电桩
    rest = [n for n in free if n not in set(charger_nodes)]
    buffers = [BufferPad(id=j, node=n) for j, n in enumerate(rest[-2:])] if len(rest) >= 2 else []

    return Layout(grid=spec, machines=machines, chargers=chargers,
                  buffers=buffers, layout_seed=seed)
