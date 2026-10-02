"""参数化布局采样器：点集+属性，同 seed 同点集（《方法设计文档》§4.1）。

输出：机台/缓冲/充电位的位置-尺寸-属性表 + 通道宽度；走廊图构建见 corridors.py。
布局类型：line（直线）/ U（U 形）/ island（双列岛式）——拓扑轴零样本的采样来源。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class MachinePad:
    id: int
    x: float
    y: float
    w: float
    h: float
    dock: tuple[float, float]      # 装卸点
    fail_rate: float = 0.001       # 每小时期望故障次数（扰动流用）
    in_cap: int = 2                # 进站缓冲容量（RCIM in/out-buffer 显式建模）
    out_cap: int = 2


@dataclass
class BufferPad:
    id: int
    x: float
    y: float
    cap: int = 12


@dataclass
class ChargerPad:
    id: int
    x: float
    y: float


@dataclass
class Layout:
    machines: list[MachinePad]
    buffers: list[BufferPad]
    chargers: list[ChargerPad]
    aisle_width: float
    char_len: float                # 归一化长度（包围盒对角线，§2.2）
    layout_type: str
    seed: int
    bbox_diag: float = 0.0

    @property
    def n_machines(self) -> int:
        return len(self.machines)


def _place(machines: list[MachinePad], rng: np.random.Generator, aisle: float) -> None:
    for i, md in enumerate(machines):
        side = 1 if i % 2 == 0 else -1
        md.dock = (md.x, md.y + (md.h / 2 + aisle * 0.7) * side)
        md.fail_rate = float(rng.uniform(0.0, 0.003))
        md.in_cap = int(rng.integers(1, 4))
        md.out_cap = int(rng.integers(1, 4))


def sample_layout(n_machines: int, layout_type: str = "line", seed: int = 0,
                  aisle_width: float = 2.8, machine_gap: float = 3.0,
                  n_buffers: int = 2, n_chargers: int = 1) -> Layout:
    """采样一个布局。同 (n_machines, layout_type, seed, ...) → 同点集（确定性）。"""
    rng = np.random.default_rng(seed)
    machines: list[MachinePad] = []
    w, h = 3.0, 2.4   # 机台尺寸
    if layout_type == "line":
        machines = [MachinePad(id=i, x=i * (w + machine_gap), y=0.0, w=w, h=h, dock=(0, 0))
                    for i in range(n_machines)]
    elif layout_type == "U":
        half = (n_machines + 1) // 2
        for i in range(half):
            machines.append(MachinePad(id=i, x=i * (w + machine_gap), y=0.0, w=w, h=h, dock=(0, 0)))
        for j in range(n_machines - half):
            machines.append(MachinePad(id=half + j, x=half * (w + machine_gap),
                                       y=(j + 1) * (h + machine_gap), w=w, h=h, dock=(0, 0)))
    else:  # island：两列平行岛
        per_col = (n_machines + 1) // 2
        for i in range(n_machines):
            col, row = i // per_col, i % per_col
            machines.append(MachinePad(id=i, x=col * (w * 3 + machine_gap),
                                       y=row * (h + machine_gap), w=w, h=h, dock=(0, 0)))
    _place(machines, rng, aisle_width)
    xs = np.array([[m.x, m.y] for m in machines], dtype=float)
    diag = float(np.hypot(*xs.max(0) - xs.min(0))) if n_machines else 1.0
    buffers = [BufferPad(id=0, x=-w - machine_gap, y=0.0),
               BufferPad(id=1, x=float(xs[:, 0].max() + w + machine_gap), y=0.0)]
    chargers = [ChargerPad(id=0, x=-w - machine_gap, y=h + machine_gap)]
    return Layout(machines=machines, buffers=buffers[:n_buffers], chargers=chargers[:n_chargers],
                  aisle_width=aisle_width, char_len=max(diag, 1e-6),
                  layout_type=layout_type, seed=seed, bbox_diag=diag)
