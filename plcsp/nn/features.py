# -*- coding: utf-8 -*-
"""token 特征的**字段定义**（spec §5.3.1）——宽度与语义的唯一真相。

⚠️ 2026-10-03 重写：此前本模块只有 `F_DYN=6` 一个桩常量，**没有任何字段定义**，
且三类 token 被迫同宽（单一 `Linear` 所致）。现按 spec §5.3.1 定死三套字段 + Global，
四段**补齐到同一宽度** `F_MAX=10` 后按序列拼接成单张张量，由 `encoder.py` 的**单个 Linear** 升维。

**归一化一律用实例静态量**（总工时 / `M_ref` / 机器数 / 包围盒对角线）——
仿真前即知，训练与推理一致。**不用 per-episode 归一化**：在线决策下不可得。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..env.instances import Instance
from ..env.layout import Layout

F_M, F_B, F_V, F_G = 7, 8, 10, 3
F_MAX = max(F_M, F_B, F_V, F_G)          # 10 —— 补齐后的统一宽度

# 各段在补齐张量 (N, F_MAX) 里占的列；超出部分恒为 0
SEG_SLICE: dict[str, slice] = {"M": slice(0, F_M), "B": slice(0, F_B),
                               "V": slice(0, F_V), "G": slice(0, F_G)}

# ⚠️ 字段名清单与宽度**必须逐段相等**——有测试守着（test_field_counts_match_declared_widths）。
# 它同时是论文附录的特征表与排错时的对照表。
FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    "M": ("backlog", "in_fill", "out_fill", "busy", "remaining_frac",
          "pm_left", "fail_rate"),
    "B": ("progress", "remaining_work", "due_margin", "finished",
          "at_machine", "in_transit", "on_agv", "weight"),
    "V": ("st_idle", "st_empty", "st_loaded", "st_down", "node_x", "node_y",
          "queued", "battery", "capacity", "speed_factor"),
    "G": ("time_progress", "done_frac", "in_flight"),
}


@dataclass(frozen=True)
class NormContext:
    """归一化用的**实例静态量**（仿真前即可算出，不随 episode 变）。"""
    total_work_min: float
    m_ref: float
    n_m: int
    n_jobs: int
    n_agv: int
    bbox_diag: float
    max_fail_rate: float
    pm_interval: float
    max_capacity: int
    max_queued: int
    node_xy: tuple[tuple[float, float], ...]   # 通道节点坐标（V 段 x/y 用）


def norm_context(inst: Instance, layout: Layout, m_ref: float) -> NormContext:
    """由实例 + 布局算出全部归一化标度（与 `SimConfig` 无关的部分）。

    ⚠️ 本函数是 `NormContext` 的**唯一构造入口**，与 `NormContext` 同模块——标度口径与
    字段定义同处一地，改一处不会漏另一处。`state_emb` 只是转出（re-export）它。
    """
    total_work = sum(min(t for _m, t in op) for job in inst.jobs for op in job)
    spec = layout.grid
    diag = float(np.hypot(spec.n_rows * (spec.cell_h + spec.aisle_w),
                          spec.n_cols * (spec.cell_w + spec.aisle_w)))
    return NormContext(
        total_work_min=max(total_work, 1e-9), m_ref=max(m_ref, 1e-9),
        n_m=inst.n_machines, n_jobs=inst.n_jobs, n_agv=len(layout.agvs or []),
        bbox_diag=max(diag, 1e-9),
        max_fail_rate=max((m.fail_rate for m in layout.machines), default=1e-9),
        pm_interval=120.0,          # 与 SimConfig.pm_interval 同源（Task 7 归一到 cfg）
        max_capacity=max((a.capacity for a in (layout.agvs or [])), default=1),
        max_queued=max(1, int(np.ceil(np.sqrt(max(inst.n_jobs, 1))))),
        node_xy=tuple(spec.node_xy(*spec.node_rc(i)) for i in range(spec.n_nodes)))
