# -*- coding: utf-8 -*-
"""token 特征的**字段定义**（spec §5.3.1）——宽度与语义的唯一真相。

⚠️ 2026-10-03 重写：此前本模块只有 `F_DYN=6` 一个桩常量，**没有任何字段定义**，
且三类 token 被迫同宽（单一 `Linear` 所致）。现按 spec §5.3.1 定死三套字段 + Global，
四段**补齐到同一宽度** `F_MAX=11` 后按序列拼接成单张张量，由 `encoder.py` 的**分段 Linear** 升维。
⚠️ 2026-10-04：① 拥堵 +1（V 段 `zone_wait`）、④ 返工 +1（B 段 `rework_cnt`）——
两个此前只被仿真执行、对网络不可见的约束。**末位追加**，既有列序不变。

**归一化一律用实例静态量**（总工时 / `M_ref` / 机器数 / 包围盒对角线）——
仿真前即知，训练与推理一致。**不用 per-episode 归一化**：在线决策下不可得。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..env.constraints import ConstraintConfig
from ..env.des import SimConfig
from ..env.instances import Instance
from ..env.layout import Layout

F_M, F_B, F_V, F_G = 7, 9, 11, 3
F_MAX = max(F_M, F_B, F_V, F_G)          # 11 —— 补齐后的统一宽度

# 各段在补齐张量 (N, F_MAX) 里占的列；超出部分恒为 0
SEG_SLICE: dict[str, slice] = {"M": slice(0, F_M), "B": slice(0, F_B),
                               "V": slice(0, F_V), "G": slice(0, F_G)}

# ⚠️ 字段名清单与宽度**必须逐段相等**——有测试守着（test_field_counts_match_declared_widths）。
# 它同时是论文附录的特征表与排错时的对照表。
# ⚠️ 2026-10-04：① 拥堵 +1（V 段 `zone_wait`）、④ 返工 +1（B 段 `rework_cnt`）。
# **一律末位追加**——既有列的下标是 ckpt 与测试的契约，不得插队。
FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    "M": ("backlog", "in_fill", "out_fill", "busy", "remaining_frac",
          "pm_left", "fail_rate"),
    "B": ("progress", "remaining_work", "due_margin", "finished",
          "at_machine", "in_transit", "on_agv", "weight", "rework_cnt"),
    "V": ("st_idle", "st_empty", "st_loaded", "st_down", "node_x", "node_y",
          "queued", "battery", "capacity", "speed_factor", "zone_wait"),
    "G": ("time_progress", "done_frac", "in_flight"),
}


@dataclass(frozen=True)
class NormContext:
    """归一化用的**实例静态量**（仿真前即可算出，不随 episode 变）。

    ⚠️ `constraints` 不是标度，但**必须与仿真同一份**：特征里"约束关掉就不存在的量"
    （③ fail_rate、⑧ due_margin、① zone_wait、④ rework_cnt）要据此静默，否则会给策略假信号
    （F2 缺陷类：③/⑧ 残留 R1；①/④ 是 R1c/R1d 同型）。
    存整份 `ConstraintConfig`（而非零散 bool）是刻意的——将来再有开关进特征层，不必改构造签名。
    """
    total_work_min: float
    m_ref: float
    n_m: int
    n_jobs: int
    n_agv: int
    bbox_diag: float
    max_fail_rate: float
    pm_interval: float
    zone_wait_limit: float       # ① 拥堵：区段等待上限 [min]（与 SimConfig 同源，见下）
    max_capacity: int
    max_queued: int
    node_xy: tuple[tuple[float, float], ...]   # 通道节点坐标（V 段 x/y 用）
    constraints: ConstraintConfig              # 建它的那一份约束（与仿真同源）


def norm_context(inst: Instance, layout: Layout, m_ref: float,
                 cfg: SimConfig | None = None,
                 constraints: ConstraintConfig | None = None) -> NormContext:
    """由实例 + 布局 + 配置算出全部归一化标度。

    ⚠️ 本函数是 `NormContext` 的**唯一构造入口**，与 `NormContext` 同模块——标度口径与
    字段定义同处一地，改一处不会漏另一处。`state_emb` 只是转出（re-export）它。

    ⚠️ **`cfg` / `constraints` 必须与跑仿真的那一份相同**：
    - `cfg`（评审 F3）：`pm_interval` 是扫描轴之一（spec §9.2 assumed 参数），写死 120 会让
      极端档（`m11_constraint_binding` 的 `pm_interval /= 10` → 12.0）下 `pm_left` **静默
      死掉**——`pm_clock` 一过 120 就被 clip 到 0，该维恒 0 且零报错。`None` = 默认 `SimConfig()`。
      同理 ① 的 `zone_wait_limit` 是 `zone_wait` 维的除数：必须取跑仿真那一份 cfg，写死会让
      改了等待上限的档静默错标度。
    - `constraints`（评审 F2）：约束开关会改变仿真的**有效范围**（⑩ 关 → 车队容量全退化为 1），
      归一标度若不跟着退化，特征就报出一个仿真里不存在的值（MK01 实测：⑩ 关时载量维报
      1/2 = 0.5，而仿真里每台车都是满容量 1）。`None` = 十约束全开（向后兼容）。
      **整份约束同时存进 `NormContext.constraints`**（R1）：③ 关时 `fail_rate`、⑧ 关时
      `due_margin`、① 关时 `zone_wait`、④ 关时 `rework_cnt` 必须静默为常量 0，特征函数据它
      判断——ctx 与仿真不同源即假信号。
    """
    c = cfg or SimConfig()
    cons = constraints or ConstraintConfig()    # None = 十约束全开（与 SimWorld 的语义一致）
    total_work = sum(min(t for _m, t in op) for job in inst.jobs for op in job)
    spec = layout.grid
    diag = float(np.hypot(spec.n_rows * (spec.cell_h + spec.aisle_w),
                          spec.n_cols * (spec.cell_w + spec.aisle_w)))
    return NormContext(
        total_work_min=max(total_work, 1e-9), m_ref=max(m_ref, 1e-9),
        n_m=inst.n_machines, n_jobs=inst.n_jobs, n_agv=len(layout.agvs or []),
        bbox_diag=max(diag, 1e-9),
        max_fail_rate=max((m.fail_rate for m in layout.machines), default=1e-9),
        pm_interval=float(c.pm_interval),   # 与 SimConfig.pm_interval 同源（评审 F3）
        zone_wait_limit=float(c.zone_wait_limit),   # ① 与 SimConfig 同源（同 F3 的理由）
        max_capacity=(max((a.capacity for a in (layout.agvs or [])), default=1)
                      if cons.heterogeneous_fleet else 1),
        max_queued=max(1, int(np.ceil(np.sqrt(max(inst.n_jobs, 1))))),
        # ⚠️ 末位是**装卸站**（节点号 = `grid.n_nodes`，格点外）——`state_emb` 按
        # `VehicleState.node` 取坐标，AGV 在站上时没有它就会 IndexError。
        node_xy=(tuple(spec.node_xy(*spec.node_rc(i)) for i in range(spec.n_nodes))
                 + (() if layout.lu is None else ((layout.lu.x, layout.lu.y),))),
        constraints=cons)
