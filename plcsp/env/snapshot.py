# -*- coding: utf-8 -*-
"""仿真状态快照（spec §5.3.1）——特征层的**唯一**原料。

设计：`SimWorld.snapshot()` 把当时的活状态**拷成纯数据**（frozen dataclass），
网络代码只读它、不碰 SimPy 对象。这样：
- 特征层可脱离仿真单测（手搓 Snapshot 即可）；
- 决策日志能只存快照（仿真继续跑，快照不跟着变）。

⚠️ **本模块不含任何归一化**——归一化在 `nn/state_emb.py`（Task 2），
因为归一化需要实例静态量（总工时/M_ref），那是特征层的依赖而非仿真层的。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MachineState:
    backlog_min: float      # 输入队列内各工序加工时长之和 [min]
    in_q_len: int
    in_cap: float           # inf 表示无界（② 关闭时）
    out_q_len: int
    out_cap: float
    busy: bool              # 是否持有加工槽
    # 在制工序的**标称**剩余 [min]：上机时刻 + 工序时长 − now，随仿真时间递减、触底为 0。
    # ⚠️ 不含换型/故障修复的墙钟延长（故长停机时可能已触底而工件仍在机台上）；无在制 = 0。
    remaining_min: float
    pm_used_min: float      # 距上次保养已累计的主轴工时（⑫）
    fail_rate: float        # ③
    prev_job: int           # 本机上一件加工的作业号；-1 = 还没加工过（⑤ 换型的依据）


@dataclass(frozen=True)
class JobState:
    done_ops: int
    total_ops: int
    remaining_min: float    # 剩余工序的**标称**加工时长之和
    finished: bool
    # 该作业当前**落在哪台机**（在制件优先）；-1 = 未开始或已派车离台。
    # ⚠️ **同一机台号可被多件命中**：在制那件 + 输出缓冲里等搬运的几件都记同一机台号。
    at_machine: int
    in_transit: bool
    on_agv: int             # 在途时所乘的车；-1 = 无


@dataclass(frozen=True)
class VehicleState:
    status: int             # 0 空闲 / 1 空载行驶 / 2 负载行驶 / 3 故障停机
    node: int               # 当前所在通道节点
    queued: int             # 本车待办任务数
    battery_frac: float     # [0,1]
    capacity: int
    speed_factor: float


@dataclass(frozen=True)
class Snapshot:
    now: float
    machines: tuple[MachineState, ...]
    jobs: tuple[JobState, ...]
    vehicles: tuple[VehicleState, ...]
    n_done: int             # 已完成作业数
    in_flight: int          # 全线在途运输任务数
