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
    # 是否持有加工槽。⚠️ 与 `remaining_min > 0` **不互为充要**：两者可同时成立
    # （持有槽但未在制切削/已触底），**Task 2 / Task 8 都不得把"busy ⟹ remaining>0"
    # 当作不变式**（返工中间态、换型段、故障修复长停都会打出 `busy=True, remaining=0`）。
    busy: bool
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
    # ⑧ 交期 `d_j`（`des.compute_due_dates` → TF/RDD 口径：`LB·τ·(1+R(2ρ_j−1))`）——**逐作业**，
    # 不再锚在参考调度上（旧口径 `τ·M_ref` 已废）。⚠️ R=0.0 的实例（标定结果里的 mk01/mk04）
    # 仍是**共同交期**，故 `job_features` 的 due_margin 维在那两个实例上依旧无作业间区分度；
    # R>0 的实例上它才有区分度。0.0 = 该快照无交期（约束关闭，或参考运行内
    # `_MREF_BUSY` 短路——两种情形都不该有值）。
    due: float
    # 该作业当前**落在哪台机**（在制件优先）；-1 = 未开始或已派车离台。
    # ⚠️ **同一机台号可被多件命中**：在制那件 + 输出缓冲里等搬运的几件都记同一机台号。
    # ⚠️ **已完成的作业仍指向末道机台**（`job_loc` 只由 transporter 清，末工序完工后不再清）。
    at_machine: int
    in_transit: bool        # 见下：表示"当前是否被某车持有"，**退回即清零**
    # 当前持有该件的车；-1 = 无。⚠️ 语义是**当前是否被某车持有**（不是"是否在运输流程里"）：
    # 退回队列即清零，故**与重试史相关**——同为"在车里排队"的物理状态，经历过退回的
    # 取 `in_transit=False`，未经历过的（transporter 已派车、还没发车的）取 `True`。
    on_agv: int


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
    # 全线在途运输任务数 = **各队列里待取** + **车上正在运送**（两种队列形状都算；
    # 已装车的批次已离开队列，故必须单列"车上"那部分，否则 bound 路径会漏计）。
    in_flight: int
