# -*- coding: utf-8 -*-
"""SimPy DES 核心（《方法设计文档》§4.2：SimPy 单一后端，无物理库）。

**十约束全部接入且可独立开关**（spec §3.3，P1b 2026-10-02；⑥ 已砍）——开关一律走
`ConstraintConfig`，关掉时**不消耗随机数**，故"关 = 该约束从未存在"。

单位：**仿真时间 = 分钟，布局坐标 = 米**（bug#13 约定）；能耗 = kWh，走 `plcsp/energy.py`。

边界（诚实声明）：
- AGV 有**空载段**（2026-10-02 补齐；此前 AGV 从上一卸货点瞬移到取货点）。
- 机器故障/AGV 故障=泊松流；⑨ 的故障**腿间检出**（不打断正在进行的行驶）。
- ⑦ 模糊运输时间、⑥ 模糊加工时间**均已砍**（见 spec §3.3 的砍除记录）。
- 完成时刻以"末工序在机台加工完毕"为准（搬运回库不计入 makespan——对齐 BKS 口径）。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING
import numpy as np
import simpy
from ..energy import (AGV_EMPTY_KW, AGV_IDLE_KW, AGV_LOADED_KW, agv_energy_kwh,
                      machine_energy_kwh, machine_params_for, total_energy_kwh)
from .corridors import shortest_node_path
from .due_dates import due_dates_for
from .layout import Layout, MachinePad
from .instances import Instance
from .transport import MATRIX, UNMAPPED_RAISE, TransportCaliber

if TYPE_CHECKING:                       # 仅为标注；运行期在 `snapshot()` 里就地导入
    from .snapshot import Snapshot

# ⚠️ 单位约定（2026-10-02 量纲对齐）：**仿真时间单位 = 分钟；布局坐标单位 = 米**。
# 此前无单位换算，运输时间 = 距离（米）÷ 速度，导致 TDT/TPT ≈ 4–6 倍（物理上不成立，
# 见 progress-log 与 spec §3.4）。此常量把 [m] / [m/s] 转成分钟。
SECONDS_PER_MIN = 60.0


def rollout(inst: Instance, layout_type: str = "line", seed_layout: int = 0, seed_chain: int = 0,
            cfg: SimConfig | None = None, op_choices: list[list[int]] | None = None,
            aisle_width: float = 1.5,
            agv_phi: list[int] | None = None, constraints=None) -> dict:
    """一次完整 episode：网格布局采样(seed_layout) → 格点距离 → SimPy(seed_chain)。

    `layout_type` **保留但忽略**（旧调用方仍传）。
    `constraints` = `ConstraintConfig`（十约束开关，spec §3.3）；None → 全开。
    """
    from .corridors import build_corridor_graph, dock_distance_matrix
    from .layout import sample_layout
    # ⚠️ aisle_width 必须同时进 SimConfig——eff_speed 读的是 cfg.aisle_width（窄道降速）。
    # 否则 `rollout(aisle_width=1.0)` 只改几何、不改速度，窄道敏感性实验**静默失效**。
    eff_cfg = cfg if cfg is not None else SimConfig(aisle_width=aisle_width)
    # ⚠️ 车队规模与载量上限必须**同步进布局**，否则布局车队 ≠ 仿真车队（静默错误）。
    layout = sample_layout(inst.n_machines, seed=seed_layout, aisle_w=aisle_width,
                           n_agv=eff_cfg.n_agv, max_agv_capacity=eff_cfg.max_agv_capacity)
    g = build_corridor_graph(layout)
    dm = dock_distance_matrix(g)
    return SimWorld(inst, layout, dm, eff_cfg, graph=g, constraints=constraints).run(
        seed_chain=seed_chain, op_choices=op_choices, agv_phi=agv_phi)


@dataclass
class SimConfig:
    n_agv: int = 3          # spec §6.3 主实验值（网格上按边际收益拐点重标）
    # AGV 车速 [m/s]。**2026-10-02 由 1.0 改为 0.5**：0.5 取自我们**已引证的** GFJSPT-MMRS
    # （spec §3.4 表："AGV 3 台，0.5 m/s"）；1.0 是无出处的拍脑袋值，且使运输被低估一倍。
    # 文献区间：0.3–0.6（IEJ 随机批量运输 FJSP）、1.2（FlexSim 车间级案例）。车速是扫描轴之一。
    agv_speed_mps: float = 0.5
    zone_hold: float = 1.0        # 缓冲满退避时长 [min]
    zone_granularity: str = "node"   # 区段粒度：node|row|col|all（越粗→区段越少→争用越强）
    zone_wait_limit: float = 8.0     # 区段申请等待上限 [min]（超时→退避重试）
    repair_time: float = 5.0      # [min]
    aisle_width: float = 1.5      # 通道宽（米）：窄通道限速 = Phase A2 真权衡来源
    # ── 约束参数（**全部 assumed**，无文献出处；见 spec §9 的 assumed 表）──
    p_rework: float = 0.05        # ④ 返工率 [1/工序]
    setup_min_default: float = 2.0   # ⑤ 异型换型时长 [min]（同作业连做 = 0）
    pm_interval: float = 120.0    # ⑫ 预防性维护间隔 [min 主轴工时]
    pm_duration: float = 10.0     # ⑫ 维护停机时长 [min]
    max_agv_capacity: int = 3     # ⑩ 车队载量上限 [件]（1 = 退化为单载）
    agv_mtbf: float = 480.0       # ⑨ AGV 平均无故障时间 [min]（8 h）
    agv_mttr: float = 10.0        # ⑨ AGV 平均修复时间 [min]
    battery_low: float = 0.20     # ⑪ 低电阈值（占容量比）
    battery_high: float = 0.80    # ⑪ 充电目标（占容量比）
    charge_kw: float = 3.0        # ⑪ 充电功率 [kW]
    # ⑧ 交期**覆盖开关**：`None` = 用 `due_dates.TF_RDD` 里**逐实例标定**的 (τ, R)。
    # 2026-10-03 ⑧ 口径重设计（TF/RDD，见 `docs/superpowers/plans/2026-10-03-due-date-redesign.md`）
    # 后，tau 从"唯一标定值"降级为**覆盖**：默认 None，只在敏感性扫描/新实例标定时显式传。
    # 旧口径 `d_j = τ·M_ref` 已废，两条死因：① 锚在**自己的**参考调度上（内生——换车队规模
    # 就换 M_ref：mk01 在 n_agv=1/3 下 109.95/103.42，交期跟着漂）；② τ=0.90 只对**参考策略**
    # 标定，而训练后策略改进 12–14%，一次性把 TWT 清零（实测 makespan 89.5 < d_j=93.08 ⟹
    # TWT ≡ 0 是恒等式）。新口径只读实例数据，与 cfg 无关。
    tau: float | None = None
    # 交期跨度（due-date range, RDD）——与 tau 同进退的覆盖开关（None = 查标定表）。
    # 两者**要么都显式传、要么都不传**：只传一个时另一个仍查表，未标定实例会显式报错。
    due_range: float | None = None
    # 矩阵口径下**未被矩阵覆盖的端点**（充电桩等）怎么算行程：`raise`（默认，显式报错）或
    # `geometry`（**声明式**降级为几何口径，并进 metrics 计数）。它进 `_cfg_key`（失效安全）。
    # ⚠️ 这**不是**"口径开关"：口径永远跟随实例（见 transport.py 的共存规则），本项只管
    # "矩阵覆盖不到的那几段"。
    transport_unmapped: str = UNMAPPED_RAISE

    @property
    def eff_speed(self) -> float:
        """有效车速：通道 <1.5m 线性降速（下限 0.4×）——"宽=快但贵"权衡。
        默认 1.5 → 倍率 1.0（历史实验逐位不变）。"""
        return min(1.0, max(0.4, self.aisle_width / 1.5))


class OpLite:
    __slots__ = ("time",)
    def __init__(self, t: float):
        self.time = t


class SimTrack:
    """快照的跟踪量（P2 Task 1）：机台与车辆在运行流里**就地写**，`SimWorld.snapshot()` 只读。

    机台/车辆的"作业进行到哪一步"只有运行流自己知道（SimPy 对象里查不出来），故单列一袋。
    `MachineSim` / `AgvSim` 只拿这个袋子（不持 `SimWorld` 引用，避免环）；
    `SimWorld` 把各跟踪列表同时挂成 `_job_progress` 等属性，快照代码直接读属性。
    """
    __slots__ = ("job_progress", "job_loc", "job_agv", "cur_op", "agv_loaded", "agv_load_n",
                 "job_rework")

    def __init__(self, n_jobs: int, n_machines: int, n_agv: int) -> None:
        self.job_progress = [0] * n_jobs                      # 已完成工序数
        self.job_loc = [-1] * n_jobs                          # 当前所在机台；-1 = 不在机台上
        self.job_agv = [-1] * n_jobs                          # 在途时所乘的车；-1 = 无
        # (工件, 加工时长, 上机时刻)：快照按 `t_start + 时长 − now` 算**标称**剩余（见 snapshot）
        self.cur_op: list[tuple[int, float, float] | None] = [None] * n_machines
        self.agv_loaded = [False] * n_agv                     # 车上是否载货（状态 2 的判据）
        self.agv_load_n = [0] * n_agv                         # 车上**在运件数**（in_flight 的"车上"部分）
        # ④ 返工：逐作业的**累计**重做次数（`MachineSim.run` 的重做环里 ++，与
        # `stats["rework_events"]` 同一分支）。⚠️ 只增不减——快照要的是"至今返了几次"。
        self.job_rework = [0] * n_jobs

    def set_load(self, aid: int, n: int) -> None:
        """一趟批次的**装上**（`n = len(batch)`）或**卸下**（`n = 0`）。

        ⚠️ 两个量**同处写**：`agv_loaded`（状态 2 的判据）由 `n > 0` 推出、`agv_load_n`
        是同一个数——不给"标志说满载、计数说空车"留下失同步的口子。
        """
        self.agv_loaded[aid] = n > 0
        self.agv_load_n[aid] = n


# ══ 信息侧约束的纯函数（⑧ 交期），放在模块级以便单测直接调用 ══

def compute_due_dates(inst: Instance, tau: float | None = None,
                      due_range: float | None = None) -> dict[int, float]:
    """⑧ 交期（TF/RDD 口径）——转调 `due_dates.due_dates_for`。

    ⚠️ **签名变了**（2026-10-03 ⑧ 重设计）：不再收 `(n_jobs, tau, m_ref)`——旧口径
    `d_j = τ·M_ref` 是共同交期且锚在参考调度上（内生），新口径 `d_j = LB·τ·(1+R(2ρ_j−1))`
    **逐作业、只读实例数据**。口径、生成式与书目见 `plcsp/env/due_dates.py` 的模块 docstring。
    `tau` / `due_range` 为 None 时查 `due_dates.TF_RDD` 的逐实例标定表（未标定实例显式报错）。
    """
    return due_dates_for(inst, tau, due_range)


def weighted_tardiness(completes: dict[int, float], due: dict[int, float],
                       weights: dict[int, float]) -> float:
    """⑧ 加权总拖期 `TWT = Σ w_j · max(0, C_j − d_j)`（spec §4.1 的目标口径）。"""
    return float(sum(weights.get(j, 1.0) * max(0.0, c - due[j])
                     for j, c in completes.items() if j in due))


_MREF_CACHE: dict = {}
_MREF_BUSY = False
# 参考运行**不依赖**的 cfg 字段：`tau` / `due_range` 只进交期（交期是 metric、不进仿真时序，
# 且参考运行内被 `_MREF_BUSY` 短路）——实测换 τ 的参考 makespan **逐位相同**。故两个都不进键：
# 否则 τ 扫描（⑧ 敏感性实验）会把同一份参考运行反复重跑。
_CFG_KEY_SKIP = ("tau", "due_range")


def _instance_key(inst: Instance) -> tuple:
    """实例内容指纹——用于缓存 `M_ref`（比 id() 稳，比文件名稳）。

    ⚠️ P4-B：**行程时间口径与矩阵必须进键**——同名 MK 与 MKT 实例只差一张矩阵，漏了它
    `reference_run` 会把几何口径的参考运行**静默喂给**矩阵口径的运行（M_ref、奖励权重、
    特征归一化一起错，且零报错）。
    """
    mat = getattr(inst, "trans_time_full", None)
    digest = (None if mat is None else
              hashlib.blake2b(np.ascontiguousarray(mat, dtype=float).tobytes(),
                              digest_size=8).hexdigest())
    return (inst.n_machines, inst.n_jobs, getattr(inst, "transport", "geometry"),
            tuple(tuple(min(t for _, t in op) for op in job) for job in inst.jobs), digest)


def _cfg_key(cfg: SimConfig) -> tuple:
    """cfg 中**影响参考运行结果**的字段指纹——缓存键的第二部分（`_instance_key` 只看实例）。

    ⚠️ 漏 cfg 会静默出错：`n_agv` / `agv_speed_mps` / `aisle_width` … 一变，参考运行的整份
    metrics 都变（实测 mk01 的 M_ref：n_agv=1/3/5 → 109.95/103.42/97.24；车速 0.5/1.0 →
    103.42/106.38），于是 f^ref 错 → 奖励权重错 → 落在哪个 Pareto 点都错。
    取**全部字段减去已知无关项**（而非白名单）：新增 cfg 字段默认进键——失效安全。
    参考运行固定 `constraints=None`（十约束全开），与调用方的 `ConstraintConfig` 无关，故后者不进键。
    """
    return tuple((f.name, getattr(cfg, f.name)) for f in fields(cfg)
                 if f.name not in _CFG_KEY_SKIP)


def reference_run(inst: Instance, cfg: SimConfig | None = None,
                  seed_layout: int = 0) -> dict:
    """参考调度（每工序取最短候选 + AGV 轮询派车）的**整份 metrics**——缓存 + 重入短路。

    ⚠️ **绝不能取被优化的那次 episode 的指标**——否则交期随策略一起漂移，
    目标退化（旧 `due_factor` 就是这么坏的：实测 MK01 tardy 恒为 10/10）。
    故单独跑一次并缓存；重入时 `_due` 会退化为"无交期"（见 `SimWorld._due`）。
    ⚠️ 由此，参考运行自身的 `tardy` / `tardy_twt` **恒为 0**——取 f^ref 者须事后按 **TF/RDD
    交期**从同一次运行的 `completes` 重算 TWT（`reward.ReferenceObjectives.of` 就是这么做的；
    交期只影响 metric、不影响动力学，故事后算 = 交期开启时的值）。
    缓存键 = 实例指纹 + `seed_layout` + **影响结果的 cfg 字段**（`_cfg_key`；`tau` / `due_range` 除外）。
    """
    c = cfg or SimConfig()
    key = _instance_key(inst) + (seed_layout,) + _cfg_key(c)
    if key in _MREF_CACHE:
        return _MREF_CACHE[key]
    global _MREF_BUSY
    if _MREF_BUSY:                   # 理论上被 `_due` 的短路挡住，此处兜底
        raise RuntimeError("reference_run 重入")
    _MREF_BUSY = True
    try:
        r = rollout(inst, seed_layout=seed_layout, seed_chain=0, cfg=c, constraints=None)
    finally:
        _MREF_BUSY = False
    _MREF_CACHE[key] = r
    return r


def reference_makespan(inst: Instance, cfg: SimConfig | None = None,
                       seed_layout: int = 0) -> float:
    """参考调度的 makespan —— `reference_run` 的薄封装（既有调用点不变）。

    ⚠️ 用途**只剩两个**（2026-10-03 ⑧ 重设计后）：① 特征归一化的 `m_ref`（`build_setup` 取它）；
    ② 奖励侧参考运行/权重口径的**同源守卫**（`joint_chain_step` 的 `layout_seed=0` 前提）。
    ⑧ 交期**不再由它导出**——交期是外生的 TF/RDD（见 `due_dates.py`），与 `M_ref` 无关。
    """
    return float(reference_run(inst, cfg, seed_layout)["makespan"])


class MachineSim:
    """机台：输入缓冲 → 换型 → 加工（故障中断-恢复）→ 保养 → 输出缓冲（满则阻塞）。

    **开关一律从 `ConstraintConfig` 读**（逐个传 bool 会在约束变多时漏参数）。
    开关关闭时**不得消耗随机数**——否则故障流被移位，"该约束从未存在"的语义就不成立。
    """

    def __init__(self, env, pad: MachinePad, rng, cfg: SimConfig, stats: dict, completes: dict,
                 events_q: simpy.Store, constraints, track: SimTrack):
        self.env, self.pad, self.rng, self.cfg, self.stats = env, pad, rng, cfg, stats
        self.completes = completes
        self.events_q = events_q
        self.con = constraints
        self.track = track              # 快照跟踪量（P2 Task 1，见 `SimTrack`）
        # SimPy 的 Store 不接受 capacity=None；无界用 inf（且下游的"缓冲满"检查须能识别 inf）
        cap_in = pad.in_cap if constraints.finite_buffer else float("inf")
        cap_out = pad.out_cap if constraints.finite_buffer else float("inf")
        self.in_q = simpy.Store(env, capacity=cap_in)
        self.out_q = simpy.Store(env, capacity=cap_out)
        self.slot = simpy.Resource(env, 1)   # 机台加工槽（故障/保养期间占用）
        self.prev_job: int | None = None     # ⑤ 换型：本机上一件加工的作业
        self.pm_clock = 0.0                  # ⑫ 距上次保养累计的主轴工时 [min]

    def _process(self, job: int, op):
        """⑤ 换型 → 加工（含故障中断-恢复）→ ⑫ 保养。**不含返工判定**（见 `run`）。"""
        with self.slot.request() as req:
            yield req
            # ⑤ 换型：与上一件**不同作业**才需换型（同作业连做 = 0；序列相关的最简形）
            if self.con.setup_time and self.prev_job is not None and self.prev_job != job:
                yield self.env.timeout(self.cfg.setup_min_default)
                self.stats["setup_min"][self.pad.id] += self.cfg.setup_min_default
            self.prev_job = job
            t = self.env.now
            finish = t + op.time
            if not self.con.machine_failure:
                yield self.env.timeout(op.time)            # ③ 关：无故障，一次跑完
            else:
                while t < finish:
                    nxt_fail = t + self.rng.exponential(1.0 / max(self.pad.fail_rate, 1e-9))
                    seg = min(nxt_fail, finish) - t
                    yield self.env.timeout(seg)
                    if nxt_fail < finish:
                        self.stats["fail_events"] += 1
                        yield self.env.timeout(self.cfg.repair_time)   # 中断-恢复
                    t += seg
            self.stats["process_time"] += op.time
            self.stats["proc_min"][self.pad.id] += op.time   # M2 能耗：按机位计的切削时长
            # ⑫ 预防性维护：主轴工时到点 → 计划停机（占机台槽，工件在外面等着）
            if self.con.maintenance:
                self.pm_clock += op.time
                if self.pm_clock >= self.cfg.pm_interval:
                    yield self.env.timeout(self.cfg.pm_duration)
                    self.stats["pm_events"] += 1
                    self.pm_clock = 0.0

    def run(self):
        while True:
            job, oi, op, is_last = yield self.in_q.get()
            self.stats["in_q_gets"] = self.stats.get("in_q_gets", 0) + 1
            # 快照跟踪（P2 Task 1）：工件上机 → 记在制工序、上机时刻与所在地
            # ⚠️ 上机时刻 = 此刻（换型/故障修复的墙钟延长都不计入剩余——标称口径）
            self.track.cur_op[self.pad.id] = (job, op.time, self.env.now)
            self.track.job_loc[job] = self.pad.id
            # ④ 返工：同件在本机**原地**重做（工件不离开机台，故不走运输）。
            # ⚠️ 必须原地——早期版本走 `in_q.put` 会**自锁**：本机是该缓冲的唯一消费者，
            # 缓冲满时 put 永久阻塞，而此时还占着加工槽 ⇒ 该机位连同工件一起卡死
            # （实测 MK07/MK10 5/5 掐表，关掉返工即 0/5）。
            while True:
                yield from self._process(job, op)
                if self.con.rework and self.rng.random() < self.cfg.p_rework:
                    self.stats["rework_events"] += 1
                    # 快照跟踪（①/④ 特征）：逐作业计数与全局事件数**同分支 ++**，
                    # 故 `Σ job_rework == stats["rework_events"]` 恒成立（有测试钉）。
                    self.track.job_rework[job] += 1
                    continue
                break
            # 快照跟踪（P2 Task 1）：工序加工完毕（返工不算进度）→ 清在制、推进度
            self.track.cur_op[self.pad.id] = None
            self.track.job_progress[job] = oi + 1
            self.stats["ops_done"] = self.stats.get("ops_done", 0) + 1
            # ⚠️ **末工序不再在这里记完工**（P4-B Task 2b）：工件还要**回装卸站**，
            # `completes[j]` = 到站时刻（由 `LuStation` 记）。两类出口同走"先发事件 →
            # 再入 out_q"的顺序（防 put→事件 顺序死锁；transporter 的 get 即等待者）。
            self.events_q.put((self.pad.id, job, oi, op, is_last))
            yield self.out_q.put((job, oi, op, is_last))            # transporter 取走（含末工序）


class _StationPad:
    """`AgvSim`/transporter 只读 `.pad.dock_node`——装卸站用这个最小形状与机台共用索引表，
    **不**把装卸站伪装成 `MachinePad`（它不加工、没有故障率/缓冲容量）。"""
    __slots__ = ("dock_node",)

    def __init__(self, node: int):
        self.dock_node = node


class LuStation:
    """装卸站（Load/Unload unit）：作业在此**入场**、末工序完工后**回站**。不加工、不改工序。

    形态、书目与"为什么不是某个交叉口"见 `layout.LuPad`。仿真里它只承担两件事：

    1. **回站落点**：AGV 把回站工件投递进 `in_q`，**到达时刻即 `completes[j]`**（makespan 口径）；
    2. **与机台共用端点号空间**：机台 `0..m-1`、装卸站 = `m`（= `machines` 表的末位），
       于是 `_transporter` / `AgvSim` 的既有索引逻辑对两类端点一视同仁。
    """

    def __init__(self, env, pad, idx: int, completes: dict, track: SimTrack):
        self.env, self.id = env, idx
        self.pad = _StationPad(pad.node)     # ⚠️ 站自己的节点号（格点外），不是它接入的交叉口
        self.completes, self.track = completes, track
        self.in_q = simpy.Store(env)         # 回站落点：无容量上限（站是终点，永不阻塞 AGV）
        self.out_q = simpy.Store(env)        # 形状与 MachineSim 对齐（transporter 只读机台的）

    def run(self):
        while True:
            job, _oi, _op, _is_last = yield self.in_q.get()
            self.completes[job] = self.env.now      # 完工 = **到达装卸站**（不是末工序下机）
            self.track.job_agv[job] = -1
            self.track.job_loc[job] = -1


def build_zone_map(layout, granularity: str) -> tuple[dict[int, int], int]:
    """按粒度构造「节点 → 区段」映射。**粒度越粗 → 区段越少 → 争用越强。**

    - `node`：每个通道节点一个区段（最细；MK10 的 5×4 网格 → 30 个）
    - `row` / `col`：整行 / 整列算一个区段（像"一条长廊一个区段"）
    - `all`：全图一个区段（极端档，用于下界）

    ⚠️ 装卸站是**格点外**节点（号 = `grid.n_nodes`），`row`/`col` 没有它的行列号 ⟹
    给它**独占**一个区段（`all` 档则并入唯一那一个）——它不是路口，不该与某一行共用区段。
    """
    spec = layout.grid
    n = spec.n_nodes                      # 格点交叉口数（装卸站不在其中）
    if granularity == "all":
        return dict.fromkeys(range(n + 1), 0), 1
    if granularity == "node":
        zof, nz = {i: i for i in range(n)}, n
    elif granularity == "row":
        zof, nz = {i: spec.node_rc(i)[0] for i in range(n)}, spec.n_rows + 1
    elif granularity == "col":
        zof, nz = {i: spec.node_rc(i)[1] for i in range(n)}, spec.n_cols + 1
    else:
        raise ValueError(f"未知区段粒度：{granularity}")
    zof[spec.lu_node] = nz                # 装卸站独占一格（node 粒度下恰好等于它自己的号）
    return zof, nz + 1


class ZoneManager:
    """区段管制：**一区段一车互斥** + 等待环检测（经典 AGV 死锁的防法）。

    ⚠️ **区段粒度是参数**（`SimConfig.zone_granularity`）：粒度越粗 → 区段数越少 → 争用越强。
    真实 AGV 系统两种都有（长廊算一个区段 vs 每个路口一个），故做成可扫描的参数。

    `zone_of`: 节点号 → 区段号 的映射（由 `SimWorld` 按粒度构造）。
    """

    def __init__(self, env: simpy.Environment, zone_of: dict[int, int], n_zones: int,
                 wait_limit: float):
        self.env, self.wait_limit, self.zone_of, self.n = env, wait_limit, zone_of, n_zones
        self.holder: dict[int, int | None] = dict.fromkeys(range(n_zones))
        self.pending: dict[int, int] = {}
        # AGV → **本次**等待的申请时刻 [min]。与 `pending` **同处增删**：只描述"此刻还在等"，
        # 一放行/超时就清（`current_wait` 的口径；累计史在 `waits`，两者不可混用）。
        self.pending_since: dict[int, float] = {}
        self._ev: dict[int, simpy.Event] = {}
        self.waits: list[float] = []      # 区段等待时长 [min]——拥堵的唯一度量

    # ── 等待环检测（不变）──
    def _wait_path_to(self, start: int, target: int) -> bool:
        node, seen = start, set()
        while node is not None and node != target:
            if node in seen:
                return False
            seen.add(node)
            z = self.pending.get(node)
            if z is None:
                return False
            node = self.holder.get(z)
        return node == target

    def would_cycle(self, agv: int, z: int) -> bool:
        return self.holder.get(z) not in (None, agv) and self._wait_path_to(self.holder[z], agv)

    # ── 申请 / 释放 ──
    def try_grant(self, agv: int, z: int) -> bool:
        if self.holder[z] == agv:
            return True
        if self.holder[z] is not None:
            return False
        self.holder[z] = agv
        return True

    def current_wait(self, agv: int) -> float:
        """本车**此刻**等待区段的时长 [min]；不在等待 = 0.0（特征层 `zone_wait` 维的唯一原料）。

        ⚠️ 与 `waits`（**完成后**追加的累计史）不同：这是"当前还在等多久"，放行/超时即清零
        （`pending_since` 与 `pending` 同处增删）。读累计史会把"等过"报成"还在等"（假信号）。
        """
        t0 = self.pending_since.get(agv)
        return 0.0 if t0 is None else max(0.0, float(self.env.now) - t0)

    def wait_zone(self, agv: int, z: int, limit: float | None = None):
        if self.try_grant(agv, z):
            return True, False
        if self.would_cycle(agv, z):
            return False, True
        self.pending[agv] = z
        t0 = self.env.now
        self.pending_since[agv] = t0
        ev = self._ev.setdefault(z, simpy.Event(self.env))
        yield ev | self.env.timeout(limit or self.wait_limit)
        self.waits.append(self.env.now - t0)
        self.pending.pop(agv, None)
        self.pending_since.pop(agv, None)
        if self.try_grant(agv, z):
            return True, False
        return False, False

    def release(self, agv: int, z: int) -> None:
        if self.holder.get(z) == agv:
            self.holder[z] = None
            ev = self._ev.get(z)
            if ev is not None and not ev.triggered:
                ev.succeed()
            self._ev[z] = simpy.Event(self.env)
        self.pending.pop(agv, None)
        self.pending_since.pop(agv, None)


class AgvSim:
    """AGV：待命 → 空载驶向取货机台 → 装载驶向卸货机台 → 投递。

    **两个行驶段**（P1b Task 2 补齐）：
    - **空载段**：从当前停位开到取货点。v0 **缺失此段**（AGV 从上一个卸货点"瞬移"到取货点），
      等于凭空多出运力、且使 AGV 只有"负载"一个状态。补齐后运输负荷才真实。
    - **负载段**：取货点 → 卸货点，沿用格点最短路。

    区段管制（① congestion）**可开关**：开时逐段申请/释放区段（持当前 → 申请下一 → 放上一），
    等待环或超时则回队重试；关时不申请，按距离一次到底。
    """

    def __init__(self, env, aid, m_dm: np.ndarray, transport: TransportCaliber, cfg: SimConfig,
                 stats: dict,
                 tasks_in, machines: list, graph, zm, constraints, spec, rng,
                 track: SimTrack, chargers=(), charger_res=(), bound: bool = False):
        self.env, self.aid, self.m_dm, self.cfg = env, aid, m_dm, cfg
        self.transport = transport          # 行程时间口径（P4-B：跟随实例，不是全局开关）
        self.stats, self.tasks_in, self.machines = stats, tasks_in, machines
        self.track = track              # 快照跟踪量（P2 Task 1，见 `SimTrack`）
        self.g, self.zm = graph, zm
        self.con = constraints              # ① congestion 等物流侧开关从这里读
        self.congestion = constraints.congestion    # ① 关 → 无区段管制
        self.bound = bound          # True: 任务按 agv_phi 绑定（每车一个 Store = L 层决策载体）
        self.pos_node: int | None = None    # 当前所在通道节点；None = 尚未出车（停在首个取货点）
        # ⑩ 异构车队：**开关关掉时倍率=1、载量=1**，即与"约束从未存在"逐位相同
        hetero = constraints.heterogeneous_fleet
        self.capacity = spec.capacity if hetero else 1
        self.speed = cfg.agv_speed_mps * (spec.speed_factor if hetero else 1.0)
        self.speed_ratio = spec.speed_factor if hetero else 1.0   # ⑩ 的**相对**倍率（关掉恒 1）
        self.battery_cap = spec.battery_kwh
        self.battery = spec.battery_kwh     # [kWh]，只在 ⑪ 开启时增减
        self.chargers, self.charger_res = list(chargers), list(charger_res)
        # ⑨ 故障：**每台车一条独立随机流**——否则关掉 AGV 故障会移位机台故障流（消融不干净）
        self.rng = rng
        self.down = False
        self.up = simpy.Event(env)
        self.up.succeed()                   # 初始可用

    def _seg_min(self, u: int, v: int) -> float:
        """节点 u→v 的**几何**行驶时长 [min]（格点最短路 ÷ 有效车速 ÷ 60）。

        ⚠️ 矩阵口径**不走这里**（矩阵已是分钟，不得再换算）——那条路走 `_leg_min`。
        """
        return float(self.m_dm[u, v]) / (self.cfg.eff_speed * self.speed) / SECONDS_PER_MIN

    def _leg_min(self, src: int, dst: int, *, count_unmapped: bool = True) -> float:
        """**整段** src→dst 的行驶时长 [min]——两种口径的唯一汇合点（不含区段拆分）。

        - 几何口径：转调 `_seg_min`（现状，逐位不变）；
        - 矩阵口径：**直接查表**（矩阵已是分钟，**不得**再套任何换算），只再乘 ⑩ 的**相对**
          速度倍率（⑩ 关 ⟹ 倍率恒 1 ⟹ 与"该约束从未存在"逐位相同）；
        - 任一端点**没有矩阵对应项**（充电桩）⟹ 按 `transport.unmapped` 处置：`raise` 显式报错；
          `geometry` **声明式**降级为几何口径并计数（metrics 带出，供表里如实声明）。
        `count_unmapped=False` 供**排序**之类的查数用（只选一个桩却把候选全计一遍会虚高）。
        ⚠️ 计数按**出发次数**记：区段争用失败而重试的腿会重复计一次（它确实又跑了一趟）。
        """
        if self.transport.mode != MATRIX:
            return self._seg_min(src, dst)
        t = self.transport.minutes(src, dst)
        if t is not None:
            return t / self.speed_ratio
        if self.transport.unmapped == UNMAPPED_RAISE:
            raise ValueError(
                f"节点 {src}→{dst} 在行程时间矩阵里**没有对应项**（矩阵只覆盖机台与装卸站）——"
                f"若要跑含充电桩的口径，请显式设 SimConfig.transport_unmapped='geometry'")
        self.stats["unmapped_legs"] = self.stats.get("unmapped_legs", 0) + 1
        got = self._seg_min(src, dst)
        self.stats["unmapped_min"] = self.stats.get("unmapped_min", 0.0) + got
        return got

    def _drain(self, leg: str, minutes: float) -> None:
        """⑪ 行驶耗电：kWh = kW × min ÷ 60（与 `energy.py` 同一套三态功率）。"""
        if not self.con.charging:
            return
        kw = AGV_EMPTY_KW if leg == "empty" else AGV_LOADED_KW
        self.battery = max(0.0, self.battery - kw * minutes / 60.0)
        self.stats["battery_min_kwh"] = min(self.stats.get("battery_min_kwh", self.battery_cap),
                                            self.battery)

    def _failures(self):
        """⑨ AGV 故障：按泊松流停机 `agv_mttr`，期间不接活。

        **腿间检出**：故障不打断正在进行的行驶（在途任务滞留在车上），
        在下一段行驶开始前生效。这是简化，但 MTBF(480 min) 远大于单段行驶时长，误差可忽略。
        """
        while True:
            yield self.env.timeout(self.rng.exponential(self.cfg.agv_mtbf))
            self.stats["agv_fail_events"] += 1
            # ⚠️ 必须**先换一个新的未触发事件**再置 down：否则等待者会反复 yield 一个
            # 已经 succeed 的事件 → 仿真时间不推进、空转成死循环（实测 pytest 直接挂死）。
            self.up = simpy.Event(self.env)
            self.down = True
            yield self.env.timeout(self.cfg.agv_mttr)
            self.down = False
            self.up.succeed()

    def _wait_up(self):
        """⑨ 若当前停机，等到修复。"""
        while self.down:
            yield self.up

    def _maybe_charge(self):
        """⑪ 低电 → 就近找**空闲**充电桩充到 `battery_high`。

        只在**待命时**充电，不在取货/送货途中中断——中断会把在途工件撂在半路。
        桩被占则试下一个（本车不排队干等，回队后下一轮再试）。
        """
        if not self.con.charging or not self.chargers:
            return
        if self.battery > self.cfg.battery_low * self.battery_cap:
            return
        src = self.pos_node if self.pos_node is not None else self.chargers[0].node
        for ch in sorted(self.chargers, key=lambda c: self._leg_min(src, c.node, count_unmapped=False)):
            res = self.charger_res[ch.id]
            if res.count >= res.capacity:       # 桩被占（SimPy 单线程，检查与申请之间无 yield）
                continue
            with res.request() as req:
                yield req
                ok, self.pos_node = yield from self._drive(src, ch.node, "empty")
                if not ok:
                    return
                need = self.cfg.battery_high * self.battery_cap - self.battery
                yield self.env.timeout(need / self.cfg.charge_kw * 60.0)   # kWh ÷ kW → h → min
                self.battery = self.cfg.battery_high * self.battery_cap
                self.stats["charge_events"] += 1
            return

    def _collect(self, q, frm: int, to: int, first: tuple) -> list:
        """⑩ 同向拼车：从**队首连续段**取走与本件同 (取货点, 卸货点) 的任务，最多 `capacity` 件。

        只看队首连续段（不搜全队列）——故**不会打乱其它任务的相对顺序**，代价是拼车机会变少。
        容量为 1 时直接返回单件（退化即"该约束从未存在"）。
        """
        batch = [first]
        while len(batch) < self.capacity and q.items:
            nxt = q.items[0]                    # 偷看队首；与下面的 get 之间无 yield，安全
            if (nxt[0], nxt[1]) != (frm, to):
                break
            batch.append((yield q.get()))
            self.stats["tasks_get"] += 1
        return batch

    def _drive(self, src: int, dst: int, leg: str):
        """把车从 `src` 节点开到 `dst` 节点，行驶时长累入 `leg` 态（"empty" / "loaded"）。

        返回 `(ok, end_node)`。`ok=False` = 区段争用超时或等待环——此时车停在 `end_node`
        且**已释放**持有的区段；**失败前已走的路程照样计入时长**（否则能耗会被低估）。
        区段管制关闭时按最短路一次到底。
        """
        if src == dst:
            return True, dst
        if not self.congestion:                 # ① 关：不申请区段，按距离直行
            seg = self._leg_min(src, dst)
            yield self.env.timeout(seg)
            self._drain(leg, seg)               # ⑪ 行驶耗电
            self.stats["travel_time"] += seg
            self.stats[f"agv_{leg}_min"] += seg
            self.stats["moves"] += 1
            return True, dst
        # 逐段申请区段：持当前 → 申请下一 → 成功才放上一 → 走这一段
        path = shortest_node_path(self.g, src, dst)
        # ⚠️ 矩阵口径下**整段时长先算出来**，逐段只决定**分摊比例**（占比之和恒为 1 ⟹
        # 各段之和 == 矩阵查表值，有测试钉）；矩阵里没有"中间走廊节点"这一说，逐段查表必失败。
        total = self._leg_min(src, dst) if self.transport.mode == MATRIX else None
        leg_geo = self._seg_min(src, dst) if total is not None else 0.0
        zseq = [self.zm.zone_of[p] for p in path]
        zseq = [z for i, z in enumerate(zseq) if i == 0 or z != zseq[i - 1]]   # 合并同一区段的连续段
        granted, cycle = yield from self.zm.wait_zone(self.aid, zseq[0], self.cfg.zone_wait_limit)
        if cycle or not granted:
            return False, src
        travel, end, prev_z = 0.0, src, zseq[0]
        for pi in range(len(path) - 1):
            z = self.zm.zone_of[path[pi + 1]]
            if z != prev_z:
                granted, cycle = yield from self.zm.wait_zone(self.aid, z, self.cfg.zone_wait_limit)
                if cycle or not granted:
                    self.zm.release(self.aid, prev_z)
                    self.stats["travel_time"] += travel
                    self.stats[f"agv_{leg}_min"] += travel
                    return False, end
                self.zm.release(self.aid, prev_z)
                prev_z = z
            seg = self._seg_min(path[pi], path[pi + 1])
            if total is not None:
                seg = total * seg / leg_geo
            yield self.env.timeout(seg)
            self._drain(leg, seg)               # ⑪ 行驶耗电
            travel += seg
            end = path[pi + 1]
            self.stats["moves"] += 1
        self.zm.release(self.aid, prev_z)
        self.stats["travel_time"] += travel
        self.stats[f"agv_{leg}_min"] += travel
        return True, dst

    def run(self):
        q = self.tasks_in[self.aid] if self.bound else self.tasks_in
        if self.con.agv_failure:
            self.env.process(self._failures())       # ⑨ 关掉时不启进程 → 不抽随机数
        while True:
            yield from self._wait_up()               # ⑨ 停机中不接活
            yield from self._maybe_charge()          # ⑪ 待命时补电
            yield from self._wait_up()
            t0 = self.env.now
            frm, to, item, path = yield q.get()
            if self.con.charging:                    # ⑪ 待命也耗电（三态口径）
                self._drain_idle(self.env.now - t0, AGV_IDLE_KW)
            self.stats["tasks_get"] += 1
            a = self.machines[frm].pad.dock_node
            b = self.machines[to].pad.dock_node
            # ── 空载段：当前停位 → 取货点 ──
            if self.pos_node is not None and self.pos_node != a:
                ok, self.pos_node = yield from self._drive(self.pos_node, a, "empty")
                if not ok:
                    self.stats["requeue"] = self.stats.get("requeue", 0) + 1
                    q.put((frm, to, item, path))
                    # 快照跟踪（P2 Task 1）：退回队列 = 这批活不在这台车上（车还是空的）
                    self.track.job_agv[item[0]] = -1
                    self.track.set_load(self.aid, 0)
                    continue
            # ⑩ 同向拼车：把队首连续的同 (取货点, 卸货点) 任务一并装走
            batch = yield from self._collect(q, frm, to, (frm, to, item, path))
            # 快照跟踪（P2 Task 1）：取到任务即记车号——FIFO 路径没有派车分支，只靠这里，
            # 否则 `JobState.in_transit` 在旧规则路径上恒为 False。
            for (_f0, _t0, _it0, _p0) in batch:
                self.track.job_agv[_it0[0]] = self.aid
            # ── 负载段：取货点 → 卸货点（整批一趟）──
            # 快照跟踪（P2 Task 1）：取货后负载行驶，本趟在运件数 = 批次大小（F5：in_flight 的"车上"部分）
            self.track.set_load(self.aid, len(batch))
            ok, self.pos_node = yield from self._drive(a, b, "loaded")
            if not ok:
                # 快照跟踪（P2 Task 1）：退回队列 = **当场**卸空——车要带着"空车"的状态在
                # 取货点干等 zone_hold，不能挂着"负载行驶"（否则状态 2 / 在途会滞留到重试）
                self.track.set_load(self.aid, 0)
                for t in batch:
                    self.track.job_agv[t[2][0]] = -1
                yield self.env.timeout(self.cfg.zone_hold)
                self.stats["requeue"] = self.stats.get("requeue", 0) + 1
                for t in batch:                      # 整批退回队列
                    q.put(t)
                continue
            self.stats["trips"] += 1
            for (_f, t2, _item2, _p2) in batch:
                self.stats["agv_del"][self.aid] += 1  # L 层状态轨迹（负载差/最后位置）
                self.stats["agv_pos"][self.aid] = t2
                # 有界输入缓冲投递：满则让步超时重试（防缓冲满阻塞拖累运输环）。
                while len(self.machines[t2].in_q.items) >= self.machines[t2].in_q.capacity:
                    yield self.env.timeout(self.cfg.zone_hold)
                yield self.machines[t2].in_q.put(_item2)
                self.track.job_agv[_item2[0]] = -1       # 快照跟踪（P2 Task 1）：投递完成，工件离车
                self.stats["deliveries"] += 1
            self.track.set_load(self.aid, 0)             # 快照跟踪（P2 Task 1）：整批卸空

    def _drain_idle(self, minutes: float, idle_kw: float) -> None:
        """⑪ 待命耗电（与 `_drain` 同一口径，只是功率取待机值）。"""
        if minutes <= 0.0:
            return
        self.battery = max(0.0, self.battery - idle_kw * minutes / 60.0)
        self.stats["battery_min_kwh"] = min(self.stats.get("battery_min_kwh", self.battery_cap),
                                            self.battery)

class SimWorld:
    """一次 episode：1 布局 + 1 实例 + 1 条独立扰动流 → 指标 dict。"""

    def __init__(self, inst: Instance, layout: Layout, m_dm: np.ndarray,
                 cfg: SimConfig | None = None, graph=None, constraints=None):
        self.inst, self.layout, self.m_dm = inst, layout, m_dm
        self.cfg = cfg or SimConfig()
        # 格点走廊图（AGV 空载段/区段路径的来源）。`graph=None` 时**自建**——
        # 此前 6 处调用点（训练环 group_rel ×3、m11、m6_*  ×2）都省略了它，
        # 在旧代码里恰好无害（AGV 用的是任务里带好的 path），一旦 AGV 需要自己算路径
        # 就会在运行期炸成 `dijkstra_path(None, …)`。自建彻底消除这个坑。
        if graph is None:
            from .corridors import build_corridor_graph
            graph = build_corridor_graph(layout)
        self.g = graph
        if constraints is None:
            from .constraints import ConstraintConfig
            constraints = ConstraintConfig()
        self.constraints = constraints      # 十约束开关（spec §3.3）
        # 行程时间口径（P4-B）：**跟随实例**（MKT 实例走矩阵、原始 MK 走几何）。只在这里
        # 认一次，整条运行链（含快照/训练栈建的世界）拿到的是同一个口径对象。
        self.transport = TransportCaliber.for_instance(inst, layout,
                                                       unmapped=self.cfg.transport_unmapped)
        self._cold_start()                  # t=0 快照骨架（P2 Task 1，run() 会整套换掉）

    def _fleet(self) -> list:
        """布局车队。⚠️ 与仿真车队不符 = 静默错误（布局车队 ≠ 仿真车队），直接报错。"""
        fleet = self.layout.agvs or []
        if len(fleet) != self.cfg.n_agv:
            raise ValueError(
                f"布局车队 {len(fleet)} 台 ≠ SimConfig.n_agv={self.cfg.n_agv}——"
                "直接构造 SimWorld 时须传 sample_layout(..., n_agv=cfg.n_agv)")
        return fleet

    def _build_entities(self, env, stats: dict, completes: dict, rng, *, bound: bool,
                        seed_chain: int = 0, charger_res=()) -> tuple[
                            list, list[simpy.Store] | simpy.Store, ZoneManager, simpy.Store]:
        """建机台 / 装卸站 / 任务队列 / 车辆 / 跟踪量，并把活引用挂到 `self`（`snapshot()` 读它们）。

        `run()` / `run_gated()` / `_cold_start()` **共用同一份**构造（逐字重复是本仓评审会判
        缺陷的形态）。⚠️ 只建对象、**不启进程**：`env.process` 的**注册顺序**决定同刻事件
        次序，故启动由调用方按原顺序做（机台 → 车辆）。返回调用方后续还要用的实体
        `(entities, tasks_in, zm, events_q)`（`self.*` 上的活引用已一并挂好）。

        ⚠️ **`entities` 比 `self.machines` 多一项**：末位是装卸站（端点号 = `inst.n_machines`）。
        任务端点、AGV 的投递目标都走这张表；而 `self.machines` **只含机台**——`snapshot()` 与
        `_energy_report` 的逐机台口径不能被装卸站污染（它不是机台）。
        """
        fleet = self._fleet()
        zof, nz = build_zone_map(self.layout, self.cfg.zone_granularity)
        zm = ZoneManager(env, zof, nz, self.cfg.zone_wait_limit)
        events_q = simpy.Store(env)
        track = SimTrack(self.inst.n_jobs, self.inst.n_machines, self.cfg.n_agv)
        machines = [MachineSim(env, self.layout.machines[i], rng, self.cfg, stats, completes,
                               events_q, self.constraints, track)
                    for i in range(self.inst.n_machines)]
        lu = LuStation(env, self.layout.lu, self.inst.n_machines, completes, track)
        entities = machines + [lu]
        if bound:
            tasks_in = [simpy.Store(env) for _ in range(self.cfg.n_agv)]   # L 层绑定：每车一队列
        else:
            tasks_in = simpy.Store(env)                                    # 旧 FIFO 规则路径
        agvs = [AgvSim(env, a, self.m_dm, self.transport, self.cfg, stats, tasks_in, entities,
                       self.g, zm, self.constraints, fleet[a],
                       np.random.default_rng([seed_chain, 1000 + a]),      # ⑨ 每车独立流
                       track, chargers=self.layout.chargers, charger_res=charger_res,
                       bound=bound)
                for a in range(self.cfg.n_agv)]
        # 活状态引用（P2 Task 1）：`snapshot()` 据此取**当时**的快照
        self.env = env
        self.completes = completes
        self.machines = machines            # ⚠️ 只含机台（装卸站在 `self.lu`，不进快照的机台段）
        self.lu = lu
        self.tasks_in = tasks_in
        self.track = track
        self.zm = zm                        # ① 拥堵：`snapshot()` 取**每车当前**区段等待时长
        self._job_progress = track.job_progress
        self._job_loc = track.job_loc
        self._job_agv = track.job_agv
        self._job_rework = track.job_rework  # ④ 返工：逐作业累计（`snapshot()` 只读）
        self._cur_op = track.cur_op
        self._agv_loaded = track.agv_loaded
        self._agv_load_n = track.agv_load_n
        self.agvs = agvs
        return entities, tasks_in, zm, events_q

    def _cold_start(self) -> None:
        """建 t=0 的空世界：**只建对象，不启进程、不抽随机数**。

        `snapshot()` 必须在 `run()` 之前也可用（决策日志与特征层都可能要"初始状态"，
        单测也直接取 t=0 快照），而 `w.machines[m].in_q` 这类活对象在 `run()` 前并不存在。
        假 stats + 全新 env ⇒ 对随后 `run()` 的行为零影响（已用指标逐位对拍证明）。
        """
        self._build_entities(simpy.Environment(), {}, {}, np.random.default_rng(0), bound=True)

    def _due(self, plans: dict[int, list[tuple[int, float]]]) -> dict[int, float]:
        """⑧ 交期（TF/RDD 口径，逐作业）。**开关关闭 → 返回空 dict**（该实例无交期，目标无拖期项）。

        旧口径 `due_factor × Σ工时` **已废弃**（spec §3.5：完全没算排队/运输/争用，
        实测 tardy 恒为 100%）。τ·M_ref 的旧口径亦已废弃（2026-10-03 ⑧ 重设计：内生且
        改进后退化）。旧字段 `due_factor` / `energy_power` 已随 P1b 清理删除。
        """
        return self._due_map()

    def _due_map(self) -> dict[int, float]:
        """交期表（只读 `self.inst` / `self.cfg`）——`_due` 与 `snapshot()` **共用此一处**，防口径漂。

        ⚠️ 无参数：交期**外生**，不随布局/车队规模/参考调度变（旧签名收 `n_jobs` 并由内部
        取 `M_ref`——那正是"交期随我们的配置漂"的病根）。
        ⚠️ `_MREF_BUSY` 短路必须留着：参考调度自身的运行里**不得**算交期——参考运行的指标
        口径把 ⑧ 当关（f^ref 的 TWT 由 `reward.ReferenceObjectives.of` 事后从**同一次运行**的
        `completes` 重算，两边同源正是 Review Focus #5 的要求）。故参考运行内本函数返回 `{}`
        （快照里对应 `JobState.due = 0.0`）。旧口径下它还是递归护栏（求交期要参考运行、
        参考运行又要求交期）；新口径已不可能递归，但"参考运行无交期"的语义要保留。
        """
        if not self.constraints.due_dates or _MREF_BUSY:
            return {}                    # 参考调度自身运行时不递归求交期
        return compute_due_dates(self.inst, self.cfg.tau, self.cfg.due_range)

    def _tardy(self, plans, completes, due) -> tuple[int, float]:
        """(误期作业数, 加权总拖期 TWT)。无交期时两者恒为 0。"""
        if not due:
            return 0, 0.0
        cnt = sum(1 for j in plans if j in completes and completes[j] > due[j])
        weights = dict.fromkeys(plans, 1.0)      # 权重：暂取等权（assumed，见 spec §9）
        return cnt, weighted_tardiness(completes, due, weights)

    def _energy_report(self, stats: dict, makespan: float) -> dict:
        """三态时长 → M2 能耗（spec §3.4，引证 GFJSPT-MMRS）。

        **占用时长的口径**（关键，防"状态统计漏一段"）：
        - 机床：只有 `proc_min`（切削）与 `setup_min`（换型，Task 4 接入）是**显式**累计的，
          其余全部归入 `idle_min`，由**闭合恒等式**给出：`makespan − proc − setup`。
          这样**三个状态之和恒等于在场时长**，不留残差。故障修复期、缓冲满阻塞期、
          等待期都自动落在空闲态——物理上也都对（机床通电但主轴不切削）。
        - AGV：`empty_min` / `loaded_min` 显式累计，`idle_min = makespan × 车数 − 两者之和`。
          待命、等待取货、缓冲区满让步、区段争用等待一并归入空闲态（车辆静止即待机功率）。

        `horizon_hit` 时 makespan 可能小于某台机的加工结束时刻，故 `idle` 取 `max(0, ·)` 兜底。
        """
        proc, setup = stats["proc_min"], stats["setup_min"]
        per_machine = []
        for m in range(len(proc)):
            p = machine_params_for(m, len(proc))
            idle = max(0.0, makespan - proc[m] - setup[m])
            per_machine.append(machine_energy_kwh(proc[m], idle, setup[m],
                                                  idle_kw=p["idle_kw"], proc_kw=p["proc_kw"],
                                                  setup_kw=p["setup_kw"]))
        machine_kwh = float(sum(per_machine))
        agv_idle = max(0.0, makespan * self.cfg.n_agv
                       - stats["agv_empty_min"] - stats["agv_loaded_min"])
        agv_kwh = agv_energy_kwh(agv_idle, stats["agv_empty_min"], stats["agv_loaded_min"])
        return {"machine_kwh": machine_kwh, "agv_kwh": agv_kwh,
                "shop_kwh": total_energy_kwh(0.0, 0.0, makespan),   # 仅车间固定项
                "total_kwh": total_energy_kwh(machine_kwh, agv_kwh, makespan),
                "machine_kwh_per_machine": per_machine,
                "agv_states_min": {"idle": agv_idle, "empty": stats["agv_empty_min"],
                                   "loaded": stats["agv_loaded_min"]}}

    def snapshot(self) -> "Snapshot":
        """当时的活状态 → 纯数据快照（**只读**，不改变任何仿真状态）。

        `_job_progress[j]` / `_job_loc[j]` / `_job_agv[j]` 由 `MachineSim` 与 `AgvSim` 维护
        （见 `run()` 里的初始化），是"作业进行到哪一步"的唯一真相。

        **口径**：`MachineState.remaining_min` 是**标称**剩余——`上机时刻 + 工序时长 − now`，
        即只算工序本身的时长，**不把换型、故障修复这些墙钟延长算进去**（故异常长的停机可能
        让它触底为 0，而工件实际还在机台上）。`JobState.remaining_min` 同理，是剩余各工序的
        **标称**最短候选工时之和。`JobState.due` = ⑧ 的交期（与 `run()` 同口径的 `_due_map`；
        约束关闭或参考运行内为 0.0）。`in_flight` = **队列里待取** + **车上在运**（两种队列形状
        都算，见下）。另外两个约束量：`JobState.rework_cnt` = 该作业**累计**返工次数（与
        `stats["rework_events"]` 同分支 ++）；`VehicleState.zone_wait` = 该车**当前**区段等待
        时长（`ZoneManager.current_wait`，不在等待 = 0.0，放行即清零）。
        """
        from .snapshot import JobState, MachineState, Snapshot, VehicleState
        ms = []
        for i, m in enumerate(self.machines):
            q = m.in_q.items
            cur = self._cur_op[i]
            ms.append(MachineState(
                backlog_min=float(sum(it[2].time for it in q)),
                in_q_len=len(q), in_cap=float(m.in_q.capacity),
                out_q_len=len(m.out_q.items), out_cap=float(m.out_q.capacity),
                busy=bool(m.slot.count),
                remaining_min=(max(0.0, cur[2] + cur[1] - float(self.env.now)) if cur else 0.0),
                pm_used_min=float(m.pm_clock), fail_rate=float(m.pad.fail_rate),
                prev_job=(-1 if m.prev_job is None else int(m.prev_job))))
        js = []
        # ⑧ 交期与 run() **同口径**（`_due_map`）：未 run 的 t=0 快照也取（交期是纯实例数据的
        # 查询，无墙钟开销）；参考运行自身运行时返回 {}（0.0 哨兵）。
        due = self._due_map()
        for j, job_ops in enumerate(self.inst.jobs):
            done = self._job_progress[j]
            js.append(JobState(
                done_ops=done, total_ops=len(job_ops),
                remaining_min=float(sum(min(t for _m, t in op) for op in job_ops[done:])),
                finished=done >= len(job_ops), due=float(due.get(j, 0.0)),
                at_machine=int(self._job_loc[j]),
                in_transit=bool(self._job_agv[j] >= 0), on_agv=int(self._job_agv[j]),
                rework_cnt=int(self._job_rework[j])))
        vs = []
        for a, agv in enumerate(self.agvs):
            vs.append(VehicleState(
                status=3 if agv.down else (2 if self._agv_loaded[a] else
                                           (1 if agv.pos_node is not None else 0)),
                node=int(agv.pos_node if agv.pos_node is not None else -1),
                queued=len(self.tasks_in[a].items) if agv.bound else 0,
                battery_frac=float(agv.battery / max(agv.battery_cap, 1e-9)),
                capacity=int(agv.capacity), speed_factor=float(agv.speed / self.cfg.agv_speed_mps),
                zone_wait=float(self.zm.current_wait(a))))
        # 在途 = **队列里待取** + **车上在运**。队列有两种形状（绑定=每车一 Store，
        # FIFO=单个共享 Store）——只看 `agv.bound` 那种形状会让 FIFO 路径恒为 0（F5）；
        # 车上那部分必须单独数，因为已装车的批次**已经离开队列**。
        queued = (sum(len(s.items) for s in self.tasks_in) if isinstance(self.tasks_in, list)
                  else len(self.tasks_in.items))
        return Snapshot(now=float(self.env.now), machines=tuple(ms), jobs=tuple(js),
                        vehicles=tuple(vs), n_done=len(self.completes),
                        in_flight=queued + sum(self._agv_load_n))

    def run(self, seed_chain: int = 0, op_choices: list[list[int]] | None = None,
            agv_phi: list[int] | None = None) -> dict:
        """op_choices[job][op_idx] = 该工序选第几个候选；缺省=每工序取最短候选（v0 调度器）。

        agv_phi[task_i] = 第 task_i 个运输任务的 AGV id（L 层决策的载体；None=旧 FIFO 规则）。
          task_i 按 transporter 生成序 0,1,2,...；未覆盖的采用轮询 (i % n_agv)（确定性兜底）。
          绑定模式：每台车一个任务队列（bound 路径）—— 与旧"空闲车接活"语义不同（基线数字仅
          None 路径口径，论文对照会注明）。
        返回 metrics：makespan / energy / fail_events / tardy / moves / deliveries。
        """
        rng = np.random.default_rng(seed_chain)
        stats = {"fail_events": 0, "process_time": 0.0, "travel_time": 0.0,
                 "moves": 0, "deliveries": 0,
                 "agv_del": [0] * self.cfg.n_agv, "agv_pos": [-1] * self.cfg.n_agv,
                 # M2 能耗的三态时长（P1b Task 2）：机床按机位计，AGV 按车队合计
                 "proc_min": [0.0] * self.inst.n_machines,
                 "setup_min": [0.0] * self.inst.n_machines,
                 "agv_empty_min": 0.0, "agv_loaded_min": 0.0,
                 # 生产侧约束的事件计数（④⑤⑫）——binding 实测与消融表的读数口径
                 "rework_events": 0, "pm_events": 0,
                 # 物流侧约束的事件计数（⑨⑩⑪）
                 "trips": 0, "charge_events": 0, "agv_fail_events": 0,
                 "battery_min_kwh": float("inf"),
                 "tasks_get": 0, "requeue": 0, "in_q_gets": 0,
                 "trans_evt": 0, "tasks_put": 0,
                 # P4-B：矩阵覆盖不到的端点（充电桩）走几何降级时的**留痕**（段数与分钟数）
                 "unmapped_legs": 0, "unmapped_min": 0.0}
        env = simpy.Environment()
        inst = self.inst
        lu_idx = inst.n_machines             # 端点号约定：机台 0..m-1、装卸站 = m
        # 计划表：job -> [机台号]（按 op_choices 或贪婪最短选择）。⚠️ 与 run_gated 同形状：
        # 选机一律走 `_pick_machine`（离线=查表），时长在那里就地查 alts。所选时长另收一份
        # 供 horizon 用（求和顺序与旧式 `sum(t for ops in plans.values() for _, t in ops)`
        # 逐位相同，见 P2 Task 5 的 A/B 对拍）。
        plans = {}
        chosen_times: list[float] = []
        for j, job_ops in enumerate(inst.jobs):
            plan = op_choices[j] if (op_choices and j < len(op_choices) and op_choices[j]) else [
                int(np.argmin([t for _, t in alts])) for alts in job_ops]
            plans[j] = [job_ops[oi][plan[oi]][0] for oi in range(len(job_ops))]
            chosen_times.extend(job_ops[oi][plan[oi]][1] for oi in range(len(job_ops)))
        completes: dict[int, float] = {}
        bound = agv_phi is not None
        charger_res = [simpy.Resource(env, 1) for _ in self.layout.chargers]  # 一桩同时只服务一车
        (entities, tasks_in, zm,
         events_q) = self._build_entities(env, stats, completes, rng, bound=bound,
                                          seed_chain=seed_chain, charger_res=charger_res)
        for e in entities:                      # ⚠️ 启动顺序不得变（同刻事件次序由注册顺序定）
            env.process(e.run())
        for agv in self.agvs:
            env.process(agv.run())
        # 全量注入（2026-10-02：分批门控已删，见 progress-log §12.5——所有作业一次投放）。
        # ⚠️ P4-B Task 2b 起投放**不是瞬移**：每个作业发一条"装卸站 → 首工序机台"事件，
        # 由 transporter 走正常派车路径生成任务（端点号 = 装卸站）。
        jkeys = list(plans.keys())
        for j in jkeys:
            env.process(self._release(env, events_q, (lu_idx, j, -1, OpLite(0.0), False)))
        env.process(self._transporter(env, events_q, tasks_in, plans, entities, stats, lu_idx,
                                      bound=bound, agv_phi=agv_phi))
        total_work = sum(chosen_times)          # = 所选候选的时长之和（口径与旧式逐位相同）
        horizon = float(total_work * 6 + 500)   # v0 护栏升格：B 层门控（cap=2）下运行可远长于
                                                # 无门控（波形化串行）；3× 护栏曾把门控运行掐
                                                # 表截断（实测 9/10 假死——horizon 不足非死锁）
        env.run(until=horizon)
        stats["horizon_hit"] = len(completes) < inst.n_jobs   # 掐表=未完成（SimPy run 必然推进至
                                                              # until：时间比较恒真，须以完成度判）
        due = self._due(plans)
        makespan = (max(completes.values()) if completes else env.now)
        energy = self._energy_report(stats, makespan)
        return {"makespan": makespan,
                "completes": dict(completes),          # 每作业完工时刻（交期校准 / TWT 需要）
                "energy": energy["total_kwh"],         # M2 引证模型（spec §3.4），单位 kWh
                "energy_breakdown": energy,
                "setup_minutes_total": float(sum(stats["setup_min"])),
                "rework_events": stats["rework_events"],
                "pm_events": stats["pm_events"],
                "trips": stats["trips"], "charge_events": stats["charge_events"],
                "agv_fail_events": stats["agv_fail_events"],
                "battery_min_kwh": (0.0 if stats["battery_min_kwh"] == float("inf")
                                    else stats["battery_min_kwh"]),
                "n_agv": self.cfg.n_agv, "fleet_size": self.layout.n_agv,
                "fail_events": stats["fail_events"],
                "tardy": self._tardy(plans, completes, due)[0],
                "tardy_twt": self._tardy(plans, completes, due)[1],
                "moves": stats["moves"], "deliveries": stats["deliveries"],
                "horizon_hit": stats.get("horizon_hit", False),
                "ops_done": stats.get("ops_done", 0), "jobs_done": len(completes),
                "task_flow": stats.get("task_flow", []),
                "agv_load": stats.get("agv_load", []),
                "dbg": {k: v for k, v in stats.items()
                        if k in ("in_q_gets", "trans_evt", "tasks_put", "tasks_get", "requeue")},
                "travel_time_total": float(stats["travel_time"]),
                "zone_wait": {"n": len(zm.waits), "total": float(sum(zm.waits)),
                              "max": float(max(zm.waits)) if zm.waits else 0.0},
                "n_zones": zm.n,
                # 口径**随结果自报**（P4-A Review Focus #1 同型）：两档的数并排放时靠这两行分辨
                "transport": self.inst.transport,
                "unmapped_legs": stats.get("unmapped_legs", 0),
                "unmapped_min": float(stats.get("unmapped_min", 0.0))}

    def run_gated(self, seed_chain: int = 0, op_choices: list[list[int]] | None = None,
                  policy_l=None, policy_s=None, online_s: bool = False) -> dict:
        """L 层门控式运行（真·事件驱动决策的同步实现）+ **在线 S 层**（P2 Task 5）。

        SimPy 单线程确定性 ⇒ transporter 生成任务时**同步调用** L 层策略，并当场把**当时的**
        活状态快照交给它（无并发 → 无需事件/Gate）。L 与 S 的回调契约对称：
        `policy_l(snap, job, frm, to, oi+1, cand_v) -> 车号`（候选 = 全车队）；`des.py` 只给原始
        材料、**不构造任何特征**（层次纪律：`env/` 不得依赖 `nn/`）。

        `online_s=False`（默认）：S 层读**预计算**的 `plans`（`op_choices` 或贪婪最短）——
        **逐位复现 P2 之前的静态行为**。
        `online_s=True`：每道工序在**前驱完成后、即将入机台时**调
        `policy_s(snap, job, oi, cand) -> 机台号` 决策（首工序无前驱 ⇒ 在 t=0 投放点决策，
        此刻各队列皆空；同刻投放的作业共享同一初始视界）。`plans` 随之退化为**决策日志**
        （`plans[job][oi]` 在决策后写入），故**决策前不得读它**。

        ⚠️ 两个回调都只拿**原始快照**（`env/` 不构造特征、不得依赖 `nn/`）——快照→特征在
        `algo/` 层做。决策留痕同理：`group_rel.roll_chain` 自记自己的 `Decision` 链
        （旧的 `dict["decision_log"]` 回传因零消费者已删，评审 M-2）。
        """
        if policy_l is None and not online_s:
            return self.run(seed_chain=seed_chain, op_choices=op_choices)
        if online_s and policy_s is None:
            raise ValueError("online_s=True 需要 policy_s 回调（S 层决策入口）")
        if online_s and op_choices is not None:
            raise ValueError("online_s=True 时计划由 policy_s 在线产生，op_choices 不生效——"
                             "两者同传会静默忽略计划，故直接报错")
        rng = np.random.default_rng(seed_chain)
        stats = {"fail_events": 0, "process_time": 0.0, "travel_time": 0.0,
                 "moves": 0, "deliveries": 0,
                 "agv_del": [0] * self.cfg.n_agv, "agv_pos": [-1] * self.cfg.n_agv,
                 # M2 能耗的三态时长（P1b Task 2）：机床按机位计，AGV 按车队合计
                 "proc_min": [0.0] * self.inst.n_machines,
                 "setup_min": [0.0] * self.inst.n_machines,
                 "agv_empty_min": 0.0, "agv_loaded_min": 0.0,
                 # 生产侧约束的事件计数（④⑤⑫）——binding 实测与消融表的读数口径
                 "rework_events": 0, "pm_events": 0,
                 # 物流侧约束的事件计数（⑨⑩⑪）
                 "trips": 0, "charge_events": 0, "agv_fail_events": 0,
                 "battery_min_kwh": float("inf"),
                 "tasks_get": 0, "requeue": 0, "in_q_gets": 0,
                 "trans_evt": 0, "tasks_put": 0,
                 # P4-B：矩阵覆盖不到的端点（充电桩）走几何降级时的**留痕**（段数与分钟数）
                 "unmapped_legs": 0, "unmapped_min": 0.0}
        env = simpy.Environment()
        inst = self.inst
        lu_idx = inst.n_machines             # 端点号约定：机台 0..m-1、装卸站 = m（同 run()）
        # 计划表：job -> [机台号]。离线 = 预填（同 run()）；在线 = 空 **决策日志**，
        # 由 `_pick_machine` 决策后写入（`None` = 尚未决策，决策前读它即 bug）。
        plans: dict[int, list] = {}
        for j, job_ops in enumerate(inst.jobs):
            if online_s:
                plans[j] = [None] * len(job_ops)
            else:
                plan = op_choices[j] if (op_choices and j < len(op_choices) and op_choices[j]) else [
                    int(np.argmin([t for _, t in alts])) for alts in job_ops]
                plans[j] = [job_ops[oi][plan[oi]][0] for oi in range(len(job_ops))]
        completes: dict[int, float] = {}
        charger_res = [simpy.Resource(env, 1) for _ in self.layout.chargers]
        (entities, tasks_in, zm,
         events_q) = self._build_entities(env, stats, completes, rng, bound=True,
                                          seed_chain=seed_chain, charger_res=charger_res)
        for e in entities:                      # ⚠️ 启动顺序不得变（同刻事件次序由注册顺序定）
            env.process(e.run())
        for agv in self.agvs:
            env.process(agv.run())
        jkeys = list(plans.keys())
        for j in jkeys:                         # 全量注入（同 run()：投放走装卸站 → 正常派车路径）
            env.process(self._release(env, events_q, (lu_idx, j, -1, OpLite(0.0), False)))
        env.process(self._transporter(env, events_q, tasks_in, plans, entities, stats, lu_idx,
                                      bound=True, policy_l=policy_l,
                                      policy_s=policy_s, online_s=online_s))
        # horizon 的工时上界：**按最短候选**估（在线时 `plans` 是决策日志、读不到时长；
        # 离线且用默认贪婪计划时该式 = 各工序所选时长之和，与旧式逐位相同）。
        # 实际选择可能更长，靠下面的 6× 余量兜底——掐表时 horizon_hit 如实置位。
        total_work = sum(min(t for _m, t in op) for job in inst.jobs for op in job)
        horizon = float(total_work * 6 + 500)   # 同 run()：门控掐表护栏（原 3× 截断 9/10 假死）
        env.run(until=horizon)
        stats["horizon_hit"] = len(completes) < inst.n_jobs   # 同 run()（原：run_gated 漏设旗标）
        due = self._due(plans)
        makespan = (max(completes.values()) if completes else env.now)
        energy = self._energy_report(stats, makespan)
        return {"makespan": makespan,
                "completes": dict(completes),          # 每作业完工时刻（同 run()）
                "energy": energy["total_kwh"],         # M2 引证模型（同 run()）
                "energy_breakdown": energy,
                "setup_minutes_total": float(sum(stats["setup_min"])),
                "rework_events": stats["rework_events"],
                "pm_events": stats["pm_events"],
                "fail_events": stats["fail_events"],
                "tardy": self._tardy(plans, completes, due)[0],
                "tardy_twt": self._tardy(plans, completes, due)[1],
                "moves": stats["moves"], "deliveries": stats["deliveries"],
                "horizon_hit": stats.get("horizon_hit", False),
                "ops_done": stats.get("ops_done", 0), "jobs_done": len(completes),
                "task_flow": stats.get("task_flow", []),
                "travel_time_total": float(stats["travel_time"]),
                # 口径随结果自报（同 run()；run_gated 此前连 zone_wait 都没带，本批不动它）
                "transport": self.inst.transport,
                "unmapped_legs": stats.get("unmapped_legs", 0),
                "unmapped_min": float(stats.get("unmapped_min", 0.0))}

    def _pick_machine(self, job: int, oi: int, policy_s, online_s: bool,
                      plans) -> tuple[int, float]:
        """**派工点**（P2 Task 5，spec §5.3.2）：选机台 + 取该候选上的加工时长。

        两个派工点（投放点的首工序、transporter 的下一工序）**共用此一处**，故"在线/离线"
        只有这一个分支——不存在"改了一处漏了另一处"的形状。

        在线（`online_s=True`）：把**此刻的活状态**快照交给 `policy_s(snap, job, oi, cand)`；
        `plans[job][oi]` 随决策写入（决策前不得读它）。离线（`online_s=False`）：读预计算的
        `plans`——不调策略、不取快照，**逐位复现旧行为**。

        ⚠️ `policy_s` 返回非候选机台 → **显式报错**（静默回退会掩盖策略/候选集不一致，
        让整条链的 logp 与动作错位而无人察觉）。
        """
        alts = self.inst.jobs[job][oi]
        cand = [m for m, _t in alts]
        if online_s:
            snap = self.snapshot()                        # 活状态（只读，见 snapshot 的契约）
            choice = int(policy_s(snap, job, oi, cand))
            if choice not in cand:
                raise ValueError(f"policy_s 选了非候选机台 {choice}；"
                                 f"job={job} oi={oi} 候选={cand}")
            plans[job][oi] = choice                       # 计划表随决策写入（决策前不得读）
        else:
            choice = plans[job][oi]
        t = next(t for m, t in alts if m == choice)
        return choice, float(t)

    @staticmethod
    def _release(env, store, item):
        yield store.put(item)

    def _transporter(self, env, events_q, tasks_in, plans, entities, stats, lu_idx: int,
                     bound: bool = False, agv_phi: list[int] | None = None,
                     policy_l=None, policy_s=None, online_s: bool = False):
        """机台完成事件 → 生成搬运任务；`task_i` = 本函数生成序。

        **三类事件**（端点号：机台 `0..m-1`、装卸站 = `lu_idx` = `m`）：

        - **投放**（`frm_idx == lu_idx`，`oi = -1`）：装卸站 → 首工序机台（**负载**段）；
        - **工序流转**（`frm_idx < m` 且非末工序）：本机台 → 下一工序机台（负载段）；
        - **回站**（`is_last`）：末工序机台 → 装卸站（**负载**段）。

        `task_flow` 的 `oi` 口径 = "这条搬运服务于第几道工序"（投放 = 0、流转 = 目标工序号、
        回站 = 末工序号 + 1）——三条路同一条判据，便于外部按同一套规则复算行程。

        **这里的取事件处即"下一工序的派工点"**（P2 Task 5）：目标机台由 `_pick_machine` 决定
        （在线 = 此刻调 `policy_s` 看活状态；离线 = 查 `plans`）。回站任务的终点是装卸站，
        **不经** `_pick_machine`（它不是候选机台，也不该进 S 层的选择空间）。

        输出缓冲满=阻塞源：out_q.put 在机台侧阻塞；此处 get 保证消费（阻塞语义=M1.1 消磨）。
        L 层（`policy_l` 非空）：`policy_l(snap, job, frm, to, oi+1, cand_v) -> 车号`——`snap` = 此刻
        的活状态，任务身份 = (作业号, 起送端点, 目标端点, 目标工序序号)，候选 = 全车队；此处
        **不构造任何特征**（F1，P2 Task 5；作业号由 P2 Task 7 补入——L 头的任务特征含"该机
        是否需换型"，需要作业号，且用 (frm, to, oi) 反查作业多义）。
        L 层（bound，`policy_l` 为空）：任务按 agv_phi[task_i] 绑定 AGV（task_i = 本函数生成序）；
        超出补轮询。
        """
        task_i = 0

        while True:
            frm_idx, job, oi, op, is_last = yield events_q.get()
            stats["trans_evt"] = stats.get("trans_evt", 0) + 1
            n_ops = len(self.inst.jobs[job])
            if is_last:                                     # 末工序完工 → 回装卸站
                yield entities[frm_idx].out_q.get()
                nxt_m, item = lu_idx, (job, n_ops, op, True)
            else:                                           # 投放（frm=装卸站）或工序流转
                if frm_idx != lu_idx:
                    yield entities[frm_idx].out_q.get()
                nxt_m, nxt_t = self._pick_machine(job, oi + 1, policy_s, online_s, plans)
                item = (job, oi + 1, OpLite(nxt_t), oi + 1 == n_ops - 1)
            from .corridors import shortest_node_path
            _path = shortest_node_path(self.g, entities[frm_idx].pad.dock_node,
                                       entities[nxt_m].pad.dock_node)   # AGV 实际经过的节点序列
            task = (frm_idx, nxt_m, item, _path)
            stats.setdefault("task_flow", []).append((job, item[1], frm_idx, nxt_m))  # L 层流导出
            stats.setdefault("agv_load", []).append((tuple(stats["agv_del"]),
                                                     tuple(stats["agv_pos"])))       # 任务时点车状态
            if policy_l is not None:
                # F1（P2 Task 5）：回调契约与 `policy_s` 对称——快照 + 任务身份 + 候选车号。
                # 此前这里现搓 11 维扁平特征（含魔数 (oi+1)/8、task_i/50，MK10 上溢出到
                # 1.75/4.40），其唯一消费者 `PolicyNet.agv_logits` 已随 Task 4 删除——
                # 即那段是**死代码**，故整体删除：特征一律由调用方从快照自造。
                # T7（P2 Task 7）：任务身份再补**作业号**——L 头的任务特征含"该机是否需换型"
                # （需作业号），而 (frm, to, oi) 反查作业**多义**（实测 MK01 51/112、
                # MK10 538/985 个键有歧义），静默取错作业 = 换型特征错，故走显式参数。
                snap = self.snapshot()                  # 派车决策时点的活状态（只读）
                cand_v = list(range(len(tasks_in)))     # 候选 = 全车队（每车一队列，都可选）
                agv = int(policy_l(snap, job, frm_idx, nxt_m, item[1], cand_v))
                if agv not in cand_v:
                    # 同 policy_s：非候选**显式报错**——负索引会静默回绕到别的车
                    raise ValueError(f"policy_l 派了不存在的车 {agv}；候选={cand_v}")
                yield tasks_in[agv].put(task)
            elif bound:
                n_agv = len(tasks_in)
                agv = agv_phi[task_i] if (agv_phi and task_i < len(agv_phi)) else (task_i % n_agv)
                yield tasks_in[agv].put(task)
            else:
                agv = -1                        # FIFO 路径：此刻还没派车（车在取货时才定）
                yield tasks_in.put(task)
            # 快照跟踪（P2 Task 1）：工件已离机台；绑定路径此刻即知派了哪台车
            self._job_loc[job] = -1
            self._job_agv[job] = agv
            task_i += 1
            stats["tasks_put"] = stats.get("tasks_put", 0) + 1
