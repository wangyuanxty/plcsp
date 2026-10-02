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

from dataclasses import dataclass
import numpy as np
import simpy
from ..energy import (AGV_EMPTY_KW, AGV_IDLE_KW, AGV_LOADED_KW, agv_energy_kwh,
                      machine_energy_kwh, machine_params_for, total_energy_kwh)
from .corridors import shortest_node_path
from .layout import Layout, MachinePad
from .instances import Instance

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


def rollout_evaluate(inst: Instance, plan: list[int] | None, seed: int = 0,
                     layout_type: str = "line", seed_layout: int = 1) -> float:
    """M3 训练环叶子评估：给定机台计划 → 一次 episode → 奖励（负 makespan，M3b 换四元权重）。

    同 (plan, seed, seed_layout) → 同值（确定性回放——树采样重放安全）。
    """
    r = rollout(inst, layout_type=layout_type, seed_layout=seed_layout,
                seed_chain=seed, op_choices=plan)
    return -float(r["makespan"])


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
    # ⑧ 交期系数 d_j = τ·M_ref。**2026-10-02 由 0.85 重标为 0.90**：0.85 是在 P1b 之前
    # 标定的，此后空载段/生产三约束/异构车队都改了 makespan，误期率漂到 26%/44%/65%。
    # 重扫结果：τ=0.90 → 20%/28%/48%，**三个实例都非退化**。
    # ⚠️ 没有任何单一 τ 能让三实例同时落进 20–40%——交期紧度本身就随实例变，这要如实报告。
    tau: float = 0.90

    @property
    def eff_speed(self) -> float:
        """有效车速：通道 <1.5m 线性降速（下限 0.4×）——"宽=快但贵"权衡。
        默认 1.5 → 倍率 1.0（历史实验逐位不变）。"""
        return min(1.0, max(0.4, self.aisle_width / 1.5))


class OpLite:
    __slots__ = ("time",)
    def __init__(self, t: float):
        self.time = t


# ══ 信息侧约束的纯函数（⑧ 交期），放在模块级以便单测直接调用 ══

def compute_due_dates(n_jobs: int, tau: float, m_ref: float) -> dict[int, float]:
    """⑧ 交期 `d_j = τ · M_ref`（spec §3.5）——同一实例所有作业**同值**（共同交期形）。"""
    return dict.fromkeys(range(n_jobs), tau * m_ref)


def weighted_tardiness(completes: dict[int, float], due: dict[int, float],
                       weights: dict[int, float]) -> float:
    """⑧ 加权总拖期 `TWT = Σ w_j · max(0, C_j − d_j)`（spec §4.1 的目标口径）。"""
    return float(sum(weights.get(j, 1.0) * max(0.0, c - due[j])
                     for j, c in completes.items() if j in due))


_MREF_CACHE: dict = {}
_MREF_BUSY = False


def _instance_key(inst: Instance) -> tuple:
    """实例内容指纹——用于缓存 `M_ref`（比 id() 稳，比文件名稳）。"""
    return (inst.n_machines, inst.n_jobs,
            tuple(tuple(min(t for _, t in op) for op in job) for job in inst.jobs))


def reference_makespan(inst: Instance, cfg: SimConfig | None = None,
                       seed_layout: int = 0) -> float:
    """⑧ 的 `M_ref` = **参考调度**（每工序取最短候选 + AGV 轮询派车）的 makespan。

    ⚠️ **绝不能取被优化的那次 episode 的 makespan**——否则交期随策略一起漂移，
    目标退化（旧 `due_factor` 就是这么坏的：实测 MK01 tardy 恒为 10/10）。
    故单独跑一次并缓存；重入时 `_due` 会退化为"无交期"（见 `SimWorld._due`）。
    """
    key = _instance_key(inst) + (seed_layout,)
    if key in _MREF_CACHE:
        return _MREF_CACHE[key]
    global _MREF_BUSY
    if _MREF_BUSY:                   # 理论上被 `_due` 的短路挡住，此处兜底
        raise RuntimeError("reference_makespan 重入")
    _MREF_BUSY = True
    try:
        r = rollout(inst, seed_layout=seed_layout, seed_chain=0,
                    cfg=cfg or SimConfig(), constraints=None)
    finally:
        _MREF_BUSY = False
    val = float(r["makespan"])
    _MREF_CACHE[key] = val
    return val


class MachineSim:
    """机台：输入缓冲 → 换型 → 加工（故障中断-恢复）→ 保养 → 输出缓冲（满则阻塞）。

    **开关一律从 `ConstraintConfig` 读**（逐个传 bool 会在约束变多时漏参数）。
    开关关闭时**不得消耗随机数**——否则故障流被移位，"该约束从未存在"的语义就不成立。
    """

    def __init__(self, env, pad: MachinePad, rng, cfg: SimConfig, stats: dict, completes: dict,
                 events_q: simpy.Store, constraints):
        self.env, self.pad, self.rng, self.cfg, self.stats = env, pad, rng, cfg, stats
        self.completes = completes
        self.events_q = events_q
        self.con = constraints
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
            # ④ 返工：同件在本机**原地**重做（工件不离开机台，故不走运输）。
            # ⚠️ 必须原地——早期版本走 `in_q.put` 会**自锁**：本机是该缓冲的唯一消费者，
            # 缓冲满时 put 永久阻塞，而此时还占着加工槽 ⇒ 该机位连同工件一起卡死
            # （实测 MK07/MK10 5/5 掐表，关掉返工即 0/5）。
            while True:
                yield from self._process(job, op)
                if self.con.rework and self.rng.random() < self.cfg.p_rework:
                    self.stats["rework_events"] += 1
                    continue
                break
            self.stats["ops_done"] = self.stats.get("ops_done", 0) + 1
            if is_last:
                self.completes[job] = self.env.now      # 末工序完成即出库（不进输出缓冲/无搬运）
                self.events_q.put((self.pad.id, job, oi, op, True))  # 批注入载体：transporter 收
                continue                                # is_last → 从 inject_q 补下一作业（否则
                                                        # 分批门控下只有首批作业会运行！）
            self.events_q.put((self.pad.id, job, oi, op, is_last))  # 先发事件：transporter 的 out_q.get 作为
            yield self.out_q.put((job, oi, op, is_last))            # 等待者立即放行 put（防 put→事件 顺序死锁）


def build_zone_map(layout, granularity: str) -> tuple[dict[int, int], int]:
    """按粒度构造「节点 → 区段」映射。**粒度越粗 → 区段越少 → 争用越强。**

    - `node`：每个通道节点一个区段（最细；MK10 的 5×4 网格 → 30 个）
    - `row` / `col`：整行 / 整列算一个区段（像"一条长廊一个区段"）
    - `all`：全图一个区段（极端档，用于下界）
    """
    spec = layout.grid
    n = spec.n_nodes
    if granularity == "node":
        return {i: i for i in range(n)}, n
    if granularity == "row":
        return {i: spec.node_rc(i)[0] for i in range(n)}, spec.n_rows + 1
    if granularity == "col":
        return {i: spec.node_rc(i)[1] for i in range(n)}, spec.n_cols + 1
    if granularity == "all":
        return dict.fromkeys(range(n), 0), 1
    raise ValueError(f"未知区段粒度：{granularity}")


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

    def wait_zone(self, agv: int, z: int, limit: float | None = None):
        if self.try_grant(agv, z):
            return True, False
        if self.would_cycle(agv, z):
            return False, True
        self.pending[agv] = z
        t0 = self.env.now
        ev = self._ev.setdefault(z, simpy.Event(self.env))
        yield ev | self.env.timeout(limit or self.wait_limit)
        self.waits.append(self.env.now - t0)
        self.pending.pop(agv, None)
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


class AgvSim:
    """AGV：待命 → 空载驶向取货机台 → 装载驶向卸货机台 → 投递。

    **两个行驶段**（P1b Task 2 补齐）：
    - **空载段**：从当前停位开到取货点。v0 **缺失此段**（AGV 从上一个卸货点"瞬移"到取货点），
      等于凭空多出运力、且使 AGV 只有"负载"一个状态。补齐后运输负荷才真实。
    - **负载段**：取货点 → 卸货点，沿用格点最短路。

    区段管制（① congestion）**可开关**：开时逐段申请/释放区段（持当前 → 申请下一 → 放上一），
    等待环或超时则回队重试；关时不申请，按距离一次到底。
    """

    def __init__(self, env, aid, m_dm: np.ndarray, cfg: SimConfig, stats: dict,
                 tasks_in, machines: list, graph, zm, constraints, spec, rng,
                 chargers=(), charger_res=(), bound: bool = False):
        self.env, self.aid, self.m_dm, self.cfg = env, aid, m_dm, cfg
        self.stats, self.tasks_in, self.machines = stats, tasks_in, machines
        self.g, self.zm = graph, zm
        self.con = constraints              # ① congestion 等物流侧开关从这里读
        self.congestion = constraints.congestion    # ① 关 → 无区段管制
        self.bound = bound          # True: 任务按 agv_phi 绑定（每车一个 Store = L 层决策载体）
        self.pos_node: int | None = None    # 当前所在通道节点；None = 尚未出车（停在首个取货点）
        # ⑩ 异构车队：**开关关掉时倍率=1、载量=1**，即与"约束从未存在"逐位相同
        hetero = constraints.heterogeneous_fleet
        self.capacity = spec.capacity if hetero else 1
        self.speed = cfg.agv_speed_mps * (spec.speed_factor if hetero else 1.0)
        self.battery_cap = spec.battery_kwh
        self.battery = spec.battery_kwh     # [kWh]，只在 ⑪ 开启时增减
        self.chargers, self.charger_res = list(chargers), list(charger_res)
        # ⑨ 故障：**每台车一条独立随机流**——否则关掉 AGV 故障会移位机台故障流（消融不干净）
        self.rng = rng
        self.down = False
        self.up = simpy.Event(env)
        self.up.succeed()                   # 初始可用

    def _seg_min(self, u: int, v: int) -> float:
        """节点 u→v 的行驶时长 [min]。距离矩阵给的是最短路，故 u,v 不必相邻。"""
        return float(self.m_dm[u, v]) / (self.cfg.eff_speed * self.speed) / SECONDS_PER_MIN

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
        for ch in sorted(self.chargers, key=lambda c: float(self.m_dm[src, c.node])):
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
            seg = self._seg_min(src, dst)
            yield self.env.timeout(seg)
            self._drain(leg, seg)               # ⑪ 行驶耗电
            self.stats["travel_time"] += seg
            self.stats[f"agv_{leg}_min"] += seg
            self.stats["moves"] += 1
            return True, dst
        # 逐段申请区段：持当前 → 申请下一 → 成功才放上一 → 走这一段
        path = shortest_node_path(self.g, src, dst)
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
                    continue
            # ⑩ 同向拼车：把队首连续的同 (取货点, 卸货点) 任务一并装走
            batch = yield from self._collect(q, frm, to, (frm, to, item, path))
            # ── 负载段：取货点 → 卸货点（整批一趟）──
            ok, self.pos_node = yield from self._drive(a, b, "loaded")
            if not ok:
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
                self.stats["deliveries"] += 1

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

    def _due(self, plans: dict[int, list[tuple[int, float]]]) -> dict[int, float]:
        """⑧ 交期 `d_j = τ·M_ref`。**开关关闭 → 返回空 dict**（该实例无交期，目标无拖期项）。

        旧口径 `due_factor × Σ工时` **已废弃**（spec §3.5：完全没算排队/运输/争用，
        实测 tardy 恒为 100%）。旧字段 `due_factor` / `energy_power` 已随 P1b 清理删除。
        """
        if not self.constraints.due_dates or _MREF_BUSY:
            return {}                    # 参考调度自身运行时不递归求 M_ref
        m_ref = reference_makespan(self.inst, self.cfg, self.layout.layout_seed)
        return compute_due_dates(len(plans), self.cfg.tau, m_ref)

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
                 "trans_evt": 0, "tasks_put": 0}
        env = simpy.Environment()
        inst = self.inst
        # 计划表：job -> [(mach, time)]（按 op_choices 或贪婪最短选择）
        plans = {}
        for j, job_ops in enumerate(inst.jobs):
            plan = op_choices[j] if (op_choices and j < len(op_choices) and op_choices[j]) else [
                int(np.argmin([t for _, t in alts])) for alts in job_ops]
            plans[j] = [(job_ops[oi][plan[oi]][0], job_ops[oi][plan[oi]][1])
                        for oi in range(len(job_ops))]
        completes: dict[int, float] = {}
        zof, nz = build_zone_map(self.layout, self.cfg.zone_granularity)
        zm = ZoneManager(env, zof, nz, self.cfg.zone_wait_limit)
        events_q = simpy.Store(env)
        machines = [MachineSim(env, self.layout.machines[i], rng, self.cfg, stats, completes,
                               events_q, self.constraints)
                    for i in range(inst.n_machines)]
        bound = agv_phi is not None
        if bound:
            tasks_in = [simpy.Store(env) for _ in range(self.cfg.n_agv)]   # L 层绑定：每车一队列
        else:
            tasks_in = simpy.Store(env)                                    # 旧 FIFO 规则路径
        for m in machines:
            env.process(m.run())
        charger_res = [simpy.Resource(env, 1) for _ in self.layout.chargers]  # 一桩同时只服务一车
        fleet = self.layout.agvs or []
        if len(fleet) != self.cfg.n_agv:      # 静默错误护栏：布局车队 ≠ 仿真车队
            raise ValueError(
                f"布局车队 {len(fleet)} 台 ≠ SimConfig.n_agv={self.cfg.n_agv}——"
                "直接构造 SimWorld 时须传 sample_layout(..., n_agv=cfg.n_agv)")
        for a in range(self.cfg.n_agv):
            env.process(AgvSim(env, a, self.m_dm, self.cfg, stats, tasks_in, machines,
                               self.g, zm, self.constraints, fleet[a],
                               np.random.default_rng([seed_chain, 1000 + a]),   # ⑨ 每车独立流
                               chargers=self.layout.chargers, charger_res=charger_res,
                               bound=bound).run())
        # 全量注入（2026-10-02：分批门控已删，见 progress-log §12.5——所有作业一次投放）
        jkeys = list(plans.keys())
        inject_q: list = []
        for j in jkeys:
            m0, t0 = plans[j][0]
            env.process(self._release(env, machines[m0].in_q, (j, 0, OpLite(t0), len(plans[j]) == 1)))
        env.process(self._transporter(env, events_q, tasks_in, plans, machines, stats, inject_q,
                                      bound=bound, agv_phi=agv_phi))
        total_work = sum(t for ops in plans.values() for _, t in ops)
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
                "n_zones": zm.n}

    def run_gated(self, seed_chain: int = 0, op_choices: list[list[int]] | None = None,
                  policy_l=None) -> dict:
        """L 层门控式运行（真·事件驱动决策的同步实现）。

        SimPy 单线程确定性 ⇒ transporter 生成任务时**同步调用** policy_l(feat)，
        此刻 stats 中的车状态（agv_del/agv_pos）即**实时值**（无并发 → 无需事件/Gate）。
        决策日志回传：返回 dict["decision_log"] = [(info, feat, agv), ...]（训练/评估用）。
        policy_l=None → 等价 run 的轮询 bound 行为（agv_phi=None 时轮询兜底）。
        """
        if policy_l is None:
            return self.run(seed_chain=seed_chain, op_choices=op_choices)
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
                 "trans_evt": 0, "tasks_put": 0}
        env = simpy.Environment()
        inst = self.inst
        plans = {}
        for j, job_ops in enumerate(inst.jobs):
            plan = op_choices[j] if (op_choices and j < len(op_choices) and op_choices[j]) else [
                int(np.argmin([t for _, t in alts])) for alts in job_ops]
            plans[j] = [(job_ops[oi][plan[oi]][0], job_ops[oi][plan[oi]][1])
                        for oi in range(len(job_ops))]
        completes: dict[int, float] = {}
        zof, nz = build_zone_map(self.layout, self.cfg.zone_granularity)
        zm = ZoneManager(env, zof, nz, self.cfg.zone_wait_limit)
        events_q = simpy.Store(env)
        machines = [MachineSim(env, self.layout.machines[i], rng, self.cfg, stats, completes,
                               events_q, self.constraints)
                    for i in range(inst.n_machines)]
        tasks_in = [simpy.Store(env) for _ in range(self.cfg.n_agv)]
        for m in machines:
            env.process(m.run())
        charger_res = [simpy.Resource(env, 1) for _ in self.layout.chargers]
        fleet = self.layout.agvs or []
        if len(fleet) != self.cfg.n_agv:
            raise ValueError(
                f"布局车队 {len(fleet)} 台 ≠ SimConfig.n_agv={self.cfg.n_agv}——"
                "直接构造 SimWorld 时须传 sample_layout(..., n_agv=cfg.n_agv)")
        for a in range(self.cfg.n_agv):
            env.process(AgvSim(env, a, self.m_dm, self.cfg, stats, tasks_in, machines,
                               self.g, zm, self.constraints, fleet[a],
                               np.random.default_rng([seed_chain, 1000 + a]),
                               chargers=self.layout.chargers, charger_res=charger_res,
                               bound=True).run())
        jkeys = list(plans.keys())
        inject_q: list = []                     # 全量注入（分批门控已删，同 run()）
        for j in jkeys:
            m0, t0 = plans[j][0]
            env.process(self._release(env, machines[m0].in_q, (j, 0, OpLite(t0), len(plans[j]) == 1)))
        dec_log: list[tuple] = []
        env.process(self._transporter(env, events_q, tasks_in, plans, machines, stats, inject_q,
                                      bound=True, policy_l=policy_l, dec_log=dec_log))
        total_work = sum(t for ops in plans.values() for _, t in ops)
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
                "decision_log": dec_log}

    @staticmethod
    def _release(env, store, item):
        yield store.put(item)

    def _transporter(self, env, events_q, tasks_in, plans, machines, stats, inject_q,
                     bound: bool = False, agv_phi: list[int] | None = None,
                     policy_l=None, dec_log: list | None = None):
        """机台完成事件：非末工序 → 从其输出缓冲取出 → 生成下一工序搬运任务（目标=下一工序机台）。

        输出缓冲满=阻塞源：out_q.put 在机台侧阻塞；此处 get 保证消费（阻塞语义=M1.1 消磨）。
        末工序 + 在途注入队列非空 → 注入下一作业（分批门控波次语义）。
        L 层（bound）：任务按 agv_phi[task_i] 绑定 AGV（task_i = 本函数生成序）；超出补轮询。
        """
        task_i = 0

        while True:
            frm_idx, job, oi, op, is_last = yield events_q.get()
            stats["trans_evt"] = stats.get("trans_evt", 0) + 1
            if is_last:
                if inject_q:
                    j2 = inject_q.pop(0)
                    m0, t0 = plans[j2][0]
                    # 注入用独立进程（非 yield 阻塞）：transporter 若在 in_q 满机上阻塞 put，
                    # 事件队列头被卡死 → 输出缓冲无人取 → 机器永不释放空间 → 永久死锁
                    # （jobs=9/10 quiescence 实测）。env.process 与首批注入同模式（_release）。
                    env.process(self._release(env, machines[m0].in_q,
                                              (j2, 0, OpLite(t0), len(plans[j2]) == 1)))
                continue
            yield machines[frm_idx].out_q.get()
            ops = plans[job]
            nxt_m, nxt_t = ops[oi + 1]
            from .corridors import shortest_node_path
            _path = shortest_node_path(self.g, machines[frm_idx].pad.dock_node,
                                       machines[nxt_m].pad.dock_node)   # AGV 实际经过的节点序列
            task = (frm_idx, nxt_m,
                    (job, oi + 1, OpLite(nxt_t), oi + 1 == len(ops) - 1), _path)
            stats.setdefault("task_flow", []).append((job, oi + 1, frm_idx, nxt_m))  # L 层流导出
            stats.setdefault("agv_load", []).append((tuple(stats["agv_del"]),
                                                     tuple(stats["agv_pos"])))       # 任务时点车状态
            if policy_l is not None:
                n_m = len(machines)
                d0, p0 = stats["agv_del"], stats["agv_pos"]
                load_d = float(d0[0] - d0[1]) / 10.0 if len(d0) > 1 else 0.0    # bug#14：n_agv=1
                pos0 = float(p0[0]) / n_m                                     # 时硬编码 [1] 越界
                pos1 = float(p0[1]) / n_m if len(p0) > 1 else float(p0[0]) / n_m
                feat = (np.array([frm_idx / n_m, nxt_m / n_m, (oi + 1) / 8.0,
                                  load_d, pos0, pos1], dtype=np.float32)
                        .tolist()
                        + [len(tasks_in[a].items) / 5.0
                           for a in range(len(tasks_in))]                       # 在途负载（通用 n_agv）
                        + [float(task_i) / 50.0, float(task_i % len(tasks_in)) / len(tasks_in)])
                                                                                # 任务序号+相位（表
                                                                                # 示缺口：轮换须可表达）
                feat = np.array(feat, dtype=np.float32)
                agv = int(policy_l(feat))
                dec_log.append(((job, oi + 1, frm_idx, nxt_m), feat, agv))
                yield tasks_in[agv].put(task)
            elif bound:
                n_agv = len(tasks_in)
                agv = agv_phi[task_i] if (agv_phi and task_i < len(agv_phi)) else (task_i % n_agv)
                yield tasks_in[agv].put(task)
            else:
                yield tasks_in.put(task)
            task_i += 1
            stats["tasks_put"] = stats.get("tasks_put", 0) + 1
