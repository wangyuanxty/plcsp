# -*- coding: utf-8 -*-
"""SimPy DES 核心（《方法设计文档》§4.2：SimPy 单一后端，无物理库）。

v0 边界（诚实声明）：
- **无区段管制**（2026-10-02 移除，见 progress-log §十五）：实测通道争用 ≤0.07%，物流瓶颈在车辆数
  而非通道容量。AGV 一次行驶到底。
- 每叶链独立扰动流：每 episode 一个 rng（seed_chain），对应理论骨架 A1。
- 机器故障=泊松流：故障期间占用机台（中断-恢复），修复时长 repair_time。
- 完成时刻以"末工序在机台加工完毕"为准（搬运回库不计入 makespan 口径——v0 简化，M1.1 对齐 BKS 口径）。
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import simpy
from .layout import Layout, MachinePad
from .instances import Instance

# ⚠️ 单位约定（2026-10-02 量纲对齐）：**仿真时间单位 = 分钟；布局坐标单位 = 米**。
# 此前无单位换算，运输时间 = 距离（米）÷ 速度，导致 TDT/TPT ≈ 4–6 倍（物理上不成立，
# 见 progress-log 与 spec §3.4）。此常量把 [m] / [m/s] 转成分钟。
SECONDS_PER_MIN = 60.0


def rollout(inst: Instance, layout_type: str = "line", seed_layout: int = 0, seed_chain: int = 0,
            cfg: SimConfig | None = None, op_choices: list[list[int]] | None = None,
            machine_gap: float = 1.0, aisle_width: float = 1.5,
            agv_phi: list[int] | None = None) -> dict:
    """一次完整 episode：网格布局采样(seed_layout) → 格点距离 → SimPy(seed_chain)。

    `layout_type` **保留但忽略**（旧调用方仍传）——计划 3 统一清理。
    """
    from .corridors import build_corridor_graph, dock_distance_matrix
    from .layout import sample_layout
    layout = sample_layout(inst.n_machines, seed=seed_layout, aisle_w=aisle_width)
    g = build_corridor_graph(layout)
    dm = dock_distance_matrix(g)
    # ⚠️ aisle_width 必须同时进 SimConfig——eff_speed 读的是 cfg.aisle_width（窄道降速）。
    # 否则 `rollout(aisle_width=1.0)` 只改几何、不改速度，窄道敏感性实验**静默失效**。
    eff_cfg = cfg if cfg is not None else SimConfig(aisle_width=aisle_width)
    return SimWorld(inst, layout, dm, eff_cfg, graph=g).run(
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
    agv_speed_mps: float = 1.0    # AGV 车速 [m/s]（布局坐标为米）
    zone_hold: float = 1.0        # 缓冲满退避时长 [min]
    zone_granularity: str = "node"   # 区段粒度：node|row|col|all（越粗→区段越少→争用越强）
    zone_wait_limit: float = 8.0     # 区段申请等待上限 [min]（超时→退避重试）
    zone_hold_limit: float = 8.0  # 区段申请等待上限（超时→释放重试）[min]
    repair_time: float = 5.0      # [min]
    due_factor: float = 1.8       # ⚠️ 已废弃（见 §3.5，改用 τ·M_ref）
    energy_power: float = 1.0     # ⚠️ 已废弃（见 §3.4，改用 M2 能耗模型）
    aisle_width: float = 1.5      # 通道宽（米）：窄通道限速 = Phase A2 真权衡来源

    @property
    def eff_speed(self) -> float:
        """有效车速：通道 <1.5m 线性降速（下限 0.4×）——"宽=快但贵"权衡。
        默认 1.5 → 倍率 1.0（历史实验逐位不变）。"""
        return min(1.0, max(0.4, self.aisle_width / 1.5))


class OpLite:
    __slots__ = ("time",)
    def __init__(self, t: float):
        self.time = t


def _try_acquire(env, res, timeout):
    req = res.request()
    events = yield req | env.timeout(timeout)
    if req in events:
        return True, req
    req.cancel()          # 关键：撤销孤儿请求，否则资源"授予"后容量永久泄漏（v0 第一大坑）
    return False, req


class MachineSim:
    """机台：输入缓冲 → 加工（故障中断-恢复）→ 输出缓冲（满则阻塞）。缓冲满=阻塞源。"""

    def __init__(self, env, pad: MachinePad, rng, cfg: SimConfig, stats: dict, completes: dict,
                 events_q: simpy.Store):
        self.env, self.pad, self.rng, self.cfg, self.stats = env, pad, rng, cfg, stats
        self.completes = completes
        self.events_q = events_q
        self.in_q = simpy.Store(env, capacity=pad.in_cap)
        self.out_q = simpy.Store(env, capacity=pad.out_cap)
        self.slot = simpy.Resource(env, 1)   # 机台加工槽（故障期间占用）

    def run(self):
        while True:
            job, oi, op, is_last = yield self.in_q.get()
            self.stats["in_q_gets"] = self.stats.get("in_q_gets", 0) + 1
            with self.slot.request() as req:
                yield req
                t = self.env.now
                finish = t + op.time
                while t < finish:
                    nxt_fail = t + self.rng.exponential(1.0 / max(self.pad.fail_rate, 1e-9))
                    seg = min(nxt_fail, finish) - t
                    yield self.env.timeout(seg)
                    if nxt_fail < finish:
                        self.stats["fail_events"] += 1
                        yield self.env.timeout(self.cfg.repair_time)   # 中断-恢复
                    t += seg
                self.stats["process_time"] += op.time
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
        return {i: spec.rc if False else spec.node_rc(i)[1] for i in range(n)}, spec.n_cols + 1
    if granularity == "all":
        return {i: 0 for i in range(n)}, 1
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
        self.holder: dict[int, int | None] = {z: None for z in range(n_zones)}
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
    """AGV：取任务 → 按格点距离行驶 → 投递。

    ⚠️ 2026-10-02：**区段管制（ZoneManager）已整条移除**。实测表明：在网格布局 + 现实车队规模下，
    通道争用占比 ≤0.07%（对照：旧 line 单环上 MK01 n_agv=4 是 32%）。结论——
    **物流的瓶颈是车辆数量（任务排队），不是通道容量（路口争用）**。
    故 AGV 不再申请区段，一次行驶到底。见 progress-log §十五 与 spec §3.3。
    """

    def __init__(self, env, aid, m_dm: np.ndarray, cfg: SimConfig, stats: dict,
                 tasks_in, machines: list, graph, zm, bound: bool = False):
        self.env, self.aid, self.m_dm, self.cfg = env, aid, m_dm, cfg
        self.stats, self.tasks_in, self.machines = stats, tasks_in, machines
        self.g, self.zm = graph, zm
        self.bound = bound          # True: 任务按 agv_phi 绑定（每车一个 Store = L 层决策载体）

    def run(self):
        q = self.tasks_in[self.aid] if self.bound else self.tasks_in
        while True:
            frm, to, item, path = yield q.get()
            self.stats["tasks_get"] = self.stats.get("tasks_get", 0) + 1
            # 逐段申请区段：持当前 → 申请下一 → 成功才放上一 → 走这一段
            zseq = [self.zm.zone_of[p] for p in path]
            zseq = [z for i, z in enumerate(zseq) if i == 0 or z != zseq[i - 1]]   # 合并同一区段的连续段
            granted, cycle = yield from self.zm.wait_zone(self.aid, zseq[0],
                                                          self.cfg.zone_wait_limit)
            if cycle or not granted:
                self.stats["requeue"] = self.stats.get("requeue", 0) + 1
                q.put((frm, to, item, path))
                continue
            travel = 0.0
            prev_z = zseq[0]
            ok = True
            for pi in range(len(path) - 1):
                z = self.zm.zone_of[path[pi + 1]]
                if z != prev_z:
                    granted, cycle = yield from self.zm.wait_zone(self.aid, z,
                                                                  self.cfg.zone_wait_limit)
                    if cycle or not granted:
                        ok = False
                        break
                    self.zm.release(self.aid, prev_z)
                    prev_z = z
                seg = (float(self.m_dm[path[pi], path[pi + 1]])
                       / (self.cfg.eff_speed * self.cfg.agv_speed_mps) / SECONDS_PER_MIN)
                yield self.env.timeout(seg)
                travel += seg
                self.stats["moves"] += 1
            self.zm.release(self.aid, prev_z)
            if not ok:
                yield self.env.timeout(self.cfg.zone_hold)
                self.stats["requeue"] = self.stats.get("requeue", 0) + 1
                q.put((frm, to, item, path))
                continue
            self.stats["travel_time"] += travel
            self.stats["agv_del"][self.aid] += 1      # L 层状态轨迹（负载差/最后位置）
            self.stats["agv_pos"][self.aid] = to
            # 有界输入缓冲投递：满则让步超时重试（防缓冲满阻塞拖累运输环）。
            while len(self.machines[to].in_q.items) >= self.machines[to].in_q.capacity:
                yield self.env.timeout(self.cfg.zone_hold)
            yield self.machines[to].in_q.put(item)
            self.stats["deliveries"] += 1

class SimWorld:
    """一次 episode：1 布局 + 1 实例 + 1 条独立扰动流 → 指标 dict。"""

    def __init__(self, inst: Instance, layout: Layout, m_dm: np.ndarray,
                 cfg: SimConfig | None = None, graph=None):
        self.inst, self.layout, self.m_dm = inst, layout, m_dm
        self.cfg = cfg or SimConfig()
        self.g = graph      # 格点走廊图（AGV 逐段路径的来源）

    def _due(self, plans: dict[int, list[tuple[int, float]]]) -> dict[int, float]:
        return {j: self.cfg.due_factor * sum(t for _, t in ops) for j, ops in plans.items()}

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
                 "agv_del": [0] * self.cfg.n_agv, "agv_pos": [-1] * self.cfg.n_agv}
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
        machines = [MachineSim(env, self.layout.machines[i], rng, self.cfg, stats, completes, events_q)
                    for i in range(inst.n_machines)]
        bound = agv_phi is not None
        if bound:
            tasks_in = [simpy.Store(env) for _ in range(self.cfg.n_agv)]   # L 层绑定：每车一队列
        else:
            tasks_in = simpy.Store(env)                                    # 旧 FIFO 规则路径
        for m in machines:
            env.process(m.run())
        for a in range(self.cfg.n_agv):
            env.process(AgvSim(env, a, self.m_dm, self.cfg, stats, tasks_in,
                               machines, self.g, zm, bound=bound).run())
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
        return {"makespan": makespan,
                "completes": dict(completes),          # 每作业完工时刻（交期校准 / TWT 需要）
                "energy": (stats["process_time"] + stats["travel_time"]) * self.cfg.energy_power,
                "fail_events": stats["fail_events"], "tardy": sum(1 for j in plans if j in completes and completes[j] > due[j]),
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
                 "agv_del": [0] * self.cfg.n_agv, "agv_pos": [-1] * self.cfg.n_agv}
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
        machines = [MachineSim(env, self.layout.machines[i], rng, self.cfg, stats, completes, events_q)
                    for i in range(inst.n_machines)]
        tasks_in = [simpy.Store(env) for _ in range(self.cfg.n_agv)]
        for m in machines:
            env.process(m.run())
        for a in range(self.cfg.n_agv):
            env.process(AgvSim(env, a, self.m_dm, self.cfg, stats, tasks_in, machines,
                               self.g, zm, bound=True).run())
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
        return {"makespan": makespan,
                "completes": dict(completes),          # 每作业完工时刻（同 run()）
                "energy": (stats["process_time"] + stats["travel_time"]) * self.cfg.energy_power,
                "fail_events": stats["fail_events"],
                "tardy": sum(1 for j in plans if j in completes and completes[j] > due[j]),
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
