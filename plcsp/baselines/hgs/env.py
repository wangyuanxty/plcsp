# -*- coding: utf-8 -*-
"""FJSPT 环境（HGS 原文 §IV-B 的 MDP）——**独立仿真器**。

⚠️ **这不是本仓 `des.py`**。选择独立仿真的理由（必须如实写进论文）：

- HGS 的动作 = 复合动作 `(O_ij, M_k, V_u)`：**先选作业的哪道工序**，再选机台与车辆。
- 本仓 DES 没有"选哪道工序"这个入口：作业在 t=0 全量投放，工序顺序由事件驱动规则涌现。
  要接上就得给 `des.py` 加优先级入口 = 改既有机制的语义。本批**不动**。
- 原文环境也比本仓 DES 简单：无拥堵、无故障、无换型、无交期、无充电；车辆单载。
  ⟹ 它对应本仓的**报告档 A（机制全关）**，不是报告档 B。

**忠实度声明**（原文没有代码，以下三处原文未写死，本实现取显式解释）：

1. **下一决策步**（原文 §IV-B3 只说"操作事件的最早释放时刻"）。本实现取：**同刻只要还有
   可行动作就继续动作**；没有可行动作时，时间前进到下一个事件（机台完工 / 车辆空出）。
   若改成"一个时刻只允许一个动作"，车队会在 t=0 被闲置——那不是调度器的合理语义。
2. **车辆到达后机台才开工**（原文把机台在指派时即视为被占用）。本实现取
   `start = 车辆载货到达时刻`，机台从该刻起被占用。
3. 原文的节点原始特征（§IV-B1）**逐字使用**：工序 7 维 / 机台 4 维 / 车辆 4 维。
   车辆位置是机台号（标量），本实现归一化为 `loc / m`（原文未说归一化方式）。

行程时间口径：**MKT 矩阵**（`inst.trans_time_full`，含装卸站 LU = 下标 0；
机台 `k` ↔ 下标 `k+1`）。空载段 = 车当前位置 → 作业当前位置；满载段 = 作业当前位置 → 目标机台。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...env.instances import Instance

_EPS = 1e-9


@dataclass
class GraphState:
    """一步决策所需的状态张量（模型侧只读）。`N` = 总工序数，`m` = 机台数，`v` = 车辆数。"""

    op_feat: np.ndarray      # (N, 7)
    mach_feat: np.ndarray    # (m, 4)
    veh_feat: np.ndarray     # (v, 4)
    om_edge: np.ndarray      # (N, m) 加工时间；不可加工 = 0
    ov_edge: np.ndarray      # (N, v) 空载时间（按车当前位置）
    mm_edge: np.ndarray      # (m, m) 满载时间
    compat: np.ndarray       # (N, m) bool：该工序能否上该机台
    eligible: np.ndarray     # (N,) bool：**动作可行**（前驱完工 ∧ 有空闲兼容机台 ∧ 有空闲车）
    idle_mach: np.ndarray    # (m,) bool
    idle_veh: np.ndarray     # (v,) bool
    n_ops: int
    n_machines: int
    n_agv: int


class FjsptEnv:
    """FJSPT 调度环境。一回合 = 把全部工序排完；目标 = makespan。"""

    def __init__(self, inst: Instance, n_agv: int, trans: np.ndarray | None = None):
        self.jobs = [list(job) for job in inst.jobs]
        self.n_jobs = inst.n_jobs
        self.n_machines = inst.n_machines
        self.n_agv = int(n_agv)
        if trans is None:
            trans = inst.trans_time_full
        if trans is None:
            raise ValueError("HGS 环境要求行程时间矩阵（inst.trans_time_full 为 None）——"
                             "几何口径实例请先挂上矩阵")
        self.trans = np.asarray(trans, dtype=float)
        if self.trans.shape != (self.n_machines + 1, self.n_machines + 1):
            raise ValueError(f"行程时间矩阵必须是 (m+1)×(m+1)（0 = 装卸站）："
                             f"实得 {self.trans.shape}，m={self.n_machines}")
        self._avg_p = [[float(np.mean([t for _, t in op])) for op in job] for job in self.jobs]
        self._proc = [[{m: t for m, t in op} for op in job] for job in self.jobs]
        self.reset()

    # ── 生命周期 ──

    def reset(self) -> None:
        self.now = 0.0
        self.op_idx = [0] * self.n_jobs          # 下一道待排工序（== len 时该作业完工）
        self.op_ready = [0.0] * self.n_jobs      # 作业可被取走的时刻（前驱完工）
        self.job_loc = [0] * self.n_jobs         # 矩阵下标：0 = 装卸站
        self.job_end = [0.0] * self.n_jobs       # 末道已排工序的完工时刻
        self.done = [[False] * len(job) for job in self.jobs]
        self.op_machine = [[-1] * len(job) for job in self.jobs]
        self.start = [[0.0] * len(job) for job in self.jobs]
        self.complete = [[0.0] * len(job) for job in self.jobs]
        self.mach_free = [0.0] * self.n_machines
        self.veh_free = [0.0] * self.n_agv
        self.veh_loc = [0] * self.n_agv          # 车辆从装卸站出发
        self.n_done = 0
        self.total_ops = sum(len(job) for job in self.jobs)
        self._job_off = np.cumsum([0] + [len(job) for job in self.jobs])[:-1]

    @property
    def makespan(self) -> float:
        return max((job[-1] for job in self.complete if job), default=0.0)

    @property
    def finished(self) -> bool:
        return self.n_done >= self.total_ops

    # ── 可行动作（原文 §IV-B2：工序前驱完工；机台兼容且空闲；车辆空闲）──

    def eligible(self) -> list[tuple[int, int]]:
        return [(j, self.op_idx[j]) for j in range(self.n_jobs)
                if self.op_idx[j] < len(self.jobs[j]) and self.op_ready[j] <= self.now + _EPS]

    def idle_machines(self, j: int, o: int) -> list[int]:
        return [k for k, _ in self.jobs[j][o] if self.mach_free[k] <= self.now + _EPS]

    def idle_vehicles(self) -> list[int]:
        return [u for u in range(self.n_agv) if self.veh_free[u] <= self.now + _EPS]

    def flat_to_job_op(self, i: int) -> tuple[int, int]:
        """扁平工序号 → `(作业号, 工序号)`。模型侧的动作是扁平号（与 `graph().eligible` 同序）。"""
        if not 0 <= i < self.total_ops:
            raise IndexError(f"扁平工序号越界：{i}（总工序数 {self.total_ops}）")
        j = int(np.searchsorted(self._job_off, i, side="right") - 1)
        return j, i - int(self._job_off[j])

    def has_action(self) -> bool:
        if not self.idle_vehicles():
            return False
        return any(self.idle_machines(j, o) for j, o in self.eligible())

    def advance(self) -> None:
        """时间前进到下一个事件（机台完工 / 车辆空出 / 作业就绪）。没有可行动作时调用。"""
        cand = [x for x in self.mach_free if x > self.now + _EPS]
        cand += [x for x in self.veh_free if x > self.now + _EPS]
        cand += [x for x in self.op_ready if x > self.now + _EPS]
        if not cand:
            raise RuntimeError("无可行动作、也没有未来事件——环境卡死（不应发生）")
        self.now = float(min(cand))

    # ── 状态转移 ──

    def step(self, j: int, o: int, k: int, u: int) -> float:
        """执行复合动作 `(O_ij, M_k, V_u)`，返回**即时奖励**（原文 §IV-B4 的 Cmax 下界差）。"""
        lb_before = self.cmax_lb()
        if (j, o) not in self.eligible():
            raise ValueError(f"工序 ({j},{o}) 当前不可行（前驱未完工或已排）")
        if k not in self.idle_machines(j, o):
            raise ValueError(f"机台 {k} 对工序 ({j},{o}) 不可行（不兼容或非空闲）")
        if u not in self.idle_vehicles():
            raise ValueError(f"车辆 {u} 当前非空闲")
        p = self._proc[j][o][k]
        t_pick = max(self.now, self.veh_free[u]) + self.trans[self.veh_loc[u], self.job_loc[j]]
        t_arr = t_pick + self.trans[self.job_loc[j], k + 1]
        end = t_arr + p
        self.mach_free[k] = end             # 机台自车辆到达起被占用
        self.veh_free[u] = t_arr
        self.veh_loc[u] = k + 1
        self.op_machine[j][o] = k
        self.start[j][o] = t_arr
        self.complete[j][o] = end
        self.done[j][o] = True
        self.op_idx[j] = o + 1
        self.op_ready[j] = end
        self.job_loc[j] = k + 1
        self.job_end[j] = end
        self.n_done += 1
        return lb_before - self.cmax_lb()

    def cmax_lb(self) -> float:
        """原文 §IV-B4 的 `C_max(s_t)`：未排工序的估计完工下界取最大。

        递归式 `C_LB(O_ij) = C_LB(O_i(j−1)) + T̄p_ij`；前驱已排时换成实际完工。终态 = 实际 Cmax。
        """
        if self.finished:
            return self.makespan
        worst = 0.0
        for j in range(self.n_jobs):
            o = self.op_idx[j]
            if o >= len(self.jobs[j]):
                continue
            base = self.complete[j][o - 1] if o > 0 else 0.0
            base += sum(self._avg_p[j][o:])
            worst = max(worst, base)
        return worst

    # ── 图状态（原文 §IV-B1 的原始特征）──

    def graph(self) -> GraphState:
        n_ops = self.total_ops
        m, v = self.n_machines, self.n_agv
        idle_mach = np.array([t <= self.now + _EPS for t in self.mach_free], dtype=bool)
        idle_veh = np.array([t <= self.now + _EPS for t in self.veh_free], dtype=bool)
        n_idle_v = int(idle_veh.sum())

        op_feat = np.zeros((n_ops, 7), dtype=float)
        compat = np.zeros((n_ops, m), dtype=bool)
        om_edge = np.zeros((n_ops, m), dtype=float)
        eligible = np.zeros(n_ops, dtype=bool)
        unsched_per_job = [len(job) - self.op_idx[j] for j, job in enumerate(self.jobs)]
        for j, job in enumerate(self.jobs):
            off = int(self._job_off[j])
            for o, alts in enumerate(job):
                i = off + o
                if self.done[j][o]:
                    op_feat[i, 0] = 1.0
                    op_feat[i, 3] = self._proc[j][o][self.op_machine[j][o]]
                else:
                    op_feat[i, 3] = self._avg_p[j][o]
                    op_feat[i, 5] = self._estimate_c(j, o)
                    op_feat[i, 6] = self._est_start(j, o)
                    eligible[i] = ((o == self.op_idx[j]) and self.op_ready[j] <= self.now + _EPS
                                   and any(idle_mach[k] for k, _ in alts) and bool(n_idle_v))
                op_feat[i, 1] = float(sum(1 for k, _ in alts if idle_mach[k]))
                op_feat[i, 2] = float(n_idle_v)
                op_feat[i, 4] = float(unsched_per_job[j])
                for k, t in alts:
                    compat[i, k] = True
                    om_edge[i, k] = t

        mach_feat = np.zeros((m, 4), dtype=float)
        for k in range(m):
            mach_feat[k, 0] = 1.0 if not idle_mach[k] else 0.0
            mach_feat[k, 1] = float(sum(1 for j, job in enumerate(self.jobs)
                                        for o in range(self.op_idx[j], len(job))
                                        if k in self._proc[j][o]))
            mach_feat[k, 2] = self.mach_free[k]
            busy = sum(self.complete[j][o] - self.start[j][o]
                       for j, job in enumerate(self.jobs)
                       for o in range(len(job))
                       if self.done[j][o] and self.op_machine[j][o] == k)
            mach_feat[k, 3] = busy / max(self.now, _EPS)

        veh_feat = np.zeros((v, 4), dtype=float)
        n_unsched = sum(unsched_per_job)
        for u in range(v):
            veh_feat[u, 0] = 1.0 if not idle_veh[u] else 0.0
            veh_feat[u, 1] = float(n_unsched)
            veh_feat[u, 2] = self.veh_free[u]
            veh_feat[u, 3] = self.veh_loc[u] / max(m, 1)

        ov_edge = np.zeros((n_ops, v), dtype=float)
        for j, job in enumerate(self.jobs):
            off = int(self._job_off[j])
            for o in range(len(job)):
                ov_edge[off + o, :] = self.trans[self.veh_loc, self.job_loc[j]]
        return GraphState(op_feat=op_feat, mach_feat=mach_feat, veh_feat=veh_feat,
                          om_edge=om_edge, ov_edge=ov_edge, mm_edge=self.trans[1:, 1:].copy(),
                          compat=compat, eligible=eligible, idle_mach=idle_mach,
                          idle_veh=idle_veh, n_ops=n_ops,
                          n_machines=m, n_agv=v)

    # ── 内部小工具 ──

    def _estimate_c(self, j: int, o: int) -> float:
        """原文的估计完工时刻 `Ĉ_i = C_ij' + Σ 未完工各道平均加工时间`。"""
        base = self.complete[j][o - 1] if o > 0 else 0.0
        return base + sum(self._avg_p[j][o:])

    def _est_start(self, j: int, o: int) -> float:
        """原文的估计开工时刻 `T^s_ij = C_ij' + Σ 之后各道平均加工时间`。"""
        base = self.complete[j][o - 1] if o > 0 else 0.0
        return base + sum(self._avg_p[j][o + 1:])
