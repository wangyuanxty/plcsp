# -*- coding: utf-8 -*-
"""**NHDRL 的状态特征**（f1–f48）—— 照 Shi et al., AEI 65 (2025) 103195 逐条实现。

出处：`docs/citation-cards.md:1528-1558`（精读时逐条抄的）。三个 agent 各读自己的子集：

| agent | 特征 | 个数 |
|---|---|---|
| **OA** | f1–f31 | 31 |
| **PA** | f1–f7 + f16–f29 + f30–f32 | 24 |
| **LA** | f8–f15 + f16–f18 + f33–f48 + f30–f32 | 30 |

⚠️ **原文的 `f32` 是 OA 给 PA 的「目标决策」**（选优化 MS 还是 TC）。本模块把它当**入参**
传进来 —— 特征层不持策略状态。

══ 关键定义（`citation-cards.md:1546` 逐字）══

    OP_i(t)  = Σ_j Σ_k λ_ijk                    已分配工序数
    CRJ_i(t) = OP_i(t) / (Ji+1)
    APT_ij   = (Σ_k α_ijk·pt_ijk) / (Σ_k α_ijk)  候选机台上的**平均**加工时长
    RTJ_i(t) = Σ_{j=OP_i(t)}^{Ji+1} APT_ij
    MCT_k(t) = max_{i,j≤OP_i(t)}(λ_ijk·ct_ij)    机台最早可用 —— 本仓 `Snapshot.mct_assigned`
    MU_k(t)  = (Σ_i Σ_{j≤OP_i(t)} λ_ijk·pt_ijk) / MCT_k(t)   —— 分子 = `assigned_min`
    ACT_v(t) = max_{i,j≤OP_i(t)}(x_ijv·lct_ij)   车最早可用 —— 本仓 `Snapshot.agv_act`
    AW_v(t)  = ΣΣ x_ijv(nct−nst+lct−lst) + AT_v  车累计行驶 —— 本仓 `VehicleState.busy_min`
    AU_v(t)  = AW_v(t) / ACT_v(t)

⚠️ **`AT_v`（故障致任务中断的额外运输时间）本仓没有单独记账** ⟹ `AW_v` 取 `busy_min`
（空载+负载行驶时长），**不含故障中断的额外时间**。这是差异，要写明。

⚠️ **f37–f48 依赖"所分配物流任务"** —— 本仓的口径是：
`l''` = 车 v 当前所在节点（`VehicleState.node`），`l'` = 被 PA 选中的机台的装卸节点。
`TV_v = leg(l'', l')`（空载段），`tt_{l'l} = ...` 的 `l`（下一程终点）**卡里没界定**
⟹ 本仓取「该工序之后工件要去的位置」不可知，**改用同一次任务内的两段**：
空载 `leg(l'', l')` + 负载 `leg(l', l_next)`，其中 `l_next` 取该作业**下一道工序的候选
机台里最早可用的那台**（无下一道则为 L/U 站）。**这一条是解释，不是原文。**
"""
from __future__ import annotations

import os

import numpy as np

from .instances import Instance
from .layout import Layout
from .snapshot import Snapshot

C_COMMISSION, UC_TRANSPORT = 7.0, 0.5
"""物流成本的常数项 / 单位运输时间成本（原文验证环境取值，见 `citation-cards.md:1494`）。"""

F_OA, F_PA, F_LA = 31, 24, 30


def _safe(x: float, lo: float = 0.0, hi: float = 1e6) -> float:
    """限幅 + 非有限值归 0。**不静默给 NaN**（NaN 会一路顺进网络）。"""
    if not np.isfinite(x):
        return 0.0
    return float(min(max(x, lo), hi))


def _ratio(a: float, b: float) -> float:
    """`a/b`，`b ≤ 0` 取 0（原文未写除零处置，见 `docs/v2/progress-log.md:266`）。"""
    return float(a) / float(b) if float(b) > 0.0 else 0.0


def _apt(inst: Instance, i: int, j: int) -> float:
    """`APT_ij` = 第 i 件第 j 道工序在**候选机台上加工时长的平均**。"""
    ops = inst.jobs[i]
    if j >= len(ops) or not ops[j]:
        return 0.0
    return float(np.mean([float(t) for _m, t in ops[j]]))


class NHFeatures:
    """把 `Snapshot` + `Instance` + `Layout` 转成三个 agent 的特征向量。"""

    def __init__(self, inst: Instance, layout: Layout,
                 c: float = C_COMMISSION, uc: float = UC_TRANSPORT,
                 speed_m_per_min: float = 60.0) -> None:
        """`speed_m_per_min` = 车速（米/分）—— 供 `_leg` 把距离折成分钟。

        ⚠️ 默认 60 m/min（= 1 m/s）与 `SimConfig.agv_speed_mps` 的默认同源；
        **换 cfg 就要换它**，否则 f33–f48 的时间量纲与仿真不一致。
        """
        self.inst, self.layout, self.c, self.uc = inst, layout, c, uc
        self.speed_m_per_min = float(speed_m_per_min)
        self.ops_tot = [len(o) for o in inst.jobs]          # Ji+1

    # ── 逐作业的量（f16–f29 的原料）──
    def _job_vecs(self, snap: Snapshot) -> tuple[list, list, list, list]:
        asg = [min(int(j.assigned_ops), t) for j, t in zip(snap.jobs, self.ops_tot)]
        crj = [_ratio(a, t) for a, t in zip(asg, self.ops_tot)]
        # ⚠️ **f24–f26 的 `APT` 只统计「未完成工件」**（2026-10-13 修）。
        # 原文（`docs/citation-cards.md:1542`）：f24–f26 =「所有**未完成工件**最早待加工工序的
        # 平均加工时间 `APT_{i,OP_i(t)+1}` 的 mean/min/max」。
        # 此前遍历**全部**工件，而 `_apt` 对 `j >= len(ops)`（完工工件）**返回 0.0**
        # ⟹ 已完工工件被当成「APT = 0」混入：**f25（min）一旦有工件完工就恒 0（死维）**，
        # **f24（mean）被稀释**（`v3_state_info.py` 实测 5 个死维之一就是它）。
        # ⚠️ `crj` / `rtj` **不筛** —— 原文那两处写的是「工件完成率」「所有工件剩余加工时间」。
        apt_oe = [_apt(self.inst, i, a) for i, a in enumerate(asg)
                  if a < self.ops_tot[i]]
        # ⚠️ **`RTJ` 求和起点**（2026-10-17，用户要求对齐）：原文
        #    `Σ_{j=OP_i(t)}^{J_i+1}` —— 从**最后一道已分配**工序起。
        #    本仓原来从 `a`（下一道未分配）起 ⟹ 差一道。
        #    `NHDRL_RTJ_LAST=1` 打开；默认 0 ⟹ 既有读数逐位不变。
        if os.environ.get("NHDRL_RTJ_LAST", "0") == "1":
            rtj = [sum(_apt(self.inst, i, k)
                       for k in range(max(0, a - 1), self.ops_tot[i]))
                   for i, a in enumerate(asg)]
        else:
            rtj = [sum(_apt(self.inst, i, k) for k in range(a, self.ops_tot[i]))
                   for i, a in enumerate(asg)]
        return asg, crj, apt_oe, rtj

    def _mach_vecs(self, snap: Snapshot) -> tuple[list, list]:
        """`MU_k` / `MCT_k`（用 `Snapshot` 的两个账本）。"""
        mu = [_ratio(a, t) for a, t in zip(snap.assigned_min, snap.mct_assigned)]
        return mu, list(snap.mct_assigned)

    def _agv_vecs(self, snap: Snapshot) -> tuple[list, list]:
        """`AU_v` / `ACT_v`（`AW_v` 取 `busy_min`）。"""
        aw = [float(v.busy_min) for v in snap.vehicles]
        act = list(snap.agv_act) if snap.agv_act else [0.0] * len(snap.vehicles)
        au = [_ratio(a, t) for a, t in zip(aw, act)]
        return au, act

    @staticmethod
    def _mmm(xs: list, lo: float = 0.0, hi: float = 1e6) -> list:
        """`[mean, min, max]`；空列表给三个 0（不是 NaN）。"""
        if not xs:
            return [0.0, 0.0, 0.0]
        a = np.asarray(xs, dtype=float)
        return [_safe(float(a.mean()), lo, hi), _safe(float(a.min()), lo, hi),
                _safe(float(a.max()), lo, hi)]

    def oa(self, snap: Snapshot, prefs: tuple[float, float] = (5.0, 5.0)) -> np.ndarray:
        """**OA 的 31 维**（`prefs` = f30/f31 的 MS/TC 优化偏好，原文 0–10 整数）。"""
        asg, crj, apt_oe, rtj = self._job_vecs(snap)
        mu, mct = self._mach_vecs(snap)
        au, act = self._agv_vecs(snap)
        n_m, n_agv = len(snap.machines), len(snap.vehicles)
        aw = [float(v.busy_min) for v in snap.vehicles]
        f = [
            float(n_m + 1),                                   # f1 K+1（含虚拟机器）
            *self._mmm(mu),                                   # f2–f4  MU_k
            *self._mmm(mct),                                  # f5–f7  MCT_k
            float(n_agv),                                     # f8  V
            float(n_agv),                                     # f9  已启用（本仓车队恒全启用）
            *self._mmm(au),                                   # f10–f12 AU_v
            *self._mmm(act),                                  # f13–f15 ACT_v
            float(sum(self.ops_tot)),                         # f16 Σ(Ji+1)
            self.c * n_agv + self.uc * float(sum(aw)),        # f17 累计物流成本
            float(sum(t - a for t, a in zip(self.ops_tot, asg))),          # f18 剩余工序
            float(sum(1 for j in snap.jobs if not j.finished)),            # f19 未完成工件
            _ratio(sum(asg), sum(self.ops_tot)),              # f20 总完成率
            *self._mmm(crj),                                  # f21–f23 CRJ_i
            *self._mmm(apt_oe),                               # f24–f26 APT
            *self._mmm(rtj),                                  # f27–f29 RTJ
            _safe(prefs[0] / 10.0, 0.0, 1.0),                 # f30 对 MS 的偏好
            _safe(prefs[1] / 10.0, 0.0, 1.0),                 # f31 对 TC 的偏好
        ]
        assert len(f) == F_OA, len(f)
        return np.asarray(f, dtype=np.float32)

    def pa(self, snap: Snapshot, oa_dec: float = 0.0,
           prefs: tuple[float, float] = (5.0, 5.0)) -> np.ndarray:
        """**PA 的 24 维** = f1–f7 + f16–f29 + f30–f32。`oa_dec` = f32（上层给的控制信号）。"""
        v = self.oa(snap, prefs)
        f = list(v[0:7]) + list(v[15:29]) + [v[29], v[30], float(oa_dec)]
        assert len(f) == F_PA, len(f)
        return np.asarray(f, dtype=np.float32)

    def la(self, snap: Snapshot, job: int, machine: int, oa_dec: float = 0.0,
           prefs: tuple[float, float] = (5.0, 5.0)) -> np.ndarray:
        """**LA 的 30 维** = f8–f15 + f16–f18 + f33–f48 + f30–f32。

        `job`/`machine` = PA 刚选中的(工件, 机台) —— f33–f48 全部围绕这一趟任务。
        """
        v = self.oa(snap, prefs)
        lu = int(self.layout.grid.lu_dock_node)
        drop = int(self.layout.machines[machine].dock_node)
        # 下一程终点：该作业下一道工序的候选机台里最早可用的那台；无则为装卸站
        oi = min(int(snap.jobs[job].assigned_ops), self.ops_tot[job] - 1)
        nxt = self.inst.jobs[job][oi + 1] if oi + 1 < self.ops_tot[job] else []
        if nxt:
            nxt_node = int(self.layout.machines[int(nxt[0][0])].dock_node)
        else:
            nxt_node = lu
        # 逐车：空载段（车当前位置 → 取货点）+ 负载段（取货点 → 下一程）
        tv, tot = [], []
        for veh in snap.vehicles:
            src = int(veh.node) if veh.node >= 0 else lu
            t_empty = self._leg(src, drop)     # 空载段：车当前位置 → 取货点
            t_load = self._leg(drop, nxt_node)  # 负载段：取货点 → 下一程
            tv.append(t_empty)
            tot.append(t_empty + t_load)
        f = list(v[7:15]) + list(v[15:18]) + [
            _safe(float(snap.now)),                            # f33 最早可运输时刻
            float(drop),                                       # f34 load 阶段终点
            float(lu),                                         # f35 non-load 阶段起点
            _safe(list(snap.mct_assigned)[machine] if machine < len(snap.mct_assigned)
                  else 0.0),                                   # f36 所分配机台的 MCT_k
            *self._mmm(tot),                                   # f37–f39 全程时间
            *[x * self.uc for x in self._mmm(tot)],            # f40–f42 对应成本
            *self._mmm(tv),                                    # f43–f45 空载段
            *[x * self.uc for x in self._mmm(tv)],             # f46–f48 对应成本
        ] + [v[29], v[30], float(oa_dec)]
        assert len(f) == F_LA, len(f)
        return np.asarray(f, dtype=np.float32)

    def _leg(self, a: int, b: int) -> float:
        """两节点间的行程时间 [min]。

        ⚠️ **这是近似**：`SimWorld.leg_min` 走的是**通道图最短路**；特征层拿不到它
        （`env/` 不构造特征、特征层不持 SimPy 对象 —— 设计 §5.3.1 的边界）。
        此处用**节点坐标的曼哈顿距离 / 车速**。对矩形网格，两者在多数节点对上一致，
        但在绕行处**曼哈顿会低估**。这是差异，要写明。
        """
        g = self.layout.grid
        n_nodes = g.n_nodes

        def xy(node: int) -> tuple[float, float]:
            return g.lu_xy if node >= n_nodes else g.node_xy(*g.node_rc(node))

        (x1, y1), (x2, y2) = xy(int(a)), xy(int(b))
        return (abs(x1 - x2) + abs(y1 - y2)) / max(self.speed_m_per_min, 1e-9)
