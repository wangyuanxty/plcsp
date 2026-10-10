# -*- coding: utf-8 -*-
"""下层规则求值的**只读视图**（步 3）——`pa`（生产派工）与 `la`（物流派车）两份。

**为什么要视图**：AEI2025 的 PA/LA 动作空间是**复合派工规则**（CDR），规则的自变量是若干
**派生标量**（平均加工时长 APT、剩余加工时间 RTJ、机台最早可用时刻、资源利用率……）。
这些量在 `env/` 侧由**活状态**现算——策略层（`hrl/`）不该读 SimPy 对象，`env/` 也不该
依赖 `nn/`。视图就是这两层之间的**契约**：`env/` 装好纯数据，`hrl/rules.py` 只读它。

**口径来源**：`docs/citation-cards.md` 的 AEI2025 卡（Shi et al., *Advanced Engineering
Informatics* 65:103195, 2025）第 3 节「动作空间」与「状态定义」。凡本实现与他们不同的地方，
**逐条写在字段注释里**，不静默偏离。

══ 决策时点（第二稿，2026-10-08）══

`_pa_dispatch` **每次只派一件**（原文："the decision point t is defined as the time when an
operation is assigned"），**LA 在同一个决策点定车**。故本模块的两份视图都在**派工那一刻**
构造，`LAView.ready_at` 是**未来**时刻。逐机/逐车的账本（`stats["pa_mct"]` /
`pa_assigned_min` / `agv_act`）**跨决策点存活**——这是"在途预留看得见"的唯一机制。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

from .instances import Instance
from .snapshot import MachineState, Snapshot

INF = float("inf")


def machine_has_room(ms: MachineState) -> bool:
    """该机台的**输入缓冲**还能不能再收一件（② 有限缓冲）。

    `ms.in_cap is None`（② 关闭）⟹ **恒 True** —— 缓冲真的无界，"满"不存在。
    ⚠️ **不得写成 `in_q_len < in_cap`**：`None < int` 是 TypeError，而把 `None` 当 0
    会让每台机恒判为满（静默错误）。

    这是"**候选机台**"的判据（清单第 1 条）：满了的机台**不进 `alts`** ⟹ 12 条 PA 规则
    都看不到它 ⟹ 不可能被派到一台收不下的机台。规则**定义**不变，变的是候选集。
    """
    return ms.in_cap is None or int(ms.in_q_len) < int(ms.in_cap)



class PAJob(NamedTuple):
    """PA 池里一个候选作业的**规则自变量**（下标 = 工件号，含不再进池者）。"""

    oi: int
    """该作业的**最早待分配工序**——口径照 AEI2025：**最低的、`plans[j][oi]` 尚未定**的下标。

    ⚠️ **不是 `done_ops`（已完成工序数）**。原文逐字：

    > *"Let **OPi(t)** denotes the operation number of job i that **have been assigned** at
    > decision point t"* / *"`CJ(t) = {i | OPi(t) < Ji + 1}`"* / *"f24 = mean_{i∈CJ(t)}(APT_{i,
    > **OPi(t)+1**})"*

    ⟹ `OP_i` 数的是**已分配**的工序数，候选是**下一道未分配的**。**一个工件被派过之后仍留在
    池里**（它的下一道未分配工序又成为候选）——这正是"池 ≈ 未完工工件数"的来源。

    ⚠️ **2026-10-08 修**：此前用的是 `done_ops`，导致一个工件被派过就被滤掉，
    实测池在 **82%** 的决策上只剩 1 个候选（§73.1）。**错的不是事实卡，是我没读原文。**

    `oi == len(alts)`（= 该作业**全部工序都已分配**）⟹ 不进池（原文的 `OP_i(t) == Ji+1`）。
    """
    done_oi: int
    """**已完成**工序数（= `job_progress`）。规则求值一般不用它；留着供诊断与
    `ready_at` 的口径判断（见下）。"""
    alts: tuple[tuple[int, float], ...]
    """本工序（`oi`）的**候选机台** `(机台, 加工时长)` —— ⚠️ **已按 ② 滤掉输入缓冲已满的机台**
    （`machine_has_room`）。**空元组有两种来由，调用方须都当作"本工件此刻派不出去"**：

    1. 该作业**全部工序都已分配**（`oi >= len(alts)`，不再进池）；
    2. 本工序的候选机台**输入缓冲全满**（② 开启时才会出现）。

    ⚠️ 第 2 种是 **2026-10-10 新引入的**（清单第 1 条）—— 此前 `alts` 恒等于实例给的
    候选集，空元组**只**有第 1 种来由。故 `hrl.rules.apply_pa_rule` 里那句
    "已完工？"的报错文案同时过期（见 `_pa_dispatch` 的池过滤）。"""
    apt: float
    """本工序的**平均**加工时长 `APT_ij = (Σ_k pt_ijk)/|候选|`（AEI2025 的 `APT_ij`）。

    ⚠️ **为什么用平均而不是最短**：复合规则是"**先**选工件、**再**选机台"，故工件规则
    必须与机台无关；AEI2025 的状态特征 f24–f26 正是 `APT_{i,OP_i+1}` 的 mean/min/max，
    证实他们也是这么用的。"""
    rem_apt: float
    """自本工序起、剩余各道工序 `APT` 之和（AEI2025 的 `RTJ_i(t)`，状态特征 f27–f29）。"""
    ready_at: float
    """该作业**可开工**的最早时刻（`EAJ` 的自变量）。

    口径：
    - `oi == 0` ⟹ 作业在装卸站待运，`now`；
    - 前道（`oi-1`）**已完成**（`oi-1 < done_oi`）⟹ 工件已可运，`now`；
    - 前道**未完成** ⟹ `now + 该机台的在制标称剩余 + 输入缓冲积压`（= 那台机器清空的时刻，
      是完工时刻的**上界**）。

    ⚠️ **在途作业按 `now` 计**（乐观）——见 §72 的登记：**这条"不可知"其实是假的**，
    快照里有 `on_agv` + 车 `node`/`status`，可以反推 ETA。**未修，登记在 §72.1。**"""
    due: float
    """⑧ 交期；`+inf` = 该实例无交期（`JobState.due` 的 0.0 哨兵已就地转成 `inf`）。"""


class PAMachine(NamedTuple):
    """一台机台的**规则自变量**。"""

    avail_at: float
    """预计空闲时刻（`EAM` 的自变量）——**≡ AEI2025 的 `MCT_k(t)`**，**纯计划口径**。

    原文：*"MCTk(t) denotes the completion time of the **last operation assigned** to machine k"*。
    实现 = 逐机账本 `stats["pa_mct"]`：PA 每次派工推进
    `mct_k ← max(mct_k, 当时的 avail, now) + pt`，**跨决策点存活**（在途预留因此看得见）。

    ⚠️ 曾取 `max(账本, 物理清空时刻)`；2026-10-08 第三次核后去掉物理项（实测逐位相同，见
    `build_pa_view` 的注释）。
    ⚠️ **不含**从工件所在地到该机台的运输时长（派工时按原文无从知道），故 `EAM` 略偏好远机台。"""
    assigned_min: float
    """已派给它的加工工时之和 [min]（`MINU` 的分子）——**≡ AEI2025 的 `Σ λ_ijk·pt_ijk`**。

    ⚠️ 是**计划**口径（派了就计），不是"已切削工时"：后者看不见在途预留。
    **分母见 `hrl.rules.minu_denominator`。**"""


def advance_avail(avail: float, now: float, proc: float) -> float:
    """把一笔 `proc` 分钟的活记进空闲时刻为 `avail` 的机台 → **新的**空闲时刻。

    式：`max(avail, now) + proc`。
    ⚠️ **机台账本与车辆账本共用这一处**（`_pa_dispatch` 与 `_emit_task`），
    两处各写一遍必然会漂——而漂了不会报错，只会让 `EAM` 悄悄用两套口径。
    """
    return max(avail, now) + proc


@dataclass(frozen=True)
class PAView:
    """PA 一次派工决策的完整输入。"""

    now: float
    jobs: tuple[PAJob, ...]
    machines: tuple[PAMachine, ...]
    snap: Snapshot | None = None
    """全量活状态。**规则不用它**——它是给步 4 的编码器留的入口（token 从快照出）。

    可为 `None`：规则的单测只需 `now/jobs/machines` 三项就能构造一份纯规则视图，
    不必造一张完整快照（`build_pa_view` 生产路径上恒非 None）。"""

def build_pa_view(snap: Snapshot, inst: Instance, plans, pa_mct, pa_assigned_min) -> PAView:
    """活快照 + 实例 + 计划表 + **PA 账本** → `PAView`（**纯函数**）。

    `plans[j][oi]` = PA 已写下的机台预留（`None` = 尚未派）；本函数**只读**它。
    ⚠️ 每个作业的 `oi` = **最低的、`plans[j][oi]` 尚未定**的下标（原文口径，见 `PAJob.oi`）——
    **不是 `done_ops`**。
    `pa_mct` / `pa_assigned_min` = 跨批账本（`stats` 里的两个逐机台列表，见 `PAMachine`）。

    ⚠️ **账本必须跨批存活**——这正是本口径相对"只读物理量"的全部好处：
    实测 MK01 有 **18/55 = 33%** 的派工时刻存在"已预留、尚未入目标机台缓冲"的件，
    只看物理量的口径在那些时刻是**瞎的**（见 `docs/progress-log.md` §71.1）。
    """
    jobs: list[PAJob] = []
    for j, ops in enumerate(inst.jobs):
        js = snap.jobs[j]
        done = int(js.done_ops)
        row = plans[j] if j in plans else [None] * len(ops)
        oi = next((p for p in range(len(ops)) if row[p] is None), len(ops))
        due = js.due if js.due > 0.0 else INF
        if oi >= len(ops):                     # 全部工序都已分配 ⟹ 不进池（原文 `OP_i == J_i+1`）
            jobs.append(PAJob(oi=oi, done_oi=done, alts=(), apt=0.0, rem_apt=0.0,
                              ready_at=snap.now, due=due))
            continue
        # ② 有限缓冲（清单第 1 条）：**输入缓冲已满的机台不进候选**。
        # ⚠️ 越界下标**放行**（沿用本函数既有的 `0 <= k < len(snap.machines)` 守卫风格）——
        # 它本来就是坏数据，本改动不该顺手改掉它的失败方式；`in_cap` 那道判断才是新增语义。
        alts = tuple((int(m), float(t)) for m, t in ops[oi]
                     if not (0 <= int(m) < len(snap.machines))
                     or machine_has_room(snap.machines[int(m)]))
        # APT：逐道工序在**各自候选机台**上的平均时长（AEI2025 的 APT_ij）
        apts = [sum(t for _m, t in ops[o]) / len(ops[o]) for o in range(oi, len(ops))]
        ready = snap.now
        if oi > 0 and oi - 1 >= done:          # 前道未完成 ⟹ 等到那台机清空
            k_prev = row[oi - 1]
            if k_prev is not None and 0 <= k_prev < len(snap.machines):
                m_prev = snap.machines[k_prev]
                ready += float(m_prev.remaining_min) + float(m_prev.backlog_min)
        jobs.append(PAJob(oi=oi, done_oi=done, alts=alts, apt=float(apts[0]),
                          rem_apt=float(sum(apts)), ready_at=ready, due=due))
    machines = tuple(
        PAMachine(
            # **≡ AEI2025 的 `MCT_k(t)`**：逐机账本，**纯计划口径**。
            # ⚠️ 曾取 `max(账本, 物理清空时刻)`——2026-10-08 第三次核后**去掉物理项**：
            # 实测（MK01，4 条 EAM 规则 × 6 seed）两者**逐位相同**，而原文是纯计划 ⟹ 无代价对齐。
            avail_at=(pa_mct[i] if i < len(pa_mct) else 0.0),
            assigned_min=float(pa_assigned_min[i]) if i < len(pa_assigned_min) else 0.0)
        for i, m in enumerate(snap.machines))
    return PAView(now=float(snap.now), jobs=tuple(jobs), machines=machines, snap=snap)


# ══════════ LA（物流派车） ══════════

class LAVehicle(NamedTuple):
    node: int
    """当前所在通道节点；`-1` = **尚未出车**。⚠️ 这种情况按"**已在取货点**"处置——
    与 `AgvSim.run` 的空载段判据（`pos_node is not None and pos_node != a`）**同一条**，
    不是另立口径。"""
    busy_min: float
    """累计**行驶**工时 [min]（`MINU` 的分子；由 `stats["agv_busy_min"]` 来）。"""
    free_at: float
    """预计空闲时刻（`stats["agv_act"]` 的账本）——**≡ AEI2025 的 `ACT_v(t)`**
    （*"the completion time of the last task assigned to AGV v"*）。

    两处用：`MINU` 的**分母**（2026-10-08 第三次核后对齐，此前误用全局 `now`）
    与 `LATEST_IN_TIME` 的到达时刻。"""


@dataclass(frozen=True)
class LAView:
    """LA 一次派车决策的完整输入。

    ⚠️ **决策时点 = 派工那一刻**（不是运输任务生成时）——照 AEI2025：运输任务**从派工就存在**，
    带一个**未来**的可运输时刻 `ct_{i,OP_i(t)}`（卡里 f33）。**这正是 `LATEST_IN_TIME`
    （Rule 3）"不晚于可运输时刻地最晚到达"有意义的前提**；§70.8 记的"塌陷"是因为
    我们把 LA 接到了完工事件上，`ready_at` 恒等于 `now`。**2026-10-08 修。**
    """

    now: float
    job: int
    oi: int
    frm_node: int
    to_node: int
    ready_at: float
    """作业的**最早可运输时刻**——前道工序在此刻完工（`oi == 0` 时 = `now`）。"""
    vehicles: tuple[LAVehicle, ...]
    leg_min: Callable[[int, int], float]
    """`leg_min(u, v) -> 分钟`——**与仿真同一条**行程时长口径（`SimWorld.leg_min`），
    规则不另造距离量。"""
    snap: Snapshot | None = None
    """全量活状态（规则不用它；留给步 4 的编码器）。"""

    def empty_leg(self, aid: int) -> float:
        """第 `aid` 台车跑完**空载段**（当前停位 → 取货点）所需时长 [min]。"""
        node = self.vehicles[aid].node
        if node < 0 or node == self.frm_node:
            return 0.0
        return float(self.leg_min(node, self.frm_node))

    def loaded_leg(self, aid: int) -> float:
        """**负载段**（取货点 → 卸货点）时长 [min]。⚠️ 与 `aid` 无关（同一任务、同一路径），
        故对**排序**是常数——但 AEI2025 的成本式含它，此处照写以便逐条对上。"""
        del aid
        if self.frm_node == self.to_node:
            return 0.0
        return float(self.leg_min(self.frm_node, self.to_node))
