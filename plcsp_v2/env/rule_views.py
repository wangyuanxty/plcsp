# -*- coding: utf-8 -*-
"""下层规则求值的**只读视图**（步 3）——`pa`（生产派工）与 `la`（物流派车）两份。

**为什么要视图**：AEI2025 的 PA/LA 动作空间是**复合派工规则**（CDR），规则的自变量是若干
**派生标量**（平均加工时长 APT、剩余加工时间 RTJ、机台最早可用时刻、资源利用率……）。
这些量在 `env/` 侧由**活状态**现算——策略层（`hrl/`）不该读 SimPy 对象，`env/` 也不该
依赖 `nn/`。视图就是这两层之间的**契约**：`env/` 装好纯数据，`hrl/rules.py` 只读它。

**口径来源**：`docs/citation-cards.md` 的 AEI2025 卡（Shi et al., *Advanced Engineering
Informatics* 65:103195, 2025）第 3 节「动作空间」与「状态定义」。凡本实现与他们不同的地方，
**逐条写在字段注释里**，不静默偏离。

══ 批内投影（本模块唯一不显然的设计）══

`_pa_dispatch` **一次把池排空**（设计 §14.3 的"一批派"）。同一批里状态**不变**，
若规则只看静态快照，`EAM` / `MINU` 会给**整批**挑同一台机台——不是规则的本意。
故 `PAView.reserved()` 在每次派工后返回一份**把这次预留记进去**的新视图，
使同批的后续派工看得到前面已派出去的活。**本视图每次 `_pa_dispatch` 调用现建一份，
跨调用不保留**（跨批的在途预留不计，理由见 `build_pa_view`）。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import NamedTuple

from .instances import Instance
from .snapshot import Snapshot

INF = float("inf")


class PAJob(NamedTuple):
    """PA 池里一个候选作业的**规则自变量**（下标 = 工件号，含已完工者）。"""

    oi: int
    """该作业的**最早待加工工序**（= 已完成工序数）。`oi == len(alts)` ⟹ 已完工。"""
    alts: tuple[tuple[int, float], ...]
    """本工序的候选 `(机台, 加工时长)`。已完工 = 空元组。"""
    apt: float
    """本工序的**平均**加工时长 `APT_ij = (Σ_k pt_ijk)/|候选|`（AEI2025 的 `APT_ij`）。

    ⚠️ **为什么用平均而不是最短**：复合规则是"**先**选工件、**再**选机台"，故工件规则
    必须与机台无关；AEI2025 的状态特征 f24–f26 正是 `APT_{i,OP_i+1}` 的 mean/min/max，
    证实他们也是这么用的。"""
    rem_apt: float
    """自本工序起、剩余各道工序 `APT` 之和（AEI2025 的 `RTJ_i(t)`，状态特征 f27–f29）。"""
    ready_at: float
    """该作业**可开工**的最早时刻（`EAJ` 的自变量）。

    口径：作业正占着某台机（`at_machine >= 0`）⟹ `now + 该机标的剩余`；否则 `now`。
    ⚠️ **在途作业按 `now` 计**（乐观）——运输到达时刻在快照里不可知。"""
    due: float
    """⑧ 交期；`+inf` = 该实例无交期（`JobState.due` 的 0.0 哨兵已就地转成 `inf`）。"""


class PAMachine(NamedTuple):
    """一台机台的**规则自变量**。"""

    avail_at: float
    """预计空闲时刻（`EAM` 的自变量）——**≡ AEI2025 的 `MCT_k(t)`**。

    口径照抄他们：`MCT_k(t) = max_{已派给 k 的工序} ct`，即"**已派工序的最晚完工时刻**"。
    我们的实现 = `max(物理清空时刻, 账本 mct_k)`：

    - **账本项**（主项）：PA 每次派工推进 `mct_k ← max(mct_k, 当时的 avail, now) + pt`，
      **跨派工批次存活**（存 `stats["pa_mct"]`）⟹ **在途预留看得见**；
    - **物理项**（下界）：`now + 在制标称剩余 + 输入缓冲积压`。账本是**计划**，会与
      实况漂（故障、保养、运输延误）⟹ 取 `max` 让实况只能把它推后、不能提前。

    ⚠️ **不含**从工件当前所在地到该机台的运输时长（派工时无从知道），故 `EAM` 略偏好远机台。"""
    assigned_min: float
    """已派给它的加工工时之和 [min]（`MINU` 的分子）——**≡ AEI2025 的 `Σ λ_ijk·pt_ijk`**。

    ⚠️ 是**计划**口径（派了就计），不是"已切削工时"：后者看不见在途预留。
    **分母见 `hrl.rules.minu_denominator`。**"""


def advance_avail(avail: float, now: float, proc: float) -> float:
    """把一笔 `proc` 分钟的活记进空闲时刻为 `avail` 的机台 → **新的**空闲时刻。

    式：`max(avail, now) + proc`。
    ⚠️ **批内投影与跨批账本共用这一处**（`PAView.reserved` 与 `_pa_dispatch` 的账本推进），
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

    def reserved(self, machine: int, proc: float) -> PAView:
        """本次派工的**投影**：返回一份把 `(机台 machine, 时长 proc)` 记进去的新视图。

        ⚠️ 只管**本批**（同一 `_pa_dispatch` 调用内的先后两笔）。
        跨批的那部分由 `stats["pa_mct"]` / `stats["pa_assigned_min"]` 账本承担——
        本视图下次被 `build_pa_view` 建出时就从账本读。**两者共用 `advance_avail`**。
        """
        ms = list(self.machines)
        old = ms[machine]
        ms[machine] = PAMachine(avail_at=advance_avail(old.avail_at, self.now, proc),
                                assigned_min=old.assigned_min + proc)
        return replace(self, machines=tuple(ms))


def build_pa_view(snap: Snapshot, inst: Instance, plans, pa_mct, pa_assigned_min) -> PAView:
    """活快照 + 实例 + 计划表 + **PA 账本** → `PAView`（**纯函数**）。

    `plans[j][oi]` = PA 已写下的机台预留（`None` = 尚未派）；本函数**只读**它。
    `pa_mct` / `pa_assigned_min` = 跨批账本（`stats` 里的两个逐机台列表，见 `PAMachine`）。

    ⚠️ **账本必须跨批存活**——这正是本口径相对"只读物理量"的全部好处：
    实测 MK01 有 **18/55 = 33%** 的派工时刻存在"已预留、尚未入目标机台缓冲"的件，
    只看物理量的口径在那些时刻是**瞎的**（见 `docs/progress-log.md` §71）。
    """
    jobs: list[PAJob] = []
    for j, ops in enumerate(inst.jobs):
        js = snap.jobs[j]
        oi = int(js.done_ops)
        if oi >= len(ops):
            jobs.append(PAJob(oi=oi, alts=(), apt=0.0, rem_apt=0.0,
                              ready_at=snap.now, due=(js.due if js.due > 0.0 else INF)))
            continue
        alts = tuple((int(m), float(t)) for m, t in ops[oi])
        # APT：逐道工序在**各自候选机台**上的平均时长（AEI2025 的 APT_ij）
        apts = [sum(t for _m, t in ops[o]) / len(ops[o]) for o in range(oi, len(ops))]
        at = int(js.at_machine)
        ready = snap.now
        if 0 <= at < len(snap.machines):
            ready += float(snap.machines[at].remaining_min)
        jobs.append(PAJob(oi=oi, alts=alts, apt=float(apts[0]), rem_apt=float(sum(apts)),
                          ready_at=ready, due=(js.due if js.due > 0.0 else INF)))
    machines = tuple(
        PAMachine(
            # AEI2025 的 `MCT_k`：账本（计划）与物理清空时刻取 max。
            avail_at=max(pa_mct[i] if i < len(pa_mct) else 0.0,
                         snap.now + float(m.remaining_min) + float(m.backlog_min)),
            assigned_min=float(pa_assigned_min[i]) if i < len(pa_assigned_min) else 0.0)
        for i, m in enumerate(snap.machines))
    return PAView(now=float(snap.now), jobs=tuple(jobs), machines=machines, snap=snap)


# ══════════ LA（物流派车） ══════════

class LATask(NamedTuple):
    """一次派车任务的身份（端点号口径同 `des.py`：机台 `0..m-1`、装卸站 = `m`）。"""

    job: int
    oi: int
    frm_node: int
    to_node: int


class LAVehicle(NamedTuple):
    node: int
    """当前所在通道节点；`-1` = **尚未出车**。⚠️ 这种情况按"**已在取货点**"处置——
    与 `AgvSim.run` 的空载段判据（`pos_node is not None and pos_node != a`）**同一条**，
    不是另立口径。"""
    busy_min: float
    """累计**行驶**工时 [min]（`MINU` 的自变量；由 `stats["agv_busy_min"]` 来）。"""


@dataclass(frozen=True)
class LAView:
    """LA 一次派车决策的完整输入。"""

    now: float
    task: LATask
    vehicles: tuple[LAVehicle, ...]
    leg_min: Callable[[int, int], float]
    """`leg_min(u, v) -> 分钟`——**与仿真同一条**行程时长口径（`SimWorld.leg_min`），
    规则不另造距离量。"""

    def empty_leg(self, aid: int) -> float:
        """第 `aid` 台车跑完**空载段**（当前停位 → 取货点）所需时长 [min]。"""
        node = self.vehicles[aid].node
        if node < 0 or node == self.task.frm_node:
            return 0.0
        return float(self.leg_min(node, self.task.frm_node))

    def loaded_leg(self, aid: int) -> float:
        """**负载段**（取货点 → 卸货点）时长 [min]。⚠️ 与 `aid` 无关（同一任务、同一路径），
        故对**排序**是常数——但 AEI2025 的成本式含它，此处照写以便逐条对上。"""
        del aid
        if self.task.frm_node == self.task.to_node:
            return 0.0
        return float(self.leg_min(self.task.frm_node, self.task.to_node))
