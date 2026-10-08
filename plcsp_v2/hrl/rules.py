# -*- coding: utf-8 -*-
"""超启发式的**规则集**（步 3）：PA 的 12 条复合派工规则 + LA 的 3 条派车规则。

规则是**纯函数**——自变量只有 `env/rule_views.py` 的 `PAView` / `LAView`（纯数据），
不读 SimPy 对象、不抽随机数、不持状态。故同一份规则既能当**基线**直接跑，
也能当**动作空间**（策略从 12 条里选 1 条，见步 4 的 PA 头）。

══ PA：12 条 = {SPT, EAJ, SRPT, EDD} × {SMPT, MINU, EAM} ══

**前 9 条逐条对应 AEI2025 的 Rule 1–9，且顺序相同**（`docs/citation-cards.md` 的 AEI2025 卡）：

| 我们的下标 | AEI2025 | 规则 |
|---|---|---|
| 0–2 | Rule 1–3 | `SPT` × {SMPT, MINU, EAM} |
| 3–5 | Rule 4–6 | `EAJ` × {SMPT, MINU, EAM} |
| 6–8 | Rule 7–9 | `SRPT` × {SMPT, MINU, EAM} |
| **9–11** | — | **`EDD` × {SMPT, MINU, EAM}（我们新增）** |

**`EDD` 是新增的第 4 条工件规则**：他们那 9 条**没有交期规则**，而 ⑧ 交期在我们的保留
约束里（设计 §2）。加一条 ⟹ 动作空间 9 → 12。**这是有意的偏离，论文里要写明。**

**工件规则口径**（AEI2025 原文："select an unfinished job and assign an appropriate
machine to process the earliest pending operation of the selected job"）：

- `SPT`  —— 最早待加工工序的加工时间 `APT` 最小
  ⚠️ **存疑，未核**：卡里只写"加工时间最短"，**没说取 `APT` 平均还是 `min_k` 最短**。
  我们取 `APT`，依据是他们的状态特征 f24–f26 是 `APT_{i,OP_i+1}` 的 mean/min/max ——
  **那是推断，不是原文。** 论文里不得声称此处一致。（登记在 §72.5）
- `EAJ`  —— **可开工时刻**最早：`max(作业就绪时刻, min_{候选机台} 机台空闲时刻)`
  ⚠️ **存疑，未核**：卡里只写"能被最早加工的工件"。**"作业就绪时刻"他们用没用、怎么算，
  卡里读不出来。** 论文里不得声称此处一致。（登记在 §72.5）
- `SRPT` —— **剩余**加工时间 `RTJ_i(t)` 最小（此项**可核**：卡里给了 `RTJ_i(t)` 的公式）
- `EDD`  —— 交期最早（**新增**）

**机台规则口径**（在**被选中工件**的候选机台里挑）：

- `SMPT` —— 加工时长最短
- `MINU` —— 利用率最低（`busy_min / max(now, MINU_TIME_FLOOR)`）
- `EAM`  —— 机台最早可用（`avail_at` 最小）

══ 平局 = **随机**（照 AEI2025） ══

AEI2025 原文："若多个工件/机器/AGV 满足条件，则**随机选一个**"。我们照做——
`apply_pa_rule(rule, view, pool, tie_rng)` 的 `tie_rng` 非空时，在**所有并列最优**的
候选里**等概率抽一个**。

⚠️ **`tie_rng` 由环境给、且是独立子流**（`default_rng([seed_chain, PA_TIE_SEED])`），
**不复用机台故障那条共享 `rng`**——复用会把故障抽样整体移位，"只改了平局"变成
"连扰动流都换了"。车队早已是各自独立的流（`[seed_chain, 1000+a]`），PA 只是照同一形状再加一条。

`tie_rng=None` = **确定性**平局（工件取池序最小、机台取机台号最小）。留给两处用：
规则单测（要钉死具体选择）、以及"平局不开随机"的消融。
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ..env.rule_views import INF, LAView, PAView

# ══════════ PA ══════════

PA_JOB_RULES: tuple[str, ...] = ("SPT", "EAJ", "SRPT", "EDD")
PA_MACHINE_RULES: tuple[str, ...] = ("SMPT", "MINU", "EAM")

PA_RULE_NAMES: tuple[str, ...] = tuple(
    f"{jr}+{mr}" for jr in PA_JOB_RULES for mr in PA_MACHINE_RULES)
N_PA_RULES = len(PA_RULE_NAMES)                     # 12

N_AEI2025_RULES = 9
"""AEI2025 的规则条数。**`PA_RULE_NAMES[:9]` 与他们的 Rule 1–9 同名同序**（有测试核对）。"""

MINU_TIME_FLOOR = 1.0
"""`MINU` 分母 `MCT_k` 的**地板** [min]——见 `minu_denominator`。

⚠️ 这是**唯一**一处保留的、与 AEI2025 不同的口径，理由是**他们原文在那个点上是空的**
（未派过活的机台 `MCT_k = 0`，除零）。别处（平局、`MINU` 分子、`EAM` 的 `MCT_k`）
**都已照他们的口径改齐**（2026-10-08，见 `docs/progress-log.md` §71）。
"""


def _resolve_pa_rule(rule: int | str) -> int:
    """规则标识（下标，或 `"SPT+SMPT"` 这样的名字）→ 下标。**非法值显式报错**。"""
    if isinstance(rule, str):
        if rule not in PA_RULE_NAMES:
            raise ValueError(f"未知的 PA 规则名 {rule!r}；可选={PA_RULE_NAMES}")
        return PA_RULE_NAMES.index(rule)
    idx = int(rule)
    if not (0 <= idx < N_PA_RULES):
        raise ValueError(f"PA 规则下标 {idx} 越界（合法 0..{N_PA_RULES - 1}）")
    return idx


def pa_job_score(rule: str, view: PAView, job: int) -> float:
    """工件规则的**打分**（越小越先）。`rule ∈ PA_JOB_RULES`。

    ⚠️ `SPT` 与 `EAJ` 的口径**存疑未核**（卡里读不出他们的确切式子）——见模块 docstring。
    论文里不得声称这两项与 AEI2025 一致。
    """
    jv = view.jobs[job]
    if rule == "SPT":
        return jv.apt
    if rule == "SRPT":
        return jv.rem_apt
    if rule == "EDD":
        return jv.due
    if rule == "EAJ":
        if not jv.alts:
            return INF                             # 已完工（不进池；防御性返回）
        return max(jv.ready_at, min(view.machines[m].avail_at for m, _t in jv.alts))
    raise ValueError(f"未知的工件规则 {rule!r}；可选={PA_JOB_RULES}")


def pa_machine_score(rule: str, view: PAView, alt: tuple[int, float]) -> float:
    """机台规则的**打分**（越小越先）。`alt = (机台号, 加工时长)`。"""
    m, t = alt
    if rule == "SMPT":
        return float(t)
    if rule == "EAM":
        return view.machines[m].avail_at
    if rule == "MINU":
        return view.machines[m].assigned_min / minu_denominator(view, m)
    raise ValueError(f"未知的机台规则 {rule!r}；可选={PA_MACHINE_RULES}")


def minu_denominator(view: PAView, machine: int) -> float:
    """`MINU` 的分母——**≡ AEI2025 的 `MCT_k(t)`**，带一处地板。

    ⚠️ **地板不是可选项**：他们的 `MU_k = (Σ已派工时)/MCT_k(t)` 在"该机台一件活都没派过"
    时 `MCT_k = 0`——**除零，而原文没写怎么处置。那个点上他们的定义是空的。**
    取 `MINU_TIME_FLOOR` 使 `t = 0` 退化为"**已派工时绝对值最小**"，
    这是该点上唯一有意义的读法（没有任何活时，利用率无从谈起）。
    **这一处保持与原文不同，论文里写明"此处原文未定义"。**
    """
    return max(view.machines[machine].avail_at, MINU_TIME_FLOOR)


def _argmin_all(keys: Sequence[float]) -> list[int]:
    """**全部**并列最小的下标（平局要用全集，不能只取第一个）。"""
    best = min(keys)
    return [i for i, k in enumerate(keys) if k == best]


def _break_tie(cands: list[int], tie_rng: np.random.Generator | None,
               fallback: Callable[[int], object]) -> int:
    """平局：`tie_rng` 非空 ⟹ **等概率抽一个**（AEI2025 的规则）；空 ⟹ 取 `fallback` 最小者。

    ⚠️ **必须走 `tie_rng` 这个独立子流**，不能借 `np.random` 全局态、也不能复用故障流
    （见模块 docstring 的「平局」一节）。
    """
    if len(cands) == 1:
        return cands[0]
    if tie_rng is None:
        return min(cands, key=fallback)
    return int(cands[int(tie_rng.integers(len(cands)))])


def apply_pa_rule(rule: int | str, view: PAView, pool: Sequence[tuple[int, int]],
                  tie_rng: np.random.Generator | None = None) -> tuple[int, int]:
    """在池里应用第 `rule` 条复合规则 → `(池下标, 机台号)`。

    **先工件、后机台**（AEI2025 的顺序：先选 unfinished job，再给它的 earliest pending
    operation 配机台）——不是联合搜索。

    **平局**：`tie_rng` 非空 = 在并列最优里**等概率抽一个**（照 AEI2025）；
    空 = 确定性（工件取池序最小、机台取机台号最小）。

    ⚠️ **不修改 `view`**：批内投影由调用方用 `PAView.reserved` 显式推进（见模块 docstring）。
    """
    idx = _resolve_pa_rule(rule)
    jr = PA_JOB_RULES[idx // len(PA_MACHINE_RULES)]
    mr = PA_MACHINE_RULES[idx % len(PA_MACHINE_RULES)]
    if not pool:
        raise ValueError("apply_pa_rule 收到空池——池非空才有决策点（调用方应先判空）")
    scores = [pa_job_score(jr, view, j) for j, _oi in pool]
    pi = _break_tie(_argmin_all(scores), tie_rng, lambda k: k)
    alts = view.jobs[pool[pi][0]].alts
    if not alts:
        raise ValueError(f"池里的作业 {pool[pi][0]} 无候选机台（已完工？）——池不该含它")
    mscores = [pa_machine_score(mr, view, mt) for mt in alts]
    mi = _break_tie(_argmin_all(mscores), tie_rng, lambda k: alts[k][0])
    return int(pi), int(alts[mi][0])


def pa_rule_policy(rule: int | str) -> Callable[[PAView, Sequence[tuple[int, int]],
                                                 np.random.Generator | None],
                                                tuple[int, int]]:
    """第 `rule` 条规则 → **`policy_pa` 回调**（契约见 `SimWorld._pa_dispatch`）。

    固定规则是**基线**：`world.run(policy_pa=pa_rule_policy("SPT+SMPT"))`。
    步 4 的 PA 头用同一个 `apply_pa_rule`，只是下标由策略给。
    第三个参数 `tie_rng` 由环境给（独立子流，见模块 docstring）。
    """
    idx = _resolve_pa_rule(rule)

    def policy(view: PAView, pool: Sequence[tuple[int, int]],
               tie_rng: np.random.Generator | None = None) -> tuple[int, int]:
        return apply_pa_rule(idx, view, pool, tie_rng)

    return policy


# ══════════ LA ══════════

LA_RULE_NAMES: tuple[str, ...] = ("MINCOST", "MINU", "LATEST_IN_TIME")
N_LA_RULES = len(LA_RULE_NAMES)                     # 3


def _resolve_la_rule(rule: int | str) -> int:
    if isinstance(rule, str):
        if rule not in LA_RULE_NAMES:
            raise ValueError(f"未知的 LA 规则名 {rule!r}；可选={LA_RULE_NAMES}")
        return LA_RULE_NAMES.index(rule)
    idx = int(rule)
    if not (0 <= idx < N_LA_RULES):
        raise ValueError(f"LA 规则下标 {idx} 越界（合法 0..{N_LA_RULES - 1}）")
    return idx


def la_score(rule: str, view: LAView, aid: int) -> tuple:
    """LA 规则的**打分**（越小越先）。返回元组，以便第三条表达两级优先。

    - `MINCOST`（AEI2025 Rule 1）—— 完成本次物流任务的运输成本最低：`空载段 + 负载段`。
      ⚠️ **负载段与车号无关**（同一任务、同一条路径）⟹ 本规则实际等价于"**就近派车**"。
      这不是实现取巧，是成本式本身如此（`tt_{l'l}` 对所有车同值）。
    - `MINU`（Rule 2）—— 利用率最低：`累计行驶工时 / max(now, 地板)`。
      ⚠️ **分母与他们的 `ACT_v(t)` 不同**（他们是"该车已派物流任务的最晚完工时刻"）、
      **分子也少了故障中断时间 `AT_v`**：我们**逐车的已派任务账本还没建**。
      **登记在 §71.4 / §72.1，不要当成已对齐。**
    - `LATEST_IN_TIME`（Rule 3）—— 原文：*"select the AGV that can **complete the no-load stage
      at the latest before the earliest transportable time of assigned operation**; and if no
      eligible AGV exists, select the AGV that can complete the no-load stage earliest"*。
      **2026-10-08 起这条不再塌陷**：LA 的决策时点移到了**派工那一刻**（§74.5），
      `view.ready_at` 是**未来**时刻，故"不晚于它"有真实内容（实测 56% 的池内条目可运输时刻
      在未来，均值提前 1.8 min、最大 6.0 min）。
    """
    if rule == "MINCOST":
        return (view.empty_leg(aid) + view.loaded_leg(aid),)
    if rule == "MINU":
        # ⚠️ 分母是全局 `now`，**不是**他们的 `ACT_v`（见上文）。
        return (view.vehicles[aid].busy_min / max(view.now, MINU_TIME_FLOOR),)
    if rule == "LATEST_IN_TIME":
        arrive = max(view.vehicles[aid].free_at, view.now) + view.empty_leg(aid)
        # 第一级：0 = 赶得上（空载段在"可运输时刻"之前跑完）、1 = 赶不上。
        # 第二级：赶得上的取**最晚**（`-arrive` 升序 = arrive 降序）；赶不上的取**最早**。
        return (0, -arrive) if arrive <= view.ready_at else (1, arrive)
    raise ValueError(f"未知的 LA 规则 {rule!r}；可选={LA_RULE_NAMES}")


def la_rule_policy(rule: int | str
                   ) -> Callable[[LAView, Sequence[int]], int]:
    """第 `rule` 条 LA 规则 → **`policy_l` 回调**：`policy_l(view, cand_v) -> 车号`。

    契约与 `policy_pa` 对称：环境装好 `LAView`（纯数据，含 `ready_at` 与行程时长口径），
    规则只读它——**规则层不认识 SimPy，也不认识布局**。
    """
    name = LA_RULE_NAMES[_resolve_la_rule(rule)]

    def policy(view: LAView, cand_v: Sequence[int]) -> int:
        if not cand_v:
            raise ValueError("LA 候选集为空——派车决策点不该出现（全车队恒可候选）")
        return int(min(cand_v, key=lambda a: (la_score(name, view, int(a)), int(a))))

    return policy
