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
from ..env.snapshot import Snapshot

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


# ══════════ v3：规则超参数化（作业侧）══════════
#   设计依据：`docs/v3/design.md` §4.2。**只做作业侧** —— 四个基本量全是分钟，
#   单位自洽、零新依赖。**机台侧留到步 2**（三个量单位不齐，要引 `M_ref`）。

LAMBDA_MAX = 1.0
"""`λ` 的上界。**必须**有界 —— 见 `docs/v3/design.md` §4.0 ②：

`X + λ(Y₁+Y₂)` 当 `λ→∞` 时退化成 `λ(Y₁+Y₂)`（`X` 的系数被压成 0）
⟹ 各规则的族**塌成同一个** ⟹ 上层选规则变多余。**有界就不塌。**
"""

DUE_MARGIN_CAP = 1e6
"""`due_margin` 的**限幅** [min]（≈ 2 年，远大于任何实例的 makespan）。

⚠️ **限幅是必须的，不是保险**：无交期的工件 `due = INF`（见 `build_pa_view`），
而 `0 * inf = nan` —— `λ = 0` 时会把整项污染成 `nan`，
**那就做不到"`λ = 0` 逐位等于原规则"**（设计 §13.2 的断言 1）。
"""

PA_JOB_MIX: dict[str, tuple[str, tuple[str, ...]]] = {
    "SPT":  ("apt",         ("wait", "due")),
    "SRPT": ("rtj",         ("wait", "due")),
    "EDD":  ("due",         ("rtj", "wait")),
    "EAJ":  ("ready_delay", ("apt", "due")),
}
"""每条作业规则的 **(主量, 两个辅量)** —— 见 `docs/v3/design.md` §4.2。

⚠️ **各规则的辅量集合要"各漏掉一个不同的量"**（§4.0 ③）：
SPT 漏 `rtj` · SRPT 漏 `apt` · EDD 漏 `apt` · EAJ 漏 `wait`。
若所有规则混同一组辅量，它们就成了同一流形的不同参数化 ⟹ 上层变多余。

⚠️ **`EAJ` 的主量必须是 `ready_delay`（时长）而不是就绪**时刻** ——
后者与辅量不同单位、不能相加。给定 `now` 时两者的 `argmin` 相同（差一个常数）。
"""


def pa_job_terms(view: PAView, job: int) -> dict[str, float]:
    """该工件的**五个量**（**全是分钟** —— 这是它们能相加的前提）。

    | 键 | 含义 |
    |---|---|
    | `apt` | 本工序加工时长（`SPT` / `EAJ` 用它） |
    | `rtj` | 剩余加工工时（`SRPT` / `EDD` 用它） |
    | `wait` | **已等待**：`max(0, now − ready_at)` |
    | `due` | **交期余量**：`due − now`，**限幅到 `DUE_MARGIN_CAP`** |
    | `ready_delay` | **还要等多久能开工**：`max(0, max(ready_at, 最早可用) − now)` |

    ⚠️ `wait` 与 `ready_delay` **不会同时为正**（前者是已就绪等了多久，后者是还没就绪）
    —— 但两者都保留：不同规则要的语义不同。
    """
    jv = view.jobs[job]
    now = view.now
    if jv.alts:
        earliest = min(view.machines[m].avail_at for m, _t in jv.alts)
        start = max(jv.ready_at, earliest)
    else:
        start = jv.ready_at                      # 已完工／无可派机台（防御性）
    return {
        "apt": float(jv.apt),
        "rtj": float(jv.rem_apt),
        "wait": max(0.0, now - jv.ready_at),
        "due": min(float(jv.due) - now, DUE_MARGIN_CAP),
        "ready_delay": max(0.0, start - now),
    }


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


def pa_job_score(rule: str, view: PAView, job: int,
                 lam: tuple[float, float] = (0.0, 0.0)) -> float:
    """工件规则的**打分**（越小越先）。`rule ∈ PA_JOB_RULES`。

    `lam = (λ₁, λ₂)`：**v3 的超参数**。`(0, 0)` ⟹ **逐位等于 v2 的原规则**
    （走下面的快路，既有读数靠这条 —— 设计 §13.2 断言 1）。

    ⚠️ **2026-10-12：上述"逐位等于 v2"对 `EAJ` 已不成立。** `ready_at` 重写后
    `EAJ` 的取值与 v2 不同（v2 的 `rule_views.py` **未动**，其读数仍可复算）。
    影响：`EAJ` 与 `wait` 辅量。**`SPT`/`SRPT`/`EDD` 仍逐位等于 v2。**
    原因与判据见 `env/rule_views.estimate_ready_at` 与 `progress-log` §37.10/§37.11。

    非零时：`score = X + λ₁·Y₁ + λ₂·Y₂`（`X`/`Y` 见 `PA_JOB_MIX`）。
    ⚠️ **`λ` 必须落在 `[0, LAMBDA_MAX]`** —— 无界会让各规则趋同（见 `LAMBDA_MAX`）。

    ⚠️ `SPT` 与 `EAJ` 的口径**存疑未核**（卡里读不出他们的确切式子）——见模块 docstring。
    论文里不得声称这两项与 AEI2025 一致。
    """
    if rule not in PA_JOB_RULES:
        raise ValueError(f"未知的工件规则 {rule!r}；可选={PA_JOB_RULES}")
    lam = (float(lam[0]), float(lam[1]))
    if not all(0.0 <= x <= LAMBDA_MAX for x in lam):
        raise ValueError(f"λ 越界（须在 [0, {LAMBDA_MAX}]）：{lam}")
    if not any(lam):
        # ── 快路：**逐位等于 v2**。36 组合扫描 / 全部既有读数走这里。
        jv = view.jobs[job]
        if rule == "SPT":
            return jv.apt
        if rule == "SRPT":
            return jv.rem_apt
        if rule == "EDD":
            return jv.due
        if not jv.alts:
            return INF                             # 已完工（不进池；防御性返回）
        return max(jv.ready_at, min(view.machines[m].avail_at for m, _t in jv.alts))
    # ── v3：混合。（`INF` 已由 `pa_job_terms` 限幅）
    main, (aux1, aux2) = PA_JOB_MIX[rule]
    t = pa_job_terms(view, job)
    return t[main] + lam[0] * t[aux1] + lam[1] * t[aux2]


PA_MACH_MIX: dict[str, tuple[str, str]] = {
    "SMPT": ("pt",          "avail_delay"),
    "MINU": ("util",        "pt_over_den"),
    "EAM":  ("avail_delay", "pt"),
}
"""每条机台规则的 **(主量, 一个辅量)** —— 见 `docs/v3/design.md` §4.3。

⚠️ **机台侧每条只有 1 个参数**（作业侧是 2 个）—— 因为机台侧**唯一的无量纲量是 `util`**，
它跟另两个（分钟）不能直接相加。**为省一个参数引外部尺度（`M_ref`）不值。**
故每条只混**同分母**的那一项。
"""


def pa_machine_terms(view: PAView, alt: tuple[int, float]) -> dict[str, float]:
    """该候选机台的**四个量**。

    | 键 | 含义 | 单位 |
    |---|---|---|
    | `pt` | 本工序在它上面的加工时长 | 分钟 |
    | `avail_delay` | **还要等多久才空闲**：`max(0, avail_at − now)` | 分钟 |
    | `util` | 利用率（`assigned_min / den`） | **无量纲** |
    | `pt_over_den` | **这活占它多久的比例**（`pt / den`） | **无量纲** |

    ⚠️ **`MINU` 混 `pt_over_den`（不是 `pt`）** —— 后者是分钟，与 `util` 不能相加。
    语义："利用率低的里面，挑**这活占它多久比例小**的"。
    """
    m, t = alt
    ms = view.machines[m]
    den = minu_denominator(view, m)
    return {
        "pt": float(t),
        "avail_delay": max(0.0, ms.avail_at - view.now),
        "util": ms.assigned_min / den,
        "pt_over_den": float(t) / den,
    }


def pa_machine_score(rule: str, view: PAView, alt: tuple[int, float],
                     lam: tuple[float, ...] = (0.0,)) -> float:
    """机台规则的**打分**（越小越先）。`alt = (机台号, 加工时长)`。

    `lam`：**v3 的超参数**（每条 1 个）。`(0,)` ⟹ **逐位等于 v2 的原规则**。
    非零时：`score = X + λ·Y`（`X`/`Y` 见 `PA_MACH_MIX`）。

    ⚠️ `EAM` 的主量在 v3 是 `avail_delay`（**时长**）而不是 `avail_at`（**时刻**）；
    它比原量差一个常数 `now` ⟹ **`argmin` 不变**（同 `pa_job_score` 里 `EAJ` 的说明）。
    """
    if rule not in PA_MACHINE_RULES:
        raise ValueError(f"未知的机台规则 {rule!r}；可选={PA_MACHINE_RULES}")
    lam1 = float(lam[0]) if len(lam) else 0.0
    if not (0.0 <= lam1 <= LAMBDA_MAX):
        raise ValueError(f"λ 越界（须在 [0, {LAMBDA_MAX}]）：{lam}")
    m, t = alt
    if not lam1:
        # ── 快路：**逐位等于 v2**
        if rule == "SMPT":
            return float(t)
        if rule == "EAM":
            return view.machines[m].avail_at
        return view.machines[m].assigned_min / minu_denominator(view, m)
    main, aux = PA_MACH_MIX[rule]
    terms = pa_machine_terms(view, alt)
    return terms[main] + lam1 * terms[aux]


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
                  tie_rng: np.random.Generator | None = None,
                  lam_job: tuple[float, ...] = (0.0, 0.0),
                  lam_mach: tuple[float, ...] = (0.0,)) -> tuple[int, int]:
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
        raise ValueError(
            "apply_pa_rule 收到空池——池非空才有决策点。"
            "⚠️ ② 有限缓冲下**池可能为空**（所有未完工作业的候选机台都满了）⟹ "
            "调用方须先判空再调本函数（`des._pa_dispatch` 的 `and v.alts` 就是那道过滤）。")
    scores = [pa_job_score(jr, view, j, lam_job) for j, _oi in pool]
    pi = _break_tie(_argmin_all(scores), tie_rng, lambda k: k)
    alts = view.jobs[pool[pi][0]].alts
    if not alts:
        raise ValueError(
            f"池里的作业 {pool[pi][0]} 无候选机台——两种来由：**全部工序已分配**，"
            "或**候选机台的输入缓冲全满**（② 开启）。两种都意味着本作业不该出现在池里"
            "（`_pa_dispatch` 的 `and v.alts` 应已滤掉它）。")
    mscores = [pa_machine_score(mr, view, mt, lam_mach) for mt in alts]
    mi = _break_tie(_argmin_all(mscores), tie_rng, lambda k: alts[k][0])
    return int(pi), int(alts[mi][0])


def pa_rule_policy(rule: int | str,
                   lam_job: tuple[float, ...] = (0.0, 0.0),
                   lam_mach: tuple[float, ...] = (0.0,)
                   ) -> Callable[[PAView, Sequence[tuple[int, int]],
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
        return apply_pa_rule(idx, view, pool, tie_rng, lam_job, lam_mach)

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


LA_MIX: dict[str, str] = {
    "MINCOST": "arrive_delay",
    "MINU": "cost_over_den",
    "LATEST_IN_TIME": "cost",
}
"""每条 LA 规则的**辅量** —— 见 `docs/v3/design.md` §4.4。

⚠️ **三条都各混**同单位**的一项**：`cost`/`arrive` 是分钟、`util`/`cost_over_den` 是无量纲。
⚠️ `LATEST_IN_TIME` **的两级结构保留**，参数只在第二级加权。
"""


def la_terms(view: LAView, aid: int) -> dict[str, float]:
    """该车的**五个量**。

    | 键 | 含义 | 单位 |
    |---|---|---|
    | `cost` | 本次任务的运输成本（`空载段 + 负载段`） | 分钟 |
    | `util` | 利用率（`busy_min / free`） | **无量纲** |
    | `cost_over_den` | 这趟活占它多久的比例（`cost / free`） | **无量纲** |
    | `arrive` | 跑完空载段到达的**时刻** | 分钟（时刻） |
    | `arrive_delay` | **还要等多久才赶得上**：`max(0, arrive − ready_at)` | 分钟 |

    ⚠️ `MINU` 混的是 **`cost_over_den`** 而不是 `cost` —— 后者是分钟，与 `util` 不能相加。
    """
    av = view.vehicles[aid]
    free = max(av.free_at, MINU_TIME_FLOOR)
    cost = view.empty_leg(aid) + view.loaded_leg(aid)
    arrive = max(av.free_at, view.now) + view.empty_leg(aid)
    return {
        "cost": cost,
        "util": av.busy_min / free,
        "cost_over_den": cost / free,
        "arrive": arrive,
        "arrive_delay": max(0.0, arrive - view.ready_at),
    }


def _la_main(rule: str, view: LAView, aid: int, terms: dict[str, float]) -> float:
    """该 LA 规则的**主量**（v3 口径）—— 与原打分同序。"""
    if rule == "MINCOST":
        return terms["cost"]
    if rule == "MINU":
        return terms["util"]
    raise ValueError(f"{rule!r} 没有主量（两级结构，见 `la_score`）")


def la_score(rule: str, view: LAView, aid: int,
             lam: tuple[float, ...] = (0.0,)) -> tuple:
    """LA 规则的**打分**（越小越先）。返回元组，以便第三条表达两级优先。

    - `MINCOST`（AEI2025 Rule 1）—— 完成本次物流任务的运输成本最低：`空载段 + 负载段`。
      ⚠️ **负载段与车号无关**（同一任务、同一条路径）⟹ 本规则实际等价于"**就近派车**"。
      这不是实现取巧，是成本式本身如此（`tt_{l'l}` 对所有车同值）。
    - `MINU`（Rule 2）—— 利用率最低：原文 `AU_v(t) = AW_v(t)/ACT_v(t)`，其中
      `AW_v = Σ x_ijv(nct−nst+lct−lst) + AT_v`（**空载段 + 负载段**的时长之和，
      `AT_v` = 故障中断时间）、`ACT_v = max lct_ij`（**该车最后一笔已派任务的完工时刻**）。

      **2026-10-08 第三次核后对齐**：
      - **分母改用 `view.vehicles[aid].free_at`（≡ `ACT_v`）**，不再用全局 `now`
        —— `free_at` 就是逐车已派任务账本（`stats["agv_act"]`），口径与 `ACT_v` 同。
        ⚠️ 我此前两稿都用 `now`，理由是"逐车账本还没建"——而账本**当时就已经有了**。
      - 分子用 `busy_min`（**实际**行驶工时）。他们的 `AW_v` 是**已派**任务的时长之和；
        新流下任务在派工那刻就订给某台车，故两者只在"计划 vs 实况"上差一点。
      - `AT_v`（故障中断时间）**恒为 0**：本仓的 ⑨ 是**段间检出**，不打断正在进行的行驶
        （`AgvSim._failures`），故"被中断的运输时间"这个量在本模型里不存在。**如实声明。**
    - `LATEST_IN_TIME`（Rule 3）—— 原文：*"select the AGV that can **complete the no-load stage
      at the latest before the earliest transportable time of assigned operation**; and if no
      eligible AGV exists, select the AGV that can complete the no-load stage earliest"*。
      **2026-10-08 起这条不再塌陷**：LA 的决策时点移到了**派工那一刻**（§74.5），
      `view.ready_at` 是**未来**时刻，故"不晚于它"有真实内容（实测 56% 的池内条目可运输时刻
      在未来，均值提前 1.8 min、最大 6.0 min）。
    """
    lam1 = float(lam[0]) if len(lam) else 0.0
    if not (0.0 <= lam1 <= LAMBDA_MAX):
        raise ValueError(f"λ 越界（须在 [0, {LAMBDA_MAX}]）：{lam}")
    if lam1 and rule in LA_RULE_NAMES:
        # ── v3：混合。**不破坏 `LATEST_IN_TIME` 的两级结构** ——
        # 第一级（0 = 赶得上 / 1 = 赶不上）是**硬分类**，参数只在第二级加权。
        terms = la_terms(view, aid)
        if rule == "LATEST_IN_TIME":
            return ((0, -terms["arrive"] + lam1 * terms["cost"])
                    if terms["arrive"] <= view.ready_at
                    else (1, terms["arrive"] + lam1 * terms["cost"]))
        return (_la_main(rule, view, aid, terms) + lam1 * terms[LA_MIX[rule]],)
    if rule == "MINCOST":
        return (view.empty_leg(aid) + view.loaded_leg(aid),)
    if rule == "MINU":
        # 分母 = 该车**已派任务账本**的空闲时刻（≡ 他们的 `ACT_v`），不是全局 `now`。
        return (view.vehicles[aid].busy_min
                / max(view.vehicles[aid].free_at, MINU_TIME_FLOOR),)
    if rule == "LATEST_IN_TIME":
        arrive = max(view.vehicles[aid].free_at, view.now) + view.empty_leg(aid)
        # 第一级：0 = 赶得上（空载段在"可运输时刻"之前跑完）、1 = 赶不上。
        # 第二级：赶得上的取**最晚**（`-arrive` 升序 = arrive 降序）；赶不上的取**最早**。
        return (0, -arrive) if arrive <= view.ready_at else (1, arrive)
    raise ValueError(f"未知的 LA 规则 {rule!r}；可选={LA_RULE_NAMES}")


# ══════════ RA：保养规则（v3 新增）══════════
#   设计依据：`docs/v3/design.md` §4.5。
#
#   ⚠️ **触发点是给定的**（`des.MachineSim._pm_after_op`：一道工序完工、下一件尚未上机）
#      —— 那是"要停机才能保养"的**唯一窗口**，**规则不改它**。
#      规则决定的只是：**在那个窗口里，保不保**。
#
#   ⚠️ **逾期是硬底线**（`pm_clock >= pm_interval` ⟹ 强制保养、**不产生决策**，见 `_pm_after_op`）
#      ⟹ 规则只能把保养**提前**，不能推迟过强制点。
#      故 `threshold >= pm_interval` 时规则**恒不主动**（退化成"只靠逾期强制"），
#      `threshold → 0` 时**每次都保**。

SIGMA_DEFAULT = 0.5
"""连续动作的**探索标准差**（**固定、不学**）—— 见 `docs/v3/design.md` §6.1。

⚠️ **为什么固定**：σ 的唯一作用是探索；而**"策略塌缩"正是 v2 这一整轮在治的病**
（`docs/v2/progress-log.md` §146）。**学 σ 等于再开一个塌缩面。**

"""


def encode_param(p: float, lo: float = 0.0, hi: float = LAMBDA_MAX) -> float:
    """参数值 → **无界动作** `u`（`decode_param` 的逆）。"""
    q = (float(p) - lo) / (hi - lo)
    if not (0.0 < q < 1.0):
        raise ValueError(f"参数 {p} 不在开区间 ({lo}, {hi}) 内 —— 边界上 `logit` 发散")
    import math
    return math.log(q / (1.0 - q))


def decode_param(u: float, lo: float = 0.0, hi: float = LAMBDA_MAX) -> float:
    """**无界动作** `u` → 有界参数 `param = lo + (hi − lo)·sigmoid(u)`（设计 §6.1）。

    ⚠️ **`logp` 必须算在 `u` 上，不是算在 `param` 上** —— `u` 才是被采样的动作；
    算在 `param` 上会漏掉 Jacobian，**策略梯度就错了**（设计 §13.3 ①）。
    本函数只负责"动作怎么解释"，不参与 `logp`。
    """
    import math
    return float(lo) + (float(hi) - float(lo)) / (1.0 + math.exp(-float(u)))


PM_RULE_NAMES: tuple[str, ...] = ("AGE", "QUEUE_EMPTY", "OLDEST")
N_PM_RULES = len(PM_RULE_NAMES)                     # 3


def _resolve_pm_rule(rule: int | str) -> str:
    """规则标识（下标，或 `"AGE"` 这样的名字）→ 名字。**非法值显式报错**。"""
    if isinstance(rule, str):
        if rule not in PM_RULE_NAMES:
            raise ValueError(f"未知的保养规则名 {rule!r}；可选={PM_RULE_NAMES}")
        return rule
    idx = int(rule)
    if not (0 <= idx < N_PM_RULES):
        raise ValueError(f"保养规则下标 {idx} 越界（合法 0..{N_PM_RULES - 1}）")
    return PM_RULE_NAMES[idx]


def pm_rule_policy(rule: int | str, threshold_min: float
                   ) -> Callable[[Snapshot, int, tuple[int, ...]], int]:
    """第 `rule` 条保养规则 → **`policy_m` 回调**（契约见 `SimWorld.run_gated`）。

    `threshold_min`：**役龄阈值的绝对值** [min 主轴工时]。
    ⚠️ 规则层**不认识 `pm_interval`** —— 比例 `A ∈ (0,1]` 到绝对值的换算由调用方做
    （`threshold = A · cfg.pm_interval`）。这样规则层只依赖 `Snapshot`，不依赖 `SimConfig`。

    | 规则 | 判据（在触发点，且**未逾期**） |
    |---|---|
    | `AGE` | `役龄 ≥ threshold` |
    | `QUEUE_EMPTY` | 同 `AGE`，**且**该机台输入队列空（现在停机不堵活） |
    | `OLDEST` | 同 `AGE`，**且**本机役龄**不小于**其它任何机台（一次只保最旧的） |

    ⚠️ **三条规则的"超参数"是同一个阈值**（设计 §4.5）—— 区别在**额外判据**，不在参数。

    ⚠️ **逾期不归本函数管**：`_pm_after_op` 在 `pm_clock >= pm_interval` 时**直接强制**、
       **不调用回调** ⟹ 本函数只会被"未逾期"的窗口调到。
    """
    from ..env.des import PM_CAND_DEFER, PM_CAND_NOW
    name = _resolve_pm_rule(rule)
    thr = float(threshold_min)
    if thr < 0.0:
        raise ValueError(f"保养阈值须 >= 0，收到 {threshold_min}")

    def policy(snap: Snapshot, mach: int, cands) -> int:
        ms = snap.machines[mach]
        if ms.pm_used_min < thr:
            return PM_CAND_DEFER
        if name == "QUEUE_EMPTY":
            if ms.in_q_len > 0:
                return PM_CAND_DEFER
        elif name == "OLDEST":
            # **不小于**其它任何机台 ⟹ 平局时都保（语义确定，不引随机）
            if any(other.pm_used_min > ms.pm_used_min for other in snap.machines):
                return PM_CAND_DEFER
        return PM_CAND_NOW

    return policy


def la_rule_policy(rule: int | str,
                   lam: tuple[float, ...] = (0.0,)
                   ) -> Callable[[LAView, Sequence[int]], int]:
    """第 `rule` 条 LA 规则 → **`policy_l` 回调**：`policy_l(view, cand_v) -> 车号`。

    契约与 `policy_pa` 对称：环境装好 `LAView`（纯数据，含 `ready_at` 与行程时长口径），
    规则只读它——**规则层不认识 SimPy，也不认识布局**。
    """
    name = LA_RULE_NAMES[_resolve_la_rule(rule)]

    def policy(view: LAView, cand_v: Sequence[int]) -> int:
        if not cand_v:
            raise ValueError("LA 候选集为空——派车决策点不该出现（全车队恒可候选）")
        # ⚠️ **平局处理**（2026-10-15 修）：此前是
        #     `min(cand_v, key=lambda a: (la_score(...), int(a)))`
        #     —— 平局按**车号升序**，而**车号是任意标签**（车队完全同构、全部在 L/U 起步）
        #     ⟹ 制造任意偏置。实测 t=0 的 16 次投放**全给 V0**、`MINCOST` 全程 76.4% 有平局、
        #     MS 差 **16%**。见 `rule_views.LAView.tie_rng` 与 §4.28。
        # ⚠️ **`view.tie_rng is None` ⟹ 旧口径（车号最小）** ⟹ v3 主线逐位不变。
        scored = [(la_score(name, view, int(a), lam), int(a)) for a in cand_v]
        best = min(s for s, _ in scored)
        tied = [a for s, a in scored if s == best]
        if len(tied) == 1:
            return int(tied[0])
        rng = getattr(view, "tie_rng", None)
        if rng is None:
            return int(min(tied))
        return int(tied[int(rng.integers(0, len(tied)))])

    return policy
