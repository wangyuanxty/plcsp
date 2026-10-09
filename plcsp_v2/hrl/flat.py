# -*- coding: utf-8 -*-
"""**扁平链**（Q1，设计 §12.2）—— 单层组 · 同步更新；两臂**只差优势从哪来**。

**问题**：*Learning Without Critics?*（arXiv:2511.03527）称
*"learned critics remain essential for long-horizon tasks"* —— **而调度是长时程。**
**⟹ 拿我们自己这套策略，只换"优势怎么算"这一个变量，其余全同。**

| 臂 | 优势 | critic | 归一化 |
|---|---|---|---|
| **A** | `R − V(s_t)`（逐决策） | **有** | **无**（原始尺度） |
| **A2** | `R − V(s_t)`（逐决策） | **有** | **有**（标准 PPO） |
| **B** | `(R − μ)/σ`（**整条链一个值**） | 无 | 有（自带） |

**⚠️ 为什么必须有 A2**：A 的优势尺度与 B 差约 **50 倍**（`grad_norm` 实测 10527 vs 199，§98.3）。
标准 PPO 会给优势做 `(A − μ)/σ`；**A 故意没做**（做了就不是"只差优势"了）。
但**一个没做归一化的 critic 臂发散，说明不了 critic 没用** —— 审稿人会写
*"This is not evidence that critics are unnecessary — it is evidence that the baseline was not tuned."*
**⟹ A2 才是那个"公平的 critic"。没有它，Q1 的结论不能进论文**（§101.4）。

**A2 的归一化口径**：`Â = (A − μ_A)/(σ_A + ε)`，**在整组更新的全部决策上算** ——
标准 PPO 的 minibatch 归一化在这里对应的就是"这一组"。

**"扁平"= 不分层**：一条链的**全部**决策（MA + PA + LA + C + M）算**一个**组，
不像两层那样按层切成两个组、交替更新。这是 v1 `joint_chain_step` 的口径。

══ 三条必须写明的口径 ══

**① 臂 A 的 GAE 为什么退化成 `R − V(s_t)`。**
本问题的奖励是**终端的**（一条链跑完才有一个标量 `R`，中间没有奖励）⟹ 取 `λ=1, γ=1` 时
`A_t = Σ_k δ_{t+k}` 望远镜求和 = `R − V(s_t)`。**设计 §12.2 写的就是 "GAE(回报, V(s))"。**
**写在这里，免得被读成"偷懒只做了个 MC 基线"。**

**② 两臂的更新规则逐字相同**：`loss = −Σ_t Â_t·logp_t`，**无 ratio、无 clip**。
设计 §12.2 的表只把"优势"列为差异；再引入 clip 就是两处差异，**读不出是哪一个起的作用**。
写了 ratio clipping 的那套叫 PPO，**本步不是**。

**③ 两臂的链是同一批种子**（`seed*1000+i`）⟹ 采样的**随机流对齐**，不是各跑各的。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ..env.reward import objective_vector, scalar_reward
from .agents import HierarchicalPolicy
from .pool import Chain
from .train import EPS, Spec, base_chain, batched_terms, group_advantage

if TYPE_CHECKING:
    from .pool import ChainPool

ALL_KINDS: tuple[str, ...] = ("MA", "PA", "LA", "C", "M")
"""一条链上的**全部**决策种类 —— 扁平链不分层 ⟹ 一个组覆盖它们全部。"""

ARMS: tuple[str, ...] = ("A", "A2", "B")
"""`A` = critic 原始优势 · `A2` = **critic + 优势归一化**（标准 PPO 口径）· `B` = 组内 z 化。"""

CRITIC_ARMS: tuple[str, ...] = ("A", "A2")
"""要 critic 头的臂。"""

NORM_ARMS: tuple[str, ...] = ("A2",)
"""要**优势归一化**的臂 —— 见模块 docstring「A2 的归一化口径」。"""

CV_DEFAULT = 0.5
"""critic 损失系数（仅臂 A）：`loss = 策略损失 + cv × critic 损失`。"""


def flat_rollout(net: HierarchicalPolicy, spec: Spec, *, n: int, seed: int,
                 dt_events: int, greedy: bool = False,
                 pool: "ChainPool | None" = None) -> list[Chain]:
    """采 `n` 条**完整链**（全层一起采、不冻结、不重放）→ `Chain` 列表。

    ⚠️ 复用 `train.base_chain` —— 它本来就是"一条完整链、全层 live"。
    **扁平链不是另一套策略，是同一套策略的另一种分组方式。**

    `greedy=True` 用于**评测**（确定性、可复现）；训练一律 `False`（要探索）。
    `pool`（`hrl/pool.py`）：给了就**多进程并行**跑这 `n` 条链（每进程 1 线程）。
    ⚠️ **并行档只支持 `greedy=False` 且 CPU** —— 评测要确定性，且一个 GPU 开多上下文只会互抢。
    """
    if pool is not None:
        if greedy:
            raise ValueError("并行档不支持 `greedy=True`（评测要确定性、且要 GPU 单档）")
        return pool.rollout(n, seed)
    out: list[Chain] = []
    for i in range(n):
        p = base_chain(net, spec, greedy=greedy, seed=seed * 1000 + i, dt_events=dt_events)
        f = objective_vector(p.metrics)                      # type: ignore[attr-defined]
        out.append(Chain(scalar_reward(f, spec.w), tuple(f), p.rec.decisions))
    return out


def grad_norm(net: HierarchicalPolicy) -> dict:
    """梯度落在**哪一块**：五个打分头 / critic 头 / 编码器（共享主干，两边都浇）。"""
    def acc(mods) -> float:
        s = 0.0
        for mod in mods:
            for p in mod.parameters():
                if p.grad is not None:
                    s += float(p.grad.pow(2).sum())
        return float(np.sqrt(s))
    heads = (net.ma, net.pa, net.la, net.ra_charge, net.ra_maint)
    crit = () if net.value_head is None else (net.value_head,)
    return {"heads": acc(heads), "critic": acc(crit), "encoder": acc((net.enc,))}


def flat_step(net: HierarchicalPolicy, spec: Spec,
              optim: torch.optim.Optimizer | None = None, *, arm: str = "B",
              n: int = 8, seed: int = 0, dt_events: int = 20, lr: float = 1e-4,
              cv: float = CV_DEFAULT, pool: "ChainPool | None" = None) -> dict:
    """**一步扁平组更新**（臂 A 或 B）。返回读数 + 梯度分布。

    两臂共用：采链 → 算优势 → `−Σ Â·logp` → 反传 → 走一步。
    **只有优势那一行不同**（见模块 docstring 的口径 ②）。

    `pool`：给了就用多进程并行采链（见 `hrl/pool.py`）。
    ⚠️ **两臂必须同档**（都带池或都不带）—— 并行档每进程 1 线程，浮点归约次序不同。
    """
    if arm not in ARMS:
        raise ValueError(f"arm={arm!r} 未知：只接受 {ARMS}")
    # ⚠️ **先查后跑**：critic 头缺了要**当场**报错，不能先花几分钟仿真再炸（快速失败纪律）。
    if arm in CRITIC_ARMS and net.value_head is None:
        raise ValueError(f"臂 {arm} 需要 critic 头 —— 构造时须给 `HierarchicalPolicy(with_value=True)`")
    if optim is None:
        optim = torch.optim.Adam(net.parameters(), lr=lr)
    runs = flat_rollout(net, spec, n=n, seed=seed, dt_events=dt_events, pool=pool)
    z = group_advantage([ch.R for ch in runs])

    optim.zero_grad(set_to_none=True)
    crit = torch.zeros((), requires_grad=True)
    rows = []
    terms: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i, ch in enumerate(runs):
        # 一次前向同时拿 logp 与 V（critic 臂）—— 见 `train.batched_terms`
        lp, v = batched_terms(net, ch.decisions, ALL_KINDS,
                              want_value=(arm in CRITIC_ARMS))
        if arm in CRITIC_ARMS:
            adv = ch.R - v.detach()                          # 逐决策：R − V(s_t)
            crit = crit + (v - ch.R).pow(2).mean()
        else:
            adv = torch.full_like(lp, float(z[i]))           # 整条链**一个**优势
        terms.append((adv, lp))
        rows.append({"R": float(ch.R), "n_dec": len(ch.decisions),
                     "makespan": float(ch.f[0]), "energy": float(ch.f[1]),
                     "twt": float(ch.f[2])})

    if arm in NORM_ARMS:
        # ⚠️ **整组一次归一化**（标准 PPO 的 minibatch 口径）。逐链各归一会把
        # "哪条链更好"这条信息（= 跨链比较）整条抹掉 —— 那正是组内相对法赖以工作的东西。
        all_adv = torch.cat([a for a, _ in terms])
        mu, sd = all_adv.mean(), all_adv.std()
        terms = [((a - mu) / (sd + EPS), lp) for a, lp in terms]

    pg = -sum((a * lp).sum() for a, lp in terms)
    loss = pg + (cv * crit if arm in CRITIC_ARMS else 0.0)
    loss.backward()
    gn = grad_norm(net)
    optim.step()
    return {"arm": arm, "loss": float(loss.detach()), "policy_loss": float(pg.detach()),
            "critic_loss": float(crit.detach()), "adv_std": float(np.std(z)),
            "r_mean": float(np.mean([ch.R for ch in runs])),
            "runs": rows, "grad_norm": gn}
