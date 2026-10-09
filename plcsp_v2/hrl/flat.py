# -*- coding: utf-8 -*-
"""**扁平链**（Q1，设计 §12.2）—— 单层组 · 同步更新；两臂**只差优势从哪来**。

**问题**：*Learning Without Critics?*（arXiv:2511.03527）称
*"learned critics remain essential for long-horizon tasks"* —— **而调度是长时程。**
**⟹ 拿我们自己这套策略，只换"优势怎么算"这一个变量，其余全同。**

| 臂 | 优势 | critic | 更新 |
|---|---|---|---|
| **A** | `R − V(s_t)`（逐决策） | **有**（`HierarchicalPolicy(with_value=True)`） | 同步 |
| **B** | `(R − μ)/σ`（**整条链一个值**） | 无 | 同步 |

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

import numpy as np
import torch

from ..env.reward import objective_vector, scalar_reward
from .agents import HierarchicalPolicy
from .rollout import NetPolicy
from .train import Spec, base_chain, batched_terms, group_advantage

ALL_KINDS: tuple[str, ...] = ("MA", "PA", "LA", "C", "M")
"""一条链上的**全部**决策种类 —— 扁平链不分层 ⟹ 一个组覆盖它们全部。"""

ARMS: tuple[str, ...] = ("A", "B")

CV_DEFAULT = 0.5
"""critic 损失系数（仅臂 A）：`loss = 策略损失 + cv × critic 损失`。"""


def flat_rollout(net: HierarchicalPolicy, spec: Spec, *, n: int, seed: int,
                 dt_events: int) -> list[tuple[float, NetPolicy, tuple]]:
    """采 `n` 条**完整链**（全层一起采、不冻结、不重放）→ `[(R, 链, 目标三元组)]`。

    ⚠️ 复用 `train.base_chain` —— 它本来就是"一条完整链、全层 live"。
    **扁平链不是另一套策略，是同一套策略的另一种分组方式。**
    """
    out: list[tuple[float, NetPolicy, tuple]] = []
    for i in range(n):
        p = base_chain(net, spec, greedy=False, seed=seed * 1000 + i, dt_events=dt_events)
        f = objective_vector(p.metrics)                      # type: ignore[attr-defined]
        out.append((scalar_reward(f, spec.w), p, f))
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
              cv: float = CV_DEFAULT) -> dict:
    """**一步扁平组更新**（臂 A 或 B）。返回读数 + 梯度分布。

    两臂共用：采链 → 算优势 → `−Σ Â·logp` → 反传 → 走一步。
    **只有优势那一行不同**（见模块 docstring 的口径 ②）。
    """
    if arm not in ARMS:
        raise ValueError(f"arm={arm!r} 未知：只接受 {ARMS}")
    # ⚠️ **先查后跑**：critic 头缺了要**当场**报错，不能先花几分钟仿真再炸（快速失败纪律）。
    if arm == "A" and net.value_head is None:
        raise ValueError("臂 A 需要 critic 头 —— 构造时须给 `HierarchicalPolicy(with_value=True)`")
    if optim is None:
        optim = torch.optim.Adam(net.parameters(), lr=lr)
    runs = flat_rollout(net, spec, n=n, seed=seed, dt_events=dt_events)
    z = group_advantage([r for r, _, _ in runs])

    optim.zero_grad(set_to_none=True)
    pg = torch.zeros((), requires_grad=True)
    crit = torch.zeros((), requires_grad=True)
    rows = []
    for i, (R, p, f) in enumerate(runs):
        # 一次前向同时拿 logp 与 V（臂 A）—— 见 `train.batched_terms`
        lp, v = batched_terms(net, p.rec.decisions, ALL_KINDS, want_value=(arm == "A"))
        if arm == "A":
            adv = R - v.detach()                             # 逐决策：R − V(s_t)
            crit = crit + (v - R).pow(2).mean()
        else:
            adv = torch.full_like(lp, float(z[i]))           # 整条链**一个**优势
        pg = pg - (adv * lp).sum()
        rows.append({"R": float(R), "n_dec": len(p.rec.decisions),
                     "makespan": float(f[0]), "energy": float(f[1]), "twt": float(f[2])})
    loss = pg + (cv * crit if arm == "A" else 0.0)
    loss.backward()
    gn = grad_norm(net)
    optim.step()
    return {"arm": arm, "loss": float(loss.detach()), "policy_loss": float(pg.detach()),
            "critic_loss": float(crit.detach()), "adv_std": float(np.std(z)),
            "r_mean": float(np.mean([r for r, _, _ in runs])),
            "runs": rows, "grad_norm": gn}
