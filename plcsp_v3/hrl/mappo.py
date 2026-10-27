# -*- coding: utf-8 -*-
"""**MAPPO 基线**（CTDE · 四个 agent · 同一个动作空间）。

## 来源

Yu, Velu, Vinitsky, Gao, Wang, Bayen, Wu. *The Surprising Effectiveness of PPO in
Cooperative Multi-Agent Games.* **NeurIPS 2021 Datasets and Benchmarks Track**.
（MAPPO = Multi-Agent PPO；CTDE = centralized training, decentralized execution。）

## 与 v3 主线的唯一差异

| 项 | v3 主线 | 本基线 |
|---|---|---|
| 优势 | 组内 z 化（A 相）/ 组内 z 化（B 相） | **集中式 critic** `R − V(s)` |
| 更新 | 两阶段交替（先 MA 后下层） | **联合 PPO**（四类决策一个 batch） |
| 目标 | `−Σ Â·logp` | **PPO clip** + value loss + entropy |
| 抽样 | 同一步内共用场景 | 同左（`flat_rollout` 的 `seed_chain`） |

**动作空间、状态特征、编码器、四个头、奖励口径全部相同** —— 这样才能把差异
归因到训练算法，而不是另一个策略网络。

## 四个 agent 的映射（事件驱动）

| agent | 决策点 | 局部观测 | 动作 |
|---|---|---|---|
| **MA** | `policy_ma` | M/B/V/G 四段池化 | 三个规则号 |
| **PA** | `policy_pa` | M ∪ B + 候选特征 | 候选下标 |
| **LA** | `policy_l` | V + 任务上下文 | 候选下标 |
| **RA** | `policy_m` | M + 候选特征 | `{NOW, DEFER}` |

**执行时只用各自的局部观测**（`NetPolicy` 已经这样接线）；
**critic 只在训练时看全局池化向量**（`HierarchicalPolicy.value_head`）。

## 回报与优势

奖励是**终局的**（一条链跑完才有一个标量 `R`）⟹ `γ=1, λ=1` 的 GAE 退化成

```
Â_t = R − V(s_t)
```

这是标准 MAPPO 在终端奖励下的形状，不是"省掉了 GAE"。
优势在**整批决策**上做一次 `(Â − μ)/σ`（标准 PPO 口径）。

## 已知口径

- PPO 的 `ratio` 用**候选集内**的 logp（与 `Decision.logp` 同源）。
- `K_eff == 1` 的强制决策 `logp = 0`，不进入 ratio 的有效梯度（熵也记 0）。
- 并行档（`ChainPool`）每步重发权重；与顺序档**不是逐位相同**（浮点归约次序）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from .agents import HierarchicalPolicy
from .flat import ALL_KINDS, flat_rollout
from .train import Spec, batched_terms_entropy

if TYPE_CHECKING:
    from .pool import ChainPool

EPS = 1e-6


@dataclass(frozen=True)
class MAPPOConfig:
    """MAPPO 的优化超参。默认值照 PPO 的常用档（不是为本问题精调过的）。"""

    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 128
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


def _update_minibatch(net: HierarchicalPolicy, optim: torch.optim.Optimizer,
                      decisions, old_lp: torch.Tensor, ret: torch.Tensor,
                      adv: torch.Tensor, idx: np.ndarray,
                      cfg: MAPPOConfig) -> dict:
    """一个 minibatch 的 PPO 更新 → 读数。**所有四个头 + encoder + critic 一起更新。**"""
    dev = old_lp.device
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=dev)
    sub = [decisions[int(i)] for i in idx]
    lp, v, ent = batched_terms_entropy(net, sub, ALL_KINDS)
    ratio = torch.exp(lp - old_lp[idx_t])
    a = adv[idx_t]
    surr = torch.min(ratio * a,
                     torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * a)
    pg = -surr.mean()
    vf = 0.5 * (v - ret[idx_t]).pow(2).mean()
    loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent.mean()
    optim.zero_grad(set_to_none=True)
    loss.backward()
    # ⚠️ **先量梯度再裁剪**：裁剪后的范数恒 ≤ 阈值，看不出有没有爆炸。
    gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm))
    optim.step()
    with torch.no_grad():
        approx_kl = float((old_lp[idx_t] - lp).mean())
        clip_frac = float(((ratio - 1.0).abs() > cfg.clip).float().mean())
    return {"n": len(sub), "loss": float(loss.detach()),
            "policy_loss": float(pg.detach()), "value_loss": float(vf.detach()),
            "entropy": float(ent.mean().detach()), "approx_kl": approx_kl,
            "clip_frac": clip_frac, "grad_norm": gn}


def mappo_step(net: HierarchicalPolicy, spec: Spec,
               optim: torch.optim.Optimizer | None = None, *,
               n: int = 8, seed: int = 0, dt_events: int = 20,
               lr: float = 1e-4, pool: "ChainPool | None" = None,
               chain_seed: int = 0, cfg: MAPPOConfig = MAPPOConfig()) -> dict:
    """采 `n` 条完整链 → 集中式 critic + PPO 更新 → 读数。

    ⚠️ `net` 必须是 `HierarchicalPolicy(with_value=True)`；缺 critic 头**当场报错**，
    不先花几分钟跑仿真。
    """
    if net.value_head is None:
        raise ValueError("MAPPO 需要集中式 critic —— 构造时须给 "
                         "`HierarchicalPolicy(with_value=True)`")
    if optim is None:
        optim = torch.optim.Adam(net.parameters(), lr=lr)
    runs = flat_rollout(net, spec, n=n, seed=seed, dt_events=dt_events,
                        pool=pool, seed_chain=chain_seed)
    decisions = [d for ch in runs for d in ch.decisions]
    if not decisions:
        raise ValueError("一条链都没有决策 —— 仿真没有接上策略")
    dev = next(net.parameters()).device
    ret = torch.as_tensor([ch.R for ch in runs for _ in ch.decisions],
                          dtype=torch.float32, device=dev)
    with torch.no_grad():
        old_lp, old_v, _ = batched_terms_entropy(net, decisions, ALL_KINDS)
    if old_v is None:
        raise ValueError("critic 没有输出 V(s)")
    old_lp = old_lp.detach()
    adv = ret - old_v.detach()
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + EPS)

    rng = np.random.default_rng(int(seed))
    order = np.arange(len(decisions))
    stats: list[dict] = []
    for _epoch in range(int(cfg.epochs)):
        rng.shuffle(order)
        for start in range(0, len(order), int(cfg.minibatch)):
            stats.append(_update_minibatch(
                net, optim, decisions, old_lp, ret, adv,
                order[start:start + int(cfg.minibatch)], cfg))
    total = sum(s["n"] for s in stats)

    def avg(key: str) -> float:
        return float(sum(s[key] * s["n"] for s in stats) / total)

    rs = np.asarray([ch.R for ch in runs], dtype=np.float64)
    return {"loss": avg("loss"), "policy_loss": avg("policy_loss"),
            "value_loss": avg("value_loss"), "entropy": avg("entropy"),
            "approx_kl": avg("approx_kl"), "clip_frac": avg("clip_frac"),
            "grad_norm": max(s["grad_norm"] for s in stats),
            "r_mean": float(rs.mean()),
            "r_std": float(rs.std(ddof=1)) if len(rs) > 1 else 0.0,
            "n_chains": len(runs), "n_decisions": len(decisions)}
