# -*- coding: utf-8 -*-
"""**目标条件化两层策略**的网络（设计 §1–§3，步 4）。

```
实例 + 布局 → token (N, F_MAX) ──► 共享 Transformer 编码器（`nn.encoder.LayoutEncoder`）
                                        │ Z_t：逐 token 隐向量
                    ┌───────────────────┴────────────────────┐
              全局池化                                    逐段硬掩码
                    ▼                                        ▼
            【MA 顶层】π_MA(Z)                        【下层三个】
              g_t ∈ {4 档}³（3 维分档）              PA(12) · LA(3) · RA 充电/保养
                    │                                        │
                    └────────► B·g 线性偏置 ◄────────────────┘
```

**四个 `B` 矩阵**（设计 §3）：`logits(动作) = base(Z) + B · g`。
`B` 的第 j 列 = "第 j 个目标把哪些动作往上抬" ⟹ **可解释**。

⚠️ **`B` 零初始化**（不是拍脑袋）——两面的代价都写下来：

- **好**：起始时 `g` **不影响任何动作** ⟹ 初始策略 = 一个不带 `g` 的普通策略，与
  "关掉顶层"的消融**同分布**；`B` 后来学出来的东西就是"`g` 到底改变了什么"，
  可解释性从第一步就是干净的；
- **代价**：**起步时扫 `g` 看不到任何变化**（有测试钉住，免得被误当成坏了）。

══ 掩码（设计 §10⑥：共享编码器 + attention 硬掩码）══

| Agent | 掩码（读哪些 token 段） | 与设计的差异 |
|---|---|---|
| **MA** | 全局（编码器给的均值池化） | — |
| **PA** | **M ∪ B**（机台 + 作业） | ✓ 与设计一致 |
| **LA** | **V**（AGV） | ⚠️ 设计还要"任务 token"——**v2 没有任务 token**（① 去后无 Z 段、也无任务段）。如实声明 |
| **RA·充电** | **V**（AGV） | ⚠️ 设计还要"桩 token"——**桩是动作不是 token**（见 `des.CHARGE_CAND_SKIP`）。如实声明 |
| **RA·保养** | **M**（机台） | — |

**"分权在决策层，不在表征层"**（设计 §10⑥ 要求论文写明）：四个 agent 共用**同一个**
编码器、同一份 `Z_t`；差别只在**读哪一段**与**自己的打分头**。
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from ..nn.encoder import LayoutEncoder
from ..nn.features import F_MAX
from .rules import N_LA_RULES, N_PA_RULES

# `g` 的表示（设计 §10④）：**每维 4 档 ⟹ 4³ = 64 个组合**。分档 logp 简单、好调试。
G_DIM = 3                # [θ_wip, θ_otr, θ_agv]
G_BINS = 4
G_NAMES = ("theta_wip", "theta_otr", "theta_agv")


def g_to_vec(bins: torch.Tensor, n_bins: int = G_BINS) -> torch.Tensor:
    """分档下标 `(B, 3)` → **连续目标向量** `(B, 3) ∈ (0, 1)`（= 档心 `(i+0.5)/n`）。

    ⚠️ **全部是比例 ⟹ 尺度无关 ⟹ 换实例定义不变**（设计 §3）。取档心而不是 `i/(n−1)`
    是为了让两端也落在开区间内——`B·g` 的偏置不该在边界上退化成 `0` 或 `1`。
    """
    return (bins.float() + 0.5) / float(n_bins)


class PolicyOut(NamedTuple):
    """一次前向的全部输出。**`g=None`（顶层不参与）时四个 `B` 项不生效**。"""

    g_logits: torch.Tensor          # (B, 3, G_BINS) —— 顶层的**分档** logits
    g_bins: torch.Tensor            # (B, 3) —— 采到的档（argmax / 采样）
    g: torch.Tensor                 # (B, 3) —— 连续目标向量（`g_to_vec(g_bins)`）
    pa: torch.Tensor                # (B, 12) —— PA 的 12 条复合规则
    la: torch.Tensor                # (B, 3)
    ra_charge: torch.Tensor         # (B, 1 + n_chargers) —— 0 = 不去充
    ra_maint: torch.Tensor          # (B, 2) —— 0 = 生产、1 = 保养


class _MaskedScoreHead(nn.Module):
    """**读自己那一段**的 token 求上下文 → 打分；再叠一条 `B·g` 的线性偏置。

    `rows` 是**段内行下标**（`(lo, hi)` 半开），由调用方按 `seg` 给——**不在 forward 里
    解析 `seg`**，因为同一份 `Z_t` 要被四个头读四次，解析只该做一次。
    """

    def __init__(self, d_model: int, n_out: int, hidden: int | None = None):
        super().__init__()
        h = hidden or max(32, d_model // 2)
        self.base = nn.Sequential(nn.Linear(d_model, h), nn.GELU(), nn.Linear(h, n_out))
        # ⚠️ **零初始化**（见模块 docstring）：起始 `g` 不影响动作
        self.b = nn.Parameter(torch.zeros(n_out, G_DIM))

    def forward(self, z: torch.Tensor, rows: tuple[int, int],
                g: torch.Tensor | None) -> torch.Tensor:
        lo, hi = rows
        if hi <= lo:
            raise ValueError(f"掩码段为空：[{lo}, {hi})——该 agent 没有可读的 token")
        ctx = z[:, lo:hi, :].mean(dim=1)            # (B, d) —— 段内均值池化
        logits = self.base(ctx)                     # (B, n_out)
        if g is not None:
            logits = logits + g @ self.b.t()        # `B·g`：第 j 列 = 目标 j 抬高哪些动作
        return logits


class MAHead(nn.Module):
    """顶层：**全局池化** → 3 维分档 logits。"""

    def __init__(self, d_model: int, n_bins: int = G_BINS):
        super().__init__()
        self.n_bins = int(n_bins)
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                 nn.Linear(d_model, G_DIM * n_bins))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled).view(-1, G_DIM, self.n_bins)


class HierarchicalPolicy(nn.Module):
    """编码器 + 四个 agent 的头（**一个 encode、四次读**）。

    `n_chargers` 决定充电头的宽度（`1 + n_chargers`）——它随布局变，故**构造时必须给**。
    """

    def __init__(self, n_chargers: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, feat_dim: int = F_MAX, n_bins: int = G_BINS):
        super().__init__()
        self.enc = LayoutEncoder(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                                 feat_dim=feat_dim)
        self.n_bins = int(n_bins)
        self.ma = MAHead(d_model, n_bins)
        self.pa = _MaskedScoreHead(d_model, N_PA_RULES)
        self.la = _MaskedScoreHead(d_model, N_LA_RULES)
        self.ra_charge = _MaskedScoreHead(d_model, 1 + int(n_chargers))
        self.ra_maint = _MaskedScoreHead(d_model, 2)

    # ── 掩码：`seg = (n_m, n_jobs, n_agv, 1)` → 各 agent 读哪一段 ──
    @staticmethod
    def masks(seg: tuple[int, ...]) -> dict[str, tuple[int, int]]:
        """逐 agent 的 token 行区间（半开）。**唯一的掩码定义处**——别处不得再算一遍。"""
        n_m, n_jobs, n_agv = int(seg[0]), int(seg[1]), int(seg[2])
        return {
            "M": (0, n_m),
            "B": (n_m, n_m + n_jobs),
            "V": (n_m + n_jobs, n_m + n_jobs + n_agv),
            "PA": (0, n_m + n_jobs),                              # M ∪ B
            "LA": (n_m + n_jobs, n_m + n_jobs + n_agv),           # V
            "RA_charge": (n_m + n_jobs, n_m + n_jobs + n_agv),    # V
            "RA_maint": (0, n_m),                                 # M
        }

    def forward(self, tok: torch.Tensor, seg: tuple[int, ...],
                g_bins: torch.Tensor | None = None, *,
                bias: torch.Tensor | None = None,
                hard: bool = True) -> PolicyOut:
        """`tok (B,N,F)` + `seg` → `PolicyOut`。

        `g_bins` 给定 ⟹ **下层条件在该目标上**（阶段 B / 执行）；`None` ⟹ 顶层现采一个
        （阶段 A / 训练时）。`hard=True`（默认）取 argmax，否则按 logits 采样。
        """
        z, pooled = self.enc(tok, seg, bias)          # (B,N,d), (B,d)
        g_logits = self.ma(pooled)                    # (B,3,n_bins)
        if g_bins is None:
            gi = (g_logits.argmax(dim=-1) if hard
                  else torch.distributions.Categorical(logits=g_logits).sample())
            g_bins = gi
        g = g_to_vec(g_bins, self.n_bins)             # (B,3)
        mk = self.masks(seg)
        return PolicyOut(
            g_logits=g_logits, g_bins=g_bins, g=g,
            pa=self.pa(z, mk["PA"], g),
            la=self.la(z, mk["LA"], g),
            ra_charge=self.ra_charge(z, mk["RA_charge"], g),
            ra_maint=self.ra_maint(z, mk["RA_maint"], g),
        )

    # ── 偏置矩阵（报告里要逐列读"目标 j 抬高了哪些动作"）──
    def b_matrices(self) -> dict[str, torch.Tensor]:
        """四个 `B` 矩阵的**当前值**（只读副本）。报表用：`B_PA[i, j]` = 目标 j 对规则 i 的偏置。"""
        return {"PA": self.pa.b.detach().clone(),
                "LA": self.la.b.detach().clone(),
                "RA_charge": self.ra_charge.b.detach().clone(),
                "RA_maint": self.ra_maint.b.detach().clone()}
