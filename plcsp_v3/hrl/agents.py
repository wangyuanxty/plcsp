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
| **LA** | **V**（AGV） | **任务身份**（逐决策上下文，见 `LA_CTX_DIM`）—— 设计要的"任务 token"**表示选错了**：任务随决策变，进不了状态张量 |
| **RA·充电** | **V**（AGV） ∪ **C**（桩） | ✓ **2026-10-09 补齐 C 段**（此前桩根本没进 token 序列） |
| **RA·保养** | **M**（机台） | — |

**"分权在决策层，不在表征层"**（设计 §10⑥ 要求论文写明）：四个 agent 共用**同一个**
编码器、同一份 `Z_t`；差别只在**读哪一段**与**自己的打分头**。

⚠️ **2026-10-09 更正**：此前这里把 LA 与 RA·充电都记成"与设计不一致"。核下来是**两回事**：
- **RA·充电**是**真缺口**（桩的占用就在快照里，只是没 token 化）⟹ **已补 C 段**；
- **LA 的"任务 token"**是**表示选错**：任务是**逐决策上下文**，特征随决策变，
  放进状态张量会让张量依赖决策 ⟹ 走**打分层输入**（v1 的 `task_feat` 就是这个做法）。
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from ..nn.encoder import LayoutEncoder
from ..nn.features import F_MAX
from .candidates import LA_CAND_DIM, PA_CAND_DIM, RA_FEAT_DIM
from .rules import (N_LA_RULES, N_PA_RULES, N_PM_RULES, PA_MACHINE_RULES)

# `g` 的表示（设计 §10④）：**每维 4 档 ⟹ 4³ = 64 个组合**。分档 logp 简单、好调试。
G_DIM = 3                # [θ_wip, θ_otr, θ_agv]
G_BINS = 4
G_NAMES = ("theta_wip", "theta_otr", "theta_agv")

LA_CTX_DIM = 4
"""LA 的**逐决策任务上下文**宽度：`(起点节点, 终点节点, 可运输时刻 − now, 工序号)`。

⚠️ **这不是 token**（设计 §10 的掩码表写成"任务 token"是**表示选错**）：任务的属性
**随决策变**，放进状态张量会让张量依赖决策。**正确做法是拼进打分层**——
`2026-10-02` 的 v1 `algo/group_rel.task_feat(...)` 就是这个口径，此处沿用。"""


def g_to_vec(bins: torch.Tensor, n_bins: int = G_BINS) -> torch.Tensor:
    """分档下标 `(B, 3)` → **连续目标向量** `(B, 3) ∈ (0, 1)`（= 档心 `(i+0.5)/n`）。

    ⚠️ **全部是比例 ⟹ 尺度无关 ⟹ 换实例定义不变**（设计 §3）。取档心而不是 `i/(n−1)`
    是为了让两端也落在开区间内——`B·g` 的偏置不该在边界上退化成 `0` 或 `1`。
    """
    return (bins.float() + 0.5) / float(n_bins)


class PolicyOut(NamedTuple):
    """一次前向的全部输出（**v3 top-K 架构**）—— 见 `docs/v3/topk-rerank-spec.md`。"""

    rule_logits: dict[str, torch.Tensor]
    """`{"PA": (B,12), "LA": (B,3), "RA": (B,3)}` —— 三个**独立**分类头的 logits。

    ⚠️ **三个独立**，不是一个联合分类 —— 与 v2 的 `MAHead`（3 个 4 档分档）同构。
    """
    rule_sel: dict[str, torch.Tensor]
    """`(B,)` —— 各头选中的规则**下标**（`hard=True` 时 argmax，否则采样）。"""
    z: torch.Tensor
    """`(B, N, d_model)` —— 编码器的逐 token 隐向量。打分头按候选行号 gather 它。"""
    value: torch.Tensor | None = None
    """(B, 1) —— **critic 的 `V(s)`**。仅在 `HierarchicalPolicy(with_value=True)` 时非空。"""


def split_pa_rule(idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """PA 的**复合规则下标** → `(作业下标, 机台下标)`。

    ⚠️ **必须与 `rules.apply_pa_rule` 的拆法逐字同源**（`idx // len(PA_MACHINE_RULES)`
    与 `idx % len(PA_MACHINE_RULES)`）—— **两处各写一遍必然会漂**（设计 §13.3 ③）。
    """
    n_m = len(PA_MACHINE_RULES)
    return idx // n_m, idx % n_m


# ══════════ v3：上层三个分类头 + 下层每规则一个参数头 ══════════
#   设计依据：`docs/v3/design.md` §3（硬路由）· §4.2–§4.6（每条规则一个 head）。
#
#   ⚠️ **本步只加模块，不动 `PolicyOut`** —— 那个接口是「换头」与「换采样」的接缝，
#      改它会让 `rollout.py` / `train.py` 同时编译不过。**留到步 4 一起改。**


class RuleSelectHeads(nn.Module):
    """**上层（MA）**：全局池化 → **三个独立**的分类头（PA / LA / RA 各一个）。

    ⚠️ **三个独立，不是一个联合分类** —— 与 v2 的 `MAHead`（3 个 4 档分档）**同构**，
    只是每组的类数从 `4` 变成 `n_rules` 里的值。v2 的这条注释仍然成立：

    > 顶层是 3 个**独立**的分档，不是一个 64 路的类别

    ⟹ 三个头各自 softmax、各自取 argmax。**因子化采样表达不了"PA 与 LA 的规则要配套"**，
    靠下层补偿（设计 §10）。

    **⚠️ 为什么一个头 12 路而不是拆成「4 作业 × 3 机台」两个头**：设计 §2 的规格写的就是
    `12`（`PA_RULE_NAMES` 的下标）。拆开会改动作空间的语义。
    """

    def __init__(self, d_model: int, n_rules: dict[str, int], hidden: int | None = None,
                 in_dim: int | None = None):
        super().__init__()
        if not n_rules:
            raise ValueError("n_rules 为空 —— 上层的动作空间不能是空的")
        h = hidden or max(32, d_model // 2)
        self.keys = tuple(n_rules)
        # ⚠️ **2026-10-19**：MA 的输入不再是全 token 均值（`encoder.py` 的 `pooled`），
        # 而是 **M/B/V/G 四段均值拼接**（`4*d_model`）。均值池化会让不同状态给出
        # 几乎相同的向量（`encoder.py:78`、progress-log §35）⟹ MA 的输出与状态无关。
        self.in_dim = int(in_dim or d_model)
        self.trunk = nn.Sequential(nn.Linear(self.in_dim, h), nn.GELU())
        self.heads = nn.ModuleDict({k: nn.Linear(h, int(n)) for k, n in n_rules.items()})

    def forward(self, pooled: torch.Tensor,
                rules: dict[str, torch.Tensor] | None = None,
                hard: bool = True) -> dict[str, torch.Tensor]:
        """`(B, d)` → `{键: (B, n_rules[键])}` 的 logits。

        `rules` / `hard` 只为与顺序条件版共用接口；独立头忽略它们。
        """
        feat = self.trunk(pooled)
        return {k: self.heads[k](feat) for k in self.keys}


class SequentialRuleSelectHeads(nn.Module):
    """**顺序条件 MA**（PA → LA；RA 先保留但不条件）。

    与 `RuleSelectHeads` 的区别：
    - LA 头的输入 = `trunk(sel_in)` **拼接 PA 规则 embedding**；
    - RA 头保持独立（先不加 RA）。

    这样策略能表达：
    `PA=EAJ+EAM` 时 LA 选 `LATEST`；`PA=EAJ+SMPT` 时 LA 选 `MINCOST`。
    """

    def __init__(self, d_model: int, n_rules: dict[str, int], hidden: int | None = None,
                 in_dim: int | None = None, rule_emb_dim: int = 8):
        super().__init__()
        # ⚠️ 默认值写 8（= 下方 `RULE_EMB_DIM`）；顺序头定义在常量之前，不能直接引用。
        if not n_rules:
            raise ValueError("n_rules 为空 —— 上层的动作空间不能是空的")
        h = hidden or max(32, d_model // 2)
        self.in_dim = int(in_dim or d_model)
        self.trunk = nn.Sequential(nn.Linear(self.in_dim, h), nn.GELU())
        self.rule_emb_pa = nn.Embedding(int(n_rules["PA"]), int(rule_emb_dim))
        self.rule_emb_la = nn.Embedding(int(n_rules["LA"]), int(rule_emb_dim))
        self.heads = nn.ModuleDict({
            "PA": nn.Linear(h, int(n_rules["PA"])),
            "LA": nn.Linear(h + int(rule_emb_dim), int(n_rules["LA"])),
            "RA": nn.Linear(h, int(n_rules["RA"])),
        })
        self.keys = tuple(n_rules)

    def forward(self, pooled: torch.Tensor,
                rules: dict[str, torch.Tensor] | None = None,
                hard: bool = True) -> dict[str, torch.Tensor]:
        """`(B,d)` + 可选的**逐行**规则 → `{PA, LA, RA}` logits。

        `rules` 给定时，LA 的 logits **按每行自己的 PA** 条件；
        不给定时用 PA 的 argmax 作为条件（只用于诊断/初始化）。
        """
        feat = self.trunk(pooled)
        pa_logits = self.heads["PA"](feat)
        if rules is not None and "PA" in rules:
            pa = rules["PA"].to(feat.device).long()
        else:
            pa = pa_logits.argmax(dim=-1)
        la_logits = self.heads["LA"](
            torch.cat([feat, self.rule_emb_pa(pa)], dim=-1))
        return {"PA": pa_logits, "LA": la_logits,
                "RA": self.heads["RA"](feat)}


RULE_EMB_DIM = 8
"""规则身份 embedding 的宽度（PA 12 路 / LA 3 路 / RA 3 路共用这个宽度）。"""

MIN_Z_NORM_K = 3
"""候选集至少这么大才做**组内 z 化**。

`K_eff == 2` 时 z 化会把 logit 差钉成 ±1，抹掉"有多确定" ⟹ 保留原始尺度。
`K_eff == 1` 时 `std` 无定义 ⟹ 跳过。
"""


def _norm_logits(s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """组内 z 化 + 掩码 → logits。**采样与重算共用这一处**。

    `s (…, K)`、`mask (…, K)` bool。z 化只在 `mask` 内做；被掩掉的位置置 `-inf`。
    ⚠️ z 化是**单调仿射**变换 ⟹ **不改变排序**，只改训练时的采样分布。
    """
    mask = mask.bool()
    valid = mask.to(s.dtype)
    n = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (s * valid).sum(dim=-1, keepdim=True) / n
    var = (((s - mean) ** 2) * valid).sum(dim=-1, keepdim=True) / n
    # ⚠️ **`sqrt(var + eps)` 而不是 `sqrt(var) + eps`**：候选分数全相等时 `var = 0`，
    # `sqrt` 在 0 点的导数是无穷 ⟹ `0 * inf = nan`，整条反传被污染（LA 在 t=0 常见：
    # 所有车同构、特征相同、打分逐位相等）。加在**根号内**让导数有限。
    normed = (s - mean) / (torch.sqrt(var + 1e-6) + 1e-6)
    out = torch.where(n >= float(MIN_Z_NORM_K), normed, s)
    return out.masked_fill(~mask, float("-inf"))


class PairScoreHead(nn.Module):
    """PA 的打分头：两个实体向量 + 候选特征 + 规则 embedding → **每候选 1 个 logit**。"""

    def __init__(self, d_model: int, cand_dim: int, n_rules: int,
                 hidden: int = 128, rule_emb_dim: int = RULE_EMB_DIM):
        super().__init__()
        self.rule_emb = nn.Embedding(int(n_rules), int(rule_emb_dim))
        self.trunk = nn.Sequential(
            nn.Linear(2 * d_model + int(cand_dim) + int(rule_emb_dim), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor,
                feats: torch.Tensor, rule: torch.Tensor) -> torch.Tensor:
        """`(B,K,d)`, `(B,K,d)`, `(B,K,C)`, `(B,)` → `(B,K)`。"""
        b, k, _ = z_a.shape
        e = self.rule_emb(rule).unsqueeze(1).expand(b, k, -1)
        x = torch.cat([z_a, z_b, feats, e], dim=-1)
        return self.trunk(x).squeeze(-1)


class SingleScoreHead(nn.Module):
    """LA / RA 的打分头：一个实体向量（+ 可选逐决策上下文）+ 候选特征 + 规则 embedding。"""

    def __init__(self, d_model: int, cand_dim: int, n_rules: int,
                 hidden: int = 64, n_out: int = 1, ctx_dim: int = 0,
                 rule_emb_dim: int = RULE_EMB_DIM):
        super().__init__()
        self.ctx_dim = int(ctx_dim)
        self.n_out = int(n_out)
        self.rule_emb = nn.Embedding(int(n_rules), int(rule_emb_dim))
        self.trunk = nn.Sequential(
            nn.Linear(d_model + self.ctx_dim + int(cand_dim) + int(rule_emb_dim),
                      int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.n_out),
        )

    def forward(self, z_a: torch.Tensor, feats: torch.Tensor,
                rule: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        """`(B,K,d)`, `(B,K,C)`, `(B,)` → `(B,K)`（`n_out=1`）或 `(B,n_out)`（RA）。"""
        b, k, _ = z_a.shape
        parts = [z_a]
        if self.ctx_dim:
            if ctx is None:
                raise ValueError(
                    f"本头声明了 ctx_dim={self.ctx_dim} 的逐决策上下文，但没给——"
                    "静默补 0 会让『这个头没接上』变成无声的坏策略")
            parts.append(ctx.unsqueeze(1).expand(b, k, -1))
        parts.append(feats)
        parts.append(self.rule_emb(rule).unsqueeze(1).expand(b, k, -1))
        out = self.trunk(torch.cat(parts, dim=-1))
        return out.squeeze(-1) if self.n_out == 1 else out.squeeze(1)


class HierarchicalPolicy(nn.Module):
    """编码器 + 三个 agent 的头（**一个 encode、多次读**）。

    ⚠️ **2026-10-10：构造签名去掉 `n_chargers`** —— 充电头删了，本网络**不再随布局变宽度**
    （唯一的变宽头就是它）。**故 `HierarchicalPolicy()` 无必需参数。**
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, feat_dim: int = F_MAX,
                 la_ctx_dim: int = LA_CTX_DIM, with_value: bool = False,
                 seq_ma: bool = False):
        super().__init__()
        self.seq_ma = bool(seq_ma)
        self.enc = LayoutEncoder(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                                 feat_dim=feat_dim)
        # ── 上层（MA）：**三个独立**分类头（设计 §2 / §4.6）──
        # ⚠️ **2026-10-19：输入是 M/B/V/G 四段均值拼接**（`4*d_model`），不是全 token 均值。
        n_rules = {"PA": N_PA_RULES, "LA": N_LA_RULES, "RA": N_PM_RULES}
        if self.seq_ma:
            # 顺序条件：PA → LA；RA 先保留但不条件。
            self.select = SequentialRuleSelectHeads(
                d_model, n_rules, in_dim=4 * d_model)
        else:
            self.select = RuleSelectHeads(
                d_model, n_rules, in_dim=4 * d_model)
        # ── 下层：**每个 agent 一个打分头**，在规则筛出的候选集内排序 ──
        #    ⚠️ 不是"每条规则一个头"：规则身份走 embedding，候选集本身由规则决定。
        self.scorer_pa = PairScoreHead(d_model, PA_CAND_DIM, N_PA_RULES, hidden=128)
        self.scorer_la = SingleScoreHead(d_model, LA_CAND_DIM, N_LA_RULES,
                                         hidden=64, ctx_dim=la_ctx_dim)
        self.scorer_ra = SingleScoreHead(d_model, RA_FEAT_DIM, N_PM_RULES,
                                         hidden=64, n_out=2)
        self.la_ctx_dim = int(la_ctx_dim)
        # **critic 的 `V(s)`**（只服务 Q1 的臂 A）。
        # ⚠️ **v3 起不再拼 `g`**（`g` 删了）⟹ 输入就是池化向量。
        # ⚠️ **默认不建**：`with_value=False` 时本属性是 `None` ⟹ 不产生参数。
        self.value_head = (nn.Sequential(nn.Linear(4 * d_model, d_model), nn.GELU(),
                                         nn.Linear(d_model, 1))
                           if with_value else None)

    # ── 掩码：`seg = (n_m, n_jobs, n_agv, 1)` → 各 agent 读哪一段 ──
    @staticmethod
    def masks(seg: tuple[int, ...]) -> dict[str, tuple[int, int]]:
        """逐 agent 的 token 行区间（半开）。**唯一的掩码定义处**——别处不得再算一遍。

        ⚠️ **2026-10-10：⑪ 去后 `seg` 退回四元**（C 段没了）⟹ `"C"` 与 `"RA_charge"`
        两个区间一并删。`RA_maint` 照旧读 M 段。
        """
        n_m, n_jobs, n_agv = (int(seg[0]), int(seg[1]), int(seg[2]))
        v_lo = n_m + n_jobs
        c_lo = v_lo + n_agv
        n_total = c_lo + 1
        return {
            "M": (0, n_m),
            "B": (n_m, v_lo),
            "V": (v_lo, c_lo),
            "G": (n_total - 1, n_total),
            "PA": (0, v_lo),                                      # M ∪ B
            "LA": (v_lo, c_lo),                                   # V（+ 任务上下文，见 `ctx_dim`）
            "RA_maint": (0, n_m),                                 # M
        }

    @staticmethod
    def _pool(z: torch.Tensor, rng: tuple[int, int], name: str) -> torch.Tensor:
        """**段内均值池化**。空段**显式报错**。

        ⚠️ **这条护栏是补回来的**：v2 的 `_MaskedScoreHead.forward` 有它，
        v3 改用 `z[:, lo:hi].mean(dim=1)` 之后**丢了** —— 而 `mean` 对**空**切片
        给的是 `nan`，**不报错** ⟹ "这个 agent 没接上"变成**无声的坏策略**。
        """
        lo, hi = int(rng[0]), int(rng[1])
        if hi <= lo:
            raise ValueError(
                f"掩码段 {name!r} 为空（{lo}:{hi}）—— 段内池化会得到 nan；"
                "静默返回会让『这个 agent 没接上』变成无声的坏策略")
        return z[:, lo:hi, :].mean(dim=1)

    def forward(self, tok: torch.Tensor, seg: tuple[int, ...], *,
                rules: dict[str, torch.Tensor] | None = None,
                bias: torch.Tensor | None = None,
                hard: bool = True) -> PolicyOut:
        """`tok (B,N,F)` + `seg` → `PolicyOut`（**v3 top-K 架构**）。

        `rules` 给定 ⟹ **规则冻结**（阶段 B / 执行）；`None` ⟹ 顶层现采一组
        （阶段 A / 训练时）。`hard=True`（默认）取 argmax，否则按 logits 采样。

        ⚠️ **`hard=False` 用的是**全局** torch RNG**
        （`torch.distributions.Categorical(logits).sample()`）—— **不受调用方的
        `torch.Generator` 控制**。**要可复现就自己传 `rules=`，或按 logits 自己采**
        （`NetPolicy._categorical` 就是这么做的）。

        ⚠️ 下层不再由本方法算 —— 候选集来自 `hrl.candidates`，打分走
        `score_pa` / `score_la` / `score_ra`。本方法只出顶层 logits 与编码器 `z`。
        """
        z, _ = self.enc(tok, seg, bias)               # (B,N,d)；均值 `pooled` 不再给 MA
        mk = self.masks(seg)
        # ⚠️ **2026-10-19：MA 的输入 = 四段均值拼接**（固定宽度、保留 M/B/V/G 的状态差异）。
        sel_in = torch.cat([self._pool(z, mk["M"], "M"),
                            self._pool(z, mk["B"], "B"),
                            self._pool(z, mk["V"], "V"),
                            self._pool(z, mk["G"], "G")], dim=-1)
        logits = self.select(sel_in, rules=rules, hard=hard)                  # {PA, LA, RA}
        sel: dict[str, torch.Tensor] = {}
        for k, lg in logits.items():
            given = None if rules is None else rules.get(k)
            if given is not None:
                sel[k] = given                        # 给定 ⟹ 冻结
            elif hard:
                sel[k] = lg.argmax(dim=-1)
            else:
                sel[k] = torch.distributions.Categorical(logits=lg).sample()
        return PolicyOut(
            rule_logits=logits, rule_sel=sel, z=z,
            value=(None if self.value_head is None else self.value_head(sel_in)),
        )

    # ── 下层打分：候选行号 → `z` 行 gather → 打分头 → 组内 z 化 ──
    @staticmethod
    def _gather(z: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """`z (B,N,d)` + `rows (B,K)` → `(B,K,d)`。"""
        rows = rows.to(z.device).long()
        idx = rows.unsqueeze(-1).expand(-1, -1, z.shape[-1])
        return z.gather(1, idx)

    def score_pa(self, z: torch.Tensor, rows_a: torch.Tensor,
                 rows_b: torch.Tensor, feats: torch.Tensor,
                 rule: torch.Tensor,
                 mask: torch.Tensor | None = None) -> torch.Tensor:
        """PA 候选打分：`(B,N,d)` + 行号 + 特征 → **`(B,K)` logits**（组内 z 化）。

        `rows_a` = 作业 token 行（`n_m + job`）；`rows_b` = 机台 token 行。
        """
        za = self._gather(z, rows_a)
        zb = self._gather(z, rows_b)
        feats = feats.to(device=z.device, dtype=z.dtype)
        rule = rule.to(device=z.device).long()
        s = self.scorer_pa(za, zb, feats, rule)
        if mask is None:
            mask = torch.ones_like(s, dtype=torch.bool)
        else:
            mask = mask.to(device=z.device).bool()
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("PA 候选集为空 —— 打分头不能对空集出分布")
        return _norm_logits(s, mask)

    def score_la(self, z: torch.Tensor, rows_a: torch.Tensor,
                 feats: torch.Tensor, rule: torch.Tensor,
                 la_ctx: torch.Tensor | None,
                 mask: torch.Tensor | None = None) -> torch.Tensor:
        """LA 候选打分：`(B,N,d)` + 车行号 + 特征 + 任务上下文 → **`(B,K)` logits**。"""
        if la_ctx is None:
            raise ValueError(
                "LA 打分缺 `la_ctx`（这一趟运什么）——没有任务就无从打分；"
                "静默补 0 会报出『每趟任务都一样』的分布")
        za = self._gather(z, rows_a)
        feats = feats.to(device=z.device, dtype=z.dtype)
        rule = rule.to(device=z.device).long()
        s = self.scorer_la(za, feats, rule, la_ctx.to(device=z.device, dtype=z.dtype))
        if mask is None:
            mask = torch.ones_like(s, dtype=torch.bool)
        else:
            mask = mask.to(device=z.device).bool()
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("LA 候选集为空 —— 打分头不能对空集出分布")
        return _norm_logits(s, mask)

    def score_ra(self, z: torch.Tensor, row: torch.Tensor,
                 feats: torch.Tensor, rule: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
        """RA 二元打分：`(B,N,d)` + 机台行 + 特征 → **`(B,2)` logits**（**不 z 化**）。

        ⚠️ 只有两个动作，z 化会把 logit 差钉死成 ±1，抹掉"有多确定"。
        允许集外的动作置 `-inf`。
        """
        za = self._gather(z, row)
        feats = feats.to(device=z.device, dtype=z.dtype)
        rule = rule.to(device=z.device).long()
        s = self.scorer_ra(za, feats, rule)
        mask = mask.to(device=z.device).bool()
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("RA 允许集为空 —— 规则过滤后一个动作都不剩")
        return s.masked_fill(~mask, float("-inf"))
