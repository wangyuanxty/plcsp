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
from .rules import (LA_RULE_NAMES, N_LA_RULES, N_PA_RULES, N_PM_RULES,
                    PA_JOB_RULES, PA_MACHINE_RULES, PM_RULE_NAMES)

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
    """一次前向的全部输出（**v3**）—— 见 `docs/v3/design.md` §3 / §4.6。"""

    rule_logits: dict[str, torch.Tensor]
    """`{"PA": (B,12), "LA": (B,3), "RA": (B,3)}` —— 三个**独立**分类头的 logits。

    ⚠️ **三个独立**，不是一个联合分类 —— 与 v2 的 `MAHead`（3 个 4 档分档）同构。
    """
    rule_sel: dict[str, torch.Tensor]
    """`(B,)` —— 各头选中的规则**下标**（`hard=True` 时 argmax，否则采样）。

    ⚠️ `PA` 的下标是 `PA_RULE_NAMES` 的（`0..11`）；拆成"作业 / 机台"用
    `split_pa_rule`（**与 `rules.apply_pa_rule` 的拆法逐字同源**）。
    """
    param_mu: dict[str, torch.Tensor]
    """下层的**连续参数 μ**（**无界**）：`{"PA": (B,3), "LA": (B,1), "RA": (B,1)}`。

    ⚠️ **是 μ，不是参数本身** —— `param = lo + (hi−lo)·sigmoid(u)` 由调用方做，
    且 **`logp` 算在 `u` 上**（设计 §6.1）：**`u` 才是被采样的动作**。

    ⚠️ **硬路由**：进入 `param_mu[k]` 的只有**各行自己选中的**那个 head 的输出
    （未选中的行不进 gather ⟹ **梯度只落在被选中的 head 上**）。
    """
    value: torch.Tensor | None = None
    """(B, 1) —— **critic 的 `V(s)`**。仅在 `HierarchicalPolicy(with_value=True)` 时非空。

    ⚠️ **两层路径不建这个头**（默认 `with_value=False` ⟹ 连模块都不创建 ⟹
    `net.parameters()` 一位不变）。它只服务 **Q1 的臂 A**（`docs/v2/design.md` §12.2）。"""


def split_pa_rule(idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """PA 的**复合规则下标** → `(作业下标, 机台下标)`。

    ⚠️ **必须与 `rules.apply_pa_rule` 的拆法逐字同源**（`idx // len(PA_MACHINE_RULES)`
    与 `idx % len(PA_MACHINE_RULES)`）—— **两处各写一遍必然会漂**（设计 §13.3 ③）。
    """
    n_m = len(PA_MACHINE_RULES)
    return idx // n_m, idx % n_m


N_PA_JOB_PARAMS = 2
"""PA **作业侧****每条规则**的参数个数（设计 §4.2）。"""
N_PA_MACH_PARAMS = 1
"""PA **机台侧****每条规则**的参数个数 —— **只有 1 个**（设计 §4.3：机台侧唯一的无量纲量是
`util`，跟另两个（分钟）不能直接相加；为省一个参数引外部尺度不值）。"""
N_LA_PARAMS = 1
"""LA **每条规则**的参数个数（设计 §4.4）。"""
N_RA_PARAMS = 1
"""RA **每条规则**的参数个数（设计 §4.5 —— 都是役龄阈值）。"""
PA_PARAM_TOTAL = N_PA_JOB_PARAMS + N_PA_MACH_PARAMS
"""PA 的**动作向量**宽度 = 被选中的作业 head 的 + 被选中的机台 head 的 = **3**。

⚠️ **不是 4×2 + 3×1 = 11** —— 每条规则只出**自己**那几个参数（硬路由）。
"""


def _route(heads: RuleParamHeads, ctx: torch.Tensor, sel: torch.Tensor,
           ctx_feat: torch.Tensor | None = None) -> torch.Tensor:
    """**按行硬路由**：从 `(B, K, P)` 里取每行自己选中的那个 head → `(B, P)`。

    ⚠️ 前向**算全部 head**（批内每行的选择可能不同），**但未选中的不进 gather**
    ⟹ **它们的梯度仍是 0** —— 这就是硬路由在批下的实现（设计 §3）。
    """
    all_out = heads.forward_all(ctx, ctx_feat)
    idx = sel.view(-1, 1, 1).expand(-1, 1, all_out.shape[-1])
    return all_out.gather(1, idx).squeeze(1)


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

    def __init__(self, d_model: int, n_rules: dict[str, int], hidden: int | None = None):
        super().__init__()
        if not n_rules:
            raise ValueError("n_rules 为空 —— 上层的动作空间不能是空的")
        h = hidden or max(32, d_model // 2)
        self.keys = tuple(n_rules)
        self.trunk = nn.Sequential(nn.Linear(d_model, h), nn.GELU())
        self.heads = nn.ModuleDict({k: nn.Linear(h, int(n)) for k, n in n_rules.items()})

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        """`(B, d)` → `{键: (B, n_rules[键])}` 的 logits。"""
        feat = self.trunk(pooled)
        return {k: self.heads[k](feat) for k in self.keys}


class RuleParamHeads(nn.Module):
    """**下层**：**每条规则一个 head** → 它自己的那几个**连续**参数（μ）。

    ⚠️ **硬路由**（设计 §3）：`forward` 收**规则名**，**只算那一个 head** ——
    没被选的 head **根本不进计算** ⟹ 上层不可替代（下层无法"学会忽略" one-hot）。

    ⚠️ **每条规则的参数个数可以不同**（设计 §4.2 / §4.4 / §4.5）——
    故 `n_params` 是一个 `{规则名: 个数}` 的字典，不是一个整数。

    ⚠️ **输出的是 μ（无界的动作 `u`），不是参数本身**。
    `param = lo + (hi − lo)·sigmoid(u)` 由调用方做；`logp` **在 `u` 上算**（设计 §6.1）。

    ⚠️ `ctx_dim`（如 LA 的"这一趟运什么"）**拼在池化向量后面**，不进 token —— 口径同
    `_MaskedScoreHead`。
    """

    def __init__(self, d_model: int, n_params: dict[str, int], hidden: int | None = None,
                 ctx_dim: int = 0):
        super().__init__()
        if not n_params:
            raise ValueError("n_params 为空 —— 下层不能没有规则")
        h = hidden or max(32, d_model // 2)
        self.keys = tuple(n_params)
        self.ctx_dim = int(ctx_dim)
        self.trunk = nn.Sequential(nn.Linear(d_model + self.ctx_dim, h), nn.GELU())
        self.heads = nn.ModuleDict({k: nn.Linear(h, int(n))
                                    for k, n in n_params.items()})

    def forward(self, pooled: torch.Tensor, rule: str,
                ctx_feat: torch.Tensor | None = None) -> torch.Tensor:
        """`(B, d)` + 规则名 → `(B, n_params[rule])` 的 **μ**。

        ⚠️ **`rule` 必须在 `n_params` 里** —— 否则显式报错（不静默退回"随便挑一个"）。
        ⚠️ **`ctx_dim > 0` 时必须给 `ctx_feat`** —— 静默补 0 会让"这个头没接上"
        变成无声的坏策略（口径同 `_MaskedScoreHead`）。
        """
        if rule not in self.heads:
            raise ValueError(f"未知的规则 {rule!r}；可选={self.keys}")
        return self.heads[rule](self.trunk(self._with_ctx(pooled, ctx_feat)))

    def _with_ctx(self, pooled: torch.Tensor,
                  ctx_feat: torch.Tensor | None) -> torch.Tensor:
        """拼逐决策上下文（`ctx_dim > 0` 时**必须**给，否则显式报错）。"""
        if not self.ctx_dim:
            return pooled
        if ctx_feat is None:
            raise ValueError(
                f"本头声明了 ctx_dim={self.ctx_dim} 的逐决策上下文，但没给——"
                "静默补 0 会让『这个头没接上』变成无声的坏策略")
        return torch.cat([pooled, ctx_feat], dim=-1)

    def forward_all(self, pooled: torch.Tensor,
                    ctx_feat: torch.Tensor | None = None) -> torch.Tensor:
        """**全部 head 一次算完** → `(B, K, n_params)`（`K = len(self.keys)`）。

        ⚠️ **为什么要它**：硬路由在**批**下要**按行分派**（每行选中的规则可能不同）
        ⟹ 前向**必须算全部 head**，再按行 gather。
        **这不削弱硬路由** —— 未选中的 head **不进 gather**，**它们的梯度仍是 0**。
        """
        feature = self.trunk(self._with_ctx(pooled, ctx_feat))
        return torch.stack([self.heads[k](feature) for k in self.keys], dim=1)


class HierarchicalPolicy(nn.Module):
    """编码器 + 三个 agent 的头（**一个 encode、多次读**）。

    ⚠️ **2026-10-10：构造签名去掉 `n_chargers`** —— 充电头删了，本网络**不再随布局变宽度**
    （唯一的变宽头就是它）。**故 `HierarchicalPolicy()` 无必需参数。**
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, feat_dim: int = F_MAX,
                 la_ctx_dim: int = LA_CTX_DIM, with_value: bool = False):
        super().__init__()
        self.enc = LayoutEncoder(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                                 feat_dim=feat_dim)
        # ── 上层（MA）：**三个独立**分类头（设计 §2 / §4.6）──
        self.select = RuleSelectHeads(d_model, {"PA": N_PA_RULES, "LA": N_LA_RULES,
                                                "RA": N_PM_RULES})
        # ── 下层：**每条规则一个 head**，**硬路由**（设计 §3 / §4.6）──
        self.pa_job = RuleParamHeads(d_model, {r: N_PA_JOB_PARAMS for r in PA_JOB_RULES})
        self.pa_mach = RuleParamHeads(d_model,
                                      {r: N_PA_MACH_PARAMS for r in PA_MACHINE_RULES})
        self.la = RuleParamHeads(d_model, {r: N_LA_PARAMS for r in LA_RULE_NAMES},
                                 ctx_dim=la_ctx_dim)
        self.ra = RuleParamHeads(d_model, {r: N_RA_PARAMS for r in PM_RULE_NAMES})
        self.la_ctx_dim = int(la_ctx_dim)
        # **critic 的 `V(s)`**（只服务 Q1 的臂 A）。
        # ⚠️ **v3 起不再拼 `g`**（`g` 删了）⟹ 输入就是池化向量。
        # ⚠️ **默认不建**：`with_value=False` 时本属性是 `None` ⟹ 不产生参数。
        self.value_head = (nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
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
        return {
            "M": (0, n_m),
            "B": (n_m, v_lo),
            "V": (v_lo, c_lo),
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
                la_ctx: torch.Tensor | None = None,
                bias: torch.Tensor | None = None,
                hard: bool = True) -> PolicyOut:
        """`tok (B,N,F)` + `seg` → `PolicyOut`（**v3**）。

        `rules` 给定 ⟹ **规则冻结**（阶段 B / 执行）；`None` ⟹ 顶层现采一组
        （阶段 A / 训练时）。`hard=True`（默认）取 argmax，否则按 logits 采样。

        ⚠️ **`hard=False` 用的是**全局** torch RNG**
        （`torch.distributions.Categorical(logits).sample()`）—— **不受调用方的
        `torch.Generator` 控制**。**要可复现就自己传 `rules=`，或按 logits 自己采**
        （`NetPolicy._categorical` 就是这么做的）。
        这条是实测踩过的坑：`NetPolicy._forward` 曾按 `greedy` 传 `hard`，
        ⟹ 每次前向都推进全局随机流 ⟹ 同 seed 的两条链不一样。
        **这两项只影响 `rule_sel`；`param_mu` 永远是 μ** —— 下层的采样由调用方做
        （设计 §6.1：**`u` 才是被采样的动作**）。

        ⚠️ **`la_ctx=None` ⟹ `param_mu["LA"]` 也是 `None`**（LA 头不算）—— 口径同 v2：
        LA 的规则自变量含这一趟的任务，**没有任务就无从打分**。补 0 等于报出一个
        "每趟任务都一样"的分布（`RuleParamHeads` 会为此显式报错）。
        好处附带：一个决策点只用得着一个头。
        """
        z, pooled = self.enc(tok, seg, bias)          # (B,N,d), (B,d)
        logits = self.select(pooled)                  # {PA, LA, RA}
        sel: dict[str, torch.Tensor] = {}
        for k, lg in logits.items():
            given = None if rules is None else rules.get(k)
            if given is not None:
                sel[k] = given                        # 给定 ⟹ 冻结
            elif hard:
                sel[k] = lg.argmax(dim=-1)
            else:
                sel[k] = torch.distributions.Categorical(logits=lg).sample()
        mk = self.masks(seg)
        c_pa = self._pool(z, mk["PA"], "PA")
        c_la = self._pool(z, mk["LA"], "LA")
        c_ra = self._pool(z, mk["RA_maint"], "RA_maint")
        ji, mi = split_pa_rule(sel["PA"])
        mu_pa = torch.cat([_route(self.pa_job, c_pa, ji),
                           _route(self.pa_mach, c_pa, mi)], dim=-1)
        return PolicyOut(
            rule_logits=logits, rule_sel=sel,
            param_mu={
                "PA": mu_pa,
                "LA": (None if la_ctx is None
                       else _route(self.la, c_la, sel["LA"], la_ctx)),
                "RA": _route(self.ra, c_ra, sel["RA"]),
            },
            value=(None if self.value_head is None else self.value_head(pooled)),
        )
