# -*- coding: utf-8 -*-
"""**两阶段交替训练**（步 5b，设计 §4–§5）。

```
阶段 A（先做）  下层确定性（**取 μ，不采样**）· 在每个顶层点采 G 组规则 · 只更新 MA 的三个分类头
阶段 B（后做）  规则冻结在 rule*（stop-gradient）· 采 M 条**只变执行**的轨迹 · 只更新下层参数头
```

⚠️ **实现顺序 = 设计 §6 的次序：A 先 B 后。**（2026-10-10 改；来历见下。）

**v2 曾反着来（B 先 A 后），理由是**：v2 的三个 `B` 矩阵**零初始化** ⟹ 起始时 `g` 不影响行为
⟹ 阶段 A 的 G 条段逐位相同 ⟹ `σ_G = 0` ⟹ 顶层拿不到梯度 ⟹ **必须先让 B 把 `B` 学非零**。

⚠️ **v3 里那条理由不存在**：选不同规则 ⟹ **走不同的 head** ⟹ 行为结构不同 ⟹
**`σ_G > 0` 从第 0 步起就成立**（设计 §6.0 ③ / §8）⟹ **不必再让 B 打头阵。**

⟹ **已改成 A 先 B 后**（`docs/v3/progress-log.md` §11）。

**（其余设计照做**：§6 的"下层确定性 + G 组规则"、"M 条 + 规则冻结"、
§10① `G=M=8`、§10③ 段回报三样。**）**

══ 两条阶段共用的原语 ══

```
从顶层点 Z_j 起，对一组规则各跑一段（回放到 Z_j 的前缀 + 该规则 + 停表）
  ⟹ N 条**段回报**（`StepCounters` 相减）
  ⟹ 组内 z 化 Â = (R − μ)/σ
  ⟹ 只对**指定那一层**的决策带梯度重算 logp ⟹ loss = −Σ Â·Σlogp
```

**"从同一个 Z_j 出发"靠重放前缀实现**（见 `hrl/replay.py`）——**不靠分叉**。

══ 规则三件的次序（**唯一一处定义**）══

`RULE_NAMES` 定 `("PA", "LA", "RA")`，`Decision` 的三个字段是 `rule_pa` / `rule_la` / `rule_ra`
（= `"rule_" + 名字小写`）。**`_rules_of`（`Decision` → 张量）与 `rules_dict`（元组 → 字典）
都从它派生** —— 两处各写一遍必然漂（设计 `docs/v3/design.md` §13.3 ③）。
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

from ..env.constraints import ConstraintConfig
from ..env.des import SimConfig, SimWorld, StepCounters
from ..env.instances import Instance
from ..env.layout import Layout
from ..nn.features import NormContext
from .agents import LA_CTX_DIM, HierarchicalPolicy
from .replay import run_branch, segment_return
from .rules import N_LA_RULES, N_PA_RULES, N_PM_RULES, SIGMA_DEFAULT
from .rollout import Decision, NetPolicy, gated_by_constraints, net_device

EPS = 1e-6
"""组内 z 化的分母下限（设计 §4.1 的 `+ε`）。"""

LOWER_KINDS = ("PA", "LA", "M")
# ⚠️ **2026-10-10：⑪ 去后 `"C"` 已删**（充电头不存在 ⟹ 链上不再有这种决策点）。
"""下层三类决策（两个 agent 的三个头）。阶段 B 只更新它们。"""


class Spec(NamedTuple):
    """一条链要的全部静态材料（实例 · 布局 · 图 · 距离 · 配置 · 归一化 · 参考目标）。"""

    inst: Instance
    layout: Layout
    graph: object
    dm: np.ndarray
    cfg: SimConfig
    ctx: NormContext
    w: tuple[float, float, float]
    """三目标的标量化权重（`reward.reward_weights(ref.as_tuple())` 派生）。

    ⚠️ **参考目标的任一分量为 0 时 `reward_weights` 会显式报错**（它写明了：
    "TWT=0 的实例须先调 τ"）。**MK01 的默认参考解就没有拖期** ⟹ 训练前必须给一个
    收紧的 `SimConfig.tau`（交期更紧 ⟹ TWT > 0），否则拿不到权重。
    **这是"先调 τ"那条纪律在训练侧的体现，不静默补一个假权重。**"""
    constraints: ConstraintConfig | None = None
    """五条约束的开关（`env.constraints.ConstraintConfig`）。`None` = 全默认（② 关、其余开）。

    ⚠️ **必须与 `ctx.constraints` 是同一份**：仿真按这一份走动力学，特征层按 `ctx.constraints`
    决定哪些量静默 —— 两份不同源就是**假信号**（见 `nn.features.norm_context` 的 docstring）。
    `q1_flat.build()` 是唯一同时构造两者的地方，它保证同源。

    ⚠️ **默认 `None` 保持向后兼容**：既有 `Spec(inst, lay, g, dm, cfg, ctx, w)` 的七元构造
    逐位不变（`world()` 传 `constraints=None` ⟹ `SimWorld` 用默认档）。
    """

    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    """三目标的 `(lo, hi)` —— **min–max 归一化**口径（照 Shi et al., AEI 65:103195 (2025) 式 51/53）。

    ⚠️ **`None`（默认）= 用原始量纲，逐位等于今日读数**（既有读数靠这一条）。
    给了 bounds ⟹ 标量化**前**先把每个目标按自己的可达范围拉到 `[0,1]`。
    见 `env.reward.normalize_minmax` 与 `progress-log.md` §147。

    **为什么要它**：原始量纲下三个目标各占的份额由**单位**决定（能耗只占 3%），
    而 `1/f^ref` 又走到另一个极端（标称 89%、实际驱动 19%）。
    """

    def world(self) -> SimWorld:
        return SimWorld(self.inst, self.layout, self.dm, self.cfg, graph=self.graph,
                        constraints=self.constraints)


def base_chain(net: HierarchicalPolicy, spec: Spec, *, greedy: bool = False,
               seed: int = 0, dt_events: int = 20) -> NetPolicy:
    """跑一条**完整**的 base 链（顶层在链内自采目标）——两阶段都从它取 `Z_j` 与前缀。"""
    p = NetPolicy(net, spec.inst, spec.layout, spec.ctx, greedy=greedy, seed=seed,
                  dt_events=dt_events)
    p.world = spec.world()                                 # type: ignore[attr-defined]
    p.metrics = p.world.run_gated(                          # type: ignore[attr-defined]
        seed_chain=0, policy_pa=p.policy_pa, policy_l=p.policy_l,
        policy_ma=p.policy_ma, dt_events=dt_events,
        # ⚠️ **按约束档撤回调**（2026-10-10）：约束关掉 ⟹ 那个决策点不存在 ⟹
        # 传 `policy_m` 会被 `run_gated` 当场拒绝。约束消融踩过。见 `gated_by_constraints`。
        **gated_by_constraints(p, spec.constraints))
    return p


def _scalar(c: StepCounters, w: tuple[float, float, float]) -> float:
    """段回报标量化：`Σ wᵢ(−fᵢ)`，`f = (完工工件数取负, 能耗, 拖期)`。**越大越好。**"""
    f = (-float(c.done_jobs), float(c.energy_kwh), float(c.twt))
    return float(sum(-wi * fi for wi, fi in zip(w, f)))


def group_advantage(rewards: list[float]) -> np.ndarray:
    """组内相对优势 `Â⁽ⁱ⁾ = (R⁽ⁱ⁾ − μ) / (σ + ε)`（设计 §4.1）。"""
    r = np.asarray(rewards, dtype=np.float64)
    return (r - r.mean()) / (r.std() + EPS)


def logp_terms(head: HierarchicalPolicy, decisions: list[Decision],
               kinds: tuple[str, ...]) -> torch.Tensor:
    """**带梯度**重算所给 kinds 的 logp（逐条前向；v1 的 `_decision_logp_terms` 同形）。

    ⚠️ 用的是**记录下来的** `(tok, seg, rules, la_ctx)`——当时的状态。
    用最新快照重算等于把策略输入换成另一个状态（决策与 logp 不再对应）。
    """
    dev = net_device(head)
    out: list[torch.Tensor] = []
    for d in decisions:
        if d.kind not in kinds:
            continue
        t = torch.as_tensor(d.tok).unsqueeze(0).to(dev)
        lc = None if d.la_ctx is None else torch.as_tensor(d.la_ctx)[None].to(dev)
        rules = _rules_of([d], dev)
        o = head(t, d.seg, rules=rules, la_ctx=lc, hard=True)
        out.append(_one_logp(o, d, 0))
    if not out:
        return torch.zeros((), requires_grad=True)
    return torch.stack(out).sum()


def batched_forward(head: HierarchicalPolicy, decisions: list[Decision],
                    kinds: tuple[str, ...]):
    """挑出 `kinds` 里的决策，**一次**前向 → `(PolicyOut, sel)`；空则 `(None, [])`。

    ⚠️ **为什么能批**：`seg`（各段行数）由**实例 + 布局**决定，**一条链里恒定** ⟹
    每条决策的 token 张量都是 `(N, F_MAX)`、**N 相同** ⟹ 可叠成 `(B, N, F_MAX)` 一次前向。
    这是 v1 `_encoder_forward_batched` 的同一招（本机实测批前向吞吐 ≈3.4×）。

    ⚠️ **`seg` 恒定这件事由 `state_emb.build_tok` 保证**：`job_features` 恒输出 `ctx.n_jobs`
    行，作业完工只改内容（`finished` 维）、**不改行数**。**将来若引入会变长的段，这里必须
    当场报错**，不能静默按第一条的 `seg` 去算别人的 token。

    ⚠️ **LA 的上下文**：给**每一条**都填 `la_ctx`（没有的填 0），LA 头于是对每条都算。
    对 PA/C/M 那几条算出来的 LA 分布**不用**（只取自己那个头的那一行）——
    **多算一点，换一次前向。**
    """
    sel = [d for d in decisions if d.kind in kinds]
    if not sel:
        return None, []
    seg = sel[0].seg
    if any(d.seg != seg for d in sel):
        raise ValueError("同一批里的 `seg` 不一致 —— 批量前向的前提是段结构恒定")
    # ⚠️ **设备跟随策略**（见 `rollout.net_device`）：搬到 `head` 参数所在的设备。
    # `tok` 存在 `Decision` 里时恒为 **numpy（主机内存）** —— 所以每次重算都要搬一次。
    dev = net_device(head)
    tok = torch.as_tensor(np.stack([d.tok for d in sel])).to(dev)
    lc = torch.as_tensor(np.stack([
        d.la_ctx if d.la_ctx is not None else np.zeros(LA_CTX_DIM, dtype=np.float32)
        for d in sel])).to(dev)
    # ⚠️ **逐行的规则**：v3 里一条链跨多个 ΔT ⟹ **批内每行的规则可能不同**。
    # `HierarchicalPolicy.forward` 支持逐行 `rules`（硬路由按行 gather）⟹ **仍是一次前向**。
    return head(tok, seg, rules=_rules_of(sel, dev), la_ctx=lc, hard=True), sel


RULE_NAMES: tuple[str, ...] = ("PA", "LA", "RA")
"""规则三件的**次序** —— `Decision` 的字段是 `"rule_" + 名字小写`。

⚠️ **唯一一处定义**：`_rules_of`（`Decision` → 逐行张量）与 `rules_dict`（元组 → 字典）
都从它派生。**两处各写一遍必然漂**（设计 §13.3 ③）。
"""


def _rules_of(decisions: list[Decision], dev) -> dict[str, torch.Tensor]:
    """`Decision` 列表 → `forward` 要的**逐行**规则张量。"""
    return {k: torch.as_tensor([int(getattr(d, "rule_" + k.lower()))
                                for d in decisions],
                               dtype=torch.long, device=dev)
            for k in RULE_NAMES}


def rules_dict(rule: tuple[int, int, int]) -> dict[str, int]:
    """`(rule_PA, rule_LA, rule_RA)` → `NetPolicy(rules=…)` 要的字典。

    ⚠️ **与 `_rules_of` 同源**（都按 `RULE_NAMES` 的次序）—— 见该常量的说明。
    """
    return {k: int(v) for k, v in zip(RULE_NAMES, rule)}


def _one_logp(o, d: Decision, i: int, sigma: float = SIGMA_DEFAULT) -> torch.Tensor:
    """一次前向的输出 `o` 的第 `i` 行 → 该决策的 **logp（带梯度）**。

    - **MA**：三个**独立**分类头的 `log_softmax` 之和（不是一个联合分类）
    - **下层**：**连续** —— `log N(u; μ, σ)`，**`u` 才是动作**（设计 §6.1 / §13.3 ①）
    """
    if d.kind == "MA":
        return sum(torch.log_softmax(o.rule_logits[k][i], dim=-1)[r]
                   for k, r in (("PA", int(d.rule_pa)), ("LA", int(d.rule_la)),
                                ("RA", int(d.rule_ra))))
    key = {"PA": "PA", "LA": "LA", "M": "RA"}[d.kind]
    mu = o.param_mu[key][i]
    if mu is None:
        raise ValueError(f"决策 {d.kind} 重算时该头为 None（`la_ctx` 没给？）")
    if d.param_u is None:
        raise ValueError(f"决策 {d.kind} 没存 `param_u` —— 连续头的 logp 重算不了")
    u = torch.as_tensor(np.asarray(d.param_u), dtype=mu.dtype, device=mu.device)
    var = float(sigma) ** 2
    return (-(u - mu) ** 2 / (2.0 * var)
            - 0.5 * float(np.log(2.0 * np.pi * var))).sum()


def _logp_rows(o, sel: list[Decision]) -> list[torch.Tensor]:
    """一次前向的输出 `o` → 按 `sel` 顺序逐条取 logp（**带梯度**）。

    ⚠️ **批内每行的规则可以不同**（一条链跨多个 ΔT）⟹ `batched_forward` 传的是
    **逐行** rules（`HierarchicalPolicy.forward` 按行硬路由）。
    ⟹ **这里按行号取自己那一条即可**，不涉及任何重排。
    """
    return [_one_logp(o, d, i) for i, d in enumerate(sel)]


def batched_terms(head: HierarchicalPolicy, decisions: list[Decision],
                  kinds: tuple[str, ...], *, want_value: bool = False):
    """**一次**前向 → `(逐决策 logp 向量 (n,), 逐决策 V 向量 (n,) 或 None)`。

    `want_value=True` 时两个向量**同序**（都按 `batched_forward` 选出的 `sel`），
    故 `−Σ_t Â_t·logp_t`（`Â = R − V`）可以直接逐元素算——**不必再前向一次**。

    ⚠️ `want_value=True` 而 `head.value_head is None` 时**显式报错**，不静默返回 0
    （静默的 0 会让"critic 没接上"变成一条无声的坏基线）。
    """
    o, sel = batched_forward(head, decisions, kinds)
    if not sel:
        return torch.zeros(0), None
    lp = torch.stack(_logp_rows(o, sel))
    if not want_value:
        return lp, None
    if o.value is None:
        raise ValueError("这个 net 没建 critic 头 —— 构造时须给 "
                         "`HierarchicalPolicy(with_value=True)`")
    return lp, o.value[:, 0]


def logp_terms_batched(head: HierarchicalPolicy, decisions: list[Decision],
                       kinds: tuple[str, ...]) -> torch.Tensor:
    """`logp_terms` 的**批量**版：所有选中的决策**一次**前向（见 `batched_forward`）。"""
    lp, _ = batched_terms(head, decisions, kinds)
    if lp.numel() == 0:
        return torch.zeros((), requires_grad=True)
    return lp.sum()


def value_terms(head: HierarchicalPolicy, decisions: list[Decision],
                kinds: tuple[str, ...]) -> torch.Tensor:
    """逐决策的 `V(s)` —— **与 logp 共用同一次前向**（见 `batched_terms`）。只服务 Q1 的臂 A。"""
    _, v = batched_terms(head, decisions, kinds, want_value=True)
    return torch.zeros(0) if v is None else v


def ma_logp_at(head: HierarchicalPolicy, d: Decision,
               rules: list[tuple[int, int, int]]) -> torch.Tensor:
    """在**同一个**顶层决策点 `d` 上，为**给定的若干组规则**各算一个 logp → `(len(rules),)`。

    ⚠️ **v2 扫的是"全 64 个档"；v3 的规则空间是 `12 × 3 × 3 = 108`，扫全空间没必要**
    —— 阶段 A 只用得到它自己采的那 `G` 组。

    ⚠️ **必须让每个成员配自己的动作**（v2 的教训）：若所有成员用同一组规则，
    组内 z 化后 `Σ_m Â_m ≡ 0` ⟹ `Σ_m Â_m·logp ≡ 0` ⟹ **损失恒为 0**。
    """
    dev = net_device(head)
    t = torch.as_tensor(d.tok).unsqueeze(0).to(dev)
    o = head(t, d.seg, hard=True)
    return torch.stack([
        sum(torch.log_softmax(o.rule_logits[k][0], dim=-1)[r]
            for k, r in (("PA", int(rp)), ("LA", int(rl)), ("RA", int(rr))))
        for rp, rl, rr in rules])


def _sample_rules(gen: torch.Generator) -> tuple[int, int, int]:
    """从**规则空间**里**均匀**采一组 `(rule_PA, rule_LA, rule_RA)`。

    ⚠️ **均匀采，不是按策略采** —— 阶段 A 的组是"同一状态下 G 个不同的目标"，
    谁好由**组内比较**决定；按策略采会把当前偏好掺进探索里（v2 同口径）。
    """
    def pick(n: int) -> int:
        return int(torch.randint(0, n, (1,), generator=gen))
    return pick(N_PA_RULES), pick(N_LA_RULES), pick(N_PM_RULES)


def goal_flat(rule: tuple[int, int, int]) -> int:
    """`(rule_PA, rule_LA, rule_RA)` → 压平下标（**只用于报表 / 落盘编码**）。

    ⚠️ **训练路径不再用它**：`ma_logp_at` 在 v3 里按传入次序返回 `(len(rules),)`，
    与优势**逐元素**相乘即可（v2 扫全空间才需要"拿压平下标去索引"）。
    """
    return ((int(rule[0]) * N_LA_RULES + int(rule[1])) * N_PM_RULES
            + int(rule[2]))


def _prefix_to(dec: list[Decision], j: int) -> tuple[list[int], tuple[int, int, int]]:
    """`Z_j` **之前**（**不含** `Z_j` 那次 MA 决策）的决策序列 + `Z_j` 处生效的规则。

    ⚠️ "不含"由 `run_branch` 的切换口径定死：`_live()` 是 `ma_calls > switch_after_ma`
    ⟹ 第 `j` 次 MA（0 起）仍走脚本，**第 `j+1` 次才换 `net`**。

    `j` 是**顶层点的下标**（0 起）。返回的规则是 `Z_j` 上**采到的**那一组（base 链的）。
    """
    ma_idx = [i for i, d in enumerate(dec) if d.kind == "MA"]
    if j >= len(ma_idx):
        raise IndexError(f"顶层点下标 {j} 越界（本链只有 {len(ma_idx)} 个）")
    # ⚠️ **v3 的前缀是 `Decision` 序列**（不是 `int` 序列）—— 一次决策是"(规则, 参数)"，
    # 一个整数装不下。`ScriptedPolicy` 直接吃它（与记录同源，不另建编码）。
    return (list(dec[:ma_idx[j]]),
            (int(dec[ma_idx[j]].rule_pa), int(dec[ma_idx[j]].rule_la),
             int(dec[ma_idx[j]].rule_ra)))


class Branch(NamedTuple):
    """一条分支的产物：**它自己的**决策链 + 停表处的累计读数。

    ⚠️ **不要用 base 链的决策链去算这条分支的 loss** —— 分支的目标不同 ⟹ 动作不同 ⟹
    状态不同 ⟹ 决策条数与内容都不同。**loss 必须用分支自己的 `decisions`。**
    """

    counters: StepCounters
    decisions: list[Decision]


def segment_from(net: HierarchicalPolicy, spec: Spec, base: NetPolicy, j: int, *,
                 rule: tuple[int, int, int], seed: int, greedy: bool,
                 dt_events: int = 20) -> Branch:
    """**从 `Z_j` 起、用固定 `rule` 跑一段**（到 `Z_{j+1}`）→ 该分支的读数与决策链。

    回放 `Z_j` 之前的决策序列（不调策略），再让 `net` + **冻结规则**接下去。
    ⚠️ 返回的 `decisions` **只含分支那一段**（`inner.rec` 是从 `Z_j` 起才记的）——
    前缀由脚本回放、不进 `inner.rec` ✓。
    """
    dec = base.rec.decisions
    prefix, _ = _prefix_to(dec, j)
    inner = NetPolicy(net, spec.inst, spec.layout, spec.ctx, rules=rules_dict(rule),
                      greedy=greedy, seed=seed, dt_events=dt_events, freeze_rules=True)
    _, counters = run_branch(spec.world(), prefix, inner,
                             switch_after_ma=j, stop_after_ma=j + 2)
    return Branch(counters=counters, decisions=inner.rec.decisions)


def _run_all_segments(net: HierarchicalPolicy, spec: Spec, base: NetPolicy,
                      plan: list[tuple[int, list[tuple[tuple[int, int, int], int, bool]]]], *,
                      dt_events: int, pool) -> list[list[Branch]]:
    """跑**整个阶段**的全部段 → 按 `plan` 的边界分组返回。

    `plan = [(j, [(rule, seed, greedy), ...]), ...]`。

    ⚠️ **必须是"一次池调用"，不是每个顶层点一次**（2026-10-09 修）：
    `SegmentPool.segments()` 每次调用都要把 `state_dict`（约 2.4 MB）发给**每个** worker 一次；
    若每顶层点调一次，`train_step` 每步会调 `2 × n_top` 次（B 阶段 n_top 次 + A 阶段 n_top 次）
    ⟹ **序列化量 = `2 × n_top × n_workers × 2.4 MB`**。
    实测 `dt_events=10`（`n_top=38`、16 进程）：**每步 ~2.9 GB**，把成本曲线压成次线性
    （段数降 2.2× 而成本只降 1.4×）。合成一次后序列化量降到 `2 × n_workers × 2.4 MB`。

    ⚠️ 段与段**互相独立且确定**（给定 `(前缀, j, goal, seed, greedy)`）⟹ **合并调用不改读数**，
    只改谁在哪个 worker 上跑。（`test_segment_pool_matches_the_sequential_path` 钉住同一条。）
    """
    if pool is None:
        return [[segment_from(net, spec, base, j, rule=r, seed=sd, greedy=gr,
                              dt_events=dt_events) for r, sd, gr in jobs]
                for j, jobs in plan]
    flat: list[tuple] = []
    for j, jobs in plan:
        prefix, _ = _prefix_to(base.rec.decisions, j)
        flat.extend((prefix, j, r, sd, gr) for r, sd, gr in jobs)
    out = pool.segments(flat)
    branches = [Branch(counters=c, decisions=d) for c, d in out]
    res: list[list[Branch]] = []
    k = 0
    for _j, jobs in plan:
        res.append(branches[k:k + len(jobs)])
        k += len(jobs)
    if k != len(branches):                       # 不许静默错位
        raise RuntimeError(f"段回填错位：切出 {k} 条，拿到 {len(branches)} 条")
    return res


# ══════════════════ 两个阶段 ══════════════════

def phase_b(net: HierarchicalPolicy, spec: Spec, base: NetPolicy, *,
            M: int = 8, seed: int = 0, dt_events: int = 20, pool=None) -> dict:
    """**阶段 B**：规则冻结在 base 链的 `rule*`（stop-gradient），M 条**只变执行**的轨迹。

    只更新**下层**（PA / LA / RA 的参数头）—— **顶层的 logp 不进图**。
    """
    n_top = sum(1 for d in base.rec.decisions if d.kind == "MA")
    loss = torch.zeros((), requires_grad=True)
    rows = []
    # ⚠️ 先把**全部**顶层点的段列出来，再**一次**交给池（见 `_run_all_segments`）
    plan_b = [(j, [(_prefix_to(base.rec.decisions, j)[1], seed * 1000 + m, False)
                   for m in range(M)]) for j in range(n_top)]
    all_b = _run_all_segments(net, spec, base, plan_b, dt_events=dt_events, pool=pool)
    for j in range(n_top):
        before = base.ma_counters[j]
        runs = all_b[j]
        segs = [segment_return(before, br.counters) for br in runs]
        adv = group_advantage([_scalar(sg, spec.w) for sg in segs])
        for a, sg in zip(adv, segs):
            rows.append({"j": j, "adv": float(a), "jobs": sg.done_jobs,
                         "energy": sg.energy_kwh, "twt": sg.twt})
        # ⚠️ **每条分支用自己的决策链**：规则相同、只有执行在变 ⟹ 组内只变执行（§6）。
        for a, br in zip(adv, runs):
            loss = loss - float(a) * logp_terms_batched(net, br.decisions, LOWER_KINDS)
    return {"loss": loss, "segments": rows, "n_top": n_top, "kind": "B"}


def phase_a(net: HierarchicalPolicy, spec: Spec, base: NetPolicy, *,
            G: int = 8, seed: int = 0, dt_events: int = 20, pool=None) -> dict:
    """**阶段 A**：下层**确定性**（连续头**取 μ，不采样**），每个顶层点采 G 组规则。

    只更新**顶层**（MA 的三个分类头）的 logp。

    ⚠️ **v3 里本阶段不再空转**：选不同规则 ⟹ **走不同的 head** ⟹ 行为结构不同 ⟹
    `σ_G > 0` 从第 0 步起成立（设计 §6.0 ③）。**v2 那个"`B=0` ⟹ G 条段全相同"的病没有对象了。**

    ⚠️ **`greedy=True` 在这里的作用是"下层取 μ"，不是"上层取 argmax"** ——
    上层的规则仍按策略**采样**（`NetPolicy.policy_ma` 只在 `freeze_rules` 时不采）。
    """
    n_top = sum(1 for d in base.rec.decisions if d.kind == "MA")
    loss = torch.zeros((), requires_grad=True)
    seg_std = []
    rules_by_j = []
    for j in range(n_top):
        gen = torch.Generator().manual_seed(seed * 1000 + j)
        rules_by_j.append([_sample_rules(gen) for _ in range(G)])
    plan_a = [(j, [(r, 0, True) for r in rules_by_j[j]]) for j in range(n_top)]
    all_a = _run_all_segments(net, spec, base, plan_a, dt_events=dt_events, pool=pool)
    for j in range(n_top):
        before = base.ma_counters[j]
        rules = rules_by_j[j]
        runs = all_a[j]
        rewards = [_scalar(segment_return(before, br.counters), spec.w) for br in runs]
        adv = group_advantage(rewards)
        seg_std.append(float(np.std(rewards)))
        # ⚠️ **顶层只更新 `Z_j` 那一个决策**（§6"只更新那一组规则的 logp"）——
        # 不是链上所有 MA 决策。`Z_j` 的决策在 **base 链**上。
        ma_idx = [i for i, d in enumerate(base.rec.decisions) if d.kind == "MA"]
        at_zj = ma_logp_at(net, base.rec.decisions[ma_idx[j]], rules)   # (G,)
        # **每个成员配自己的动作**：`Σ_m Â_m·logp(a_m)`，`a_m` 互不相同 ⟹ 不为 0
        loss = loss - (torch.as_tensor(adv) * at_zj).sum()
    return {"loss": loss, "seg_std": seg_std, "n_top": n_top, "kind": "A"}


def train_step(net: HierarchicalPolicy, spec: Spec, optim: torch.optim.Optimizer | None,
               *, M: int = 8, G: int = 8, seed: int = 0, dt_events: int = 20,
               lr: float = 1e-4, pool=None, loss_threads: int = 8) -> dict:
    """一次**交替**：**先 A 后 B**（= 设计 §6 的次序；见模块 docstring）。

    每阶段各自：跑组 → 算优势 → **只对那一层**带梯度重算 → 反传 → 走一步。
    """
    if optim is None:
        optim = torch.optim.Adam(net.parameters(), lr=lr)
    # ⚠️ **两种负载要两种线程数**（实测 §106.7）：
    #   · `base_chain` = **每次一条决策的串行链** —— 1 线程最快（24 线程慢 2.7×，§99.2）；
    #   · `logp 前向 + backward` = **一大批矩阵运算** —— 8 线程快 2.9×。
    # 同一个进程里混着这两种负载，所以要**按阶段切**，不能全局设一个值。
    # ⚠️ 只在**带池**时切：`pool=None` 时段也在父进程跑（= 串行链那类），
    # 换成多线程反而更慢 —— 那条路保持调用方的设置不变。
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    base = base_chain(net, spec, greedy=False, seed=seed, dt_events=dt_events)
    out: dict = {"base_makespan": float(base.metrics["makespan"])}   # type: ignore[attr-defined]
    torch.set_num_threads(max(1, loss_threads) if pool is not None else prev)
    try:
        for name, fn, kw in (("A", phase_a, {"G": G}), ("B", phase_b, {"M": M})):
            optim.zero_grad(set_to_none=True)
            res = fn(net, spec, base, seed=seed, dt_events=dt_events, pool=pool, **kw)
            res["loss"].backward()
            grad = _grad_norm(net, name)
            optim.step()
            out[name] = {k: v for k, v in res.items() if k != "loss"}
            out[name]["loss"] = float(res["loss"].detach())
            out[name]["grad_norm"] = grad
    finally:
        torch.set_num_threads(prev)
    return out


def _grad_norm(net: HierarchicalPolicy, phase: str) -> dict:
    """**哪一层拿到了梯度**（本步的验收判据之一：阶段 B 不许动 MA，阶段 A 不许动下层）。"""
    def norm(mods) -> float:
        s = 0.0
        for mod in mods:
            for p in mod.parameters():
                if p.grad is not None:
                    s += float(p.grad.pow(2).sum())
        return float(np.sqrt(s))
    # ⚠️ **v3 的层界**：上层 = 那三个**分类头**（`select`）；
    # 下层 = 四个**参数头**（PA 的作业侧 / 机台侧 + LA + RA）。
    lower = (net.pa_job, net.pa_mach, net.la, net.ra)
    upper = (net.select,)
    return {"upper": norm(upper), "lower": norm(lower),
            "expect": ("lower" if phase == "B" else "upper")}
