# -*- coding: utf-8 -*-
"""**两阶段交替训练**（步 5b，设计 §4–§5）。

```
阶段 B（先做）  目标冻结在 g*  · 采 M 条**只变执行**的轨迹  · 只更新 PA/LA/RA 的 logp
阶段 A（后做）  下层冻结成 greedy · 在每个顶层点采 G 个目标 · 只更新 MA 的 logp
```

⚠️ **顺序是 B 先 A 后 —— 设计写的是"先 A 后 B"**（§4.2/§4.3 的编号次序）。理由（§93/§95 裁定）：

> **`B` 零初始化 ⟹ 起始时 `g` 完全不影响行为** ⟹ 阶段 A 采 G 个目标得到 **G 个完全相同的段回报**
> ⟹ `σ_G = 0` ⟹ **顶层拿不到梯度**。
> **`B` 必须先在阶段 B 里被学成非零，阶段 A 才有信号。**

⚠️ **而"`B` 零初始化"本身是我们的设计选择，不是框架要求**（设计文档没规定；§83.2 选定，§95 裁定）。
换来的是"**初始策略 = 不带 `g` 的普通策略**"（与关掉顶层的消融同分布）。
**空转只到第一次阶段 B 为止**；`B` 一非零，阶段 A 的 `σ` 就 > 0。
**故：若改成小随机初始化，本条顺序约束随即消失。**

**（其余设计照做**：§4.2 的"下层确定性 + G 个目标"、§4.3 的"M 条 + 目标冻结"、
§10① `G=M=8`、§10③ 段回报三样。**）**

══ 两条阶段共用的原语 ══

```
从顶层点 Z_j 起，对一组目标各跑一段（回放到 Z_j 的前缀 + 该目标 + 停表）
  ⟹ N 条**段回报**（`StepCounters` 相减）
  ⟹ 组内 z 化 Â = (R − μ)/σ
  ⟹ 只对**指定那一层**的决策带梯度重算 logp ⟹ loss = −Σ Â·Σlogp
```

**"从同一个 Z_j 出发"靠重放前缀实现**（见 `hrl/replay.py`）——**不靠分叉**。
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

from ..env.des import SimConfig, SimWorld, StepCounters
from ..env.instances import Instance
from ..env.layout import Layout
from ..nn.features import NormContext
from .agents import G_BINS, G_DIM, LA_CTX_DIM, HierarchicalPolicy
from .replay import run_branch, segment_return
from .rollout import Decision, NetPolicy, net_device

EPS = 1e-6
"""组内 z 化的分母下限（设计 §4.1 的 `+ε`）。"""

LOWER_KINDS = ("PA", "LA", "C", "M")
"""下层四类决策（三个 agent 的四个头）。阶段 B 只更新它们。"""


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

    def world(self) -> SimWorld:
        return SimWorld(self.inst, self.layout, self.dm, self.cfg, graph=self.graph)


def base_chain(net: HierarchicalPolicy, spec: Spec, *, greedy: bool = False,
               seed: int = 0, dt_events: int = 20) -> NetPolicy:
    """跑一条**完整**的 base 链（顶层在链内自采目标）——两阶段都从它取 `Z_j` 与前缀。"""
    p = NetPolicy(net, spec.inst, spec.layout, spec.ctx, greedy=greedy, seed=seed,
                  dt_events=dt_events)
    p.world = spec.world()                                 # type: ignore[attr-defined]
    p.metrics = p.world.run_gated(                          # type: ignore[attr-defined]
        seed_chain=0, policy_pa=p.policy_pa, policy_l=p.policy_l,
        policy_c=p.policy_c, policy_m=p.policy_m, policy_ma=p.policy_ma,
        dt_events=dt_events)
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

    ⚠️ 用的是**记录下来的** `(tok, seg, g_bins, la_ctx)`——当时的状态。
    用最新快照重算等于把策略输入换成另一个状态（决策与 logp 不再对应）。
    """
    dev = net_device(head)
    out: list[torch.Tensor] = []
    for d in decisions:
        if d.kind not in kinds:
            continue
        t = torch.as_tensor(d.tok).unsqueeze(0).to(dev)
        gb = torch.as_tensor([list(d.g_bins)], device=dev)
        lc = None if d.la_ctx is None else torch.as_tensor(d.la_ctx)[None].to(dev)
        o = head(t, d.seg, gb, la_ctx=lc, hard=True)
        if d.kind == "MA":
            # ⚠️ **顶层是 3 个**独立**的分档，不是一个 64 路的类别**（§10④"每维 4 档"）。
            # `action` 存的是压平下标 `b0*16 + b1*4 + b2` ⟹ 这里要**解回三档**再逐维取 logp。
            # （展平后当 64 路取 logp 会与 3 路各自的 logp **不是同一个数**，
            #  `g_logits.view(-1)` 更只有 12 项 ⟹ 越界。实测踩过。）
            b0, rem = divmod(int(d.action), G_BINS ** 2)
            b1, b2 = divmod(rem, G_BINS)
            lp = sum(torch.log_softmax(o.g_logits[0, i], dim=-1)[b]
                     for i, b in enumerate((b0, b1, b2)))
            out.append(lp)
            continue
        row = {"PA": o.pa, "LA": o.la, "C": o.ra_charge, "M": o.ra_maint}[d.kind]
        if row is None:
            raise ValueError(f"决策 {d.kind} 重算时该头为 None（`la_ctx` 没给？）")
        out.append(torch.log_softmax(row[0], dim=-1)[d.action])
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
    gb = torch.as_tensor([list(d.g_bins) for d in sel], device=dev)
    lc = torch.as_tensor(np.stack([
        d.la_ctx if d.la_ctx is not None else np.zeros(LA_CTX_DIM, dtype=np.float32)
        for d in sel])).to(dev)
    return head(tok, seg, gb, la_ctx=lc, hard=True), sel


def _logp_rows(o, sel: list[Decision]) -> list[torch.Tensor]:
    """一次前向的输出 `o` → 按 `sel` 顺序逐条取 logp（**带梯度**）。

    ⚠️ **顶层是 3 个**独立**的分档，不是一个 64 路的类别**（§10④"每维 4 档"）。
    `action` 存的是压平下标 `b0*16 + b1*4 + b2` ⟹ 这里要**解回三档**再逐维取 logp。
    （展平后当 64 路取 logp 会与 3 路各自的 logp **不是同一个数**，
     `g_logits.view(-1)` 更只有 12 项 ⟹ 越界。实测踩过。）
    """
    terms: list[torch.Tensor] = []
    for i, d in enumerate(sel):
        if d.kind == "MA":
            b0, rem = divmod(int(d.action), G_BINS ** 2)
            b1, b2 = divmod(rem, G_BINS)
            terms.append(sum(torch.log_softmax(o.g_logits[i, j], dim=-1)[b]
                             for j, b in enumerate((b0, b1, b2))))
            continue
        row = {"PA": o.pa, "LA": o.la, "C": o.ra_charge, "M": o.ra_maint}[d.kind]
        if row is None:
            raise ValueError(f"决策 {d.kind} 重算时该头为 None")
        terms.append(torch.log_softmax(row[i], dim=-1)[d.action])
    return terms


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


def ma_logp_at(head: HierarchicalPolicy, d: Decision) -> torch.Tensor:
    """在**同一个**顶层决策点 `d` 上，为**每个分档组合**算 logp —— 返回 `(G_BINS**3,)`。

    ⚠️ **为什么不能只算一个**：阶段 A 的 G 个成员**共享 `Z_j`**，
    区别只是"取了哪个 goal"。组内 z 化后 `Σ_m Â_m ≡ 0` ⟹
    若每个成员都用**同一个**动作的 logp，`Σ_m Â_m·logp = logp·Σ_m Â_m ≡ 0` ⟹ **loss 恒为 0**。
    **必须让每个成员配自己的动作**，logp 才有区分度。
    （标准 GRPO 里每个成员有**自己的一整条轨迹**，所以不撞这个问题；我们这里是"同一状态、多动作"，
    正是把它当**单步 contextual bandit** 来更新。）
    """
    dev = net_device(head)
    t = torch.as_tensor(d.tok).unsqueeze(0).to(dev)
    gb = torch.as_tensor([list(d.g_bins)], device=dev)
    o = head(t, d.seg, gb, la_ctx=None, hard=True)
    lps = []
    for flat in range(G_BINS ** G_DIM):
        b0, rem = divmod(flat, G_BINS ** 2)
        b1, b2 = divmod(rem, G_BINS)
        lps.append(sum(torch.log_softmax(o.g_logits[0, i], dim=-1)[b]
                       for i, b in enumerate((b0, b1, b2))))
    return torch.stack(lps)


def goal_flat(goal: tuple[int, int, int]) -> int:
    """`(b0,b1,b2)` → 压平下标（与 `NetPolicy.policy_ma` 记的 `action` 同一编码）。"""
    return int(goal[0]) * G_BINS ** 2 + int(goal[1]) * G_BINS + int(goal[2])


def _prefix_to(dec: list[Decision], j: int) -> tuple[list[int], tuple[int, int, int]]:
    """`Z_j` 之前（含 `Z_j` 那次 MA 决策）的动作序列 + `Z_j` 处生效的 `g`。

    `j` 是**顶层点的下标**（0 起）。返回的 `g` 是 `Z_j` 上**采到的**那个（base 链的）。
    """
    ma_idx = [i for i, d in enumerate(dec) if d.kind == "MA"]
    if j >= len(ma_idx):
        raise IndexError(f"顶层点下标 {j} 越界（本链只有 {len(ma_idx)} 个）")
    return [d.action for d in dec[:ma_idx[j]]], dec[ma_idx[j]].g_bins


class Branch(NamedTuple):
    """一条分支的产物：**它自己的**决策链 + 停表处的累计读数。

    ⚠️ **不要用 base 链的决策链去算这条分支的 loss** —— 分支的目标不同 ⟹ 动作不同 ⟹
    状态不同 ⟹ 决策条数与内容都不同。**loss 必须用分支自己的 `decisions`。**
    """

    counters: StepCounters
    decisions: list[Decision]


def segment_from(net: HierarchicalPolicy, spec: Spec, base: NetPolicy, j: int, *,
                 goal: tuple[int, int, int], seed: int, greedy: bool,
                 dt_events: int = 20) -> Branch:
    """**从 `Z_j` 起、用固定 `goal` 跑一段**（到 `Z_{j+1}`）→ 该分支的读数与决策链。

    回放 `Z_j` 之前的动作序列（不调策略），再让 `net` + `freeze_g` 接下去。
    ⚠️ 返回的 `decisions` **只含分支那一段**（`inner.rec` 是从 `Z_j` 起才记的）——
    前缀由脚本回放、不进 `inner.rec` ✓。
    """
    dec = base.rec.decisions
    prefix, _ = _prefix_to(dec, j)
    inner = NetPolicy(net, spec.inst, spec.layout, spec.ctx, g_bins=goal,
                      greedy=greedy, seed=seed, dt_events=dt_events, freeze_g=True)
    _, counters = run_branch(spec.world(), prefix, inner,
                             switch_after_ma=j, stop_after_ma=j + 2)
    return Branch(counters=counters, decisions=inner.rec.decisions)


def _run_segments(net: HierarchicalPolicy, spec: Spec, base: NetPolicy, j: int,
                  jobs: list[tuple[tuple[int, int, int], int, bool]], *,
                  dt_events: int, pool) -> list[Branch]:
    """跑 `Z_j` 上的一批段 —— `pool=None` 就地跑，给了就用 `SegmentPool` 并行。

    `jobs = [(goal, seed, greedy), ...]`。**两条路产出同形的 `Branch`**。
    ⚠️ 并行档在**子进程**里跑仿真与采样，**loss 仍在父进程算**（那要梯度）。
    """
    if pool is None:
        return [segment_from(net, spec, base, j, goal=g, seed=sd, greedy=gr,
                             dt_events=dt_events) for g, sd, gr in jobs]
    prefix, _ = _prefix_to(base.rec.decisions, j)
    out = pool.segments([(prefix, j, g, sd, gr) for g, sd, gr in jobs])
    return [Branch(counters=c, decisions=d) for c, d in out]


# ══════════════════ 两个阶段 ══════════════════

def phase_b(net: HierarchicalPolicy, spec: Spec, base: NetPolicy, *,
            M: int = 8, seed: int = 0, dt_events: int = 20, pool=None) -> dict:
    """**阶段 B**：目标冻结在 base 链的 `g`（stop-gradient），M 条**只变执行**的轨迹。

    只更新**下层**（PA/LA/C/M）的 logp —— **顶层的 logp 不进图**。
    """
    n_top = sum(1 for d in base.rec.decisions if d.kind == "MA")
    loss = torch.zeros((), requires_grad=True)
    rows = []
    for j in range(n_top):
        _prefix, goal = _prefix_to(base.rec.decisions, j)
        before = base.ma_counters[j]
        runs = _run_segments(net, spec, base, j,
                             [(goal, seed * 1000 + m, False) for m in range(M)],
                             dt_events=dt_events, pool=pool)
        segs = [segment_return(before, br.counters) for br in runs]
        adv = group_advantage([_scalar(sg, spec.w) for sg in segs])
        for a, sg in zip(adv, segs):
            rows.append({"j": j, "adv": float(a), "jobs": sg.done_jobs,
                         "energy": sg.energy_kwh, "twt": sg.twt})
        # ⚠️ **每条分支用自己的决策链**：目标相同、只有执行在变 ⟹ 组内只变执行（§4.3）。
        for a, br in zip(adv, runs):
            loss = loss - float(a) * logp_terms_batched(net, br.decisions, LOWER_KINDS)
    return {"loss": loss, "segments": rows, "n_top": n_top, "kind": "B"}


def phase_a(net: HierarchicalPolicy, spec: Spec, base: NetPolicy, *,
            G: int = 8, seed: int = 0, dt_events: int = 20, pool=None) -> dict:
    """**阶段 A**：下层**确定性**（greedy），每个顶层点采 G 个目标。

    只更新**顶层**（MA）的 logp。⚠️ `B=0` 时 G 条段的回报会全相同 ⟹ `σ=0` ⟹ `Â≡0` ⟹
    本阶段拿不到梯度（**这是为什么先做 B**，见模块 docstring）。
    """
    n_top = sum(1 for d in base.rec.decisions if d.kind == "MA")
    loss = torch.zeros((), requires_grad=True)
    seg_std = []
    for j in range(n_top):
        before = base.ma_counters[j]
        gen = torch.Generator().manual_seed(seed * 1000 + j)
        goals = [tuple(int(x) for x in torch.randint(0, G_BINS, (G_DIM,), generator=gen))
                 for _ in range(G)]
        runs = _run_segments(net, spec, base, j, [(g, 0, True) for g in goals],
                             dt_events=dt_events, pool=pool)
        rewards = [_scalar(segment_return(before, br.counters), spec.w) for br in runs]
        adv = group_advantage(rewards)
        seg_std.append(float(np.std(rewards)))
        # ⚠️ **顶层只更新 `Z_j` 那一个决策**（§4.2"只更新 g 的 logp"）——
        # 不是链上所有 MA 决策。`Z_j` 的决策在 **base 链**上。
        ma_idx = [i for i, d in enumerate(base.rec.decisions) if d.kind == "MA"]
        at_zj = ma_logp_at(net, base.rec.decisions[ma_idx[j]])       # (64,)
        idx = torch.as_tensor([goal_flat(g) for g in goals])
        # **每个成员配自己的动作**：`Σ_m Â_m·logp(a_m)`，`a_m` 互不相同 ⟹ 不为 0
        loss = loss - (torch.as_tensor(adv) * at_zj[idx]).sum()
    return {"loss": loss, "seg_std": seg_std, "n_top": n_top, "kind": "A"}


def train_step(net: HierarchicalPolicy, spec: Spec, optim: torch.optim.Optimizer | None,
               *, M: int = 8, G: int = 8, seed: int = 0, dt_events: int = 20,
               lr: float = 1e-4, pool=None, loss_threads: int = 8) -> dict:
    """一次**交替**：**先 B 后 A**（顺序的理由见模块 docstring）。

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
        for name, fn, kw in (("B", phase_b, {"M": M}), ("A", phase_a, {"G": G})):
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
    lower = (net.pa, net.la, net.ra_charge, net.ra_maint)
    upper = (net.ma,)
    return {"upper": norm(upper), "lower": norm(lower),
            "expect": ("lower" if phase == "B" else "upper")}
