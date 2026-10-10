# -*- coding: utf-8 -*-
"""**把两层策略接到仿真上**（步 5a）——网络 → `policy_*` 回调，并**冻结决策**供带梯度重算。

设计参照 v1 的既有模式（`plcsp/algo/group_rel.py` 的 `Decision` + `chain_logp`）：
**无梯度地跑一条链，把每个决策点的输入原样记下来；带梯度的 logp 事后**批量重算**。**

⚠️ **v2 比 v1 简单一处**：v1 的 `Decision` 要存**逐候选特征**（S 头挑机台、L 头挑车、R 头挑路），
因为动作是"从 n 个实体里挑一个"。**v2 的动作是"从固定几条规则里挑一条"** ⟹
头直接出固定宽度的 logits，**不需要逐候选特征**。`Decision` 因此只剩 `(kind, tok, seg, g, action, logp)`。

══ 四类决策点 ══

| kind | 触发 | 动作 | logits |
|---|---|---|---|
| **MA** | 定时器（每 `dt_events` 个完工）+ 扰动（③ 机障 / ⑨ AGV 故障） | 3 维各 4 档 | `(B,3,4)` |
| **PA** | 一道工序被分配的时刻（= transporter 的每个事件；**每次恰派一件**） | 12 条复合规则选 1 | `(B,12)` |
| **LA** | **两个点**（见下） | 3 条派车规则选 1 | `(B,3)` |
| **C / M** | ~~RA 的两个头~~ ⟹ **⑪ 去后只剩 M**（机台完工且下一件未上机） | `1+n_桩` / 2 | —— |

⚠️ **LA 有**两个**触发点**（2026-10-09 核清；此前本表只写了一个，且"与 PA 同一决策点"与实测不符）：

| # | 触发 | 在哪 | MK01 实测 |
|---|---|---|---|
| ① | **紧跟 PA 派工**（PA 真派出一件才触发） | `des.py` 主循环，PA 之后 | **55**（= 工序数） |
| ② | **`_emit_task` 的兜底** —— 那一程**没有预订车**时 | `_emit_task` 里的 `elif policy_l is not None` | **10**（= 作业数） |

**②是哪几程**：**回站程**。`des.py` 明写"**回站不由 PA 决定**（终点恒为装卸站）⟹ 不走决策点，直接发"
⟹ 回站程没有 `booking` ⟹ 落到兜底 ⟹ **再调一次 LA**。
**合计 55 + 10 = 65** —— 与实测的 `LA 65` 逐数对上。

⚠️ **②有个实质含义**：那条路上件**已在手** ⟹ `ready_at ≡ now` ⟹
**`LATEST_IN_TIME` 塌陷成 `MINCOST`**（`des.py` 自己的注释写了"这是预期的"）。
**⟹ 3 条 LA 规则里有 1 条在回站程上退化。**（开放线索，见 `docs/v2/progress-log.md`）

⚠️ **`policy_l` 拿到的是"规则下标"而不是车号** —— 规则由 `hrl.rules.la_rule_policy` 落到具体车上。
同理 `policy_pa` 拿到规则下标后由 `rules.apply_pa_rule` 落到 `(池下标, 机台)`。
**这样"策略的输出"与"规则的输出"分得干干净净**：策略只在**规则层**做选择。
"""
from __future__ import annotations

from typing import NamedTuple

import math

import numpy as np
import torch

from ..env.des import PM_CANDS          # ⚠️ `PM_CANDS` 的**唯一定义处**在 `env/des.py`（动作码语义）
from ..env.rule_views import LAView, PAView
from ..env.snapshot import Snapshot
from ..nn.features import NormContext
from ..nn.state_emb import build_tok
from .agents import HierarchicalPolicy, split_pa_rule
from .rules import (LA_RULE_NAMES, N_LA_RULES, N_PA_RULES, SIGMA_DEFAULT,
                    apply_pa_rule, decode_param, la_rule_policy, pm_rule_policy)


class Decision(NamedTuple):
    """一个决策点的**全部打分输入**（采样时冻结）——事后据此带梯度重算 logp。

    存的是**当时的**上下文，不是事后重建的：仿真状态在变，用最新快照重算等于把策略输入
    换成另一个状态（决策与 logp 不再对应）。与 v1 同名同义，**只是不要 `cand_feat`**（见模块 docstring）。
    """

    kind: str                       # "MA" | "PA" | "LA" | "M"
    tok: np.ndarray                 # 当时的 token 特征 (N, F_MAX)（**恒为 numpy / 主机内存**）
    seg: tuple[int, ...]            # 当时的段长
    rule_pa: int                    # **当时生效的**三个规则号
    rule_la: int                    # ⚠️ MA 那条 = **它自己刚采的**；下层那条 = **上层给的**
    rule_ra: int
    param_u: np.ndarray | None      # **无界动作 `u`**（下层）—— `logp` 算在它上面（设计 §6.1）。MA = `None`
    logp: float                     # 采样时的 logp（MA = 三个分类的 `log_softmax` 之和）
    la_ctx: np.ndarray | None = None  # 仅 LA：逐决策任务上下文 (LA_CTX_DIM,)


def net_device(net) -> torch.device:
    """`net` 的参数在哪个设备上 —— **设备跟随策略**（v1 `policy.device` 的同一口径）。

    ⚠️ **2026-10-09 补**：v2 此前**完全没有设备支持**（`hrl/` 里 0 处 `cuda`/`device`）
    ⟹ 模型恒跑在 CPU，且**编码器里那条 CUDA 图快路永远触发不了**
    （`LayoutEncoder._graph_ok` 的头一条就是 `tok_feat.is_cuda`）。
    **v1 有这套东西（`policy.device` · `forward_enc` 搬运 · `worker_device`），v2 重写时丢了。**
    实测代价：单次前向 CPU 6.5 ms vs 图上 0.383 ms —— 见 progress-log §99。

    调用方只需 `net.to(device)`，**不需要往每个函数传设备**。
    """
    return next(net.parameters()).device


class Recorder:
    """按**发生顺序**收集 `Decision`。**顺序即链**——事后按同一顺序重算 logp。"""

    def __init__(self) -> None:
        self.decisions: list[Decision] = []

    def __len__(self) -> int:
        return len(self.decisions)

    def kinds(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.decisions:
            out[d.kind] = out.get(d.kind, 0) + 1
        return out


class NetPolicy:
    """把 `HierarchicalPolicy` 包成仿真要的那组回调，并记录决策。

    ⚠️ **采样在 `torch.no_grad()` 下做**（rollout 不该建图）；logp 记的是**采样时**的值，
    带梯度的重算在 `recompute_logp` 里另起一次前向（与 v1 的 `_decision_logp_terms` 同一套形状）。
    """

    def __init__(self, net: HierarchicalPolicy, inst, layout, ctx: NormContext,
                 *, rules: dict[str, int] | None = None, greedy: bool = False,
                 seed: int = 0, record: bool = True, dt_events: int = 20,
                 freeze_rules: bool = False, sigma: float = SIGMA_DEFAULT,
                 pm_interval: float = 120.0):
        self.net, self.inst, self.layout, self.ctx = net, inst, layout, ctx
        self.greedy = bool(greedy)
        # 顶层的定时器步长（§10②）。**存在策略对象上**：`run_gated` 与
        # `hrl.replay.run_branch` 都从它取，避免"两处各写一个 ΔT"。
        self.dt_events = int(dt_events)
        # ⚠️ **`freeze_rules`**：顶层**照常被触发**（`ma_decisions` 照增、`stop_after_ma` 照能用，
        # 因为停表靠的就是这个计数），**但不重新采规则** —— 冻结在构造时给的 `rules` 上。
        # 为什么需要它：设计 §6 阶段 B 的"条件在**固定的** `rule*`"要的是"顶层不再决策"，
        # 而停表/分段要的是"顶层**照常发生**"。**两件事分开** ⟹ 这个开关。
        self.freeze_rules = bool(freeze_rules)
        # 连续动作的探索标准差（**固定、不学** —— 见 `rules.SIGMA_DEFAULT`）
        self.sigma = float(sigma)
        # ⑫ 的保养间隔 —— `policy_m` 要把**比例** `A` 换算成**绝对阈值** `A·pm_interval`
        self.pm_interval = float(pm_interval)
        self.rec = Recorder() if record else None
        # ⚠️ **采样器必须与 logits 同设备**（`torch.multinomial` 要求 generator 与输入一致）。
        # CPU 档 `torch.Generator(device='cpu')` 与旧写法 `torch.Generator()` **逐位相同**
        # ⟹ 默认档一位不变（v1 `group_rel.py:735` 同口径）。
        self.device = net_device(net)
        self.gen = torch.Generator(device=self.device).manual_seed(int(seed))
        # 当前生效的三个规则号（0 = PA / 1 = LA / 2 = RA 的默认档）
        self.cur: dict[str, int] = ({"PA": 0, "LA": 0, "RA": 0} if rules is None
                                    else {k: int(v) for k, v in rules.items()})
        self.ma_reasons: list[str] = []
        # 每个顶层点的**累计读数**（与 `ma_reasons` 同长、同序）——段回报取增量用
        self.ma_counters: list = []

    # ── 内部：一次前向（no_grad，只为采样）──
    def _cur_rules(self) -> dict[str, torch.Tensor]:
        """**当前生效**的三个规则号 → `forward` 要的逐行张量（恒 1 行）。"""
        return {k: torch.as_tensor([int(self.cur[k])], device=self.device)
                for k in self.cur}

    def _forward(self, snap: Snapshot, la_ctx: np.ndarray | None = None):
        """一次前向 → `(PolicyOut, tok, seg)`。

        ⚠️ **必须传 `rules=self._cur_rules()`**（2026-10-10 修一条实测踩过的 bug）。

        为什么不传不行：`HierarchicalPolicy.forward(rules=None)` 会按
        **`rule_logits.argmax()`** 去路由 `param_mu`（走哪个参数 head）。而本类里
        **规则是 `policy_ma` 用 `self.gen` 采的**（`self.cur`），两者**一般不同**
        ⟹ **参数与规则错位**：`apply_pa_rule(self.cur["PA"], …, lam)` 用的是采样的规则，
        而 `lam` 却来自**另一条规则**的 head。**这正是设计 §13.3 ③ 点名的那件事。**

        ⚠️ **为什么之前没被测出来**：
        - `greedy=True` 时 `policy_ma` 走 argmax ⟹ `self.cur` **恰好就是** argmax ⟹ 两条路一致；
        - `greedy=False` 时错位是**确定性**的 ⟹ "同 seed 复现"两边错到同一处 ⟹ 仍然相同。

        ⚠️ **附带效果**：传了 `rules` ⟹ `forward` 不再走
        `Categorical(logits).sample()`（那条路用**全局** torch RNG，不受 `self.gen` 控制）。
        不传 `rules` 时每次前向都推进全局随机流，`flat_rollout` 连调两次读数就分岔
        ⟹ 那是**症状**；本条才是根因。

        ⚠️ `policy_ma` 调本方法时 `self.cur` 还是**旧的** —— 但它只读 `rule_logits`
        （不受 `rules` 影响）✓ 无害。
        """
        tok, seg = build_tok(snap, self.inst, self.layout, self.ctx)
        # ⚠️ **搬到 `policy.device`**（设备跟随策略，见 `net_device`）。CPU 档是恒等操作。
        t = torch.as_tensor(tok).unsqueeze(0).to(self.device)
        lc = (None if la_ctx is None
              else torch.as_tensor(la_ctx, dtype=torch.float32)[None].to(self.device))
        with torch.no_grad():
            return (self.net(t, seg, rules=self._cur_rules(), la_ctx=lc, hard=True),
                    tok, seg)

    def _categorical(self, row: torch.Tensor) -> tuple[int, float]:
        """**分类头**取一个动作 → `(下标, logp)`。`greedy` = argmax。"""
        prob = torch.softmax(row, dim=-1)
        a = (int(torch.argmax(row)) if self.greedy
             else int(torch.multinomial(prob, 1, generator=self.gen)))
        return a, float(torch.log(prob[a].clamp_min(1e-12)))

    def _sample_param(self, mu_row: torch.Tensor, kind: str, tok, seg,
                      la_ctx=None) -> np.ndarray:
        """**连续头**采样一组参数：`u = μ + σ·ε`（`greedy` 时 **`u = μ`，不采样**）。

        ⚠️ **`greedy` 时取 `μ`** —— 那是连续策略的"确定性模式"。
        **搞错（照旧采样）的话，阶段 A 的 `σ_G` 会混进下层自己的采样噪声**
        （设计 §13.3 ②）。

        ⚠️ **`logp` 算在 `u` 上**（设计 §13.3 ①）—— `u` 才是被采样的动作；
        `param = decode_param(u)` 只是"动作怎么解释"，**不参与 `logp`**。
        """
        mu = mu_row[0]
        if self.greedy:
            u = mu
        else:
            u = mu + self.sigma * torch.randn(mu.shape, generator=self.gen,
                                              device=mu.device)
        var = float(self.sigma) ** 2
        lp = float((-(u - mu) ** 2 / (2.0 * var) - 0.5 * math.log(2.0 * math.pi * var)).sum())
        arr = np.asarray(u.detach().cpu().numpy(), dtype=np.float32)
        if self.rec is not None:
            self.rec.decisions.append(Decision(
                kind=kind, tok=tok, seg=tuple(seg),
                rule_pa=self.cur["PA"], rule_la=self.cur["LA"], rule_ra=self.cur["RA"],
                param_u=arr, logp=lp,
                la_ctx=None if la_ctx is None else np.asarray(la_ctx, dtype=np.float32)))
        return arr

    # ── 顶层（MA）：采 `g` ──
    def policy_ma(self, snap: Snapshot, reason: str,
                  counters=None) -> tuple[int, int, int]:
        """采**一组规则**（PA / LA / RA 各一个）并**就地生效**。

        `counters`（`env.des.StepCounters`）= 该顶层点的**累计读数**；本方法把它连同
        规则一起记下 —— 段回报靠"两条累计读数相减"（见 `hrl/replay.py`）。
        """
        self.ma_reasons.append(str(reason))
        self.ma_counters.append(counters)
        if self.freeze_rules:
            c = self.cur                         # 冻结：顶层照常触发，但不改规则
            return (c["PA"], c["LA"], c["RA"])
        out, tok, seg = self._forward(snap)
        sel: dict[str, int] = {}
        logps: list[float] = []
        for k, lg in out.rule_logits.items():
            a, lp = self._categorical(lg[0])
            sel[k] = a
            logps.append(lp)
        self.cur = sel
        if self.rec is not None:
            self.rec.decisions.append(Decision(
                kind="MA", tok=tok, seg=tuple(seg),
                rule_pa=sel["PA"], rule_la=sel["LA"], rule_ra=sel["RA"],
                param_u=None, logp=float(sum(logps))))
        return (sel["PA"], sel["LA"], sel["RA"])

    # ── PA：12 条规则选 1 → `apply_pa_rule` 落到 (池下标, 机台) ──
    def policy_pa(self, view: PAView, pool, tie_rng=None):
        out, tok, seg = self._forward(view.snap)
        u = self._sample_param(out.param_mu["PA"], "PA", tok, seg)
        # `u` 的布局 = [作业侧 2 个, 机台侧 1 个]（见 `agents.PA_PARAM_TOTAL`）
        lam_job = (decode_param(float(u[0])), decode_param(float(u[1])))
        lam_mach = (decode_param(float(u[2])),)
        return apply_pa_rule(self.cur["PA"], view, pool, tie_rng, lam_job, lam_mach)

    # ── LA：3 条规则选 1 → `la_rule_policy` 落到车号 ──
    def policy_l(self, view: LAView, cand_v):
        lc = self._la_ctx(view)
        out, tok, seg = self._forward(view.snap, lc)
        u = self._sample_param(out.param_mu["LA"], "LA", tok, seg, la_ctx=lc)
        return int(la_rule_policy(self.cur["LA"], (decode_param(float(u[0])),))(view, cand_v))

    def _la_ctx(self, view: LAView) -> np.ndarray:
        """LA 的**逐决策任务上下文**（≡ `agents.LA_CTX_DIM`）：
        `(起点节点, 终点节点, 可运输时刻 − now, 工序号)`，逐项归一。"""
        d = max(float(self.ctx.bbox_diag), 1e-9)
        m = max(float(self.ctx.m_ref), 1e-9)
        return np.array([view.frm_node / d, view.to_node / d,
                         float(np.clip((view.ready_at - view.now) / m, -2.0, 2.0)),
                         float(np.clip(view.oi / 8.0, -1.0, 4.0))], dtype=np.float32)

    # ── RA 的头（**⑪ 去后只剩保养一个**）──
    # ⚠️ **2026-10-10：⑪ 去后 `policy_c`（充电头）已删** —— 车队无电量、无充电桩。
    def policy_m(self, snap: Snapshot, mach: int, cands):
        # ⚠️ **v3 里 `mach` 用得上** —— `pm_rule_policy` 拿它取该机台的役龄/队列
        # （v2 是 `del mach`，因为那时的保养头只看全局池化）。
        out, tok, seg = self._forward(snap)
        u = self._sample_param(out.param_mu["RA"], "M", tok, seg)
        # 参数是**比例** `A ∈ (0,1)` ⟹ 绝对阈值 = `A · pm_interval`（设计 §4.5）
        thr = decode_param(float(u[0])) * self.pm_interval
        return int(pm_rule_policy(self.cur["RA"], thr)(snap, int(mach), cands))


def gated_by_constraints(pol, constraints) -> dict:
    """按**约束档**决定哪些决策回调挂上 —— **约束关掉 ⟹ 那个决策点不存在 ⟹ 回调必须是 `None`。**

    ⚠️ **为什么必须有这一层**：`run_gated` 对"给了回调、但该约束关着"是**显式报错**
    （"给策略一个死动作只会污染链 logp"，见 `env/des.SimWorld.run_gated`）。
    而训练器**无条件**挂全部回调 ⟹ **约束组消融里关了 ⑫ 的那几组一跑就炸**。

    **实测踩过**：`b_ablation` 的冒烟在 `--arms None` 上当场 `ValueError` ——
    它关掉了 `maintenance`，而 `base_chain` 照旧传 `policy_m`。

    ⚠️ 返回的是**关键字参数字典**，调用方 `**` 展开进 `run_gated`。
    ⚠️ 目前只有 `maintenance` 受约束门控（⑪ 删除后充电头不存在；
    剩下 PA / LA / MA 三个回调与约束开关无关）。
    """
    on_m = constraints is None or bool(constraints.maintenance)
    return {"policy_m": pol.policy_m if on_m else None}


# `PM_CANDS` / `LA_RULE_NAMES` 由本模块转出（explicit re-export）——调用方从一处取就够了
__all__ = ["Decision", "LA_RULE_NAMES", "N_LA_RULES", "N_PA_RULES", "NetPolicy",
           "PM_CANDS", "Recorder", "gated_by_constraints", "split_pa_rule", "torch"]
