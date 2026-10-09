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
| **C / M** | RA 的两个头（AGV 待命 / 机台完工且下一件未上机） | `1+n_桩` / 2 | —— |

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

import numpy as np
import torch

from ..env.des import PM_CANDS          # ⚠️ `PM_CANDS` 的**唯一定义处**在 `env/des.py`（动作码语义）
from ..env.rule_views import LAView, PAView
from ..env.snapshot import Snapshot
from ..nn.features import NormContext
from ..nn.state_emb import build_tok
from .agents import G_BINS, G_DIM, HierarchicalPolicy, g_to_vec
from .rules import LA_RULE_NAMES, N_LA_RULES, N_PA_RULES, apply_pa_rule, la_rule_policy


class Decision(NamedTuple):
    """一个决策点的**全部打分输入**（采样时冻结）——事后据此带梯度重算 logp。

    存的是**当时的**上下文，不是事后重建的：仿真状态在变，用最新快照重算等于把策略输入
    换成另一个状态（决策与 logp 不再对应）。与 v1 同名同义，**只是不要 `cand_feat`**（见模块 docstring）。
    """

    kind: str                       # "MA" | "PA" | "LA" | "C" | "M"
    tok: np.ndarray                 # 当时的 token 特征 (N, F_MAX)（**恒为 numpy / 主机内存**）
    seg: tuple[int, ...]            # 当时的段长
    g_bins: tuple[int, int, int]    # 当时生效的 `g`（MA 自己那条存它采到的档）
    action: int                     # 所取的动作。MA = 三维档压平（b0*16+b1*4+b2）；其余 = 规则/动作码下标
    logp: float                     # 采样时的 logp（MA = 三维之和）
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
                 *, g_bins: tuple[int, int, int] | None = None, greedy: bool = False,
                 seed: int = 0, record: bool = True, dt_events: int = 20,
                 freeze_g: bool = False):
        self.net, self.inst, self.layout, self.ctx = net, inst, layout, ctx
        self.greedy = bool(greedy)
        # 顶层的定时器步长（§10②）。**存在策略对象上**：`run_gated` 与
        # `hrl.replay.run_branch` 都从它取，避免"两处各写一个 ΔT"。
        self.dt_events = int(dt_events)
        # ⚠️ **`freeze_g`**：顶层**照常被触发**（`ma_decisions` 照增、`stop_after_ma` 照能用，
        # 因为停表靠的就是这个计数），**但不重新采 `g`** —— `g` 冻结在构造时给的值上。
        # 为什么需要它：设计 §4.3 的"条件在**固定的** `g\*`"要的是"顶层不再决策"，
        # 而停表/分段要的是"顶层**照常发生**"。**两件事分开** ⟹ 这个开关。
        self.freeze_g = bool(freeze_g)
        self.rec = Recorder() if record else None
        # ⚠️ **采样器必须与 logits 同设备**（`torch.multinomial` 要求 generator 与输入一致）。
        # CPU 档 `torch.Generator(device='cpu')` 与旧写法 `torch.Generator()` **逐位相同**
        # ⟹ 默认档一位不变（v1 `group_rel.py:735` 同口径）。
        self.device = net_device(net)
        self.gen = torch.Generator(device=self.device).manual_seed(int(seed))
        # 当前生效的目标：给定时 = **条件在固定 `g`**（阶段 B / 执行）；`None` = 由 MA 现采
        self.g_bins: tuple[int, int, int] | None = (None if g_bins is None
                                                    else (int(g_bins[0]), int(g_bins[1]),
                                                          int(g_bins[2])))
        self.ma_reasons: list[str] = []
        # 每个顶层点的**累计读数**（与 `ma_reasons` 同长、同序）——段回报取增量用
        self.ma_counters: list = []

    # ── 内部：一次前向（no_grad，只为采样）──
    def _forward(self, snap: Snapshot, la_ctx: np.ndarray | None = None):
        tok, seg = build_tok(snap, self.inst, self.layout, self.ctx)
        # ⚠️ **搬到 `policy.device`**（设备跟随策略，见 `net_device`）。CPU 档是恒等操作。
        t = torch.as_tensor(tok).unsqueeze(0).to(self.device)
        gb = (None if self.g_bins is None
              else torch.as_tensor([list(self.g_bins)], device=self.device))
        lc = (None if la_ctx is None
              else torch.as_tensor(la_ctx, dtype=torch.float32)[None].to(self.device))
        with torch.no_grad():
            return self.net(t, seg, gb, la_ctx=lc, hard=self.greedy), tok, seg

    def _pick(self, logits: torch.Tensor, kind: str, tok, seg, action_override=None,
              la_ctx=None, slot: int | None = None) -> int:
        """从 `(1, n)` 的 logits 里取一个动作（greedy=argmax，否则按分布采样），并记录。"""
        row = logits[0] if slot is None else logits[0, slot]
        prob = torch.softmax(row, dim=-1)
        a = int(torch.argmax(row)) if self.greedy else int(torch.multinomial(prob, 1, generator=self.gen))
        logp = float(torch.log(prob[a].clamp_min(1e-12)))
        if self.rec is not None:
            gb = self.g_bins if self.g_bins is not None else (0, 0, 0)
            self.rec.decisions.append(Decision(
                kind=kind, tok=tok, seg=tuple(seg), g_bins=tuple(gb),
                action=int(a), logp=logp,
                la_ctx=None if la_ctx is None else np.asarray(la_ctx, dtype=np.float32)))
        return a

    # ── 顶层（MA）：采 `g` ──
    def policy_ma(self, snap: Snapshot, reason: str,
                  counters=None) -> tuple[int, int, int]:
        """采一组 `g`（3 维各 4 档）并**就地生效**（下层下一次决策就用它）。

        `counters`（`env.des.StepCounters`）= 该顶层点的**累计读数**；本方法把它连同
        `g` 一起记下 —— 段回报靠"两条累计读数相减"（见 `hrl/replay.py`）。
        """
        self.ma_reasons.append(str(reason))
        self.ma_counters.append(counters)
        if self.freeze_g and self.g_bins is not None:
            return self.g_bins                      # 冻结：顶层照常触发，但不改 `g`
        out, tok, seg = self._forward(snap)
        gl = out.g_logits[0]                      # (3, 4)
        bins, logps = [], []
        for i in range(G_DIM):
            prob = torch.softmax(gl[i], dim=-1)
            b = (int(torch.argmax(gl[i])) if self.greedy
                 else int(torch.multinomial(prob, 1, generator=self.gen)))
            bins.append(b)
            logps.append(float(torch.log(prob[b].clamp_min(1e-12))))
        self.g_bins = (bins[0], bins[1], bins[2])
        if self.rec is not None:
            flat = bins[0] * (G_BINS ** 2) + bins[1] * G_BINS + bins[2]
            self.rec.decisions.append(Decision(
                kind="MA", tok=tok, seg=tuple(seg), g_bins=tuple(self.g_bins),
                action=flat, logp=float(sum(logps))))
        return self.g_bins

    # ── PA：12 条规则选 1 → `apply_pa_rule` 落到 (池下标, 机台) ──
    def policy_pa(self, view: PAView, pool, tie_rng=None):
        out, tok, seg = self._forward(view.snap)
        rule = self._pick(out.pa, "PA", tok, seg)
        return apply_pa_rule(rule, view, pool, tie_rng)

    # ── LA：3 条规则选 1 → `la_rule_policy` 落到车号 ──
    def policy_l(self, view: LAView, cand_v):
        lc = self._la_ctx(view)
        out, tok, seg = self._forward(view.snap, lc)
        rule = self._pick(out.la, "LA", tok, seg, la_ctx=lc)
        return int(la_rule_policy(rule)(view, cand_v))

    def _la_ctx(self, view: LAView) -> np.ndarray:
        """LA 的**逐决策任务上下文**（≡ `agents.LA_CTX_DIM`）：
        `(起点节点, 终点节点, 可运输时刻 − now, 工序号)`，逐项归一。"""
        d = max(float(self.ctx.bbox_diag), 1e-9)
        m = max(float(self.ctx.m_ref), 1e-9)
        return np.array([view.frm_node / d, view.to_node / d,
                         float(np.clip((view.ready_at - view.now) / m, -2.0, 2.0)),
                         float(np.clip(view.oi / 8.0, -1.0, 4.0))], dtype=np.float32)

    # ── RA 的两个头 ──
    def policy_c(self, snap: Snapshot, aid: int, cands):
        del aid
        out, tok, seg = self._forward(snap)
        return self._pick(out.ra_charge, "C", tok, seg)      # 与 `charge_cands` 同序

    def policy_m(self, snap: Snapshot, mach: int, cands):
        del mach
        out, tok, seg = self._forward(snap)
        return self._pick(out.ra_maint, "M", tok, seg)       # 与 `PM_CANDS` 同序


# `PM_CANDS` / `LA_RULE_NAMES` 由本模块转出（explicit re-export）——调用方从一处取就够了
__all__ = ["Decision", "LA_RULE_NAMES", "N_LA_RULES", "N_PA_RULES", "NetPolicy",
           "PM_CANDS", "Recorder", "g_to_vec", "torch"]
