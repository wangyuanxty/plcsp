# -*- coding: utf-8 -*-
"""**把策略接到仿真上**（v3 top-K 架构）——网络 → `policy_*` 回调，并**冻结决策**供重算。

每类下层决策的动作空间现在是**规则筛出的候选集**：

| kind | 候选集 | 动作 |
|---|---|---|
| **PA** | `pa_candidates`：复合规则的前 K=16 个 `(工件, 机台)` | 候选下标 → `(池下标, 机台)` |
| **LA** | `la_candidates`：规则的前 K=6 台车 | 候选下标 → 车号 |
| **RA** | `ra_candidates`：规则过滤后的 `{NOW, DEFER}` | 0 / 1 |
| **MA** | —— | 三个规则号（不变） |

⚠️ **logp 一律在候选集内归一化**：`log_softmax(logits)[chosen]`。
⚠️ **`K_eff == 1` 的强制决策不抽签、不推进 `self.gen`** —— 否则后续采样流整体移位。
⚠️ **greedy 的分数并列**用逐决策 `topk_rng` 打破，**不按实体号**（同 §4.28 的纪律）。
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

from ..env.des import PM_CANDS          # ⚠️ `PM_CANDS` 的**唯一定义处**在 `env/des.py`
from ..env.rule_views import LAView, PAView
from ..env.snapshot import Snapshot
from ..nn.features import NormContext
from ..nn.state_emb import build_tok
from .agents import HierarchicalPolicy
from .candidates import (K_LA, K_PA, la_candidates, pa_candidates,
                         ra_candidates, ra_features, topk_rng)
from .rules import (LA_RULE_NAMES, N_LA_RULES, N_PA_RULES, apply_pa_rule,
                    la_rule_policy)


class CandRec(NamedTuple):
    """一个决策点的**候选集**（冻结在决策那一刻，供带梯度重算 logp）。"""

    rows_a: np.ndarray
    """PA: 作业 token 行 `n_m + job` `(K,)`；LA: 车行 `n_m+n_jobs+agv` `(K,)`；
    RA: 机台行 `(1,)`。"""
    rows_b: np.ndarray | None
    """仅 PA：机台 token 行 `(K,)`。"""
    feats: np.ndarray
    """PA `(K,9)` / LA `(K,4)` / RA `(1,5)`。"""
    mask: np.ndarray
    """PA/LA `(K,)`（恒 True）；RA `(2,)` 允许集。"""
    act: int
    """选中下标；RA 为动作码 0/1。"""


class Decision(NamedTuple):
    """一个决策点的**全部打分输入**（采样时冻结）——事后据此带梯度重算 logp。"""

    kind: str                       # "MA" | "PA" | "LA" | "M"
    tok: np.ndarray                 # 当时的 token 特征 (N, F_MAX)（**恒为 numpy**）
    seg: tuple[int, ...]            # 当时的段长
    rule_pa: int                    # **当时生效的**三个规则号
    rule_la: int
    rule_ra: int
    logp: float                     # 采样时的 logp（K 内归一化）
    cand: CandRec | None = None     # 下层：候选集；MA = None
    act: tuple[int, ...] | None = None
    """回放用的**具体动作**：PA `(pool_idx, machine)` / LA `(agv,)` /
    RA `(action_code,)` / MA = None。"""
    la_ctx: np.ndarray | None = None  # 仅 LA：逐决策任务上下文 (LA_CTX_DIM,)


def net_device(net) -> torch.device:
    """`net` 的参数在哪个设备上 —— **设备跟随策略**（v1 `policy.device` 的同一口径）。"""
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
    """把 `HierarchicalPolicy` 包成仿真要的那组回调，并记录决策。"""

    def __init__(self, net: HierarchicalPolicy, inst, layout, ctx: NormContext,
                 *, rules: dict[str, int] | None = None, greedy: bool = False,
                 seed: int = 0, record: bool = True, dt_events: int = 20,
                 freeze_rules: bool = False, seed_chain: int = 0,
                 force_top1: bool = False,
                 fixed_rules: dict[str, int] | None = None,
                 pa_limit: int | None = None,
                 exact_rule: bool = False,
                 tie_seed: int | None = None):
        self.net, self.inst, self.layout, self.ctx = net, inst, layout, ctx
        self.greedy = bool(greedy)
        # ⚠️ **消融开关**：强制取规则候选的第 0 个（规则 top-1），绕过打分头。
        #    只给诊断用；默认 False ⟹ 逐位不变。
        self.force_top1 = bool(force_top1)
        self.dt_events = int(dt_events)
        self.seed_chain = int(seed_chain)
        self.freeze_rules = bool(freeze_rules)
        # ⚠️ **固定某几个头**（单训 PA 诊断用）：这些键不采样，始终用给定值。
        self.fixed_rules = ({} if fixed_rules is None
                            else {k: int(v) for k, v in fixed_rules.items()})
        bad = set(self.fixed_rules) - {"PA", "LA", "RA"}
        if bad:
            raise ValueError(f"fixed_rules 含未知头 {sorted(bad)}")
        # ⚠️ **限制 PA 的可选规则数**（默认 None = 全部 12 条）。
        #    只影响采样与 logp；`PA_RULE_NAMES` 本身不改。
        if pa_limit is not None:
            pa_limit = int(pa_limit)
            if not 1 <= pa_limit <= N_PA_RULES:
                raise ValueError(
                    f"pa_limit={pa_limit} 越界：应在 1..{N_PA_RULES}")
        self.pa_limit = pa_limit
        # ⚠️ **`exact_rule`（诊断）**：PA/LA 不走 top-K 候选 + 打分头，
        #    直接调用 `rules.apply_pa_rule` / `rules.la_rule_policy`，并**沿用环境给的
        #    `tie_rng`** —— 这才是 CDR 基线的那条执行路径（含随机平局）。
        #    `force_top1` 取的是候选集第 0 个，**不是**规则的 argmin，见 `candidates.py`。
        #    默认 False ⟹ 既有读数逐位不变。
        self.exact_rule = bool(exact_rule)
        # ⚠️ **平局流种子**（2026-10-20）：`None` ⟹ 跟随 `seed_chain`（逐位不变）。
        #    非 None ⟹ 同一场景换平局流（评测精度：一个场景跑 T 条平局流）。
        #    ⚠️ 它**不影响** `self.gen`（策略采样流）与候选构造的 `topk_rng`
        #    —— 候选构造仍由 `seed_chain` 决定，换平局流不改候选集。
        self.tie_seed = None if tie_seed is None else int(tie_seed)
        self.rec = Recorder() if record else None
        self.device = net_device(net)
        self.gen = torch.Generator(device=self.device).manual_seed(int(seed))
        if rules is None:
            self.cur = {"PA": 0, "LA": 0, "RA": 0}
            self.cur.update(self.fixed_rules)
        else:
            self.cur = {k: int(v) for k, v in rules.items()}
        self.ma_reasons: list[str] = []
        self.ma_counters: list = []
        # 逐 kind 的决策序号 —— 候选构造 rng 的 `decision_index`（每局从 0 起）。
        self._n_dec: dict[str, int] = {"PA": 0, "LA": 0, "RA": 0}
        # **一跳前向缓存**：PA 与 LA 在同一个决策点、同一规则下共用一个 `z`。
        # key = (快照 hash, 当前规则)。快照是 frozen dataclass，`hash` 覆盖全部字段。
        self._fwd_key = None
        self._fwd_val = None

    # ── 内部：一次前向（no_grad，只为采样 / 取 z）──
    def _cur_rules(self) -> dict[str, torch.Tensor]:
        return {k: torch.as_tensor([int(self.cur[k])], device=self.device)
                for k in self.cur}

    def _forward(self, snap: Snapshot):
        """一次前向 → `(PolicyOut, tok, seg)`。只出顶层 logits 与编码器 `z`。

        ⚠️ **一跳缓存**：PA 与 LA 在同一决策点、同一规则下，快照相同 ⟹ 编码器
        `z` 与 token 逐位相同。缓存把 LA 的那次前向省掉（实测 LA 决策数 ≈ PA + 回站数）。
        缓存键**含 `self.cur`**：MA 换规则后必须重算 `rule_sel`。
        """
        key = (hash(snap), tuple(sorted(self.cur.items())))
        if key == self._fwd_key:
            return self._fwd_val
        tok, seg = build_tok(snap, self.inst, self.layout, self.ctx)
        t = torch.as_tensor(tok).unsqueeze(0).to(self.device)
        with torch.no_grad():
            val = (self.net(t, seg, rules=self._cur_rules(), hard=True), tok, seg)
        self._fwd_key, self._fwd_val = key, val
        return val

    def _categorical(self, row: torch.Tensor) -> tuple[int, float]:
        """**顶层（MA）**分类头取一个动作 → `(下标, logp)`。`greedy` = argmax。"""
        prob = torch.softmax(row, dim=-1)
        a = (int(torch.argmax(row)) if self.greedy
             else int(torch.multinomial(prob, 1, generator=self.gen)))
        return a, float(torch.log(prob[a].clamp_min(1e-12)))

    def _pick_candidate(self, row: torch.Tensor,
                        rng: np.random.Generator) -> tuple[int, float]:
        """候选集内取一个动作 → `(下标, logp)`。

        - `n_valid == 1`：强制；`logp=0`；**不抽签**（不推进 `self.gen`）。
        - `greedy=True`：argmax；**分数并列**用逐决策 `rng` 均匀抽。
        - `greedy=False`：按 `softmax(row)` 用 `self.gen` 抽。
        """
        finite = torch.isfinite(row)
        n_valid = int(finite.sum())
        if n_valid <= 0:
            raise ValueError("候选集为空 —— 策略不能在空集上取动作")
        if n_valid == 1:
            a = int(torch.nonzero(finite, as_tuple=False)[0, 0])
            return a, 0.0
        lp = torch.log_softmax(row, dim=-1)
        if self.greedy:
            best = torch.max(row[finite])
            tied = torch.nonzero(row == best, as_tuple=False).flatten().tolist()
            a = (int(tied[int(rng.integers(len(tied)))]) if len(tied) > 1
                 else int(tied[0]))
        else:
            a = int(torch.multinomial(torch.softmax(row, dim=-1), 1,
                                      generator=self.gen))
        return a, float(lp[a])

    def _record(self, kind: str, tok, seg, lp: float, cand: CandRec,
                act: tuple[int, ...], la_ctx: np.ndarray | None = None) -> None:
        if self.rec is None:
            return
        self.rec.decisions.append(Decision(
            kind=kind, tok=tok, seg=tuple(seg),
            rule_pa=self.cur["PA"], rule_la=self.cur["LA"], rule_ra=self.cur["RA"],
            logp=float(lp), cand=cand, act=act,
            la_ctx=None if la_ctx is None else np.asarray(la_ctx, dtype=np.float32)))

    # ── 顶层（MA）：采一组规则 ──
    def policy_ma(self, snap: Snapshot, reason: str,
                  counters=None) -> tuple[int, int, int]:
        """采**一组规则**（PA / LA / RA 各一个）并**就地生效**。"""
        self.ma_reasons.append(str(reason))
        self.ma_counters.append(counters)
        if self.freeze_rules:
            c = self.cur
            return (c["PA"], c["LA"], c["RA"])
        out, tok, seg = self._forward(snap)
        if getattr(self.net, "seq_ma", False):
            return self._policy_ma_seq(out, tok, seg)
        sel: dict[str, int] = {}
        logps: list[float] = []
        for k, lg in out.rule_logits.items():
            if k in self.fixed_rules:
                # ⚠️ 固定的头不采样、不进 logp：单训 PA 时 LA/RA 不参与梯度。
                sel[k] = self.fixed_rules[k]
                continue
            row = lg[0]
            if k == "PA" and self.pa_limit is not None:
                # ⚠️ 只在前 `pa_limit` 条规则里采样；`PA_RULE_NAMES` 不变。
                row = row[:self.pa_limit]
            a, lp = self._categorical(row)
            sel[k] = a
            logps.append(lp)
        self.cur = sel
        if self.rec is not None:
            self.rec.decisions.append(Decision(
                kind="MA", tok=tok, seg=tuple(seg),
                rule_pa=sel["PA"], rule_la=sel["LA"], rule_ra=sel["RA"],
                logp=float(sum(logps)), cand=None, act=None))
        return (sel["PA"], sel["LA"], sel["RA"])

    def _policy_ma_seq(self, out, tok, seg) -> tuple[int, int, int]:
        """**顺序条件 MA**：先采 PA，再用采样到的 PA 条件采 LA。"""
        sel: dict[str, int] = {}
        logps: list[float] = []
        # ── PA ──
        if "PA" in self.fixed_rules:
            pa = self.fixed_rules["PA"]
        else:
            row = out.rule_logits["PA"][0]
            if self.pa_limit is not None:
                row = row[:self.pa_limit]
            pa, lp = self._categorical(row)
            logps.append(lp)
        sel["PA"] = pa
        # ── LA：条件在采样到的 PA 上 ──
        if "LA" in self.fixed_rules:
            la = self.fixed_rules["LA"]
        else:
            t = torch.as_tensor(tok).unsqueeze(0).to(self.device)
            rt = {k: torch.as_tensor([int(sel.get(k, self.cur[k]))],
                                     device=self.device) for k in ("PA", "LA", "RA")}
            rt["PA"] = torch.as_tensor([int(pa)], device=self.device)
            with torch.no_grad():
                o2 = self.net(t, seg, rules=rt, hard=True)
            la, lp = self._categorical(o2.rule_logits["LA"][0])
            logps.append(lp)
        sel["LA"] = la
        # ── RA：先不加 RA；固定的 RA 直接取，否则用独立头 ──
        if "RA" in self.fixed_rules:
            ra = self.fixed_rules["RA"]
        else:
            ra, lp = self._categorical(out.rule_logits["RA"][0])
            logps.append(lp)
        sel["RA"] = ra
        self.cur = sel
        if self.rec is not None:
            self.rec.decisions.append(Decision(
                kind="MA", tok=tok, seg=tuple(seg),
                rule_pa=sel["PA"], rule_la=sel["LA"], rule_ra=sel["RA"],
                logp=float(sum(logps)), cand=None, act=None))
        return (sel["PA"], sel["LA"], sel["RA"])
    # ── 精确规则执行（诊断档）：与 CDR 基线**同一条代码路径** ──
    def _exact_pa(self, view: PAView, pool, tie_rng):
        """`apply_pa_rule` + **环境给的** `tie_rng` —— CDR 就是这个。

        ⚠️ **必须记录决策**（2026-10-20 修）：分支回放（`run_branch`）靠
        `Decision.act` 前缀重演；不记 ⟹ 前缀里没有 PA，j≥1 直接崩
        （`回放的决策序列用完了`）。
        ⚠️ `logp` 记 **0.0**：本档不训练下层，这个数**没有分布含义**。
        """
        if not pool:
            raise ValueError("PA 池为空 —— `run_gated` 不该在空池上派工")
        pi, m = (int(x) for x in apply_pa_rule(self.cur["PA"], view, pool, tie_rng))
        _out, tok, seg = self._forward(view.snap)
        self._record("PA", tok, seg, 0.0, None, (pi, m))
        return pi, m

    def _exact_la(self, view: LAView, cand_v):
        """`la_rule_policy` 同款：规则打分 + `view.tie_rng` 随机平局。"""
        if not cand_v:
            raise ValueError("LA 候选集为空 —— 派车决策点不该出现（全车队恒可候选）")
        a = int(la_rule_policy(self.cur["LA"])(view, cand_v))
        _out, tok, seg = self._forward(view.snap)
        self._record("LA", tok, seg, 0.0, None, (a,))
        return a

    def policy_pa(self, view: PAView, pool, tie_rng=None):
        if self.exact_rule:
            return self._exact_pa(view, pool, tie_rng)
        idx = self._n_dec["PA"]
        self._n_dec["PA"] += 1
        rng = topk_rng(self.seed_chain, "PA", idx)
        cands = pa_candidates(view, pool, self.cur["PA"],
                              k=K_PA, m_ref=self.ctx.m_ref, rng=rng)
        if not cands:
            raise ValueError(
                "PA 候选集为空 —— 池非空且每个池成员都应有候选机台；"
                "空集说明池过滤或 `alts` 有不一致")
        out, tok, seg = self._forward(view.snap)
        rows_a = torch.as_tensor(
            np.asarray([self.ctx.n_m + c.job for c in cands], dtype=np.int64)
        )[None].to(self.device)
        rows_b = torch.as_tensor(
            np.asarray([c.mach for c in cands], dtype=np.int64)
        )[None].to(self.device)
        feats = torch.as_tensor(np.stack([c.feats for c in cands]))[None].to(self.device)
        rule = torch.as_tensor([int(self.cur["PA"])], dtype=torch.long,
                               device=self.device)
        with torch.no_grad():
            logits = self.net.score_pa(out.z, rows_a, rows_b, feats, rule)[0]
            if self.force_top1:
                a = 0
                lp = (0.0 if logits.numel() == 1
                      else float(torch.log_softmax(logits, dim=-1)[0]))
            else:
                a, lp = self._pick_candidate(logits, rng)
        c = cands[a]
        self._record("PA", tok, seg, lp, CandRec(
            rows_a=rows_a[0].cpu().numpy(), rows_b=rows_b[0].cpu().numpy(),
            feats=feats[0].cpu().numpy(),
            mask=np.ones(len(cands), dtype=bool), act=a), (int(c.pi), int(c.mach)))
        return int(c.pi), int(c.mach)

    # ── LA：规则筛 top-K → 打分 → 车号 ──
    def policy_l(self, view: LAView, cand_v):
        if self.exact_rule:
            return self._exact_la(view, cand_v)
        idx = self._n_dec["LA"]
        self._n_dec["LA"] += 1
        rng = topk_rng(self.seed_chain, "LA", idx)
        cands = la_candidates(view, self.cur["LA"], cand_v,
                              k=K_LA, m_ref=self.ctx.m_ref, rng=rng)
        if not cands:
            raise ValueError("LA 候选集为空 —— 派车决策点不该出现（全车队恒可候选）")
        lc = self._la_ctx(view)
        out, tok, seg = self._forward(view.snap)
        rows_a = torch.as_tensor(
            np.asarray([self.ctx.n_m + self.ctx.n_jobs + c.agv for c in cands],
                       dtype=np.int64)
        )[None].to(self.device)
        feats = torch.as_tensor(np.stack([c.feats for c in cands]))[None].to(self.device)
        rule = torch.as_tensor([int(self.cur["LA"])], dtype=torch.long,
                               device=self.device)
        lc_t = torch.as_tensor(lc, dtype=torch.float32)[None].to(self.device)
        with torch.no_grad():
            logits = self.net.score_la(out.z, rows_a, feats, rule, lc_t)[0]
            if self.force_top1:
                a = 0
                lp = (0.0 if logits.numel() == 1
                      else float(torch.log_softmax(logits, dim=-1)[0]))
            else:
                a, lp = self._pick_candidate(logits, rng)
        c = cands[a]
        self._record("LA", tok, seg, lp, CandRec(
            rows_a=rows_a[0].cpu().numpy(), rows_b=None,
            feats=feats[0].cpu().numpy(),
            mask=np.ones(len(cands), dtype=bool), act=a), (int(c.agv),), la_ctx=lc)
        return int(c.agv)

    def _la_ctx(self, view: LAView) -> np.ndarray:
        """LA 的**逐决策任务上下文**（≡ `agents.LA_CTX_DIM`）：
        `(起点节点, 终点节点, 可运输时刻 − now, 工序号)`，逐项归一。"""
        d = max(float(self.ctx.bbox_diag), 1e-9)
        m = max(float(self.ctx.m_ref), 1e-9)
        return np.array([view.frm_node / d, view.to_node / d,
                         float(np.clip((view.ready_at - view.now) / m, -2.0, 2.0)),
                         float(np.clip(view.oi / 8.0, -1.0, 4.0))], dtype=np.float32)

    # ── RA：规则过滤 {NOW, DEFER} → 打分 → 动作码 ──
    def policy_m(self, snap: Snapshot, mach: int, cands):
        idx = self._n_dec["RA"]
        self._n_dec["RA"] += 1
        rng = topk_rng(self.seed_chain, "RA", idx)
        allowed, mask = ra_candidates(snap, self.cur["RA"], mach, cands)
        if not allowed:
            raise ValueError("RA 允许集为空 —— 规则过滤后一个动作都不剩")
        out, tok, seg = self._forward(snap)
        row = torch.as_tensor([int(mach)], dtype=torch.long)[None].to(self.device)
        feats = torch.as_tensor(ra_features(snap, int(mach), self.ctx))
        feats = feats[None, None].to(self.device)
        rule = torch.as_tensor([int(self.cur["RA"])], dtype=torch.long,
                               device=self.device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool)[None].to(self.device)
        with torch.no_grad():
            logits = self.net.score_ra(out.z, row, feats, rule, mask_t)[0]
            if self.force_top1:
                a = int(allowed[0])          # NOW if allowed, else DEFER
                lp = (0.0 if len(allowed) == 1
                      else float(torch.log_softmax(logits, dim=-1)[a]))
            else:
                a, lp = self._pick_candidate(logits, rng)
        if a not in cands:                       # 防御性：掩码应已挡住
            raise ValueError(f"RA 选了非候选动作 {a}；候选={tuple(cands)}")
        self._record("M", tok, seg, lp, CandRec(
            rows_a=row[0].cpu().numpy(), rows_b=None,
            feats=feats[0].cpu().numpy(), mask=mask, act=a), (int(a),))
        return int(a)


def gated_by_constraints(pol, constraints) -> dict:
    """按**约束档**决定哪些决策回调挂上。

    ⚠️ 约束关掉 ⟹ 那个决策点不存在 ⟹ 回调必须是 `None`
    （`run_gated` 对"给了回调、但该约束关着"是**显式报错**）。
    """
    on_m = constraints is None or bool(constraints.maintenance)
    return {"policy_m": pol.policy_m if on_m else None}


__all__ = ["CandRec", "Decision", "LA_RULE_NAMES", "N_LA_RULES", "N_PA_RULES",
           "NetPolicy", "PM_CANDS", "Recorder", "gated_by_constraints", "torch"]
