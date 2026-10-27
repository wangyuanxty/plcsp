# -*- coding: utf-8 -*-
"""**精排到底有没有东西可学**：规则的前 K 个候选里，argmin 是不是最好的？

══ 为什么必须测 ══

v3 的架构是「规则粗排 top-K → 打分头精排」。这是搜索里的标准做法。
但精排能学，前提是**粗排留下的候选里确实有比 argmin 更好的**。
这条前提**从没测过**：

- 我们测过「换**规则**」的收益（§37.32，中位 9.2 min）；
- 没测过「在同一条规则的前 K 个候选里换一个」的收益。

若 argmin 恒为最好 ⟹ 精排**没有可学的答案**，不是学不会。

══ 测法 ══

1. 固定规则 + `force_top1` 跑一局（= 候选第 0 个，确定性子种子）。
2. 记下每个 PA 决策点的**候选集**。
3. 抽若干决策点 `t`：在第 `t` 点**只把动作换成候选 `c`**，其余照旧，跑完整局。
4. 报 `c=0` 与 `c=k` 的 makespan 差。

⚠️ **平局被彻底排除**：全档 `greedy` + `force_top1` + 固定 `topk_rng`，
没有任何随机平局流。所以差异**只来自那一个候选**。

⚠️ 这是**单步偏离**，不是可叠加的上界（同 §37.32 的口径）。

══ 用法 ══

`D:/anaconda/python.exe -m plcsp_v3.repro.v3_topk_oracle`
环境变量：`V3_TOPK_INST`（默认 mk01）· `V3_TOPK_STRIDE`（默认 10）· `V3_TOPK_MAX`（默认 8 个点）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from plcsp_v3.env.constraints import ConstraintConfig          # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy              # noqa: E402
from plcsp_v3.hrl.candidates import K_PA, pa_candidates, topk_rng  # noqa: E402
from plcsp_v3.hrl.rollout import NetPolicy                      # noqa: E402
from plcsp_v3.hrl.rules import LA_RULE_NAMES, PA_RULE_NAMES      # noqa: E402
from plcsp_v3.q1_flat import build                               # noqa: E402

INST = os.environ.get("V3_TOPK_INST", "mk01")
STRIDE = int(os.environ.get("V3_TOPK_STRIDE", "10"))
MAXPTS = int(os.environ.get("V3_TOPK_MAX", "8"))
DT = 20
PAIR = PA_RULE_NAMES.index("EAJ+EAM"), LA_RULE_NAMES.index("LATEST_IN_TIME")


class ForcedPA(NetPolicy):
    """在**第 `at` 次** PA 决策上，强制返回候选 `idx`；其余走 `force_top1`。"""

    def __init__(self, *a, at: int | None = None, idx: int = 0, **kw):
        super().__init__(*a, **kw)
        self.at = at
        self.idx = idx
        self.calls = 0
        self.cands_seen: list[int] = []

    def policy_pa(self, view, pool, tie_rng=None):
        forced = self.at is not None and self.calls == self.at
        if forced:
            # 与 `super()` 内部**同一条**候选构造流（同 `seed_chain`、同决策序号）
            idx = self._n_dec["PA"]
            rng = topk_rng(self.seed_chain, "PA", idx)
            cands = pa_candidates(view, pool, self.cur["PA"], k=K_PA,
                                  m_ref=self.ctx.m_ref, rng=rng)
            self.cands_seen.append(len(cands))
        pi, m = super().policy_pa(view, pool, tie_rng)   # 照常推进内部状态
        self.calls += 1
        if not forced:
            return pi, m
        c = cands[min(self.idx, len(cands) - 1)]
        return int(c.pi), int(c.mach)


def _run(net, sp, *, at=None, idx=0):
    rules = {"PA": PAIR[0], "LA": PAIR[1], "RA": 0}
    p = ForcedPA(net, sp.inst, sp.layout, sp.ctx, greedy=True, seed=0,
                 dt_events=DT, fixed_rules=rules, force_top1=True,
                 at=at, idx=idx)
    p.world = sp.world()                                # type: ignore[attr-defined]
    return float(p.world.run_gated(
        seed_chain=1, policy_pa=p.policy_pa, policy_l=p.policy_l,
        dt_events=DT)["makespan"]), p.calls


def main() -> None:
    cons = ConstraintConfig(maintenance=False)
    sp, _ = build(seed=0, instance=INST, tau=None, pm_interval=30,
                  target_failures=4, n_agv=3, finite_buffer=True,
                  layout_seed=200, reward_norm="minmax", constraints=cons)
    torch.manual_seed(0)
    net = HierarchicalPolicy()
    net.eval()                       # force_top1 ⟹ 网络权重不影响动作，只走候选构造

    base_ms, n_pa = _run(net, sp)
    print(f"**top-K oracle** · {INST} · 规则 "
          f"{PA_RULE_NAMES[PAIR[0]]}+{LA_RULE_NAMES[PAIR[1]]}"
          f" · force_top1 · 场景 1 · 保养关", flush=True)
    print(f"  基线 makespan **{base_ms:.1f}** · PA 决策 {n_pa} 个\n", flush=True)

    ts = list(range(0, n_pa, max(1, STRIDE)))[:MAXPTS]
    print(f"  {'t':>3}{'K':>4}{'c=0':>9}{'最好 c':>8}{'最好 MS':>10}"
          f"{'改善':>9}{'变差':>8}", flush=True)
    print("  " + "-" * 52, flush=True)
    gains, wins, kbest, losses = [], 0, [], []
    for t in ts:
        ms0, _ = _run(net, sp, at=t, idx=0)
        row = [_run(net, sp, at=t, idx=c)[0] for c in range(1, K_PA)]
        allms = [ms0] + row
        best_c = int(np.argmin(allms))
        best_ms = float(np.min(allms))
        g = ms0 - best_ms                     # 正 = 有更好的候选
        w = float(np.max(allms)) - ms0        # 正 = 选错要多花多少
        gains.append(g)
        losses.append(w)
        if best_c != 0:
            wins += 1
            kbest.append(best_c)
        print(f"  {t:>3}{K_PA:>4}{ms0:>9.1f}{best_c:>8}{best_ms:>10.1f}"
              f"{g:>9.2f}{-w:>8.1f}", flush=True)
    g = np.asarray(gains)
    lw = np.asarray(losses)
    print(f"\n  **argmin 就是最好的：{len(ts) - wins} / {len(ts)}**"
          f"（这些点上精排无可学）", flush=True)
    print(f"  改善：中位 {np.median(g):.2f} min　最大 {g.max():.2f} min"
          f"　均值 {g.mean():.2f} min", flush=True)
    print(f"  **选错的代价**：中位 {np.median(lw):.2f} min　最大 {lw.max():.2f} min"
          f"　均值 {lw.mean():.2f} min", flush=True)
    print(f"  **上下行不对称：均值 改善 {g.mean():.2f} vs 代价 {lw.mean():.2f}"
          f"（{lw.mean() / max(g.mean(), 1e-9):.1f}×）**", flush=True)
    if kbest:
        print(f"  更好的候选下标：{kbest}（0 = argmin）", flush=True)
    print("\n  ⚠️ 单步偏离**不可叠加** ⟹ 这些改善是**上界的方向**，不是可实现总量。",
          flush=True)


if __name__ == "__main__":
    main()
