# -*- coding: utf-8 -*-
"""**代理指标 vs 真实目标** —— 9 条规则在两个排序下名次一致吗？

══ 为什么要这个 ══

最小版（跨链 GAE 已修）**曲线在升**（判据 1 ✅），makespan 也从 198.1 降到 182.8，
**但它偏好 `EAJ+MINU`（真实 makespan 排第 6，201.9），而不是最好的 `EAJ+SMPT`（153.3）。**

**假设**：训练优化的不是真实 makespan，而是**代理** `Σd_ms = CM末 − CM首`
（奖励 `−d_ms` 的望远镜求和，corr 与 makespan 只有 **0.951**）。
- **可抓的信号**：换对规则好 **2.7 min（1.7%）**
- **代理的残差**：**~5%**

**信噪比 < 1 ⟹ 策略停在"代理下看起来好"的那条，而不是真实最好那条。**

══ 判据 ══

逐条规则算 `(真实 makespan, Σd_ms)`，比两个排序。
**若 `EAJ+MINU` 在 `Σd_ms` 下排名明显高于 makespan 下的第 6 ⟹ 假设成立。**
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # 仓根

import numpy as np  # noqa: E402

from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.rules import (PA_RULE_NAMES, apply_pa_rule,  # noqa: E402
                                la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402

LA_FIXED = 0


def cm_of(snap) -> float:
    return float(snap.now) + max(
        (float(m.remaining_min) + float(m.backlog_min) for m in snap.machines), default=0.0)


class Cap:
    def __init__(self, rule: int) -> None:
        self.rule, self.cms = rule, []

    def policy_pa(self, view, pool, tie_rng=None):
        self.cms.append(cm_of(view.snap))
        return apply_pa_rule(self.rule, view, pool, None)

    def policy_l(self, view, cand_v):
        return int(la_rule_policy(LA_FIXED, (0.0,))(view, cand_v))


def main() -> None:
    a, b = int(os.environ.get("V3_A", "16")), int(os.environ.get("V3_B", "140"))
    n_sc = int(os.environ.get("V3_N_SC", "20"))
    inst = named(b, a)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    scs = [100_000 + 13 * i for i in range(n_sc)]

    ms, prox = {}, {}
    for k in range(9):
        mk, px = [], []
        for sc in scs:
            c = Cap(k)
            m = spec.world().run_gated(seed_chain=sc, policy_pa=c.policy_pa,
                                       policy_l=c.policy_l, dt_events=0)
            mk.append(float(m["makespan"]))
            px.append(float(c.cms[-1] - c.cms[0]))     # Σd_ms，望远镜求和
        ms[k], prox[k] = float(np.mean(mk)), float(np.mean(px))

    rank_ms = {k: r for r, k in enumerate(sorted(ms, key=lambda i: ms[i]), 1)}
    rank_px = {k: r for r, k in enumerate(sorted(prox, key=lambda i: prox[i]), 1)}
    print(f"Exp({a}/{b}) · {n_sc} 场景\n")
    print(f"  {'规则':<12}{'真实 makespan':>14}{'名次':>6}"
          f"{'代理 Σd_ms':>13}{'名次':>6}{'差':>5}")
    print("  " + "-" * 58)
    for k in sorted(ms, key=lambda i: ms[i]):
        d = rank_ms[k] - rank_px[k]
        print(f"  {PA_RULE_NAMES[k]:<12}{ms[k]:>14.1f}{rank_ms[k]:>6}"
              f"{prox[k]:>13.1f}{rank_px[k]:>6}{d:>+5}")

    X = np.array([ms[k] for k in range(9)])
    Y = np.array([prox[k] for k in range(9)])
    print(f"\n  corr(makespan, Σd_ms) = **{np.corrcoef(X, Y)[0, 1]:+.4f}**")
    tau = float(np.corrcoef([rank_ms[k] for k in range(9)],
                            [rank_px[k] for k in range(9)])[0, 1])
    print(f"  名次的 Spearman 相关 = **{tau:+.4f}**")
    bad = [(PA_RULE_NAMES[k], rank_ms[k], rank_px[k])
           for k in range(9) if abs(rank_ms[k] - rank_px[k]) >= 2]
    print(f"\n  名次差 ≥ 2 的规则（{len(bad)} 条）："
          f"{'　'.join(f'{n}: {a_}→{b_}' for n, a_, b_ in bad) or '无'}")

    # 真实最优 vs 代理最优
    k_ms = min(ms, key=lambda i: ms[i])
    k_px = min(prox, key=lambda i: prox[i])
    print(f"\n  ⟹ 真实最优 = **{PA_RULE_NAMES[k_ms]}**（{ms[k_ms]:.1f}）"
          f"　代理最优 = **{PA_RULE_NAMES[k_px]}**（{ms[k_px]:.1f}）")
    if k_ms == k_px:
        print("  ⟹ 两者**相同** ⟹ 优化代理**不会**把策略领到别的规则上（本假设不成立）。")
    else:
        print("  ⟹ 两者不同 ⟹ **优化代理会把策略领到另一条规则上。**")


if __name__ == "__main__":
    main()
