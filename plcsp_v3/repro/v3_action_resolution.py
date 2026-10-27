# -*- coding: utf-8 -*-
"""**动作空间在决策点上的真实分辨率** —— 基线看不见的那个量。

══ 为什么要测这个 ══

基线（27 条 CDR）把一条规则**整局固定**，所以它只证明了「不同规则 ⟹ 不同结局」。
它**看不见**的是：**在单个决策点上，换规则到底换不换得动东西。**

这正好是"只有学习会中招"的那一类病：

- 若一个决策点上 9 条规则**挑中同一件同一台机** ⟹ 该点的动作空间是**退化的** ⟹
  PPO 在该点拿到的优势恒为 0 ⟹ **没有梯度**（基线完全不受影响）。
- 若挑中的 (件, 机) 不同但**结局的即时代价相同** ⟹ 同样没有梯度。

做法：跑一局**固定规则**（保证轨迹合法），在 `policy_pa` 里**截获每一个 `(view, pool)`**，
再离线对这同一个 view 把 9 条规则**全算一遍**，比较它们各自挑中的 `(工件, 机台)`。
"""
import os
import sys
from collections import Counter

sys.path.insert(0, r"D:\research\DeepReinforcementLearningScheduling")

import numpy as np  # noqa: E402

from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.rules import (PA_MACHINE_RULES, PA_RULE_NAMES,  # noqa: E402
                                apply_pa_rule, la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402

N_RULE = 9


class Capture:
    """跑一局固定规则，顺手把每个决策点的 `(view, pool)` 留下。"""

    def __init__(self, rule: int = 0) -> None:
        self.rule, self.caps = rule, []

    def policy_pa(self, view, pool, tie_rng=None):
        self.caps.append((view, pool))
        return apply_pa_rule(self.rule, view, pool, None)

    def policy_l(self, view, cand_v):
        return int(la_rule_policy(0, (0.0,))(view, cand_v))


def main() -> None:
    a, b = int(os.environ.get("V3_A", "16")), int(os.environ.get("V3_B", "140"))
    n_sc = int(os.environ.get("V3_N_SC", "6"))
    inst = named(b, a)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)

    picks, npool, n_job, n_mach = [], [], [], []
    for i in range(n_sc):
        c = Capture(0)
        spec.world().run_gated(seed_chain=100_000 + 13 * i, policy_pa=c.policy_pa,
                               policy_l=c.policy_l, dt_events=0)
        for view, pool in c.caps:
            nxt = []
            for k in range(N_RULE):
                pi, m = apply_pa_rule(k, view, pool, None)
                j = pool[int(pi)][0]
                nxt.append((int(j), int(m)))
            picks.append(nxt)
            npool.append(len(pool))
            n_job.append(len({x[0] for x in nxt}))
            n_mach.append(len({x[1] for x in nxt}))

    picks = np.asarray(picks)                 # (决策点数, 9, 2)
    n = len(picks)
    print(f"Exp({a}/{b}) · {n_sc} 个场景 · **{n} 个决策点** · 每个点上把 9 条规则全算一遍\n")

    pair_uniq = np.array([len({(int(p[0]), int(p[1])) for p in row}) for row in picks])
    print(f"  池大小（该点可选工件数）     均值 {np.mean(npool):.2f}"
          f"　最小 {min(npool)}　最大 {max(npool)}")
    print(f"  9 条规则挑中的**不同工件数**  均值 {np.mean(n_job):.2f}")
    print(f"  9 条规则挑中的**不同机台数**  均值 {np.mean(n_mach):.2f}")
    print(f"  9 条规则挑中的**不同 (件,机)** 均值 **{np.mean(pair_uniq):.2f}**"
          f"　中位 {np.median(pair_uniq):.0f}")

    dist = Counter(pair_uniq.tolist())
    print("\n  不同 (件,机) 的条数分布：")
    for k in sorted(dist):
        print(f"      {k} 种　{dist[k]:>5} 个决策点　{dist[k] / n:6.1%}")

    one = float(np.mean(pair_uniq == 1))
    print(f"\n  ⟹ **{one:.1%} 的决策点上，9 条规则挑中的是完全相同的一件一机**"
          f"（该点动作空间退化）")

    # 工件那一半 / 机台那一半分别看：退化可能只在其中一半
    print("\n  拆开看：")
    print(f"      工件全同（9 条规则挑同一件）  {np.mean(np.array(n_job) == 1):6.1%}")
    print(f"      机台全同（9 条规则挑同一台）  {np.mean(np.array(n_mach) == 1):6.1%}")

    # 熵：动作在 9 条规则上的分散程度（0 = 全挤在一个动作）
    ent = []
    for row in picks:
        c = Counter((int(p[0]), int(p[1])) for p in row)
        pr = np.array(list(c.values())) / len(row)
        ent.append(float(-(pr * np.log(pr)).sum()))
    print(f"\n  逐决策点「9 条规则的落点」熵：均值 **{np.mean(ent):.4f}**"
          f"　（log 9 = {np.log(9):.4f}）")
    print(f"  规则清单：{PA_RULE_NAMES}  机台规则 {PA_MACHINE_RULES}")


if __name__ == "__main__":
    main()
