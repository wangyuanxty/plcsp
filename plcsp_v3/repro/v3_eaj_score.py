# -*- coding: utf-8 -*-
"""**EAJ 的真实打分有没有分辨力** —— 量的是 `pa_job_score("EAJ", ...)` 本身。

⚠️ 上一版探针只量了 `ready_at` 这一半。`rules.py:138` 的 EAJ 其实是

    start = max(ready_at, min_{候选机台} avail_at)      # avail_at = 机台账本 pa_mct
    score = max(0, start − now)                        # 越小越先

机台那一半（`avail_at`）也是活的，所以必须整式一起量。

**判据**：若 EAJ 打分在池内大面积并列 ⟹ EAJ 无法区分候选 ⟹ 退回平局规则
（`tie_rng=None` 取池序最小）⟹ **EAJ 名义上是状态相关规则，实际退化成固定规则。**
"""
import os
import sys
from collections import Counter

sys.path.insert(0, r"D:\research\DeepReinforcementLearningScheduling")

import numpy as np  # noqa: E402

from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.rules import (PA_JOB_RULES, apply_pa_rule,  # noqa: E402
                                la_rule_policy, pa_job_score)
from plcsp_v3.q1_flat import build  # noqa: E402

TOL = 1e-9


class Capture:
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

    rows = []
    for i in range(n_sc):
        c = Capture(0)
        spec.world().run_gated(seed_chain=100_000 + 13 * i, policy_pa=c.policy_pa,
                               policy_l=c.policy_l, dt_events=0)
        rows += c.caps
    n = len(rows)

    print(f"Exp({a}/{b}) · {n_sc} 个场景 · **{n} 个决策点**\n")
    print(f"  {'工件规则':<6}{'打分极差':>10}{'并列最优数':>11}"
          f"{'池大小':>8}{'选池首占比':>11}{'有效候选数':>11}")
    print("  " + "-" * 62)
    detail = {}
    for rule in PA_JOB_RULES:
        span, ties, size, first, eff = [], [], [], [], []
        for view, pool in rows:
            s = np.array([pa_job_score(rule, view, j) for j, _oi in pool], dtype=float)
            s = np.round(s, 9)
            lo = s.min()
            at = np.flatnonzero(s <= lo + TOL)
            span.append(float(s.max() - s.min()))
            ties.append(len(at))
            size.append(len(pool))
            first.append(float(0 in at))          # 池首恰好是并列最优
            eff.append(len(np.unique(s)))         # 有效候选数
        detail[rule] = (np.mean(span), np.mean(ties), np.mean(size),
                        np.mean(first), np.mean(eff))
        print(f"  {rule:<8}{np.mean(span):>10.3f}{np.mean(ties):>11.2f}"
              f"{np.mean(size):>8.2f}{np.mean(first):>11.1%}{np.mean(eff):>11.2f}")

    # 决定性：`tie_rng=None`（确定性平局 ⟹ 取池序最小）下各规则实际挑中池首的比例
    print("\n  ⚠️ **确定性平局下，规则实际挑中池首（工件号最小）的比例**：")
    for k, rule in enumerate(PA_JOB_RULES):
        got = [int(apply_pa_rule(k * 3, view, pool, None)[0]) for view, pool in rows]
        print(f"      {rule:<6} {np.mean([g == 0 for g in got]):6.1%}")

    print("\n  ⚠️ **环境用的是随机平局**（`des._pa_tie_rng` 独立子流）⟹ 并列时等概率抽一个。")
    print("     所以「EAJ 退化」对**基线**表现为「在并列集里随机选」——")
    print("     它仍然**不看状态**，只是把确定性换成了随机性。")
    d = Counter(int(round(x)) for x in
                [len(np.unique(np.round([pa_job_score('EAJ', v, j) for j, _o in p], 9)))
                 for v, p in rows])
    print("\n  EAJ 的**有效候选数**分布（打分不同的个数）：")
    for k in sorted(d):
        print(f"      {k} 个　{d[k]:>5} 点　{d[k] / n:6.1%}")


if __name__ == "__main__":
    main()
