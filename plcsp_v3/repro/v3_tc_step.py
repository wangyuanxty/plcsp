# -*- coding: utf-8 -*-
"""**TC 支修好后有没有逐步信号** —— 先量，再提改法。

`Snapshot.vehicles[v].busy_min` = 累计（空载 + 负载）行驶工时。
原文 Eq.2 的 TC 增量就是这一项 ⟹ `tc_cum(snap) = uc · Σ_v busy_min`，
逐步增量 = 相邻决策点的差。

判据：增量的**非零占比**与**不同取值数**。
- 若几乎恒为 0 ⟹ 修了也没用，得另想办法。
- 若非零占比可观 ⟹ TC 支有信号，OA 的「锁死在 TC 支」自毁循环被打破。
"""
import os
import sys
from collections import Counter

sys.path.insert(0, r"D:\research\DeepReinforcementLearningScheduling")

import numpy as np  # noqa: E402

from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.rules import apply_pa_rule, la_rule_policy  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

UC = 0.5


def tc_cum(snap) -> float:
    return UC * sum(float(v.busy_min) for v in snap.vehicles)


class Capture:
    def __init__(self, rule: int = 0, lar: int = 0) -> None:
        self.rule, self.lar, self.vals = rule, lar, []

    def policy_pa(self, view, pool, tie_rng=None):
        self.vals.append(tc_cum(view.snap))
        return apply_pa_rule(self.rule, view, pool, None)

    def policy_l(self, view, cand_v):
        return int(la_rule_policy(self.lar, (0.0,))(view, cand_v))


def main() -> None:
    a, b = int(os.environ.get("V3_A", "16")), int(os.environ.get("V3_B", "140"))
    n_sc = int(os.environ.get("V3_N_SC", "6"))
    inst = named(b, a)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)

    all_steps, per_run = [], []
    for i in range(n_sc):
        for rule, lar in ((3, 0), (3, 1), (3, 2), (0, 0), (7, 2), (9, 1)):
            c = Capture(rule, lar)
            spec.world().run_gated(seed_chain=100_000 + 13 * i, policy_pa=c.policy_pa,
                                   policy_l=c.policy_l, dt_events=0)
            v = np.asarray(c.vals, dtype=float)
            s = np.diff(v)
            all_steps += s.tolist()
            if rule == 3 and lar == 0:
                per_run.append(float(v[-1] - v[0]))

    S = np.asarray(all_steps, dtype=float)
    n = S.size
    print(f"Exp({a}/{b}) · {n_sc} 场景 × 6 组(PA,LA) · **{n} 个逐步增量**\n")
    print(f"  TC 累计增量 Σ 的极差       {S.min():+.3f} – {S.max():+.3f}")
    print(f"  **非零占比**               {float(np.mean(np.abs(S) > 1e-12)):.1%}")
    print(f"  **不同取值数**             {len(np.unique(np.round(S, 9)))}")
    nz = S[np.abs(S) > 1e-12]
    if nz.size:
        print(f"  非零部分  均值 {nz.mean():+.4f}　最小 {nz.min():+.4f}　最大 {nz.max():+.4f}")
    d = Counter(np.round(S, 6).tolist())
    print("\n  取值分布（前 8）：")
    for k, c in d.most_common(8):
        print(f"      {k:+.6f}　{c:>6} 次　{c / n:6.1%}")
    print(f"\n  对照：一局的累计 TC 增量展开（{n_sc} 个场景，PA=EAJ+SMPT, LA=0）：")
    for i, x in enumerate(per_run):
        print(f"      #{i}　{x:+.3f}")
    print("\n  ⟹ 与 MS 支对照：MS 支的逐步增量**不同取值 322 种**（§37.14 实测）。")


if __name__ == "__main__":
    main()
