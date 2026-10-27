# -*- coding: utf-8 -*-
"""**状态到底有没有「局内」分辨力** —— 四个版本全塌成常数，这个必须先排除。

══ 要回答的问题 ══

「上层按状态选规则」要成立，**同一个局内、不同决策点上的状态必须不同**。
若 PA 看到的 24 维特征在**一局之内几乎不变**，那么：

- 网络每次拿到的输入几乎一样 ⟹ **它连"按状态给不同动作"在原理上都做不到**
  （同样的输入只能给同样的输出）；
- ⟹ **塌成CDR不是"学不好"，是**被输入逼出来的**。**

══ 判据 ══

逐维比较两种离散度：

- **局内** `std_i` = 第 i 维在**一局 123 个决策点上**的标准差；
- **局间** `std_s` = 该维**各局均值之间**的标准差（20 个场景）。

`局内 / 局间` 越小 ⟹ 该维在一局内几乎不动、只在局与局之间动
⟹ **策略只能学会"分场景"，学不到"分决策点"**。
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # 仓根

import numpy as np  # noqa: E402

from plcsp_v3.env.nhdrl_features import F_PA, NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.rules import apply_pa_rule, la_rule_policy  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

RULE, LA_FIXED = 3, 0          # EAJ+SMPT（实测最好）


class Cap:
    def __init__(self, nf) -> None:
        self.nf, self.feats = nf, []

    def policy_pa(self, view, pool, tie_rng=None):
        self.feats.append(np.asarray(self.nf.pa(view.snap, 0.0, (5.0, 5.0)), dtype=float))
        return apply_pa_rule(RULE, view, pool, None)

    def policy_l(self, view, cand_v):
        return int(la_rule_policy(LA_FIXED, (0.0,))(view, cand_v))


def main() -> None:
    a, b = int(os.environ.get("V3_A", "16")), int(os.environ.get("V3_B", "140"))
    n_sc = int(os.environ.get("V3_N_SC", "20"))
    inst = named(b, a)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    nf = NHFeatures(inst, spec.layout)

    per_run = []
    for i in range(n_sc):
        c = Cap(nf)
        spec.world().run_gated(seed_chain=100_000 + 13 * i, policy_pa=c.policy_pa,
                               policy_l=c.policy_l, dt_events=0)
        per_run.append(np.stack(c.feats))
    print(f"Exp({a}/{b}) · {n_sc} 场景 · 每局 {per_run[0].shape[0]} 个决策点 · "
          f"PA 特征 **{F_PA} 维**\n")

    within = np.array([X.std(axis=0) for X in per_run])          # (n_sc, 24)
    means = np.array([X.mean(axis=0) for X in per_run])          # (n_sc, 24)
    across = means.std(axis=0)                                   # (24,)
    w, ac = within.mean(axis=0), across
    ratio = np.where(ac > 0, w / np.maximum(ac, 1e-12), np.inf)

    print(f"  {'维':>3}{'局内 std':>11}{'局间 std':>11}{'局内/局间':>11}"
          f"{'局内极差':>11}{'局内恒定?':>10}")
    print("  " + "-" * 58)
    for i in range(F_PA):
        rng = float(np.mean([float(X[:, i].max() - X[:, i].min()) for X in per_run]))
        print(f"  {i:>3}{w[i]:>11.5f}{ac[i]:>11.5f}{ratio[i]:>11.3f}{rng:>11.5f}"
              f"{'   ⚠️ 是' if rng < 1e-9 else '':>10}")

    const = int(np.sum([float(np.mean([float(X[:, i].max() - X[:, i].min())
                                       for X in per_run])) < 1e-9 for i in range(F_PA)]))
    print(f"\n  ⟹ **{F_PA} 维里有 {const} 维**在一局之内**完全不变**（局内极差 = 0）")
    print(f"  ⟹ 局内/局间 的中位数 = **{float(np.median(ratio)):.3f}**"
          f"　（< 1 ⟹ 该维的局内变化小于局间变化）")
    print(f"  ⟹ 整数维（计数类）占比 {(np.abs(per_run[0] - np.round(per_run[0])) < 1e-9).mean():.1%}")

    # 决策点上到底有多少个不同的输入？
    uniq = Counter(len(np.unique(np.round(X, 6), axis=0)) for X in per_run)
    print("\n  每局**互不相同**的特征向量个数（去重后）：")
    for k in sorted(uniq):
        print(f"      {k} 个 unique / {per_run[0].shape[0]} 个决策点　{uniq[k]} 局")
    print("\n  ⟹ 若 unique 数远小于决策点数，说明**大量决策点喂给网络的是同一个输入** ——"
          "\n     那些点上网络只能给同一个动作，**与「按状态选规则」无关**。")


if __name__ == "__main__":
    main()
