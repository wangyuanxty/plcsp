# -*- coding: utf-8 -*-
"""**在同一个决策点上换规则，结局差多少？** —— 「题面有没有可学的状态依赖」的直接读数。

══ 为什么要这个 ══

四个版本（v3 最小版 / v3 全版 / 复现最小版 / 复现全版）**全部塌成CDR**。
已排除：仿真内核（JobShopLab 仲裁）、动作空间（决策点上有 4.05 种不同 `(件,机)`）、
状态特征（局内/局间 std 中位数 **15.18**，124/124 个决策点输入互不相同，§37.25）。

**只剩一个候选：优势估计 / 学习更新。** 但在下结论之前，得先确认**题面本身有没有肉**：

> **在第 `t` 个决策点换一条规则，终局 makespan 会变吗？**
> - **不变** ⟹ 该点的动作对结局没有影响 ⟹ **没有任何东西可学**，塌成常数是必然的。
> - **会变** ⟹ 有肉可吃，而策略学不到 ⟹ **病灶在优势估计/更新**。

══ 做法（不训练，纯评估）══

固定规则 A 跑第 `1..t−1` 个决策点，**从第 `t` 个起换成规则 B**，比终局 makespan。
对 9 条规则 B 各跑一遍 ⟹ 该点的 9 个终局值，**看它们的散布**。

⚠️ 仿真对 `(seed_chain, 策略)` 是确定的 ⟹ 用**同一条链**跑，只在该点分叉，除规则外无别的差别。
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

BASE, LA_FIXED = 3, 0            # 基准 = EAJ+SMPT（实测最好）
N_PA = 9


class SwitchAt:
    """前 `t−1` 个决策点用 `base`，**第 `t` 个起**换成 `after`。"""

    def __init__(self, base: int, after: int, t: int) -> None:
        self.base, self.after, self.t, self.n = base, after, t, 0

    def policy_pa(self, view, pool, tie_rng=None):
        self.n += 1
        return apply_pa_rule(self.after if self.n >= self.t else self.base,
                             view, pool, None)

    def policy_l(self, view, cand_v):
        return int(la_rule_policy(LA_FIXED, (0.0,))(view, cand_v))


def main() -> None:
    a, b = int(os.environ.get("V3_A", "16")), int(os.environ.get("V3_B", "140"))
    n_sc = int(os.environ.get("V3_N_SC", "4"))
    inst = named(b, a)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    scs = [100_000 + 13 * i for i in range(n_sc)]

    # 先量一局有几个决策点
    cnt = SwitchAt(BASE, BASE, 0)
    spec.world().run_gated(seed_chain=scs[0], policy_pa=cnt.policy_pa,
                           policy_l=cnt.policy_l, dt_events=0)
    total = cnt.n
    ts = [int(total * f) for f in (0.15, 0.3, 0.5, 0.7, 0.85)]
    print(f"Exp({a}/{b}) · {n_sc} 场景 · 每局 **{total} 个决策点** · "
          f"基准规则 = {PA_RULE_NAMES[BASE]}\n")
    print("  在决策点 t 处把规则换成 B，终局 makespan 的散布（9 条 B）：\n")
    print(f"  {'t':>5}{'最好':>9}{'最差':>9}{'极差':>9}{'极差/基准':>11}"
          f"{'基准(B=A)':>11}{'最好比基准好':>13}")
    print("  " + "-" * 68)

    allspread = []
    for t in ts:
        ms = np.zeros(N_PA)
        for k in range(N_PA):
            vals = []
            for sc in scs:
                p = SwitchAt(BASE, k, t)
                vals.append(float(spec.world().run_gated(
                    seed_chain=sc, policy_pa=p.policy_pa,
                    policy_l=p.policy_l, dt_events=0)["makespan"]))
            ms[k] = float(np.mean(vals))
        base_ms = ms[BASE]                 # B = A ⟹ 等于纯基准，自检用
        spread = float(ms.max() - ms.min())
        gain = base_ms - float(ms.min())
        allspread.append((t, spread, gain))
        print(f"  {t:>5}{ms.min():>9.1f}{ms.max():>9.1f}{spread:>9.1f}"
              f"{spread / base_ms:>10.1%}{base_ms:>11.1f}{gain:>13.1f}")

    sp = np.array([s for _t, s, _g in allspread])
    gn = np.array([g for _t, _s, g in allspread])
    print(f"\n  ⟹ 极差/基准 的均值 = **{float(np.mean(sp)) / float(ms[BASE]):.1%}**"
          f"　最大 {float(sp.max()) / float(ms[BASE]):.1%}")
    print(f"  ⟹ 「换到最好规则」相对基准的改善：均值 **{float(gn.mean()):.1f}** min"
          f"　最大 **{float(gn.max()):.1f}** min")
    print("\n  ⚠️ 这是**下界**：只换一次。真·每决策点 oracle 只会更好（不会更差）。")
    print("  ⟹ 若极差显著大于 0 ⟹ **题面有肉**，学不到就是**优势估计/更新**的问题；")
    print("     若极差 ≈ 0     ⟹ **该点无论选什么都一样**，塌成常数是必然。")


if __name__ == "__main__":
    main()
