# -*- coding: utf-8 -*-
"""**实例规模是不是「常数近乎最优」的原因** —— 一个变量：工件数。

══ 为什么查这个 ══

我一整天都在 `Exp(16/140)` 上测 —— **16 个工件 / 18 台机**，工件比机台还少，
车间**根本不拥挤**，派工规则自然没什么可分的。
**原文 Table 11 用的是 `Exp(50/425)` … `Exp(90/801)`（50–90 工件 / 18 机）。**

⚠️ **`(a/b)` 的 `a` = 工件数**（`shi_instances.named(b, a)`），`b` = 含虚拟工序的总工序数，
**真工序 = b − a**。

判据：**规则极差** 与 **每场景余量** 随规模怎么变。
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

N_SC, N_RULE = 8, 9


def main() -> None:
    print("9 条规则 × 8 场景 · 实例规模逐个放大\n")
    print(f"  {'实例':<14}{'工件':>5}{'机':>4}{'真工序':>7}"
          f"{'最好 CDR':>12}{'ms':>8}{'规则极差':>10}{'每场景余量':>12}")
    print("  " + "-" * 74)
    for a, b in ((16, 140), (30, 261), (50, 425), (90, 801)):
        inst = named(b, a)
        real = sum(len(o) for o in inst.jobs)
        spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                        finite_buffer=True, layout_seed=0, reward_norm="minmax",
                        tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
        scs = [100_000 + 13 * i for i in range(N_SC)]
        M = np.zeros((N_RULE, N_SC))
        for k in range(N_RULE):
            for i, sc in enumerate(scs):
                m = spec.world().run_gated(
                    seed_chain=sc,
                    policy_pa=lambda v, p, tr, k=k: apply_pa_rule(k, v, p, tr),
                    policy_l=la_rule_policy(0), dt_events=0)
                M[k, i] = float(m["makespan"])
        glob = M.mean(axis=1)
        g = int(glob.argmin())
        print(f"  Exp({a}/{b})<{inst.n_jobs:>4}{inst.n_machines:>4}{real:>7}"
              f"{PA_RULE_NAMES[g]:>12}{glob[g]:>8.1f}"
              f"{(glob.max() - glob.min()) / glob[g]:>10.1%}"
              f"{(glob[g] - M.min(axis=0).mean()) / glob[g]:>12.2%}", flush=True)


if __name__ == "__main__":
    main()
