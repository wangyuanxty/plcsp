# -*- coding: utf-8 -*-
"""**加工时间范围 2–10 vs 2–20** —— 用户提出：加工时间差别更极端，两难会不会出现（2026-10-17）。

⚠️ **这是改原文规格的敏感度测试** —— 原文 §5.1 明确写 *"randomly determined between 2 and 10"*。
   **本扫描的结果不得当复现读数。**

**用户的理由**：
```
现在 pt ∈ [2, 10]，快慢只差 **5 倍** ⟹ 挑快机台的收益小 ⟹ 「省运输 vs 保工期」的张力小
改成 [2, 20]，差 **10 倍** ⟹ 挑快机台的收益翻倍 ⟹ 张力应该变大
```

**判据**：`MS付` / `TC付`（原文 1.43 / 1.43；本仓 2–10 时 1.36 / 1.21）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from plcsp_v3.env.constraints import ConstraintConfig  # noqa: E402
from plcsp_v3.env.shi_instances import eval_instance  # noqa: E402
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, apply_pa_rule,  # noqa: E402
                                la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

EIDX = int(os.environ.get("PR_EIDX", "199"))
SHOPLEN = float(os.environ.get("PR_SHOPLEN", "240"))
ALTS_MAX = int(os.environ.get("PR_ALTSMAX", "10"))
N_INST = int(os.environ.get("PR_NINST", "5"))
HIS = [int(x) for x in os.environ.get("PR_HIS", "10,20").split(",")]


def build_spec(inst):
    return build(seed=0, pm_interval=30, target_failures=4, n_agv=6,
                 finite_buffer=True, layout_seed=0, reward_norm="minmax",
                 tau=2.70, sim_overrides={"due_range": 0.6, "la_tie_random": 1.0,
                                          "agv_mttr_lo": 10.0, "agv_mttr_hi": 10.0,
                                          "agv_plan_pos": 1.0, "agv_act_book": 1.0},
                 inst_obj=inst, layout_scheme="random", **dims_from_shop_len(SHOPLEN, inst.n_machines),
                 shi2025=True,
                 constraints=ConstraintConfig(finite_buffer=False, machine_failure=False,
                                              maintenance=False, due_dates=False,
                                              agv_failure=False))[0]


def main() -> None:
    print(f"**加工时间范围敏感度** · Exp(16/140) · 车间 {SHOPLEN:.0f} m · "
          f"**忠实口径** · {N_INST} 实例均值\n"
          "  ⚠️ **改原文规格**（原文 2–10）—— 结果**不得当复现读数**。\n")
    print(f"  {'pt 上限':>8}{'MS@MS最好':>10}{'TC@MS最好':>10}{'MS@TC最好':>10}"
          f"{'TC@TC最好':>10}{'MS付':>7}{'TC付':>7}{'前沿':>7}")
    print("  " + "-" * 76)
    for hi in HIS:
        c = np.zeros(4)
        pm, pt, fr = [], [], []
        for k in range(N_INST):
            inst = eval_instance(140, 16, EIDX, idx=EIDX + k, alts_dist="uniform",
                                 alts_max=ALTS_MAX, pt_dist="uniform", pt_hi=hi)
            spec = build_spec(inst)
            ms, tc = [], []
            for kk in range(9):
                for j in range(len(LA_RULE_NAMES)):
                    w = spec.world()
                    m = w.run_gated(seed_chain=100000,
                                    policy_pa=lambda v, p, tr=None, _k=kk:
                                        apply_pa_rule(_k, v, p, tr),
                                    policy_l=la_rule_policy(j), dt_events=0)
                    ms.append(float(m["makespan"]))
                    tc.append(float(m["logistics_cost"]))
            ms, tc = np.asarray(ms), np.asarray(tc)
            i_ms, i_tc = int(np.argmin(ms)), int(np.argmin(tc))
            c += np.array([ms[i_ms], tc[i_ms], ms[i_tc], tc[i_tc]])
            pm.append(ms[i_tc] / ms[i_ms])
            pt.append(tc[i_ms] / tc[i_tc])
            front = [i for i in range(len(ms))
                     if not any((ms[q] <= ms[i] and tc[q] <= tc[i]) and
                                (ms[q] < ms[i] or tc[q] < tc[i]) for q in range(len(ms)))]
            fr.append(len(front))
        a, b, cc, dd = c / N_INST
        print(f"  {'2–' + str(hi):>8}{a:>10.1f}{b:>10.1f}{cc:>10.1f}{dd:>10.1f}"
              f"{np.mean(pm):>7.2f}{np.mean(pt):>7.2f}{np.mean(fr):>7.1f}", flush=True)
    print("\n  ⚠️ 原文：`MS付 / TC付 = 1.43 / 1.43`。")
    print("  ⚠️ 本仓 2–10 时（同口径）：1.36 / 1.21。")


if __name__ == "__main__":
    main()
