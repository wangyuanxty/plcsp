# -*- coding: utf-8 -*-
"""**`TC付` 只有 1.21 的分解** —— 省下来的是负载段还是空载段（2026-10-17）。

══ 问题 ══

**忠实口径**（有向走廊 + 中点接入，`shi2025=True` 自动开）下的读数：

```
MS 最好那条：(MS 328.5, TC 539.8)
TC 最好那条：(MS 443.0, TC 446.1)      → `MS付` 1.35 · `TC付` **1.21**

原文：       CDR (610, 420) ／ RN (428, 600)   → 1.43 / **1.43**
```

**`TC = 42 + 0.5 × (空载 + 负载)`** ⟹ 拆开看是哪一段没省下来：
- **负载段** ← 由**机台分配**定（哪台机做哪道工序）
- **空载段** ← 由**派车**定（哪台车去取货）

⚠️ 两者由**不同的决策**决定 ⟹ 「为什么只省一半」要看是哪一段卡住。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from plcsp_v3.env.constraints import ConstraintConfig  # noqa: E402
from plcsp_v3.env.shi_instances import eval_instance  # noqa: E402
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, apply_pa_rule, la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

EIDX = int(os.environ.get("DE_EIDX", "199"))
SHOPLEN = float(os.environ.get("DE_SHOPLEN", "240"))
ALTS_MAX = int(os.environ.get("DE_ALTSMAX", "10"))
N_INST = int(os.environ.get("DE_NINST", "5"))


def build_spec(inst):
    return build(seed=0, pm_interval=30, target_failures=4, n_agv=6,
                 finite_buffer=True, layout_seed=0, reward_norm="minmax",
                 tau=2.70, sim_overrides={"due_range": 0.6, "la_tie_random": 1.0,
                                          "agv_mttr_lo": 10.0, "agv_mttr_hi": 10.0,
                                          "agv_plan_pos": 1.0, "agv_act_book": 1.0},
                 inst_obj=inst, layout_scheme="random", **dims_from_shop_len(SHOPLEN, inst.n_machines),
                 shi2025=True,          # ⟹ directed_corridors / mid_edge_dock 自动开
                 constraints=ConstraintConfig(finite_buffer=False, machine_failure=False,
                                              maintenance=False, due_dates=False,
                                              agv_failure=False))[0]


def main() -> None:
    print(f"**TC 的分解** · Exp(16/140) · 车间 {SHOPLEN:.0f} m · "
          f"**忠实口径**（有向走廊 + 中点接入）· {N_INST} 实例均值\n")
    ms_all, tc_all, emp_all, ld_all = [], [], [], []
    for k in range(N_INST):
        inst = eval_instance(140, 16, EIDX, idx=EIDX + k,
                             alts_dist="uniform", alts_max=ALTS_MAX, pt_dist="uniform")
        spec = build_spec(inst)
        for kk in range(9):
            for j in range(len(LA_RULE_NAMES)):
                w = spec.world()
                m = w.run_gated(seed_chain=100000,
                                policy_pa=lambda v, p, tr=None, _k=kk:
                                    apply_pa_rule(_k, v, p, tr),
                                policy_l=la_rule_policy(j), dt_events=0)
                ms_all.append(float(m["makespan"]))
                tc_all.append(float(m["logistics_cost"]))
                st = m["energy_breakdown"]["agv_states_min"]
                emp_all.append(float(st["empty"]))
                ld_all.append(float(st["loaded"]))
    ms, tc = np.asarray(ms_all), np.asarray(tc_all)
    emp, ld = np.asarray(emp_all), np.asarray(ld_all)
    i_ms, i_tc = int(np.argmin(ms)), int(np.argmin(tc))
    print(f"  {'':<14}{'MS':>8}{'TC':>8}{'空载':>9}{'负载':>9}{'运输合计':>10}")
    print("  " + "-" * 60)
    for tag, i in (("MS 最好", i_ms), ("TC 最好", i_tc)):
        print(f"  {tag:<14}{ms[i]:>8.1f}{tc[i]:>8.1f}{emp[i]:>9.1f}{ld[i]:>9.1f}"
              f"{(emp[i] + ld[i]):>10.1f}")
    print("  " + "-" * 60)
    print(f"  {'比值':<14}{ms[i_tc] / ms[i_ms]:>8.2f}{tc[i_ms] / tc[i_tc]:>8.2f}"
          f"{emp[i_ms] / emp[i_tc]:>9.2f}{ld[i_ms] / ld[i_tc]:>9.2f}"
          f"{(emp[i_ms] + ld[i_ms]) / (emp[i_tc] + ld[i_tc]):>10.2f}")
    print("\n  ⟹ 哪一列的比值大，**哪一段就是没省下来的那一段**。")
    print("     负载段 ← 机台分配（PA）；空载段 ← 派车（LA）。")
    print("  ⚠️ 原文的 `600 / 420 = 1.43` 是**含 42 的成本比**，不是运输比；")
    print(f"     换算成运输比：`2(600−42) / 2(420−42)` = **{2 * (600 - 42) / (2 * (420 - 42)):.2f}**")


if __name__ == "__main__":
    main()
