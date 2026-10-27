# -*- coding: utf-8 -*-
"""**AGV 台数诊断**：车越忙，「运输换工期」会不会出现（2026-10-17，用户提出的推论）。

══ 推论链 ══

```
用户：「如果车不忙，MINCOST 肯定是最优」
⟹ 车不忙 ⟹ LA 层无决策内容 ⟹ **想有权衡，车必须忙**
⟹ 因为只有车忙时，「现在派哪台车」才会拖慢别的活
```

**本仓 AGV 利用率 ≈ 50%（很空）；原文要有 43% 的「运输换工期」，车一定更忙。**
**⟹ 直接把 AGV 数减下去，看耦合会不会出现。**

⚠️ **原文 6 台是固定的** —— 本扫描是**诊断**，不是改口径：
   目的是确认「车忙」是不是那个缺失的耦合，**结果不得当成复现读数。**

══ 判据 ══

```
TC付 = TC(MS最好那条) ÷ TC(TC最好那条)   ← 要好工期，运输付几倍（原文 1.43）
MS比 = 最好 CDR 的 MS ÷ RN 的 MS          ← >1 才是原文方向（原文 1.43）
AGV 利用率 = 运输工时 ÷ (台数 × MS)
```
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import v3_hv3 as H  # noqa: E402
from plcsp_v3.env.constraints import ConstraintConfig  # noqa: E402
from plcsp_v3.env.shi_instances import EVAL_IDX, eval_instance  # noqa: E402
from plcsp_v3.hrl.rules import LA_RULE_NAMES, PA_RULE_NAMES  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

EIDX = int(os.environ.get("AG_EIDX", str(EVAL_IDX)))
SHOPLEN = float(os.environ.get("AG_SHOPLEN", "240"))
MAXES = [int(x) for x in os.environ.get("AG_MAXES", "10").split(",")]
AGVS = [int(x) for x in os.environ.get("AG_AGVS", "6,5,4,3").split(",")]
N_RUN = int(os.environ.get("AG_RN", "10"))


def build_spec(inst, n_agv: int):
    ovr = {"due_range": 0.6, "la_tie_random": 1.0,
           "agv_mttr_lo": 10.0, "agv_mttr_hi": 10.0,
           "agv_plan_pos": 1.0, "agv_act_book": 1.0}
    return build(seed=0, pm_interval=30, target_failures=4, n_agv=n_agv,
                 finite_buffer=True, layout_seed=0, reward_norm="minmax",
                 tau=2.70, sim_overrides=ovr, inst_obj=inst,
                 layout_scheme="random", **dims_from_shop_len(SHOPLEN, inst.n_machines),
                 shi2025=False,
                 constraints=ConstraintConfig(finite_buffer=False, machine_failure=False,
                                              maintenance=False, due_dates=False,
                                              agv_failure=False))[0]


def main() -> None:
    print("**AGV 台数诊断** · Exp(16/140) · 4×5 · 车间 "
          f"{SHOPLEN:.0f} m · 候选均匀 · 两个账本开 · 故障关\n"
          "  ⚠️ **原文 6 台是固定的** —— 本扫描只作诊断，**不得当复现读数**。\n", flush=True)
    print(f"  {'alts_max':>9}{'车数':>6}{'MS最好':>9}{'@TC':>9}{'TC最好':>9}{'@MS':>9}"
          f"{'MS付':>8}{'TC付':>8}{'MS比':>8}{'AGV利用':>9}")
    print("  " + "-" * 84)
    for mx in MAXES:
        inst = eval_instance(140, 16, EIDX, idx=EIDX, alts_dist="uniform",
                             alts_max=mx, pt_dist="uniform")
        for na in AGVS:
            spec = build_spec(inst, na)
            rows = []
            for k in range(9):
                for j in range(len(LA_RULE_NAMES)):
                    ms, tc = H.run_cdr(spec, k, j)
                    rows.append((PA_RULE_NAMES[k], LA_RULE_NAMES[j], ms, tc))
            ms = np.array([r[2] for r in rows])
            tc = np.array([r[3] for r in rows])
            rn_ms = float(np.mean([H.run_rn(spec, i)[0] for i in range(N_RUN)]))
            i_ms, i_tc = int(np.argmin(ms)), int(np.argmin(tc))
            # 运输工时 = 2·(TC − c·Σy_v)；Σy_v 这里取实际车数（每台都在用）。
            transport = 2.0 * (tc[i_tc] - 7.0 * na)
            util = transport / (na * ms[i_ms]) if na and ms[i_ms] > 0 else 0.0
            print(f"  {mx:>9d}{na:>6d}{ms[i_ms]:>9.1f}{tc[i_ms]:>9.1f}"
                  f"{tc[i_tc]:>9.1f}{ms[i_tc]:>9.1f}"
                  f"{ms[i_tc] / ms[i_ms]:>8.2f}{tc[i_ms] / tc[i_tc]:>8.2f}"
                  f"{ms.min() / rn_ms:>8.2f}{util:>9.1%}", flush=True)
    print("\n  ⚠️ 原文：`TC付 = 1.43` · `MS比 = 1.43` · `AGV 利用率` 反推 ≈ 20%（按其 TC 420 / MS 610）")
    print("  ⚠️ `MS付`/`TC付` 是 **CDR 内部**的两个角 —— 原文没有这一对，只作描述。")


if __name__ == "__main__":
    main()
