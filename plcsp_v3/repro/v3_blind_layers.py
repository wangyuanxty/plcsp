# -*- coding: utf-8 -*-
"""**分层盲化**：把 `EAJ` 与 `MINU` 对账本的依赖**分开**（2026-10-17，用户要求）。

══ 上一层的结论 ══

```
正常（有账本）  ：`EAJ+MINU+MINCOST` **同时**是 MS 最小和 TC 最小 ⟹ 前沿 = 1 点
整块盲化（全 0）：前沿变成 3 条，MS 最好与 TC 最好**分成两条规则**
⟹ **通吃的优势来自计划账本**
```

**但账本被**两条规则**共用，本脚本把它拆开：**

| 开关 | 效果 | 受影响 |
|---|---|---|
| `pa_blind_eaj` | `ready_at` / `transportable_at` 里的 `pa_mct` → 0 | **只 EAJ** |
| `pa_blind_minu` | `PAMachine.assigned_min` → 0 ⟹ `MINU ≡ 0`（全并列） | **只 MINU** |
| `pa_blind` | 两者 + `avail_at` 全 0 | EAJ · MINU · EAM |

⚠️ 三个开关默认 `False` ⟹ 既有读数逐位不变。

══ 判据 ══

```
MS 最好与 TC 最好**是不是同一条** ⟹ 通吃还在不在
Pareto 前沿几条                  ⟹ 前沿有没有宽度
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
from plcsp_v3.env.shi_instances import (EVAL_IDX, N_AGV_SHI,  # noqa: E402
                                        eval_instance)
from plcsp_v3.hrl.rules import LA_RULE_NAMES, PA_RULE_NAMES  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

EIDX = int(os.environ.get("LB_EIDX", str(EVAL_IDX)))
SHOPLEN = float(os.environ.get("LB_SHOPLEN", "240"))
ALTS_MAX = int(os.environ.get("LB_ALTSMAX", "5"))
CASES = [("正常", {}),
         ("只盲 EAJ", {"pa_blind_eaj": 1.0}),
         ("只盲 MINU", {"pa_blind_minu": 1.0}),
         ("两个都盲", {"pa_blind_eaj": 1.0, "pa_blind_minu": 1.0})]


def build_spec(inst, extra: dict):
    ovr = {"due_range": 0.6, "la_tie_random": 1.0,
           "agv_mttr_lo": 10.0, "agv_mttr_hi": 10.0,
           "agv_plan_pos": 1.0, "agv_act_book": 1.0}
    ovr.update(extra)
    return build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                 finite_buffer=True, layout_seed=0, reward_norm="minmax",
                 tau=2.70, sim_overrides=ovr, inst_obj=inst,
                 layout_scheme="random", **dims_from_shop_len(SHOPLEN, inst.n_machines),
                 shi2025=False,
                 constraints=ConstraintConfig(finite_buffer=False, machine_failure=False,
                                              maintenance=False, due_dates=False,
                                              agv_failure=False))[0]


def main() -> None:
    inst = eval_instance(140, 16, EIDX, idx=EIDX, alts_dist="uniform",
                         alts_max=ALTS_MAX, pt_dist="uniform")
    print(f"**分层盲化** · Exp(16/140) · 车间 {SHOPLEN:.0f} m · alts_max={ALTS_MAX}\n"
          f"  {'情形':<12}{'MS 最好':<26}{'TC 最好':<26}{'通吃?':>7}{'前沿':>6}"
          f"{'MS跨度':>14}{'TC跨度':>14}")
    print("  " + "-" * 108)
    for tag, extra in CASES:
        spec = build_spec(inst, extra)
        rows = []
        for k in range(9):
            for j in range(len(LA_RULE_NAMES)):
                ms, tc = H.run_cdr(spec, k, j)
                rows.append((PA_RULE_NAMES[k], LA_RULE_NAMES[j], ms, tc))
        ms = np.array([r[2] for r in rows])
        tc = np.array([r[3] for r in rows])
        i_ms, i_tc = int(np.argmin(ms)), int(np.argmin(tc))
        front = [i for i in range(len(rows))
                 if not any((ms[q] <= ms[i] and tc[q] <= tc[i]) and
                            (ms[q] < ms[i] or tc[q] < tc[i]) for q in range(len(rows)))]
        a = f"{rows[i_ms][0].split('+')[0]}+{rows[i_ms][1][:6]} {ms[i_ms]:.0f}/{tc[i_ms]:.0f}"
        b = f"{rows[i_tc][0].split('+')[0]}+{rows[i_tc][1][:6]} {ms[i_tc]:.0f}/{tc[i_tc]:.0f}"
        same = "**是**" if i_ms == i_tc else "否"
        print(f"  {tag:<12}{a:<26}{b:<26}{same:>7}{len(front):>6}"
              f"{ms.min():>7.0f}–{ms.max():<7.0f}{tc.min():>7.0f}–{tc.max():<7.0f}",
              flush=True)
    print("\n  ⚠️ 「通吃?」= MS 最好与 TC 最好是不是**同一条规则**（是 ⟹ 前沿塌成一个点）。")
    print("  ⚠️ 三个开关默认 False ⟹ 既有读数逐位不变。")


if __name__ == "__main__":
    main()
