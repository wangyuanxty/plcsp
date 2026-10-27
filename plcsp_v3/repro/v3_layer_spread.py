# -*- coding: utf-8 -*-
"""**LA 层到底有多重要** —— 固定 PA 换 LA 的跨度 vs 固定 LA 换 PA 的跨度（2026-10-17）。

══ 回答用户 ══

> *"你的意思是 LA 的三条规则选谁都差不多？"*

**本脚本把两个方向分开量：**

```
PA 跨度：固定一条 LA，9 条 PA 之间 MS/TC 的极差
LA 跨度：固定一条 PA，3 条 LA 之间 MS/TC 的极差
```

⚠️ **不再从「RN vs 最好 CDR」推 LA 的权重** —— RN 同时随机了 PA 与 LA，**不可归因**。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import v3_hv3 as H  # noqa: E402
from plcsp_v3.env.constraints import ConstraintConfig  # noqa: E402
from plcsp_v3.env.shi_instances import (EVAL_IDX, eval_instance)  # noqa: E402
from plcsp_v3.hrl.rules import LA_RULE_NAMES, PA_RULE_NAMES  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

EIDX = int(os.environ.get("SP_EIDX", str(EVAL_IDX)))
SHOPLEN = float(os.environ.get("SP_SHOPLEN", "240"))
ALTS_MAX = int(os.environ.get("SP_ALTSMAX", "10"))


def build_spec(inst):
    ovr = {"due_range": 0.6, "la_tie_random": 1.0,
           "agv_mttr_lo": 10.0, "agv_mttr_hi": 10.0,
           "agv_plan_pos": 1.0, "agv_act_book": 1.0}
    return build(seed=0, pm_interval=30, target_failures=4, n_agv=6,
                 finite_buffer=True, layout_seed=0, reward_norm="minmax",
                 tau=2.70, sim_overrides=ovr, inst_obj=inst,
                 layout_scheme="random", **dims_from_shop_len(SHOPLEN, inst.n_machines), shi2025=False,
                 constraints=ConstraintConfig(finite_buffer=False, machine_failure=False,
                                              maintenance=False, due_dates=False,
                                              agv_failure=False))[0]


def main() -> None:
    inst = eval_instance(140, 16, EIDX, idx=EIDX, alts_dist="uniform",
                         alts_max=ALTS_MAX, pt_dist="uniform")
    spec = build_spec(inst)
    MS = np.zeros((9, 3))
    TC = np.zeros((9, 3))
    for k in range(9):
        for j in range(3):
            MS[k, j], TC[k, j] = H.run_cdr(spec, k, j)
    print(f"**PA vs LA 的跨度** · Exp(16/140) · 车间 {SHOPLEN:.0f} m · "
          f"alts_max={ALTS_MAX} · 6 车\n")
    print("  ── 每个 PA 换 3 条 LA 的跨度（同一批工件、同一个 PA）──")
    print(f"  {'PA 规则':<14}{'MS 三档':<28}{'跨度':>8}{'TC 三档':<28}{'跨度':>8}")
    print("  " + "-" * 86)
    la_ms, la_tc = [], []
    for k in range(9):
        r_ms = MS[k].max() / MS[k].min() - 1
        r_tc = TC[k].max() / TC[k].min() - 1
        la_ms.append(r_ms)
        la_tc.append(r_tc)
        ms_s = " ".join(f"{v:6.0f}" for v in MS[k])
        tc_s = " ".join(f"{v:6.0f}" for v in TC[k])
        print(f"  {PA_RULE_NAMES[k]:<14}{ms_s:<28}{r_ms:>7.0%}{tc_s:<28}{r_tc:>7.0%}")
    print("  " + "-" * 86)
    print(f"  **LA 跨度均值**：MS **{np.mean(la_ms):.1%}** · TC **{np.mean(la_tc):.1%}**"
          f"　（中位 {np.median(la_ms):.1%} / {np.median(la_tc):.1%}）")

    print("\n  ── 每条 LA 固定时，9 条 PA 的跨度 ──")
    pa_ms, pa_tc = [], []
    for j in range(3):
        r_ms = MS[:, j].max() / MS[:, j].min() - 1
        r_tc = TC[:, j].max() / TC[:, j].min() - 1
        pa_ms.append(r_ms)
        pa_tc.append(r_tc)
        print(f"  {LA_RULE_NAMES[j]:<16}MS {MS[:, j].min():.0f}–{MS[:, j].max():.0f}"
              f"（跨度 {r_ms:.0%}）　TC {TC[:, j].min():.0f}–{TC[:, j].max():.0f}"
              f"（跨度 {r_tc:.0%}）")
    print(f"  **PA 跨度均值**：MS **{np.mean(pa_ms):.1%}** · TC **{np.mean(pa_tc):.1%}**")


if __name__ == "__main__":
    main()
