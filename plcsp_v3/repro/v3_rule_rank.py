# -*- coding: utf-8 -*-
"""**27 条 CDR 的完整排名** —— 找出「MS 与 TC 同时最好」的那条到底是哪条规则（2026-10-17）。

══ 背景 ══

扫描表里出现过一格退化读数：

```
alts_max=5 · 车间 240 m
  MS最好 = 307.0  @TC = 416.0
  TC最好 = 416.0  @MS = 307.0      ⟹ **同一条规则两个目标都最好** ⟹ 前沿塌成一个点
```

**本脚本把 27 条逐条打出来**，并标出：谁支配谁、Pareto 前沿有几条、那条「双料最好」是谁。
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

EIDX = int(os.environ.get("CR_EIDX", str(EVAL_IDX)))
SHOPLEN = float(os.environ.get("CR_SHOPLEN", "240"))
TOPN = int(os.environ.get("CR_TOPN", "12"))


def build_spec(inst):
    ovr = {"due_range": 0.6, "la_tie_random": 1.0,
           "agv_mttr_lo": 10.0, "agv_mttr_hi": 10.0,
           "agv_plan_pos": 1.0, "agv_act_book": 1.0}
    return build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                 finite_buffer=True, layout_seed=0, reward_norm="minmax",
                 tau=2.70, sim_overrides=ovr, inst_obj=inst,
                 layout_scheme="random", **dims_from_shop_len(SHOPLEN, inst.n_machines),
                 shi2025=False,
                 constraints=ConstraintConfig(finite_buffer=False, machine_failure=False,
                                              maintenance=False, due_dates=False,
                                              agv_failure=False))[0]


def main() -> None:
    for mx in (5, 10):
        inst = eval_instance(140, 16, EIDX, idx=EIDX, alts_dist="uniform",
                             alts_max=mx, pt_dist="uniform")
        spec = build_spec(inst)
        rows = []
        for k in range(9):
            for j in range(len(LA_RULE_NAMES)):
                ms, tc = H.run_cdr(spec, k, j)
                rows.append((PA_RULE_NAMES[k], LA_RULE_NAMES[j], ms, tc))
        ms = np.array([r[2] for r in rows])
        tc = np.array([r[3] for r in rows])
        i_ms, i_tc = int(np.argmin(ms)), int(np.argmin(tc))
        # Pareto 前沿（两目标都最小）
        front = [i for i in range(len(rows))
                 if not any((ms[q] <= ms[i] and tc[q] <= tc[i]) and
                            (ms[q] < ms[i] or tc[q] < tc[i]) for q in range(len(rows)))]
        print(f"\n════ alts_max = {mx} · 车间 {SHOPLEN:.0f} m · 候选均值 "
              f"{np.mean([len(o) for j in inst.jobs for o in j]):.2f} 台/工序 ════")
        print(f"  {'#':>3} {'PA 规则':<12}{'LA 规则':<16}{'MS':>8}{'TC':>8}   标记")
        print("  " + "-" * 62)
        for rank, i in enumerate(np.argsort(ms)[:TOPN]):
            marks = []
            if i == i_ms:
                marks.append("**MS 最好**")
            if i == i_tc:
                marks.append("**TC 最好**")
            if i in front:
                marks.append("前沿")
            print(f"  {rank + 1:>3} {rows[i][0]:<12}{rows[i][1]:<16}"
                  f"{ms[i]:>8.1f}{tc[i]:>8.1f}   {' · '.join(marks)}")
        print(f"\n  **MS 最好**：{rows[i_ms][0]} + {rows[i_ms][1]}"
              f"　(MS {ms[i_ms]:.1f}, TC {tc[i_ms]:.1f})")
        print(f"  **TC 最好**：{rows[i_tc][0]} + {rows[i_tc][1]}"
              f"　(MS {ms[i_tc]:.1f}, TC {tc[i_tc]:.1f})")
        same = "**同一条（前沿塌成一个点）**" if i_ms == i_tc else "不同"
        print(f"  ⟹ {same}　·　**Pareto 前沿 {len(front)} 条**")
        if len(front) > 1:
            print("  前沿成员：" + "；".join(
                f"{rows[i][0]}+{rows[i][1]} ({ms[i]:.0f},{tc[i]:.0f})" for i in front))


if __name__ == "__main__":
    main()
