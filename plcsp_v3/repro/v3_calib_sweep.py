# -*- coding: utf-8 -*-
"""**候选上限 `alts_max` × 车间长度 —— 同时回答「1–5 行不行」与「新默认下 TC 靶在哪」**（2026-10-17）。

══ 为什么要一起做 ══

```
2026-10-17 起复现默认 = 候选**均匀 {1..alts_max}**（原文没给分布，取假设最少的一档）
而 `shop_len = 240` 是 2026-10-16 在**幂律候选（alts_pow=−2）**下标定的
⟹ **默认一改，那把尺子就过期了** —— TC 最好从 408 涨到 476（同一档 4×5·240·故障关）
```

**⟹ 单独问「1–5 行不行」会被这个漂移污染。故两轴一起扫。**

══ 口径（必须与标定档一致才可比）══

```
布局 4×5（`layout_cols=None` → grid_shape(18) = (4,5)）· scheme="random"
候选 均匀 {1..alts_max} · pt 均匀 [2,10]
**两个账本都开**：`agv_plan_pos=1` · `agv_act_book=1`（§4.37 的两处修复）
故障关 · idx=199
```

⚠️ 5×5 那批扫描**没开**这两个账本，故它们的数与这里**不可并列**。

══ 判据 ══

```
① `TC最好` 与「每趟」：原文 = **420 / 5.40 min** —— 看哪一档回到靶上
② `MS付` / `TC付`（**CDR 内部**的两个角）与 `MS比`（最好 CDR ÷ RN）
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
from plcsp_v3.hrl.rules import LA_RULE_NAMES, N_AEI2025_RULES  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

EIDX = int(os.environ.get("CA_EIDX", str(EVAL_IDX)))
MAXES = [int(x) for x in os.environ.get("CA_MAXES", "5,10").split(",")]
LENS = [float(x) for x in os.environ.get("CA_LENS", "180,200,220,240").split(",")]
NOFAIL = os.environ.get("CA_NOFAIL", "1") == "1"
N_RUN = int(os.environ.get("CA_RN", "10"))
SC_FIX = int(os.environ.get("CA_SC0", "100000"))


def build_spec(inst, shop_len: float):
    ovr = {"due_range": 0.6, "la_tie_random": 1.0,
           "agv_mttr_lo": 10.0, "agv_mttr_hi": 10.0,
           # ⚠️ **§4.37 的两处修复** —— 这两条必须开，否则口径与标定档不同源。
           "agv_plan_pos": 1.0, "agv_act_book": 1.0}
    kw = (dict(shi2025=False,
               constraints=ConstraintConfig(finite_buffer=False, machine_failure=False,
                                            maintenance=False, due_dates=False,
                                            agv_failure=False))
          if NOFAIL else dict(shi2025=True))
    return build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                 finite_buffer=True, layout_seed=0, reward_norm="minmax",
                 tau=2.70, sim_overrides=ovr, inst_obj=inst,
                 layout_scheme="random", **dims_from_shop_len(shop_len, inst.n_machines), **kw)[0]


def main() -> None:
    print("**候选上限 × 车间长度** · Exp(16/140) · 4×5 · 候选均匀 {{1..max}} · pt 均匀 · "
          f"故障={'关' if NOFAIL else '开'} · **两个账本都开**\n", flush=True)
    print(f"  {'alts_max':>9}{'车间[m]':>9}{'候选均值':>10}{'MS最好':>9}{'@TC':>9}"
          f"{'TC最好':>9}{'@MS':>9}{'每趟':>7}{'MS付':>8}{'TC付':>8}{'MS比':>8}")
    print("  " + "-" * 96)
    for mx in MAXES:
        for sl in LENS:
            inst = eval_instance(140, 16, EIDX, idx=EIDX, alts_dist="uniform",
                                 alts_max=mx, pt_dist="uniform")
            nk = np.mean([len(op) for job in inst.jobs for op in job])
            spec = build_spec(inst, sl)
            ms, tc = [], []
            for k in range(N_AEI2025_RULES):
                for j in range(len(LA_RULE_NAMES)):
                    m_, t_ = H.run_cdr(spec, k, j)
                    ms.append(m_)
                    tc.append(t_)
            ms, tc = np.asarray(ms), np.asarray(tc)
            rn_ms = float(np.mean([H.run_rn(spec, i)[0] for i in range(N_RUN)]))
            i_ms, i_tc = int(np.argmin(ms)), int(np.argmin(tc))
            per_trip = 2.0 * (tc[i_tc] - 42.0) / 140.0
            print(f"  {mx:>9d}{sl:>9.0f}{nk:>10.2f}{ms[i_ms]:>9.1f}{tc[i_ms]:>9.1f}"
                  f"{tc[i_tc]:>9.1f}{ms[i_tc]:>9.1f}{per_trip:>7.2f}"
                  f"{ms[i_tc] / ms[i_ms]:>8.2f}{tc[i_ms] / tc[i_tc]:>8.2f}"
                  f"{ms.min() / rn_ms:>8.2f}", flush=True)
    print("\n  ⚠️ 原文：`TC最好 = 420` · `每趟 = 5.40 min` · `MS付/TC付 = 1.43/1.43`（**CDR-vs-RN**）"
          " · `MS比 = 1.43`")
    print("  ⚠️ `MS付`/`TC付` 是 **CDR 内部**的两个角 —— **原文没有这一对**，只作描述。")
    print("  ⚠️ 换 `alts_max` 会换实例（候选抽样流不同）⟹ 行间不是同一批工件，只有统计意义。")


if __name__ == "__main__":
    main()
