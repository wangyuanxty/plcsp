# -*- coding: utf-8 -*-
"""**多规模的 MS/TC 权衡** —— 机位/走廊显式 · 候选 8–16 · 0.5 m/s · **装卸 0**。

══ 为什么扫这个（2026-10-17，用户要求）══

原文训练分布是 `I ∈ [10,40]`。训练前先确认：
**不同规模上，27 条 CDR 内部的两个角是否仍存在权衡。**

判据（只在 CDR 内部）：
```
MS付 = MS(TC 最好) / MS(MS 最好)
TC付 = TC(MS 最好) / TC(TC 最好)
原文：1.43 / 1.43
```

⚠️ **不是** §4.40.1 的 CDR-vs-RN `MS比`。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from plcsp_v3.env.shi_instances import (POOL_STRIDE, PT_REDRAW_SEED,  # noqa: E402
                                        make_shi_instance)
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, PA_RULE_NAMES,  # noqa: E402
                                apply_pa_rule, la_rule_policy)
from plcsp_v3.q1_flat import build, dims_from_shop_len  # noqa: E402

JOBS = [int(x) for x in os.environ.get(
    "SZ_JOBS", "10,16,22,28,34,40").split(",")]
N_INST = int(os.environ.get("SZ_NINST", "3"))
SHOPLEN = float(os.environ.get("SZ_SHOPLEN", "200"))
# ⚠️ **新布局口径**（2026-10-18）：四项都给了就**覆盖** `shop_len`。
#    均衡点：机位 15 m · 走廊 10 m · L/U 25 m ⟹ 间距 25 m、车间 150 × 125 m。
CELL = os.environ.get("SZ_CELL", "")
AISLE = os.environ.get("SZ_AISLE", "")
LU = os.environ.get("SZ_LU", "")
ALTS_MIN = int(os.environ.get("SZ_ALTSMIN", "8"))
ALTS_MAX = int(os.environ.get("SZ_ALTSMAX", "16"))
SPEED = float(os.environ.get("SZ_SPEED", "0.5"))
HANDLING = float(os.environ.get("SZ_H", "0.0"))
N_AGV = int(os.environ.get("SZ_NAGV", "6"))
SEED0 = 100_000


def build_spec(inst):
    kw = dict(layout_scheme="random", layout_cols=5, layout_rows=5,
              shi2025=True,
              **dims_from_shop_len(SHOPLEN, inst.n_machines, layout_cols=5))
    if CELL:
        kw.update(cell_w=float(CELL), cell_h=float(CELL),
                  aisle_w=float(AISLE), lu_connector_m=float(LU))
    return build(
        seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV,
        finite_buffer=True, layout_seed=0, reward_norm="none", tau=2.70,
        sim_overrides={
            "due_range": 0.6, "la_tie_random": 1.0,
            "agv_plan_pos": 1.0, "agv_act_book": 1.0,
            "skip_same_machine_move": 1.0,
            "agv_speed_mps": SPEED,
            "agv_handling_lo": HANDLING, "agv_handling_hi": HANDLING,
        },
        inst_obj=inst, **kw)[0]


def run_cdr(spec, k: int, j: int) -> tuple[float, float]:
    m = spec.world().run_gated(
        seed_chain=SEED0,
        policy_pa=lambda v, p, tr=None, _k=k: apply_pa_rule(_k, v, p, tr),
        policy_l=la_rule_policy(j), dt_events=0)
    return float(m["makespan"]), float(m["logistics_cost"])


def main() -> None:
    layout_tag = (f"机位 {CELL} m · 走廊 {AISLE} m · L/U {LU} m"
                  if CELL else f"车间 {SHOPLEN:.0f} m")
    print(f"**多规模权衡** · {layout_tag} · 候选 {ALTS_MIN}–{ALTS_MAX} · "
          f"车速 {SPEED} m/s · h={HANDLING * 60:.1f} s · {N_AGV} 车\n"
          f"  {N_INST} 实例/规模 · 每条 27 CDR · 故障 1–8 / 修复 5–10\n"
          f"  原文：`MS付 / TC付 = 1.43 / 1.43`（`Exp(16/140)`）\n", flush=True)
    print(f"  {'工件':>4}{'真工序':>7}{'MS最好':>9}{'它的TC':>9}"
          f"{'TC最好':>9}{'它的MS':>9}{'MS付':>7}{'TC付':>7}{'前沿':>6}")
    print("  " + "-" * 76)
    for si, jobs in enumerate(JOBS):
        acc = []
        n_ops = 0
        for k in range(N_INST):
            seed = POOL_STRIDE * (1000 + 10 * si + k)
            rot = PT_REDRAW_SEED + 1000 + 10 * si + k
            inst = make_shi_instance(
                jobs, seed=seed, rot_seed=rot, alts_dist="uniform",
                alts_min=ALTS_MIN, alts_max=ALTS_MAX, pt_dist="uniform",
                pt_lo=2, pt_hi=10)
            n_ops = sum(len(j) for j in inst.jobs)
            spec = build_spec(inst)
            rows = [(PA_RULE_NAMES[pk] + "+" + LA_RULE_NAMES[lj],
                     *run_cdr(spec, pk, lj))
                    for pk in range(9) for lj in range(len(LA_RULE_NAMES))]
            named = rows
            ms = np.array([r[1] for r in rows])
            tc = np.array([r[2] for r in rows])
            i_ms, i_tc = int(np.argmin(ms)), int(np.argmin(tc))
            front = [i for i in range(len(rows))
                     if not any((ms[q] <= ms[i] and tc[q] <= tc[i]) and
                                (ms[q] < ms[i] or tc[q] < tc[i])
                                for q in range(len(rows)))]
            bm, bt = named[i_ms], named[i_tc]
            print(f"    · Exp({jobs}/{n_ops}) 实例 {k + 1}/{N_INST}")
            print(f"        最好 MS：{bm[0]:<30} MS={bm[1]:>8.1f}"
                  f"  它的 TC={bm[2]:>8.1f}")
            print(f"        最好 TC：{bt[0]:<30} TC={bt[2]:>8.1f}"
                  f"  它的 MS={bt[1]:>8.1f}")
            print(f"        MS付={ms[i_tc] / ms[i_ms]:.2f}  "
                  f"TC付={tc[i_ms] / tc[i_tc]:.2f}  前沿={len(front)}",
                  flush=True)
            acc.append((ms[i_ms], tc[i_ms], ms[i_tc], tc[i_tc],
                        ms[i_tc] / ms[i_ms], tc[i_ms] / tc[i_tc], len(front)))
        a = np.asarray(acc, dtype=float).mean(axis=0)
        # ⚠️ 列序：MS最好 · 它的TC · **TC最好** · **它的MS** · MS付 · TC付 · 前沿
        #    `acc` 里存的是 `(ms_best, tc_of_ms_best, ms_of_tc_best, tc_best, ...)`
        #    ⟹ 第三、四列要**交换**打印（2026-10-18 修：此前把 403.7 打进了 TC 列）。
        print(f"  {jobs:>4}{n_ops:>7}{a[0]:>9.1f}{a[1]:>9.1f}"
              f"{a[3]:>9.1f}{a[2]:>9.1f}{a[4]:>7.2f}{a[5]:>7.2f}{a[6]:>6.1f}",
              flush=True)
    print("\n  ⚠️ `MS付` / `TC付` 是 **27 条 CDR 内部**的两个角，不是 CDR-vs-RN。")
    print("  ⚠️ 单实例比值噪声大；看跨规模的趋势，不把单格当定论。")


if __name__ == "__main__":
    main()
