# -*- coding: utf-8 -*-
"""**mk01–mk10 的 27 条 CDR 全扫（MS / TC）**。

用途：在**新的机位口径**下，逐实例看 27 条 CDR 的两个目标。

口径：

| 项 | 值 |
|---|---|
| 实例 | `load_mk(mk01..mk10)`，加工数据一字不改 |
| 布局 | 机位 15 × 15 m · 走廊 10 m · L/U 连接段 25 m |
| 排布 | `grid`（逐行填）+ `grid_shape(n_machines)` |
| 车队 | 6 台 |
| 车速 | 0.5 m/s |
| 装卸 | 10 s / 次（= 1/6 min） |
| 约束 | `shi2025=True`（只留 AGV 故障；装卸进 tt） |
| 场景 | `100_000 + 13i`，默认 3 个 |
| 目标 | `makespan`（MS）· `logistics_cost`（TC, Eq.2） |

⚠️ MK 的加工数据没有“工件类别”，也没有回站 dummy op；每个作业独立。
⚠️ 这里用的是 **v3 几何布局**，不是 MKT 的行程矩阵。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from plcsp_v3.env.instances import load_mk  # noqa: E402
from plcsp_v3.hrl.rules import (N_AEI2025_RULES, PA_RULE_NAMES,  # noqa: E402
                                apply_pa_rule, la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402

N_SCEN = int(os.environ.get("MKC_SCEN", "3"))
N_AGV = int(os.environ.get("MKC_AGV", "6"))
SPEED = float(os.environ.get("MKC_SPEED", "0.5"))
HANDLING_MIN = float(os.environ.get("MKC_HANDLING_MIN", "0.0"))
CELL = float(os.environ.get("MKC_CELL", "15.0"))
AISLE = float(os.environ.get("MKC_AISLE", "10.0"))
LU = float(os.environ.get("MKC_LU", "25.0"))
SCHEME = os.environ.get("MKC_SCHEME", "grid")

LA_NAMES = ("MINCOST", "MINU", "LATEST_IN_TIME")


def make_spec(name: str):
    inst = load_mk(name)
    spec, _ = build(
        seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV,
        finite_buffer=True, layout_seed=0, reward_norm="none",
        tau=2.70,
        sim_overrides={
            "due_range": 0.6,
            "agv_speed_mps": SPEED,
            "agv_handling_lo": HANDLING_MIN,
            "agv_handling_hi": HANDLING_MIN,
            "agv_plan_pos": 1.0,
            "agv_act_book": 1.0,
            "skip_same_machine_move": True,
        },
        inst_obj=inst, layout_scheme=SCHEME,
        cell_w=CELL, cell_h=CELL, aisle_w=AISLE, lu_connector_m=LU,
        shi2025=True)
    return inst, spec


def run_one(spec, k: int, j: int, scs: list[int]) -> tuple[float, float]:
    def pol_pa(view, pool, tie_rng=None):
        return apply_pa_rule(k, view, pool, tie_rng)

    pol_l = la_rule_policy(j)
    rows = [spec.world().run_gated(seed_chain=sc, policy_pa=pol_pa,
                                   policy_l=pol_l, dt_events=0)
            for sc in scs]
    return (float(np.mean([float(r["makespan"]) for r in rows])),
            float(np.mean([float(r["logistics_cost"]) for r in rows])))


def nondominated(pts: list[tuple[float, float]]) -> list[int]:
    out = []
    for i, (mi, ti) in enumerate(pts):
        if not any((mj <= mi and tj <= ti) and (mj < mi or tj < ti)
                   for j, (mj, tj) in enumerate(pts) if j != i):
            out.append(i)
    return out


def main() -> None:
    names = [f"mk{k:02d}" for k in range(1, 11)]
    scs = [100_000 + 13 * i for i in range(N_SCEN)]
    print(f"mk01–mk10 · 27 CDR · {N_SCEN} 场景 · "
          f"机位 {CELL:.0f}m · 走廊 {AISLE:.0f}m · L/U {LU:.0f}m · "
          f"v={N_AGV} · speed={SPEED}\n", flush=True)
    print(f"  {'实例':<6} | {'工件':>4} | {'机台':>4} | "
          f"{'最好 MS 规则':<26} | {'MS':>8} | "
          f"{'最好 TC 规则':<26} | {'TC':>8} | "
          f"{'MS付':>6} | {'TC付':>6} | {'非支配':>5}")
    print("  " + "-" * 128)

    for name in names:
        inst, spec = make_spec(name)
        rows: list[tuple[str, str, float, float]] = []
        for k in range(N_AEI2025_RULES):
            for j, la in enumerate(LA_NAMES):
                ms, tc = run_one(spec, k, j, scs)
                rows.append((PA_RULE_NAMES[k], la, ms, tc))
        pts = [(r[2], r[3]) for r in rows]
        i_ms = min(range(len(rows)), key=lambda i: pts[i][0])
        i_tc = min(range(len(rows)), key=lambda i: pts[i][1])
        ms_corner = pts[i_tc][0] / max(pts[i_ms][0], 1e-9)
        tc_corner = pts[i_ms][1] / max(pts[i_tc][1], 1e-9)
        nd = nondominated(pts)
        print(f"  {name:<6} | {inst.n_jobs:>4} | {inst.n_machines:>4} | "
              f"{rows[i_ms][0] + '+' + rows[i_ms][1]:<26} | {pts[i_ms][0]:>8.1f} | "
              f"{rows[i_tc][0] + '+' + rows[i_tc][1]:<26} | {pts[i_tc][1]:>8.1f} | "
              f"{ms_corner:>6.2f} | {tc_corner:>6.2f} | {len(nd):>5}", flush=True)

    print("\n  ⚠️ MS付 = MS(TC 最好) ÷ MS(MS 最好) · "
          "TC付 = TC(MS 最好) ÷ TC(TC 最好)。原文两角 ≈ 1.43 / 1.43。")


if __name__ == "__main__":
    main()
