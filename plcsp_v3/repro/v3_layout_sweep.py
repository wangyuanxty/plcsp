# -*- coding: utf-8 -*-
"""**5×5 布局 × 车间长度 100–200 m 扫描**（2026-10-17，用户给定）。

══ 用户的两条指令 ══

> *"布局改成5*5吧， 25个格子里面随机选18个"*
> *"然后重新从100 -200 扫描下 长度"*

**为什么这两条是配套的**：5×5 比现在的 4×5 多一行 ⟹ **趟更长** ⟹ 缩 `shop_len` 正好补偿。
**而 §4.17.2 那张表说：车间越小、`RN/CDR` 越高**（`scale` 12.5→2 时 0.51→**0.89**）
—— **所以这个方向是有依据的，不是新猜。**

══ 判据（唯一有原文锚的那条）══

```
最好 CDR 的 MS ÷ RN 的 MS
  原文 Fig. 6(b)(a)：610 / 428 = **1.43**   ⟹ **RN 在 MS 上赢规则**
  本仓现状：**0.74**                        ⟹ 方向反着
```

⚠️ **`shop_len = 240` 是 2026-10-16 由「TC ≈ 420 靶」标定的**；本扫描**故意离开那个值**
去看形状怎么走，**故 TC 会脱靶** —— 这是**有意的**，不是失误。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import v3_hv3 as H  # noqa: E402
from plcsp_v3.env import layout as layout_mod  # noqa: E402
from plcsp_v3.env.constraints import ConstraintConfig  # noqa: E402
from plcsp_v3.env.corridors import (build_corridor_graph,  # noqa: E402
                                    dock_distance_matrix)
from plcsp_v3.env.shi_instances import (EVAL_IDX, N_AGV_SHI,  # noqa: E402
                                        eval_instance)
from plcsp_v3.hrl.rules import LA_RULE_NAMES, N_AEI2025_RULES  # noqa: E402
from plcsp_v3.q1_flat import SHI_AISLE_W, build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

EIDX = int(os.environ.get("LS2_EIDX", str(EVAL_IDX)))
LENS = [float(x) for x in os.environ.get("LS2_LENS", "100,120,140,160,180,200").split(",")]
ROWS = int(os.environ.get("LS2_ROWS", "5"))
COLS = int(os.environ.get("LS2_COLS", "5"))
N_COLS_ALTS = 10          # `alts_max`
NOFAIL = os.environ.get("LS2_NOFAIL", "1") == "1"
N_RUN = int(os.environ.get("LS2_RN", "10"))
SC_FIX = int(os.environ.get("LS2_SC0", "100000"))


def trip_stats(shop_len: float, rows: int, cols: int) -> tuple[float, float]:
    """该布局下**机台两两**行程时长的 (均值, 中位) [min] —— 只建布局与距离矩阵。"""
    pitch = shop_len / (cols + 1)
    side = pitch - SHI_AISLE_W
    lay = layout_mod.sample_layout(
        N_AGV_SHI + 12, seed=0, cell_w=side, cell_h=side, aisle_w=SHI_AISLE_W,
        lu_connector_m=pitch, n_cols=cols, n_rows=rows, cap_fixed=None,
        scheme="random", n_agv=N_AGV_SHI)
    dm = dock_distance_matrix(build_corridor_graph(lay))
    dock = [int(p.dock_node) for p in lay.machines]
    eff = min(1.0, max(0.4, SHI_AISLE_W / 1.5))
    a = np.array(dock)[:, None]
    v = dm[a, a.T] / (eff * 0.5) / 60.0
    return float(np.mean(v)), float(np.median(v))


def build_spec(inst, shop_len: float, rows: int, cols: int):
    ovr = {"due_range": 0.6, "la_tie_random": 1.0,
           "agv_mttr_lo": 10.0, "agv_mttr_hi": 10.0}
    kw = (dict(shi2025=False,
               constraints=ConstraintConfig(finite_buffer=False, machine_failure=False,
                                            maintenance=False, due_dates=False,
                                            agv_failure=False))
          if NOFAIL else dict(shi2025=True))
    return build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                 finite_buffer=True, layout_seed=0, reward_norm="minmax",
                 tau=2.70, sim_overrides=ovr, inst_obj=inst,
                 layout_scheme="random", **dims_from_shop_len(shop_len, inst.n_machines, layout_cols=cols), layout_cols=cols,
                 layout_rows=rows, **kw)[0]


def main() -> None:
    inst = eval_instance(140, 16, EIDX, idx=EIDX,
                         alts_pow=-2.0, alts_dist="uniform", pt_dist="uniform")
    nk = np.mean([len(op) for job in inst.jobs for op in job])
    print(f"**{ROWS}×{COLS} 布局 × 车间长度扫描** · Exp(16/140) · idx={EIDX} · "
          f"候选分布=均匀 · 故障={'关' if NOFAIL else '开'}\n"
          f"  ⚠️ 候选均值 {nk:.2f} 台/工序 · 每个长度跑 27 条 CDR + {N_RUN} 局 RN\n", flush=True)
    print(f"  {'车间[m]':>8}{'MS最好':>9}{'@TC':>9}{'TC最好':>9}{'@MS':>9}"
          f"{'每趟':>7}{'MS付':>8}{'TC付':>8}{'MS付∪':>8}{'TC付∪':>8}{'MS比':>8}")
    print("  " + "-" * 92)
    for sl in LENS:
        mu, med = trip_stats(sl, ROWS, COLS)
        spec = build_spec(inst, sl, ROWS, COLS)
        ms, tc = [], []
        for k in range(N_AEI2025_RULES):
            for j in range(len(LA_RULE_NAMES)):
                m_, t_ = H.run_cdr(spec, k, j)
                ms.append(m_)
                tc.append(t_)
        ms, tc = np.asarray(ms), np.asarray(tc)
        # ⚠️ **RN 的 TC 也要收**（2026-10-17）：原文那两个角是「CDR 与 RN」，
        #    所以「并集的两个角」必须把 RN 算进去，否则口径不同（用户指出）。
        rn_pts = np.asarray([H.run_rn(spec, i) for i in range(N_RUN)])
        rn_ms = float(rn_pts[:, 0].mean())
        ratio = ms.min() / rn_ms
        # ⚠️ **前沿的两个角**（用户 2026-10-17 提出）—— 比 `corr` 干净得多：
        #    不含 HV、不含归一化、不含基准。就是「为拿最好的 MS，TC 要付几倍」与反向。
        i_ms, i_tc = int(np.argmin(ms)), int(np.argmin(tc))
        pay_ms = ms[i_tc] / ms[i_ms]        # 为拿最好的 TC，MS 付几倍（**仅 CDR**）
        pay_tc = tc[i_ms] / tc[i_tc]        # 为拿最好的 MS，TC 付几倍（**仅 CDR**）
        # ⚠️ **并集口径**（27 CDR ∪ RN 的 N_RUN 局）—— 与原文的构成一致
        all_pts = np.vstack([np.column_stack([ms, tc]), rn_pts])
        j_ms, j_tc = int(np.argmin(all_pts[:, 0])), int(np.argmin(all_pts[:, 1]))
        pay_ms_u = all_pts[j_tc, 0] / all_pts[j_ms, 0]
        pay_tc_u = all_pts[j_ms, 1] / all_pts[j_tc, 1]
        per_trip = 2.0 * (tc[i_tc] - 42.0) / 140.0
        flag = "  ← 翻" if ratio > 1.0 else ""
        print(f"  {sl:>8.0f}{ms[i_ms]:>9.1f}{tc[i_ms]:>9.1f}{ms[i_tc]:>9.1f}"
              f"{tc[i_tc]:>9.1f}{per_trip:>7.2f}{pay_ms:>8.2f}{pay_tc:>8.2f}"
              f"{pay_ms_u:>8.2f}{pay_tc_u:>8.2f}{ratio:>8.2f}{flag}", flush=True)
    print("\n  ⚠️ **`MS付` / `TC付` = 27 条 CDR **内部**的两个角**（不含 RN）。")
    print("  ⚠️ **`MS付∪` / `TC付∪` = 27 条 CDR ∪ RN 的并集的两个角** —— **与原文的构成一致**")
    print("     （原文那两个点是：最好的 CDR 610/420 与 RN 428/600）。")
    print("     **原文 = 610/428 = 1.43 · 600/420 = 1.43 **（并集口径）。")
    print("  ⚠️ `MS比` = 最好 CDR 的 MS ÷ RN 的 MS：**> 1 ⟹ RN 赢规则**（原文 **1.43**）。")
    print("  ⚠️ 本扫描**故意**离开标定尺度 240 m（那是按 TC 靶标的），**故 TC 会脱靶**。")


if __name__ == "__main__":
    main()
