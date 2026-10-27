# -*- coding: utf-8 -*-
"""**权衡宽度 vs 实例规模** —— 用原文自己的 18 个实例档（2026-10-17）。

══ 为什么扫这个 ══

**规模是唯一一个「原文明确跑过、我们没量过它跟权衡的关系」的轴。**
大实例里工件多、争用强 —— **两难可能自己变宽，不需要我造任何参数。**

⚠️ 前面已经证过：
- 换几何（排布 / 间距 / 网格形状 / 车间长度）⟹ 无效，因为**没有机制跟位置联动**
- 换分布（候选机数 / pt 分布）⟹ 无效
- 换车数 ⟹ 方向对但非单调
- **加「快慢↔远近」耦合（`pt_pos_corr`）⟹ 有效**（`TC付` 1.18 → 1.34），**但原文没有这个参数**

**⟹ 本扫描不加任何参数，只换规模。**

══ 判据 ══

```
`MS付` / `TC付`：前沿两个角（原文 1.43 / 1.43）
`前沿`：Pareto 点数
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
from plcsp_v3.env.shi_instances import eval_instance  # noqa: E402
from plcsp_v3.hrl.rules import LA_RULE_NAMES, PA_RULE_NAMES  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

# 原文 Table 9 的实例档：`Exp(工件数 / 工序数含虚拟)`。
SIZES = [(int(a), int(b)) for a, b in
         (x.split("/") for x in os.environ.get(
             "SZ_SIZES", "7/59,10/86,16/140,20/173,30/261").split(","))]
EIDX = int(os.environ.get("SZ_EIDX", "199"))
SHOPLEN = float(os.environ.get("SZ_SHOPLEN", "240"))
ALTS_MAX = int(os.environ.get("SZ_ALTSMAX", "10"))
N_RUN = int(os.environ.get("SZ_RN", "10"))
# ⚠️ **多实例**（2026-10-17 加）：单实例的比值噪声大，要确认 1.2–1.3 站得住。
N_INST = int(os.environ.get("SZ_NINST", "1"))


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
    print(f"**权衡宽度 vs 规模** · 车间 {SHOPLEN:.0f} m · alts_max={ALTS_MAX} · 6 车\n"
          "  原文：`MS付 / TC付 = 1.43 / 1.43`（`Exp(16/140)`）\n", flush=True)
    print(f"  {'实例':>14}{'真工序':>7}{'MS@MS最好':>10}{'TC@MS最好':>10}"
          f"{'MS@TC最好':>10}{'TC@TC最好':>10}{'运输比':>8}{'MS付':>7}{'TC付':>7}")
    print("  " + "-" * 92)
    for jobs, tot in SIZES:
        pm, pt, fr, mr, n_ops = [], [], [], [], 0
        c_ms = np.zeros(4)
        tr_ratio = []
        for k_inst in range(N_INST):
            inst = eval_instance(tot, jobs, EIDX, idx=EIDX + k_inst,
                                 alts_dist="uniform", alts_max=ALTS_MAX,
                                 pt_dist="uniform")
            n_ops = sum(len(j) for j in inst.jobs)
            spec = build_spec(inst)
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
            rn_ms = float(np.mean([H.run_rn(spec, i)[0] for i in range(N_RUN)]))
            c_ms += np.array([ms[i_ms], tc[i_ms], ms[i_tc], tc[i_tc]])
            # 运输工时 = 2·(TC − 42)；两边同乘 0.5，比值不受影响
            tr_ratio.append((tc[i_ms] - 42.0) / (tc[i_tc] - 42.0))
            pm.append(ms[i_tc] / ms[i_ms])
            pt.append(tc[i_ms] / tc[i_tc])
            fr.append(len(front))
            mr.append(ms.min() / rn_ms)
        a, b, c, d = c_ms / N_INST
        print(f"  {f'Exp({jobs}/{tot})':>14}{n_ops:>7d}"
              f"{a:>10.1f}{b:>10.1f}{c:>10.1f}{d:>10.1f}"
              f"{np.mean(tr_ratio):>8.2f}"
              f"{np.mean(pm):>7.2f}{np.mean(pt):>7.2f}   "
              f"({N_INST} 实例 · 前沿 {np.mean(fr):.1f})", flush=True)
    print("\n  ⚠️ `MS付` / `TC付` 是 **CDR 内部**的两个角。")
    print("  ⚠️ 本扫描**不加任何参数** —— 只换实例规模。")


if __name__ == "__main__":
    main()
