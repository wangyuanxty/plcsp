# -*- coding: utf-8 -*-
"""**原文 Fig. 4 的排布 vs 本仓的随机撒** —— 复现层从没用过 Fig. 4（2026-10-17）。

══ 为什么现在才跑 ══

```
`layout_scheme` 一直是 `"random"`（机位随机撒）。
理由（`layout.py:68`）：
  「`pt` 与候选集都独立随机 ⟹ 机台位置与能力统计独立
   ⟹ 换排布 ≡ 换编号 ⟹ 所有统计量不变」

**而那条理由的前提 ——「pt/候选与位置无关」—— 只是**我们的假设**，原文没这么说。**
原文说 *"The overall shop layout and transfer routes between two locations are
**fixed and known**, as shown in **Fig. 4**"*。
⟹ **Fig. 4 的具体排布原文给了，我们一次没用过。**
```

**⟹ 如果那个排布里「同类零件的可用机台集中在一侧」或「快的机台成组」，**
**随机撒就把这个结构抹掉了。**

══ 判据 ══

```
`MS付` / `TC付`：前沿两个角 —— 原文 1.43 / 1.43
`前沿`：Pareto 点数
`通吃?`：MS 最好与 TC 最好是不是同一条
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
from plcsp_v3.env.shi_instances import (EVAL_IDX, eval_instance)  # noqa: E402
from plcsp_v3.hrl.rules import LA_RULE_NAMES, PA_RULE_NAMES  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

EIDX = int(os.environ.get("FP_EIDX", str(EVAL_IDX)))
SHOPLEN = float(os.environ.get("FP_SHOPLEN", "240"))
ALTS_MAX = int(os.environ.get("FP_ALTSMAX", "10"))
N_INST = int(os.environ.get("FP_NINST", "3"))
N_RUN = int(os.environ.get("FP_RN", "10"))


def build_spec(inst, scheme: str):
    ovr = {"due_range": 0.6, "la_tie_random": 1.0,
           "agv_mttr_lo": 10.0, "agv_mttr_hi": 10.0,
           "agv_plan_pos": 1.0, "agv_act_book": 1.0}
    return build(seed=0, pm_interval=30, target_failures=4, n_agv=6,
                 finite_buffer=True, layout_seed=0, reward_norm="minmax",
                 tau=2.70, sim_overrides=ovr, inst_obj=inst,
                 layout_scheme=scheme, **dims_from_shop_len(SHOPLEN, inst.n_machines), shi2025=False,
                 constraints=ConstraintConfig(finite_buffer=False, machine_failure=False,
                                              maintenance=False, due_dates=False,
                                              agv_failure=False))[0]


def main() -> None:
    print(f"**Fig. 4 排布 vs 随机撒** · Exp(16/140) · 车间 {SHOPLEN:.0f} m · "
          f"alts_max={ALTS_MAX} · 6 车 · **{N_INST} 个实例均值 ± sd**\n", flush=True)
    print(f"  {'排布':<10}{'MS付':>8}{'±':>6}{'TC付':>8}{'±':>6}"
          f"{'前沿':>7}{'通吃?':>8}{'MS比':>8}")
    print("  " + "-" * 64)
    for scheme in ("random", "fig4"):
        pm, pt, fr, same_n, mr = [], [], [], 0, []
        for k_inst in range(N_INST):
            inst = eval_instance(140, 16, EIDX, idx=EIDX + k_inst,
                                 alts_dist="uniform", alts_max=ALTS_MAX,
                                 pt_dist="uniform")
            spec = build_spec(inst, scheme)
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
            pm.append(ms[i_tc] / ms[i_ms])
            pt.append(tc[i_ms] / tc[i_tc])
            fr.append(len(front))
            same_n += int(i_ms == i_tc)
            mr.append(ms.min() / rn_ms)
        print(f"  {scheme:<10}{np.mean(pm):>8.2f}{np.std(pm):>6.2f}"
              f"{np.mean(pt):>8.2f}{np.std(pt):>6.2f}"
              f"{np.mean(fr):>7.1f}{same_n:>8d}{np.mean(mr):>8.2f}", flush=True)
    print("\n  ⚠️ `通吃?` 列 = {N_INST} 个实例里有几个「MS 最好与 TC 最好同一条」。")
    print("  ⚠️ 原文：`MS付 / TC付 = 1.43 / 1.43`。")
    print("  ⚠️ `fig4` = 原文 Fig. 4 逐格读出的排布（`layout.FIG4_PLACEMENT`），**尺寸原文没给**。")


if __name__ == "__main__":
    main()
