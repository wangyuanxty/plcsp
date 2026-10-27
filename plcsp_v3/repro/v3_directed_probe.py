# -*- coding: utf-8 -*-
"""**有向走廊 vs 无向走廊** —— 照原文 Fig. 4 的箭头（2026-10-17，用户发现）。

══ 发现 ══

**原文 Fig. 4 的走廊不是全双向的**（第 14 页高倍渲染逐条读箭头）：

```
横向：每条边**两端都有箭头** ⟹ **双向**
纵向：**一整条只有一个方向**，逐列交替
      竖线 c = 0 ↑ · 1 ↓ · 2 ↑ · 3 ↓ · 4 ↑
⟹ **d(A→B) ≠ d(B→A)**，AGV 的「位置」带上方向性
```

**而本仓的 `build_corridor_graph` 是 `nx.Graph`（无向）⟹ 这个结构从来没有过。**

⚠️ **绕行上限**：纵向单向时，往上走若在「向下」列，要先横到相邻「向上」列、上、再横回来
   ⟹ **最多多两个横向 hop**（240 m 车间里 = 80 m ≈ 2.7 min）。**效应真实但有上限。**

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

EIDX = int(os.environ.get("DC_EIDX", "199"))
SHOPLEN = float(os.environ.get("DC_SHOPLEN", "240"))
ALTS_MAX = int(os.environ.get("DC_ALTSMAX", "10"))
N_INST = int(os.environ.get("DC_NINST", "5"))
N_RUN = int(os.environ.get("DC_RN", "10"))


def build_spec(inst, directed: bool, mid_edge: bool = False):
    ovr = {"due_range": 0.6, "la_tie_random": 1.0,
           "agv_mttr_lo": 10.0, "agv_mttr_hi": 10.0,
           "agv_plan_pos": 1.0, "agv_act_book": 1.0}
    return build(seed=0, pm_interval=30, target_failures=4, n_agv=6,
                 finite_buffer=True, layout_seed=0, reward_norm="minmax",
                 tau=2.70, sim_overrides=ovr, inst_obj=inst,
                 layout_scheme="random", **dims_from_shop_len(SHOPLEN, inst.n_machines),
                 directed_corridors=directed, mid_edge_dock=mid_edge, shi2025=False,
                 constraints=ConstraintConfig(finite_buffer=False, machine_failure=False,
                                              maintenance=False, due_dates=False,
                                              agv_failure=False))[0]


def main() -> None:
    print(f"**走廊方向 × 机台接入点** · Exp(16/140) · 车间 {SHOPLEN:.0f} m · "
          f"alts_max={ALTS_MAX} · 6 车 · **{N_INST} 实例均值**\n")
    print(f"  {'走廊':<6}{'接入':<6}{'MS@MS最好':>10}{'TC@MS最好':>10}"
          f"{'MS@TC最好':>10}{'TC@TC最好':>10}{'MS付':>7}{'TC付':>7}{'前沿':>7}")
    print("  " + "-" * 80)
    for tag, d, med in (("无向", False, False), ("有向", True, False),
                        ("无向", False, True), ("有向", True, True)):
        c = np.zeros(4)
        pm, pt, fr = [], [], []
        for k in range(N_INST):
            inst = eval_instance(140, 16, EIDX, idx=EIDX + k,
                                 alts_dist="uniform", alts_max=ALTS_MAX,
                                 pt_dist="uniform")
            spec = build_spec(inst, d, med)
            rows = []
            for kk in range(9):
                for j in range(len(LA_RULE_NAMES)):
                    ms, tc = H.run_cdr(spec, kk, j)
                    rows.append((PA_RULE_NAMES[kk], LA_RULE_NAMES[j], ms, tc))
            ms = np.array([r[2] for r in rows])
            tc = np.array([r[3] for r in rows])
            i_ms, i_tc = int(np.argmin(ms)), int(np.argmin(tc))
            c += np.array([ms[i_ms], tc[i_ms], ms[i_tc], tc[i_tc]])
            pm.append(ms[i_tc] / ms[i_ms])
            pt.append(tc[i_ms] / tc[i_tc])
            front = [i for i in range(len(rows))
                     if not any((ms[q] <= ms[i] and tc[q] <= tc[i]) and
                                (ms[q] < ms[i] or tc[q] < tc[i]) for q in range(len(rows)))]
            fr.append(len(front))
        a, b, cc, dd = c / N_INST
        print(f"  {tag:<6}{'中点' if med else '路口':<6}{a:>10.1f}{b:>10.1f}"
              f"{cc:>10.1f}{dd:>10.1f}{np.mean(pm):>7.2f}{np.mean(pt):>7.2f}"
              f"{np.mean(fr):>7.1f}", flush=True)
    print("\n  ⚠️ 原文：`MS付 / TC付 = 1.43 / 1.43`。")
    print("  ⚠️ `接入=路口` = 旧行为（机台在路口，两条走廊都能直接进）。")
    print("     `接入=中点` = 照 Fig. 4（机台只接横走廊，必须先横走到路口才能转竖）。")


if __name__ == "__main__":
    main()
