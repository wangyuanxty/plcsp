# -*- coding: utf-8 -*-
"""**泄露检验（盲化）**：把 PA 的计划账本清零，看 `EAJ+MINU+MINCOST` 还支不支配（2026-10-17）。

══ 用户的问题 ══

> *"这条规则是泄露给他实际不应该看到的信息了吗？"*
> （指的是 `alts_max=5` 下**同时**做到 MS 最小与 TC 最小的 `EAJ+MINU+MINCOST`）

**已查**：三个分量都只用「已派任务账本（`pa_mct` ≡ 原文 `MCT_k`）」与「当前位置」，无未来信息。

**⚠️ 但 `EAJ` 与 `MINU` 共用**同一个账本** —— 若那个账本比「物理实况」乐观，
规则会系统性地把活送到「**计划上**最闲」的机器。

══ 本脚本的做法 ══

`SimConfig.pa_blind = True` ⟹ `_pa_dispatch` 把**喂给规则视图**的
`pa_mct` / `pa_assigned_min` 换成**全 0**（真账本仍按原口径推进）。

⟹ **规则看不到「已经派了什么」**：
   `EAJ` 退化成「都用 now」· `MINU` 退化成「全并列 → 平局随机」· `EAM` 同理。

**判据**：同一批 27 条，在**盲化**下再排一次 ——
```
仍然出现「同一条同时最好」或前沿很薄 ⟹ 优势**不来自账本**
前沿变宽、那条不再支配     ⟹ 优势**来自账本** ⟹ 要查账本口径
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

EIDX = int(os.environ.get("BL_EIDX", str(EVAL_IDX)))
SHOPLEN = float(os.environ.get("BL_SHOPLEN", "240"))
ALTS_MAX = int(os.environ.get("BL_ALTSMAX", "5"))


def build_spec(inst, blind: bool):
    ovr = {"due_range": 0.6, "la_tie_random": 1.0,
           "agv_mttr_lo": 10.0, "agv_mttr_hi": 10.0,
           "agv_plan_pos": 1.0, "agv_act_book": 1.0,
           "pa_blind": 1.0 if blind else 0.0}
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
    for blind in (False, True):
        spec = build_spec(inst, blind)
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
        tag = "**盲化（看不到已派账本）**" if blind else "正常（有账本）"
        print(f"\n════ {tag} · alts_max={ALTS_MAX} · 车间 {SHOPLEN:.0f} m ════")
        print(f"  MS 最好：{rows[i_ms][0]}+{rows[i_ms][1]}  ({ms[i_ms]:.1f}, {tc[i_ms]:.1f})")
        print(f"  TC 最好：{rows[i_tc][0]}+{rows[i_tc][1]}  ({ms[i_tc]:.1f}, {tc[i_tc]:.1f})")
        print(f"  ⟹ **{'同一条（前沿=1 点）' if i_ms == i_tc else '不同'}**"
              f" · Pareto 前沿 **{len(front)} 条**")
        print(f"  ⟹ MS 跨度 {ms.min():.1f}–{ms.max():.1f}"
              f" · TC 跨度 {tc.min():.1f}–{tc.max():.1f}"
              f" · **TC 可压 {tc.max() / tc.min() - 1:.1%}**")
        if len(front) > 1:
            print("  前沿：" + "；".join(
                f"{rows[i][0]}+{rows[i][1]}({ms[i]:.0f},{tc[i]:.0f})" for i in front))


if __name__ == "__main__":
    main()
