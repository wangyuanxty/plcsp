# -*- coding: utf-8 -*-
"""**`pt ∈ [2,20]` × 车间长度** —— 找与原文量级更接近的 (MS, TC)（2026-10-17，用户要求）。

⚠️ **`pt` 上界 20 违反原文规格**（原文 §5.1 写 *"between 2 and 10"*）。
   **本扫描是敏感度测试，结果不得当复现读数。**

══ 为什么值得扫 ══

```
2–10 时（车间 240）：MS 最好 328.5 · TC 最好 446.1
原文               ：MS 最好 610   · TC 最好 420

pt 拉到 2–20 后，机台总工时约翻倍 ⟹ MS 上升
⟹ 有可能 **MS 更接近原文的 610、TC 仍接近 420**
```

**判据：哪一档的 `(MS最好, TC最好)` 同时接近原文的 `(610, 420)`。**
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from plcsp_v3.env.shi_instances import eval_instance  # noqa: E402
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, apply_pa_rule,  # noqa: E402
                                la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

EIDX = int(os.environ.get("P2_EIDX", "199"))
LENS = [float(x) for x in os.environ.get("P2_LENS", "100,140,180,220,260,300").split(",")]
ALTS_MAX = int(os.environ.get("P2_ALTSMAX", "10"))
ALTS_MIN = int(os.environ.get("P2_ALTSMIN", "1"))
PT_HI = int(os.environ.get("P2_PTHI", "20"))
PT_LO = int(os.environ.get("P2_PTLO", "2"))
# ⚠️ **AGV 台数**（2026-10-17 加）：原文 **6**。
NAGV = int(os.environ.get("P2_NAGV", "6"))
N_INST = int(os.environ.get("P2_NINST", "3"))
# ⚠️ **装卸随机**（2026-10-17，用户指定 1–3 分钟）：`hi = 0` ⟹ 不开（旧行为）。
HLO = float(os.environ.get("P2_HLO", "1"))
HHI = float(os.environ.get("P2_HHI", "3"))
# ⚠️ **AGV 车速 [m/s]**（2026-10-17，用户指定 1.0）：`SimConfig.agv_speed_mps`，默认 0.5。
SPEED = float(os.environ.get("P2_SPEED", "0.5"))


def build_spec(inst, shop_len):
    return build(seed=0, pm_interval=30, target_failures=4, n_agv=NAGV,
                 finite_buffer=True, layout_seed=0, reward_norm="none",
                 tau=2.70, sim_overrides={"due_range": 0.6, "la_tie_random": 1.0,
                                          # ⚠️ **2026-10-17 修**：这里此前写死
                                          #    `agv_mttr_lo=10, agv_mttr_hi=10`，
                                          #    把原文的**修复 5–10 min** 覆盖掉了。
                                          #    已删 —— `shi2025=True` 会给原文口径。
                                          "agv_plan_pos": 1.0, "agv_act_book": 1.0,
                                          # ⚠️ **同机台相邻工序省略搬运**（2026-10-17，照原文）。
                                          "skip_same_machine_move": 1.0,
                                          "agv_handling_lo": HLO, "agv_handling_hi": HHI,
                                          "agv_speed_mps": SPEED},
                 inst_obj=inst, layout_scheme="random", **dims_from_shop_len(shop_len, inst.n_machines, layout_cols=5),
                 # ⚠️ **5×5，25 格里随机选 18**（2026-10-17 用户指定）。
                 layout_cols=5, layout_rows=5,
                 # ⚠️ **故障开**（2026-10-17 修）：原文每实例 **1–8 次**、修复 **5–10 min**
                 #    （`q1_flat.build(shi2025=True)` 的默认口径）。
                 #    ⚠️ 此前本脚本写死 `agv_failure=False`（照 `v3_greedy_la.py` 的模板），
                 #    **违反原文规格** —— 已改回。
                 shi2025=True)[0]


def main() -> None:
    print(f"**pt {PT_LO}–{PT_HI} × 5×5 × 车间长度 · {NAGV} 台 AGV** · Exp(16/140) · "
          f"**{N_INST} 实例均值**\n"
          f"  ⚠️ **pt {PT_LO}–{PT_HI}**（原文 2–10）—— 敏感度测试，不是复现。\n"
          f"  布局：**5×5（25 格）× 随机选 18** · 其余配置照原文\n"
          f"  原文：MS 最好 **610** · 它的 TC **420**（即 TC 最好）\n", flush=True)
    print(f"  {'车间[m]':>8}{'MS最好':>9}{'它的TC':>9}{'TC最好':>9}{'它的MS':>9}"
          f"{'MS付':>7}{'TC付':>7}{'前沿':>7}")
    print("  " + "-" * 66)
    for sl in LENS:
        a_ms, a_tc, pm, pt, fr = [], [], [], [], []
        for k in range(N_INST):
            inst = eval_instance(140, 16, EIDX, idx=EIDX + k, alts_dist="uniform",
                                 alts_max=ALTS_MAX, alts_min=ALTS_MIN,
                                 pt_dist="uniform", pt_hi=PT_HI, pt_lo=PT_LO)
            spec = build_spec(inst, sl)
            ms, tc = [], []
            for kk in range(9):
                for j in range(len(LA_RULE_NAMES)):
                    w = spec.world()
                    m = w.run_gated(seed_chain=100000,
                                    policy_pa=lambda v, p, tr=None, _k=kk:
                                        apply_pa_rule(_k, v, p, tr),
                                    policy_l=la_rule_policy(j), dt_events=0)
                    ms.append(float(m["makespan"]))
                    tc.append(float(m["logistics_cost"]))
            ms, tc = np.asarray(ms), np.asarray(tc)
            i_ms, i_tc = int(np.argmin(ms)), int(np.argmin(tc))
            # ⚠️ **四个原始数**（用户指定）：MS 最好 · 它的 TC · TC 最好 · 它的 MS
            a_ms.append((ms[i_ms], tc[i_ms]))
            a_tc.append((ms[i_tc], tc[i_tc]))
            pm.append(ms[i_tc] / ms[i_ms])
            pt.append(tc[i_ms] / tc[i_tc])
            front = [i for i in range(len(ms))
                     if not any((ms[q] <= ms[i] and tc[q] <= tc[i]) and
                                (ms[q] < ms[i] or tc[q] < tc[i]) for q in range(len(ms)))]
            fr.append(len(front))
        b_ms = np.asarray(a_ms)
        b_tc = np.asarray(a_tc)
        print(f"  {sl:>8.0f}{b_ms[:, 0].mean():>9.1f}{b_ms[:, 1].mean():>9.1f}"
              f"{b_tc[:, 1].mean():>9.1f}{b_tc[:, 0].mean():>9.1f}"
              f"{np.mean(pm):>7.2f}{np.mean(pt):>7.2f}{np.mean(fr):>7.1f}", flush=True)
    print("\n  ⚠️ 前四列是**四个原始数**（用户指定）；后两列是它们的比值。")
    print("  ⚠️ 原文：MS 最好 **610** · 它的 TC **420** · TC 最好 **420** · 它的 MS **610**")
    print("     （原文只给了「最好 CDR」一个点 = (610, 420)，与 RN 的 (428, 600)）")


if __name__ == "__main__":
    main()
