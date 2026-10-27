# -*- coding: utf-8 -*-
"""**打破通吃的方向：让「均衡」变成运输的代价**（2026-10-17，用户提出的方向）。

══ 为什么是这个方向 ══

用户指出：**换参数会换「哪条规则」通吃，但换不掉「总有一条通吃」。**
故结论不能落在某条规则上，必须落在**集合的性质**上。

```
本仓实测（`v3_blind_layers.py`）：
  有账本   MS 307.0 · TC 416.0（**同一条规则两维都最好**）
  盲 EAJ   MS 306.3 · TC 490.7      ← 工期 −0.2%，运输 **+18%**
  盲 MINU  MS 304.0 · TC 468.0      ← 工期 −1.0%，运输 **+13%**

⟹ **账本不改变「工期能做多短」，它改变「做这个工期要花多少运输」。**
⟹ 也就是说：**运输是「机台均衡」的副产品，不是它的代价。**
```

**这一条是结构性的：**
```
`pt` 逐 (机台, 工序) 独立随机 ⟹ 「快」与「位置」无关
⟹ `MINU` 均衡负载时挑到的机台，**平均距离不增不减**
⟹ **均衡不要运输成本** ⟹ MS 与 TC 同向 ⟹ 总有一条规则同时最好
```

**⟹ 本脚本把两者绑起来：`pt_pos_corr < 0` ⟹ **远处的机台更快**。**
```
`MINU` 的分子（已派工时）涨得慢 ⟹ 比率低 ⟹ 它**持续往远处送活**
⟹ **均衡必须付运输** ⟹ 两目标反向 ⟹ 没有规则能同时最好
```

⚠️ **原文没有这一条**（Fig. 4 只给拓扑、没尺寸；`pt` 只说 2–10 随机）。
   它是**我方推定的补充**，目的是检验「通吃」是不是由「运输与均衡解耦」造成的。

══ 判据 ══

```
`MS付`/`TC付`：前沿两个角（要好工期，运输付几倍）—— 原文 1.43
`通吃?`：MS 最好与 TC 最好是不是同一条（**是 = 没打破**）
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

EIDX = int(os.environ.get("PC_EIDX", str(EVAL_IDX)))
SHOPLEN = float(os.environ.get("PC_SHOPLEN", "240"))
ALTS_MAX = int(os.environ.get("PC_ALTSMAX", "10"))
CORRS = [float(x) for x in os.environ.get("PC_CORRS", "0,-0.5,-1.0,-1.5").split(",")]
N_RUN = int(os.environ.get("PC_RN", "10"))
# ⚠️ **多实例**（2026-10-17 加）：单实例的比值噪声很大，用户要看能不能站住。
N_INST = int(os.environ.get("PC_NINST", "1"))


def build_spec(inst, corr: float):
    ovr = {"due_range": 0.6, "la_tie_random": 1.0,
           "agv_mttr_lo": 10.0, "agv_mttr_hi": 10.0,
           "agv_plan_pos": 1.0, "agv_act_book": 1.0}
    return build(seed=0, pm_interval=30, target_failures=4, n_agv=6,
                 finite_buffer=True, layout_seed=0, reward_norm="minmax",
                 tau=2.70, sim_overrides=ovr, inst_obj=inst,
                 layout_scheme="random", **dims_from_shop_len(SHOPLEN, inst.n_machines),
                 pt_pos_corr=corr, shi2025=False,
                 constraints=ConstraintConfig(finite_buffer=False, machine_failure=False,
                                              maintenance=False, due_dates=False,
                                              agv_failure=False))[0]


def main() -> None:
    print(f"**pt × 位置相关性** · Exp(16/140) · 车间 {SHOPLEN:.0f} m · "
          f"alts_max={ALTS_MAX} · 6 车 · **{N_INST} 个实例的均值 ± 标准差**\n"
          "  corr < 0 = **远处的机台更快** ⟹ MINU 会往远处送活 ⟹ 均衡要付运输\n", flush=True)
    print(f"  {'corr':>7}{'MS付':>9}{'±':>7}{'TC付':>9}{'±':>7}{'前沿':>8}{'MS比':>9}")
    print("  " + "-" * 60)
    for corr in CORRS:
        pm, pt, fr, mr = [], [], [], []
        for k_inst in range(N_INST):
            inst = eval_instance(140, 16, EIDX, idx=EIDX + k_inst,
                                 alts_dist="uniform", alts_max=ALTS_MAX,
                                 pt_dist="uniform")
            spec = build_spec(inst, corr)
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
            mr.append(ms.min() / rn_ms)
        print(f"  {corr:>7.2f}{np.mean(pm):>9.2f}{np.std(pm):>7.2f}"
              f"{np.mean(pt):>9.2f}{np.std(pt):>7.2f}"
              f"{np.mean(fr):>8.1f}{np.mean(mr):>9.2f}", flush=True)
    print("\n  ⚠️ 原文：`MS付 / TC付 = 1.43 / 1.43`（图 6 的 CDR 与 RN 两点）。")
    print("  ⚠️ `pt_pos_corr` **原文没有** —— 这是我方推定的补充，不是复现口径。")
    print("  ⚠️ 用户 2026-10-17 定下的判据：**不比 `MS比`**（原文自己也是一半一半），")
    print("     只要 **MS 与 TC 有权衡**（`MS付`/`TC付` 明显 > 1）。")


if __name__ == "__main__":
    main()
