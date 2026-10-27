# -*- coding: utf-8 -*-
"""**分布族扫描**：候选机台数 × 加工时间 = 2 × 2（2026-10-16）。

══ 用户提出 ══

> *"工序能放几个机台改正态分布吧。加工时间你是什么分布？是不是也应该正态分布"*

```
现状：候选机台数 k ~ **幂律** p(k) ∝ k²  ⟹ 往 10 偏（均值 7.94）
      加工时间   ~ **均匀整数 [2, 10]**（原文 "在 2–10 之间随机" 的自然读法）

⚠️ 两者的**分布族原文都没给** —— 原文 §5.1 逐字只说：
   *"the available machines for each operation is determined based on the
     corresponding **component type**"* / *"processing time … randomly determined
     between 2 and 10"*
   （⚠️ 本仓此前引过 *"Each machine group is fixed"* —— **全文检索零命中，原文没有这句**。）
```

**⚠️ 先验**：候选机台数已经扫过 6 档（均值 1.04 → 7.94），方向一次没翻；
**正态落在那个区间内** ⟹ 先验低。**但 pt 的方差是真新轴**（两族均值都是 6，只改方差）。

**判据**：与 `v3_alts_sweep.py` 同 —— 用有原文锚的那一条：

```
最后一列 = 最好 CDR 的 MS ÷ RN 的 MS
  < 1 ⟹ 规则赢 RN（= 现状）    > 1 ⟹ **RN 赢规则**（= 原文的方向，那里是 610/428 = 1.43）
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
from plcsp_v3.q1_flat import SHI_SHOP_LEN, build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

EIDX = int(os.environ.get("DS_EIDX", str(EVAL_IDX)))
SHOPLEN = float(os.environ.get("DS_SHOPLEN", str(SHI_SHOP_LEN)))
ALTS_POW = float(os.environ.get("DS_ALTSPOW", "-2"))
NOFAIL = os.environ.get("DS_NOFAIL", "1") == "1"
N_RUN = int(os.environ.get("DS_RN", "10"))
SC_FIX = int(os.environ.get("DS_SC0", "100000"))
# ⚠️ 换实例时，新旧实例的 `pt` 抽样流不同；跑多个实例看结论稳不稳。
N_INST = int(os.environ.get("DS_EVALN", "5"))


def build_spec(inst):
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
                 layout_scheme="random", **dims_from_shop_len(SHOPLEN, inst.n_machines), **kw)[0]


def main() -> None:
    print(f"**分布族扫描** · Exp(16/140) · 车间 {SHOPLEN:.0f} m · "
          f"故障={'关' if NOFAIL else '开'} · 每个格子 {N_INST} 个实例的均值\n", flush=True)
    print(f"  {'候选分布':>10}{'pt 分布':>10}{'候选均值':>10}{'pt 均值':>9}{'pt 极差':>9}"
          f"{'MS 最好':>10}{'±':>8}{'TC 最好':>10}{'corr':>9}"
          f"{'CDR-MS/RN':>11}{'±':>8}")
    print("  " + "-" * 106)
    for ad in ("uniform", "power", "normal"):
        for pd_ in ("uniform", "normal"):
            ratios, corrs, nk, pv, msmin, tcmin = [], [], [], [], [], []
            for k_inst in range(N_INST):
                # ⚠️ `n_inst` 只用于越界检查（约束是 `n_inst ≤ idx`）：给 `EIDX`，
                #    于是 `idx = EIDX, EIDX+1, …` 全部合法（都 ≥ `n_inst` ⟹ 不在训练池里）。
                inst = eval_instance(140, 16, EIDX, idx=EIDX + k_inst,
                                     alts_pow=ALTS_POW, alts_dist=ad, pt_dist=pd_)
                nk.append(np.mean([len(op) for job in inst.jobs for op in job]))
                pv.extend(t for job in inst.jobs for op in job for _m, t in op)
                spec = build_spec(inst)
                ms_i, tc_i = [], []
                for k in range(N_AEI2025_RULES):
                    for j in range(len(LA_RULE_NAMES)):
                        m_, t_ = H.run_cdr(spec, k, j)
                        ms_i.append(m_)
                        tc_i.append(t_)
                rn_ms = float(np.mean([H.run_rn(spec, i)[0] for i in range(N_RUN)]))
                ms_i, tc_i = np.asarray(ms_i), np.asarray(tc_i)
                # ⚠️ **比值按实例算，再平均** —— 把 5 个实例的解**汇总**再取 min/mean
                #    是**混口径**（不同实例的 MS 量级不同），不得那样做。
                ratios.append(ms_i.min() / rn_ms)
                corrs.append(float(np.corrcoef(ms_i, tc_i)[0, 1]))
                msmin.append(ms_i.min())
                tcmin.append(tc_i.min())
            r = np.asarray(ratios)
            print(f"  {ad:>10}{pd_:>10}{np.mean(nk):>10.2f}"
                  f"{np.asarray(pv).mean():>9.2f}{np.asarray(pv).max() - np.asarray(pv).min():>9.0f}"
                  f"{np.mean(msmin):>10.1f}{np.std(msmin):>9.1f}"
                  f"{np.mean(tcmin):>10.1f}{np.mean(corrs):>+9.3f}"
                  f"{np.mean(r):>9.2f}{np.std(r):>8.2f}", flush=True)
    print("\n  ⚠️ 判据 = 最后一列：**> 1 ⟹ RN 赢规则（原文的方向，610/428 = 1.43）**；< 1 ⟹ 方向反着。")
    print("  ⚠️ `候选均值` 是每道工序能用几台机；`pt 极差` 反映分布的方差。")
    print("  ⚠️ 两族 pt 的**均值都是 6** ⟹ 机台总工时不变，只改 `SMPT` 能挑出多少差异。")


if __name__ == "__main__":
    main()
