# -*- coding: utf-8 -*-
"""**候选机台数扫描 —— 每次量 `corr(MS, TC)`**（2026-10-16）。

══ 要回答的问题 ══

用户提出的判据（`open-question-nhdrl.md` §4.40）：

```
原文：随机挑规则（RN）在 **MS 上赢**展示的那条 CDR 43%，在 **TC 上输** 30%
      ⟹ **规则族内部有真权衡**
本仓：27 条 CDR 的 `corr(MS, TC)` = **+0.475** ⟹ **没有权衡**
      ⟹ 最好的那条同时支配其余 26 条 ⟹ 一条静态规则就够 ⟹ **策略没有状态可学**
```

**⟹ 要打的就是：让 `corr(MS, TC)` 从正的翻成负的。**

⚠️ **此前扫 `alts_pow` 量的是「杠杆」（散度），不是「相关性」—— 两个不同的统计量。**
散度小 ≠ 负相关。**本脚本补的正是这一个。**

══ 为什么挑 `alts_pow` ══

它是我方选的、原文没给、且**直接决定规则有多少自由度**的**唯一**一个量：
- **正 ⟹ 候选往 1 偏**（专用工序 ⟹ 机台选择被锁死 ⟹ 能动 TC 的只剩派车）
- **负 ⟹ 候选往 10 偏**（通用工序 ⟹ 机台任选）

⚠️ 现值 `−2.0` 的依据是「对上原文的 43%」—— 而那个 43% 是**方法与方法的差**，
被误当成**族内散度**的靶（§4.40）。**依据不成立。**

⚠️ `alts_pow = 0` ⟹ 走**旧的均匀 2–4 台**那条路（不是幂律的某一档），见 `make_shi_routings`。
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

EIDX = int(os.environ.get("LS_EIDX", str(EVAL_IDX)))
RANGE = os.environ.get("LS_RANGE", "-2,0,1,2,3,4")
SHOPLEN = float(os.environ.get("LS_SHOPLEN", str(SHI_SHOP_LEN)))
NOFAIL = os.environ.get("LS_NOFAIL", "1") == "1"
SC_FIX = int(os.environ.get("LS_SC0", "100000"))


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
    print(f"**候选机台数扫描** · Exp(16/140) · eval idx={EIDX} · 车间 {SHOPLEN:.0f} m · "
          f"故障={'关' if NOFAIL else '开'}\n"
          f"  27 条 CDR 各跑 1 局（**CDR 是确定性的** ⟹ 1 局即够）· "
          f"量的是 `corr(MS, TC)`\n", flush=True)
    print(f"  {'alts_pow':>9}{'候选数均值':>12}{'MS 最好':>10}{'MS 最差':>10}"
          f"{'TC 最好':>10}{'TC 最差':>10}{'corr(MS,TC)':>13}{'散度MS':>9}{'散度TC':>9}")
    print("  " + "-" * 92)
    for pow_ in [float(x) for x in RANGE.split(",")]:
        inst = eval_instance(140, 16, 20, idx=EIDX, alts_pow=pow_)
        n_alts = np.mean([len(op) for job in inst.jobs for op in job])
        spec = build_spec(inst)
        ms, tc = [], []
        for k in range(N_AEI2025_RULES):
            for j in range(len(LA_RULE_NAMES)):
                m_, t_ = H.run_cdr(spec, k, j)
                ms.append(m_)
                tc.append(t_)
        ms, tc = np.asarray(ms), np.asarray(tc)
        corr = float(np.corrcoef(ms, tc)[0, 1])
        lev_ms = (ms.max() - ms.min()) / ms.min()
        lev_tc = (tc.max() - tc.min()) / tc.min()
        flag = "  ← **负相关**" if corr < 0 else ""
        print(f"  {pow_:>9.1f}{n_alts:>12.2f}{ms.min():>10.1f}{ms.max():>10.1f}"
              f"{tc.min():>10.1f}{tc.max():>10.1f}{corr:>+13.3f}"
              f"{lev_ms:>8.0%}{lev_tc:>9.0%}{flag}", flush=True)
    print("\n  ⚠️ 27 个点的 `corr` 噪声 ≈ ±1/√27 ≈ **±0.19** —— 判读要带这个界。")
    print("  ⚠️ 判据是**定性**的：原文 RN 在 MS 上赢 CDR 43%、在 TC 上输 30% ⟹ **负相关**。")
    print("  ⚠️ 散度（杠杆）与相关性**不是一个量** —— 前者小不代表后者为负。")


if __name__ == "__main__":
    main()
