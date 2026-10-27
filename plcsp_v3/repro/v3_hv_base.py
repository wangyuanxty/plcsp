# -*- coding: utf-8 -*-
"""**HV 的归一化基准里没有「策略」** —— 这一条能不能解释「CDR 的 HV 异常地高」。

══ 起因 ══

本仓一直报 CDR 最好 HV ≈ 0.87–0.99（原文 0.264）。查过很多轮，结论都是「基准不是主因」
（§4.26 → §4.27 → §4.34）。**但那些轮次全都建立在同一个错误前提上：我们不知道"策略能到哪"。**

- 原归一说辞：「基准 = 该实例上**全部方法的所有解**」—— 而实际传进去的只有 `RN + 27 CDR`，
  **策略没在里面**（还没训出来）。
- 没有上限锚 ⟹ **CDR 自己就成了 MS 轴的下界** ⟹ 总有一条解落在 `x = 0` 或 `y = 0` ⟹ HV 天然接近 1。

**§4.35 量出「同口径能做到的是 163.3 min」，第一次有了上限锚。**

══ 本脚本做什么 ══

跑同一个实例的 `RN + 27 CDR`（与 `v3_rn_cdr.py` 同口径），然后**在基准里加一个代理点**
（= 策略将来大概能到的位置），看 HV 怎么走：

| 基准 | 说明 |
|---|---|
| A = `RN ∪ 27 CDR` | **本仓一直用的口径**（= 现状） |
| B = A ∪ {(MS, TC)} 扫参 | 把「能做到的点」加进去 |

⚠️ **代理点不是策略的读数** —— 我们还没有训出来的策略。
它只是**一个位置**，用来量「基准里有没有上限锚」这件事值多少 HV。**不得当结论并列。**

用法：`python plcsp_v3/repro/v3_hv_base.py`
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import v3_hv3 as H  # noqa: E402
from plcsp_v3.env.constraints import ConstraintConfig  # noqa: E402
from plcsp_v3.env.shi_instances import EVAL_IDX, N_AGV_SHI, eval_instance  # noqa: E402
from plcsp_v3.hrl.mo_metrics import (REF_NORM, hypervolume_2d,  # noqa: E402
                                     normalize_minmax, pareto_front)
from plcsp_v3.hrl.rules import LA_RULE_NAMES, N_AEI2025_RULES  # noqa: E402
from plcsp_v3.q1_flat import SHI_SHOP_LEN, build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

EIDX = int(os.environ.get("HB_EIDX", str(EVAL_IDX)))
N_RUN = int(os.environ.get("HB_RUN", "20"))
SHOPLEN = float(os.environ.get("HB_SHOPLEN", str(SHI_SHOP_LEN)))
# ⚠️ 默认与 `v3_greedy_la.py` 一致：**AGV 故障关**（那个 163.3 出自确定性模型）。
NOFAIL = os.environ.get("HB_NOFAIL", "1") == "1"
# 构造式可行解（`exact_cpsat.greedy_schedule`，§4.35）：MS 163.3 · TC = 42 + 0.5×578.6
PROXY = (float(os.environ.get("HB_MS", "163.3")), float(os.environ.get("HB_TC", "331.3")))


def build_spec(inst):
    """与 `v3_rn_cdr.py` / `v3_greedy_la.py` 逐字同口径。"""
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
    inst = eval_instance(140, 16, 20, idx=EIDX)
    spec = build_spec(inst)
    print(f"实例 eval idx={EIDX} · 车间 {SHOPLEN:.0f} m · "
          f"**AGV 故障 = {'关' if NOFAIL else '开'}**\n")
    rn = np.asarray([H.run_rn(spec, i) for i in range(N_RUN)])
    cdr = np.asarray([H.run_cdr(spec, k, j)
                      for k in range(N_AEI2025_RULES)
                      for j in range(len(LA_RULE_NAMES))])
    print(f"实例 eval idx={EIDX} · RN {N_RUN} 局 · CDR {len(cdr)} 条\n")
    print(f"  RN   MS {rn[:, 0].mean():.1f} ± {rn[:, 0].std():.1f}"
          f"　TC {rn[:, 1].mean():.1f} ± {rn[:, 1].std():.1f}")
    print(f"  CDR  MS [{cdr[:, 0].min():.1f}, {cdr[:, 0].max():.1f}]"
          f"　TC [{cdr[:, 1].min():.1f}, {cdr[:, 1].max():.1f}]\n")

    def hv_all(extra: list[np.ndarray]) -> tuple[float, float, tuple]:
        normed, (lo, hi) = normalize_minmax([rn[None, :][0], cdr, *extra])
        rn_n, cdr_n = normed[0], normed[1]
        return (float(hypervolume_2d(pareto_front(rn_n), REF_NORM)),
                float(hypervolume_2d(pareto_front(cdr_n), REF_NORM)),
                (lo, hi))

    hv_rn0, hv_cdr0, (lo0, hi0) = hv_all([])
    print("  ══ 基准 A = `RN ∪ 27 CDR`（本仓一直用的口径）══")
    print(f"    MS [{lo0[0]:.1f}, {hi0[0]:.1f}]　TC [{lo0[1]:.1f}, {hi0[1]:.1f}]")
    print(f"    RN HV **{hv_rn0:.4f}**　CDR HV **{hv_cdr0:.4f}**"
          f"　RN/CDR **{hv_rn0 / hv_cdr0:.2f}×**")

    print("\n  ══ 基准 B = A ∪ {代理点}（把「能做到的」也当解）══")
    print(f"    {'代理点 (MS, TC)':<22}{'RN HV':>9}{'CDR HV':>9}{'RN/CDR':>9}"
          f"{'基准 MS 下界':>13}")
    sweep = [(302.0, 442.0), (250.0, 400.0), (200.0, 360.0), PROXY, (140.0, 300.0),
             (100.0, 250.0), (41.0, 200.0)]
    for pt in sweep:
        hv_rn, hv_cdr, (lo, hi) = hv_all([np.asarray(pt)[None, :]])
        tag = " ← 构造式可行解" if pt == PROXY else ("（≈ 现状最好 CDR）" if pt[0] > 300
                                                    else "")
        print(f"    ({pt[0]:>6.1f}, {pt[1]:>6.1f})     {hv_rn:>9.4f}{hv_cdr:>9.4f}"
              f"{hv_rn / hv_cdr:>9.2f}{lo[0]:>13.1f}{tag}")
    print("\n  ⚠️ 原文 `Exp(16/140)`：NHDRL 0.8225 · **RN 0.3378 · CDRs 0.2642**"
          "（**RN/CDR = 1.28**）")
    print("  ⚠️ 代理点**不是**策略读数，只是一个**位置** —— 用来量「基准里有没有上限锚」。")
    print("  ⚠️ 原文**从没说过**基准按谁；本仓此前取「RN ∪ CDR」是**漏了策略**，不是原文教的。")


if __name__ == "__main__":
    main()
