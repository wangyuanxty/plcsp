# -*- coding: utf-8 -*-
"""**「为省运输而集中」的极端派工** —— 省 TC 要付多少 MS（2026-10-17，用户提出的机制）。

══ 用户的判断 ══

> *"要让节省 tc 造成的 ms 代价高，需要节省 tc 导致机台负载不均匀，
>   因为加工时间差别并不极端"*

**实测支持这个判断：**
```
TC 的节省**全部**来自派车（空载 163 → 355，2.18×）
机台选择对运输几乎没影响（负载 609 → 644，**1.06×**）
⟹ 「省 TC」现在不需要动机台 ⟹ 也就不需要付 MS
```

**⟹ 而加工时间只差 2–10（5 倍）⟹ 挑近机台**不会**因为加工慢而拖 MS；
拖 MS 的方式是**把活堆在那几台上** ⟹ 排队。**

══ 本脚本造一个极端派工（**诊断用，不是基线**）══

```
PA：每道工序，在候选机台里选**离件当前位置最近**的那台（**完全不看加工时间**）
LA：MINCOST
⟹ 「为省 TC 而集中到近机台」的极端形态
```

**它给出那个极端点的 (MS, TC)。与 27 条 CDR 的前沿一比，就知道：**
```
· 若它落在「TC 更低、MS 更高」的那一角 ⟹ **这条两难存在**，策略有空间
· 若它两维都更差 ⟹ 这条机制不成立
```
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from plcsp_v3.env.constraints import ConstraintConfig  # noqa: E402
from plcsp_v3.env.shi_instances import eval_instance  # noqa: E402
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, apply_pa_rule, la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

EIDX = int(os.environ.get("NM_EIDX", "199"))
SHOPLEN = float(os.environ.get("NM_SHOPLEN", "240"))
ALTS_MAX = int(os.environ.get("NM_ALTSMAX", "10"))
N_INST = int(os.environ.get("NM_NINST", "5"))


def build_spec(inst):
    return build(seed=0, pm_interval=30, target_failures=4, n_agv=6,
                 finite_buffer=True, layout_seed=0, reward_norm="minmax",
                 tau=2.70, sim_overrides={"due_range": 0.6, "la_tie_random": 1.0,
                                          "agv_mttr_lo": 10.0, "agv_mttr_hi": 10.0,
                                          "agv_plan_pos": 1.0, "agv_act_book": 1.0},
                 inst_obj=inst, layout_scheme="random", **dims_from_shop_len(SHOPLEN, inst.n_machines),
                 shi2025=True,
                 constraints=ConstraintConfig(finite_buffer=False, machine_failure=False,
                                              maintenance=False, due_dates=False,
                                              agv_failure=False))[0]


def make_nearest_pa(world, lay):
    """极端派工：候选机台里选**离件当前位置最近**的那台（**不看加工时间**）。"""
    lu = int(lay.lu.dock_node)
    dock = [int(p.dock_node) for p in lay.machines]

    def policy(view, pool, tie_rng=None):
        plans = world._pa_plans
        best = None
        for pi, (j, oi) in enumerate(pool):
            jv = view.jobs[j]
            if not jv.alts:
                continue
            org = lu if oi == 0 else dock[int(plans[j][oi - 1])]
            m_best, d_best = -1, None
            for m, _pt in jv.alts:
                d = 0.0 if org == dock[int(m)] else world.leg_min(org, dock[int(m)])
                if d_best is None or d < d_best:
                    m_best, d_best = int(m), d
            # 同一距离下用「最早可开工」打破平局（只影响确定性，不引入新信息）
            key = (d_best, jv.ready_at)
            if best is None or key < best[0]:
                best = (key, pi, m_best)
        assert best is not None
        return best[1], best[2]

    return policy


def run(w, policy_pa, policy_l):
    m = w.run_gated(seed_chain=100000, policy_pa=policy_pa, policy_l=policy_l,
                    dt_events=0)
    return float(m["makespan"]), float(m["logistics_cost"])


def main() -> None:
    print(f"**「最近机台」探针** · Exp(16/140) · 车间 {SHOPLEN:.0f} m · "
          f"**忠实口径** · **逐实例比较**\n")
    print(f"  {'实例':>6}{'CDR_MS最好':>11}{'CDR_TC最好':>11}{'最近机台':>16}"
          f"{'它被CDR支配?':>13}{'比CDR的TC':>10}")
    print("  " + "-" * 72)
    ratios = []
    dominated = 0
    for k in range(N_INST):
        inst = eval_instance(140, 16, EIDX, idx=EIDX + k,
                             alts_dist="uniform", alts_max=ALTS_MAX, pt_dist="uniform")
        spec = build_spec(inst)
        cms, ctc = [], []
        for kk in range(9):
            for j in range(LA_RULE_NAMES.__len__()):
                w = spec.world()
                a, b = run(w, lambda v, p, tr=None, _k=kk: apply_pa_rule(_k, v, p, tr),
                           la_rule_policy(j))
                cms.append(a)
                ctc.append(b)
        w = spec.world()
        nms, ntc = run(w, make_nearest_pa(w, spec.layout), la_rule_policy(0))
        cms, ctc = np.asarray(cms), np.asarray(ctc)
        i_ms, i_tc = int(np.argmin(cms)), int(np.argmin(ctc))
        # 最近机台那个点，是否被 27 条里**任何一条**同时压住？
        dom = bool(((cms <= nms) & (ctc <= ntc)).any())
        dominated += int(dom)
        ratios.append((nms, ntc, cms[i_ms], ctc[i_ms], cms[i_tc], ctc[i_tc]))
        print(f"  {k:>6}{cms[i_ms]:>11.1f}{ctc[i_ms]:>11.1f}"
              f"{f'({nms:.1f}, {ntc:.1f})':>16}{'**是**' if dom else '否':>13}"
              f"{ntc / ctc[i_tc]:>10.2f}", flush=True)
    r = np.asarray(ratios)
    print("  " + "-" * 72)
    print(f"  逐实例：最近机台的 TC ÷ CDR 的 TC 最好 = **均值 {np.mean(r[:, 1] / r[:, 5]):.2f}**"
          f"（{np.min(r[:, 1] / r[:, 5]):.2f}–{np.max(r[:, 1] / r[:, 5]):.2f}）")
    print(f"  被 CDR 支配的实例数：**{dominated}/{N_INST}**")
    print("\n  ⚠️ **必须逐实例比** —— 拿「跨实例最小值」去比「均值」是错的口径。")
    print("  ⚠️ 「最近机台」是**诊断探针**，不是基线、不进 27 条。")


if __name__ == "__main__":
    main()
