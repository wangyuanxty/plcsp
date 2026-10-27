# -*- coding: utf-8 -*-
"""**平局频率探针**：规则筛出来的候选集合有多大？

对 PA 的 12 条规则、LA 的 3 条规则，逐决策点统计：

```
PA：最高分并列的 (工件, 机台) 对数
LA：最高分并列的 AGV 数
```

用途：判断“规则筛出平局集合、下层打破平局”这个架构有没有东西可学。

- 平局集合恒为 1 ⟹ 下层没有动作；
- 平局集合经常 > 1 ⟹ 下层有空间。

⚠️ RA（保养）是**二元决策**，不是“多台机台并列”⟹ 本探针不统计。
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import v3_nhdrl as N  # noqa: E402
from plcsp_v3.env.shi_instances import eval_instance  # noqa: E402
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, PA_JOB_RULES,  # noqa: E402
                                PA_MACHINE_RULES, PA_RULE_NAMES,
                                _argmin_all, apply_pa_rule, la_rule_policy,
                                la_score, pa_job_score, pa_machine_score)
from plcsp_v3.q1_flat import build  # noqa: E402

N_SC = int(os.environ.get("TIE_SCEN", "5"))
BASE_PA = os.environ.get("TIE_BASE_PA", "EAJ+SMPT,SPT+MINU,SRPT+EAM")
BASES = [PA_RULE_NAMES.index(x) for x in BASE_PA.split(",")]
LAM_J, LAM_M = (0.0, 0.0), (0.0,)


class TieCap:
    def __init__(self, base_pa: int) -> None:
        self.base_pa = base_pa
        self.pa_pair = defaultdict(list)     # rule -> [pair tie size]
        self.pa_job = defaultdict(list)      # rule -> [job tie size]
        self.la = defaultdict(list)          # rule -> [AGV tie size]

    def policy_pa(self, view, pool, tie_rng=None):
        n_job_rules = len(PA_MACHINE_RULES)
        for idx in range(len(PA_RULE_NAMES)):
            jr = PA_JOB_RULES[idx // n_job_rules]
            mr = PA_MACHINE_RULES[idx % n_job_rules]
            js = [pa_job_score(jr, view, j, LAM_J) for j, _oi in pool]
            jt = _argmin_all(js)
            pair = 0
            for pi in jt:
                alts = view.jobs[pool[pi][0]].alts
                ms = [pa_machine_score(mr, view, mt, LAM_M) for mt in alts]
                pair += len(_argmin_all(ms))
            self.pa_job[idx].append(len(jt))
            self.pa_pair[idx].append(pair)
        return apply_pa_rule(self.base_pa, view, pool, tie_rng)

    def policy_l(self, view, cand):
        for j, name in enumerate(LA_RULE_NAMES):
            sc = [la_score(name, view, a, (0.0,)) for a in cand]
            self.la[j].append(len(_argmin_all(sc)))
        return la_rule_policy(0)(view, cand)


def _row(name, vals) -> str:
    a = np.asarray(vals, dtype=float)
    n = len(a)
    return (f"  {name:<14}{n:>6}{np.mean(a > 1):>9.1%}"
            f"{np.mean(a):>9.2f}{np.max(a):>6.0f}")


def main() -> None:
    inst = eval_instance(140, 16, 20, **N._inst_kw())
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N.NAGV,
                    finite_buffer=True, layout_seed=0, reward_norm="none",
                    tau=2.70, sim_overrides=N._sim_over(), inst_obj=inst,
                    layout_scheme="random", layout_cols=N.LAYOUT_COLS,
                    layout_rows=N.LAYOUT_ROWS, shi2025=N.SHI,
                    cell_w=N.CELL or None, cell_h=N.CELL or None,
                    aisle_w=N.AISLE or None, lu_connector_m=N.LU or None)
    scs = [100_000 + 13 * i for i in range(N_SC)]
    print(f"**平局频率** · {N_SC} 场景 · base PA = "
          f"{', '.join(PA_RULE_NAMES[b] for b in BASES)}\n", flush=True)
    print(f"  {'规则':<14}{'决策':>6}{'平局>1':>9}{'均值':>9}{'最大':>6}")
    print("  " + "-" * 44)

    for b in BASES:
        cap = TieCap(b)
        for sc in scs:
            spec.world().run_gated(seed_chain=sc, policy_pa=cap.policy_pa,
                                   policy_l=cap.policy_l, dt_events=0)
        print(f"  base = {PA_RULE_NAMES[b]}", flush=True)
        for idx, name in enumerate(PA_RULE_NAMES):
            print(_row(name, cap.pa_pair[idx]), flush=True)
        print("  ── LA（base 轨迹同一条）──", flush=True)
        for j, name in enumerate(LA_RULE_NAMES):
            print(_row(name, cap.la[j]), flush=True)
        print(flush=True)

    print("  ⚠️ 平局>1 是**该规则最高分并列**的比例；"
          "PA 的均值是并列 (工件,机台) 对数。")
    print("  ⚠️ RA 是二元决策（保/不保），不是多机台并列 ⟹ 不统计。")


if __name__ == "__main__":
    main()
