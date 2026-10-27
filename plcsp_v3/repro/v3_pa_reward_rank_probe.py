# -*- coding: utf-8 -*-
"""**PA 训练奖励是否把 EAJ+SMPT 排在最好** —— 逐规模、逐规则算 `R_MS`。

对每个规模、每条固定 PA 规则（LA=MINCOST）：

```
final MS                 真实目标
R_MS = Σ γ^t · r_ms      训练用的 Eq.50–53（MS 支）
```

若 `R_MS` 最好的规则 ≠ `final MS` 最好的规则 ⟹ **奖励口径把 PA 带偏**。
若 `R_MS` 也把 EAJ+SMPT 排最好、但策略不选 ⟹ **探索/优化问题**（logits 太平等）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import v3_nhdrl as N  # noqa: E402
from plcsp_v3.env.shi_instances import (POOL_STRIDE, PT_REDRAW_SEED,  # noqa: E402
                                        make_shi_instance)
from plcsp_v3.hrl.rules import (N_AEI2025_RULES, PA_RULE_NAMES,  # noqa: E402
                                apply_pa_rule, la_rule_policy)
from plcsp_v3.q1_flat import build, dims_from_shop_len  # noqa: E402

SIZES = [int(x) for x in os.environ.get(
    "PRR_SIZES", "10,16,22,28,34,40").split(",")]
N_RUN = int(os.environ.get("PRR_RUN", "5"))
GAMMA = 0.99


def build_spec(inst):
    kw = dict(seed=0, pm_interval=30, target_failures=4, n_agv=N.NAGV,
              finite_buffer=True, layout_seed=0, reward_norm="none",
              tau=2.70, sim_overrides=N._sim_over(), inst_obj=inst,
              layout_scheme="random",
              **dims_from_shop_len(200.0, inst.n_machines,
                                   layout_cols=N.LAYOUT_COLS),
              layout_rows=N.LAYOUT_ROWS, shi2025=N.SHI)
    if N.CELL > 0.0:
        kw.update(cell_w=N.CELL, cell_h=N.CELL, aisle_w=N.AISLE,
                  lu_connector_m=N.LU)
    return build(**kw)[0]


class Cap:
    def __init__(self, k: int) -> None:
        self.k = k
        self.rew = N.RewardFn()
        self.prev = None
        self.R = 0.0
        self.t = 0

    def __call__(self, view, pool, tie_rng=None):
        cm, tc = N.cm_of(view.snap), N.tc_cum(view.snap)
        if self.prev is not None:
            r_ms, _ = self.rew.both(cm - self.prev[0], tc - self.prev[1],
                                    (5.0, 5.0))
            self.R += (GAMMA ** self.t) * r_ms
            self.t += 1
        self.prev = (cm, tc)
        return apply_pa_rule(self.k, view, pool, tie_rng)

    def finish(self, makespan: float, snap) -> None:
        tc = N.tc_cum(snap)
        r_ms, _ = self.rew.both(makespan - self.prev[0],
                                tc - self.prev[1], (5.0, 5.0))
        self.R += (GAMMA ** self.t) * r_ms


def main() -> None:
    print("**PA 训练奖励 vs 最终 MS** · 逐规模 · OA=MS · LA=MINCOST\n", flush=True)
    print(f"  {'工件':>4}{'MS最好':>16}{'R最好':>16}"
          f"{'rank corr':>11}{'R(EAJ+SMPT)名次':>16}")
    for size in SIZES:
        inst = make_shi_instance(
            size, seed=POOL_STRIDE * (2000 + size),
            rot_seed=PT_REDRAW_SEED + 2000 + size,
            alts_dist="uniform", alts_min=N.ALTS_MIN, alts_max=N.ALTS_MAX,
            pt_dist="uniform", pt_lo=2, pt_hi=10)
        spec = build_spec(inst)
        scs = [100_000 + 13 * i for i in range(N_RUN)]
        ms, rr = [], []
        for k in range(N_AEI2025_RULES):
            mk, rk = [], []
            for sc in scs:
                cap = Cap(k)
                w = spec.world()
                m = w.run_gated(seed_chain=sc, policy_pa=cap,
                                policy_l=la_rule_policy(0), dt_events=0)
                cap.finish(float(m["makespan"]), w.snapshot())
                mk.append(float(m["makespan"]))
                rk.append(float(cap.R))
            ms.append(float(np.mean(mk)))
            rr.append(float(np.mean(rk)))
        ms_a, rr_a = np.asarray(ms), np.asarray(rr)
        k_ms = int(np.argmin(ms_a))
        k_r = int(np.argmax(rr_a))
        # Spearman：R 越大越好，MS 越小越好
        rank_ms = np.argsort(np.argsort(ms_a))
        rank_r = np.argsort(np.argsort(-rr_a))
        rho = float(np.corrcoef(rank_ms, rank_r)[0, 1])
        j = PA_RULE_NAMES.index("EAJ+SMPT")
        print(f"  {size:>4}{PA_RULE_NAMES[k_ms]:>16}"
              f"{PA_RULE_NAMES[k_r]:>16}{rho:>11.3f}"
              f"{int(rank_r[j]) + 1:>16}", flush=True)
        print("      MS: " + " ".join(f"{x:.1f}" for x in ms_a), flush=True)
        print("      R : " + " ".join(f"{x:.2f}" for x in rr_a), flush=True)
    print("\n  ⚠️ R = Σγ^t·r_ms；MS 越小越好，R 越大越好。")
    print("  ⚠️ 名次 1 = 最好。")


if __name__ == "__main__":
    main()
