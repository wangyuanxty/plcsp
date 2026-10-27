# -*- coding: utf-8 -*-
"""**奖励排序 vs 最终 MS 排序** —— 解释 PA 为什么不按规模切换。

对 9 条固定 PA 规则（LA=MINCOST），在变规模实例上比较：
```
final MS               真实目标
R_base  = Σ γ^t·r_t    训练用的 Eq.50–53 逐步奖励
R_bonus = R_base + 20·(LB_MS/MS + LB_TC/TC)
```
判据：
- 若 `R_base` 的最优规则 ≠ `final MS` 的最优规则 ⟹ **奖励口径把 PA 带偏**；
- 若 `R_bonus` 把排序拉回 `final MS` ⟹ **终局 bonus 是在修这个**。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import v3_nhdrl as N  # noqa: E402
from plcsp_v3.env.nhdrl_features import NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import (POOL_STRIDE, PT_REDRAW_SEED,  # noqa: E402
                                        make_shi_instance)
from plcsp_v3.hrl.rules import (PA_RULE_NAMES, apply_pa_rule,  # noqa: E402
                                la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

GAMMA = 0.99
N_INST = int(os.environ.get("RA_NINST", "30"))
JOBS_LO = int(os.environ.get("RA_JOBS_LO", "10"))
JOBS_HI = int(os.environ.get("RA_JOBS_HI", "40"))
TERM = float(os.environ.get("RA_TERM", "20"))


def lb_bounds(spec) -> tuple[float, float]:
    inst, lay = spec.inst, spec.layout
    nf = NHFeatures(inst, lay,
                    speed_m_per_min=spec.cfg.agv_speed_mps * 60.0)
    lb_ms = max(sum(min(float(t) for _m, t in op) for op in job)
                for job in inst.jobs)
    lu = int(lay.grid.lu_dock_node)
    s = 0.0
    for job in inst.jobs:
        f = min(nf._leg(lu, lay.machines[int(m)].dock_node)
                for m, _t in job[0])
        last_leg = min(nf._leg(lay.machines[int(m)].dock_node, lu)
                       for m, _t in job[-1])
        s += f + last_leg
    return lb_ms, 7.0 + 0.5 * s


class ConstRule:
    """固定 PA 规则 + 记录 Eq.50–53 的逐步奖励（OA=MS）。"""

    def __init__(self, k: int) -> None:
        self.k, self.rew = k, N.RewardFn()
        self.prev: tuple[float, float] | None = None
        self.acc, self.t = 0.0, 0

    def __call__(self, view, pool, tie_rng=None):
        cm, tc = N.cm_of(view.snap), N.tc_cum(view.snap)
        if self.prev is not None:
            r = self.rew(cm - self.prev[0], tc - self.prev[1], True,
                         (5.0, 5.0))
            self.acc += (GAMMA ** self.t) * r
            self.t += 1
        self.prev = (cm, tc)
        return apply_pa_rule(self.k, view, pool, tie_rng)

    def finish(self, makespan: float, snap) -> None:
        tc = N.tc_cum(snap)
        r = self.rew(makespan - self.prev[0], tc - self.prev[1], True,
                     (5.0, 5.0))
        self.acc += (GAMMA ** self.t) * r


def main() -> None:
    rng = np.random.default_rng([20261018, 11])
    rows = []              # (rule, mean MS, mean R_base, mean R_bonus)
    for k in range(9):
        ms_l, rb_l, rbon_l = [], [], []
        for i in range(N_INST):
            jobs = int(rng.integers(JOBS_LO, JOBS_HI + 1))
            inst = make_shi_instance(
                jobs, seed=POOL_STRIDE * (7000 + i),
                rot_seed=PT_REDRAW_SEED + 7000 + i,
                alts_dist="uniform", alts_min=N.ALTS_MIN,
                alts_max=N.ALTS_MAX, pt_dist="uniform", pt_lo=2, pt_hi=10)
            spec = build(seed=0, pm_interval=30, target_failures=4,
                         n_agv=N.NAGV, finite_buffer=True, layout_seed=0,
                         reward_norm="none", tau=2.70,
                         sim_overrides=N._sim_over(), inst_obj=inst,
                         layout_scheme="random", **dims_from_shop_len(200.0, inst.n_machines, layout_cols=N.LAYOUT_COLS),
                         layout_cols=N.LAYOUT_COLS,
                         layout_rows=N.LAYOUT_ROWS, shi2025=N.SHI)[0]
            pol = ConstRule(k)
            w = spec.world()
            m = w.run_gated(seed_chain=100_000, policy_pa=pol,
                            policy_l=la_rule_policy(0), dt_events=0)
            ms = float(m["makespan"])
            pol.finish(ms, w.snapshot())
            lb_ms, lb_tc = lb_bounds(spec)
            bonus = TERM * (lb_ms / max(ms, 1e-9)
                            + lb_tc / (N.tc_cum(w.snapshot())
                                       + 7.0 * N.NAGV))
            ms_l.append(ms)
            rb_l.append(pol.acc)
            rbon_l.append(pol.acc + bonus)
        rows.append((PA_RULE_NAMES[k], float(np.mean(ms_l)),
                     float(np.mean(rb_l)), float(np.mean(rbon_l))))
        print(f"  {PA_RULE_NAMES[k]:<12} MS {np.mean(ms_l):>7.1f}"
              f"  R_base {np.mean(rb_l):>8.2f}"
              f"  R_bonus {np.mean(rbon_l):>8.2f}", flush=True)

    ms = np.array([r[1] for r in rows])
    rb = np.array([r[2] for r in rows])
    rbon = np.array([r[3] for r in rows])
    print(f"\n**{N_INST} 实例 · 规模 {JOBS_LO}–{JOBS_HI}**\n")
    print(f"  final MS 最好：**{rows[int(np.argmin(ms))][0]}**"
          f"（{ms.min():.1f}）")
    print(f"  R_base  最好：**{rows[int(np.argmax(rb))][0]}**"
          f"（{rb.max():.2f}）")
    print(f"  R_bonus 最好：**{rows[int(np.argmax(rbon))][0]}**"
          f"（{rbon.max():.2f}）")
    # Spearman 秩相关
    def spearman(a, b):
        ra = np.argsort(np.argsort(a))
        rbx = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rbx)[0, 1])
    print(f"\n  rank corr(final MS, R_base)  = {spearman(ms, -rb):+.3f}")
    print(f"  rank corr(final MS, R_bonus) = {spearman(ms, -rbon):+.3f}")
    print("\n  ⚠️ 1 = 排序完全一致；0 = 无关；负 = 反向。")


if __name__ == "__main__":
    main()
