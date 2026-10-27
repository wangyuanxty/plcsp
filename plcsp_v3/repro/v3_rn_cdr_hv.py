# -*- coding: utf-8 -*-
"""**RN vs 27 CDR 的 HV** —— 2–4 与 8–16 两档候选，18 个复现实例。

协议（对齐外部复现 `evaluate.py`）：

```
每个实例：RN 跑 N_RUN 次（每次换故障场景 + 换随机动作种子）
          27 条 CDR 各跑 N_RUN 次（每次换故障场景）
归一化：该实例上 RN ∪ 全部 CDR 的 min–max
参考点：(1.1, 1.1)，HV ÷ 1.21
```

输出：

```
RN HV
CDR 并集 HV       （把 27 条的所有点放一起取 Pareto）
最好单条 CDR HV   （27 条各自算，取最大 —— 对 CDR 最有利的口径）
谁更高
```

⚠️ 这只是 **RN vs CDR**，不含训练策略。
⚠️ 故障场景：第 i 次运行用 `seed_chain = SC_FIX + i`（与外部“每组不同”对齐）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from plcsp_v3.env.shi_instances import (POOL_STRIDE, PT_REDRAW_SEED,  # noqa: E402
                                        make_shi_instance)
from plcsp_v3.hrl.mo_metrics import hypervolume_2d, pareto_front  # noqa: E402
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, N_AEI2025_RULES,  # noqa: E402
                                apply_pa_rule, la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402

N_RUN = int(os.environ.get("RCH_RUN", "10"))
N_INST = int(os.environ.get("RCH_NINST", "3"))
JOBS = [int(x) for x in os.environ.get(
    "RCH_JOBS", "10,16,22,28,34,40").split(",")]
SC_FIX = 100_000
REF = (1.1, 1.1)
REF_AREA = REF[0] * REF[1]
ALTS = [(2, 4), (8, 16)]


def build_spec(inst, alts_min: int, alts_max: int):
    return build(
        seed=0, pm_interval=30, target_failures=4, n_agv=6,
        finite_buffer=True, layout_seed=0, reward_norm="none", tau=2.70,
        sim_overrides={
            "due_range": 0.6, "la_tie_random": 1.0,
            "agv_plan_pos": 1.0, "agv_act_book": 1.0,
            "skip_same_machine_move": 1.0,
            "agv_speed_mps": 0.5,
            "agv_handling_lo": 0.0, "agv_handling_hi": 0.0,
        },
        inst_obj=inst, layout_scheme="random",
        layout_cols=5, layout_rows=5,
        cell_w=20.0, cell_h=20.0, aisle_w=15.0, lu_connector_m=30.0,
        shi2025=True)[0]


def _mt(m: dict) -> tuple[float, float]:
    return float(m["makespan"]), float(m["logistics_cost"])


def run_rn(spec, i: int) -> tuple[float, float]:
    rng = np.random.default_rng([SC_FIX, 99, i])

    def pol_pa(view, pool, tie_rng=None):
        return apply_pa_rule(int(rng.integers(0, N_AEI2025_RULES)),
                             view, pool, tie_rng)

    def pol_l(view, cand):
        return int(la_rule_policy(int(rng.integers(0, len(LA_RULE_NAMES))))(
            view, cand))

    m = spec.world().run_gated(seed_chain=SC_FIX + i, policy_pa=pol_pa,
                               policy_l=pol_l, dt_events=0)
    return _mt(m)


def run_cdr(spec, k: int, j: int, i: int) -> tuple[float, float]:
    m = spec.world().run_gated(
        seed_chain=SC_FIX + i,
        policy_pa=lambda v, p, tr=None, _k=k: apply_pa_rule(_k, v, p, tr),
        policy_l=la_rule_policy(j), dt_events=0)
    return _mt(m)


def hv(P: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    Q = (P - lo) / np.maximum(hi - lo, 1e-12)
    return hypervolume_2d(pareto_front(Q), REF) / REF_AREA


def main() -> None:
    for alts_min, alts_max in ALTS:
        print(f"\n══ 候选 {alts_min}–{alts_max} · {N_RUN} 次/方法 ══", flush=True)
        print(f"  {'实例':<12}{'工序':>5}{'RN HV':>9}{'CDR并集':>9}"
              f"{'最好单条':>9}{'谁高':>6}")
        print("  " + "-" * 54)
        wins = 0
        total = 0
        for si, jobs in enumerate(JOBS):
            acc = []
            for k in range(N_INST):
                seed = POOL_STRIDE * (1000 + 10 * si + k)
                rot = PT_REDRAW_SEED + 1000 + 10 * si + k
                inst = make_shi_instance(
                    jobs, seed=seed, rot_seed=rot, alts_dist="uniform",
                    alts_min=alts_min, alts_max=alts_max,
                    pt_dist="uniform", pt_lo=2, pt_hi=10)
                spec = build_spec(inst, alts_min, alts_max)
                n_ops = sum(len(j) for j in inst.jobs)
                rn = np.asarray([run_rn(spec, i) for i in range(N_RUN)])
                cdr = [[np.asarray([run_cdr(spec, pk, lj, i)
                                    for i in range(N_RUN)])
                        for lj in range(len(LA_RULE_NAMES))]
                       for pk in range(N_AEI2025_RULES)]
                allp = np.vstack([rn, *[x for row in cdr for x in row]])
                lo, hi = allp.min(0), allp.max(0)
                rn_hv = hv(rn, lo, hi)
                cdr_all = np.vstack([x for row in cdr for x in row])
                cdr_union = hv(cdr_all, lo, hi)
                cdr_best = max(hv(P, lo, hi) for row in cdr for P in row)
                win = rn_hv > cdr_best
                wins += int(win)
                total += 1
                acc.append((rn_hv, cdr_union, cdr_best))
                print(f"  Exp({jobs}/{n_ops}) #{k + 1:<2}"
                      f"{n_ops:>5}{rn_hv:>9.4f}{cdr_union:>9.4f}"
                      f"{cdr_best:>9.4f}{'RN' if win else 'CDR':>6}", flush=True)
            a = np.asarray(acc, dtype=float).mean(axis=0)
            print(f"  {jobs:>4} 均值{'':>5}{a[0]:>9.4f}{a[1]:>9.4f}"
                  f"{a[2]:>9.4f}{'':>6}", flush=True)
        print(f"\n  RN 胜过最好单条 CDR：**{wins}/{total}**", flush=True)


if __name__ == "__main__":
    main()
