# -*- coding: utf-8 -*-
"""**top-K 覆盖率探针**：规则的前 K 个候选里，有没有最终 MS 最好的那个？

对 PA / LA 的规则：

1. 用 base 规则跑一条链，记录每个决策点的 top-`K_max` 候选；
2. 在抽样的决策点 t，逐个候选做**单点偏离**：
   - 前缀照 base；
   - 第 t 点强制成该候选；
   - 之后继续照 base；
   - 记录最终 MS。
3. 对每个 K，统计“最终 MS 最好的候选落在 top-K 内”的比例。

⚠️ “最好”只在 top-`K_max` 内选，不是全部合法动作。
⚠️ 单点偏离，不是全局最优。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import v3_nhdrl as N  # noqa: E402
from plcsp_v3.env.shi_instances import eval_instance  # noqa: E402
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, PA_JOB_RULES,  # noqa: E402
                                PA_MACHINE_RULES, PA_RULE_NAMES,
                                apply_pa_rule, la_rule_policy, la_score,
                                pa_job_score, pa_machine_score)
from plcsp_v3.q1_flat import build  # noqa: E402

N_SC = int(os.environ.get("TOPK_SCEN", "5"))
N_T = int(os.environ.get("TOPK_T", "8"))
K_PA = int(os.environ.get("TOPK_KPA", "16"))
K_LA = int(os.environ.get("TOPK_KLA", "6"))
BASE_PA = PA_RULE_NAMES.index(os.environ.get("TOPK_PA", "EAJ+SMPT"))
BASE_LA = LA_RULE_NAMES.index(os.environ.get("TOPK_LA", "MINCOST"))
LAM_J, LAM_M = (0.0, 0.0), (0.0,)


def topk_pa(view, pool, rule: int, k: int):
    """按规则（先工件、后机台）的字典序排前 k 个 `(j, oi, m)`。"""
    n_mr = len(PA_MACHINE_RULES)
    jr = PA_JOB_RULES[rule // n_mr]
    mr = PA_MACHINE_RULES[rule % n_mr]
    rows = []
    for pi, (j, oi) in enumerate(pool):
        js = pa_job_score(jr, view, j, LAM_J)
        for mi, (m, _t) in enumerate(view.jobs[j].alts):
            ms = pa_machine_score(mr, view, (m, _t), LAM_M)
            rows.append((js, ms, pi, mi, int(m), int(oi), int(j)))
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    return [(r[6], r[5], r[4]) for r in rows[:k]]      # (j, oi, m)


def topk_la(view, cand, rule: int, k: int):
    rows = []
    for a in cand:
        sc = la_score(LA_RULE_NAMES[rule], view, a, (0.0,))
        rows.append((sc, int(a)))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [a for _s, a in rows[:k]]


class Capture:
    def __init__(self) -> None:
        self.pa, self.la = [], []

    def policy_pa(self, view, pool, tie_rng=None):
        self.pa.append(topk_pa(view, pool, BASE_PA, K_PA))
        return apply_pa_rule(BASE_PA, view, pool, tie_rng)

    def policy_l(self, view, cand):
        self.la.append(topk_la(view, cand, BASE_LA, K_LA))
        return int(la_rule_policy(BASE_LA)(view, cand))


class OPA:
    def __init__(self, t: int, j: int, oi: int, m: int) -> None:
        self.t, self.j, self.oi, self.m, self.n = t, j, oi, m, 0

    def policy_pa(self, view, pool, tie_rng=None):
        i = self.n
        self.n += 1
        if i == self.t:
            return list(pool).index((self.j, self.oi)), self.m
        return apply_pa_rule(BASE_PA, view, pool, tie_rng)

    def policy_l(self, view, cand):
        return int(la_rule_policy(BASE_LA)(view, cand))


class OLA:
    def __init__(self, t: int, a: int) -> None:
        self.t, self.a, self.n = t, a, 0

    def policy_pa(self, view, pool, tie_rng=None):
        return apply_pa_rule(BASE_PA, view, pool, tie_rng)

    def policy_l(self, view, cand):
        i = self.n
        self.n += 1
        if i == self.t and self.a in cand:
            return int(self.a)
        return int(la_rule_policy(BASE_LA)(view, cand))


def _run(spec, sc, pa, pl):
    return float(spec.world().run_gated(
        seed_chain=sc, policy_pa=pa, policy_l=pl, dt_events=0)["makespan"])


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
    print(f"**top-K 覆盖率** · base PA={PA_RULE_NAMES[BASE_PA]} · "
          f"base LA={LA_RULE_NAMES[BASE_LA]} · {N_SC} 场景 · "
          f"每场景抽 {N_T} 个决策点\n", flush=True)

    ranks_pa, ranks_la = [], []
    for sc in scs:
        cap = Capture()
        _run(spec, sc, cap.policy_pa, cap.policy_l)

        for t in np.unique(np.linspace(0, len(cap.pa) - 1, N_T,
                                       dtype=int)):
            cand = cap.pa[t]
            ms = [_run(spec, sc, OPA(int(t), j, oi, m).policy_pa,
                       OLA(-1, -1).policy_l) for j, oi, m in cand]
            ranks_pa.append(int(np.argmin(ms)))

        for t in np.unique(np.linspace(0, len(cap.la) - 1, N_T,
                                       dtype=int)):
            cand = cap.la[t]
            ms = [_run(spec, sc, OPA(-1, 0, 0, 0).policy_pa,
                       OLA(int(t), a).policy_l) for a in cand]
            ranks_la.append(int(np.argmin(ms)))
        print(f"  sc={sc}  PA 决策 {len(cap.pa)} · LA 决策 {len(cap.la)}",
              flush=True)

    for tag, ranks, kmax in (("PA", ranks_pa, K_PA),
                             ("LA", ranks_la, K_LA)):
        r = np.asarray(ranks)
        print(f"\n  ── {tag}（{len(r)} 个采样决策）──")
        print(f"  {'K':>4}{'覆盖率':>9}")
        for k in range(1, kmax + 1):
            print(f"  {k:>4}{np.mean(r < k):>9.1%}")
        print(f"  最好候选的平均名次：{np.mean(r) + 1:.2f}")
    print("\n  ⚠️ “最好”只在 top-K_max 内选；单点偏离，不是全局最优。")


if __name__ == "__main__":
    main()
