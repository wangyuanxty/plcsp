# -*- coding: utf-8 -*-
"""NSGA-II vs 36 CDR —— 同一组留出随机实例、同一 v3 布局/配置/场景。

⚠️ 口径差异（必须随结果报出）：
- NSGA-II 走 `env.des.rollout`（**离线机台分配 + FIFO 派车**），
  染色体只编码机台选择，不编码工序顺序、不编码 LA 规则。
- CDR 走 `run_gated`（**在线 PA/LA 规则**）。
- 因此这是“同一 DES、同一实例/布局/场景”下的**不同决策子空间**对照，
  不是逐位可比。

NSGA-II 是**每实例单独优化**；CDR 是固定规则。
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
from pymoo.algorithms.moo.nsga2 import NSGA2  # noqa: E402
from pymoo.core.problem import Problem  # noqa: E402
from pymoo.operators.crossover.sbx import SBX  # noqa: E402
from pymoo.operators.mutation.pm import PM  # noqa: E402
from pymoo.operators.sampling.rnd import IntegerRandomSampling  # noqa: E402
from pymoo.optimize import minimize  # noqa: E402
from pymoo.util.nds.non_dominated_sorting import find_non_dominated  # noqa: E402

from plcsp_v3.baselines.nsga2 import decode, gene_bounds  # noqa: E402
from plcsp_v3.env.des import rollout  # noqa: E402
from plcsp_v3.env.instances import gen_mk_like  # noqa: E402
from plcsp_v3.env.reward import objective_vector  # noqa: E402
from plcsp_v3.hrl.cdr import cdr_pairs, run_cdr  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

N_VAL = int(os.environ.get("NSGA_NVAL", "10"))
VAL_SEED = 5000
LAYOUT_BASE = 500
N_AGV = 3
SC = 1
POP = int(os.environ.get("NSGA_POP", "16"))
GEN = int(os.environ.get("NSGA_GEN", "10"))


def _specs():
    out = []
    for i in range(N_VAL):
        inst = gen_mk_like(seed=VAL_SEED + i)
        sp, _ = build(seed=0, inst_obj=inst, tau=2.0, pm_interval=30,
                      target_failures=4, n_agv=N_AGV, finite_buffer=True,
                      layout_seed=LAYOUT_BASE + i, reward_norm="minmax",
                      sim_overrides={"due_range": 0.6})
        out.append((f"val{i}", sp))
    return out


class V3Problem(Problem):
    def __init__(self, spec):
        self.spec = spec
        xl, xu = gene_bounds(spec.inst, spec.cfg.n_agv, False)
        super().__init__(n_var=len(xl), n_obj=3, xl=xl, xu=xu, vtype=int)
        self.n_eval = 0

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.empty((len(X), 3), dtype=float)
        for i in range(len(X)):
            ms, _phi = decode(X[i], self.spec.inst, self.spec.cfg.n_agv, False)
            r = rollout(self.spec.inst, layout=self.spec.layout, seed_chain=SC,
                        cfg=self.spec.cfg, constraints=self.spec.constraints,
                        op_choices=ms)
            F[i] = objective_vector(r)
        self.n_eval += len(X)
        out["F"] = F


def _nsga(spec):
    prob = V3Problem(spec)
    algo = NSGA2(pop_size=POP,
                 sampling=IntegerRandomSampling(),
                 crossover=SBX(prob=0.9, eta=15, vtype=int),
                 mutation=PM(eta=20, vtype=int),
                 eliminate_duplicates=True)
    t0 = time.perf_counter()
    res = minimize(prob, algo, ("n_gen", GEN), seed=0, verbose=False)
    wall = time.perf_counter() - t0
    F = np.atleast_2d(np.asarray(res.F, dtype=float))
    if len(F) > 1:
        F = F[find_non_dominated(F)]
    k = int(np.argmin(F[:, 0]))
    return F[k], len(F), prob.n_eval, wall


def main() -> None:
    specs = _specs()
    print(f"**NSGA-II vs 36 CDR** · {N_VAL} 留出实例 · layout 500+i · "
          f"场景 {SC} · pop={POP} × gen={GEN}\n", flush=True)

    pairs = cdr_pairs(12)
    cdr = np.empty((len(pairs), len(specs), 3), dtype=float)
    for pi, (pa, la) in enumerate(pairs):
        for mi, (_n, sp) in enumerate(specs):
            cdr[pi, mi] = objective_vector(
                run_cdr(sp.world(), pa, la, seed_chain=SC))
    best_pi = int(np.argmin(cdr[:, :, 0].mean(axis=1)))
    oracle = np.asarray([cdr[int(np.argmin(cdr[:, mi, 0])), mi]
                         for mi in range(len(specs))], dtype=float)
    print(f"  CDR 最好单条：{pairs[best_pi][0]}+{pairs[best_pi][1]}", flush=True)
    print(f"  CDR 最好单条均值：MS {cdr[best_pi, :, 0].mean():.1f}  "
          f"E {cdr[best_pi, :, 1].mean():.2f}  TWT {cdr[best_pi, :, 2].mean():.1f}",
          flush=True)
    print(f"  CDR per-instance oracle：MS {oracle[:, 0].mean():.1f}  "
          f"E {oracle[:, 1].mean():.2f}  TWT {oracle[:, 2].mean():.1f}\n",
          flush=True)

    print(f"  {'instance':<8}{'NSGA MS':>10}{'NSGA E':>9}{'NSGA TWT':>10}"
          f"{'CDR MS':>10}{'front':>7}{'n_eval':>8}{'wall':>7}", flush=True)
    print("  " + "-" * 72, flush=True)
    rows = []
    for mi, (name, sp) in enumerate(specs):
        f, nf, ne, wall = _nsga(sp)
        rows.append(f)
        print(f"  {name:<8}{f[0]:>10.1f}{f[1]:>9.2f}{f[2]:>10.1f}"
              f"{oracle[mi, 0]:>10.1f}{nf:>7}{ne:>8}{wall:>6.1f}s", flush=True)
    rows = np.asarray(rows, dtype=float)
    print("  " + "-" * 72, flush=True)
    print(f"  {'mean':<8}{rows[:, 0].mean():>10.1f}{rows[:, 1].mean():>9.2f}"
          f"{rows[:, 2].mean():>10.1f}{oracle[:, 0].mean():>10.1f}", flush=True)
    win = int((rows[:, 0] < oracle[:, 0]).sum())
    win_b = int((rows[:, 0] < cdr[best_pi, :, 0]).sum())
    print(f"\n  NSGA-II 在 MS 上赢 per-instance oracle：{win}/{len(specs)}", flush=True)
    print(f"  NSGA-II 在 MS 上赢最好单条 CDR：{win_b}/{len(specs)}", flush=True)
    print("  ⚠️ NSGA-II 只编码机台分配；CDR 走在线 PA/LA。子空间不同。", flush=True)


if __name__ == "__main__":
    main()
