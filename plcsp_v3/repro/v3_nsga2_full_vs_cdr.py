# -*- coding: utf-8 -*-
"""完整编码 NSGA-II vs 36 CDR。

染色体 = **机台块 + 工序优先级块 + AGV 块**：
- 机台块：每道工序的候选序号（`op_choices`）；
- 工序优先级块：每道工序一个整数 key（越小越先），传给 `rollout(seq_rank=...)`；
- AGV 块：每趟运输任务的车号（`rollout(agv_phi=...)`，bound 派车）。

⚠️ 仍是**离线**基线（整张染色体开跑前定死）；CDR 是**在线** PA/LA。
这是方法类别差异，不是实现缺口。

⚠️ `nsga2.py` 的旧注释写“DES 没有工序排序入口”——**那是错的**：
`des.rollout` 从 2026-10-06 起就有 `seq_rank`，本脚本使用它。
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

from plcsp_v3.baselines.nsga2 import _gene_int, transport_task_count  # noqa: E402
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
POP = int(os.environ.get("NSGA_POP", "40"))
GEN = int(os.environ.get("NSGA_GEN", "30"))
AGV_GENES = os.environ.get("NSGA_AGV", "1") == "1"


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


def _bounds(inst, n_agv: int, agv_genes: bool):
    ma = [len(alts) - 1 for job in inst.jobs for alts in job]
    n_ops = len(ma)
    os_blk = [n_ops - 1] * n_ops
    if not agv_genes:
        xu = ma + os_blk
        return np.zeros(len(xu), dtype=int), np.asarray(xu, dtype=int), n_ops
    nt = transport_task_count(inst)
    xu = ma + os_blk + [n_agv - 1] * nt
    return np.zeros(len(xu), dtype=int), np.asarray(xu, dtype=int), n_ops


def _decode(x, inst, n_agv: int, agv_genes: bool):
    x = np.asarray(x).reshape(-1)
    n_ops = sum(len(job) for job in inst.jobs)
    op_choices, ranks = [], []
    k = 0
    for job in inst.jobs:
        mrow, rrow = [], []
        for alts in job:
            mrow.append(_gene_int(x[k], len(alts) - 1))
            rrow.append(_gene_int(x[n_ops + k], n_ops - 1))
            k += 1
        op_choices.append(mrow)
        ranks.append(rrow)
    phi = None
    if agv_genes:
        nt = transport_task_count(inst)
        phi = [_gene_int(v, n_agv - 1) for v in x[2 * n_ops:2 * n_ops + nt]]
    return op_choices, ranks, phi


class FullProblem(Problem):
    def __init__(self, spec):
        self.spec = spec
        xl, xu, self.n_ops = _bounds(spec.inst, spec.cfg.n_agv, AGV_GENES)
        super().__init__(n_var=len(xl), n_obj=3, xl=xl, xu=xu, vtype=int)
        self.n_eval = 0

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.empty((len(X), 3), dtype=float)
        for i in range(len(X)):
            ms, ranks, phi = _decode(X[i], self.spec.inst,
                                     self.spec.cfg.n_agv, AGV_GENES)
            r = rollout(self.spec.inst, layout=self.spec.layout, seed_chain=SC,
                        cfg=self.spec.cfg, constraints=self.spec.constraints,
                        op_choices=ms, seq_rank=ranks, agv_phi=phi)
            F[i] = objective_vector(r)
        self.n_eval += len(X)
        out["F"] = F


def _nsga(spec):
    prob = FullProblem(spec)
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
    print(f"**完整 NSGA-II vs 36 CDR** · {N_VAL} 留出实例 · layout 500+i · "
          f"场景 {SC} · pop={POP} × gen={GEN} · AGV块={AGV_GENES}\n", flush=True)

    pairs = cdr_pairs(12)
    cdr = np.empty((len(pairs), len(specs), 3), dtype=float)
    for pi, (pa, la) in enumerate(pairs):
        for mi, (_n, sp) in enumerate(specs):
            cdr[pi, mi] = objective_vector(
                run_cdr(sp.world(), pa, la, seed_chain=SC))
    best_pi = int(np.argmin(cdr[:, :, 0].mean(axis=1)))
    oracle = np.asarray([cdr[int(np.argmin(cdr[:, mi, 0])), mi]
                         for mi in range(len(specs))], dtype=float)
    print(f"  CDR 最好单条：{pairs[best_pi][0]}+{pairs[best_pi][1]}  "
          f"MS {cdr[best_pi, :, 0].mean():.1f}", flush=True)
    print(f"  CDR per-instance oracle：MS {oracle[:, 0].mean():.1f}\n",
          flush=True)

    print(f"  {'instance':<8}{'NSGA MS':>10}{'NSGA E':>9}{'NSGA TWT':>10}"
          f"{'CDR MS':>10}{'front':>7}{'n_eval':>9}{'wall':>8}", flush=True)
    print("  " + "-" * 74, flush=True)
    rows = []
    for mi, (name, sp) in enumerate(specs):
        f, nf, ne, wall = _nsga(sp)
        rows.append(f)
        print(f"  {name:<8}{f[0]:>10.1f}{f[1]:>9.2f}{f[2]:>10.1f}"
              f"{oracle[mi, 0]:>10.1f}{nf:>7}{ne:>9}{wall:>7.0f}s", flush=True)
    rows = np.asarray(rows, dtype=float)
    print("  " + "-" * 74, flush=True)
    print(f"  {'mean':<8}{rows[:, 0].mean():>10.1f}{rows[:, 1].mean():>9.2f}"
          f"{rows[:, 2].mean():>10.1f}{oracle[:, 0].mean():>10.1f}", flush=True)
    win = int((rows[:, 0] < oracle[:, 0]).sum())
    win_b = int((rows[:, 0] < cdr[best_pi, :, 0]).sum())
    print(f"\n  NSGA-II 赢 per-instance oracle：{win}/{len(specs)}", flush=True)
    print(f"  NSGA-II 赢最好单条 CDR：{win_b}/{len(specs)}", flush=True)
    print("  ⚠️ NSGA-II 是离线（整张染色体定死）；CDR 是在线 PA/LA。", flush=True)


if __name__ == "__main__":
    main()
