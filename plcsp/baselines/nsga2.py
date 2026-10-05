# -*- coding: utf-8 -*-
"""NSGA-II 对位基线（spec §6.1 的 ③ 层）——**包装 pymoo**，评价走本仓 DES。

## 来源与许可证

- **算法**：Deb, Pratap, Agarwal, Meyarivan. *A fast and elitist multiobjective genetic
  algorithm: NSGA-II.* IEEE Transactions on Evolutionary Computation 6(2):182–197, 2002.
- **实现**：pymoo 0.6.2（Blank, Deb. *pymoo: Multi-Objective Optimization in Python.*
  IEEE Access 8:89497–89509, 2020），**Apache License 2.0**。
  本模块**只做包装**：编码/解码、调用本仓仿真、记账。选择/交叉/变异/非支配排序由 pymoo 提供。
  安装：`pip install pymoo==0.6.2`（依赖 moocore / autograd / cma / alive_progress，均不升 numpy）。

## 目标与评价口径

- **目标 = 本仓的三目标**（`plcsp/env/reward.objective_vector`）：
  makespan / energy（kWh）/ TWT。三者都是越小越好。
- **评价 = `rollout(inst, seed_chain=..., cfg=..., constraints=REPORT_TIERS[tier],
  op_choices=...)`**。这是本仓的**规则派车**入口（AGV 共享队列、空闲车接活），
  与 ① 规则层、`m14_mkt_reference` 的两档参考表**同一入口**。
- **不用 `run_gated`**：那是**在线策略**入口（DRL 的 S/L/R/M/C/B 回调）。
  离线元启发式没有可交给它的策略；用它就要改 NSGA-II 的方法。故不选。
- **可选 `agv_genes=True`**：额外编码每趟运输任务的车号，走 `run(agv_phi=...)`。
  那条路径是 **bound 派车**（每车一队列）——与 DRL 的 L 头同一条路径。
  默认关：默认档与 ① 规则层同口径，且标准 NSGA-II 不编码车号。

## 决策变量（染色体）

- **机台块**：每道工序一个整数 = 该工序候选表的序号。上界 = 候选数 − 1。
  MK01 的规模 = 55 基因（每工序候选 1–3 个）。
- ⚠️ **不编码工序顺序（OS）**：本仓 DES 在 t=0 全量投放作业，工序顺序由事件驱动规则
  涌现。仿真**没有**接收外部工序优先级的入口。要加就得改 `des.py`——本批不做。
  这是本基线的**已知范围限制**，必须写进论文。
- （可选）**车号块**：长度 = Σ每作业(工序数 + 1)（= 运输任务总数；MK01 = 65）。

## 两档口径（spec §6.1 / P4-B）

- `A-MKT` = `REPORT_TIERS["A-MKT"]`（十机制全关）。与已发表数字同口径，**确定性**
  （无故障/返工），单条随机链即可复现。
  ⚠️ 档 A 关着 ⑧ 交期 ⟹ **TWT 恒为 0**，实际只有两个活目标。
- `B-Full` = 机制全开。三目标都活（TWT 非零）；但故障/返工是随机的 ⟹
  整轮评价固定**同一条** `seed_chain`（共同随机数），换链就是换场景。
- 矩阵实例上档 B 开着 ⑪ 充电，而充电桩在矩阵里没有对应项 ⟹ `transport_unmapped="geometry"`
  声明式降级。降级段数/分钟数由仿真逐次报出，本模块累计后随结果带出。

## 代价（必须随结果报出）

NSGA-II 的代价按**仿真次数**算：`pop_size × (n_gen + 1)` 是理论上界
（初代 + 每代子代）。`eliminate_duplicates` 会去掉重复个体，故**实测次数可能更少**。
本模块逐次计数，`Nsga2Result.n_eval` 是**实测**值。
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import find_non_dominated

from ..env.constraints import REPORT_TIERS
from ..env.des import SimConfig, rollout
from ..env.instances import Instance
from ..env.reward import objective_vector

PYMOO_VERSION = "0.6.2"
OBJECTIVE_NAMES = ("makespan", "energy_kwh", "twt")


@dataclass(frozen=True)
class EvalSpec:
    """一轮 NSGA-II 的**评价口径**。口径是问题的属性，必须随结果一起报出。

    ⚠️ **三个 `SimConfig` 级开关（2026-10-05，A 主对比）默认全 `False` = 逐位等于既有读数**：
    DRL 的两档（`m13 --constraints None/Full`）跑在 `SimConfig(multi_drop=True,
    agv_failover=True, machine_age_failure=…)` 上（`experiment-plan.md` §9.7），基线必须
    **同一套动力学**才可比——`multi_drop` 改交付模型、`agv_failover` 改 ⑨ 停机期间的任务
    归属、`machine_age_failure` 改 ③ 的故障率。缺一个就是"两个问题各跑各的"。
    """

    tier: str = "A-MKT"
    seed_chain: int = 0
    n_agv: int = 3
    transport_unmapped: str = "geometry"
    agv_genes: bool = False
    multi_drop: bool = False
    agv_failover: bool = False
    machine_age_failure: bool = False

    def cfg(self) -> SimConfig:
        return SimConfig(n_agv=self.n_agv, transport_unmapped=self.transport_unmapped,
                         multi_drop=self.multi_drop, agv_failover=self.agv_failover,
                         machine_age_failure=self.machine_age_failure)


def transport_task_count(inst: Instance) -> int:
    """运输任务总数 = Σ每作业(工序数 + 1)。

    三条任务来源（`des._transporter` 的 docstring）：投放（1/作业）、工序流转（工序数 − 1）、
    回站（1/作业）。合计 = Σ工序数 + 作业数。MK01 = 55 + 10 = 65（有测试钉住）。
    """
    return sum(len(job) + 1 for job in inst.jobs)


def chromosome_length(inst: Instance, agv_genes: bool = False) -> int:
    n_ops = sum(len(job) for job in inst.jobs)
    return n_ops + (transport_task_count(inst) if agv_genes else 0)


def gene_bounds(inst: Instance, n_agv: int, agv_genes: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """基因下界/上界。机台块的上界 = 候选数 − 1；车号块的上界 = n_agv − 1。"""
    xu = [len(alts) - 1 for job in inst.jobs for alts in job]
    if agv_genes:
        if n_agv < 1:
            raise ValueError(f"n_agv 必须 ≥ 1，收到 {n_agv}")
        xu += [n_agv - 1] * transport_task_count(inst)
    return np.zeros(len(xu), dtype=int), np.array(xu, dtype=int)


def _gene_int(value: float, hi: int) -> int:
    """浮点基因 → 整数（四舍五入后裁剪）。pymoo 的整型算子给整数，此处只做失效安全。"""
    return int(min(max(int(round(float(value))), 0), hi))


def decode(x, inst: Instance, n_agv: int = 3,
           agv_genes: bool = False) -> tuple[list[list[int]], list[int] | None]:
    """基因 → `(op_choices, agv_phi)`。`agv_phi=None` 时 `run` 走 FIFO 共享队列。

    `op_choices[job][oi]` = 该工序候选表的序号（`des.run` 的既有契约）。
    `agv_phi[task_i]` = 第 task_i 个运输任务的 AGV 号——task_i = `_transporter` 的生成序。
    """
    x = np.asarray(x).reshape(-1)
    k = 0
    op_choices: list[list[int]] = []
    for job in inst.jobs:
        row: list[int] = []
        for alts in job:
            if k >= len(x):
                raise ValueError(f"基因长度不足：机台块需要 ≥ {k + 1} 个基因，收到 {len(x)}")
            row.append(_gene_int(x[k], len(alts) - 1))
            k += 1
        op_choices.append(row)
    if not agv_genes:
        return op_choices, None
    n_tasks = transport_task_count(inst)
    if k + n_tasks > len(x):
        raise ValueError(f"基因长度不足：车号块需要 {n_tasks} 个基因（总计 {k + n_tasks}），"
                         f"收到 {len(x)}")
    phi = [_gene_int(v, n_agv - 1) for v in x[k:k + n_tasks]]
    return op_choices, phi


class MachineAssignmentProblem(Problem):
    """pymoo 问题：基因 = 机台选择（+ 可选车号块），目标 = 本仓 DES 的三目标。

    ⚠️ **逐个个体串行仿真**（`_evaluate` 里 for 循环）——本仓仿真是纯 Python/SimPy，
    单进程多线程无收益（线程钉 1，见包入口）。代价按**仿真次数**记账。
    """

    def __init__(self, inst: Instance, spec: EvalSpec):
        self.inst = inst
        self.spec = spec
        if spec.tier not in REPORT_TIERS:
            raise ValueError(f"未知档位：{spec.tier}；可选 {sorted(REPORT_TIERS)}")
        self.constraints = REPORT_TIERS[spec.tier]
        self.cfg = spec.cfg()
        xl, xu = gene_bounds(inst, spec.n_agv, spec.agv_genes)
        super().__init__(n_var=len(xl), n_obj=len(OBJECTIVE_NAMES),
                         xl=xl, xu=xu, vtype=int)
        # ── 记账（必须随结果报出的代价读数）──
        self.n_eval = 0             # 实测仿真次数
        self.sim_s = 0.0            # 仿真耗时合计 [s]
        self.horizon_hits = 0       # 掐表（未完工）的仿真次数
        self.unmapped_legs = 0      # 矩阵未覆盖、几何降级补的段数合计

    def evaluate_one(self, x) -> tuple[float, float, float]:
        """一个基因 → 三目标。仿真入口 = `rollout`（规则派车，见模块 docstring）。"""
        ms, phi = decode(x, self.inst, self.spec.n_agv, self.spec.agv_genes)
        t0 = time.perf_counter()
        r = rollout(self.inst, seed_chain=self.spec.seed_chain, cfg=self.cfg,
                    constraints=self.constraints, op_choices=ms, agv_phi=phi)
        self.sim_s += time.perf_counter() - t0
        self.horizon_hits += int(bool(r["horizon_hit"]))
        self.unmapped_legs += int(r["unmapped_legs"])
        return objective_vector(r)

    def _evaluate(self, X, out, *args, **kwargs) -> None:
        F = np.empty((X.shape[0], len(OBJECTIVE_NAMES)), dtype=float)
        for i in range(X.shape[0]):
            F[i, :] = self.evaluate_one(X[i])
        self.n_eval += X.shape[0]
        out["F"] = F


@dataclass(frozen=True)
class Nsga2Result:
    """一轮 NSGA-II 的结果：**非支配前沿** + 代价读数 + 口径标签。"""

    F: np.ndarray            # (n_points, 3) 目标值，列 = makespan / energy / TWT
    X: np.ndarray            # (n_points, n_var) 前沿点基因
    tier: str
    seed: int
    seed_chain: int
    pop_size: int
    n_gen: int
    n_eval: int              # 实测仿真次数
    wall_s: float            # 墙钟合计 [s]
    sim_s: float             # 仿真耗时合计 [s]
    horizon_hits: int
    unmapped_legs: int
    agv_genes: bool
    # 评价的 `SimConfig` 级开关（`EvalSpec` 的三个）——随结果一起报，别让口径只活在调用参数里
    multi_drop: bool = False
    agv_failover: bool = False
    machine_age_failure: bool = False

    def summary(self) -> str:
        """一段可粘进报告的文本（口径 + 前沿 + 代价）。"""
        theory = self.pop_size * (self.n_gen + 1)
        lines = [
            f"NSGA-II（pymoo {PYMOO_VERSION}，Apache-2.0）｜档 {self.tier}"
            f"｜seed={self.seed}｜sim_seed_chain={self.seed_chain}"
            f"｜pop={self.pop_size}×gen={self.n_gen}｜车号块={self.agv_genes}",
            f"cfg：multi_drop={self.multi_drop}"
            f"｜agv_failover={self.agv_failover}｜machine_age_failure={self.machine_age_failure}",
            f"目标 = {OBJECTIVE_NAMES[0]} / {OBJECTIVE_NAMES[1]} / {OBJECTIVE_NAMES[2]}"
            f"（本仓 DES，rollout 规则派车）",
            f"代价：理论仿真次数 = pop×(gen+1) = {theory}；实测 = {self.n_eval}"
            f"；墙钟 = {self.wall_s:.1f} s；仿真合计 = {self.sim_s:.1f} s"
            f"（{self.sim_s / max(self.n_eval, 1) * 1000:.0f} ms/次）",
        ]
        if self.horizon_hits:
            lines.append(f"⚠️ 掐表（未完工）仿真次数 = {self.horizon_hits}")
        if self.unmapped_legs:
            lines.append(f"⚠️ 矩阵未覆盖、几何降级补的段数合计 = {self.unmapped_legs}")
        lines.append(f"Pareto 前沿：{self.F.shape[0]} 个点")
        for row in self.F:
            lines.append("  " + "  ".join(f"{name}={v:.2f}" for name, v in zip(OBJECTIVE_NAMES, row)))
        return "\n".join(lines)


def run_nsga2(inst: Instance, spec: EvalSpec, *, pop_size: int = 8, n_gen: int = 2,
              seed: int = 0) -> Nsga2Result:
    """跑一轮 NSGA-II，返回**非支配前沿**（对最终种群再做一次非支配排序）。

    ⚠️ `seed` 只控制 pymoo 的随机流；`spec.seed_chain` 控制仿真随机链。
    两者都固定 ⟹ 结果可复现（仿真是确定性的，见测试）。
    """
    if pop_size < 2:
        raise ValueError(f"pop_size 必须 ≥ 2，收到 {pop_size}")
    if n_gen < 1:
        raise ValueError(f"n_gen 必须 ≥ 1，收到 {n_gen}")
    problem = MachineAssignmentProblem(inst, spec)
    algo = NSGA2(pop_size=pop_size,
                 sampling=IntegerRandomSampling(),
                 crossover=SBX(prob=0.9, eta=15, vtype=int),
                 mutation=PM(eta=20, vtype=int),
                 eliminate_duplicates=True)
    t0 = time.perf_counter()
    res = minimize(problem, algo, ("n_gen", n_gen), seed=int(seed), verbose=False)
    wall = time.perf_counter() - t0
    F_all = np.atleast_2d(np.asarray(res.F, dtype=float))
    X_all = np.atleast_2d(np.asarray(res.X, dtype=float))
    front = find_non_dominated(F_all) if len(F_all) > 1 else np.arange(len(F_all))
    F, X = F_all[front], X_all[front]
    order = np.lexsort((F[:, 2], F[:, 1], F[:, 0]))
    return Nsga2Result(F=F[order], X=X[order], tier=spec.tier, seed=int(seed),
                       seed_chain=spec.seed_chain, pop_size=pop_size, n_gen=n_gen,
                       n_eval=problem.n_eval, wall_s=wall, sim_s=problem.sim_s,
                       horizon_hits=problem.horizon_hits, unmapped_legs=problem.unmapped_legs,
                       agv_genes=spec.agv_genes, multi_drop=spec.multi_drop,
                       agv_failover=spec.agv_failover,
                       machine_age_failure=spec.machine_age_failure)


def main() -> None:
    ap = argparse.ArgumentParser(description="NSGA-II 对位基线（包装 pymoo，评价走本仓 DES）")
    ap.add_argument("--inst", default="mk01", help="MK 实例名（默认走 MKT 口径）")
    ap.add_argument("--geometry", action="store_true",
                    help="用原始 MK 几何口径实例（默认用 MKT 矩阵口径实例）")
    ap.add_argument("--tier", default="A-MKT", choices=sorted(REPORT_TIERS))
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--gens", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0, help="pymoo 随机流")
    ap.add_argument("--seed-chain", type=int, default=0, help="仿真随机链")
    ap.add_argument("--n-agv", default="default",
                    help='"default" = 该实例的车数设定（MKT: v=m；几何: 3）')
    ap.add_argument("--agv-genes", action="store_true", help="额外编码每趟任务的车号")
    # ⚠️ 三个 `SimConfig` 级开关（2026-10-05，A 主对比）默认全关 ⟹ 既有读数逐位不变。
    #    DRL 两档跑在它们上面（§9.3/§9.7），基线要同口径时打开对应的那几个。
    ap.add_argument("--multi-drop", action="store_true",
                    help="⑩ multi-drop 行程模型（DRL 两档都开；缺它就是另一个交付模型）")
    ap.add_argument("--agv-failover", action="store_true",
                    help="⑨ 故障 failover（⑨ 关时惰性）")
    ap.add_argument("--machine-age-failure", action="store_true",
                    help="③ 役龄故障率（档 B 开、档 A 关）")
    args = ap.parse_args()

    if args.geometry:
        from ..env.instances import load_mk
        inst = load_mk(args.inst)
        default_v = 3
    else:
        from ..env.mkt import load_mkt
        mkt = load_mkt(args.inst)
        inst, default_v = mkt.base, mkt.n_agv
    v = default_v if args.n_agv == "default" else int(args.n_agv)
    spec = EvalSpec(tier=args.tier, seed_chain=args.seed_chain, n_agv=v,
                    agv_genes=args.agv_genes, multi_drop=args.multi_drop,
                    agv_failover=args.agv_failover,
                    machine_age_failure=args.machine_age_failure)
    res = run_nsga2(inst, spec, pop_size=args.pop, n_gen=args.gens, seed=args.seed)
    print(f"实例={args.inst}（{'几何' if args.geometry else 'MKT 矩阵'}口径，v={v}）")
    print(res.summary())


if __name__ == "__main__":
    main()
