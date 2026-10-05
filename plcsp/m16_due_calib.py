# -*- coding: utf-8 -*-
"""交期标定：在 (τ, R) 网格上为每个实例选出冻结的 (tardiness factor, due-date range)。

标定规则（⑧ 重设计计划 Task 2，**原样实现**）
--------------------------------------------
在 (τ, R) 网格上求满足下式者，取

    |参考误期率 − target_ref| + |改进后误期率 − 0.40| + 0.02·R

**最小**的一对，约束为：

- `lo ≤ 参考策略误期率 ≤ hi`（默认 `0.20 ≤ · ≤ 0.75`）
- `lo ≤ 改进 `improve`（默认 0.88，即"策略改进 12%"）后的误期率 ≤ hi`

**为什么要"改进后也不许退化"这一条**：这正是旧口径 `d_j = τ·M_ref` 的死因——τ=0.90 只对着
**参考策略**标定，而训练后策略改进约 12–14%，一次性把 TWT 清零（实测 makespan 89.5 <
`d_j = 93.08` ⟹ TWT ≡ 0）。收紧 τ 是**保守方向**（只会让我们的 TWT 变大），故本规则里
不存在"调参数美化结果"的激励。

网格与常数（**计划未指定，本实现显式写死在此，便于复核**）
----------------------------------------------------------
- `TAU_GRID`（几何口径）= 0.50 → 6.00，步长 0.05（覆盖实测 1.55–5.85 且两端留余量）
- **矩阵口径的 τ 上限单独取**（`TAU_HI_BY_CALIBER`）：矩阵已是分钟、且装卸站往返也走矩阵，
  参考 makespan 相对 LB 大一个量级。实测（P4-B Task 3 Step 1 探针，`seed_chain=0`、
  `n_agv=m`、`ConstraintConfig().with_off("due_dates")`）10 个实例的 `M_ref / LB`：
  mk01 26.61、mk02 27.74、mk03 19.44、mk04 27.55、mk05 6.03、mk06 51.94、mk07 11.99、
  mk08 11.09、mk09 15.78、mk10 55.39 ⟹ 最大 55.39，×1.5 = 83.09 ⟹ 上限取 **85.0**。
  ⚠️ **不得**顺手抬高几何口径的上限——搜索空间一开，冻结表可能选出不同的 (τ,R)。
- `R_GRID` = 0.20 → **0.80**，步长 0.10（**下限 0.20**：R=0 会退化成共同交期；**上限 0.80**：
  R>1 会产出负交期——两条理由不同，见下方常数处注释）
- 改进后误期率的目标 `TARGET_IMP = 0.40`、R 罚项 `R_PENALTY = 0.02`（规则原文）
- 平局取 **(τ 小, R 小)** 者（遍历按升序 + 严格小于）——保证产物可复现

口径与 cfg 条件（P4-B）
----------------------
- 口径**跟随实例**（`inst.transport`）：原始 MK = `geometry`（缺省），MKT = `matrix`。本脚本按
  实例标签选 τ 网格与打印的表名，**不**由 cfg 决定口径。
- 几何口径的默认 cfg 仍是 `SimConfig()`（n_agv=3、车速 0.5、通道 1.5 m…）——**数字逐位不变**。
- 矩阵口径的默认 cfg = `SimConfig(n_agv=m, transport_unmapped="geometry")`：v = m 是
  HGS/HA-DQN 的口径；⑪ 充电在标定配置里是**开着**的，而充电桩在矩阵里没有对应项 ⟹
  必须显式选**声明式**几何降级（并被计数），否则标定第一趟就报错。
- "误期率落带是 cfg 条件"这条继续成立：换车队规模/车速会改参考调度 ⟹ 同一个 `d_j` 下误期率变。

跑法::

    PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m16_due_calib
    PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m16_due_calib --instances mk01 --target-ref 0.45
    PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m16_due_calib --transport matrix

把打印出的表贴回 `plcsp/env/due_dates.py`（几何表 `TF_RDD`、矩阵表 `TF_RDD_MATRIX`）；
`plcsp/tests/test_due_calib.py` 与 `plcsp/tests/test_due_calib_mkt.py` 会逐实例核对
"冻结的表 == 脚本重算的结果"（Review Focus #6：表是可复现产物，不是手抄魔数）。
"""
from __future__ import annotations

import argparse

import numpy as np

from .env.constraints import ConstraintConfig
from .env.des import SimConfig, rollout
from .env.due_dates import tf_rdd_due_dates, workload_lower_bound
from .env.instances import Instance, load_mk

TAU_LO, TAU_HI, TAU_STEP = 0.50, 6.00, 0.05
# ⚠️ 网格上限**按行程时间口径**取（P4-B Task 3 Step 1 实测）：矩阵口径的行程是分钟的整数级
#    （几何口径 ~0.2 min/段），参考 makespan 大数倍 ⟹ 同一个 LB 下 τ 要同比放大。
#    实测最大 `M_ref / LB` = 55.39（mk10）⟹ ×1.5 = 83.09 ⟹ 取 85.0（实测比值表见模块 docstring）。
#    ⚠️ **不要**抬高几何口径的上限（`TAU_HI`）——搜索空间一开，冻结表可能选出不同的 (τ,R)，
#    作废 §21.2 的冻结表（`test_frozen_table_is_reproduced_by_the_calibration_script` 盯着）。
TAU_HI_BY_CALIBER: dict[str, float] = {"geometry": TAU_HI, "matrix": 85.0}
# ⚠️ `R_LO = 0.20`（**下限，不是可行域边界**）：R = 0 意味着交期退化成**共同交期**
#    （所有作业同值），而"逐作业、有跨度"是用户对本口径的**明确选择**。
#    实测：R=0 在 10 个实例上**并非不可行**（mk10 在 τ∈[2.70,2.90] 就可行）——
#    早先"mk10 无 R 不可行"的判断是把 `τ·W_j` 与 `τ·LB` 两种参数化混为一谈得出的，**是错的**。
#    故 R 下限的理由**不是可行性**，而是 ① 落实逐作业口径 ② 让 `due_margin` 维不再冗余。
#    代价：mk01/mk04 的 (τ,R) 会从 (2.50,0.0)/(3.85,0.0) 变成下方表里的值。
R_LO, R_HI, R_STEP = 0.20, 0.80, 0.10
# ⚠️ `R_HI = 0.80`（**上限**）：`d_j = LB·τ·(1 + R·(2ρ_j − 1))` 在 `R > 1` 时**最低倍率越过零点**，
#    会产出**负交期**。实测踩到过：mk04 在 R=1.2 下最短作业交期 −35.2，而当时只看两个误期率、
#    没看交期本身，一路绿着发了出去。上限 0.8 ⟹ 最低倍率恒 ≥ 0.2 > 0。
#    `tests/test_due_dates.py::test_due_dates_are_strictly_positive` 钉住；`tf_rdd_due_dates`
#    里另有一条显式守卫（双保险：标定网格改了也不会静默发出）。
TARGET_IMP = 0.40          # 改进后误期率的目标（规则原文）
R_PENALTY = 0.02           # R 的罚项系数（规则原文：交期跨度越大越保守）


def _grid(lo: float, hi: float, step: float) -> np.ndarray:
    """闭区间网格（`np.arange` 的浮点端点上取整，避免 5.999999 这类残渣）。"""
    n = int(round((hi - lo) / step)) + 1
    return np.round(lo + step * np.arange(n), 6)


TAU_GRID = _grid(TAU_LO, TAU_HI, TAU_STEP)      # 几何口径的网格（既有，不动）
R_GRID = _grid(R_LO, R_HI, R_STEP)

# 口径标签（与 `plcsp/env/due_dates.py` 的 `TABLE_BY_CALIBER` 同一套词）。
GEOMETRY, MATRIX = "geometry", "matrix"


def _caliber_of(inst: Instance) -> str:
    """实例的行程时间口径标签（缺省 `geometry`）——决定用哪张 τ 网格、贴回哪张表。"""
    return getattr(inst, "transport", GEOMETRY)


def tau_grid(caliber: str) -> np.ndarray:
    """该口径的 τ 网格（步长/下限共用，**上限按口径取**——理由见 `TAU_HI_BY_CALIBER`）。

    口径未知 ⟹ 显式报错，**不得**静默退回几何网格（那会让矩阵口径在 6.00 就截断，
    选出一个"看起来可行"但其实够不到 τ 真值的点，且零报错）。
    """
    try:
        hi = TAU_HI_BY_CALIBER[caliber]
    except KeyError:
        raise ValueError(
            f"未知行程时间口径 {caliber!r}——无法确定 τ 网格上限"
            f"（已知：{sorted(TAU_HI_BY_CALIBER)}）") from None
    return _grid(TAU_LO, hi, TAU_STEP)


def reference_completes(inst: Instance, cfg: SimConfig | None = None,
                        constraints: ConstraintConfig | None = None) -> dict[int, float]:
    """参考策略（每工序取最短候选 + AGV 轮询）的完成时刻 `C_j`，`seed_chain=0`。

    ⚠️ **关掉 ⑧ 跑**（`due_dates=False`）：交期是 metric、不进时序——实测 ⑧ 开/关的
    `completes` 与 makespan **逐位相同**（mk01/mk04 对拍）。关掉还解开一个鸡生蛋：
    `_due_map` 会去查 `TF_RDD`，未标定实例在标定它的那一刻就会报错（Task 3 接线后）。

    ⚠️ **cfg 条件**：`cfg=None` → `SimConfig()`（几何口径的默认：n_agv=3、车速 0.5、通道 1.5 m…），
    故**冻结的 (τ,R) 只对默认 cfg 保证"两个误期率落在 [0.20, 0.75]"**。换车队规模/车速/通道宽会改
    参考调度的 `completes` ⟹ 同一个 `d_j` 下的误期率随之变（评审实测 mk01 在 **n_agv=1**：参考误期率
    **1.00**、改进 12% 后 **0.80**，**双双出带**）。
    ⚠️ 两件事必须分开说：**交期本身是外生的**（`d_j` 与 cfg 无关，`test_due_dates_do_not_depend_on_fleet_size`
    钉住），**但"误期率落在带内"是 cfg 条件的**。P4 计划的 `n_agv` 1/3/5 扫描**必须先在新 cfg 下重标**，
    或至少如实报告哪些点出带（不得沿用本表的默认-cfg 保证）。
    ⚠️ P4-B：矩阵口径必须显式传 cfg（`SimConfig(n_agv=m, transport_unmapped="geometry")`）——
    默认的 `SimConfig()` 跑的是几何口径的行程，矩阵实例上会给出错的 `C_j`。
    """
    c = SimConfig() if cfg is None else cfg
    cons = ConstraintConfig().with_off("due_dates") if constraints is None else constraints
    return rollout(inst, seed_chain=0, cfg=c, constraints=cons)["completes"]


def _rates(inst: Instance, tau: float, due_range: float, completes: dict[int, float],
           improve: float) -> tuple[float, float]:
    """(参考策略误期率, 改进 `improve` 后的误期率)——同一份 `C_j` 上的两个计数。"""
    due = tf_rdd_due_dates(inst, tau, due_range)
    n = inst.n_jobs
    ref = sum(1 for j in range(n) if completes[j] > due[j]) / n
    imp = sum(1 for j in range(n) if completes[j] * improve > due[j]) / n
    return ref, imp


def calibrate_report(inst: Instance, *, cfg: SimConfig | None = None, target_ref: float = 0.45,
                     improve: float = 0.88, lo: float = 0.20, hi: float = 0.75) -> dict:
    """网格搜索的主实现：返回 `{tau, due_range, ref, imp, lb, score}`。

    与 `calibrate` 的分工：本函数把中间读数（两个误期率）也带出来供 CLI 打表；
    `calibrate` 是计划规定的薄封装（只要 `(τ, R)`）。

    ⚠️ τ 网格与参考运行**都按实例的口径**取（`inst.transport`）：矩阵实例必须显式传
    `cfg=SimConfig(n_agv=m, transport_unmapped="geometry")`，否则参考运行走的是几何行程。
    """
    C = reference_completes(inst, cfg=cfg)
    grid = tau_grid(_caliber_of(inst))
    best = None
    for tau in grid:
        for R in R_GRID:
            ref, imp = _rates(inst, float(tau), float(R), C, improve)
            if not (lo <= ref <= hi and lo <= imp <= hi):
                continue
            score = abs(ref - target_ref) + abs(imp - TARGET_IMP) + R_PENALTY * float(R)
            # 严格小于 ⟹ 平局保留先到者（τ 升序、R 升序）⟹ 产物可复现
            if best is None or score < best["score"]:
                best = {"tau": float(tau), "due_range": float(R), "ref": ref,
                        "imp": imp, "score": float(score)}
    if best is None:
        hi_cal = TAU_HI_BY_CALIBER[_caliber_of(inst)]
        raise ValueError(
            f"({TAU_LO}–{hi_cal}, {R_LO}–{R_HI}) 网格上无可行 (τ,R)：约束 "
            f"{lo} ≤ 两个误期率 ≤ {hi} 太紧，或参考策略的完成时刻分布太窄/太宽。"
            "请放宽 lo/hi 或扩大网格。")
    best["lb"] = workload_lower_bound(inst)
    return best


def calibrate(inst: Instance, *, cfg: SimConfig | None = None, target_ref: float = 0.45,
              improve: float = 0.88, lo: float = 0.20, hi: float = 0.75) -> tuple[float, float]:
    """标定一个实例，返回 `(τ, R)`——规则与网格见模块 docstring。"""
    r = calibrate_report(inst, cfg=cfg, target_ref=target_ref, improve=improve, lo=lo, hi=hi)
    return r["tau"], r["due_range"]


def _cli_cfg(transport: str, n_agv: str, m: int) -> SimConfig:
    """CLI 的标定 cfg：几何口径保持 `SimConfig()`（既有数字逐位不变）；矩阵口径给 v 与降级策略。

    矩阵口径**必须**显式选 `transport_unmapped="geometry"`：标定配置里 ⑪ 充电是开着的，
    而充电桩在矩阵里没有对应项——不选就 `raise`（这是**声明式**降级并计数，不是静默降级）。
    """
    if transport == GEOMETRY:
        return SimConfig() if n_agv == "default" else SimConfig(n_agv=int(n_agv))
    v = m if n_agv == "default" else int(n_agv)
    return SimConfig(n_agv=v, transport_unmapped="geometry")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default=",".join(f"mk{i:02d}" for i in range(1, 11)))
    ap.add_argument("--transport", choices=(GEOMETRY, MATRIX), default=GEOMETRY,
                    help="行程时间口径：geometry（原始 MK，默认）或 matrix（MKT）")
    ap.add_argument("--n-agv", default="default",
                    help='"default" = 几何口径的 SimConfig()（n_agv=3）／'
                         '矩阵口径的 v=m（HGS/HA-DQN 口径）；也可给整数')
    ap.add_argument("--target-ref", type=float, default=0.45,
                    help="参考策略误期率的目标值（评分项之一）")
    ap.add_argument("--improve", type=float, default=0.88,
                    help="策略改进比例：C_j × improve（默认 0.88 = 改进 12%%）")
    args = ap.parse_args()
    table_name = "TF_RDD" if args.transport == GEOMETRY else "TF_RDD_MATRIX"
    tau_hi = TAU_HI_BY_CALIBER[args.transport]
    print(f"口径 = {args.transport}｜标定规则 target_ref={args.target_ref} improve={args.improve}｜"
          f"网格 τ {TAU_LO}–{tau_hi}/{TAU_STEP}｜R {R_LO}–{R_HI}/{R_STEP}｜"
          f"评分 |ref−{args.target_ref}| + |imp−{TARGET_IMP}| + {R_PENALTY}·R")
    if args.transport == MATRIX:
        print("⚠️ 矩阵口径的 cfg = SimConfig(n_agv=v, transport_unmapped='geometry')："
              "⑪ 充电开着而充电桩无矩阵项 ⟹ **声明式**几何降级 + 计数。")
    print(f"{'inst':<7}{'v':>3}{'LB':>9}{'tau':>7}{'R':>6}{'ref':>8}{'imp':>8}{'score':>9}")
    rows: dict[str, tuple[float, float]] = {}
    for name in args.instances.split(","):
        if args.transport == MATRIX:
            from .env.mkt import load_mkt
            inst = load_mkt(name).base
        else:
            inst = load_mk(name)
        m = inst.n_machines
        v = m if args.n_agv == "default" else int(args.n_agv)
        cfg = _cli_cfg(args.transport, args.n_agv, m)
        r = calibrate_report(inst, cfg=cfg, target_ref=args.target_ref, improve=args.improve)
        rows[name] = (r["tau"], r["due_range"])
        print(f"{name:<7}{v:>3}{r['lb']:>9.2f}{r['tau']:>7.2f}{r['due_range']:>6.1f}"
              f"{r['ref']:>8.2f}{r['imp']:>8.2f}{r['score']:>9.3f}", flush=True)
    print(f"\n# ── 贴回 plcsp/env/due_dates.py 的 {table_name} ──")
    print(f"{table_name}: dict[str, tuple[float, float]] = {{")
    for name, (tau, rr) in rows.items():
        print(f'    "{name}": ({tau:.2f}, {rr:.1f}),')
    print("}")


if __name__ == "__main__":
    main()
