# -*- coding: utf-8 -*-
"""交期紧度 k 的校准（bug#12 修复后重做）。

背景：原交期 `d_j = 1.8 × P_j`（P_j = 作业自身总加工时间）**完全没考虑排队/运输/争用**——
实测官方 MK01 上 makespan=405 而交期只有 16–40，**tardy 恒等于 10/10，目标退化无信号**。

改用**参考调度校准**的 TWK 形式：
    d_j = r_j + k · P_j        r_j = 0（静态实例）
k 由"参考调度下目标误期率"反解。本脚本扫描 k，报告误期率曲线。

参考调度 = DES 默认口径：每工序取最短候选机台 + AGV 轮询派车（`rollout(op_choices=None)`）。

用法：
    python -m plcsp.m9_due_calib                 # 三实例 × 多 k
    python -m plcsp.m9_due_calib --seeds 20
输出：stdout 表格（不写文件）
"""
from __future__ import annotations

import argparse
import statistics as st

from .env.des import SimConfig, rollout
from .env.instances import load_mk

INSTANCES = ("mk01", "mk07", "mk10")
K_GRID = (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 80.0)
TAU_GRID = (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10)
C_GRID = (1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0, 4.0, 5.0, 6.0)
TARGET_LO, TARGET_HI = 0.20, 0.40          # "有信号但不全崩"的目标区间


def shortest_candidate_workload(inst) -> list[float]:
    """P_j = 作业 j 在最短候选口径下的总加工时间。"""
    return [sum(min(t for _, t in op) for op in job) for job in inst.jobs]


def sweep(inst_name: str, seeds: int) -> list[tuple[float, float, float]]:
    """返回 [(k, 误期率均值, 误期率标准差), ...]。"""
    inst = load_mk(inst_name)
    p_j = shortest_candidate_workload(inst)
    cfg = SimConfig()

    # 参考调度：跑 seeds 次，收集每作业完工时刻
    completes_per_seed: list[list[float]] = []
    makespans: list[float] = []
    for s in range(seeds):
        r = rollout(inst, seed_chain=s, cfg=cfg)
        if r["horizon_hit"] or r["jobs_done"] != inst.n_jobs:
            continue                        # 掐表/未完成 → 丢弃，不污染校准
        completes_per_seed.append([r["completes"][j] for j in range(inst.n_jobs)])
        makespans.append(r["makespan"])
    if not completes_per_seed:
        raise RuntimeError(f"{inst_name}: 参考调度无有效样本（全部掐表/未完成）")

    rows: list[tuple[float, float, float]] = []
    for k in K_GRID:
        due = [k * p for p in p_j]
        rates = [
            sum(1 for j in range(inst.n_jobs) if c[j] > due[j]) / inst.n_jobs
            for c in completes_per_seed
        ]
        rows.append((k, st.mean(rates), st.pstdev(rates)))
    print(f"[{inst_name}] n={inst.n_jobs}x{inst.n_machines}  "
          f"P_j∈[{min(p_j):.0f},{max(p_j):.0f}]  "
          f"makespan={st.mean(makespans):.1f}±{st.pstdev(makespans):.1f}  "
          f"有效样本 {len(completes_per_seed)}/{seeds}")
    return rows


def sweep_tau(inst_name: str, seeds: int) -> list[tuple[float, float, float]]:
    """交期 = τ · M_ref（参考工期比例形式）。返回 [(τ, 误期率均值, sd), ...]。

    与 TWK 形式（k·P_j）的差别：τ 直接可读（"交期 = 参考工期的百分之几"），
    且不依赖实例的工时尺度 → 有望跨实例统一。
    """
    inst = load_mk(inst_name)
    cfg = SimConfig()
    completes_per_seed: list[list[float]] = []
    m_refs: list[float] = []
    for s in range(seeds):
        r = rollout(inst, seed_chain=s, cfg=cfg)
        if r["horizon_hit"] or r["jobs_done"] != inst.n_jobs:
            continue
        completes_per_seed.append([r["completes"][j] for j in range(inst.n_jobs)])
        m_refs.append(r["makespan"])
    if not completes_per_seed:
        raise RuntimeError(f"{inst_name}: 参考调度无有效样本")
    m_ref = st.mean(m_refs)

    rows: list[tuple[float, float, float]] = []
    for tau in TAU_GRID:
        due = tau * m_ref
        rates = [
            sum(1 for c in cs if c > due) / inst.n_jobs
            for cs in completes_per_seed
        ]
        rows.append((tau, st.mean(rates), st.pstdev(rates)))
    print(f"[{inst_name}] M_ref={m_ref:.1f}  "
          f"完工时刻∈[{min(min(c) for c in completes_per_seed):.0f},"
          f"{max(max(c) for c in completes_per_seed):.0f}]")
    return rows


def sweep_twk_delivery(inst_name: str, seeds: int) -> list[tuple[float, float, float]]:
    """形式 C：d_j = c · (TPT_j + TDT_j)  —— **AEI 103216 原文用的 TWK 形式**。

    出处（原文逐字）："This paper adopts the Total Work Content (TWK) method [61], and defines
    the due date of job as d_i = a_i + c·(TPT_i + TDT_i), where c is the due date tightness
    (c = 1.5), TPT_i is the total processing time, and TDT_i is the total delivery time."
    —— 静态实例 a_i = 0。

    与形式 A（k·P_j）的差别：**计入运输时间 TDT**。我们的旧公式 d = 1.8·P_j 丢掉了 TDT，
    这正是"交期 16–40 而 makespan 405"的根因。

    TDT_j 近似 = 沿参考计划相邻工序的 dock 距离之和 ÷ 有效车速（不叠加区段等待）。
    """
    from .env.corridors import build_corridor_graph, dock_distance_matrix
    from .env.layout import sample_layout

    inst = load_mk(inst_name)
    cfg = SimConfig()
    n_m = inst.n_machines
    lay = sample_layout(n_m, "line", 1)
    dm = dock_distance_matrix(build_corridor_graph(lay))

    # 参考计划：每工序取最短候选 → 该作业经过的机台序列
    tpt, tdt = [], []
    for job in inst.jobs:
        chosen = [min(op, key=lambda x: x[1])[0] for op in job]      # 每工序最短候选机台
        tpt.append(sum(min(t for _, t in op) for op in job))
        hop = 0.0
        for a, b in zip(chosen, chosen[1:]):
            hop += float(dm[a, b]) / cfg.eff_speed / cfg.agv_speed_mps / 60.0   # [m]→[min]
        tdt.append(hop)

    completes_per_seed: list[list[float]] = []
    for s in range(seeds):
        r = rollout(inst, seed_chain=s, cfg=cfg)
        if r["horizon_hit"] or r["jobs_done"] != inst.n_jobs:
            continue
        completes_per_seed.append([r["completes"][j] for j in range(inst.n_jobs)])
    if not completes_per_seed:
        raise RuntimeError(f"{inst_name}: 参考调度无有效样本")

    rows: list[tuple[float, float, float]] = []
    for c in C_GRID:
        due = [c * (tpt[j] + tdt[j]) for j in range(inst.n_jobs)]
        rates = [sum(1 for j in range(inst.n_jobs) if cs[j] > due[j]) / inst.n_jobs
                 for cs in completes_per_seed]
        rows.append((c, st.mean(rates), st.pstdev(rates)))
    print(f"[{inst_name}] TPT∈[{min(tpt):.0f},{max(tpt):.0f}]  "
          f"TDT∈[{min(tdt):.0f},{max(tdt):.0f}]  "
          f"TPT+TDT∈[{min(a+b for a,b in zip(tpt,tdt)):.0f},"
          f"{max(a+b for a,b in zip(tpt,tdt)):.0f}]")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--mode", choices=["k", "tau", "twk", "both", "all"], default="all")
    args = ap.parse_args()

    if args.mode in ("k", "both", "all"):
        print("=" * 62)
        print("形式 A：d_j = k · P_j   （TWK，但**不含运输**）")
        print("=" * 62)
        _report(lambda n: sweep(n, args.seeds), "k")
        print()

    if args.mode in ("tau", "both", "all"):
        print("=" * 62)
        print("形式 B：d_j = τ · M_ref  （参考工期比例）")
        print("=" * 62)
        _report(lambda n: sweep_tau(n, args.seeds), "τ")
        print()

    if args.mode in ("twk", "both", "all"):
        print("=" * 62)
        print("形式 C：d_j = c · (TPT_j + TDT_j)   （AEI 103216 原文形式，**含运输**）")
        print("=" * 62)
        _report(lambda n: sweep_twk_delivery(n, args.seeds), "c")
        print()


def _report(sweep_fn, label: str) -> None:
    for name in INSTANCES:
        rows = sweep_fn(name)
        print(f"    {label:>7} {'误期率':>9} {'sd':>7}   判定")
        for x, mean, sd in rows:
            if mean < TARGET_LO:
                tag = "过松（信号弱）"
            elif mean > TARGET_HI:
                tag = "过紧（近全崩）"
            else:
                tag = "*** 目标区间 ***"
            print(f"    {x:>7.2f} {mean:>9.3f} {sd:>7.3f}   {tag}")
        print()


if __name__ == "__main__":
    main()
