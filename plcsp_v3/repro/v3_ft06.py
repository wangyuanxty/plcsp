# -*- coding: utf-8 -*-
"""**我的仿真跑 ft06**（零运输矩阵）—— 与 JobShopLab 的 `fifo 68 / spt 88 / mwkr 83` 对照。

实例来源：`jobshoplab/data/jssp_instances/spec_files/ft06`（OR-Library 标准格式）。
**已知最优 = 55**（同一目录 `solutions/ft06_sol` 是完整调度，可复核）。

退化配置（让两边语义尽量一致）：
- 运输 = **全零矩阵**（瞬时运输 ⟹ AGV 车型差异消失）
- 缓冲 = **无界**（`finite_buffer=False`，对照 JobShopLab 的 `capacity=2^63-1`）
- 无预防性保养、无故障
"""
import os
import sys

sys.path.insert(0, r"D:\research\DeepReinforcementLearningScheduling")

import numpy as np  # noqa: E402

from plcsp_v3.env.instances import Instance  # noqa: E402
from plcsp_v3.hrl.rules import PA_RULE_NAMES, apply_pa_rule, la_rule_policy  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

ORLIB = r"D:\research\DeepReinforcementLearningScheduling\jobshoplab\data\jssp_instances\spec_files\ft06"
SOL = r"D:\research\DeepReinforcementLearningScheduling\jobshoplab\data\jssp_instances\spec_files\solutions\ft06_sol"


def load_orlib(path: str) -> Instance:
    """OR-Library JSSP 文本 → `Instance`。每道工序**只有一台候选机**（= 经典 JSSP）。"""
    toks: list[str] = []
    for line in open(path, encoding="utf-8"):
        if line.lstrip().startswith("#"):
            continue
        toks += line.split()
    n, m = int(toks[0]), int(toks[1])
    idx = 2
    jobs = []
    for _ in range(n):
        ops = []
        for _ in range(m):
            mac, dur = int(toks[idx]), float(toks[idx + 1])
            idx += 2
            ops.append([(mac, dur)])
        jobs.append(ops)
    return Instance(n_jobs=n, n_machines=m, jobs=jobs, source=path)


def opt_makespan(path: str, inst: Instance) -> float | None:
    """`*_sol` 是每工序的开工时刻表 —— **但它的列序与实例工序序对不上**（见下），故只作参考。

    ⚠️ **实测：按同一列序读，j1 的第 3 道工序在 t=0 开工、第 1 道在 t=40 ⟹ 违反前后道约束。**
    ⟹ 该文件的列语义未知，**不得当作最优解的依据**。ft06 的已知最优 55 另见
    JobShopLab README 的对照表（`lb` 列），本脚本只**引用**它，未自行验证。
    """
    try:
        S = np.loadtxt(path)
    except Exception:
        return None
    if S.shape[0] != inst.n_jobs:
        return None
    bad = 0
    for j, ops in enumerate(inst.jobs):
        for oi in range(1, len(ops)):
            if S[j, oi] < S[j, oi - 1]:
                bad += 1
    return float("nan") if bad else float(
        max(S[j, oi] + ops[oi][0][1] for j, ops in enumerate(inst.jobs)
            for oi in range(len(ops))))


def main() -> None:
    inst = load_orlib(ORLIB)
    n = inst.n_jobs
    chk = opt_makespan(SOL, inst)
    print(f"ft06：{n} 作业 × {inst.n_machines} 机")
    print("  已知最优 = **55**（引自 JobShopLab README 的 `lb` 列；本脚本**未**自行验证）")
    print(f"  `solutions/ft06_sol` 自查：{'列序与工序序不符，弃用' if chk != chk else f'给出 {chk:.0f}'}")
    print("  工序（机, 时长）：")
    for j, ops in enumerate(inst.jobs):
        print(f"    j{j}: " + " ".join(f"({a[0][0]},{a[0][1]:.0f})" for a in ops))

    full = np.zeros((inst.n_machines + 1, inst.n_machines + 1), dtype=float)
    inst.transport = "matrix"
    inst.trans_time_full = full

    n_agv = int(os.environ.get("V3_AGV", "6"))
    spec, _ = build(seed=0, pm_interval=None, target_failures=0, n_agv=n_agv,
                    finite_buffer=False, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    print(f"\n  我的仿真：{n_agv} AGV · 零运输矩阵 · 无界缓冲 · 无保养/故障\n")
    print(f"  {'PA 规则':<12}{'makespan':>10}")
    print("  " + "-" * 24)
    for k in range(12):
        m = spec.world().run_gated(seed_chain=0,
                                   policy_pa=lambda v, p, tr, k=k: apply_pa_rule(k, v, p, None),
                                   policy_l=la_rule_policy(0), dt_events=0)
        print(f"  {PA_RULE_NAMES[k]:<12}{float(m['makespan']):>10.1f}")
    print("\n  ⟹ **合法性判据：任何 makespan 必须 ≥ 55**（低于 ⟹ 仿真算错）")
    print("  ⟹ JobShopLab 同实例零运输：fifo 68 · spt 88 · mwkr 83")


if __name__ == "__main__":
    main()
