# -*- coding: utf-8 -*-
"""**基线栏**（照原文口径）：RN + **27 条 CDRs** → Pareto 解集 → SP / HV。

原文协议（`docs/citation-cards.md:1633-1680`）：

> - **27 条复合派工规则（CDRs）** —— 由 **PA 的 9 个动作与 LA 的 3 个动作两两组合**生成
> - **RN**（Random version of NHDRL）—— **每个决策点随机为每个 agent 选动作**
> - 比较协议：每实例各方法独立运行 **20 次** → Pareto 支配 → 比较 HV 与 SP

⚠️ **本脚本修正了上一版的两处口径错误**（上一版只跑 9 条 CDR、RN 只随机 PA）：
  ① `CDRs` 由 9 条改为 **27 条**（PA 9 × LA 3）
  ② `RN` 由"只随机 PA"改为 **PA 与 LA 都随机**

⚠️ **OA 的随机在本基线里是空操作**：基线是**纯规则**（不训练），OA 只决定"优化 MS 还是
TC"，而在无学习的情形下它不改变任何执行动作。**原文的 RN 是"NHDRL 不训练"，其 OA 随机才
有含义；本档的 RN 是"规则随机"，故 OA 栏不适用。这一条要写明。**
"""
import os
import sys

sys.path.insert(0, r"D:\research\DeepReinforcementLearningScheduling")

import numpy as np  # noqa: E402

from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.mo_metrics import hypervolume_2d, pareto_front, spacing  # noqa: E402
from plcsp_v3.hrl.rules import (PA_RULE_NAMES, apply_pa_rule,  # noqa: E402
                                la_rule_policy, pm_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402

C_COMMISSION, UC_TRANSPORT = 7.0, 0.5
N_RUNS = int(os.environ.get("V3_NRUNS", "20"))
N_PA, N_LA = 9, 3                      # 原文的 Rule 1–9 与 LA 的 3 条


def tc_of(m: dict, n_agv: int) -> float:
    st = m["energy_breakdown"]["agv_states_min"]
    return C_COMMISSION * n_agv + UC_TRANSPORT * (float(st["empty"]) + float(st["loaded"]))


def run_pt(spec, sc: int, pa: int | None, la: int | None, seed: int) -> tuple[float, float]:
    """一局 → (MS, TC)。`pa is None` ⟹ **PA 与 LA 都随机**（原文的 RN）。"""
    if pa is None:
        gen = np.random.default_rng(seed)

        def pol(view, pool, tie_rng=None):
            return apply_pa_rule(int(gen.integers(N_PA)), view, pool, tie_rng)

        def pol_l(view, cand_v):
            return int(la_rule_policy(int(gen.integers(N_LA)), (0.0,))(view, cand_v))
    else:
        def pol(view, pool, tie_rng=None):
            return apply_pa_rule(int(pa), view, pool, tie_rng)
        pol_l = la_rule_policy(int(la))

    m = spec.world().run_gated(seed_chain=sc, policy_pa=pol, policy_l=pol_l,
                               policy_m=pm_rule_policy(0, 30.0), dt_events=0)
    return float(m["makespan"]), tc_of(m, N_AGV_SHI)


def main() -> None:
    a, b = int(sys.argv[1]), int(sys.argv[2])       # Exp(a/b)
    inst = named(b, a)
    real = sum(len(o) for o in inst.jobs)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    scs = [100_000 + 13 * i for i in range(N_RUNS)]
    print(f"Exp({a}/{b}) · 真工序 {real} · 18 机 · {N_AGV_SHI} AGV · "
          f"{N_RUNS} 次 × (RN + {N_PA * N_LA} 条 CDR)\n", flush=True)

    pts: dict[str, list] = {}
    pts["RN"] = [run_pt(spec, sc, None, None, 7000 + i) for i, sc in enumerate(scs)]
    print(f"  RN（PA+LA 都随机）  完成　"
          f"MS {min(p[0] for p in pts['RN']):.1f}–{max(p[0] for p in pts['RN']):.1f}",
          flush=True)
    for pa in range(N_PA):
        for la in range(N_LA):
            nm = f"CDR-{pa * N_LA + la + 1}"
            pts[nm] = [run_pt(spec, sc, pa, la, 0) for sc in scs]
        print(f"  PA={PA_RULE_NAMES[pa]:<11} × {N_LA} 条 LA  完成", flush=True)

    allp = np.array([p for v in pts.values() for p in v])
    ref = tuple(allp.max(0) * 1.1)
    print(f"\n  参考点（所有点最大值 ×1.1）= ({ref[0]:.1f}, {ref[1]:.1f})\n")
    print(f"  {'方法':<22}{'前沿点':>7}{'MS 范围':>17}{'TC 范围':>15}{'HV':>10}{'SP':>9}")
    print("  " + "-" * 82)
    best = None
    for name, P in pts.items():
        F = pareto_front(P)
        hv, sp = hypervolume_2d(F, ref), spacing(F)
        if best is None or hv > best[1]:
            best = (name, hv)
        print(f"  {name:<22}{len(F):>7}"
              f"{f'{F[:, 0].min():.1f}–{F[:, 0].max():.1f}':>17}"
              f"{f'{F[:, 1].min():.1f}–{F[:, 1].max():.1f}':>15}"
              f"{hv:>10.1f}{sp:>9.4f}", flush=True)
    print(f"\n  ⟹ HV 最高：**{best[0]}** = {best[1]:.1f}")
    print("  ⟹ `CDRs` 基线取 27 条里的最好那条（原文：表中只列表现最好的 3 条）")


if __name__ == "__main__":
    main()
