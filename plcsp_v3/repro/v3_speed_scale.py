# -*- coding: utf-8 -*-
"""**运输时间的尺度是不是「常数最优」的原因** —— 一个变量：`agv_speed_mps`。

══ 要判的 ══

若「规则之间的区分度」与 AGV 速度无关 ⟹ **运输尺度不是原因**，那条候选划掉。
若速度放慢后区分度显著变大 ⟹ 我原来的尺度**选小了**（原文 `tt_ll'` 没给数，是我自造的）。

⚠️ **只改 `agv_speed_mps`**（`sim_overrides`），其余全不动。
⚠️ 速度一变，`build()` 里的参考运行（36 条 CDR 扫 `objective_bounds`）也会跟着变 —— 那是**应当的**。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # 仓根

import numpy as np  # noqa: E402

from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.rules import (PA_RULE_NAMES, apply_pa_rule,  # noqa: E402
                                la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402

BASE_SPEED = 0.5          # `SimConfig.agv_speed_mps` 默认
N_SC, N_RULE = 20, 9


def main() -> None:
    inst = named(140, 16)
    scs = [100_000 + 13 * i for i in range(N_SC)]
    print("Exp(16/140) · 9 条规则 × 20 场景 · **只改 agv_speed_mps**\n")
    print(f"  {'倍率':>5}{'车速':>9}{'单程':>8}{'运输/加工':>10}"
          f"{'规则极差':>10}{'每场景余量':>12}{'AGV忙时':>9}")
    print("  " + "-" * 66)

    for f in (1.0, 3.0, 10.0):
        sp = BASE_SPEED / f
        spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                        finite_buffer=True, layout_seed=0, reward_norm="minmax",
                        tau=2.70, sim_overrides={"due_range": 0.6, "agv_speed_mps": sp},
                        inst_obj=inst)
        M = np.zeros((N_RULE, N_SC))
        busy, mk = 0.0, 0.0
        for k in range(N_RULE):
            for i, sc in enumerate(scs):
                m = spec.world().run_gated(
                    seed_chain=sc,
                    policy_pa=lambda v, p, tr, k=k: apply_pa_rule(k, v, p, tr),
                    policy_l=la_rule_policy(0), dt_events=0)
                M[k, i] = float(m["makespan"])
                if k == 0:
                    b = m["energy_breakdown"]["agv_states_min"]
                    busy += float(b["empty"]) + float(b["loaded"])
                    mk += float(m["makespan"])
        glob = M.mean(axis=1)
        g = int(glob.argmin())
        spread = (glob.max() - glob.min()) / glob[g]
        head = (glob[g] - M.min(axis=0).mean()) / glob[g]
        # 单程：总行驶分钟 / (2 × 搬运次数)，搬运 ≈ 工序数 + 工件数（含回站）
        n_legs = 2 * (sum(len(o) for o in inst.jobs) + inst.n_jobs)
        led = busy / n_legs
        pt = float(np.mean([t for ops in inst.jobs for alts in ops for _m, t in alts]))
        print(f"  {f:>5.0f}×{sp:>9.3f}{led:>8.2f}{led / pt:>10.1%}"
              f"{spread:>10.1%}{head:>12.2%}{busy / (N_SC * mk):>9.1%}")
        print(f"        最好 CDR {PA_RULE_NAMES[g]}={glob[g]:.1f}"
              f"　最差 {PA_RULE_NAMES[int(glob.argmax())]}={glob.max():.1f}")

    print("\n  ⟹ 「规则极差」= 最好 CDR与最差常数之差 / 最好 CDR")
    print("  ⟹ 「每场景余量」= 每场景事后选最好的均值，相对全局一条最好 CDR")
    print("  ⟹ 两者若不随倍率显著变化 ⟹ **运输尺度不是原因**")


if __name__ == "__main__":
    main()
