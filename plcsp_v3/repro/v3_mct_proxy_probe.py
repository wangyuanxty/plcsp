# -*- coding: utf-8 -*-
"""**`cm_of` 代理是否与真 MS 同序** —— 27 条 CDR 的「最终 MCT」vs「真 makespan」。

奖励里的 `cm_of(snap) = max(snap.mct_assigned)`（机台完工），
不含运输/回站。真 MS 是 `m["makespan"]`。

逐规模比较两条排序：
```
true MS 最好规则
final MCT 最好规则
```
若两者不同 ⟹ **奖励在优化错的目标**。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import v3_nhdrl as N  # noqa: E402
from plcsp_v3.env.shi_instances import (POOL_STRIDE, PT_REDRAW_SEED,  # noqa: E402
                                        make_shi_instance)
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, PA_RULE_NAMES,  # noqa: E402
                                apply_pa_rule, la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

SIZES = [int(x) for x in os.environ.get("MP_SIZES", "10,16,40").split(",")]


def main() -> None:
    for size in SIZES:
        inst = make_shi_instance(
            size, seed=POOL_STRIDE * (2000 + size),
            rot_seed=PT_REDRAW_SEED + 2000 + size,
            alts_dist="uniform", alts_min=N.ALTS_MIN, alts_max=N.ALTS_MAX,
            pt_dist="uniform", pt_lo=2, pt_hi=10)
        spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=6,
                        finite_buffer=True, layout_seed=0, reward_norm="none",
                        tau=2.70, sim_overrides=N._sim_over(), inst_obj=inst,
                        layout_scheme="random", **dims_from_shop_len(200.0, inst.n_machines, layout_cols=N.LAYOUT_COLS),
                        layout_cols=N.LAYOUT_COLS, layout_rows=N.LAYOUT_ROWS,
                        shi2025=N.SHI)
        rows = []
        for pk in range(9):
            for lj in range(len(LA_RULE_NAMES)):
                w = spec.world()
                m = w.run_gated(
                    seed_chain=100_000,
                    policy_pa=lambda v, p, tr=None, _k=pk: apply_pa_rule(
                        _k, v, p, tr),
                    policy_l=la_rule_policy(lj), dt_events=0)
                snap = w.snapshot()
                mct = max((float(x) for x in snap.mct_assigned), default=0.0)
                rows.append((PA_RULE_NAMES[pk] + "+" + LA_RULE_NAMES[lj],
                             float(m["makespan"]), mct,
                             float(m["logistics_cost"])))
        by_ms = sorted(rows, key=lambda r: r[1])
        by_mct = sorted(rows, key=lambda r: r[2])
        rho = np.corrcoef([r[1] for r in rows], [r[2] for r in rows])[0, 1]
        print(f"\n**{size} 工件 · {sum(len(o) for o in inst.jobs)} 真工序**"
              f" · corr(true MS, final MCT) = {rho:.3f}", flush=True)
        print(f"  {'真 MS 前 3':<32}{'final MCT 前 3':<32}")
        print("  " + "-" * 66)
        for a, b in zip(by_ms[:3], by_mct[:3]):
            print(f"  {a[0]:<28}{a[1]:>7.1f}   "
                  f"{b[0]:<28}{b[2]:>7.1f}", flush=True)
        print(f"  ⟹ 真 MS 最好 **{by_ms[0][0]}** · "
              f"MCT 最好 **{by_mct[0][0]}**"
              f"　{'✅ 同一条' if by_ms[0][0] == by_mct[0][0] else '❌ 不同'}",
              flush=True)


if __name__ == "__main__":
    main()
