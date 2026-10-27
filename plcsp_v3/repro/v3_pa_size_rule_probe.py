# -*- coding: utf-8 -*-
"""**PA 为什么在变规模训练里选不出 EAJ** —— 同一实例上，策略 vs 9 条固定 PA。

对每个规模：

1. 用**同一条实例、同一组场景**跑 PA-only 策略（greedy，OA=MS、LA=MINCOST）；
2. 跑 9 条固定 PA（LA=MINCOST）；
3. 比较：策略 MS、策略最常用规则、该规则的固定 MS、本规模最好规则。

判据：

- 若 EAJ 只在某些规模最好，而策略在其他规模用 SPT/SRPT
  ⟹ **平均最优 ≠ 逐规模最优**，不是“训不出来”。
- 若 EAJ 在每个规模都最好，而策略仍不选
  ⟹ 训练/探索问题。
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import v3_nhdrl as N  # noqa: E402
from plcsp_v3.env.nhdrl_features import NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import (POOL_STRIDE, PT_REDRAW_SEED,  # noqa: E402
                                        make_shi_instance)
from plcsp_v3.hrl.rules import (N_AEI2025_RULES, PA_RULE_NAMES,  # noqa: E402
                                apply_pa_rule, la_rule_policy)
from plcsp_v3.q1_flat import build, dims_from_shop_len  # noqa: E402

CKPT = os.environ.get("PSR_CKPT", os.path.join(
    os.environ.get("TEMP", "/tmp"), "hdrl3_pa_ms_20_15.pt"))
SIZES = [int(x) for x in os.environ.get(
    "PSR_SIZES", "10,16,22,28,34,40").split(",")]
N_RUN = int(os.environ.get("PSR_RUN", "5"))


def build_spec(inst):
    kw = dict(seed=0, pm_interval=30, target_failures=4, n_agv=N.NAGV,
              finite_buffer=True, layout_seed=0, reward_norm="none",
              tau=2.70, sim_overrides=N._sim_over(), inst_obj=inst,
              layout_scheme="random",
              **dims_from_shop_len(200.0, inst.n_machines,
                                   layout_cols=N.LAYOUT_COLS),
              layout_rows=N.LAYOUT_ROWS, shi2025=N.SHI)
    if N.CELL > 0.0:
        kw.update(cell_w=N.CELL, cell_h=N.CELL, aisle_w=N.AISLE,
                  lu_connector_m=N.LU)
    return build(**kw)[0]


def main() -> None:
    if not os.path.exists(CKPT):
        raise SystemExit(f"⚠️ 找不到 ckpt `{CKPT}`")
    net = N.NHDRL()
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        net.load_state_dict(obj["state_dict"])
        feat_stats = obj.get("feat_stats")
    else:
        net.load_state_dict(obj)
        feat_stats = None
    net.eval()
    print(f"**PA 逐规模：策略 vs 9 条固定 PA** · ckpt `{CKPT}`\n", flush=True)
    print("  " + " " * 18
          + "".join(f"{n:>12}" for n in PA_RULE_NAMES[:N_AEI2025_RULES]))
    for size in SIZES:
        inst = make_shi_instance(
            size, seed=POOL_STRIDE * (2000 + size),
            rot_seed=PT_REDRAW_SEED + 2000 + size,
            alts_dist="uniform", alts_min=N.ALTS_MIN, alts_max=N.ALTS_MAX,
            pt_dist="uniform", pt_lo=2, pt_hi=10)
        spec = build_spec(inst)
        nf = NHFeatures(inst, spec.layout,
                        speed_m_per_min=spec.cfg.agv_speed_mps * 60.0)
        scs = [100_000 + 13 * i for i in range(N_RUN)]
        # 9 条固定 PA（LA = MINCOST）
        rule_ms = []
        for k in range(9):
            vals = [float(spec.world().run_gated(
                seed_chain=sc,
                policy_pa=lambda v, p, tr=None, _k=k: apply_pa_rule(
                    _k, v, p, tr),
                policy_l=la_rule_policy(0), dt_events=0)["makespan"])
                for sc in scs]
            rule_ms.append(float(np.mean(vals)))
        # 策略
        cnt: Counter = Counter()
        pol_ms = []
        for sc in scs:
            r = N.Rollout(net, spec, nf, N.RewardFn(),
                          torch.Generator().manual_seed(0), greedy=True,
                          feat_stats=feat_stats, force_oa=0, force_la=0)
            w = spec.world()
            m = w.run_gated(seed_chain=sc, policy_pa=r.policy_pa,
                            policy_l=r.policy_l, dt_events=0)
            r.finish(w.snapshot(), float(m["makespan"]))
            for b in r.buf["pa"]:
                cnt[b[1]] += 1
            pol_ms.append(float(m["makespan"]))
        pms = float(np.mean(pol_ms))
        top_k = cnt.most_common(1)[0][0]
        best_k = int(np.argmin(rule_ms))
        print(f"  {size:>3} 工件 策略 MS {pms:>7.1f}"
              f"  最常用 {PA_RULE_NAMES[top_k]:<12}"
              f"（它的固定 MS {rule_ms[top_k]:>7.1f}）"
              f"  本规模最好 {PA_RULE_NAMES[best_k]:<12}"
              f"（{rule_ms[best_k]:>7.1f}）"
              f"  差 {pms - rule_ms[best_k]:>+7.1f}", flush=True)
        print("  " + " " * 10
              + "".join(f"{x:>12.1f}" for x in rule_ms), flush=True)
    print("\n  ⚠️ OA=MS、LA=MINCOST，策略与固定规则同场景。")
    print("  ⚠️ 若 EAJ 只在部分规模最好 ⟹ 变规模训练的最优是折中，不是 bug。")


if __name__ == "__main__":
    main()
