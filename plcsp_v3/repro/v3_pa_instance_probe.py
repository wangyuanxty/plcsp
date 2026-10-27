# -*- coding: utf-8 -*-
"""**同一规模、不同实例** —— PA 的动作分布会不会随实例变？

固定 16 工件，换 6 组加工时间/类型指派。
OA / LA 强制 0/0，只看 PA。

判据：
- 6 个实例的 PA 分布**相同** ⟹ PA 不看实例状态（常数策略）；
- 分布**随实例变** ⟹ 条件化成立，问题在选错规则。
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
from plcsp_v3.hrl.rules import PA_RULE_NAMES  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

CKPT = os.environ.get("PI_CKPT", os.path.join(
    os.environ.get("TEMP", "/tmp"), "nhdrl3_iso_vary.pt"))
SIZE = int(os.environ.get("PI_SIZE", "16"))
N_INST = int(os.environ.get("PI_NINST", "6"))
N_RUN = int(os.environ.get("PI_RUN", "5"))


def main() -> None:
    if not os.path.exists(CKPT):
        raise SystemExit(f"⚠️ 找不到 ckpt `{CKPT}`")
    net = N.NHDRL()
    _obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(_obj, dict) and "state_dict" in _obj:
        net.load_state_dict(_obj["state_dict"])
        feat_stats = _obj.get("feat_stats")
    else:
        net.load_state_dict(_obj)
        feat_stats = None
    net.eval()
    print(f"**同规模不同实例** · {SIZE} 工件 · {N_INST} 实例 · "
          f"{N_RUN} 场景/实例\n", flush=True)
    print(f"  {'实例':>4}{'真工序':>7}{'平均MS':>9}  PA 分布（greedy）")
    print("  " + "-" * 72)
    for s in range(N_INST):
        inst = make_shi_instance(
            SIZE, seed=POOL_STRIDE * (3000 + s),
            rot_seed=PT_REDRAW_SEED + 3000 + s,
            alts_dist="uniform", alts_min=N.ALTS_MIN, alts_max=N.ALTS_MAX,
            pt_dist="uniform", pt_lo=2, pt_hi=10)
        spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N.NAGV,
                        finite_buffer=True, layout_seed=0, reward_norm="none",
                        tau=2.70, sim_overrides=N._sim_over(), inst_obj=inst,
                        layout_scheme="random", **dims_from_shop_len(200.0, inst.n_machines, layout_cols=N.LAYOUT_COLS),
                        layout_cols=N.LAYOUT_COLS, layout_rows=N.LAYOUT_ROWS,
                        shi2025=N.SHI)
        nf = NHFeatures(inst, spec.layout,
                        speed_m_per_min=spec.cfg.agv_speed_mps * 60.0)
        cnt: Counter = Counter()
        mss = []
        for k in range(N_RUN):
            r = N.Rollout(net, spec, nf, N.RewardFn(),
                          torch.Generator().manual_seed(0), greedy=True,
                          feat_stats=feat_stats, force_oa=0, force_la=0)
            w = spec.world()
            m = w.run_gated(seed_chain=100_000 + 13 * k,
                            policy_pa=r.policy_pa, policy_l=r.policy_l,
                            dt_events=0)
            r.finish(w.snapshot(), float(m["makespan"]))
            for b in r.buf["pa"]:
                cnt[b[1]] += 1
            mss.append(float(m["makespan"]))
        n = max(1, sum(cnt.values()))
        top = "　".join(f"{PA_RULE_NAMES[k]}:{v / n:.0%}"
                        for k, v in cnt.most_common(3))
        print(f"  {s:>4}{sum(len(o) for o in inst.jobs):>7}"
              f"{np.mean(mss):>9.1f}  {top}", flush=True)
    print("\n  ⚠️ OA/LA 强制 0/0，只看 PA。")
    print("  ⚠️ 若 6 个实例分布相同 ⟹ PA 不看实例状态。")


if __name__ == "__main__":
    main()
