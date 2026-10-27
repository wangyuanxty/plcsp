# -*- coding: utf-8 -*-
"""**PA 是否按规模条件化** —— 同一个 ckpt，在 10/16/22/28/34/40 上跑 greedy PA。

OA / LA 强制固定（0 / 0），只看 PA 的动作分布。

判据：
- 各规模 PA 分布**相同** ⟹ 策略没有用规模特征；
- 分布**随规模变** ⟹ 条件化成立，问题在选错了规则。
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
from plcsp_v3.q1_flat import build, dims_from_shop_len  # noqa: E402

CKPT = os.environ.get("PS_CKPT", os.path.join(
    os.environ.get("TEMP", "/tmp"), "nhdrl3_iso_vary.pt"))
SIZES = [int(x) for x in os.environ.get("PS_SIZES", "10,16,22,28,34,40").split(",")]
N_RUN = int(os.environ.get("PS_RUN", "5"))


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
    print(f"**PA 按规模条件化探针** · ckpt `{CKPT}` · {N_RUN} 场景/规模\n",
          flush=True)
    print(f"  {'工件':>4}{'真工序':>7}{'平均MS':>9}  PA 分布（greedy）· `Tanh` 前幅度")
    print("  " + "-" * 72)
    for size in SIZES:
        inst = make_shi_instance(
            size, seed=POOL_STRIDE * (2000 + size),
            rot_seed=PT_REDRAW_SEED + 2000 + size,
            alts_dist="uniform", alts_min=N.ALTS_MIN, alts_max=N.ALTS_MAX,
            pt_dist="uniform", pt_lo=2, pt_hi=10)
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
        spec, _ = build(**kw)
        nf = NHFeatures(inst, spec.layout,
                        speed_m_per_min=spec.cfg.agv_speed_mps * 60.0)
        cnt: Counter = Counter()
        mss = []
        pre_all: list[np.ndarray] = []
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
            if r.buf["pa"]:
                F = torch.as_tensor(np.stack([b[0] for b in r.buf["pa"]]),
                                    dtype=torch.float32)
                with torch.no_grad():
                    pre = net.pa.actor[2](
                        net.pa.actor[1](net.pa.actor[0](F))).numpy()
                pre_all.append(pre)
        n = max(1, sum(cnt.values()))
        top = "　".join(f"{PA_RULE_NAMES[k]}:{v / n:.0%}"
                        for k, v in cnt.most_common(4))
        pre = np.concatenate(pre_all) if pre_all else np.zeros(1)
        print(f"  {size:>4}{sum(len(o) for o in inst.jobs):>7}"
              f"{np.mean(mss):>9.1f}  {top}"
              f"　|pre|中位 {np.median(np.abs(pre)):.2f}"
              f" ≥3:{np.mean(np.abs(pre) >= 3):.0%}", flush=True)
    print("\n  ⚠️ OA/LA 强制 0/0，只看 PA。")
    print("  ⚠️ 若各规模分布相同 ⟹ PA 没有用规模特征。")


if __name__ == "__main__":
    main()
