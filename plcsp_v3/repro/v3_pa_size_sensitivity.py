# -*- coding: utf-8 -*-
"""**只改规模特征，看 PA 的 logits 动不动**。

PA 的 24 维里，规模相关的是：
```
索引 7  = f16 总工序数 Σ(Ji+1)
索引 9  = f18 剩余工序数
索引 10 = f19 未完成工件数
```
做法：
1. 在各规模上跑一局，取 PA 标准化特征的均值；
2. 以 16 工件的均值为底，只替换索引 7/9/10；
3. 看 PA `Tanh` 前的 logits 与 argmax 是否变化。
"""
import os
import sys

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

CKPT = os.environ.get("PZ_CKPT", os.path.join(
    os.environ.get("TEMP", "/tmp"), "nhdrl3_iso_vary.pt"))
SIZES = [int(x) for x in os.environ.get("PZ_SIZES", "10,16,22,28,34,40").split(",")]
SIZE_IDX = [7, 9, 10]


def build_spec(size: int):
    inst = make_shi_instance(
        size, seed=POOL_STRIDE * (2000 + size),
        rot_seed=PT_REDRAW_SEED + 2000 + size,
        alts_dist="uniform", alts_min=N.ALTS_MIN, alts_max=N.ALTS_MAX,
        pt_dist="uniform", pt_lo=2, pt_hi=10)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N.NAGV,
                    finite_buffer=True, layout_seed=0, reward_norm="none",
                    tau=2.70, sim_overrides=N._sim_over(), inst_obj=inst,
                    layout_scheme="random", **dims_from_shop_len(200.0, inst.n_machines, layout_cols=N.LAYOUT_COLS),
                    layout_cols=N.LAYOUT_COLS, layout_rows=N.LAYOUT_ROWS,
                    shi2025=N.SHI)
    return spec, NHFeatures(inst, spec.layout,
                            speed_m_per_min=spec.cfg.agv_speed_mps * 60.0)


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

    means = {}
    for size in SIZES:
        spec, nf = build_spec(size)
        r = N.Rollout(net, spec, nf, N.RewardFn(),
                      torch.Generator().manual_seed(0), greedy=True,
                      feat_stats=feat_stats, force_oa=0, force_la=0)
        w = spec.world()
        w.run_gated(seed_chain=100_000, policy_pa=r.policy_pa,
                    policy_l=r.policy_l, dt_events=0)
        F = np.stack([b[0] for b in r.buf["pa"]])
        means[size] = F.mean(axis=0)
        print(f"  规模 {size:>2}：PA 特征 {F.shape[0]} 点", flush=True)

    base = means[16].copy()
    print(f"\n  {'规模':>4}{'f16/f18/f19（标准化后）':>28}{'argmax':>20}"
          f"{'logits 变化 L2':>15}")
    print("  " + "-" * 70)
    def logits(f):
        with torch.no_grad():
            x = torch.as_tensor(f[None], dtype=torch.float32)
            return net.pa.actor[2](net.pa.actor[1](net.pa.actor[0](x)))[0].numpy()

    base_log = logits(base)
    for size in SIZES:
        h = base.copy()
        h[SIZE_IDX] = means[size][SIZE_IDX]
        lg = logits(h)
        print(f"  {size:>4}{np.array2string(h[SIZE_IDX], precision=2):>28}"
              f"{PA_RULE_NAMES[int(np.argmax(lg))]:>20}"
              f"{np.linalg.norm(lg - base_log):>15.3f}", flush=True)
    print("\n  ⚠️ 底 = 16 工件的均值；只替换 f16/f18/f19。")
    print("  ⚠️ L2 ≈ 0 且 argmax 不变 ⟹ PA 不看规模特征。")


if __name__ == "__main__":
    main()
