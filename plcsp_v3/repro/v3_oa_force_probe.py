# -*- coding: utf-8 -*-
"""**强制 OA 选 MS / TC** —— 判断 OA 的 TC 偏好是不是「对的」。

同一 PA/LA 策略、同一实例、同一 20 场景：

```
unforced  → OA 按训练后的策略
force_ms  → OA 恒选 MS
force_tc  → OA 恒选 TC
```

判据：
- 如果 force_ms 的最终 MS 明显更好、TC 也可接受 ⟹ **OA 学偏了**；
- 如果 force_ms 更差 ⟹ **不是 OA 的问题，是 PA/LA 或奖励口径的问题**。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import v3_nhdrl as N  # noqa: E402
from plcsp_v3.env.nhdrl_features import NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import eval_instance  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

CKPT = os.environ.get("OF_CKPT", os.path.join(
    os.environ.get("TEMP", "/tmp"), "nhdrl3_vary.pt"))
N_RUN = int(os.environ.get("OF_RUN", "20"))


class ConstantActor(nn.Module):
    """固定输出两个 logits —— 用来把 OA 钉死在 MS（0）或 TC（1）。"""

    def __init__(self, pick: int) -> None:
        super().__init__()
        v = [10.0, -10.0] if pick == 0 else [-10.0, 10.0]
        self.register_buffer("v", torch.tensor([v]))

    def forward(self, f):
        return self.v.expand(f.shape[0], -1)


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

    inst = eval_instance(140, 16, 20, **N._inst_kw())
    spec = build(seed=0, pm_interval=30, target_failures=4, n_agv=N.NAGV,
                 finite_buffer=True, layout_seed=0, reward_norm="none",
                 tau=2.70, sim_overrides=N._sim_over(), inst_obj=inst,
                 layout_scheme="random", **dims_from_shop_len(200.0, inst.n_machines, layout_cols=N.LAYOUT_COLS),
                 layout_cols=N.LAYOUT_COLS, layout_rows=N.LAYOUT_ROWS,
                 shi2025=N.SHI)[0]
    nf = NHFeatures(inst, spec.layout,
                    speed_m_per_min=spec.cfg.agv_speed_mps * 60.0)
    scs = [100_000 + 13 * i for i in range(N_RUN)]
    orig = net.oa.actor

    print(f"**强制 OA** · {N_RUN} 场景 · `Exp(16/140)`\n", flush=True)
    print(f"  {'模式':<10}{'MS 均值':>10}{'MS 最小':>10}{'TC 均值':>10}{'TC 最小':>10}")
    print("  " + "-" * 50)
    for tag, pick in (("unforced", None), ("force_ms", 0), ("force_tc", 1)):
        if pick is None:
            net.oa.actor = orig
        else:
            net.oa.actor = ConstantActor(pick).to(
                next(net.parameters()).device)
        ms, tc = [], []
        for sc in scs:
            r = N.Rollout(net, spec, nf, N.RewardFn(),
                          torch.Generator().manual_seed(0), eps=0.0,
                          feat_stats=feat_stats)
            w = spec.world()
            m = w.run_gated(seed_chain=sc, policy_pa=r.policy_pa,
                            policy_l=r.policy_l, dt_events=0)
            ms.append(float(m["makespan"]))
            tc.append(float(m["logistics_cost"]))
        print(f"  {tag:<10}{np.mean(ms):>10.1f}{np.min(ms):>10.1f}"
              f"{np.mean(tc):>10.1f}{np.min(tc):>10.1f}", flush=True)
    net.oa.actor = orig
    print("\n  ⚠️ 强制只改 OA；PA/LA 仍是同一套权重。")
    print("  ⚠️ 该实例上 27 CDR 最好 MS ≈ 276.2（EAJ+SMPT+MINCOST）。")


if __name__ == "__main__":
    main()
