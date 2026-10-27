# -*- coding: utf-8 -*-
"""打分头消融 + RA 隔离（不训练）。

四组，同 ckpt、同 MK01–10、同 layout 200+i、同场景 1–2：

| # | scorer | RA |
|---|---|---|
| A | top-K（打分头） | 学出来的 `policy_m` |
| B | top-1（规则首选） | 学出来的 `policy_m` |
| C | top-K | 规则驱动（`policy_m=None`） |
| D | top-1 | 规则驱动 |

A−B = 打分头贡献；A−C = 学出来的 RA 贡献。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from plcsp_v3.env.reward import objective_vector  # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy  # noqa: E402
from plcsp_v3.hrl.rollout import NetPolicy  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

MK_NAMES = tuple(f"mk{i:02d}" for i in range(1, 11))
N_SC = 2
SC0 = 1
N_AGV = 3
DT = 20
_CUSTOM = os.environ.get("V3_DIAG_CKPT", "")
CKPTS = ({"custom": _CUSTOM} if _CUSTOM else {
    "final": os.path.join(os.environ.get("TEMP", ""), "v3_pilot500_seed0.pt"),
    "peak375": os.path.join(os.environ.get("TEMP", ""),
                            "v3_pilot500_seed0_e0375.pt"),
})


def _specs():
    out = []
    for i, name in enumerate(MK_NAMES):
        sp, _ = build(seed=0, instance=name, tau=None, pm_interval=30,
                      target_failures=4, n_agv=N_AGV, finite_buffer=True,
                      layout_seed=200 + i, reward_norm="minmax")
        out.append((name, sp))
    return out


def _run(net, specs, force_top1: bool, rule_ra: bool):
    vals = []
    for _name, sp in specs:
        for k in range(N_SC):
            p = NetPolicy(net, sp.inst, sp.layout, sp.ctx, greedy=True,
                          seed=0, dt_events=DT, seed_chain=SC0 + k,
                          force_top1=force_top1)
            kw = {} if rule_ra else {"policy_m": p.policy_m}
            m = sp.world().run_gated(
                seed_chain=SC0 + k, policy_pa=p.policy_pa,
                policy_l=p.policy_l, policy_ma=p.policy_ma,
                dt_events=DT, **kw)
            vals.append(objective_vector(m))
    return np.asarray(vals, dtype=float).mean(axis=0)


def main() -> None:
    specs = _specs()
    print("**打分头消融 + RA 隔离** · MK01–10 · layout 200+i · 场景 1–2\n",
          flush=True)
    print(f"  {'ckpt':<8}{'scorer':<8}{'RA':<9}{'MS':>8}{'E':>8}{'TWT':>10}",
          flush=True)
    print("  " + "-" * 52, flush=True)
    for tag, ck in CKPTS.items():
        if not os.path.exists(ck):
            print(f"  ⚠️ 缺 ckpt {ck}", flush=True)
            continue
        obj = torch.load(ck, map_location="cpu", weights_only=False)
        net = HierarchicalPolicy()
        net.load_state_dict(obj["state_dict"])
        net.eval()
        rows = {}
        for scorer, ft1 in (("topK", False), ("top1", True)):
            for ra, rule_ra in (("learned", False), ("rule", True)):
                v = _run(net, specs, ft1, rule_ra)
                rows[(scorer, ra)] = v
                print(f"  {tag:<8}{scorer:<8}{ra:<9}{v[0]:>8.1f}{v[1]:>8.2f}"
                      f"{v[2]:>10.1f}", flush=True)
        a, b = rows[("topK", "learned")], rows[("top1", "learned")]
        c, d = rows[("topK", "rule")], rows[("top1", "rule")]
        print(f"  {tag} · 打分头贡献（topK−top1，learned RA）: "
              f"MS {a[0] - b[0]:+.1f}  E {a[1] - b[1]:+.2f}  "
              f"TWT {a[2] - b[2]:+.1f}", flush=True)
        print(f"  {tag} · RA 贡献（learned−rule，topK）: "
              f"MS {a[0] - c[0]:+.1f}  E {a[1] - c[1]:+.2f}  "
              f"TWT {a[2] - c[2]:+.1f}", flush=True)
        print(f"  {tag} · 打分头贡献（topK−top1，rule RA）: "
              f"MS {c[0] - d[0]:+.1f}  E {c[1] - d[1]:+.2f}  "
              f"TWT {c[2] - d[2]:+.1f}\n", flush=True)


if __name__ == "__main__":
    main()
