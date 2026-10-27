# -*- coding: utf-8 -*-
"""MA 头的 logits 熵：判断「塌缩」是权重 one-hot，还是 argmax 恰好恒定。

对每个 MA 决策，重放当时的 tok/seg，取三个分类头的 softmax：

- 熵 → 0 ⟹ 策略输出近乎 one-hot；
- 熵高但 argmax 不变 ⟹ 有不确定性，但偏好一直指向同一个组合。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

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
H_MAX = {"PA": float(np.log(12)), "LA": float(np.log(3)), "RA": float(np.log(3))}


def _specs():
    out = []
    for i, name in enumerate(MK_NAMES):
        sp, _ = build(seed=0, instance=name, tau=None, pm_interval=30,
                      target_failures=4, n_agv=N_AGV, finite_buffer=True,
                      layout_seed=200 + i, reward_norm="minmax")
        out.append((name, sp))
    return out


def _collect(net, specs):
    rec = {k: {"h": [], "top1": []} for k in ("PA", "LA", "RA")}
    for _name, sp in specs:
        for k in range(N_SC):
            p = NetPolicy(net, sp.inst, sp.layout, sp.ctx, greedy=True,
                          seed=0, dt_events=DT, seed_chain=SC0 + k)
            sp.world().run_gated(
                seed_chain=SC0 + k, policy_pa=p.policy_pa,
                policy_l=p.policy_l, policy_m=p.policy_m,
                policy_ma=p.policy_ma, dt_events=DT)
            for d in p.rec.decisions:
                if d.kind != "MA":
                    continue
                t = torch.as_tensor(d.tok).unsqueeze(0)
                with torch.no_grad():
                    o = net(t, d.seg, hard=True)
                for key, lg in o.rule_logits.items():
                    prob = torch.softmax(lg[0], dim=-1).clamp_min(1e-12)
                    rec[key]["h"].append(float(-(prob * prob.log()).sum()))
                    rec[key]["top1"].append(float(prob.max()))
    return rec


def _report(tag, rec):
    print(f"\n══ {tag} ══", flush=True)
    for key in ("PA", "LA", "RA"):
        h = np.asarray(rec[key]["h"], dtype=float)
        p = np.asarray(rec[key]["top1"], dtype=float)
        if not len(h):
            print(f"  {key}: 无 MA 决策", flush=True)
            continue
        print(f"  {key}  head（{len(h)} 个 MA 决策，H_max={H_MAX[key]:.2f}）", flush=True)
        print(f"    熵: mean {h.mean():.3f}  min {h.min():.3f}  max {h.max():.3f}"
              f"  归一化均值 {h.mean() / H_MAX[key]:.1%}", flush=True)
        print(f"    top-1 概率: mean {p.mean():.3f}  min {p.min():.3f}  "
              f"max {p.max():.3f}", flush=True)
        print(f"    top-1 > 0.99 的比例: {np.mean(p > 0.99):.1%}；"
              f"> 0.90: {np.mean(p > 0.90):.1%}", flush=True)


def main() -> None:
    specs = _specs()
    print("**MA logits 熵** · MK01–10 · layout 200+i · 场景 1–2 · greedy",
          flush=True)
    for tag, ck in CKPTS.items():
        if not os.path.exists(ck):
            print(f"\n  ⚠️ 缺 ckpt {ck}", flush=True)
            continue
        obj = torch.load(ck, map_location="cpu", weights_only=False)
        net = HierarchicalPolicy()
        net.load_state_dict(obj["state_dict"])
        net.eval()
        _report(tag, _collect(net, specs))


if __name__ == "__main__":
    main()
