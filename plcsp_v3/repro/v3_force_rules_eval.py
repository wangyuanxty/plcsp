# -*- coding: utf-8 -*-
"""不训练定位：把 MA 的规则强制成最好 CDR 的规则，看差距在哪。

同一组 10 个留出随机实例、场景 1–2：

| 配置 | MA 规则 | RA |
|---|---|---|
| own | 模型自己选 | 学出来的 |
| forced + rule RA | `EAJ+SMPT + LATEST_IN_TIME` | 规则驱动 |
| forced + learned RA | 同上 | 学出来的 |
| forced + top1 + rule RA | 同上，绕过打分头 | 规则驱动 |
| CDR | 固定 `EAJ+SMPT + LATEST_IN_TIME` | 规则驱动 |

若 forced 接近 CDR ⟹ 差距在 MA 的规则选择（主要是 LA）。
若 forced 仍差 ⟹ 差距在下层打分头 / 网络。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from plcsp_v3.env.instances import gen_mk_like  # noqa: E402
from plcsp_v3.env.reward import objective_vector  # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy  # noqa: E402
from plcsp_v3.hrl.cdr import run_cdr  # noqa: E402
from plcsp_v3.hrl.rollout import NetPolicy  # noqa: E402
from plcsp_v3.hrl.rules import LA_RULE_NAMES, PA_RULE_NAMES, PM_RULE_NAMES  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

CKPT = os.environ.get(
    "V3_DIAG_CKPT",
    os.path.join(os.environ.get("TEMP", ""), "v3_span3_200_seed0.pt"))
N_VAL = 10
VAL_SEED = 5000
LAYOUT_BASE = 500
N_AGV = 3
DT = 20
BEST_PA = "EAJ+SMPT"
BEST_LA = "LATEST_IN_TIME"
RA_RULE = "QUEUE_EMPTY"


def _specs():
    out = []
    for i in range(N_VAL):
        inst = gen_mk_like(seed=VAL_SEED + i)
        sp, _ = build(seed=0, inst_obj=inst, tau=2.0, pm_interval=30,
                      target_failures=4, n_agv=N_AGV, finite_buffer=True,
                      layout_seed=LAYOUT_BASE + i, reward_norm="minmax",
                      sim_overrides={"due_range": 0.6})
        out.append((f"val{i}", sp))
    return out


def _net():
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = HierarchicalPolicy()
    net.load_state_dict(obj["state_dict"])
    net.eval()
    return net


def _run(net, sp, sc, *, rules=None, ma=False, ra=True, top1=False):
    p = NetPolicy(net, sp.inst, sp.layout, sp.ctx, rules=rules, greedy=True,
                  seed=0, dt_events=DT, seed_chain=sc,
                  freeze_rules=rules is not None, force_top1=top1)
    kw = {"policy_m": p.policy_m} if ra else {}
    if ma:
        kw["policy_ma"] = p.policy_ma
    m = sp.world().run_gated(seed_chain=sc, policy_pa=p.policy_pa,
                             policy_l=p.policy_l, dt_events=DT, **kw)
    return objective_vector(m)


def main() -> None:
    net = _net()
    specs = _specs()
    forced = {"PA": PA_RULE_NAMES.index(BEST_PA),
              "LA": LA_RULE_NAMES.index(BEST_LA),
              "RA": PM_RULE_NAMES.index(RA_RULE)}
    cfgs = {
        "own (MA free)": dict(ma=True, ra=True),
        "own (MA free) + top1": dict(ma=True, ra=True, top1=True),
        "forced + rule RA": dict(rules=forced, ra=False),
        "forced + learned RA": dict(rules=forced, ra=True),
        "forced + top1 + rule RA": dict(rules=forced, ra=False, top1=True),
    }
    res = {}
    for name, kw in cfgs.items():
        vals = []
        for _n, sp in specs:
            for sc in (1, 2):
                vals.append(_run(net, sp, sc, **kw))
        res[name] = np.asarray(vals, dtype=float).mean(0)
    cdr = []
    for _n, sp in specs:
        for sc in (1, 2):
            cdr.append(objective_vector(run_cdr(
                sp.world(), BEST_PA, BEST_LA, seed_chain=sc)))
    res[f"CDR {BEST_PA}+{BEST_LA}"] = np.asarray(cdr, dtype=float).mean(0)

    print(f"ckpt = {CKPT}", flush=True)
    print(f"**强制规则定位** · {N_VAL} 留出实例 · 场景 1–2 · "
          f"forced = {BEST_PA} + {BEST_LA} + {RA_RULE}\n", flush=True)
    print(f"  {'config':<30}{'MS':>9}{'E':>8}{'TWT':>10}", flush=True)
    print("  " + "-" * 58, flush=True)
    for k, v in res.items():
        print(f"  {k:<30}{v[0]:>9.1f}{v[1]:>8.2f}{v[2]:>10.1f}", flush=True)


if __name__ == "__main__":
    main()
