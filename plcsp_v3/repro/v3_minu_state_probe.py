# -*- coding: utf-8 -*-
"""**逐状态检查：策略选 `EAJ+MINU` 是合理还是错？**

对象：`hdrl3_pa_msfixed_s5.pt`（固定尺度 MS · logit 5 · 2–4 · 新特征）。

做法（单点偏离）：

1. 用策略跑一条链，记录每个 PA 决策点的 greedy 规则；
2. 对策略选了 `EAJ+SMPT` 或 `EAJ+MINU` 的点 `t`：
   - 前缀照策略跑；
   - 第 `t` 点强制成 `EAJ+SMPT` / `EAJ+MINU`；
   - 之后继续用策略。
3. 比较两条反事实的**最终 MS**。

判据：

- 策略选 MINU 时，MINU 的最终 MS 更低 ⟹ 状态依赖，合理；
- SMPT 更低 ⟹ 策略在该点选错。
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
from plcsp_v3.env.shi_instances import eval_instance  # noqa: E402
from plcsp_v3.hrl.rules import (PA_RULE_NAMES, apply_pa_rule,  # noqa: E402
                                la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402

CKPT = os.environ.get("MSP_CKPT", os.path.join(
    os.environ.get("TEMP", "/tmp"), "hdrl3_pa_msfixed_s5.pt"))
N_SC = int(os.environ.get("MSP_SCEN", "10"))
SMPT = PA_RULE_NAMES.index("EAJ+SMPT")
MINU = PA_RULE_NAMES.index("EAJ+MINU")
KEEP = (SMPT, MINU)


class Override:
    """前 `t` 个 PA 决策照策略；第 `t` 个强制成 `k`；之后继续照策略。"""

    def __init__(self, r, t: int, k: int) -> None:
        self.r, self.t, self.k, self.n = r, t, k, 0

    def policy_pa(self, view, pool, tie_rng=None):
        i = self.n
        self.n += 1
        if i == self.t:
            return apply_pa_rule(self.k, view, pool, tie_rng)
        return self.r.policy_pa(view, pool, tie_rng)


def main() -> None:
    if not os.path.exists(CKPT):
        raise SystemExit(f"⚠️ 找不到 ckpt `{CKPT}`")
    inst = eval_instance(140, 16, 20, **N._inst_kw())
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N.NAGV,
                    finite_buffer=True, layout_seed=0, reward_norm="none",
                    tau=2.70, sim_overrides=N._sim_over(), inst_obj=inst,
                    layout_scheme="random", layout_cols=N.LAYOUT_COLS,
                    layout_rows=N.LAYOUT_ROWS, shi2025=N.SHI,
                    cell_w=N.CELL or None, cell_h=N.CELL or None,
                    aisle_w=N.AISLE or None, lu_connector_m=N.LU or None)
    nf = NHFeatures(inst, spec.layout,
                    speed_m_per_min=spec.cfg.agv_speed_mps * 60.0)
    net = N.NHDRL()
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    net.load_state_dict(obj["state_dict"] if "state_dict" in obj else obj)
    feat_stats = obj.get("feat_stats") if isinstance(obj, dict) else None
    net.eval()

    scs = [100_000 + 13 * i for i in range(N_SC)]
    acc = {k: {"n": 0, "better": 0, "worse": 0, "tie": 0, "regret": []}
           for k in KEEP}
    base_ms = []
    print(f"**逐状态：策略 vs 单点换规则** · ckpt `{CKPT}` · {N_SC} 场景\n",
          flush=True)
    for sc in scs:
        r0 = N.Rollout(net, spec, nf, N.RewardFn(),
                       torch.Generator().manual_seed(0), greedy=True,
                       feat_stats=feat_stats, force_oa=0, force_la=0)
        w0 = spec.world()
        m0 = w0.run_gated(seed_chain=sc, policy_pa=r0.policy_pa,
                          policy_l=la_rule_policy(0), dt_events=0)
        base_ms.append(float(m0["makespan"]))
        acts = [int(b[1]) for b in r0.buf["pa"]]
        n_use = 0
        for t, a in enumerate(acts):
            if a not in KEEP:
                continue
            n_use += 1
            ms = {}
            for k in KEEP:
                r = N.Rollout(net, spec, nf, N.RewardFn(),
                              torch.Generator().manual_seed(0), greedy=True,
                              feat_stats=feat_stats, force_oa=0, force_la=0)
                ov = Override(r, t, k)
                m = spec.world().run_gated(seed_chain=sc, policy_pa=ov.policy_pa,
                                           policy_l=la_rule_policy(0),
                                           dt_events=0)
                ms[k] = float(m["makespan"])
            best = min(ms, key=ms.get)
            d = acc[a]
            d["n"] += 1
            d["regret"].append(ms[a] - ms[best])
            if ms[a] == ms[best]:
                d["better" if a == best else "tie"] += 1
            else:
                d["worse"] += 1
        print(f"  sc={sc} 决策 {len(acts)} · 参与 {n_use} · "
              f"base MS {base_ms[-1]:.1f}", flush=True)

    print(f"\n  base 策略 MS 均值 {np.mean(base_ms):.1f}\n")
    for k in KEEP:
        d = acc[k]
        n = max(1, d["n"])
        print(f"  {PA_RULE_NAMES[k]} 被选 {d['n']} 次：")
        print(f"    它更好 {d['better'] / n:.1%}　更差 {d['worse'] / n:.1%}"
              f"　并列 {d['tie'] / n:.1%}")
        print(f"    平均后悔 {np.mean(d['regret']):.2f} min"
              f"　最大 {np.max(d['regret']):.1f} min")
    print("\n  ⚠️ 单点偏离：只改第 t 点，之后仍用策略。不是全局最优。")


if __name__ == "__main__":
    main()
