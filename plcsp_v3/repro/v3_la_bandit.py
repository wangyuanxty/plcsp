# -*- coding: utf-8 -*-
"""**LA 的三臂老虎机**（最小复现）：3 条派车规则 · 整局回报 · PA 固定 EAJ。

先写真值（3 条规则各跑整局），再训 780 个参数的 LA 头，看它收敛到哪一条。

**3 个类、整局标量回报 —— 这是能想出来的最小分类问题。**
它收敛到错的那条 ⟹ 不是「信号太弱」，是有东西真的错了。
"""
import sys
from pathlib import Path

R = Path(r"D:\research\DeepReinforcementLearningScheduling")
sys.path.insert(0, str(R))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from plcsp_v3.env.reward import objective_vector, scalar_reward  # noqa: E402
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, apply_pa_rule,  # noqa: E402
                                la_rule_policy, pm_rule_policy)
from plcsp_v3.nn.state_emb import build_tok  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

PA_FIX, RA_FIX, THR = 3, 0, 30.0          # PA = EAJ+SMPT
N_LA = len(LA_RULE_NAMES)
SCR = 100_000


def episode(spec, la: int, sc: int):
    box: list = []

    def policy_pa(view, pool, tie_rng=None):
        if not box:
            tok, _ = build_tok(view.snap, spec.inst, spec.layout, spec.ctx)
            box.append(tok)
        return apply_pa_rule(PA_FIX, view, pool, tie_rng)

    m = spec.world().run_gated(seed_chain=sc, policy_pa=policy_pa,
                               policy_l=la_rule_policy(la),
                               policy_m=pm_rule_policy(RA_FIX, THR), dt_events=0)
    f = objective_vector(m)
    return box[0], f, scalar_reward(f, spec.w, bounds=spec.bounds)


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    torch.manual_seed(0)
    spec, net = build(seed=0, pm_interval=30, target_failures=4,
                      finite_buffer=True, layout_seed=0, reward_norm="minmax")

    print("═══ 真值：3 条 LA 规则各跑整局（PA=EAJ+SMPT 固定）═══", flush=True)
    for sc in (SCR, SCR + 1, SCR + 2):
        row = []
        for la in range(N_LA):
            _t, f, r = episode(spec, la, sc)
            row.append(f"{LA_RULE_NAMES[la]}={f[0]:6.2f}")
        print(f"  world {sc}: " + "   ".join(row), flush=True)
    print()

    for p in net.parameters():
        p.requires_grad_(False)
    for p in net.select.heads["LA"].parameters():
        p.requires_grad_(True)
    optim = torch.optim.Adam(list(net.select.heads["LA"].parameters()), lr=1e-2)

    tok, _f, _r = episode(spec, 0, SCR)
    seg = (int(spec.inst.n_machines), int(spec.inst.n_jobs), int(spec.ctx.n_agv), 1)
    t = torch.as_tensor(tok).unsqueeze(0)
    print(f"═══ 训 LA 头（{sum(p.numel() for p in net.select.heads['LA'].parameters())}"
          f" 参数 · lr=1e-2 · {N_LA} 臂 · 整局回报）═══", flush=True)
    print(f"{'step':>5} | " + "  ".join(f"p({n[:5]})" for n in LA_RULE_NAMES)
          + " | argmax", flush=True)

    for step in range(1, steps + 1):
        sc = SCR + step
        lg = net(t, seg, hard=True).rule_logits["LA"][0]
        logp_all = torch.log_softmax(lg, dim=-1)
        Rs = np.array([episode(spec, la, sc)[2] for la in range(N_LA)])
        adv = (Rs - Rs.mean()) / (Rs.std() + 1e-6)
        optim.zero_grad(set_to_none=True)
        loss = -(logp_all * torch.as_tensor(adv, dtype=torch.float32)).sum()
        loss.backward()
        optim.step()
        if step % max(1, steps // 10) == 0 or step == 1:
            with torch.no_grad():
                lg = net(t, seg, hard=True).rule_logits["LA"][0]
                p = torch.softmax(lg, dim=-1)
            print(f"{step:>5} | " + "  ".join(f"{float(x):.4f}" for x in p)
                  + f" | {LA_RULE_NAMES[int(lg.argmax())]}", flush=True)


if __name__ == "__main__":
    main()
