# -*- coding: utf-8 -*-
"""**最小可训模型**：一次决策 · 12 路选择 · 整局回报。

```
开局（t=0 快照）── net.select.heads["PA"] ──► 一条 PA 规则 a
                                             a 固定跑完整局（LA/RA 用固定规则）
                              reward = scalar_reward(objective_vector(m), w, bounds)
loss = −Σ_a Â_a · logp(a)          Â = 组内 z 化
```

12 臂老虎机，12×64+12 = 780 个参数。**它必须收敛到平均最好的那条规则。**
真值（同配置、整局、纯规则）：`EAJ+EAM` / `EAJ+SMPT` 最好。

**学不会 ⟹ loss / 优化器 / token 接线里有 bug。**
"""
import sys
from pathlib import Path

R = Path(r"D:\research\DeepReinforcementLearningScheduling")
sys.path.insert(0, str(R))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from plcsp_v3.env.reward import objective_vector, scalar_reward  # noqa: E402
from plcsp_v3.hrl.rules import (PA_RULE_NAMES, apply_pa_rule,  # noqa: E402
                                la_rule_policy, pm_rule_policy)
from plcsp_v3.nn.state_emb import build_tok  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

N_RULES = 12
SCR = 100_000


def episode(spec, a: int, sc: int):
    """开局的状态 + 用规则 `a` 跑完整局的回报。"""
    box: list = []

    def policy_pa(view, pool, tie_rng=None):
        if not box:
            tok, _ = build_tok(view.snap, spec.inst, spec.layout, spec.ctx)
            box.append(tok)
        return apply_pa_rule(a, view, pool, tie_rng)

    m = spec.world().run_gated(seed_chain=sc, policy_pa=policy_pa,
                               policy_l=la_rule_policy(0),
                               policy_m=pm_rule_policy(0, 30.0), dt_events=0)
    return box[0], scalar_reward(objective_vector(m), spec.w, bounds=spec.bounds)


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    torch.manual_seed(0)
    spec, net = build(seed=0, pm_interval=30, target_failures=4,
                      finite_buffer=True, layout_seed=0, reward_norm="minmax")
    for p in net.parameters():
        p.requires_grad_(False)
    for p in net.select.heads["PA"].parameters():
        p.requires_grad_(True)
    optim = torch.optim.Adam(list(net.select.heads["PA"].parameters()), lr=1e-3)

    # 开局状态与规则、与世界都无关（t=0 什么都没发生）⟹ 跑一局把它取出来即可
    tok, _r0 = episode(spec, 0, SCR)
    seg = (int(spec.inst.n_machines), int(spec.inst.n_jobs), int(spec.ctx.n_agv), 1)
    t = torch.as_tensor(tok).unsqueeze(0)
    print(f"开局 tok {tok.shape} · seg={seg} · 12 臂 · 只训 "
          f"{sum(p.numel() for p in net.select.heads['PA'].parameters())} 个参数",
          flush=True)
    print(f"{'step':>4} | {'argmax':<12} {'p(argmax)':>9} | {'最佳规则':<12} "
          f"{'其整局 R':>9} | {'本轮 R 极差':>10}", flush=True)

    for step in range(1, steps + 1):
        sc = SCR + step
        # ⚠️ **带梯度算一次**（12 臂共用同一次前向，不是每臂各算一次）
        lg = net(t, seg, hard=True).rule_logits["PA"][0]
        logp_all = torch.log_softmax(lg, dim=-1)
        Rs = [episode(spec, a, sc)[1] for a in range(N_RULES)]
        Rs_np = np.array(Rs)
        adv = (Rs_np - Rs_np.mean()) / (Rs_np.std() + 1e-6)
        optim.zero_grad(set_to_none=True)
        loss = -(logp_all * torch.as_tensor(adv, dtype=torch.float32)).sum()
        loss.backward()
        optim.step()
        if step % max(1, steps // 10) == 0 or step == 1:
            with torch.no_grad():
                lg = net(t, seg, hard=True).rule_logits["PA"][0]
                p = torch.softmax(lg, dim=-1)
            ai = int(lg.argmax())
            best = int(np.argmax(Rs_np))
            print(f"{step:>4} | {PA_RULE_NAMES[ai]:<12} {float(p[ai]):9.4f} | "
                  f"{PA_RULE_NAMES[best]:<12} {Rs_np[best]:9.4f} | "
                  f"{Rs_np.max() - Rs_np.min():10.4f}", flush=True)

    print()
    with torch.no_grad():
        lg = net(t, seg, hard=True).rule_logits["PA"][0]
        order = np.argsort(-lg.numpy())
    print("训练后 12 条规则的概率（降序）：")
    p = torch.softmax(lg, dim=-1).numpy()
    for a in order[:5]:
        print(f"  {PA_RULE_NAMES[a]:<12} p={p[a]:.4f}  logit={float(lg[a]):+.4f}")


if __name__ == "__main__":
    main()
