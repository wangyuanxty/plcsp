# -*- coding: utf-8 -*-
"""**最小模型 · 第二层**：规则在每个 ΔT 点各选一次，**回报仍是整局**。

与 `v3_bandit.py` 只差一件事：决策从 1 次变成每个顶层点 1 次。
与真实 `phase_a` 只差一件事：**回报是整局，不是段**。

```
跑 N 条链（每条：每个 ΔT 采一条 PA 规则）
  R_i = scalar_reward(objective_vector(m_i), w, bounds)         ← **整局**
  Â_i = 组内 z 化
  loss = −Σ_i Â_i · Σ_j logp(a_{i,j})                           ← 链级 score
```

**若它收敛到 EAJ 系 ⟹ 段回报是唯一的病。**
**若它也退化   ⟹ 病在「一条链多个决策共用一个标量」。**
"""
import sys
from pathlib import Path

R = Path(r"D:\research\DeepReinforcementLearningScheduling")
sys.path.insert(0, str(R))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from plcsp_v3.env.reward import objective_vector, scalar_reward  # noqa: E402
from plcsp_v3.hrl import train as T  # noqa: E402
from plcsp_v3.hrl.rules import PA_RULE_NAMES  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

N_CHAIN, N_RULES, SCR = 8, 12, 100_000


def ma_logp(net, d) -> torch.Tensor:
    """一个 MA 决策的 logp（**带梯度**）：三个头的 `log_softmax` 之和。"""
    t = torch.as_tensor(d.tok).unsqueeze(0)
    o = net(t, d.seg, hard=True)
    return sum(torch.log_softmax(o.rule_logits[k][0], dim=-1)[int(r)]
               for k, r in (("PA", d.rule_pa), ("LA", d.rule_la), ("RA", d.rule_ra)))


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    dt = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    torch.manual_seed(0)
    spec, net = build(seed=0, pm_interval=30, target_failures=4,
                      finite_buffer=True, layout_seed=0, reward_norm="minmax")
    for p in net.parameters():
        p.requires_grad_(False)
    for p in net.select.parameters():
        p.requires_grad_(True)
    optim = torch.optim.Adam(list(net.select.parameters()), lr=1e-3)

    def policy_argmax(net, sc):
        """一条 greedy 链上，每个 MA 点选了什么（看状态有没有用）。"""
        b = T.base_chain(net, spec, greedy=True, seed=0, dt_events=dt, seed_chain=sc)
        mas = [d for d in b.rec.decisions if d.kind == "MA"]
        seq = [d.rule_pa for d in mas]
        las = sorted({d.rule_la for d in mas})
        ras = sorted({d.rule_ra for d in mas})
        return seq, las, ras, float(b.metrics["makespan"])

    print(f"N_CHAIN={N_CHAIN} · dt_events={dt} · 只训 net.select · **整局回报**",
          flush=True)
    print(f"{'step':>4} | {'PA 序列':<26} | {'LA 用到':<8} {'RA 用到':<8} | "
          f"{'评测 ms':>8} | {'R̄':>8}", flush=True)

    for step in range(1, steps + 1):
        chains = []
        for i in range(N_CHAIN):
            p = T.base_chain(net, spec, greedy=False, seed=step * 100 + i,
                             dt_events=dt, seed_chain=SCR + step * 100 + i)
            r = scalar_reward(objective_vector(p.metrics), spec.w, bounds=spec.bounds)
            chains.append((r, p))
        Rs = np.array([r for r, _ in chains])
        adv = (Rs - Rs.mean()) / (Rs.std() + 1e-6)
        optim.zero_grad(set_to_none=True)
        loss = torch.zeros(())
        n = 0
        for a, (_r, p) in zip(adv, chains):
            for d in p.rec.decisions:
                if d.kind == "MA":
                    loss = loss - float(a) * ma_logp(net, d)
                    n += 1
        (loss / max(n, 1)).backward()
        optim.step()
        if step % max(1, steps // 8) == 0 or step == 1:
            seq, las, ras, ms = policy_argmax(net, SCR)
            txt = "".join(PA_RULE_NAMES[a].split("+")[0][:2] for a in seq)
            print(f"{step:>4} | {txt:<26} | {str(las):<8} {str(ras):<8} | "
                  f"{ms:8.2f} | {Rs.mean():+8.4f}", flush=True)

    print()
    seq, las, ras, ms = policy_argmax(net, SCR)
    print(f"训练后 greedy 链：PA 序列 = {seq}")
    print(f"                  LA 用到 = {las}   RA 用到 = {ras}")
    print(f"                  ms = {ms:.2f}")


if __name__ == "__main__":
    main()
