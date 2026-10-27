# -*- coding: utf-8 -*-
"""**只变一处：决策点落在一个固定状态，还是落在变化的状态上。**

两臂共用**同一套**：3 路机台规则（工件规则固定 `EAJ`）· 整局回报 · 组内 z 化 ·
只训 `net.select.heads["PA"]` · 同 lr · 同步数 · 同 5 个评测世界。

```
--once   决策只有 1 次，状态 = t=0 开局快照   （每条链输入**完全相同** —— 已知能训对）
--each   决策在每个顶层点各 1 次，状态**随 ΔT 变化**
```

**若 `--once` 对而 `--each` 错 ⟹ 病在「输入变化」那一环**（编码器 / 池化 / 状态不可分）。
"""
import sys

sys.path.insert(0, r"D:\research\DeepReinforcementLearningScheduling")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from plcsp_v3.env.reward import objective_vector, scalar_reward  # noqa: E402
from plcsp_v3.hrl.rules import (PA_MACHINE_RULES, PA_RULE_NAMES,  # noqa: E402
                                apply_pa_rule, la_rule_policy, pm_rule_policy)
from plcsp_v3.nn.state_emb import build_tok  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

LA_FIX, RA_FIX, THR = 0, 0, 30.0
EVAL_W = list(range(500_000, 500_005))
EAJ = next(i for i, n in enumerate(PA_RULE_NAMES) if n.startswith("EAJ+"))
ARMS3 = [EAJ + m for m in range(len(PA_MACHINE_RULES))]      # EAJ+SMPT / MINU / EAM
IDX = torch.tensor(ARMS3)


def fixed_ms(spec, pa, sc):
    def pp(view, pool, tie_rng=None):
        return apply_pa_rule(pa, view, pool, tie_rng)
    m = spec.world().run_gated(seed_chain=sc, policy_pa=pp,
                               policy_l=la_rule_policy(LA_FIX),
                               policy_m=pm_rule_policy(RA_FIX, THR), dt_events=0)
    return objective_vector(m)[0]


def masked_logp(net, tok, seg):
    """12 路 logits 里只取 `ARMS3` 这三路 → `log_softmax`（**带梯度**）。"""
    t = torch.as_tensor(tok).unsqueeze(0)
    lg = net(t, seg, hard=True).rule_logits["PA"][0][IDX]
    return torch.log_softmax(lg, dim=-1)


def masked_pick(net, gen, tok, seg) -> int:
    """在这 3 路里按 softmax 采一个（返回 **ARMS3 内的下标**）。"""
    with torch.no_grad():
        p = torch.softmax(masked_logp(net, tok, seg), dim=-1)
    return int(torch.multinomial(p, 1, generator=gen))


def run(mode: str, steps: int, lr: float, ref: float) -> None:
    torch.manual_seed(0)
    spec, net = build(seed=0, pm_interval=30, target_failures=4, finite_buffer=True,
                      layout_seed=0, reward_norm="minmax")
    for p in net.parameters():
        p.requires_grad_(False)
    for p in net.select.heads["PA"].parameters():
        p.requires_grad_(True)
    optim = torch.optim.Adam(list(net.select.heads["PA"].parameters()), lr=lr)

    def rollout(seed: int, sc: int, greedy: bool):
        """跑一局 → (回报, [(tok, seg, 3 路内下标)])。"""
        gen = torch.Generator().manual_seed(seed)
        rec: list = []
        picks: list = []
        held = [0]

        def policy_pa(view, pool, tie_rng=None):
            tok, seg = build_tok(view.snap, spec.inst, spec.layout, spec.ctx)
            if mode == "each" or not picks:
                k = (int(torch.argmax(masked_logp(net, tok, seg)))
                     if greedy else masked_pick(net, gen, tok, seg))
                held[0] = k
                picks.append(k)
                if not greedy:
                    rec.append((tok, seg, k))
            return apply_pa_rule(ARMS3[held[0]], view, pool, tie_rng)

        m = spec.world().run_gated(seed_chain=sc, policy_pa=policy_pa,
                                   policy_l=la_rule_policy(LA_FIX),
                                   policy_m=pm_rule_policy(RA_FIX, THR), dt_events=0)
        return objective_vector(m), rec, picks

    init = float(np.mean([rollout(0, sc, True)[0][0] for sc in EVAL_W]))
    print(f"\n{'=' * 70}\nmode={mode} steps={steps} lr={lr} · init 均值 ms = {init:.2f}"
          f"   （最优固定 {ref:.2f}）", flush=True)
    for step in range(1, steps + 1):
        loss = torch.zeros(())
        Rs, recs = [], []
        for i in range(8):
            f, rec, _pk = rollout(step * 100 + i, 800_000 + step * 100 + i, False)
            Rs.append(scalar_reward(f, spec.w, bounds=spec.bounds))
            recs.append(rec)
        adv = (np.array(Rs) - np.mean(Rs)) / (np.std(Rs) + 1e-6)
        n = 0
        for a, rec in zip(adv, recs):
            for tok, seg, k in rec:
                loss = loss - float(a) * masked_logp(net, tok, seg)[k]
                n += 1
        if n:
            (loss / n).backward()
            optim.step()
        if step % max(1, steps // 5) == 0 or step == 1:
            ms = [rollout(0, sc, True)[0][0] for sc in EVAL_W]
            _r, _rc, picks = rollout(0, EVAL_W[0], True)
            print(f"  step {step:>3}  均值 ms = {np.mean(ms):7.2f}   "
                  f"决策数/链 = {len(picks)}   用到 = "
                  f"{sorted({PA_RULE_NAMES[ARMS3[k]] for k in picks})}", flush=True)
    ms = float(np.mean([rollout(0, sc, True)[0][0] for sc in EVAL_W]))
    print(f"  ==> 训练后 均值 ms = {ms:.2f}   最优固定 = {ref:.2f}   "
          f"**{'不退化 ✓' if ms <= ref + 3 else '仍退化 ✗'}**", flush=True)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "each"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
    torch.manual_seed(0)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, finite_buffer=True,
                    layout_seed=0, reward_norm="minmax")
    ref = float(np.mean([min(fixed_ms(spec, r, sc) for r in ARMS3) for sc in EVAL_W]))
    print(f"3 路机台规则 · 最优固定均值 = {ref:.2f}")
    for m in (["once", "each"] if mode == "both" else [mode]):
        run(m, steps, lr, ref)


if __name__ == "__main__":
    main()
