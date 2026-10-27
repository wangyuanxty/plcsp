# -*- coding: utf-8 -*-
"""5 分钟 bug 检查：重置 select 头后，LA 头到底有没有梯度？

做两件事：
1. 在同一个 base 链的 3 个顶层点上，固定 PA/RA、只换 3 条 LA 规则，
   跑终局段，比较最终 MS。看 LA 信号是否真的存在。
2. 用这 3 条规则做一个小 phase A（重置后的 select 头），
   backward，打印 PA/LA/RA 头与 encoder 的梯度范数。
   LA 头梯度为 0 ⟹ 代码 bug；非 0 ⟹ 训练设置问题。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from plcsp_v3.env.instances import gen_mk_like  # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy, RuleSelectHeads  # noqa: E402
from plcsp_v3.hrl.replay import segment_return  # noqa: E402
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, N_LA_RULES, N_PA_RULES,  # noqa: E402
                                N_PM_RULES, PA_RULE_NAMES, PM_RULE_NAMES)
from plcsp_v3.hrl.train import (base_chain, group_advantage, ma_logp_at,  # noqa: E402
                                segment_from)
from plcsp_v3.q1_flat import build  # noqa: E402

CKPT = os.path.join(os.environ.get("TEMP", ""), "v3_mafix200_seed0.pt")
DT = 20
N_TOP = 3
PA = PA_RULE_NAMES.index("EAJ+SMPT")
RA = PM_RULE_NAMES.index("QUEUE_EMPTY")


def _grad_norm(module) -> float:
    s = 0.0
    for p in module.parameters():
        if p.grad is not None:
            s += float(p.grad.pow(2).sum())
    return float(np.sqrt(s))


def main() -> None:
    inst = gen_mk_like(seed=0)
    sp, _ = build(seed=0, inst_obj=inst, tau=2.0, pm_interval=30,
                  target_failures=4, n_agv=3, finite_buffer=True,
                  layout_seed=100, reward_norm="minmax",
                  sim_overrides={"due_range": 0.6})
    sp = sp._replace(w=(1.0, 0.0, 0.0))
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = HierarchicalPolicy()
    net.load_state_dict(obj["state_dict"])
    # 重置 select：从塌缩的 MA 继续训是错的。
    net.select = RuleSelectHeads(net.enc.d_model,
                                 {"PA": N_PA_RULES, "LA": N_LA_RULES,
                                  "RA": N_PM_RULES},
                                 in_dim=4 * net.enc.d_model)
    for p in net.parameters():
        p.requires_grad_(False)
    for p in net.enc.parameters():
        p.requires_grad_(True)
    for p in net.select.parameters():
        p.requires_grad_(True)
    net.train()

    base = base_chain(net, sp, greedy=True, seed=0, dt_events=DT,
                      seed_chain=100_001, force_top1=True)
    ma_idx = [i for i, d in enumerate(base.rec.decisions) if d.kind == "MA"]
    n_top = min(N_TOP, len(ma_idx))
    rules = [(PA, la, RA) for la in range(N_LA_RULES)]

    print(f"ckpt = {CKPT}（encoder 保留，select 重置）", flush=True)
    print(f"实例 = {inst.n_jobs}×{inst.n_machines} · n_top 前 {n_top} 个", flush=True)
    print(f"固定 PA={PA_RULE_NAMES[PA]} · RA={PM_RULE_NAMES[RA]} · 只换 LA\n",
          flush=True)

    loss = torch.zeros((), requires_grad=True)
    for j in range(n_top):
        rewards = []
        ms = []
        for r in rules:
            br = segment_from(net, sp, base, j, rule=r, seed=0, greedy=True,
                              dt_events=DT, seed_chain=base.seed_chain,
                              terminal=True, force_top1=True)
            sg = segment_return(base.ma_counters[j], br.counters)
            rewards.append(float(-sg.d_ms))
            ms.append(float(br.counters.ms_committed))
        d = base.rec.decisions[ma_idx[j]]
        at = ma_logp_at(net, d, rules)
        adv = group_advantage(rewards)
        loss = loss - (torch.as_tensor(adv) * at).sum()
        print(f"  j={j}  " + "  ".join(
            f"{LA_RULE_NAMES[k]}: MS {ms[k]:.1f} (adv {adv[k]:+.2f})"
            for k in range(len(rules))), flush=True)

    loss.backward()
    print("\n  梯度范数：", flush=True)
    for k in ("PA", "LA", "RA"):
        print(f"    {k} head = {_grad_norm(net.select.heads[k]):.6f}", flush=True)
    print(f"    select trunk = {_grad_norm(net.select.trunk):.6f}", flush=True)
    print(f"    encoder = {_grad_norm(net.enc):.6f}", flush=True)
    print(f"    loss = {float(loss.detach()):.4f}", flush=True)


if __name__ == "__main__":
    main()
