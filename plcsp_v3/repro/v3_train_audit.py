# -*- coding: utf-8 -*-
"""**PA/LA 训练通路的数值审计**（2026-10-20）。

══ 问什么 ══

用户问：「PA/LA 的训练到底有没有 bug？」

本脚本只查**通路**，不查学习效果。四条判据：

| # | 判据 | 不通过意味着 |
|---|---|---|
| 1 | 分支里每条决策的**记录 logp** == 带梯度重算的 logp | 训练用的分布不是执行时的分布 |
| 2 | 阶段 B 的 loss 对 `scorer_pa` / `scorer_la` 的梯度 **非零** | 梯度没接上 |
| 3 | 同一条分支重跑一次 ⟹ 逐位相同 | 分支不确定 |
| 4 | 优势的符号与段回报的符号一致 | 更新方向反了 |

⚠️ 判据 1 是**最关键的一条**：如果记录 logp 与重算 logp 不等，
阶段 B 实际在优化一个**与执行策略无关**的目标。

══ 用法 ══

`D:/anaconda/python.exe -m plcsp_v3.repro.v3_train_audit`
ckpt 由 `V3_AUDIT_CKPT` 指定（默认主模型 span1）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from plcsp_v3.env.instances import gen_mk_like                 # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy              # noqa: E402
from plcsp_v3.hrl.train import (LOWER_KINDS, _prefix_to,        # noqa: E402
                                _run_all_segments, _scalar, base_chain,
                                batched_terms, group_advantage,
                                segment_return)
from plcsp_v3.q1_flat import build                               # noqa: E402

DT = 20
M = 4
CKPT = os.environ.get(
    "V3_AUDIT_CKPT", os.path.join(os.environ.get("TEMP", ""),
                                  "v3_mafix200_seed0.pt"))


def main() -> None:
    if not os.path.exists(CKPT):
        print(f"⚠️ 缺 ckpt {CKPT}", flush=True)
        return
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = HierarchicalPolicy(seq_ma=bool(obj.get("args", {}).get("seq_ma")))
    net.load_state_dict(obj["state_dict"])
    net.train()                                   # 训练态（BN/dropout 与训练一致）

    sp, _ = build(seed=0, inst_obj=gen_mk_like(seed=0), tau=2.0,
                  pm_interval=30, target_failures=4, n_agv=3,
                  finite_buffer=True, layout_seed=100, reward_norm="minmax",
                  sim_overrides={"due_range": 0.6})
    print(f"**训练通路审计** · {os.path.basename(CKPT)} · "
          f"{sp.inst.n_jobs}×{sp.inst.n_machines} · M={M}\n", flush=True)

    base = base_chain(net, sp, greedy=False, seed=0, dt_events=DT, seed_chain=0)
    n_top = sum(1 for d in base.rec.decisions if d.kind == "MA")
    print(f"  base makespan {base.metrics['makespan']:.1f} · 顶层点 {n_top}",
          flush=True)

    j = min(6, n_top - 1)
    rule = _prefix_to(base.rec.decisions, j)[1]
    plan = [(j, [(rule, 5000 + m, False) for m in range(M)])]
    runs = _run_all_segments(net, sp, base, plan, dt_events=DT, pool=None)[0]
    before = base.ma_counters[j]
    segs = [segment_return(before, br.counters) for br in runs]
    rr = np.array([_scalar(sg, sp.w) for sg in segs])
    adv = group_advantage(rr.tolist())

    # ── 判据 1：记录 logp vs 重算 logp ──
    print("\n  ── 判据 1：记录 logp == 重算 logp ──", flush=True)
    worst = 0.0
    n_cmp = 0
    for m, br in enumerate(runs):
        sel = [d for d in br.decisions if d.kind in LOWER_KINDS]
        if not sel:
            continue
        lp, _ = batched_terms(net, br.decisions, LOWER_KINDS)
        rec = np.array([d.logp for d in sel], dtype=np.float64)
        got = lp.detach().numpy().astype(np.float64)
        diff = float(np.max(np.abs(rec - got))) if len(sel) else 0.0
        worst = max(worst, diff)
        n_cmp += len(sel)
        print(f"    分支 {m}: {len(sel):>3} 条　PA "
              f"{sum(d.kind == 'PA' for d in sel):>3}　LA "
              f"{sum(d.kind == 'LA' for d in sel):>3}　"
              f"max|Δlogp| = {diff:.3e}", flush=True)
    print(f"  **总计 {n_cmp} 条决策，最大偏差 {worst:.3e}** "
          f"{'✅' if worst < 1e-4 else '❌ 通路有问题'}", flush=True)

    # ── 判据 2：梯度非零 ──
    print("\n  ── 判据 2：loss 对两个打分头的梯度 ──", flush=True)
    net.zero_grad(set_to_none=True)
    loss = torch.zeros((), requires_grad=True)
    for a, br in zip(adv, runs):
        lp, _ = batched_terms(net, br.decisions, LOWER_KINDS)
        if lp.numel():
            loss = loss - float(a) * lp.sum()
    loss.backward()
    for name in ("scorer_pa", "scorer_la", "scorer_ra", "enc", "select"):
        mod = getattr(net, name)
        s = sum(float(p.grad.pow(2).sum()) for p in mod.parameters()
                if p.grad is not None)
        print(f"    {name:<12} grad_norm = {s ** 0.5:.6f}", flush=True)
    net.zero_grad(set_to_none=True)

    # ── 判据 3：分支可复现 ──
    print("\n  ── 判据 3：同一分支重跑逐位相同 ──", flush=True)
    runs2 = _run_all_segments(net, sp, base, plan, dt_events=DT, pool=None)[0]
    same = all(
        [(d.kind, d.act) for d in a.decisions] ==
        [(d.kind, d.act) for d in b.decisions]
        for a, b in zip(runs, runs2))
    print(f"    四个分支的决策序列相同：{'✅' if same else '❌'}", flush=True)

    # ── 判据 4：优势符号 ──
    print("\n  ── 判据 4：优势 vs 段回报 ──", flush=True)
    order = np.argsort(rr)
    print(f"    段回报（升序）：{[round(float(x), 3) for x in rr[order]]}",
          flush=True)
    print(f"    对应优势      ：{[round(float(x), 3) for x in adv[order]]}",
          flush=True)
    ok = np.all(np.diff(adv[order]) > 0)
    print(f"    优势随回报单调递增：{'✅' if ok else '❌'}", flush=True)


if __name__ == "__main__":
    main()
