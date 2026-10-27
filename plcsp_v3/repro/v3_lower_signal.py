# -*- coding: utf-8 -*-
"""**阶段 B 到底有没有信号**：用主线配置测组内回报的离散度。

══ 为什么 ══

`tests/test_train.py` 记过一张表：MK01 · 随机初始化 · `dt_events=50` 时，
4 个顶层点里 **3 个的 `σ_M` = 0** ⟹ 组内优势全 0 ⟹ 下层一个梯度都没有。
但那是**测试档**。主线用的是训练好的 ckpt + `dt_events=20` + `span=1`。

本脚本用**主线配置**重测同一件事：

- 每个顶层点 `j` 跑 `M` 条分支（规则冻结、只有执行在变）；
- 报该点段回报的 **σ / 极差 / 不同取值个数**；
- 报该段内 PA/LA 决策数，以及 `K_eff > 1` 的比例（能变的决策占多少）。

判据：σ 恒 0 的点 ⟹ 那些步**完全没梯度**。σ 普遍很小 ⟹ 优势≈噪声。

══ 口径 ══

ckpt 由 `V3_SIG_CKPT` 指定（默认主模型 span1）。
实例：随机 `gen_mk_like(seed=0)` · layout_seed=100 · `dt_events=20` · `span=1`。
⚠️ **训练实例**，不是 MK 零样本；这里只问"信号在不在"，不问泛化。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from plcsp_v3.env.instances import gen_mk_like                # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy              # noqa: E402
from plcsp_v3.hrl.train import (_prefix_to, _run_all_segments,   # noqa: E402
                                _scalar, base_chain, segment_return)
from plcsp_v3.q1_flat import build                               # noqa: E402

M_BRANCH = int(os.environ.get("V3_SIG_M", "8"))
DT = int(os.environ.get("V3_SIG_DT", "20"))
SPAN = int(os.environ.get("V3_SIG_SPAN", "1"))
CKPT = os.environ.get(
    "V3_SIG_CKPT", os.path.join(os.environ.get("TEMP", ""),
                                "v3_mafix200_seed0.pt"))


def main() -> None:
    if not os.path.exists(CKPT):
        print(f"⚠️ 缺 ckpt {CKPT}", flush=True)
        return
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = HierarchicalPolicy(seq_ma=bool(obj.get("args", {}).get("seq_ma")))
    net.load_state_dict(obj["state_dict"])
    net.eval()

    sp, _ = build(seed=0, inst_obj=gen_mk_like(seed=0), tau=2.0,
                  pm_interval=30, target_failures=4, n_agv=3,
                  finite_buffer=True, layout_seed=100, reward_norm="minmax",
                  sim_overrides={"due_range": 0.6})
    print(f"**阶段 B 信号** · {os.path.basename(CKPT)} · M={M_BRANCH}"
          f" · dt={DT} · span={SPAN}", flush=True)
    print(f"  实例 {sp.inst.n_jobs}×{sp.inst.n_machines} · "
          f"{sum(len(o) for o in sp.inst.jobs)} 工序\n", flush=True)

    base = base_chain(net, sp, greedy=True, seed=0, dt_events=DT, seed_chain=0)
    n_top = sum(1 for d in base.rec.decisions if d.kind == "MA")
    print(f"  base makespan {base.metrics['makespan']:.1f} · 顶层点 {n_top}\n",
          flush=True)

    plan = [(j, [(_prefix_to(base.rec.decisions, j)[1], 1000 + m, False)
                 for m in range(M_BRANCH)]) for j in range(n_top)]
    all_b = _run_all_segments(net, sp, base, plan, dt_events=DT, pool=None,
                              span=SPAN)
    print(f"  {'j':>3}{'段回报 均值':>12}{'σ':>9}{'极差':>9}{'不同值':>8}"
          f"{'决策数':>8}{'K>1 比例':>10}{'σ=0':>6}", flush=True)
    print("  " + "-" * 68, flush=True)
    zero = 0
    sig = []
    for j in range(n_top):
        before = base.ma_counters[j]
        segs = [segment_return(before, br.counters) for br in all_b[j]]
        r = np.array([_scalar(sg, sp.w) for sg in segs], dtype=float)
        s = float(r.std(ddof=0))
        k1 = [int(np.count_nonzero(d.cand.mask)) for br in all_b[j]
              for d in br.decisions if d.cand is not None]
        k_frac = float(np.mean([x > 1 for x in k1])) if k1 else float("nan")
        n_dec = int(np.mean([len([d for d in br.decisions
                                  if d.kind in ("PA", "LA", "M")])
                             for br in all_b[j]]))
        if s == 0.0:
            zero += 1
        sig.append(s)
        print(f"  {j:>3}{r.mean():>12.3f}{s:>9.3f}"
              f"{float(r.max() - r.min()):>9.3f}{len(np.unique(np.round(r, 6))):>8}"
              f"{n_dec:>8}{k_frac:>10.1%}{'是' if s == 0.0 else '':>6}",
              flush=True)
    print(f"\n  **σ=0 的顶层点：{zero} / {n_top}**"
          f"（这些点对下层贡献**恰好为 0** 的梯度）", flush=True)
    print(f"  σ 中位数 {np.median(sig):.3f}　均值 {np.mean(sig):.3f}", flush=True)


if __name__ == "__main__":
    main()
