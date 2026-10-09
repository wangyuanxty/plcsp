# -*- coding: utf-8 -*-
"""**Q1 跑数器**（设计 §12.2–§12.4）——扁平链两臂的读数。

```
python -m plcsp_v2.q1_flat --arms A B --seeds 0 1 2 --steps 200
```

**问的是**：*Learning Without Critics?*（arXiv:2511.03527）称
*"learned critics remain essential for long-horizon tasks"* —— **而调度是长时程。**
臂 A 有 critic（`R − V(s_t)`），臂 B 是现行 GRPO 口径（组内 z 化）。**其余完全相同。**

**测三样**（§12.4）：**训练曲线方差 · 收敛速度 · 扰动（③⑨）下的最终业绩。**
⚠️ **必须多种子** —— 单种子说明不了稳定性，而"稳定性"正是这里要测的东西。

══ 输出 ══

JSONL，每行一条记录（`kind` 区分）：
- `train`：每步的 `step / r_mean / r_std / loss / heads_grad / critic_grad / enc_grad`；
- `eval`：每 `--eval-every` 步的 **greedy 评测**（`makespan / energy / twt` 的均值与标准误）；
- `final`：跑完后的**扰动评测**（⑨ 按 `scaled_agv_mtbf` 激活到每局 ~4 次故障）。

⚠️ **默认写到系统临时目录**（项目硬规则 1：`*.jsonl` 不进项目根）。`--out` 可覆盖。

══ 一条必须写明的口径 ══

**两臂的优势尺度不同，这是构造性的**：臂 B 的 `z(R)` 天生均值 0、标准差 1；
臂 A 的 `R − V(s_t)` 在 `V` 还没学好时**约等于 `R`（≈ −40）**，比 z 值大 ~25 倍。
**不改。** 理由：Adam 的更新是 `lr·m̂/(√v̂+ε)`，**对整体的尺度因子近似不变**，
故两臂的**更新幅度可比**；而尺度差本身就是"学出来的基线 vs 组内基线"的一部分。
⚠️ **这条要在论文里写明**，别让审稿人以为两个梯度范数差 50 倍是我漏了归一化。
（实测 `n=8`：臂 B `heads_grad≈199`，臂 A `heads_grad≈10527`。）
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .env import layout as layout_mod
from .env.corridors import build_corridor_graph, dock_distance_matrix
from .env.des import SimConfig, reference_makespan, scaled_agv_mtbf
from .env.instances import load_mk
from .env.reward import ReferenceObjectives, reward_weights
from .hrl.agents import HierarchicalPolicy
from .hrl.flat import flat_rollout, flat_step
from .hrl.train import Spec
from .nn.features import norm_context

DEFAULT_OUT = os.path.join(tempfile.gettempdir(), "q1_flat.jsonl")

TARGET_FAILURES = 4.0
"""⑨ 的**激活口径**（`scaled_agv_mtbf`）：每局期望故障次数。

⚠️ 为什么必须激活：`SimConfig.agv_mtbf` 默认 480 min，而 MK01 一局只有 ~100–170 min
⟹ **默认档下 ⑨ 几乎不发生**，"故障下稳不稳"就测了个空。"""


def build(seed: int, *, tau: float = 1.0, disturb: bool = False,
          n_agv: int = 3, with_value: bool = False):
    """→ `(spec, net)`。**两臂走同一个 `build`** ⟹ 实例/布局/编码器/约束档全同。

    `disturb=True`：⑨ 按 `TARGET_FAILURES` 激活（只用于**最终评测**，不用于训练）。
    """
    torch.manual_seed(seed)
    inst = load_mk("mk01")
    cfg = SimConfig(tau=tau, n_agv=n_agv)
    if disturb:
        # ⚠️ `SimConfig` 是**普通 dataclass**（不是 frozen/NamedTuple）⟹ 用 `dataclasses.replace`，
        # 不用 `_replace`（那是 NamedTuple 的方法，写错是 AttributeError 而不是静默）。
        cfg = dataclasses.replace(cfg, agv_mtbf=scaled_agv_mtbf(inst, TARGET_FAILURES))
    lay = layout_mod.sample_layout(inst.n_machines, seed=seed, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    dm = dock_distance_matrix(g)
    ctx = norm_context(inst, lay, reference_makespan(inst, cfg), cfg)
    w = reward_weights(ReferenceObjectives.of(inst, cfg).as_tuple())
    net = HierarchicalPolicy(n_chargers=len(lay.chargers), with_value=with_value)
    return Spec(inst, lay, g, dm, cfg, ctx, w), net


def evaluate(net: HierarchicalPolicy, spec: Spec, *, n: int, seed: int,
             dt_events: int) -> dict:
    """**greedy** 评测 `n` 条链 → 三目标的均值与标准误（越小越好）。"""
    prev = net.training
    net.eval()                                   # 关 dropout/BN（若有）——评测要确定性
    try:
        runs = flat_rollout(net, spec, n=n, seed=seed, dt_events=dt_events, greedy=True)
    finally:
        net.train(prev)
    f = np.array([x[2] for x in runs], dtype=np.float64)
    return {"n": n,
            "makespan": float(f[:, 0].mean()), "makespan_se": float(f[:, 0].std(ddof=1)
                                                                   / np.sqrt(n)) if n > 1 else 0.0,
            "energy": float(f[:, 1].mean()),
            "twt": float(f[:, 2].mean()),
            "R": float(np.mean([x[0] for x in runs]))}


def run_one(arm: str, seed: int, *, steps: int, n: int, dt_events: int,
            eval_every: int, eval_n: int, rec) -> None:
    """一条 `(臂, 种子)` 的完整训练 + 评测。`rec` = 写一行的回调。"""
    spec, net = build(seed, with_value=(arm == "A"))
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    rec({"kind": "start", "arm": arm, "seed": seed, "steps": steps, "n": n,
         "dt_events": dt_events, "n_params": sum(p.numel() for p in net.parameters())})
    base = evaluate(net, spec, n=eval_n, seed=0, dt_events=dt_events)
    rec({"kind": "eval", "arm": arm, "seed": seed, "step": 0, **base})

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        out = flat_step(net, spec, optim, arm=arm, n=n, seed=seed * 10000 + step,
                        dt_events=dt_events)
        gn = out["grad_norm"]
        rs = [r["R"] for r in out["runs"]]
        # ⚠️ `flat_step` 返回的 loss 是**整组 8 条链的和**（真实被优化的量）。这里**除以 `n`**
        # 再记，否则两个数会读错：
        #   · 臂 A 的 `loss ≈ −9.4e4` 看着吓人，其实是 8 条链之和（每条 ≈ −1.2e4）；
        #   · 臂 B 的 `loss ≈ −12.6` **不是"没学"** —— 组内 z 化使 `Σ_i z_i ≡ 0`，
        #     8 条链的贡献**互相抵消**。**这才是 GRPO 的正常形状**，别当成 bug。
        rec({"kind": "train", "arm": arm, "seed": seed, "step": step,
             "r_mean": float(np.mean(rs)), "r_std": float(np.std(rs, ddof=1)) if n > 1 else 0.0,
             "loss_per_chain": out["loss"] / n,
             "policy_loss_per_chain": out["policy_loss"] / n,
             "critic_loss_per_chain": out["critic_loss"] / n,
             # ⚠️ `critic_loss_per_chain` ≈ `E[(V−R)²]`：初值 ≈ R² ≈ 1300 ⟹ 说明 **`V` 从 0 附近起步**
             # （critic 头是默认小初值）。随训练下降才说明 critic 在学。
             "heads_grad": gn["heads"], "critic_grad": gn["critic"],
             "enc_grad": gn["encoder"],
             "makespan": float(np.mean([r["makespan"] for r in out["runs"]])),
             "elapsed_s": time.perf_counter() - t0})
        if eval_every and step % eval_every == 0:
            ev = evaluate(net, spec, n=eval_n, seed=0, dt_events=dt_events)
            rec({"kind": "eval", "arm": arm, "seed": seed, "step": step, **ev})

    # 最终：**扰动**（⑨ 激活）下的业绩
    dspec, _ = build(seed, disturb=True, with_value=(arm == "A"))
    dist = evaluate(net, dspec, n=eval_n, seed=1, dt_events=dt_events)
    rec({"kind": "final", "arm": arm, "seed": seed, "total_s": time.perf_counter() - t0,
         "clean": evaluate(net, spec, n=eval_n, seed=1, dt_events=dt_events),
         "disturbed": dist, "agv_mtbf": dspec.cfg.agv_mtbf})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Q1：扁平链两臂（设计 §12.2–§12.4）")
    ap.add_argument("--arms", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n", type=int, default=8, help="组大小（设计 §12.2 的 G）")
    ap.add_argument("--dt-events", type=int, default=20, help="顶层定时器步长（§10②）")
    ap.add_argument("--eval-every", type=int, default=25, help="0 = 不中途评测")
    ap.add_argument("--eval-n", type=int, default=3, help="评测链数（greedy）")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        def rec(d: dict) -> None:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            fh.flush()
            print(json.dumps(d, ensure_ascii=False), flush=True)

        for seed in a.seeds:
            for arm in a.arms:
                run_one(arm, seed, steps=a.steps, n=a.n, dt_events=a.dt_events,
                        eval_every=a.eval_every, eval_n=a.eval_n, rec=rec)
    print(f"\n写出：{out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
