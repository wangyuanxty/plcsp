# -*- coding: utf-8 -*-
"""**D1 跑数器**：目标条件化到底起没起作用？（v2 实验计划 D1）

```
python -m plcsp_v2.d1_goal --ckpt checkpoints/d5/two_s0.pt
```

**问的是**：v2 的核心主张 ——「MA 顶层出 `g`，下层条件在 `g` 上」**有用** —— 真的成立吗？

══ ⚠️ 一个已核的陷阱：必须**训练后**再测 ══

**`B` 零初始化 ⟹ 起始时 `g` 完全不影响任何动作**（§93/§95）。
**⟹ 在未训练的网上"关掉 `g`"≡"本来就没开" ⟹ 测不出任何东西。**

**故本实验是「拿一个训好的模型做评测消融」，不是重训。**

══ 五个变体（同一个模型，只改 `g` 怎么来）══

| 变体 | `g` 从哪来 | 对应的问题 |
|---|---|---|
| **`free`** | **MA 每步自采**（正常） | 基线 |
| `fixed_111` | 冻结在中档 `(1,1,1)` | 「顶层关掉，给个不变的 `g`」 |
| `fixed_000` / `fixed_333` | 冻结在两端的角 | 同上，但 `g` 取极端值 |
| **`b_zero`** | **把四个 `B` 置零** | 「`g` **完全**断开」——比冻结更强 |

**判据**：

| 结果 | 结论 |
|---|---|
| **`free` 与各 `fixed_*` / `b_zero` 差不多** | ⚠️ **`g` 没起作用 —— 顶层是装饰** |
| **`free` 明显更好** | ✅ **目标条件化有贡献** |
| 各 `fixed_*` 之间**差很多** | ✅ **`g` 是有语义的方向**（不是随便一个常数） |

⚠️ **`b_zero` 是"分布外"的读法**：模型是在 `B≠0` 下训的，把 `B` 归零会让下层的 logits 整体偏移
——**掉分可能来自"没见过"，不只是"少了 `g`"。** 故**它是旁证，不是主判据**；
主判据是 `free` vs `fixed_*`（两者都在分布内）。

⚠️ 顺带记 **MA 实际采到的档**（`free` 变体）—— 「顶层学到了偏好哪一档」本身是 D6 的一半。
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .env.reward import objective_vector
from .hrl.agents import HierarchicalPolicy
from .hrl.rollout import NetPolicy
from .hrl.train import Spec
from .q1_flat import build

VARIANTS = ("free", "fixed_000", "fixed_111", "fixed_333", "b_zero")


def zero_b(net: HierarchicalPolicy) -> None:
    """把四个 `B` 矩阵归零 —— **`g` 与下层动作彻底断开**（设计 §3 的注入项）。

    ⚠️ **原地改**，调用方要自己负责恢复（本脚本用 `deepcopy`）。
    """
    with torch.no_grad():
        for head in (net.pa, net.la, net.ra_charge, net.ra_maint):
            head.b.zero_()


def _run_one_chain(net: HierarchicalPolicy, spec: Spec, *, variant: str, seed: int,
                   dt_events: int) -> tuple[tuple[float, float, float], list[tuple]]:
    """一条 **greedy** 链 → `(目标三元组, MA 采到的档列表)`。"""
    if variant == "free" or variant == "b_zero":
        gb, freeze = None, False
    else:
        gb = tuple(int(ch) for ch in variant.split("_")[1])       # "fixed_111" → (1,1,1)
        freeze = True
    p = NetPolicy(net, spec.inst, spec.layout, spec.ctx, g_bins=gb, greedy=True,
                  seed=seed, dt_events=dt_events, freeze_g=freeze)
    p.world = spec.world()                                        # type: ignore[attr-defined]
    m = p.world.run_gated(seed_chain=0, policy_pa=p.policy_pa, policy_l=p.policy_l,  # type: ignore[attr-defined]
                          policy_c=p.policy_c, policy_m=p.policy_m, policy_ma=p.policy_ma,
                          dt_events=dt_events)
    bins = [tuple(int(x) for x in d.g_bins) for d in p.rec.decisions if d.kind == "MA"]
    return objective_vector(m), bins


def evaluate_variant(net: HierarchicalPolicy, spec: Spec, *, variant: str, n: int,
                     seed: int, dt_events: int) -> dict:
    net = copy.deepcopy(net)                                      # ⚠️ `b_zero` 是原地改，必须隔离
    if variant == "b_zero":
        zero_b(net)
    prev = net.training
    net.eval()
    try:
        runs = [_run_one_chain(net, spec, variant=variant, seed=seed * 1000 + i,
                               dt_events=dt_events) for i in range(n)]
    finally:
        net.train(prev)
    f = np.array([r[0] for r in runs], dtype=np.float64)
    bins = [b for _f, bs in runs for b in bs]
    return {"variant": variant, "n": n,
            "makespan": float(f[:, 0].mean()), "energy": float(f[:, 1].mean()),
            "twt": float(f[:, 2].mean()),
            # ⚠️ 上面三项**越小越好**，下面这个**越大越好**（与 `reward` 的约定相反）
            "R": float(np.mean([sum(-wi * fi for wi, fi in zip(spec.w, row))
                                for row in f])),
            "n_ma": len(bins),
            "ma_bins": [list(b) for b in bins[:40]]}               # 只留前若干条，别把日志撑爆


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D1：关掉 g（v2 实验计划 D1）")
    ap.add_argument("--ckpt", nargs="+", required=True, help="一个或多个 .pt（同 seed 的 spec）")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--n-eval", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1, help="评测链的种子基")
    ap.add_argument("--dt-events", type=int, default=20)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    torch.set_num_threads(1)

    rows = []
    for path in a.ckpt:
        ck = torch.load(path, weights_only=False)
        spec, _ = build(int(ck["seed"]))          # ⚠️ 同 seed ⟹ 同布局/同归一化
        net = HierarchicalPolicy(n_chargers=ck["n_chargers"], with_value=ck["with_value"])
        net.load_state_dict(ck["state_dict"])
        print(f"\n=== {Path(path).name}（arm={ck['arm']} seed={ck['seed']} "
              f"steps={ck['steps']}）===")
        for v in a.variants:
            r = evaluate_variant(net, spec, variant=v, n=a.n_eval, seed=a.seed,
                                 dt_events=a.dt_events)
            r["ckpt"] = Path(path).name
            r["arm"] = ck["arm"]
            r["train_seed"] = ck["seed"]
            rows.append(r)
            print(f"  {v:<10} ms={r['makespan']:7.2f}  twt={r['twt']:8.2f}  "
                  f"R={r['R']:9.3f}  MA 决策 {r['n_ma']} 条")

    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n写出：{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
