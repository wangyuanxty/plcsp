# -*- coding: utf-8 -*-
"""**G · 三目标 Pareto 跑数器**（`docs/v2/experiment-plan.md` §G）—— **扫 `g`，不重训**。

```
python -m plcsp_v3.g_pareto --ckpt <Full_s0.pt> --n-eval 3
```

══ ⭐ 为什么本跑数器**不按权重训 M 个策略** ══

计划 §G 的原做法是"加权和分解，扫 M 个 `prefs` 权重向量，**每个训一个策略**"。
**v2 有一个便宜得多、而且更贴合本文主张的做法**：

> **`g` 就是"这次要什么"的连续条件。** 把同一个已训策略在 `g` 的 64 个档
> （3 维 × 4 档）上各跑一遍，**一次训练就得到 64 个运行点** ⟹ 直接读前沿。

**这正是目标条件化的回报**：一个策略覆盖整条前沿，而不是每个偏好点一个策略。
⚠️ **但这件事本身要如实测**：若 64 个点**挤成一团**（= `g` 没起作用），
那说明条件化没学到东西 —— **这恰好是 D1 要测的**，两个实验互为佐证。

⚠️ **v1 的前沿退化成两个点**（§52.9.3），根因是**总能耗与 makespan 近秩等价（ρ=0.978）**。
**本跑数器顺带核这一条**（报三目标两两的 Spearman ρ）。**若 v2 重演，如实写，别掩饰。**

⚠️ 只评测、不训练 ⟹ 每次调用只花分钟级。默认写到系统临时目录（硬规则 1）。
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .env.reward import objective_vector
from .hrl.agents import G_BINS, G_DIM, HierarchicalPolicy
from .hrl.rollout import NetPolicy
from .q1_flat import (build, constraints_from_ckpt, missing_env_warning,
                      resolve_train_env)

DEFAULT_OUT = os.path.join(tempfile.gettempdir(), "g_pareto.jsonl")

OBJ = ("makespan", "energy", "twt")
"""三目标，**一律越小越好**。"""



def run_g(net: HierarchicalPolicy, spec, g_bins: tuple[int, int, int], *, n: int,
          seed: int, dt_events: int) -> tuple[tuple[float, float, float], list]:
    """在**固定 `g`** 下跑 `n` 条 greed 链 → 三目标均值 + MA 采到的档（诊断用）。

    ⚠️ **`freeze_g=True` 是必须的**：否则顶层会在链内重新采 `g`，
    "这一条链条件在哪个 `g` 上"就说不清了 —— 而本节的全部意义就在于**控制 `g`**。
    """
    f = np.zeros(3, dtype=np.float64)
    bins: list = []
    for i in range(n):
        p = NetPolicy(net, spec.inst, spec.layout, spec.ctx, g_bins=g_bins,
                      greedy=True, seed=seed * 1000 + i, dt_events=dt_events, freeze_g=True)
        p.world = spec.world()                              # type: ignore[attr-defined]
        m = p.world.run_gated(                               # type: ignore[attr-defined]
            seed_chain=0, policy_pa=p.policy_pa, policy_l=p.policy_l,
            policy_m=p.policy_m, policy_ma=p.policy_ma, dt_events=dt_events)
        f += np.array(objective_vector(m), dtype=np.float64)
        bins += [tuple(int(x) for x in d.g_bins) for d in p.rec.decisions if d.kind == "MA"]
    return tuple((f / n).tolist()), bins


def pareto_front(points: list[tuple[float, float, float]]) -> list[int]:
    """非支配点的小标（**三目标一律越小越好**）。O(n²)，n=64 足够。

    `i` 被支配 ⟺ 存在 `j` 在**三项都不差于** `i`、且**至少一项严格更好**。
    """
    keep = []
    for i, a in enumerate(points):
        dominated = any(
            all(b[k] <= a[k] + 1e-12 for k in range(3)) and any(b[k] < a[k] - 1e-12 for k in range(3))
            for j, b in enumerate(points) if j != i)
        if not dominated:
            keep.append(i)
    return keep


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 秩相关（不引 scipy：本仓依赖里没有它，且这里只要一个数）。"""
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    d = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / d) if d > 1e-12 else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="G：三目标 Pareto（扫 g，不重训）")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--n-eval", type=int, default=3, help="每个 g 档的链数")
    ap.add_argument("--seed", type=int, default=1, help="评测链的种子基")
    ap.add_argument("--dt-events", type=int, default=20)
    ap.add_argument("--only-diag", action="store_true",
                    help="只扫对角 4 档（000/111/222/333）——冒烟用")
    # ⚠️ **必须还原训练环境**（2026-10-10 加）—— 见 `q1_flat.resolve_train_env`
    ap.add_argument("--pm-interval", type=float, default=None)
    ap.add_argument("--agv-n-failures", type=int, default=0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    torch.set_num_threads(1)

    combos = ([(k, k, k) for k in range(G_BINS)] if a.only_diag
              else list(itertools.product(range(G_BINS), repeat=G_DIM)))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with out.open("a", encoding="utf-8") as fh:
        def rec(d: dict) -> None:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            fh.flush()

        for path in a.ckpt:
            ck = torch.load(path, weights_only=False)
            cons = constraints_from_ckpt(ck)
            env = resolve_train_env(ck, pm_interval=a.pm_interval,
                                    agv_n_failures=a.agv_n_failures)
            warn = missing_env_warning(ck, env)
            if warn:
                print(warn, file=sys.stderr)
            spec, _ = build(int(ck["seed"]), constraints=cons, **env)
            net = HierarchicalPolicy(with_value=ck["with_value"])
            net.load_state_dict(ck["state_dict"])
            net.eval()
            print(f"\n=== {Path(path).name}（arm={ck['arm']} seed={ck['seed']} "
                  f"steps={ck['steps']}）· {len(combos)} 个 g 档 ===", flush=True)

            pts, meta = [], []
            for gb in combos:
                f, bins = run_g(net, spec, gb, n=a.n_eval, seed=a.seed,
                                dt_events=a.dt_events)
                pts.append(f)
                meta.append({"g": list(gb), "f": list(f), "n_ma": len(bins),
                             "ma_bins": [list(b) for b in bins[:10]]})
                print(f"  g={gb}  ms={f[0]:7.2f}  E={f[1]:6.3f}  twt={f[2]:8.2f}", flush=True)

            front = pareto_front(pts)
            arr = np.array(pts)
            rho = {f"{OBJ[i]}-{OBJ[j]}": round(_spearman(arr[:, i], arr[:, j]), 3)
                   for i in range(3) for j in range(i + 1, 3)}
            # ⚠️ **先判"退化"再报前沿**：所有点相同时 `pareto_front` 会返回**全部**点
            # （没有谁被谁支配），报成"64/64 非支配"是**误导** —— 那不是一条前沿，
            # 是一个点。**实测踩过**：`g` 无影响时 64 个档逐位相同。
            spread = float((arr.max(axis=0) - arr.min(axis=0)).max())
            degenerate = spread <= 1e-9
            rec({"kind": "pareto", "ckpt": Path(path).name, "arm": ck["arm"],
                 "train_seed": ck["seed"], "n_eval": a.n_eval, "n_g": len(combos),
                 "degenerate": degenerate, "max_spread": spread,
                 "points": meta, "front_idx": front, "spearman": rho})
            if degenerate:
                print(f"\n  ⟹ ⚠️⚠️ **退化成一点**：{len(pts)} 个 `g` 档给出**逐位相同**的三目标"
                      f"（极差 {spread:.3e}）。\n"
                      "     **这不是「前沿只有一个点」，是「`g` 对动作完全没影响」。**\n"
                      "     设计 §95：`B` 零初始化，`g` 影响动作是**学出来的**。\n"
                      "     若训练跑完了还是这样，去看 `B` 的量级（`d6_bmat`）——"
                      "**多半是它太小**（见 progress-log §124）。\n"
                      "     ⚠️ 此处的 ρ=1.0 是**退化造成的**，**不是** v1 那条"
                      "「能耗与 makespan 近秩等价」（§52.9.3）——别混为一谈。\n")
            else:
                print(f"\n  ⟹ **非支配点 {len(front)}/{len(pts)}**："
                      f"{[meta[i]['g'] for i in front]}")
                print(f"  ⟹ 三目标两两 Spearman ρ：{rho}")
                print("     ⚠️ v1 的 `energy` 与 `makespan` ρ=0.978（近秩等价，前沿退化成两点）。"
                      "**这里若也接近 1，就是同一件事重演，如实写。**\n")
            rows.append({"ckpt": Path(path).name, "n_front": len(front),
                         "degenerate": degenerate, "rho": rho})

    print(f"写出：{out}", file=sys.stderr)
    return 0


if __name__ == "__main__":      # ⚠️ Windows `spawn` 需要（本文件不建池，但保持一致）
    raise SystemExit(main())
