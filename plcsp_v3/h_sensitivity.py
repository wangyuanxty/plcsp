# -*- coding: utf-8 -*-
"""**H · 敏感性分析跑数器**（`docs/v2/experiment-plan.md` §H）—— **同策略在变参数的环境里重评**。

```
python -m plcsp_v3.h_sensitivity --ckpt <Full_s0.pt> --n-eval 3
```

══ 问的是什么 ══

**不是**"换个参数重训会怎样"（那是 §H 之外的事），而是：

> **策略训练时用的参数，与部署时真实的不一样 —— 业绩掉多少？**

⚠️ 这一点决定了本跑数器的正确口径：**布局与训练时相同（同 seed）、只改被扫的那个参数**。
`build(seed, ...)` 的布局只由 `(seed, n_machines, n_agv)` 决定 ⟹ 其余参数随便改，
几何不动。**唯一的例外是 `n_agv`** —— 它改车队规模，故单列。

⚠️ **特征也会跟着变**（`pm_interval` 是 `pm_left` 维的除数、`aisle_width` 影响归一化……）——
**这是有意的**：部署时策略看得到真实参数，看得见却没见过，正是要测的东西。

══ ⚠️ 本跑数器报的是"退化"，不是"敏感性系数" ══

输出里给每个参数一行：**相对基准的百分比变化**。**不报方差分析、不报显著性** ——
3 个 seed × 3 条链撑不起那些，报了就是过度声称。**要给区间就多跑 seed，别在这里编。**

⚠️ 默认写到系统临时目录（硬规则 1）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

from .hrl.agents import HierarchicalPolicy
from .q1_flat import (build, constraints_from_ckpt, evaluate,
                      missing_env_warning, resolve_train_env)

DEFAULT_OUT = os.path.join(tempfile.gettempdir(), "h_sensitivity.jsonl")

GRID: dict[str, list] = {
    # 🔴 计划 §H 标为最要紧的两个
    "pm_interval": [15.0, 30.0, 60.0, 120.0],
    "pm_duration": [5.0, 10.0, 20.0],
    # 🟠
    "agv_speed_mps": [0.25, 0.5, 1.0],
    "aisle_width": [1.0, 1.5, 2.0],
    "zone_hold": [0.5, 1.0, 2.0],
    # 🟡 开关型
    "agv_failover": [False, True],
    "machine_age_failure": [False, True],
}
"""逐参数的扫描点。**基准值由 `--base` 给**（默认 `pm_interval=30`，与训练参数组一致）。

⚠️ `zone_hold` 同时是**② 的让步轮询间隔** —— 它变大 ⟹ 阻塞时的重试更粗 ⟹
`buffer_block_min` 会跟着变。**读那一行的"退化"时要知道这不只是"等待更久"。**
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="H：敏感性（同策略重评）")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--n-eval", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1, help="评测链的种子基")
    ap.add_argument("--dt-events", type=int, default=20)
    ap.add_argument("--base", type=float, default=30.0,
                    help="`pm_interval` 的基准值（训练用的那个；本跑数器其余格点照 `GRID`）")
    ap.add_argument("--pm-interval", type=float, default=None,
                    help="训练时的 ⑫ 保养间隔（旧 ckpt 无 `env` 时必须给）")
    ap.add_argument("--agv-n-failures", type=int, default=0,
                    help="训练时的 ⑨ 按次数注入")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    torch.set_num_threads(1)

    grid = dict(GRID)
    grid["pm_interval"] = sorted({a.base, *grid["pm_interval"]})

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        def rec(d: dict) -> None:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            fh.flush()

        for path in a.ckpt:
            ck = torch.load(path, weights_only=False)
            cons = constraints_from_ckpt(ck)
            seed = int(ck["seed"])
            net = HierarchicalPolicy(with_value=ck["with_value"])
            net.load_state_dict(ck["state_dict"])
            net.eval()
            print(f"\n=== {Path(path).name}（arm={ck['arm']} seed={seed} "
                  f"steps={ck['steps']}）===", flush=True)

            # ⚠️ **基准必须在同一套 `build` 参数下取** —— 否则"变化"里混进了别的东西。
            # ⚠️ **基准 = 训练时的环境**，必须还原（否则"退化"里混进了"环境换了"）
            base_env = resolve_train_env(ck, pm_interval=a.pm_interval,
                                         agv_n_failures=a.agv_n_failures)
            if a.pm_interval is None and "pm_interval" not in base_env:
                base_env["pm_interval"] = a.base
            warn = missing_env_warning(ck, base_env)
            if warn:
                print(warn, file=sys.stderr)
            bspec, _ = build(seed, constraints=cons, **base_env)
            b = evaluate(net, bspec, n=a.n_eval, seed=a.seed, dt_events=a.dt_events)
            print(f"  基准（pm_interval={a.base} · agv_n_failures=4）"
                  f"  ms={b['makespan']:7.2f}  E={b['energy']:6.3f}  twt={b['twt']:8.2f}",
                  flush=True)
            rec({"kind": "baseline", "ckpt": Path(path).name, "arm": ck["arm"],
                 "train_seed": seed, "n_eval": a.n_eval, **b})

            for pname, values in grid.items():
                for v in values:
                    if pname == "pm_interval" and float(v) == float(a.base):
                        continue                     # 基准已跑过
                    spec, _ = build(seed, constraints=cons,
                                    sim_overrides={pname: v}, **base_env)
                    r = evaluate(net, spec, n=a.n_eval, seed=a.seed, dt_events=a.dt_events)
                    d = {"kind": "point", "ckpt": Path(path).name, "arm": ck["arm"],
                         "train_seed": seed, "param": pname, "value": v,
                         **r,
                         # ⚠️ **百分比变化**，不是"敏感性系数" —— 不夸大。
                         "d_makespan_pct": float(100 * (r["makespan"] / b["makespan"] - 1)),
                         "d_energy_pct": float(100 * (r["energy"] / b["energy"] - 1)),
                         "d_twt_pct": float(100 * (r["twt"] / b["twt"] - 1))}
                    rec(d)
                    print(f"  {pname}={v!s:<7} ms={r['makespan']:7.2f}"
                          f" ({d['d_makespan_pct']:+6.1f}%)  "
                          f"E={r['energy']:6.3f} ({d['d_energy_pct']:+6.1f}%)  "
                          f"twt={r['twt']:8.2f} ({d['d_twt_pct']:+7.1f}%)", flush=True)

            # ⚠️ **`n_agv` 单列**：它换的是**车队**，布局的 V 段行数跟着变
            # （注意力能吃变长序列，故网络不炸；但这是"换个车队"不是"调个参数"）。
            for n_agv in (2, 3, 4):
                if n_agv == 3:
                    continue
                spec, _ = build(seed, constraints=cons, n_agv=n_agv, **base_env)
                r = evaluate(net, spec, n=a.n_eval, seed=a.seed, dt_events=a.dt_events)
                d = {"kind": "point", "ckpt": Path(path).name, "arm": ck["arm"],
                     "train_seed": seed, "param": "n_agv", "value": n_agv, **r,
                     "d_makespan_pct": float(100 * (r["makespan"] / b["makespan"] - 1)),
                     "d_energy_pct": float(100 * (r["energy"] / b["energy"] - 1)),
                     "d_twt_pct": float(100 * (r["twt"] / b["twt"] - 1))}
                rec(d)
                print(f"  n_agv={n_agv}      ms={r['makespan']:7.2f}"
                      f" ({d['d_makespan_pct']:+6.1f}%)  "
                      f"E={r['energy']:6.3f} ({d['d_energy_pct']:+6.1f}%)  "
                      f"twt={r['twt']:8.2f} ({d['d_twt_pct']:+7.1f}%)", flush=True)

    print(f"写出：{out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
