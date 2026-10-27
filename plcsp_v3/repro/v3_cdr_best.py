# -*- coding: utf-8 -*-
"""**最好固定规则到底是多少**：36 条 CDR 的精度档扫描。

══ 为什么重测 ══

我们一直用 **260.0** 当"最好单条 CDR"。但那个数是**只测了 3 个规则对**得来的
（`EAJ+EAM+LATEST` / `EAJ+SMPT+MINCOST` / `EAJ+SMPT+LATEST`）。

2026-10-20 发现：初始化 ckpt 在精确执行下报 **248.4**，规则是
`EAJ+EAM + MINCOST` —— 它**不在**那 3 个对里。

⟹ **基线错了。** 本脚本重扫全部 36 条，口径与 `evaluate` 一致：
MK01–10 · layout 200+i · 场景 1 · 保养关 · **每条规则跑 T 条平局流**。

══ 两阶段（省时间）══

1. **粗扫**：36 条 × 10 实例 × `T1` 条平局流 → 前 `TOPN` 名；
2. **精测**：前 `TOPN` 条 × 10 实例 × `T2` 条平局流。

══ 用法 ══

`D:/anaconda/python.exe -m plcsp_v3.repro.v3_cdr_best`
环境变量：`V3_CDR_T1`（默认 3）· `V3_CDR_T2`（默认 20）· `V3_CDR_TOPN`（默认 5）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from plcsp_v3.env.constraints import ConstraintConfig        # noqa: E402
from plcsp_v3.env.reward import objective_vector              # noqa: E402
from plcsp_v3.hrl.cdr import cdr_pairs, run_cdr               # noqa: E402
from plcsp_v3.q1_flat import TIE_STREAM_BASE, build           # noqa: E402

MK_NAMES = tuple(f"mk{i:02d}" for i in range(1, 11))
T1 = int(os.environ.get("V3_CDR_T1", "3"))
T2 = int(os.environ.get("V3_CDR_T2", "20"))
TOPN = int(os.environ.get("V3_CDR_TOPN", "5"))
SC0 = 1
N_AGV = 3


def _specs():
    cons = ConstraintConfig(maintenance=False)
    out = []
    for i, name in enumerate(MK_NAMES):
        sp, _ = build(seed=0, instance=name, tau=None, pm_interval=30,
                      target_failures=4, n_agv=N_AGV, finite_buffer=True,
                      layout_seed=200 + i, reward_norm="minmax", constraints=cons)
        out.append(sp)
    return out


def _sweep(specs, pairs, T):
    """`pairs × 实例 × T` 条平局流 → 逐对的逐实例均值 `(n_pairs, 10, 3)`。"""
    out = np.zeros((len(pairs), len(specs), 3))
    for k, (pa, la) in enumerate(pairs):
        for i, sp in enumerate(specs):
            vals = [objective_vector(run_cdr(sp.world(), pa, la, seed_chain=SC0,
                                             tie_seed=TIE_STREAM_BASE + j))
                    for j in range(T)]
            out[k, i] = np.mean(vals, axis=0)
        print(f"    {k + 1}/{len(pairs)} {pa}+{la}  MS {out[k, :, 0].mean():.1f}",
              flush=True)
    return out


def main() -> None:
    specs = _specs()
    pairs = cdr_pairs(12)
    print(f"**36 条 CDR 精度扫描** · MK01–10 · layout 200+i · 场景 {SC0}"
          f" · 保养关\n  粗扫：{len(pairs)} 条 × {T1} 条平局流", flush=True)
    r1 = _sweep(specs, pairs, T1)
    order = np.argsort(r1[:, :, 0].mean(axis=1))
    print(f"\n  粗扫前 {TOPN}：", flush=True)
    for k in order[:TOPN]:
        print(f"    {pairs[k][0]}+{pairs[k][1]:<16} MS {r1[k, :, 0].mean():.1f}",
              flush=True)
    top = [pairs[k] for k in order[:TOPN]]
    print(f"\n  精测：前 {TOPN} 条 × {T2} 条平局流", flush=True)
    r2 = _sweep(specs, top, T2)
    print(f"\n  {'规则':<32}{'MS':>8}{'E':>8}{'TWT':>9}", flush=True)
    print("  " + "-" * 58, flush=True)
    best = None
    for k in np.argsort(r2[:, :, 0].mean(axis=1)):
        m = r2[k].mean(axis=0)
        print(f"  {top[k][0] + '+' + top[k][1]:<32}{m[0]:>8.1f}{m[1]:>8.2f}"
              f"{m[2]:>9.1f}", flush=True)
        if best is None:
            best = m
    print(f"\n  **最好固定规则 MS = {best[0]:.1f}**（口径：{T2} 条平局流）",
          flush=True)


if __name__ == "__main__":
    main()
