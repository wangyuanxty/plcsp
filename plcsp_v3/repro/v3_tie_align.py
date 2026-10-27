# -*- coding: utf-8 -*-
"""**平局对齐 + 评测精度**：同一条规则，多种执行路径，T 条平局流。

══ 为什么要测 ══

此前固定规则对照表混了三件事（`progress-log.md` §38.23）：

| 路径 | 平局怎么断 | 候选怎么来 |
|---|---|---|
| CDR（`cdr.run_cdr`） | 环境 `tie_rng`，**随机** | 全池 |
| 策略 `force_top1` | 确定性：排序后取第 0 个 | 规则排序的前 `K_PA=16` |
| 策略 `exact_rule` | 环境 `tie_rng`，**随机** | 全池 |

`force_top1` 的 index 0 **不是规则的 argmin**。`exact_rule` 直接调
`rules.apply_pa_rule` + 环境 `tie_rng` —— 与 CDR **逐位同一条代码路径**。

══ 评测精度 ══

同一规则、同一场景，只换平局流：逐实例 std **17.8 min**（§38.23 实测）。
`n=2` 个场景的 SE ≈ 12.6 ⟹ **测不出 8 min 的效应**。
本脚本对**每一个实例**跑 `T` 条平局流（`V3_TIE_STREAMS`，默认 20），
报**逐实例均值的跨实例均值 ± 标准误**。

⚠️ 场景（`seed_chain`）固定为 `SC0`：本脚本只控**平局**这一条随机源。
跨场景方差另行用 `n>1` 控制（见 `evaluate(..., n=..., tie_streams=...)`）。

══ 口径 ══

MK01–10 · `layout_seed=200+i` · 场景 `seed_chain=1` · **保养关**
· 三目标 `(MS, Energy, TWT)`。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from plcsp_v3.env.constraints import ConstraintConfig          # noqa: E402
from plcsp_v3.env.reward import objective_vector                # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy              # noqa: E402
from plcsp_v3.hrl.cdr import run_cdr                            # noqa: E402
from plcsp_v3.q1_flat import TIE_STREAM_BASE, build, evaluate    # noqa: E402

MK_NAMES = tuple(f"mk{i:02d}" for i in range(1, 11))
SC0 = 1
N_SC = 1
DT = 20
N_AGV = 3
T = int(os.environ.get("V3_TIE_STREAMS", "20"))

PAIRS = (
    ("EAJ+EAM", "LATEST_IN_TIME"),
    ("EAJ+SMPT", "MINCOST"),
    ("EAJ+SMPT", "LATEST_IN_TIME"),
)


def _specs():
    cons = ConstraintConfig(maintenance=False)
    out = []
    for i, name in enumerate(MK_NAMES):
        sp, _ = build(seed=0, instance=name, tau=None, pm_interval=30,
                      target_failures=4, n_agv=N_AGV, finite_buffer=True,
                      layout_seed=200 + i, reward_norm="minmax",
                      constraints=cons)
        out.append((name, sp))
    return out


def _fixed(pa: str, la: str):
    from plcsp_v3.hrl.rules import LA_RULE_NAMES, PA_RULE_NAMES
    return {"PA": PA_RULE_NAMES.index(pa), "LA": LA_RULE_NAMES.index(la),
            "RA": 0}


def _eval(net, sps, rules, **kw) -> np.ndarray:
    """逐实例：T 条平局流的均值 → `(10, 3)`。"""
    rows = []
    for _n, sp in sps:
        r = evaluate(net, sp, n=N_SC, seed=SC0, dt_events=DT,
                     fixed_rules=rules, tie_streams=T, **kw)
        rows.append((r["makespan"], r["energy"], r["twt"]))
    return np.asarray(rows, dtype=float)


def _cdr(sps, pa, la) -> np.ndarray:
    rows = []
    for _n, sp in sps:
        vals = [objective_vector(run_cdr(sp.world(), pa, la, seed_chain=SC0,
                                         tie_seed=TIE_STREAM_BASE + j))
                for j in range(T)]
        rows.append(np.mean(vals, axis=0))
    return np.asarray(rows, dtype=float)


def _ms(m: np.ndarray) -> str:
    """`(10,3)` → `MS均值 ± 跨实例SE`。"""
    se = m[:, 0].std(ddof=1) / np.sqrt(len(m))
    return f"{m[:, 0].mean():7.1f} ±{se:5.1f}"


def main() -> None:
    t0 = time.perf_counter()
    sps = _specs()
    torch.manual_seed(0)
    net = HierarchicalPolicy()
    net.eval()
    print(f"**平局对齐 · 精度档** · MK01–10 · layout 200+i · 场景 {SC0}"
          f" · 保养关 · **{T} 条平局流**\n", flush=True)
    print(f"  {'rule pair':<32}{'path':<9}{'MS (mean±SE)':>17}"
          f"{'E':>8}{'TWT':>9}", flush=True)
    print("  " + "-" * 78, flush=True)
    for pa, la in PAIRS:
        rules = _fixed(pa, la)
        cdr = _cdr(sps, pa, la)
        ex = _eval(net, sps, rules, exact_rule=True)
        t1 = _eval(net, sps, rules, force_top1=True)
        for tag, m in (("CDR", cdr), ("exact", ex), ("top1", t1)):
            print(f"  {pa + '+' + la:<32}{tag:<9}{_ms(m):>17}"
                  f"{m[:, 1].mean():>8.2f}{m[:, 2].mean():>9.1f}", flush=True)
        d = ex[:, 0] - cdr[:, 0]
        d2 = t1[:, 0] - ex[:, 0]
        se_d = d.std(ddof=1) / np.sqrt(len(d))
        se_d2 = d2.std(ddof=1) / np.sqrt(len(d2))
        print(f"  {'':<42}exact−CDR {d.mean():+7.2f} (SE {se_d:4.2f})"
              f"　逐实例相同 {int((np.abs(d) < 1e-9).sum())}/{len(d)}",
              flush=True)
        print(f"  {'':<42}top1−exact {d2.mean():+7.2f} (SE {se_d2:4.2f})",
              flush=True)
    print(f"\n  总用时 {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
