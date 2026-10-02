# -*- coding: utf-8 -*-
"""约束 binding 实测工具（P1b Task 3）：**每个约束接入后立刻量它动不动**。

动机（P1a 的教训）："拥堵建完才发现不 binding"。同一把尺子（事件率 × 时长 = 期望事件数）
粗估显示 ③ 机器故障在 MK10 上期望仅 ~0.12 次/episode——**而实测 ×10 就有 +9.8%（粗估错了）**。
故本工具只信实测。

## 两列，答两个不同的问题

| 列 | 问的是 | 依据 |
|---|---|---|
| **关态** | 该约束**在默认参数下**是否 binding | `ConstraintConfig.with_off(name)`——就是消融表那一下 |
| **极端档** | 把该约束的参数**推到极端**是否 binding | 布局/配置参数外推，不改 `des.py` 逻辑 |

两列都 < 2% 才是"这个约束在本工况下真的无所谓"；只有关态 < 2% 而极端档 ≫ 2%，说明是**默认参数太小**，
不是机制无关——机器故障就是这一类。

**关态的附加作用**：未接线的开关在关态会给出**恰好 0.00%**（配置变了但没人读它）。
这是 Task 4–6 的待办清单，也是"关了但没真关"（Review Focus #2）的自动探针。

## 噪声地板（重要）

不同配置下随机数流的消耗长度不同，**同种子 ≠ 共同随机数**（Q2，见 progress-log §十六）。
故关态 Δ 里含"流移位"噪声——实测过 ±1.5% 量级且方向不定。故一并打印**种子间标准差**：
`|Δ|` 没超过 `σ` 就别当信号读。

用法：
    python -m plcsp.m11_constraint_binding --all --seeds 5
    python -m plcsp.m11_constraint_binding --constraint charging --constraint maintenance
"""
from __future__ import annotations

import argparse
import math
import statistics as st
from typing import Callable

from .env.constraints import ConstraintConfig
from .env.corridors import build_corridor_graph, dock_distance_matrix
from .env.des import SimConfig, SimWorld
from .env.instances import load_mk
from .env.layout import sample_layout

TARGETS = ("mk01", "mk10")
BINDING_PCT = 2.0          # 判据（计划 Global Constraints）：|Δ%| < 2 ⟹ 不 binding
# ⚠️ 已接入 `des.py` 的约束——用于消歧"关态零影响"是"从不触发"还是"未接线"。
# **Task 4/5/6 每接一个都要把它加进来**，否则该约束会被误报成"未接线"。
WIRED = ("congestion", "finite_buffer", "machine_failure")

CONSTRAINTS = ("congestion", "finite_buffer", "machine_failure", "rework", "setup_time",
               "fuzzy_processing", "due_dates", "agv_failure", "heterogeneous_fleet",
               "charging", "maintenance")
LABELS = {"congestion": "①拥堵", "finite_buffer": "②有限缓冲", "machine_failure": "③机器故障",
          "rework": "④返工", "setup_time": "⑤换型", "fuzzy_processing": "⑥模糊加工",
          "due_dates": "⑧交期", "agv_failure": "⑨AGV故障", "heterogeneous_fleet": "⑩异构车队",
          "charging": "⑪充电", "maintenance": "⑫预防性维护"}


# ── 极端档探针：只改布局/配置参数，**不改 des.py 逻辑** ──
# 签名 (lay, cfg) -> None，就地修改。未接入的约束没有探针（会打印"探针缺失"）。

def _probe_machine_failure(lay, cfg) -> None:
    for m in lay.machines:
        m.fail_rate *= 100.0            # 期望次数从 ~0.1 拉到 ~10


def _probe_finite_buffer(lay, cfg) -> None:
    for m in lay.machines:
        m.in_cap = m.out_cap = 1        # 极端：每台机只有 1 个位置


def _probe_congestion(lay, cfg) -> None:
    cfg.zone_granularity = "all"        # 极端：全图一个区段（争用最强档）


PROBES: dict[str, tuple[str, Callable]] = {
    "machine_failure": ("fail_rate ×100", _probe_machine_failure),
    "finite_buffer": ("缓冲 cap 全=1", _probe_finite_buffer),
    "congestion": ("区段粒度 all", _probe_congestion),
}


def run_variant(inst, cfg: SimConfig, constraints: ConstraintConfig, seeds: int,
                *, mutate: Callable | None = None, seed_layout: int = 0) -> dict | None:
    """同布局同种子跑 seeds 次。**逐种子保留指标**（只留均值就算不出种子间 σ）。"""
    ms, en, td = [], [], []
    for s in range(seeds):
        lay = sample_layout(inst.n_machines, seed=seed_layout, aisle_w=cfg.aisle_width)
        if mutate is not None:
            mutate(lay, cfg)
        g = build_corridor_graph(lay)
        dm = dock_distance_matrix(g)
        r = SimWorld(inst, lay, dm, cfg, graph=g, constraints=constraints).run(seed_chain=s)
        if r["horizon_hit"] or r["jobs_done"] != inst.n_jobs:
            continue
        ms.append(r["makespan"]); en.append(r["energy"]); td.append(r["tardy"])
    if not ms:
        return None
    return {"makespan": ms, "energy": en, "tardy": td}


def _paired_delta(base: dict, var: dict) -> tuple[float, float]:
    """配对差值（逐种子）→ (均值%, 种子间 σ%)。σ 是噪声地板的估计。"""
    ds = [(v - b) / b * 100.0 for b, v in zip(base["makespan"], var["makespan"])]
    return st.mean(ds), (st.stdev(ds) if len(ds) > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--constraint", action="append", default=None,
                    help="只测这些约束（可重复）；缺省配合 --all")
    ap.add_argument("--all", action="store_true", help="测全部约束")
    ap.add_argument("--instances", default=None,
                    help="逗号分隔的实例名；缺省 mk01,mk10")
    args = ap.parse_args()

    if not args.constraint and not args.all:
        ap.error("须给 --all 或至少一个 --constraint")
    todo = list(CONSTRAINTS) if args.all else list(args.constraint)
    unknown = [c for c in todo if c not in CONSTRAINTS]
    if unknown:
        ap.error(f"未知约束：{unknown}；可选 {list(CONSTRAINTS)}")
    targets = tuple(args.instances.split(",")) if args.instances else TARGETS

    for name in targets:
        inst = load_mk(name)
        print(f"===== {name}（{inst.n_machines} 机位，{args.seeds} 种子）=====")
        base = run_variant(inst, SimConfig(), ConstraintConfig(), args.seeds)
        if base is None:
            print("  基线无有效样本\n"); continue
        print(f"  基线  makespan={st.mean(base['makespan']):7.1f}  "
              f"energy={st.mean(base['energy']):6.2f} kWh  tardy={st.mean(base['tardy']):4.2f}")
        print(f"  {'约束':<14}{'关态 Δmks':>12}{'(σ)':>8}   {'极端档':<14}{'Δmks':>9}   判定")

        for c in todo:
            off = ConstraintConfig().with_off(c)
            go = run_variant(inst, SimConfig(), off, args.seeds)
            if go is None:
                print(f"  {LABELS[c]:<14}{'（无有效样本）':>12}")
                continue
            d_off, sd_off = _paired_delta(base, go)

            if c in PROBES:
                label, probe = PROBES[c]
                ge = run_variant(inst, SimConfig(), ConstraintConfig(), args.seeds, mutate=probe)
                if ge is None:
                    ext_lab, d_ext = label, float("nan")
                else:
                    ext_lab, d_ext = label, _paired_delta(base, ge)[0]
            else:
                ext_lab, d_ext = "探针缺失", float("nan")

            # 判定：两列都 < 2% ⟹ 不 binding（计划 Global Constraints 的判据）
            has_ext = not math.isnan(d_ext)
            worst = max(abs(d_off), abs(d_ext) if has_ext else 0.0)
            if d_off == 0.0 and (not has_ext or abs(d_ext) < BINDING_PCT):
                # 关态**逐位相同** = 该约束没改变任何结果。两种可能，本工具分不出来：
                # (a) 开关未接线（Task 4–6 的待办）；(b) 已接线但本工况下从不触发。
                # 用 WIRED 常量消歧：在 WIRED 里 = (b)，不在 = (a)。
                verdict = "⚠ 未接线" if c not in WIRED else "关态零影响"
            elif worst < BINDING_PCT:
                verdict = "不 binding"
            else:
                verdict = "binding"
            ext_s = "     n/a" if not has_ext else f"{d_ext:+8.2f}%"
            print(f"  {LABELS[c]:<14}{d_off:+11.2f}%{sd_off:7.2f}   {ext_lab:<14}{ext_s}   {verdict}")
        print()


if __name__ == "__main__":
    main()
