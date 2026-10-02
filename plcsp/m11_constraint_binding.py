# -*- coding: utf-8 -*-
"""约束 binding 实测（闸 1 的实证版）：把已实现的约束推到极端，看指标动不动。

动机：P1a 的教训是"拥堵建完才发现不 binding"。同一把尺子（事件率 × 时长 = 期望事件数）
粗估显示 ③ 机器故障 在 MK10 上期望仅 ~0.12 次/episode、② 有限缓冲疑不 binding。
**本脚本用实测验证这些粗估**，避免在计划 3 里重蹈"先建后测"。

判据：把某约束的参数推到极端，若 makespan/energy **变化 < 2%**，判其**不 binding**。

注意：本脚本直接调内部件（`sample_layout` → 改参数 → `SimWorld.run`），
而非 `rollout()`——因为 `rollout` 在函数内 import，monkeypatch 打不中。

用法：python -m plcsp.m11_constraint_binding [--seeds 5]
输出：stdout 表格（不写文件）
"""
from __future__ import annotations

import argparse
import statistics as st

from .env.corridors import build_corridor_graph, dock_distance_matrix
from .env.des import SimConfig, SimWorld
from .env.instances import load_mk
from .env.layout import sample_layout

TARGETS = ("mk01", "mk10")


def run_variant(inst, cfg: SimConfig, seeds: int, *, mutate=None, seed_layout: int = 0):
    """采样布局 → 可选地改布局参数 → 跑 seeds 次。返回 (makespan, energy, tardy) 均值或 None。"""
    ms, en, td = [], [], []
    for s in range(seeds):
        lay = sample_layout(inst.n_machines, seed=seed_layout, aisle_w=cfg.aisle_width)
        if mutate is not None:
            mutate(lay)
        g = build_corridor_graph(lay)
        dm = dock_distance_matrix(g)
        r = SimWorld(inst, lay, dm, cfg).run(seed_chain=s)
        if r["horizon_hit"] or r["jobs_done"] != inst.n_jobs:
            continue
        ms.append(r["makespan"]); en.append(r["energy"]); td.append(r["tardy"])
    if not ms:
        return None
    return st.mean(ms), st.mean(en), st.mean(td)


def _mult_fail(mult: float):
    def _m(lay):
        for m in lay.machines:
            m.fail_rate *= mult
    return _m


def _cap_one(lay):
    for m in lay.machines:
        m.in_cap = m.out_cap = 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    for name in TARGETS:
        inst = load_mk(name)
        print(f"===== {name} =====")
        base = run_variant(inst, SimConfig(), args.seeds)
        if base is None:
            print("  基线无有效样本\n"); continue
        bm, be, bt = base
        # 期望故障次数（粗估）：机台数 × 平均故障率 × makespan 小时数
        avg_rate = 0.0015                                   # U(0, 0.003) 的均值
        exp_fail = inst.n_machines * avg_rate * (bm / 60.0)
        print(f"  基线                      makespan={bm:8.1f}  energy={be:8.1f}  tardy={bt:4.1f}")
        print(f"    └ 粗估期望故障次数/episode ≈ {exp_fail:.2f}")

        for mult in (10.0, 100.0, 1000.0):
            got = run_variant(inst, SimConfig(), args.seeds, mutate=_mult_fail(mult))
            if got is None:
                print(f"  fail_rate ×{mult:<7.0f}        （无有效样本）"); continue
            m, e, t = got
            d = (m - bm) / bm * 100
            flag = "  ← 不 binding" if abs(d) < 2.0 else ""
            print(f"  fail_rate ×{mult:<7.0f}        makespan={m:8.1f} ({d:+6.2f}%)"
                  f"  energy={e:8.1f} ({(e-be)/be*100:+6.2f}%){flag}")

        got = run_variant(inst, SimConfig(), args.seeds, mutate=_cap_one)
        if got is None:
            print("  缓冲 cap 全=1             （无有效样本）")
        else:
            m, e, t = got
            d = (m - bm) / bm * 100
            flag = "  ← 不 binding" if abs(d) < 2.0 else ""
            print(f"  缓冲 cap 全=1             makespan={m:8.1f} ({d:+6.2f}%)"
                  f"  energy={e:8.1f} ({(e-be)/be*100:+6.2f}%){flag}")
        print()


if __name__ == "__main__":
    main()
