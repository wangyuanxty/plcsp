# -*- coding: utf-8 -*-
"""网格环境下的 n_agv 标定（P1a 验收，2026-10-02 改版）。

⚠️ 判据已换：旧判据是"区段等待占比落在 15–35%"，但**拥堵机制已整条砍除**
（实测通道争用 ≤0.07%，见 progress-log §十五）。物流的结构性瓶颈是**车辆数量**（任务排队）。

新判据：**makespan 对车辆数的边际收益拐点** —— 车辆数增加带来的改善开始饱和的那个点。
再多加车只是浪费产能，不改善交付。

用法：python -m plcsp.m10_grid_calib [--seeds 10]
输出：stdout 表格（不写文件）
"""
from __future__ import annotations

import argparse
import statistics as st

from .env.des import SimConfig, rollout
from .env.instances import load_mk

INSTANCES = ("mk01", "mk07", "mk10")
AGV_GRID = (1, 2, 3, 4, 6, 8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    print("实例   n_agv  makespan(sd)        运输总时长  运输占比  相对上一档改善")
    for name in INSTANCES:
        inst = load_mk(name)
        prev: float | None = None
        for na in AGV_GRID:
            cfg = SimConfig(n_agv=na)
            ms, tv = [], []
            for s in range(args.seeds):
                r = rollout(inst, seed_chain=s, cfg=cfg)
                if r["horizon_hit"] or r["jobs_done"] != inst.n_jobs:
                    continue
                ms.append(r["makespan"]); tv.append(r["travel_time_total"])
            if not ms:
                print(f"{name:6} {na:5}  （无有效样本）")
                continue
            m, sd, t = st.mean(ms), st.pstdev(ms), st.mean(tv)
            imp = "—" if prev is None else f"{(prev - m) / prev * 100:+6.2f}%"
            print(f"{name:6} {na:5}  {m:8.1f} ({sd:5.1f})  {t:10.2f}  {100*t/m:7.2f}%  {imp}")
            prev = m
        print()


if __name__ == "__main__":
    main()
