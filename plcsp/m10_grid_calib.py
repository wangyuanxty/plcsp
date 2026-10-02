# -*- coding: utf-8 -*-
"""网格环境下的 n_agv 重标与拥堵实测（P1a 验收）。

背景：spec §3.3 的拥堵数据（n_agv=4 等待占比 32%）是在**旧的 line 单环**上测的；
换网格后 AGV 可绕行，该数会变，故 n_agv 必须重标（spec §6.3 已标注待重标）。

标定口径（spec §6.3）：取"拥堵显著但系统未过饱和"的工况——
**等待占比 15–35%，且 requeue ≈ 0**（不触发超时重试）。

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

    print("实例   n_agv  makespan      等待次数  等待总时长  等待占比  未释放  最长单次  requeue")
    for name in INSTANCES:
        inst = load_mk(name)
        for na in AGV_GRID:
            cfg = SimConfig(n_agv=na)
            ms, n_, tot, mx, rq = [], [], [], [], []
            leak = 0
            for s in range(args.seeds):
                r = rollout(inst, seed_chain=s, cfg=cfg)
                if r["horizon_hit"] or r["jobs_done"] != inst.n_jobs:
                    continue
                zw = r["zone_wait"]
                ms.append(r["makespan"]); n_.append(zw["n"])
                tot.append(zw["total"]); mx.append(zw["max"])
                rq.append(r["dbg"].get("requeue", 0))
                leak += 0 if r["zone_holders_free"] else 1
            if not ms:
                print(f"{name:6} {na:5}  （无有效样本）")
                continue
            m = st.mean(ms); t = st.mean(tot)
            print(f"{name:6} {na:5}  {m:11.1f}  {st.mean(n_):9.1f}  {t:10.2f}  "
                  f"{100*t/(m*na):7.2f}%  {leak:6d}  {st.mean(mx):8.2f}  {st.mean(rq):7.1f}")


if __name__ == "__main__":
    main()
