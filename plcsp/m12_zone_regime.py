# -*- coding: utf-8 -*-
"""区段争用的「binding 边界」扫描（P1a 补测）。

背景：默认参数下争用 ≤0.07%，但那是**一个工况点**。本脚本扫「运输负荷相对车队产能」这条轴，
找出**争用开始 binding 的阈值**——比"拥堵不重要"强得多的表述。

杠杆：**AGV 车速**（降速 = 运输时间变长 = 负荷上去；重载/安全限速，可辩护）
      × **车队规模**（车越多 → 同时在动的车越多 → 争用概率上去）

输出每格的：AGV 利用率、等待占比、区段粒度 node 下的等待。

用法：python -m plcsp.m12_zone_regime [--seeds 5]
"""
from __future__ import annotations

import argparse
import statistics as st

from .env.des import SimConfig, rollout
from .env.instances import load_mk

SPEEDS = (1.0, 0.5, 0.2, 0.1)      # m/s
AGVS = (2, 4, 8)
INSTANCES = ("mk10",)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    for name in INSTANCES:
        inst = load_mk(name)
        print(f"===== {name}（区段粒度 node，等待占比 = 争用）=====")
        print("  车速   车数   利用率    等待占比   最长等待   超时重试   makespan")
        for v in SPEEDS:
            for na in AGVS:
                cfg = SimConfig(n_agv=na, agv_speed_mps=v)
                util, wait, mx, rq, ms = [], [], [], [], []
                for s in range(args.seeds):
                    r = rollout(inst, seed_chain=s, cfg=cfg)
                    if r["horizon_hit"] or r["jobs_done"] != inst.n_jobs:
                        continue
                    util.append(r["travel_time_total"] / (r["makespan"] * na))
                    wait.append(r["zone_wait"]["total"] / (r["makespan"] * na))
                    mx.append(r["zone_wait"]["max"])
                    rq.append(r["dbg"].get("requeue", 0))
                    ms.append(r["makespan"])
                if not ms:
                    print(f"  {v:4.1f}  {na:3d}   （无有效样本）")
                    continue
                print(f"  {v:4.1f}  {na:3d}   {100*st.mean(util):6.2f}%  "
                      f"{100*st.mean(wait):8.3f}%  {st.mean(mx):8.2f}  "
                      f"{st.mean(rq):8.1f}  {st.mean(ms):9.1f}")
            print()


if __name__ == "__main__":
    main()
