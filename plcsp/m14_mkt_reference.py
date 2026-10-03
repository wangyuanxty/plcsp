# -*- coding: utf-8 -*-
"""MKT 上跑我们的参考调度，与已发表数字并列（P4-A Task 3）。

⚠️ **本脚本不把 trans_time 接进仿真**（那是 P4-B 的设计决定）——故 `ours` 是
**原始 MK 几何**下跑出来的，只在**量级**上与已发表数字可比。见计划「已知边界」。
输出 stdout 表，**不写文件**。

⚠️ 术语：本模块描述自己的行为用平实说法——「与已发表数字并列」「复现保真度」，
**不用"对拍"**（用户明令：不自造术语）。

口径警示（Review Focus #3）：
- `benchmark` = **对照列**（HGS / HA-DQN / HF2021）的口径 = **MKT**。
- `ours_benchmark` = **我们那一列**跑在什么口径上。本批 = `"MK-raw-geometry"`（trans_time 未接入）。
- **两列不同口径 ⟹ 本表的 `ours` 与对照列不可直接比**，只判量级。
  P4-B 接上 trans_time 后须把 `ours_benchmark` 改为 `"MKT"`——
  `plcsp/tests/test_mkt_reference.py::test_our_column_declares_its_own_basis_and_is_not_mkt_yet`
  就是为此设的绊线（到时会红，逼人同步改标签）。
"""
from __future__ import annotations

import argparse
import statistics as st

from .env.des import SimConfig, rollout
from .env.instances import load_mk
from .env.mkt import MKT_PUBLISHED

# `ours` 那一列的口径标签。**接上 trans_time（P4-B）后必须改成 "MKT"**（有绊线测试盯着）。
OURS_BENCHMARK_RAW_MK = "MK-raw-geometry"


def reference_table(names: tuple[str, ...], n_agv: int | None = None,
                    seeds: int = 3) -> list[dict]:
    """每实例一行：我们（参考调度）+ 已发表数字。

    `n_agv=None` → 该实例的 **v = m**（HGS/HA-DQN 口径）。表里**必带车辆数列**，
    且 `benchmark` 自报对照列是 MKT、`ours_benchmark` 自报我们那列的口径——防与原始 MK 混表。

    ⚠️ 参考调度 = `rollout` 的默认口径（每工序取最短候选 + AGV 轮询派车），**未训练**；
       它是"链路能不能跑、量级对不对"的探针，不是我们的方法。
    """
    rows: list[dict] = []
    for name in names:
        base = load_mk(name)
        v = base.n_machines if n_agv is None else int(n_agv)
        cfg = SimConfig(n_agv=v)
        got = [rollout(base, seed_chain=s, cfg=cfg)["makespan"] for s in range(seeds)]
        rows.append({"inst": name, "benchmark": "MKT", "n_agv": v,
                     "ours": float(st.mean(got)),
                     "ours_benchmark": OURS_BENCHMARK_RAW_MK,
                     **MKT_PUBLISHED[name]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="mk01,mk07,mk10")
    ap.add_argument("--n-agv", default="default",
                    help='"default" = v=m（HGS/HA-DQN 口径）；也可给整数（2 = HF2021 口径）')
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    v = None if args.n_agv == "default" else int(args.n_agv)
    rows = reference_table(tuple(args.instances.split(",")), n_agv=v, seeds=args.seeds)
    print(f"⚠️ ours 口径 = {OURS_BENCHMARK_RAW_MK}（trans_time 未接入仿真，P4-B 才接）"
          f"｜对照列口径 = MKT｜**两者不同口径，只判量级，不可直接比**")
    print(f"{'实例':<7}{'基准':<6}{'车辆数':>7}{'我们(参考)':>12}{'HGS':>9}{'HA-DQN':>9}{'HF2021*':>10}")
    print("  * HF2021 的 LAHC 用 2 台车，与其余列**不同口径**，不可直接比（§19.7d 存疑点②）")
    for r in rows:
        print(f"{r['inst']:<7}{r['benchmark']:<6}{r['n_agv']:>7}{r['ours']:>12.1f}"
              f"{r['HGS_JMS2024']:>9.1f}{r['HA_DQN_CIS2025']:>9.1f}{r['HF2021_LAHC']:>10.1f}")


if __name__ == "__main__":
    main()
