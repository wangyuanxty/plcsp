# -*- coding: utf-8 -*-
"""MKT 上跑我们的参考调度，与已发表数字并列（P4-A Task 3；P4-B Task 4 改为**两档**）。

⚠️ **P4-B 起运输走 MKT 行程时间矩阵**（`load_mkt` 造的实例自带 matrix 口径）——本表的
`ours` 与对照列**同口径**，两列可以直接比。P4-A 的「ours 跑在原始 MK 几何上 ⟹ 两列不同口径、
只判量级」告诫**已作废**。

**两档（用户裁定 2026-10-03）**：每一 (实例 × 档) 一行。

- 档 A `A-MKT` = `ABLATION_GROUPS["None"]`——MKT（FJSP + 运输）**没有对应项**的机制全关，
  是**退化特例**，与已发表数字同口径；
- 档 B `B-Full` = `ABLATION_GROUPS["Full"]`——机制全开。
  两档的差别**只在机制**，运输口径相同（都是矩阵）。

⚠️ 档 B 开着 ⑪ 充电，而充电桩在矩阵里**没有对应项** ⟹ 显式选 `transport_unmapped="geometry"`
（**声明式**几何降级），并把未映射路段的**段数与分钟数逐行带出**——论文必须如实声明
「这一档里有 x% 的行程是几何口径补的」，不得说成「运输全部来自矩阵」。
⚠️ `horizon_hit` 逐行带出：矩阵口径下 makespan 大数倍，掐表会伪装成「更慢」。

⚠️ 术语：本模块描述自己的行为用平实说法——「与已发表数字并列」「复现保真度」，
**不用"对拍"**（用户明令：不自造术语）。

口径声明：
- `benchmark` = **对照列**（HGS / HA-DQN / HF2021）的口径 = **MKT**。
- `ours_benchmark` = **我们那一列**跑在什么口径上 = **MKT**（P4-B 已把矩阵接进仿真）。
  P4-A 埋的绊线 `test_our_column_declares_its_own_basis_and_is_not_mkt_yet` 已按它自己写明的
  要求**翻转**为 `test_our_column_declares_the_mkt_basis`（**改写非删除**，计数不减）。
"""
from __future__ import annotations

import argparse
import statistics as st

from .env.constraints import REPORT_TIERS
from .env.des import SimConfig, rollout
from .env.mkt import MKT_PUBLISHED, MKT_PUBLISHED_AGV, load_mkt

# `ours` 那一列的口径标签（P4-B：矩阵已接进仿真 ⟹ 与对照列同为 MKT）。
OURS_BENCHMARK = "MKT"


def _fleet_spec_str(spec: str | int, m: int) -> str:
    """车数设定的可读写法：`"m"` → `v=m(=6)`；整数 → `v=2`。"""
    return f"v=m(={m})" if spec == "m" else f"v={int(spec)}"


def _same_fleet(spec: str | int, v: int, m: int) -> bool:
    """该对照列的车数是否与 `ours`（本表用的是 `v`）**一致**。

    ⚠️ 一致才可比——不一致的列并排摆着只是参考，不能当成同口径对手（Review Focus #1）。
    """
    return v == m if spec == "m" else int(spec) == v


def reference_table(names: tuple[str, ...], n_agv: int | None = None, seeds: int = 3,
                    tiers: tuple[str, ...] = ("A-MKT", "B-Full")) -> list[dict]:
    """每 (实例 × 档) 一行：我们（参考调度）+ 已发表数字。

    `n_agv=None` → 该实例的 **v = m**（HGS/HA-DQN 口径）。表里**必带车辆数列**，
    且 `benchmark` 自报对照列是 MKT、`ours_benchmark` 自报我们那列的口径——防与原始 MK 混表。
    `comparable_fleet` 是**逐列算出来的**同车数判定（不是写死的脚注）：摘要表里
    HF2021 是异类，切到 `--n-agv 2` 后异类变成 HGS/HA-DQN——写死的脚注会**恰好说反**。

    ⚠️ 行程时间**一律来自矩阵**（`load_mkt` 造的实例自带 matrix 口径）。
    ⚠️ 档 B 开着 ⑪ 充电，而充电桩在矩阵里**没有对应项** ⟹ 显式选
       `transport_unmapped='geometry'` 并把段数/分钟数带进行里（**声明式**降级，不静默）。
    ⚠️ `horizon_hit` 必须随行带出：矩阵口径下 makespan 大数倍，掐表会伪装成「更慢」。
    ⚠️ 参考调度 = `rollout` 的默认口径（每工序取最短候选 + AGV 轮询派车），**未训练**；
       它是「链路能不能跑、量级对不对」的探针，不是我们的方法。
    """
    rows: list[dict] = []
    for name in names:
        mkt = load_mkt(name, n_agv=n_agv)
        v, m = mkt.n_agv, mkt.layout_m
        for tier in tiers:
            cfg = SimConfig(n_agv=v, transport_unmapped="geometry")
            got = [rollout(mkt.base, seed_chain=s, cfg=cfg,
                           constraints=REPORT_TIERS[tier]) for s in range(seeds)]
            ms = [r["makespan"] for r in got]
            rows.append({"inst": name, "tier": tier, "benchmark": "MKT",
                         "transport": got[0]["transport"],      # ⚠️ 取自仿真自报，不写死
                         "n_agv": v, "layout_m": m,
                         "ours": float(st.mean(ms)), "ours_benchmark": OURS_BENCHMARK,
                         # 三目标里的能耗（kWh）逐行带出——两档的运输/约束差别必须看得到
                         "energy": float(st.mean([r["energy"] for r in got])),
                         "unmapped_legs": sum(r["unmapped_legs"] for r in got),
                         "unmapped_min": sum(r["unmapped_min"] for r in got),
                         "horizon_hit": any(r["horizon_hit"] for r in got),
                         "published_agv": dict(MKT_PUBLISHED_AGV),
                         "comparable_fleet": {k: _same_fleet(s, v, m)
                                              for k, s in MKT_PUBLISHED_AGV.items()},
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
    print(f"⚠️ 口径声明：本表两档都跑在**行程时间矩阵**（MKT）上；ours 口径 = {OURS_BENCHMARK}，"
          f"与对照列**同口径**（P4-A 的「两列不同口径、只判量级」告诫已作废）")
    print("档 A-MKT = ABLATION_GROUPS['None']（MKT 没有对应项的机制全关，退化特例）；"
          "档 B-Full = 机制全开。两档的差别**只在机制**，运输口径相同。")
    print("⚠️ 未映射路段：档 B 的充电桩在矩阵里没有对应项 ⟹ 那些腿用**几何口径**补"
          "（声明式降级）。下表逐行给段数与分钟数——论文须如实声明其占比。")
    print(f"{'实例':<7}{'档':<8}{'基准':<6}{'车辆数':>7}{'我们(参考)':>12}"
          f"{'未映射(段/min)':>16}{'掐表':>6}{'HGS':>9}{'HA-DQN':>9}{'HF2021':>10}")
    for r in rows:
        print(f"{r['inst']:<7}{r['tier']:<8}{r['benchmark']:<6}{r['n_agv']:>7}{r['ours']:>12.1f}"
              f"{r['unmapped_legs']:>5}/{r['unmapped_min']:<10.1f}{str(r['horizon_hit']):>6}"
              f"{r['HGS_JMS2024']:>9.1f}{r['HA_DQN_CIS2025']:>9.1f}{r['HF2021_LAHC']:>10.1f}")
        m = r["layout_m"]
        specs = " / ".join(f"{k}({_fleet_spec_str(s, m)})" for k, s in r["published_agv"].items())
        ok = [k for k, same in r["comparable_fleet"].items() if same]
        bad = [k for k, same in r["comparable_fleet"].items() if not same]
        print(f"    车数：ours v={r['n_agv']}｜{specs}")
        print(f"    与 ours **同车数**（口径可比的对照）：{ok or '（无）'}")
        print(f"    与 ours **不同车数**（只作参考，不可当同口径对手）：{bad or '（无）'}")


if __name__ == "__main__":
    main()
