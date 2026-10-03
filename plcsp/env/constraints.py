"""约束配置（spec §3.3）：**十一个**约束的开关 + 五组消融配置。

设计：**开关是纯配置，不含逻辑**——`des.py` 读它决定行为。
5 组消融 = 5 个配置实例，**改配置不改代码**，避免消融时引入代码差异。

① 拥堵于 commit `eb1d1da` 按参数化粒度恢复（见 spec §3.3.1），故计入开关。
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ConstraintConfig:
    """十个约束的开关。默认全开 = 论文主配置（spec §3.3）。

    ⑥ 模糊加工**已砍**（2026-10-03，实测不 binding）——故为 10 而非 11。
    """
    # A 类：基建已有
    congestion: bool = True           # ① AGV 拥堵 / 区段冲突 / 死锁（粒度见 SimConfig.zone_granularity）
    finite_buffer: bool = True        # ② 有限缓冲（in_cap/out_cap）
    machine_failure: bool = True      # ③ 机器故障（泊松 + 中断恢复）
    # B 类：近乎免费
    rework: bool = True               # ④ 工件返工
    setup_time: bool = True           # ⑤ 换型 / 准备时间（顺序相关）
    due_dates: bool = True            # ⑧ 交期 / 拖期（TF/RDD：LB·τ·(1+R(2ρ−1))，见 due_dates.py）
    agv_failure: bool = True          # ⑨ AGV 故障
    # C 类：真花钱
    heterogeneous_fleet: bool = True  # ⑩ 异构车队（载重/速度/多载量）
    charging: bool = True             # ⑪ 充电 / 电量
    maintenance: bool = True          # ⑫ 预防性维护（计划性停机）

    def with_off(self, *names: str) -> "ConstraintConfig":
        """返回**关掉指定开关**的副本（不可变风格，不原地改）。

        用法：`cfg.with_off("rework", "setup_time")`。
        取 `*names` 而非 `**kw` 是刻意的——后者要写 `with_(rework=True)` 才能关，
        值被忽略、靠 key 表意，极易读反。
        """
        for n in names:
            if not hasattr(self, n):
                raise ValueError(f"未知约束开关：{n}")
        return replace(self, **dict.fromkeys(names, False))


_FULL = ConstraintConfig()

ABLATION_GROUPS: dict[str, ConstraintConfig] = {
    "Full": _FULL,
    "-物流": _FULL.with_off("congestion", "agv_failure", "heterogeneous_fleet", "charging"),
    "-生产": _FULL.with_off("finite_buffer", "machine_failure", "rework",
                            "setup_time", "maintenance"),   # ② 机台缓冲属生产侧
    "-信息": _FULL.with_off("due_dates"),
    "None": _FULL.with_off("congestion", "finite_buffer", "machine_failure", "rework",
                           "setup_time", "due_dates",
                           "agv_failure", "heterogeneous_fleet", "charging", "maintenance"),
}

# ── P4-B：两档报告（用户裁定 2026-10-03）──
# 档 A「MKT 口径」= 文献（MKT = FJSP + 运输）没有对应项的机制全关 **⟹ 恰好等于 None 组**，
#   故**复用**它而不是新造配置（单一真相；改动只走 `with_off`）。
# 档 B「完整口径」= 机制全开，运输口径**相同**（都是行程时间矩阵）——两档的差别**只在机制**。
# ⚠️ 十个机制 ①拥堵 ②有限缓冲 ③机台故障 ④返工 ⑤换型 ⑧交期 ⑨AGV故障 ⑩异构车队 ⑪充电 ⑫维护
#   在 MKT 里一个都没有对应项。将来加第 11 个机制时若它 MKT 也没有对应项，
#   `REPORT_TIERS["A-MKT"]` 必须同步——`test_two_tier_report.py` 按 **dataclass 字段**逐个查，
#   不依赖本注释或任何硬编码的名字表。
REPORT_TIERS: dict[str, ConstraintConfig] = {
    "A-MKT": ABLATION_GROUPS["None"],
    "B-Full": ABLATION_GROUPS["Full"],
}
