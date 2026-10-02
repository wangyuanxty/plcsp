"""约束配置（spec §3.3）：**十一个**约束的开关 + 五组消融配置。

设计：**开关是纯配置，不含逻辑**——`des.py` 读它决定行为。
5 组消融 = 5 个配置实例，**改配置不改代码**，避免消融时引入代码差异。

① 拥堵于 commit `eb1d1da` 按参数化粒度恢复（见 spec §3.3.1），故计入开关。
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ConstraintConfig:
    """十一个约束的开关。默认全开 = 论文主配置（spec §3.3）。"""
    # A 类：基建已有
    congestion: bool = True           # ① AGV 拥堵 / 区段冲突 / 死锁（粒度见 SimConfig.zone_granularity）
    finite_buffer: bool = True        # ② 有限缓冲（in_cap/out_cap）
    machine_failure: bool = True      # ③ 机器故障（泊松 + 中断恢复）
    # B 类：近乎免费
    rework: bool = True               # ④ 工件返工
    setup_time: bool = True           # ⑤ 换型 / 准备时间（顺序相关）
    fuzzy_processing: bool = True     # ⑥ 模糊加工时间（三角模糊数，**非**正态）
    due_dates: bool = True            # ⑧ 交期 / 拖期（τ·M_ref）
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
        return replace(self, **{n: False for n in names})


_FULL = ConstraintConfig()

ABLATION_GROUPS: dict[str, ConstraintConfig] = {
    "Full": _FULL,
    "-物流": _FULL.with_off("congestion", "agv_failure", "heterogeneous_fleet", "charging"),
    "-生产": _FULL.with_off("finite_buffer", "machine_failure", "rework",
                            "setup_time", "maintenance"),   # ② 机台缓冲属生产侧
    "-信息": _FULL.with_off("fuzzy_processing", "due_dates"),
    "None": _FULL.with_off("congestion", "finite_buffer", "machine_failure", "rework",
                           "setup_time", "fuzzy_processing", "due_dates",
                           "agv_failure", "heterogeneous_fleet", "charging", "maintenance"),
}
