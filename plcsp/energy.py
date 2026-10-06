# -*- coding: utf-8 -*-
"""M2 能耗模型（spec §3.4）——参数全部引证 GFJSPT-MMRS（SWEVO 99:102181, 2025）。

单位：功率 [kW]，时长 [min]，能耗 [kWh]。
**换算关键：kWh = kW × (min / 60)** —— 这是本模块唯一易错处（漏 /60 会差 60 倍）。

机床按**三态**计：待机 / 加工 / 换型。AGV 按**三态**计：待机 / 空载 / 负载。
车间另有固定功率 P₀（照明/空压等），按 makespan 计。

两处**须在论文里声明的假设**（见 spec §9 的 assumed 表）：
1. **换型功率 = 待机功率**——原文献未单列换型功率。换型时主轴不切削但机床通电，取空载值是下界。
2. **取档内低速值**——原文献每档给了两个转速档（如 M1–M4 为 1500/4500 rpm）对应的功率，
   我们不建模主轴转速，故取**较低值**（保守下界）。`MACHINE_TIERS` 保留了区间端点，
   敏感性分析可直接换成高端值。
"""
from __future__ import annotations

# ── 引证常量（GFJSPT-MMRS, SWEVO 99:102181, 2025；spec §3.4 表）──
# 机床三档（按原文献机位分组）：rpm 区间 + 切削/空载功率区间 [kW]
MACHINE_TIERS: dict[str, dict[str, tuple[int, int] | tuple[float, float]]] = {
    "M1_M4": {"rpm": (1500, 4500), "proc": (0.951, 1.17), "idle": (0.74, 0.90)},
    "M5_M6": {"rpm": (2000, 6000), "proc": (0.30, 0.60), "idle": (0.24, 0.50)},
    "M7_M10": {"rpm": (1000, 4000), "proc": (0.20, 0.56), "idle": (0.16, 0.36)},
}
# 档位边界：原文献 10 台机床分组为 M1–M4 / M5–M6 / M7–M10（0-based 机位，逐字照抄）
_PAPER_TIERS = ("M1_M4", "M1_M4", "M1_M4", "M1_M4",
                "M5_M6", "M5_M6",
                "M7_M10", "M7_M10", "M7_M10", "M7_M10")
# ⚠️ 外推部分（assumed）：原文献只有 10 台机位的参数表，但 **MK10 有 15 台**。
# 第 11 台起按三档**循环**指派（M1_M4 → M5_M6 → M7_M10 → …）。这是本模块唯一的非引证假设，
# 已列 spec §9 的 assumed 表；敏感性分析应覆盖"档位指派方式"对能耗的影响。
_CYCLE_TIERS = ("M1_M4", "M5_M6", "M7_M10")
MAX_MACHINES = len(_PAPER_TIERS) + len(_CYCLE_TIERS) * 2      # 16：覆盖到 MK 实例族上限


def _tier_of(idx: int) -> str:
    """0-based 机位 → 档位名。前 10 台照抄原文献，其后循环（见 `_CYCLE_TIERS` 注释）。"""
    if idx < len(_PAPER_TIERS):
        return _PAPER_TIERS[idx]
    return _CYCLE_TIERS[(idx - len(_PAPER_TIERS)) % len(_CYCLE_TIERS)]

# AGV 三态 [kW]
AGV_IDLE_KW, AGV_EMPTY_KW, AGV_LOADED_KW = 0.1, 0.2, 0.5
# 车间固定功率 [kW]
SHOP_FIXED_KW = 0.4


def _kwh(power_kw: float, minutes: float) -> float:
    """kW × min → kWh。**漏掉 /60 是 60 倍错误。**"""
    return power_kw * (minutes / 60.0)


def machine_params_for(idx: int, n_machines: int) -> dict[str, float | str]:
    """机位号 → 该机台的能耗参数。

    机位分组**沿用原文献**：第 1–4 台取 M1_M4 档、5–6 台取 M5_M6、7–10 台取 M7_M10；
    第 11 台起为**外推**（三档循环，见 `_CYCLE_TIERS`）。超出 `MAX_MACHINES` 显式报错。
    """
    if not 0 <= idx < n_machines:
        raise ValueError(f"机位号越界：{idx}（共 {n_machines} 台）")
    if n_machines > MAX_MACHINES:
        raise ValueError(
            f"机位数 {n_machines} 超出本模型覆盖范围（最多 {MAX_MACHINES} 台）——"
            "须先扩表或改用其他引证来源，不得静默外推。")
    tier = _tier_of(idx)
    t = MACHINE_TIERS[tier]
    return {"tier": tier, "proc_kw": t["proc"][0], "idle_kw": t["idle"][0],
            "setup_kw": t["idle"][0]}      # 换型功率取空载（原文献未单列）


def machine_energy_kwh(proc_min: float, idle_min: float, setup_min: float,
                       *, idle_kw: float, proc_kw: float, setup_kw: float) -> float:
    """机床三态能耗。`setup_kw` 无文献值，暂取 `idle_kw`（见模块 docstring 假设 1）。"""
    return (_kwh(proc_kw, proc_min) + _kwh(idle_kw, idle_min) + _kwh(setup_kw, setup_min))


def agv_energy_kwh(idle_min: float, empty_min: float, loaded_min: float) -> float:
    """AGV 三态能耗。"""
    return (_kwh(AGV_IDLE_KW, idle_min) + _kwh(AGV_EMPTY_KW, empty_min)
            + _kwh(AGV_LOADED_KW, loaded_min))


# ── 分项导出（2026-10-06）──
# 动机：实测 `energy / makespan` 几乎恒定（progress-log §52.9.3 的 Pareto 前沿退化成两点）。
# 要判断"哪一项随 makespan 走"，就得把三态各自的值拿出来。**只加函数，不改上面两条既有公式。**

def machine_energy_breakdown_kwh(proc_min: float, idle_min: float, setup_min: float,
                                 *, idle_kw: float, proc_kw: float, setup_kw: float) -> dict[str, float]:
    """机床三态的**分项** [kWh]，键 = `proc` / `idle` / `setup`。

    与 `machine_energy_kwh` **同一批浮点、同一累加序**（proc → idle → setup）：
    `proc + idle + setup` 逐位等于该函数的返回值（防"分项加了、总量漂了"）。
    """
    return {"proc": _kwh(proc_kw, proc_min),
            "idle": _kwh(idle_kw, idle_min),
            "setup": _kwh(setup_kw, setup_min)}


def agv_energy_breakdown_kwh(idle_min: float, empty_min: float,
                             loaded_min: float) -> dict[str, float]:
    """AGV 三态的**分项** [kWh]，键 = `idle` / `empty` / `loaded`（同上，求和 == `agv_energy_kwh`）。"""
    return {"idle": _kwh(AGV_IDLE_KW, idle_min),
            "empty": _kwh(AGV_EMPTY_KW, empty_min),
            "loaded": _kwh(AGV_LOADED_KW, loaded_min)}


def total_energy_kwh(machine_kwh: float, agv_kwh: float, makespan_min: float) -> float:
    """总能耗 = 机床 + AGV + 车间固定（固定项按 makespan 计）。"""
    return machine_kwh + agv_kwh + _kwh(SHOP_FIXED_KW, makespan_min)
