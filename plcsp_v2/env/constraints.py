"""约束配置（spec §3.3）：**五条**约束的开关 + 五组消融配置。

设计：**开关是纯配置，不含逻辑**——`des.py` 读它决定行为。
5 组消融 = 5 个配置实例，**改配置不改代码**，避免消融时引入代码差异。

⚠️ **编号一律沿用 v1 的历史编号，不重排**（②③⑧⑨⑫ 与 spec / 两份 progress-log
一一对应，重排会让旧记录对不上号）。**故本文件里编号是不连续的**——
`② ③ ⑧ ⑨ ⑫` 就是全部五条。

⚠️ **v2 的五条约束**（2026-10-10 起）：

| 号 | 名 | 开关 |
|---|---|---|
| ② | 有限缓冲 | `finite_buffer` |
| ③ | 机器故障 | `machine_failure` |
| ⑧ | 交期 / 拖期 | `due_dates` |
| ⑨ | AGV 故障 | `agv_failure` |
| ⑫ | 预防性维护 | `maintenance` |

**v2 相对 v1 删掉的**：① 拥堵、④ 返工、⑤ 换型、⑥⑦ 模糊、⑩ 异构车队/拼批、
**⑪ 充电**（2026-10-10 删，三条依据见 `des.SimConfig` 顶部那处删除标记）。
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ConstraintConfig:
    """**五条**约束的开关。默认全开 = 论文主配置（spec §3.3）。

    ⚠️ **`finite_buffer` 是例外**：它默认 `False`（见字段上的注释）。
    """
    # A 类：基建已有
    # ⚠️ **v2：① 拥堵开关已删**（2026-10-08）—— 区段机制整个移除
    # ⚠️ **v2：② 有限缓冲开关曾删**（2026-10-08），**2026-10-10 加回**。
    # ⚠️ **2026-10-10：默认值由 `False` 翻成 `True`**（用户裁定「**在报告档打开**」）。
    # **依据**：② 是五条里**唯一默认档就 binding** 的那条（MK01 makespan 73.01 vs 75.87、
    # AGV 侧阻塞 58.0 min）。**一个默认关着的约束，在论文里说不出口** ——
    # 而 AEI2025 的假设 4 恰恰批评"约束建模不现实"，我们不该在这条上留同一个把柄。
    #
    # ⚠️ **本条**覆盖**本仓的通用纪律「机制开关默认关 ⟹ 既有读数逐位不变」。
    # 为什么覆盖是安全的：**只有本开关**翻了默认；其余四条逐位不变。
    # 代价是**2026-10-10 之前的读数全部作废**（凡是走默认档跑出来的）。
    #
    # ⚠️ **它还让 `SimWorld(constraints=None)` 的契约第一次为真** ——
    # `des.SimWorld` 的 docstring 写着「`None` → **全开**」，
    # 而 ② 关着时 `None` 其实**不是全开**（少一条）。翻过来才对得上。
    finite_buffer: bool = True
    machine_failure: bool = True      # ③ 机器故障（泊松 + 中断恢复）
    # B 类：近乎免费
    # ⚠️ **v2：④ 返工开关已删**（2026-10-08）
    # ⚠️ **v2：⑤ 换型开关已删**（2026-10-08）—— 约束去掉后无对应机制，开关无意义
    due_dates: bool = True            # ⑧ 交期 / 拖期（TF/RDD：LB·τ·(1+R(2ρ−1))，见 due_dates.py）
    agv_failure: bool = True          # ⑨ AGV 故障
    # C 类：真花钱
    # ⚠️ **v2：⑩ 异构车队（含拼批）开关已删**（2026-10-08）
    # ⚠️ **2026-10-10：⑪ 充电 / 电量开关已删** —— 机制整条移除（用户裁定）。
    #    删它的三条依据见 `des.SimConfig` 顶部那处删除标记。
    maintenance: bool = True          # ⑫ 预防性维护（计划性停机）

    def with_off(self, *names: str) -> "ConstraintConfig":
        """返回**关掉指定开关**的副本（不可变风格，不原地改）。

        用法：`cfg.with_off("agv_failure", "due_dates")`。
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
    "-物流": _FULL.with_off("agv_failure"),
    "-生产": _FULL.with_off("finite_buffer", "machine_failure", "maintenance"),
    "-信息": _FULL.with_off("due_dates"),
    # ⚠️ **`None` = 五条全关** —— 必须**逐条列全**。
    # 2026-10-10 的教训：② 的默认值翻成 `True` 之后，本组**漏列 `finite_buffer`**
    # 就会带着 ② 跑，而它叫"None"。**`with_off` 是减法，不是置零** ——
    # 新加一条默认开的约束时，**本组必须同步**。
    "None": _FULL.with_off("finite_buffer",
                           "machine_failure",
                           "due_dates",
                           "agv_failure", "maintenance"),
}

# ── P4-B：两个报告档（用户裁定 2026-10-03）──
# 报告档 A「MKT 口径」= 文献（MKT = FJSP + 运输）没有对应项的机制全关 **⟹ 恰好等于 None 组**，
#   故**复用**它而不是新造配置（单一真相；改动只走 `with_off`）。
# 报告档 B「完整口径」= 机制全开，运输口径**相同**（都是行程时间矩阵）——两个报告档的差别**只在机制**。
# ⚠️ **五条机制** ②有限缓冲 ③机台故障 ⑧交期 ⑨AGV故障 ⑫维护 在 MKT 里一个都没有对应项。
#   将来加第 6 个机制时若它 MKT 也没有对应项，
#   `REPORT_TIERS["A-MKT"]` 必须同步——`test_two_tier_report.py` 按 **dataclass 字段**逐个查，
#   不依赖本注释或任何硬编码的名字表。
REPORT_TIERS: dict[str, ConstraintConfig] = {
    "A-MKT": ABLATION_GROUPS["None"],
    "B-Full": ABLATION_GROUPS["Full"],
}
