# -*- coding: utf-8 -*-
"""行程时间口径（transport-time caliber）：几何最短路 or MKT 机台间矩阵（P4-B Task 1）。

**共存规则（用户裁定，2026-10-03）**：口径**跟随实例**，不是全局开关——MKT 实例用矩阵、
原始 MK 实例用几何。任何"全局开关"都允许**同一实例跑出两套行程时间**，那是静默错配；
口径是问题的属性，不是运行配置。

矩阵口径的三条硬约束（**都来自数据本身**，见 `plcsp/data/mkt/README.md`）：

1. 矩阵**已经是分钟**——**不得**再套几何口径的 `距离 / (eff_speed·车速) / 60` 换算；
2. 原始矩阵是 `(m+1) × (m+1)`，**第 0 行/列是装卸站（LU）**，机台 i ↔ 下标 **i+1**；
   本模块只接受**含 LU 的全矩阵**——喂进 `load_mkt_layout(m)` 的默认（已丢 LU）结果会**显式报错**；
3. 矩阵**非对称**（`m=4` 是唯一例外，见 README §2）⟹ 查表**必须保序**。

⚠️ **装卸站（LU）的下标 0 现在有消费者**（P4-B Task 2b，2026-10-03 用户裁定）：作业**在装卸站
入场、在装卸站完工**，故任务里会出现"站→首工序机台"与"末工序机台→站"两条**负载段**，它们查的
正是矩阵的第 0 行/列。`layout.lu` 的节点由 `from_matrix` 自动映射到下标 0——不需要调用方记得传，
也就没有"忘了传"的口子。若某布局确实没有装卸站（`layout.lu is None`，只应出现在手搓的测试夹具里），
则下标 0 空置，`test_slot_zero_is_the_load_unload_station_and_nothing_else` 钉住"站 ↔ 0"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .layout import Layout

GEOMETRY = "geometry"            # 几何口径：格点最短路距离 ÷ 有效车速（现状；原始 MK 走这条）
MATRIX = "matrix"                # 矩阵口径：查 MKT 行程时间表（单位：分钟）
UNMAPPED_RAISE = "raise"         # 端点无矩阵对应项 ⟹ 显式报错（默认）
UNMAPPED_GEOMETRY = "geometry"   # 端点无矩阵对应项 ⟹ **声明式**降级为几何口径（并被计数）


@dataclass(frozen=True)
class TransportCaliber:
    """行程时间口径。**不可变**：一次运行内不得换口径（换了 = 同一实例两套行程时间）。"""

    mode: str = GEOMETRY
    matrix: np.ndarray | None = None                              # (m+1)×(m+1)，含 LU
    node_slot: Mapping[int, int] = field(default_factory=dict)    # 通道节点 → 下标（1..m）
    unmapped: str = UNMAPPED_RAISE

    def __post_init__(self) -> None:
        if self.mode not in (GEOMETRY, MATRIX):
            raise ValueError(f"未知行程时间口径：{self.mode!r}（只认 {GEOMETRY!r} / {MATRIX!r}）")
        if self.unmapped not in (UNMAPPED_RAISE, UNMAPPED_GEOMETRY):
            raise ValueError(f"未知未映射端点策略：{self.unmapped!r}"
                             f"（只认 {UNMAPPED_RAISE!r} / {UNMAPPED_GEOMETRY!r}）")
        if self.mode != MATRIX:
            return
        if self.matrix is None:
            raise ValueError("矩阵口径必须给 matrix（(m+1)×(m+1)，含 LU）")
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"行程时间矩阵必须是**方阵**，实得 {tuple(self.matrix.shape)}")
        if not self.node_slot:
            raise ValueError("矩阵口径必须给节点→下标映射（node_slot）")
        want = max(self.node_slot.values()) + 1
        if self.matrix.shape[0] != want:
            raise ValueError(
                f"行程时间矩阵必须是 (m+1)×(m+1)（**第 0 行/列是装卸站 LU**）：机台最大下标 "
                f"{want - 1} ⟹ 应有 {want} 行/列，实得 {self.matrix.shape[0]}——若你手上是 "
                f"`load_mkt_layout(m)` 的默认结果，那是**已丢 LU** 的 m×m，不得当全矩阵用")
        if min(self.node_slot.values()) < 0:
            raise ValueError("矩阵下标必须 ≥ 0（0 = 装卸站的位置）")
        if len(set(self.node_slot.values())) != len(self.node_slot):
            raise ValueError("两个通道节点映射到了同一个矩阵下标——节点→下标不是一一对应")

    # ── 构造 ──
    @staticmethod
    def geometry() -> "TransportCaliber":
        """几何口径（缺省）：原始 MK 实例走它。"""
        return TransportCaliber(mode=GEOMETRY)

    @staticmethod
    def from_matrix(matrix: np.ndarray, layout: Layout,
                    *, unmapped: str = UNMAPPED_RAISE) -> "TransportCaliber":
        """按**布局**建矩阵口径：机台 i 的 `dock_node` ↔ 矩阵下标 **i+1**，装卸站 ↔ 下标 **0**。

        ⚠️ 机台号必须与**实例机台号**一致：`sample_layout` 按格子顺序放机台（`machines[i]` 即
        实例机台 i），故这里直接用 `enumerate`。⚠️ 对齐的是**下标**，不是坐标——MKT 矩阵的
        机台编号来自 Brandimarte 的 `.fjs`，两边同源同序（P4-A 已核"加工数据一字不改"）。
        ⚠️ 装卸站**由布局自动带入**（`layout.lu`）——不设开关、不让调用方记得传：
        "忘了映射装卸站"会是静默错行（整条 LU 段查错格），不是显式错误。
        """
        node_slot = {int(mp.dock_node): i + 1 for i, mp in enumerate(layout.machines)}
        if len(node_slot) != len(layout.machines):
            raise ValueError("机台的 dock_node 有重复——节点→矩阵下标不是一一对应")
        lu = getattr(layout, "lu", None)
        if lu is not None:
            if int(lu.node) in node_slot:
                raise ValueError("装卸站节点与某台机台的 dock_node 重合——节点→下标不是一一对应")
            node_slot[int(lu.node)] = 0        # 矩阵第 0 行/列就是装卸站
        return TransportCaliber(mode=MATRIX, matrix=np.asarray(matrix, dtype=float),
                                node_slot=node_slot, unmapped=unmapped)

    @staticmethod
    def for_instance(inst, layout: Layout, *, unmapped: str = UNMAPPED_RAISE) -> "TransportCaliber":
        """按**实例自带的口径标签**建口径——共存规则的**唯一入口**。

        - `transport == "geometry"`（缺省）⟹ 几何口径，**忽略**任何矩阵；
        - `transport == "matrix"` ⟹ 必须有 `(m+1)×(m+1)` 的**含 LU 全矩阵**，否则**显式报错**
          （静默退回几何 = "同一实例两套行程时间"，正是本设计要堵的口子）。
        """
        mode = getattr(inst, "transport", GEOMETRY)
        if mode == GEOMETRY:
            return TransportCaliber.geometry()
        if mode != MATRIX:
            raise ValueError(f"实例的 transport 标签未知：{mode!r}（只认 {GEOMETRY!r} / {MATRIX!r}）")
        full = getattr(inst, "trans_time_full", None)
        if full is None:
            raise ValueError(
                "实例标了 transport='matrix' 却没有 trans_time_full——口径标签与数据不符"
                "（矩阵口径的实例必须由 `plcsp.env.mkt.load_mkt` 造出）")
        want = inst.n_machines + 1
        if tuple(full.shape) != (want, want):
            raise ValueError(
                f"实例 {inst.source} 的行程时间矩阵应为 (m+1)×(m+1)={want}×{want}，"
                f"实得 {tuple(full.shape)}")
        return TransportCaliber.from_matrix(full, layout, unmapped=unmapped)

    # ── 查询 ──
    def slot_of(self, node: int) -> int | None:
        """通道节点 → 矩阵下标；无对应项（充电桩等）→ None。"""
        return self.node_slot.get(int(node))

    def minutes(self, u: int, v: int) -> float | None:
        """矩阵口径下 u→v 的行程时长 [min]；任一端点无对应项 → None（由调用方按 `unmapped` 处置）。

        ⚠️ **不做任何单位换算**：矩阵本身就是分钟。⚠️ **保序**查表（矩阵非对称）。
        """
        if self.mode != MATRIX:
            raise ValueError("几何口径没有矩阵查表——调用方应先判 mode（或走 AgvSim._leg_min）")
        su, sv = self.slot_of(u), self.slot_of(v)
        if su is None or sv is None:
            return None
        return float(self.matrix[su, sv])
