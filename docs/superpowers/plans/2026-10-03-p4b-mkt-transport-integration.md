# P4-B MKT 行程时间矩阵接入仿真 + 两档口径报告 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 P4-A 落盘的 **MKT 机台间行程时间矩阵**接进 `AgvSim`（今天仿真一律用布局几何算行程时间），使我们的数字能按文献在用的同一口径写进同一张表；并交付**两档报告**（档 A = MKT 口径的退化形态、档 B = 机制全开）。

**Architecture:** 新增一个**口径对象** `TransportCaliber`（几何 / 矩阵两态），口径**跟随实例**（用户裁定：MKT 实例走矩阵、原始 MK 实例走几何，不做全局开关）。矩阵作为**实例数据**挂在 `Instance` 上（`transport` 标签 + `trans_time_full`），于是 `rollout` / `SimWorld` / `build_setup` / `reference_run` / 训练栈的**签名一概不变**——矩阵口径自动贯通，没有"忘了传"的口子。`AgvSim` 的行程时间收敛到**一个**方法 `_leg_min`。

**Tech Stack:** Python 3.12.4（`D:/anaconda/python.exe`）｜CPU-only｜numpy 1.26.4｜simpy｜pytest 7.4.4｜Windows 11

**Spec:** `docs/superpowers/specs/2026-10-02-plcsp-rebuild-design.md`（§3.2 布局 / §3.5 交期 / §6.1 基线 / §6.2 消融）
**依据记录**：`docs/progress-log.md` **§19.7d**（MKT 口径核实）、**§20.3 / §20.5**（`ours` 与对照列不同口径）、**§21.7**（MKT 静默复用 MK 标定的隐患）
**上游计划**：`docs/superpowers/plans/2026-10-03-p4a-mkt-benchmark-alignment.md` 的「已知边界」表（HGS 不在本批）

## Global Constraints

- **项目根** = `D:\research\DeepReinforcementLearningScheduling`。**根目录只允许有目录**，不得新增根级文件；**禁止**把 `*.log` / `*.jsonl` / `figs/` 等临时产物写进包目录或根目录。
- **Python** 一律用 `D:/anaconda/python.exe`；前缀 `PYTHONIOENCODING=utf-8`。**CPU-only**。
- **单位约定（不得改动）**：仿真时间 = 分钟，布局坐标 = 米，能耗 = kWh。**矩阵已经是分钟**——**不得**对它套几何口径的 `距离 / (eff_speed · 车速) / 60` 换算（本批的头号缺陷形态）。
- **回归门禁**：`plcsp/tests/` 现有 **283 项必须始终全绿**，且 `ruff check plcsp/` 保持 clean。⚠️ **全套 `plcsp/tests` 约 6 分钟**（`test_end_to_end_a.py` 独占 4 分多），每个任务末尾都要跑，知情安排墙钟。
- **行尾**：本仓是 **38 CRLF / 58 LF** 的混合，**无 `.gitattributes`**。**改既有文件一律用 Edit 工具**（不得整文件重写、不得转换行尾）；新建文件跟随同目录多数派——实测：`plcsp/env/` **CRLF 6 / LF 4 ⟹ 新文件用 CRLF**；`plcsp/tests/` **LF 25 / CRLF 2 ⟹ LF**；`plcsp/` 顶层 **LF 8 / CRLF 1 ⟹ LF**；`docs/` 与 `docs/superpowers/plans/` **全 LF**。
- **提交信息格式**：`<type>: <description>`，**不添加任何署名/生成标识**。
- **⚠️ 术语纪律（用户明令）**：**不自造术语**。用文献与既有文档承认的词——「行程时间矩阵」「口径」「退化特例」「复现保真度」；**不要**用"对拍"这类随手词。引用文献给**完整题名 + 期刊 + 卷期页 + 年**，拿不到原文就写**待核**，不得编造。
- **测试约定（本仓既有风格，必须遵守）**：`pytest.mark.unit`；**中文 docstring 说明这条测试为什么存在**；**钉反属性**（断言"某件事**不会**静默发生"）；带 `⚠️` 的断言要写清它挡的是哪一类错误实现。
- **判据纪律（本项目反复吃亏的一条）**：测试名/docstring 声称的属性**必须真的被断言覆盖**（已发生 6 次"名不副实"）。本计划里每条测试都要求：**若把实现改错，它必须变红**——写完测试后**自己变异一次**（改坏实现看它红不红），在报告里写明变异结果。

## Review Focus

以下八类是本批最可能出错、且**光看名字会以为已经覆盖**的：每一条都在**它所属任务**里配了直接断言。

1. **矩阵被"顺手"再换算一次**。矩阵是分钟；若 `_leg_min` 沿用几何口径的 `/ (eff_speed·车速) / 60`，全部行程缩到 1/60 上下，**不报错**，只是数字变得"好看"。→ Task 2 的**逐位对拍**测试（Σ 查表值）。
2. **LU 的 off-by-one 换个地方复发**。P4-A 挡的是"读文件时裁错方向"；本批的新风险是**节点→矩阵下标**（机台 i ↔ 下标 **i+1**，第 0 位是 LU）。喂进 `load_mkt_layout(m)` 的默认（已丢 LU）结果同样**不报错**、只是整体错行一行。→ Task 1 的形状守卫 + 具体非对称项测试 + "无任何节点映射到下标 0"。
3. **口径跟随实例的规则被绕过**。给几何实例塞矩阵 / 给 `matrix` 实例没矩阵 / 事后加一个"全局矩阵开关"——三者都是"同一实例跑出两套行程时间"的入口。→ Task 2 的三条显式报错测试 + 一条反属性（`SimConfig` 里**不得**出现口径开关）。
4. **参考运行缓存串味**。`_instance_key` 若不含口径与矩阵，`reference_run` 会把**几何口径**的参考运行静默喂给矩阵口径的调用方（M_ref、奖励权重、特征归一化一起错）。→ Task 2 的"两个口径的参考 makespan 必须不等"。
5. **⑩/⑪ 在矩阵口径下静默失效或反噬**。⑩ 的速度倍率若没接进矩阵查表，"异构车队"退化成同构（机制变常数）；⑪ 的充电桩在矩阵里**没有对应项**，若静默取 0 或静默猜一个机台，运输负荷被凭空抹掉。⚠️ **反方向也要防**：矩阵口径下单车一趟 episode 会耗 ~3.5 kWh（0.5 kW × 约 342 min 负载行驶 + 空载功率），与电池容量 2–4 kWh 同量级 ⟹ **⑪ 从"结构性死约束"（§5.7 的 0.27 kWh）变成 binding**，充电腿会真的出现。→ Task 2 的直接算术测试（倍率）+ 未映射端点的 **raise 路径**与**计数路径**两条 + 强制低电的两条（在 Task 4）。
6. **交期标定静默复用**。`load_mkt(name)` 内部就是 `load_mk(name)` ⟹ 文件名主干相同，按主干取键会让 MKT 实例命中原始 MK 的 (τ, R)（两个不同问题用同一把尺）。→ Task 2 的"不得静默等于 MK 的值" + Task 3 的"两表不同值、且由脚本可复现"。
7. **区段拆分下"各段之和 ≠ 矩阵值"**。① 开时行程被拆成逐节点段，而**中间走廊节点在矩阵里没有对应项**；逐段查表会 KeyError 或静默退回几何。→ Task 2 的"① 开时各段之和**逐位**等于矩阵查表值"。
8. **掐表（horizon）被当成正常结果**。矩阵口径下行程是分钟的整数级（几何口径约 0.2 min/腿），makespan 会大数倍；`horizon = 总工时×6 + 500` 这个固定护栏可能截断运行，而截断结果**看起来只是"更慢"**。→ Task 2 把 `horizon_hit` 带进 metrics 检查、Task 4 的表里**逐行打印并断言为假**。

---

### Task 1: 行程时间口径对象（几何 / 矩阵两态）

**Files:**
- Create: `plcsp/env/transport.py`（新文件 → **CRLF**，跟随 `plcsp/env/` 多数派）
- Test: `plcsp/tests/test_transport_caliber.py`（**LF**）

**Interfaces:**
- Consumes: `plcsp.env.mkt.load_mkt_layout`（数据）、`plcsp.env.layout.Layout`（机台的 `dock_node`）
- Produces:
  - 常量 `GEOMETRY = "geometry"` / `MATRIX = "matrix"` / `UNMAPPED_RAISE = "raise"` / `UNMAPPED_GEOMETRY = "geometry"`
  - `TransportCaliber`（frozen dataclass）：`mode: str` / `matrix: np.ndarray | None` / `node_slot: Mapping[int, int]` / `unmapped: str`
  - `TransportCaliber.geometry() -> TransportCaliber`
  - `TransportCaliber.from_matrix(matrix: np.ndarray, layout: Layout, *, unmapped: str = UNMAPPED_RAISE) -> TransportCaliber`
  - `caliber.slot_of(node: int) -> int | None`、`caliber.minutes(u: int, v: int) -> float | None`
  - （Task 2 再加 `for_instance(inst, layout, *, unmapped)`——它需要 `Instance` 上本批新增的两个字段）

- [ ] **Step 1: 写失败测试**

创建 `plcsp/tests/test_transport_caliber.py`：

```python
"""行程时间口径对象的测试（P4-B Task 1）。

口径的两条硬约束来自数据本身（`plcsp/data/mkt/README.md`）：矩阵是 `(m+1)×(m+1)`、
**第 0 行/列是装卸站（LU）**、机台 i ↔ 下标 i+1；矩阵**已是分钟**、且**非对称**
（`m=4` 是唯一例外）。本文件只测**对象本身**，接进仿真的测试在 `test_transport_wiring.py`。
"""
from __future__ import annotations

import pytest

from plcsp.env.layout import sample_layout
from plcsp.env.mkt import load_mkt_layout
from plcsp.env.transport import TransportCaliber


def _layout(m: int, seed: int = 0):
    return sample_layout(m, seed=seed, n_agv=m)


@pytest.mark.unit
def test_geometry_caliber_has_no_matrix():
    """几何口径是**缺省**：原始 MK 实例走它，不得带上任何矩阵。"""
    c = TransportCaliber.geometry()
    assert c.mode == "geometry" and c.matrix is None and not c.node_slot


@pytest.mark.unit
def test_dropped_lu_matrix_is_rejected_loudly():
    """⚠️ Review Focus #2：喂进**已丢 LU** 的 m×m 矩阵必须**显式报错**。

    `load_mkt_layout(m)` 的默认返回是丢过 LU 的 `m×m`；若它被当全矩阵用，机台 i 会整体错行
    一行——数字看着合理、全程不报错，正是 P4-A 同类 off-by-one 的复发形态。
    """
    lay = _layout(6)
    with pytest.raises(ValueError, match="LU"):
        TransportCaliber.from_matrix(load_mkt_layout(6), lay)      # 丢 LU 的 6×6


@pytest.mark.unit
def test_non_square_matrix_is_rejected():
    """裁错方向（`(m+1)×m` 之类）也要在**构造时**就报错，不得等到查表才崩。"""
    lay = _layout(6)
    bad = load_mkt_layout(6, drop_lu=False)[:, 1:]                  # 7×6
    with pytest.raises(ValueError, match="方阵"):
        TransportCaliber.from_matrix(bad, lay)


@pytest.mark.unit
def test_slot_zero_is_reserved_for_the_load_unload_station():
    """⚠️ 下标 0 是装卸站（LU）——**本项目今天没有任何节点该映射到它**（尚无 LU 运输）。

    这条挡的是"把某台机台挂到下标 0"的实现错误（会让所有机台整体错行）。
    """
    lay = _layout(6)
    c = TransportCaliber.from_matrix(load_mkt_layout(6, drop_lu=False), lay)
    assert 0 not in set(c.node_slot.values()), "有节点映射到了 LU 位（下标 0）"
    assert set(c.node_slot.values()) == set(range(1, 7))


@pytest.mark.unit
@pytest.mark.parametrize("m", [5, 6, 15])
def test_machine_i_maps_to_matrix_row_i_plus_one(m: int):
    """⚠️ Review Focus #2：机台 i ↔ 矩阵下标 **i+1**（+1 是 LU 占的第 0 位）。"""
    lay = _layout(m)
    c = TransportCaliber.from_matrix(load_mkt_layout(m, drop_lu=False), lay)
    for i, mp in enumerate(lay.machines):
        assert c.slot_of(mp.dock_node) == i + 1, f"机台 {i} 的矩阵下标不是 {i + 1}"


@pytest.mark.unit
def test_matrix_values_are_used_verbatim():
    """⚠️ Review Focus #1 的对象侧：查表值必须**逐位等于**文件里的数（分钟），不得有任何换算。"""
    lay = _layout(6)
    raw = load_mkt_layout(6, drop_lu=False)
    c = TransportCaliber.from_matrix(raw, lay)
    for i in range(6):
        for j in range(6):
            assert c.minutes(lay.machines[i].dock_node, lay.machines[j].dock_node) == float(raw[i + 1, j + 1])


@pytest.mark.unit
def test_asymmetric_entry_survives_the_node_mapping():
    """⚠️ Review Focus #2：**保序查表**——用一条实测非对称项钉住（转置/错行都会红）。

    10 机布局：1 基机台号 M[1][2] = 3 而 M[2][1] = 12（`progress-log §19.7d`、P4-A F3 已核）。
    本测试走**节点→下标**这条路（本批的新增映射层），比直接读文件多挡一层。
    """
    lay = _layout(10)
    c = TransportCaliber.from_matrix(load_mkt_layout(10, drop_lu=False), lay)
    a, b = lay.machines[0].dock_node, lay.machines[1].dock_node
    assert c.minutes(a, b) == pytest.approx(3.0)
    assert c.minutes(b, a) == pytest.approx(12.0)
    assert c.minutes(a, b) != c.minutes(b, a), "矩阵被对称化了"


@pytest.mark.unit
def test_unmapped_node_is_reported_not_guessed():
    """⚠️ Review Focus #5：充电桩在矩阵里**没有对应项** ⟹ 返回 None，**不得**猜最近机台或取 0。"""
    lay = _layout(6)
    c = TransportCaliber.from_matrix(load_mkt_layout(6, drop_lu=False), lay)
    assert lay.chargers, "该布局应有充电桩（本测试的前提）"
    charger_node = lay.chargers[0].node
    assert c.slot_of(charger_node) is None
    assert c.minutes(lay.machines[0].dock_node, charger_node) is None
    assert c.minutes(charger_node, lay.machines[0].dock_node) is None


@pytest.mark.unit
def test_unknown_unmapped_policy_is_rejected():
    """策略字符串写错必须当场报错——否则运行期的"降级"会变成静默行为。"""
    lay = _layout(6)
    with pytest.raises(ValueError, match="未映射"):
        TransportCaliber.from_matrix(load_mkt_layout(6, drop_lu=False), lay, unmapped="fallback")


@pytest.mark.unit
def test_geometry_caliber_has_no_matrix_lookup():
    """几何口径下查表是**编程错误**（调用方必须先判 mode）——不得静默返回几何值。"""
    with pytest.raises(ValueError, match="矩阵"):
        TransportCaliber.geometry().minutes(0, 1)


@pytest.mark.unit
def test_duplicate_dock_nodes_are_rejected():
    """机台共用 dock 节点时"节点→机台"不是函数，必须报错（否则整张表错行且难查）。"""

    class _FakeM:
        def __init__(self, node):
            self.dock_node = node

    class _FakeLayout:
        machines = [_FakeM(3), _FakeM(3)]

    with pytest.raises(ValueError, match="重复"):
        TransportCaliber.from_matrix(load_mkt_layout(4, drop_lu=False), _FakeLayout())
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_transport_caliber.py -q --no-header`
Expected: FAIL — `ModuleNotFoundError: No module named 'plcsp.env.transport'`

- [ ] **Step 3: 实现 `plcsp/env/transport.py`**

```python
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

⚠️ **LU 今天没有消费者**：本项目的 AGV 任务只在**机台之间**搬运（作业在首工序机台入场、
在末工序机台完工，见 `des.py` 的 `AgvSim.run`），没有"从装卸站取件/送回装卸站"这一段。
全矩阵进来是为了**保住数据**（将来要加 LU 运输时不必回头改数据层），不是现在用得上——
`node_slot` 里没有任何节点映射到下标 0，`test_slot_zero_is_reserved_for_the_load_unload_station`
钉住这一点（加了 LU 运输它会红，逼人同时改口径与文档）。
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
        if min(self.node_slot.values()) < 1:
            raise ValueError("节点不得映射到下标 0——那是装卸站（LU）的位置，本项目尚无 LU 运输")

    # ── 构造 ──
    @staticmethod
    def geometry() -> "TransportCaliber":
        """几何口径（缺省）：原始 MK 实例走它。"""
        return TransportCaliber(mode=GEOMETRY)

    @staticmethod
    def from_matrix(matrix: np.ndarray, layout: Layout,
                    *, unmapped: str = UNMAPPED_RAISE) -> "TransportCaliber":
        """按**布局**建矩阵口径：机台 i 的 `dock_node` ↔ 矩阵下标 **i+1**。

        ⚠️ 机台号必须与**实例机台号**一致：`sample_layout` 按格子顺序放机台（`machines[i]` 即
        实例机台 i），故这里直接用 `enumerate`。⚠️ 对齐的是**下标**，不是坐标——MKT 矩阵的
        机台编号来自 Brandimarte 的 `.fjs`，两边同源同序（P4-A 已核"加工数据一字不改"）。
        """
        node_slot = {int(mp.dock_node): i + 1 for i, mp in enumerate(layout.machines)}
        if len(node_slot) != len(layout.machines):
            raise ValueError("机台的 dock_node 有重复——节点→矩阵下标不是一一对应")
        return TransportCaliber(mode=MATRIX, matrix=np.asarray(matrix, dtype=float),
                                node_slot=node_slot, unmapped=unmapped)

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
```

- [ ] **Step 4: 跑测试确认通过 + 全套回归**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_transport_caliber.py -q --no-header`
Expected: PASS（10 项）

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header`
Expected: 全绿（**283 + 10 = 293**，约 6 分钟）

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m ruff check plcsp/`
Expected: `All checks passed!`

> ⚠️ **变异自检（必做，写进报告）**：把 `node_slot` 的 `i + 1` 改成 `i`（少加一），确认
> `test_machine_i_maps_to_matrix_row_i_plus_one` 与 `test_matrix_values_are_used_verbatim` **变红**；
> 改回。若不变红，说明这两条测试是"名不副实"的，必须先修测试。

- [ ] **Step 5: 提交**

```bash
git add plcsp/env/transport.py plcsp/tests/test_transport_caliber.py
git commit -m "feat: 行程时间口径对象（几何/矩阵两态）+ 节点→矩阵下标映射（P4-B Task 1）"
```

---

### Task 2: 把矩阵接进仿真（口径跟随实例、缓存键、交期键）

**Files:**
- Modify: `plcsp/env/instances.py`（`Instance` 加 `transport` 标签与 `trans_time_full`）
- Modify: `plcsp/env/mkt.py`（`load_mkt` 造矩阵实例；`MktInstance.trans_time` 改成**属性**）
- Modify: `plcsp/env/transport.py`（加 `for_instance`）
- Modify: `plcsp/env/des.py`（`SimConfig.transport_unmapped`；`AgvSim._leg_min` / `_seg_min` / `_drive` / `_maybe_charge`；`_build_entities` 传口径；`_instance_key`；两个 stats 字典与 metrics）
- Modify: `plcsp/env/due_dates.py`（标定表按**口径**分表；错误信息指明脚本与开关）
- Test: `plcsp/tests/test_transport_wiring.py`（**LF**）

**Interfaces:**
- Consumes: Task 1 的 `TransportCaliber`
- Produces:
  - `Instance.transport: str = "geometry"`、`Instance.trans_time_full: np.ndarray | None = None`
  - `TransportCaliber.for_instance(inst, layout, *, unmapped=UNMAPPED_RAISE) -> TransportCaliber`
  - `SimConfig.transport_unmapped: str = UNMAPPED_RAISE`
  - `AgvSim._leg_min(src: int, dst: int, *, count_unmapped: bool = True) -> float`
  - metrics 新增：`transport: str` / `unmapped_legs: int` / `unmapped_min: float`
  - `due_dates.TF_RDD_MATRIX: dict[str, tuple[float, float]] = {}`（Task 3 填充）+ `TABLE_BY_CALIBER`；
    `due_dates_for` 改为**按口径选表**（键仍是文件名主干，`_instance_name` 语义不变）

- [ ] **Step 1: 写失败测试**

创建 `plcsp/tests/test_transport_wiring.py`：

```python
"""矩阵口径接进仿真的测试（P4-B Task 2）。

⚠️ 本文件的核心是**直接钉住算术**（逐位对拍 Σ 矩阵查表值），不靠端到端间接推断——
本项目的教训是"测试名声称的属性 > 实际验证的内容"已发生 6 次。
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import simpy

from plcsp.env.constraints import ConstraintConfig
from plcsp.env.corridors import build_corridor_graph, dock_distance_matrix
from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import load_mk
from plcsp.env.layout import AgvSpec, sample_layout
from plcsp.env.mkt import load_mkt
from plcsp.env.transport import TransportCaliber


def _agv(inst, *, speed_factor: float, hetero: bool, unmapped: str = "raise", n_agv: int = 2):
    """造一台**不跑仿真**的 AgvSim，只用来直接调 `_leg_min`（算术判据不该绕道端到端）。"""
    from plcsp.env.des import AgvSim, SimTrack, ZoneManager, build_zone_map

    cfg = SimConfig(n_agv=n_agv, transport_unmapped=unmapped)
    lay = sample_layout(inst.n_machines, seed=0, n_agv=n_agv)
    g = build_corridor_graph(lay)
    zof, nz = build_zone_map(lay, cfg.zone_granularity)
    env = simpy.Environment()
    con = ConstraintConfig() if hetero else ConstraintConfig(heterogeneous_fleet=False)
    spec = AgvSpec(id=0, speed_factor=speed_factor, capacity=1, battery_kwh=3.0)
    cal = TransportCaliber.for_instance(inst, lay, unmapped=unmapped)
    agv = AgvSim(env, 0, dock_distance_matrix(g), cal, cfg, {}, simpy.Store(env), [], g,
                 ZoneManager(env, zof, nz, cfg.zone_wait_limit), con, spec,
                 np.random.default_rng(0), SimTrack(1, inst.n_machines, n_agv),
                 chargers=lay.chargers)
    return agv, lay


def _exact_caliber_constraints():
    """精确对拍需要的三个前提：① 关（不拆区段）、⑩ 关（载量 1、倍率 1）、⑪ 关。

    ⚠️ **⑪ 必须关**：矩阵口径下单车一趟 episode 会耗掉 ~3.5 kWh（0.5 kW × ~342 min 负载 +
    0.2 kW × 空载），与电池容量 2–4 kWh 同量级 ⟹ 车**会**去充电桩，而充电桩没有矩阵项
    （默认策略 `raise` 会直接报错、`geometry` 会往总额里混入几何腿）。既有几何口径测试没这个问题
    （整段 episode 只耗 0.27 kWh，见 §5.7），**不要**照抄它的约束配置。
    """
    return ConstraintConfig().with_off("congestion", "heterogeneous_fleet", "charging")


# ── 主线：逐位对拍 ──

@pytest.mark.unit
def test_matrix_travel_time_equals_the_lookup_sum_exactly():
    """⭐ Review Focus #1：矩阵口径下 `travel_time_total` 必须**逐位等于** Σ 矩阵查表值。

    这条同时挡住三类错误实现：① 又除了一次 `60·车速`（数会小 ~60 倍）；② 用了丢过 LU 的错行
    矩阵；③ 几何口径的 `m_dm` 还在被读。口径：① 关、⑩ 关、⑪ 关、单车
    （空载段 = 上一卸货点 → 本次取货点）。
    """
    mkt = load_mkt("mk01")
    t = mkt.trans_time                                  # m×m 机台间矩阵（分钟）
    r = rollout(mkt.base, seed_chain=1, cfg=SimConfig(n_agv=1), constraints=_exact_caliber_constraints())

    loaded, empty, prev = 0.0, 0.0, None
    for _j, _oi, frm, to in r["task_flow"]:
        loaded += float(t[frm][to])
        if prev is not None:
            empty += float(t[prev][frm])
        prev = to
    assert len(r["task_flow"]) > 0, "本 episode 应有搬运任务"
    assert r["travel_time_total"] == pytest.approx(loaded + empty, rel=1e-9), (
        f"矩阵口径的行程 {r['travel_time_total']:.2f} ≠ 查表值 {loaded + empty:.2f}"
        "——疑似又套了一次几何换算（或读的不是这张矩阵）")


@pytest.mark.unit
def test_zone_split_preserves_the_matrix_total():
    """⚠️ Review Focus #7：① **开着**时行程被拆成逐节点区段，但各段之和仍须等于矩阵查表值。

    矩阵没有"中间走廊节点"的概念；整段矩阵时长按**几何占比**摊到各段（占比和恒为 1）。
    单车 ⟹ 无区段争用 ⟹ 不会中途失败，故等式应**逐位**成立。⑪ 关（理由同上一条）。
    """
    mkt = load_mkt("mk01")
    t = mkt.trans_time
    cons = ConstraintConfig().with_off("heterogeneous_fleet", "charging")   # ① 保持默认开
    r = rollout(mkt.base, seed_chain=1, cfg=SimConfig(n_agv=1), constraints=cons)
    loaded, empty, prev = 0.0, 0.0, None
    for _j, _oi, frm, to in r["task_flow"]:
        loaded += float(t[frm][to])
        if prev is not None:
            empty += float(t[prev][frm])
        prev = to
    assert r["zone_wait"]["n"] >= 0                            # 若 ① 根本没生效，下面的等式无意义
    assert r["travel_time_total"] == pytest.approx(loaded + empty, rel=1e-9)


@pytest.mark.unit
def test_matrix_minutes_are_not_rescaled_by_speed_or_aisle_width():
    """⚠️ Review Focus #1：矩阵**已是分钟**——改绝对车速/通道宽**不得**改变矩阵口径的行程。

    几何口径下两者**必须**改变行程（见 `test_agv_travel.py`），这里钉的是相反的属性。
    """
    mkt = load_mkt("mk01")
    cons = _exact_caliber_constraints()
    fast = rollout(mkt.base, seed_chain=1,
                   cfg=SimConfig(n_agv=1, agv_speed_mps=0.5, aisle_width=1.5), constraints=cons)
    slow = rollout(mkt.base, seed_chain=1,
                   cfg=SimConfig(n_agv=1, agv_speed_mps=0.05, aisle_width=0.6), constraints=cons)
    assert fast["travel_time_total"] == slow["travel_time_total"], "矩阵口径被车速/通道宽重标了"
    assert fast["makespan"] == slow["makespan"]


# ── ⑩ 的相对倍率必须仍然生效（否则机制变常数）──

@pytest.mark.unit
def test_relative_speed_factor_still_applies_under_the_matrix():
    """⚠️ Review Focus #5：⑩ 的**相对**速度倍率在矩阵口径下必须生效。

    矩阵是"标准车速"下的分钟数；车队内部的快慢差（±20%）仍要表达出来，否则"异构车队"
    在 MKT 口径里静默退化成同构车队（机制变常数、消融表照绿）。
    """
    mkt = load_mkt("mk01")
    t = mkt.trans_time
    agv1, lay = _agv(mkt.base, speed_factor=1.0, hetero=True)
    agv2, _ = _agv(mkt.base, speed_factor=2.0, hetero=True)     # 同一 seed ⟹ 布局与 dock 节点相同
    u, v = lay.machines[0].dock_node, lay.machines[1].dock_node
    assert agv1._leg_min(u, v) == pytest.approx(float(t[0][1]))
    assert agv2._leg_min(u, v) == pytest.approx(float(t[0][1]) / 2.0)


@pytest.mark.unit
def test_speed_factor_is_inert_when_heterogeneous_fleet_is_off():
    """⑩ 关 ⟹ 倍率恒 1（与"该约束从未存在"逐位相同）——反方向钉一次，防"开关关不掉"。"""
    mkt = load_mkt("mk01")
    t = mkt.trans_time
    agv, lay = _agv(mkt.base, speed_factor=2.0, hetero=False)
    u, v = lay.machines[0].dock_node, lay.machines[1].dock_node
    assert agv._leg_min(u, v) == pytest.approx(float(t[0][1]))


# ── 未映射端点（充电桩）的两条路径 ──

@pytest.mark.unit
def test_unmapped_endpoint_raises_by_default():
    """⚠️ Review Focus #5：矩阵没有对应项的端点**默认显式报错**，不得静默猜一个值。"""
    mkt = load_mkt("mk01")
    agv, lay = _agv(mkt.base, speed_factor=1.0, hetero=True, unmapped="raise")
    with pytest.raises(ValueError, match="没有对应项"):
        agv._leg_min(lay.machines[0].dock_node, lay.chargers[0].node)


@pytest.mark.unit
def test_unmapped_endpoint_falls_back_to_geometry_and_is_counted():
    """⚠️ Review Focus #5：显式选 `geometry` 策略时**降级要留痕**（计数 + 时长都进 metrics）。

    口径：这一档是"矩阵覆盖不到的路段用几何补"，必须在表里声明——**不得**无声降级。
    """
    mkt = load_mkt("mk01")
    agv, lay = _agv(mkt.base, speed_factor=1.0, hetero=True, unmapped="geometry")
    u = lay.machines[0].dock_node
    got = agv._leg_min(u, lay.chargers[0].node)
    assert got > 0.0
    assert agv.stats["unmapped_legs"] == 1 and agv.stats["unmapped_min"] == pytest.approx(got)


# ── 口径跟随实例（不是全局开关）──

@pytest.mark.unit
def test_no_global_switch_can_turn_the_matrix_on():
    """⚠️ Review Focus #3：`SimConfig` 里**不得**出现任何口径/矩阵开关（用户裁定）。

    若日后有人加 `SimConfig.use_mkt_matrix`，这条会红——那正是"同一实例跑出两套行程时间"的入口。
    """
    names = {f.name for f in dataclasses.fields(SimConfig)}
    bad = sorted(n for n in names if "matrix" in n or "mkt" in n.lower())
    assert not bad, f"SimConfig 里出现了口径开关：{bad}（口径只能由实例携带）"


@pytest.mark.unit
def test_matrix_label_without_matrix_data_raises():
    """⚠️ Review Focus #3：标了 `matrix` 却没有矩阵 ⟹ 显式报错，**不得**静默退回几何。"""
    inst = load_mk("mk01")
    inst.transport = "matrix"                     # 手工制造"标签与数据不符"
    with pytest.raises(ValueError, match="trans_time_full"):
        TransportCaliber.for_instance(inst, sample_layout(6, seed=0, n_agv=3))


@pytest.mark.unit
def test_geometry_instance_never_gets_a_matrix_caliber():
    """⚠️ Review Focus #3：几何实例 + 任何布局 ⟹ 几何口径（矩阵进不来）。"""
    lay = sample_layout(6, seed=0, n_agv=3)
    c = TransportCaliber.for_instance(load_mk("mk01"), lay)
    assert c.mode == "geometry" and c.matrix is None


@pytest.mark.unit
def test_rollout_declares_its_transport_caliber_in_the_metrics():
    """口径必须**随结果自报**——否则两档口径的数并排放时无从分辨（P4-A Review Focus #1 同型）。"""
    mk = rollout(load_mk("mk01"), seed_chain=1, cfg=SimConfig(n_agv=3))["transport"]
    mkt = rollout(load_mkt("mk01").base, seed_chain=1, cfg=SimConfig(n_agv=6))["transport"]
    assert mk == "geometry" and mkt == "matrix"


@pytest.mark.unit
def test_reference_runs_of_the_two_calibers_do_not_share_the_cache():
    """⚠️ Review Focus #4：同名 MK 与 MKT 实例的参考运行**不得共用缓存**。

    `_instance_key` 若不含口径与矩阵，`reference_run` 会把**几何口径**的参考运行静默喂给
    矩阵口径的调用方（M_ref、奖励权重、特征归一化一起错，且零报错）。
    """
    from plcsp.env.des import reference_run

    cfg = SimConfig(n_agv=6)
    a = reference_run(load_mk("mk01"), cfg=cfg, seed_layout=0)["makespan"]
    b = reference_run(load_mkt("mk01").base, cfg=cfg, seed_layout=0)["makespan"]
    assert a != b, "两个口径的参考运行跑出同一 makespan——缓存串味（_instance_key 漏了口径/矩阵）"
    assert b > a, "矩阵是分钟量级，矩阵口径的参考 makespan 应显著更大"


# ── 交期标定表：MKT 不得命中 MK 的条目 ──

@pytest.mark.unit
def test_due_date_table_is_selected_by_caliber():
    """⚠️ Review Focus #6：交期标定表必须**按口径选**——MKT 不得读到原始 MK 那本尺。

    `load_mkt(name).base.source` 与 `load_mk(name).source` 是同一个文件名（加工数据一字不改），
    故**文件名主干相同**；分口径靠的是"读哪张表"，不是键长什么样。
    """
    from plcsp.env.due_dates import _instance_name, due_dates_for

    mk, mkt = load_mk("mk01"), load_mkt("mk01").base
    assert _instance_name(mk) == _instance_name(mkt) == "mk01"     # 主干确实相同（这正是危险所在）
    assert due_dates_for(mk)                                        # 几何表有 mk01

    inst = load_mk("mk01")
    inst.transport = "mkt-ish"                                      # 未知口径
    with pytest.raises(ValueError, match="口径"):
        due_dates_for(inst)


@pytest.mark.unit
def test_mkt_never_silently_reuses_the_mk_calibration():
    """⚠️ Review Focus #6：MKT 实例要么**显式报错**（未标定），要么给出与 MK **不同**的交期。

    唯一不允许的是"静默等于 MK 的值"——那是两个不同问题共用一把尺。
    （Task 3 落 `TF_RDD_MATRIX` 后本测试走"值不同"那一支；本测试**不需要改写**。）
    """
    from plcsp.env.due_dates import due_dates_for

    mk, mkt = load_mk("mk01"), load_mkt("mk01").base
    try:
        d_mkt = due_dates_for(mkt)
    except ValueError as e:
        assert "matrix" in str(e), f"报错必须说清是哪个口径下未标定，实得：{e}"
        d_mkt = None                       # 未标定：显式报错是正确行为
    if d_mkt is not None:
        assert d_mkt != due_dates_for(mk), "MKT 交期与 MK 逐位相同——标定表被静默复用了"


@pytest.mark.unit
def test_mkt_instance_still_presents_the_dropped_view():
    """`MktInstance.trans_time` 改成属性后，P4-A 交付的语义**不得变**：仍是丢 LU 的 m×m。"""
    mkt = load_mkt("mk01")
    from plcsp.env.mkt import load_mkt_layout
    assert mkt.trans_time.shape == (6, 6)
    assert np.array_equal(mkt.trans_time, load_mkt_layout(6))
    assert mkt.base.trans_time_full.shape == (7, 7), "全矩阵（含 LU）必须保在实例上"
```

> ⚠️ 上面 `_agv(...)` 造的是**不跑仿真**的 AgvSim（`run()` 从未被调用），故 `machines=[]`、
> `stats={}` 都合法——它只用来直接调 `_leg_min`。算术判据不该绕道端到端。

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_transport_wiring.py -q --no-header`
Expected: FAIL — `AttributeError`（`Instance` 没有 `transport`）/ `TypeError`（`AgvSim` 没有 `transport` 参数）

- [ ] **Step 3: 实现**

**(a) `plcsp/env/instances.py`** —— `Instance` 末尾追加两个字段（**放在最后**，位置参数兼容）：

```python
@dataclass
class Instance:
    n_jobs: int
    n_machines: int
    jobs: list[list[list[tuple[int, float]]]]   # job -> op -> [(mach, time), ...]
    source: str = ""
    # ── P4-B：**行程时间口径跟随实例**（用户裁定）──
    # `transport` 是口径标签（"geometry" | "matrix"）；矩阵本身在 `trans_time_full`——
    # 口径是**问题的属性**，不是运行配置，故它随实例走、`rollout`/`SimWorld` 的签名一概不变。
    transport: str = "geometry"
    trans_time_full: np.ndarray | None = None   # (m+1)×(m+1)，**第 0 行/列是装卸站（LU）**，单位分钟
```

（`from __future__ import annotations` 已在文件头，`np.ndarray` 只作标注、无需在模块级 import numpy。）

**(b) `plcsp/env/mkt.py`** —— `MktInstance` 去掉 `trans_time` 字段、改为属性；`load_mkt` 造矩阵实例：

```python
@dataclass(frozen=True)
class MktInstance:
    """MKT 实例 = 原始 MK + 行程时间矩阵 + 车辆数。

    ⚠️ **车辆数不是数据文件里的字段**（§19.7d）：HF2021 用 2 台、HGS 与 HA-DQN 用 v=m。
    **不同车数下的 Cmax 不可比**，故它只能是显式参数、且必须随结果一起报出去。
    ⚠️ P4-B 起矩阵**挂在 `base` 上**（`base.trans_time_full`，含 LU 的全矩阵）——正是它让
    "口径跟随实例"贯通整条训练/评估栈；`trans_time` 是本对象提供的**丢 LU 视图**（P4-A 语义不变）。
    """

    base: Instance
    n_agv: int
    layout_m: int

    @property
    def trans_time(self) -> np.ndarray:
        """机台间矩阵（丢 LU 的 m×m）——`base.trans_time_full[1:, 1:]` 的**视图**（只读用）。"""
        return self.base.trans_time_full[1:, 1:]

    @property
    def name(self) -> str:
        return f"{self.base.source}（MKT, m={self.layout_m}, v={self.n_agv}）"


def load_mkt(name: str, n_agv: int | None = None) -> MktInstance:
    """载入 MKT 实例。`n_agv=None` -> **v = m**（HGS/HA-DQN 的设定）。

    ⚠️ **加工数据一字不改**（有测试逐作业比对）：布局矩阵与车辆数是 MKT 相对 MK 的**全部**增量。
    ⚠️ 布局按**机台数**选（不是按实例名）——故 mk06 与 mk10 共用同一份 15 机布局。
    ⚠️ P4-B 起把实例标成 **matrix 口径**并挂上**含 LU 的全矩阵**（`drop_lu=False`）——
       `AgvSim` 的节点→矩阵下标映射需要第 0 位是 LU 的那个全矩阵。
    """
    base = load_mk(name)                      # 加工数据一字不改（有测试钉住）
    m = base.n_machines
    full = load_mkt_layout(m, drop_lu=False)  # ⚠️ **含 LU**：丢 LU 的 m×m 会被口径对象拒绝
    base = replace(base, transport="matrix", trans_time_full=full)
    return MktInstance(base=base, n_agv=(m if n_agv is None else int(n_agv)), layout_m=m)
```

（文件头补 `from dataclasses import dataclass, replace`。）

**(c) `plcsp/env/transport.py`** —— 追加 `for_instance`（放在 `from_matrix` 之后）：

```python
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
```

**(d) `plcsp/env/des.py`**：

1. 模块头 import：`from .transport import (MATRIX, UNMAPPED_GEOMETRY, UNMAPPED_RAISE, TransportCaliber)`；stdlib 加 `import hashlib`。
2. `SimConfig` 末尾（`due_range` 之后）加：

```python
    # 矩阵口径下**未被矩阵覆盖的端点**（充电桩等）怎么算行程：`raise`（默认，显式报错）或
    # `geometry`（**声明式**降级为几何口径，并进 metrics 计数）。它进 `_cfg_key`（失效安全）。
    # ⚠️ 这**不是**"口径开关"：口径永远跟随实例（见 transport.py 的共存规则），本项只管
    # "矩阵覆盖不到的那几段"。
    transport_unmapped: str = UNMAPPED_RAISE
```

3. `AgvSim.__init__`：签名在 `m_dm` 之后插 `transport: TransportCaliber`，函数体加：

```python
        self.transport = transport
        self.speed_ratio = spec.speed_factor if hetero else 1.0   # ⑩ 的**相对**倍率（关掉恒 1）
```

4. `AgvSim._seg_min` —— **函数体一字不改**（保既有数字逐位不变），只改 docstring：

```python
    def _seg_min(self, u: int, v: int) -> float:
        """节点 u→v 的**几何**行驶时长 [min]（格点最短路 ÷ 有效车速 ÷ 60）。

        ⚠️ 矩阵口径**不走这里**（矩阵已是分钟，不得再换算）——那条路走 `_leg_min`。
        """
        return float(self.m_dm[u, v]) / (self.cfg.eff_speed * self.speed) / SECONDS_PER_MIN
```

5. 新增 `_leg_min`（放在 `_seg_min` 之后）：

```python
    def _leg_min(self, src: int, dst: int, *, count_unmapped: bool = True) -> float:
        """**整段** src→dst 的行驶时长 [min]——两种口径的唯一汇合点（不含区段拆分）。

        - 几何口径：转调 `_seg_min`（现状，逐位不变）；
        - 矩阵口径：**直接查表**（矩阵已是分钟，**不得**再套任何换算），只再乘 ⑩ 的**相对**
          速度倍率（⑩ 关 ⟹ 倍率恒 1 ⟹ 与"该约束从未存在"逐位相同）；
        - 任一端点**没有矩阵对应项**（充电桩）⟹ 按 `transport.unmapped` 处置：`raise` 显式报错；
          `geometry` **声明式**降级为几何口径并计数（metrics 带出，供表里如实声明）。
        `count_unmapped=False` 供**排序**之类的查数用（只选一个桩却把候选全计一遍会虚高）。
        ⚠️ 计数按**出发次数**记：区段争用失败而重试的腿会重复计一次（它确实又跑了一趟）。
        """
        if self.transport.mode != MATRIX:
            return self._seg_min(src, dst)
        t = self.transport.minutes(src, dst)
        if t is not None:
            return t / self.speed_ratio
        if self.transport.unmapped == UNMAPPED_RAISE:
            raise ValueError(
                f"节点 {src}→{dst} 在行程时间矩阵里**没有对应项**（矩阵只覆盖机台与装卸站）——"
                f"若要跑含充电桩的口径，请显式设 SimConfig.transport_unmapped='geometry'")
        self.stats["unmapped_legs"] = self.stats.get("unmapped_legs", 0) + 1
        got = self._seg_min(src, dst)
        self.stats["unmapped_min"] = self.stats.get("unmapped_min", 0.0) + got
        return got
```

6. `_drive` —— 区段开与关两条路的时长来源改为"整段先算、按几何占比摊"：

```python
        if not self.congestion:                 # ① 关：不申请区段，一次到底
            seg = self._leg_min(src, dst)
            ...                                  # 以下整段不变（timeout / _drain / stats）
        # 逐段申请区段：持当前 → 申请下一 → 成功才放上一 → 走这一段
        path = shortest_node_path(self.g, src, dst)
        # ⚠️ 矩阵口径下**整段时长先算出来**，逐段只决定**分摊比例**（占比之和恒为 1 ⟹
        # 各段之和 == 矩阵查表值，有测试钉）；矩阵里没有"中间走廊节点"这一说，逐段查表必失败。
        total = self._leg_min(src, dst) if self.transport.mode == MATRIX else None
        leg_geo = self._seg_min(src, dst) if total is not None else 0.0
        ...
            seg = self._seg_min(path[pi], path[pi + 1])
            if total is not None:
                seg = total * seg / leg_geo
```

7. `_maybe_charge` 的充电桩排序改用同一入口（几何口径下与旧的按距离排序**同序**，因为只差一个正的常数因子）：

```python
        for ch in sorted(self.chargers, key=lambda c: self._leg_min(src, c.node, count_unmapped=False)):
```

> ℹ️ **能耗自动跟随矩阵**：`_drain(leg, seg)` 读的就是这里算出的分钟数（kWh = kW × min ÷ 60），
> 故矩阵口径的行程变长 ⟹ AGV 能耗同比变长，两档的 `energy` 会真的不同（Task 4 有测试钉）。

8. `SimWorld.__init__`：`_cold_start()` 之前加

```python
        self.transport = TransportCaliber.for_instance(inst, layout,
                                                       unmapped=self.cfg.transport_unmapped)
```

`_build_entities` 的 `AgvSim(...)` 调用在 `self.m_dm` 之后传 `self.transport`。

9. 两个 stats 字典（`run()` 与 `run_gated()`）各加 `"unmapped_legs": 0, "unmapped_min": 0.0`；两个返回的 metrics 字典各加：

```python
                "transport": self.inst.transport,
                "unmapped_legs": stats.get("unmapped_legs", 0),
                "unmapped_min": float(stats.get("unmapped_min", 0.0)),
```

10. `_instance_key` —— 把**口径与矩阵**纳入指纹：

```python
def _instance_key(inst: Instance) -> tuple:
    """实例内容指纹——用于缓存 `M_ref`（比 id() 稳，比文件名稳）。

    ⚠️ P4-B：**行程时间口径与矩阵必须进键**——同名 MK 与 MKT 实例只差一张矩阵，漏了它
    `reference_run` 会把几何口径的参考运行**静默喂给**矩阵口径的运行（M_ref、奖励权重、
    特征归一化一起错，且零报错）。
    """
    mat = getattr(inst, "trans_time_full", None)
    digest = (None if mat is None else
              hashlib.blake2b(np.ascontiguousarray(mat, dtype=float).tobytes(),
                              digest_size=8).hexdigest())
    return (inst.n_machines, inst.n_jobs, getattr(inst, "transport", "geometry"),
            tuple(tuple(min(t for _, t in op) for op in job) for job in inst.jobs), digest)
```

**(e) `plcsp/env/due_dates.py`** —— 标定表**按口径分表**（键仍是文件名主干，既有 `TF_RDD` 一字不动）：

```python
# ══ 标定表：逐实例标定的 (τ, R)，**按行程时间口径分表** ══
# ⚠️ P4-B：`load_mkt(name)` 内部就是 `load_mk(name)`（加工数据一字不改）⟹ **文件名主干相同**，
# 只按主干取键会让 MKT 实例**静默命中**原始 MK 的 (τ,R)——那是两个不同问题共用一把尺。
# 故键仍是主干（"mk01"），**分口径靠选哪张表**：口径未知 ⟹ 显式报错，不得回退到几何表。
TF_RDD: dict[str, tuple[float, float]] = { ... 既有 10 项，不动 ... }

# 矩阵口径（MKT）的 (τ,R)：由 `plcsp/m16_due_calib.py --transport matrix` 标定后落盘（Task 3）。
# 在落盘前它是空的 ⟹ 矩阵口径的 `due_dates_for` 会**显式报错**（不会退回几何表）。
TF_RDD_MATRIX: dict[str, tuple[float, float]] = {}

TABLE_BY_CALIBER: dict[str, dict[str, tuple[float, float]]] = {
    "geometry": TF_RDD,
    "matrix": TF_RDD_MATRIX,
}


def _caliber_of(inst: Instance) -> str:
    """实例的行程时间口径标签（缺省 geometry）——决定用哪张标定表。"""
    return getattr(inst, "transport", "geometry")


def _instance_name(inst: Instance) -> str | None:
    """从 `Instance.source` 取标定表键（如 `Mk01.fjs` → `mk01`）；取不出返回 None。

    ⚠️ 用文件**名**而非内容指纹：标定表本来就是逐 **实例名** 的（mk01–mk10）。
    `gen_random` 的 source 是描述串，必不命中。
    """
    src = (inst.source or "").strip()
    if not src:
        return None
    return Path(src).stem.lower()
```

`due_dates_for` 的查表改为：

```python
    if tau is None or due_range is None:
        caliber = _caliber_of(inst)
        try:
            table = TABLE_BY_CALIBER[caliber]
        except KeyError:
            raise ValueError(
                f"未知行程时间口径 {caliber!r}——无法确定用哪张交期标定表"
                f"（已知：{sorted(TABLE_BY_CALIBER)}）") from None
        key = _instance_name(inst)
        calib = table.get(key) if key else None
        if calib is None:
            raise ValueError(
                f"实例 {key or inst.source or '<无 source>'} 在 **{caliber}** 口径下未标定交期 "
                f"(τ, R)——{caliber} 表只覆盖 {sorted(table)}。矩阵口径请跑 "
                f"`plcsp/m16_due_calib.py --transport matrix`；或显式传入 tau= 与 due_range=。")
        ...
```

- [ ] **Step 4: 跑测试确认通过 + 全套回归 + 变异自检**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_transport_wiring.py -q --no-header`
Expected: PASS

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header`
Expected: 全绿（**293 + 约 16**，约 6 分钟）
> ⚠️ **若 `test_travel_time_is_small_fraction_of_makespan` 或别的既有测试红了**：先确认它跑的是**几何口径**（原始 MK）——本任务不该改变几何口径的任何数字（`_seg_min` 逐字未动）。红了就是回归，**不要**改测试。

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m ruff check plcsp/`

> ⚠️ **变异自检（必做，写进报告）**：① 在 `_leg_min` 的矩阵分支里加回 `/ self.cfg.eff_speed / SECONDS_PER_MIN`，确认 `test_matrix_travel_time_equals_the_lookup_sum_exactly` 变红；② 把 `_drive` 的区段分摊注释掉（整段走 `_seg_min`），确认 `test_zone_split_preserves_the_matrix_total` 变红；③ 在 `_leg_min` 去掉 `speed_ratio`，确认速度倍率测试变红。三条都要变红，改回后全绿。

- [ ] **Step 5: 提交**

```bash
git add plcsp/env/instances.py plcsp/env/mkt.py plcsp/env/transport.py plcsp/env/des.py plcsp/env/due_dates.py plcsp/tests/test_transport_wiring.py
git commit -m "feat: MKT 行程时间矩阵接入 AgvSim（口径跟随实例、缓存键与交期键分口径）（P4-B Task 2）"
```

---

### Task 3: MKT 口径的交期标定（冻结 `TF_RDD_MATRIX`）

**Files:**
- Modify: `plcsp/m16_due_calib.py`（加 `--transport` / `--n-agv`；网格上限按口径取；贴回块打印分口径表）
- Modify: `plcsp/env/due_dates.py`（填入 `TF_RDD_MATRIX` 的冻结值）
- Test: `plcsp/tests/test_due_calib_mkt.py`（**LF**）

**Interfaces:**
- Consumes: Task 2 的矩阵口径（`load_mkt(name).base` + `SimConfig(transport_unmapped=...)`）
- Produces:
  - `reference_completes(inst, cfg: SimConfig | None = None, constraints=None) -> dict[int, float]`（**签名变**：收 cfg）
  - `calibrate_report(inst, *, cfg=None, target_ref=0.45, improve=0.88, lo=0.20, hi=0.75) -> dict`、`calibrate(inst, *, cfg=None, ...) -> tuple[float, float]`（**签名变**）
  - `TAU_HI_BY_CALIBER: dict[str, float]` 与 `_grid` 的按口径调用
  - `plcsp/env/due_dates.py::TF_RDD_MATRIX` 的 10 个冻结项

- [ ] **Step 1: 先量再定网格上限（**不得跳过**）**

矩阵口径下行程是**分钟的整数级**，而几何口径约 **0.2 min/腿**（`d/(eff·speed)/60`：6 m ÷ 0.5 m/s ÷ 60）——
参考 makespan 会大数倍，而 `d_j = LB·τ·(1+R(2ρ−1))` 里 `LB` 只读加工数据（不变）⟹ **τ 会同比变大**。
既有 `TAU_HI = 6.00` 是给几何口径留的余量，**很可能不够**（本机粗估：mk01 的矩阵参考 makespan 或达 200–300，而 LB = 25.5 ⟹ τ 需 ~8–12）。

先跑一条探针（用 Task 2 的接线；把比值**记下来**，它是 TAU_HI 的依据，也是论文里"τ 为什么这么大"的证据）：

```bash
PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -c "
from plcsp.env.constraints import ConstraintConfig
from plcsp.env.des import SimConfig, rollout
from plcsp.env.due_dates import workload_lower_bound
from plcsp.env.mkt import load_mkt
cons = ConstraintConfig().with_off('due_dates')
for n in ('mk01', 'mk06', 'mk07', 'mk10'):
    mkt = load_mkt(n); m = mkt.layout_m
    r = rollout(mkt.base, seed_chain=0,
                cfg=SimConfig(n_agv=m, transport_unmapped='geometry'), constraints=cons)
    lb = workload_lower_bound(mkt.base)
    print(n, 'M_ref=', round(r['makespan'], 2), 'LB=', round(lb, 2),
          'ratio=', round(r['makespan'] / lb, 2), 'horizon_hit=', r['horizon_hit'])
"
```

**本机粗估**（写此计划时按矩阵均值 7.3 min/腿、mk01 有 45 条负载腿 + 约等量空载腿、v=m 算得）：
mk01 的 `M_ref(MKT)/LB ≈ 8–10`（几何口径同实例 ≈ 4.05，τ 现为 2.45）。**若实测比值比粗估更大**，按实测取。

据此定 `TAU_HI_BY_CALIBER["matrix"]`（= 实测最大 ratio × 1.5 向上取整；例如实测 ratio ≤ 12 ⟹ 取 **20.0**），
并把实测比值写进脚本注释与报告。
⚠️ **不得顺手抬高几何口径的 `TAU_HI`**——搜索空间一开，几何表可能选出不同的 (τ,R)，作废 §21.2 的冻结表（`test_frozen_table_is_reproduced_by_the_calibration_script` 会红）。

- [ ] **Step 2: 写失败测试**

创建 `plcsp/tests/test_due_calib_mkt.py`：

```python
"""MKT（矩阵）口径的交期标定测试（P4-B Task 3）。

为什么要有这张表：矩阵口径下参考 makespan 大数倍 ⟹ 交期若沿用几何口径的 (τ,R)，
误期率会整体塌到 0（TWT 变死目标，`progress-log §20.4-2` 的老病）。
"""
from __future__ import annotations

import pytest

from plcsp.env.mkt import load_mkt

MK10 = [f"mk{i:02d}" for i in range(1, 11)]


@pytest.mark.unit
def test_matrix_table_exists_and_is_not_the_geometry_table():
    """⚠️ Review Focus #6：两张表**必须都在**，且值不得相同——口径不同 ⟹ 尺不同。"""
    from plcsp.env.due_dates import TF_RDD, TF_RDD_MATRIX

    assert sorted(TF_RDD) == MK10
    assert sorted(TF_RDD_MATRIX) == MK10, "矩阵口径的表没落盘（或缺项）"
    same = [n for n in MK10 if TF_RDD[n] == TF_RDD_MATRIX[n]]
    assert not same, f"这些实例两个口径的 (τ,R) 完全相同（疑似照抄）：{same}"


@pytest.mark.unit
def test_mkt_due_dates_actually_read_the_matrix_table():
    """⚠️ Review Focus #6 的**直接判据**：MKT 实例算出的交期必须等于"用矩阵表的 (τ,R)"那一组。

    只断言"两表不同值"还不够——若 `due_dates_for` 读了另一张表，值也会不同且不报错。
    这里把"用的是哪组参数"钉死：等于矩阵表的那组，且**不等于**几何表的那组。
    """
    from plcsp.env.due_dates import TF_RDD, TF_RDD_MATRIX, due_dates_for, tf_rdd_due_dates

    mkt = load_mkt("mk01").base
    tau_m, r_m = TF_RDD_MATRIX["mk01"]
    tau_g, r_g = TF_RDD["mk01"]
    assert due_dates_for(mkt) == tf_rdd_due_dates(mkt, tau_m, r_m)
    assert due_dates_for(mkt) != tf_rdd_due_dates(mkt, tau_g, r_g)


@pytest.mark.unit
@pytest.mark.parametrize("name", MK10)
def test_matrix_table_is_reproduced_by_the_script(name: str):
    """⚠️ Review Focus：`TF_RDD_MATRIX` 必须是标定脚本的**可复现产物**，不是手抄魔数。"""
    from plcsp.env.des import SimConfig
    from plcsp.env.due_dates import TF_RDD_MATRIX
    from plcsp.m16_due_calib import calibrate

    mkt = load_mkt(name)
    cfg = SimConfig(n_agv=mkt.layout_m, transport_unmapped="geometry")
    tau, r = calibrate(mkt.base, cfg=cfg)
    tab_tau, tab_r = TF_RDD_MATRIX[name]
    assert tab_tau == pytest.approx(tau, abs=1e-9), f"{name}: 表 τ={tab_tau} ≠ 脚本 {tau}"
    assert tab_r == pytest.approx(r, abs=1e-9), f"{name}: 表 R={tab_r} ≠ 脚本 {r}"


@pytest.mark.unit
@pytest.mark.parametrize("name", ["mk01", "mk07", "mk10"])
def test_mkt_due_dates_make_twt_alive_at_the_reference_point(name: str):
    """判据：矩阵口径下，**参考策略**的误期率必须落在标定带内（否则目标又死了）。

    这条与几何口径的 `test_calibrated_due_dates_stay_live_after_improvement` 同型，
    只是换了口径——**不得**复用几何表来通过它。
    """
    from plcsp.env.des import SimConfig, rollout
    from plcsp.env.due_dates import due_dates_for

    mkt = load_mkt(name)
    d = due_dates_for(mkt.base)
    cfg = SimConfig(n_agv=mkt.layout_m, transport_unmapped="geometry")
    C = rollout(mkt.base, seed_chain=0, cfg=cfg)["completes"]
    rate = sum(1 for j in C if C[j] > d[j]) / mkt.base.n_jobs
    assert 0.15 < rate < 0.85, f"{name}: 矩阵口径参考误期率 {rate:.2f}——交期又成死目标"
```

- [ ] **Step 3: 跑测试确认失败**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_due_calib_mkt.py -q --no-header`
Expected: FAIL — `AssertionError: 矩阵口径的表没落盘（或缺项）` / `TypeError: calibrate() got an unexpected keyword argument 'cfg'`

- [ ] **Step 4: 实现**

**(a) `plcsp/m16_due_calib.py`**：

- `TAU_LO, TAU_HI, TAU_STEP` 保留（几何口径的网格**不得**动），新增：

```python
# ⚠️ 网格上限**按口径**取：矩阵口径的行程是分钟的整数级（几何口径 ~0.2 min/腿），参考 makespan
# 大数倍 ⟹ 同一个 LB 下 τ 要同比放大。`matrix` 的值 = Step 1 实测最大 ratio × 1.5 向上取整
# （本机粗估 ratio≤12 ⟹ 20.0；**实测后按实测填，并把实测比值记在下一行**）。
# ⚠️ **不要**抬高几何口径的上限——搜索空间一开，冻结表可能选出不同的 (τ,R)，作废 §21.2 的表。
TAU_HI_BY_CALIBER: dict[str, float] = {"geometry": TAU_HI, "matrix": 20.0}
```

- `_grid` 保持；`TAU_GRID` 改名/保留为几何口径的网格，另加 `tau_grid(caliber: str) -> np.ndarray`。
- `reference_completes(inst, cfg=None, constraints=None)`：把写死的 `SimConfig()` 换成参数（默认 `None` → `SimConfig()`，**几何口径的数字逐位不变**）。
- `calibrate_report` / `calibrate` 加 `cfg=None`，并把 `TAU_GRID` 换成 `tau_grid(caliber)`（口径从 `cfg.transport_unmapped`/实例标签推，或直接由 CLI 传）。
- CLI 加 `--transport {geometry,matrix}` 与 `--n-agv`（默认 `"default"`；`matrix` 口径下 `"default"` = **v = m**，即 HGS/HA-DQN 口径）。`matrix` 口径的默认 cfg = `SimConfig(n_agv=m, transport_unmapped="geometry")`（⑪ 充电在标定配置里是开的，桩没有矩阵项 ⟹ 必须显式选降级策略**并计数**）。
- 贴回块按口径打印：

```python
    print(f"\n# ── 贴回 plcsp/env/due_dates.py 的 {table_name} ──")
    print(f"{table_name}: dict[str, tuple[float, float]] = {{")
```

- 模块 docstring 补：网格上限按口径取的理由与**实测比值**；`matrix` 口径的 cfg 条件（v=m、`transport_unmapped='geometry'`）；**"误期率落带是 cfg 条件"** 这条继续成立。

**(b) `plcsp/env/due_dates.py`**：把脚本输出的 10 项填进 `TF_RDD_MATRIX`，并在表上方写明：标定 cfg（`n_agv = m`、`transport_unmapped='geometry'`、`ConstraintConfig().with_off('due_dates')`）、网格、脚本命令。

- [ ] **Step 5: 跑测试确认通过 + 全套回归**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_due_calib_mkt.py plcsp/tests/test_due_calib.py -q --no-header`
Expected: PASS（含既有几何表不动）

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header`
Expected: 全绿（约 6 分钟；**实测本任务的 3 条 TWT 测试会跑矩阵 rollout，报告里如实报墙钟增量**）

> ⚠️ **若某实例在网格上无可行 (τ,R)**：**不得**放宽 `lo/hi` 或改评分规则去凑数。如实把该实例
> 留在表外（`due_dates_for` 会显式报错），在报告与 progress-log 里写明"哪个实例、卡在哪个约束"，
> 并在论文口径里注明该实例只能报 ⑧ 关闭的那一档。

- [ ] **Step 6: 提交**

```bash
git add plcsp/m16_due_calib.py plcsp/env/due_dates.py plcsp/tests/test_due_calib_mkt.py
git commit -m "feat: MKT 口径交期标定 + 冻结 TF_RDD_MATRIX（P4-B Task 3）"
```

---

### Task 4: 两档报告（档 A = MKT 口径、档 B = 完整口径）+ m14 绊线翻转

> **档 A 就是 `ABLATION_GROUPS["None"]`**：MKT = FJSP + 运输，本项目的**十个**机制（①拥堵 ②有限缓冲
> ③机台故障 ④返工 ⑤换型 ⑧交期 ⑨AGV故障 ⑩异构车队 ⑪充电 ⑫维护）**全都没有对应项** ⟹ 档 A 的机制侧
> 恰好等于"全关"。故**复用现有配置对象，不新造**（单一真相；`test_constraints.py` 已钉 "None 组全关"）。
> 档 B = `ABLATION_GROUPS["Full"]`（全开），运输仍走矩阵。两档的区别**只在机制**，口径（矩阵）相同。

**Files:**
- Modify: `plcsp/env/constraints.py`（加 `REPORT_TIERS`）
- Modify: `plcsp/m14_mkt_reference.py`（改用 MKT 实例；带 `tier` 列；`ours_benchmark` 改 `"MKT"`）
- Modify: `plcsp/tests/test_mkt_reference.py`（**改写绊线**，不删除——计数不减）
- Test: `plcsp/tests/test_two_tier_report.py`（**LF**）

**Interfaces:**
- Consumes: Task 2/3 的矩阵口径与 `TF_RDD_MATRIX`
- Produces:
  - `constraints.REPORT_TIERS: dict[str, ConstraintConfig]`（`{"A-MKT": ABLATION_GROUPS["None"], "B-Full": ABLATION_GROUPS["Full"]}`）
  - `m14.reference_table(names, n_agv=None, seeds=3, tiers=("A-MKT", "B-Full")) -> list[dict]`，每行 = (实例 × 档)，字段含 `tier` / `transport="matrix"` / `ours_benchmark="MKT"` / `unmapped_legs` / `unmapped_min` / `horizon_hit`

- [ ] **Step 1: 写失败测试**

创建 `plcsp/tests/test_two_tier_report.py`：

```python
"""两档口径报告的测试（P4-B Task 4）。"""
from __future__ import annotations

import dataclasses

import pytest

from plcsp.env.constraints import ABLATION_GROUPS, ConstraintConfig, REPORT_TIERS


@pytest.mark.unit
def test_tier_a_is_exactly_the_existing_none_group():
    """★ 用户裁定的落点：档 A（MKT 口径）**就是** `ABLATION_GROUPS["None"]`——不新造配置。

    十个机制（①拥堵 ②有限缓冲 ③机台故障 ④返工 ⑤换型 ⑧交期 ⑨AGV故障 ⑩异构车队 ⑪充电 ⑫维护）
    MKT 一个都没有 ⟹ "MKT 没有对应项的机制全关" == "全关"。
    """
    assert REPORT_TIERS["A-MKT"] is ABLATION_GROUPS["None"], "档 A 不是复用而是新造了一个配置"
    assert REPORT_TIERS["B-Full"] is ABLATION_GROUPS["Full"]


@pytest.mark.unit
def test_tier_a_turns_off_every_dataclass_field():
    """⚠️ **按 dataclass 字段**逐个查（不是查 `test_constraints.py` 里硬编码的 10 个名字）。

    将来加第 11 个机制时，若它 MKT 也没有对应项而 `ABLATION_GROUPS["None"]` 忘了关，
    档 A 就不再是"MKT 口径"了——这条会红。
    """
    on = [f.name for f in dataclasses.fields(ConstraintConfig) if getattr(REPORT_TIERS["A-MKT"], f.name)]
    assert not on, f"档 A 里还开着的机制：{on}——若该机制 MKT 确实有对应项，请补进文档并说明理由"


@pytest.mark.unit
def test_both_tiers_run_on_the_matrix_and_differ():
    """两档都必须**跑得出且不同**——相同则说明机制没接进仿真（消融表的老毛病）。"""
    from plcsp.env.mkt import load_mkt
    from plcsp.m14_mkt_reference import reference_table

    rows = {r["tier"]: r for r in reference_table(("mk01",), n_agv=6, seeds=1)}
    a, b = rows["A-MKT"], rows["B-Full"]
    assert a["transport"] == b["transport"] == "matrix"
    assert a["makespan"] != b["makespan"], "两档跑出同一 Cmax——机制没接上"
    assert a["energy"] != b["energy"], "两档 energy 相同——运输/约束没进能耗"


@pytest.mark.unit
def test_tier_a_has_no_unmapped_legs_and_tier_b_declares_its_own():
    """⚠️ Review Focus #5：档 A **不该**有未映射路段（⑪ 关 ⟹ 车不会开去充电桩）。

    档 B 开着 ⑪，可能产生未映射路段——**计数与分钟数必须随行带出**，供论文如实声明
    "这一档里有 x% 的行程是几何口径补的"。
    """
    from plcsp.m14_mkt_reference import reference_table

    rows = {r["tier"]: r for r in reference_table(("mk01",), n_agv=6, seeds=1)}
    assert rows["A-MKT"]["unmapped_legs"] == 0, "档 A 出现未映射路段——有 MKT 没有的机制在驱使 AGV"
    for tier, r in rows.items():
        assert "unmapped_legs" in r and "unmapped_min" in r, f"{tier} 没带未映射计数"
        assert r["unmapped_min"] >= 0.0


@pytest.mark.unit
def test_forced_low_battery_under_matrix_raises_by_default():
    """⚠️ Review Focus #5 的**不happy 路径**：矩阵口径 + ⑪ 开 + 默认策略 ⟹ 充电腿必须**显式报错**。

    用 `battery_low=0.99 / battery_high=1.0` **力迫**第一趟之后就充电（默认参数下矩阵口径的
    耗电也确实会到这一档，见 §5.7 的对比），把"静默猜一个值"这条路堵死。
    """
    from plcsp.env.des import SimConfig, rollout
    from plcsp.env.mkt import load_mkt

    mkt = load_mkt("mk01")
    with pytest.raises(ValueError, match="没有对应项"):
        rollout(mkt.base, seed_chain=1,
                cfg=SimConfig(n_agv=6, battery_low=0.99, battery_high=1.0,
                              transport_unmapped="raise"),
                constraints=ConstraintConfig())


@pytest.mark.unit
def test_forced_low_battery_under_matrix_counts_the_geometry_fallback():
    """⚠️ Review Focus #5：显式选 `geometry` 时充电腿**跑得通且留痕**（段数 + 分钟都 > 0）。

    这一档就是**档 B 的真实形态**（⑪ 开、矩阵无充电桩项）——论文里必须如实声明这一部分的占比。
    """
    from plcsp.env.des import SimConfig, rollout
    from plcsp.env.mkt import load_mkt

    mkt = load_mkt("mk01")
    r = rollout(mkt.base, seed_chain=1,
                cfg=SimConfig(n_agv=6, battery_low=0.99, battery_high=1.0,
                              transport_unmapped="geometry"),
                constraints=ConstraintConfig())
    assert r["charge_events"] > 0, "力迫低电却没充过电——本测试的前提不成立"
    assert r["unmapped_legs"] > 0 and r["unmapped_min"] > 0.0


@pytest.mark.unit
def test_rows_carry_tier_and_caliber_labels():
    """⚠️ Review Focus：两档的数并排放时必须**逐行自报档位与口径**（P4-A Review Focus #3 同型）。"""
    from plcsp.m14_mkt_reference import reference_table

    for r in reference_table(("mk01",), n_agv=6, seeds=1):
        assert r["tier"] in ("A-MKT", "B-Full")
        assert r["ours_benchmark"] == "MKT"
        assert r["benchmark"] == "MKT"          # 对照列的口径（未变）


@pytest.mark.unit
def test_no_horizon_truncation_in_the_tested_configs():
    """⚠️ Review Focus #8：矩阵口径下 makespan 大数倍，固定护栏可能**掐表**。

    掐表的结果只是"看起来更慢"，会被当成正常读数——故必须逐行检查。
    """
    from plcsp.m14_mkt_reference import reference_table

    for r in reference_table(("mk01",), n_agv=6, seeds=1):
        assert r["horizon_hit"] is False, f"{r['tier']} 掐表了——先按声明的方式抬护栏（它是护栏不是物理量）"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_two_tier_report.py -q --no-header`
Expected: FAIL — `ImportError: cannot import name 'REPORT_TIERS'`

- [ ] **Step 3: 实现**

**(a) `plcsp/env/constraints.py`** 末尾追加：

```python
# ── P4-B：两档报告（用户裁定 2026-10-03）──
# 档 A「MKT 口径」= 文献（MKT = FJSP + 运输）没有对应项的机制全关 **⟹ 恰好等于 None 组**，
#   故**复用**它而不是新造配置（单一真相；改动只走 `with_off`）。
# 档 B「完整口径」= 机制全开，运输口径**相同**（都是行程时间矩阵）——两档的差别**只在机制**。
REPORT_TIERS: dict[str, ConstraintConfig] = {
    "A-MKT": ABLATION_GROUPS["None"],
    "B-Full": ABLATION_GROUPS["Full"],
}
```

**(b) `plcsp/m14_mkt_reference.py`**：

- 模块 docstring 与 `OURS_BENCHMARK_RAW_MK` 一并改：`OURS_BENCHMARK = "MKT"`（**绊线测试翻转**）；写清
  「本表两档都跑在**矩阵口径**上；档 A 是退化特例（= `ABLATION_GROUPS["None"]`），档 B 机制全开；
  未映射路段用几何口径补的段数与分钟数逐行带出」。
- `reference_table` 改为：

```python
def reference_table(names: tuple[str, ...], n_agv: int | None = None, seeds: int = 3,
                    tiers: tuple[str, ...] = ("A-MKT", "B-Full")) -> list[dict]:
    """每 (实例 × 档) 一行：我们（参考调度）+ 已发表数字。

    ⚠️ 行程时间**一律来自矩阵**（`load_mkt` 造的实例自带 matrix 口径）。
    ⚠️ 档 B 开着 ⑪ 充电，而充电桩在矩阵里**没有对应项** ⟹ 显式选
       `transport_unmapped='geometry'` 并把段数/分钟数带进行里（**声明式**降级，不静默）。
    ⚠️ `horizon_hit` 必须随行带出：矩阵口径下 makespan 大数倍，掐表会伪装成"更慢"。
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
                         "ours": float(st.mean(ms)), "ours_benchmark": "MKT",
                         "unmapped_legs": sum(r["unmapped_legs"] for r in got),
                         "unmapped_min": sum(r["unmapped_min"] for r in got),
                         "horizon_hit": any(r["horizon_hit"] for r in got),
                         "published_agv": dict(MKT_PUBLISHED_AGV),
                         "comparable_fleet": {k: _same_fleet(s, v, m)
                                              for k, s in MKT_PUBLISHED_AGV.items()},
                         **MKT_PUBLISHED[name]})
    return rows
```

- `main()` 的表头加「档」「未映射(段/min)」两列，并先打印一段**口径声明**：本表 ours 与对照列**同口径**
  （MKT 矩阵）；档 A = `ABLATION_GROUPS["None"]`（退化特例）、档 B = 全开；未映射路段（充电桩）用几何口径补。

**(c) `plcsp/tests/test_mkt_reference.py`** —— 把 `test_our_column_declares_its_own_basis_and_is_not_mkt_yet`
**改写成** `test_our_column_declares_the_mkt_basis`（断言 `== "MKT"`），并在 docstring 里写明"这是 P4-A 埋的
绊线，本批按它的要求翻转"。⚠️ **改写不删除**（计数不减）。

- [ ] **Step 4: 跑脚本，把表贴进报告**

Run:
```bash
PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_two_tier_report.py plcsp/tests/test_mkt_reference.py -q --no-header
PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m14_mkt_reference --instances mk01,mk07,mk10 --n-agv default --seeds 3
```
Expected: 两档表落 stdout（6 行 = 3 实例 × 2 档），并带未映射声明。**把 stdout 原样贴进 progress-log**（数字不得手抄/调整）。

- [ ] **Step 5: 全套回归 + 提交**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header` → 全绿
Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m ruff check plcsp/` → clean

```bash
git add plcsp/env/constraints.py plcsp/m14_mkt_reference.py plcsp/tests/test_mkt_reference.py plcsp/tests/test_two_tier_report.py
git commit -m "feat: 两档口径表（档 A = ABLATION_GROUPS['None']、档 B = 全开）+ m14 绊线翻转（P4-B Task 4）"
```

---

### Task 5: 文档（spec / INDEX / progress-log）

**Files:**
- Modify: `docs/superpowers/specs/2026-10-02-plcsp-rebuild-design.md`（§3.2 补"行程时间口径"；§3.5 补矩阵口径的 (τ,R) 表；§6.1/§6.2 写两档与 HGS 归属）
- Modify: `docs/INDEX.md`（新增 §5.11；更新 §5 待办里已完成的 `trans_time` 项）
- Modify: `docs/progress-log.md`（新增 **§二十二**）

- [ ] **Step 1: spec §3.2 补一小节「行程时间口径（P4-B）」**

必须含：① 共存规则（口径**跟随实例**；原始 MK 几何、MKT 矩阵；**没有**全局开关及其理由）；
② 矩阵口径的三条硬约束（已是分钟、`(m+1)×(m+1)` 含 LU 且机台 i ↔ 下标 i+1、非对称需保序）；
③ 未映射端点（充电桩）的两种策略与**档 B 用的是"声明式几何降级 + 计数"**；
④ ① 开时区段拆分的**按几何占比摊**规则（占比和恒 1 ⟹ 各段之和 = 矩阵查表值）；
⑤ **诚实边界**：LU 今天没有消费者（作业在首工序机台入场、末工序机台完工）——**矩阵的 LU 行/列目前不计入 makespan**。

- [ ] **Step 2: spec §3.5 补矩阵口径的 (τ,R)**

贴 `TF_RDD_MATRIX` 与标定 cfg（`n_agv = m`、`transport_unmapped='geometry'`）、网格上限按口径取的理由与**实测比值**，
并保留既有那句"误期率落带是 cfg 条件"。

- [ ] **Step 3: spec §6.1/§6.2 写两档与 HGS 归属**

档 A = `ABLATION_GROUPS["None"]`（退化特例，与已发表数字同口径）、档 B = 全开；两档都跑矩阵。
明确写：**HGS 从零实现不在本批**（见 P4-A 计划「已知边界」），它是独立的下一批；本批交付的口径是它的**前置**。

- [ ] **Step 4: INDEX 与 progress-log**

- `docs/INDEX.md`：新增 **§5.11 P4-B MKT 行程时间矩阵接入（2026-10-03 完成）**（交付物、两档表读数、已知边界）；
  §5 的待办第 2 条把"`trans_time` 接进 `AgvSim`"划掉、只留 HGS。
- `docs/progress-log.md` 新增 **§二十二**，必须含：① 两档表**原样 stdout**；② MKT (τ,R) 冻结表与网格上限的实测比值；
  ③ MKT 参考 makespan vs LB 的比值实测（决定 τ 量级的那条）；④ 未映射路段的段数/分钟数占比；
  ⑤ 全套回归计数与墙钟；⑥ **开放线索**（见下方"已知边界"表逐条登记为线索）。

- [ ] **Step 5: 全套回归 + 提交**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header` → 全绿
Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m ruff check plcsp/` → clean

```bash
git add docs/superpowers/specs/2026-10-02-plcsp-rebuild-design.md docs/INDEX.md docs/progress-log.md
git commit -m "docs: P4-B 矩阵口径接线的实测记录、两档表与 spec 补节（P4-B Task 5）"
```

---

## 完成后的状态

- **行程时间口径跟随实例**：MKT 实例走矩阵（分钟，不再被任何几何换算重标），原始 MK 走几何（数字逐位不变）。
- **两档报告**：档 A = `ABLATION_GROUPS["None"]`（MKT 没有对应项的机制全关 ⟹ 与已发表数字同口径的退化特例）、
  档 B = 机制全开 + 矩阵运输；两档都带**档位/口径/车数/未映射路段**列，`horizon_hit` 逐行可见。
- **交期按口径分表**：MKT 不再静默复用 MK 的 (τ,R)，矩阵口径有自己冻结、可复现的标定表。
- `m14` 的绊线按 P4-A 的要求翻转成 `ours_benchmark == "MKT"`。

## 已知边界（**必须带进下一批 / 论文口径声明**）

| 项 | 说明 |
|---|---|
| **LU（装卸站）运输未建模** | 矩阵的 LU 行/列（下标 0）**今天没有消费者**：作业在首工序机台入场、末工序机台完工，没有"从装卸站取件/回送"这段。故我们的 makespan **不含回库运输**。若已发表口径含这一段，我们仍系统性偏低（量级 ≈ 一条矩阵项）。**HGS 那批须先核这一点**，再决定是否加 LU 运输（那会改问题定义、动全部读数）|
| **空载段是我们的建模选择** | 我们的 AGV 有"上一卸货点 → 本次取货点"的空载段（v0 没有，P1b 补的）。两档都带它。若已发表模型不含空载重定位，档 A 仍非逐位对齐——同属 HGS 那批要核的事 |
| **矩阵口径下"车速 × 车队"扫描轴对运输失效** | 矩阵已是绝对分钟，`agv_speed_mps` / `aisle_width` 不再影响行程（有测试钉）。spec §3.3.3 的"运输不在关键路径"结论必须**在矩阵口径下重测**——粗估运输负荷涨约一个数量级（mk01：45 条负载腿 × 均值 7.6 min ≈ 342 AGV·min，几何口径下同实例只有 ~9 min）|
| **⑪ 充电从"结构性死约束"变成 binding** | §5.7 的判定（"episode 耗电 0.27 kWh ≪ 电池 2–4 kWh"）**只在几何口径下成立**。矩阵口径下负载行驶 ≈ 342 min × 0.5 kW + 空载 ≈ 0.2 kW ⟹ 单车一趟 ~3.5 kWh，与容量同量级 ⟹ 充电事件与充电腿会真的出现（`charge_kw=3.0`，每次充电还要占几分钟到十几分钟）。论文的 §3.3.2 binding 判定表须按口径分列，**不得沿用几何口径的结论**|
| **HGS 从零实现** | 不在本批（P4-A 计划已裁定为独立下一批）。本批交付的**矩阵口径与两档表是它的前置** |
| **训练后的两档数字** | 本批的两档表是**参考调度**（未训练）口径。训练后的数字要重训——且 MKT 实例的 `due_margin` 输入因新交期而变，**旧 checkpoint 在 MKT 实例上不能直接沿用** |
| **未映射路段（充电桩）是混合口径** | 档 B 开着 ⑪，而矩阵没有充电桩项 ⟹ 那些腿用几何口径补（**声明式**，段数与分钟数随行带出）。论文里必须如实写，不得说成"运输全部来自矩阵" |
| **`des.py` 体积** | 已 1100+ 行（超本仓 800 行指引）。本批不拆（拆分会与本批的接口改动打架），留给独立的重构批 |
| **`HF2021_LAHC` 的数字仍未核** | P4-A 的开放线索：该列占 `MKT_PUBLISHED` 的三分之一，是第三方转述，进论文前必须核原文 |
| **评估侧同源 guard 仍缺** | `m13_train_a._make_eval_fn` 的 `constraints=None` 未加守卫（P4-A 已登记）——本批未动 |
