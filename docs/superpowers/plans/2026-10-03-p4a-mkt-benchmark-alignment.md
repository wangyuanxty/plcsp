# P4-A 采用 MKT 基准 + 消融链路单种子打通 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把我们的实例口径对齐到文献在用的 **MKT 基准**（Brandimarte MK + 机台间行程时间），使我们的数字能与 HGS / HA-DQN 等已发表方法**写进同一张表**；同时用单种子把 5 组消融的链路打通。

**Architecture:** 纯数据层新增（MKT 布局矩阵落盘 + 适配器），**不改我们的问题定义**——MKT = 原始 MK 的加工数据（一字不改）+ 一张机台间行程时间矩阵 + 一个车辆数设定。仿真栈、编码器、训练器一概不动。

**Tech Stack:** Python 3.12.4（`D:/anaconda/python.exe`）｜CPU-only｜numpy 1.26.4｜pytest 7.4.4｜Windows 11

**Spec:** `docs/superpowers/specs/2026-10-02-plcsp-rebuild-design.md`（§3.2 布局 / §6.1 基线 / §6.2 消融）
**依据记录**：`docs/progress-log.md` **§19.7c**（第 4 项基线筛选结果）与 **§19.7d**（MKT 口径核实结论）

## Global Constraints

- **项目根** = `D:\research\DeepReinforcementLearningScheduling`。**根目录只允许有目录**，不得新增根级文件。
- **Python** 一律用 `D:/anaconda/python.exe`。**CPU-only**。
- **单位约定（不得改动）**：仿真时间 = 分钟，布局坐标 = 米，能耗 = kWh。
- **回归门禁**：`plcsp/tests/` 现有 **178 项必须始终全绿**。
- **⚠️ 行尾：保持每个文件**既有**的行尾，不得转换。** 本仓 HEAD 是 **18 CRLF / 24 LF 的混合**。**改既有文件一律用 Edit 工具**；新建 `.py` 跟随同目录多数派（`plcsp/env/` 下的新建文件看 `des.py` = CRLF 还是多数派，以多数为准并在报告里写明你的判断）。
- **禁止生成临时产物**（`*.log`/`*.jsonl`/`figs/`）到包目录或根目录。
- **提交信息格式**：`<type>: <description>`，**不添加任何署名/生成标识**。
- **⚠️ 术语纪律（用户明令）**：**不自造术语**。描述本计划做的事用平实说法——「采用 MKT 基准」「与已发表数字并列」「复现保真度」，**不要用"对拍"这类我先前随手用的词**。引用文献给**完整题名 + 期刊 + 卷期页 + 年**，不得只写代号。

## Review Focus

以下六类是任务的测试覆盖不到、但最可能出错的：

1. **车辆数不一致导致数字不可比**。HF2021 用 **2 台**、HGS 与 HA-DQN 用 **v = m**。同一实例、不同车数下 Cmax 差异巨大（HF2021 的 LAHC 在 MKT01 上 187，HGS 是 153）。期望行为：车辆数**只能是显式参数**，产出的表**必须带"车辆数"列**，且不得把两档混进同一行。
2. **LU 行列的 off-by-one**。HGS 的读法是 `trans_time[1:, 1:]`（第 0 行/列是装卸站）。若忘记丢、或丢错方向，距离整体偏移且**不报错**。期望行为：有测试直接钉住"丢 LU 后矩阵维度 = 机台数 × 机台数"与"对角为 0"。
3. **MKT 的数字与原始 MK 的数字混进同一张表**。MK01 的 BKS 是 **40**，MKT01 的已发表数字是 **97–187**。两者**不同口径**。期望行为：任何产表都**标注用的是哪一套**；有断言挡住"同一张表里两套实例混用"。
4. **"行程时间随机 2–10" 与实发矩阵矛盾**（§19.7d 存疑点①）。网站文字与 HGS 正文都写 2–10，但实发矩阵值域到 17。期望行为：**实测矩阵值域并如实记录**，不得照抄"2–10"。
5. **许可**。`github.com/msh0576/FJSPT-Scheduler` **无 LICENSE**；作者站是一手数据源。期望行为：**只取数据文件、不复制该仓代码**；数据来源与许可在数据目录留 README。
6. **机器数与布局文件不匹配**。MK01 是 6 机、MK07 是 5 机、MK10 是 15 机。期望行为：适配器按实例**实际**机台数选布局文件，选不到就**显式报错**，不静默回退。

---

### Task 1: MKT 布局矩阵落盘

**Files:**
- Create: `plcsp/data/mkt/layouts/*.txt`（约 12 个文件）
- Create: `plcsp/data/mkt/README.md`（来源 / 许可 / 存疑点）
- Test: `plcsp/tests/test_mkt_data.py`

**Interfaces:**
- Consumes: 无
- Produces: `plcsp/data/mkt/layouts/{m}_machine_layout.txt`（`m ∈ {4,5,6,8,10,11,12,13,15,16,17,18}`）——纯文本，首行起为 `(m+1) × (m+1)` 行程时间矩阵，**第 0 行/列是装卸站（LU）**；模块常量 `MKT_LAYOUT_DIR: Path`

> ⚠️ **先测再钉**：§19.7d 报的"6 机值域到 17、15 机到 15"是**侦查 agent 的一面之词**，本任务须**自己量一遍**再写进断言。若实测与它不符，**以你实测为准**并在报告里指出差异。

- [ ] **Step 1: 取得数据文件**

数据来源（**只取数据文件，不要复制任何代码**）：

- 一手：作者站 `https://fastmanufacturingproject.wordpress.com/2019/04/11/fjspt-instances/`（页面附 `4to18machines_layouts-1.pdf`）
- 备选：`https://github.com/msh0576/FJSPT-Scheduler` 的 `BenchmarkDataset/dataset/{m}_machine_layout.txt`（**该仓无 LICENSE，只作数据来源，不要引入其代码**）

把 12 个布局矩阵放到 `plcsp/data/mkt/layouts/`。**若只能拿到 PDF**，自己转成纯文本（每行空格分隔的数字，勿加表头）。

- [ ] **Step 2: 写失败测试**

创建 `plcsp/tests/test_mkt_data.py`：

```python
"""MKT 布局数据的测试（P4-A Task 1）。

数据来源与口径见 `plcsp/data/mkt/README.md`；存疑点见 `progress-log §19.7d`。
"""
from __future__ import annotations

import numpy as np
import pytest

from plcsp.env.mkt import MKT_LAYOUT_DIR, load_mkt_layout

NEEDED = (4, 5, 6, 8, 10, 11, 12, 13, 15, 16, 17, 18)


@pytest.mark.unit
def test_all_layout_files_present():
    """12 个机台数的布局矩阵必须齐——少了任何一个，对应规模的实例都跑不了。"""
    missing = [m for m in NEEDED if not (MKT_LAYOUT_DIR / f"{m}_machine_layout.txt").exists()]
    assert not missing, f"缺布局文件：{missing}（目录 {MKT_LAYOUT_DIR}）"


@pytest.mark.unit
@pytest.mark.parametrize("m", NEEDED)
def test_layout_shape_is_m_plus_1_square(m: int):
    """原始矩阵是 (m+1)×(m+1)：**第 0 行/列是装卸站（LU）**（§19.7d）。"""
    raw = load_mkt_layout(m, drop_lu=False)
    assert raw.shape == (m + 1, m + 1), f"{m} 机布局维度应为 {m+1}，实得 {raw.shape}"


@pytest.mark.unit
@pytest.mark.parametrize("m", [6, 10, 15])
def test_drop_lu_gives_m_by_m_with_zero_diagonal(m: int):
    """⚠️ Review Focus #2：丢 LU 后必须是 m×m 且**对角为 0**（同机台间无行程）。"""
    t = load_mkt_layout(m, drop_lu=True)
    assert t.shape == (m, m)
    assert np.allclose(np.diag(t), 0.0), f"{m} 机布局丢 LU 后对角非 0：{np.diag(t)}"


@pytest.mark.unit
def test_matrix_is_asymmetric():
    """⚠️ 行程时间矩阵**非对称**（§19.7d 实测：10 机布局 M[1][2]=3 而 M[2][1]=12）。

    若实现里误做了对称化，这个测试必须红。
    """
    t = load_mkt_layout(10, drop_lu=True)
    assert not np.allclose(t, t.T), "行程时间矩阵应非对称——疑似被对称化了"


@pytest.mark.unit
def test_values_are_measured_not_assumed():
    """⚠️ Review Focus #4：值域**实测**后钉住，不得照抄文献的「随机 2–10」。

    §19.7d 的侦查报「6 机到 17、15 机到 15」，但那是二手读数。此处按**本机实测**钉住，
    并在断言消息里打印实测区间，便于日后核对。
    """
    for m, expected_max in ((6, 17), (15, 15)):
        t = load_mkt_layout(m, drop_lu=True)
        got = float(t.max())
        assert got == pytest.approx(expected_max, abs=0.0), (
            f"{m} 机布局实测最大值 {got}，与 §19.7d 记录的 {expected_max} 不符——"
            "若数据源换了版本，请更新 progress-log 而不是改这个断言")


@pytest.mark.unit
def test_missing_machine_count_raises():
    """⚠️ Review Focus #6：取不到的机台数必须**显式报错**，不得静默回退。"""
    with pytest.raises((FileNotFoundError, ValueError)):
        load_mkt_layout(7)          # 7 不在 4..18 的已发布集合里
```

- [ ] **Step 3: 跑测试确认失败**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_mkt_data.py -q --no-header`
Expected: FAIL — `ModuleNotFoundError: No module named 'plcsp.env.mkt'`

- [ ] **Step 4: 写 `plcsp/data/mkt/README.md`**

内容必须含：
1. **数据来源**（作者站 URL；若从 HGS 仓取，注明并写明该仓**无 LICENSE、只作数据来源**）
2. **口径**：`(m+1)×(m+1)`，第 0 行/列为装卸站；**实测值域**（你量的，不是抄的）
3. **三条存疑点**（逐字引自 `progress-log §19.7d`）：① "2–10" 与实发矩阵矛盾；② 车辆数各家用得不同（HF2021=2，HGS/HA-DQN=v=m）；③ HGS 仓日志 ≠ 论文 Table IV
4. **引用**：Homayouni, S. M., Fontes, D. B. M. M. *Production and transport scheduling in flexible job shop manufacturing systems.* **Journal of Global Optimization 79(2):463–502, 2021.**

- [ ] **Step 5: 实现 `plcsp/env/mkt.py`（本任务只需 `load_mkt_layout` 与 `MKT_LAYOUT_DIR`）**

```python
# -*- coding: utf-8 -*-
"""MKT 基准的数据层（Brandimarte MK + 机台间行程时间）。

**MKT = 原始 MK 的加工数据（一字不改）+ 一张机台间行程时间矩阵 + 一个车辆数设定**
（核实于 progress-log §19.7d）。故本模块**只补运输**，不碰加工数据。

口径（与 HGS 的读取方式一致）：
布局文件是 `(m+1)×(m+1)`，**第 0 行/列是装卸站（Load/Unload）**；
机台间行程时间 = 丢 LU 后的 `m×m` 子矩阵（`raw[1:, 1:]`）。
⚠️ 该矩阵**非对称**；⚠️ 对角线为 0。

数据来源与许可见 `plcsp/data/mkt/README.md`。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MKT_LAYOUT_DIR = Path(__file__).resolve().parent.parent / "data" / "mkt" / "layouts"


def load_mkt_layout(n_machines: int, *, drop_lu: bool = True) -> np.ndarray:
    """读 `{n}_machine_layout.txt`。

    `drop_lu=True`（默认）返回 `m×m` 的机台间矩阵；`False` 返回含装卸站的 `(m+1)×(m+1)` 原矩阵。
    **取不到该机台数就显式报错**——不得回退到邻近规模（Review Focus #6）。
    """
    p = MKT_LAYOUT_DIR / f"{n_machines}_machine_layout.txt"
    if not p.exists():
        raise FileNotFoundError(
            f"MKT 布局文件不存在：{p}（已发布集合为 4–18 中的 12 个机台数，见 "
            f"{MKT_LAYOUT_DIR}）——不得静默回退到其它规模")
    m = np.loadtxt(p, dtype=float)
    return m[1:, 1:] if drop_lu else m
```

- [ ] **Step 6: 跑测试确认通过 + 全套回归**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header`
Expected: 全绿（178 + 本任务新增）

- [ ] **Step 7: 提交**

```bash
git add plcsp/data/mkt/ plcsp/env/mkt.py plcsp/tests/test_mkt_data.py
git commit -m "feat: MKT 基准布局数据落盘 + 读取（P4-A Task 1）"
```

---

### Task 2: MKT 实例适配器

**Files:**
- Modify: `plcsp/env/mkt.py`（加 `MktInstance` 与 `load_mkt`）
- Test: `plcsp/tests/test_mkt_instance.py`

**Interfaces:**
- Consumes: Task 1 的 `load_mkt_layout`
- Produces:
  - `MktInstance`（frozen dataclass）：`base: Instance`（原始 MK）/ `trans_time: np.ndarray (m,m)` / `n_agv: int` / `layout_m: int`
  - `load_mkt(name: str, n_agv: int | None = None) -> MktInstance` —— `n_agv=None` 时取 **v = m**（HGS/HA-DQN 的设定）
  - `MKT_PUBLISHED: dict[str, dict[str, float]]` —— 已发表的逐实例数字（出处见 Review Focus #3）

- [ ] **Step 1: 写失败测试**

创建 `plcsp/tests/test_mkt_instance.py`：

```python
"""MKT 实例适配器的测试（P4-A Task 2）。"""
from __future__ import annotations

import numpy as np
import pytest

from plcsp.env.instances import load_mk
from plcsp.env.mkt import MKT_PUBLISHED, load_mkt


@pytest.mark.unit
@pytest.mark.parametrize("name,m", [("mk01", 6), ("mk07", 5), ("mk10", 15)])
def test_processing_data_is_identical_to_raw_mk(name: str, m: int):
    """⚠️ 核心断言：**MKT 的加工数据与原始 MK 逐位相同**（§19.7d 已核实的事实）。

    MKT 相对 MK 的增量**只有**行程时间矩阵与车辆数——若这里不等，说明适配器动了加工数据。
    """
    inst = load_mkt(name)
    base = load_mk(name)
    assert inst.base.n_jobs == base.n_jobs
    assert inst.base.n_machines == base.n_machines
    for j, (a, b) in enumerate(zip(inst.base.jobs, base.jobs)):
        assert a == b, f"{name} 作业 {j} 的工序数据被改动了"


@pytest.mark.unit
@pytest.mark.parametrize("name,m", [("mk01", 6), ("mk07", 5), ("mk10", 15)])
def test_transport_matrix_matches_machine_count(name: str, m: int):
    """行程时间矩阵维度必须等于**该实例实际的机台数**（Review Focus #6）。"""
    inst = load_mkt(name)
    assert inst.trans_time.shape == (m, m)
    assert inst.layout_m == m


@pytest.mark.unit
def test_default_fleet_is_v_equals_m():
    """⚠️ Review Focus #1：默认车辆数 = **m**（HGS 与 HA-DQN 的设定，由 HA-DQN 表 11 的
    J-M-A 列读出：10-6-6 / 20-5-5 / 20-15-15）。"""
    assert load_mkt("mk01").n_agv == 6
    assert load_mkt("mk07").n_agv == 5
    assert load_mkt("mk10").n_agv == 15


@pytest.mark.unit
def test_fleet_size_is_explicit_in_the_result():
    """⚠️ Review Focus #1：车辆数**只能显式传**，且结果里带着它——不同车数下 Cmax 不可比。"""
    a = load_mkt("mk01", n_agv=2)
    assert a.n_agv == 2
    assert a.n_agv != load_mkt("mk01").n_agv


@pytest.mark.unit
def test_published_numbers_are_labelled_by_benchmark():
    """⚠️ Review Focus #3：已发表数字必须**标注基准**，防与原始 MK 混表。

    原始 MK01 的 BKS 是 40，而 MKT01 的已发表数字是 97–187——两套口径。
    """
    ref = MKT_PUBLISHED["mk01"]
    assert set(ref) >= {"HGS_JMS2024", "HA_DQN_CIS2025", "HF2021_LAHC"}
    assert 90.0 < ref["HGS_JMS2024"] < 200.0, "HGS 在 MKT01 上报 153，不该落在原始 MK 的量级"
    assert ref["HGS_JMS2024"] > 40.0, "MKT01 的数字不得与原始 MK01 的 BKS(40) 混淆"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_mkt_instance.py -q --no-header`
Expected: FAIL — `ImportError: cannot import name 'load_mkt'`

- [ ] **Step 3: 实现（追加到 `plcsp/env/mkt.py`）**

```python
# ── 已发表的逐实例数字（⚠️ 均为 **MKT** 口径，不是原始 MK）──
# HGS：Moon, Lee, Park. *Learning-enabled Flexible Job-shop Scheduling for Scalable Smart
#      Manufacturing.* Journal of Manufacturing Systems 77:356–367, 2024，Table IV 的 HGS 列。
# HA-DQN：Dong, Wan, Zeng. *A heuristic-assisted deep reinforcement learning algorithm for
#      flexible job shop scheduling with transport constraints.* Complex & Intelligent
#      Systems 11:210, 2025，Table 8。
# HF2021 LAHC：Homayouni & Fontes, J. Global Optimization 79(2):463–502, 2021（其车辆数为 2，
#      **与 HGS/HA-DQN 的 v=m 不同口径**——见 progress-log §19.7d 存疑点②）。
MKT_PUBLISHED: dict[str, dict[str, float]] = {
    "mk01": {"HGS_JMS2024": 153.0, "HA_DQN_CIS2025": 97.0, "HF2021_LAHC": 187.0},
    "mk02": {"HGS_JMS2024": 104.0, "HA_DQN_CIS2025": 71.0, "HF2021_LAHC": 148.0},
    "mk03": {"HGS_JMS2024": 267.0, "HA_DQN_CIS2025": 235.0, "HF2021_LAHC": 371.0},
    "mk04": {"HGS_JMS2024": 139.0, "HA_DQN_CIS2025": 129.0, "HF2021_LAHC": 225.0},
    "mk05": {"HGS_JMS2024": 374.0, "HA_DQN_CIS2025": 259.0, "HF2021_LAHC": 312.0},
    "mk06": {"HGS_JMS2024": 217.0, "HA_DQN_CIS2025": 158.0, "HF2021_LAHC": 389.5},
    "mk07": {"HGS_JMS2024": 348.0, "HA_DQN_CIS2025": 213.0, "HF2021_LAHC": 291.0},
    "mk08": {"HGS_JMS2024": 812.0, "HA_DQN_CIS2025": 669.0, "HF2021_LAHC": 846.0},
    "mk09": {"HGS_JMS2024": 529.0, "HA_DQN_CIS2025": 502.0, "HF2021_LAHC": 794.0},
    "mk10": {"HGS_JMS2024": 409.0, "HA_DQN_CIS2025": 357.0, "HF2021_LAHC": 712.5},
}


@dataclass(frozen=True)
class MktInstance:
    """MKT 实例 = 原始 MK + 行程时间矩阵 + 车辆数。

    ⚠️ **车辆数不是数据文件里的字段**（§19.7d）：HF2021 用 2 台、HGS 与 HA-DQN 用 v=m。
    **不同车数下的 Cmax 不可比**，故它只能是显式参数、且必须随结果一起报出去。
    """
    base: Instance
    trans_time: np.ndarray
    n_agv: int
    layout_m: int

    @property
    def name(self) -> str:
        return f"{self.base.source}（MKT, m={self.layout_m}, v={self.n_agv}）"


def load_mkt(name: str, n_agv: int | None = None) -> MktInstance:
    """载入 MKT 实例。`n_agv=None` → **v = m**（HGS/HA-DQN 的设定）。"""
    base = load_mk(name)                      # 加工数据一字不改（有测试钉住）
    m = base.n_machines
    trans = load_mkt_layout(m)
    return MktInstance(base=base, trans_time=trans, n_agv=(m if n_agv is None else int(n_agv)),
                       layout_m=m)
```

（`from dataclasses import dataclass`、`from .instances import Instance, load_mk` 加到文件头。）

- [ ] **Step 4: 跑测试确认通过 + 全套回归 + 提交**

```bash
PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header
git add plcsp/env/mkt.py plcsp/tests/test_mkt_instance.py
git commit -m "feat: MKT 实例适配器（P4-A Task 2）——加工数据不动，只补行程时间与车辆数"
```

> ⚠️ **本任务只产出数据对象，不接进仿真。** 把 `trans_time` 接进 `AgvSim` 是 P4-B 的活
> （要决定"行程时间矩阵"与现有"布局最短路距离"如何共存——**那是一个设计决定，不在本批**）。

---

### Task 3: 在 MKT 上跑我们的参考调度，并与已发表数字并列

**Files:**
- Create: `plcsp/m14_mkt_reference.py`
- Test: `plcsp/tests/test_mkt_reference.py`

**Interfaces:**
- Consumes: Task 2 的 `load_mkt` / `MKT_PUBLISHED`
- Produces: 一张表，列为 `实例 | 车辆数 | 我们（参考调度）| HGS | HA-DQN | HF2021(车辆数不同)`

- [ ] **Step 1: 写失败测试**

创建 `plcsp/tests/test_mkt_reference.py`：

```python
"""MKT 上跑参考调度的测试（P4-A Task 3）。"""
from __future__ import annotations

import pytest

from plcsp.env.mkt import MKT_PUBLISHED, load_mkt


@pytest.mark.unit
def test_reference_table_has_fleet_column():
    """⚠️ Review Focus #1：产出的表**必须带"车辆数"列**——否则不同车数的数字会混在一起比。"""
    from plcsp.m14_mkt_reference import reference_table

    rows = reference_table(("mk01",), n_agv=2)
    assert rows, "表为空"
    assert "n_agv" in rows[0], f"表缺车辆数列：{list(rows[0])}"
    assert rows[0]["n_agv"] == 2


@pytest.mark.unit
def test_reference_table_labels_the_benchmark():
    """⚠️ Review Focus #3：表必须自报用的是 MKT（不是原始 MK），防混表。"""
    from plcsp.m14_mkt_reference import reference_table

    rows = reference_table(("mk01",), n_agv=6)
    assert rows[0]["benchmark"] == "MKT"


@pytest.mark.unit
def test_our_reference_is_in_the_published_ballpark():
    """量级核验：我们的参考调度（最短候选 + 轮询派车）应落在已发表数字的**同一量级**。

    这一步**只判量级**（训练后的 DRL 当然比未训练的参考调度好）：
    若我们跑出 40 上下 ⟹ 行程时间根本没接上；若跑出几千 ⟹ 单位错了。
    """
    from plcsp.m14_mkt_reference import reference_table

    r = reference_table(("mk01",), n_agv=6)[0]
    pub = MKT_PUBLISHED["mk01"]["HGS_JMS2024"]
    assert 0.3 * pub < r["ours"] < 5.0 * pub, (
        f"MK01(MKT) 参考调度 Cmax={r['ours']:.1f}，已发表 HGS={pub}——量级不符")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_mkt_reference.py -q --no-header`
Expected: FAIL — `ModuleNotFoundError: No module named 'plcsp.m14_mkt_reference'`

- [ ] **Step 3: 实现 `plcsp/m14_mkt_reference.py`**

```python
# -*- coding: utf-8 -*-
"""MKT 上跑我们的参考调度，与已发表数字并列（P4-A Task 3）。

⚠️ **本脚本不把 trans_time 接进仿真**（那是 P4-B 的设计决定）——故 `ours` 是
**原始 MK 几何**下跑出来的，只在**量级**上与已发表数字可比。见计划「已知边界」。
输出 stdout 表，**不写文件**。
"""
from __future__ import annotations

import argparse
import statistics as st

from .algo.setup import build_setup
from .env.des import SimConfig, rollout
from .env.instances import load_mk
from .env.mkt import MKT_PUBLISHED


def reference_table(names: tuple[str, ...], n_agv: int | None = None,
                    seeds: int = 3) -> list[dict]:
    """每实例一行：我们（参考调度）+ 已发表数字。

    `n_agv=None` → 该实例的 **v = m**（HGS/HA-DQN 口径）。表里**必带车辆数列**，
    且 `benchmark` 字段自报是 MKT——防与原始 MK 的数字混表。
    """
    rows: list[dict] = []
    for name in names:
        base = load_mk(name)
        v = base.n_machines if n_agv is None else int(n_agv)
        cfg = SimConfig(n_agv=v)
        got = [rollout(base, seed_chain=s, cfg=cfg)["makespan"] for s in range(seeds)]
        rows.append({"inst": name, "benchmark": "MKT", "n_agv": v,
                     "ours": float(st.mean(got)), **MKT_PUBLISHED[name]})
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
    print(f"{'实例':<7}{'基准':<6}{'车辆数':>7}{'我们(参考)':>12}{'HGS':>9}{'HA-DQN':>9}{'HF2021*':>10}")
    print("  * HF2021 的 LAHC 用 2 台车，与其余列**不同口径**，不可直接比（§19.7d 存疑点②）")
    for r in rows:
        print(f"{r['inst']:<7}{r['benchmark']:<6}{r['n_agv']:>7}{r['ours']:>12.1f}"
              f"{r['HGS_JMS2024']:>9.1f}{r['HA_DQN_CIS2025']:>9.1f}{r['HF2021_LAHC']:>10.1f}")


if __name__ == "__main__":
    main()
```

**"ours" 怎么算**：`plcsp/algo/setup.build_setup` + `rollout` 的参考口径（每工序取最短候选 + AGV 轮询）。
⚠️ **本任务不把 `trans_time` 接进仿真**（那是 P4-B）——故 `ours` 是**原始 MK 几何**下跑出来的。
→ 因此 `ours` 只在**量级**上可比，测试断言的正是量级（`0.3×` 到 `5×`）。
**报告里必须写清这一点**：本表证明的是"链路通、量级合理"，**不是"我们已经能对上 MKT"**。

- [ ] **Step 4: 跑脚本，把表写进报告**

Run:
```bash
PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m14_mkt_reference --instances mk01,mk07,mk10 --n-agv default
```

- [ ] **Step 5: 跑全套回归 + 提交**

```bash
PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header
git add plcsp/m14_mkt_reference.py plcsp/tests/test_mkt_reference.py
git commit -m "feat: MKT 参考调度表（P4-A Task 3）——与已发表数字并列，带车辆数列"
```

---

### Task 4: 消融链路单种子打通

**Files:**
- Create: `plcsp/m15_ablation_smoke.py`
- Test: `plcsp/tests/test_ablation_smoke.py`

**Interfaces:**
- Consumes: `plcsp/env/constraints.ABLATION_GROUPS`、`plcsp/algo/setup.build_setup`、`plcsp/algo/group_rel.joint_chain_step`
- Produces: 5 组 × 3 实例 × **1 种子** = **15 次训练**的表

> ⚠️ **本任务用原始 MK，不用 MKT**（用户裁定：原始 MK 保留跑我们自己的指标/消融；
> MKT 用于与已发表数字比较）。表里**必须标注**用的是哪一套。

- [ ] **Step 1: 写失败测试**

创建 `plcsp/tests/test_ablation_smoke.py`：

```python
"""消融链路单种子打通的测试（P4-A Task 4）。

⚠️ 本测试**只跑 2 组**（不是 5 组）以保住分钟级墙钟；完整的 5 组 × 3 实例由脚本承担。
"""
from __future__ import annotations

import pytest

from plcsp.env.constraints import ABLATION_GROUPS


@pytest.mark.unit
def test_ablation_groups_are_five():
    """spec §6.2 定的是 5 组（Full / −物流 / −生产 / −信息 / None）。"""
    assert set(ABLATION_GROUPS) == {"Full", "-物流", "-生产", "-信息", "None"}


@pytest.mark.unit
def test_two_groups_run_and_differ():
    """⚠️ Review Focus：两组必须**跑得出且不同**——若相同，说明 constraints 没接进训练。

    （`§19.7` 的 F1 已把 constraints 接进 `joint_chain_step`；本测试是端到端的再确认。）
    """
    from plcsp.m15_ablation_smoke import run_group
    from plcsp.env.instances import load_mk

    inst = load_mk("mk01")
    full = run_group(inst, ABLATION_GROUPS["Full"], seed=0, steps=3, G=2)
    none = run_group(inst, ABLATION_GROUPS["None"], seed=0, steps=3, G=2)
    assert full["jobs_done"] == inst.n_jobs and none["jobs_done"] == inst.n_jobs
    assert full["makespan"] != none["makespan"], "两组跑出同一 makespan——constraints 没生效"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_ablation_smoke.py -q --no-header`
Expected: FAIL — `ModuleNotFoundError: No module named 'plcsp.m15_ablation_smoke'`

- [ ] **Step 3: 实现 `plcsp/m15_ablation_smoke.py`**

```python
# -*- coding: utf-8 -*-
"""消融链路单种子打通（P4-A Task 4）。

⚠️ 用**原始 MK**（非 MKT）——用户裁定：原始 MK 跑我们自己的指标与消融，MKT 用于与已发表数字比较。
⚠️ 墙钟：`joint_chain_step(G=8)` 实测 MK01 ≈ 18.2 s/步、MK10 ≈ 86–94 s/步（P2）。
   **5 组 × 3 实例 × 100 步不可能分钟级跑完**——故测试只跑 2 组 × 3 步，本脚本是长跑。
"""
from __future__ import annotations

import argparse
import time

from .algo.group_rel import joint_chain_step, roll_chain
from .algo.setup import build_setup
from .env.constraints import ABLATION_GROUPS, ConstraintConfig
from .env.des import SimConfig
from .env.instances import Instance, load_mk
from .env.reward import ReferenceObjectives
from .nn.state_emb import norm_context


def run_group(inst: Instance, constraints: ConstraintConfig, *, seed: int,
              steps: int, G: int, cfg: SimConfig | None = None) -> dict:
    """跑一组消融 `steps` 步，返回**末步**的指标（由 `roll_chain` 的 metrics 带出）。"""
    cfg = cfg or SimConfig()
    lay, dm, ctx = build_setup(inst, cfg, constraints=constraints)
    ref = ReferenceObjectives.of(inst, cfg)
    pol = _policy_for(ctx)
    for s in range(steps):
        joint_chain_step(pol, inst, lay, dm, cfg, ctx, ref,
                         seed=seed * 1000 + s, G=G, constraints=constraints)
    _, met = roll_chain(inst, lay, dm, cfg, pol, seed=seed * 1000 + steps, ctx=ctx,
                        constraints=constraints, sample=False)
    return {"makespan": met["makespan"], "energy": met["energy"],
            "tardy_twt": met["tardy_twt"], "jobs_done": met["jobs_done"]}


def _policy_for(ctx) -> "object":
    """建一个带编码器的 `PolicyNet`（两个打分头只在 `enc is not None` 时存在）。"""
    from .algo.policy import PolicyNet
    from .nn.encoder import LayoutEncoder
    return PolicyNet(enc=LayoutEncoder())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="mk01,mk07,mk10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30,
                    help="默认 30：P2 实测 MK01 上 ~25 步即饱和（progress-log §19.2）")
    ap.add_argument("--G", type=int, default=8)
    args = ap.parse_args()
    print(f"基准 = MK（原始，非 MKT）｜种子 {args.seed}｜{args.steps} 步｜G={args.G}")
    print(f"{'组':<8}{'实例':<7}{'Cmax':>10}{'energy':>10}{'TWT':>10}{'秒':>8}")
    for name in args.instances.split(","):
        inst = load_mk(name)
        for gname, cons in ABLATION_GROUPS.items():
            t0 = time.time()
            r = run_group(inst, cons, seed=args.seed, steps=args.steps, G=args.G)
            print(f"{gname:<8}{name:<7}{r['makespan']:>10.1f}{r['energy']:>10.2f}"
                  f"{r['tardy_twt']:>10.1f}{time.time() - t0:>8.1f}", flush=True)


if __name__ == "__main__":
    main()
```

CLI：`--instances mk01,mk07,mk10 --seed 0 --steps 100 --G 8`，
输出表：`组 | 实例 | Cmax | energy | TWT | r_mean`，**表头标注 benchmark=MK（原始）**。

⚠️ **墙钟**：`joint_chain_step(G=8)` MK01 ≈ 18.2 s/步、MK10 ≈ 86–94 s/步（P2 实测）。
故 100 步 × 3 实例 × 5 组 **不可能在分钟级跑完**——**脚本是长跑，测试只跑 2 组 × 3 步**。
脚本默认 `--steps` 设小些（如 30，即"~25 步饱和"的读数附近，见 `§19.2`），并在报告里报**实测墙钟**。

- [ ] **Step 4: 跑通并把表写进报告**

```bash
PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m15_ablation_smoke --instances mk01 --steps 30 --G 8
```
Expected: 5 组 × 1 实例跑通，表落 stdout

- [ ] **Step 5: 跑全套回归 + 提交 + 更新文档**

```bash
PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header
git add -A && git commit -m "feat: 消融链路单种子打通（P4-A Task 4）"
```
并在 `docs/progress-log.md` 新增一节记：5 组的实测读数、墙钟、以及 `--steps` 的选取依据。

---

## 完成后的状态

- **MKT 数据与适配器就位**——我们能在文献承认的同一基准上产出数字
- **与已发表数字并列的表**（HGS / HA-DQN / HF2021），**带车辆数列**、**标注基准**
- **消融链路单种子打通**，5 组能跑出不同结果

## 已知边界（**必须带进 P4-B**）

| 项 | 说明 |
|---|---|
| **本批不把 `trans_time` 接进仿真** | `m14` 的 `ours` 是**原始 MK 几何**下的数——只证明链路通、量级合理，**不是"已能对上 MKT"**。接进 `AgvSim` 是 P4-B 的设计决定（现有"布局最短路距离"与"行程时间矩阵"如何共存）|
| **车辆数是设定、不是数据** | 我们默认 v=m（HGS/HA-DQN 口径），并报一档 v=2（HF2021 口径）作敏感性。**两档不得混进同一行。** |
| **HGS 未实现** | 第 4 项基线（HGS 从零实现，JMS 77:356–367）**不在本批**，是独立的下一批 |
| **消融只有 1 个种子** | 用户裁定"先单 seed 跑通"，全矩阵（75 run）待链路确认后再谈 |
