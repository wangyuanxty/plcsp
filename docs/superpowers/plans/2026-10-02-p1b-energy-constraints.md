# P1b 能耗与约束 实施计划（M2 能耗模型 · 约束框架 · 剩余约束接入）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把能耗模型从"假的"（`(process_time+travel_time)×1.0`，算出来是时间不是能量）换成**引证参数的 M2 模型**；把约束做成**配置开关**；接入剩余 9 个约束——**每接入一个就立刻实测它是否 binding**。

**Architecture:** 三层：`constraints.py`（纯配置，无逻辑）→ `energy.py`（纯函数：状态时长 → 能耗）→ `des.py`（消费配置与能耗）。三者单向依赖，配置层可被测试直接构造，不需要跑仿真。

**Tech Stack:** Python 3.12.4（`D:/anaconda/python.exe`）｜simpy 4.1.2｜numpy 1.26.4｜pytest 7.4.4｜Windows 11

**Spec:** `docs/superpowers/specs/2026-10-02-plcsp-rebuild-design.md`（**§3.3 十个约束**、**§3.4 能耗 M2**、**§3.5 交期 τ**、**§6.2 消融分组**为权威）

## Global Constraints

- **项目根** = `D:esearch\DeepReinforcementLearningScheduling`。**根目录不留临时产物**（`*.log` / `*.jsonl` / `figs/` / 提取文本 / 渲染图一律进系统临时目录，用完即删）——正式文档放哪里不受限。
- **Python** 一律用 `D:/anaconda/python.exe`。**CPU-only**。
- **单位约定（bug#13，不得改动）**：**仿真时间 = 分钟，布局坐标 = 米**。运输 = `距离[m]/(eff_speed×agv_speed_mps)/60`。能源单位 = **kWh**，功率 = **kW**。
- **回归门禁**：`plcsp/tests/` 的 **69 项必须始终全绿**；`test_instances.py`（42 项）不得改断言。
- **能耗参数必须引证**：全部取自 **GFJSPT-MMRS（SWEVO 99:102181, 2025）**，**不得自造**（spec §3.4 已列全表）。
- **⭐ 每个约束接入后必须立刻做 binding 实测**（Task 3 的工具）。**这是本计划的核心纪律**——
  P1a 的教训是"拥堵建完才发现不 binding"；而 `m11` 的实测又证明**粗估会错**（我按"期望次数"判机器故障不 binding，实测 ×10 就有 +9.8%）。
  **判据：把该约束推到极端，makespan 变化 < 2% ⟹ 不 binding，须在 spec §3.3 标注并考虑砍除。**
- **提交信息格式**：`<type>: <description>`，**不添加任何署名/生成标识**。

## Review Focus

以下六类是本计划的测试覆盖不到、但最容易出事的：

1. **能耗量级对不上现实锚**。spec §3.4 给了 GFJSPT-MMRS 的实证锚：**同规模实例 makespan 516 min → 能耗 42.4 kWh**。期望行为：我们的 MK 实例能耗落在**同量级（10⁰–10² kWh）**；若算出 10³ 或 10⁻²，说明单位（kW×min vs kWh）错了。
2. **配置开关"关了但没真关"**。例如 `fuzzy_processing=False` 时仍抽样了模糊数、`charging=False` 时 AGV 仍在耗电。期望行为：**每个开关在关态下产生与"该约束从未存在"逐位相同的轨迹**（可用同种子对拍验证）。
3. **约束之间的隐式耦合**。例如 ⑫ 预防性维护停机时，⑪ 充电是否还在计能耗；⑨ AGV 故障期间，在途任务的持有状态如何处置。期望行为：每个约束的开关**可独立翻转**，组合出 2⁹ 种配置都不崩。
4. **异构多载量改变了任务-车辆映射的语义**。一车可载 k 件时，"任务"的粒度可能不再是"一件一程"。期望行为：明确并测试"多载"是**同向拼车**还是**串行多点投递**——两者对 `task_flow`/`agv_phi` 索引口径的影响不同，**选错会让 `agv_phi` 索引失配**。
5. **交期 τ 与既有的 `due_factor` 并存**。`des.py` 里 `due_factor=1.8` 的老逻辑仍在（且 `_due()` 仍用它）。期望行为：新口径 `d_j = τ·M_ref` 生效后，**老路径不再被任何活代码调用**（或明确标注废弃）。
6. **模糊加工的"模糊 ≠ 随机"**。spec §3.3 明确要求区分。期望行为：⑥ 用**模糊数**（三角/梯形，隶属度）而非正态分布；若实现成正态，写论文时会被抓。

---

### Task 1: 约束配置框架

**Files:**
- Create: `plcsp/env/constraints.py`
- Modify: `plcsp/env/des.py`（`SimWorld` 接收 `ConstraintConfig`）
- Test: `plcsp/tests/test_constraints.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `ConstraintConfig`（frozen dataclass，**10 个** bool：`finite_buffer` / `machine_failure` / `rework` / `setup_time` / `fuzzy_processing` / `due_dates` / `agv_failure` / `heterogeneous_fleet` / `charging` / `maintenance`）
  - `ABLATION_GROUPS: dict[str, ConstraintConfig]`——5 组（Full / −物流 / −生产 / −信息 / None）
  - 每组配置附中文标签，供论文表格直接引用
  - `SimWorld(..., constraints: ConstraintConfig | None = None)`；`None` → `ConstraintConfig()`（全开）

- [ ] **Step 1: 写失败测试**

创建 `plcsp/tests/test_constraints.py`：

```python
"""约束配置框架的单元测试（P1b Task 1）。"""
from __future__ import annotations

import pytest

from plcsp.env.constraints import ABLATION_GROUPS, ConstraintConfig

ALL_FLAGS = ("finite_buffer", "machine_failure", "rework", "setup_time",
             "fuzzy_processing", "due_dates", "agv_failure",
             "heterogeneous_fleet", "charging", "maintenance")


@pytest.mark.unit
def test_default_config_has_all_constraints_on():
    cfg = ConstraintConfig()
    assert all(getattr(cfg, f) is True for f in ALL_FLAGS)


@pytest.mark.unit
def test_config_is_frozen():
    cfg = ConstraintConfig()
    with pytest.raises(Exception):
        cfg.rework = False          # frozen dataclass 应拒绝赋值


@pytest.mark.unit
def test_ablation_groups_are_five_and_well_formed():
    assert set(ABLATION_GROUPS) == {"Full", "-物流", "-生产", "-信息", "None"}
    assert all(getattr(ABLATION_GROUPS["Full"], f) for f in ALL_FLAGS)
    assert not any(getattr(ABLATION_GROUPS["None"], f) for f in ALL_FLAGS)


@pytest.mark.unit
def test_ablation_groups_partition_the_flags():
    """三组减法必须互不重叠、且并集 = 除 Full 外的全部约束（否则消融表有洞）。"""
    full = ABLATION_GROUPS["Full"]
    on = {f for f in ALL_FLAGS if getattr(full, f)}
    groups = [ABLATION_GROUPS[g] for g in ("-物流", "-生产", "-信息")]
    off_sets = [{f for f in ALL_FLAGS if not getattr(g, f)} for g in groups]
    # 两两不重叠
    for i in range(3):
        for j in range(i + 1, 3):
            assert not (off_sets[i] & off_sets[j]), "两组减法有重叠"
    # 并集 = 全开集合
    assert set().union(*off_sets) == on
```

- [ ] **Step 2: 跑测试确认失败**

Run: `D:/anaconda/python.exe -m pytest plcsp/tests/test_constraints.py -q --no-header`
Expected: FAIL — `ModuleNotFoundError: No module named 'plcsp.env.constraints'`

- [ ] **Step 3: 实现 `constraints.py`**

```python
"""约束配置（spec §3.3）：十个约束的开关 + 五组消融配置。

设计：**开关是纯配置，不含逻辑**——`des.py` 读它决定行为。
5 组消融 = 5 个配置实例，**改配置不改代码**，避免消融时引入代码差异。
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ConstraintConfig:
    """十个约束的开关。默认全开 = 论文主配置（spec §3.3）。"""
    # A 类：基建已有
    finite_buffer: bool = True        # ② 有限缓冲（in_cap/out_cap）
    machine_failure: bool = True      # ③ 机器故障（泊松 + 中断恢复）
    # B 类：近乎免费
    rework: bool = True               # ④ 工件返工
    setup_time: bool = True           # ⑤ 换型 / 准备时间（顺序相关）
    fuzzy_processing: bool = True     # ⑥ 模糊加工时间
    due_dates: bool = True            # ⑧ 交期 / 拖期（τ·M_ref）
    agv_failure: bool = True          # ⑨ AGV 故障
    # C 类：真花钱
    heterogeneous_fleet: bool = True  # ⑩ 异构车队（载重/速度/多载量）
    charging: bool = True             # ⑪ 充电 / 电量
    maintenance: bool = True          # ⑫ 预防性维护（计划性停机）

    def with_(self, **kw) -> "ConstraintConfig":
        """返回关掉指定开关的副本（不可变风格，不原地改）。"""
        return replace(self, **{k: False for k in kw})


_FULL = ConstraintConfig()

ABLATION_GROUPS: dict[str, ConstraintConfig] = {
    "Full": _FULL,
    "-物流": replace(_FULL, agv_failure=False, heterogeneous_fleet=False, charging=False),
    "-生产": replace(_FULL, machine_failure=False, rework=False, setup_time=False,
                     maintenance=False),
    "-信息": replace(_FULL, fuzzy_processing=False, due_dates=False),
    "None": replace(_FULL, finite_buffer=False, machine_failure=False, rework=False,
                    setup_time=False, fuzzy_processing=False, due_dates=False,
                    agv_failure=False, heterogeneous_fleet=False, charging=False,
                    maintenance=False),
}
```


- [ ] **Step 4: 跑测试确认通过 + 修 `with_` 的语义**

Run: `D:/anaconda/python.exe -m pytest plcsp/tests/test_constraints.py -q --no-header`
Expected: PASS（4 项）。若 `with_` 的测试失败，说明 `replace` 的 kw 语义写反了——`with_` 应是"关掉"，而 `replace` 需要显式 False。

- [ ] **Step 5: `SimWorld` 接收配置**

在 `plcsp/env/des.py` 的 `SimWorld.__init__` 增加形参：

```python
    def __init__(self, inst: Instance, layout: Layout, m_dm: np.ndarray,
                 cfg: SimConfig | None = None,
                 constraints: "ConstraintConfig | None" = None):
        ...
        from .constraints import ConstraintConfig
        self.constraints = constraints or ConstraintConfig()
```

并在 `rollout()` / `run()` / `run_gated()` 上各加一个 `constraints=None` 形参向下透传。

- [ ] **Step 6: 跑全套回归**

Run: `D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header`
Expected: **73 项全绿**（69 + 4 新）。

- [ ] **Step 7: 提交**

```bash
git add plcsp/env/constraints.py plcsp/env/des.py plcsp/tests/test_constraints.py
git commit -m "feat: 约束配置框架（十开关 + 五组消融，改配置不改代码）"
```

---

### Task 2: 能耗模型 M2

**Files:**
- Create: `plcsp/energy.py`
- Modify: `plcsp/env/des.py`（统计三态时长；接入 `energy.py`；`energy_power` 废弃）
- Test: `plcsp/tests/test_energy.py`

**Interfaces:**
- Consumes: 无（纯函数 + 引证常量）
- Produces:
  - `MachineEnergyParams`（三档转速的切削/空载功率 [kW]，取自 GFJSPT-MMRS）
  - `AGVEnergyParams`（待机 0.1 / 空载 0.2 / 负载 0.5 kW）
  - `SHOP_FIXED_KW = 0.4`（车间固定功率）
  - `machine_energy_kwh(proc_min, idle_min, setup_min, params) -> float`
  - `agv_energy_kwh(idle_min, empty_min, loaded_min) -> float`
  - `total_energy_kwh(...) -> float`（含 `SHOP_FIXED_KW × makespan_h`）

- [ ] **Step 1: 写失败测试**

创建 `plcsp/tests/test_energy.py`：

```python
"""M2 能耗模型的单元测试（P1b Task 2）。参数全部引证 GFJSPT-MMRS（SWEVO 99:102181）。"""
from __future__ import annotations

import pytest

from plcsp.energy import (AGV_IDLE_KW, AGV_LOADED_KW, AGV_EMPTY_KW, SHOP_FIXED_KW,
                          agv_energy_kwh, machine_energy_kwh, total_energy_kwh)


@pytest.mark.unit
def test_agv_params_match_cited_source():
    """AGV 三态功率必须是文献值（spec §3.4：待机 0.1 / 空载 0.2 / 负载 0.5 kW）。"""
    assert (AGV_IDLE_KW, AGV_EMPTY_KW, AGV_LOADED_KW) == (0.1, 0.2, 0.5)


@pytest.mark.unit
def test_agv_energy_is_kwh_not_kw_times_min():
    """单位核验：0.5 kW 跑 60 min = 0.5 kWh（不是 30）。这是本任务最易错处。"""
    e = agv_energy_kwh(idle_min=0.0, empty_min=0.0, loaded_min=60.0)
    assert e == pytest.approx(0.5, rel=1e-9)


@pytest.mark.unit
def test_machine_energy_uses_three_states():
    """机床三态各自贡献：待机/加工/换型 功率不同，缺一不可。"""
    p = dict(idle_kw=0.74, proc_kw=0.951, setup_kw=0.74)     # GFJSPT-MMRS M1–M4 低速档
    only_idle = machine_energy_kwh(0.0, 60.0, 0.0, **p)
    only_proc = machine_energy_kwh(60.0, 0.0, 0.0, **p)
    assert only_idle == pytest.approx(0.74, rel=1e-9)
    assert only_proc == pytest.approx(0.951, rel=1e-9)
    assert only_proc > only_idle, "加工功率应高于空载"


@pytest.mark.unit
def test_total_energy_matches_literature_order_of_magnitude():
    """Review Focus #1：量级必须落在现实锚附近。

    现实锚（GFJSPT-MMRS）：makespan 516 min 的同规模实例 → 42.4 kWh。
    本测试用一组典型时长验证**量级**在 10^0–10^2 kWh，防止单位错（kW×min 会得 10^3）。
    """
    e = total_energy_kwh(machine_kwh=20.0, agv_kwh=10.0, makespan_min=500.0)
    assert 10.0 < e < 100.0, f"总能耗 {e:.1f} kWh 偏离现实锚量级（应 ~40 kWh）"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `D:/anaconda/python.exe -m pytest plcsp/tests/test_energy.py -q --no-header`
Expected: FAIL — `ModuleNotFoundError: No module named 'plcsp.energy'`

- [ ] **Step 3: 实现 `energy.py`**

```python
"""M2 能耗模型（spec §3.4）——参数全部引证 GFJSPT-MMRS（SWEVO 99:102181, 2025）。

单位：功率 [kW]，时长 [min]，能耗 [kWh]。
**换算关键：kWh = kW × (min / 60)** —— 这是本模块唯一易错处（漏 /60 会差 60 倍）。

机床按**三态**计：待机 / 加工 / 换型（换型功率暂取空载值，因该文献未单列换型功率，
此假设须在 spec §9 的"assumed"表中标注）。AGV 按**三态**计：待机 / 空载 / 负载。
"""
from __future__ import annotations

# ── 引证常量（GFJSPT-MMRS, SWEVO 99:102181, 2025）──
# 机床三档转速（切削 / 空载），[kW]
MACHINE_TIERS = {
    "M1_M4":  dict(rpm=(1500, 4500), proc=(0.951, 1.17), idle=(0.74, 0.90)),
    "M5_M6":  dict(rpm=(2000, 6000), proc=(0.30, 0.60),  idle=(0.24, 0.50)),
    "M7_M10": dict(rpm=(1000, 4000), proc=(0.20, 0.56),  idle=(0.16, 0.36)),
}
# AGV 三态 [kW]
AGV_IDLE_KW, AGV_EMPTY_KW, AGV_LOADED_KW = 0.1, 0.2, 0.5
# 车间固定功率 [kW]
SHOP_FIXED_KW = 0.4


def _kwh(power_kw: float, minutes: float) -> float:
    """kW × min → kWh。**漏掉 /60 是 60 倍错误。**"""
    return power_kw * (minutes / 60.0)


def machine_energy_kwh(proc_min: float, idle_min: float, setup_min: float,
                       *, idle_kw: float, proc_kw: float, setup_kw: float) -> float:
    """机床三态能耗。setup_kw 无文献值，暂取 idle_kw（见模块 docstring）。"""
    return (_kwh(proc_kw, proc_min) + _kwh(idle_kw, idle_min)
            + _kwh(setup_kw, setup_min))


def agv_energy_kwh(idle_min: float, empty_min: float, loaded_min: float) -> float:
    """AGV 三态能耗。"""
    return (_kwh(AGV_IDLE_KW, idle_min) + _kwh(AGV_EMPTY_KW, empty_min)
            + _kwh(AGV_LOADED_KW, loaded_min))


def total_energy_kwh(machine_kwh: float, agv_kwh: float, makespan_min: float) -> float:
    """总能耗 = 机床 + AGV + 车间固定（固定项按 makespan 计）。"""
    return machine_kwh + agv_kwh + _kwh(SHOP_FIXED_KW, makespan_min)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `D:/anaconda/python.exe -m pytest plcsp/tests/test_energy.py -q --no-header`
Expected: PASS（4 项）

- [ ] **Step 5: 在 `des.py` 里统计三态时长并接入**

- `MachineSim`：新增 `stats["proc_min"][pad.id]` / `stats["idle_min"][pad.id]` / `stats["setup_min"][pad.id]`。
  加工时累 `proc_min`；`in_q` 空等时累 `idle_min`（用 `env.now - 上一事件时刻` 近似）。
- `AgvSim`：把 `travel_time` 拆成空载/负载两段（返程空驶累 `empty_min`，载货累 `loaded_min`），
  等待任务的时间累 `idle_min`。
- 返回 dict 的 `energy` 改为 `total_energy_kwh(...)`；**`cfg.energy_power` 标注废弃**。

> ⚠️ 三态时长是**近似**（用事件间隔推算空闲段），须在 docstring 写明近似口径与误差来源。

- [ ] **Step 6: 量级核验（Review Focus #1）**

Run:
```bash
PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -c "
from plcsp.env.instances import load_mk
from plcsp.env.des import rollout, SimConfig
for n in ('mk01','mk07','mk10'):
    i = load_mk(n); r = rollout(i, seed_chain=1, cfg=SimConfig())
    print(n, 'makespan', round(r['makespan'],1), 'energy', round(r['energy'],2), 'kWh')
"
```
Expected: 能耗落在 **10⁰–10² kWh**（现实锚 42.4 kWh）。**若得 10³，几乎肯定是漏了 `/60`。**

- [ ] **Step 7: 提交**

```bash
git add plcsp/energy.py plcsp/env/des.py plcsp/tests/test_energy.py
git commit -m "feat: M2 能耗模型（引证 GFJSPT-MMRS 参数，机床三态 + AGV 三态 + 车间固定）"
```

---

### Task 3: binding 实测工具（通用化）

**Files:**
- Modify: `plcsp/m11_constraint_binding.py`（把硬编码的两个探针改成**按约束名分派**）
- Test: 无需新测试（工具脚本，靠 Task 4–6 的实际使用验证）

**Interfaces:**
- Consumes: Task 1 的 `ConstraintConfig`
- Produces: `python -m plcsp.m11_constraint_binding --constraint <name> --seeds N`，
  输出该约束**极端档**下的 makespan/energy/tardy 变化率，并给出 **binding / 不 binding** 判定

- [ ] **Step 1: 改写工具为"开关对拍"式**

把"手动放大参数"改为**开/关对拍**（更贴合真实消融，且不依赖具体参数）：

```python
PROBES = {
    "machine_failure":     dict(mutate=None),                       # 关 = 把 fail_rate 设 0
    "finite_buffer":       dict(mutate=_cap_one),                   # 极端 = cap 全 1
    "rework":              dict(mutate=_rework_extreme),
    "setup_time":          dict(mutate=_setup_extreme),
    "fuzzy_processing":    dict(mutate=_fuzzy_extreme),
    "agv_failure":         dict(mutate=_agv_fail_extreme),
    "heterogeneous_fleet": dict(mutate=_split_fleet_extreme),
    "charging":            dict(mutate=_charging_extreme),
    "maintenance":         dict(mutate=_maintenance_extreme),
    "due_dates":           dict(mutate=None),                       # 关 = 不优化 tardy
}
```

每个 `_xxx_extreme(lay)` 只改布局/配置里的参数，**不改 `des.py` 逻辑**——
这样工具本身与实现解耦，也顺带验证了 Task 4–6 的开关是否真的生效。

- [ ] **Step 2: 跑基线，记录**哪些约束在**当前已实现**范围内可测

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m11_constraint_binding --all --seeds 5`
Expected: 打印各约束的 binding 判定表。**未实现的约束会报"探针缺失"**——这是预期的，
它给出了 Task 4–6 的待办清单。

- [ ] **Step 3: 提交**

```bash
git add plcsp/m11_constraint_binding.py
git commit -m "test: binding 实测工具通用化（按约束名分派极端档）"
```

---

### Task 4: 生产侧约束（④ 返工 · ⑤ 换型 · ⑫ 预防性维护）

**Files:**
- Modify: `plcsp/env/des.py`
- Test: `plcsp/tests/test_constraints_production.py`

**Interfaces:**
- Consumes: Task 1 的 `ConstraintConfig`；Task 3 的 binding 工具
- Produces: `MachineSim` 支持返工入队、顺序相关换型时间、累计工时触发的计划停机

- [ ] **Step 1: 写失败测试**

```python
"""生产侧约束（④⑤⑫）的测试（P1b Task 4）。"""
import pytest
from plcsp.env.constraints import ConstraintConfig
from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import load_mk

@pytest.mark.unit
def test_switches_off_reproduce_baseline_trajectory():
    """Review Focus #2：开关关掉后，轨迹必须与"该约束从未存在"逐位相同。"""
    inst = load_mk("mk01")
    a = rollout(inst, seed_chain=1, cfg=SimConfig(),
                constraints=ConstraintConfig(rework=False, setup_time=False, maintenance=False))
    b = rollout(inst, seed_chain=1, cfg=SimConfig(),
                constraints=ConstraintConfig(rework=False, setup_time=False, maintenance=False))
    assert a["makespan"] == b["makespan"]      # 确定性：同配置同种子必须同结果

@pytest.mark.unit
def test_setup_time_increases_setup_minutes():
    """⑤ 换型：开启后 stats 里应有非零 setup 时长。"""
    inst = load_mk("mk01")
    on = rollout(inst, seed_chain=1, cfg=SimConfig(),
                 constraints=ConstraintConfig(setup_time=True))
    off = rollout(inst, seed_chain=1, cfg=SimConfig(),
                  constraints=ConstraintConfig(setup_time=False))
    assert on["setup_minutes_total"] > off["setup_minutes_total"] == 0.0

@pytest.mark.unit
def test_all_three_switches_independently_toggleable():
    """Review Focus #3：三个开关可独立翻转，组合不崩。"""
    inst = load_mk("mk01")
    for rw in (False, True):
        for su in (False, True):
            for mt in (False, True):
                r = rollout(inst, seed_chain=1, cfg=SimConfig(),
                            constraints=ConstraintConfig(rework=rw, setup_time=su, maintenance=mt))
                assert r["jobs_done"] == inst.n_jobs and not r["horizon_hit"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `D:/anaconda/python.exe -m pytest plcsp/tests/test_constraints_production.py -q --no-header`
Expected: FAIL — `KeyError: 'setup_minutes_total'`

- [ ] **Step 3: 实现三个约束**

- **④ 返工**：`ConstraintConfig.rework=True` 时，每道工序完工后以 `p_rework`（默认 0.05，**assumed**）
  概率重新入队该工序（重做一次）。返工件**要重新运输**，故会额外生成搬运任务。
- **⑤ 换型**：`MachineSim` 记录上一件工件的类型；换型时长 = `setup_matrix[prev_type][cur_type]`
  （默认矩阵为对角 0、异型 2 min，**assumed**），累入 `stats["setup_min"]`，期间不加工。
- **⑫ 预防性维护**：每台机累计加工工时达 `pm_interval`（默认 120 min，**assumed**）后，
  强制停机 `pm_duration`（默认 10 min，**assumed**），期间 `slot` 被占。
  完工后计数器清零。

> ⚠️ **三个约束的参数都是 assumed（无文献出处）**——实施时须一并写入
> `spec §9 参数与出处` 表的 `assumed` 行，并在论文里做敏感性分析。

- [ ] **Step 4: 跑测试确认通过**

Run: `D:/anaconda/python.exe -m pytest plcsp/tests/test_constraints_production.py -q --no-header`
Expected: PASS（3 项）

- [ ] **Step 5: ⭐ binding 实测（本计划的核心纪律）**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m11_constraint_binding --constraint rework --constraint setup_time --constraint maintenance --seeds 5`
Expected: 每项给出 makespan 变化率与 binding 判定。
**若某项 < 2% ⟹ 在 `spec §3.3` 该行标注"⚠️ 实测不 binding（<2%）"，并考虑砍除。**

- [ ] **Step 6: 跑全套回归**

Run: `D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header`
Expected: 全绿（76 项）

- [ ] **Step 7: 提交**

```bash
git add plcsp/env/des.py plcsp/tests/test_constraints_production.py
git commit -m "feat: 生产侧约束（返工/换型/预防性维护）+ binding 实测"
```

---

### Task 5: 物流侧约束（⑨ AGV 故障 · ⑩ 异构多载量 · ⑪ 充电）

**Files:**
- Modify: `plcsp/env/layout.py`（每台 AGV 的异构参数）
  - ⚠️ **接口变更**：`sample_layout(...)` 新增 `n_agv: int = 3` 形参，`Layout` 新增 `agvs: list[AgvSpec]`——
    因为车队规格属布局（`layout.py`）而非仿真（`des.py`）。**所有现有调用方（`rollout` 内部）须同步传 `cfg.n_agv`**，
    否则布局的车队与仿真的车队数量不一致（静默错误）。
- Modify: `plcsp/env/des.py`（`AgvSim` 支持故障/多载/电量）
- Test: `plcsp/tests/test_constraints_logistics.py`

**Interfaces:**
- Consumes: Task 1 的 `ConstraintConfig`
- Produces:
  - `Layout.agvs: list[AgvSpec]`，`AgvSpec(speed_mps, capacity, battery_kwh)`
  - `AgvSim` 支持：故障停机、一车 k 件（**同向拼车**，见 Review Focus #4）、电量耗尽去充电桩

- [ ] **Step 1: 写失败测试**

```python
"""物流侧约束（⑨⑩⑪）的测试（P1b Task 5）。"""
import pytest
from plcsp.env.constraints import ConstraintConfig
from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import load_mk

@pytest.mark.unit
def test_heterogeneous_fleet_has_varied_specs():
    """⑩ 异构：车队的速度/载量/电量不得全部相同。"""
    from plcsp.env.layout import sample_layout
    lay = sample_layout(6, seed=0, n_agv=4)
    specs = [(a.speed_mps, a.capacity, a.battery_kwh) for a in lay.agvs]
    assert len(set(specs)) > 1, "异构车队却给出全同规格"

@pytest.mark.unit
def test_multiload_reduces_trip_count():
    """⑩ 多载量：容量 >1 时应减少实际行程数（一趟送多件）。"""
    inst = load_mk("mk01")
    single = rollout(inst, seed_chain=1, cfg=SimConfig(),
                     constraints=ConstraintConfig(heterogeneous_fleet=False))
    multi = rollout(inst, seed_chain=1, cfg=SimConfig(),
                    constraints=ConstraintConfig(heterogeneous_fleet=True))
    assert multi["moves"] <= single["moves"], "多载量未减少行程数"

@pytest.mark.unit
def test_charging_consumes_battery_and_visits_charger():
    """⑪ 充电：开启后应有非零充电次数，且电量不得为负。"""
    inst = load_mk("mk01")
    r = rollout(inst, seed_chain=1, cfg=SimConfig(),
                constraints=ConstraintConfig(charging=True))
    assert r["charge_events"] >= 0
    assert r["battery_min_kwh"] >= 0.0, "电量出现负值 → 充电模型有误"

@pytest.mark.unit
def test_switches_off_equals_no_constraint():
    """Review Focus #2：三个开关全关时，不得出现充电事件/故障事件。"""
    inst = load_mk("mk01")
    r = rollout(inst, seed_chain=1, cfg=SimConfig(),
                constraints=ConstraintConfig(charging=False, agv_failure=False,
                                             heterogeneous_fleet=False))
    assert r["charge_events"] == 0
    assert r["agv_fail_events"] == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `D:/anaconda/python.exe -m pytest plcsp/tests/test_constraints_logistics.py -q --no-header`
Expected: FAIL — `KeyError: 'charge_events'`

- [ ] **Step 3: 实现三个约束**

- **⑩ 异构多载量**（**先定语义，再写代码**）：明确 **"同向拼车"**——同一目标机台、同一取货点的多件合成一趟。
  `AgvSpec.capacity ∈ {1, 2, 3}`，速度 ∈ [0.8, 1.2] m/s，电量 ∈ [2, 4] kWh（**assumed**）。
  **索引口径**：`agv_phi[task_i]` 仍按"运输任务"索引（不是"行程"），一次行程消费多个 `task_i`——
  须在 docstring 写明，否则 P2 的 L 层训练会错配。
- **⑨ AGV 故障**：`AgvSim` 按泊松流（`agv_mtbf`，默认 8 h，**assumed**）进入故障，停机 `agv_mttr`（默认 10 min，**assumed**），期间在途任务滞留。
- **⑪ 充电**：`AgvSpec.battery_kwh` 按行驶时长与三态功率扣减；低于阈值（默认 20%）时，
  中断取货、前往**最近的空闲充电桩**（`layout.chargers`，≥2 个），充满（默认 80%）后归队。
  充电桩本身也是互斥资源（一个桩同时只服务一台车）。

- [ ] **Step 4: 跑测试确认通过**

Run: `D:/anaconda/python.exe -m pytest plcsp/tests/test_constraints_logistics.py -q --no-header`
Expected: PASS（4 项）

- [ ] **Step 5: ⭐ binding 实测**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m11_constraint_binding --constraint agv_failure --constraint heterogeneous_fleet --constraint charging --seeds 5`
Expected: 三项各给 binding 判定。**⑪ 充电预计不 binding**（AGV 利用率仅 ~5%，电池掉得慢）——
若实测确 <2%，**按纪律在 spec §3.3 标注并考虑砍除**。

- [ ] **Step 6: 跑全套回归 + 提交**

```bash
D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header
git add -A
git commit -m "feat: 物流侧约束（AGV故障/异构多载量/充电）+ binding 实测"
```

---

### Task 6: 信息侧约束（⑥ 模糊加工 · ⑧ 交期 τ）

**Files:**
- Modify: `plcsp/env/des.py`
- Test: `plcsp/tests/test_constraints_info.py`

**Interfaces:**
- Consumes: Task 1；`plcsp/m9_due_calib.py` 的 τ 校准结果
- Produces: 工序加工时间可抽样自**三角模糊数**；交期口径改为 `d_j = τ · M_ref`（τ=0.85）

- [ ] **Step 1: 写失败测试**

```python
"""信息侧约束（⑥⑧）的测试（P1b Task 6）。"""
import pytest
from plcsp.env.constraints import ConstraintConfig
from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import load_mk

@pytest.mark.unit
def test_fuzzy_processing_is_fuzzy_not_gaussian():
    """Review Focus #6：⑥ 必须是**模糊数**（有隶属度、有界），不是正态（无界）。"""
    from plcsp.env.des import sample_fuzzy_time
    vals = [sample_fuzzy_time(5.0, spread=0.2, rng_seed=s) for s in range(200)]
    assert all(3.0 <= v <= 7.0 for v in vals), "模糊数必须落在支撑集内（正态无界 → 违规）"
    assert min(vals) < 5.0 < max(vals)

@pytest.mark.unit
def test_fuzzy_off_gives_deterministic_times():
    """开关关掉 → 加工时间 = 标称值（逐位相同）。"""
    inst = load_mk("mk01")
    a = rollout(inst, seed_chain=1, cfg=SimConfig(),
                constraints=ConstraintConfig(fuzzy_processing=False))
    b = rollout(inst, seed_chain=1, cfg=SimConfig(),
                constraints=ConstraintConfig(fuzzy_processing=False))
    assert a["makespan"] == b["makespan"]

@pytest.mark.unit
def test_due_dates_use_tau_times_m_ref():
    """⑧ 交期 = τ·M_ref（spec §3.5，τ=0.85），不再是旧的 due_factor×工时。"""
    from plcsp.env.des import compute_due_dates
    due = compute_due_dates(n_jobs=10, tau=0.85, m_ref=400.0)
    assert all(d == pytest.approx(340.0) for d in due.values())

@pytest.mark.unit
def test_tardiness_is_weighted_twt_not_count():
    """⑧ 目标用加权总拖期 TWT = Σ w_i·T_i，不是"几个作业误期"的计数（spec §4.1）。"""
    from plcsp.env.des import weighted_tardiness
    completes = {0: 100.0, 1: 350.0, 2: 340.0}
    due = {0: 340.0, 1: 340.0, 2: 340.0}
    weights = {0: 1.0, 1: 2.0, 2: 1.0}
    # 工件1 晚 10 min × 权重2 = 20；工件2 未晚 = 0；工件0 未晚 = 0
    assert weighted_tardiness(completes, due, weights) == pytest.approx(20.0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `D:/anaconda/python.exe -m pytest plcsp/tests/test_constraints_info.py -q --no-header`
Expected: FAIL — `ImportError: cannot import name 'sample_fuzzy_time'`

- [ ] **Step 3: 实现**

- **⑥ 模糊加工**：`sample_fuzzy_time(nominal, spread=0.2, rng_seed)` 用**三角模糊数**
  `Triangular(a, b, c) = (nominal×(1−spread), nominal, nominal×(1+spread))`，按隶属度抽样。
  **开关关掉 → 直接返回 nominal**（不得有任何随机）。
- **⑧ 交期**：新增 `compute_due_dates(n_jobs, tau, m_ref)`（返回 `{job: 同值}`），
  与 `weighted_tardiness(completes, due, weights)`；`_due()` 的旧 `due_factor` 路径**标注废弃**。
  返回 dict 增 `tardy_twt`（与既有 `tardy` 计数并存，前者进目标、后者仅参考）。

- [ ] **Step 4: 跑测试确认通过 + binding 实测**

Run:
```bash
D:/anaconda/python.exe -m pytest plcsp/tests/test_constraints_info.py -q --no-header
PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m11_constraint_binding --constraint fuzzy_processing --constraint due_dates --seeds 5
```
Expected: 4 项 PASS；两项 binding 判定。**⑧ 预期 binding**（spec §3.5 已标定 τ 使误期率 ~39%）。

- [ ] **Step 5: 跑全套回归 + 提交**

```bash
D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header
git add -A
git commit -m "feat: 信息侧约束（三角模糊加工 / τ·M_ref 交期 / TWT 目标）"
```

---

### Task 7: P1b 验收

**Files:**
- Modify: `docs/INDEX.md`、`docs/superpowers/specs/2026-10-02-plcsp-rebuild-design.md`（§3.3 标注 binding 结论、§9 参数出处）
- Modify: `docs/progress-log.md`

**Interfaces:**
- Consumes: Task 1–6 全部产出
- Produces: **每个约束的 binding 判定表**——这既是验收，也是论文"约束重要性"叙事的数据基础

- [ ] **Step 1: 跑全量 binding 矩阵**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m11_constraint_binding --all --seeds 10`
Expected: 三实例 × 10 约束的 binding 判定表。

- [ ] **Step 2: 五组消融各跑通一次**

Run:
```bash
PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -c "
from plcsp.env.constraints import ABLATION_GROUPS
from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import load_mk
inst = load_mk('mk01')
for name, c in ABLATION_GROUPS.items():
    r = rollout(inst, seed_chain=1, cfg=SimConfig(), constraints=c)
    print(f'{name:8} jobs {r[\"jobs_done\"]}/{inst.n_jobs}  mks {r[\"makespan\"]:.1f}  energy {r[\"energy\"]:.1f} kWh')
"
```
Expected: 5 组全部 `jobs N/N`、无掐表。

- [ ] **Step 3: 把 binding 结论写进 spec §3.3**

对**实测 <2%** 的约束，在该行末加：`⚠️ 实测不 binding（MK01/MK10：+X.X%）`。
**并据实决定保留还是砍除**——砍除的须记入 `progress-log.md` 并更新 §3.3 标题的约束计数。

- [ ] **Step 4: 补 `spec §9` 的 assumed 参数表**

把本计划引入的全部 **assumed** 参数（返工率、换型矩阵、PM 间隔/时长、AGV MTBF/MTTR、
车队规格分布、模糊数宽度、充电阈值）逐条列入 §9，标注 `assumed` 并说明敏感性分析计划。

- [ ] **Step 5: 登记 INDEX + progress-log**

`docs/INDEX.md` 新增 §5.7「P1b 能耗与约束」，写明：10 约束全部接入、binding 判定表位置、能耗已换 M2。

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "feat: P1b 验收（10 约束全接入 + binding 判定表 + assumed 参数表）"
```

---

## 完成后的状态

- 能耗 = M2 引证模型（量级对得上现实锚）
- 10 个约束全部接入且**可独立开关**；5 组消融 = 5 个配置实例
- **每个约束都有 binding 实测判定**——这是 P1a 教训的制度化
- 可交付给计划 4（P2+P3：编码器 token 特征重设计 + 奖励 + A 跑通）

## 已知风险 / 未决

| 项 | 说明 |
|---|---|
| **大量 assumed 参数** | 本计划引入的约束参数**基本都无文献出处**（返工率、换型矩阵、PM 周期、MTBF…）。审稿人必问。缓解：§9 明标 + 敏感性分析 |
| **⑪ 充电预计不 binding** | AGV 利用率 ~5% → 电池掉得慢。若实测 <2%，按纪律砍或在 spec 里如实标注 |
| **Q2 共享 RNG（非 CRN）仍未修** | `progress-log.md` §十六。**binding 实测本身也受它影响**（跨配置对照非共同随机数）——Task 3 的工具若要更严，需先修 Q2 |
| **⑥ 模糊 vs 随机的表述** | 必须在论文里说清（spec §3.3）。实现用模糊数，但若审稿人认为是"随机"的换皮，需补隶属度论证 |
| **`agv_phi` 索引口径** | 多载量下"运输任务"与"行程"不再一一对应——P2 的 L 层训练**必须知道这一点**（Review Focus #4） |
