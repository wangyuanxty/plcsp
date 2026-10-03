# ⑧ 交期口径重设计（TF/RDD）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ⑧ 交期从「`d_j = τ·M_ref`（共同交期、锚在自己的参考调度上）」换成**外生、逐作业、有跨度**的 TF/RDD 口径，使 TWT 在策略改进后**不再退化成恒 0**。

**Architecture:** 只改**目标读数**（交期是 metric，不进仿真时序）。新增一个纯数据模块 `plcsp/env/due_dates.py`（作业工时、负荷下界、TF/RDD 生成、标定表）+ 一个标定脚本。`des.py` / `reward.py` / `snapshot.py` 改为调用它；`SimConfig.tau` 从"唯一标定值"降级为**覆盖开关**。

**Tech Stack:** Python 3.12.4（`D:/anaconda/python.exe`）｜CPU-only｜numpy 1.26.4｜pytest 7.4.4｜Windows 11

**Spec:** `docs/superpowers/specs/2026-10-02-plcsp-rebuild-design.md` **§3.5**（本计划**取代**该节）
**依据**：`docs/progress-log.md` **§20.4-2 / §20.5**（死目标实测）与 **§21**（根因与标定实测）

## Global Constraints

- **项目根** = `D:esearch\DeepReinforcementLearningScheduling`。**根目录不留临时产物**（`*.log` / `*.jsonl` / `figs/` / 提取文本 / 渲染图一律进系统临时目录，用完即删）——正式文档放哪里不受限。
- **Python** 一律 `D:/anaconda/python.exe`；**CPU-only**。
- **单位约定（不得改动）**：仿真时间 = 分钟，布局坐标 = 米，能耗 = kWh。距离→分钟 = `d_m / (eff_speed·speed) / 60`。
- **回归门禁**：`plcsp/tests/` 现有 **231 项必须全绿**（除本计划明确要求改写的两项，见 Task 3）。
- **行尾**：本仓是 CRLF/LF 混合。**改既有文件用 Edit**；新 `.py` 跟随同目录多数派（`plcsp/env/` **平局**，取同侪 `des.py`/`instances.py` = **CRLF**；`plcsp/*.py` = **LF**）。
- **禁止临时产物**到包目录或根目录。
- **提交格式** `<type>: <description>`，**无署名/生成标识**。
- **⚠️ 术语纪律（用户明令）**：**不自造术语**。用文献承认的词：「**总工时（total work content, TWK）**」「**交期松紧因子（tardiness factor, TF）**」「**交期跨度（due-date range, RDD）**」。引用给完整题名 + 期刊 + 卷期页 + 年。

## Review Focus

1. **改的是读数还是仿真？** ⑧ 只影响目标，**不得改变任何事件时序**。期望：`makespan` 在 ⑧ 开/关、在任何 (τ,R) 下**逐位相同**（已有 `test_due_dates_switch_is_not_a_dead_flag` 钉，本计划须保持它绿）。
2. **`_CFG_KEY_SKIP` / `_ref_cfg_key` 的 tau 项**。`tau` 现在还有 `R` 同伴，且语义从"唯一标定值"变成"覆盖开关"。期望：两处指纹都覆盖 `(tau, due_range)`，且**参考运行的缓存键仍不含它们**（参考运行与交期无关，实测过）。
3. **未标定实例**（`gen_random` 造的、或新增的 MK 之外实例）不得静默用错值。期望：显式报错或走一个**写明出处**的默认档，不得 `KeyError` 崩在深处、也不得静默取 0。
4. **`test_features.py` 的"due_margin 是冗余维"测试**。逐作业交期下该维**不再冗余**——那条测试必须改写而不是删掉，且改写后要能证明"现在有区分度"。
5. **`ReferenceObjectives` 的 TWT** 必须与新交期**同源**。期望：`of()` 里的 `compute_due_dates` 换成新入口，且 `matches()` 指纹覆盖新参数。
6. **标定表必须是可复现的产物**，不是手抄的魔数。期望：标定脚本 + 测试断言表与脚本一致（或在容差内）。

---

### Task 1: TF/RDD 交期生成（纯函数 + 标定表）

**Files:**
- Create: `plcsp/env/due_dates.py`
- Test: `plcsp/tests/test_due_dates.py`

**Interfaces:**
- Consumes: `plcsp.env.instances.Instance`
- Produces:
  - `total_work_content(inst) -> list[float]` —— `W_j` = 作业 j 各工序最短候选工时之和
  - `workload_lower_bound(inst) -> float` —— `LB = max(ΣW_j / m, max_j W_j)`
  - `tf_rdd_due_dates(inst, tau: float, due_range: float) -> dict[int, float]` —— `d_j = LB·(τ)·(1 + R·(2ρ_j − 1))`，`ρ_j` = `W_j` 升序排名归一化到 [0,1]
  - `TF_RDD: dict[str, tuple[float, float]]` —— 逐实例标定的 `(τ, R)`（mk01–mk10，值见 Task 2 标定结果）
  - `due_dates_for(inst, tau: float | None = None, due_range: float | None = None) -> dict[int, float]`

- [ ] **Step 1: 写失败测试**

```python
"""TF/RDD 交期口径的测试（⑧ 重设计 Task 1）。"""
from __future__ import annotations

import numpy as np
import pytest

from plcsp.env.instances import Instance, gen_random, load_mk


@pytest.mark.unit
def test_total_work_content_is_min_over_alternatives():
    """W_j = 该作业各工序**最短候选工时**之和（纯实例数据，与任何调度无关）。"""
    from plcsp.env.due_dates import total_work_content

    inst = Instance(n_jobs=1, n_machines=2, jobs=[[[(0, 5.0), (1, 9.0)], [(1, 3.0)]]])
    assert total_work_content(inst) == [8.0]


@pytest.mark.unit
def test_workload_lower_bound_is_a_lower_bound():
    """LB 是 makespan 的下界：不得大于参考调度（否则口径自相矛盾）。"""
    from plcsp.env.des import SimConfig, reference_makespan
    from plcsp.env.due_dates import workload_lower_bound

    for name in ("mk01", "mk07", "mk10"):
        inst = load_mk(name)
        assert workload_lower_bound(inst) <= reference_makespan(inst, SimConfig())


@pytest.mark.unit
def test_due_dates_are_per_job_not_common():
    """逐作业交期：不同作业的 d_j **必须不同**（共同交期是旧口径的退化形式）。"""
    from plcsp.env.due_dates import tf_rdd_due_dates

    inst = load_mk("mk01")
    d = tf_rdd_due_dates(inst, tau=2.5, due_range=0.8)
    assert len(set(round(v, 6) for v in d.values())) > 1, "还是共同交期"


@pytest.mark.unit
def test_due_dates_are_exogenous():
    """⚠️ Review Focus 的核心：**交期不得随我们的调度变**。

    旧口径 `d_j = τ·M_ref` 锚在自己的参考调度上——换车队规模就换 M_ref（实测 mk01：
    n_agv=3 → 103.42，n_agv=1 → 109.95），交期跟着动。新口径只读实例数据。
    """
    from plcsp.env.due_dates import due_dates_for

    inst = load_mk("mk01")
    a = due_dates_for(inst)
    b = due_dates_for(inst)
    assert a == b, "同实例两次调用不同值"
    # 与 cfg 无关——签名里根本没有 cfg
    import inspect
    assert "cfg" not in inspect.signature(due_dates_for).parameters
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_due_dates.py -q --no-header`
Expected: FAIL — `No module named 'plcsp.env.due_dates'`

- [ ] **Step 3: 实现 `plcsp/env/due_dates.py`**

要点（实现者按此写，docstring 要写清**为什么**）：
- `total_work_content`：`[sum(min(t for _, t in alts) for alts in job) for job in inst.jobs]`
- `workload_lower_bound`：`max(sum(W)/inst.n_machines, max(W))`
- `tf_rdd_due_dates`：
  ```
  order = W_j 升序的作业号序列
  ρ_j = order.index(j) / (n_jobs − 1)          # n_jobs == 1 时取 0.5
  d_j = LB · tau · (1 + R·(2ρ_j − 1))
  ```
- `TF_RDD` 表 + `due_dates_for(inst, tau=None, due_range=None)`：显式传入 ≥ 表；表里没有的实例 → **显式报错**（Review Focus #3），错误信息要说明"该实例未标定，请跑 `plcsp/m16_due_calib.py`"。
- 模块 docstring 必须写清：口径出处（TF/RDD 交期生成，见 Task 3 的书目）、**为什么不用 `M_ref`**（内生 + 基准太弱导致 TWT 恒 0）、以及"LB 是纯实例数据"。

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_due_dates.py -q --no-header`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add plcsp/env/due_dates.py plcsp/tests/test_due_dates.py
git commit -m "feat: TF/RDD 交期生成（⑧ 重设计 Task 1）——外生、逐作业、有跨度"
```

---

### Task 2: 标定脚本 + 冻结 (τ, R) 表

**Files:**
- Create: `plcsp/m16_due_calib.py`
- Modify: `plcsp/env/due_dates.py`（把标定结果写进 `TF_RDD`）
- Test: `plcsp/tests/test_due_calib.py`

**Interfaces:**
- Produces: `calibrate(inst, *, target_ref: float = 0.45, improve: float = 0.88, lo: float = 0.20, hi: float = 0.75) -> tuple[float, float]`

**标定规则（必须原样实现，且写进 docstring）**：
> 在 `(τ, R)` 网格上求满足下式者，取 `|参考误期率 − target_ref| + |改进后误期率 − 0.40| + 0.02R` 最小的一对：
> - `0.20 ≤ 参考策略误期率 ≤ 0.75`
> - `0.20 ≤ 改进 12%（`C_j × 0.88`）后的误期率 ≤ 0.75`
>
> **为什么要"改进后也不许退化"这一条**：这正是旧口径的死因——τ=0.90 只对着**参考策略**标定，
> 而训练后策略改进约 12–14%，一次性把 TWT 清零。收紧 τ 是**保守方向**（只会让我们的 TWT 变大），
> 不存在"调参数美化结果"的激励。

- [ ] **Step 1: 写失败测试**

```python
"""交期标定的测试（⑧ 重设计 Task 2）。"""
from __future__ import annotations

import pytest

from plcsp.env.instances import load_mk

MK10 = [f"mk{i:02d}" for i in range(1, 11)]
IMPROVE = 0.88


@pytest.mark.unit
@pytest.mark.parametrize("name", MK10)
def test_calibrated_due_dates_stay_live_after_improvement(name: str):
    """⭐⭐ **本计划的判据测试**：交期在策略改进 12% 后**不得退化**。

    旧口径 `d_j = τ·M_ref` 正是在这里死的：训练后 makespan 89.5 < d_j=93.08 ⟹ TWT ≡ 0。
    新口径下，参考策略与"改进 12% 的策略"的误期率都必须落在 (0.15, 0.85)。
    """
    from plcsp.env.des import SimConfig, rollout
    from plcsp.env.due_dates import due_dates_for

    inst = load_mk(name)
    d = due_dates_for(inst)
    C = rollout(inst, seed_chain=0, cfg=SimConfig())["completes"]
    n = inst.n_jobs
    ref = sum(1 for j in C if C[j] > d[j]) / n
    imp = sum(1 for j in C if C[j] * IMPROVE > d[j]) / n
    assert 0.15 < ref < 0.85, f"{name} 参考策略误期率 {ref:.2f} 退化"
    assert 0.15 < imp < 0.85, f"{name} 改进 12% 后误期率 {imp:.2f} 退化（目标失效）"


@pytest.mark.unit
@pytest.mark.parametrize("name", MK10)
def test_twt_is_positive_after_improvement(name: str):
    """TWT 在改进后必须**非零**——这是"目标还活着"的直接读数。"""
    from plcsp.env.des import SimConfig, rollout
    from plcsp.env.due_dates import due_dates_for

    inst = load_mk(name)
    d = due_dates_for(inst)
    C = rollout(inst, seed_chain=0, cfg=SimConfig())["completes"]
    twt = sum(max(0.0, C[j] * IMPROVE - d[j]) for j in C)
    assert twt > 0.0, f"{name}: 改进 12% 后 TWT 恒 0——目标又死了"


@pytest.mark.unit
def test_uncalibrated_instance_raises():
    """⚠️ Review Focus #3：没标定的实例必须**显式报错**，不得静默取默认值。"""
    from plcsp.env.due_dates import due_dates_for
    from plcsp.env.instances import gen_random

    with pytest.raises((KeyError, ValueError)):
        due_dates_for(gen_random(4, 3, seed=0))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_due_calib.py -q --no-header`
Expected: FAIL — `TF_RDD` 里还没有标定值

- [ ] **Step 3: 实现标定脚本并把结果写进 `TF_RDD`**

`m16_due_calib.py` 打表（实例 | LB | τ | R | 参考误期率 | 改进后误期率），CLI `--instances --target-ref --improve`。
**最终冻结值**（2026-10-03 定案；由 `m16_due_calib.py` 在 R∈[0.2, 0.8] 网格上选出，**脚本输出即此表**——`test_frozen_table_is_reproduced_by_the_calibration_script` 逐实例对拍）：

| inst | τ | R | | inst | τ | R |
|---|---|---|---|---|---|---|
| mk01 | 2.45 | 0.2 | | mk06 | 5.60 | 0.2 |
| mk02 | 2.65 | 0.3 | | mk07 | 1.90 | 0.6 |
| mk03 | 3.40 | 0.5 | | mk08 | 2.50 | 0.8 |
| mk04 | 4.00 | 0.7 | | mk09 | 2.45 | 0.8 |
| mk05 | 1.55 | 0.6 | | mk10 | 3.00 | 0.7 |

⚠️ **τ 跨实例 1.55–5.60（3.6 倍）**——这是 TWK 类口径的固有代价（§3.5 早就实测到同类现象）。
**论文里必须如实写**，并给 R 的作用。

> ⚠️ **更正（实施者查出，已核实）**：计划原文写「mk10 的无 R 不可行性就是证据」——**这句是错的**。
> R=0 在 mk10 上**可行**（τ∈[2.70,2.90] 就落进速率带）。原判断是把 `τ·W_j`（离散比值格）与
> `τ·LB`（连续格）两种参数化混为一谈得出的。**R 下限的真实理由是「落实逐作业口径」**（用户明选），
> 不是可行性；标定里因此加了 `R_LO = 0.20`，另加 `R_HI = 0.80`（R>1 会产出负交期，实测踩到：
> mk04 在 R=1.2 下最短作业 −35.2）。
>
> ⚠️ **首轮值已作废**（原表：mk01 2.50/0.0、mk02 2.65/0.3、mk03 3.35/0.6、mk04 3.85/0.0、
> mk05 1.55/0.6、mk06 5.60/0.2、mk07 1.90/0.6、mk08 2.50/0.8、mk09 2.45/0.8、mk10 3.30/1.4）——
> 那是 R_LO=0 网格上的产物：mk01/mk04 落到 R=0、退化成共同交期，与"逐作业"的既定口径矛盾；
> 且实施者按规则重算时 mk03/mk10 与原文不一致（两对可行但评分更差）。上表是**最终值**。

- [ ] **Step 4: 跑测试确认通过 + 提交**

```bash
PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_due_calib.py -q --no-header
git add plcsp/m16_due_calib.py plcsp/env/due_dates.py plcsp/tests/test_due_calib.py
git commit -m "feat: 交期标定脚本 + 冻结 (tau,R) 表（⑧ 重设计 Task 2）"
```

---

### Task 3: 接线（`des` / `reward` / `snapshot`）+ 改写两条旧测试

**Files:**
- Modify: `plcsp/env/des.py`（`compute_due_dates`、`SimConfig.tau`、`_due_map`、`_CFG_KEY_SKIP`）
- Modify: `plcsp/env/reward.py`（`ReferenceObjectives.of`、`_ref_cfg_key`）
- Modify: `plcsp/env/snapshot.py`（`JobState.due` 的注释）
- Modify: `plcsp/tests/test_constraints_info.py`（`test_due_dates_use_tau_times_m_ref` 改写）
- Modify: `plcsp/tests/test_features.py`（`test_due_margin_is_redundant_with_global_time_progress` 改写）
- Test: `plcsp/tests/test_due_wiring.py`

**Interfaces:**
- `SimConfig.tau: float | None = None`（`None` = 用 `TF_RDD` 表）＋ 新增 `SimConfig.due_range: float | None = None`
- `des.compute_due_dates(inst, tau=None, due_range=None) -> dict[int, float]`（**签名变了**：不再收 `n_jobs/m_ref`）
- `SimWorld._due_map()` 无参（改为读 `self.inst` 与 `self.cfg`）

- [ ] **Step 1: 写失败测试**

```python
"""⑧ 接线的测试（⑧ 重设计 Task 3）。"""
from __future__ import annotations

import pytest

from plcsp.env.instances import load_mk


@pytest.mark.unit
def test_due_dates_do_not_depend_on_fleet_size():
    """⚠️ Review Focus 的核心：**换车队规模不得改变交期**。

    旧口径下 mk01 的 M_ref 随 n_agv 变（n_agv=3 → 103.42，n_agv=1 → 109.95），交期跟着变——
    等于"交期随我们的配置漂"。新口径只读实例数据。
    """
    from plcsp.env.des import SimConfig
    from plcsp.env.due_dates import due_dates_for

    inst = load_mk("mk01")
    assert due_dates_for(inst) == due_dates_for(inst, tau=None, due_range=None)
    # 通过 SimWorld 的 _due_map 再验一次（真正被仿真用的那条路）
    from plcsp.env.des import SimWorld
    from plcsp.algo.setup import build_setup
    d1 = SimWorld(inst, *build_setup(inst, SimConfig(n_agv=1))[:2],
                  SimConfig(n_agv=1))._due_map()
    d3 = SimWorld(inst, *build_setup(inst, SimConfig(n_agv=3))[:2],
                  SimConfig(n_agv=3))._due_map()
    assert d1 == d3, "交期随车队规模变了——锚又回到内生量上了"


@pytest.mark.unit
def test_makespan_is_unchanged_by_due_date_caliber():
    """⚠️ Review Focus #1：⑧ 是 metric，**不得改变任何事件时序**。"""
    from plcsp.env.des import SimConfig, rollout

    inst = load_mk("mk01")
    a = rollout(inst, seed_chain=1, cfg=SimConfig())
    b = rollout(inst, seed_chain=1, cfg=SimConfig(tau=2.0, due_range=1.0))
    c = rollout(inst, seed_chain=1, cfg=SimConfig(tau=9.0, due_range=0.0))
    assert a["makespan"] == b["makespan"] == c["makespan"]
    assert a["tardy_twt"] != c["tardy_twt"], "换 τ 读数没变——τ 根本没接上"


@pytest.mark.unit
def test_reference_objectives_twt_uses_the_same_due_dates():
    """⚠️ Review Focus #5：`ReferenceObjectives` 的 TWT 必须与仿真同源。"""
    from plcsp.env.des import SimConfig, reference_run
    from plcsp.env.reward import ReferenceObjectives

    inst = load_mk("mk01")
    cfg = SimConfig()
    ref = ReferenceObjectives.of(inst, cfg)
    r = reference_run(inst, cfg)
    assert ref.twt == pytest.approx(r["tardy_twt"]), "ref 的 TWT 与参考运行不同源"


@pytest.mark.unit
def test_due_margin_is_no_longer_redundant():
    """⚠️ Review Focus #4：逐作业交期下 `due_margin` 维**不再**与 `time_progress` 线性等价。

    旧口径（共同交期 `d_j = τ·M_ref`）下 `due_margin = τ − time_progress`，对所有作业 token
    同值 ⟹ 冗余维。新口径下逐作业取不同值 ⟹ 有区分度。
    """
    from plcsp.env.des import SimConfig, reference_run
    from plcsp.env.due_dates import due_dates_for

    inst = load_mk("mk01")
    d = due_dates_for(inst)
    assert len(set(round(v, 6) for v in d.values())) > 1
```

- [ ] **Step 2: 跑测试确认失败** → Expected: FAIL（`_due_map()` 还要参数 / `ReferenceObjectives.twt` 不同源）

- [ ] **Step 3: 接线**
  - `des.py`：`SimConfig.tau: float | None = None`；新增 `due_range: float | None = None`；`_CFG_KEY_SKIP` 改为 `("tau", "due_range")`（参考运行仍与交期无关——**实测过**，换 τ 参考 makespan 逐位不变）；`compute_due_dates(inst, tau=None, due_range=None)` 转调 `due_dates_for`；`SimWorld._due_map()` 去参。
  - `reward.py`：`_ref_cfg_key` 覆盖 `(tau, due_range)`；`of()` 里 `compute_due_dates(inst, c.tau, c.due_range)`。
  - `snapshot.py`：更新 `JobState.due` 的注释（不再"同值"）。
  - `test_constraints_info.py::test_due_dates_use_tau_times_m_ref` → 改写为断言 **`d_j = LB·τ·(1+R(2ρ_j−1))`** 的具体值（逐作业）。
  - `test_features.py::test_due_margin_is_redundant_with_global_time_progress` → 改写为 **`test_due_margin_distinguishes_jobs`**（断言不同作业的 `due_margin` 不同）。

- [ ] **Step 4: 全套回归**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header`
Expected: 全绿（231 + 新增 − 0；两条改写的不减少计数）

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: ⑧ 交期接线 TF/RDD（⑧ 重设计 Task 3）——tau 降级为覆盖开关，两条旧测试改写"
```

---

### Task 4: 重跑消融 + 更新文档

**Files:**
- Modify: `docs/superpowers/specs/2026-10-02-plcsp-rebuild-design.md`（**§3.5 整节重写**）
- Modify: `docs/progress-log.md`（新增 §二十一）
- Modify: `docs/INDEX.md`（§5.10）

- [ ] **Step 1: 重跑消融（MK01、30 步、G=8、种子 0）**

```bash
PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m15_ablation_smoke --instances mk01 --steps 30 --G 8
```
Expected: 五组 TWT **不再全是 0.0**；Cmax 与 §20.4 的表**不同**（`due_margin` 特征变了 ⇒ 策略输入变了，这是预期内的）。

- [ ] **Step 2: 重写 spec §3.5**

必须含：旧口径与它的死因（**实测**：训练后 makespan 89.5 < d_j 93.08 ⟹ TWT ≡ 0 是恒等式）、新口径的公式、`(τ,R)` 表、
标定规则（含"改进后不许退化"这条约束及其**保守方向**论证）、以及书目。书目按 TF/RDD 的出处给完整信息（实现者须自行核实并给完整题名+期刊+卷期页+年；**拿不到原文就写"待核"**，不得编）。

- [ ] **Step 3: progress-log 新增 §二十一**：根因、标定实测表、mk10 的无 R 不可行性、重跑读数、开放线索。

- [ ] **Step 4: 全套回归 + 提交**

```bash
PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header
git add -A && git commit -m "docs: ⑧ 交期口径重设计的实测记录与 spec §3.5 重写（⑧ 重设计 Task 4）"
```

---

## 完成后的状态

- ⑧ 交期**外生**（只读实例数据）、**逐作业**、**有跨度**，且在策略改进 12% 后仍不退化为恒 0
- `due_margin` 维从冗余维变成有区分度的维
- `(τ, R)` 表由脚本可复现地生成，不是手抄魔数

## 已知边界（**必须带进 P4-B**）

| 项 | 说明 |
|---|---|
| **τ 跨实例 1.55–5.60** | TWK 类口径的固有代价；论文必须如实写。⚠️ R 的作用**不是**"mk10 无 R 不可行"（那是错的，见 Task 2 的更正框）——R 是**落实逐作业口径**（下限 0.2）与**避免负交期**（上限 0.8） |
| **只对 10 个 MK 实例标定过** | 新实例（含 MKT 口径下的变体）须重跑标定脚本；`due_dates_for` 对未标定实例**显式报错**。⚠️ `TF_RDD` 按 `source` 文件名主干取键 ⟹ `load_mkt` 会静默命中同名 MK 条目——P4-B 接 `trans_time` 前必须先解决 |
| **旧读数全部作废** | `due_margin` 是编码器输入 ⇒ 策略输入变了 ⇒ 所有训练读数（含 §20.4 消融表、P2 的 A 端到端）**都要重跑**才能再引用 |
| **`improve=0.88` 是乐观假设，不是"设计余量"** | 实测 mk01 改进 13%–18%（§20.4 `89.5/103.4≈0.87`、§19.2 `85.0/103.4≈0.82`），取 12% 是往**弱**的一端假设，而饱和风险随改进增大 ⟹ 偏乐观。计划里写的 `92.0/103.4=0.89` 在本仓查不到出处 |
| **mk02/mk06 余量最小且无真实策略复核** | 均匀缩放比 `min_j d_j / makespan`：mk02 **0.61**（改进 39% 即饱和）、mk06 **0.64**（36%）——比 mk01 的 0.48 更早饱和；目前只有解析代理（`C_j×0.88`），P4-B 须用真实 checkpoint 复核 |
| **TWT 对远好于参考的策略仍会饱和** | 交期是绝对阈值：mk01 的 `−生产` 消融组 Cmax 47.8 < 最早交期 49.98 ⟹ 整组 TWT ≡ 0（⑧ 开着）。标定的"改进 12% 不退化"只覆盖参考附近；论文须写明适用区间（见 progress-log §21.5） |
