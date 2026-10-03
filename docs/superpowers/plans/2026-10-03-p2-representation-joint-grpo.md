# P2+P3 表征重建与联合链 GRPO 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把"输入全零、编码器不在活路径上、两头各训各的"的骨架，改造成 spec §5.3 定义的**在线两决策 + 共享编码器 + 联合链 GRPO + 三目标奖励**，并让 A（加权标量化）在 MK01 上端到端跑通。

**Architecture:** 四层单向依赖：`snapshot.py`（仿真状态 → 纯数据）→ `features.py`/`state_emb.py`（纯数据 → 张量）→ `encoder.py`/`policy.py`（张量 → 两头打分）→ `group_rel.py`（联合链训练）。仿真层（`des.py`）只负责**在派工点回调策略并记录决策**，不含任何网络代码。

**Tech Stack:** Python 3.12.4（`D:/anaconda/python.exe`）｜torch 2.14.0+cpu｜simpy 4.1.2｜numpy 1.26.4｜pytest 7.4.4｜Windows 11

**Spec:** `docs/superpowers/specs/2026-10-02-plcsp-rebuild-design.md`（**§5.3 为权威**；§4.2 奖励、§5.1 编码器、§5.2 算法）

## Global Constraints

- **项目根** = `D:esearch\DeepReinforcementLearningScheduling`。**根目录不留临时产物**（`*.log` / `*.jsonl` / `figs/` / 提取文本 / 渲染图一律进系统临时目录，用完即删）——正式文档放哪里不受限。
- **Python** 一律用 `D:/anaconda/python.exe`。**CPU-only**。
- **单位约定（bug#13，不得改动）**：仿真时间 = 分钟，布局坐标 = 米，能耗 = kWh。
- **回归门禁**：`plcsp/tests/` 的 **107 项必须始终全绿**。
- **⚠️ 行尾：保持每个文件**既有**的行尾，不得转换。** 本仓 HEAD 是 **混合的**（18 个 CRLF / 24 个 LF，实测于 `8c67dcc`）——不存在"统一是 LF"这回事。
  转换会令 diff 从真实改动虚增成全文重写。**改既有文件一律用 Edit 工具**（它保留原行尾）；新建文件跟随同目录既有文件的多数派。
- **禁止在根目录/包目录留临时产物**（`*.log`、`*.jsonl`、`figs/`）——已在 `.gitignore` 中，但**也不要生成**。
- **提交信息格式**：`<type>: <description>`，**不添加任何署名/生成标识**。

## Review Focus

以下六类是本计划的测试覆盖不到、但最容易出事的：

1. **归一化尺度跨实例崩**。MK01 与 MK10 的总工时差 **12 倍**（153 vs 1847 min）。期望行为：**每一维都除以实例静态量**（总工时 / `M_ref` / 机器数 / 包围盒对角线），故同一物理量在两实例上的取值域可重叠；若某维漏归一化，MK10 上该维量级压倒其余维，网络退化成单维策略。
2. **三段特征索引错位**。`tok_feat` 按 `[M | B | V | G]` 拼接，任何一处的顺序/宽度不一致都会**静默错配**（B 段第 3 行写成了机台量也不会报错）。P1a 有前科（"机台号当节点号"）。期望行为：`seg` 由**各段实际长度**推出而非硬编码；有测试逐段校验首行。
3. **在线 S 层悄悄改变了口径**。静态计划下 `plans` 是常量，在线后它是**决策日志**。期望行为：**关闭在线开关时，同种子逐位等于改造前的静态行为**（对拍测试）。
4. **`f^ref` 跨实例不可比**。奖励权重 `wᵢ = 1/fᵢ^ref` 是**按实例**算的，故奖励绝对值跨实例不可比。期望行为：权重只在单实例内使用；训练/评估若跨实例混用会静默出错，须有断言挡住。
5. **联合 logp 的梯度是假的**。若 L 头的 logp 被 `detach` 或走了 no_grad 路径，"联合链"就只是名义上的。期望行为：**从 `logp_L` 反传到编码器参数，梯度非零**。
6. **决策日志的内存与确定性**。一条链有 ~460 个决策（240 S + 220 L），每个决策要存**当时的特征快照**（因为仿真状态在变）。期望行为：同种子重放同一决策日志；日志大小与实例规模成线性而非平方。

---

### Task 1: 仿真状态快照（`snapshot.py`）

**Files:**
- Create: `plcsp/env/snapshot.py`
- Modify: `plcsp/env/des.py`（`SimWorld.snapshot()`）
- Test: `plcsp/tests/test_snapshot.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `MachineState`（frozen dataclass）：`backlog_min: float` / `in_q_len: int` / `in_cap: float` / `out_q_len: int` / `out_cap: float` / `busy: bool` / `remaining_min: float` / `pm_used_min: float` / `fail_rate: float` / `prev_job: int`
  - `JobState`：`done_ops: int` / `total_ops: int` / `remaining_min: float` / `finished: bool` / `at_machine: int` / `in_transit: bool` / `on_agv: int`
  - `VehicleState`：`status: int`（0 空闲/1 空载/2 负载/3 故障）/ `node: int` / `queued: int` / `battery_frac: float` / `capacity: int` / `speed_factor: float`
  - `Snapshot`：`now: float` / `machines: tuple` / `jobs: tuple` / `vehicles: tuple` / `n_done: int` / `in_flight: int`
  - `SimWorld.snapshot() -> Snapshot` —— **只读**，不得改变仿真状态

> ⚠️ 本任务**只做数据搬运**，不含任何归一化与神经网络（归一化在 Task 2）。

- [ ] **Step 1: 写失败测试**

创建 `plcsp/tests/test_snapshot.py`：

```python
"""仿真状态快照的测试（P2 Task 1）。快照是特征层的唯一原料，且必须是只读的。"""
from __future__ import annotations

import pytest

from plcsp.env.des import SimWorld, SimConfig, rollout
from plcsp.env.instances import load_mk
from plcsp.env.layout import sample_layout
from plcsp.env.corridors import build_corridor_graph, dock_distance_matrix


def _world(name: str = "mk01"):
    inst = load_mk(name)
    lay = sample_layout(inst.n_machines, seed=0, n_agv=3)
    dm = dock_distance_matrix(build_corridor_graph(lay))
    return inst, SimWorld(inst, lay, dm, SimConfig(), graph=build_corridor_graph(lay))


@pytest.mark.unit
def test_snapshot_shapes_match_instance():
    """快照的三段长度必须等于实例的机台数/作业数/车队数。"""
    inst, w = _world()
    snap = w.snapshot()
    assert len(snap.machines) == inst.n_machines
    assert len(snap.jobs) == inst.n_jobs
    assert len(snap.vehicles) == SimConfig().n_agv
    assert snap.now == 0.0, "未跑仿真时 now 应为 0"


@pytest.mark.unit
def test_snapshot_is_read_only():
    """取快照不得改变仿真状态——对拍两次必须逐位相同。"""
    inst, w = _world()
    a, b = w.snapshot(), w.snapshot()
    assert a == b


@pytest.mark.unit
def test_job_state_tracks_progress_and_location():
    """跑一段后：已完成作业的 done_ops == total_ops；未开始的 at_machine == -1。"""
    inst, w = _world()
    w.run(seed_chain=1)
    snap = w.snapshot()
    for j, js in enumerate(snap.jobs):
        assert js.total_ops == len(inst.jobs[j])
        assert 0 <= js.done_ops <= js.total_ops
        if js.done_ops == 0:
            assert js.at_machine == -1 and not js.in_transit
        if js.finished:
            assert js.done_ops == js.total_ops


@pytest.mark.unit
def test_machine_backlog_equals_sum_of_queued_op_times():
    """机台积压 = 输入队列内各工序的加工时长之和（这是最强的一条特征，来不得含糊）。"""
    inst, w = _world()
    snap = w.snapshot()
    for m, ms in enumerate(snap.machines):
        q = w.machines[m].in_q.items
        assert ms.in_q_len == len(q)
        assert ms.backlog_min == pytest.approx(sum(op.time for (_j, _oi, op, _l) in q))
        assert ms.backlog_min == 0.0, "初始队列应为空"


@pytest.mark.unit
def test_vehicle_status_is_one_of_four_values():
    inst, w = _world()
    snap = w.snapshot()
    assert all(v.status in (0, 1, 2, 3) for v in snap.vehicles)
    assert all(0.0 <= v.battery_frac <= 1.0 for v in snap.vehicles)
    assert all(v.capacity >= 1 for v in snap.vehicles)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_snapshot.py -q --no-header`
Expected: FAIL — `ModuleNotFoundError: No module named 'plcsp.env.snapshot'`

- [ ] **Step 3: 实现 `snapshot.py`**

```python
# -*- coding: utf-8 -*-
"""仿真状态快照（spec §5.3.1）——特征层的**唯一**原料。

设计：`SimWorld.snapshot()` 把当时的活状态**拷成纯数据**（frozen dataclass），
网络代码只读它、不碰 SimPy 对象。这样：
- 特征层可脱离仿真单测（手搓 Snapshot 即可）；
- 决策日志能只存快照（仿真继续跑，快照不跟着变）。

⚠️ **本模块不含任何归一化**——归一化在 `nn/state_emb.py`（Task 2），
因为归一化需要实例静态量（总工时/M_ref），那是特征层的依赖而非仿真层的。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MachineState:
    backlog_min: float      # 输入队列内各工序加工时长之和 [min]
    in_q_len: int
    in_cap: float           # inf 表示无界（② 关闭时）
    out_q_len: int
    out_cap: float
    busy: bool              # 是否持有加工槽
    remaining_min: float    # 当前工序剩余加工时长（未加工 = 0）
    pm_used_min: float      # 距上次保养已累计的主轴工时（⑫）
    fail_rate: float        # ③
    prev_job: int           # 本机上一件加工的作业号；-1 = 还没加工过（⑤ 换型的依据）


@dataclass(frozen=True)
class JobState:
    done_ops: int
    total_ops: int
    remaining_min: float    # 剩余工序的**标称**加工时长之和
    finished: bool
    at_machine: int         # 当前所在机台；-1 = 未开始
    in_transit: bool
    on_agv: int             # 在途时所乘的车；-1 = 无


@dataclass(frozen=True)
class VehicleState:
    status: int             # 0 空闲 / 1 空载行驶 / 2 负载行驶 / 3 故障停机
    node: int               # 当前所在通道节点
    queued: int             # 本车待办任务数
    battery_frac: float     # [0,1]
    capacity: int
    speed_factor: float


@dataclass(frozen=True)
class Snapshot:
    now: float
    machines: tuple[MachineState, ...]
    jobs: tuple[JobState, ...]
    vehicles: tuple[VehicleState, ...]
    n_done: int             # 已完成作业数
    in_flight: int          # 全线在途运输任务数
```

在 `des.py` 的 `SimWorld` 上加方法（放在 `_energy_report` 之后）：

```python
    def snapshot(self) -> "Snapshot":
        """当时的活状态 → 纯数据快照（**只读**，不改变任何仿真状态）。

        `_job_progress[j]` / `_job_loc[j]` / `_job_agv[j]` 由 `MachineSim` 与 `AgvSim` 维护
        （见 `run()` 里的初始化），是"作业进行到哪一步"的唯一真相。
        """
        from .snapshot import JobState, MachineState, Snapshot, VehicleState
        ms = []
        for i, m in enumerate(self.machines):
            q = m.in_q.items
            cur = self._cur_op[i]
            ms.append(MachineState(
                backlog_min=float(sum(it[2].time for it in q)),
                in_q_len=len(q), in_cap=float(m.in_q.capacity),
                out_q_len=len(m.out_q.items), out_cap=float(m.out_q.capacity),
                busy=bool(m.slot.count),
                remaining_min=float(cur[1]) if cur else 0.0,
                pm_used_min=float(m.pm_clock), fail_rate=float(m.pad.fail_rate),
                prev_job=(-1 if m.prev_job is None else int(m.prev_job))))
        js = []
        for j, job_ops in enumerate(self.inst.jobs):
            done = self._job_progress[j]
            js.append(JobState(
                done_ops=done, total_ops=len(job_ops),
                remaining_min=float(sum(min(t for _m, t in op) for op in job_ops[done:])),
                finished=done >= len(job_ops), at_machine=int(self._job_loc[j]),
                in_transit=bool(self._job_agv[j] >= 0), on_agv=int(self._job_agv[j])))
        vs = []
        for a, agv in enumerate(self.agvs):
            vs.append(VehicleState(
                status=3 if agv.down else (2 if self._agv_loaded[a] else
                                           (1 if agv.pos_node is not None else 0)),
                node=int(agv.pos_node if agv.pos_node is not None else -1),
                queued=len(self.tasks_in[a].items) if agv.bound else 0,
                battery_frac=float(agv.battery / max(agv.battery_cap, 1e-9)),
                capacity=int(agv.capacity), speed_factor=float(agv.speed / self.cfg.agv_speed_mps)))
        return Snapshot(now=float(self.env.now), machines=tuple(ms), jobs=tuple(js),
                        vehicles=tuple(vs), n_done=len(self.completes),
                        in_flight=sum(v.queued for v in vs))
```

配套：`SimWorld.run()` 里需要新增并挂上四个跟踪量（`self.env = env` 也要留住引用）：

```python
        self.env = env
        self.machines = machines
        self.agvs = [AgvSim(...) for a in range(self.cfg.n_agv)]   # 原来是直接 env.process(...)
        self.tasks_in = tasks_in
        self._job_progress = [0] * inst.n_jobs      # 已完成工序数
        self._job_loc = [-1] * inst.n_jobs          # 当前所在机台
        self._job_agv = [-1] * inst.n_jobs          # 在途时所乘的车
        self._cur_op: list[tuple[int, float] | None] = [None] * inst.n_machines  # (工件, 剩余)
        self._agv_loaded = [False] * self.cfg.n_agv
```

四个跟踪量的更新点（**这是本任务最容易漏的地方，逐条对应**）：

| 何时 | 更新什么 |
|---|---|
| `MachineSim.run` 取出工件、进入加工槽 | `_cur_op[pad.id] = (job, op.time)`；`_job_loc[job] = pad.id` |
| 加工完成、`_cur_op` 清零 | `_cur_op[pad.id] = None`；`_job_progress[job] = oi + 1` |
| `_transporter` 生成运输任务 | `_job_agv[job] = agv`（派车后）；`_job_loc[job] = -1` |
| `AgvSim` 投递完成 | `_job_agv[job] = -1`；`_agv_loaded[a]` 置 False |
| `AgvSim` 取货后开始负载行驶 | `_agv_loaded[a] = True` |

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_snapshot.py -q --no-header`
Expected: PASS（5 项）

- [ ] **Step 5: 跑全套回归**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header`
Expected: **112 项全绿**（107 + 5 新）

- [ ] **Step 6: 提交**

```bash
git add plcsp/env/snapshot.py plcsp/env/des.py plcsp/tests/test_snapshot.py
git commit -m "feat: 仿真状态快照（P2 Task 1）——特征层的唯一原料"
```

---

### Task 2: token 特征构造（`features.py` + `state_emb.py`）

**Files:**
- Modify: `plcsp/nn/features.py`（重写：字段定义 + 宽度）
- Modify: `plcsp/nn/state_emb.py`（重写：快照 → 张量）
- Test: `plcsp/tests/test_features.py`

**Interfaces:**
- Consumes: Task 1 的 `Snapshot` / `MachineState` / `JobState` / `VehicleState`
- Produces:
  - `F_M, F_B, F_V, F_G = 7, 8, 10, 3`（模块级常量，**宽度唯一真相**）
  - `FEATURE_NAMES: dict[str, tuple[str, ...]]`（每段字段名，供论文附录与错位排查）
  - `NormContext`（frozen dataclass）：`total_work_min` / `m_ref` / `n_m` / `n_jobs` / `n_agv` / `bbox_diag` / `max_fail_rate` / `pm_interval` / `max_weight` / `max_queued`
  - `norm_context(inst, layout, m_ref) -> NormContext`
  - `F_MAX = 10`（= max(7,8,10,3)）、`SEG_SLICE: dict[str, slice]`（各段在补齐张量里占的列）
  - `build_tok(snap, inst, layout, ctx) -> tuple[np.ndarray, tuple[int,int,int,int]]`（**单张 `(N, F_MAX)`** + `seg`）
  - `machine_features(snap, ctx) -> np.ndarray` `(n_m, F_M)`
  - `job_features(snap, ctx) -> np.ndarray` `(n_jobs, F_B)`
  - `vehicle_features(snap, ctx) -> np.ndarray` `(n_agv, F_V)`
  - `global_features(snap, ctx) -> np.ndarray` `(1, F_G)`

> ⚠️ **输入是单张张量**（spec §5.3.1）：四段各自的 per-token 特征**补齐到 `F_MAX` 后按序列拼接**。
> 列语义在四段间**重叠**（第 3 列在 M 段是"在加工"、在 B 段是"已完成"），靠**类型嵌入**解耦——
> 故 Task 3 的类型嵌入是**必需**的，去掉即失效。

- [ ] **Step 1: 写失败测试**

创建 `plcsp/tests/test_features.py`：

```python
"""token 特征构造的测试（P2 Task 2）。"""
from __future__ import annotations

import numpy as np
import pytest

from plcsp.env.des import SimConfig, SimWorld
from plcsp.env.instances import load_mk
from plcsp.env.layout import sample_layout
from plcsp.env.corridors import build_corridor_graph, dock_distance_matrix
from plcsp.nn.features import F_B, F_G, F_M, F_MAX, F_V, FEATURE_NAMES, norm_context
from plcsp.nn.state_emb import build_tok


def _ctx_and_snap(name="mk01", done=False):
    inst = load_mk(name)
    lay = sample_layout(inst.n_machines, seed=0, n_agv=3)
    g = build_corridor_graph(lay)
    w = SimWorld(inst, lay, dock_distance_matrix(g), SimConfig(), graph=g)
    if done:
        w.run(seed_chain=1)
    return inst, lay, w, norm_context(inst, lay, m_ref=100.0)


@pytest.mark.unit
def test_field_counts_match_declared_widths():
    """⚠️ Review Focus #2：字段名清单与声明宽度必须逐段相等——防静默错位。"""
    assert len(FEATURE_NAMES["M"]) == F_M == 7
    assert len(FEATURE_NAMES["B"]) == F_B == 8
    assert len(FEATURE_NAMES["V"]) == F_V == 10
    assert len(FEATURE_NAMES["G"]) == F_G == 3


@pytest.mark.unit
def test_build_tok_is_one_padded_tensor():
    """输入是**单张** `(N, F_MAX)`——四段按行拼接、列不足处补零（spec §5.3.1）。"""
    inst, lay, w, ctx = _ctx_and_snap()
    tok, seg = build_tok(w.snapshot(), inst, lay, ctx)
    n_m, n_b, n_v, n_g = seg
    assert seg == (inst.n_machines, inst.n_jobs, SimConfig().n_agv, 1)
    assert tok.shape == (n_m + n_b + n_v + n_g, F_MAX)
    # 补零列必须**恒为 0**（M 段 7:、B 段 8:、G 段 3:）
    assert np.all(tok[:n_m, F_M:] == 0.0)
    assert np.all(tok[n_m:n_m + n_b, F_B:] == 0.0)
    assert np.all(tok[-n_g:, F_G:] == 0.0) 


@pytest.mark.unit
def test_features_are_finite_and_bounded():
    """归一化后不得出现 NaN/inf，且不应有远超 [0,1] 量级的失控维。"""
    inst, lay, w, ctx = _ctx_and_snap(done=True)
    tok, _ = build_tok(w.snapshot(), inst, lay, ctx)
    assert np.isfinite(tok).all(), "出现 NaN/inf"
    assert np.abs(tok).max() < 20.0, f"有维失控：max|·|={np.abs(tok).max():.1f}"


@pytest.mark.unit
def test_backlog_feature_is_normalized_by_total_work():
    """⚠️ Review Focus #1：积压维必须除以实例总工时——否则 MK01 与 MK10 差 12 倍。"""
    inst_s, _, ws, ctx_s = _ctx_and_snap("mk01")
    inst_l, _, wl, ctx_l = _ctx_and_snap("mk10")
    assert ctx_l.total_work_min > ctx_s.total_work_min * 10, "两实例总工时应有量级差"
    # 同一物理积压量在两实例上应映到相近的归一化值
    assert ctx_s.total_work_min == pytest.approx(153.0, rel=0.05)
    assert ctx_l.total_work_min == pytest.approx(1847.0, rel=0.05)


@pytest.mark.unit
def test_seg_lengths_come_from_actual_rows_not_hardcoded():
    """⚠️ Review Focus #2：seg 由各段**实际行数**推出，不得硬编码实例规模。"""
    inst, lay, w, ctx = _ctx_and_snap("mk10")
    tok, seg = build_tok(w.snapshot(), inst, lay, ctx)
    assert seg[0] == inst.n_machines == 15
    assert seg[1] == inst.n_jobs == 20
    assert tok.shape[0] == sum(seg)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_features.py -q --no-header`
Expected: FAIL — `ImportError: cannot import name 'F_M'`

- [ ] **Step 3: 重写 `features.py`**

```python
# -*- coding: utf-8 -*-
"""token 特征的**字段定义**（spec §5.3.1）——宽度与语义的唯一真相。

⚠️ 2026-10-03 重写：此前本模块只有 `F_DYN=6` 一个桩常量，**没有任何字段定义**，
且三类 token 被迫同宽（单一 `Linear` 所致）。现按 spec §5.3.1 定死三套字段 + Global，
四段**补齐到同一宽度** `F_MAX=10` 后按序列拼接成单张张量，由 `encoder.py` 的**单个 Linear** 升维。

**归一化一律用实例静态量**（总工时 / `M_ref` / 机器数 / 包围盒对角线）——
仿真前即知，训练与推理一致。**不用 per-episode 归一化**：在线决策下不可得。
"""
from __future__ import annotations

from dataclasses import dataclass

F_M, F_B, F_V, F_G = 7, 8, 10, 3
F_MAX = max(F_M, F_B, F_V, F_G)          # 10 —— 补齐后的统一宽度

# 各段在补齐张量 (N, F_MAX) 里占的列；超出部分恒为 0
SEG_SLICE: dict[str, slice] = {"M": slice(0, F_M), "B": slice(0, F_B),
                               "V": slice(0, F_V), "G": slice(0, F_G)}

# ⚠️ 字段名清单与宽度**必须逐段相等**——有测试守着（test_field_counts_match_declared_widths）。
# 它同时是论文附录的特征表与排错时的对照表。
FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    "M": ("backlog", "in_fill", "out_fill", "busy", "remaining_frac",
          "pm_left", "fail_rate"),
    "B": ("progress", "remaining_work", "due_margin", "finished",
          "at_machine", "in_transit", "on_agv", "weight"),
    "V": ("st_idle", "st_empty", "st_loaded", "st_down", "node_x", "node_y",
          "queued", "battery", "capacity", "speed_factor"),
    "G": ("time_progress", "done_frac", "in_flight"),
}


@dataclass(frozen=True)
class NormContext:
    """归一化用的**实例静态量**（仿真前即可算出，不随 episode 变）。"""
    total_work_min: float
    m_ref: float
    n_m: int
    n_jobs: int
    n_agv: int
    bbox_diag: float
    max_fail_rate: float
    pm_interval: float
    max_capacity: int
    max_queued: int
    node_xy: tuple[tuple[float, float], ...]   # 通道节点坐标（V 段 x/y 用）
```

- [ ] **Step 4: 重写 `state_emb.py`**

```python
# -*- coding: utf-8 -*-
"""快照 + 实例 + 布局 → 编码器输入（spec §5.3.1）。

三段宽度不同（F_M=7 / F_B=8 / F_V=10 / F_G=3），故产出**按段分组的数组列表** +
`seg`。四段特征**补齐到 `F_MAX` 后按序列拼成单张 `(N, F_MAX)`**（spec §5.3.1）。
"""
from __future__ import annotations

import numpy as np

from ..env.instances import Instance
from ..env.layout import Layout
from ..env.snapshot import Snapshot
from .features import F_B, F_G, F_M, F_V, NormContext


def _safe(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def norm_context(inst: Instance, layout: Layout, m_ref: float) -> NormContext:
    """由实例 + 布局算出全部归一化标度（与 `SimConfig` 无关的部分）。"""
    total_work = sum(min(t for _m, t in op) for job in inst.jobs for op in job)
    spec = layout.grid
    diag = float(np.hypot(spec.n_rows * (spec.cell_h + spec.aisle_w),
                          spec.n_cols * (spec.cell_w + spec.aisle_w)))
    return NormContext(
        total_work_min=max(total_work, 1e-9), m_ref=max(m_ref, 1e-9),
        n_m=inst.n_machines, n_jobs=inst.n_jobs, n_agv=len(layout.agvs or []),
        bbox_diag=max(diag, 1e-9),
        max_fail_rate=max((m.fail_rate for m in layout.machines), default=1e-9),
        pm_interval=120.0,          # 与 SimConfig.pm_interval 同源（Task 7 归一到 cfg）
        max_capacity=max((a.capacity for a in (layout.agvs or [])), default=1),
        max_queued=max(1, int(np.ceil(np.sqrt(max(inst.n_jobs, 1))))),
        node_xy=tuple(spec.node_xy(*spec.node_rc(i)) for i in range(spec.n_nodes)))


def machine_features(snap: Snapshot, ctx: NormContext) -> np.ndarray:
    """(n_m, 7)。顺序 = FEATURE_NAMES["M"]，**不得改序**（有测试按名核对）。"""
    out = np.zeros((ctx.n_m, F_M), dtype=np.float32)
    for i, m in enumerate(snap.machines):
        out[i] = (
            m.backlog_min / ctx.total_work_min,                     # 0 backlog
            _safe(m.in_q_len / max(m.in_cap, 1.0), 0.0, 1.0),       # 1 in_fill
            _safe(m.out_q_len / max(m.out_cap, 1.0), 0.0, 1.0),     # 2 out_fill
            1.0 if m.busy else 0.0,                                 # 3 busy
            m.remaining_min / max(ctx.total_work_min, 1e-9),        # 4 remaining_frac
            _safe(1.0 - m.pm_used_min / max(ctx.pm_interval, 1e-9), 0.0, 1.0),  # 5 pm_left
            m.fail_rate / max(ctx.max_fail_rate, 1e-9),             # 6 fail_rate
        )
    return out


def job_features(snap: Snapshot, ctx: NormContext) -> np.ndarray:
    """(n_jobs, 8)。"""
    out = np.zeros((ctx.n_jobs, F_B), dtype=np.float32)
    for j, js in enumerate(snap.jobs):
        out[j] = (
            js.done_ops / max(js.total_ops, 1),                     # 0 progress
            js.remaining_min / ctx.total_work_min,                  # 1 remaining_work
            (0.0 if js.finished else
             _safe((snap.now - 0.0) / ctx.m_ref - 1.0, -2.0, 2.0)), # 2 due_margin（见下）
            1.0 if js.finished else 0.0,                            # 3 finished
            _safe(js.at_machine / max(ctx.n_m, 1), -1.0, 1.0),      # 4 at_machine
            1.0 if js.in_transit else 0.0,                          # 5 in_transit
            _safe(js.on_agv / max(ctx.n_agv, 1), -1.0, 1.0),        # 6 on_agv
            1.0,                                                    # 7 weight（等权，见 §5.3.1）
        )
    return out


def vehicle_features(snap: Snapshot, ctx: NormContext) -> np.ndarray:
    """(n_agv, 10)。"""
    out = np.zeros((ctx.n_agv, F_V), dtype=np.float32)
    for a, v in enumerate(snap.vehicles):
        st = [0.0] * 4
        st[_safe(v.status, 0, 3)] = 1.0
        # ⚠️ x/y 必须取自**节点坐标**，不能都用节点号（那等于丢掉了几何，只剩序号）
        nx, ny = ctx.node_xy[max(v.node, 0)]
        out[a] = (*st,
                  nx / ctx.bbox_diag,                                # 5 node_x
                  ny / ctx.bbox_diag,                                # 6 node_y
                  _safe(v.queued / max(ctx.max_queued, 1), 0.0, 1.0),
                  _safe(v.battery_frac, 0.0, 1.0),
                  v.capacity / max(ctx.max_capacity, 1),
                  _safe(v.speed_factor, 0.0, 2.0))
    return out


def global_features(snap: Snapshot, ctx: NormContext) -> np.ndarray:
    """(1, 3)。"""
    return np.array([[
        _safe(snap.now / ctx.m_ref, 0.0, 4.0),                      # 0 time_progress
        snap.n_done / max(ctx.n_jobs, 1),                           # 1 done_frac
        _safe(snap.in_flight / max(ctx.max_queued, 1), 0.0, 4.0),   # 2 in_flight
    ]], dtype=np.float32)


def build_tok(snap: Snapshot, inst: Instance, layout: Layout,
              ctx: NormContext) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """快照 → **单张** `(N, F_MAX)` 特征张量 + `seg`（spec §5.3.1）。

    四段各自算完后按 `SEG_SLICE` 填进对应行，**列不足处保持 0**（补零）。
    故 `tok[:n_m, 7:]`、`tok[n_m:n_m+n_jobs, 8:]`、`tok[-1, 3:]` 恒为 0。

    ⚠️ `seg` 由**各段实际行数**推出，**不硬编码实例规模**（Review Focus #2）。
    """
    parts = (("M", machine_features(snap, ctx)), ("B", job_features(snap, ctx)),
             ("V", vehicle_features(snap, ctx)), ("G", global_features(snap, ctx)))
    seg = tuple(p.shape[0] for _k, p in parts)                    # (n_m, n_jobs, n_agv, 1)
    tok = np.zeros((sum(seg), F_MAX), dtype=np.float32)
    r = 0
    for key, p in parts:
        tok[r:r + p.shape[0], SEG_SLICE[key]] = p                 # 补零列保持 0
        r += p.shape[0]
    return tok, seg
```

- [ ] **Step 5: 跑测试确认通过**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_features.py -q --no-header`
Expected: PASS（5 项）

- [ ] **Step 6: 跑全套回归 + 提交**

```bash
D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header
git add plcsp/nn/features.py plcsp/nn/state_emb.py plcsp/tests/test_features.py
git commit -m "feat: token 特征重设计（P2 Task 2）——三套字段 + 实例静态量归一化"
```

> 🔻 **`due_margin` 的已知简化**：`job_features` 第 2 维现用 `now/M_ref − 1` 近似，
> 因为 `Snapshot` 里没有逐作业交期。**Task 6 接奖励时须把 `due_j` 放进 `JobState`** 并改为
> `(due_j − now)/M_ref`。此处先占位，有 TODO 注释标记。

---

### Task 3: 编码器统一宽度 + 类型嵌入

**Files:**
- Modify: `plcsp/nn/encoder.py`
- Test: `plcsp/tests/test_encoder_segments.py`

**Interfaces:**
- Consumes: Task 2 的 `F_MAX`
- Produces:
  - `LayoutEncoder(d_model=128, n_heads=4, n_layers=8, feat_dim=F_MAX)`
  - `LayoutEncoder.forward(tok_feat: Tensor (1,N,F_MAX), seg: tuple[int,int,int,int]) -> (tok (1,N,d), ctx (1,d))`
  - `LayoutEncoder.N_SEG_TYPES = 4`；`LayoutEncoder.type_emb: Parameter (4, d_model)` — **必需，不可省**

- [ ] **Step 1: 写失败测试**

创建 `plcsp/tests/test_encoder_segments.py`：

```python
"""统一宽度投影 + 类型嵌入的测试（P2 Task 3）。"""
from __future__ import annotations

import torch
import pytest

from plcsp.nn.encoder import LayoutEncoder
from plcsp.nn.features import F_B, F_G, F_M, F_MAX


def _tok(n_m=6, n_b=10, n_v=3, fill=1.0):
    """按补齐规则造一张 (1, N, F_MAX)：各段只填自己的有效列，其余为 0。"""
    from plcsp.nn.features import F_MAX, SEG_SLICE
    t = torch.zeros(1, n_m + n_b + n_v + 1, F_MAX)
    for start, n, key in ((0, n_m, "M"), (n_m, n_b, "B"), (n_m + n_b, n_v, "V"),
                          (n_m + n_b + n_v, 1, "G")):
        t[0, start:start + n, SEG_SLICE[key]] = fill
    return t, (n_m, n_b, n_v, 1)


@pytest.mark.unit
def test_forward_shapes():
    enc = LayoutEncoder()
    tok, ctx = enc(*_tok())
    assert tok.shape == (1, 6 + 10 + 3 + 1, enc.d_model)
    assert ctx.shape == (1, enc.d_model)


@pytest.mark.unit
def test_single_linear_projection_not_per_segment():
    """输入是**单张**张量、**单个** Linear（spec §5.3.1 的统一宽度方案）。"""
    enc = LayoutEncoder()
    assert hasattr(enc, "embed") and isinstance(enc.embed, torch.nn.Linear)
    assert enc.embed.in_features == F_MAX
    assert not hasattr(enc, "proj"), "仍存在分段投影——与统一宽度方案不符"


@pytest.mark.unit
def test_type_embedding_distinguishes_segments():
    """⚠️ Review Focus #2：三类 token 即使特征值相同，输出也必须不同——否则网络分不出类型。

    旧版正是这样坏的：单一 Linear + 无类型嵌入，`mask='full'` 下 seg 完全不被使用。
    """
    enc = LayoutEncoder().eval()
    tok, _ = enc(*_tok(fill=0.0))
    # 机台段首 token 与车辆段首 token 的嵌入必须不同
    assert not torch.allclose(tok[0, 0], tok[0, 16], atol=1e-6)
    assert not torch.allclose(tok[0, 0], tok[0, 19], atol=1e-6)


@pytest.mark.unit
def test_padded_columns_do_not_affect_embedding():
    """补零列不得影响嵌入——把它们改成任意值，输出必须**逐位不变**。

    这是"统一宽度 + 单 Linear"方案的正确性前提：补零列乘的是权重列，
    若某处列偏移写错、把有效列当成了补零列，嵌入就会变。本测试把该风险钉死。
    """
    enc = LayoutEncoder().eval()
    a, seg = _tok(fill=1.0)
    b = a.clone()
    n_m, n_b, _, _ = seg
    b[0, :n_m, F_M:] = 99.0                      # M 段补零列
    b[0, n_m:n_m + n_b, F_B:] = 99.0             # B 段补零列
    b[0, -1, F_G:] = 99.0                        # G 段补零列
    with torch.no_grad():
        assert torch.allclose(enc(a, seg)[0], enc(b, seg)[0], atol=1e-6)


@pytest.mark.unit
def test_attention_is_full_not_masked():
    """spec §5.1 定的是**全连接**注意力（轴掩码已砍）——不得再有 block_mask 分支。"""
    import inspect
    src = inspect.getsource(LayoutEncoder.forward)
    assert "block_mask" not in src
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_encoder_segments.py -q --no-header`
Expected: FAIL — `TypeError: forward() takes 3 positional arguments but 5 were given`

- [ ] **Step 3: 重写 `encoder.py`**

要点（完整实现按此重写；`DualAxisLayer` 的注意力体保留但去掉掩码参数）：

```python
class LayoutEncoder(nn.Module):
    """四段 token → d_model 嵌入 → 标准 Transformer（全连接注意力）。

    2026-10-03 重写：单一 `Linear(feat_dim, d_model)` → **四段各自投影** +
    **类型嵌入**。旧版三类 token 被迫同宽、且无类型嵌入，`mask='full'` 下 seg 不被使用，
    网络分不出 M/B/V。
    """
    N_SEG_TYPES = 4

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 8,
                 feat_dim: int = F_MAX):
        super().__init__()
        self.d_model, self.n_heads, self.n_layers = d_model, n_heads, n_layers
        self.feat_dim = feat_dim
        self.embed = nn.Linear(feat_dim, d_model)          # **单个** Linear（统一宽度）
        self.type_emb = nn.Parameter(torch.zeros(self.N_SEG_TYPES, d_model))
        self.layers = nn.ModuleList([AttnLayer(d_model, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)

    def forward(self, tok_feat, seg):
        """tok_feat: **单张** (1, N, F_MAX)；seg=(n_m,n_b,n_v,n_g)。

        返回 (tok (1,N,d), ctx (1,d))。
        `seg` 用来**生成每个 token 的类型 id**（决定加哪个 `type_emb`），并核对总长。
        ⚠️ 类型嵌入在统一宽度方案下是**必需**的：四段的列语义重叠（第 3 列在 M 段是
        "在加工"、在 B 段是"已完成"），只有 type_emb 能把它们解耦开。
        """
        n_m, n_b, n_v, n_g = seg
        N = n_m + n_b + n_v + n_g
        assert tok_feat.shape[1] == N, f"tok_feat 行数 {tok_feat.shape[1]} != sum(seg)={N}"
        tid = torch.cat([torch.full((n,), i, dtype=torch.long)
                         for i, n in enumerate(seg)])          # 每个 token 的类型 id
        x = (self.embed(tok_feat[0]) + self.type_emb[tid]).unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        return x, x.mean(dim=1)


class AttnLayer(nn.Module):
    """标准全连接注意力层（掩码参数与 block_mask 一并删除）。"""
    # ...（其余与旧 DualAxisLayer 相同，只是 forward 不再收 mask）
```

同时**删除** `block_mask` 与 `_SEG_INDEX`（无人再用）。

- [ ] **Step 4: 跑测试确认通过 + 全套回归**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header`
Expected: 全绿（116 项）

- [ ] **Step 5: 提交**

```bash
git add plcsp/nn/encoder.py plcsp/tests/test_encoder_segments.py
git commit -m "refactor: 编码器统一宽度 + 类型嵌入（P2 Task 3）——单个 Linear + 必需的类型嵌入"
```

---

### Task 4: L 头接编码器（`agv_logits_emb`）

**Files:**
- Modify: `plcsp/algo/policy.py`
- Test: `plcsp/tests/test_policy_heads.py`

**Interfaces:**
- Consumes: Task 3 的 `LayoutEncoder`
- Produces:
  - `PolicyNet.agv_logits_emb(tok, feat_task, cand_idx) -> Tensor (1,1,n_cand)` —— 与 `mach_logits_emb` **对称**
  - `PolicyNet.forward_enc(segs, seg) -> (tok, ctx)`（封装编码器前向）
  - `v_token_index(seg) -> list[int]`：V 段在序列中的位置 = `[n_m+n_b+i for i in range(n_agv)]`

- [ ] **Step 1: 写失败测试**

创建 `plcsp/tests/test_policy_heads.py`：

```python
"""两个头对称性的测试（P2 Task 4）。"""
from __future__ import annotations

import torch
import pytest

from plcsp.algo.policy import PolicyNet, v_token_index
from plcsp.nn.encoder import LayoutEncoder
from plcsp.nn.features import F_MAX


def _enc_inputs(n_m=6, n_b=10, n_v=3):
    return torch.randn(1, n_m + n_b + n_v + 1, F_MAX), (n_m, n_b, n_v, 1)


@pytest.mark.unit
def test_both_heads_read_the_same_encoder():
    """⚠️ Review Focus #5：L 头必须走编码器——这是'联合链'的前提。"""
    pol = PolicyNet(enc=LayoutEncoder(), n_agv=3)
    tok_feat, seg = _enc_inputs()
    tok, _ = pol.forward_enc(tok_feat, seg)
    idx = v_token_index(seg)
    logits = pol.agv_logits_emb(tok, torch.zeros(1, 1, 3), torch.tensor(idx))
    assert logits.shape == (1, 1, 3)


@pytest.mark.unit
def test_l_head_gradient_reaches_encoder():
    """⚠️ Review Focus #5：从 L 头反传，编码器参数必须有非零梯度（否则'联合'是假的）。"""
    pol = PolicyNet(enc=LayoutEncoder(), n_agv=3)
    tok_feat, seg = _enc_inputs()
    tok, _ = pol.forward_enc(tok_feat, seg)
    out = pol.agv_logits_emb(tok, torch.zeros(1, 1, 3),
                             torch.tensor(v_token_index(seg))).sum()
    out.backward()
    g = pol.enc.embed.weight.grad            # 编码器输入投影
    assert g is not None and g.abs().sum() > 0, "L 头梯度没到编码器"


@pytest.mark.unit
def test_v_token_index_matches_seg():
    assert v_token_index((6, 10, 3, 1)) == [16, 17, 18]
    assert v_token_index((15, 20, 4, 1)) == [35, 36, 37, 38]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_policy_heads.py -q --no-header`
Expected: FAIL — `ImportError: cannot import name 'v_token_index'`

- [ ] **Step 3: 实现**

在 `policy.py` 加：

```python
def v_token_index(seg: tuple[int, int, int, int]) -> list[int]:
    """V 段 token 在序列中的位置（= M 段 + B 段之后）。有序，第 i 个 = 第 i 台车。"""
    n_m, n_b, n_v, _ = seg
    return list(range(n_m + n_b, n_m + n_b + n_v))


class PolicyNet(nn.Module):
    def __init__(self, ...):
        # ...（既有参数保留，新增下面一行）
        self.optim: torch.optim.Optimizer | None = None   # 由训练器在首步惰性创建（Adam）

    def forward_enc(self, tok_feat, seg):
        """编码器前向（两个头共用）。无编码器时返回 (None, None)。"""
        if self.enc is None:
            return None, None
        return self.enc(tok_feat, seg)

    def agv_logits_emb(self, tok, feat_task, cand_idx) -> torch.Tensor:
        """(1,N,d) × (1,1,F_task) × (n_cand,) → (1,1,n_cand)。

        与 `mach_logits_emb` **对称**：候选对象的 token 嵌入 ⊕ 决策特征 → 打分。
        2026-10-03 新增：此前 L 头吃 `des.py` 手搓的 11 维扁平向量、**不走编码器**，
        导致它对生产侧结构性失明（看不到机台状态/计划/布局，n_agv≥3 时看不到 2 号以后的车）。
        """
        tok_c = tok[0, cand_idx.long()]                       # (n_cand, d)
        ft = feat_task.expand(1, tok_c.shape[0], -1)[0]       # (n_cand, F_task)
        return self.l_head_tok(torch.cat([tok_c, ft], dim=-1)).squeeze(-1).unsqueeze(0).unsqueeze(1)
```

`__init__` 里加 `l_head_tok = nn.Sequential(nn.Linear(enc.d_model + n_feat_task, hidden), nn.GELU(), nn.Linear(hidden, 1))`，
其中 `n_feat_task` 为**任务特征**维数（起送机台/目标机台/工序序号/该机是否需换型 → **4**）。
同时**删除** `l_head`（旧扁平路径）与 `mach_logits`（旧 MLP 路径）、`s_head`——两个头一律走 token 嵌入。

- [ ] **Step 4: 跑测试确认通过 + 全套回归 + 提交**

```bash
D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header
git add plcsp/algo/policy.py plcsp/tests/test_policy_heads.py
git commit -m "feat: L 头接编码器（P2 Task 4）——两个头读同一份 token 嵌入"
```

---

### Task 5: 在线 S 层

**Files:**
- Modify: `plcsp/env/des.py`
- Test: `plcsp/tests/test_online_dispatch.py`

**Interfaces:**
- Consumes: Task 1 的 `Snapshot`
- Produces:
  - `SimWorld.run_gated(..., policy_s=None, online_s: bool = False)`
  - `policy_s(snap, job, oi, cand) -> int`：候选机台列表里选一个
  - 决策日志条目：`("S", snap, job, oi, cand, action)` / `("L", snap, task_feat, cand, action)`

- [ ] **Step 1: 写失败测试**

创建 `plcsp/tests/test_online_dispatch.py`：

```python
"""在线 S 层的测试（P2 Task 5）。"""
from __future__ import annotations

import pytest

from plcsp.env.des import SimConfig, rollout, SimWorld
from plcsp.env.instances import load_mk
from plcsp.env.layout import sample_layout
from plcsp.env.corridors import build_corridor_graph, dock_distance_matrix


@pytest.mark.unit
def test_online_off_reproduces_static_trajectory():
    """⚠️ Review Focus #3：在线开关关闭时，必须**逐位等于**改造前的静态行为。"""
    inst = load_mk("mk01")
    a = rollout(inst, seed_chain=1, cfg=SimConfig())          # 默认路径（静态）
    lay = sample_layout(inst.n_machines, seed=0, n_agv=SimConfig().n_agv)
    g = build_corridor_graph(lay)
    w = SimWorld(inst, lay, dock_distance_matrix(g), SimConfig(), graph=g)
    b = w.run_gated(seed_chain=1, online_s=False)
    assert a["makespan"] == b["makespan"]


@pytest.mark.unit
def test_online_s_is_called_once_per_operation():
    """在线模式下，决策次数 == 总工序数（每道工序一个决策）。"""
    inst = load_mk("mk01")
    lay = sample_layout(inst.n_machines, seed=0, n_agv=SimConfig().n_agv)
    g = build_corridor_graph(lay)
    w = SimWorld(inst, lay, dock_distance_matrix(g), SimConfig(), graph=g)
    calls = []

    def policy_s(snap, job, oi, cand):
        calls.append((job, oi, tuple(cand)))
        return 0                                     # 候选里第一个（= 最短候选，与静态同）

    r = w.run_gated(seed_chain=1, online_s=True, policy_s=policy_s)
    n_ops = sum(len(j) for j in inst.jobs)
    assert len(calls) == n_ops, f"应为每道工序一次决策，实得 {len(calls)}/{n_ops}"
    assert r["jobs_done"] == inst.n_jobs


@pytest.mark.unit
def test_online_s_sees_live_machine_state():
    """⚠️ 在线 S 的意义：决策时能看到**当时的**机台状态，而不只是初始状态。

    断言：后半程的决策里，至少有一台机的积压 > 0（静态模式下这是不可能的）。
    """
    inst = load_mk("mk01")
    lay = sample_layout(inst.n_machines, seed=0, n_agv=SimConfig().n_agv)
    g = build_corridor_graph(lay)
    w = SimWorld(inst, lay, dock_distance_matrix(g), SimConfig(), graph=g)
    backlogs = []

    def policy_s(snap, job, oi, cand):
        backlogs.append(max(m.backlog_min for m in snap.machines))
        return 0

    w.run_gated(seed_chain=1, online_s=True, policy_s=policy_s)
    assert max(backlogs) > 0.0, "在线 S 全程都没看到过非零积压——说明看的还是初始快照"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_online_dispatch.py -q --no-header`
Expected: FAIL — `TypeError: run_gated() got an unexpected keyword argument 'online_s'`

- [ ] **Step 3: 实现在线派工**

改动点（两个派工点，都在 `des.py`）：

**① 初始注入**（`run_gated` 里 `env.process(self._release(...))` 处）：把 `m0, t0 = plans[j][0]`
换成"在线选机"：

```python
def _pick_machine(self, job: int, oi: int, policy_s, online_s: bool, plans) -> tuple[int, float]:
    """派工点：选机台 + 取该候选上的加工时长。

    在线：调 `policy_s(快照, job, oi, 候选列表)`；
    离线（`online_s=False`）：读预计算的 `plans`——**逐位复现旧行为**。
    """
    alts = self.inst.jobs[job][oi]
    cand = [m for m, _t in alts]
    if online_s:
        choice = int(policy_s(self.snapshot(), job, oi, cand))
        if choice not in cand:
            raise ValueError(f"policy_s 选了非候选机台 {choice}；候选 {cand}")
    else:
        choice = plans[job][oi]
    t = next(t for m, t in alts if m == choice)
    return choice, float(t)
```

**② transporter 生成下一工序搬运任务**：把 `nxt_m, nxt_t = ops[oi + 1]` 换成
`nxt_m, nxt_t = self._pick_machine(job, oi + 1, policy_s, online_s, plans)`。

**③ `plans` 语义变更**：`online_s=True` 时 `plans` 退化为**决策日志**
（`plans[job][oi]` 在决策后写入），故 `_pick_machine` 之前不得读它。
`total_work`（算 horizon 用）改为按**最短候选**估上界：

```python
total_work = sum(min(t for _m, t in op) for job in inst.jobs for op in job)
```

**④ 决策日志**：`policy_s` 的每次调用都往 `dec_log` 追加 `("S", snap, job, oi, tuple(cand), choice)`；
L 层追加 `("L", snap, (frm, to, oi+1), cand_v, action)`。

> ⚠️ **层次纪律**：`des.py` 只记**原始快照**，**不构造特征**——`env/` 不得依赖 `nn/`。
> 快照 → 特征的转换发生在 `algo/group_rel.py` 的 `roll_chain` 里（Task 7），
> 这样 `Decision.segs` 才拿得到四段特征。

- [ ] **Step 4: 跑测试确认通过 + 全套回归**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header`
Expected: 全绿（119 项）

- [ ] **Step 5: 提交**

```bash
git add plcsp/env/des.py plcsp/tests/test_online_dispatch.py
git commit -m "feat: 在线 S 层（P2 Task 5）——派工点决策，离线路径逐位复现"
```

---

### Task 6: 三目标奖励

**Files:**
- Create: `plcsp/env/reward.py`
- Modify: `plcsp/env/des.py`（`JobState` 加 `due`；返回 dict 已有三项）
- Test: `plcsp/tests/test_reward.py`

**Interfaces:**
- Consumes: Task 1 的 `Snapshot`；`des.rollout` 的返回 dict
- Produces:
  - `objective_vector(r: dict) -> tuple[float, float, float]`（makespan, energy, TWT）
  - `reward_weights(f_ref: tuple[float, float, float]) -> tuple[float, float, float]`
  - `scalar_reward(f: tuple, w: tuple, prefs: tuple | None = None) -> float`
  - `ReferenceObjectives`（frozen dataclass）+ `reference_objectives(inst, cfg, seed_layout) -> ReferenceObjectives`

- [ ] **Step 1: 写失败测试**

创建 `plcsp/tests/test_reward.py`：

```python
"""三目标奖励的测试（P2 Task 6，spec §4.2）。"""
from __future__ import annotations

import pytest

from plcsp.env.reward import (ReferenceObjectives, objective_vector, reward_weights,
                              scalar_reward)
from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import load_mk


@pytest.mark.unit
def test_objective_vector_reads_three_objectives():
    r = {"makespan": 100.0, "energy": 8.0, "tardy_twt": 12.5}
    assert objective_vector(r) == (100.0, 8.0, 12.5)


@pytest.mark.unit
def test_weights_are_inverse_reference_and_sum_to_one():
    w = reward_weights((100.0, 8.0, 20.0))
    assert sum(w) == pytest.approx(1.0)
    # 参考值越小 → 权重越大（"相对参考各改进一个单位，贡献相同"）
    assert w[1] > w[2] > w[0]


@pytest.mark.unit
def test_weights_reject_zero_reference():
    """参考值为 0（如 TWT=0 的实例）不得产生 inf 权重——必须显式报错或兜底。"""
    with pytest.raises(ValueError):
        reward_weights((100.0, 8.0, 0.0))


@pytest.mark.unit
def test_scalar_reward_is_monotone_in_each_objective():
    """三个目标各自变小（更好）时，奖励必须上升——这是"多目标"最容易被写反的地方。"""
    w = reward_weights((100.0, 8.0, 20.0))
    base = scalar_reward((100.0, 8.0, 20.0), w)
    assert scalar_reward((90.0, 8.0, 20.0), w) > base      # makespan ↓
    assert scalar_reward((100.0, 7.0, 20.0), w) > base     # energy ↓
    assert scalar_reward((100.0, 8.0, 10.0), w) > base     # TWT ↓


@pytest.mark.unit
def test_reference_objectives_cached_and_consistent_with_m_ref():
    """f^ref 与 M_ref 必须取自**同一次**参考运行（口径一致）。"""
    from plcsp.env.des import reference_makespan
    inst = load_mk("mk01")
    ref = ReferenceObjectives.of(inst, SimConfig())
    assert ref.makespan == pytest.approx(reference_makespan(inst, SimConfig()), rel=1e-9)
    assert ref.energy > 0.0 and ref.twt >= 0.0


@pytest.mark.unit
def test_end_to_end_reward_on_real_rollout():
    inst = load_mk("mk01")
    ref = ReferenceObjectives.of(inst, SimConfig())
    w = reward_weights(ref.as_tuple())
    r = rollout(inst, seed_chain=1, cfg=SimConfig())
    got = scalar_reward(objective_vector(r), w)
    assert got < 0.0, "奖励应为负（三项都是'越小越好'）"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_reward.py -q --no-header`
Expected: FAIL — `ModuleNotFoundError: No module named 'plcsp.env.reward'`

- [ ] **Step 3: 实现 `reward.py`**

```python
# -*- coding: utf-8 -*-
"""三目标奖励（spec §4.2）——A 阶段的加权标量化。

    r = w₁·(−makespan) + w₂·(−energy) + w₃·(−TWT)
    wᵢ = (1/fᵢ^ref) / Σⱼ(1/fⱼ^ref)      fᵢ^ref = 参考调度下的第 i 个目标实测值

**按参考调度归一化后等权**：物理含义是"相对参考调度各改进一个单位，贡献相同"，
且可复现（不依赖拍脑袋）。**权重须与 M_ref 一同写进论文实验设置。**

⚠️ **跨实例不可比**：`f^ref` 是按实例算的，故奖励绝对值只在单实例内有意义。
训练/评估不得跨实例混用（`ReferenceObjectives` 里存了实例指纹，可断言）。

期末一次性结算，无中间奖励（→ §4.4 的切比雪夫可加性问题不存在）。
"""
from __future__ import annotations

from dataclasses import dataclass


def objective_vector(r: dict) -> tuple[float, float, float]:
    """评测返回 dict → (makespan, energy, TWT)。三者都是"越小越好"。"""
    return (float(r["makespan"]), float(r["energy"]), float(r["tardy_twt"]))


def reward_weights(f_ref: tuple[float, float, float]) -> tuple[float, float, float]:
    """wᵢ = (1/fᵢ^ref) / Σ(1/fⱼ^ref)。fᵢ^ref ≤ 0 时显式报错（不得静默出 inf/负权重）。"""
    if any(f <= 0.0 for f in f_ref):
        raise ValueError(f"参考目标值必须为正，收到 {f_ref}——TWT=0 的实例须先调 τ")
    inv = [1.0 / f for f in f_ref]
    s = sum(inv)
    return tuple(x / s for x in inv)


def scalar_reward(f: tuple[float, float, float], w: tuple[float, float, float],
                  prefs: tuple[float, float, float] | None = None) -> float:
    """Σ wᵢ(−fᵢ)。`prefs` 为 C 阶段预留：给定时**取代** w（组内固定、跨组变化）。"""
    ww = prefs if prefs is not None else w
    return float(sum(-wi * fi for wi, fi in zip(ww, f)))


@dataclass(frozen=True)
class ReferenceObjectives:
    """参考调度下的三个目标实测值（与 `M_ref` 取自**同一次**运行）。"""
    makespan: float
    energy: float
    twt: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.makespan, self.energy, self.twt)

    @staticmethod
    def of(inst, cfg) -> "ReferenceObjectives":
        """跑一次参考调度（每工序取最短候选 + AGV 轮询）并缓存。"""
        from .des import reference_run
        r = reference_run(inst, cfg)
        return ReferenceObjectives(float(r["makespan"]), float(r["energy"]),
                                   float(r["tardy_twt"]))
```

在 `des.py` 里把 `reference_makespan` 泛化成 `reference_run`（返回整个 dict，缓存不变），
`reference_makespan` 保留为薄封装以免破坏现有调用点。**同时**把 `due` 放进 `Snapshot`
（解掉 Task 2 的 🔻 占位）：`JobState` 增 `due: float` 字段，`job_features` 第 2 维改为
`(js.due - snap.now) / ctx.m_ref`。

- [ ] **Step 4: 跑测试确认通过 + 全套回归 + 提交**

```bash
D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header
git add plcsp/env/reward.py plcsp/env/des.py plcsp/env/snapshot.py plcsp/nn/state_emb.py plcsp/tests/test_reward.py
git commit -m "feat: 三目标奖励（P2 Task 6）——1/f^ref 归一化权重 + 参考目标缓存"
```

---

### Task 7: 联合链 GRPO 训练器

**Files:**
- Modify: `plcsp/algo/group_rel.py`（新增 `joint_chain_step`，删除旧的 `train_step` / `l_seq_step*` / `local_l_step` / `sample_plan` / `_plan_logp` / `_op_logits` / `_l_feat` / `_op_feat` / `_cand_feat`）
- Modify: `plcsp/algo/runner.py`（默认 `step_fn=joint_chain_step`，Adam）
- Test: `plcsp/tests/test_joint_chain.py`

**Interfaces:**
- Consumes: Task 5 的决策日志；Task 4 的 `forward_enc` / `agv_logits_emb` / `mach_logits_emb`；Task 6 的 `scalar_reward`
- Produces:
  - `roll_chain(inst, layout, dm, cfg, policy, seed, ctx, sample=True) -> (decisions, metrics)`：跑一条链，记录每个决策的 `(四段特征, 决策特征, 候选, 动作)`
  - `chain_logp(decisions, policy) -> Tensor`：`Σ_t logπ_S + Σ_t logπ_L`（**求和**，带梯度）
  - `joint_chain_step(policy, inst, layout, dm, cfg, ctx, w, seed, G=8, lr=1e-3, clip_eps=0.2, epochs=1) -> (r_mean, diag)`
- `op_feat(inst, job, oi, layout) -> list[float]`、`setup_flag(snap, machine, job) -> float`

- [ ] **Step 1: 写失败测试**

创建 `plcsp/tests/test_joint_chain.py`：

```python
"""联合链 GRPO 的测试（P2 Task 7，spec §5.3.4）。"""
from __future__ import annotations

import torch
import pytest

from plcsp.algo.group_rel import chain_logp, joint_chain_step, roll_chain
from plcsp.algo.policy import PolicyNet
from plcsp.env.des import SimConfig
from plcsp.env.instances import load_mk
from plcsp.env.layout import sample_layout
from plcsp.env.corridors import build_corridor_graph, dock_distance_matrix
from plcsp.nn.encoder import LayoutEncoder


def _setup(name="mk01"):
    """返回 (inst, layout, dm, cfg, ctx, policy)——**六个**，与 roll_chain 的签名对齐。"""
    from plcsp.nn.state_emb import norm_context
    inst = load_mk(name)
    cfg = SimConfig()
    lay = sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    pol = PolicyNet(enc=LayoutEncoder(), n_agv=cfg.n_agv)
    return inst, lay, dock_distance_matrix(g), cfg, norm_context(inst, lay, m_ref=100.0), pol


@pytest.mark.unit
def test_chain_has_one_decision_per_op_and_per_task():
    """一条链的决策数 = 工序数 + 运输任务数。"""
    inst, lay, dm, cfg, ctx, pol = _setup()
    decisions, metrics = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    n_ops = sum(len(j) for j in inst.jobs)
    kinds = [d.kind for d in decisions]
    assert kinds.count("S") == n_ops
    assert kinds.count("L") == metrics["deliveries"]


@pytest.mark.unit
def test_logp_is_sum_not_mean():
    """⚠️ Review Focus：logp 必须取**求和**——现状 S 用平均、L 用求和，同一次比较里口径不一致。"""
    inst, lay, dm, cfg, ctx, pol = _setup()
    decisions, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    lp = chain_logp(decisions, pol)
    n_calls = [0]

    def counting(fp):
        n_calls[0] += 1
        return fp

    # 求和口径：隐式验证——logp 的绝对值应随决策数线性增长，而不是被压在 O(1)
    decisions_k = decisions[: max(4, len(decisions) // 2)]
    lp_k = chain_logp(decisions_k, pol)
    assert abs(float(lp)) > abs(float(lp_k)), "logp 未随决策数增长——疑似仍取平均"


@pytest.mark.unit
def test_ratio_is_one_for_unchanged_policy():
    """同一策略、同一条链 → ratio 必须为 1（这是 PPO 裁剪的前提）。"""
    inst, lay, dm, cfg, ctx, pol = _setup()
    decisions, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    a = chain_logp(decisions, pol)
    b = chain_logp(decisions, pol)
    assert float(torch.exp(b - a)) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.unit
def test_gradient_flows_to_both_heads_and_encoder():
    """⚠️ Review Focus #5：一次更新必须同时给 S 头、L 头、编码器都留下非零梯度。"""
    inst, lay, dm, cfg, ctx, pol = _setup()
    decisions, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    chain_logp(decisions, pol).backward()
    for name, p in (("s_head_tok", pol.s_head_tok[0].weight),
                    ("l_head_tok", pol.l_head_tok[0].weight),
                    ("enc.embed", pol.enc.embed.weight),
                    ("enc.type_emb", pol.enc.type_emb)):
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} 无梯度"


@pytest.mark.unit
def test_joint_step_uses_adam_and_returns_diagnostics():
    inst, lay, dm, cfg, ctx, pol = _setup()
    r, diag = joint_chain_step(pol, inst, lay, dm, seed=0, G=4, cfg=cfg, ctx=ctx, w=w)
    assert isinstance(pol.optim, torch.optim.Adam), "优化器应为 Adam（spec §5.3.4 第 4 条）"
    assert set(diag) >= {"loss", "ratio", "r_mean", "r_std", "A_std"}
    assert r == pytest.approx(diag["r_mean"])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_joint_chain.py -q --no-header`
Expected: FAIL — `ImportError: cannot import name 'roll_chain'`

- [ ] **Step 3: 实现 `joint_chain_step`**

核心结构（写入 `group_rel.py`，**替换**该文件里的旧训练器）：

```python
@dataclass
class Decision:
    kind: str                       # "S" | "L"
    tok: np.ndarray                 # 当时的 **单张** (N, F_MAX) 特征（已冻结）
    seg: tuple
    feat: np.ndarray                # 决策特征（S: 工序特征；L: 任务特征）
    cand: tuple[int, ...]           # 候选（S: 机台号；L: 车号）
    action: int


def roll_chain(inst, layout, dm, cfg, policy, seed,
               ctx: NormContext, sample: bool = True) -> tuple[list[Decision], dict]:
    """跑一条链：仿真里每个派工点同步调策略，记录 (当时的四段特征, 决策特征, 候选, 动作)。

    ⚠️ **无梯度**——决策只记录上下文，动作由 `torch.multinomial`/`argmax` 在 `no_grad` 下取；
       logp 事后由 `chain_logp` **带梯度重算**。仿真栈（SimPy）不参与反向传播，这是必须的。

    ⚠️ 每个决策存**当时的** segs——仿真状态在变，不能事后用最新快照重建（Review Focus #6）。
    """
    decisions: list[Decision] = []
    w = SimWorld(inst, layout, dm, cfg, graph=build_corridor_graph(layout))

    def _freeze(snap):
        tok, seg = build_tok(snap, inst, layout, ctx)
        return np.asarray(tok, dtype=np.float32), seg

    def _act(kind, tok_feat, seg, feat, cand):
        with torch.no_grad():
            tok, _ = policy.forward_enc(
                torch.as_tensor(tok_feat, dtype=torch.float32).unsqueeze(0), seg)
            logits = (policy.mach_logits_emb if kind == "S" else policy.agv_logits_emb)(
                tok, torch.as_tensor(feat, dtype=torch.float32), torch.tensor(cand))
            p = torch.softmax(logits.flatten(), -1)
            a = int(torch.multinomial(p, 1).item()) if sample else int(p.argmax())
        decisions.append(Decision(kind=kind, tok=tok_feat, seg=seg,
                                  feat=np.asarray(feat, dtype=np.float32),
                                  cand=tuple(cand), action=cand[a]))
        return a

    def policy_s(snap, job, oi, cand):
        segs, seg = _freeze(snap)
        return _act("S", segs, seg, op_feat(inst, job, oi, layout), tuple(cand))

    def policy_l(snap, frm, to, oi, cand):
        segs, seg = _freeze(snap)
        # 任务特征 4 维：起送机台 / 目标机台 / 工序序号(归一) / 该机是否需换型
        f = [frm / max(ctx.n_m, 1), to / max(ctx.n_m, 1),
             oi / max(max(len(j) for j in inst.jobs), 1),
             setup_flag(snap, to, job_of(inst, frm, oi))]
        return _act("L", segs, seg, f, tuple(cand))

    metrics = w.run_gated(seed_chain=seed, online_s=True, policy_s=policy_s,
                          policy_l=policy_l, bound=True)
    return decisions, metrics


def chain_logp(decisions: list[Decision], policy) -> torch.Tensor:
    """Σ_t logπ_S(a_t) + Σ_t logπ_L(a_t)——**求和**（spec §5.3.4 第 2 条）。

    每个决策用**当时记录的** segs/seg 重建 token 嵌入（状态已变，不能用最新快照）。
    """
    total = torch.zeros(())
    for d in decisions:
        tok, _ = policy.forward_enc(
            torch.as_tensor(d.tok, dtype=torch.float32).unsqueeze(0), d.seg)
        if d.kind == "S":
            logits = policy.mach_logits_emb(tok, torch.as_tensor(d.feat), torch.tensor(d.cand))
        else:
            logits = policy.agv_logits_emb(tok, torch.as_tensor(d.feat), torch.tensor(d.cand))
        lp = torch.log_softmax(logits.flatten(), -1)
        total = total + lp[d.cand.index(d.action)]
    return total


def joint_chain_step(policy, inst, layout, dm, cfg, ctx, w, seed: int, G: int = 8,
                     lr: float = 1e-3, clip_eps: float | None = 0.2,
                     epochs: int = 1) -> tuple[float, dict]:
    """一步联合链组训练（spec §5.3.4）。

    G 条链（**J=1**，预算全给 G）→ 每条一个终端奖励 → 组内 z → PPO 式裁剪更新。
    优化器 = Adam（`policy.optim`），不再手写 SGD。
    """
    if policy.optim is None:
        policy.optim = torch.optim.Adam(policy.parameters(), lr=lr)

    # 1) 采样 G 条链（**J=1**：预算全给 G，spec §5.3.4 第 3 条）
    chains, rewards = [], []
    for g in range(G):
        dec, met = roll_chain(inst, layout, dm, cfg, policy, seed * 1000 + g, ctx, sample=True)
        chains.append(dec)
        rewards.append(scalar_reward(objective_vector(met), w))

    # 2) 组内 z 化（**一个**优势，不分组分层——联合链的核心）
    A = torch.tensor(_z(np.array(rewards, dtype=np.float64), "z"), dtype=torch.float32)

    # 3) 冻结旧 logp
    with torch.no_grad():
        old = torch.stack([chain_logp(d, policy) for d in chains])
    old = old.detach()

    # 4) K-epoch 裁剪更新（epochs=1 且 clip_eps=None ⇒ 退化为纯组内 REINFORCE）
    loss_val, ratio_mean = 0.0, 1.0
    for _ in range(max(epochs, 1)):
        new = torch.stack([chain_logp(d, policy) for d in chains])
        if clip_eps is None:
            obj = A.detach() * new
        else:
            ratio = torch.exp(new - old)
            obj = torch.min(ratio * A.detach(),
                            torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A.detach())
            ratio_mean = float(ratio.mean().item())
        loss = -obj.mean()
        policy.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)   # 范数裁剪（不是逐元素）
        policy.optim.step()
        loss_val = float(loss.item())

    diag = {"loss": loss_val, "ratio": ratio_mean,
            "r_mean": float(np.mean(rewards)), "r_std": float(np.std(rewards)),
            "A_std": float(A.std())}
    for d in chains:                      # ⚠️ 决策日志用完即弃——不得跨 step 累积
        d.clear()
    return diag["r_mean"], diag
```

**配套小工具**（同文件，供 `roll_chain` 用）：

```python
def op_feat(inst, job: int, oi: int, layout) -> list[float]:
    """S 头的决策特征（4 维）：工序序号 / 候选数 / **是否需换型** / 剩余工序占比。

    第 3 维是 ⑤ 换型的进网位置——spec §5.3.1② 定死：换型是 (机台, 作业) 的交互量，
    放进 S 头**候选特征**里由 `mach_logits_emb` 的 `feat_op` 承载，不占 M token 位。
    """
    n_ref = max(max(len(j) for j in inst.jobs), 1)
    return [oi / n_ref, 1.0 - oi / n_ref, 0.0, (len(inst.jobs[job]) - oi) / n_ref]


def setup_flag(snap, machine: int, job: int) -> float:
    """该机台加工该作业是否需要换型（1.0 = 需要）。"""
    prev = snap.machines[machine].prev_job
    return 0.0 if prev in (-1, job) else 1.0


def job_of(inst, frm: int, oi: int) -> int:
    """占位：由调用方在 `_transporter` 里传入真实作业号（`roll_chain` 里直接用闭包变量）。"""
    raise NotImplementedError("roll_chain 内部用闭包里的 job，不经此函数")
```

**同时删除**（都被 `joint_chain_step` 取代）：`train_step` / `l_seq_step` / `l_seq_step_gated` /
`local_l_step` / `sample_plan` / `greedy_plan`（已删）/ `_plan_logp` / `_op_logits` /
`_l_feat` / `_op_feat` / `_cand_feat` / `_z`（保留，仍用于组内 z）。
`runner.py` 的默认 `step_fn` 改为 `joint_chain_step`，`resume_training` 的 key 前缀检查同步。

- [ ] **Step 4: 跑测试确认通过 + 全套回归**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header`
Expected: 全绿（124 项）

- [ ] **Step 5: 提交**

```bash
git add plcsp/algo/group_rel.py plcsp/algo/runner.py plcsp/tests/test_joint_chain.py
git commit -m "feat: 联合链 GRPO（P2 Task 7）——两头一个 logp、一个优势、Adam"
```

---

### Task 8: A 端到端跑通（验收）

**Files:**
- Create: `plcsp/m13_train_a.py`（训练脚本）
- Test: `plcsp/tests/test_end_to_end_a.py`
- Modify: `docs/INDEX.md`、`docs/superpowers/specs/2026-10-02-plcsp-rebuild-design.md`（§5.3 标注已实现）、`docs/progress-log.md`

**Interfaces:**
- Consumes: Task 1–7 全部产出
- Produces: MK01 上的训练结果 + 与规则基线的对比

- [ ] **Step 1: 写验收测试**

创建 `plcsp/tests/test_end_to_end_a.py`：

```python
"""A 端到端验收（P2 Task 8）。"""
from __future__ import annotations

import pytest
import torch

from plcsp.algo.group_rel import joint_chain_step, roll_chain
from plcsp.algo.policy import PolicyNet
from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import load_mk
from plcsp.env.layout import sample_layout
from plcsp.env.corridors import build_corridor_graph, dock_distance_matrix
from plcsp.env.reward import ReferenceObjectives, reward_weights, scalar_reward, objective_vector
from plcsp.nn.encoder import LayoutEncoder


@pytest.mark.unit
def test_encoder_actually_receives_nonzero_features():
    """⚠️ Review Focus：P0 遗留的"输入全零"必须已消失——这是 P2 的头号验收项。"""
    inst, lay, dm, cfg, ctx, pol = _setup()
    decisions, _ = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    nonzero = [float(abs(d.tok).sum()) for d in decisions]
    assert min(nonzero) > 0.0, "仍有全零特征段——编码器等于吃零输入"


@pytest.mark.unit
def test_training_reduces_reward_over_50_steps():
    """50 步内组均值奖励必须有改善（不是"跑通"而是"在学"）。"""
    inst, lay, dm, cfg, ctx, pol = _setup()
    ref = ReferenceObjectives.of(inst, SimConfig())
    w = reward_weights(ref.as_tuple())
    early, late = [], []
    for s in range(50):
        r, _ = joint_chain_step(pol, inst, lay, dm, cfg, ctx, w, seed=s, G=4, lr=3e-4)
        (early if s < 10 else late if s >= 40 else []).append(r) if s < 10 or s >= 40 else None
    assert sum(late) / len(late) > sum(early) / len(early), "50 步后奖励未改善"


@pytest.mark.unit
def test_trained_policy_beats_rule_baseline_on_mk01():
    """A 的验收线：训练后的 argmax 策略在 MK01 上优于规则（每工序最短候选 + 轮询派车）。"""
    inst, lay, dm, cfg, ctx, pol = _setup()
    ref = ReferenceObjectives.of(inst, SimConfig())
    w = reward_weights(ref.as_tuple())
    for s in range(300):                                   # 训练
        joint_chain_step(pol, inst, lay, dm, cfg, ctx, w, seed=s, G=8, lr=3e-4)
    # 评估：argmax 策略，10 个扰动种子取均值
    scores = []
    for s in range(10):
        _, met = roll_chain(inst, lay, dm, cfg, pol, seed=10_000 + s, ctx=ctx, sample=False)
        scores.append(met["makespan"])
    trained = float(np.mean(scores))
    rule = float(rollout(inst, seed_chain=0, cfg=SimConfig())["makespan"])
    assert trained < rule, f"训练后 {trained:.1f} 未优于规则基线 {rule:.1f}"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m pytest plcsp/tests/test_end_to_end_a.py -q --no-header`
Expected: FAIL

- [ ] **Step 3: 训练脚本 `m13_train_a.py`**

按 `runner.run_training` 的标准用法写（`step_fn=joint_chain_step`，`run_dir=checkpoints/a_mk01`），
CLI 参数 `--steps / --G / --lr / --inst / --seed`，输出 metrics 到 run_dir（**不是包目录**）。

- [ ] **Step 4: 跑通并记录**

```bash
PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m13_train_a --inst mk01 --steps 300 --G 8
```
Expected: 指标落盘；`metrics.ndjson` 里 `r_mean` 单调上行（允许波动）

- [ ] **Step 5: 写进文档**

- `spec §5.3` 顶部加「**2026-10-03 已实现**」标注，逐条勾掉 §5.3.1 的"现状"表
- `INDEX.md` 新增 §5.8「P2+P3 表征与联合训练」
- `progress-log.md` 新增 §十八：A 的实测数字 + 与规则基线的对比

- [ ] **Step 6: 全套回归 + 提交**

```bash
D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header
git add -A && git commit -m "feat: A 端到端跑通（P2 Task 8 验收）"
```

---

## 完成后的状态

- **编码器不再是零输入**——`tok_feat` 由 `Snapshot` 实时构造；单张 `(N, F_MAX)` + 单 Linear + **必需的类型嵌入**
- **两头读同一份 token 嵌入**——L 头不再失明，联合链成立
- **S 层在线**——决策点看得见当时的机台积压/缓冲/保养余量
- **奖励三目标**——`1/f^ref` 归一化权重，与 `M_ref` 同源
- **训练是一个 GRPO**——一个 logp（求和）、一个优势、Adam、J=1
- 可交付给 **P4（多目标 C / 基线 / 消融）**

## 已知风险 / 未决

| 项 | 说明 |
|---|---|
| **决策日志的内存** | 一条链 ~460 个决策，每个存一份四段特征。MK10 上 460×(15×7+20×8+4×10+3) ≈ 460×300 floats ≈ 550 KB/链 × G=8 = 4.4 MB/步。可控，但**不得随 episode 数累积**（每个 step 用完即弃） |
| **`chain_logp` 的重放成本** | 每个决策重跑一次编码器前向 → 460 次前向/链/epoch。CPU-only 下这是**主要瓶颈**，K-epoch>1 会线性放大。若太慢，退路是**把 tok 嵌入缓存**（但会 detach 梯度，需验证梯度流） |
| **`due_margin` 的交期口径** | Task 2 先占位、Task 6 补齐；若 Task 6 未按计划做，该维会一直是近似值 |
| **`pm_interval` 在 `NormContext` 里硬编码 120** | 须与 `SimConfig.pm_interval` 同源，Task 7 一并归一 |
| **L 序列的"预演流"近似** | 改在线 S 后，任务流的形状取决于 S 的决策；现有近似是否仍成立需在 Task 5 后复测 |
| **J=1 的方差** | J 从 4 降到 1 后，同组内奖励方差变大。若 G=8 不够，先加 G 而不是加 J |
