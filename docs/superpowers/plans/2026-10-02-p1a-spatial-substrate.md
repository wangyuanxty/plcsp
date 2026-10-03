# P1a 空间基底 实施计划（网格布局 · 格点走廊 · 路径级 zone 申请）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把车间的空间模型从"一维单环走廊 + 任意 zone 分配"换成**二维网格布局 + 格点走廊 + 沿实际路径的逐段 zone 申请**，使拥堵（约束①）与充电（约束⑪）建立在真实几何之上。

**Architecture:** 三层单向依赖：`layout.py`（网格几何：机位/通道节点/充电桩）→ `corridors.py`（格点图 + 最短路）→ `des.py`（AGV 沿路径逐段申请 zone）。每层都有独立的可测接口，上层只依赖下层的公开签名。

**Tech Stack:** Python 3.12.4（`D:/anaconda/python.exe`）｜networkx 3.6.1（已有）｜numpy 1.26.4｜simpy 4.1.2｜pytest 7.4.4｜Windows 11

**Spec:** `docs/superpowers/specs/2026-10-02-plcsp-rebuild-design.md`（**§3.2 布局**为本文档的权威依据；§3.3 约束①；§10.2 机制候选）

## Global Constraints

- **项目根** = `D:esearch\DeepReinforcementLearningScheduling`。**根目录不留临时产物**（`*.log` / `*.jsonl` / `figs/` / 提取文本 / 渲染图一律进系统临时目录，用完即删）——正式文档放哪里不受限。
- **Python 解释器**一律用 `D:/anaconda/python.exe`。**CPU-only，torch 2.14.0+cpu**。
- **单位约定（bug#13 已定，不得改动）**：**仿真时间单位 = 分钟，布局坐标单位 = 米**。运输时间 = `距离[m] / (窄道倍率 × agv_speed_mps) / 60`。`SECONDS_PER_MIN = 60.0` 已定义于 `plcsp/env/des.py`。
- **回归门禁**：`plcsp/tests/` 的 **49 项必须始终全绿**。其中 `test_instances.py`（42 项）锁定官方 Brandimarte 实例，**任何任务都不得修改其断言**。
- **夹具自包含**：实例读 `plcsp/data/brandimarte/*.fjs`（随包），**不得回退到 `third_party/`**。
- **旧地图必须消失**：`line` / `U` / `island` 三种布局类型与"单环走廊"是上一代设计，本计划完成后**不得残留**。
- **提交信息格式**：`<type>: <description>`，type ∈ {feat, fix, refactor, docs, test, chore, perf, ci}。**不添加任何署名/生成标识**。

## Review Focus

以下五类是计划的测试覆盖不到、但最容易出事的：

1. **格点图上"任意两节点不可达"**。机器数少于格位数时会出现空格；若某台机的 dock 落在被围死的角落，最短路会抛 `NetworkXNoPath`。期望行为：**所有机台 dock 两两可达**，且 `sample_layout` 后即可验证，不必等到仿真。
2. **逐段 zone 申请在长路径上死锁**。改成"持当前、申请下一个"后，等待图会出现真实环（经典 AGV 死锁）。期望行为：`ZoneManager` 的环检测**能在授权前拒绝**（返回 `cycle=True`），AGV 退避重试，**整个 episode 仍能完成全部作业**（`jobs_done == n_jobs`）。
3. **释放顺序错误导致 zone 泄漏**。逐段前进时必须释放上一个持有的节点，否则一次 episode 后所有节点的 holder 都不为空，下一次 rollout 直接卡死。期望行为：episode 结束时 `ZoneManager.holder` **全部为 None**。
4. **距离矩阵与逐段路径不一致**。若 `dock_distance_matrix` 用最短路、而 AGV 实际逐段走的却是另一条路，拥堵模式会与距离口径矛盾。期望行为：**AGV 走的路径就是距离矩阵用的那条最短路**（同一函数产出）。
5. **充电桩落在不可达或与机台重叠的节点上**。期望行为：所有充电桩节点在格点图中存在、互不重合、且与任一机台 dock 不同。
6. **`route_phi` 与 `agv_phi` 索引错配**。两者都按"第 task_i 个运输任务"索引；若计数口径不一致（例如一个在 `_transporter` 生成时递增、另一个在 `AgvSim` 取任务时递增），路线会**错配到别的任务上**，而**仿真照样跑完、指标照样出**——典型的静默错误。期望行为：两个 `phi` 数组的索引由**同一处**递增的计数器产生；`Task 3 Step 4b` 的 `test_route_phi_is_a_decision_variable` 只覆盖"路线被消费"，**索引一致性须靠人工核对计数点**。

---

### Task 1: 网格布局生成器

**Files:**
- Modify: `plcsp/env/layout.py`（整体重写）
- Test: `plcsp/tests/test_layout_grid.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `GridSpec(n_rows, n_cols, cell_w=3.0, cell_h=2.4, aisle_w=1.5)`，含
    `n_nodes -> int`、`node_id(r, c) -> int`、`node_rc(node_id) -> (r, c)`、`node_xy(r, c) -> (float, float)`
  - `MachinePad(id, x, y, w, h, dock_node: int, fail_rate: float, in_cap: int, out_cap: int)`
  - `ChargerPad(id, node: int)`、`BufferPad(id, node: int)`
  - `Layout(grid: GridSpec, machines, chargers, buffers, layout_seed: int)`，含 `n_machines -> int`
  - `sample_layout(n_machines, seed=0, *, cell_w=3.0, cell_h=2.4, aisle_w=1.5, n_chargers=2) -> Layout`

- [ ] **Step 1: 写失败测试**

创建 `plcsp/tests/test_layout_grid.py`：

```python
"""网格布局生成器的单元测试（P1a Task 1）。"""
from __future__ import annotations

import pytest

from plcsp.env.layout import GridSpec, sample_layout


@pytest.mark.unit
def test_gridspec_node_count_and_ids():
    spec = GridSpec(n_rows=2, n_cols=3)
    assert spec.n_nodes == (2 + 1) * (3 + 1) == 12
    assert spec.node_id(0, 0) == 0
    assert spec.node_id(0, 3) == 3
    assert spec.node_id(1, 0) == 4
    assert spec.node_rc(spec.node_id(2, 3)) == (2, 3)


@pytest.mark.unit
@pytest.mark.parametrize("n_machines", [5, 6, 10, 14, 20])
def test_layout_places_all_machines_on_distinct_nodes(n_machines: int):
    lay = sample_layout(n_machines, seed=0)
    assert lay.n_machines == n_machines
    nodes = [m.dock_node for m in lay.machines]
    assert len(set(nodes)) == n_machines, "机台 dock 节点必须互不重合"
    assert all(0 <= n < lay.grid.n_nodes for n in nodes)


@pytest.mark.unit
def test_layout_has_at_least_two_distinct_chargers():
    lay = sample_layout(6, seed=0, n_chargers=2)
    assert len(lay.chargers) >= 2
    nodes = [c.node for c in lay.chargers]
    assert len(set(nodes)) == len(nodes), "充电桩不得落在同一节点"
    machine_nodes = {m.dock_node for m in lay.machines}
    assert not (set(nodes) & machine_nodes), "充电桩不得与机台 dock 重合"


@pytest.mark.unit
def test_layout_is_deterministic():
    a, b = sample_layout(10, seed=7), sample_layout(10, seed=7)
    assert [m.dock_node for m in a.machines] == [m.dock_node for m in b.machines]
    assert [c.node for c in a.chargers] == [c.node for c in b.chargers]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `D:/anaconda/python.exe -m pytest plcsp/tests/test_layout_grid.py -q --no-header`
Expected: FAIL — `ImportError: cannot import name 'GridSpec'`

- [ ] **Step 3: 重写 `layout.py`**

整体替换 `plcsp/env/layout.py`：

```python
"""网格布局生成器（spec §3.2）：机位网格 + 通道格点 + 充电桩。

形态来源：AEI 103216 Fig. 2（该文借用 Cai et al., IJPR 61(4):1373-1393, 2023）。
- 机台占**格子**，通道是**格子之间的格线** → 通道天然不穿过机台（v0 缺陷自动消失）
- 通道图节点 = (n_rows+1) × (n_cols+1) 个交叉口
- zone = 通道节点（一节点一车互斥，见 des.ZoneManager）

单位：坐标 = 米（bug#13 已定单位约定）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """网格规格。机位 n_rows × n_cols；通道节点 (n_rows+1) × (n_cols+1)。"""
    n_rows: int
    n_cols: int
    cell_w: float = 3.0      # 机位宽 [m]
    cell_h: float = 2.4      # 机位高 [m]
    aisle_w: float = 1.5     # 通道宽 [m]（< 1.5 触发窄道降速，见 des.eff_speed）

    @property
    def n_nodes(self) -> int:
        return (self.n_rows + 1) * (self.n_cols + 1)

    def node_id(self, r: int, c: int) -> int:
        return r * (self.n_cols + 1) + c

    def node_rc(self, node: int) -> tuple[int, int]:
        return divmod(node, self.n_cols + 1)

    def node_xy(self, r: int, c: int) -> tuple[float, float]:
        """节点 (r,c) 的坐标：节点在格子**左上角**，间距 = 机位 + 通道。"""
        return (c * (self.cell_w + self.aisle_w), r * (self.cell_h + self.aisle_w))


@dataclass
class MachinePad:
    id: int
    x: float
    y: float
    w: float
    h: float
    dock_node: int            # 该机台装卸点所在的通道节点
    fail_rate: float = 0.001  # ⚠️ assumed（无文献出处），见 spec §9
    in_cap: int = 2           # ⚠️ assumed
    out_cap: int = 2          # ⚠️ assumed


@dataclass
class ChargerPad:
    id: int
    node: int


@dataclass
class BufferPad:
    id: int
    node: int


@dataclass
class Layout:
    grid: GridSpec
    machines: list[MachinePad]
    chargers: list[ChargerPad]
    buffers: list[BufferPad]
    layout_seed: int

    @property
    def n_machines(self) -> int:
        return len(self.machines)


def grid_shape(n_machines: int) -> tuple[int, int]:
    """机位数 → (n_rows, n_cols)：尽量方，行数 ≤ 列数。"""
    n_cols = max(1, math.ceil(math.sqrt(n_machines)))
    n_rows = max(1, math.ceil(n_machines / n_cols))
    return n_rows, n_cols


def sample_layout(n_machines: int, seed: int = 0, *, cell_w: float = 3.0,
                  cell_h: float = 2.4, aisle_w: float = 1.5,
                  n_chargers: int = 2) -> Layout:
    """采样一个网格布局。同 (n_machines, seed, 尺寸) → 同结果（确定性）。

    机台按格子顺序（逐行）占用前 n_machines 个格子；dock 落在该格左上角节点。
    充电桩从**未被机台占用**的节点里确定性选取（按节点号升序跳过已用）。
    """
    rng = np.random.default_rng(seed)
    n_rows, n_cols = grid_shape(n_machines)
    spec = GridSpec(n_rows=n_rows, n_cols=n_cols, cell_w=cell_w, cell_h=cell_h, aisle_w=aisle_w)

    machines: list[MachinePad] = []
    for i in range(n_machines):
        r, c = divmod(i, n_cols)
        node = spec.node_id(r, c)
        x, y = spec.node_xy(r, c)
        machines.append(MachinePad(
            id=i, x=x, y=y, w=cell_w, h=cell_h, dock_node=node,
            fail_rate=float(rng.uniform(0.0, 0.003)),      # assumed，见 spec §9
            in_cap=int(rng.integers(1, 4)),                # assumed
            out_cap=int(rng.integers(1, 4)),               # assumed
        ))

    used = {m.dock_node for m in machines}
    free = [n for n in range(spec.n_nodes) if n not in used]
    if len(free) < n_chargers:
        raise ValueError(f"自由节点不足：需 {n_chargers} 个充电桩，仅 {len(free)} 个可用")
    # 确定性分散：在自由节点里等间隔取，避免全挤在一角
    idx = [round(k * (len(free) - 1) / max(1, n_chargers - 1)) for k in range(n_chargers)] \
        if n_chargers > 1 else [0]
    charger_nodes = [free[i] for i in idx]
    chargers = [ChargerPad(id=j, node=n) for j, n in enumerate(charger_nodes)]

    # 缓冲/装卸：取最后两个自由节点（网格外侧），避开充电桩
    rest = [n for n in free if n not in set(charger_nodes)]
    buffers = [BufferPad(id=j, node=n) for j, n in enumerate(rest[-2:])] if len(rest) >= 2 else []

    return Layout(grid=spec, machines=machines, chargers=chargers,
                  buffers=buffers, layout_seed=seed)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `D:/anaconda/python.exe -m pytest plcsp/tests/test_layout_grid.py -q --no-header`
Expected: PASS（4 项）

- [ ] **Step 5: 提交**

```bash
git add plcsp/env/layout.py plcsp/tests/test_layout_grid.py
git commit -m "feat: 网格布局生成器（spec §3.2）取代 line/U/island"
```

---

### Task 2: 格点走廊图与距离矩阵

**Files:**
- Modify: `plcsp/env/corridors.py`（整体重写）
- Test: `plcsp/tests/test_corridors_lattice.py`

**Interfaces:**
- Consumes: Task 1 的 `Layout` / `GridSpec`（`node_id` / `node_rc` / `node_xy` / `n_nodes`）
- Produces:
  - `build_corridor_graph(layout: Layout) -> nx.Graph`——节点为 `int` 通道节点号，属性 `pos`；边为横/竖相邻，权重 = 段长 [m]
  - `dock_distance_matrix(g: nx.Graph) -> np.ndarray`——形状 `(n_nodes, n_nodes)`，最短路距离
  - `shortest_node_path(g: nx.Graph, src: int, dst: int) -> list[int]`——含首尾的节点序列
  - **`k_shortest_paths(g, src, dst, k=3) -> list[list[int]]`**——**候选路径集**（按长度升序）。
    这是**路线决策的动作空间**（用户 2026-10-02 选定：策略输出路线，见 spec §1 边界声明
    "任务分配 + 路线走廊选择"）。P2 接策略头，P1a 只提供集合与管道。

- [ ] **Step 1: 写失败测试**

创建 `plcsp/tests/test_corridors_lattice.py`：

```python
"""格点走廊图与距离矩阵的单元测试（P1a Task 2）。"""
from __future__ import annotations

import networkx as nx
import pytest

from plcsp.env.corridors import (build_corridor_graph, dock_distance_matrix,
                                 shortest_node_path)
from plcsp.env.layout import sample_layout


@pytest.mark.unit
def test_lattice_is_connected_and_sized():
    lay = sample_layout(6, seed=0)
    g = build_corridor_graph(lay)
    assert g.number_of_nodes() == lay.grid.n_nodes
    assert nx.is_connected(g), "格点图必须连通"


@pytest.mark.unit
@pytest.mark.parametrize("n_machines", [5, 6, 10, 14, 20])
def test_all_machine_docks_mutually_reachable(n_machines: int):
    """Review Focus #1：任意两机台 dock 必须可达（不得抛 NetworkXNoPath）。"""
    lay = sample_layout(n_machines, seed=0)
    g = build_corridor_graph(lay)
    dm = dock_distance_matrix(g)
    nodes = [m.dock_node for m in lay.machines]
    for a in nodes:
        for b in nodes:
            assert dm[a, b] < float("inf"), f"dock {a} → {b} 不可达"


@pytest.mark.unit
def test_lattice_distance_matches_manhattan_on_full_grid():
    """空格不影响格点结构：3x2 网格上相邻节点距离应等于段长。"""
    lay = sample_layout(6, seed=0)
    g = build_corridor_graph(lay)
    s = lay.grid
    # 相邻横节点
    d_h = dock_distance_matrix(g)[s.node_id(0, 0), s.node_id(0, 1)]
    assert d_h == pytest.approx(s.cell_w + s.aisle_w)
    # 相邻竖节点
    d_v = dock_distance_matrix(g)[s.node_id(0, 0), s.node_id(1, 0)]
    assert d_v == pytest.approx(s.cell_h + s.aisle_w)


@pytest.mark.unit
def test_shortest_node_path_returns_contiguous_walk():
    """Review Focus #4：路径必须是图上真实相邻的一串节点（无跳跃）。"""
    lay = sample_layout(14, seed=0)
    g = build_corridor_graph(lay)
    src, dst = lay.machines[0].dock_node, lay.machines[-1].dock_node
    path = shortest_node_path(g, src, dst)
    assert path[0] == src and path[-1] == dst
    assert len(path) >= 2
    for u, v in zip(path, path[1:]):
        assert g.has_edge(u, v), f"路径上 {u}→{v} 并非图上的边"


@pytest.mark.unit
def test_path_length_equals_matrix_entry():
    """Review Focus #4：逐段路径的总长必须等于距离矩阵给出的最短路。"""
    lay = sample_layout(14, seed=0)
    g = build_corridor_graph(lay)
    dm = dock_distance_matrix(g)
    src, dst = lay.machines[0].dock_node, lay.machines[-1].dock_node
    total = sum(g[u][v]["weight"] for u, v in zip(*(lambda p: (p, p[1:]))(
        shortest_node_path(g, src, dst))))
    assert total == pytest.approx(dm[src, dst], rel=1e-9)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `D:/anaconda/python.exe -m pytest plcsp/tests/test_corridors_lattice.py -q --no-header`
Expected: FAIL — `ImportError: cannot import name 'shortest_node_path'`

- [ ] **Step 3: 重写 `corridors.py`**

整体替换 `plcsp/env/corridors.py`：

```python
"""格点走廊图与距离矩阵（spec §3.2）。

节点 = 通道交叉口（编号 0..n_nodes-1）；边 = 横/竖相邻节点，权重 = 段长 [m]。
机台/充电桩/缓冲都**挂在**某个通道节点上（`dock_node` / `node`），故距离矩阵按节点算。

与上一代的区别：旧版是"相邻 dock + 首尾闭环"的**单环**，且连线可能穿过机台；
新版通道在格子**之间**、机台在格子**之内**，两问题同时消失（无需栅格 A*）。
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .layout import Layout


def build_corridor_graph(layout: Layout) -> nx.Graph:
    """格点图：节点带 pos 属性；横/竖相邻边，权重 = 欧氏段长 [m]。"""
    spec = layout.grid
    g = nx.Graph()
    for r in range(spec.n_rows + 1):
        for c in range(spec.n_cols + 1):
            g.add_node(spec.node_id(r, c), pos=spec.node_xy(r, c))
    for r in range(spec.n_rows + 1):
        for c in range(spec.n_cols + 1):
            u = spec.node_id(r, c)
            if c + 1 <= spec.n_cols:                     # 横向
                v = spec.node_id(r, c + 1)
                g.add_edge(u, v, weight=abs(spec.node_xy(r, c + 1)[0] - spec.node_xy(r, c)[0]))
            if r + 1 <= spec.n_rows:                     # 纵向
                v = spec.node_id(r + 1, c)
                g.add_edge(u, v, weight=abs(spec.node_xy(r + 1, c)[1] - spec.node_xy(r, c)[1]))
    return g


def dock_distance_matrix(g: nx.Graph) -> np.ndarray:
    """节点两两最短路距离矩阵 (n_nodes, n_nodes) [m]。不可达处为 inf。"""
    nodes = sorted(g.nodes)
    n = len(nodes)
    dm = np.full((n, n), np.inf, dtype=float)
    idx = {node: i for i, node in enumerate(nodes)}
    for src in nodes:
        lengths = nx.single_source_dijkstra_path_length(g, src, weight="weight")
        for dst, dist in lengths.items():
            dm[idx[src], idx[dst]] = float(dist)
    return dm


def shortest_node_path(g: nx.Graph, src: int, dst: int) -> list[int]:
    """两节点间最短路的节点序列（含首尾）。

    Review Focus #4：AGV 的逐段 zone 申请必须走**这条**路径——与距离矩阵同源，
    否则拥堵模式会与距离口径矛盾。
    """
    return list(nx.dijkstra_path(g, src, dst, weight="weight"))


def k_shortest_paths(g: nx.Graph, src: int, dst: int, k: int = 3) -> list[list[int]]:
    """**候选路径集**（简单路径，按长度升序，最多 k 条）。

    这是**路线决策的动作空间**（spec §1 边界声明："任务分配 + 路线走廊选择"）：
    策略在 (frm, to) 的这 k 条候选里选一条 → 主动绕开拥堵。
    网格上通常有多条（曼哈顿等长最短路 + 绕行路），故候选集天然非空且 >1。

    注意：候选数可能 < k（图小或端点相邻时），此时返回实际条数。
    """
    out: list[list[int]] = []
    for p in nx.shortest_simple_paths(g, src, dst, weight="weight"):
        out.append(list(p))
        if len(out) >= k:
            break
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `D:/anaconda/python.exe -m pytest plcsp/tests/test_corridors_lattice.py -q --no-header`
Expected: PASS（5 项，其中 `test_all_machine_docks_mutually_reachable` 参数化 5 例）

- [ ] **Step 5: 提交**

```bash
git add plcsp/env/corridors.py plcsp/tests/test_corridors_lattice.py
git commit -m "feat: 格点走廊图与最短路距离矩阵（取代单环走廊）"
```

---

### Task 3: 路径级逐段 zone 申请

**Files:**
- Modify: `plcsp/env/des.py`（`ZoneManager` 与 `AgvSim.run`；`rollout` 的 距离矩阵调用；`SimConfig.n_zones` 移除）
- Test: `plcsp/tests/test_zone_path.py`

**Interfaces:**
- Consumes: Task 2 的 `build_corridor_graph` / `dock_distance_matrix` / `shortest_node_path`
- Produces:
  - `ZoneManager(env, n_nodes, wait_limit)`——zone 编号 = 通道节点号（`0..n_nodes-1`）
  - `AgvSim` 逐段推进：持 `cur` → 申请 `nxt` → 成功后释放 `prev`
  - `SimWorld.run()/run_gated()` 返回 `zone_wait` 字段（**保持既有签名**），新增 `n_zones_used`

- [ ] **Step 1: 写失败测试**

创建 `plcsp/tests/test_zone_path.py`：

```python
"""路径级逐段 zone 申请的测试（P1a Task 3）。"""
from __future__ import annotations

import pytest

from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import load_mk


@pytest.mark.unit
def test_episode_completes_on_grid():
    """网格上跑通：全部作业完成、无掐表。"""
    inst = load_mk("mk01")
    r = rollout(inst, seed_chain=1, cfg=SimConfig(n_agv=4))
    assert r["jobs_done"] == inst.n_jobs
    assert r["horizon_hit"] is False


@pytest.mark.unit
def test_no_zone_leak_after_episode():
    """Review Focus #3：episode 结束后不得有 zone 仍被持有。"""
    inst = load_mk("mk01")
    r = rollout(inst, seed_chain=1, cfg=SimConfig(n_agv=4))
    assert r["zone_holders_free"] is True, "有 zone 未释放 → 下一轮 rollout 会卡死"


@pytest.mark.unit
def test_zone_wait_is_reported_and_nonzero_under_contention():
    """Review Focus #2：多车时应有真实的区段等待（路径级争用可见）。"""
    inst = load_mk("mk01")
    r = rollout(inst, seed_chain=1, cfg=SimConfig(n_agv=4))
    zw = r["zone_wait"]
    assert zw["n"] > 0, "路径级 zone 申请应产生等待记录"
    assert zw["total"] > 0.0


@pytest.mark.unit
def test_single_agv_has_no_zone_contention():
    """单车不可能与自己争用 → 等待次数应为 0（交叉口一次只被一车申请）。"""
    inst = load_mk("mk01")
    r = rollout(inst, seed_chain=1, cfg=SimConfig(n_agv=1))
    assert r["zone_wait"]["n"] == 0


@pytest.mark.unit
def test_route_phi_is_a_decision_variable():
    """**路线是决策变量**（spec §1 边界声明）：传入 route_phi 全选第 2 条候选，
    总行驶时间不得短于全选最短路（即路线确实被消费了，不是摆设）。"""
    # Arrange
    inst = load_mk("mk07")
    cfg = SimConfig(n_agv=4)
    r_short = rollout(inst, seed_chain=1, cfg=cfg)          # 缺省 = 最短路
    n_tasks = int(r_short["dbg"].get("trans_evt", 0))

    # Act
    r_long = rollout(inst, seed_chain=1, cfg=cfg, route_phi=[1] * n_tasks)

    # Assert
    assert r_short["jobs_done"] == inst.n_jobs and r_long["jobs_done"] == inst.n_jobs
    assert r_long["travel_time_total"] >= r_short["travel_time_total"], \
        "选了更长候选路径却行驶更少 → route_phi 没被消费"
    assert r_short["travel_time_total"] > 0.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `D:/anaconda/python.exe -m pytest plcsp/tests/test_zone_path.py -q --no-header`
Expected: FAIL — `KeyError: 'zone_holders_free'`

- [ ] **Step 3: 改 `ZoneManager`——zone 编号改为通道节点**

在 `plcsp/env/des.py` 中，把 `ZoneManager.__init__` 的 `n_zones` 形参改名为 `n_nodes`（语义：zone 数 = 通道节点数），并在类 docstring 补一句：

```python
class ZoneManager:
    """通道节点互斥 + **资源等待环检测**（死锁 = 等待环）。

    **zone 编号即通道节点号**（spec §3.2）：AGV 沿实际最短路逐段推进，
    持有当前节点、申请下一个、成功后释放上一个 → 等待图反映真实路径，
    经典 AGV 环状死锁在此结构下才可能出现，由 DFS 环检测在授权前拒绝。
    """
```

- [ ] **Step 4: 改 `AgvSim.run`——逐段推进**

把 `plcsp/env/des.py::AgvSim.run` 中**原来那两行伪造的 zone 选择**：

```python
            seq = sorted({frm % self.zm.n, to % self.zm.n})
```

**连同任务元组的解包**，一起改。任务元组由 3 元组变 **4 元组**：`(frm, to, item, path)`——
`path` 即**策略选定的路线**（节点序列）。所以先改解包：

```python
            frm, to, item, path = yield q.get()
```

再把它之后"一口气把 `seq` 全部申请下来"的整段循环替换为**沿该路径逐段推进**：

```python
            # ① 先占住起点节点
            granted, cycle = yield from self.zm.wait_zone(
                self.aid, path[0], self.cfg.zone_hold_limit)
            if cycle or not granted:
                self.stats["requeue"] = self.stats.get("requeue", 0) + 1
                q.put((frm, to, item, path))
                continue
            # ② 逐段推进：申请下一节点 → 成功后释放上一节点 → 走这一段
            prev = path[0]
            ok = True
            for prev_i, nxt in enumerate(path[1:]):
                granted, cycle = yield from self.zm.wait_zone(
                    self.aid, nxt, self.cfg.zone_hold_limit)
                if cycle or not granted:
                    ok = False
                    break
                seg_m = float(self.m_dm[path[prev_i], nxt])        # 上一节点 → 本节点 [m]
                seg_min = seg_m / (self.cfg.eff_speed * self.cfg.agv_speed_mps) / SECONDS_PER_MIN
                self.zm.release(self.aid, prev)                     # 拿到下一段才放上一段
                prev = nxt
                yield self.env.timeout(seg_min)
                self.stats["travel_time"] += seg_min
                self.stats["moves"] += 1
            if not ok:
                self.zm.release(self.aid, prev)                     # Review Focus #3：失败路径也必须释放
                yield self.env.timeout(self.cfg.zone_hold)
                self.stats["requeue"] = self.stats.get("requeue", 0) + 1
                q.put((frm, to, item, path))
                continue
            self.zm.release(self.aid, prev)                         # 到达终点 → 释放最后一个
```

> 语义：AGV **任一时刻最多持 2 个节点**（当前 + 已预约的下一节点），且"拿到下一段才放上一段"——
> 这正是经典 AGV **环状等待**（AGV1 持 B 等 C、AGV2 持 C 等 B）得以出现的结构，也是 `ZoneManager`
> 的等待图 DFS 环检测真正派上用场的地方。
>
> ⚠️ **`shortest_node_path` 在这个任务里不再被 `AgvSim` 调用**——路径改由任务元组携带
> （来源见 Step 4b）。`shortest_node_path` 仍作为**默认路线**（无策略时）保留。

- [ ] **Step 4b: 在 `_transporter` 里生成候选路径并选定路线**

`AgvSim` 只消费路径，**路径在任务生成处决定**。改 `plcsp/env/des.py::SimWorld._transporter`：

1. **每 episode 预计算一次候选路径缓存**（路径只取决于 `(frm, to)`，不要每个任务重算）：

```python
        # 候选路径缓存：{(frm, to): [path0, path1, ...]}，按长度升序（spec §1 边界声明：
        # 路线是决策变量）。K = 候选数上限。
        K_PATHS = 3
        path_cache: dict[tuple[int, int], list[list[int]]] = {}
        def paths_for(a: int, b: int) -> list[list[int]]:
            key = (a, b)
            if key not in path_cache:
                cand = k_shortest_paths(self.g, a, b, K_PATHS)
                path_cache[key] = cand if cand else [shortest_node_path(self.g, a, b)]
            return path_cache[key]
```

2. **任务入队时携带选定路径**。把原来 `tasks_in.put((frm, to, item))` 之类的写法改为：

```python
                cands = paths_for(frm, to)
                # route_phi[task_i] = 候选路径编号（策略输出；None/缺省 → 0 = 最短路）
                ri = route_phi[task_i] if route_phi else 0
                ri = min(max(0, ri), len(cands) - 1)          # 越界兜底：退回最短路
                q.put((frm, to, item, cands[ri]))
```

其中 `route_phi` 是与既有 `agv_phi` **同构的形参**（同样是"第 task_i 个运输任务的决策"），
默认 `None`。`task_i` 的计数方式与 `agv_phi` **必须完全一致**（同一个生成序），否则路线与 AGV 会错配。

3. **`run()` / `run_gated()` 的签名各增一个 `route_phi: list[int] | None = None`**，
   与 `agv_phi` 并列，并透传给 `_transporter`。

> 这样 P2 只需让 `policy_l` 多输出一个路线头，并把选中的编号写进 `route_phi`——
> **`AgvSim` 与 `ZoneManager` 一行都不用再动**。

- [ ] **Step 5: `AgvSim` 构造与 `SimWorld` 传递走廊图**

`AgvSim.__init__` 增加 `graph` 形参（放在 `m_dm` 之后）：

```python
    def __init__(self, env, aid, m_dm: np.ndarray, cfg: SimConfig, stats: dict,
                 zm: ZoneManager, tasks_in, machines: list, graph, bound: bool = False):
        self.env, self.aid, self.m_dm, self.cfg = env, aid, m_dm, cfg
        self.stats, self.zm, self.tasks_in, self.machines = stats, zm, tasks_in, machines
        self.g = graph                  # 格点走廊图（逐段路径的来源）
        self.bound = bound
```

`SimWorld.__init__` 增加 `graph=None` 形参并存为 `self.g`；`run()` 与 `run_gated()` 中
每一处 `AgvSim(...)` 构造都补传 `graph=self.g`。若 `graph` 为 `None`（旧调用方），
在 `AgvSim.__init__` 里 `self.g = graph if graph is not None else nx.Graph()` 并**抛错**更好——
本计划一律显式传图。

- [ ] **Step 6: 在返回值里加 `zone_holders_free`**

`SimWorld.run()` 与 `run_gated()` 的返回 dict 中，与 `zone_wait` 并列加：

```python
                "zone_holders_free": all(v is None for v in zm.holder.values()),
                "n_zones_used": zm.n,
                "travel_time_total": float(stats["travel_time"]),   # 路线决策的效果可见
```

- [ ] **Step 7: 让 `rollout` 支持 `route_phi` 并改用格点图**

`plcsp/env/des.py::rollout` 的签名增加 `route_phi: list[int] | None = None`（与 `agv_phi` 并列），
并透传给 `SimWorld.run`；同时把布局与距离矩阵换成格点版：

```python
def rollout(inst: Instance, layout_type: str = "line", seed_layout: int = 0, seed_chain: int = 0,
            cfg: SimConfig | None = None, op_choices: list[list[int]] | None = None,
            machine_gap: float = 1.0, aisle_width: float = 1.5,
            agv_phi: list[int] | None = None,
            route_phi: list[int] | None = None) -> dict:
    ...
    layout = sample_layout(inst.n_machines, seed=seed_layout, aisle_w=aisle_width)
    g = build_corridor_graph(layout)
    dm = dock_distance_matrix(g)
    return SimWorld(inst, layout, dm, cfg or SimConfig(), graph=g).run(
        seed_chain=seed_chain, op_choices=op_choices, agv_phi=agv_phi, route_phi=route_phi)
```

> `layout_type` 形参**保留但忽略**（旧调用方仍在传）——计划 3 统一清理。
> 同时把 `rollout` 里 `from .corridors import build_corridor_graph, dock_distance_matrix`
> 与 `from .corridors import k_shortest_paths, shortest_node_path` 的导入补齐。

- [ ] **Step 8: 跑测试确认通过**

Run: `D:/anaconda/python.exe -m pytest plcsp/tests/test_zone_path.py -q --no-header`
Expected: PASS（4 项）

- [ ] **Step 9: 跑全套回归**

Run: `D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header`
Expected: **全部通过**。若 `test_instances.py` 变红，**说明改动越界了**——它只依赖实例加载，不该受布局影响。

- [ ] **Step 10: 提交**

```bash
git add plcsp/env/des.py plcsp/tests/test_zone_path.py
git commit -m "feat: AGV 沿真实路径逐段申请 zone（取代 frm%n / to%n 的伪造分配）"
```


---

### Task 4: P1a 验收与基线重标

**Files:**
- Create: `plcsp/m10_grid_calib.py`
- Modify: `docs/INDEX.md`（登记 P1a 完成）
- Modify: `docs/progress-log.md`（记录 `n_agv` 重标结果）

**Interfaces:**
- Consumes: Task 1–3 的全部产出
- Produces: 网格环境下的 `n_agv` 推荐值 + 拥堵实测曲线（供 spec §6.3 回填）

- [ ] **Step 1: 写重标脚本**

创建 `plcsp/m10_grid_calib.py`：

```python
# -*- coding: utf-8 -*-
"""网格环境下的 n_agv 重标与拥堵实测（P1a 验收）。

背景：spec §3.3 的拥堵数据（n_agv=4 等待占比 32%）是在**旧的 line 单环**上测的；
换网格后 AGV 可绕行，该数会变，故 n_agv 必须重标（spec §6.3 已标注待重标）。

用法：python -m plcsp.m10_grid_calib [--seeds 10]
输出：stdout 表格（不写文件）
"""
from __future__ import annotations

import argparse
import statistics as st

from .env.des import SimConfig, rollout
from .env.instances import load_mk

INSTANCES = ("mk01", "mk07")
AGV_GRID = (1, 2, 3, 4, 6, 8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    print("实例   n_agv  makespan      等待次数  等待总时长  等待占比  未释放zone  最长单次")
    for name in INSTANCES:
        inst = load_mk(name)
        for na in AGV_GRID:
            cfg = SimConfig(n_agv=na)
            ms, n_, tot, mx = [], [], [], []
            leak = 0
            for s in range(args.seeds):
                r = rollout(inst, seed_chain=s, cfg=cfg)
                if r["horizon_hit"] or r["jobs_done"] != inst.n_jobs:
                    continue
                zw = r["zone_wait"]
                ms.append(r["makespan"]); n_.append(zw["n"])
                tot.append(zw["total"]); mx.append(zw["max"])
                leak += 0 if r["zone_holders_free"] else 1
            if not ms:
                print(f"{name:6} {na:5}  （无有效样本）")
                continue
            m = st.mean(ms); t = st.mean(tot)
            print(f"{name:6} {na:5}  {m:11.1f}  {st.mean(n_):9.1f}  {t:10.2f}  "
                  f"{100*t/(m*na):7.2f}%  {leak:10d}  {st.mean(mx):8.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑重标**

Run: `PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m10_grid_calib --seeds 10`
Expected: 打印三实例 × 6 档 `n_agv` 的表；**`未释放zone` 一列全为 0**（Review Focus #3）。

- [ ] **Step 3: 判定并记录 `n_agv`**

按 spec §6.3 的口径选取：**等待占比落在 15–35% 且 `requeue ≈ 0`**（拥堵显著、系统未过饱和）。
把选中的 `n_agv` 与实测曲线写进 `docs/progress-log.md` 新增一节，并在
`docs/superpowers/specs/2026-10-02-plcsp-rebuild-design.md` §6.3 把"⏳ 待网格布局后重标"改为实测值。

- [ ] **Step 4: 更新 INDEX**

在 `docs/INDEX.md` 的 §5.5 之后新增：

```markdown
## 5.6 P1a 空间基底（2026-10-02 完成）

- **网格布局**：机位 `n_rows × n_cols`，通道为格线；**通道不穿过机台**（v0 缺陷消失，无需栅格 A*）。
- **zone 重定义**：由"3 个抽象资源"改为**通道节点**（一节点一车互斥）。
- **路径级申请**：AGV 沿真实最短路**逐段推进**（持当前、申请下一个、释放上一个）——取代旧的 `frm % n / to % n` 伪造分配。
- **充电桩 ≥2**，落在不同通道节点。
- **`n_agv` 已按网格重标**：见 `progress-log.md`（旧值 4 系 line 单环下所测，不可沿用）。
```

- [ ] **Step 5: 端到端复检 + 全套回归**

Run:
```bash
PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -c "
from plcsp.env.instances import load_mk
from plcsp.env.des import rollout, SimConfig
for n in ('mk01','mk07','mk10'):
    i=load_mk(n); r=rollout(i, seed_chain=1, cfg=SimConfig(n_agv=4))
    print(n, r['jobs_done'],'/',i.n_jobs, 'mks', round(r['makespan'],1),
          'horizon_hit', r['horizon_hit'], 'zone_free', r['zone_holders_free'])
"
D:/anaconda/python.exe -m pytest plcsp/tests/ -q --no-header | tail -3
```
Expected: 三实例全完成、`zone_free True`；测试全绿。

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "feat: P1a 空间基底验收（网格 + 路径级 zone + n_agv 重标）"
```

---

## 完成后的状态

- 车间空间模型 = 二维网格；通道与机台分离；拥堵建立在**真实路径**上
- `line`/`U`/`island` 与单环走廊**已消失**
- `n_agv` 已按网格重标，spec §6.3 回填
- **可交付给计划 3（能耗 M2 + 约束框架 + 11 约束接入）**

## 已知风险 / 未决

| 项 | 说明 |
|---|---|
| **拥堵量级会变** | 网格多路径 → 比 line 单环**更轻**。§10.2 的"车队规模—吞吐非单调"机制候选**必须在网格上复现**；**若消失，机制候选即失效**——这是本计划最重要的观察项 |
| **`n_chargers` 取 2** | spec §3.2 定了"≥2"；具体值可在 §6.3 敏感性里扫 |
| **`fail_rate` / `in_cap` / `out_cap` 仍是 assumed** | 沿用旧采样器的随机区间，**无文献出处**（spec §9 已列）。计划 3 的参数—出处表要处理 |
| **`m1_smoke.py` 等旧脚本** | `Layout` 字段变了（`layout_type`/`char_len` 消失），它们会断。**P1a 不负责修**——计划 3 一并处理 |
