# -*- coding: utf-8 -*-
"""A-MKT 报告档的**精确解**基线（OR-Tools CP-SAT）——给策略一把认证过的尺子。

## 这个模块回答什么问题

策略在 `REPORT_TIERS["A-MKT"]` 上的 makespan 比最优解差多少？该档把十个机制全关，
只剩下 **FJSP + 行程时间矩阵**。于是在该档上，**策略能实现的调度是本模型可行集的子集**：

    J* ≤ J_policy      ⟹     (J_policy − J*)/J* 是**认证过的**最优性间隙上界。

⚠️ **精确解不是竞争者，是尺子**：它的值不依赖策略的架构、种子或消融——换一个策略，
`J*` 不变。故报告时**必须同时给出对偶界**：它才是"证明过的"那一半（`J* ≥ dual_bound`）。

## 依据的模型

**主模型 = Yao, Y., Liu, Q., Li, X., Gao, L. _Constraint Programming for AGV and Machine
Integrated Scheduling Problem in Flexible Manufacturing System._ IEEE Transactions on
Automation Science and Engineering 23:2378–2390, 2026（DOI 10.1109/TASE.2025.3650678；
PDF 在 `references/TASE2026_AGVMachine_IntegratedScheduling_CP.pdf`）的 NCP2**，
用 CP-SAT 重写。NCP2 的判据：**相邻工序若在同一机台上，则该次运输任务根本不存在**，
机台前后由直接先后约束 (30) 连起来（论文 §IV.D）。逐条对应关系：

| NCP2 | 本文实现 |
|---|---|
| (19) `alternative(ProTask_i, [ProInMac_i,j])` | 每工序每候选机台一个 `NewOptionalIntervalVar` + `AddExactlyOne` |
| (20)(21) `noOverlap(TaskInMacSeq_j)` | `AddNoOverlap`（机台侧） |
| (22)(23)(24)(25) 运输任务的存在性与 AGV 指派 | `present` 布尔 + `Σ_k (1 − skip[k][t]) == present[t]` |
| (26)(27) `noOverlap(TaskInVehSeq_k, Tt)` | **见下：CP-SAT 没有转移矩阵版 noOverlap** |
| (28)(29)(30) 运输↔加工的时间耦合 | `mv_start ≥ E_prev`、`S ≥ mv_end`、`S ≥ E_prev` |

### 唯一的真坑：AGV 序的转移时间

CP Optimizer 的 `noOverlap(seq, Tt)` 用转移矩阵表达 AGV 空载行程；CP-SAT **没有**对应物。
本模块按 OR-Tools 官方 contrib 例程（`examples/contrib/scheduling_with_transitions_sat.py`）
的路线，用 **`AddCircuit` + 字面弧**表达：每台车一张有向图，节点 = {起点 S_k} ∪ 运输任务 ∪
{终点 E_k}；弧 (a→b) 为真 ⟹ `mv_end[a] + 空载(a→b) ≤ mv_start[b]`。任务未被该车服务时走
**自环**（`AddCircuit` 认自环为"该节点不在回路里"——已实测）。

空载时长依赖**车上一趟的卸货点** `d[a]` 与本趟的取货点 `o[t]`——两者都是决策变量。
本模块不枚举机台对（那会是 O(T²·|MS|²) 条约束；实测 MK01 需 14 513 条、MK03 需 203 997 条），
而是每 (车, 任务) 引入一个**取货前车位置** `pd`，用 `AddElement` 查表取时长：
`tt[k][t] = T[pd[k][t], o[t]]`。于是弧只需 2 条约束：`pd == d[a]` 与
`mv_end[a] + tt ≤ mv_start[b]`。

## 与**本仓** A-MKT 口径的两处刻意对齐

1. **回站段**（`include_return=True`，默认）：本仓的 `completes[j]` = 工件**回到装卸站**的时刻
   （`des.LuStation`），故每条作业在末工序后还有一条"末机台 → 装卸站"的负载段。
   论文模型止于末机台。加上这一段后模型**仍是** DES 的松弛（DES 本来就有这一段），但**更紧**。
2. **车的初始位置**（`initial_travel="free"`，默认）：`AgvSim.pos_node is None` 时**跳过空载段**
   （首趟任务不付空载时间）——即"车在首个取货点待命"。论文假设车全部停在仓库。
   默认取 `free` 是**逐字复刻 DES**（严格的松弛）；`lu` 是论文口径（更紧，但只在"每台车首趟
   都是装卸站取货"时才是 DES 的松弛——本档 v ≤ 作业数时成立，`docs` 有论断）。

## 已知的松弛（模型比 DES 松的地方，故 J* 只可能**偏小**，不会偏大）

- NCP2 的 `noOverlap(seq, Tt)` **只在车序转移里**收空载时间；若"上一工序完工"是紧的那一边，
  模型允许运输任务在 `E_prev` 那一刻就开始（DES 要求 `E_prev + 空载`）。`exact_deadhead=True`
  补上这条（仍是 DES 的松弛，但更紧）。
- 同机台相邻工序：NCP2 不收 AGV（DES 要一趟零时长交接）。

## 数据前提

CP-SAT 是整数求解器。本模块把**加工时长与行程时间**同乘一个公共分母 `scale`（= 各值分母的
LCM；整数数据 ⟹ 1）后取整——**不做任何四舍五入**，故缩放不引入误差，`J*/scale` 仍是精确解。

## 两个读数的地位（别把它们并排看）

- **`dual_bound`（对偶界）= 认证的下界**：`J* ≥ dual_bound` 是 CP-SAT **证明过的**。
  故 `(J_policy − dual_bound)/dual_bound` 对**任何**策略都成立——这才是本模块的产物。
- **`objective`（最好解）= 上界**：来自 `greedy_schedule` 的 warm start 或求解器的搜索。
  它**不是**认证的（除非状态是 OPTIMAL，那时两者相等、间隙归零）。

**构造式 warm start**（`greedy_schedule` + `hint_schedule`）：启发式列表调度，确定性、无随机数。
它的 makespan 是**真调度**的 makespan ⟹ 可安全地当地平线（`cmax ≤ 它`），也灌进 `add_hint`
当初始解。⚠️ **提示是完整提示**：CP-SAT 对不完整的 `add_hint` 只喂一个 repair 搜索
（实测在大实例上等于没用）；实测完整提示能让 MK01 在 15 s 内直接报出构造解那一条。

用法见 `main()`；测试见 `plcsp/tests/test_baseline_exact.py`。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Sequence

from ..env.constraints import REPORT_TIERS
from ..env.instances import Instance

# ── 口径常量（字符串而非 bool：调用点读得出"选了哪一档"）──
INIT_FREE = "free"      # 车在首趟任务的取货点待命（= DES 语义，默认）
INIT_LU = "lu"          # 车全部停在装卸站（= 论文假设）
MAX_SCALE = 10 ** 6     # 公共分母上限：超过就说明数据不是"有限位小数"，显式报错
GAP_TOL = 1e-9


class OrtoolsMissing(RuntimeError):
    """未装 ortools。安装会改动门禁解释器环境，故**不自动装**——见模块 docstring 与报告。"""


def _ortools():
    """延迟导入 ortools：门禁解释器没装它时，本模块仍可被导入（测试用 importorskip）。"""
    try:
        from ortools.sat.python import cp_model
    except ImportError as e:                                   # pragma: no cover - 门禁无 ortools
        raise OrtoolsMissing(
            "精确解基线需要 ortools（CP-SAT）。门禁解释器 D:/anaconda/python.exe **没有**它，"
            "且本模块**不自动安装**（那会改动门禁环境）。装法："
            "D:/anaconda/python.exe -m pip install ortools —— 请先征得同意。"
        ) from e
    return cp_model


# ══════════════════════ 数据 → 整数时间 ══════════════════════

def _denominator(v: float) -> int:
    """一个时间量的分母（整数 ⟹ 1）。用 limit_denominator 认有限位小数，无四舍五入。"""
    f = Fraction(float(v)).limit_denominator(MAX_SCALE)
    if abs(float(f) - float(v)) > 1e-9:
        raise ValueError(f"时间量 {v!r} 不是有限位小数（分母 > {MAX_SCALE}）——"
                         "本模块只做**精确**整数缩放，不做四舍五入（四舍五入会破坏下界性）")
    return f.denominator


def time_scale(inst: Instance) -> int:
    """加工时长与行程时间的公共分母（LCM）。整数数据 → 1（本仓 MK01–MK05/MK07 即此）。"""
    dens = {_denominator(t) for job in inst.jobs for alts in job for _, t in alts}
    mat = inst.trans_time_full
    if mat is not None:
        dens |= {_denominator(float(x)) for x in mat.ravel()}
    scale = 1
    for d in dens:
        scale = scale * d // math.gcd(scale, d)
    if scale > MAX_SCALE:
        raise ValueError(f"公共分母 {scale} 超过上限 {MAX_SCALE}")
    return scale


def travel_matrix(inst: Instance) -> list[list[int]]:
    """行程时间矩阵 → **整数** `T[u][v]`，下标口径：**0 = 装卸站（LU）**、机台 m ↦ m+1。

    ⚠️ 对角线**必须置 0**：`AgvSim._drive` 在 `src == dst` 时**直接返回**（不查表），
    而 MKT 的 `m=17` 矩阵对角第 6 项恰是 4（上游数据如此，见 `plcsp/data/mkt/README.md` §2）。
    不置 0 会凭空给"同点往返"加 4 分钟 ⟹ 模型比 DES 紧 ⟹ 下界性失效。
    """
    mat = inst.trans_time_full
    if mat is None:
        raise ValueError("精确解基线只支持矩阵口径（`inst.trans_time_full` 缺失）——"
                         "几何口径请先用 `plcsp.env.transport` 造出机台间行程时间矩阵")
    scale = time_scale(inst)
    n = int(mat.shape[0])
    out = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in range(n):
            out[u][v] = 0 if u == v else int(round(float(mat[u, v]) * scale))
    return out


# ══════════════════════ 口径 ══════════════════════

@dataclass(frozen=True)
class ExactSpec:
    """一次精确求解的**口径**。口径是问题的属性，必须随结果一起报出。"""

    tier: str = "A-MKT"
    n_agv: int = 3
    time_limit_s: float = 3600.0
    # 与 `EvalSpec` 同名的两个 SimConfig 级开关——精确解只支持全关的 A-MKT 档
    transport_unmapped: str = "geometry"
    # ── 模型变体（默认 = 论文 NCP2 + 本仓回站口径）──
    include_return: bool = True            # 末工序后"末机台 → 装卸站"负载段
    same_machine_transport: bool = False   # True = 同机台相邻工序也生成零时长运输任务（DES 逐字口径）
    initial_travel: str = INIT_FREE        # 车首趟任务的空载口径：free（DES）/ lu（论文）
    exact_deadhead: bool = False           # 补"空载从上一工序完工时刻起算"（更紧，仍是松弛）
    sym_break: bool = True                 # 同型车对称性破缺
    workers: int = 8
    random_seed: int = 0
    # 上界（= 地平线）。None ⟹ 由规则基线（`rollout`）给；给不出时用构造式粗界。
    upper_bound: float | None = None


# ══════════════════════ 运输任务表 ══════════════════════

@dataclass(frozen=True)
class _Task:
    """一条运输任务。`o_op`/`d_op` = 决定取货点/卸货点的工序（None 表示装卸站）。

    - **到达任务** `kind="arr"`（作业 j 的第 i 道工序）：`o_op=(j,i-1)`（i=0 时为 None=站）、
      `d_op=(j,i)`。存在性由机台选择决定（NCP2：同机台则不存在）。
    - **回站任务** `kind="ret"`（作业 j）：`o_op=(j,last)`、`d_op=None`（站）。恒存在。
    """

    idx: int
    kind: str
    job: int
    op: int
    o_op: tuple[int, int] | None
    d_op: tuple[int, int] | None
    rank: int          # 该任务在本题作业路线里的序号（用于弧剪枝）


def _task_table(inst: Instance, include_return: bool) -> list[_Task]:
    """按 `_transporter` 的生成序造任务表：作业 0..n-1 的投放/流转，最后是各作业的回站。

    ⚠️ **顺序不影响模型**（任务之间只有先后约束），但保持与 `transport_task_count` 同构
    便于人读与对拍。
    """
    tasks: list[_Task] = []
    idx = 0
    for j, job in enumerate(inst.jobs):
        for i in range(len(job)):
            tasks.append(_Task(idx=idx, kind="arr", job=j, op=i,
                               o_op=(j, i - 1) if i > 0 else None, d_op=(j, i), rank=i))
            idx += 1
    if include_return:
        for j, job in enumerate(inst.jobs):
            tasks.append(_Task(idx=idx, kind="ret", job=j, op=len(job) - 1,
                               o_op=(j, len(job) - 1), d_op=None, rank=len(job)))
            idx += 1
    return tasks


# ══════════════════════ 模型 ══════════════════════

@dataclass
class ModelBundle:
    """建好的模型 + 抽取解所需的全部句柄（**不做求解**——求解在 `solve()`）。"""

    model: Any
    spec: ExactSpec
    inst: Instance
    scale: int
    tasks: list[_Task]
    n_tasks: int
    horizon: int                          # 时间上界（已缩放）
    op_start: list[list[Any]]             # [作业][工序] → IntVar
    op_end: list[list[Any]]
    op_mach: list[list[Any]]              # [作业][工序] → [BoolVar]，候选表同序
    op_node: list[list[Any]]              # [作业][工序] → IntVar（节点号 = 机台号+1）
    mv_start: list[Any]                   # [任务] → IntVar
    mv_end: list[Any]
    mv_dur: list[Any]                     # [任务] → IntVar（负载段时长 = T[o][d]）
    mv_orig: list[Any]                    # [任务] → IntVar（取货点节点号）
    mv_idx: list[Any]                     # [任务] → IntVar（o·n+d，Element 的下标）
    present: list[Any]                    # [任务] → BoolVar（1 = 该运输任务存在）
    skip: list[list[Any]]                 # [车][任务] → BoolVar（1 = 该车不服务此任务）
    serve: list[list[Any]]                # [车][任务] → BoolVar（= 1 − skip）
    cmax: Any
    n_arcs: int
    build_s: float
    # 回路句柄（提示 warm start 用）：每车的 S→t 弧 / 自环 / 闲置 / 任务间弧 / 空载量
    circuit: dict[str, Any] = field(default_factory=dict)

    def stats(self) -> dict[str, Any]:
        proto = self.model.proto
        return {"n_tasks": self.n_tasks, "n_arcs": self.n_arcs,
                "n_vars": len(proto.variables), "n_constraints": len(proto.constraints),
                "horizon": self.horizon, "scale": self.scale, "build_s": self.build_s}


def _job_lower_bound(inst: Instance, n_agv: int, include_return: bool) -> float:
    """构造式下界（作业侧 + 机台侧）——只为给 CP-SAT 一个起始对偶界，**不影响正确性**。

    - 作业侧：Σ 每工序最短加工时长（+ 若含回站，还要一趟从站出发、一趟回站的行程下界）。
      行程下界取矩阵里"站 → 任意机台"与"任意机台 → 站"的最小值——对任意调度都成立。
    - 机台侧：只能上某一台机台的工序，其加工时长之和 ≤ 该机台占用 ≤ Cmax。
    """
    lb = 0.0
    mat = inst.trans_time_full
    n = int(mat.shape[0])
    to_m = min(float(mat[0, v]) for v in range(1, n)) if n > 1 else 0.0
    from_m = min(float(mat[u, 0]) for u in range(1, n)) if n > 1 else 0.0
    for job in inst.jobs:
        lb = max(lb, sum(min(t for _, t in alts) for alts in job))
        if include_return:
            lb = max(lb, sum(min(t for _, t in alts) for alts in job) + to_m + from_m)
    solo: dict[int, float] = {}
    for job in inst.jobs:
        for alts in job:
            if len(alts) == 1:
                solo[alts[0][0]] = solo.get(alts[0][0], 0.0) + alts[0][1]
    if solo:
        lb = max(lb, max(solo.values()))
    return lb


def build_model(inst: Instance, spec: ExactSpec) -> ModelBundle:
    """建 CP-SAT 模型（不求解）。`inst` 必须是**矩阵口径**实例（`load_mkt(...).base`）。"""
    # ⚠️ 口径守卫放在 `_ortools()` **之前**：没装 ortools 的机器也要能测出这些错误，
    #    否则守卫测试会被 importorskip 一起跳过（那就没人钉住它们了）。
    if spec.tier not in REPORT_TIERS:
        raise ValueError(f"未知报告档 {spec.tier!r}；可选 {sorted(REPORT_TIERS)}")
    if spec.tier != "A-MKT":
        raise ValueError(
            f"精确解基线只支持 A-MKT 档（十机制全关）；收到 {spec.tier!r}。"
            "B-8 档开着随机机制（③ 故障、④ 返工、⑨ AGV 故障），**不是确定性优化问题**"
            "——没有单点最优解可证。")
    if getattr(inst, "transport", None) != "matrix" or inst.trans_time_full is None:
        raise ValueError("精确解基线要求矩阵口径实例（用 `plcsp.env.mkt.load_mkt(...).base`）")
    if spec.n_agv < 1:
        raise ValueError(f"n_agv 必须 ≥ 1，收到 {spec.n_agv}")
    if spec.initial_travel not in (INIT_FREE, INIT_LU):
        raise ValueError(f"未知 initial_travel={spec.initial_travel!r}（只认 {INIT_FREE}/{INIT_LU}）")

    cp_model = _ortools()
    t0 = time.perf_counter()
    scale = time_scale(inst)
    T = travel_matrix(inst)
    n_node = len(T)                       # 含装卸站：(m+1)
    free_idx = n_node                     # 合成节点："车在首趟取货点待命"（用时恒 0）
    flat = [T[u][v] for u in range(n_node) for v in range(n_node)] + [0] * n_node
    m = inst.n_machines
    tasks = _task_table(inst, spec.include_return)
    nT = len(tasks)

    # ── 地平线：上界（覆盖 J*，因为 J* ≤ 规则基线）──
    ub = spec.upper_bound if spec.upper_bound is not None else _crude_upper_bound(inst, spec)
    horizon = int(math.ceil(ub * scale)) + 1

    model = cp_model.CpModel()

    # ── 加工侧：每工序一组候选机台（NCP2 (19) alternative）──
    op_start: list[list[Any]] = []
    op_end: list[list[Any]] = []
    op_mach: list[list[Any]] = []
    op_dur: list[list[Any]] = []
    op_node: list[list[Any]] = []
    mach_intervals: dict[int, list[Any]] = {mm: [] for mm in range(m)}
    for j, job in enumerate(inst.jobs):
        s_row, e_row, y_row, d_row, n_row = [], [], [], [], []
        for i, alts in enumerate(job):
            s = model.new_int_var(0, horizon, f"s_{j}_{i}")
            e = model.new_int_var(0, horizon, f"e_{j}_{i}")
            dur = model.new_int_var(1, horizon, f"d_{j}_{i}")
            y_alt = [model.new_bool_var(f"y_{j}_{i}_{mm}") for mm, _ in alts]
            for (mm, p), y in zip(alts, y_alt):
                # ⚠️ 所有候选**共用** (s, e)：y 为真时 CP-SAT 保证 e = s + p（整数）
                mach_intervals[mm].append(
                    model.new_optional_interval_var(s, int(round(p * scale)), e, y, f"iv_{j}_{i}_{mm}"))
            model.add_exactly_one(y_alt)
            p_scaled = [int(round(p * scale)) for _, p in alts]
            model.add(dur == cp_model.LinearExpr.sum(
                [int(pp) * y for pp, y in zip(p_scaled, y_alt)]))
            model.add(e == s + dur)
            node = model.new_int_var(1, m, f"nd_{j}_{i}")   # 节点号 = **机台号 + 1**（矩阵口径）
            # ⚠️ 必须用**机台号**（`alts[k][0]`），不是候选表的下标——两者只在候选表从机台 0
            #    连续开始时才碰巧相等（bug#14：B 实例的单候选 [(1,10)] 曾被算成节点 1 = 机台 0）
            model.add(node == cp_model.LinearExpr.sum(
                [(mm + 1) * y for mm, y in zip([a[0] for a in alts], y_alt)]))
            s_row.append(s), e_row.append(e), y_row.append(y_alt)
            d_row.append(dur), n_row.append(node)
        op_start.append(s_row), op_end.append(e_row), op_mach.append(y_row)
        op_dur.append(d_row), op_node.append(n_row)
    for mm in range(m):                                     # (20)(21) noOverlap
        model.add_no_overlap(mach_intervals[mm])

    # ── 运输侧：存在性 + 时长 ──
    mv_start, mv_end, present, origin, dest = [], [], [], [], []
    mv_dur, mv_orig, mv_idx = [], [], []
    for t in tasks:
        pv = model.new_bool_var(f"pres_{t.idx}")
        o_expr = 0 if t.o_op is None else op_node[t.o_op[0]][t.o_op[1]]
        d_expr = 0 if t.d_op is None else op_node[t.d_op[0]][t.d_op[1]]
        if t.kind == "ret" or t.op == 0 or spec.same_machine_transport:
            model.add(pv == 1)                              # 恒存在
        else:
            # NCP2 (24)：同机台 ⟹ 无运输任务。
            # ⚠️ 按**机台号**取交集比较（不是候选表下标）——两张候选表可以不同长、不同序
            #    （bug#15：按下标 zip 会把"机台 2"和"机台 3"当成同一个）。
            prev_map = {mm: y for (mm, _), y in zip(inst.jobs[t.job][t.op - 1], op_mach[t.job][t.op - 1])}
            cur_map = {mm: y for (mm, _), y in zip(inst.jobs[t.job][t.op], op_mach[t.job][t.op])}
            common = sorted(set(prev_map) & set(cur_map))
            if not common:
                model.add(pv == 1)               # 候选无交集 ⟹ 必异机台 ⟹ 必存在
            else:
                same_bits = []
                for mm in common:
                    z = model.new_bool_var(f"same_{t.idx}_{mm}")
                    model.add_bool_and([prev_map[mm], cur_map[mm]]).only_enforce_if(z)
                    model.add_bool_or([prev_map[mm].Not(), cur_map[mm].Not()]).only_enforce_if(z.Not())
                    same_bits.append(z)
                same = model.new_bool_var(f"sameany_{t.idx}")
                model.add_bool_or(same_bits).only_enforce_if(same)
                for b in same_bits:
                    model.add_implication(b, same)
                model.add(pv + same == 1)
        ms = model.new_int_var(0, horizon, f"ms_{t.idx}")
        me = model.new_int_var(0, horizon, f"me_{t.idx}")
        dr = model.new_int_var(0, horizon, f"dt_{t.idx}")
        o_var = model.new_int_var(0, n_node - 1, f"o_{t.idx}")
        model.add(o_var == o_expr)
        # 负载段时长 = T[o][d]（对角线已置 0 ⟹ 同点出行 0 分钟，与 DES `_drive` 一致）
        idx_var = model.new_int_var(0, n_node * n_node - 1, f"oi_{t.idx}")
        model.add(idx_var == o_var * n_node + d_expr)
        model.add_element(idx_var, flat[:n_node * n_node], dr)
        model.add(me == ms + dr)
        # (28) 运输先于加工：(29) 运输后于上一工序完工
        # ⚠️ (28)(29) 都必须**挂在存在性上**：同机台的运输任务**不存在**，其 `ms`/`me` 是
        #   占位值（构造解与提示里取 0）。若无条件加 (29)，不存在的任务会被迫 `ms ≥ E_prev`
        #   ⟹ 构造解与模型对不上（bug#16：warm start 因此被整支否掉，白丢一个初始解）。
        #   任务不存在时工序之间只剩 (30) `S ≥ E_prev`，那才是 NCP2 §IV.D 的原意。
        if t.kind == "arr":
            model.add(op_start[t.job][t.op] >= me).only_enforce_if(pv)          # (28)
            if t.op > 0:
                model.add(ms >= op_end[t.job][t.op - 1]).only_enforce_if(pv)    # (29)
        else:                                               # 回站段恒存在 ⟹ 无需挂条件
            model.add(ms >= op_end[t.job][t.op])
        mv_start.append(ms), mv_end.append(me), present.append(pv)
        mv_dur.append(dr), mv_orig.append(o_var), mv_idx.append(idx_var)
        origin.append(o_var), dest.append(d_expr)

    # ── 工序之间的直接先后（NCP2 (30)）：同机台时是唯一连接，异机台时被 (28)(29) 蕴含 ──
    for j, job in enumerate(inst.jobs):
        for i in range(1, len(job)):
            model.add(op_start[j][i] >= op_end[j][i - 1])

    # ── 车侧：每车一张回路（弧 = "b 紧接在 a 之后"），自环 = "这趟活不归我" ──
    n_arcs = 0
    skip = [[None] * nT for _ in range(spec.n_agv)]
    serve = [[None] * nT for _ in range(spec.n_agv)]
    first: list[Any] = []
    heads: list[list[Any]] = []                             # 提示（warm start）要用
    pd_all: list[list[Any]] = []
    tt_all: list[list[Any]] = []
    arc_lit: list[dict[tuple[int, int], Any]] = [{} for _ in range(spec.n_agv)]
    idle_vars: list[Any] = []
    tail_all: list[list[Any]] = []
    for k in range(spec.n_agv):
        s_node, e_node = 0, nT + 1                          # 本车专用编号空间
        arcs: list[tuple[int, int, Any]] = []
        head = [model.new_bool_var(f"st_{k}_{t}") for t in range(nT)]   # S→t
        slf = [model.new_bool_var(f"sk_{k}_{t}") for t in range(nT)]    # 自环 = 不服务
        tail = [model.new_bool_var(f"en_{k}_{t}") for t in range(nT)]   # t→E
        idle = model.new_bool_var(f"idle_{k}")
        pd = [model.new_int_var(0, n_node, f"pd_{k}_{t}") for t in range(nT)]   # 取货前车位置
        tt = [model.new_int_var(0, horizon, f"tt_{k}_{t}") for t in range(nT)]  # 空载时长
        heads.append(head), pd_all.append(pd), tt_all.append(tt)
        idle_vars.append(idle), tail_all.append(tail)
        for t in range(nT):
            skip[k][t] = slf[t]
            serve[k][t] = slf[t].Not()
            arcs.append((s_node, t + 1, head[t]))
            arcs.append((t + 1, e_node, tail[t]))
            arcs.append((t + 1, t + 1, slf[t]))
            # 空载 = T[车上一趟卸货点][本趟取货点]；free 起点用合成行（恒 0）
            model.add_element(pd[t] * n_node + origin[t], flat, tt[t])
            first_pd = free_idx if spec.initial_travel == INIT_FREE else 0
            model.add(pd[t] == first_pd).only_enforce_if(head[t])
            # 首趟任务也要付空载（`free` 行恒 0 ⟹ 该约束自动成立；`lu` 下才有内容）
            model.add(mv_start[t] >= tt[t]).only_enforce_if(head[t])
        arcs.append((s_node, e_node, idle))
        arcs.append((e_node, s_node, True))
        for a in range(nT):
            for b in range(nT):
                if a == b:
                    continue
                ta, tb = tasks[a], tasks[b]
                if ta.job == tb.job and ta.rank >= tb.rank:
                    continue                                # 同作业逆序不可能：车序不可违反作业先后
                lit = model.new_bool_var(f"arc_{k}_{a}_{b}")
                arcs.append((a + 1, b + 1, lit))
                arc_lit[k][(a, b)] = lit
                # 弧为真 ⟹ 车从 a 的卸货点出发、且 a 卸完并空驶到 b 的取货点后才能装 b
                model.add(pd[b] == dest[a]).only_enforce_if(lit)
                model.add(mv_end[a] + tt[b] <= mv_start[b]).only_enforce_if(lit)
                if spec.exact_deadhead:
                    # DES 逐字：空载从"任务可出发"起算（车若更早空着，同样从这里起算）
                    model.add(mv_start[b] >= _ready_var(tb, op_end) + tt[b]).only_enforce_if(lit)
                n_arcs += 1
        model.add_circuit(arcs)
        # 首趟任务序号（+1；0 = 本车闲置）——对称破缺用
        fst = model.new_int_var(0, nT, f"first_{k}")
        model.add(fst == cp_model.LinearExpr.sum([(t + 1) * head[t] for t in range(nT)]))
        first.append(fst)

    # (25)：存在的运输任务恰由一台车服务
    for t in range(nT):
        model.add(cp_model.LinearExpr.sum([serve[kk][t] for kk in range(spec.n_agv)])
                  == present[t])

    # ── 对称破缺：同型车可互换，按"首趟任务序号"排序（0 = 空闲）──
    # 合法：任意解的车辆按 fst 重排后仍可行（车队同型、初始待命位置口径相同）。
    if spec.sym_break:
        for k in range(spec.n_agv - 1):
            model.add(first[k] <= first[k + 1])

    # ── 目标：makespan = 各作业回站/末工序完完工时刻的最大值 ──
    if spec.include_return:
        comp = [mv_end[t.idx] for t in tasks if t.kind == "ret"]
    else:
        comp = [op_end[j][len(job) - 1] for j, job in enumerate(inst.jobs)]
    cmax = model.new_int_var(0, horizon, "cmax")
    model.add_max_equality(cmax, comp)
    lb_scaled = int(math.floor(_job_lower_bound(inst, spec.n_agv, spec.include_return) * scale))
    if lb_scaled > 0:
        model.add(cmax >= lb_scaled)
    model.minimize(cmax)

    bundle = ModelBundle(
        model=model, spec=spec, inst=inst, scale=scale, tasks=tasks, n_tasks=nT,
        horizon=horizon, op_start=op_start, op_end=op_end, op_mach=op_mach, op_node=op_node,
        mv_start=mv_start, mv_end=mv_end, present=present, skip=skip, serve=serve,
        mv_dur=mv_dur, mv_orig=mv_orig, mv_idx=mv_idx,
        cmax=cmax, n_arcs=n_arcs, build_s=time.perf_counter() - t0,
        circuit={"head": heads, "tail": tail_all, "skip": skip, "idle": idle_vars,
                 "arc": arc_lit, "pd": pd_all, "tt": tt_all})
    return bundle


def _ready_var(tb: _Task, op_end: list[list[Any]]) -> Any:
    """任务 b 的"可出发时刻" = 它服务的那道工序的**上一道工序**完工时刻（无则 0）。

    运输任务 b 服务于工序 (j,i)：b 的取货点就是工序 i−1 所在的机台（i=0 时为装卸站），
    故"任务可出发"= `op_end[j][i−1]`（i=0 → 0，恒可出发）。回站任务的取货点是末工序所在机台。
    """
    if tb.kind == "ret":
        return op_end[tb.job][tb.op]
    if tb.op == 0:
        return 0
    return op_end[tb.job][tb.op - 1]


def _node_of(inst: Instance, plan: list[list[tuple[int, float]]],
             op: tuple[int, int] | None) -> int:
    """工序 → 行程时间矩阵下标（机台 m ↦ m+1）；`None`（= 装卸站）↦ 0。"""
    return 0 if op is None else plan[op[0]][op[1]][0] + 1


def greedy_schedule(inst: Instance, n_agv: int, *, include_return: bool = True,
                    same_machine_transport: bool = False, initial_travel: str = INIT_FREE,
                    choices: Sequence[Sequence[int]] | None = None) -> dict[str, Any]:
    """**构造式列表调度**：给 CP-SAT 一个可行初始解（warm start）与一个紧地平线。

    规则（全部确定性，无随机数）：
    - 机台选择 = 该工序**最短加工时长**的候选（与 DES 的缺省 `op_choices` 同款）；`choices` 可覆盖。
    - 反复挑"**最早能完工**"的下一条运输任务：可选车、可选起始时刻 = `max(车空闲, 任务可出发)
      + 空载`（首趟任务空载 = 0，与 DES 的 `pos_node is None` 同款）。
    - 到达任务的工序随即上机：`S = max(运输完, 机台空闲)`；回站任务即作业完工。

    返回值形状与 `extract_schedule` 一致，可直接过 `validate_schedule`。
    ⚠️ 它只是**启发式**：**不保证最优**，也**不参与下界**——只用来给求解器一个起点。
    """
    mat = inst.trans_time_full

    def tv(u: int, v: int) -> float:
        return 0.0 if u == v else float(mat[u, v])

    m = inst.n_machines
    plan: list[list[tuple[int, float]]] = []
    for j, job in enumerate(inst.jobs):
        row = []
        for i, alts in enumerate(job):
            k = (min(range(len(alts)), key=lambda q: alts[q][1])
                 if choices is None else int(choices[j][i]))
            k = min(max(k, 0), len(alts) - 1)
            row.append((int(alts[k][0]), float(alts[k][1])))
        plan.append(row)

    tasks = _task_table(inst, include_return)
    route: dict[int, list[int]] = {j: [t.idx for t in tasks if t.job == j]
                                   for j in range(inst.n_jobs)}
    op_s = [[0.0] * len(job) for job in inst.jobs]
    op_e = [[0.0] * len(job) for job in inst.jobs]
    mach_free = [0.0] * m
    agv_free = [0.0] * n_agv
    agv_node: list[int | None] = [None] * n_agv
    info: list[dict[str, Any] | None] = [None] * len(tasks)
    pos = {j: 0 for j in range(inst.n_jobs)}

    def ready_of(t: _Task) -> float:
        if t.kind == "ret":
            return op_e[t.job][t.op]
        return 0.0 if t.op == 0 else op_e[t.job][t.op - 1]

    def start_op(t: _Task, arrive: float) -> float:
        mm, p = plan[t.job][t.op]
        st = max(arrive, mach_free[mm])
        op_s[t.job][t.op] = st
        op_e[t.job][t.op] = st + p
        mach_free[mm] = st + p
        return st + p

    while any(pos[j] < len(route[j]) for j in range(inst.n_jobs)):
        best: tuple[float, int, int, float, float, bool] | None = None
        for j in range(inst.n_jobs):
            p = pos[j]
            if p >= len(route[j]):
                continue
            t = tasks[route[j][p]]
            origin = _node_of(inst, plan, t.o_op)
            dest = _node_of(inst, plan, t.d_op)
            # NCP2：同机台相邻工序 ⟹ 该运输任务不存在（零时长交接，不占车）
            present = not (t.kind == "arr" and t.op > 0 and not same_machine_transport
                           and plan[t.job][t.op - 1][0] == plan[t.job][t.op][0])
            rdy = ready_of(t)
            if not present:
                fin = max(rdy, mach_free[plan[t.job][t.op][0]]) + plan[t.job][t.op][1]
                cand = (fin, -1, t.idx, rdy, 0.0, False)
            else:
                d = int(round(tv(origin, dest)))
                pick = None
                for k in range(n_agv):
                    if agv_node[k] is None:
                        # 首趟任务：free = 车就等在取货点（DES）；lu = 车在装卸站（论文）
                        dh = 0.0 if initial_travel == INIT_FREE else tv(0, origin)
                    else:
                        dh = tv(agv_node[k], origin)
                    st = max(agv_free[k], rdy) + dh
                    fin = st + d
                    if pick is None or fin < pick[0]:
                        pick = (fin, k, st)
                assert pick is not None
                cand = (pick[0], pick[1], t.idx, pick[2], d, True)
            if best is None or cand[0] < best[0]:
                best = cand
        assert best is not None
        fin, k, tid, st, dur, present = best
        t = tasks[tid]
        if present:
            agv_free[k] = fin
            agv_node[k] = _node_of(inst, plan, t.d_op)
            info[tid] = {"idx": tid, "kind": t.kind, "job": t.job, "op": t.op,
                         "origin": _node_of(inst, plan, t.o_op),
                         "dest": _node_of(inst, plan, t.d_op), "present": 1, "agv": k,
                         "start": st, "end": fin}
        else:
            info[tid] = {"idx": tid, "kind": t.kind, "job": t.job, "op": t.op,
                         "origin": _node_of(inst, plan, t.o_op),
                         "dest": _node_of(inst, plan, t.d_op), "present": 0, "agv": -1,
                         "start": 0.0, "end": 0.0}
        # ⚠️ **只有到达任务才排工序**：回站任务的 `op` 是末工序，再排一次会把末工序的完工
        #    时刻推到回站之后（构造解自相矛盾——由 `validate_schedule` 当场抓住）。
        if t.kind == "arr":
            start_op(t, fin if present else ready_of(t))
        pos[t.job] += 1
    # 完工 = 回站到达；不含回站段时 = 末工序下机（与 `build_model` 的目标口径一致）
    completes = ([max(i["end"] for i in info if i is not None and i["kind"] == "ret"
                      and i["job"] == j) for j in range(inst.n_jobs)]
                 if include_return else [op_e[j][-1] for j in range(inst.n_jobs)])
    return {"ops": [[[plan[j][i][0], op_s[j][i], op_e[j][i]] for i in range(len(job))]
                    for j, job in enumerate(inst.jobs)],
            "tasks": [i for i in info if i is not None],
            "completes": completes, "makespan": max(completes)}


def hint_schedule(bundle: ModelBundle, sched: dict[str, Any]) -> None:
    """把一份**构造解**灌成模型提示（`add_hint`，**不是约束**）。

    ⚠️ **必须把每个变量都提示到**：CP-SAT 会打印 "The solution hint is incomplete: X out of Y
    non fixed variables hinted"，**不完整的提示它不拿来当初始解**（只喂给一个 repair 搜索，
    实测在大实例上等于没用）。故本函数逐个变量给值，包括 O(T²) 条弧（未选中的显式提示 0）。

    提示不成立时 CP-SAT 只是忽略它 ⟹ 本函数**只影响速度，不影响正确性**。
    两处对齐：① 车的编号按 `first[k]`（= 首趟任务序号 +1）升序重排——与 `sym_break` 的约束
    同序，否则提示会被那条约束整个否掉；② `pd`/`tt` 按同一套位置口径给值（首趟取决于
    `initial_travel`）。
    """
    model, s = bundle.model, bundle.scale
    inst = bundle.inst
    n_node = inst.n_machines + 1
    # ── 加工侧：机台选择 / 起止 / 时长 / 节点号 ──
    for j, job in enumerate(inst.jobs):
        for i, alts in enumerate(job):
            mm = int(sched["ops"][j][i][0])
            for k, y in enumerate(bundle.op_mach[j][i]):
                model.add_hint(y, 1 if int(alts[k][0]) == mm else 0)
            model.add_hint(bundle.op_start[j][i], int(round(sched["ops"][j][i][1] * s)))
            model.add_hint(bundle.op_end[j][i], int(round(sched["ops"][j][i][2] * s)))
            model.add_hint(bundle.op_node[j][i], mm + 1)
    # ── 运输侧：存在性 / 起止 / 时长 / 取货点 / Element 下标 ──
    served: dict[int, list[int]] = {}
    dest_of: dict[int, int] = {}
    org_of: dict[int, int] = {}
    start_of: dict[int, float] = {}
    for t in sched["tasks"]:
        ti = t["idx"]
        org, dst = int(t["origin"]), int(t["dest"])
        org_of[ti], dest_of[ti], start_of[ti] = org, dst, float(t["start"])
        model.add_hint(bundle.present[ti], int(t["present"]))
        model.add_hint(bundle.mv_start[ti], int(round(t["start"] * s)))
        model.add_hint(bundle.mv_end[ti], int(round(t["end"] * s)))
        model.add_hint(bundle.mv_dur[ti], int(round((t["end"] - t["start"]) * s)))
        model.add_hint(bundle.mv_orig[ti], org)
        model.add_hint(bundle.mv_idx[ti], org * n_node + dst)
        if t["present"] == 1 and t["agv"] >= 0:
            served.setdefault(int(t["agv"]), []).append(ti)
    # ── 车的编号按 `first[k]`（首趟任务序号 +1；空闲 = 0）升序——与 `sym_break` 同序 ──
    seq_by_old = {k: sorted(tis, key=lambda ti: start_of[ti]) for k, tis in served.items()}
    order = sorted(range(bundle.spec.n_agv),
                   key=lambda k: (seq_by_old[k][0] + 1) if seq_by_old.get(k) else 0)
    relabel = {old: new for new, old in enumerate(order)}
    seq_by_new = {relabel[old]: tis for old, tis in seq_by_old.items()}
    # ── 车侧：回路（含 O(T²) 条弧的显式 0）/ 自环 / 首末弧 / 空载 ──
    start_node = 0 if bundle.spec.initial_travel == INIT_LU else n_node     # n_node = free 行
    for k in range(bundle.spec.n_agv):
        seq = seq_by_new.get(k, [])
        ins = set(seq)
        model.add_hint(bundle.circuit["idle"][k], 0 if seq else 1)
        for t in range(bundle.n_tasks):
            model.add_hint(bundle.skip[k][t], 0 if t in ins else 1)
            model.add_hint(bundle.circuit["head"][k][t], 1 if (seq and seq[0] == t) else 0)
            model.add_hint(bundle.circuit["tail"][k][t], 1 if (seq and seq[-1] == t) else 0)
        nxt = {a: b for a, b in zip(seq, seq[1:])}
        for (a, b), lit in bundle.circuit["arc"][k].items():
            model.add_hint(lit, 1 if nxt.get(a) == b else 0)
        for idx, ti in enumerate(seq):
            cur = start_node if idx == 0 else dest_of[seq[idx - 1]]
            model.add_hint(bundle.circuit["pd"][k][ti], cur)
            model.add_hint(bundle.circuit["tt"][k][ti],
                           0 if cur == n_node else int(round(float(
                               inst.trans_time_full[cur, org_of[ti]] * s))))
    model.add_hint(bundle.cmax, int(round(float(sched["makespan"]) * s)))


def _crude_upper_bound(inst: Instance, spec: ExactSpec) -> float:
    """构造式上界（不跑仿真时的兜底）：串行做完所有活 + 全部运输的时长之和。

    ⚠️ 它**只当地平线**（必须 ≥ J*，否则会切掉最优解，下界性就没了）。故取一个显然够大的量。
    """
    T = travel_matrix(inst)
    tmax = max(max(r) for r in T)
    proc = sum(max(t for _, t in alts) for job in inst.jobs for alts in job)
    n_tasks = sum(len(job) for job in inst.jobs) + (len(inst.jobs) if spec.include_return else 0)
    return float(proc + 2 * n_tasks * tmax + 1)


def rule_upper_bound(inst: Instance, spec: ExactSpec, *, seed_chain: int = 0) -> dict[str, Any]:
    """跑一次**规则基线**（`rollout`，与本仓基线同一入口），拿它的 makespan 当地平线/参照。

    ⚠️ 它同时是"认证间隙"的分母侧读数：`gap = (J_rule − J*)/J*`。但它**不是**下界——
    只有 `J*` 才是。故本函数的结果只作参照报出。
    """
    from ..env.des import SimConfig, rollout
    r = rollout(inst, seed_chain=seed_chain,
                cfg=SimConfig(n_agv=spec.n_agv, transport_unmapped=spec.transport_unmapped),
                constraints=REPORT_TIERS[spec.tier])
    return {"makespan": float(r["makespan"]), "horizon_hit": bool(r["horizon_hit"]),
            "tasks_put": int(r["dbg"]["tasks_put"]), "unmapped_legs": int(r["unmapped_legs"])}


# ══════════════════════ 求解 ══════════════════════

@dataclass
class ExactResult:
    """一次精确求解的读数。**objective 是上界、dual_bound 是下界**，两者相等才算证明最优。"""

    instance: str
    tier: str
    n_agv: int
    status: str
    objective: float | None          # 最好可行解（None = 一个都没有）
    dual_bound: float | None         # CP-SAT 的对偶界（**认证的下界**）
    wall_s: float
    time_limit_s: float
    scale: int
    n_tasks: int
    n_arcs: int
    build_s: float
    horizon: float
    rule_makespan: float | None = None
    rule_horizon_hit: bool | None = None
    greedy_makespan: float | None = None     # 构造式解（warm start 的来源）——不是下界
    schedule: dict[str, Any] | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def gap_abs(self) -> float | None:
        if self.objective is None or self.dual_bound is None:
            return None
        return max(0.0, self.objective - self.dual_bound)

    @property
    def gap_rel(self) -> float | None:
        """自证间隙 `(UB − LB)/UB`——0 表示**已证明最优**。"""
        if self.objective is None or self.gap_abs is None or self.objective <= 0:
            return None
        return self.gap_abs / self.objective

    def certified_gap_vs(self, policy_makespan: float) -> float | None:
        """认证间隙 `(J_policy − LB)/LB`——用**对偶界**而非最好解，故对策略永远成立。"""
        if self.dual_bound is None or self.dual_bound <= 0:
            return None
        return (policy_makespan - self.dual_bound) / self.dual_bound

    def summary(self) -> str:
        o = "—" if self.objective is None else f"{self.objective:.2f}"
        b = "—" if self.dual_bound is None else f"{self.dual_bound:.2f}"
        g = "—" if self.gap_rel is None else f"{self.gap_rel * 100:.2f}%"
        lines = [
            f"精确解基线（CP-SAT，{self.tier} 档）｜实例={self.instance}｜车数 v={self.n_agv}",
            f"状态={self.status}｜最好解 UB={o}｜对偶界 LB={b}｜自证间隙={g}"
            f"｜墙钟={self.wall_s:.1f} s（上限 {self.time_limit_s:.0f} s）",
            f"模型：运输任务 {self.n_tasks}｜弧 {self.n_arcs}｜地平线 {self.horizon:.0f}"
            f"｜缩放 1/{self.scale}｜建模 {self.build_s:.1f} s",
        ]
        if self.greedy_makespan is not None:
            lines.append(f"构造式 warm start（启发式上界，**不是下界**）={self.greedy_makespan:.2f}")
        if self.rule_makespan is not None:
            cg = self.certified_gap_vs(self.rule_makespan)
            lines.append(f"规则基线 makespan={self.rule_makespan:.2f}"
                         f"（掐表={self.rule_horizon_hit}）"
                         f"｜认证间隙 (J_rule − LB)/LB="
                         + ("—" if cg is None else f"{cg * 100:.2f}%"))
        return "\n".join(lines)


def solve(inst: Instance, spec: ExactSpec, *, name: str = "",
          rule_bound: bool = True, warm_start: bool = True,
          want_schedule: bool = True) -> ExactResult:
    """建模型 → 跑 CP-SAT → 抽取解与对偶界。

    地平线（= 时间的上界）取**三个上界里最小的**：构造式粗界、`greedy_schedule`、规则基线。
    三者都是**真调度**的 makespan ⟹ 都 ≥ J*，用它当地平线不会切掉最优解。
    `warm_start=True` 时还把构造解灌成 `add_hint`（只是提示，不改变可行集）。
    """
    cp_model = _ortools()
    rule_ms: float | None = None
    rule_hit: bool | None = None
    upper = spec.upper_bound

    greedy: dict[str, Any] | None = None
    if warm_start or upper is None:
        g = greedy_schedule(inst, spec.n_agv, include_return=spec.include_return,
                            same_machine_transport=spec.same_machine_transport,
                            initial_travel=spec.initial_travel)
        try:                                    # 构造解必须自己先过独立校验器，否则不采信
            validate_schedule(inst, g, spec.n_agv,
                              same_machine_transport=spec.same_machine_transport)
        except AssertionError as e:             # pragma: no cover - 构造式的自检
            raise RuntimeError(f"构造解未通过独立校验（这是本模块的 bug）：{e}") from e
        greedy = g
        if upper is None:
            upper = float(g["makespan"])
    if rule_bound:
        rb = rule_upper_bound(inst, spec)
        rule_ms, rule_hit = rb["makespan"], rb["horizon_hit"]
        if not rule_hit and upper is not None:
            upper = min(upper, float(rb["makespan"]))
    eff = _with_ub(spec, upper) if upper is not None else spec
    bundle = build_model(inst, eff)
    if warm_start and greedy is not None:
        hint_schedule(bundle, greedy)
        # 构造解的 makespan 是**真上界** ⟹ 加成硬约束不切最优解，但能立刻收紧目标界
        bundle.model.add(bundle.cmax <= int(math.ceil(float(greedy["makespan"]) * bundle.scale)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(spec.time_limit_s)
    solver.parameters.num_workers = int(max(1, spec.workers))
    solver.parameters.random_seed = int(spec.random_seed)
    t0 = time.perf_counter()
    st = solver.solve(bundle.model)
    wall = time.perf_counter() - t0
    status = solver.status_name(st)

    objective = None
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        objective = float(solver.objective_value) / bundle.scale
    # ⚠️ **对偶界在 UNKNOWN 下同样有效**（CP-SAT 会报当前证明的下界）——它是本模块的主产物，
    #    绝不能因为"还没找到可行解"就不报（那会把唯一可用的认证读数丢掉）。
    try:
        bound = float(solver.best_objective_bound) / bundle.scale
    except (RuntimeError, ValueError):                          # pragma: no cover - 求解器未启动
        bound = None
    if st == cp_model.INFEASIBLE:
        bound = None

    sched = None
    if want_schedule and objective is not None:
        sched = extract_schedule(bundle, solver)

    res = ExactResult(
        instance=name or getattr(inst, "source", ""), tier=spec.tier, n_agv=spec.n_agv,
        status=status, objective=objective, dual_bound=bound, wall_s=wall,
        time_limit_s=float(spec.time_limit_s), scale=bundle.scale, n_tasks=bundle.n_tasks,
        n_arcs=bundle.n_arcs, build_s=bundle.build_s, horizon=float(bundle.horizon),
        rule_makespan=rule_ms, rule_horizon_hit=rule_hit, schedule=sched,
        model_kwargs={"include_return": spec.include_return,
                      "same_machine_transport": spec.same_machine_transport,
                      "initial_travel": spec.initial_travel,
                      "exact_deadhead": spec.exact_deadhead, "sym_break": spec.sym_break})
    res.greedy_makespan = None if greedy is None else float(greedy["makespan"])
    return res


def _with_ub(spec: ExactSpec, ub: float) -> ExactSpec:
    from dataclasses import replace
    return replace(spec, upper_bound=float(ub))


# ══════════════════════ 抽解 + 独立校验 ══════════════════════

def extract_schedule(bundle: ModelBundle, solver: Any) -> dict[str, Any]:
    """从已求解的 CP-SAT 模型抽出调度（单位 = **本仓时间**，已除以缩放）。"""
    s = bundle.scale
    inst = bundle.inst
    ops: list[list[list[float | int]]] = []
    for j, job in enumerate(inst.jobs):
        row = []
        for i, alts in enumerate(job):
            y = bundle.op_mach[j][i]
            k = next((idx for idx, v in enumerate(y) if solver.value(v) == 1), None)
            if k is None:                                    # pragma: no cover - 解必给完整指派
                raise RuntimeError(f"解里作业 {j} 工序 {i} 没有选中任何机台")
            row.append([int(alts[k][0]), solver.value(bundle.op_start[j][i]) / s,
                        solver.value(bundle.op_end[j][i]) / s])
        ops.append(row)
    tasks = []
    for t in bundle.tasks:
        agv = next((k for k in range(bundle.spec.n_agv)
                    if solver.value(bundle.serve[k][t.idx]) == 1), None)
        mch = -1 if t.d_op is None else int(ops[t.d_op[0]][t.d_op[1]][0])
        tasks.append({"idx": t.idx, "kind": t.kind, "job": t.job, "op": t.op,
                      "origin": 0 if t.o_op is None else int(ops[t.o_op[0]][t.o_op[1]][0]) + 1,
                      "dest": 0 if t.d_op is None else mch + 1,
                      "present": int(solver.value(bundle.present[t.idx])),
                      "agv": -1 if agv is None else agv,
                      "start": solver.value(bundle.mv_start[t.idx]) / s,
                      "end": solver.value(bundle.mv_end[t.idx]) / s})
    comp = ([t for t in tasks if t["kind"] == "ret"] if bundle.spec.include_return
            else [{"job": j, "end": ops[j][-1][2]} for j in range(inst.n_jobs)])
    return {"ops": ops, "tasks": tasks,
            "completes": [max(t["end"] for t in comp if t["job"] == j)
                          for j in range(inst.n_jobs)],
            "makespan": solver.value(bundle.cmax) / s}


def validate_schedule(inst: Instance, sched: dict[str, Any], n_agv: int, *,
                      same_machine_transport: bool = False) -> float:
    """**独立**校验一份调度（不依赖模型）：满足则返回 makespan，违反则**显式报错**。

    逐条核对 DES 的语义（`des.AgvSim` / `des._transporter` / `des.LuStation`）：
    工序候选合法、机台不重叠、作业先后、运输行程 = 矩阵查表、AGV 不重叠且空载 = 上一卸货点
    到本趟取货点的行程、完工 = 回站到达。任何一条不成立都抛错——不静默。
    """
    tol = 1e-6
    T = inst.trans_time_full
    m = inst.n_machines
    ops = sched["ops"]
    errs: list[str] = []

    def tv(u: int, v: int) -> float:
        return 0.0 if u == v else float(T[u, v])

    # 1) 候选合法 + 时长
    for j, job in enumerate(inst.jobs):
        for i, alts in enumerate(job):
            mm, s, e = ops[j][i]
            cand = dict((int(a), float(p)) for a, p in alts)
            if mm not in cand:
                errs.append(f"作业{j}工序{i} 选了非候选机台 {mm}")
            elif abs((e - s) - cand[mm]) > tol:
                errs.append(f"作业{j}工序{i} 时长不符：{e - s} ≠ {cand[mm]}")
            if s < -tol or e < s - tol:
                errs.append(f"作业{j}工序{i} 时间非法：s={s} e={e}")

    # 2) 作业先后
    for j, job in enumerate(inst.jobs):
        for i in range(1, len(job)):
            if ops[j][i][1] < ops[j][i - 1][2] - tol:
                errs.append(f"作业{j}工序{i} 早于上一工序完工")

    # 3) 机台不重叠
    for mm in range(m):
        used = sorted((ops[j][i][1], ops[j][i][2], j, i)
                      for j, job in enumerate(inst.jobs) for i in range(len(job))
                      if ops[j][i][0] == mm)
        for (s1, e1, j1, i1), (s2, e2, j2, i2) in zip(used, used[1:]):
            if s2 < e1 - tol:
                errs.append(f"机台{mm} 重叠：({j1},{i1}) {s1}-{e1} 与 ({j2},{i2}) {s2}-{e2}")

    # 4) 运输任务：存在性判据 + 行程 + 与加工任务的时间耦合（NCP2 (24)(28)(29)(30)）
    by_job: dict[int, list[dict[str, Any]]] = {}
    for t in sched["tasks"]:
        by_job.setdefault(t["job"], []).append(t)
        # 存在判据（NCP2 (22)(24)）：投放段与回站段恒存在；工序流转段**异机台才存在**
        if t["kind"] == "ret" or t["op"] == 0 or same_machine_transport:
            want = True
        else:
            want = ops[t["job"]][t["op"] - 1][0] != ops[t["job"]][t["op"]][0]
        if want and t["present"] != 1:
            errs.append(f"任务{t['idx']} 应存在却标为不存在")
        if not want and t["present"] != 0:
            errs.append(f"任务{t['idx']} 应不存在（同机台）却标为存在")
        if t["present"] != 1:
            continue
        if abs((t["end"] - t["start"]) - tv(t["origin"], t["dest"])) > tol:
            errs.append(f"任务{t['idx']} 行程不符：{t['end'] - t['start']} ≠ "
                        f"T[{t['origin']},{t['dest']}]={tv(t['origin'], t['dest'])}")
        if t["kind"] == "arr":
            if t["op"] > 0 and t["start"] < ops[t["job"]][t["op"] - 1][2] - tol:
                errs.append(f"任务{t['idx']} 早于上一工序完工")
            if ops[t["job"]][t["op"]][1] < t["end"] - tol:
                errs.append(f"任务{t['idx']} 晚于其工序开工")
        else:
            if t["start"] < ops[t["job"]][t["op"]][2] - tol:
                errs.append(f"回站任务{t['idx']} 早于末工序完工")

    # 5) 每车序：不重叠 + 空载时间（首趟任务不付空载 —— 与 DES `pos_node is None` 同口径）
    for t in sched["tasks"]:
        if t["present"] == 1 and t["agv"] < 0:
            errs.append(f"任务{t['idx']} 存在但没有车服务")
        if t["present"] == 0 and t["agv"] >= 0:
            errs.append(f"任务{t['idx']} 不存在却被派了车 {t['agv']}")
    for k in range(n_agv):
        mine = sorted([t for t in sched["tasks"] if t["present"] == 1 and t["agv"] == k],
                      key=lambda t: t["start"])
        for a, b in zip(mine, mine[1:]):
            dead = tv(a["dest"], b["origin"])
            if b["start"] < a["end"] + dead - tol:
                errs.append(f"车{k}：任务{a['idx']}→{b['idx']} 空载不足"
                            f"（{b['start'] - a['end']} < {dead}）")

    # 6) 完工 = 回站到达
    for j in range(inst.n_jobs):
        rets = [t["end"] for t in by_job[j] if t["kind"] == "ret"]
        want_done = max(rets) if rets else ops[j][-1][2]     # 无回站段 ⟹ 完工 = 末工序下机
        if "completes" in sched and abs(sched["completes"][j] - want_done) > tol:
            errs.append(f"作业{j} 完工时刻与回站/末工序不符")
    ms = max(sched["completes"]) if sched.get("completes") else 0.0
    if abs(ms - sched["makespan"]) > tol:
        errs.append(f"makespan {sched['makespan']} ≠ max 完工 {ms}")
    if errs:
        raise AssertionError("调度校验失败：\n  " + "\n  ".join(errs))
    return float(ms)


# ══════════════════════ CLI ══════════════════════

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="A-MKT 报告档的精确解基线（CP-SAT，NCP2 模型）——给策略一把认证过的尺子")
    ap.add_argument("--inst", default="mk01", help="MK 实例名（走 MKT 矩阵口径）")
    ap.add_argument("--n-agv", default="default", help='"default" = 该实例的 v=m')
    ap.add_argument("--tier", default="A-MKT", choices=sorted(REPORT_TIERS))
    ap.add_argument("--time-limit", type=float, default=3600.0, help="CP-SAT 时间上限 [s]")
    ap.add_argument("--workers", type=int, default=8, help="CP-SAT 并行 worker 数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="结果 JSON 输出路径（留空 = 只打印）")
    # 模型变体
    ap.add_argument("--no-return", action="store_true", help="去掉回站段（论文 NCP2 原样）")
    ap.add_argument("--same-machine-transport", action="store_true",
                    help="同机台相邻工序也建零时长运输任务（DES 逐字口径）")
    ap.add_argument("--initial-travel", default=INIT_FREE, choices=[INIT_FREE, INIT_LU])
    ap.add_argument("--exact-deadhead", action="store_true",
                    help="空载从上一工序完工时刻起算（更紧，仍是 DES 的松弛）")
    ap.add_argument("--no-sym-break", action="store_true")
    ap.add_argument("--no-rule-bound", action="store_true",
                    help="不跑规则基线（地平线仍取构造解的 makespan，只是少一个对照读数）")
    args = ap.parse_args(argv)

    from ..env.mkt import load_mkt
    mkt = load_mkt(args.inst)
    inst = mkt.base
    v = mkt.n_agv if args.n_agv == "default" else int(args.n_agv)
    spec = ExactSpec(tier=args.tier, n_agv=v, time_limit_s=args.time_limit, workers=args.workers,
                     random_seed=args.seed, include_return=not args.no_return,
                     same_machine_transport=args.same_machine_transport,
                     initial_travel=args.initial_travel, exact_deadhead=args.exact_deadhead,
                     sym_break=not args.no_sym_break)
    res = solve(inst, spec, name=args.inst, rule_bound=not args.no_rule_bound)
    print(res.summary())
    if res.schedule is not None:
        ms = validate_schedule(inst, res.schedule, v,
                               same_machine_transport=args.same_machine_transport)
        print(f"调度校验通过（独立复算 makespan={ms:.2f}）")
    payload = {k: v_ for k, v_ in vars(res).items() if k != "schedule"}
    payload["gap_abs"] = res.gap_abs
    payload["gap_rel"] = res.gap_rel
    if res.rule_makespan is not None:
        payload["certified_gap_vs_rule"] = res.certified_gap_vs(res.rule_makespan)
    payload["schedule"] = res.schedule
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"结果已写入 {args.out}")
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())
