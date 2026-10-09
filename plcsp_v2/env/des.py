# -*- coding: utf-8 -*-
"""SimPy DES 核心（《方法设计文档》§4.2：SimPy 单一后端，无物理库）。

**十约束全部接入且可独立开关**（spec §3.3，P1b 2026-10-02；⑥ 已砍）——开关一律走
`ConstraintConfig`，关掉时**不消耗随机数**，故"关 = 该约束从未存在"。

单位：**仿真时间 = 分钟，布局坐标 = 米**（bug#13 约定）；能耗 = kWh，走 `plcsp/energy.py`。

边界（诚实声明）：
- AGV 有**空载段**（2026-10-02 补齐；此前 AGV 从上一卸货点瞬移到取货点）。
- 机器故障/AGV 故障=泊松流；⑨ 的故障**段间检出**（不打断正在进行的行驶）。
- ⑦ 模糊运输时间、⑥ 模糊加工时间**均已砍**（见 spec §3.3 的砍除记录）。
- 完成时刻以**工件运抵装卸站**为准（P4-B Task 2b）：作业在装卸站入场、末工序完工后回站，
  `completes[j]` = 到站时刻 ⟹ makespan **含入场段与回站段的运输**（不是"末工序下机即完工"）。
  ⚠️ 已发表 MKT 数字的 `Cmax` 止于**末工序完工**且数据管线**丢 LU**——两者的边界不同。
- **停表条件 = 完工**（2026-10-07 修复）：`env.run(until=done | timeout(horizon))`——末件到站即停，
  `horizon`（6×工时+500）退居真·死锁护栏。此前 `until=horizon` 会把完工后的空转尾巴跑完，
  事件计数（实测 `agv_fail_events`）按护栏而非按 episode 累积——详见
  `SimWorld.run` 的停表注释；停表时刻由 metrics 的 `stop_time` 自报。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Callable, NamedTuple
import numpy as np
import simpy
from ..energy import (agv_energy_breakdown_kwh, agv_energy_kwh,
                      machine_energy_breakdown_kwh, machine_energy_kwh,
                      machine_params_for, total_energy_kwh)
from .due_dates import due_dates_for
from .layout import Layout, MachinePad
from .instances import Instance
from .rule_views import LAView, advance_avail, build_pa_view   # 步 3：规则求值的只读视图（纯数据）
from .transport import MATRIX, UNMAPPED_RAISE, TransportCaliber

if TYPE_CHECKING:                       # 仅为标注；运行期在 `snapshot()` 里就地导入
    from .snapshot import Snapshot

# ⚠️ 单位约定（2026-10-02 量纲对齐）：**仿真时间单位 = 分钟；布局坐标单位 = 米**。
# 此前无单位换算，运输时间 = 距离（米）÷ 速度，导致 TDT/TPT ≈ 4–6 倍（物理上不成立，
# 见 progress-log 与 spec §3.4）。此常量把 [m] / [m/s] 转成分钟。
SECONDS_PER_MIN = 60.0

PA_TIE_SEED = 2000
"""PA 规则**平局**随机源的子种子（步 3）——`default_rng([seed_chain, PA_TIE_SEED])`。

⚠️ **必须与既有流分开**：机台故障消费共享 `rng`（`default_rng(seed_chain)`），车队消费
`[seed_chain, 1000+a]`。PA 若借共享那条，会把故障抽样**整体移位**——"只改了平局"就变成
"连扰动流都换了"。独立子流使"平局开/关"成为一次**干净的消融**。
"""

# ⚠️ **AEI2025 式(48) 的成本常数**（`uc = 0.5`、`c = 7`，见其 §5.1 参数设置）。
# 只用于 `metrics["logistics_cost"]` 这个**报告桥**——我们的目标函数里**没有**成本项。
# 名字带来源，避免被误读成"我们自己的参数"。
DISTURBANCE = -1
"""事件流的**扰动哨兵**（`frm_idx` 位）——不是工序完工，而是"开一个决策点"。
原文的决策点定义：*"the time when an operation is assigned **or an AGV breakdown occurs**"*。"""

class StepCounters(NamedTuple):
    """**顶层决策点上的累计读数**（设计 §10③ 的段回报要的就是它的**增量**）。

    ⚠️ **只在顶层点算**（每局 ~3–4 次）⟹ 能耗那一步的开销可忽略。
    ⚠️ `makespan_so_far` 取 `env.now`——**不是真 makespan**（中途无定义，§10③ 明确不用它）；
    它只作诊断，段回报里不出现。
    """

    now: float
    done_ops: int
    done_jobs: int
    energy_kwh: float
    twt: float


AGV_FAIL_N_SEED = 3000
"""⑨ **按次数注入**（`SimConfig.agv_n_failures`）的独立子流：`default_rng([seed_chain, AGV_FAIL_N_SEED])`。

⚠️ **必须与既有流分开**（与 `PA_TIE_SEED` 同一条纪律）：机台故障消费 `default_rng(seed_chain)`、
车队消费 `[seed_chain, 1000+a]`、PA 平局消费 `[seed_chain, PA_TIE_SEED]`。
借任何一条都会把它**整体移位** ——"只改了故障注入方式"就变成"顺手改了别的流"。"""

DISTURBANCE_MACHINE = -2
"""**机障 ③** 的扰动哨兵。设计 §10⑤：顶层 MA 的触发 = **定时器 + 机障 ③ + AGV 故障 ⑨** ——
⑨ 的哨兵是 `DISTURBANCE`，③ 的在这里。两者都只是"开一个决策点/顶层决策点"，不产生件。"""

AEI2025_AGV_ACTIVATION_COST = 7.0      # c：启用一台 AGV 的费用
AEI2025_UNIT_TRANSPORT_COST = 0.5      # uc：单位运输时间成本


def rollout(inst: Instance, layout_type: str = "line", seed_layout: int = 0, seed_chain: int = 0,
            cfg: SimConfig | None = None, op_choices: list[list[int]] | None = None,
            aisle_width: float = 1.5,
            agv_phi: list[int] | None = None, constraints=None,
            layout: Layout | None = None,
            seq_rank: list[list[int | None]] | None = None,
            policy_pa=None) -> dict:
    """一次完整 episode：网格布局采样(seed_layout) → 格点距离 → SimPy(seed_chain)。

    `layout_type` **保留但忽略**（旧调用方仍传）。
    `constraints` = `ConstraintConfig`（十约束开关，spec §3.3）；None → 全开。

    ⚠️ **`layout`（布局覆盖，2026-10-06）**：给定时用它，**不再现采样**——布局级属性
    （`MachinePad.fail_rate` / `in_cap`…）这才进得了本入口。
    `None`（默认）= 用 `seed_layout` 现采样 = **今日行为，逐位不变**（既有读数靠它）。
    给定时 `seed_layout` / `aisle_width` 不再参与布局构造（几何由该布局定死；`cfg` 仍定
    动力学，车队规模不符由 `SimWorld._fleet` 显式报错）。

    ⚠️ **`seq_rank`（工序排序入口，2026-10-06）**：`seq_rank[job][oi] -> int`（越小越先；
    `None` = 该工序未覆盖 = `+inf`）。**基线侧**（NSGA-II / HGS 的 OS 段）由此把外部算好的
    工序顺序注入非延迟解码器——语义是"**缓冲内优先级**"（机台在已入缓冲的件里挑），
    不是"等指定的下一件"。`None`（默认）= 裸 FIFO 取件 = **逐位不变**。

    ⚠️ **本机制是 v1 的遗产，v2 保留它只有一个用途：给基线的 OS 段留入口。**
    - **不是 v2 的策略动作**：v2 的四个 agent 是 MA / PA / LA / RA，**没有排序头（Q）**；
    - **契约与解码口径的出处是 v1 的 spec** ——
      `docs/superpowers/specs/2026-10-06-sequencing-design.md` §5。
      **那份文档只描述 v1**（顶部有版本横幅）；**v2 是有意复用它，不是"照它实现 v2"**。
      ⚠️ v2 的 `docs/v2/design.md` 里**没有排序这一节** —— 这处复用**此前没登记**。
    """
    from .corridors import build_corridor_graph, dock_distance_matrix
    from .layout import sample_layout
    # ⚠️ aisle_width 必须同时进 SimConfig——eff_speed 读的是 cfg.aisle_width（窄道降速）。
    # 否则 `rollout(aisle_width=1.0)` 只改几何、不改速度，窄道敏感性实验**静默失效**。
    eff_cfg = cfg if cfg is not None else SimConfig(aisle_width=aisle_width)
    # ⚠️ 车队规模必须**同步进布局**，否则布局车队 ≠ 仿真车队（静默错误）。
    # ⚠️ **v2：⑩ 去后不再透传 `max_agv_capacity`**（2026-10-08）—— 单车单载，载量恒 1。
    if layout is None:
        layout = sample_layout(inst.n_machines, seed=seed_layout, aisle_w=aisle_width,
                               n_agv=eff_cfg.n_agv)
    g = build_corridor_graph(layout)
    dm = dock_distance_matrix(g)
    return SimWorld(inst, layout, dm, eff_cfg, graph=g, constraints=constraints).run(
        seed_chain=seed_chain, op_choices=op_choices, agv_phi=agv_phi, seq_rank=seq_rank,
        policy_pa=policy_pa)


@dataclass
class SimConfig:
    n_agv: int = 3          # spec §6.3 主实验值（网格上按边际收益拐点重标）
    # AGV 车速 [m/s]。**2026-10-02 由 1.0 改为 0.5**：0.5 取自我们**已引证的** GFJSPT-MMRS
    # （spec §3.4 表："AGV 3 台，0.5 m/s"）；1.0 是无出处的拍脑袋值，且使运输被低估一倍。
    # 文献区间：0.3–0.6（IEJ 随机批量运输 FJSP）、1.2（FlexSim 车间级案例）。车速是扫描轴之一。
    agv_speed_mps: float = 0.5
    # ⚠️ **v2：① 去后 `zone_granularity` / `zone_wait_limit` 已删**（2026-10-08）。
    # `zone_hold` **保留**——它是通用退避时长，另有三处在用（⑨ failover 重试、
    # 负载段失败后退避、待命轮询），与区段无关。
    zone_hold: float = 1.0        # 通用退避时长 [min]（原为区段而设，v2 下另作他用）
    repair_time: float = 5.0      # [min]
    aisle_width: float = 1.5      # 通道宽（米）：窄通道限速 = Phase A2 真权衡来源
    # ── 约束参数（**全部 assumed**，无文献出处；见 spec §9 的 assumed 表）──
    # ⚠️ v2：④ 返工已删（2026-10-08）—— `p_rework` 一并去掉
    # ⚠️ v2：⑤ 换型已删（2026-10-08）—— `setup_min_default` 一并去掉
    pm_interval: float = 120.0    # ⑫ 预防性维护间隔 [min 主轴工时]
    pm_duration: float = 10.0     # ⑫ 维护停机时长 [min]
    # ⚠️ **v2：⑩ 已删**（2026-10-08）—— `max_agv_capacity`（车队载量上限）与
    # `multi_drop`（多卸货点行程模型）两个参数一并去掉。车队同构单车单载。
    agv_mtbf: float = 480.0       # ⑨ AGV 平均无故障时间 [min]（8 h）
    agv_mttr: float = 10.0        # ⑨ AGV 平均修复时间 [min]
    # ⑨ **按次数注入**（0 = 关，用上面的 `agv_mtbf` 泊松流）。
    # ⚠️ **口径照 Shi et al. 2025**（AEI 65:103195）：该文**不设 MTBF**，而是
    #   "实例中每个实例的**故障次数在 1–8 之间随机**；故障在**最短调度周期**内随机发生
    #    并随机指派给某台 AGV；每次修理时间在 5–10 随机"。
    # `>0` 时：`SimWorld` 在 `[0, reference_makespan]` 上**均匀抽 N 个时刻**，
    # 再**轮转指派**给各车（保证均衡），本车只负责到点触发。
    # ⚠️ **实际注入次数可能少于 N**：抽出的时刻若落在**完工之后**就不发生（Shi 同此局限）。
    # 期望次数 ≠ 固定次数的差别：泊松的 sd = √N（N=4 时实际 2–6 次），本档恒为 N。
    agv_n_failures: int = 0
    # ⚠️ **按次数注入的抽签上界**（`[0, agv_fail_horizon]` 上均匀抽）。
    # **必须由调用方填**（`q1_flat.build` 会填），`_build_entities` **不去自己算** ——
    # 理由：算它要跑一次参考解，而**参考解自己也会走 `_build_entities`** ⟹ 递归 ⟹
    # `reference_run` 的 `_MREF_BUSY` 守卫当场抛"重入"（实测踩过）。
    # ⚠️ 该值的算法：`reference_makespan(inst, cfg_不含故障注入)` —— 用**没注入**的 cfg，
    # 否则"注入改动力学 ⟹ 改 T ⟹ 改注入"自指。
    # `agv_n_failures > 0` 时**必须给**，否则显式报错（不静默退化）。
    agv_fail_horizon: float | None = None
    # ⑨ **故障 failover 开关**（2026-10-05）：`True` = 停机期间把该车**手上/队列里**的任务
    # 退回队列（bound 配置转交未停机的其它车，FIFO 配置放回共享队列）——任务不再"车趴多久卡多久"。
    # ⚠️ **这是改动力学**：打开后 makespan / travel / requeue 都会变，是 ⑨ 的代价口径从
    # "仅停机时间"改为"停机 + 重新派车 + 队列重排"的**定义性后果**（此前是低估口径）。
    # ⚠️ 时点不变：故障仍只在**段间**（不持任何区段锁）生效——持锁停机 = 死锁，绝不做。
    # 默认 False ⟹ 与今日逐位相同（既有读数靠它）。
    agv_failover: bool = False
    # ③ **役龄故障率开关**（2026-10-05）：`True` 时 `fail_rate` 从常数变成**役龄的函数**
    # （Weibull 风险函数的递增段，自变量 = `pm_clock`——**当前状态里已有的量**，不引历史）。
    # 曲线的定义与理由见 `machine_fail_rate` 的 docstring 与 `mechanism-designs.md`。
    # ⚠️ **改动力学**：故障的"何时"变了（保养后一段时间更不易坏、役龄越大越易坏）。
    # ⚠️ 打开要求 ③ `machine_failure` 与 ⑫ `maintenance` 都开（否则曲线没有故障可作用 /
    #    没有役龄量）——`SimWorld.__init__` 显式报错，不静默。
    # 默认 False ⟹ 与今日逐位相同（既有读数靠它）。
    machine_age_failure: bool = False
    # Weibull 形状参数 β（**assumed**，无核实出处；敏感性扫 1.5/2.0/3.0）。
    # β = 1 退化回常数率（与关态同式）；β < 1（递减风险）不在讨论范围，显式报错。
    machine_age_beta: float = 2.0
    # ⚠️ **2026-10-10：⑪ 充电机制整条删除**（用户裁定：「删机制 反正不触发 不要了」）。
    # 原 `battery_low` / `battery_high` / `charge_kw` 三个 cfg 量一并删。
    # **删它的依据**：MK01/MK07/MK10 默认档下 `charge_events` 恒 0（一局 76–170 min，
    # 满电出发永远到不了低电阈值）；要让它起作用得把 `soc0` 压到 ≤0.25 —— 那是**为触发而调参**，
    # 不是实例自带的约束。**能量目标不受影响**（`energy.py` 按三态时间计，不读电池）。
    # ⑧ 交期**覆盖开关**：`None` = 用 `due_dates.TF_RDD` 里**逐实例标定**的 (τ, R)。
    # 2026-10-03 ⑧ 口径重设计（TF/RDD，见 `docs/superpowers/plans/2026-10-03-due-date-redesign.md`）
    # 后，tau 从"唯一标定值"降级为**覆盖**：默认 None，只在敏感性扫描/新实例标定时显式传。
    # 旧口径 `d_j = τ·M_ref` 已废，两条死因：① 锚在**自己的**参考调度上（内生——换车队规模
    # 就换 M_ref：mk01 在 n_agv=1/3 下 109.95/103.42，交期跟着漂）；② τ=0.90 只对**参考策略**
    # 标定，而训练后策略改进 12–14%，一次性把 TWT 清零（实测 makespan 89.5 < d_j=93.08 ⟹
    # TWT ≡ 0 是恒等式）。新口径只读实例数据，与 cfg 无关。
    tau: float | None = None
    # 交期跨度（due-date range, RDD）——与 tau 同进退的覆盖开关（None = 查标定表）。
    # 两者**要么都显式传、要么都不传**：只传一个时另一个仍查表，未标定实例会显式报错。
    due_range: float | None = None
    # 矩阵口径下**未被矩阵覆盖的端点**怎么算行程：`raise`（默认，显式报错）或
    # `geometry`（**声明式**降级为几何口径，并进 metrics 计数）。它进 `_cfg_key`（失效安全）。
    # ⚠️ 这**不是**"口径开关"：口径永远跟随实例（见 transport.py 的共存规则），本项只管
    # "矩阵覆盖不到的那几段"。
    # ⚠️ **2026-10-10**：本项原来举的例子是**充电桩**（桩在格点外、矩阵不覆盖）。⑪ 删后
    # 桩不存在了，但**本项保留** —— 它是"矩阵覆盖不全"的通用守卫，与具体有哪些端点无关。
    transport_unmapped: str = UNMAPPED_RAISE

    @property
    def eff_speed(self) -> float:
        """有效车速：通道 <1.5m 线性降速（下限 0.4×）——"宽=快但贵"权衡。
        默认 1.5 → 倍率 1.0（历史实验逐位不变）。"""
        return min(1.0, max(0.4, self.aisle_width / 1.5))


class OpLite:
    __slots__ = ("time",)
    def __init__(self, t: float):
        self.time = t


class SimTrack:
    """快照的跟踪量（P2 Task 1）：机台与车辆在运行流里**就地写**，`SimWorld.snapshot()` 只读。

    机台/车辆的"作业进行到哪一步"只有运行流自己知道（SimPy 对象里查不出来），故单列一袋。
    `MachineSim` / `AgvSim` 只拿这个袋子（不持 `SimWorld` 引用，避免环）；
    `SimWorld` 把各跟踪列表同时挂成 `_job_progress` 等属性，快照代码直接读属性。
    """
    __slots__ = ("job_progress", "job_loc", "job_agv", "cur_op", "agv_loaded", "agv_load_n")

    def __init__(self, n_jobs: int, n_machines: int, n_agv: int) -> None:
        self.job_progress = [0] * n_jobs                      # 已完成工序数
        self.job_loc = [-1] * n_jobs                          # 当前所在机台；-1 = 不在机台上
        self.job_agv = [-1] * n_jobs                          # 在途时所乘的车；-1 = 无
        # (工件, 加工时长, 上机时刻)：快照按 `t_start + 时长 − now` 算**标称**剩余（见 snapshot）
        self.cur_op: list[tuple[int, float, float] | None] = [None] * n_machines
        self.agv_loaded = [False] * n_agv                     # 车上是否载货（状态 2 的判据）
        self.agv_load_n = [0] * n_agv                         # 车上**在运件数**（in_flight 的"车上"部分）
        # ⚠️ **v2：④ 返工的 `job_rework` 已删**（2026-10-08）

    def set_load(self, aid: int, n: int) -> None:
        """一趟批次的**装上**（`n = len(batch)`）或**卸下**（`n = 0`）。

        ⚠️ 两个量**同处写**：`agv_loaded`（状态 2 的判据）由 `n > 0` 推出、`agv_load_n`
        是同一个数——不给"标志说满载、计数说空车"留下失同步的口子。
        """
        self.agv_loaded[aid] = n > 0
        self.agv_load_n[aid] = n


# ══ 信息侧约束的纯函数（⑧ 交期），放在模块级以便单测直接调用 ══

def compute_due_dates(inst: Instance, tau: float | None = None,
                      due_range: float | None = None) -> dict[int, float]:
    """⑧ 交期（TF/RDD 口径）——转调 `due_dates.due_dates_for`。

    ⚠️ **签名变了**（2026-10-03 ⑧ 重设计）：不再收 `(n_jobs, tau, m_ref)`——旧口径
    `d_j = τ·M_ref` 是共同交期且锚在参考调度上（内生），新口径 `d_j = LB·τ·(1+R(2ρ_j−1))`
    **逐作业、只读实例数据**。口径、生成式与书目见 `plcsp/env/due_dates.py` 的模块 docstring。
    `tau` / `due_range` 为 None 时查 `due_dates.TF_RDD` 的逐实例标定表（未标定实例显式报错）。
    """
    return due_dates_for(inst, tau, due_range)


def weighted_tardiness(completes: dict[int, float], due: dict[int, float],
                       weights: dict[int, float]) -> float:
    """⑧ 加权总拖期 `TWT = Σ w_j · max(0, C_j − d_j)`（spec §4.1 的目标口径）。"""
    return float(sum(weights.get(j, 1.0) * max(0.0, c - due[j])
                     for j, c in completes.items() if j in due))


_MREF_CACHE: dict = {}
_MREF_BUSY = False
# 参考运行**不依赖**的 cfg 字段：`tau` / `due_range` 只进交期（交期是 metric、不进仿真时序，
# 且参考运行内被 `_MREF_BUSY` 短路）——实测换 τ 的参考 makespan **逐位相同**。故两个都不进键：
# 否则 τ 扫描（⑧ 敏感性实验）会把同一份参考运行反复重跑。
_CFG_KEY_SKIP = ("tau", "due_range")


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


def _cfg_key(cfg: SimConfig) -> tuple:
    """cfg 中**影响参考运行结果**的字段指纹——缓存键的第二部分（`_instance_key` 只看实例）。

    ⚠️ 漏 cfg 会静默出错：`n_agv` / `agv_speed_mps` / `aisle_width` … 一变，参考运行的整份
    metrics 都变（实测 mk01 的 M_ref：n_agv=1/3/5 → 109.95/103.42/97.24；车速 0.5/1.0 →
    103.42/106.38），于是 f^ref 错 → 奖励权重错 → 落在哪个 Pareto 点都错。
    取**全部字段减去已知无关项**（而非白名单）：新增 cfg 字段默认进键——失效安全。
    参考运行固定 `constraints=None`（十约束全开），与调用方的 `ConstraintConfig` 无关，故后者不进键。
    """
    return tuple((f.name, getattr(cfg, f.name)) for f in fields(cfg)
                 if f.name not in _CFG_KEY_SKIP)


def reference_run(inst: Instance, cfg: SimConfig | None = None,
                  seed_layout: int = 0) -> dict:
    """参考调度（每工序取最短候选 + AGV 轮询派车）的**整份 metrics**——缓存 + 重入短路。

    ⚠️ **绝不能取被优化的那次 episode 的指标**——否则交期随策略一起漂移，
    目标退化（旧 `due_factor` 就是这么坏的：实测 MK01 tardy 恒为 10/10）。
    故单独跑一次并缓存；重入时 `_due` 会退化为"无交期"（见 `SimWorld._due`）。
    ⚠️ 由此，参考运行自身的 `tardy` / `tardy_twt` **恒为 0**——取 f^ref 者须事后按 **TF/RDD
    交期**从同一次运行的 `completes` 重算 TWT（`reward.ReferenceObjectives.of` 就是这么做的；
    交期只影响 metric、不影响动力学，故事后算 = 交期开启时的值）。
    缓存键 = 实例指纹 + `seed_layout` + **影响结果的 cfg 字段**（`_cfg_key`；`tau` / `due_range` 除外）。
    """
    c = cfg or SimConfig()
    key = _instance_key(inst) + (seed_layout,) + _cfg_key(c)
    if key in _MREF_CACHE:
        return _MREF_CACHE[key]
    global _MREF_BUSY
    if _MREF_BUSY:                   # 理论上被 `_due` 的短路挡住，此处兜底
        raise RuntimeError("reference_run 重入")
    _MREF_BUSY = True
    try:
        r = rollout(inst, seed_layout=seed_layout, seed_chain=0, cfg=c, constraints=None)
    finally:
        _MREF_BUSY = False
    _MREF_CACHE[key] = r
    return r


def scaled_agv_mtbf(inst: Instance, target_failures: float = 4.0,
                    cfg: SimConfig | None = None, seed_layout: int = 0) -> float:
    """按"**每局期望故障次数**"反推 `agv_mtbf`（⑨ 的**激活**口径，2026-10-09）。

    ⚠️ **为什么需要它**：`SimConfig.agv_mtbf` 默认 480 min，而 MK01 的 makespan 只有 ~73 min
    ⟹ **10 个 seed 合计只触发 1 次故障**（实测）。**⑨ 是小实例上等于不存在的一条约束**——
    这正是 §65 记过的"约束不触发是这个领域系统性的沉默"。

    口径：一局里 `n_agv` 条独立泊松流，期望故障数 `= n_agv × T / MTBF`，其中 `T` 取
    **参考调度的 makespan**（纯实例数据，与布局/策略无关 ⟹ 同一实例的配置可复现）。
    故 `MTBF = n_agv × T / target_failures`。

    ⚠️ **与 AEI2025 的扰动模型不同**（他们的：每实例故障总次数在 **1–8** 随机、时间随机、
    随机指派某台车、修复时间 5–10）。我们保留泊松可靠性模型（更标准），
    但**用本函数把期望次数对齐到同一量级**——否则"⑨ 是主打扰动"这句话在我们的小实例上不成立。
    """
    c = cfg or SimConfig()
    t = reference_makespan(inst, c, seed_layout=seed_layout)
    return max(1.0, float(c.n_agv) * float(t) / max(target_failures, 1e-9))


def reference_makespan(inst: Instance, cfg: SimConfig | None = None,
                       seed_layout: int = 0) -> float:
    """参考调度的 makespan —— `reference_run` 的薄封装（既有调用点不变）。

    ⚠️ 用途**只剩两个**（2026-10-03 ⑧ 重设计后）：① 特征归一化的 `m_ref`（`build_setup` 取它）；
    ② 奖励侧参考运行/权重口径的**同源守卫**（v1 的 `joint_chain_step` 曾以 `layout_seed=0`
    为前提；**v2 删了 `algo/`，那个消费者已不存在** —— 本项在 v2 里**暂无消费者**，
    留给步 5 的训练侧）。
    ⑧ 交期**不再由它导出**——交期是外生的 TF/RDD（见 `due_dates.py`），与 `M_ref` 无关。
    """
    return float(reference_run(inst, cfg, seed_layout)["makespan"])


# ⑫ 维护头（M）的候选（动作码）：0 = 现在保养，1 = 不保养。**动作语义的唯一真相在此**
# ——`MachineSim._pm_after_op` 按它执行；（v1 里 `algo/group_rel._pm_cand_feat` 按同一行序造
# 候选特征，**v2 删了 `algo/`**，特征侧在步 5 由 `hrl/` 自建，**行序仍以本常量为准**）。
# 候选**是动作**而不是实体（机台）：两个候选都属于同一台机台。
PM_CAND_NOW = 0
PM_CAND_DEFER = 1
PM_CANDS: tuple[int, int] = (PM_CAND_NOW, PM_CAND_DEFER)


def machine_fail_rate(base_rate: float, age: float, beta: float, eta: float) -> float:
    """③ 役龄故障率 [1/min]——**Weibull 风险函数的递增段**，按保养周期**均值匹配**。

        λ(a) = base · β · (a / η)^(β−1)         （a ≥ 0，η > 0，β ≥ 1）

    性质（有测试逐条钉）：
    - **β = 1 ⟹ λ ≡ base**（与今天的常数模型同式，内置退化自检点）；
    - **β > 1 ⟹ λ 严格递增**（役龄越大越易故障）= 可靠性工程里的"耗损期"形态；
    - **[0, η] 上的均值 = base**：机台按 η 周期保养时，开关开/关的**平均**故障频次一致，
      差别只在"何时"（不让"频次变了"混进"分布变了"）；
    - λ(0) = 0（β > 1 的 Weibull 性质）——**纯耗损模型**，不含早期/随机失效（如实声明）。

    ⚠️ 自变量 `age` **只能是 `pm_clock`**（主轴工时累计、保养归零——当前状态里已有的量）。
    **不许**引入"距上次故障多久""上次保养距今"这类需要记历史的量：那会把 MDP 变成有记忆
    过程（用户裁定，见 `mechanism-designs.md`）。

    ⚠️ **引文待核**：Weibull, W. *A statistical distribution function of wide applicability*,
    Journal of Applied Mechanics 18(3):293–297, 1951（**卷期页待核**）。形状参数 β 一律
    **assumed**（敏感性扫 1.5/2.0/3.0）。
    """
    if beta < 1.0:
        raise ValueError(
            f"machine_age_beta={beta} < 1：本项只用 Weibull 风险函数的**递增段**（β ≥ 1）——"
            "递减风险（早期失效）不在讨论范围，也不该拿它当'退化曲线'。")
    if eta <= 0.0:
        raise ValueError(f"役龄尺度 eta={eta} 必须 > 0（本仓取 SimConfig.pm_interval）。")
    a = max(float(age), 0.0)
    return float(base_rate) * beta * (a / float(eta)) ** (beta - 1.0)

# ⚠️ **2026-10-10：⑪ 充电头（C）整条删除** —— 原 `CHARGE_CAND_SKIP` 与
# `charge_cands(n_chargers)` 在此定义，随机制一并删（`nn/features.py` 的 "C" 段、
# `hrl/agents.py` 的 `ra_charge` 头、`hrl/rollout.py` 的 `policy_c` 同批删）。
# **动作语义的唯一真相**曾在本行——现在那个动作不存在了。




# ══ 工序排序入口 —— ⚠️ **v1 的遗产，v2 只为基线的 OS 段保留**（见 `rollout` 的 `seq_rank` 注释）
# 契约与解码口径的出处是 **v1 的 spec** `docs/superpowers/specs/2026-10-06-sequencing-design.md`
# §3（那份文档**只描述 v1**）。**v2 没有排序头（Q）**，本入口不参与 v2 的策略。

# 排序回调的契约（spec §3.2）：`decide(机台号, 候选作业元组) -> 候选**序号**`。
# 候选 = 缓冲内各件、**按到达序**（与 `in_q.items` 同序；序号 0 = FIFO 队首）。
# ⚠️ 动作是**序号**不是作业号：重复作业号在票面上仍有歧义，而序号永远唯一。
SeqDecide = Callable[[int, "tuple[int, ...]"], int]


def check_seq_rank(inst: Instance, seq_rank) -> list[list[float]]:
    """校验外部工序优先表，并把未覆盖的工序填成 `+inf`（spec §5.1 的解码口径）。

    `seq_rank[job][oi] -> int`（**越小越先**）；`None` 或行内缺口 = 该工序未覆盖 = `+inf`。
    返回**规范化副本**（长度 = `n_jobs × 各作业工序数`，元素一律 `int` 或 `inf`）。

    ⚠️ **形态不符即报错，绝不静默忽略**：静默忽略会把"基线的 OS 段没接上"伪装成
    "接上了但没用"——正是本仓反复吃亏的那类**接线缺口**（spec §60.4②）。
    故长度、逐行长度、元素类型三者逐条查，且错误信息带**具体下标**。
    """
    if not isinstance(seq_rank, (list, tuple)):
        raise TypeError(f"seq_rank 必须是 list[list[int|None]]，得到 {type(seq_rank).__name__}")
    if len(seq_rank) != inst.n_jobs:
        raise ValueError(f"seq_rank 的行数 {len(seq_rank)} ≠ 实例作业数 {inst.n_jobs}"
                         "（每个作业一行，行内按工序号索引）")
    out: list[list[float]] = []
    for j, row in enumerate(seq_rank):
        if not isinstance(row, (list, tuple)):
            raise TypeError(f"seq_rank[{j}] 必须是 list[int|None]，得到 {type(row).__name__}")
        n_ops = len(inst.jobs[j])
        if len(row) != n_ops:
            raise ValueError(f"seq_rank[{j}] 长度 {len(row)} ≠ 作业 {j} 的工序数 {n_ops}")
        vals: list[float] = []
        for oi, v in enumerate(row):
            if v is None:
                vals.append(float("inf"))
            elif isinstance(v, int) and not isinstance(v, bool):
                vals.append(float(v))
            else:
                raise TypeError(f"seq_rank[{j}][{oi}] 必须是 int 或 None（+inf），"
                                f"得到 {type(v).__name__}：{v!r}")
        out.append(vals)
    return out


class _PriorityStore(simpy.Store):
    """**可重排**的输入缓冲（spec §3.2）：只在给了排序入口时才替换裸 `simpy.Store`。

    覆写 `_do_get`——SimPy 4.1.2 的真实签名是 `_do_get(self, event)`（源码已核）。
    它是**唯一**的弹件点：`StoreGet.__init__`（机台空闲、缓冲已有件）与 `StorePut` 事件的
    回调 `_trigger_get`（机台在等、件刚到）两条路径都经过它 ⟹ 一处覆写覆盖两种情形。

    ⚠️ **决策在"取件这一刻"做**，候选集 = 当时 `items` 里的件。**不得**改成"投递时冻结的
    逐件优先级"（spec §3.1）：那是**另一个时刻**的状态，且会让候选打分失去归一化——本仓
    骨架是"一组候选一次 softmax"，逐件独立算分再比大小不是概率分布。
    ⚠️ `capacity` **必须**透传给 `super().__init__`：AGV 侧"缓冲满则让步重试"的两处循环
    ⚠️ **v2：这两处引用已失效**（2026-10-08）—— ② 去后缓冲无界，`in_q.capacity` 恒 `inf`，
    原"缓冲满则让步重试"的两个循环已删。
    ⚠️ **`LuStation.in_q` 不得换成本类**（spec §3.3 第 8 行）：装卸站到站即完工，不是加工缓冲。
    """

    def __init__(self, env, capacity, decide: SeqDecide, mach: int, stats: dict) -> None:
        super().__init__(env, capacity=capacity)
        self.decide, self.mach, self.stats = decide, mach, stats

    def _do_get(self, event) -> None:
        n = len(self.items)
        if n >= 2:
            # 只在**真有得选**（≥ 2 件）时调回调（spec §3.2）——由 Store 侧强制，
            # 不由基线/策略侧自觉（同 R/B 头的"候选 < 2 不记决策"纪律）。
            idx = self.decide(self.mach, tuple(it[0] for it in self.items))
            if type(idx) is not int or not 0 <= idx < n:
                # 非候选动作**显式报错**（同其余六头的写法）：静默回退会掩盖
                # "调用方与候选集不一致"这类错。
                raise ValueError(
                    f"排序回调返回的动作码非法：{idx!r}（机台 {self.mach}，候选 {n} 件）。"
                    f"动作 = 候选序号，取值域 [0, {n})。")
            if idx != 0:
                # `seq_reorders` = 真选了非 FIFO 队首的次数（"决策真的发生了"的硬证据，
                # 不是"回调被调了"）。`.get` 兜底：手搓 stats 的夹具不带新键。
                self.stats["seq_reorders"] = self.stats.get("seq_reorders", 0) + 1
            event.succeed(self.items.pop(idx))
        elif n == 1:
            event.succeed(self.items.pop(0))
        # n == 0：不 succeed（事件留在 `get_queue` 里等件）——与 `Store._do_get` 同语义。
        return None


class MachineSim:
    """机台：输入缓冲 → 换型 → 加工（故障中断-恢复）→ 保养 → 输出缓冲（满则阻塞）。

    **开关一律从 `ConstraintConfig` 读**（逐个传 bool 会在约束变多时漏参数）。
    开关关闭时**不得消耗随机数**——否则故障流被移位，"该约束从未存在"的语义就不成立。

    ⚠️ **`pm`（⑫ 维护头，M）**：`None`（默认）= 规则驱动，保养在主轴工时到点时**自动**触发
    ——逐位等于今日行为；非 None = 策略驱动，一道工序加工完毕、下一件尚未上机时问它
    {现在保养, 不保养}（见 `_pm_after_op`）。回调只拿机台号，快照与特征由 `algo/` 层自取
    （层次纪律：`env/` 不得依赖 `nn/`）。

    ⚠️ **`seq`（工序排序入口，2026-10-06）**：`None`（默认）= 输入缓冲是**裸 `simpy.Store`**
    （FIFO 取件，逐位等于今日行为）；非 None = `_PriorityStore`，取件那刻在已入缓冲的件里
    挑回调指定的那件。**关态在构造层面就不出现 `_PriorityStore`**（spec §3.4）——
    判据可写成 `type(m.in_q) is simpy.Store`。
    """

    def __init__(self, env, pad: MachinePad, rng, cfg: SimConfig, stats: dict, completes: dict,
                 events_q: simpy.Store, constraints, track: SimTrack, pm=None,
                 seq: SeqDecide | None = None, machine_disturbance: bool = False):
        self.env, self.pad, self.rng, self.cfg, self.stats = env, pad, rng, cfg, stats
        self.completes = completes
        self.events_q = events_q
        # ③ 机障是否向事件流发**扰动哨兵**（顶层 MA 的触发之一，设计 §10⑤）。
        # ⚠️ **默认 False** ⟹ 不往 `events_q` 多插事件 ⟹ **既有读数逐位不变**；
        # 只有 `run_gated(policy_ma=...)` 才打开（见 `_build_entities`）。
        self.machine_disturbance = bool(machine_disturbance)
        self.con = constraints
        self.track = track              # 快照跟踪量（P2 Task 1，见 `SimTrack`）
        self.pm = pm                    # ⑫ 维护头回调（None = 规则驱动，见类 docstring）
        self.seq = seq                  # 工序排序回调（None = 裸 FIFO，见类 docstring）
        # ② **有限缓冲**（2026-10-10 加回）：`finite_buffer` 关 ⟹ `inf` ⟹ **逐位不变**。
        # ⚠️ `capacity` 必须透传给 `Store`（含 `_PriorityStore`）—— AGV 侧"满则让步重试"
        # 读 `in_q.capacity`；不透传时它恒 `inf`，那个循环永远不进门（v1 的注释同此）。
        cap_in = float(pad.in_cap) if constraints.finite_buffer else float("inf")
        cap_out = float(pad.out_cap) if constraints.finite_buffer else float("inf")
        self.in_q = (simpy.Store(env, capacity=cap_in) if seq is None else
                     _PriorityStore(env, capacity=cap_in, decide=seq,
                                    mach=pad.id, stats=stats))
        self.out_q = simpy.Store(env, capacity=cap_out)
        self.slot = simpy.Resource(env, 1)   # 机台加工槽（故障/保养期间占用）
        # ⚠️ v2：⑤ 换型的 `prev_job` 已删（2026-10-08）
        self.pm_clock = 0.0                  # ⑫ 距上次保养累计的主轴工时 [min]
        # ③ 役龄故障率（默认关）：开时 `fail_rate` 按 `pm_clock` 放大（见 `machine_fail_rate`）。
        self.aging = bool(cfg.machine_age_failure)

    def _process(self, job: int, op):
        """加工（含故障中断-恢复）→ ⑫ 保养。**不含返工判定**（见 `run`）。

        ⚠️ **v2：⑤ 换型已删**（2026-10-08）—— 原换型段（`prev_job` 不同则等 `setup_min_default`）
        已移除；`stats["setup_min"]` 恒 0（energy 的 setup 项因此恒 0，属惰性残留，步 2 清）。
        """
        with self.slot.request() as req:
            yield req
            t = self.env.now
            finish = t + op.time
            if not self.con.machine_failure:
                yield self.env.timeout(op.time)            # ③ 关：无故障，一次跑完
            else:
                # ③ 役龄故障率（开关开；前置已由 SimWorld 校验）：**一次工序内 τ 不变**——
                # `pm_clock` 只在工序完成后累加（见下），故在循环外算一次即可。
                # 关态 `rate` 就是 `pad.fail_rate`（同一数值）⟹ 抽签序列逐位不变。
                rate = float(self.pad.fail_rate)
                if self.aging:
                    rate = machine_fail_rate(rate, self.pm_clock, self.cfg.machine_age_beta,
                                             self.cfg.pm_interval)
                while t < finish:
                    nxt_fail = t + self.rng.exponential(1.0 / max(rate, 1e-9))
                    seg = min(nxt_fail, finish) - t
                    yield self.env.timeout(seg)
                    if nxt_fail < finish:
                        self.stats["fail_events"] += 1
                        # ③ 机障 → 顶层 MA 的触发之一（设计 §10⑤）。**只发哨兵**：
                        # 无 `policy_ma` 时下游收到它只记一个计数、什么都不做 ⟹ 既有读数不变。
                        if self.machine_disturbance:
                            self.events_q.put((DISTURBANCE_MACHINE, self.pad.id, -1, None,
                                               False))
                        yield self.env.timeout(self.cfg.repair_time)   # 中断-恢复
                    t += seg
            self.stats["process_time"] += op.time
            self.stats["proc_min"][self.pad.id] += op.time   # M2 能耗：按机位计的切削时长
            # ⑫ 预防性维护：主轴工时到点 → 计划停机（占机台槽，工件在外面等着）
            if self.con.maintenance:
                self.pm_clock += op.time
                if self.pm is None:                     # 规则驱动（默认）——逐位等于今日
                    if self.pm_clock >= self.cfg.pm_interval:
                        yield self.env.timeout(self.cfg.pm_duration)
                        self.stats["pm_events"] += 1
                        # T3 ⑫：被阈值强制触发的那部分。`.get` 兜底——手搓 stats 的测试夹具
                        # 不带新键（同 `buffer_block_min` 的既有写法）
                        self.stats["pm_events_forced"] = self.stats.get("pm_events_forced", 0) + 1
                        self.pm_clock = 0.0
                else:                                   # 策略驱动（⑫ 维护头）
                    yield from self._pm_after_op()

    def _pm_after_op(self):
        """⑫ 维护头（M）的**决策点**（策略配置）：一道工序**加工完毕、下一件尚未上机**。

        ⚠️ 时点**就是规则配置检查的那个时点**（`_process` 里主轴工时累加后、`run` 把工件交输出
        缓冲前）：此刻工件仍占着加工槽、下一件进不来——"要停机才能保养"的唯一窗口。放在同一
        时点，使"规则 → 策略"只改**谁决定**，不改时点语义：
        - **逾期**（`pm_clock >= pm_interval`）→ 强制保养，且**不产生决策**：规则是硬底线，
          策略只能把保养提前，不能推迟过强制点（设计 §⑫「推迟：强制停机落在更晚」的语义
          = 让强制停机落在规则原本的位置）；
        - 未逾期 → 回调答 {现在保养, 不保养}（候选动作码见 `PM_CANDS`）。

        ⚠️ 回调**只拿机台号**：快照与候选特征由 `algo/` 层自取（层次纪律，同 `run_gated`），
        且它必须与规则同记账（`pm_events` / `pm_duration` / `pm_clock` 归零）——否则两条
        驱动路径的指标不可比。
        ⚠️ **`pm_events` 拆三键（T3 ⑫）**：`pm_events_forced` = 逾期强制触发的那部分、
        `pm_events_chosen` = 策略在未逾期时**主动**选的那部分，`pm_events` = 两者之和
        （既有读数不变）。T3 的 ⑫ 激活量**只取 forced**——罚"策略主动保养"等于罚它刚拿到
        的动作（`mechanism-designs.md` §T3 的关键定义）。
        """
        if self.pm_clock >= self.cfg.pm_interval:
            yield self.env.timeout(self.cfg.pm_duration)
            self.stats["pm_events"] += 1
            # 逾期：规则是硬底线，策略只能提前。`.get` 兜底（手搓 stats 的夹具不带新键）
            self.stats["pm_events_forced"] = self.stats.get("pm_events_forced", 0) + 1
            self.pm_clock = 0.0
        elif self.pm(self.pad.id) == PM_CAND_NOW:
            yield self.env.timeout(self.cfg.pm_duration)
            self.stats["pm_events"] += 1
            # 未逾期：策略主动选的（T3 不罚这部分）
            self.stats["pm_events_chosen"] = self.stats.get("pm_events_chosen", 0) + 1
            self.pm_clock = 0.0

    def _take_in(self):
        """从输入缓冲取件，顺带记**排序机会率**（spec §3.6：关态也要计）。

        `seq_gets_ge2` 的口径 = **取件这一刻**缓冲内的件数 ≥ 2（候选集 = 弹件前的 `items`）。
        缓冲非空时 `simpy.Store.get()` **同步**跑完 `_do_get`（`StoreGet.__init__` →
        `_trigger_get`）且机台不挂起 ⟹ 此处读到的 `len(items)` 与 `_do_get` 看到的候选集
        **逐位相同**；缓冲空时 `get()` 排队、件到达后在**事件帧**里弹件，裸 `Store` 上读不到
        弹件前的长度（spec §3.4 明令关态不得引入子类）⟹ 该路径按"刚到的那 1 件"计。
        ⚠️ 于是**同刻多件到达且机台在等**时本键会**少计**（spec §3.7 第 3 条的场景）；
        代价只是机会率略偏保守，不影响"机制是不是死的"的判断（⑩ 拼批头的教训）。
        """
        n = len(self.in_q.items)                # ⚠️ 必须在 `get()` **之前**读
        item = yield self.in_q.get()
        if n >= 2:
            self.stats["seq_gets_ge2"] = self.stats.get("seq_gets_ge2", 0) + 1
        return item

    def run(self):
        while True:
            job, oi, op, is_last = yield from self._take_in()
            self.stats["in_q_gets"] = self.stats.get("in_q_gets", 0) + 1
            # 快照跟踪（P2 Task 1）：工件上机 → 记在制工序、上机时刻与所在地
            # ⚠️ 上机时刻 = 此刻（换型/故障修复的墙钟延长都不计入剩余——标称口径）
            self.track.cur_op[self.pad.id] = (job, op.time, self.env.now)
            self.track.job_loc[job] = self.pad.id
            # ⚠️ **v2：④ 返工已删**（2026-10-08）—— 原重做环（`rng < p_rework` 则原地 `continue`）
            # 整个移除。顺带消失的是它那个自锁风险（原地重做是当年为绕开 `in_q.put` 死锁而选的）。
            yield from self._process(job, op)
            # 快照跟踪（P2 Task 1）：工序加工完毕 → 清在制、推进度
            self.track.cur_op[self.pad.id] = None
            self.track.job_progress[job] = oi + 1
            self.stats["ops_done"] = self.stats.get("ops_done", 0) + 1
            # ⚠️ **末工序不再在这里记完工**（P4-B Task 2b）：工件还要**回装卸站**，
            # `completes[j]` = 到站时刻（由 `LuStation` 记）。两类出口同走"先发事件 →
            # 再入 out_q"的顺序（防 put→事件 顺序死锁；transporter 的 get 即等待者）。
            self.events_q.put((self.pad.id, job, oi, op, is_last))
            # ② **有限缓冲**：`out_q` 满 ⟹ **机台被堵**（加工完了但件送不走）⟹ 记被迫等待。
            # ⚠️ 这与 AGV 侧那个 `buffer_block_min`（`in_q` 满）**是 ② 的两半** ——
            # 实测 MK 上 `in_q` 那半恒 0，效应全在这里。
            if self.con.finite_buffer and self.out_q.capacity != float("inf"):
                _t_blk = self.env.now
                while len(self.out_q.items) >= self.out_q.capacity:
                    yield self.env.timeout(self.cfg.zone_hold)
                self.stats["out_block_min"] = (self.stats.get("out_block_min", 0.0)
                                               + (self.env.now - _t_blk))
            yield self.out_q.put((job, oi, op, is_last))            # transporter 取走（含末工序）


class _StationPad:
    """`AgvSim`/transporter 只读 `.pad.dock_node`——装卸站用这个最小形状与机台共用索引表，
    **不**把装卸站伪装成 `MachinePad`（它不加工、没有故障率/缓冲容量）。"""
    __slots__ = ("dock_node",)

    def __init__(self, node: int):
        self.dock_node = node


class LuStation:
    """装卸站（Load/Unload unit）：作业在此**入场**、末工序完工后**回站**。不加工、不改工序。

    形态、书目与"为什么不是某个交叉口"见 `layout.LuPad`。仿真里它只承担两件事：

    1. **回站落点**：AGV 把回站工件投递进 `in_q`，**到达时刻即 `completes[j]`**（makespan 口径）；
    2. **与机台共用端点号空间**：机台 `0..m-1`、装卸站 = `m`（= `machines` 表的末位），
       于是 `_transporter` / `AgvSim` 的既有索引逻辑对两类端点一视同仁。

    ⚠️ **完工事件（2026-10-07，停表修复）**：`done` 非空时，末件到站**当场** `succeed()` 它。
    这是"episode 到此为止"的**唯一**判据——`run()` / `run_gated()` 拿它当 `env.run(until=…)`
    的停表条件（`horizon` 退居真·死锁护栏）。理由与护栏的关系见 `SimWorld.run` 的模块注释。
    `n_jobs=None` = 不装这个事件（手搓夹具的既有用法逐字不变）。
    """

    def __init__(self, env, pad, idx: int, completes: dict, track: SimTrack,
                 done=None, n_jobs: int | None = None):
        self.env, self.id = env, idx
        self.pad = _StationPad(pad.node)     # ⚠️ 站自己的节点号（格点外），不是它接入的交叉口
        self.completes, self.track = completes, track
        self.done, self.n_jobs = done, n_jobs
        self.in_q = simpy.Store(env)         # 回站落点：无容量上限（站是终点，永不阻塞 AGV）
        self.out_q = simpy.Store(env)        # 形状与 MachineSim 对齐（transporter 只读机台的）

    def run(self):
        while True:
            job, _oi, _op, _is_last = yield self.in_q.get()
            self.completes[job] = self.env.now      # 完工 = **到达装卸站**（不是末工序下机）
            self.track.job_agv[job] = -1
            self.track.job_loc[job] = -1
            # ⚠️ 末件到站 ⟹ 没有活可派了：成功的**是 simpy.Event 不是进程**，同刻后续事件
            #    照常入队；`triggered` 守卫使重复成功不可能（重复 `succeed()` 会 raise）。
            if (self.done is not None and not self.done.triggered
                    and len(self.completes) >= self.n_jobs):
                self.done.succeed()


# ⚠️ **v2：① 拥堵去后 `build_zone_map` 与 `ZoneManager` 已删**（2026-10-08）——
# 无区段机制，AGV 行驶不再申请/释放区段。


class AgvSim:
    """AGV：待命 → 空载驶向取货机台 → 装载驶向卸货机台 → 投递。

    **两个行驶段**（P1b Task 2 补齐）：
    - **空载段**：从当前停位开到取货点。v0 **缺失此段**（AGV 从上一个卸货点"瞬移"到取货点），
      等于凭空多出运力、且使 AGV 只有"负载"一个状态。补齐后运输负荷才真实。
    - **负载段**：取货点 → 卸货点，沿用格点最短路。

    区段管制（① congestion）**可开关**：开时逐段申请/释放区段（持当前 → 申请下一 → 放上一），
    等待环或超时则回队重试；关时不申请，按距离一次到底。

    **路线决策（R，2026-10-04 恢复）**：① 开且 `route` 回调非空时，每段行驶先问策略在 k 条
    候选路径里选哪条（`SimWorld._make_route_fn`）；`route=None`（默认）时恒走最短路——
    **逐位等于"路线头从未存在"**（`route_logits` 当年随 ① 被砍、① 恢复后漏恢复，见
    `docs/progress-log.md` §27.3/§28）。

    ⚠️ **2026-10-10：⑪ 充电决策（C）整条删除**（用户裁定）。原 `charge` 回调、
    `_maybe_charge`、`_charge_at`、`_depleted`、`_recover_from_depletion` 一并删，
    车队**不再有电量**。见 `SimConfig` 那处删除标记列的三条依据。
    """

    def __init__(self, env, aid, m_dm: np.ndarray, transport: TransportCaliber, cfg: SimConfig,
                 stats: dict,
                 tasks_in, machines: list, graph, constraints, spec, rng,
                 track: SimTrack, bound: bool = False,
                 fleet: list | None = None, events_q=None,
                 fail_times: tuple[float, ...] = ()):
        self.env, self.aid, self.m_dm, self.cfg = env, aid, m_dm, cfg
        self.transport = transport          # 行程时间口径（P4-B：跟随实例，不是全局开关）
        self.stats, self.tasks_in, self.machines = stats, tasks_in, machines
        self.track = track              # 快照跟踪量（P2 Task 1，见 `SimTrack`）
        self.g = graph
        self.con = constraints              # 物流侧开关（⑨ 等）从这里读
        self.bound = bound          # True: 任务按 agv_phi 绑定（每车一个 Store = L 层决策载体）
        # ⚠️ **按次数注入**的预排时刻（空 = 用 `agv_mtbf` 泊松流）。见 `SimConfig.agv_n_failures`。
        self.fail_times = tuple(float(t) for t in fail_times)
        # ⚠️ **v2：⑩ 的 `multi_drop` 与拼批头（B）回调已删**（2026-10-08）

        # ⚠️ **v2：① 去后路线头（R）与 `route` 回调已删**（2026-10-08）—— `_drive` 恒走最短路。
        # ⚠️ **2026-10-10：⑪ 去后 `charge` 回调已删**（见类 docstring）。
        # ⑨/决策点：故障要**通知** transporter 开一个决策点（原文："the decision point t is
        # defined as the time when an operation is assigned **or an AGV breakdown occurs**"）。
        # `None` = 不通知（手搓夹具的既有用法逐字不变）。
        self.events_q = events_q
        self.pos_node: int | None = None    # 当前所在通道节点；None = 尚未出车（停在首个取货点）
        # ⚠️ **v2：⑩ 异构车队已删**（2026-10-08）—— 车队**同构、单车单载**：
        # 载量恒 1（`capacity` 字段整个去掉），速度取 `cfg.agv_speed_mps`（无倍率）。
        self.speed = cfg.agv_speed_mps
        # ⚠️ **2026-10-10：⑪ 去后车队无电量** —— 原 `self.battery_cap`、`soc0` 的区间守卫、
        # `self.battery`、`self.chargers` / `self.charger_res` 全部删除。
        # 车队现在**完全同构**：`AgvSpec` 只剩 `id`，绝对车速取 `cfg.agv_speed_mps`，载量恒 1。
        # `spec` 参数**保留**（调用点按位传参；`AgvSpec` 仍按车号区分身份）。
        # ⑨ 故障：**每台车一条独立随机流**——否则关掉 AGV 故障会移位机台故障流（消融不干净）
        self.rng = rng
        self.down = False
        self.up = simpy.Event(env)
        self.up.succeed()                   # 初始可用
        # ⑨ failover（`SimConfig.agv_failover`，默认关）：停机期间把手上/队列里的任务退回。
        # `fleet` = 本世界的**同一份**车队列表（`_build_entities` 边建边填；run 之前一定填满）
        # ——转交前要查目标车是否停机（转给停机车会在两辆停机车之间乒乓）。
        self.failover = bool(cfg.agv_failover)
        self.fleet: list = fleet if fleet is not None else []
        # ⚠️ **v2：`free_q` 已撤**（2026-10-08）—— PA 派工不再等空车（见 `_pa_dispatch`）。
        self._rr = 0                        # 转交轮转指针（本车实例级、确定性）
        # ⑨ 的三个故障场景（AEI2025 §3.2）：**故障发生在车的哪个即时状态**。
        # SC1 = 空闲、SC2 = 空载段中、SC3 = 负载段中。`_failures` 据此判定，
        # `_drive` 负责进出本段时置位/复位——**行驶中打断**靠 `simpy.Interrupt`。
        self.stage = "idle"
        self._main = None                   # 本车主进程（`_failures` 打断它）

    def _seg_min(self, u: int, v: int) -> float:
        """节点 u→v 的**几何**行驶时长 [min]（格点最短路 ÷ 有效车速 ÷ 60）。

        ⚠️ 矩阵口径**不走这里**（矩阵已是分钟，不得再换算）——那条路走 `_leg_min`。
        """
        return float(self.m_dm[u, v]) / (self.cfg.eff_speed * self.speed) / SECONDS_PER_MIN

    def _leg_min(self, src: int, dst: int, *, count_unmapped: bool = True) -> float:
        """**整段** src→dst 的行驶时长 [min]——两种口径的唯一汇合点（不含区段拆分）。

        - 几何口径：转调 `_seg_min`（现状，逐位不变）；
        - 矩阵口径：**直接查表**（矩阵已是分钟，**不得**再套任何换算）。
          ⚠️ **修（步 3）**：原式 `t / self.speed_ratio` 的 `self.speed_ratio` 是 ⑩ 的
          产物——⑩ 删除时定义行被一并删掉，**却漏了这处使用**（矩阵口径一查表就
          `AttributeError`；MK 实例走几何口径，故一直没暴露）。车队同构 ⟹ 倍率恒 1，
          直接返回查表值；
        - 任一端点**没有矩阵对应项** ⟹ 按 `transport.unmapped` 处置：`raise` 显式报错；
          `geometry` **声明式**降级为几何口径并计数（metrics 带出，供表里如实声明）。
        `count_unmapped=False` 供**排序**之类的查数用（只选一个端点却把候选全计一遍会虚高）。
        ⚠️ 计数按**出发次数**记：区段争用失败而重试的段会重复计一次（它确实又跑了一趟）。
        ⚠️ **2026-10-10**：本项原来举的例子是**充电桩**（桩曾是唯一在矩阵外的端点）。
        ⑪ 删后桩不存在，但**守卫保留**——矩阵覆盖不全时仍要走同一条报错/降级路径。
        """
        if self.transport.mode != MATRIX:
            return self._seg_min(src, dst)
        t = self.transport.minutes(src, dst)
        if t is not None:
            return t
        if self.transport.unmapped == UNMAPPED_RAISE:
            raise ValueError(
                f"节点 {src}→{dst} 在行程时间矩阵里**没有对应项**（矩阵只覆盖机台与装卸站）——"
                "若要跑矩阵不覆盖的端点，请显式设 SimConfig.transport_unmapped='geometry'")
        self.stats["unmapped_legs"] = self.stats.get("unmapped_legs", 0) + 1
        got = self._seg_min(src, dst)
        self.stats["unmapped_min"] = self.stats.get("unmapped_min", 0.0) + got
        return got

    def leg_min(self, src: int, dst: int) -> float:
        """`_leg_min(..., count_unmapped=False)` 的**公开**入口（步 3）。

        LA 的规则求值要"车到取货点要多久"，必须与仿真**同一条**口径（`shortest_node_path`
        的同一张矩阵、同一套换算）——另造一个距离函数就是给口径漂移留口子。
        借首台车的实现即可：本口径与车号无关（v2 车队同构 ⟹ `speed` 逐车相同）。
        """
        return self._leg_min(src, dst, count_unmapped=False)

    # ⚠️ **2026-10-10：⑪ 去后 `_drain`（行驶耗电）已删** —— 车队无电量，没有可耗的东西。
    # 它的两个调用点（`_drive` 的中断补记与段末记账）同步删除。

    def _failures(self):
        """⑨ AGV 故障：按泊松流停机 `agv_mttr`，期间不接活。

        **段间检出**：故障不打断正在进行的行驶，在下一段行驶开始前的**段间**生效。
        这是简化，但 MTBF(480 min) 远大于单段行驶时长，误差可忽略。

        ⚠️ **`SimConfig.agv_failover=True`** 时语义加强：车在段间停机期间，**手上与队列里的
        任务退回/转交别的车**（`_handoff_queued` / `_requeue_hand`），不再"车趴多久卡多久"。
        时点**不变**——仍只在段间、且车不持任何区段锁（持锁停机 = 同区段的车永久等待 = 死锁）。
        默认关时本函数**逐字**是今日行为（只设 down/up）。
        """
        if self.fail_times:
            # ⚠️ **按次数注入**（`SimConfig.agv_n_failures`，口径照 Shi et al. 2025）：
            # 时刻由 `SimWorld` 一次性抽好并指派到车，本车只负责到点触发。
            for t in self.fail_times:
                yield self.env.timeout(max(0.0, t - self.env.now))
                yield from self._do_failure()
            return
        while True:
            yield self.env.timeout(self.rng.exponential(self.cfg.agv_mtbf))
            yield from self._do_failure()

    def _do_failure(self):
        """一次故障的**全部后果**（两条注入路径共用，两份拷贝必然漂）。"""
        self.stats["agv_fail_events"] += 1
        # ── ⑨ 的三个场景（AEI2025 §3.2）：按车的**即时状态**判定 ──
        # SC1 空闲 · SC2 空载段中 · SC3 负载段中。计数逐场景入 stats，供报告如实声明分布。
        scen = {"idle": "sc1", "empty": "sc2", "loaded": "sc3"}[self.stage]
        self.stats[f"agv_fail_{scen}"] = self.stats.get(f"agv_fail_{scen}", 0) + 1
        # ⚠️ 必须**先换一个新的未触发事件**再置 down：否则等待者会反复 yield 一个
        # 已经 succeed 的事件 → 仿真时间不推进、空转成死循环（实测 pytest 直接挂死）。
        self.up = simpy.Event(self.env)
        self.down = True
        if self.events_q is not None:
            # 扰动事件（哨兵：`frm_idx = -1`）——transporter 收到后**开一个决策点**。
            yield self.events_q.put((DISTURBANCE, self.aid, -1, None, False))
        # ⚠️ **SC2/SC3：打断正在进行的行驶**（`_drive` 里接 `simpy.Interrupt`）。
        # 只在 `stage != "idle"` 时打断——空闲时车可能停在任意 yield 点上，
        # 盲目打断会把 Interrupt 抛到无关的等待里。
        if self.stage != "idle" and self._main is not None:
            self._main.interrupt(("agv_breakdown", self.stage))
        yield self.env.timeout(self.cfg.agv_mttr)
        self.down = False
        self.up.succeed()

    def _wait_up(self):
        """⑨ 若当前停机，等到修复。"""
        while self.down:
            yield self.up

    def _up_targets(self) -> list[int]:
        """可接管的**其它车**（未停机），从本车 +1 起轮转——确定性顺序。

        ⚠️ 只收未停机的车：转给停机车会在两辆停机车之间乒乓（任务原地打转、计数虚增，
        且双方都在等 `up` 事件，谁也不会先跑）。
        """
        n = len(self.fleet) or self.cfg.n_agv
        out = []
        for k in range(1, n):
            j = (self.aid + k) % n
            if j < len(self.fleet) and not self.fleet[j].down:
                out.append(j)
        return out

    def _handoff_queued(self):
        """⑨ failover **检查点 1**（循环顶、段间、不持锁）：把本车队列里的任务转交别的车。

        只在 `bound`（每车一 Store）且存在未停机的其它车时动作；FIFO 配置的共享队列本来就
        人人可取（无需动作），其它车全停机时也不动（留在本车队列，等任一车恢复后再转交）。
        """
        if not self.failover or not self.bound or not self.down:
            return
        targets = self._up_targets()
        if not targets:
            return
        q = self.tasks_in[self.aid]
        while q.items:
            item = yield q.get()
            j = targets[self._rr % len(targets)]
            self._rr += 1
            yield self.tasks_in[j].put(item)
            self.track.job_agv[item[2][0]] = j            # 快照：改指新持有者
            self.stats["agv_failover_tasks"] = self.stats.get("agv_failover_tasks", 0) + 1

    def _requeue_hand(self, items):
        """⑨ failover **检查点 2/3**（段间、不持锁）：把本车**手上**的任务退回/转交。

        `bound`：转交未停机的其它车；其它车全停机 ⟹ 放回**本车队列**（车不再手拿任务，
        恢复后可跑）。FIFO：放回**共享队列队尾**（会改变 FIFO 相对顺序，如实记明）。
        记账：`agv_failover_tasks` + `track.job_agv`（新持有者 / −1）。
        """
        if not self.failover or not items:
            return
        if not self.bound:
            for item in items:
                yield self.tasks_in.put(item)
                self.track.job_agv[item[2][0]] = -1
                self.stats["agv_failover_tasks"] = self.stats.get("agv_failover_tasks", 0) + 1
            return
        targets = self._up_targets()
        own = self.tasks_in[self.aid]
        for item in items:
            if targets:
                j = targets[self._rr % len(targets)]
                self._rr += 1
                yield self.tasks_in[j].put(item)
                self.track.job_agv[item[2][0]] = j
            else:
                yield own.put(item)
                self.track.job_agv[item[2][0]] = self.aid
            self.stats["agv_failover_tasks"] = self.stats.get("agv_failover_tasks", 0) + 1

    def _wait_up_or_failover(self):
        """⑨ 段间停机等待。`failover` 关 ⟹ **逐字**等于 `_wait_up`（默认配置逐位不变）。

        `failover` 开：停机期间反复尝试转交本车队列里的任务（检查点 1）。轮询间隔 =
        `zone_hold`（确定性；停机是有界事件，轮询代价可忽略）。**只等待、不行驶**——
        车不持任何区段锁，故不会死锁。
        """
        if not self.failover:
            yield from self._wait_up()
            return
        while self.down:
            yield from self._handoff_queued()
            yield self.up | self.env.timeout(self.cfg.zone_hold)

    # ⚠️ **2026-10-10：⑪ 去后 `_depleted` / `_recover_from_depletion` 已删** ——
    # 车队无电量，"耗尽 ⟹ 不可用"这个门与它的恢复环都不存在了。
    # 原实现（低电阈值判据 + 反复补电直到 `battery_high × cap`）随机制一并删。
    # ⚠️ 顺带消失的统计键：`agv_dry_events`（耗尽次数）。

    # ⚠️ **2026-10-10：⑪ 去后 `_charge_at` / `_maybe_charge` 已删** ——
    # 车队无电量，没有"开去桩、排队、充到目标"这件事。
    # ⚠️ 顺带消失的统计键：`charge_events`（充电次数）、`battery_min_kwh`（最低电量）。

    def _collect(self, q, frm: int, to: int, first: tuple) -> list:
        """取一个任务（**单车单载**）。

        ⚠️ **v2：⑩ 拼车/拼批已删**（2026-10-08）—— 载量恒 1，故本函数**永远只返回单件**。
        原实现在容量 > 1 时从队首连续段拼同 (取货点, 卸货点) 的任务；那段逻辑随 ⑩ 一并移除。
        保留函数形状（仍是生成器、仍返回 list）是为了让 `run` 的调用点不用改形态。
        """
        del q, frm, to                      # 签名保持以对齐既有调用点；v2 下不再使用
        if False:                           # 保持"是生成器"的形态（`yield from` 调用点不变）
            yield None
        return [first]

    # ⚠️ **v2：⑩ 拼批 / multi-drop 已删**（2026-10-08）——
    # `_same_frm_indices` / `_take_indices` / `_collect_multi` / `_count_batch` /
    # `_collect_batch` / `_deliver_multi` 六个方法一并移除。单车单载，一趟一件。

    def _drive(self, src: int, dst: int, leg: str):
        """把车从 `src` 节点开到 `dst` 节点，行驶时长累入 `leg` 态（"empty" / "loaded"）。

        返回 `(ok, end_node)`。`ok=False` = 区段争用超时或等待环——此时车停在 `end_node`
        且**已释放**持有的区段；**失败前已走的路程照样计入时长**（否则能耗会被低估）。
        区段管制关闭时按最短路一次到底。

        ⚠️ **路线（R）在此处决断**（2026-10-04 恢复）：`self.route` 非空时路径由策略在 k 条
        候选里选（`route(aid, src, dst, leg) -> 节点序列`）；为空时走 `shortest_node_path`，
        **不多取快照、不多抽随机数** ⟹ 关态逐位等于今日（硬要求）。
        ⚠️ 矩阵口径**不需要为绕行改公式**：整段时长 `total`（矩阵查表）按几何占比分摊到各段
        （`seg = total * seg / leg_geo`），故绕行的总时长自动按"几何长度比"变长，而最短路
        那条的数值与今日逐位相同。
        """
        if src == dst:
            return True, dst
        # ⚠️ **v2：① 拥堵与路线头（R）都已删**（2026-10-08）—— 按距离直行，一次走到底。
        # ⚠️ 矩阵口径下**整段时长一次算出**（原来是逐段按几何占比分摊——那时要逐段查表）。
        seg = self._leg_min(src, dst)
        self.stage = leg                        # ⑨：SC2（空载中）/ SC3（负载中）据此判定
        t0 = self.env.now
        try:
            yield self.env.timeout(seg)
        except simpy.Interrupt:                 # ⚠️ ⑨ SC2/SC3：行驶中途故障
            spent = min(self.env.now - t0, seg)
            self.stats["travel_time"] += spent
            self.stats[f"agv_{leg}_min"] += spent
            self.stats["agv_busy_min"][self.aid] += spent
            # ⚠️ **被打断 = 本段没跑完** ⟹ 从未走完的那部分要单独记（`moves` 不计这一次）。
            self.stats["agv_interrupted_min"] = (
                self.stats.get("agv_interrupted_min", 0.0) + (seg - spent))
            # 被中断任务**已消耗的运输时间**（原文的 `at`，计入 TC）。
            self.stats["agv_at_min"] = self.stats.get("agv_at_min", 0.0) + spent
            # ── 修理点（SC2/SC3）：原文 *"moved to a location closer to itself between the
            #    starting location and destination location"* —— **取本段两端里离它更近的一端**。
            #    ⚠️ 原文这句话本身有歧义（"离自己更近"既可读作"路上离自己最近的点"=原地，
            #    也可读作"两端里更近的一端"）；我们取后者，因为它的**目的**是
            #    *"avoid the interference of mid-road repair to other shop activities"*
            #    ——那只有移开路面才成立。如实声明。
            here = self._node_along(src, dst, spent / max(seg, 1e-9))
            near_src = self._leg_min(here, src) <= self._leg_min(here, dst)
            self.pos_node = src if near_src else dst
            self.stage = "idle"
            return False, self.pos_node
        self.stage = "idle"
        self.stats["travel_time"] += seg
        self.stats[f"agv_{leg}_min"] += seg
        # 步 3：**逐车**累计行驶工时（LA 的 `MINU` 规则的自变量）。与上面两个车队合计
        # 键**同处写**——分两处记会有"合计说跑了、逐车说没跑"的失同步口子。
        self.stats["agv_busy_min"][self.aid] += seg
        self.stats["moves"] += 1
        return True, dst

    def _node_along(self, src: int, dst: int, frac: float) -> int:
        """沿 `src→dst` 的格点最短路按**时长比例**定位到某个节点（只在被打断时调用）。

        ⚠️ 只在 `simpy.Interrupt` 分支里算——正常行驶不付这份代价。
        """
        from .corridors import shortest_node_path
        path = shortest_node_path(self.g, src, dst)
        if len(path) <= 2:
            return path[-1] if frac >= 0.5 else path[0]
        cum, tot = [0.0], 0.0
        for u, v in zip(path, path[1:]):
            tot += self._leg_min(u, v, count_unmapped=False)
            cum.append(tot)
        if tot <= 0.0:
            return path[-1] if frac >= 0.5 else path[0]
        t = frac * tot
        for k, c in enumerate(cum):
            if c >= t:
                return int(path[min(k, len(path) - 1)])
        return int(path[-1])

    def run(self):
        q = self.tasks_in[self.aid] if self.bound else self.tasks_in
        self._main = self.env.active_process      # ⑨ 打断目标（见 `_failures`）
        if self.con.agv_failure:
            self.env.process(self._failures())       # ⑨ 关掉时不启进程 → 不抽随机数
        while True:
            yield from self._wait_up_or_failover()   # ⑨ 停机中不接活（failover 打开时含转交）
            # ⚠️ **2026-10-10：⑪ 去后**，原「耗尽门」(`_depleted` / `_recover_from_depletion`)、
            # 「待命补电」(`_maybe_charge`)、「待命耗电」(`_drain_idle`) 四处调用一并删除
            # —— 车队无电量，不存在耗尽，也不存在补电。`agv_dry_events` 统计键随之消失。
            # ⚠️ 原处**第二次** `_wait_up_or_failover()` 也已删：它当时只为"补电之后再确认
            # 是否停机"而设。现在两步之间**没有任何 yield**，而 `_wait_up_or_failover` 在
            # `down=False` 时恒立即返回（`_wait_up` 的 `while self.down`）⟹ 它就是空操作。
            frm, to, item, path = yield q.get()
            self.stats["tasks_get"] += 1
            if self.failover and self.down:
                # ⑨ failover **检查点 2**（段间、不持锁）：等任务期间趴窝 ⟹ 刚取到的任务
                # 不留在车上（今日行为是"带着货跑完整趟"，不真实）。
                yield from self._requeue_hand([(frm, to, item, path)])
                continue
            a = self.machines[frm].pad.dock_node
            b = self.machines[to].pad.dock_node
            # ── 空载段：当前停位 → 取货点 ──
            if self.pos_node is not None and self.pos_node != a:
                ok, self.pos_node = yield from self._drive(self.pos_node, a, "empty")
                if not ok:
                    self.stats["requeue"] = self.stats.get("requeue", 0) + 1
                    q.put((frm, to, item, path))
                    # 快照跟踪（P2 Task 1）：退回队列 = 这批活不在这台车上（车还是空的）
                    self.track.job_agv[item[0]] = -1
                    self.track.set_load(self.aid, 0)
                    continue
            # ⚠️ **v2：⑩ 去后单车单载**（2026-10-08）—— 一趟一件，无拼车/拼批分支
            batch = yield from self._collect(q, frm, to, (frm, to, item, path))
            # 快照跟踪（P2 Task 1）：取到任务即记车号——FIFO 路径没有派车分支，只靠这里，
            # 否则 `JobState.in_transit` 在旧规则路径上恒为 False。
            for (_f0, _t0, _it0, _p0) in batch:
                self.track.job_agv[_it0[0]] = self.aid
            self.track.set_load(self.aid, len(batch))
            # ── 负载段：取货点 → 卸货点（整批一趟，**单卸货点**：逐位不变）──
            # 快照跟踪（P2 Task 1）：取货后负载行驶，本趟在运件数 = 批次大小（F5：in_flight 的"车上"部分）
            if self.failover and self.down:
                # ⑨ failover **检查点 3**（段间、不持锁）：空载段期间趴窝 ⟹ 整批退回，
                # 不带货趴窝（今日行为是照跑负载段）。
                self.track.set_load(self.aid, 0)
                yield from self._requeue_hand(batch)
                continue
            ok, self.pos_node = yield from self._drive(a, b, "loaded")
            if not ok and self.pos_node != b:
                # SC3 且**修理点 = 取货点**（离自己更近的那一端是起点）⟹ 带着货退回队列。
                # 快照跟踪（P2 Task 1）：退回队列 = **当场**卸空——车要带着"空车"的状态在
                # 取货点干等 zone_hold，不能挂着"负载行驶"（否则状态 2 / 在途会滞留到重试）
                self.track.set_load(self.aid, 0)
                for t in batch:
                    self.track.job_agv[t[2][0]] = -1
                yield self.env.timeout(self.cfg.zone_hold)
                self.stats["requeue"] = self.stats.get("requeue", 0) + 1
                for t in batch:                      # 整批退回队列
                    q.put(t)
                continue
            if not ok:
                # ⚠️ **SC3 且修理点 = 卸货点**：原文 *"If this load job moves synchronously to
                # the end location of load-stage, it will **not be seen as an interrupted task**
                # and can continue to be processed directly on the assigned machine."*
                # ⟹ 工件已随车同步到达终点，**照常投递**（只多等一次修复）。
                self.stats["agv_sc3_delivered"] = self.stats.get("agv_sc3_delivered", 0) + 1
                yield from self._wait_up_or_failover()
            self.stats["trips"] += 1
            for (_f, t2, _item2, _p2) in batch:
                self.stats["agv_del"][self.aid] += 1  # L 层状态轨迹（负载差/最后位置）
                self.stats["agv_pos"][self.aid] = t2
                # ② **有限缓冲**：满则让步超时重试（**照 v1 逐字**）——不直接 `put` 阻塞，
                # 是为了让"被迫等待时长"可测（`buffer_block_min` = ② 的激活量）。
                # ⚠️ 两处投递路径（本函数 + `run` 的单卸货点段）**各记一处**，漏一处即低估一半。
                if self.con.finite_buffer:
                    _t_block = self.env.now
                    while (len(self.machines[t2].in_q.items)
                           >= self.machines[t2].in_q.capacity):
                        yield self.env.timeout(self.cfg.zone_hold)
                    self.stats["buffer_block_min"] = (
                        self.stats.get("buffer_block_min", 0.0) + (self.env.now - _t_block))
                yield self.machines[t2].in_q.put(_item2)
                self.track.job_agv[_item2[0]] = -1       # 快照跟踪（P2 Task 1）：投递完成，工件离车
                self.stats["deliveries"] += 1
            self.track.set_load(self.aid, 0)             # 快照跟踪（P2 Task 1）：整批卸空

    # ⚠️ **2026-10-10：⑪ 去后 `_drain_idle`（待命耗电）已删** —— 车队无电量。
    # 它的调用点（`run` 里取完任务后按等待时长记账）同步删除。


class SimWorld:
    """一次 episode：1 布局 + 1 实例 + 1 条独立扰动流 → 指标 dict。"""

    def __init__(self, inst: Instance, layout: Layout, m_dm: np.ndarray,
                 cfg: SimConfig | None = None, graph=None, constraints=None):
        self.inst, self.layout, self.m_dm = inst, layout, m_dm
        self.cfg = cfg or SimConfig()
        # 格点走廊图（AGV 空载段/区段路径的来源）。`graph=None` 时**自建**——
        # 此前 6 处调用点（训练环 group_rel ×3、m11、m6_*  ×2）都省略了它，
        # 在旧代码里恰好无害（AGV 用的是任务里带好的 path），一旦 AGV 需要自己算路径
        # 就会在运行期炸成 `dijkstra_path(None, …)`。自建彻底消除这个坑。
        if graph is None:
            from .corridors import build_corridor_graph
            graph = build_corridor_graph(layout)
        self.g = graph
        if constraints is None:
            from .constraints import ConstraintConfig
            constraints = ConstraintConfig()
        self.constraints = constraints      # 十约束开关（spec §3.3）
        # ③ 役龄故障率的**前置**（显式报错，不静默降级）：
        # - ③ 关 ⟹ 根本没有故障，"率随龄上升"是死开关；
        # - ⑫ 关 ⟹ `pm_clock` 不累加（设计如此）⟹ **役龄量不存在**，曲线没有自变量。
        if self.cfg.machine_age_failure:
            if not self.constraints.machine_failure:
                raise ValueError(
                    "SimConfig.machine_age_failure=True 但 ③ machine_failure 关：没有故障可"
                    "'随龄上升'——请开 ③ 或关本开关。")
            if not self.constraints.maintenance:
                raise ValueError(
                    "SimConfig.machine_age_failure=True 但 ⑫ maintenance 关：`pm_clock` 在 ⑫ 关时"
                    "根本不累加（设计如此）⟹ 役龄量不存在，曲线没有自变量。请开 ⑫ 或关本开关。")
        # 行程时间口径（P4-B）：**跟随实例**（MKT 实例走矩阵、原始 MK 走几何）。只在这里
        # 认一次，整条运行链（含快照/训练栈建的世界）拿到的是同一个口径对象。
        self.transport = TransportCaliber.for_instance(inst, layout,
                                                       unmapped=self.cfg.transport_unmapped)
        # ⑧ 交期表**本世界内**缓存（见 `_due_map`）：交期外生，而 `snapshot()` 每个决策点都读它。
        self._due_key: tuple | None = None
        self._due_val: dict[int, float] = {}
        # 逐作业**标称剩余工时**的查表（见 `_remaining_min`）：纯实例数据，与仿真状态无关。
        self._rem_tbl: list[list[float]] | None = None
        # 步 3：本局 stats 的活引用（由 `_build_entities` 挂上）。此处先置空 dict，
        # 使 `snapshot()` 在任何路径上都有可读对象（缺键走 `.get` 兜底）。
        self._live_stats: dict = {}
        # 步 3：PA 平局随机源（每局在 `run`/`run_gated` 里重播，见 `PA_TIE_SEED`）。
        self._pa_tie_rng: np.random.Generator | None = None
        self._cold_start()                  # t=0 快照骨架（P2 Task 1，run() 会整套换掉）

    def _fleet(self) -> list:
        """布局车队。⚠️ 与仿真车队不符 = 静默错误（布局车队 ≠ 仿真车队），直接报错。"""
        fleet = self.layout.agvs or []
        if len(fleet) != self.cfg.n_agv:
            raise ValueError(
                f"布局车队 {len(fleet)} 台 ≠ SimConfig.n_agv={self.cfg.n_agv}——"
                "直接构造 SimWorld 时须传 sample_layout(..., n_agv=cfg.n_agv)")
        return fleet

    def _build_entities(self, env, stats: dict, completes: dict, rng, *, bound: bool,
                        seed_chain: int = 0, pm=None,
                        seq: SeqDecide | None = None,
                        done=None, machine_disturbance: bool = False) -> tuple[
                            list, list[simpy.Store] | simpy.Store, simpy.Store]:
        """建机台 / 装卸站 / 任务队列 / 车辆 / 跟踪量，并把活引用挂到 `self`（`snapshot()` 读它们）。

        `run()` / `run_gated()` / `_cold_start()` **共用同一份**构造（逐字重复是本仓评审会判
        缺陷的形态）。⚠️ 只建对象、**不启进程**：`env.process` 的**注册顺序**决定同刻事件
        次序，故启动由调用方按原顺序做（机台 → 车辆）。返回调用方后续还要用的实体
        `(entities, tasks_in, events_q)`（`self.*` 上的活引用已一并挂好）。

        ⚠️ **`entities` 比 `self.machines` 多一项**：末位是装卸站（端点号 = `inst.n_machines`）。
        任务端点、AGV 的投递目标都走这张表；而 `self.machines` **只含机台**——`snapshot()` 与
        `_energy_report` 的逐机台口径不能被装卸站污染（它不是机台）。

        ⚠️ `route`（R 层）：`None` = 不启用路线决策（`run()` / `_cold_start()` 恒如此）；
        ⚠️ **v2：`route`（R 头）已删**（2026-10-08）。
        ⚠️ `pm`（⑫ 维护头）：同上，`None` = 规则驱动；`run_gated` 传 `_make_pm_fn(...)` 的
        返回值。透传给每台 `MachineSim`，`_pm_after_op` 消费。
        ⚠️ **2026-10-10：⑪ 去后 `charge` / `charger_res` 两个参数已删** —— 车队无电量、
        也没有充电桩，C 头与它的桩占用表都不存在了。
        ⚠️ **v2：⑩ 去后本类不再有 `batch` 参数**（2026-10-08）。
        ⚠️ `seq`（工序排序）：`None` = 输入缓冲是裸 `simpy.Store`（逐位等于今日行为）；
        `run()` 传 `_make_seq_decide(...)` 的返回值。透传给每台 `MachineSim`，
        **⚠️ 明确不传 `LuStation`**（spec §3.3 第 8 行：装卸站到站即完工，不是加工缓冲）。
        ⚠️ `done`（完工事件）：透传给 `LuStation`，末件到站时成功；调用方拿它当停表条件。
        `None` = 不装（手搓夹具的既有用法逐字不变）。
        """
        specs = self._fleet()
        # ⚠️ **v2：① 去后无 `ZoneManager`**（2026-10-08）
        events_q = simpy.Store(env)
        track = SimTrack(self.inst.n_jobs, self.inst.n_machines, self.cfg.n_agv)
        machines = [MachineSim(env, self.layout.machines[i], rng, self.cfg, stats, completes,
                               events_q, self.constraints, track, pm=pm, seq=seq,
                               machine_disturbance=machine_disturbance)
                    for i in range(self.inst.n_machines)]
        lu = LuStation(env, self.layout.lu, self.inst.n_machines, completes, track,
                       done=done, n_jobs=self.inst.n_jobs)
        entities = machines + [lu]
        if bound:
            tasks_in = [simpy.Store(env) for _ in range(self.cfg.n_agv)]   # L 层绑定：每车一队列
        else:
            tasks_in = simpy.Store(env)                                    # 旧 FIFO 规则路径
        # ⚠️ 车队列表**边建边填**（每个 AgvSim 拿同一份列表的引用）：⑨ failover 转交前要查
        #    目标车是否停机。`run()` 在所有车建好之后才启动，故运行时列表一定是满的。
        # ⚠️ ⑨ **按次数注入**（`SimConfig.agv_n_failures` > 0）：在 `[0, 参考解 makespan]` 上
        # **均匀抽 N 个时刻**，再**轮转指派**给各车（保证均衡，不让某台车独吞）。
        # 口径照 Shi et al. 2025：该文不设 MTBF，直接规定每实例的故障次数。
        # ⚠️ 抽在 `[0, T_ref]` 上 ⟹ **实际注入次数可能 < N**（落在完工之后的就不发生），Shi 同此局限。
        # ⚠️ `T_ref` 走 `reference_makespan` —— 它**有缓存**（`_MREF_CACHE`），故这里不是每局重跑。
        per_agv: dict[int, list[float]] = {}
        if self.cfg.agv_n_failures > 0:
            # ⚠️ **不许在这里算参考解**（会递归，见 `SimConfig.agv_fail_horizon`）——
            # 要求调用方填好，缺了就显式报错，不静默退化成一个拍脑袋的上界。
            if self.cfg.agv_fail_horizon is None:
                raise ValueError(
                    "SimConfig.agv_n_failures > 0 但 agv_fail_horizon=None ——"
                    "抽签上界必须由调用方填（用 `reference_makespan(inst, 不含注入的 cfg)`）。"
                    "在本层自己算会造成 `reference_run` 递归（实测抛『重入』）。")
            fn_rng = np.random.default_rng([seed_chain, AGV_FAIL_N_SEED])
            t_ref = float(self.cfg.agv_fail_horizon)
            ts = np.sort(fn_rng.uniform(0.0, t_ref, size=int(self.cfg.agv_n_failures)))
            order = fn_rng.permutation(self.cfg.n_agv)
            for i, t in enumerate(ts):
                per_agv.setdefault(int(order[i % self.cfg.n_agv]), []).append(float(t))
        agvs: list[AgvSim] = []
        for a in range(self.cfg.n_agv):
            agvs.append(AgvSim(env, a, self.m_dm, self.transport, self.cfg, stats, tasks_in,
                               entities, self.g, self.constraints, specs[a],
                               np.random.default_rng([seed_chain, 1000 + a]),   # ⑨ 每车独立流
                               track, bound=bound, fleet=agvs,
                               events_q=events_q, fail_times=tuple(per_agv.get(a, ()))))
        # 活状态引用（P2 Task 1）：`snapshot()` 据此取**当时**的快照
        self.env = env
        self.completes = completes
        # ⚠️ **2026-10-10：⑪ 去后 `self.charger_res` 已删**（桩占用的活引用不存在了）。
        self.machines = machines            # ⚠️ 只含机台（装卸站在 `self.lu`，不进快照的机台段）
        self.lu = lu
        self.tasks_in = tasks_in
        self.track = track
        self._job_progress = track.job_progress
        self._job_loc = track.job_loc
        self._job_agv = track.job_agv
        # ⚠️ v2：④ 返工的 `_job_rework` 已删（2026-10-08）
        self._cur_op = track.cur_op
        self._agv_loaded = track.agv_loaded
        self._agv_load_n = track.agv_load_n
        self.agvs = agvs
        # 步 3：本局的 stats **活引用**（快照要读其中的累计量：`agv_busy_min`）。
        # 只在 `_build_entities` 这一处挂——`run()` / `run_gated()` / `_cold_start()`
        # 三条路径共用它，不存在"某条路径忘了挂"的形状。`_cold_start` 传的是空 dict ⟹
        # 快照对缺失键有 `.get` 兜底（见 `snapshot()`）。
        self._live_stats = stats
        return entities, tasks_in, events_q

    def leg_min(self, src: int, dst: int) -> float:
        """节点 `src → dst` 的行驶时长 [min]（步 3：**LA 规则求值**的公开入口）。

        转调首台车的 `AgvSim.leg_min`——同一套行程时长口径（矩阵/几何 + 未映射处置），
        与仿真**逐位同源**。车队同构 ⟹ 该口径与车号无关，故借哪台都一样。
        """
        return self.agvs[0].leg_min(src, dst)

    def _step_counters(self, stats: dict) -> "StepCounters":
        """顶层决策点上的**累计读数**（段回报取增量用）。见 `StepCounters`。

        ⚠️ `_tardy` 只把 `plans` 当**作业号的来源**用（`for j in plans` / `dict.fromkeys`），
        故这里不必拿真的计划表 —— 用全体作业号即可。**别改成传 `None`**（会 TypeError）。
        """
        now = float(self.env.now)
        er = self._energy_report(stats, max(now, 1e-9))
        due = self._due_map()
        twt = self._tardy(dict.fromkeys(range(self.inst.n_jobs)), self.completes, due)[1]
        return StepCounters(now=now, done_ops=int(stats.get("ops_done", 0)),
                            done_jobs=len(self.completes),
                            energy_kwh=float(er.get("total_kwh", 0.0)), twt=float(twt))

    def _node_of(self, endpoint: int) -> int:
        """端点号 → 通道节点（机台 `0..m-1`、装卸站 = `m`）。"""
        e = int(endpoint)
        return (int(self.layout.machines[e].dock_node) if e < self.inst.n_machines
                else int(self.layout.lu.dock_node))

    def _la_view(self, job: int, oi: int, frm: int, to: int, ready_at: float) -> LAView:
        """LA 一次派车决策的只读视图——**在"派工那一刻"构造**（步 3 第二稿）。

        ⚠️ **时点变了**：此前在"运输任务生成时"（= 前道完工之后）构造，故 `ready_at ≡ now`，
        Rule 3 塌陷成 Rule 1（§70.8）。现在 LA 与 PA **同一决策点**（原文的 PA→LA 顺序），
        `ready_at` 是**未来**时刻 ⟹ Rule 3 的"不晚于可运输时刻地最晚到达"有真实内容。

        ⚠️ 逐车位置取**当前**节点：车若已有排队任务，其"完成后所在位置"是任务终点而非当前
        节点——**未建模**（登记 §72.1）。逐车空闲时刻用账本 `stats["agv_act"]`。
        """
        from .rule_views import LAVehicle
        snap = self.snapshot()
        act = self._live_stats.get("agv_act") or []
        return LAView(
            now=float(snap.now), job=int(job), oi=int(oi),
            frm_node=self._node_of(frm), to_node=self._node_of(to),
            ready_at=float(ready_at),
            vehicles=tuple(LAVehicle(node=int(v.node), busy_min=float(v.busy_min),
                                     free_at=float(act[a]) if a < len(act) else 0.0)
                           for a, v in enumerate(snap.vehicles)),
            leg_min=self.leg_min, snap=snap)

    def _cold_start(self) -> None:
        """建 t=0 的空世界：**只建对象，不启进程、不抽随机数**。

        `snapshot()` 必须在 `run()` 之前也可用（决策日志与特征层都可能要"初始状态"，
        单测也直接取 t=0 快照），而 `w.machines[m].in_q` 这类活对象在 `run()` 前并不存在。
        假 stats + 全新 env ⇒ 对随后 `run()` 的行为零影响（已用指标逐位对拍证明）。
        """
        self._build_entities(simpy.Environment(), {}, {}, np.random.default_rng(0), bound=True)

    def _due(self, plans: dict[int, list[tuple[int, float]]]) -> dict[int, float]:
        """⑧ 交期（TF/RDD 口径，逐作业）。**开关关闭 → 返回空 dict**（该实例无交期，目标无拖期项）。

        旧口径 `due_factor × Σ工时` **已废弃**（spec §3.5：完全没算排队/运输/争用，
        实测 tardy 恒为 100%）。τ·M_ref 的旧口径亦已废弃（2026-10-03 ⑧ 重设计：内生且
        改进后退化）。旧字段 `due_factor` / `energy_power` 已随 P1b 清理删除。
        """
        return self._due_map()

    def _due_map(self) -> dict[int, float]:
        """交期表（只读 `self.inst` / `self.cfg`）——`_due` 与 `snapshot()` **共用此一处**，防口径漂。

        ⚠️ 无参数：交期**外生**，不随布局/车队规模/参考调度变（旧签名收 `n_jobs` 并由内部
        取 `M_ref`——那正是"交期随我们的配置漂"的病根）。
        ⚠️ `_MREF_BUSY` 短路必须留着：参考调度自身的运行里**不得**算交期——参考运行的指标
        口径把 ⑧ 当关（f^ref 的 TWT 由 `reward.ReferenceObjectives.of` 事后从**同一次运行**的
        `completes` 重算，两边同源正是 Review Focus #5 的要求）。故参考运行内本函数返回 `{}`
        （快照里对应 `JobState.due = 0.0`）。旧口径下它还是递归护栏（求交期要参考运行、
        参考运行又要求交期）；新口径已不可能递归，但"参考运行无交期"的语义要保留。
        """
        if not self.constraints.due_dates or _MREF_BUSY:
            return {}                    # 参考调度自身运行时不递归求交期
        # ⚠️ **本世界内缓存**（2026-10-04 性能批次）：本函数上半段自己写着交期"**外生**"——
        # 只取决于 (实例, τ, R)，与仿真状态无关。而 `snapshot()` **每个决策点**都要读它：
        # 实测 `due_dates_for` 75.9 µs/次 × 960 次/步 = 73 ms（整步的 3.8%），全在重算同一张表。
        # 缓存后每回合只算一次（8 条链 8 次，不是 960 次）。
        # ⚠️ 键**必须带 (τ, R)**：`SimConfig` 不是 frozen，τ 扫描实验会中途改它。
        # ⚠️ 返回的是**同一个 dict**——调用方只读、不得改（既有调用方 `snapshot()` / `_tardy` /
        #    `weighted_tardiness` 都只读，有测试钉口径）。
        key = (self.cfg.tau, self.cfg.due_range)
        if key != self._due_key:
            self._due_val = compute_due_dates(self.inst, self.cfg.tau, self.cfg.due_range)
            self._due_key = key
        return self._due_val

    def _remaining_min(self) -> list[list[float]]:
        """`[job][done]` = 该作业从第 `done` 道工序起的**标称**剩余工时之和——供 `snapshot()` 查表。

        ⚠️ **纯实例数据，与仿真状态无关**，故每世界算一次即可。否则 `snapshot()` 每个决策点
        现算一次：实测 32.1 µs/次 × 960 次/步 = 31 ms（整步的 1.5%）；预计算成表后 1.3 µs/次。
        口径与 `due_dates.total_work_content` 同族（`min(t for _m, t in op)`），
        与原式 `sum(min(...) for op in job_ops[done:])` **逐位相同**（同一批浮点、同一累加序）。
        """
        if self._rem_tbl is None:
            self._rem_tbl = [
                [sum(min(t for _m, t in op) for op in job[d:]) for d in range(len(job) + 1)]
                for job in self.inst.jobs]
        return self._rem_tbl

    def _tardy(self, plans, completes, due) -> tuple[int, float]:
        """(误期作业数, 加权总拖期 TWT)。无交期时两者恒为 0。"""
        if not due:
            return 0, 0.0
        cnt = sum(1 for j in plans if j in completes and completes[j] > due[j])
        weights = dict.fromkeys(plans, 1.0)      # 权重：暂取等权（assumed，见 spec §9）
        return cnt, weighted_tardiness(completes, due, weights)

    def _energy_report(self, stats: dict, makespan: float) -> dict:
        """三态时长 → M2 能耗（spec §3.4，引证 GFJSPT-MMRS）。

        **占用时长的口径**（关键，防"状态统计漏一段"）：
        - 机床：只有 `proc_min`（切削）与 `setup_min`（换型，Task 4 接入）是**显式**累计的，
          其余全部归入 `idle_min`，由**闭合恒等式**给出：`makespan − proc − setup`。
          这样**三个状态之和恒等于在场时长**，不留残差。故障修复期、缓冲满阻塞期、
          等待期都自动落在空闲态——物理上也都对（机床通电但主轴不切削）。
        - AGV：`empty_min` / `loaded_min` 显式累计，`idle_min = makespan × 车数 − 两者之和`。
          待命、等待取货、缓冲区满让步、区段争用等待一并归入空闲态（车辆静止即待机功率）。

        `horizon_hit` 时 makespan 可能小于某台机的加工结束时刻，故 `idle` 取 `max(0, ·)` 兜底。

        **分项导出（2026-10-06，只加键）**：`machine_states_kwh` / `agv_states_kwh` 给出三态各自
        的 [kWh]。动机是 `energy / makespan` 近乎恒定（`progress-log.md` §52.9.3 的前沿退化），
        需要知道**哪一项**随 makespan 走。上面所有旧键的表达式一字未动 ⟹ `total_kwh`（=奖励里的
        `energy`）**逐位不变**，有 `tests/test_energy.py` 的黄金值测试钉死。
        """
        proc, setup = stats["proc_min"], stats["setup_min"]
        per_machine = []
        machine_states_kwh = {"proc": 0.0, "idle": 0.0, "setup": 0.0}
        idle_by_machine = []
        for m in range(len(proc)):
            p = machine_params_for(m, len(proc))
            idle = max(0.0, makespan - proc[m] - setup[m])
            idle_by_machine.append(idle)
            per_machine.append(machine_energy_kwh(proc[m], idle, setup[m],
                                                  idle_kw=p["idle_kw"], proc_kw=p["proc_kw"],
                                                  setup_kw=p["setup_kw"]))
            by_state = machine_energy_breakdown_kwh(proc[m], idle, setup[m],
                                                    idle_kw=p["idle_kw"], proc_kw=p["proc_kw"],
                                                    setup_kw=p["setup_kw"])
            for state, kwh in by_state.items():
                machine_states_kwh[state] += kwh
        machine_kwh = float(sum(per_machine))
        agv_idle = max(0.0, makespan * self.cfg.n_agv
                       - stats["agv_empty_min"] - stats["agv_loaded_min"])
        agv_kwh = agv_energy_kwh(agv_idle, stats["agv_empty_min"], stats["agv_loaded_min"])
        return {"machine_kwh": machine_kwh, "agv_kwh": agv_kwh,
                "shop_kwh": total_energy_kwh(0.0, 0.0, makespan),   # 仅车间固定项
                "total_kwh": total_energy_kwh(machine_kwh, agv_kwh, makespan),
                "machine_kwh_per_machine": per_machine,
                "agv_states_min": {"idle": agv_idle, "empty": stats["agv_empty_min"],
                                   "loaded": stats["agv_loaded_min"]},
                # ── 分项（新键；旧键一字未动）──
                "machine_states_kwh": machine_states_kwh,
                "machine_states_min": {"proc": list(proc), "idle": idle_by_machine,
                                       "setup": list(setup)},
                "agv_states_kwh": agv_energy_breakdown_kwh(
                    agv_idle, stats["agv_empty_min"], stats["agv_loaded_min"])}

    def snapshot(self) -> "Snapshot":
        """当时的活状态 → 纯数据快照（**只读**，不改变任何仿真状态）。

        `_job_progress[j]` / `_job_loc[j]` / `_job_agv[j]` 由 `MachineSim` 与 `AgvSim` 维护
        （见 `run()` 里的初始化），是"作业进行到哪一步"的唯一真相。

        **口径**：`MachineState.remaining_min` 是**标称**剩余——`上机时刻 + 工序时长 − now`，
        即只算工序本身的时长，**不把换型、故障修复这些墙钟延长算进去**（故异常长的停机可能
        让它触底为 0，而工件实际还在机台上）。`JobState.remaining_min` 同理，是剩余各工序的
        **标称**最短候选工时之和。`JobState.due` = ⑧ 的交期（与 `run()` 同口径的 `_due_map`；
        约束关闭或参考运行内为 0.0）。`in_flight` = **队列里待取** + **车上在运**（两种队列形状
        都算，见下）。⚠️ **v2：④ 返工删后已无 `rework_cnt`**（2026-10-08）。
        ⚠️ **v2：① 去后 `VehicleState.zone_wait` 已删**（2026-10-08）。
        `queued_tasks` = **队列内容**（逐件身份，⑩ 拼批头的原料；两种队列形状都接，见下）。
        """
        from .snapshot import JobState, MachineState, QueuedTask, Snapshot, VehicleState
        ms = []
        for i, m in enumerate(self.machines):
            q = m.in_q.items
            cur = self._cur_op[i]
            ms.append(MachineState(
                backlog_min=float(sum(it[2].time for it in q)),
                in_q_len=len(q),
                out_q_len=len(m.out_q.items),
                busy=bool(m.slot.count),
                remaining_min=(max(0.0, cur[2] + cur[1] - float(self.env.now)) if cur else 0.0),
                pm_used_min=float(m.pm_clock), fail_rate=float(m.pad.fail_rate),
                # ② 有限缓冲：容量只在**约束开启**时报，否则报 `None`（= 无界，真值）。
                # ⚠️ **不得恒报 `pad.in_cap`**：② 关时两个 `Store` 无界，报一个容量会让
                # `build_pa_view` 把没满的机台滤掉、让 `in_queue` 维按一个不存在的上限归一。
                # ⚠️ **`self.constraints`，不是 `self.con`** —— 后者是 `AgvSim` / `MachineSim`
                # 的字段名；`snapshot()` 是 `SimWorld` 的方法。（实测踩过：写成 `self.con`
                # 在**有策略**的路径上才炸 —— 规则档不调 `snapshot()`，所以规则档冒烟是绿的。）
                in_cap=(int(m.pad.in_cap) if self.constraints.finite_buffer else None)))
        js = []
        # ⑧ 交期与 run() **同口径**（`_due_map`）：未 run 的 t=0 快照也取（交期是纯实例数据的
        # 查询，无墙钟开销）；参考运行自身运行时返回 {}（0.0 哨兵）。
        due = self._due_map()
        rem = self._remaining_min()          # 预计算表（见 `_remaining_min`）：每快照省 32 µs
        for j, job_ops in enumerate(self.inst.jobs):
            done = self._job_progress[j]
            js.append(JobState(
                done_ops=done, total_ops=len(job_ops),
                remaining_min=float(rem[j][done]),
                finished=done >= len(job_ops), due=float(due.get(j, 0.0)),
                at_machine=int(self._job_loc[j]),
                in_transit=bool(self._job_agv[j] >= 0), on_agv=int(self._job_agv[j]),
                ))
        vs = []
        # 步 3：逐车累计行驶工时（LA 的 `MINU` 自变量）。`_cold_start` 与手搓 stats 的夹具
        # 不带此键 ⟹ `.get` 兜底成"全 0"（= 还没跑过，真值，不是哨兵）。
        agv_busy = self._live_stats.get("agv_busy_min") or []
        for a, agv in enumerate(self.agvs):
            vs.append(VehicleState(
                status=3 if agv.down else (2 if self._agv_loaded[a] else
                                           (1 if agv.pos_node is not None else 0)),
                node=int(agv.pos_node if agv.pos_node is not None else -1),
                queued=len(self.tasks_in[a].items) if agv.bound else 0,
                # ⚠️ **2026-10-10：⑪ 去后 `battery_frac` 已删**（车队无电量）。
                busy_min=float(agv_busy[a]) if a < len(agv_busy) else 0.0))
        # 在途 = **队列里待取** + **车上在运**。队列有两种形状（绑定=每车一 Store，
        # FIFO=单个共享 Store）——只看 `agv.bound` 那种形状会让 FIFO 路径恒为 0（F5）；
        # 车上那部分必须单独数，因为已装车的批次**已经离开队列**。
        queued = (sum(len(s.items) for s in self.tasks_in) if isinstance(self.tasks_in, list)
                  else len(self.tasks_in.items))
        # ⑩ 拼批头：队列**内容**（逐件身份，队列序）。两种形状都接——`bound` 每车一张表
        # 标 `veh = 车号`；FIFO 共享表标 `veh = -1`。任务身份口径 = `_transporter` 的
        # `task = (frm, to, item, _path)`，**不另造**（见 `snapshot.QueuedTask`）。
        qt: list[QueuedTask] = []
        for a, store in (list(enumerate(self.tasks_in)) if isinstance(self.tasks_in, list)
                         else [(-1, self.tasks_in)]):
            for t in store.items:
                item = t[2]
                qt.append(QueuedTask(veh=int(a), job=int(item[0]), frm=int(t[0]),
                                     to=int(t[1]), oi=int(item[1])))
        return Snapshot(now=float(self.env.now), machines=tuple(ms), jobs=tuple(js),
                        vehicles=tuple(vs), n_done=len(self.completes),
                        in_flight=queued + sum(self._agv_load_n),
                        # ⚠️ **v2：① 去后 `zone_holder` / `zone_wait` 已删**（2026-10-08）
                        # ⚠️ **2026-10-10：⑪ 去后 `chargers=` 已删**（无充电桩，C 段 token 同批删）。
                        queued_tasks=tuple(qt))

    # ⚠️ **v2：① 去后 `_zone_wait_max` 已删**（2026-10-08）

    def _make_seq_decide(self, rank: list[list[float]]) -> SeqDecide:
        """把**静态工序优先表**编成 `decide(mach, jobs) -> 候选序号`（spec §5.1）。

        `rank[job][oi]` 越小越先；`+inf` = 未覆盖（`check_seq_rank` 已规范化）。
        候选 = 缓冲内各件、按到达序；**同 rank 取序号最小的**（= FIFO 队首）——
        确定性由 `min` 的"首个最小"语义保证，不另设随机源。
        `oi` 从 `SimTrack.job_progress` 取（**与 `_job_progress` 同源、不新增记账**）；
        `self.track` 由 `_build_entities` 建出，故这里**惰性**读属性、不预取。
        ⚠️ 本函数是**纯函数**：只看 `rank` 与"各作业进行到第几道工序"，不读仿真状态
        （缓冲件数、机台工况一概不看）——静态表天然如此；在线路径走 `policy_q` 闭包（§5.1）。
        """
        def decide(mach: int, jobs: tuple[int, ...]) -> int:
            # ⚠️ `mach` 在本解码下用不到（静态表与机台无关）；保留形参是为了与在线 Q 头
            #    同型（spec §3.2 的契约只有一份），也让"机台"进得了将来的状态依赖规则。
            _ = mach
            prog = self.track.job_progress
            return min(range(len(jobs)), key=lambda i: rank[jobs[i]][prog[jobs[i]]])
        return decide

    def run(self, seed_chain: int = 0, op_choices: list[list[int]] | None = None,
            agv_phi: list[int] | None = None,
            seq_rank: list[list[int | None]] | None = None,
            policy_pa=None) -> dict:
        """op_choices[job][op_idx] = 该工序选第几个候选；缺省=每工序取最短候选（v0 调度器）。

        agv_phi[task_i] = 第 task_i 个运输任务的 AGV id（L 层决策的载体；None=旧 FIFO 规则）。
          task_i 按 transporter 生成序 0,1,2,...；未覆盖的采用轮询 (i % n_agv)（确定性兜底）。
          绑定模式：每台车一个任务队列（bound 路径）—— 与旧"空闲车接活"语义不同（基线数字仅
          None 路径口径，论文对照会注明）。

        `seq_rank[job][oi]` = 该工序的**优先级**（越小越先；`None` = 未覆盖 = `+inf`）。
          **None（默认）= 输入缓冲是裸 `simpy.Store`，FIFO 取件，逐位不变**（spec §3.4）。
          非 None = 机台在**已入缓冲的件**里挑 rank 最小的先加工（同 rank 取 FIFO 队首）——
          即"缓冲内优先级"，机台只要缓冲非空就永不闲置（非延迟解码器，spec §2.1(A)）。
          形态不符由 `check_seq_rank` **显式报错**（不静默忽略）。
        ⚠️ `seq_rank` **不进 `SimConfig`**（同 `op_choices` 的同类处置，spec §3.5）：
          进 cfg 就会进 `_cfg_key` ⟹ 参考运行缓存 `M_ref` 全部失效，且
          `EvalSpec.from_cfg` / `check_same_dynamics` 按名枚举、看不见新字段（静默缺口）。
          实例规模的数据不是运行时标度参数。

        ⚠️ **`policy_pa`（生产派工头，PA；步 3）**：`None`（默认）= 不启用——`plans` 走
          离线预填，**逐位等于 step 2 之前的行为**。非空 = `_pa_dispatch` 在「未完工工件的
          最早待加工工序」池里挑一件**并定机台**写进 `plans`（预留表），transporter 只读。
          契约：`policy_pa(view, pool, tie_rng) -> (池下标, 机台号)`，
          `view` = `env/rule_views.PAView`，`tie_rng` = **环境给的平局随机源**
          （独立子流 `[seed_chain, PA_TIE_SEED]`，照 AEI2025 的随机平局）。
          固定规则用 `hrl.rules.pa_rule_policy(...)` 造。

        返回 metrics：makespan / energy / fail_events / tardy / moves / deliveries / seq_*。
        """
        seq_decide = (None if seq_rank is None
                      else self._make_seq_decide(check_seq_rank(self.inst, seq_rank)))
        rng = np.random.default_rng(seed_chain)
        # 步 3：PA 平局随机源——**独立子流**，每局重播（见 `PA_TIE_SEED` 的注释；
        # 每次派工重建会让 55 次派工抽到同一串数，那是错的）。
        self._pa_tie_rng = np.random.default_rng([seed_chain, PA_TIE_SEED])
        stats = {"fail_events": 0, "process_time": 0.0, "travel_time": 0.0,
                 "moves": 0, "deliveries": 0,
                 "agv_del": [0] * self.cfg.n_agv, "agv_pos": [-1] * self.cfg.n_agv,
                 # M2 能耗的三态时长（P1b Task 2）：机床按机位计，AGV 按车队合计
                 "proc_min": [0.0] * self.inst.n_machines,
                 "setup_min": [0.0] * self.inst.n_machines,
                 "agv_empty_min": 0.0, "agv_loaded_min": 0.0,
                 # 步 3：**逐车**累计行驶工时（LA 的 `MINU` 规则的自变量；快照逐车报出）
                 "agv_busy_min": [0.0] * self.cfg.n_agv,
                 # 步 3 第二稿：逐车**已派任务账本**（预计空闲时刻；LA 的 `LATEST_IN_TIME` 用）
                 "agv_act": [0.0] * self.cfg.n_agv,
                 # 生产侧约束的事件计数（④⑤⑫）——binding 实测与消融表的读数口径
                 # ⚠️ T3 ⑫：`pm_events` = forced + chosen（拆键见 `MachineSim._pm_after_op`）。
                 "pm_events": 0,
                 "pm_events_forced": 0, "pm_events_chosen": 0,
                 # 物流侧约束的事件计数（⑨）
                 "trips": 0, "agv_fail_events": 0,
                 # ② 有限缓冲的两半**激活量**（被迫等待累计 [min]）：
                 # `buffer_block_min` = AGV 投递撞上 `in_q` 满；`out_block_min` = 机台撞上 `out_q` 满
                 "buffer_block_min": 0.0,
                 "out_block_min": 0.0,
                 # ⑨ failover 的记账：故障期间被退回/转交的任务件数（默认关 ⟹ 恒 0）
                 "agv_failover_tasks": 0,
                 # ⑨ 的三个故障场景（SC1 空闲 / SC2 空载段中 / SC3 负载段中）逐场景计数；
                 # `agv_at_min` = 被中断任务**已消耗的运输时间**（原文的 `at`，计入 TC）；
                 # `agv_sc3_delivered` = SC3 里"修理点恰为终点 ⟹ 不算中断"的次数。
                 "agv_fail_sc1": 0, "agv_fail_sc2": 0, "agv_fail_sc3": 0,
                 "agv_sc3_delivered": 0, "agv_at_min": 0.0, "agv_interrupted_min": 0.0,
                 # ⚠️ **2026-10-10：⑪ 去后 `battery_min_kwh` 初值已删**（车队无电量）。
                 "tasks_get": 0, "requeue": 0, "in_q_gets": 0,
                 "trans_evt": 0, "tasks_put": 0,
                 # P4-B：矩阵覆盖不到的端点走几何降级时的**留痕**（段数与分钟数）
                 "unmapped_legs": 0, "unmapped_min": 0.0,
                 # 工序排序入口的记账（spec §3.6）。**关态也计**：机会率要在关态测
                 # （开着测会与策略/优先级表的实际选择混淆）。
                 "seq_gets_ge2": 0, "seq_reorders": 0}
        env = simpy.Environment()
        inst = self.inst
        lu_idx = inst.n_machines             # 端点号约定：机台 0..m-1、装卸站 = m
        # 计划表：job -> [机台号]（按 op_choices 或贪婪最短选择）。⚠️ 与 run_gated 同形状：
        # 选机一律走 `_pick_machine`（离线=查表），时长在那里就地查 alts。所选时长另收一份
        # 供 horizon 用（求和顺序与旧式 `sum(t for ops in plans.values() for _, t in ops)`
        # 逐位相同，见 P2 Task 5 的 A/B 对拍）。
        plans = {}
        chosen_times: list[float] = []
        for j, job_ops in enumerate(inst.jobs):
            plan = op_choices[j] if (op_choices and j < len(op_choices) and op_choices[j]) else [
                int(np.argmin([t for _, t in alts])) for alts in job_ops]
            # ⚠️ **v2**：给了 `policy_pa` 时**留空**（`None`）—— 由 PA 派工器逐道填（见 `_pa_dispatch`）。
            # 否则 `plans` 预填满 ⟹ PA 的池恒空 ⟹ 它一次都不会被调用（实测：派工 0 次）。
            plans[j] = ([None] * len(job_ops) if policy_pa is not None
                        else [job_ops[oi][plan[oi]][0] for oi in range(len(job_ops))])
            # horizon 的工时估计仍用贪婪计划（PA 在线时读不到时长，用最坏/最优估均可）
            chosen_times.extend(job_ops[oi][plan[oi]][1] for oi in range(len(job_ops)))
        completes: dict[int, float] = {}
        done = env.event()                      # 完工事件：末件到站即成功（见下"停表"注释）
        bound = agv_phi is not None
        (entities, tasks_in,
         events_q) = self._build_entities(env, stats, completes, rng, bound=bound,
                                          seed_chain=seed_chain,
                                          seq=seq_decide, done=done)
        for e in entities:                      # ⚠️ 启动顺序不得变（同刻事件次序由注册顺序定）
            env.process(e.run())
        for agv in self.agvs:
            env.process(agv.run())
        # 全量注入（2026-10-02：分批门控已删，见 progress-log §12.5——所有作业一次投放）。
        # ⚠️ P4-B Task 2b 起投放**不是瞬移**：每个作业发一条"装卸站 → 首工序机台"事件，
        # 由 transporter 走正常派车路径生成任务（端点号 = 装卸站）。
        jkeys = list(plans.keys())
        for j in jkeys:
            env.process(self._release(env, events_q, (lu_idx, j, -1, OpLite(0.0), False)))
        env.process(self._transporter(env, events_q, tasks_in, plans, entities, stats, lu_idx,
                                      bound=bound, agv_phi=agv_phi, policy_pa=policy_pa))
        total_work = sum(chosen_times)          # = 所选候选的时长之和（口径与旧式逐位相同）
        horizon = float(total_work * 6 + 500)   # v0 护栏升格：B 层门控（cap=2）下运行可远长于
                                                # 无门控（波形化串行）；3× 护栏曾把门控运行掐
                                                # 表截断（实测 9/10 假死——horizon 不足非死锁）
        # ⚠️ **停表修复（2026-10-07）：完工即停，horizon 退居真·死锁护栏。**
        # 缺陷：`env.run(until=horizon)` 总把仿真时间**推到护栏**才停。报告档 B（矩阵口径）
        # 实测 MK01 makespan 711 min / horizon 1418 min ⟹ 完工后还空转 ~707 min。空转期间
        # **事件进程照跑**：`AgvSim._failures` 每抽一次就 `agv_fail_events += 1`（每车一条独立
        # 泊松流）⟹ 该计数按**护栏**累积而非按 episode：n_agv=6 时实测 25 次，真 episode 的
        # 期望仅 6×711/485 ≈ 8.8 次（MK07 更甚：44 次 vs 5×1527/485 ≈ 15.7）。
        # 受影响的是**尾巴里还会触发的事件计数**（AGV 故障；以及尾巴里仍在跑的行程/充电记账）；
        # `fail_events`/`pm_events` 只在加工中触发，末件回站前全部结束 ⟹ 不动。
        # 修法：末件到站（`LuStation.run` 记 `completes` 处）成功 `done`；停表条件用 simpy 的
        # `|` 合成（`done | env.timeout(horizon)`）——跑完即停，永不到站的真死锁仍由护栏兜住。
        # ⚠️ `horizon_hit` 的判据**不变**（`len(completes) < n_jobs`）：完成度是"做完没有"，
        # 不是"停表条件谁触发"——护栏那一刻到站的末件照样算跑完（SimPy 的 run 必然推进到
        # `until`，旧的"时间到即掐表"写法恒真，见 run_gated 同注释）。
        env.run(until=done | env.timeout(horizon))
        stats["horizon_hit"] = len(completes) < inst.n_jobs   # 掐表=未完成（SimPy run 必然推进至
                                                              # until：时间比较恒真，须以完成度判）
        due = self._due(plans)
        makespan = (max(completes.values()) if completes else env.now)
        energy = self._energy_report(stats, makespan)
        return {"makespan": makespan,
                "completes": dict(completes),          # 每作业完工时刻（交期校准 / TWT 需要）
                "energy": energy["total_kwh"],         # M2 引证模型（spec §3.4），单位 kWh
                "energy_breakdown": energy,
                "setup_minutes_total": float(sum(stats["setup_min"])),
                                "pm_events": stats["pm_events"],
                # ⑫ 拆键（T3）：forced = 被阈值强制触发（T3 的激活量）、chosen = 策略主动选
                "pm_events_forced": stats["pm_events_forced"],
                "pm_events_chosen": stats["pm_events_chosen"],
                "trips": stats["trips"],
                # ② 有限缓冲的激活量（**两处返回字典都要列** —— `metrics` 是逐键列的，
                # 漏一个键就会 `m.get(k, 0)` 取到 0，把"有阻塞"读成"没阻塞"。实测踩过。）
                "buffer_block_min": stats.get("buffer_block_min", 0.0),
                "out_block_min": stats.get("out_block_min", 0.0),
                "agv_fail_events": stats["agv_fail_events"],
                # ⑨ failover：故障期间退回/转交的任务件数（默认关 ⟹ 恒 0）
                "agv_failover_tasks": stats["agv_failover_tasks"],
                # ⚠️ v2：⑩ 去后 batch_* 三个读数已删（2026-10-08）
                # ⚠️ **2026-10-10：⑪ 去后 `battery_min_kwh` 读数已删**（车队无电量）。
                "n_agv": self.cfg.n_agv, "fleet_size": self.layout.n_agv,
                # 完成度守卫（`reward._incomplete_objectives`）用的三个量：掐表时长 + 规模。
                # `horizon` = 本次运行的**护栏**（6×工时+500），是全部完工时刻的严格上界——
                # 跑完时它**不被取到**（停表在 makespan），故它是一个上界而不是实测值。
                # `stop_time` = `env.run` 的**实际停表时刻**（跑完 = makespan；掐表 = horizon）：
                # 停表修复后"仿真跑到哪儿"这件事必须可观测，否则护栏与实测分不开。
                "horizon": float(horizon),
                "stop_time": float(env.now),
                "n_jobs": self.inst.n_jobs, "n_machines": self.inst.n_machines,
                "fail_events": stats["fail_events"],
                "tardy": self._tardy(plans, completes, due)[0],
                "tardy_twt": self._tardy(plans, completes, due)[1],
                "moves": stats["moves"], "deliveries": stats["deliveries"],
                # 工序排序入口的读数（spec §3.6）：机会率 = `seq_gets_ge2 / dbg["in_q_gets"]`；
                # `seq_reorders` = 真选了非 FIFO 队首的次数（"决策真的发生了"的硬证据）。
                # 两条路径的 keyset 必须一致，故此处与本入口同键；`run_gated` 没有排序回调
                # ⟹ 它的 `seq_reorders` 恒 0，而 `seq_gets_ge2` 照计（机台侧无条件记账）。
                "seq_gets_ge2": stats.get("seq_gets_ge2", 0),
                "seq_reorders": stats.get("seq_reorders", 0),
                "horizon_hit": stats.get("horizon_hit", False),
                "ops_done": stats.get("ops_done", 0), "jobs_done": len(completes),
                # 步 3：PA 派工次数（**"派工器真的被调用了"的硬证据**——步 2 曾实测为 0，
                # 原因是 `plans` 被离线路径预填满、池恒空。见 progress-log §69）。
                "pa_assigns": stats.get("pa_assigns", 0),
                # 扰动触发的决策点数（原文决策点定义的**另一半**，2026-10-08 补齐）
                "dist_events": stats.get("dist_events", 0),
                # ③ 机障发出的扰动哨兵数（顶层触发之一）+ 顶层实际决策次数
                "mf_dist_events": stats.get("mf_dist_events", 0),
                "ma_decisions": stats.get("ma_decisions", 0),
                # ⑨ 的三个故障场景分布 + 中断记账（`at` 是 TC 的一项，见式(48)）
                "agv_fail_sc1": stats.get("agv_fail_sc1", 0),
                "agv_fail_sc2": stats.get("agv_fail_sc2", 0),
                "agv_fail_sc3": stats.get("agv_fail_sc3", 0),
                "agv_sc3_delivered": stats.get("agv_sc3_delivered", 0),
                "agv_at_min": float(stats.get("agv_at_min", 0.0)),
                "agv_interrupted_min": float(stats.get("agv_interrupted_min", 0.0)),
                "task_flow": stats.get("task_flow", []),
                "agv_load": stats.get("agv_load", []),
                "dbg": {k: v for k, v in stats.items()
                        if k in ("in_q_gets", "trans_evt", "tasks_put", "tasks_get", "requeue",
                                 "pa_assigns", "dist_events", "mf_dist_events",
                                 "ma_decisions")},
                "travel_time_total": float(stats["travel_time"]),
                # 物流成本（**借 AEI2025 式(48) 的成本模型**，只为让读数可与其表并列）：
                #   `TC = c·Σ y_v + uc·(Σ(nct−nst+lct−lst) + at)`
                # - `Σ y_v = n_agv`：本仓**车队固定、全部常开**（⑩ 异构车队删后无启用决策）；
                # - `Σ(空载段+负载段) = travel_time_total`（逐车累计行驶工时的车队合计）；
                # - `at`（故障中断时间）**恒 0**：⑨ 段间检出、不打断正在进行的行驶。
                # ⚠️ **这不是我们的目标函数**（我们的是 makespan + energy + TWT），
                # 只是**报告桥**——不加它，任何"比较 TC"的说法都不成立。见 §77.4 第 2 行。
                "logistics_cost": (
                    AEI2025_AGV_ACTIVATION_COST * self.cfg.n_agv
                    + AEI2025_UNIT_TRANSPORT_COST * (float(stats["travel_time"])
                                                     + float(stats.get("agv_at_min", 0.0)))),
                # ⚠️ **v2：① 去后 `zone_wait` / `n_zones` 已删**（2026-10-08）
                # 口径**随结果自报**（P4-A Review Focus #1 同型）：两种口径的数并排放时靠这两行分辨
                "transport": self.inst.transport,
                "unmapped_legs": stats.get("unmapped_legs", 0),
                "unmapped_min": float(stats.get("unmapped_min", 0.0))}

    def run_gated(self, seed_chain: int = 0, op_choices: list[list[int]] | None = None,
                  policy_l=None, policy_s=None, online_s: bool = False,
                  policy_m=None, policy_pa=None,
                  policy_ma=None, dt_events: int = 0,
                  stop_after_ma: int = 0) -> dict:
        """L 层门控式运行（真·事件驱动决策的同步实现）+ **在线 S 层**（P2 Task 5）+ 在线 R 层。

        SimPy 单线程确定性 ⇒ transporter 生成任务时**同步调用** L 层策略，并当场把**当时的**
        活状态快照交给它（无并发 → 无需事件/Gate）。S/L/R 的回调契约对称：
        `policy_l(snap, job, frm, to, oi+1, cand_v) -> 车号`（候选 = 全车队）；
        `policy_s(snap, job, oi, cand) -> 机台号`；
        `des.py` 只给原始材料、**不构造任何特征**（层次纪律：`env/` 不得依赖 `nn/`）。

        `online_s=False`（默认）：S 层读**预计算**的 `plans`（`op_choices` 或贪婪最短）——
        **逐位复现 P2 之前的静态行为**。
        `online_s=True`：每道工序在**前驱完成后、即将入机台时**调
        `policy_s(snap, job, oi, cand) -> 机台号` 决策（首工序无前驱 ⇒ 在 t=0 投放点决策，
        此刻各队列皆空；同刻投放的作业共享同一初始视界）。`plans` 随之退化为**决策日志**
        （`plans[job][oi]` 在决策后写入），故**决策前不得读它**。

        ⚠️ **v2：路线头（R）与 `policy_r` / `route_k` 已删**（2026-10-08）——
        ① 拥堵去掉后"选远路严格更差"，路线决策没有意义。

        ⚠️ **`policy_m`（⑫ 维护头，M）**：`None`（默认）= 规则驱动 ⟹ 逐位等于今日行为；
        非空 = **策略驱动**：一道工序加工完毕、下一件尚未上机时（`MachineSim._pm_after_op`）
        由策略在 {现在保养, 不保养} 里选。契约与 S/L/R 对称：
        `policy_m(snap, mach, PM_CANDS) -> 动作码`（`PM_CANDS[0]` = 现在保养）。
        ⚠️ 逾期（`pm_clock >= pm_interval`）**不进回调**——强制保养由仿真直接执行，
        不产生决策：规则是硬底线，策略只能把保养提前、不能推迟过点。
        ⚠️ `policy_m` 非空但 ⑫ `maintenance` 关闭 → **显式报错**：⑫ 关时 `pm_clock` 根本
        不累加（决策点不存在），给策略一个死动作只会污染链 logp。同 `policy_r` 的形态。
        ⚠️ **报告配置（`REPORT_TIERS` 的 `B-8` 档）关 ⑫ ⟹ 本头在报告配置里不可达**
        （非空即报错）。代码**保留**：消融 / 机制验证档（短 `pm_interval`）仍可显式打开 ⑫
        用她；报告结果里没有她——这是"两机制从不激活、移出报告配置"的代价，如实声明。

        ⚠️ **2026-10-10：⑪ 充电头 `policy_c` 整条删除**（用户裁定「删机制 反正不触发 不要了」）
        —— 连同原「规则驱动 vs 策略驱动」两档、`charging` 关闭时的显式报错、
        与 `_make_charge_fn` 一并删。车队无电量、无充电桩，这个决策点不存在。
        （原「报告配置里本就不可达」的那段说明随之作废：**不是不可达，是没了**。）

        ⚠️ **v2：⑩ 拼批头（B）与 `policy_b` 已删**（2026-10-08）—— `SimConfig.multi_drop`
        与拼批一起移除，本入口不再有第六个回调。

        ⚠️ **`policy_pa`（生产派工头，PA）**：`None`（默认）= 不启用（离线预填计划）；
        非空 = **策略/规则驱动**：`_pa_dispatch` 在「未完工工件的最早待加工工序」池里挑
        **一件并定机台**，写进 `plans`（预留），transporter 只读。契约见 `_pa_dispatch`：
        `policy_pa(view, pool, tie_rng) -> (池下标, 机台号)`，`view` = `env/rule_views.PAView`，
        `tie_rng` = 平局随机源（独立子流）。
        ⚠️ 与 `online_s=True` **互斥**（显式报错）：两者都定机台，会互相覆盖同一格预留。

        ⚠️ 多个回调都只拿**原始快照**（`env/` 不构造特征、不得依赖 `nn/`）——快照→特征在
        `algo/` 层做。决策留痕同理：`group_rel.roll_chain` 自记自己的 `Decision` 链
        （旧的 `dict["decision_log"]` 回传因零消费者已删，评审 M-2）。
        """
        if (policy_l is None and not online_s and policy_m is None
                and policy_pa is None):
            return self.run(seed_chain=seed_chain, op_choices=op_choices, policy_pa=policy_pa)
        if online_s and policy_s is None:
            raise ValueError("online_s=True 需要 policy_s 回调（S 层决策入口）")
        if online_s and op_choices is not None:
            raise ValueError("online_s=True 时计划由 policy_s 在线产生，op_choices 不生效——"
                             "两者同传会静默忽略计划，故直接报错")
        if online_s and policy_pa is not None:
            raise ValueError(
                "online_s=True 与 policy_pa 同传：步 3 起**机台由 PA 定**（复合规则的机台那一半），"
                "`_pick_machine` 只读 PA 写下的预留；再开在线 S 层会让它当场覆盖同一格预留，"
                "两个机台决定互相打架。请二选一。")
        pm_fn = None
        if policy_m is not None:
            if not self.constraints.maintenance:
                raise ValueError(
                    "policy_m 非空但 ⑫ 维护关闭：⑫ 关时 `pm_clock` 根本不累加、保养事件恒 0"
                    "（决策点不存在），给策略一个死动作只会污染链 logp。"
                    "请开 ⑫ 或传 policy_m=None。")
            pm_fn = self._make_pm_fn(policy_m)
        # ⚠️ **2026-10-10：⑪ 去后 `charge_fn` 与它的 `charging` 守卫已删**
        # （原为"`policy_c` 非空但 ⑪ 关闭 ⟹ 显式报错"——那个决策点现在不存在了）。
        # ⚠️ v2：⑩ 去后 `policy_b` / `batch_fn` 已删（2026-10-08）
        rng = np.random.default_rng(seed_chain)
        # 步 3：PA 平局随机源——**独立子流**，每局重播（见 `PA_TIE_SEED` 的注释；
        # 每次派工重建会让 55 次派工抽到同一串数，那是错的）。
        self._pa_tie_rng = np.random.default_rng([seed_chain, PA_TIE_SEED])
        stats = {"fail_events": 0, "process_time": 0.0, "travel_time": 0.0,
                 "moves": 0, "deliveries": 0,
                 "agv_del": [0] * self.cfg.n_agv, "agv_pos": [-1] * self.cfg.n_agv,
                 # M2 能耗的三态时长（P1b Task 2）：机床按机位计，AGV 按车队合计
                 "proc_min": [0.0] * self.inst.n_machines,
                 "setup_min": [0.0] * self.inst.n_machines,
                 "agv_empty_min": 0.0, "agv_loaded_min": 0.0,
                 # 步 3：**逐车**累计行驶工时（LA 的 `MINU` 规则的自变量；快照逐车报出）
                 "agv_busy_min": [0.0] * self.cfg.n_agv,
                 # 步 3 第二稿：逐车**已派任务账本**（预计空闲时刻；LA 的 `LATEST_IN_TIME` 用）
                 "agv_act": [0.0] * self.cfg.n_agv,
                 # 生产侧约束的事件计数（④⑤⑫）——binding 实测与消融表的读数口径
                 # ⚠️ T3 ⑫：`pm_events` = forced + chosen（拆键见 `MachineSim._pm_after_op`）。
                 "pm_events": 0,
                 "pm_events_forced": 0, "pm_events_chosen": 0,
                 # 物流侧约束的事件计数（⑨）
                 "trips": 0, "agv_fail_events": 0,
                 # ② 有限缓冲的两半**激活量**（被迫等待累计 [min]）：
                 # `buffer_block_min` = AGV 投递撞上 `in_q` 满；`out_block_min` = 机台撞上 `out_q` 满
                 "buffer_block_min": 0.0,
                 "out_block_min": 0.0,
                 # ⑨ failover 的记账：故障期间被退回/转交的任务件数（默认关 ⟹ 恒 0）
                 "agv_failover_tasks": 0,
                 # ⑨ 的三个故障场景（SC1 空闲 / SC2 空载段中 / SC3 负载段中）逐场景计数；
                 # `agv_at_min` = 被中断任务**已消耗的运输时间**（原文的 `at`，计入 TC）；
                 # `agv_sc3_delivered` = SC3 里"修理点恰为终点 ⟹ 不算中断"的次数。
                 "agv_fail_sc1": 0, "agv_fail_sc2": 0, "agv_fail_sc3": 0,
                 "agv_sc3_delivered": 0, "agv_at_min": 0.0, "agv_interrupted_min": 0.0,
                 # ⚠️ **2026-10-10：⑪ 去后 `battery_min_kwh` 初值已删**（车队无电量）。
                 "tasks_get": 0, "requeue": 0, "in_q_gets": 0,
                 "trans_evt": 0, "tasks_put": 0,
                 # P4-B：矩阵覆盖不到的端点走几何降级时的**留痕**（段数与分钟数）
                 "unmapped_legs": 0, "unmapped_min": 0.0,
                 # 工序排序入口的记账（spec §3.6）。**关态也计**：机会率要在关态测
                 # （开着测会与策略/优先级表的实际选择混淆）。
                 "seq_gets_ge2": 0, "seq_reorders": 0}
        env = simpy.Environment()
        inst = self.inst
        lu_idx = inst.n_machines             # 端点号约定：机台 0..m-1、装卸站 = m（同 run()）
        # 计划表：job -> [机台号]。离线 = 预填（同 run()）；在线 = 空 **决策日志**，
        # 由 `_pick_machine` 决策后写入（`None` = 尚未决策，决策前读它即 bug）。
        plans: dict[int, list] = {}
        for j, job_ops in enumerate(inst.jobs):
            if online_s or policy_pa is not None:
                # 在线 S 层：`plans` 是**决策日志**（`_pick_machine` 决策后写入）。
                # 步 3 的 PA：`plans` 是**预留表**（`_pa_dispatch` 批量填满，transporter 只读）。
                # 两条路的共同点 = 决策前不得读它。
                plans[j] = [None] * len(job_ops)
            else:
                plan = op_choices[j] if (op_choices and j < len(op_choices) and op_choices[j]) else [
                    int(np.argmin([t for _, t in alts])) for alts in job_ops]
                plans[j] = [job_ops[oi][plan[oi]][0] for oi in range(len(job_ops))]
        completes: dict[int, float] = {}
        done = env.event()                      # 完工事件（同 run()：末件到站即成功）
        (entities, tasks_in,
         events_q) = self._build_entities(env, stats, completes, rng, bound=True,
                                          seed_chain=seed_chain,
                                          pm=pm_fn,
                                          done=done,
                                          machine_disturbance=policy_ma is not None)
        for e in entities:                      # ⚠️ 启动顺序不得变（同刻事件次序由注册顺序定）
            env.process(e.run())
        for agv in self.agvs:
            env.process(agv.run())
        jkeys = list(plans.keys())
        for j in jkeys:                         # 全量注入（同 run()：投放走装卸站 → 正常派车路径）
            env.process(self._release(env, events_q, (lu_idx, j, -1, OpLite(0.0), False)))
        stop_ev = env.event()                   # 跑到第 k 个顶层点就停（§4.2 的"跑 ΔT"）
        env.process(self._transporter(env, events_q, tasks_in, plans, entities, stats, lu_idx,
                                      bound=True, policy_l=policy_l,
                                      policy_s=policy_s, online_s=online_s,
                                      policy_pa=policy_pa,
                                      policy_ma=policy_ma, dt_events=dt_events,
                                      stop_after_ma=stop_after_ma, stop_event=stop_ev))
        # horizon 的工时上界：**按最短候选**估（在线时 `plans` 是决策日志、读不到时长；
        # 离线且用默认贪婪计划时该式 = 各工序所选时长之和，与旧式逐位相同）。
        # 实际选择可能更长，靠下面的 6× 余量兜底——掐表时 horizon_hit 如实置位。
        total_work = sum(min(t for _m, t in op) for job in inst.jobs for op in job)
        horizon = float(total_work * 6 + 500)   # 同 run()：门控掐表护栏（原 3× 截断 9/10 假死）
        # 停表条件同 run()：`done | timeout(horizon)`（末件到站即停，护栏只兜真死锁）。
        # ⚠️ 两处调用点的停表形状**必须一致**——只改一处 = 另一条路径的计数继续按护栏虚增。
        env.run(until=done | env.timeout(horizon) | stop_ev)
        stats["horizon_hit"] = len(completes) < inst.n_jobs   # 同 run()（原：run_gated 漏设旗标）
        due = self._due(plans)
        makespan = (max(completes.values()) if completes else env.now)
        energy = self._energy_report(stats, makespan)
        return {"makespan": makespan,
                "completes": dict(completes),          # 每作业完工时刻（同 run()）
                "energy": energy["total_kwh"],         # M2 引证模型（同 run()）
                "energy_breakdown": energy,
                "setup_minutes_total": float(sum(stats["setup_min"])),
                                "pm_events": stats["pm_events"],
                # ⑫ 拆键（T3）：forced = 被阈值强制触发（T3 的激活量）、chosen = 策略主动选。
                # 规则配置 forced == pm_events、chosen == 0（规则不会主动提前保养）。
                "pm_events_forced": stats["pm_events_forced"],
                "pm_events_chosen": stats["pm_events_chosen"],
                # 物流侧约束的事件计数（⑨）——run_gated 此前不带，⑨ 的效果读数要用。
                "trips": stats["trips"],
                # ② 有限缓冲的激活量（**两处返回字典都要列** —— `metrics` 是逐键列的，
                # 漏一个键就会 `m.get(k, 0)` 取到 0，把"有阻塞"读成"没阻塞"。实测踩过。）
                "buffer_block_min": stats.get("buffer_block_min", 0.0),
                "out_block_min": stats.get("out_block_min", 0.0),
                "agv_fail_events": stats["agv_fail_events"],
                # ⑨ failover：故障期间退回/转交的任务件数（默认关 ⟹ 恒 0）
                "agv_failover_tasks": stats["agv_failover_tasks"],
                # ⚠️ v2：⑩ 去后 batch_* 三个读数已删（2026-10-08）
                # ⚠️ **2026-10-10：⑪ 去后 `battery_min_kwh` 读数已删**（车队无电量）。
                # 车队规模 + 完成度守卫（`reward._incomplete_objectives`）需要的三个量。
                # ⚠️ `n_agv` / `fleet_size` 此前只在 `run()` 里有——两条路径的 metrics 键集
                #    必须一致（奖励守卫读 run_gated 的返回）。
                "n_agv": self.cfg.n_agv, "fleet_size": self.layout.n_agv,
                # ⚠️ 与 run() 同键同义：`horizon` = 护栏上界、`stop_time` = 实际停表时刻
                # （停表修复：跑完时停在 makespan，不再推进到护栏）。
                "horizon": float(horizon),
                "stop_time": float(env.now),
                "n_jobs": self.inst.n_jobs, "n_machines": self.inst.n_machines,
                "fail_events": stats["fail_events"],
                "tardy": self._tardy(plans, completes, due)[0],
                "tardy_twt": self._tardy(plans, completes, due)[1],
                "moves": stats["moves"], "deliveries": stats["deliveries"],
                # 工序排序入口的读数（spec §3.6）：机会率 = `seq_gets_ge2 / dbg["in_q_gets"]`；
                # `seq_reorders` = 真选了非 FIFO 队首的次数（"决策真的发生了"的硬证据）。
                # 两条路径的 keyset 必须一致，故此处与本入口同键；`run_gated` 没有排序回调
                # ⟹ 它的 `seq_reorders` 恒 0，而 `seq_gets_ge2` 照计（机台侧无条件记账）。
                "seq_gets_ge2": stats.get("seq_gets_ge2", 0),
                "seq_reorders": stats.get("seq_reorders", 0),
                "horizon_hit": stats.get("horizon_hit", False),
                "ops_done": stats.get("ops_done", 0), "jobs_done": len(completes),
                # 步 3：PA 派工次数（**"派工器真的被调用了"的硬证据**——步 2 曾实测为 0，
                # 原因是 `plans` 被离线路径预填满、池恒空。见 progress-log §69）。
                "pa_assigns": stats.get("pa_assigns", 0),
                # 扰动触发的决策点数（原文决策点定义的**另一半**，2026-10-08 补齐）
                "dist_events": stats.get("dist_events", 0),
                # ③ 机障发出的扰动哨兵数（顶层触发之一）+ 顶层实际决策次数
                "mf_dist_events": stats.get("mf_dist_events", 0),
                "ma_decisions": stats.get("ma_decisions", 0),
                # ⑨ 的三个故障场景分布 + 中断记账（`at` 是 TC 的一项，见式(48)）
                "agv_fail_sc1": stats.get("agv_fail_sc1", 0),
                "agv_fail_sc2": stats.get("agv_fail_sc2", 0),
                "agv_fail_sc3": stats.get("agv_fail_sc3", 0),
                "agv_sc3_delivered": stats.get("agv_sc3_delivered", 0),
                "agv_at_min": float(stats.get("agv_at_min", 0.0)),
                "agv_interrupted_min": float(stats.get("agv_interrupted_min", 0.0)),
                "task_flow": stats.get("task_flow", []),
                "travel_time_total": float(stats["travel_time"]),
                # 物流成本（**借 AEI2025 式(48) 的成本模型**，只为让读数可与其表并列）：
                #   `TC = c·Σ y_v + uc·(Σ(nct−nst+lct−lst) + at)`
                # - `Σ y_v = n_agv`：本仓**车队固定、全部常开**（⑩ 异构车队删后无启用决策）；
                # - `Σ(空载段+负载段) = travel_time_total`（逐车累计行驶工时的车队合计）；
                # - `at`（故障中断时间）**恒 0**：⑨ 段间检出、不打断正在进行的行驶。
                # ⚠️ **这不是我们的目标函数**（我们的是 makespan + energy + TWT），
                # 只是**报告桥**——不加它，任何"比较 TC"的说法都不成立。见 §77.4 第 2 行。
                "logistics_cost": (
                    AEI2025_AGV_ACTIVATION_COST * self.cfg.n_agv
                    + AEI2025_UNIT_TRANSPORT_COST * (float(stats["travel_time"])
                                                     + float(stats.get("agv_at_min", 0.0)))),
                # ⚠️ **v2：① 去后 `zone_wait` / `n_zones` 已删**（2026-10-08）
                # 口径随结果自报（同 run()）
                "transport": self.inst.transport,
                "unmapped_legs": stats.get("unmapped_legs", 0),
                "unmapped_min": float(stats.get("unmapped_min", 0.0))}

    def _pick_machine(self, job: int, oi: int, policy_s, online_s: bool,
                      plans) -> tuple[int, float]:
        """**派工点**（P2 Task 5，spec §5.3.2）：选机台 + 取该候选上的加工时长。

        两个派工点（投放点的首工序、transporter 的下一工序）**共用此一处**，故"在线/离线"
        只有这一个分支——不存在"改了一处漏了另一处"的形状。

        在线（`online_s=True`）：把**此刻的活状态**快照交给 `policy_s(snap, job, oi, cand)`；
        `plans[job][oi]` 随决策写入（决策前不得读它）。离线（`online_s=False`）：读预计算的
        `plans`——不调策略、不取快照，**逐位复现旧行为**。

        ⚠️ `policy_s` 返回非候选机台 → **显式报错**（静默回退会掩盖策略/候选集不一致，
        让整条链的 logp 与动作错位而无人察觉）。
        """
        alts = self.inst.jobs[job][oi]
        cand = [m for m, _t in alts]
        if online_s:
            snap = self.snapshot()                        # 活状态（只读，见 snapshot 的契约）
            choice = int(policy_s(snap, job, oi, cand))
            if choice not in cand:
                raise ValueError(f"policy_s 选了非候选机台 {choice}；"
                                 f"job={job} oi={oi} 候选={cand}")
            plans[job][oi] = choice                       # 计划表随决策写入（决策前不得读）
        else:
            choice = plans[job][oi]
        t = next(t for m, t in alts if m == choice)
        return choice, float(t)

    # ⚠️ **v2：① 与路线头去后 `_make_route_fn` 已删**（2026-10-08）

    def _make_pm_fn(self, policy_m):
        """构造 **⑫ 维护决策（M）** 的回调 `pm(mach) -> 动作码`（一局一个）。

        契约与 S/L/R 对称：`env/` 只给**原始快照**（此刻的活状态）与机台号，**不构造特征**
        （层次纪律：`env/` 不得依赖 `nn/`）——M 头的决策特征与逐候选特征由 `algo/` 侧从
        快照自造（见 `group_rel.pm_feat` / `_pm_cand_feat`）。

        ⚠️ 逾期（`pm_clock >= pm_interval`）**不进本函数**：强制保养在
        `MachineSim._pm_after_op` 里直接执行、不产生决策——规则是硬底线。
        ⚠️ 非候选动作**显式报错**（同 `policy_s` / `policy_l` / `policy_r`）：静默回退会
        掩盖策略/候选集不一致，让整条链的 logp 与动作错位而无人察觉。
        """
        def pm(mach: int) -> int:
            snap = self.snapshot()                  # 只读（见 `snapshot()` 的契约）
            choice = int(policy_m(snap, int(mach), PM_CANDS))
            if choice not in PM_CANDS:
                raise ValueError(f"policy_m 选了非候选动作 {choice}；候选={PM_CANDS}"
                                 f"（机台 {mach}）")
            return choice

        return pm

    # ⚠️ **2026-10-10：⑪ 去后 `_make_charge_fn` 已删**（车队无电量、无充电桩）。
    # 原实现：`charge(aid, cands)` 回调，把当时的快照 + 本车号 + 候选动作码交给策略，
    # 并对非候选动作显式报错。

    # ⚠️ **v2：⑩ 去后 `_make_batch_fn` 已删**（2026-10-08）—— 无拼批头

    @staticmethod
    def _release(env, store, item):
        yield store.put(item)

    def _pa_dispatch(self, plans, policy_pa, stats):
        """**决策点（PA）** —— 在「未完工工件的最早**未分配**工序」池里挑一件，**并给它定机台**。
        **每次调用恰派一件**（原文口径），返回 `(工件, 工序, 机台)`；池空 ⟹ 返回 `None`。

        ══ 池的口径：照 AEI2025 原文逐字（2026-10-08 第二稿）══

        > *"Let **OPi(t)** denotes the operation number of job i that **have been assigned** at
        > decision point t"* / *"`CJ(t) = {i | OPi(t) < Ji + 1}`"*
        > *"f24 = mean_{i∈CJ(t)}(APT_{i, **OPi(t)+1**})"*

        **池 = 每个"还有未分配工序"的作业，取其最低未分配工序下标** —— `oi` **不是**
        `done_ops`。⟹ **一个工件被派过之后仍在池里**（它的下一道未分配工序又成为候选），
        池恒 ≈ 未完工工件数。

        ⚠️ **第一稿错在这里**：用 `done_ops` 当 `oi` ⟹ 在制/在途工件的预留早已写好、被滤掉
        ⟹ 实测池在 **82%** 的决策上只剩 **1** 个候选（§73.1），四条工件规则退化成同一个动作。
        **错的不是事实卡，是我没读原文。**

        ══ 为什么"恰派一件"而不是"排空" ══

        新口径下池**永不为空**（派掉一件，该作业的下一道未分配工序立刻补位）⟹
        排空 = **一次调用就把整张调度表派完**。原文的决策点是"一道工序**被分配**的时刻"，
        **一件一次**。⟹ 也因此**没有"批内"**——同一批的后续派工看得到前面预留这件事不再存在
        （旧口径下池永不为空、排空即等于一次派完整张表）。

        ⚠️ **没被派到的工件会等**：它的件停在源机台的输出缓冲里，直到某个决策点派到它。
        **这正是原文的语义**（PA 挑哪件，哪件才推进）。

        ══ 契约 ══

        `policy_pa(view, pool, tie_rng) -> (池下标, 机台号)`——`view` 是
        `env/rule_views.PAView`（纯数据；`view.snap` 留给步 4 的编码器）；
        `tie_rng` 是**环境的平局随机源**（独立子流 `[seed_chain, PA_TIE_SEED]`，每局重播，
        见 `PA_TIE_SEED` 的注释——**不得改回共享 `rng`**）。

        `None` = 不启用 PA（离线路径：`plans` 已预填 ⟹ 本函数不被调用）。

        ══ 账本（`stats["pa_mct"]` / `stats["pa_assigned_min"]`）══

        按 AEI2025 的 `MCT_k` / `Σλ·pt` 推进，**跨决策点存活**：
        `mct[k] ← max(mct[k], 物理清空时刻, now) + pt`、`assigned_min[k] += pt`。
        ⚠️ 原文的 `MCT_k` 是**纯计划**；我们取与物理清空时刻的 `max`（实况只能把它推后）。
        """
        led_mct = self._live_stats.setdefault("pa_mct", [0.0] * self.inst.n_machines)
        led_assigned = self._live_stats.setdefault("pa_assigned_min",
                                                   [0.0] * self.inst.n_machines)
        if policy_pa is None:
            raise ValueError("_pa_dispatch 需要 policy_pa；离线路径不应调用它")
        view = build_pa_view(self.snapshot(), self.inst, plans, led_mct, led_assigned)
        # 池 = 还有未分配工序的作业（`view.jobs[j].oi` 已是"最低未分配下标"，见 `PAJob.oi`）
        # ⚠️ **`and v.alts`（2026-10-10，清单第 2 条）**：② 开启时，候选机台**全满**的作业
        # 也**派不出去**（`build_pa_view` 已把满机台从 `alts` 滤掉 ⟹ `alts` 空）。
        # **规则定义没变，变的是候选集。**
        #
        # ⚠️ **池因此可能第一次为空**——② 关闭时它恒非空（每个未完工作业都有非空 `alts`）。
        # 空池 ⟹ 返回 `None` ⟹ 调用方**本轮不派工、等下一个事件**（`des.run` 主循环的
        # `if got is not None`）。**这不是死锁**：机台完工事件会腾出缓冲位、池随即非空。
        # 真出现"所有机台都在等"的环（需要 `out_cap` 也 binding），由 `horizon` 如实截断
        # ——与 `_recover_from_depletion` 同一条纪律：**不静默放行，也不死循环**。
        pool = tuple((j, v.oi) for j, v in enumerate(view.jobs)
                     if v.oi < len(self.inst.jobs[j]) and v.alts)
        if not pool:
            return None
        pi, m = policy_pa(view, pool, self._pa_tie_rng)
        pi = int(pi)
        if not (0 <= pi < len(pool)):
            raise ValueError(f"policy_pa 选了不存在的池下标 {pi}；池={pool}")
        cand = [mm for mm, _t in view.jobs[pool[pi][0]].alts]
        if int(m) not in cand:
            raise ValueError(
                f"policy_pa 给作业 {pool[pi][0]} 的第 {pool[pi][1]} 道工序选了非候选"
                f"机台 {m}；候选={cand}")
        m = int(m)
        j, oi = pool[pi]
        ready_at = float(view.jobs[j].ready_at)      # 该工序的**可运输时刻**（给 LA 的 f33）
        plans[j][oi] = m
        proc = next(t for mm, t in view.jobs[j].alts if mm == m)
        # 账本推进（AEI2025 的 MCT_k / Σλ·pt）
        led_mct[m] = advance_avail(view.machines[m].avail_at, view.now, float(proc))
        led_assigned[m] += float(proc)
        stats["pa_assigns"] = stats.get("pa_assigns", 0) + 1
        return j, oi, m, ready_at

    def _transporter(self, env, events_q, tasks_in, plans, entities, stats, lu_idx: int,
                     bound: bool = False, agv_phi: list[int] | None = None,
                     policy_l=None, policy_s=None, online_s: bool = False,
                     policy_pa=None, policy_ma=None, dt_events: int = 0,
                     stop_after_ma: int = 0, stop_event=None):
        """机台完成事件 → 生成搬运任务；`task_i` = 本函数生成序。

        **三类事件**（端点号：机台 `0..m-1`、装卸站 = `lu_idx` = `m`）：

        - **投放**（`frm_idx == lu_idx`，`oi = -1`）：装卸站 → 首工序机台（**负载**段）；
        - **工序流转**（`frm_idx < m` 且非末工序）：本机台 → 下一工序机台（负载段）；
        - **回站**（`is_last`）：末工序机台 → 装卸站（**负载**段）。

        `task_flow` 的 `oi` 口径 = "这条搬运服务于第几道工序"（投放 = 0、流转 = 目标工序号、
        回站 = 末工序号 + 1）——三条路同一条判据，便于外部按同一套规则复算行程。

        **这里的取事件处即"下一工序的派工点"**（P2 Task 5）：目标机台由 `_pick_machine` 决定
        （在线 = 此刻调 `policy_s` 看活状态；离线 = 查 `plans`）。回站任务的终点是装卸站，
        **不经** `_pick_machine`（它不是候选机台，也不该进 S 层的选择空间）。

        输出缓冲满=阻塞源：out_q.put 在机台侧阻塞；此处 get 保证消费（阻塞语义=M1.1 消磨）。
        L 层（`policy_l` 非空）：`policy_l(snap, job, frm, to, oi+1, cand_v) -> 车号`——`snap` = 此刻
        的活状态，任务身份 = (作业号, 起送端点, 目标端点, 目标工序序号)，候选 = 全车队；此处
        **不构造任何特征**（F1，P2 Task 5；作业号由 P2 Task 7 补入——L 头的任务特征含"该机
        是否需换型"，需要作业号，且用 (frm, to, oi) 反查作业多义）。
        L 层（bound，`policy_l` 为空）：任务按 agv_phi[task_i] 绑定 AGV（task_i = 本函数生成序）；
        超出补轮询。

        ══ v2：**批量派工**（2026-10-08，设计 §14.3；步 3 加规则集）══

        `policy_pa` 非空时，机台不再由 `_pick_machine` 当场选：**PA 提前把 `plans` 填满**
        （`_pa_dispatch`），transporter 只**读预留**（`online_s` 必须为 False，见 `run` 的校验）。
        分工 = AEI2025 的 PA/LA：**PA 定 (工件, 机台)，LA 定车**。

        ⚠️ 回站任务（`is_last`）的终点恒为装卸站 ⟹ **不参与机台选择**，也不在 PA 的池里
        （池只含"还没定机台的待加工工序"，回站不是工序）。
        """
        task_i = 0
        # ═══ 统一事件流（2026-10-08 第二稿）═══
        # 三条驱动方式（**同一套流**，只换"谁定机台"）：
        #   · `policy_pa` 非空 —— 决策点由 PA 派一件（新路径，见 `_pa_dispatch`）；
        #   · `online_s=True` —— 决策点由 S 层当场定机台（旧消融路径）；
        #   · 都不给 —— `plans` 离线预填，本函数只读它（**逐位等于旧行为**）。
        # `pending[j] = (oi, src)`：件**已产出**、等目的地；一旦 `plans[j][oi]` 有值就发车。
        # `booking[(j, oi)]` = 该程**已订的车**（LA 在派工那一刻定的，见 §74.5）。
        pending: dict[int, tuple[int, int]] = {}
        booking: dict[tuple[int, int], int] = {}
        last_ma_ops = -1          # 上一次**定时器**触发顶层时的 `ops_done`（防同刻重复触发）
        while True:
            frm_idx, job, oi, op, is_last = yield events_q.get()
            stats["trans_evt"] = stats.get("trans_evt", 0) + 1
            n_ops = len(self.inst.jobs[job]) if job >= 0 else 0
            if frm_idx < 0:
                # ⚠️ **扰动事件**：`DISTURBANCE`（AGV 故障 ⑨）或 `DISTURBANCE_MACHINE`（机障 ③）。
                # 它们**不产生件**、也不改 `pending`，只是让下层/顶层**在扰动那一刻再决策一次**。
                # ⑨：原文的决策点定义含它（见 §77.4 第 5 行）；**LA 侧对"趴窝车已订任务的重订"
                # 未实现**，登记 §77.5。③：设计 §10⑤ 把机障列为**顶层**的触发之一。
                stats["dist_events"] = stats.get("dist_events", 0) + 1
                if frm_idx == DISTURBANCE_MACHINE:
                    stats["mf_dist_events"] = stats.get("mf_dist_events", 0) + 1
            elif is_last:                                   # 末工序完工 → 回装卸站
                # ⚠️ 回站**不由 PA 决定**（终点恒为装卸站）⟹ 不走决策点，直接发。
                # 这是与原文的一处小差异：他们的 dummy 工序（`J_i+1`）也进 PA 的动作。
                yield entities[frm_idx].out_q.get()
                yield from self._emit_task(tasks_in, entities, stats, frm_idx, lu_idx,
                                           (job, n_ops, op, True), job, policy_l,
                                           bound, agv_phi, task_i)
                task_i += 1
            else:
                # 件**此刻已在 `out_q`**（机台是先 put 事件再 put 件，同一事件帧）⟹ 不阻塞。
                if frm_idx != lu_idx:
                    yield entities[frm_idx].out_q.get()
                pending[job] = (oi + 1, frm_idx)             # 件已到，等目的地
            # ── 顶层（MA）触发：**定时器 + 扰动**（设计 §10⑤）──
            # `policy_ma(snap, reason) -> g_bins`；`reason ∈ {"timer", "disturbance"}`。
            # ⚠️ 触发点放在**决策点之前**：这样本次决策用到的就是刚定下的 `g`。
            # `dt_events = 0` ⟹ 定时器关（只有扰动触发）。
            if policy_ma is not None:
                od = int(stats.get("ops_done", 0))
                # ⚠️ **第一次也触发**（`ma_decisions == 0`）：设计 §4.2 说"顶层**每 ΔT** 输出 g"
                # ⟹ 第一段（t=0 起）也得有个 `g` 管着，否则开局那 ΔT 个完工事件没有目标。
                first = int(stats.get("ma_decisions", 0)) == 0
                timer_due = (dt_events > 0 and od > 0 and od % dt_events == 0
                             and od != last_ma_ops)
                if frm_idx < 0 or timer_due or first:
                    if timer_due:
                        last_ma_ops = od
                    stats["ma_decisions"] = stats.get("ma_decisions", 0) + 1
                    policy_ma(self.snapshot(),
                              "initial" if first else
                              ("disturbance" if frm_idx < 0 else "timer"),
                              self._step_counters(stats))
            # ── **停表**：跑到第 `stop_after_ma` 个顶层点就停（§4.2 的"跑 ΔT"靠它）──
            # ⚠️ 停表**不保留状态**（SimPy 帧不可复制，§13 #1 属实）⟹ 分叉靠**重放前缀**：
            # 从 t=0 用**已记录的动作**重跑到这里、再让新目标接下去。停表只用来**读读数增量**。
            if stop_event is not None and stop_after_ma > 0 \
                    and int(stats.get("ma_decisions", 0)) >= stop_after_ma \
                    and not stop_event.triggered:
                stop_event.succeed()
                return
            if policy_pa is not None:
                # ── 决策点：**恰派一件**（原文："the decision point t is defined as the time
                #    when an operation is assigned"）──
                got = self._pa_dispatch(plans, policy_pa, stats)
                if got is not None and policy_l is not None:
                    # ⚠️ **LA 在派工那一刻定车**（原文的 PA→LA 同决策点顺序）：此刻算出的
                    # `ready_at` 是**未来**时刻 ⟹ `LATEST_IN_TIME` 的"不晚于它地最晚到达"有内容。
                    # 车**先订下**，件到了（下面的冲账）才进它的队列。
                    j3, oi3, m3, ready3 = got
                    src3 = lu_idx if oi3 == 0 else int(plans[j3][oi3 - 1])
                    cand3 = list(range(len(tasks_in)))
                    agv3 = int(policy_l(self._la_view(j3, oi3, src3, m3, ready3), cand3))
                    if agv3 not in cand3:
                        raise ValueError(f"policy_l 派了不存在的车 {agv3}；候选={cand3}")
                    booking[(j3, oi3)] = agv3
            elif online_s and job >= 0:
                # 旧消融路径：S 层当场定机台（`_pick_machine` 把结果写进 `plans`）
                self._pick_machine(job, oi + 1, policy_s, True, plans)
            # ── 冲账：所有"件已到 + 目的地已定"的作业，此刻发车 ──
            for j2 in list(pending):
                oi2, src2 = pending[j2]
                m2 = plans[j2][oi2]
                if m2 is None:
                    continue
                n2 = len(self.inst.jobs[j2])
                t2 = next(t for mm, t in self.inst.jobs[j2][oi2] if mm == m2)
                item2 = (j2, oi2, OpLite(t2), oi2 == n2 - 1)
                yield from self._emit_task(tasks_in, entities, stats, src2, m2, item2,
                                           j2, policy_l, bound, agv_phi, task_i,
                                           agv=booking.pop((j2, oi2), None))
                task_i += 1
                del pending[j2]

    def _emit_task(self, tasks_in, entities, stats, frm: int, to: int, item, job: int,
                   policy_l, bound: bool, agv_phi, task_i: int, agv: int | None = None):
        """建路径 + 派车（`policy_l` / 绑定 / FIFO 三条路）。旧路径与新路径**共用此一处**。

        `agv` 非空 = **已订的车**（新路径：LA 在**派工那一刻**定的，见 `_pa_dispatch` 的注释）
        ⟹ 直接进它的队列，不再问一次策略——同一程问两次会拿到两个答案。
        """
        from .corridors import shortest_node_path
        _path = shortest_node_path(self.g, entities[frm].pad.dock_node,
                                   entities[to].pad.dock_node)   # AGV 实际经过的节点序列
        task = (frm, to, item, _path)
        stats.setdefault("task_flow", []).append((job, item[1], frm, to))  # L 层流导出
        stats.setdefault("agv_load", []).append((tuple(stats["agv_del"]),
                                                 tuple(stats["agv_pos"])))       # 任务时点车状态
        if agv is not None:
            yield tasks_in[agv].put(task)
        elif policy_l is not None:
            # 无预订时的兜底（`online_s` / 离线路径 + `policy_l`）：此刻件**已在手**，
            # 故 `ready_at ≡ now`——`LATEST_IN_TIME` 在这条路上仍会塌陷成 `MINCOST`，
            # **这是预期的**（那条路本来就不是新流）。新流走上面的 `agv` 分支。
            view = self._la_view(job, item[1], frm, to, float(self.env.now))
            cand_v = list(range(len(tasks_in)))         # 候选 = 全车队（每车一队列，都可选）
            agv = int(policy_l(view, cand_v))
            if agv not in cand_v:
                # 同 policy_s：非候选**显式报错**——负索引会静默回绕到别的车
                raise ValueError(f"policy_l 派了不存在的车 {agv}；候选={cand_v}")
            yield tasks_in[agv].put(task)
        elif bound:
            n_agv = len(tasks_in)
            agv = agv_phi[task_i] if (agv_phi and task_i < len(agv_phi)) else (task_i % n_agv)
            yield tasks_in[agv].put(task)
        else:
            agv = -1                        # FIFO 路径：此刻还没派车（车在取货时才定）
            yield tasks_in.put(task)
        # 快照跟踪（P2 Task 1）：工件已离机台；绑定路径此刻即知派了哪台车
        self._job_loc[job] = -1
        self._job_agv[job] = agv
        if agv >= 0:
            # 步 3 第二稿：逐车**已派任务账本**（预计空闲时刻）。估时用**负载段**时长——
            # 空载段取决于车的当前停位，此处不反推（登记 §72.1）。
            est = self.leg_min(self._node_of(frm), self._node_of(to))
            act = self._live_stats.setdefault("agv_act", [0.0] * self.cfg.n_agv)
            act[agv] = advance_avail(act[agv], float(self.env.now), est)
        stats["tasks_put"] = stats.get("tasks_put", 0) + 1
