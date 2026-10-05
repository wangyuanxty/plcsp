# -*- coding: utf-8 -*-
"""T3 拉格朗日乘子：**冻结的预算表** + 对偶上升的状态机（设计 `docs/t3-design.md`）。

机制（一句话）：奖励加罚项，λ 对偶上升，无 critic::

    r' = r − Σᵢ λᵢ · âᵢ
    λᵢ ← clip(λᵢ + η · (âᵢ − bᵢ), 0, λ_max)          # 标准对偶上升（带防发散上界）

口径（设计的 §1–§4，**本文与实现同源**）
----------------------------------------
- **激活量只算「被迫发作」**：罚"策略主动做的动作"= 惩罚它刚拿到的动作，自相矛盾。
  故 ⑪ 用 `agv_dry_events`（耗尽停机，**不是** `charge_events`）、⑫ 用 `pm_events_forced`
  （被阈值强制触发，**不是** `pm_events` 总数）。
- **适用范围 = 可控的六条**：① 拥堵、② 有限缓冲、③ 机器故障、⑤ 换型、⑪ 充电、⑫ 维护。
  ④ 返工、⑨ AGV 故障**不纳入**（无记忆、不可控——罚它们只是加噪）。
  ⚠️ ③ 纳入的前提是**策略能通过 ⑫ 提前保养影响役龄**（§44 的役龄故障率），故 ③ 与 ⑫
  共用一条守卫（`check_influenceable`）：没有维护头就没有可作用的手段。
- **归一化**：`âᵢ = aᵢ / aᵢ^ref`，`aᵢ^ref` = **参考调度**上同一口径的激活量，**冻进表**。
  于是 `bᵢ` 是 [0, 1] 的目标激活率（`BUDGET_RATIO` = 参考水平的比例），
  λ 的尺度跨约束可比。
- **逐实例标定、冻结成表**（同 `due_dates.TF_RDD`）：未标定的实例**显式报错**，
  不静默取默认值。

⚠️ **标定环境（表是环境条件的，必须写清）**
--------------------------------------------
`aᵢ^ref` 在**同一个参考运行**上一次性测出，环境 = 本模块的 `calibration_cfg` +
`calibration_layout`：

- **默认电池放不空**（2–4 kWh，实测 mk01 全程每车耗 0.1–0.27 kWh）⟹ ⑪ 的
  `agv_dry_events ≡ 0`，没有参考水平可归一化；
- **默认 `pm_interval=120` 在 mk01/mk02 从不逾期**（每机 ~25.5 主轴分钟）⟹ ⑫ 的
  `pm_events_forced ≡ 0`。

故标定环境显式取**机制真的活的配置**：`pm_interval=T3_PM_INTERVAL`（短间隔）+
`battery_low=T3_BATTERY_LOW`（0，规则配置不到 0 不补电）+ `T3_BATTERY_KWH`（小电池）。
六条约束在该环境下**全部非零**（`test_t3_budget.py` 逐实例核对表 == 脚本重算）。
⚠️ 换 cfg / 换布局（电池是**布局**的属性）⟹ `aᵢ^ref` 变，须重跑 `plcsp/m17_t3_calib.py`。
本表的保证与 ⑧ 交期的 (τ,R) 同型：**"激活量落带"是 cfg 条件的**，表本身是冻结产物。

⚠️ **③ 的口径差（如实声明，不是静默错位）**：标定环境**没有开役龄模型**
（`machine_age_failure=False`）——实测开着它 + 短间隔保养时，规则配置自己就把役龄压在低位，
mk01 的参考故障数从 2 掉到 **0**，既无法归一化、也会把预算标到 0 附近（正是设计 §4.2 的
上界退化形态）。故 ③ 的 `aᵢ^ref` 是"**无记忆故障率**下的参考次数"，而 T3 训练配置**必须开
役龄**（③ 的可控性守卫）——两者的差在语义上等价于"把 ③ 的预算锚在**不保养**的行为水平上"，
b = 参考的一半意味着"允许的被迫故障数"（mk01 = 1 次）；策略保养得好 ⟹ â 低于 b ⟹ λ 落 0
（不咬人），保养得差 ⟹ 故障上升 ⟹ λ 咬人。这正是 ③ 想要的语义。

书目
----
对偶上升 / 拉格朗日乘子法是约束强化学习里"代价约束"的标准做法（CPO / PPO-Lagrangian
一族）。⚠️ **本模块不引具体书目**——用户已裁定"PPO-Lagrangian 是历史叫法、拉格朗日与值函数
正交"，本仓是 GRPO-Lagrangian（不为此引入 critic）；正式引用待 §12 的引证流程核过再写。
"""
from __future__ import annotations

import numpy as np

from .constraints import ConstraintConfig
from .corridors import build_corridor_graph, dock_distance_matrix
from .des import SimConfig, SimWorld
from .due_dates import _instance_name          # 同一套"标定表键"口径（文件名主干，小写）
from .instances import Instance
from .layout import AgvSpec, Layout, sample_layout

# 行程时间口径标签（与 `due_dates.TABLE_BY_CALIBER` 同一套词）。
GEOMETRY, MATRIX = "geometry", "matrix"

# ══ 六条受控约束（顺序 = 表列的**唯一真相**；表里的元组按此序） ══
T3_CONSTRAINTS: tuple[str, ...] = (
    "congestion",        # ① 拥堵      → zone_wait["total"]   （被迫等待时长 [min]）
    "finite_buffer",     # ② 有限缓冲  → buffer_block_min    （被迫等待时长 [min]）
    "machine_failure",   # ③ 机器故障  → fail_events         （次数）
    "setup_time",        # ⑤ 换型      → setup_minutes_total （分钟）
    "charging",          # ⑪ 充电      → agv_dry_events      （耗尽停机次数）
    "maintenance",       # ⑫ 维护      → pm_events_forced    （被阈值强制触发的次数）
)
# 中文标签（打印用；编号沿用 spec §3.3 的原始编号，不重排——见 constraints.py 的说明）。
T3_LABELS: dict[str, str] = {
    "congestion": "① 拥堵", "finite_buffer": "② 有限缓冲", "machine_failure": "③ 机器故障",
    "setup_time": "⑤ 换型", "charging": "⑪ 充电", "maintenance": "⑫ 维护",
}
# 每条约束的**激活量只算被迫发作**——上表右列即口径。⚠️ ⑪ 不用 `charge_events`
# （那是策略主动充的次数）、⑫ 不用 `pm_events`（含策略主动保养）——理由见模块 docstring。
ACTIVATION_SOURCES: dict[str, str] = {
    "congestion": 'metrics["zone_wait"]["total"]',
    "finite_buffer": 'metrics["buffer_block_min"]',
    "machine_failure": 'metrics["fail_events"]',
    "setup_time": 'metrics["setup_minutes_total"]',
    "charging": 'metrics["agv_dry_events"]',
    "maintenance": 'metrics["pm_events_forced"]',
}

# ══ 对偶上升的默认参数 ══
# `bᵢ` = `BUDGET_RATIO × aᵢ^ref`（= 归一化后的目标水平，设计 §4.1 第 2 步的"目标激活率"）。
# 0.5 = 设计 §4.1 的例子值（"如参考调度的 50%"）。**它同时是双侧判据的旋钮**：
# 下界（防"约束不再咬人"）要求策略改进后 â 仍不得为 0，上界（防"咬死动作"）要求动作分布
# 不退化——标定脚本打印网格供复核，逐约束的目标率**留作开放线索**（设计 §7）。
BUDGET_RATIO = 0.5
# `η`（对偶上升步长）：λ 每步变动 `η·(â−b)`，而 `â−b` 是 O(1) 的量（â = 参考水平的比例）。
# 取 0.1 ⟹ 典型情形（â 是 b 的两倍）每步 λ +0.05，几十步内到 O(1)——与奖励同量级。
# ⚠️ **待扫**（设计 §7）：太大 ⟹ λ 震荡；太小 ⟹ 几百步还到不了预算。默认值给一个可用的起点。
T3_ETA = 0.1
# λ 上界（防发散）。⚠️ **不该被触发**：`r' = r − Σλᵢâᵢ` 里 λ∈[0,10]、â ~ O(1) 时的罚项已达
# 奖励本身的量级；真被顶到上界说明标定错了或罚项被组内 z 化吃掉太多（设计 §3.3/§3.4）。
T3_LAMBDA_MAX = 10.0

# ══ 标定环境（`aᵢ^ref` 的测量条件；见模块 docstring） ══
T3_PM_INTERVAL = 10.0     # ⑫：默认 120 在 mk01/mk02 从不逾期 ⟹ 短间隔验证档
T3_BATTERY_KWH = 0.10     # ⑪：默认 2–4 kWh 放不空 ⟹ 小电池验证档（同 test_charge_head 的口径）
T3_BATTERY_LOW = 0.0      # ⑪：规则配置的补电阈值必须为 0，否则车到不了 0（同 §33.4）


# ══ 冻结表：逐实例、逐口径的参考激活量 `aᵢ^ref`（表列序 = `T3_CONSTRAINTS`） ══
# 由 `plcsp/m17_t3_calib.py` 在标定环境下的**参考调度**（每工序取最短候选 + AGV 轮询）上
# 一次跑测出，`seed_layout=0`、`seed_chain=0`；**不是手抄的魔数**——
# `plcsp/tests/test_t3_budget.py` 逐实例与脚本重算对拍（同 ⑧ 交期的纪律）。
# 新增实例必须重跑标定脚本，否则 `t3_ref_vector` 显式报错。
#
# 标定命令：
#   PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m17_t3_calib --transport geometry
#   PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m17_t3_calib --transport matrix
#
# 实测（2026-10-05，几何口径、10/10 实例、六条约束全部非零）：
#   mk01 (1.7505, 309.0, 2, 94, 2, 11)      mk02 (1.7768, 170.0, 1, 102, 3, 12)
#   mk03 (3.4195, 1382.0, 3, 280, 10, 57)   mk04 (3.2706, 784.0, 1, 166, 6, 25)
#   mk05 (2.4722, 1319.0, 1, 200, 3, 55)    mk06 (2.3785, 722.0, 2, 282, 9, 30)
#   mk07 (1.9046, 1075.0, 2, 190, 4, 45)    mk08 (2.5991, 2455.0, 4, 426, 19, 171)
#   mk09 (4.4207, 2503.0, 4, 456, 15, 164)  mk10 (4.0717, 1332.0, 7, 462, 15, 138)
# ⚠️ 量纲差异极大（① 是分钟级的 1.75–4.4、② 是 170–2503 分钟、③ 是 1–7 次）——
# 这正是必须归一化的理由（λ 的尺度跨约束不可比）；归一化后六条一律是"占参考水平的比例"。
A_REF: dict[str, tuple[float, ...]] = {           # geometry 口径（原始 MK）
    "mk01": (1.7505, 309.0000, 2.0000, 94.0000, 2.0000, 11.0000),
    "mk02": (1.7768, 170.0000, 1.0000, 102.0000, 3.0000, 12.0000),
    "mk03": (3.4195, 1382.0000, 3.0000, 280.0000, 10.0000, 57.0000),
    "mk04": (3.2706, 784.0000, 1.0000, 166.0000, 6.0000, 25.0000),
    "mk05": (2.4722, 1319.0000, 1.0000, 200.0000, 3.0000, 55.0000),
    "mk06": (2.3785, 722.0000, 2.0000, 282.0000, 9.0000, 30.0000),
    "mk07": (1.9046, 1075.0000, 2.0000, 190.0000, 4.0000, 45.0000),
    "mk08": (2.5991, 2455.0000, 4.0000, 426.0000, 19.0000, 171.0000),
    "mk09": (4.4207, 2503.0000, 4.0000, 456.0000, 15.0000, 164.0000),
    "mk10": (4.0717, 1332.0000, 7.0000, 462.0000, 15.0000, 138.0000),
}
A_REF_MATRIX: dict[str, tuple[float, ...]] = {    # matrix 口径（MKT）——**只收六条全活的实例**
    # ⚠️ 实测（2026-10-05，`--transport matrix`，标定环境 = n_agv=m + 几何降级 + 短间隔 + 小电池）：
    # **10 个实例里只有 4 个的六条参考激活量全 > 0**。另 6 个的 ② 有限缓冲为 **0**
    # （矩阵口径的行程是分钟级 ⟹ AGV 慢，机台把输入缓冲排空了才等到下一趟投递）；
    # mk02/mk04 的 ③ 机器故障也同时为 0。**参考水平为 0 ⟹ 无法归一化**（`â = a/0` 无定义），
    # 故这些实例**不进表**——查表时**显式报错**，绝不静默当作"该约束恒不激活"。
    # 这是"表是环境条件的"的又一处体现（同 ⑧ 交期的矩阵表）；要覆盖这些实例须另选标定环境
    # （更紧的缓冲 / 更长的 episode）——登记为开放线索（`docs/progress-log.md` §49）。
    "mk01": (1688.9377, 1.0000, 1.0000, 92.0000, 77.0000, 12.0000),
    "mk05": (1572.2857, 27.0000, 4.0000, 198.0000, 78.0000, 55.0000),
    "mk08": (11167.3504, 7.0000, 6.0000, 418.0000, 347.0000, 169.0000),
    "mk10": (19350.5470, 2.0000, 6.0000, 434.0000, 435.0000, 139.0000),
}

TABLE_BY_CALIBER: dict[str, dict[str, tuple[float, ...]]] = {
    GEOMETRY: A_REF,
    MATRIX: A_REF_MATRIX,
}


def caliber_of(inst: Instance) -> str:
    """实例的行程时间口径标签（缺省 `geometry`）——决定查哪张标定表。"""
    return getattr(inst, "transport", GEOMETRY)


def instance_key(inst: Instance) -> str | None:
    """标定表键 = 实例文件名主干（`Mk01.fjs` → `mk01`）；取不出返回 None。

    与 `due_dates._instance_name` 同一口径（同一函数），故两张标定表的键一致。
    """
    return _instance_name(inst)


def calibration_cfg(inst: Instance, caliber: str | None = None) -> SimConfig:
    """标定环境的 `SimConfig`（表口径的**唯一真相**，脚本与测试共用）。

    - 几何口径：`SimConfig(pm_interval=10, battery_low=0)`（其余默认）；
    - 矩阵口径：再加 `n_agv=m`（HGS/HA-DQN 口径）与 `transport_unmapped="geometry"`
      （⑪ 充电桩没有矩阵项 ⟹ 声明式几何降级，同 `m16_due_calib._cli_cfg`）。
    """
    cal = caliber or caliber_of(inst)
    if cal == GEOMETRY:
        return SimConfig(pm_interval=T3_PM_INTERVAL, battery_low=T3_BATTERY_LOW)
    if cal == MATRIX:
        return SimConfig(n_agv=inst.n_machines, transport_unmapped=GEOMETRY,
                         pm_interval=T3_PM_INTERVAL, battery_low=T3_BATTERY_LOW)
    raise ValueError(
        f"未知行程时间口径 {cal!r}——无法确定 T3 标定环境的 cfg（已知：{sorted(TABLE_BY_CALIBER)}）")


def calibration_layout(inst: Instance, cfg: SimConfig, seed_layout: int = 0) -> Layout:
    """标定环境的布局：默认采样布局 + **车队电池全换小电池**（⑪ 激活量的前提）。

    ⚠️ 电池是**布局**（`AgvSpec`）的属性、不是 cfg 的字段——故标定环境必须同时钉住
    cfg（`calibration_cfg`）与布局（本函数），只钉一个 ⟹ `aᵢ^ref` 不可复现。
    """
    lay = sample_layout(inst.n_machines, seed=seed_layout, aisle_w=cfg.aisle_width,
                        n_agv=cfg.n_agv, max_agv_capacity=cfg.max_agv_capacity)
    for i, a in enumerate(lay.agvs):
        lay.agvs[i] = AgvSpec(id=a.id, speed_factor=a.speed_factor,
                              capacity=a.capacity, battery_kwh=T3_BATTERY_KWH)
    return lay


def activation_from_metrics(met: dict) -> dict[str, float]:
    """从一次 episode 的 metrics dict 取出六条约束的**原始激活量**（未归一化）。

    键名缺失即**显式报错**（不静默取 0——静默取 0 会让"读错指标名"变成"该约束从不激活"，
    两者在表里长得一模一样）。
    """
    out: dict[str, float] = {}
    for name in T3_CONSTRAINTS:
        try:
            if name == "congestion":
                v = float(met["zone_wait"]["total"])
            elif name == "finite_buffer":
                v = float(met["buffer_block_min"])
            elif name == "machine_failure":
                v = float(met["fail_events"])
            elif name == "setup_time":
                v = float(met["setup_minutes_total"])
            elif name == "charging":
                v = float(met["agv_dry_events"])
            else:                                   # maintenance
                v = float(met["pm_events_forced"])
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"metrics 里取不到 {T3_LABELS[name]} 的激活量（{ACTIVATION_SOURCES[name]}）："
                f"{e!r}——键名漂了就是「该约束从不激活」的假象，故不静默取 0。") from e
        out[name] = v
    return out


def reference_activation(inst: Instance, cfg: SimConfig | None = None,
                         caliber: str | None = None, seed_layout: int = 0,
                         seed_chain: int = 0) -> dict[str, float]:
    """**参考调度**上的六条激活量 `aᵢ^ref`（标定脚本与对拍测试的唯一入口）。

    参考调度 = `SimWorld.run()`（每工序取最短候选 + AGV 轮询），与 `des.reference_run` 同一
    口径；`cfg=None` ⟹ `calibration_cfg(inst)`、布局 ⟹ `calibration_layout(...)`（小电池）。
    ⚠️ **关 ⑧ 交期**跑（`with_off("due_dates")`，同 `m16_due_calib.reference_completes`）：
    交期是 metric、不进时序，关掉还解开"标定一个新实例要先有交期表"的鸡生蛋。
    """
    c = cfg or calibration_cfg(inst, caliber)
    lay = calibration_layout(inst, c, seed_layout)
    g = build_corridor_graph(lay)
    dm = dock_distance_matrix(g)
    cons = ConstraintConfig().with_off("due_dates")
    met = SimWorld(inst, lay, dm, c, graph=g, constraints=cons).run(seed_chain=seed_chain)
    if met.get("horizon_hit") or met.get("jobs_done") != inst.n_jobs:
        raise RuntimeError(
            f"T3 标定的参考运行没跑完（horizon_hit={met.get('horizon_hit')}、"
            f"jobs={met.get('jobs_done')}/{inst.n_jobs}）——掐表下的激活量不可作参考水平。")
    return activation_from_metrics(met)


def t3_ref_vector(inst: Instance, caliber: str | None = None) -> np.ndarray:
    """实例的 `aᵢ^ref` 向量（表列序 = `T3_CONSTRAINTS`）。**未标定实例显式报错。**

    逐元素守卫 `aᵢ^ref > 0`：为 0 时 `â = a / aᵢ^ref` 无定义（参考水平上该约束从不激活，
    就没有"占参考水平的比例"可言）——报错，不静默产生 inf / NaN。
    """
    cal = caliber or caliber_of(inst)
    try:
        table = TABLE_BY_CALIBER[cal]
    except KeyError:
        raise ValueError(
            f"未知行程时间口径 {cal!r}——无法确定用哪张 T3 预算表"
            f"（已知：{sorted(TABLE_BY_CALIBER)}）") from None
    key = instance_key(inst)
    row = table.get(key) if key else None
    if row is None:
        raise ValueError(
            f"实例 {key or inst.source or '<无 source>'} 在 **{cal}** 口径下未标定 T3 参考激活量 "
            f"——{cal} 表只覆盖 {sorted(table)}。请跑 "
            f"`PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m17_t3_calib "
            f"--transport {cal}` 并把输出贴回 plcsp/env/t3_budget.py。")
    a = np.asarray(row, dtype=np.float64)
    if a.shape != (len(T3_CONSTRAINTS),):
        raise ValueError(f"T3 表行长度 {a.shape} ≠ 约束数 {len(T3_CONSTRAINTS)}（{key}/{cal}）"
                         "——表列序 = T3_CONSTRAINTS，长度不符即表与实现漂了。")
    bad = [T3_LABELS[n] for n, v in zip(T3_CONSTRAINTS, a) if not v > 0.0]
    if bad:
        raise ValueError(
            f"实例 {key}/{cal} 的 T3 参考激活量在 {'、'.join(bad)} 上为 0（或负）——"
            "归一化 â = a / a^ref 无定义。请重跑标定脚本（标定环境见 t3_budget 模块 docstring），"
            "或把该约束移出 T3 约束集（`keep=` 参数）。")
    return a


def normalized_activation(met: dict, a_ref: np.ndarray, keep: tuple[str, ...]) -> np.ndarray:
    """一条链的归一化激活量 `âᵢ = aᵢ / aᵢ^ref`（只取 `keep` 列，序同上）。"""
    raw = activation_from_metrics(met)
    return np.asarray([raw[n] for n in keep], dtype=np.float64) / a_ref


def dual_ascent(lam: np.ndarray, ahat: np.ndarray, b: np.ndarray, eta: float,
                lam_max: float = T3_LAMBDA_MAX) -> np.ndarray:
    """一步标准对偶上升：`λ ← clip(λ + η·(â − b), 0, λ_max)`（纯函数，可单测）。

    性质（验收判据 3）：`â > b ⟹ λ 升`；`â < b ⟹ λ 降`；**下界 0 恒成立**（约束不罚负）。
    """
    lam = np.asarray(lam, dtype=np.float64)
    out = lam + float(eta) * (np.asarray(ahat, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return np.clip(out, 0.0, float(lam_max))


def check_influenceable(keep, pm_head: bool, charge_head: bool,
                        machine_age_failure: bool) -> None:
    """**可控性守卫**（单一真相；`joint_chain_step` 与 CLI 共用）——不可控的约束不纳入 T3。

    理由与 ④⑨ 不纳入同一条：罚"策略无法控制的事"只会加噪。三条的**作用手段**分别是：

    - **⑫ 维护**：提前保养 = M 头的动作 ⟹ 需要 `pm_head`（规则配置在阈值处自动发生，策略无动作）；
    - **③ 机器故障**：役龄故障率随 `pm_clock` 上升、提前保养把它归零（设计 §1.1）⟹ 需要
      `pm_head` **且** `cfg.machine_age_failure`（后者关时故障率是常数，保养不改变故障）；
    - **⑪ 充电**：早充 = C 头的动作 ⟹ 需要 `charge_head`（规则配置按低电阈值补电，策略无动作）。

    不在 `keep` 里的约束不受限。违规**显式报错**，不静默剔除（静默剔除会让"T3 到底罚了
    什么"与配置不符而无人察觉）。
    """
    if "maintenance" in keep and not pm_head:
        raise ValueError(
            "T3 含 ⑫ 维护，但 pm_head=False：规则配置的保养在阈值处自动发生、策略没有任何"
            "动作能改变它 ⟹ 罚 `pm_events_forced` 是罚策略无法控制的事（④⑨ 不纳入的"
            "同一条理由）。请开 pm_head，或把 maintenance 移出 T3 约束集。")
    if "machine_failure" in keep and not (pm_head and machine_age_failure):
        raise ValueError(
            "T3 含 ③ 机器故障，但（pm_head=False 或 machine_age_failure=False）："
            "③ 之所以可控，是因为役龄故障率随 `pm_clock` 上升、而策略能提前保养把它归零"
            "（设计 §1.1）；两者缺一 ⟹ 故障率与策略动作无关，罚它只是加噪。"
            "请同时开 pm_head 与 machine_age_failure，或把 machine_failure 移出 T3 约束集。")
    if "charging" in keep and not charge_head:
        raise ValueError(
            "T3 含 ⑪ 充电，但 charge_head=False：规则配置按低电阈值补电、策略没有'何时充'"
            "的动作 ⟹ 罚 `agv_dry_events` 基本不可控。请开 charge_head，"
            "或把 charging 移出 T3 约束集。")


# ══ 退化守卫（上界，设计 §4.2）的判据 ══
# 预算过紧 ⟹ T3 把动作推向极端（⑪ 永远充 / ⑫ 永远保养 / ① 永远绕路）。判据**盯动作分布**、
# 不盯 λ 的大小（设计 §1.1：③ 与 ⑫ 推同一个动作，单看 λ 会低估推动力）。
# 使用率落到 [LO, HI] 之外 = 退化成单一取值。0.05/0.95 是"~0 / ~1"的**可操作化**（暂定值，
# 设计只写"既非 ~0 也非 ~1"；真正的判据是"不是恒同一动作"，本阈值是它的宽容版本）。
T3_USAGE_LO, T3_USAGE_HI = 0.05, 0.95


def usage_is_degenerate(usage: float, lo: float = T3_USAGE_LO,
                        hi: float = T3_USAGE_HI) -> bool:
    """该动作使用率是否退化（落在 [lo, hi] 之外）——上界守卫的直接判据。"""
    return not (float(lo) <= float(usage) <= float(hi))


def action_usage(decisions, kind: str, is_active) -> float:
    """`kind` 类决策里取 `is_active(action)` 的比例（无该类决策 → 0.0）。

    ⑪：动作码 ≠ `CHARGE_CAND_SKIP` = 去了充电桩；⑫：动作码 = `PM_CAND_NOW` = 主动提前保养。
    ⚠️ "无决策"与"决策全是非激活"在本读数里都记 0.0——"链里根本没有该类决策"这条由
    调用方另行判（有决策数才谈得上分布）。
    """
    ds = [d for d in decisions if d.kind == kind]
    if not ds:
        return 0.0
    return sum(1 for d in ds if is_active(int(d.action))) / len(ds)


class T3Budget:
    """一个实例的 T3 预算：`aᵢ^ref` + `bᵢ` + 实际启用的约束子集（`keep`）。

    `keep` 是**现实约束**的显式表达：某个头没开时，对应的约束不可控，纳入 T3 就是
    "罚策略无法控制的事"（④⑨ 不纳入的同一条理由）——故由调用方显式给出，默认全集。
    """

    def __init__(self, inst: Instance, keep: tuple[str, ...] | None = None,
                 ratio: float = BUDGET_RATIO, caliber: str | None = None):
        keep = tuple(T3_CONSTRAINTS if keep is None else keep)
        for n in keep:
            if n not in T3_CONSTRAINTS:
                raise ValueError(f"未知 T3 约束 {n!r}——可选：{T3_CONSTRAINTS}")
        if not 0.0 < float(ratio) < 1.0:
            raise ValueError(f"ratio={ratio} 非法：预算比例必须落在 (0, 1)——"
                             "0 = 要求零激活（把动作咬死），1 = 等于参考水平（机制空转）。")
        a_ref = t3_ref_vector(inst, caliber)
        idx = [T3_CONSTRAINTS.index(n) for n in keep]
        self.keep: tuple[str, ...] = keep
        self.a_ref: np.ndarray = a_ref[idx]
        self.b: np.ndarray = np.full(len(keep), float(ratio))   # 归一化后的目标水平
        self.ratio = float(ratio)

    def __len__(self) -> int:
        return len(self.keep)

    def labels(self) -> list[str]:
        return [T3_LABELS[n] for n in self.keep]

    def describe(self) -> str:
        """一行自报（打印/日志用）——a^ref、b、约束名。"""
        parts = [f"{T3_LABELS[n]}: a^ref={a:.4g}, b={self.ratio:.2f}"
                 for n, a in zip(self.keep, self.a_ref)]
        return "｜".join(parts)


class T3Lagrangian:
    """λ 的**跨步状态**（初值 0；设计 §3.4：它不是每步重置）。

    ⚠️ **持有者 = `runner.run_training`**（设计点名"最容易写错的一处"）：训练环把
    `step_kwargs()` 注进每一步，再用 `advance(diag)` 把 `joint_chain_step` 算出的下一步 λ
    收回来。`joint_chain_step` 只读 λ、不改对象——单调、非负由 `dual_ascent` 保证。
    ⚠️ `--resume` 不恢复 λ（ckpt 现在也不存优化器状态，设计 §7 的开放线索）：从 0 重来。
    """

    def __init__(self, budget: T3Budget, eta: float = T3_ETA, lam0=None,
                 lam_max: float = T3_LAMBDA_MAX):
        if eta <= 0.0:
            raise ValueError(f"eta={eta} 非法：对偶上升步长必须 > 0")
        self.budget, self.eta, self.lam_max = budget, float(eta), float(lam_max)
        self.lam = (np.zeros(len(budget), dtype=np.float64) if lam0 is None
                    else np.asarray(lam0, dtype=np.float64).copy())
        if self.lam.shape != (len(budget),):
            raise ValueError(f"lam0 形状 {self.lam.shape} ≠ 约束数 {len(budget)}")

    def step_kwargs(self) -> dict:
        """注进 `joint_chain_step` 的三个 T3 形参（当前 λ 的**副本**）。"""
        return {"t3_lambda": self.lam.copy(), "t3_budget": self.budget, "t3_eta": self.eta}

    def advance(self, diag: dict) -> None:
        """从一步的 diag 收下 λ 的下一步值（`joint_chain_step` 算好的）。"""
        nxt = diag.get("t3_lambda")
        if nxt is None:
            raise KeyError("diag 里没有 t3_lambda——T3 开着但 joint_chain_step 没回传 λ "
                           "（接线漂了；不许静默当成'λ 不变'）。")
        self.lam = np.asarray(nxt, dtype=np.float64)
