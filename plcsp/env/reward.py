# -*- coding: utf-8 -*-
"""三目标奖励（spec §4.2）——A 阶段的加权标量化。

    r = w₁·(−makespan) + w₂·(−energy) + w₃·(−TWT)
    wᵢ = (1/fᵢ^ref) / Σⱼ(1/fⱼ^ref)      fᵢ^ref = 参考调度下的第 i 个目标实测值

**按参考调度归一化后等权**：物理含义是"相对参考调度各改进一个单位，贡献相同"，
且可复现（不依赖拍脑袋）。**权重须与 M_ref 一同写进论文实验设置。**

⚠️ **跨实例不可比**：`f^ref` 是按实例算的，故奖励绝对值只在单实例内有意义。
训练/评估不得跨实例混用；**同一实例换 cfg 亦然**（`f^ref` 随 n_agv/车速/通道宽…变）。
`ReferenceObjectives` 同时存**实例指纹与 cfg 指纹**，`matches(inst, cfg)` 可断言。

期末一次性结算，无中间奖励（→ §4.4 的切比雪夫可加性问题不存在）。

⚠️ **完成度守卫**（2026-10-06）：`horizon_hit=True`（episode 没跑完）时，`objective_vector`
返回三个**结构上界**，不返回实测值。理由与三条界的来历见 `_incomplete_objectives`。
`horizon_hit=False` 时逐位等于今日读数——**全部既有读数靠这一条**（有测试钉住）。

⚠️ `ReferenceObjectives.of` **不走** `objective_vector`（它从同一次参考运行的 `completes`
另算 TWT）⟹ 本次守卫**不影响** `f^ref` / 奖励权重 `w`。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..energy import AGV_EMPTY_KW, AGV_IDLE_KW, AGV_LOADED_KW, MACHINE_TIERS, SHOP_FIXED_KW

# 上限界用的功率常数（与 `energy.py` 同源；只用于构造**上界**，不是新参数）。
# 取各档区间的**上端**（比模型实际用的低端更保守）⟹ 该界对"换高端值"的敏感性扫描也成立。
_MAX_MACHINE_KW = max(max(t["proc"][1], t["idle"][1]) for t in MACHINE_TIERS.values())
_MAX_AGV_KW = max(AGV_IDLE_KW, AGV_EMPTY_KW, AGV_LOADED_KW)


def _incomplete_objectives(r: dict) -> tuple[float, float, float]:
    """未跑完的 episode → 三个**占优上界**：每个分量都严格大于任何跑完 episode 的同名分量。

    三条界的来历（都用 `des.py` 自己的定义，不引新假设）：

    1. **makespan** ≤ `horizon`（停表条件 = `done | timeout(horizon)`：末件到站即停、只有真死锁
       才走到护栏；护栏仍是一切完工时刻的上界，`des.LuStation.run` 的完成事件）⟹ 取 `horizon + 1`。
    2. **energy** ≤ (最贵机床功率 × 机台数 + 最贵 AGV 功率 × 车数 + 车间固定) × makespan ÷ 60
       ——机床三态时长之和与 AGV 三态时长之和都**恒等于 makespan**（`_energy_report` 的
       闭合口径）⟹ 再取 `horizon + 1` 得严格上界。
    3. **TWT** = Σ_j w_j·max(0, C_j − d_j) ≤ Σ_j C_j ≤ 作业数 × makespan（等权）⟹ 同取 `+1`。

    ⚠️ 三项都用 `horizon + 1`，不用 `horizon`：完工时刻 ≤ horizon，`+1` 保证**严格**大于，
    不依赖"某个事件不会恰好发生在 horizon 这一刻"。
    ⚠️ 缺键时**显式报错**，不静默退回实测值——静默退回正是本守卫要修的那个 bug。
    """
    missing = [k for k in ("horizon", "n_machines", "n_agv", "n_jobs") if k not in r]
    if missing:
        raise ValueError(
            f"完成度守卫缺少 metrics 键 {missing}（`horizon_hit=True` 时必需）——"
            "`des.SimWorld.run` / `run_gated` 的返回都带这些键；手搓 metrics 请补齐。")
    h = float(r["horizon"]) + 1.0
    p_kw = (_MAX_MACHINE_KW * int(r["n_machines"]) + _MAX_AGV_KW * int(r["n_agv"])
            + SHOP_FIXED_KW)
    return (h, p_kw * h / 60.0, float(r["n_jobs"]) * h)


def objective_vector(r: dict) -> tuple[float, float, float]:
    """评测返回 dict → (makespan, energy, TWT)。三者都是"越小越好"。

    ⚠️ **完成度守卫**（2026-10-06）：`r["horizon_hit"]` 为真时返回 `_incomplete_objectives(r)`
    的上界三元组；为假时返回实测三元组（**逐位不变**）。

    **为什么必须守**：没跑完时三项全都"看起来更好"——makespan 是**部分完工的最大值**、
    energy 按（更小的）makespan 计、TWT 只算已完工的作业（甚至可以为 0）。
    ⟹ **"少干活 / 让车队趴窝"是奖励吸引子**（⑪ 验证档实测：2/10 完工却报 46.22，
    见 `progress-log.md` §52.9.2）。

    **为什么用分量上界**，而不是"makespan 取 horizon"（只罚一项）、固定大罚、或按完成比例缩放：
    奖励是三项的**非负加权和**（`scalar_reward`）。只有**每个分量都严格更差**，才能保证
    "未跑完的奖励严格差于**任何**跑完的 episode"——只罚一项时，另外两项的"假改善"
    （尤其 energy 权重在 MK01 上占 ~0.89）仍可能把总奖励拉回去。固定大罚要选一个跨实例的
    魔数；按完成比例缩放不给出绝对序。本做法只用到实例自己的量与 `energy.py` 的引证常数。

    ⚠️ 缺 `horizon_hit` 键的 dict（手搓的）按"跑完"处理——与今日行为一致。
    """
    if r.get("horizon_hit", False):
        return _incomplete_objectives(r)
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
    """Σ wᵢ(−fᵢ)。`prefs`（spec §6.1 预留的签名）：给定时**取代** `w`（不是相乘）。

    A 阶段用它表达"单目标配置"——one-hot `(1,0,0)` = 纯 makespan；`None` = 用 f^ref 派生的 `w`
    （既有读数靠这条，逐位不变）。组内 z 化会把总尺度消掉，故 one-hot 的原始量纲
    （makespan ~10²、energy ~10¹、TWT ~10¹）不进优势，只改目标的相对权重。
    """
    ww = prefs if prefs is not None else w
    return float(sum(-wi * fi for wi, fi in zip(ww, f)))


def _ref_cfg_key(cfg) -> tuple:
    """`ReferenceObjectives` 的 cfg 指纹 = `des._cfg_key` + **(tau, due_range)**。

    ⚠️ 与 `_cfg_key`（`reference_run` 的缓存键）差**两项**：参考运行的动力学与交期无关
    （故缓存键跳过 τ / R，免得敏感性扫描反复重跑），但 f^ref 的 **TWT 分量按新的 TF/RDD
    交期事后算**（见 `of`）——交期口径一变 TWT 就变，w 跟着变。两个**都要**补回：只补 τ
    会漏掉 R（同样改 d_j）。其余字段（n_agv / 车速 / 通道宽…）直接复用 `_cfg_key` 的枚举，
    新增 cfg 字段默认进键（失效安全）。
    """
    from .des import SimConfig, _cfg_key
    c = cfg or SimConfig()                  # None = 默认 cfg（同 `of` / `reference_run` 的语义）
    return _cfg_key(c) + (("tau", c.tau), ("due_range", c.due_range))


@dataclass(frozen=True)
class ReferenceObjectives:
    """参考调度下的三个目标实测值（与 `M_ref` 取自**同一次**运行）。"""
    makespan: float
    energy: float
    twt: float
    # 实例指纹（口径同 `des._instance_key`）：奖励绝对值只在**同一实例**内有意义，
    # 训练/评估跨实例混用须被 `matches` 挡住。
    inst_key: tuple = ()
    # cfg 指纹（口径同 `des._cfg_key` + `tau`，见 `_ref_cfg_key`）：**同实例换 cfg，f^ref 也变**
    # （实测 mk01 的 M_ref：n_agv=1/3/5 → 109.95/103.42/97.24；车速 0.5/1.0 → 103.42/106.38），
    # 而 w 由 f^ref 算出——只挡实例不挡 cfg，P4 扫 n_agv 时会**静默**沿用旧 w，
    # "按参考调度归一化"在这条扫描轴上不成立且零报错（评审 I-1）。
    cfg_key: tuple = ()

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.makespan, self.energy, self.twt)

    def matches(self, inst, cfg) -> bool:
        """本参考值是否与 `(inst, cfg)` 同源（**跨实例 + 跨 cfg** 混用检查）。

        ⚠️ `cfg` **必传**（`None` = 默认 `SimConfig()`，与 `of` / `reference_run` 的 None 语义
        一致——**不是**"跳过 cfg 检查"）：只查实例会漏掉"同实例换 cfg"这条，而它是**静默**的。
        故不给默认值：省略即 TypeError，逼调用方明确 cfg 口径。
        """
        from .des import _instance_key
        return self.inst_key == _instance_key(inst) and self.cfg_key == _ref_cfg_key(cfg)

    @staticmethod
    def of(inst, cfg) -> "ReferenceObjectives":
        """跑一次参考调度（每工序取最短候选 + AGV 轮询）并缓存。

        ⚠️ TWT 必须**事后**按 **TF/RDD 交期**从**同一次运行**的 `completes` 算出：
        参考运行内 `_due()` 被 `_MREF_BUSY` 短路（参考运行的指标口径把 ⑧ 当关——旧口径下
        求交期要跑参考运行、跑参考运行又要求交期，会递归），故 `r["tardy_twt"]` **恒为 0**
        ——直接取它 f^ref 就不是"实测值"，且 `reward_weights` 会对 0 显式报错。交期只影响
        metric、不影响仿真动力学，故事后算 = 交期开启时的值。
        ⚠️ 交期入口与仿真**同源**（`compute_due_dates` → `due_dates.due_dates_for`）：
        两边各写一份口径就会静默错位（Review Focus #5）。
        """
        from .des import (_instance_key, SimConfig, compute_due_dates, reference_run,
                          weighted_tardiness)
        r = reference_run(inst, cfg)
        c = cfg or SimConfig()
        due = compute_due_dates(inst, c.tau, c.due_range)
        weights = dict.fromkeys(range(inst.n_jobs), 1.0)   # 等权，口径同 `SimWorld._tardy`
        twt = weighted_tardiness(r["completes"], due, weights)
        return ReferenceObjectives(float(r["makespan"]), float(r["energy"]), float(twt),
                                   _instance_key(inst), _ref_cfg_key(c))
