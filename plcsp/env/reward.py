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
