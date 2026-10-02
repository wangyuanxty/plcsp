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
    # 实例指纹（口径同 `des._instance_key`）：奖励绝对值只在**同一实例**内有意义，
    # 训练/评估跨实例混用须被 `matches` 挡住。
    inst_key: tuple = ()

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.makespan, self.energy, self.twt)

    def matches(self, inst) -> bool:
        """本参考值是否取自 `inst` 的参考运行（跨实例混用检查）。"""
        from .des import _instance_key
        return self.inst_key == _instance_key(inst)

    @staticmethod
    def of(inst, cfg) -> "ReferenceObjectives":
        """跑一次参考调度（每工序取最短候选 + AGV 轮询）并缓存。

        ⚠️ TWT 必须**事后**按 `d_j = τ·M_ref` 从**同一次运行**的 `completes` 算出：
        参考运行内 `_due()` 被 `_MREF_BUSY` 短路（防"求 M_ref 要跑 run、run 又要 M_ref"的
        递归），故 `r["tardy_twt"]` **恒为 0**——直接取它 f^ref 就不是"实测值"，
        且 `reward_weights` 会对 0 显式报错。交期只影响 metric、不影响仿真动力学，
        故事后算 = 交期开启时的值。
        """
        from .des import (_instance_key, SimConfig, compute_due_dates, reference_run,
                          weighted_tardiness)
        r = reference_run(inst, cfg)
        c = cfg or SimConfig()
        due = compute_due_dates(inst.n_jobs, c.tau, float(r["makespan"]))
        weights = dict.fromkeys(range(inst.n_jobs), 1.0)   # 等权，口径同 `SimWorld._tardy`
        twt = weighted_tardiness(r["completes"], due, weights)
        return ReferenceObjectives(float(r["makespan"]), float(r["energy"]), float(twt),
                                   _instance_key(inst))
