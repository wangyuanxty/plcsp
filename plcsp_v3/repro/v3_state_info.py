# -*- coding: utf-8 -*-
"""**24 维特征在「一局之内」动不动** —— 决定策略能不能按状态选动作。

══ 为什么测这个 ══

已确立（`v3_headroom.py` 读数 1）：奖励 `−d(cm)` **强烈**依赖动作（逐决策点上 9 条规则
的 `cm` 差到 60%）。所以**信号不是问题**。

但「奖励有信号」≠「策略能按状态选」。策略要能分状态，**特征必须随状态变**。
每一个版本都塌成固定规则。若特征是**一局之内几乎不变**的，那策略**只能**塌 ——
不是学不会，是**没有可用的输入**。

══ 可疑点 ══

（原口径）标准化 `μ/σ` 曾在 **12 规则 × 8 场景**上标定 —— 那是**跨场景**的尺度。
`Standardizer` **已于 2026-10-13 删除**（§37.34.5：它不修任何东西），
本脚本保留「局内 σ / 标定 σ」这一栏，只为**记录当初那个可疑点**。
若 `σ_局内 ≪ σ_标定`，那标准化把**跨场景的差**当成了 1，
把**局内真正要用的差**压到了 0 附近 —— 等于给网络喂一个近常量。

══ 判据 ══

- `局内 / 标定` 的逐维中位数：接近 1 ⟹ 特征在局内的动态范围与标定同量级（正常）。
  **远小于 1 ⟹ 局内几乎不动 ⟹ 这就是病灶。**
- 「局内会动的维数 / 24」：若只剩一两维在动，策略没有可分的输入。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # 仓根

import numpy as np  # noqa: E402

from plcsp_v3.env.nhdrl_features import F_PA, NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.rules import (N_AEI2025_RULES, apply_pa_rule,  # noqa: E402
                                la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402

N_SC = int(os.environ.get("SI_SC", "8"))


def collect(spec, nf, k: int, sc: int) -> np.ndarray:
    """跑一条固定规则，逐步收**原始** 24 维特征。"""
    rows: list[np.ndarray] = []

    def pol(view, pool, tie_rng=None):
        rows.append(np.asarray(nf.pa(view.snap, 0.0, (5.0, 5.0)), dtype=float))
        return apply_pa_rule(k, view, pool, tie_rng)

    spec.world().run_gated(seed_chain=sc, policy_pa=pol,
                           policy_l=la_rule_policy(0), dt_events=0)
    return np.stack(rows)


def main() -> None:
    inst = named(140, 16)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    nf = NHFeatures(inst, spec.layout)
    scs = [100_000 + 13 * i for i in range(N_SC)]
    print(f"24 维特征在局内的动态范围 · Exp(16/140) · {N_SC} 场景 · 固定规则 "
          f"{N_AEI2025_RULES} 条 · F_PA={F_PA}\n", flush=True)

    # ── 标定用的池（与 `M.calibrate` 同口径：12 规则 × 8 场景）──
    pool = np.concatenate([collect(spec, nf, k, sc)
                           for k in range(N_AEI2025_RULES) for sc in scs])
    sd_cal = np.where(pool.std(axis=0) > 1e-9, pool.std(axis=0), 1.0)

    # ── 局内：同一场景同一规则，逐决策点 ──
    FS = [collect(spec, nf, 3, sc) for sc in scs]      # 规则 3 = EAJ+SMPT
    IS = np.stack([F.std(axis=0) for F in FS])
    IR = np.stack([F.max(axis=0) - F.min(axis=0) for F in FS])

    ratio = np.median(IS, axis=0) / sd_cal
    print(f"  局内决策点数：{len(FS[0])}（各场景相同）")
    print("\n  逐维 **局内 σ / 标定 σ**：")
    print(f"    中位数 {np.median(ratio):.4f}　"
          f"最小 {ratio.min():.4f}　最大 {ratio.max():.4f}")
    print(f"    **比值 < 0.1 的维数：{int((ratio < 0.1).sum())} / {F_PA}**")
    print(f"    **比值 < 0.01 的维数：{int((ratio < 0.01).sum())} / {F_PA}**")
    print("\n  逐维明细：")
    print(f"    {'dim':>4}{'局内σ':>11}{'标定σ':>11}{'σ比':>9}"
          f"{'局内极差':>11}{'极差比':>9}")
    for i in range(F_PA):
        print(f"    {i:>4}{np.median(IS[:, i]):>11.4f}{sd_cal[i]:>11.4f}"
              f"{ratio[i]:>9.4f}{np.median(IR[:, i]):>11.4f}"
              f"{np.median(IR[:, i]) / sd_cal[i]:>9.4f}")
    print("\n  ⚠️ 判据：**σ比 远小于 1 ⟹ 局内特征几乎不动 ⟹ 策略没有可分的输入**")


if __name__ == "__main__":
    main()
