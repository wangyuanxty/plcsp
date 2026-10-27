# -*- coding: utf-8 -*-
"""**奖励对齐探针** —— 直接量 `corr(ΣR, 最终 makespan)`。

**为什么必须先做这一步**：训练的目标是 `max ΣR`。若 `ΣR` 与最终 makespan **不同向**，
那么"学不出来"就不是网络的错，是**奖励口径的错** —— 再换结构、换 lr、换探索都救不回来。

做法：**固定一条规则跑一局**（不给网络任何自由度），遍历 (9 条 PA 规则 × 8 个场景)，
记录 `ΣR`（用 `v3_nhdrl.RewardFn`，与训练**同一个类**）与该局的最终 makespan。

- 若 `corr(ΣR, MS) < 0` ⟹ 奖励方向对。
- 若 `corr(ΣR, MS) ≥ 0` ⟹ **奖励方向错**，训练在朝反方向爬。
- 另外打印**每条规则的 `ΣR` 均值 vs 该规则的 MS 均值** —— 看奖励**偏好哪条规则**，
  和**哪条规则真的最好**是否一致。这两列不一致 = 学出来的必然是错的规则。
"""
import os
import sys

sys.path.insert(0, r"D:\research\DeepReinforcementLearningScheduling")

import numpy as np  # noqa: E402

from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.rules import (PA_RULE_NAMES, apply_pa_rule,  # noqa: E402
                                la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402

sys.path.insert(0, __import__("os").path.dirname(
    __import__("os").path.abspath(__file__)))   # 同目录（脚本已入库）
from v3_nhdrl import RewardFn, cm_of  # noqa: E402

N_RULE, N_SC, LA_RULE = 9, 8, 0


class Probe:
    """只算奖励的壳子 —— 不给网络自由度，纯固定规则。

    除 `ΣR` 外还记**三个原料量**，用来把"奖励被压扁"归因到具体一项：

    - `T`  = 决策点数（`ΣR` 里 `0.9` 那一项累加的次数）
    - `Sd` = `Σ d_ms`（增量之和；由 `cm_of` 望远镜相消，恒等于 `CM末 − CM首`）
    - `hi` = `RewardFn` 里跑完后的 `ms_hi`（运行观测极大值 = 归一化分母的来源）
    """

    def __init__(self, rew: RewardFn, rule: int) -> None:
        self.rew, self.rule = rew, rule
        self.R, self._cm = 0.0, None
        self.T, self.Sd = 0, 0.0

    def policy_pa(self, view, pool, tie_rng=None):
        cm = cm_of(view.snap)
        if self._cm is not None:
            # ⚠️ 与 `Rollout._close` 逐字同参：`oa_ms=True`（只看 MS 支），`prefs=(5,5)`
            d = cm - self._cm
            self.R += self.rew(d, 0.0, True, (5.0, 5.0))
            self.T += 1
            self.Sd += d
        self._cm = cm
        return apply_pa_rule(self.rule, view, pool, tie_rng)

    def policy_l(self, view, cand_v):
        return int(la_rule_policy(LA_RULE, (0.0,))(view, cand_v))


def main() -> None:
    a, b = int(os.environ.get("V3_A", "16")), int(os.environ.get("V3_B", "140"))
    inst = named(b, a)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    # ⚠️ **整个探针共用一个 `RewardFn`** —— 与训练同一约束（尺度必须统一，否则比的是尺度）
    rew = RewardFn()
    scs = [100_000 + 13 * i for i in range(N_SC)]

    print(f"Exp({a}/{b}) · 9 条 PA 规则 × {N_SC} 个场景 · LA 固定 rule {LA_RULE}\n", flush=True)
    print(f"  {'PA 规则':<13}{'ΣR 均值':>11}{'T 均值':>9}{'Σd 均值':>10}"
          f"{'MS 均值':>10}{'MS 最小':>10}")
    print("  " + "-" * 64)
    R, M, T, S = [], [], [], []
    for k in range(N_RULE):
        rk, mk, tk, sk = [], [], [], []
        for sc in scs:
            p = Probe(rew, k)
            m = spec.world().run_gated(seed_chain=sc, policy_pa=p.policy_pa,
                                       policy_l=p.policy_l, dt_events=0)
            rk.append(p.R)
            mk.append(float(m["makespan"]))
            tk.append(p.T)
            sk.append(p.Sd)
        R += rk
        M += mk
        T += tk
        S += sk
        print(f"  {PA_RULE_NAMES[k]:<13}{np.mean(rk):>11.4f}{np.mean(tk):>9.1f}"
              f"{np.mean(sk):>10.2f}{np.mean(mk):>10.1f}{min(mk):>10.1f}", flush=True)

    R, M = np.asarray(R), np.asarray(M)
    T, S = np.asarray(T), np.asarray(S)

    def _c(x: np.ndarray) -> str:
        return f"{np.corrcoef(x, M)[0, 1]:+.4f}"

    print("\n  ══ 归因：把 `ΣR = 0.9·(T − Σd/H)` 拆开 ══")
    print(f"  corr(ΣR, MS)   = **{_c(R)}**　← 当前奖励（负 = 方向对）")
    print(f"  corr(T,  MS)   = {_c(T)}　← 决策点数项。**正相关 ⟹ 奖励在奖「决策多」**")
    print(f"  corr(Σd, MS)   = {_c(S)}　← 增量项。**正相关 ⟹ 这一项方向对**")
    print(f"  corr(−Σd, MS)  = {_c(-S)}　← 若只留增量项，分辨力如何")
    print("\n  量程（越大越有分辨力，用 CV 比）：")
    for nm, x in (("ΣR", R), ("T", T), ("Σd", S), ("MS", M)):
        print(f"    {nm:<4} {x.min():>9.2f} – {x.max():>9.2f}"
              f"　极差/均值 = {np.ptp(x) / abs(x.mean()):.4f}")
    print("\n  corr(T, MS) 若为正：规则让决策点变多 ⟹ `0.9·T` 项抬 ΣR ⟹ "
          "**策略被引向「拖长流程」**（与「缩短 makespan」相反）。")

    # 奖励偏好的规则 vs 真正最好的规则
    per_r = np.array([R[i * N_SC:(i + 1) * N_SC].mean() for i in range(N_RULE)])
    per_m = np.array([M[i * N_SC:(i + 1) * N_SC].mean() for i in range(N_RULE)])
    print(f"\n  奖励偏好的规则（ΣR 最大）= **{PA_RULE_NAMES[int(per_r.argmax())]}**"
          f"（ΣR {per_r.max():.4f}）")
    print(f"  真正最好的规则（MS 最小）= **{PA_RULE_NAMES[int(per_m.argmin())]}**"
          f"（MS {per_m.min():.1f}）")
    if int(per_r.argmax()) != int(per_m.argmin()):
        print("  ⟹ **两者不一致** —— 奖励会把策略领到错的规则上。")
    else:
        print("  ⟹ 两者一致，奖励在这一层没有指错方向。")


if __name__ == "__main__":
    main()
