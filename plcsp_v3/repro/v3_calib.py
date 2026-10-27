# -*- coding: utf-8 -*-
"""**标定两支的典型单步尺度** `S_ms` / `S_tc` —— 给「各除各的尺度」那版奖励用。

为什么需要：`raw` 口径下两支的单步代价差 **2.6 倍**（§37.19.2），OA 于是白拿 TC 的便宜。
除以各自的典型尺度后，**两支的单步典型量都是 1** ⟹ 回报水平相同 ⟹ OA 的选择不再被
"哪支扣得少"主导。

⚠️ **标定要在"训练开始时那个分布"上做** —— 用**固定规则**跑（≈ 随机初始策略的行为），
不用训好的策略（那会把学坏的口径再喂回去）。
"""
import os
import sys

sys.path.insert(0, r"D:\research\DeepReinforcementLearningScheduling")

import numpy as np  # noqa: E402

from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.rules import apply_pa_rule, la_rule_policy  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

sys.path.insert(0, __import__("os").path.dirname(
    __import__("os").path.abspath(__file__)))   # 同目录（脚本已入库）
from v3_nhdrl import UC_TRANSPORT, cm_of, tc_cum  # noqa: E402


class Cap:
    def __init__(self, rule: int, lar: int) -> None:
        self.rule, self.lar, self.cm, self.tc = rule, lar, None, None
        self.d, self.t = [], []

    def policy_pa(self, view, pool, tie_rng=None):
        cm, tc = cm_of(view.snap), tc_cum(view.snap)
        if self.cm is not None:
            self.d.append(cm - self.cm)
            self.t.append((tc - self.tc) / UC_TRANSPORT)   # 换回分钟
        self.cm, self.tc = cm, tc
        return apply_pa_rule(self.rule, view, pool, None)

    def policy_l(self, view, cand_v):
        return int(la_rule_policy(self.lar, (0.0,))(view, cand_v))


def main() -> None:
    a, b = int(os.environ.get("V3_A", "16")), int(os.environ.get("V3_B", "140"))
    n_sc = int(os.environ.get("V3_N_SC", "4"))
    inst = named(b, a)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)

    D, T = [], []
    for i in range(n_sc):
        for k in range(9):
            for lar in (0, 1, 2):
                c = Cap(k, lar)
                spec.world().run_gated(seed_chain=100_000 + 13 * i,
                                       policy_pa=c.policy_pa, policy_l=c.policy_l,
                                       dt_events=0)
                D += c.d
                T += c.t
    D, T = np.abs(np.asarray(D)), np.abs(np.asarray(T))
    print(f"Exp({a}/{b}) · {n_sc} 场景 × 9 PA 规则 × 3 LA 规则 · **{D.size} 个单步量**\n")
    for nm, X in (("MS 支  |d_ms|          [min]", D), ("TC 支  |tc_step|/uc    [min]", T)):
        nz = X[X > 1e-12]
        print(f"  {nm}：均值 {X.mean():.4f}　中位 {np.median(X):.4f}"
              f"　非零占比 {nz.size / X.size:.1%}")
        if nz.size:
            print(f"      非零部分：均值 {nz.mean():.4f}　中位 {np.median(nz):.4f}"
                  f"　p90 {np.percentile(nz, 90):.4f}")
    print(f"\n  ⟹ **建议 S_ms = {D.mean():.4f}　S_tc = {T.mean():.4f}**"
          f"（两支的「每步典型量」都归到 1）")
    print(f"  ⟹ 现值（未归一）两支均值比 = {D.mean() / T.mean():.2f} 倍"
          f"　—— §37.19.2 在随机初始化下量到 2.6 倍，同向同量级")


if __name__ == "__main__":
    main()
