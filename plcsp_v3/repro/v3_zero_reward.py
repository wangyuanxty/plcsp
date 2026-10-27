# -*- coding: utf-8 -*-
"""**奖励到底有没有值** —— 逐决策点看 OA 选了哪支、那一支给了多少奖励。

══ 为什么查它 ══

`Rollout._close` 把**同一个** `r` 同时发给 OA / PA / LA 三个缓冲：

    r = self.rew(cm - self._cm, 0.0, oa_a == 0, self.prefs)

`oa_a == 0` ⟹ 走 **MS 支**（`x = −d_ms`，有信号）。
`oa_a == 1` ⟹ 走 **TC 支**，而 `tc_step` **恒传 0.0**（本仓没有逐步物流账本 ——
模块 docstring 里登记的"缺口，不是近似"）⟹ **该决策的奖励恒为 0**。

⟹ 若 OA 大面积选 TC，**PA/LA 的奖励就是一片 0**，优势恒 0，梯度恒 0。

原文的 TC 支**有**逐步增量（Eq.2 的 `Σ(nct−nst+lct−lst)` 是逐任务的）⟹ **这一条是复现缺口**。

══ 判据 ══

- `oa_a == 0` 的占比。若接近 1 ⟹ 这条不是病根。
- 奖励为 0 的决策点占比。
- 奖励的取值分布（非零部分）。
"""
import os
import sys
from collections import Counter

sys.path.insert(0, r"D:\research\DeepReinforcementLearningScheduling")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from plcsp_v3.env.nhdrl_features import NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

sys.path.insert(0, __import__("os").path.dirname(
    __import__("os").path.abspath(__file__)))   # 同目录（脚本已入库）
from v3_nhdrl import NHDRL, Rollout, RewardFn  # noqa: E402


def main() -> None:
    a, b = int(os.environ.get("V3_A", "16")), int(os.environ.get("V3_B", "140"))
    n_sc = int(os.environ.get("V3_N_SC", "8"))
    ck = os.environ.get("V3_CKPT", "")
    inst = named(b, a)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    nf = NHFeatures(inst, spec.layout)
    net = NHDRL()
    if ck and os.path.exists(ck):
        net.load_state_dict(torch.load(ck, map_location="cpu"))
        print(f"已载入 {ck}")
    else:
        print("⚠️ 随机初始化（对照）")
    net.eval()

    rew = RewardFn()
    oa_cnt, r_vals = Counter(), []
    per_sc = []
    for i in range(n_sc):
        r = Rollout(net, spec, nf, rew, torch.Generator().manual_seed(0), greedy=True)
        spec.world().run_gated(seed_chain=100_000 + 13 * i, policy_pa=r.policy_pa,
                               policy_l=r.policy_l, dt_events=0)
        oa_cnt.update(int(row[1]) for row in r.buf["oa"])
        rv = np.array([row[4] for row in r.buf["oa"]], dtype=float)
        r_vals += rv.tolist()
        per_sc.append((len(rv), float(np.mean(rv)), float(np.mean(np.abs(rv) > 0))))

    R = np.asarray(r_vals)
    n_oa = sum(oa_cnt.values())
    print(f"\n{n_sc} 个场景 · **{n_oa} 个决策点**\n")
    print(f"  OA 动作分布：{dict(sorted(oa_cnt.items()))}"
          f"　（0 = 优化 MS，1 = 优化 TC）")
    if n_oa:
        print(f"  **OA 选「优化 MS」的占比：{oa_cnt[0] / n_oa:.1%}**")
    print(f"\n  奖励取值：非零占比 {float(np.mean(np.abs(R) > 0)):.1%}"
          f"　均值 {R.mean():+.4f}　标准差 {R.std():.4f}")
    print(f"            最小 {R.min():+.4f}　最大 {R.max():+.4f}"
          f"　**不重复取值 {len(np.unique(np.round(R, 6)))} 种**")
    if float(np.mean(np.abs(R) > 0)) > 0:
        nz = R[np.abs(R) > 0]
        print(f"  非零部分：均值 {nz.mean():+.4f}　最小 {nz.min():+.4f}　最大 {nz.max():+.4f}")
    print("\n  逐场景（决策点数 / 奖励均值 / 非零占比）：")
    for i, (n, m, nz) in enumerate(per_sc):
        print(f"      #{i}　{n:>4} 点　均值 {m:+.4f}　非零 {nz:6.1%}")

    print("\n  ⟹ 若「OA 选 MS 的占比」远小于 1，则**其余决策点的奖励恒为 0**，")
    print("     PA/LA 在这部分点上**拿不到任何梯度**。这是复现缺口（原文 TC 支有逐步增量）。")


if __name__ == "__main__":
    main()
