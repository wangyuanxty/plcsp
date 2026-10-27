# -*- coding: utf-8 -*-
"""**未归一化时，随机初始化锁到哪条规则？**（只测初始化，**不训练**）

══ 为什么要有这个 ══

去掉 `Standardizer` 后实测：随机初始化的 `softmax` 是个 **delta**（最大概率 1.000、熵 0.000）。
策略从第 0 步起就**恒选一个动作、且与状态无关**，**锁到哪条完全由 seed 决定**。

⟹ 于是 `shi2025_min2.py` 每一档的起点都是**抽签**：
- 抽到 `EAJ+SMPT`（147.6）⟹ 判据当场 ✅，**但那是初始化给的**
- 抽到 `SRPT+MINU`（279.6）⟹ 起点就是最差常数

**本脚本把这张签表列出来**，好挑出「锁到坏规则」的 seed，拿去回答那个关键问题：
**锁坏了，训练能不能学回来？**
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # 仓根

import numpy as np  # noqa: E402
import torch  # noqa: E402

import shi2025_min_pa as M  # noqa: E402
from plcsp_v3.env.nhdrl_features import NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.rules import (N_AEI2025_RULES, PA_RULE_NAMES,  # noqa: E402
                                apply_pa_rule, la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402

N_SEED = int(os.environ.get("LOCK_SEED", "10"))
KS = tuple(int(x) for x in os.environ.get("LOCK_KS", "2,3,5,9").split(","))
# K 阶梯的均匀间隔取法，与 `shi2025_min2.py` 一致
RANK = [3, 0, 5, 2, 4, 6, 8, 1, 7]


def act_of(k: int) -> tuple[int, ...]:
    return (tuple(RANK[round(i * (len(RANK) - 1) / (k - 1))] for i in range(k))
            if k > 1 else (RANK[0],))


def main() -> None:
    inst = named(140, 16)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    nf = NHFeatures(inst, spec.layout)
    scs = [100_000 + 13 * i for i in range(20)]

    ms = {}
    for k in range(N_AEI2025_RULES):
        ms[k] = float(np.mean([float(spec.world().run_gated(
            seed_chain=sc, policy_pa=lambda v, p, tr, k=k: apply_pa_rule(k, v, p, tr),
            policy_l=la_rule_policy(0), dt_events=0)["makespan"]) for sc in scs]))
    print("  12 条规则的 20 场景均值："
          + "　".join(f"{PA_RULE_NAMES[k]}:{ms[k]:.1f}" for k in range(N_AEI2025_RULES)))

    # 一条固定规则的轨迹 ⟹ 拿一批真实状态（与训练同分布）
    rows: list = []

    def pol(view, pool, tie_rng=None):
        rows.append(np.asarray(nf.pa(view.snap, 0.0, (5.0, 5.0)), dtype=float))
        return apply_pa_rule(3, view, pool, tie_rng)

    spec.world().run_gated(seed_chain=9_000_001, policy_pa=pol,
                           policy_l=la_rule_policy(0), dt_events=0)
    F = torch.as_tensor(np.stack(rows), dtype=torch.float32)
    print(f"  取样状态：{len(rows)} 个（规则 EAJ+SMPT 的轨迹）\n")

    for K in KS:
        ACT = act_of(K)
        print(f"  ══ K = {K} ══　动作集合 "
              + " ".join(f"{PA_RULE_NAMES[a]}({ms[a]:.0f})" for a in ACT))
        print(f"    {'seed':>5}{'锁到':>14}{'概率':>8}{'该规则纯用':>12}"
              f"{'最差常数':>10}{'相对最好':>10}")
        for sd in range(N_SEED):
            torch.manual_seed(sd)
            net = M.Head(M.F_PA)
            net.pi[2] = torch.nn.Linear(M.HID_A, K)
            with torch.no_grad():
                p = torch.softmax(net.pi(F), -1).mean(0).numpy()
            j = int(p.argmax())
            a = ACT[j]
            tag = "最好" if a == 3 else f"+{ms[a] / ms[3] - 1:.1%}"
            print(f"    {sd:>5}{PA_RULE_NAMES[a]:>14}{p[j]:>8.3f}"
                  f"{ms[a]:>12.1f}{tag:>10}")
        print(flush=True)


if __name__ == "__main__":
    main()
