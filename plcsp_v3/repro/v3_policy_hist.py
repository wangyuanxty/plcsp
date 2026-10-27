# -*- coding: utf-8 -*-
"""**策略直方图** —— 训练完的 net 到底还在不在看状态。

判据：逐决策点记 PA 的 **argmax**。

- **恒为 1 个动作** ⟹ 塌成CDR ⟹ **与"上层按状态选规则"这件事无关**，
  HV 高低只反映它挑中了哪条 CDR。
- **多个动作且有结构** ⟹ 真的在按状态切换。

不需要新代码：`Rollout.buf[k]` 每行是 `(state, a, logp, v, r)`，取 `[1]` 就是动作序列。
"""
import os
import sys
from collections import Counter

sys.path.insert(0, r"D:\research\DeepReinforcementLearningScheduling")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from plcsp_v3.env.nhdrl_features import F_LA, F_OA, F_PA, NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.rules import PA_RULE_NAMES  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

sys.path.insert(0, __import__("os").path.dirname(
    __import__("os").path.abspath(__file__)))   # 同目录（脚本已入库）
from v3_nhdrl import NHDRL, Rollout, RewardFn  # noqa: E402

N_SC = int(os.environ.get("V3_N_SC", "8"))


def main() -> None:
    a, b = int(os.environ.get("V3_A", "16")), int(os.environ.get("V3_B", "140"))
    ck = os.environ.get("V3_CKPT", "")
    inst = named(b, a)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    nf = NHFeatures(inst, spec.layout)
    net = NHDRL()
    assert (F_OA, F_PA, F_LA) == (31, 24, 30)
    if ck and os.path.exists(ck):
        net.load_state_dict(torch.load(ck, map_location="cpu"))
        print(f"已载入 {ck}")
    else:
        print("⚠️ 未载入权重 —— 随机初始化")
    net.eval()

    acts: dict[str, Counter] = {"oa": Counter(), "pa": Counter(), "la": Counter()}
    states: list = []
    for i in range(N_SC):
        r = Rollout(net, spec, nf, RewardFn(),
                    torch.Generator().manual_seed(0), greedy=True)
        spec.world().run_gated(seed_chain=100_000 + 13 * i, policy_pa=r.policy_pa,
                               policy_l=r.policy_l, dt_events=0)
        for k in acts:
            acts[k].update(int(row[1]) for row in r.buf[k])
        states += [row[0] for row in r.buf["pa"]]

    print(f"\n{N_SC} 个场景，逐决策点的 **argmax** 直方图\n")
    for k in ("oa", "pa", "la"):
        c = acts[k]
        n = sum(c.values())
        print(f"  {k.upper()}：{n} 个决策点，{len(c)} 种动作")
        for act, cnt in c.most_common(6):
            nm = PA_RULE_NAMES[act] if k == "pa" else str(act)
            print(f"      {act:>2} {nm:<12} {cnt:>5}　{cnt / n:6.1%}")
        print()

    # ⚠️ **关键澄清**：argmax 用了几种动作，**不等于**学到了状态依赖。
    # 若 logits 几乎相等，argmax 只是在"并列"里被浮点噪声挑一个 ⟹ 直方图看着"分散"，其实是瞎。
    # 判据：`ln(动作数)` = 均匀分布的熵。PA 是 `ln 9 = 2.197`。
    with torch.no_grad():
        S = torch.as_tensor(np.stack(states), dtype=torch.float32)
        lg = net.pa.actor(S)
        rng = (lg.max(-1).values - lg.min(-1).values)
        ent = torch.distributions.Categorical(logits=lg).entropy()
    n_pa = len(acts["pa"])
    print("  ══ PA 的「是不是真在选」══")
    print(f"    logit 极差  均值 {float(rng.mean()):.4f}　"
          f"中位 {float(rng.median()):.4f}　最大 {float(rng.max()):.4f}")
    print(f"    softmax 熵  均值 {float(ent.mean()):.4f}　"
          f"（**均匀分布 = ln 9 = {np.log(9):.4f}**）")
    print(f"    ⟹ 熵越接近 {np.log(9):.4f}、极差越接近 0 ⟹ 9 个动作**几乎并列**，"
          f"argmax 是噪声不是决策。")
    print(f"  ⟹ PA 用了 **{n_pa}** 种动作"
          + ("　**塌成CDR**" if n_pa == 1 else "　有状态依赖" if n_pa > 2 else ""))


if __name__ == "__main__":
    main()
