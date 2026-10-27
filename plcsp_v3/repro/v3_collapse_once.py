# -*- coding: utf-8 -*-
"""**坍缩是不是一次更新完成的？优势归一化是不是放大器？**

══ 起因 ══

最小版（跨链 GAE 已修、`lr=1e-3`）**`step 1` 就已经 `EAJ+MINU` 90%** ——
9 个动作、随机初始化，本该 ~11% 均摊。**坍缩发生在第一批更新，与训练量无关。**

**待查机制**：`ADV = (ADV − mean)/(std + 1e-6)` ——
奖励几乎恒定 ⟹ 原始 `ADV.std()` 极小 ⟹ **除一个极小数把噪声放大到 O(1)**。

══ 这个脚本量什么 ══

1. 随机初始化下，**更新前**的策略分布（应接近均匀）；
2. 第一批的 **原始 advantage 的 std**、奖励的 std；
3. **一次 `update` 后**的策略分布 —— 看它一步跳多远；
4. **不归一化**时，同一次更新后跳多远（对照）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # 仓根

import numpy as np  # noqa: E402
import torch  # noqa: E402

from plcsp_v3.env.nhdrl_features import NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

import shi2025_min_pa as M  # noqa: E402  （同目录）

LA_FIXED, N_CHAIN = 0, 4


def run_rollout(net, spec, nf):
    chains = []
    for _i in range(N_CHAIN):
        r = M.Roll(net, spec, nf)
        spec.world().run_gated(seed_chain=9_000_001, policy_pa=r.policy_pa,
                               policy_l=r.policy_l, dt_events=0)
        chains.append(r.buf)
    return chains


def dist_of(net, chains) -> np.ndarray:
    """策略在**这批状态**上的平均动作概率（9 维）。"""
    S = torch.as_tensor(np.stack([b[0] for ch in chains for b in ch]),
                        dtype=torch.float32)
    with torch.no_grad():
        return torch.softmax(net.pi(S), -1).mean(0).numpy()


def one_update(net, chains, lr, normalize: bool, epochs=4) -> dict:
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    S, A, LP0, ADV, RET = [], [], [], [], []
    for buf in chains:
        adv, ret = M.gae([b[4] for b in buf], [b[3] for b in buf])
        S += [b[0] for b in buf]
        A += [b[1] for b in buf]
        LP0 += [b[2] for b in buf]
        ADV += list(adv)
        RET += list(ret)
    S = torch.as_tensor(np.stack(S), dtype=torch.float32)
    A = torch.as_tensor(A, dtype=torch.long)
    LP0 = torch.as_tensor(LP0, dtype=torch.float32)
    raw = np.asarray(ADV)
    ADVt = torch.as_tensor(raw, dtype=torch.float32)
    if normalize:
        ADVt = (ADVt - ADVt.mean()) / (ADVt.std() + 1e-6)
    RETt = torch.as_tensor(np.asarray(RET), dtype=torch.float32)
    for _ in range(epochs):
        d = net.dist(S)
        ratio = torch.exp(d.log_prob(A) - LP0)
        loss = (-torch.min(ratio * ADVt,
                           torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * ADVt).mean()
                + 0.5 * ((net.value(S) - RETt) ** 2).mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return {"adv_std_raw": float(raw.std()), "adv_mean_raw": float(raw.mean()),
            "adv_std_norm": float(ADVt.std())}


def main() -> None:
    a, b = 16, 140
    inst = named(b, a)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    nf = NHFeatures(inst, spec.layout)
    names = M.PA_RULE_NAMES
    print(f"Exp({a}/{b}) · 每条链 {0} · 场景数 1（组内 4 链同场景）\n")

    for lr in (1e-3, 1e-4):
        for norm in (True, False):
            torch.manual_seed(0)
            net = M.Head()
            chains = run_rollout(net, spec, nf)
            r = np.asarray([b_[4] for ch in chains for b_ in ch])
            before = dist_of(net, chains)
            st = one_update(net, chains, lr, normalize=norm)
            after = dist_of(net, chains)
            print(f"  lr={lr:g}　归一化={'开' if norm else '**关**'}")
            print(f"      奖励：mean {r.mean():+.5f}　**std {r.std():.5f}**"
                  f"　不同取值 {len(np.unique(np.round(r, 9)))}")
            print(f"      advantage：原始 std **{st['adv_std_raw']:.5f}**"
                  f"　归一化后 std {st['adv_std_norm']:.3f}"
                  f"　放大倍数 {st['adv_std_norm'] / max(st['adv_std_raw'], 1e-12):.0f}×")
            print("      更新前 分布 top3："
                  + " ".join(f"{names[k]}:{before[k]:.2f}"
                             for k in np.argsort(-before)[:3]))
            print("      更新后 分布 top3："
                  + " ".join(f"{names[k]}:{after[k]:.2f}"
                             for k in np.argsort(-after)[:3]))
            tv = 0.5 * float(np.abs(after - before).sum())
            print(f"      **一次更新的总变差距离 = {tv:.3f}**"
                  f"　（0 = 没动，1 = 完全换边）\n")


if __name__ == "__main__":
    main()
