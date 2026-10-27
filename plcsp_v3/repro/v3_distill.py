# -*- coding: utf-8 -*-
"""**监督蒸馏**：把「逐点该用哪条」的答案**直接喂给同一个 24 维网络**，再评它能不能赢常数。

══ 为什么做这个（GLM 的 A 实验）══

已确立的三条：
- **逐场景 oracle** 比CDR好 **15.37%** —— 但那 20 条证人各不相同，**不可部署**（§37.40）；
- **单一 `(t,k)`**（992 个候选里挑的）好 4.69% —— **留出集上只剩 ~1.5%**，即**过拟合**（§37.45）；
- 24 维状态对「该用哪条」有**弱**信息（35.1% vs 基线 23.4%，§37.41）。

**本脚本一刀切开「估计器提不出」与「特征/策略类表达不了」：**

用**监督**（交叉熵）把答案键灌进**同一个 `M.Head`（24 维输入、含 Tanh）**，
**不经过 PPO**。然后按场景**留出**评测。

| 结果 | 含义 |
|---|---|
| **监督策略能明显赢过CDR** | 信息**可表达** ⟹ 病灶在 **PPO 估计器**（信用分配 / δ 噪声） |
| **监督策略也赢不了** | 信息**表达不了** ⟹ 病灶在**特征 / 策略类**，**别再调 RL 超参** |

══ 答案键的口径（比 GLM 说的多一项）══

⚠️ 每个 `(场景, t)` 有 **9 个选项**：`A[sc,t,k]`（偏离到规则 `k`，k≠3）**以及 `base[sc]`**
（**保持根规则 `EAJ+SMPT`**）。**少了「保持」这一项，网络学不出「这里不该切」。**

⚠️ 标签有**并列**（§37.41：72% 的点上最好与次好差 0.00 分钟）⟹
**交叉熵学的是「并列集合里的一个」，不是唯一真值。** 这一栏的读数要按此打折。

══ 评测口径 ══

**确定性平局**（与答案键同源，`base` 也是确定性平局下的数），**不与 `147.6`（随机平局）相减**。
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
from plcsp_v3.hrl.rules import PA_RULE_NAMES, apply_pa_rule, la_rule_policy  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

ROOT = 3
NPZ = os.environ.get("DIS_NPZ", os.path.join(os.environ.get("TEMP", "/tmp"),
                                             "devscan.npz"))
EPOCHS = int(os.environ.get("DIS_EPOCHS", "400"))
N_RUN = int(os.environ.get("DIS_RUN", "8"))


def train_ce(net, X, Y, epochs: int) -> None:
    """交叉熵把 (状态 → 该用哪条) 灌进网络。**不经过 PPO。**"""
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(Y, dtype=torch.long)
    for _ in range(epochs):
        loss = torch.nn.functional.cross_entropy(net.pi(xt), yt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def eval_policy(spec, nf, net, scs) -> float:
    """greedy 跑一遍，返回 makespan 均值。**确定性平局**（与答案键同源）。"""
    ms = []
    for sc in scs:
        def pol(view, pool, tie_rng=None, _n=net, _nf=nf):
            with torch.no_grad():
                f = np.asarray(_nf.pa(view.snap, 0.0, (5.0, 5.0)), dtype=np.float32)
                a = int(_n.pi(torch.as_tensor(f[None])).argmax())
            return apply_pa_rule(a, view, pool, None)

        ms.append(float(spec.world().run_gated(
            seed_chain=sc, policy_pa=pol, policy_l=la_rule_policy(0),
            dt_events=0)["makespan"]))
    return float(np.mean(ms))


def main() -> None:
    d = np.load(NPZ)
    A, X, base, scs_in = d["A"], d["X"], d["base"], d["scs"]
    n_sc, T, n_r = A.shape
    alts = [k for k in range(n_r) if k != ROOT]
    print(f"监督蒸馏 · 从 `{NPZ}` 读入 · {n_sc} 场景 × {T} 决策点 · "
          f"{len(alts)} 条备选 + **保持根规则** ⟹ 每点 {len(alts) + 1} 个选项\n")

    # ── 答案键：(状态 → 9 选 1) ──
    Xs, Ys, Gs = [], [], []
    for i in range(n_sc):
        for t in range(T):
            row = A[i, t, alts]
            if not np.isfinite(row).all() or not np.isfinite(X[i, t]).all():
                continue
            opts = list(row) + [base[i]]            # ⚠️ 末项 = **保持根规则**
            k = int(np.argmin(opts))
            Xs.append(X[i, t])
            Ys.append(ROOT if k == len(alts) else alts[k])
            Gs.append(i)
    Xs = np.asarray(Xs, dtype=float)
    Ys = np.asarray(Ys)
    Gs = np.asarray(Gs)
    cnt = np.bincount(Ys, minlength=n_r)
    print(f"  样本 {len(Ys)}　多数类基线 {cnt.max() / len(Ys):.1%}")
    print("  标签分布：" + "　".join(f"{PA_RULE_NAMES[k]}×{cnt[k]}"
                                      for k in np.argsort(-cnt) if cnt[k] > 0))

    inst = named(140, 16)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    nf = NHFeatures(inst, spec.layout)
    scs = [int(x) for x in scs_in]
    base_mean = float(base.mean())
    print(f"\n  CDR（确定性平局 · 同 {n_sc} 场景）= **{base_mean:.1f}**\n")

    # ── 按场景留出：一半训、一半评 ──
    print(f"  {'划分':<10}{'训练样本':>10}{'留出场景数':>12}{'留出 makespan':>15}{'赢常数':>10}")
    print("  " + "-" * 58)
    wins = []
    for r in range(N_RUN):
        rng = np.random.default_rng(r)
        perm = rng.permutation(n_sc)
        tr_sc, te_sc = perm[:n_sc // 2], perm[n_sc // 2:]
        m_tr = np.isin(Gs, tr_sc)
        torch.manual_seed(r)
        net = M.Head(M.F_PA)
        train_ce(net, Xs[m_tr], Ys[m_tr], EPOCHS)
        ev = eval_policy(spec, nf, net, [scs[i] for i in te_sc])
        win = (base[te_sc].mean() - ev) / base[te_sc].mean()
        wins.append(win)
        print(f"  第 {r + 1} 次{'':<6}{int(m_tr.sum()):>10}{len(te_sc):>12}"
              f"{ev:>15.1f}{win:>9.2%}", flush=True)

    wins = np.array(wins)
    print("  " + "-" * 58)
    print(f"\n  **留出集上赢CDR**：中位 {np.median(wins):.2%}　均值 {wins.mean():.2%}"
          f"　>0 占比 {(wins > 0).mean():.0%}")
    print("\n  对照：单一 `(t,k)` 全 20 场景内挑+评 = 4.69%；留出估计 ≈ 1.4–1.5%")
    print("       逐场景 oracle = 15.37%（**不可部署**）")
    print("\n  判读：")
    print("    **明显赢** ⟹ 信息可表达，病灶在 PPO 估计器（信用分配 / δ 噪声）")
    print("    **赢不了** ⟹ 信息表达不了，病灶在特征 / 策略类 —— **别再调 RL 超参**")


if __name__ == "__main__":
    main()
