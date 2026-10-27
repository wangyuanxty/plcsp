# -*- coding: utf-8 -*-
"""**「该用哪条规则」这条信息，在不在 24 维状态里？**

══ 为什么问这个 ══

已确立（§37.32）：**余量存在（≥5%）、最优规则逐点变（8 条备选全都在某些点上最好）**。
加上 §37.31.1（奖励有动作信号）与 §37.31.2（特征局内会动）——
**四件事都对了，而 K=9 的策略还是输给常数（157.9 vs 147.6）。**

**⟹ 只剩两种可能，而现有读分数不出来：**

| 可能 | 含义 | 下一步 |
|---|---|---|
| **A. 信息在状态里，学习器学不会** | 病灶在 **PPO / 样本 / 超参** | 改学习器 |
| **B. 信息不在状态里** | 病灶在 **24 维特征**（没编进决定「哪条规则好」的量） | 改特征 |

**本脚本一刀切开 A 与 B，而且不经过 PPO、不经过任何补丁。**

══ 测法 ══

读数 3（`v3_headroom.py`）已经给出**答案纸**：根轨迹（`EAJ+SMPT`）上每个决策点 `t`，
逐条换规则跑完，**哪条规则最终 makespan 最好**。

缺的是那些 `t` 上的 **24 维特征**。两边拼起来 ⟹ 一份监督数据集 `(特征 → 该用哪条规则)`。

**判据：拿分类器的交叉验证准确率（按场景分组，防泄漏）比「多数类基线」。**

- **明显高于基线** ⟹ **信息在状态里（A）** ⟹ 策略学不会，病灶在学习器。
- **≈ 基线** ⟹ **信息不在状态里（B）** ⟹ 病灶在特征。

⚠️ **两个对照必须有，否则读数不可解读：**
1. **多数类基线** —— 标签若是「八成都是同一条规则」，那 80% 准确率什么也没说明。
2. **只用 `t`（决策点序号）当输入的基线** —— 若它也能达到同样准确率，
   那「状态依赖」其实是「**时间依赖**」，特征的功劳没那么大。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # 仓根

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_score  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from plcsp_v3.env.nhdrl_features import NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.rules import (N_AEI2025_RULES, PA_RULE_NAMES,  # noqa: E402
                                apply_pa_rule, la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402

ROOT = 3                                   # EAJ+SMPT（20 场景实测最好）
N_SC = int(os.environ.get("SP_SC", "3"))
STRIDE = int(os.environ.get("SP_STRIDE", "2"))


def collect(spec, nf, sc: int) -> np.ndarray:
    """跑根轨迹：收**每个决策点的 24 维特征**，返回 `[决策点数, 24]`。

    ⚠️ **确定性平局**（`tie_rng=None`）—— 与 `v3_headroom.py` 读数 3 的干净版同口径。
    """
    rows: list[np.ndarray] = []

    def pol(view, pool, tie_rng=None):
        rows.append(np.asarray(nf.pa(view.snap, 0.0, (5.0, 5.0)), dtype=float))
        return apply_pa_rule(ROOT, view, pool, None)

    spec.world().run_gated(seed_chain=sc, policy_pa=pol,
                           policy_l=la_rule_policy(0), dt_events=0)
    return np.stack(rows)


def deviation(spec, sc: int, t_star: int, alt: int) -> float:
    """在第 `t_star` 个决策点只换 `alt`，其余用 `ROOT`。"""
    cnt = {"i": 0}

    def pol(view, pool, tie_rng=None):
        i = cnt["i"]
        cnt["i"] += 1
        return apply_pa_rule(alt if i == t_star else ROOT, view, pool, None)

    return float(spec.world().run_gated(seed_chain=sc, policy_pa=pol,
                                        policy_l=la_rule_policy(0),
                                        dt_events=0)["makespan"])


def main() -> None:
    inst = named(140, 16)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    nf = NHFeatures(inst, spec.layout)
    scs = [100_000 + 13 * i for i in range(N_SC)]
    alts = [k for k in range(N_AEI2025_RULES) if k != ROOT]
    print(f"「该用哪条规则」在不在状态里 · Exp(16/140) · {N_SC} 场景 · "
          f"每 {STRIDE} 步取一点 · 确定性平局 · 根规则 {PA_RULE_NAMES[ROOT]}\n", flush=True)

    # ⚠️ `SP_NPZ` 给定 ⟹ **直接读 `v3_deviation_scan.py` 存下的数据**（2480 个样本），
    # 不再重跑仿真。186 样本那版**不可信**（低于多数类基线 = 过拟合的签名，样本不够）。
    npz = os.environ.get("SP_NPZ", "")
    X, Y, G, T, P = [], [], [], [], []
    if npz:
        d = np.load(npz)
        A, XA, scs = d["A"], d["X"], d["scs"]           # A:(N,T,12) X:(N,T,24)
        for i in range(A.shape[0]):
            for t in range(A.shape[1]):
                row = A[i, t, alts]
                if not np.isfinite(row).all() or not np.isfinite(XA[i, t]).all():
                    continue
                s = np.sort(row)
                X.append(XA[i, t])
                Y.append(alts[int(np.argmin(row))])
                G.append(i)
                T.append(t / A.shape[1])
                P.append(float(s[1] - s[0]))            # 最好 → 次好（标签可分性）
        P = np.asarray(P)
        print(f"  从 `{npz}` 读入：{len(Y)} 个样本 · {len(scs)} 个场景\n", flush=True)

        # ── ⚠️ **标签可分性对照**（2480 样本那批加的）──
        # 若每个 `(场景, t)` 上 8 条备选的 makespan **彼此很接近**，
        # 那「哪条最好」就是**掷硬币** ⟹ 分类器测不出信息是**当然的**，
        # **不能据此判「特征里没信息」**。没有这一栏，上面那张准确率表读不出结论。
        g12, spr = [], []
        for i in range(A.shape[0]):
            for t in range(A.shape[1]):
                row = A[i, t, alts]
                if np.isfinite(row).all():
                    s = np.sort(row)
                    g12.append(s[1] - s[0])          # 最好 → 次好
                    spr.append(s[-1] - s[0])         # 最好 → 最差
        g12, spr = np.asarray(g12), np.asarray(spr)
        print("  ⚠️ **标签可分性**（该不该信上面那张表，取决于这一栏）：")
        print(f"    每个 (场景,t) 上 8 条备选的 **最好→次好** 差："
              f"中位 {np.median(g12):.2f} 分钟")
        for q in (0.5, 1.0, 2.0, 5.0):
            print(f"      差 < {q:>4.1f} 分钟的占比："
                  f"**{(g12 < q).mean():.1%}**")
        print(f"    最好→最差 的差（全部 8 条）：中位 {np.median(spr):.2f} 分钟")
        # ⚠️ **并列是致命的**：`np.argmin` 并列时返回**第一个下标** ⟹ 标签被**下标顺序**
        # 决定，不是被「哪条更好」决定。**证据**：标签里最多的是 `SPT+SMPT`（下标 0）。
        # 所以下面这张表必须**按「唯一最优 + 差距够大」筛过**才可读。
        print("\n  ⚠️ **按标签质量分层**（只有前几行是「信息论上可学」的）：")
        for name, keep in (
            ("全部样本", np.ones(len(g12), dtype=bool)),
            ("唯一最优（无并列）", g12 > 1e-9),
            ("唯一最优且差 ≥ 1 分钟", g12 >= 1.0),
            ("唯一最优且差 ≥ 5 分钟", g12 >= 5.0),
        ):
            print(f"    {name:<24}保留 {keep.sum():>5} / {len(keep)}"
                  f"　（{keep.mean():>5.1%}）")
        print("    ⟹ **75%+ 的点上「最好」与「次好」并列 ⟹ 用 argmin 造标签 = 造随机标签。**")
        print("    ⟹ 上面那张准确率表**只该看「唯一最优」那几行**。\n", flush=True)
    else:
        for si, sc in enumerate(scs):
            F = collect(spec, nf, sc)
            L = len(F)
            ts = list(range(0, L, STRIDE))
            base = deviation(spec, sc, -1, ROOT)
            for t in ts:
                ms = np.array([deviation(spec, sc, t, k) for k in alts])
                X.append(F[t])
                Y.append(alts[int(ms.argmin())])
                G.append(si)
                T.append(t / L)
            print(f"  场景 {si + 1}/{N_SC}　{sc}　决策点 {L}　取 {len(ts)} 点"
                  f"　根 makespan {base:.1f}", flush=True)

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y)
    G = np.asarray(G)
    T = np.asarray(T).reshape(-1, 1)
    cnt = np.bincount(Y, minlength=N_AEI2025_RULES)
    maj = cnt.max() / len(Y)
    print(f"\n  样本 {len(Y)}　特征 {X.shape[1]} 维　标签 {len(cnt[cnt > 0])} 类")
    print("  标签分布：" + "　".join(f"{PA_RULE_NAMES[k]}×{cnt[k]}"
                                      for k in np.argsort(-cnt) if cnt[k] > 0))
    print(f"\n  **多数类基线 {maj:.1%}**（全押同一类）\n", flush=True)

    cv = GroupKFold(n_splits=min(N_SC, 5))
    cands = {
        "只用决策点序号 t": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "24 维特征 · 逻辑回归": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "24 维特征 · MLP(64,)": make_pipeline(
            StandardScaler(), MLPClassifier((64,), max_iter=3000, random_state=0)),
    }
    print(f"  {'输入':<26}{'交叉验证准确率':>14}")
    print("  " + "-" * 42)
    for name, clf in cands.items():
        Feat = T if name.startswith("只用") else X
        acc = cross_val_score(clf, Feat, Y, groups=G, cv=cv, scoring="accuracy")
        print(f"  {name:<26}{acc.mean():>13.1%}　（±{acc.std():.3f}）", flush=True)

    # ⚠️ **上表在「全部样本」上算，而其中 68% 的标签是被 `argmin` 的下标顺序决定的
    # 随机标签（并列）⟹ 那张表**不可读**。下面按标签质量分层重做。**
    if npz and P.size:
        print("\n  ══ 按标签质量分层（**只有前几行可读**）══")
        print(f"  {'层':<22}{'样本':>7}{'多数类':>9}{'24维·LR':>10}{'只用 t':>9}")
        for nm, keep in (("全部（含并列）", np.ones(len(Y), dtype=bool)),
                         ("唯一最优", P > 1e-9),
                         ("唯一最优 · 差≥1min", P >= 1.0),
                         ("唯一最优 · 差≥5min", P >= 5.0)):
            if keep.sum() < 40:
                print(f"  {nm:<22}{int(keep.sum()):>7}　样本太少，跳过")
                continue
            Xs, Ys, Gs, Ts = X[keep], np.asarray(Y)[keep], np.asarray(G)[keep], \
                T[keep] if isinstance(T, np.ndarray) else np.asarray(T)[keep]
            c = np.bincount(Ys, minlength=N_AEI2025_RULES)
            maj = c.max() / len(Ys)
            ng = len(set(Gs.tolist()))
            cvk = GroupKFold(n_splits=min(ng, 5))
            a1 = cross_val_score(cands["24 维特征 · 逻辑回归"],
                                 Xs, Ys, groups=Gs, cv=cvk, scoring="accuracy")
            a2 = cross_val_score(cands["只用决策点序号 t"],
                                 Ts, Ys, groups=Gs, cv=cvk, scoring="accuracy")
            print(f"  {nm:<22}{len(Ys):>7}{maj:>9.1%}"
                  f"{a1.mean():>10.1%}{a2.mean():>9.1%}", flush=True)
        print("  ⟹ **判据**：看「唯一最优」那两行 —— 24 维特征是否**明显高于多数类**。")
        print("  ⟹ 「全部（含并列）」那一行**不要读**，它是随机标签。")

    # ⚠️ **状态好不好分？**（2026-10-13，回答「是不是状态很难分」）
    # 在**同一场景内**找特征空间里**离得最近**的另一个决策点，看两者标签一不一致。
    # 若「最近的邻居」都对不上，那**不是分类器的错，是状态本身分不开**。
    # ⚠️ 只在「唯一最优」层上做 —— 并列的标签是被下标决定的（见上）。
    if npz and P.size:
        keep = P > 1e-9
        Xk, Yk, Gk = X[keep], np.asarray(Y)[keep], np.asarray(G)[keep]
        mu, sd = Xk.mean(axis=0), Xk.std(axis=0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        Z = (Xk - mu) / sd
        nn_d, nn_ok = [], []
        for g in np.unique(Gk):
            m = Gk == g
            if m.sum() < 3:
                continue
            A_ = Z[m]
            Y_ = Yk[m]
            dmat = np.sqrt(((A_[:, None, :] - A_[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(dmat, np.inf)
            j = dmat.argmin(axis=1)
            nn_d += list(dmat[np.arange(len(j)), j])
            nn_ok += list(Y_ == Y_[j])
        nn_d = np.asarray(nn_d)
        nn_ok = np.asarray(nn_ok)
        p = np.bincount(Yk) / len(Yk)
        chance = float((p ** 2).sum())
        print("\n  ⚠️ **状态好不好分**（同场景内最近邻，`唯一最优` 层）：")
        print(f"    最近邻距离（标准化后）：中位 **{np.median(nn_d):.3f}**"
              f"　10% 分位 {np.percentile(nn_d, 10):.3f}"
              f"　90% 分位 {np.percentile(nn_d, 90):.3f}")
        print(f"    最近邻的**标签一致率：{nn_ok.mean():.1%}**"
              f"　（随机配对 = {chance:.1%}）")
        print("    ⟹ 一致率 ≈ 随机 ⟹ **状态分不开**（不是分类器的错）")
        print("    ⟹ 一致率 ≫ 随机 ⟹ 状态能分，问题在**全局泛化**而非局部可分")

    print("\n  ⚠️ 判据：")
    print(f"    「24 维特征」**明显高于**多数类基线 {maj:.1%} ⟹ **信息在状态里（A）**，病灶在学习器")
    print("    「24 维特征」**≈** 基线 ⟹ **信息不在状态里（B）**，病灶在特征")
    print("    另看「只用序号 t」那一行：若它也能达到同样准确率 ⟹ 那是**时间依赖**，不是状态依赖")


if __name__ == "__main__":
    main()
