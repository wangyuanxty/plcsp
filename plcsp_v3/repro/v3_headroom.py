# -*- coding: utf-8 -*-
"""**动作到底有没有用**：两条读数，都不经过网络。

══ 为什么测这个 ══

K 阶梯（`shi2025_min2.py`）读出：动作 2 → 9 条，选对率 100% → 38.4%。
我一直在解释成「学得慢」。**但那个解释没有证据**：它默认奖励里**带着**动作信号，
只是网络没学到。**这个脚本把这个前提本身拿去测。**

══ 两条读数 ══

1. **`cm` 轨迹分岔** —— 9 条规则跑**同一个场景**，逐步记 `cm(t)`。
   `cm` 就是奖励里的那个量（`reward = −(cm(t) − cm(t−1))`）。
   - 若 9 条轨迹**前段完全重合** ⟹ 前段每步的奖励**与动作无关** ⟹ 那些步只贡献噪声。
   - 分岔点越晚，**有信号的决策点占比越低** ⟹ 需要的样本越多。

2. **⚠️ 已作废（2026-10-13，§37.31.3 自纠）：9 × 20 配对矩阵。**
   那 20 个「场景」只是**同一个实例**的 20 次**机障抽样**（`seed_chain` = 故障实现）
   ⟹ **同一个问题问 20 遍，答案当然一样**。它是**循环论证**：
   - 它只约束「先知道整个场景再挑一条规则」这类策略（**是下界，不是天花板**）；
   - **而派工策略按**决策点**选动作，根本不在这条轴上。**
   **保留只为记录这个错**；**不要读它的结论**。
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # 仓根

import numpy as np  # noqa: E402

from plcsp_v3.env.nhdrl_features import NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.rules import (N_AEI2025_RULES, N_PA_RULES,  # noqa: E402
                                PA_RULE_NAMES, apply_pa_rule, la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402

N_SC = int(os.environ.get("ROOM_SC", "20"))


def cm_of(snap) -> float:
    """**与训练里用的一模一样的量**（`shi2025_min2.RollK.policy_pa`）。"""
    return float(snap.now) + max(
        (float(m.remaining_min) + float(m.backlog_min) for m in snap.machines),
        default=0.0)


def trace_rule(spec, k: int, sc: int, trace=None) -> float:
    """跑一条固定规则；`trace` 给定时逐步记 `cm`。"""
    def pol(view, pool, tie_rng=None):
        if trace is not None:
            trace.append(cm_of(view.snap))
        return apply_pa_rule(k, view, pool, tie_rng)

    return float(spec.world().run_gated(seed_chain=sc, policy_pa=pol,
                                        policy_l=la_rule_policy(0),
                                        dt_events=0)["makespan"])


def part_a(spec) -> None:
    """读数 1：9 条规则的 `cm` 轨迹。"""
    sc = 100_000
    tr = {}
    for k in range(N_PA_RULES):
        t: list = []
        trace_rule(spec, k, sc, t)
        tr[k] = np.asarray(t)
    L = min(len(v) for v in tr.values())
    T = np.stack([tr[k][:L] for k in range(N_PA_RULES)])       # 9 × L
    print(f"\n  【读数 1】`cm` 轨迹 · 场景 {sc} · 取公共前 {L} 个决策点")
    print("  各规则决策点总数：" + "　".join(f"{PA_RULE_NAMES[k]}:{len(tr[k])}"
                                              for k in range(N_PA_RULES)))
    print("\n  决策点 t　　cm 的最小/最大（9 条规则）　　极差/均值")
    for t in [0, 1, 2, 5, 10, L // 4, L // 2, 3 * L // 4, L - 1]:
        col = T[:, t]
        m = float(col.mean())
        print(f"  {t:>8}　　{m:>8.1f} ~ {col.max():>8.1f}　　"
              f"{(col.max() - col.min()) / m:>8.2%}")
    # ⭐ 关键：逐决策点看「9 条规则给出同一个 cm 的比例」
    same = np.mean([np.allclose(T[:, t], T[0, t]) for t in range(L)])
    rng = (T.max(axis=0) - T.min(axis=0)) / T.mean(axis=0)
    print(f"\n  **前 {L} 个决策点里，9 条规则的 cm 数值相同的比例：{same:.1%}**")
    print(f"  极差/均值：中位数 {np.median(rng):.2%}　"
          f"前 1/4 段中位数 {np.median(rng[:L // 4]):.2%}　"
          f"后 1/4 段中位数 {np.median(rng[3 * L // 4:]):.2%}")


def part_b(spec, scs) -> None:
    """⚠️ **读数 2：已作废**（循环论证，见文件头）。保留只为记录这个错。"""
    M = np.zeros((N_PA_RULES, len(scs)))
    for k in range(N_PA_RULES):
        for i, sc in enumerate(scs):
            M[k, i] = trace_rule(spec, k, sc)
        print(f"  参照 {k + 1}/{N_PA_RULES}　{PA_RULE_NAMES[k]:<12}"
              f"{M[k].mean():>8.1f}", flush=True)
    glob = M.mean(axis=1)
    g = int(glob.argmin())
    per_sc = M.min(axis=0)
    d = glob[g] - per_sc                                   # 配对差（正 = oracle 更好）
    print(f"\n  【读数 2】9 × {len(scs)} 配对矩阵")
    print(f"  全局最好 CDR {PA_RULE_NAMES[g]}　{glob[g]:.1f}")
    print(f"  逐场景 oracle（每场景挑**该场景**最好的规则）　{per_sc.mean():.1f}")
    print(f"  **配对差 {d.mean():.2f}　= {d.mean() / glob[g]:.2%}**　"
          f"（逐场景差的标准差 {d.std(ddof=1):.2f}　标准误 "
          f"{d.std(ddof=1) / np.sqrt(len(scs)):.2f}）")
    cnt = Counter(int(np.argmin(M[:, i])) for i in range(len(scs)))
    print("  逐场景最好规则："
          + "　".join(f"{PA_RULE_NAMES[k]}×{v}" for k, v in cnt.most_common()))
    print(f"\n  逐场景极差（9 条规则 max−min）："
          f"中位数 {np.median(M.max(0) - M.min(0)):.1f}　"
          f"最小 {(M.max(0) - M.min(0)).min():.1f}")
    print("\n  场景　" + "".join(f"{PA_RULE_NAMES[k][:9]:>11}"
                                  for k in np.argsort(glob)))
    for i in range(len(scs)):
        print(f"  {i:>4}　" + "".join(f"{M[k, i]:>11.1f}" for k in np.argsort(glob)))


def part_c(spec, sc: int = 100_000, root: int = 3) -> None:
    """读数 3（**正确的轴**）：**逐个决策点**的单步偏离。

    ⚠️ **读数 2 是循环论证**（2026-10-13 自纠）：那 20 个「场景」只是**同一个实例**的
    20 次机障抽样（`seed_chain` = 故障实现，见 `q1_flat.py:259`）。同一个问题，
    最好规则当然一样。**它只约束「先知道整个场景再挑一条规则」这类策略**，
    **不是天花板，是下界**，而且压根不是派工策略所在的那条轴。

    **本读数才是对的那条轴**：`t` = 决策点序号。在第 `t` 个决策点**只换一条规则**、
    之后恢复根规则，看最终 makespan。若「最好的那条规则」随 `t` 变 ⟹ **有状态依赖**。

    ⚠️ **它是**松**上界**：只能单步偏离、不能叠加，故「逐点取最好」的组合不可实现。
    但**若连这个松上界都≈ 0，就真没有余量**；若很大，则**策略有活可干**。
    """
    n_alt = [k for k in range(N_AEI2025_RULES) if k != root]
    ts = list(range(int(os.environ.get("ROOM_T0", "0")),
                    int(os.environ.get("ROOM_T1", "124"))))
    # ⚠️ **必须钉死平局**（2026-10-13）：`tie_rng` 是**整局一条流**，只在出现平局时抽。
    # 不同规则触发的平局**次数不同** ⟹ 在第 `t` 点换规则会把**后面所有决策点的平局整体移位**
    # ⟹ 测到的就不是「改这一个决策」。实测平局分量与效果同量级（同规则 153.27 vs 147.59，20 场景均值）。
    # `ROOM_TIE=det` ⟹ 两边都传 `None`（确定性平局），偏离只改一个决策。
    det = os.environ.get("ROOM_TIE", "rand") == "det"

    def one(t_star: int, alt: int) -> float:
        cnt = {"i": 0}

        def pol(view, pool, tie_rng=None):
            i = cnt["i"]
            cnt["i"] += 1
            return apply_pa_rule(alt if i == t_star else root, view, pool,
                                 None if det else tie_rng)

        return float(spec.world().run_gated(seed_chain=sc, policy_pa=pol,
                                            policy_l=la_rule_policy(0),
                                            dt_events=0)["makespan"])

    base = one(-1, root)
    print(f"\n  【读数 3】**逐决策点**单步偏离 · 场景 {sc} · 根规则 "
          f"{PA_RULE_NAMES[root]} · 平局 **{'确定性' if det else '环境随机'}**"
          f" · 根 makespan **{base:.1f}**")
    print(f"  共 {len(ts)} 个决策点 × {len(n_alt)} 条备选 = "
          f"{len(ts) * len(n_alt)} 局\n", flush=True)

    A = np.full((len(ts), N_AEI2025_RULES), np.nan)
    for a, t in enumerate(ts):
        for k in n_alt:
            A[a, k] = one(t, k)
        if (a + 1) % 20 == 0:
            print(f"    t 进度 {a + 1}/{len(ts)}", flush=True)

    best_gain = np.nanmin(A, axis=1) - base          # 负 = 比根规则好
    best_k = np.array([n_alt[int(np.nanargmin(A[a, n_alt]))] for a in range(len(ts))])
    print("\n  逐决策点：换了以后最好能**比根规则好多少**（分钟，负 = 更好）")
    print(f"    **改善的决策点数：{int((best_gain < -1e-9).sum())} / {len(ts)}**"
          f"　（相等 {int((np.abs(best_gain) <= 1e-9).sum())}）"
          f"　**变差的：{int((best_gain > 1e-9).sum())}**")
    print(f"    改善量：中位数 {np.median(best_gain[best_gain < 0]) if (best_gain < 0).any() else 0:.1f}"
          f"　最好 {best_gain.min():.1f}　最坏 {best_gain.max():.1f}")
    print("    逐点最好规则分布："
          + "　".join(f"{PA_RULE_NAMES[k]}×{int((best_k == k).sum())}"
                      for k in n_alt if (best_k == k).any()))
    print(f"\n    **松上界**：若逐点都能取到最好的那条 ⟹ "
          f"{base + best_gain.mean():.1f}（比根规则好 "
          f"{best_gain.mean() / base:.2%}）")
    print("    ⚠️ 该值是**上界**（单步偏离不可叠加）；但它若≈ 0，则**确实没有余量**。")


def main() -> None:
    inst = named(140, 16)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    nf = NHFeatures(inst, spec.layout)          # noqa: F841  （保持与训练同构）
    scs = [100_000 + 13 * i for i in range(N_SC)]
    print(f"动作有没有用 · Exp(16/140) · {N_SC} 场景 · 9 条规则"
          f"　**不经过网络、不训练**")
    if os.environ.get("ROOM_ONLY", "c") == "c":     # 只跑读数 3（1/2 已出读数）
        part_c(spec)
        return
    part_a(spec)
    part_b(spec, scs)
    part_c(spec)


if __name__ == "__main__":
    main()
