# -*- coding: utf-8 -*-
"""**20 个场景的「单步偏离」扫描** —— 量出「最优解与最好 CDR 的差距**至少**多大」。

══ 为什么做这个 ══

§37.32 只在一个场景（`100000`）上扫过。由此我说了「余量约 5%」——**那是错的**：
5% 是 `mean_t(单步偏离收益)`，一个**不可叠加的均值**，推不出任何东西（已撤回）。

**真正站得住的是「最好的那一次单步偏离」**：
992 局（124 决策点 × 8 备选）里最终 makespan 最低的那**一局**。
它是**一条能跑出来的调度方案**，不是聚合量。

```
确定性平局 · 场景 100000：根规则 157.3　最好的一次偏离 130.0
  ⟹ 存在 130.0 的方案 ⟹ **最优解 ≤ 130.0** ⟹ 与CDR 的差距 ≥ **17.4%**
```

**本脚本把这件事在 20 个场景上重做一遍**，报**逐场景**的差距，再汇总。

══ 口径（重要）══

⚠️ **全程确定性平局（`tie_rng=None`）**。理由：随机平局下 `tie_rng` 是整局一条流，
只在出现平局时抽；不同规则触发的平局次数不同 ⟹ **在第 `t` 点换规则会把后面所有决策点的
平局整体移位** ⟹ 测到的不是「改这一个决策」。故基准与偏离**都用确定性平局**，内部自洽。

⚠️ **参照另测**：同时报「确定性平局下的最好 CDR」，**不与 `147.6`（随机平局口径）相减**。
（同规则 20 场景：确定性 153.27 vs 随机 147.59。）

══ 复用 ══

本脚本**不新增判定逻辑** —— `one()` 与 `v3_headroom.py` 读数 3 的同源，只是搬到多进程。
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # 仓根

import numpy as np  # noqa: E402

from plcsp_v3.env.nhdrl_features import F_PA, NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.rules import (N_AEI2025_RULES, PA_RULE_NAMES,  # noqa: E402
                                apply_pa_rule, la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402

ROOT = int(os.environ.get("DS_ROOT", "3"))
N_SC = int(os.environ.get("DS_SC", "20"))
N_WORK = int(os.environ.get("DS_WORK", "6"))
T_MAX = int(os.environ.get("DS_TMAX", "124"))
# ⚠️ **两种口径，是**两族不同的策略**，不是同一件事的两个写法**（2026-10-13）：
#   `point`（原）：**只在第 `t` 个决策点**换成 `k`，之后回 `ROOT`
#     —— 这不是一个策略（124 个局部改动**不可叠加**，§37.32；照它做实测比不切差 12–17%）
#   `tail`：**从第 `t` 个决策点起一直用 `k`**
#     —— **这是一个真正的策略族**，也是 GLM 说的「录前缀、换 `k`、重放到局末」
# 结果分开存盘，互不覆盖。
MODE = os.environ.get("DS_MODE", "point")
NPZ = os.path.join(os.environ.get("TEMP", "/tmp"),
                   "devscan.npz" if MODE == "point" else f"devscan_{MODE}.npz")
_G: dict = {}


def _init() -> None:
    """每个 worker 各建一次世界（SimPy 对象不可跨进程共享）。"""
    inst = named(140, 16)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)
    _G["spec"] = spec
    _G["nf"] = NHFeatures(inst, spec.layout)


def _dev(args: tuple) -> tuple:
    """`(场景, 决策点 t, 备选规则 k)` ⟹ `(场景, t, k, makespan)`。`t=-1` = 不偏离。

    ⚠️ `MODE` 决定「偏离」的含义 —— 见文件头的说明。**两种是两族不同的策略。**
    """
    sc, t_star, k = args
    cnt = {"i": 0}

    def pol(view, pool, tie_rng=None):
        i = cnt["i"]
        cnt["i"] += 1
        use = (i >= t_star) if MODE == "tail" else (i == t_star)
        return apply_pa_rule(k if use else ROOT, view, pool, None)

    ms = float(_G["spec"].world().run_gated(
        seed_chain=sc, policy_pa=pol, policy_l=la_rule_policy(0),
        dt_events=0)["makespan"])
    return sc, t_star, k, ms


def main() -> None:
    scs = [100_000 + 13 * i for i in range(N_SC)]
    alts = [k for k in range(N_AEI2025_RULES) if k != ROOT]
    tasks = [(sc, t, k) for sc in scs for t in range(T_MAX) for k in alts]
    print(f"单步偏离扫描 · {N_SC} 场景 × {T_MAX} 决策点 × {len(alts)} 备选 "
          f"= **{len(tasks)} 局** · 根规则 {PA_RULE_NAMES[ROOT]} · "
          f"确定性平局 · 模式 **{MODE}** · {N_WORK} 进程\n", flush=True)

    base = {}
    with ProcessPoolExecutor(max_workers=N_WORK, initializer=_init) as ex:
        for sc, _t, _k, ms in ex.map(_dev, [(sc, -1, ROOT) for sc in scs]):
            base[sc] = ms
            print(f"  根 makespan · 场景 {sc} = {ms:.1f}", flush=True)
        A = {sc: np.full((T_MAX, N_AEI2025_RULES), np.nan) for sc in scs}
        n = 0
        for sc, t, k, ms in ex.map(_dev, tasks, chunksize=8):
            A[sc][t, k] = ms
            n += 1
            if n % 2000 == 0:
                print(f"    进度 {n}/{len(tasks)}", flush=True)

    print(f"\n  {'场景':>8}{'根(常数)':>10}{'最好一次偏离':>14}{'差距':>9}"
          f"{'改善的点数':>11}   最好那一次是（决策点, 规则）")
    print("  " + "-" * 78)
    rows = []
    for i, sc in enumerate(scs):
        M = A[sc]
        bn = np.nanmin(M, axis=1)                  # 每个决策点上的最好备选
        j = int(np.nanargmin(bn))
        k = alts[int(np.nanargmin(M[j, alts]))]
        gap = (base[sc] - bn.min()) / base[sc]
        rows.append((base[sc], bn.min(), gap, int((bn < base[sc] - 1e-9).sum())))
        print(f"  {i:>8}{base[sc]:>10.1f}{bn.min():>14.1f}{gap:>9.2%}"
              f"{rows[-1][3]:>8}/{T_MAX}   t={j} · {PA_RULE_NAMES[k]}", flush=True)

    R = np.array(rows, dtype=float)
    print("  " + "-" * 78)
    print(f"  {'均值':>8}{R[:, 0].mean():>10.1f}{R[:, 1].mean():>14.1f}"
          f"{R[:, 2].mean():>9.2%}{R[:, 3].mean():>8.1f}/{T_MAX}")
    print(f"  {'中位':>8}{np.median(R[:, 0]):>10.1f}{np.median(R[:, 1]):>14.1f}"
          f"{np.median(R[:, 2]):>9.2%}{np.median(R[:, 3]):>8.1f}/{T_MAX}")
    print(f"\n  **逐场景差距**：最小 {R[:, 2].min():.2%}　最大 {R[:, 2].max():.2%}"
          f"　20 个场景**全部为正**：{bool((R[:, 2] > 0).all())}")

    # ⚠️ **上面那个「最好一次偏离」是**逐场景 oracle** —— 20 条证人各不相同
    # （`t*` 从 0 到 93、`k*` 有 6 种）⟹ **它不构成对「单策略最优值」的界。**
    # 下面这一段才是**单一可部署策略**：所有场景共用同一对 `(t, k)`。
    base_mean = float(np.mean([base[sc] for sc in scs]))
    big = np.stack([A[sc] for sc in scs])              # (20, T, 12)，ROOT 列为 NaN
    print("\n  ══ 单一策略：**所有场景都用同一对 `(t, k)`** ══")
    print(f"  CDR 20 场景均值（本口径）= **{base_mean:.1f}**")
    flat = big.reshape(len(scs), -1)
    tt = np.repeat(np.arange(T_MAX), N_AEI2025_RULES)
    kk = np.tile(np.arange(N_AEI2025_RULES), T_MAX)
    means = np.nanmean(flat, axis=0)
    order = np.argsort(means)
    print("  最好的单一 `(t, k)`（按 20 场景均值升序，前 10）：")
    for o in order[:10]:
        print(f"      t={tt[o]:>3} · {PA_RULE_NAMES[int(kk[o])]:<12}"
              f"{means[o]:>8.1f}　vs 常数 {base_mean:.1f}　"
              f"**{(base_mean - means[o]) / base_mean:+.2%}**")
    best_o = int(order[0])
    win = int(np.sum(means < base_mean - 1e-9))
    print(f"\n  **最好的单一策略**：t={tt[best_o]} · {PA_RULE_NAMES[int(kk[best_o])]} "
          f"⟹ {means[best_o]:.1f}　比CDR好 "
          f"**{(base_mean - means[best_o]) / base_mean:.2%}**")
    print(f"  在全部 {len(means)} 个 `(t, k)` 里，**赢过CDR 的**：{win} 个"
          f"（{win / len(means):.1%}）")

    # ── 存盘：给 `v3_state_probe.py` 用（2480 个带标签样本）──
    # ⚠️ **主进程自己要建一份 spec/nf**（2026-10-13 修）：`_G` 只在 worker 里由
    # `initializer=_init` 填，主进程里是空的 ⟹ 早先两版在这里崩过两次
    # （`KeyError: 'spec'`、`'Spec' object has no attribute 'nf'`）。
    # ⚠️ **`A` 先单独存一次**：特征收集是后加的，它崩了不该把 19840 局的仿真白跑掉。
    out = NPZ
    np.savez_compressed(out, A=big, base=np.array([base[sc] for sc in scs]),
                        scs=np.array(scs))
    print(f"\n  已存 `{out}`（A/base）", flush=True)
    if "spec" not in _G:
        _init()
    X = np.full((len(scs), T_MAX, F_PA), np.nan)
    for i, sc in enumerate(scs):
        rows_f: list = []

        def pol(view, pool, tie_rng=None, _r=rows_f):
            _r.append(np.asarray(_G["nf"].pa(view.snap, 0.0, (5.0, 5.0)),
                                 dtype=float))
            return apply_pa_rule(ROOT, view, pool, None)

        _G["spec"].world().run_gated(seed_chain=sc, policy_pa=pol,
                                     policy_l=la_rule_policy(0), dt_events=0)
        n = min(T_MAX, len(rows_f))
        X[i, :n] = np.stack(rows_f[:n])
    np.savez_compressed(out, A=big, X=X,
                        base=np.array([base[sc] for sc in scs]),
                        scs=np.array(scs))
    print(f"  已补 X 并重存 `{out}`（供 `v3_state_probe.py` 用 2480 样本）")

    print("\n  ⟹ ⚠️ **上行「最好一次偏离」是**逐场景 oracle**，20 条证人各不相同，"
          "**不构成单策略的界**。")
    print("  ⟹ 只有**单一 `(t,k)` 那一行**才是可部署策略的证据。")
    print("\n  ⚠️ 确定性平局口径；**不与 `147.6`（随机平局 20 场景均值）相减**。")


if __name__ == "__main__":
    main()
