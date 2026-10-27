# -*- coding: utf-8 -*-
"""**状态依赖探针**：把「单点偏离的收益」按**状态**分箱。

══ 要判的 ══

`open-question-nhdrl.md` §4.1 量过：**单点族余量中位 +1.51% < 判据噪声 1.8%**
⟹ 「在某一步换一条规则」在我们实例上**没有收益** ⟹ **状态依赖弱**。

**但那是"整体中位数"，没有分箱。** 本探针问的是：

> **那 1.51% 是不是集中在**某些状态**下？还是均匀地摊平在全部决策点上？**

- **集中在某些状态** ⟹ **找到该放大的维度**（那正是"制造状态依赖"该用的杠杆）
- **摊平** ⟹ 确认"状态依赖真的为零"，只能改实例结构（§4.9 设计里的「路径 B」）

══ 口径 ══

**单点族**（照 §37.40）：从 t=0 用**基准规则**跑到第 `t` 个决策点 → 那一点换成规则 `k`
→ 之后**立刻回基准规则**。`ΔMS(t,k) = MS(偏离) − MS(基准)`，**负 = 换过去更好**。

**基准规则** = `BASE_PA` + `BASE_LA`。**每局只偏离一次**（其余全同 ⟹ 干净单变量）。

══ ⚠️ 不变量 ══

- 本探针**只读**，不改仓库任何默认值
- 用的是 `run_gated` + `apply_pa_rule` / `la_rule_policy`，与其它 repro 脚本同源
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from plcsp_v3.env.shi_instances import (N_AGV_SHI, N_ALTS_HI,  # noqa: E402
                                        N_ALTS_LO, named)
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, N_AEI2025_RULES,  # noqa: E402
                                PA_RULE_NAMES, apply_pa_rule, la_rule_policy)
from plcsp_v3.q1_flat import SHI_SHOP_LEN, build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

# ⚠️ **只用原文的 9 条 PA 规则**（`PA_RULE_NAMES` 有 12 条 —— 后 3 条 `EDD+*` 是我们加的，
#    见 `hrl/rules.py:17`）。偏离目标含 `EDD` 就不是原文的 CDR 集合了。
PA9 = PA_RULE_NAMES[:N_AEI2025_RULES]

BASE_PA = os.environ.get("DV_BASE_PA", "EAJ+SMPT")
BASE_LA = os.environ.get("DV_BASE_LA", "LATEST_IN_TIME")
N_SC = int(os.environ.get("DV_NSC", "4"))
SCHEME = os.environ.get("DV_SCHEME", "random")
# ⚠️ **`DV_SHOPLEN`（2026-10-16）**：车间长度 [m]，取代旧的 `DV_SCALE`（无因次倍率）。
SHOPLEN = float(os.environ.get("DV_SHOPLEN", str(SHI_SHOP_LEN)))
SHI = os.environ.get("DV_SHI", "1") == "1"
# ⚠️ **每局偏离的决策点**（等间隔取 `DV_NT` 个）—— 全部 124 个太贵
N_T = int(os.environ.get("DV_NT", "12"))
# ⚠️ **路线专属子集**（`cluster_size`）：该零件类型只能用这几台机。
#    `None` = 不限制（每道工序从全部 18 台里抽）。**原文只说"可用机器固定"，没给哪几台。**
CS = os.environ.get("DV_CLUSTER", "")
CS = int(CS) if CS else None
# ⚠️ **每道工序的候选机台数**（`DV_NALT="lo-hi"`）：原文只说"可用机器固定"，**没给台数**。
#    宽度窄 ⟹ `EAM` 与 `MINU` 几乎总选同一台 ⟹ 规则之间没有可翻转的东西。
NALT = os.environ.get("DV_NALT", "")
ALT_LO, ALT_HI = ((int(x) for x in NALT.split("-")) if NALT
                  else (N_ALTS_LO, N_ALTS_HI))
# ⚠️ **重尾 `α_ijk`**：`DV_NARROW="n_narrow-wide_lo-wide_hi"`，例 `"3-4-8"`。
#    空（默认）⟹ 均匀 `(ALT_LO, ALT_HI)`，**逐位不变**。
NARROW = os.environ.get("DV_NARROW", "")
N_NARROW, WIDE_LO, WIDE_HI = (((int(x) for x in NARROW.split("-")))
                              if NARROW else (0, None, None))
QUIET = os.environ.get("DV_QUIET", "0") == "1"
DIMS = ("pool", "mach_busy", "alts_min", "alts_med")


def probe_state(view, pool) -> dict:
    """一个决策点的**状态特征**（只读视图，纯函数）。"""
    now = view.now
    alts = [len(view.jobs[j].alts) for j, _ in pool]
    return {
        "pool": float(len(pool)),                       # 未完工工件数
        "mach_busy": float(np.mean([m.avail_at > now for m in view.machines])),
        "alts_min": float(min(alts)) if alts else 0.0,  # 最窄的候选集
        "alts_med": float(np.median(alts)) if alts else 0.0,
        "due": float(np.mean([min(view.jobs[j].due - now, 1e6) for j, _ in pool])),
    }


def run(spec, sc, *, dev_pa=None, dev_la=None, at=None, record=False):
    """跑一局。`dev_pa`/`dev_la` 在第 `at` 个 PA 决策点生效，其余用基准规则。"""
    cnt = [0]
    st: list[dict] = []

    def pol_pa(view, pool, tie_rng=None):
        i = cnt[0]
        cnt[0] += 1
        if record:
            st.append(probe_state(view, pool))
        if dev_pa is not None and i == at:
            return apply_pa_rule(dev_pa, view, pool, tie_rng)
        return apply_pa_rule(BASE_PA, view, pool, tie_rng)

    def pol_la(view, cv):
        if dev_la is not None and cnt[0] - 1 == at:
            return int(la_rule_policy(dev_la)(view, cv))
        return int(la_rule_policy(BASE_LA)(view, cv))

    m = spec.world().run_gated(seed_chain=sc, policy_pa=pol_pa,
                               policy_l=pol_la, dt_events=0)
    return float(m["makespan"]), st


def main() -> None:
    inst = named(140, 16, n_alts_lo=ALT_LO, n_alts_hi=ALT_HI, cluster_size=CS,
                 narrow_n=N_NARROW, wide_lo=WIDE_LO, wide_hi=WIDE_HI)
    if N_NARROW:
        from collections import Counter
        wd = sorted(Counter(len(o) for j in inst.jobs for o in j).items())
        print(f"  候选宽度分布：{wd}", flush=True)
    spec = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                 finite_buffer=True, layout_seed=0, reward_norm="minmax",
                 tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst,
                 layout_scheme=SCHEME, **dims_from_shop_len(SHOPLEN, inst.n_machines), shi2025=SHI)[0]
    scs = [100_000 + 13 * i for i in range(N_SC)]
    print(f"状态依赖探针 · 基准 = {BASE_PA} + {BASE_LA} · cluster={CS} · "
          f"α={ALT_LO}-{ALT_HI} · {N_SC} 场景 × {N_T} 个偏离点\n", flush=True)

    recs = []          # (state, kind, name, dms)
    for sc in scs:
        ms0, st = run(spec, sc, record=True)
        n = len(st)
        if n == 0:
            continue
        ats = np.unique(np.linspace(0, n - 1, min(N_T, n)).astype(int))
        for at in ats:
            for k, nm in enumerate(PA9):
                if nm == BASE_PA:
                    continue
                ms, _ = run(spec, sc, dev_pa=k, at=int(at))
                recs.append((st[at], "PA", nm, ms - ms0))
            for j, nm in enumerate(LA_RULE_NAMES):
                if nm == BASE_LA:
                    continue
                ms, _ = run(spec, sc, dev_la=j, at=int(at))
                recs.append((st[at], "LA", nm, ms - ms0))
        if not QUIET:
            print(f"  场景 {sc} 完（{n} 个决策点）基准 MS {ms0:.1f}", flush=True)

    dms = np.asarray([r[3] for r in recs])
    print(f"\n  总样本 {len(recs)} 个 (t, k) 对")
    print(f"  ΔMS：中位 {np.median(dms):+8.2f}　均值 {dms.mean():+8.2f}　"
          f"<0（换过去更好）的占比 {(dms < 0).mean():.1%}")
    print("  ⚠️ **上两行只是背景** —— 它们把「规则」这一维**坍缩**掉了，"
          "正好会把要找的东西抹平。判据在下面。")

    # ══ 正确的判据 ══
    # ⚠️ **`<0 占比 ≈ 50%` 不是"状态依赖强"，是"规则之间无法区分"。**
    #    要判的是：**各状态箱里，哪条规则最好（argmin）是否不同。**
    #    故必须按 **(规则, 箱)** 二维拆开，看每箱的 argmin。
    # ⚠️ **判据**：要分清两件事 ——
    #    ① **基准选错**：同一条替代规则在**所有**箱都好（那不是状态依赖，换基准就好）
    #    ② **真翻转**：**不同**箱里**不同**规则最好（**那才是状态依赖**）
    #    ⟹ 必须把**每条规则在各箱的 mean** 并排打出来，看**排名是否互换**。
    all_names = sorted({r[2] for r in recs},
                       key=lambda nm: float(np.mean([r[3] for r in recs if r[2] == nm])))
    print("\n  ═══ 每条替代规则在各箱的 mean ΔMS（**负 = 比基准好**）═══")
    print("  列按**总体** mean 从好到坏排；`*` = 该箱的 argmin")
    argmins = {}
    for dim in DIMS:
        v = np.asarray([r[0][dim] for r in recs])
        q = np.quantile(v, [0, .25, .5, .75, 1.0])
        print(f"\n  ── 按 `{dim}` 分四分位 ──")
        print(f"    {'箱':<7}{'n':>5}　{'（基准）':>8}"
              + "".join(f"{nm:>9}" for nm in all_names))
        for b in range(4):
            lo, hi = q[b], q[b + 1]
            m = (v >= lo) & (v <= hi if b == 3 else v < hi)
            if m.sum() < 30:
                continue
            per: dict[str, list[float]] = {}
            for r_, keep in zip(recs, m):
                if keep:
                    per.setdefault(r_[2], []).append(r_[3])
            means = {nm: float(np.mean(x)) for nm, x in per.items()}
            means["（基准）"] = 0.0          # 基准自己必进候选（ΔMS 恒 0）
            best = min(means, key=lambda nm: means[nm])
            argmins[(dim, b)] = best
            cells = "".join(
                (f"{means[nm]:>+8.1f}" + ("*" if nm == best else " "))
                if nm in means else f"{'—':>9}" for nm in all_names)
            print(f"    Q{b+1:<6}{int(m.sum()):>5}　"
                  f"{'+0.0' + ('*' if best == '（基准）' else ' '):>8}{cells}")

    # ⚠️ **判据**：各箱的 argmin 若**不同** ⟹ 状态依赖存在。
    #    **全部都是「（基准）」** ⟹ 基准在所有状态下都最优 ⟹ 状态依赖为零。
    uniq = sorted(set(argmins.values()))
    print(f"\n  **SUMMARY** cluster={CS} α={ALT_LO}-{ALT_HI}　n={len(recs)}")
    print(f"    **各箱 argmin 的取值集合 = {uniq}**（共 {len(argmins)} 箱）")
    if uniq == ["（基准）"]:
        print("    ⟹ **每一箱的 argmin 都是「基准」** "
              "⟹ 基准在所有状态下最优 ⟹ **状态依赖为零**")
    else:
        n_base = sum(1 for x in argmins.values() if x == "（基准）")
        print(f"    ⟹ **{len(argmins) - n_base}/{len(argmins)} 箱的 argmin 不是基准** "
              f"⟹ **状态依赖存在**（那些状态下换规则更好）")


if __name__ == "__main__":
    main()
