# -*- coding: utf-8 -*-
"""**`EAJ` 的替代口径** —— 原文唯一没定义的那条规则。

══ 为什么查它 ══

`hrl/rules.py:29-31` 自己登记：

> `EAJ` —— **可开工时刻**最早：`max(作业就绪时刻, min_{候选机台} 机台空闲时刻)`
> ⚠️ **存疑，未核**：卡里只写"能被最早加工的工件"。**"作业就绪时刻"他们用没用、怎么算，
> 卡里读不出来。**

**原文逐字**（Table 3 Rule 4）：*"EAJ (select the job that can be processed earliest)"* —— 五个词。

**⚠️ 而它同时是我们 27 条里排名第 3 的 PA 工件规则**（`EAJ+SMPT+LATEST`）。
**若我们的 `EAJ` 实现得比原文强，那 9 条 `EAJ` 系 CDR 就整体偏强 ⟹ CDR 集合偏强。**

**可变处很具体**：`ready_at` 里那一段运输 —— `estimate_ready_at` 三支各自加的
`min_{候选机台} leg(src → 该机台装卸点)`。

| 口径 | 工件侧的量 |
|---|---|
| **`orig`**（本仓现值） | `ready_at` = `avail + min leg`（**可开工**） |
| **`alt`** | `transportable_at` = `avail`（**可运输**，不含运输段） |

⚠️ **只读探针**：规则在本地重实现，**不改 `hrl/rules.py`、不动 v3 主线**。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import v3_hv3 as H  # noqa: E402
from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.mo_metrics import (pareto_front,  # noqa: E402
                                     normalize_minmax, hypervolume_2d, REF_NORM)
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, N_AEI2025_RULES,  # noqa: E402
                                PA_JOB_RULES, PA_MACHINE_RULES,
                                la_rule_policy, pa_machine_score)
from plcsp_v3.q1_flat import SHI_SHOP_LEN, build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

N_SC = int(os.environ.get("EA_NSC", "20"))
ALT = os.environ.get("EA_ALT", "1") == "1"       # 是否跑替代口径


def job_key(jr: str, view, j: int, alt: bool) -> float:
    """工件侧打分 —— `alt=True` 时 `EAJ` 换用**不含运输段**的口径。"""
    jv = view.jobs[j]
    if jr == "SPT":
        return float(jv.apt)
    if jr == "SRPT":
        return float(jv.rem_apt)
    if jr == "EDD":
        return float(jv.due)
    if jr == "EAJ":
        base = jv.transportable_at if alt else jv.ready_at
        return float(max(base, min(view.machines[m].avail_at for m, _t in jv.alts))
                     if jv.alts else base)
    raise ValueError(jr)


def make_policy(jr: str, mr: str, alt: bool):
    def pol(view, pool, tie_rng=None):
        ks = [job_key(jr, view, j, alt) for j, _oi in pool]
        best = min(ks)
        cand = [i for i, k in enumerate(ks) if k == best]
        pi = (int(cand[int(tie_rng.integers(len(cand)))]) if tie_rng is not None
              else cand[0])
        alts = view.jobs[pool[pi][0]].alts
        ms = [pa_machine_score(mr, view, mt) for mt in alts]
        bm = min(ms)
        cm = [i for i, k in enumerate(ms) if k == bm]
        mi = (int(cm[int(tie_rng.integers(len(cm)))]) if tie_rng is not None
              else cm[0])
        return int(pi), int(alts[mi][0])
    return pol


def run_cdr(spec, jr, mr, la, scs, alt):
    pol = make_policy(jr, mr, alt)
    pol_l = la_rule_policy(la)
    rows = [spec.world().run_gated(seed_chain=sc, policy_pa=pol,
                                   policy_l=pol_l, dt_events=0) for sc in scs]
    return (float(np.mean([float(r["makespan"]) for r in rows])),
            float(np.mean([float(r["logistics_cost"]) for r in rows])))


def run_rn(spec, i, scs, alt):
    rng = np.random.default_rng([H.SC_FIX, 99, i])

    def pol_pa(view, pool, tie_rng=None):
        k = int(rng.integers(0, N_AEI2025_RULES))
        return make_policy(PA_JOB_RULES[k // 3],
                           PA_MACHINE_RULES[k % 3], alt)(view, pool, tie_rng)

    rows = [spec.world().run_gated(
        seed_chain=H.SC_FIX, policy_pa=pol_pa,
        policy_l=lambda v, cv: int(la_rule_policy(int(rng.integers(0, 3)))(v, cv)),
        dt_events=0) for _ in [0]]
    return (float(rows[0]["makespan"]), float(rows[0]["logistics_cost"]))


def main() -> None:
    inst = named(140, 16)
    spec = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                 finite_buffer=True, layout_seed=0, reward_norm="minmax",
                 tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst,
                 layout_scheme="fig4", **dims_from_shop_len(SHI_SHOP_LEN, inst.n_machines), shi2025=True)[0]
    scs = [100_000 + 13 * i for i in range(N_SC)]

    for alt in ([False, True] if ALT else [False]):
        tag = "**alt（`EAJ` 不含运输段）**" if alt else "orig（本仓现值）"
        pts, tg = [], []
        for ji, jr in enumerate(PA_JOB_RULES[:3]):
            for mi, mr in enumerate(PA_MACHINE_RULES):
                for li in range(len(LA_RULE_NAMES)):
                    pts.append(run_cdr(spec, jr, mr, li, scs, alt))
                    tg.append((jr, mr, li))
        pts = np.asarray(pts)
        rn = np.asarray([run_rn(spec, i, scs, alt) for i in range(N_SC)])
        normed, (lo, hi) = normalize_minmax([rn, pts])
        hv_rn = hypervolume_2d(pareto_front(normed[0]), REF_NORM)
        hv_c = np.asarray([hypervolume_2d(pareto_front(P), REF_NORM)
                           for P in normed[1:]])
        b = int(np.argmax(hv_c))
        ms_all = np.asarray([p[0] for p in pts])
        bms = int(ms_all.argmin())
        print(f"\n═══ {tag} ═══")
        print(f"  RN  MS {rn[:,0].mean():.1f}　HV {hv_rn:.4f}")
        print(f"  CDR MS 最小：{tg[bms][0]}+{tg[bms][1]}+{LA_RULE_NAMES[tg[bms][2]]}"
              f"　({pts[bms,0]:.1f}, {pts[bms,1]:.1f})")
        print(f"  CDR HV 最大：{tg[b][0]}+{tg[b][1]}+{LA_RULE_NAMES[tg[b][2]]}"
              f"　HV {hv_c[b]:.4f}")
        print(f"  **RN/CDR（HV）= {hv_rn / hv_c[b]:.2f}×**"
              f"　RN_MS/CDR_MS = {rn[:,0].mean() / pts[bms,0]:.2f}")
        # `EAJ` 系 vs 其余两系 的 MS 均值
        eaj = np.mean([p[0] for p, t in zip(pts, tg) if t[0] == "EAJ"])
        oth = np.mean([p[0] for p, t in zip(pts, tg) if t[0] != "EAJ"])
        print(f"  `EAJ` 系 9 条的 MS 均值 {eaj:.1f}　vs 其余 18 条 {oth:.1f}"
              f"　（差 {eaj - oth:+.1f}）")
        print(f"  归一化基准：MS [{lo[0]:.1f}, {hi[0]:.1f}]　TC [{lo[1]:.1f}, {hi[1]:.1f}]")


if __name__ == "__main__":
    main()
