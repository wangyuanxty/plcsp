# -*- coding: utf-8 -*-
"""**把「构造式列表调度」的 LA 搬进 DES** —— 只改派车那一层，PA 不动。

══ 要回答的问题（`open-question-nhdrl.md` §4.35）══

§4.35 量出：本仓最好 CDR 的 MS 是 **301.7**，而同口径（同样的矩阵、同样的机台选择规则）
的构造式可行解只要 **163.3** —— **差 1.85×**，与原文「最好 CDR 612 ／ NHDRL 300 = 2.04×」同形。

⚠️ **那个 163.3 出自 `plcsp/baselines/exact_cpsat.py` 的 `greedy_schedule`**（另一套模型 +
自带独立校验器）。它只证明「模型里存在这么一份调度」，**没有在本仓 DES 里跑过**。
本脚本补那一步。

══ 搬的是哪一半 ══

`greedy_schedule` 的 LA 是**选最早完工的车**：

    dh_v = 空载(车 v 当前位置 → 取货点)
    st_v = max(车 v 空闲时刻, 件的最早可运输时刻) + dh_v
    选 argmin_v st_v                     （负载段与 v 无关，是常数，不参与排序）

⚠️ **对照本仓现有的三条 LA 规则**：
- `MINCOST`       = `空载 + 负载` ⟹ 等价于**就近**（不看车的忙闲、不看件好了没）
- `MINU`          = 利用率最低（**连距离都不看**）
- `LATEST_IN_TIME`= 赶得上就最晚到（**不看路程**）

**三条里只有 `MINCOST` 沾距离，而它不扣「车还得等多久」和「件还没好」。**

══ 三条臂（其余一切逐字相同）══

| 臂 | PA | LA |
|---|---|---|
| 0 | `EAJ+SMPT`（规则，= 现有最好的一族） | `MINCOST`（规则）—— 应复现 §4.33.2 的 304.3 |
| 1 | `EAJ+SMPT`（规则） | **构造式 LA** |
| 2 | `SPT+SMPT`（规则） | **构造式 LA** |

**判据**：臂 1 若掉到 ~200 以下 ⟹ 缺口的**大半在 LA**；若仍在 ~300 ⟹ LA 不是主因，
得往 PA 的**任务选择**那一半找。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))   # 仓根
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from plcsp_v3.env.constraints import ConstraintConfig  # noqa: E402
from plcsp_v3.env.shi_instances import (EVAL_IDX, N_AGV_SHI,  # noqa: E402
                                        Instance, eval_instance)
from plcsp_v3.hrl.rules import PA_RULE_NAMES, apply_pa_rule  # noqa: E402
from plcsp_v3.q1_flat import SHI_SHOP_LEN, build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

A = int(os.environ.get("GP_A", "16"))
B = int(os.environ.get("GP_B", "140"))
N_INST = int(os.environ.get("GP_NINST", "20"))
EIDX = int(os.environ.get("GP_EIDX", str(EVAL_IDX)))
SHOPLEN = float(os.environ.get("GP_SHOPLEN", str(SHI_SHOP_LEN)))
SCHEME = os.environ.get("GP_SCHEME", "random")
# ⚠️ `GP_NOFAIL=1` ⟹ 关 ⑨ AGV 故障（与 `exact_cpsat` 那台确定性求解器同口径）。
NOFAIL = os.environ.get("GP_NOFAIL", "1") == "1"
# ⚠️ **`GP_PLANPOS`（2026-10-16）**：LA 的距离项取**已派任务之后的预计位置**
#    （原文 `l''` = *"完成上一任务后的位置"*）而不是物理当前位置。默认开。
PLANPOS = os.environ.get("GP_PLANPOS", "1") == "1"
# ⚠️ **`GP_ACTBOOK`（2026-10-16）**：`agv_act` 改在**派车那一刻**推进（原文 `ACT_v(t)`
#    的 "the last task **assigned**"）。默认开。
ACTBOOK = os.environ.get("GP_ACTBOOK", "1") == "1"
N_RUN = int(os.environ.get("GP_RUN", "3"))
SC_FIX = int(os.environ.get("GP_SC0", "100000"))


def pa_rule_index(name: str) -> int:
    return PA_RULE_NAMES.index(name)


def make_greedy_la():
    """**构造式 LA**：选**最早完工**的车。口径与 `exact_cpsat.greedy_schedule` 逐条对齐。

    ⚠️ 负载段 `loaded_leg(v)` 对每台车同值（同一任务、同一路径）⟹ **对排序是常数**，
    但**照写**，以便与构造式逐条对上。
    """

    def policy(view, cand_v):
        best_v, best_fin = -1, None
        for v in cand_v:
            st = max(view.vehicles[v].free_at, view.ready_at) + view.empty_leg(v)
            fin = st + view.loaded_leg(v)
            if best_fin is None or fin < best_fin:
                best_v, best_fin = int(v), fin
        assert best_v >= 0
        return best_v

    return policy


def make_greedy_pa(world, lay):
    """**构造式 PA**：在池里挑使「**最早完工**」最小的那道工序，并给它定机台。

    与 `exact_cpsat.greedy_schedule` 的联合决策**逐条对齐**：

        机台 m = 该工序**加工时长最短**的候选        （≡ 本仓的 `SMPT`）
        origin = 件**当前所在**的机台（`oi == 0` ⟹ 装卸站）
        loaded = leg(origin → m)
        fin(j, oi) = min_v [ max(车 v 空闲, 件可运输) + 空载(车 v 位置 → origin) + loaded ]
        取 argmin fin 的那道工序（连同它选定的 m）

    ⚠️ **`origin` 从 `world._pa_plans` 取**（`des.py:1879`：给了 `policy_pa` 时它挂在 world 上）。
    池的口径保证 `oi` 之前的工序**都已分配** ⟹ `plans[j][oi-1]` 必非空。
    **这是本仓 PA 结构里唯一拿得到「件在哪」的入口** —— `PAView` 里没有这个字段。

    ⚠️ **不要与 `world` 分开构造**：`leg_min` 必须与仿真**同一条**口径（仓库纪律）。
    """
    lu = int(lay.lu.dock_node)
    dock = [int(p.dock_node) for p in lay.machines]

    def policy(view, pool, tie_rng=None):
        plans = getattr(world, "_pa_plans", None)
        if plans is None:
            raise RuntimeError("构造式 PA 需要 `world._pa_plans` —— 它只在给了 `policy_pa` 时挂上")
        snap = view.snap
        # ⚠️ **位置口径与 `_la_view` 同一条**（`cfg.agv_plan_pos`）：
        #    已派任务有目的地就取它，否则取物理位置。两处不同源 = 假信号。
        planned = world._live_stats.get("agv_plan_pos") or []
        nodes = [(int(planned[v]) if v < len(planned) and int(planned[v]) >= 0
                  else int(snap.vehicles[v].node)) for v in range(len(snap.vehicles))]
        act = list(snap.agv_act) if snap.agv_act else [0.0] * len(nodes)

        def dead(u: int, org: int) -> float:
            # ⚠️ `u < 0` = **尚未出车** ⟹ 按「已在取货点」处置（与 `LAView.empty_leg` 同一条）。
            return 0.0 if (u < 0 or u == org) else world.leg_min(u, org)

        best = None
        for pi, (j, oi) in enumerate(pool):
            jv = view.jobs[j]
            if not jv.alts:
                continue
            m, _pt = min(jv.alts, key=lambda a: a[1])       # ≡ SMPT
            org = lu if oi == 0 else dock[int(plans[j][oi - 1])]
            dst = dock[int(m)]
            loaded = 0.0 if org == dst else world.leg_min(org, dst)
            fin = min(max(act[v], jv.transportable_at) + dead(nodes[v], org)
                      for v in range(len(nodes))) + loaded
            if best is None or fin < best[0]:
                best = (fin, pi, int(m))
        assert best is not None, "池非空但没有可用候选"
        return best[1], best[2]

    return policy


def build_spec(inst: Instance):
    ovr = {"due_range": 0.6, "la_tie_random": 1.0,
           "agv_mttr_lo": 10.0, "agv_mttr_hi": 10.0,
           "agv_plan_pos": 1.0 if PLANPOS else 0.0,
           "agv_act_book": 1.0 if ACTBOOK else 0.0}
    kw = (dict(shi2025=False,
               constraints=ConstraintConfig(finite_buffer=False, machine_failure=False,
                                            maintenance=False, due_dates=False,
                                            agv_failure=False))
          if NOFAIL else dict(shi2025=True))
    return build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                 finite_buffer=True, layout_seed=0, reward_norm="minmax",
                 tau=2.70, sim_overrides=ovr, inst_obj=inst,
                 layout_scheme=SCHEME, **dims_from_shop_len(SHOPLEN, inst.n_machines), **kw)[0]


def run(spec, k: int, policy_l) -> tuple[float, float]:
    w = spec.world()
    m = w.run_gated(seed_chain=SC_FIX,
                    policy_pa=lambda v, p, tr=None: apply_pa_rule(k, v, p, tr),
                    policy_l=policy_l, dt_events=0)
    return float(m["makespan"]), float(m["logistics_cost"])


def _empty_min(m: dict) -> float:
    """空载段总时长 [min]——**只为诊断**。`run_gated` 把它挂在 `energy_breakdown` 下。"""
    v = m.get("agv_empty_min")
    if v is None:
        v = (m.get("energy_breakdown") or {}).get("agv_states_min", {}).get("empty")
    return float(v) if v is not None else 0.0


def main() -> None:
    inst = eval_instance(B, A, N_INST, idx=EIDX)
    spec = build_spec(inst)
    greedy = make_greedy_la()
    print(f"**构造式 LA 搬进 DES** · Exp({A}/{B}) · eval idx={EIDX} · 车间 {SHOPLEN:.0f} m · "
          f"scheme={SCHEME} · {N_AGV_SHI} 台车 · **AGV 故障 = {'关' if NOFAIL else '开'}** · "
          f"**位置口径 = {'已派任务目的地' if PLANPOS else '物理当前位'}** · "
          f"**账本进位 = {'派车那刻' if ACTBOOK else '任务发出时'}** · "
          f"{N_RUN} 局/臂\n", flush=True)
    print(f"  {'臂':<4}{'PA':<12}{'LA':<14}{'MS':>9}{'TC':>9}{'运输 min':>10}{'空载 min':>10}")
    arms = [(0, "EAJ+SMPT", "MINCOST", None, ""),
            (1, "EAJ+SMPT", "构造式", greedy, ""),
            (2, "SPT+SMPT", "构造式", greedy, ""),
            # ⭐ **臂 3 = 构造式 PA + 构造式 LA**（与 `exact_cpsat.greedy_schedule` 对齐的那一对）
            (3, "**构造式**", "构造式", greedy, "pa")]
    rows = []
    for tag, pa_name, la_name, pol, kind in arms:
        if pol is None:
            from plcsp_v3.hrl.rules import la_rule_policy
            pol = la_rule_policy(0)                       # 0 = MINCOST
        k = -1 if kind == "pa" else pa_rule_index(pa_name)
        ms, tc, emp = [], [], []
        for _i in range(N_RUN):
            w = spec.world()
            if kind == "pa":
                # ⚠️ **必须与 `w` 同建**：`leg_min` 要同一条口径，`_pa_plans` 也挂在 `w` 上。
                pa = make_greedy_pa(w, spec.layout)
            else:
                pa = lambda v, p, tr=None, _k=k: apply_pa_rule(_k, v, p, tr)  # noqa: E731
            m = w.run_gated(seed_chain=SC_FIX, policy_pa=pa, policy_l=pol, dt_events=0)
            ms.append(float(m["makespan"]))
            tc.append(float(m["logistics_cost"]))
            emp.append(_empty_min(m))
        mm, tt, ee = np.mean(ms), np.mean(tc), np.mean(emp)
        rows.append((tag, pa_name, la_name, mm, tt, ee))
        print(f"  {tag:<4}{pa_name:<12}{la_name:<14}{mm:>9.1f}{tt:>9.1f}"
              f"{2 * (tt - 42.0):>10.1f}{ee:>10.1f}", flush=True)
    base = rows[0][3]
    print(f"\n  ⭐ 臂 1 / 臂 0 = **{rows[1][3] / base:.2f}×**"
          f"　臂 2 / 臂 0 = **{rows[2][3] / base:.2f}×**"
          f"　**臂 3 / 臂 0 = {rows[3][3] / base:.2f}×**")
    print("  对照靶（`exact_cpsat.greedy_schedule`，确定性模型）= **163.3 min**"
          " · 本仓最好 CDR = 301.7 min ⟹ 倍率 1.85×")


if __name__ == "__main__":
    main()
