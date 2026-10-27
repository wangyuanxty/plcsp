# -*- coding: utf-8 -*-
"""**扫 `tt` 强度** —— 「原文的 CDR 为什么比我们弱」的最后一个候选解释。

══ 要判的 ══

```
tt ↓（运输变便宜）
  ⟹ 运输不再是瓶颈
  ⟹ LA 选谁无所谓（车都不忙、距离都不远）
  ⟹ **Rule 3 的优势 ↓**
  ⟹ **RN/CDR ↑ ？**
```

**成立** ⟹ 原文那边 `tt` 比我们弱，就是那个缺口。
**不成立** ⟹ 运输强度不是那个变量，得换思路。

══ ⚠️ 为什么以前的扫描没回答这个 ══

| 扫描 | 量的是 | 缺 |
|---|---|---|
| `scale` 6→22（§37.50） | MS/TC 比值（对 Fig. 6 的靶） | **没量 RN** |
| 车速 0.5→0.05（`v3_speed_scale.py`） | 规则极差 / 余量 | 只扫**放慢**方向 |
| `scale` 4→16（§37.51.4） | 27 条 CDR 的前沿 | **没量 RN** |

**⟹ 「RN/CDR 作为 `tt` 强度的函数」从来没量过。**

══ ⚠️ 用机位/走廊尺寸扫，不用 `agv_speed` ══

`tt ∝ pitch / speed`（`pitch = cell + aisle`）。
旧代码用 `layout_scale`（已删除）；现在直接给
`cell_w=3s · cell_h=2.4s · aisle_w=1.5s · lu_connector_m=3s`，
与旧口径逐位相同。**这是诊断，用来判"这条路通不通"。**
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import v3_hv3 as H  # noqa: E402
from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, N_AEI2025_RULES,  # noqa: E402
                                PA_RULE_NAMES, apply_pa_rule, la_rule_policy)
from plcsp_v3.q1_flat import build  # noqa: E402

SCALES = [float(x) for x in os.environ.get("TT_SCALES", "4,6,8,12.5,20").split(",")]
# ⚠️ **`n_agv` 是与 `tt` 正交的旋钮**（2026-10-15 加）：`tt` 缩放 ⟹ 近似**相似变换**
#    （实测 `R3≠R1` 跨 5 档恒定 26–30%）⟹ 改不动比值。
#    而 **车队规模**改的是**资源稀缺度**，不是相似变换。
#    ⚠️ 原文写死 `V = 6` ⟹ **这一扫是诊断，不是方案**。
AGVS = [int(x) for x in os.environ.get("TT_AGVS", str(N_AGV_SHI)).split(",")]
PA = os.environ.get("TT_PA", "EAJ+SMPT")
N_SC = int(os.environ.get("TT_NSC", "20"))
# ⚠️ **`sim_overrides` 逃生口**：`TT_OVR="agv_load_min=1.5,agv_unload_min=1.5"`。
#    用它扫**装卸时间**等进 `SimConfig` 的量。
OVR = os.environ.get("TT_OVR", "")
SIM_OVR = {"due_range": 0.6}
if OVR:
    SIM_OVR.update({k: float(v) for k, v in
                    (kv.split("=") for kv in OVR.split(","))})


def probe_busy(spec, scs):
    """跑一遍 `PA` + `LATEST_IN_TIME`，退回逐决策点的车队忙度与选车分歧。"""
    rec = []
    inner_lit = la_rule_policy("LATEST_IN_TIME")
    inner_mc = la_rule_policy("MINCOST")

    def pol(view, cand):
        rec.append(dict(
            busy_any=float(np.mean([v.free_at > view.now for v in view.vehicles])),
            diff=int(inner_lit(view, cand)) != int(inner_mc(view, cand)),
        ))
        return int(inner_lit(view, cand))

    ms, legs, mk = [], 0.0, 0.0
    for i, sc in enumerate(scs):
        m = spec.world().run_gated(
            seed_chain=sc,
            policy_pa=lambda v, p, tr=None: apply_pa_rule(PA, v, p, tr),
            policy_l=pol, dt_events=0)
        ms.append(float(m["makespan"]))
        if i == 0:
            b = m["energy_breakdown"]["agv_states_min"]
            legs = float(b["empty"]) + float(b["loaded"])
            mk = float(m["makespan"])
    return (float(np.mean(ms)), float(np.mean([r["busy_any"] for r in rec])),
            float(np.mean([r["diff"] for r in rec])), legs, mk)


def main() -> None:
    inst = named(140, 16)
    scs = [100_000 + 13 * i for i in range(N_SC)]
    n_legs = 2 * (sum(len(o) for o in inst.jobs) + inst.n_jobs)
    print(f"扫 tt 强度 / 车队规模 · Exp(16/140) · {N_SC} 场景 · PA={PA}\n")
    print(f"  {'scale':>7}{'n_agv':>7}{'单程/min':>10}{'忙车占比':>10}{'R3≠R1':>8}"
          f"{'R3的MS':>9}{'R1的MS':>9}{'RN的MS':>9}{'CDR最好MS':>11}{'RN/CDR':>9}"
          f"**{'corr(MS,TC)':>14}**")
    print("  " + "-" * 106)

    for s in SCALES:
        for na in AGVS:
            spec = build(seed=0, pm_interval=30, target_failures=4, n_agv=na,
                         finite_buffer=True, layout_seed=0, reward_norm="minmax",
                         tau=2.70, sim_overrides=SIM_OVR, inst_obj=inst,
                         layout_scheme=os.environ.get("TT_SCHEME", "random"),
                         cell_w=3.0 * s, cell_h=2.4 * s,
                         aisle_w=1.5 * s, lu_connector_m=3.0 * s,
                         shi2025=True)[0]
            ms_lit, busy, diff, legs, mk = probe_busy(spec, scs)
            # R1 = MINCOST 的 MS（同一 PA）
            ms_mc = float(np.mean([
                float(spec.world().run_gated(
                    seed_chain=sc,
                    policy_pa=lambda v, p, tr=None: apply_pa_rule(PA, v, p, tr),
                    policy_l=la_rule_policy("MINCOST"), dt_events=0)["makespan"])
                for sc in scs]))

            rn = np.asarray([H.run_rn(spec, i) for i in range(N_SC)])
            cdr_pts, cdr_tag = [], []
            for k in range(N_AEI2025_RULES):
                for j in range(len(LA_RULE_NAMES)):
                    cdr_pts.append(np.asarray(
                        [H.run_cdr(spec, k, j) for _ in range(N_SC)]))
                    cdr_tag.append((k, j))

            # ⚠️ **只比物理量，不比 HV** —— HV 的归一化基准每档不同，
            #    且 **HV 比与 MS 比方向会相反**（见 `open-question-nhdrl.md` §4.11.3）。
            rn_ms = float(rn[:, 0].mean())
            ms_all = np.asarray([P[:, 0].mean() for P in cdr_pts])
            tc_all = np.asarray([P[:, 1].mean() for P in cdr_pts])
            bms = int(np.argmin(ms_all))
            # ⚠️ **两个目标的权衡强度**：在**全部 27 个解**上算 `corr(MS, TC)`。
            #    **负 = 两目标权衡**（Pareto 前沿才有意义）；**正 = 同向**（前沿退化）。
            #    ⚠️ **不能用非支配子集的两端算斜率** —— 那是两个离散点，没有统计意义。
            corr = float(np.corrcoef(ms_all, tc_all)[0, 1])
            print(f"  {s:>7.1f}{na:>7d}{legs / n_legs:>10.2f}{busy:>10.1%}{diff:>8.1%}"
                  f"{ms_lit:>9.1f}{ms_mc:>9.1f}"
                  f"{rn_ms:>9.1f}{ms_all[bms]:>11.1f}{rn_ms / ms_all[bms]:>9.2f}"
                  f"**{corr:>+14.3f}**"
                  f"  （最好 CDR = {PA_RULE_NAMES[cdr_tag[bms][0]]}"
                  f"+{LA_RULE_NAMES[cdr_tag[bms][1]]}）", flush=True)

    print("\n  ⚠️ 原文 `Exp(16/140)` HV：RN 0.3378 · CDRs 0.2642 ⟹ 1.28×"
          "（**HV 不可跨档比**，此处只作背景）")
    print("  ⚠️ 判据（**看物理量**）：")
    print("     ① `tt ↓` ⟹ `忙车占比` ↓")
    print("     ② `tt ↓` ⟹ **`R1的MS / R3的MS` → 1**（Rule 3 优势消失）")
    print("     ③ 那两条成立 ⟹ **`RN/CDR` 应上扬**  ⟸ **这是要判的那一条**")


if __name__ == "__main__":
    main()
