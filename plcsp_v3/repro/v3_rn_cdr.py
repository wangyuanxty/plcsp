# -*- coding: utf-8 -*-
"""**只算 RN 与 27 条 CDR 的两个目标 + HV**（不需要训练、不需要 ckpt）。

══ 为什么单独一个文件 ══

`v3_hv3.py` 里的 `run_rn` / `run_cdr` 是对的（它扫满 27 条），但那个脚本**还要一个策略 ckpt**
才跑得完。**RN 与 CDR 都是纯基线** —— 它们两个的数不需要任何训练。

⚠️ **本文件不改 `v3_hv3.py`**；它自己建实例，口径与该文件逐字一致
（`SC_FIX` · `N_RUN=20` · 归一化基准 = 全部解 · 参考点 `(1,1)`）。

⚠️ **评测实例走 `eval_instance`**（不在训练池里），与 `v3_nhdrl.py` / `shi2025_min2.py` 同源。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))   # 仓根
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # repro/ 自身

import numpy as np  # noqa: E402

import v3_hv3 as H  # noqa: E402
from plcsp_v3.env.constraints import ConstraintConfig  # noqa: E402
from plcsp_v3.env.shi_instances import (EVAL_IDX, N_AGV_SHI,  # noqa: E402
                                        eval_instance)
from plcsp_v3.hrl.mo_metrics import (REF_NORM, hypervolume_2d,  # noqa: E402
                                     normalize_minmax, pareto_front)
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, N_AEI2025_RULES,  # noqa: E402
                                PA_RULE_NAMES)
from plcsp_v3.q1_flat import SHI_SHOP_LEN, build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

A = int(os.environ.get("RC_A", "16"))
B = int(os.environ.get("RC_B", "140"))
N_INST = int(os.environ.get("RC_NINST", "20"))
# ⚠️ **`RC_EIDX`（2026-10-15 加）**：换另一个**评测实例**。默认 = `EVAL_IDX` ⟹ 逐位不变。
EIDX = int(os.environ.get("RC_EIDX", str(EVAL_IDX)))
SCHEME = os.environ.get("RC_SCHEME", "random")
# ⚠️ **`RC_SHOPLEN`（2026-10-16）**：车间长度 [m]，取代旧的 `RC_SCALE`（无因次倍率）。
#    默认 = `SHI_SHOP_LEN`（由 Fig. 6(b) 的 TC ≈ 420 标定）。见 `SHI_SHOP_LEN` 的注释。
SHOPLEN = float(os.environ.get("RC_SHOPLEN", str(SHI_SHOP_LEN)))
SHI = os.environ.get("RC_SHI", "1") == "1"
# ⚠️ **`RC_NOFAIL=1`（2026-10-16 加）**：把 **AGV 故障（⑨）关掉**，其余 Shi 口径逐条不变。
#
# **为什么需要它**：`exact_cpsat` 那台精确求解器（见 `v3_exact_ms.py`）是**确定性**模型，
# 没有故障 ⟹ 拿它的读数跟本仓「注入了故障」的读数比是**不同口径**。
# `build(shi2025=True)` 的故障块**无法**被 `sim_overrides` 压成 0（它写在 `over.update`
# **之后**，且 `not agv_n_failures` 时无条件抽 1–8 次）⟹ 只能改走 `shi2025=False`
# + **显式 `constraints=`**，把那一条开关关掉、其余四条照旧关（= 与 `shi2025` 分支同档）。
# ⚠️ **默认关 ⟹ 既有读数逐位不变。**
NOFAIL = os.environ.get("RC_NOFAIL", "") == "1"
OVR = os.environ.get("RC_OVR", "")
SIM_OVR = {"due_range": 0.6}
if OVR:
    # 形如 `agv_load_min=0.75,agv_unload_min=0.75,agv_mttr_lo=5,agv_mttr_hi=10`
    SIM_OVR.update({k: float(v) for k, v in
                    (kv.split("=") for kv in OVR.split(","))})
if NOFAIL:
    # `la_tie_random` 与 `agv_mttr_*` 是 `shi2025` 分支设的，这里逐条补上。
    SIM_OVR["la_tie_random"] = 1.0
    SIM_OVR["agv_mttr_lo"] = SIM_OVR["agv_mttr_hi"] = 10.0



def main() -> None:
    # ⚠️ `RC_INST=old` ⟹ 用**池里那个实例 0**（`named`，= 2026-10-14 之前的口径），
    #    用来复现 `layout_scale=12.5` 当初标定时看到的 best CDR (567, 612)。
    #    `new`（默认）⟹ `eval_instance`（不在训练池，见 §4.7.2）。
    # ⚠️ **重尾 `α_ijk`**：`RC_NARROW="n_narrow-wide_lo-wide_hi"`，例 `"3-4-8"`。
    #    空（默认）⟹ 均匀 2–4 台，**逐位不变**。见 `shi_instances.make_shi_routings`。
    #    ⚠️ **算在分支外面** —— 两条分支（`old`/`new`）都要用（2026-10-15 修：
    #    一度写在 `else` 里，`RC_INST=old` 时参数根本没传进去，读数与默认**逐位相同**）。
    _nr = os.environ.get("RC_NARROW", "")
    kw = ({} if not _nr else
          dict(zip(("narrow_n", "wide_lo", "wide_hi"),
                   (int(x) for x in _nr.split("-")))))
    if os.environ.get("RC_INST", "new") == "old":
        from plcsp_v3.env.shi_instances import named
        inst = named(B, A, **kw)
        tag = "池实例0（旧口径）" + (f" · 重尾 α={_nr}" if _nr else "")
    else:
        # ⚠️ **`RC_EIDX`（2026-10-15 加）**：换**另一个**评测实例 —— 用来查「读数是不是
        #    这一个实例的特例」。默认 `EVAL_IDX` ⟹ 既有读数**逐位不变**。
        inst = eval_instance(B, A, N_INST, idx=EIDX, **kw)
        tag = (f"eval idx={EIDX}（不在训练池）"
               + (f" · 重尾 α={_nr}" if _nr else ""))
    # ⚠️ **`RC_NOFAIL=1`**：走 `shi2025=False` + 显式 `constraints=`，只把 **⑨ AGV 故障**
    #    关掉（其余三条本来就关着）—— 用来与 `exact_cpsat` 那台**确定性**求解器同口径。
    _bk = (dict(shi2025=False,
                constraints=ConstraintConfig(finite_buffer=False, machine_failure=False,
                                             maintenance=False, due_dates=False,
                                             agv_failure=False))
           if NOFAIL else dict(shi2025=SHI))
    spec = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                 finite_buffer=True, layout_seed=0, reward_norm="minmax",
                 tau=2.70, sim_overrides=SIM_OVR, inst_obj=inst,
                 layout_scheme=SCHEME, **dims_from_shop_len(SHOPLEN, inst.n_machines), **_bk)[0]
    print(f"RN / 27 CDR · Exp({A}/{B}) · 实例 = {tag} · "
          f"真工序 {sum(len(o) for o in inst.jobs)} · {H.N_RUN} 次运行\n"
          f"  scheme={SCHEME} · 车间长={SHOPLEN:.0f} m · shi2025={not NOFAIL} · "
          f"**AGV 故障 = {'关' if NOFAIL else '开'}** · "
          f"sim_overrides={SIM_OVR}"
          f" · seed_chain={H.SC_FIX}（全部方法同一个实例+同一组故障）\n",
          flush=True)

    rn = np.asarray([H.run_rn(spec, i) for i in range(H.N_RUN)])
    print(f"  RN    {H.N_RUN} 局完成", flush=True)
    n_cdr = N_AEI2025_RULES * len(LA_RULE_NAMES)
    cdr_pts, cdr_tag = [], []
    for k in range(N_AEI2025_RULES):
        for j in range(len(LA_RULE_NAMES)):
            cdr_pts.append(np.asarray([H.run_cdr(spec, k, j) for _ in range(H.N_RUN)]))
            cdr_tag.append((k, j))
    print(f"  CDR   {n_cdr} 条 × {H.N_RUN} 局 = {n_cdr * H.N_RUN} 局完成\n", flush=True)

    normed, (lo, hi) = normalize_minmax([rn, *cdr_pts])
    rn_n, cdr_n = normed[0], normed[1:]
    print(f"  归一化基准（该实例上全部解）：MS [{lo[0]:.1f}, {hi[0]:.1f}]"
          f"　TC [{lo[1]:.1f}, {hi[1]:.1f}]　参考点 {REF_NORM}")
    print("  ⚠️ 原文**没说**基准按谁 —— 本仓取「全部解」，登记在案。\n")

    hv_rn = hypervolume_2d(pareto_front(rn_n), REF_NORM)
    hv_cdr = np.asarray([hypervolume_2d(pareto_front(P), REF_NORM) for P in cdr_n])
    ms = np.asarray([P[:, 0].mean() for P in cdr_pts])
    tc = np.asarray([P[:, 1].mean() for P in cdr_pts])

    print(f"  **RN   {H.N_RUN} 次**：MS 均值 {rn[:, 0].mean():.1f}　TC 均值 {rn[:, 1].mean():.1f}"
          f"　Pareto 点数 {len(pareto_front(rn_n))}　**HV {hv_rn:.4f}**")
    # ⚠️ **两个目标的权衡强度**：在**全部 27 个解**上算 `corr(MS, TC)`。
    #    **负 = 两目标权衡**（Pareto 前沿才有意义）；**正 = 同向**（前沿退化）。
    #    ⚠️ 27 个点的 corr 标准误 ≈ 1/√27 ≈ **±0.19** —— 判读要带这个噪声界。
    corr = float(np.corrcoef(ms, tc)[0, 1])
    print(f"  **`corr(MS, TC)`（27 条）= {corr:+.3f}**"
          f"　⚠️ 27 点的噪声 ≈ ±0.19")
    print("\n  ── 27 条 CDR（按 MS 升序前 6）──")
    print(f"  {'PA 规则':<12}{'LA 规则':<16}{'MS':>8}{'TC':>10}{'HV':>10}")
    for i in np.argsort(ms)[:6]:
        k, j = cdr_tag[i]
        print(f"  {PA_RULE_NAMES[k]:<12}{LA_RULE_NAMES[j]:<16}"
              f"{ms[i]:>8.1f}{tc[i]:>10.1f}{hv_cdr[i]:>10.4f}")
    best = int(np.argmax(hv_cdr))
    k, j = cdr_tag[best]
    print(f"\n  **CDR 最好（单条 HV 口径）**：{PA_RULE_NAMES[k]}+{LA_RULE_NAMES[j]}"
          f"　MS {ms[best]:.1f} / TC {tc[best]:.1f}　**HV {hv_cdr[best]:.4f}**")
    bms = int(np.argmin(ms))
    k2, j2 = cdr_tag[bms]
    print(f"  **CDR 最好（MS 口径）**：{PA_RULE_NAMES[k2]}+{LA_RULE_NAMES[j2]}"
          f"　MS {ms[bms]:.1f} / TC {tc[bms]:.1f}　HV {hv_cdr[bms]:.4f}")

    # ⚠️ **两种「CDRs」口径都要报** —— 原文没写清 Table 9 的 `CDRs` 是哪一个：
    #   ① **单条最好**：27 条各自的 HV 取最大（`v3_hv3.py` 用的口径）
    #   ② **27 条并集**：把 27 个解当一个方法的解集（`v3_hv.py` 用的口径）
    #   两者差很多 ⟹ **不写清就不可比**。
    union = np.vstack(cdr_pts)                       # 27 条的原始 (MS, TC)
    cdr_all = np.vstack(cdr_n)                       # 同上，归一化后
    hv_union = hypervolume_2d(pareto_front(cdr_all), REF_NORM)
    print(f"  **CDR 全部 27 条并集**：MS {union[:, 0].min():.1f}–{union[:, 0].max():.1f}"
          f"　**HV {hv_union:.4f}**")

    print(f"\n  **RN / 单条最好 CDR = {hv_rn / hv_cdr.max():.2f}×**"
          f"　**RN / 27 条并集 = {hv_rn / hv_union:.2f}×**"
          f"　（原文 Exp(16/140)：RN 0.3378 · CDRs 0.2642 ⟹ **1.28×**）")
    print(f"  CDR 单条 HV：中位 {np.median(hv_cdr):.4f}　最小 {hv_cdr.min():.4f}")
    print("\n  ⚠️ 原文那两个数是**他们的实例**上的，**不可与本仓并列** —— 此处只比**比值**。")


if __name__ == "__main__":
    main()
