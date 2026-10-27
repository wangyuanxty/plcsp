# -*- coding: utf-8 -*-
"""**簇间距扫描** —— 车间非均匀（左右两簇，中间一条宽通道）时，权衡会不会出现（2026-10-16）。

══ 用户提出 ══

> *"如果机台不是均匀分布呢？"*

**而这不是之前两条裁定挡得住的：**

| 你裁过的 | 挡得住吗 |
|---|---|
| 「换排布 ≡ 给机台重新编号 ⟹ 统计量不变」（2026-10-15） | ✗ **那是置换**；本脚本改的是**距离矩阵本身** |
| 「正方形节点间距 + L/U 占 1 格」 | ✗ **那是各向同性**；本脚本改的是**间距不均匀** |

**机制**：均匀网格里 19 个节点两两差不多远（实测中位 **4.0** / max 9.3 min）
⟹ 「选哪台机」对 TC 的影响被**压平** ⟹ 27 条规则在 TC 上拉不开。
**簇间距让「距离」第一次成为一件有后果的事。**

⚠️ **原文没有依据** —— Fig. 4 只给拓扑，**尺寸一个字都没有**。

══ ⚠️ 必须做的归一：每个簇间距都把 `shop_len` 重新标定 ══

**簇间距会把总运输量抬高**（跨簇的每趟都变长）⟹ 若不补偿，就分不清
「间距变了」和「运输量变了」。故对每个 `gap` **二分 `shop_len`**，
使**机台两两行程时长的均值**回到与 `gap=0` 时同一水平（≈ 5.46 min，§4.33.3 的标定值）。

══ 判据 ══

**不用 `corr(MS,TC)`**（它被「两个都差」的大簇支配，§4.40 已证它误导），也**不用 PCA 方向**
（那条靶是我编的，已撤回）。**用有原文锚的那一条**：

```
原文：**RN 在 MS 上赢**展示的那条 CDR **30%**（428 vs 610），在 TC 上输 43%
本仓（gap=0）：**最好 CDR 在 MS 上赢 RN 26%** —— **方向反了**
```

⚠️ **PCA 方向照量，但只作描述，不当靶。**
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import v3_hv3 as H  # noqa: E402
from plcsp_v3.env import layout as layout_mod  # noqa: E402
from plcsp_v3.env.constraints import ConstraintConfig  # noqa: E402
from plcsp_v3.env.corridors import (build_corridor_graph,  # noqa: E402
                                    dock_distance_matrix)
from plcsp_v3.env.shi_instances import (EVAL_IDX, N_AGV_SHI,  # noqa: E402
                                        eval_instance)
from plcsp_v3.hrl.rules import LA_RULE_NAMES, N_AEI2025_RULES  # noqa: E402
from plcsp_v3.q1_flat import SHI_AISLE_W, build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

EIDX = int(os.environ.get("CS_EIDX", str(EVAL_IDX)))
GAPS = [float(x) for x in os.environ.get("CS_GAPS", "0,60,160,300").split(",")]
SPLIT = int(os.environ.get("CS_SPLIT", "3"))
TARGET_TRIP = float(os.environ.get("CS_TRIP", "5.46"))    # §4.33.3 的标定值
NOFAIL = os.environ.get("CS_NOFAIL", "1") == "1"
N_RN = int(os.environ.get("CS_RN", "10"))
N_COLS = 5
SC_FIX = int(os.environ.get("CS_SC0", "100000"))


def mean_trip(shop_len: float, gap: float) -> float:
    """该 (shop_len, gap) 下**机台两两**行程时长的均值 [min] —— 只建布局与距离矩阵，不跑仿真。"""
    pitch = shop_len / (N_COLS + 1)
    side = pitch - SHI_AISLE_W
    lay = layout_mod.sample_layout(
        N_AGV_SHI + 12, seed=0, cell_w=side, cell_h=side, aisle_w=SHI_AISLE_W,
        lu_connector_m=pitch, n_cols=None, cap_fixed=None,
        scheme="random", n_agv=N_AGV_SHI,
        cluster_gap=gap, cluster_split=SPLIT)
    dm = dock_distance_matrix(build_corridor_graph(lay))
    dock = [int(p.dock_node) for p in lay.machines]
    eff = min(1.0, max(0.4, SHI_AISLE_W / 1.5))
    a = np.array(dock)[:, None]
    return float(np.mean(dm[a, a.T]) / (eff * 0.5) / 60.0)


def calibrate(gap: float) -> float:
    """二分 `shop_len` 使 `mean_trip ≈ TARGET_TRIP`。"""
    lo, hi = 20.0, 600.0
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if mean_trip(mid, gap) < TARGET_TRIP:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_spec(inst, shop_len: float, gap: float):
    ovr = {"due_range": 0.6, "la_tie_random": 1.0,
           "agv_mttr_lo": 10.0, "agv_mttr_hi": 10.0}
    kw = (dict(shi2025=False,
               constraints=ConstraintConfig(finite_buffer=False, machine_failure=False,
                                            maintenance=False, due_dates=False,
                                            agv_failure=False))
          if NOFAIL else dict(shi2025=True))
    return build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                 finite_buffer=True, layout_seed=0, reward_norm="minmax",
                 tau=2.70, sim_overrides=ovr, inst_obj=inst,
                 layout_scheme="random", **dims_from_shop_len(shop_len, inst.n_machines),
                 cluster_gap=gap, cluster_split=SPLIT, **kw)[0]


def pca_angle(ms: np.ndarray, tc: np.ndarray) -> float:
    """27 个点**标准化后**第一主成分与 MS 轴的夹角 [度]。+45° = 沿对角线；−45° = 沿权衡方向。"""
    X = np.column_stack([(ms - ms.mean()) / ms.std(), (tc - tc.mean()) / tc.std()])
    w, v = np.linalg.eigh(np.cov(X.T))
    p = v[:, int(np.argmax(w))]
    return float(np.degrees(np.arctan2(p[1], p[0])))


def main() -> None:
    inst = eval_instance(140, 16, 20, idx=EIDX, alts_pow=-2.0)
    print(f"**簇间距扫描** · Exp(16/140) · idx={EIDX} · 列 {N_COLS} · "
          f"分簇在列 {SPLIT} · 故障={'关' if NOFAIL else '开'}\n"
          f"  每个 gap 都把 shop_len 二分标定到 **行程均值 ≈ {TARGET_TRIP:.2f} min**"
          f"（否则分不清「间距变了」和「运输量变了」）\n", flush=True)
    print(f"  {'簇间距[m]':>10}{'标定车间[m]':>13}{'行程均值':>10}{'趟数':>7}"
          f"{'MS 最好':>10}{'TC 最好':>10}{'MS 最差':>10}"
          f"{'corr':>9}{'PCA角':>8}{'RN MS':>9}{'CDR-MS/RN':>11}")
    print("  " + "-" * 108)
    for gap in GAPS:
        sl = calibrate(gap)
        trip = mean_trip(sl, gap)
        spec = build_spec(inst, sl, gap)
        ms, tc = [], []
        for k in range(N_AEI2025_RULES):
            for j in range(len(LA_RULE_NAMES)):
                m_, t_ = H.run_cdr(spec, k, j)
                ms.append(m_)
                tc.append(t_)
        ms, tc = np.asarray(ms), np.asarray(tc)
        rn_ms = np.mean([H.run_rn(spec, i)[0] for i in range(N_RN)])
        corr = float(np.corrcoef(ms, tc)[0, 1])
        ang = pca_angle(ms, tc)
        ratio = ms.min() / rn_ms
        flag = "  ← **方向翻了**" if ratio > 1.0 else ""
        print(f"  {gap:>10.0f}{sl:>13.1f}{trip:>10.2f}{len(ms):>7}"
              f"{ms.min():>10.1f}{tc.min():>10.1f}{ms.max():>10.1f}"
              f"{corr:>+9.3f}{ang:>+8.1f}{rn_ms:>9.1f}{ratio:>11.2f}{flag}", flush=True)
    print("\n  ⚠️ 判据 = 最后一列（`CDR 最好 MS ÷ RN 的 MS`）：**< 1 ⟹ 规则赢 RN（方向反了）**；")
    print("     **> 1 ⟹ RN 赢规则（= 原文的方向）**。原文那格是 610/428 = **1.43**。")
    print("  ⚠️ `PCA 角` 只作描述，**不当靶**（那条靶是我编的，已撤回，见 §4.40）。")


if __name__ == "__main__":
    main()
