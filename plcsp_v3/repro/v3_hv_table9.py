# -*- coding: utf-8 -*-
"""**Table 9 的 18 个实例** —— NHDRL / RN / CDRs 三个 HV，逐实例。

实例清单（`docs/v3/open-question-nhdrl.md` §4.27.2 + SP 文本里的 `Exp(36/316)`）：

```
Exp(7/59)  Exp(8/70)   Exp(10/86)  Exp(11/96)  Exp(16/140) Exp(18/156)
Exp(20/173) Exp(21/185) Exp(22/192) Exp(23/203) Exp(25/215) Exp(27/238)
Exp(28/246) Exp(30/261) Exp(31/272) Exp(32/287) Exp(34/305) Exp(36/316)
```

⚠️ **实例是用 `named` 生成的合成实例**，只对齐 `(工件数 a, 总工序数 b)`；
原文是真实航空车间数据，我们没有那张表。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import v3_hv3 as H  # noqa: E402
import v3_nhdrl as N  # noqa: E402
from plcsp_v3.env.nhdrl_features import NHFeatures  # noqa: E402
from plcsp_v3.env.shi_instances import named  # noqa: E402
from plcsp_v3.hrl.mo_metrics import (REF_NORM, hypervolume_2d,  # noqa: E402
                                     normalize_minmax, pareto_front)
from plcsp_v3.q1_flat import build  # noqa: E402
from plcsp_v3.q1_flat import dims_from_shop_len  # noqa: E402

TABLE9 = [(7, 59), (8, 70), (10, 86), (11, 96), (16, 140), (18, 156),
          (20, 173), (21, 185), (22, 192), (23, 203), (25, 215), (27, 238),
          (28, 246), (30, 261), (31, 272), (32, 287), (34, 305), (36, 316)]
N_RUN = int(os.environ.get("HV9_RUN", "20"))
CKPT = os.environ.get("HV9_CKPT", os.path.join(
    os.environ.get("TEMP", "/tmp"), "nhdrl3_vary.pt"))
SHOPLEN = float(os.environ.get("HV9_SHOPLEN", "200"))
SCHEME = os.environ.get("HV9_SCHEME", "random")
SHI = os.environ.get("HV9_SHI", "1") == "1"


def main() -> None:
    if not os.path.exists(CKPT):
        raise SystemExit(f"⚠️ 找不到 ckpt `{CKPT}` —— 先跑 `v3_nhdrl.py` 训练。")
    net = N.NHDRL()
    _obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(_obj, dict) and "state_dict" in _obj:
        net.load_state_dict(_obj["state_dict"])
        feat_stats = _obj.get("feat_stats")
    else:
        net.load_state_dict(_obj)
        feat_stats = None
    net.eval()
    print(f"**Table 9 · 18 实例 · 每方法 {N_RUN} 次**\n"
          f"  策略 ckpt = `{CKPT}`　特征标准化 = {feat_stats is not None}\n",
          flush=True)
    print(f"  {'实例':>13}{'真工序':>7}{'NHDRL HV':>10}{'RN HV':>8}"
          f"{'CDRs HV':>9}")
    print("  " + "-" * 50)
    rows = []
    t0 = __import__("time").time()
    for i, (a, b) in enumerate(TABLE9):
        inst = named(b, a, seed=1000 + i, alts_dist="uniform",
                     alts_min=N.ALTS_MIN, alts_max=N.ALTS_MAX,
                     pt_dist="uniform", pt_lo=2, pt_hi=10)
        n_ops = sum(len(o) for o in inst.jobs)
        spec, _ = build(seed=0, pm_interval=30, target_failures=4,
                     n_agv=N.NAGV, finite_buffer=True, layout_seed=0,
                     reward_norm="none", tau=2.70,
                     sim_overrides=N._sim_over(), inst_obj=inst,
                     layout_scheme=SCHEME, **dims_from_shop_len(SHOPLEN, inst.n_machines, layout_cols=N.LAYOUT_COLS),
                     layout_cols=N.LAYOUT_COLS, layout_rows=N.LAYOUT_ROWS,
                     shi2025=SHI)
        nf = NHFeatures(inst, spec.layout,
                        speed_m_per_min=spec.cfg.agv_speed_mps * 60.0)
        P_nh = np.asarray([H.run_policy(spec, nf, net, k, feat_stats)
                           for k in range(N_RUN)])
        P_rn = np.asarray([H.run_rn(spec, k) for k in range(N_RUN)])
        P_cd = np.asarray([H.run_cdr(spec, k, j)
                           for k in range(H.N_AEI2025_RULES)
                           for j in range(H.N_LA_RULES)])
        normed, _ = normalize_minmax([P_nh, P_rn, P_cd])
        hvs = []
        for P in normed:
            hvs.append(hypervolume_2d(pareto_front(P), REF_NORM))
        rows.append((a, b, n_ops, *hvs))
        print(f"  {f'Exp({a}/{b})':>13}{n_ops:>7}{hvs[0]:>10.4f}"
              f"{hvs[1]:>8.4f}{hvs[2]:>9.4f}", flush=True)
    arr = np.asarray([(r[3], r[4], r[5]) for r in rows], dtype=float)
    print("  " + "-" * 50)
    print(f"  {'均值':>13}{'':>7}{arr[:, 0].mean():>10.4f}"
          f"{arr[:, 1].mean():>8.4f}{arr[:, 2].mean():>9.4f}")
    print(f"\n  用时 {__import__('time').time() - t0:.0f} s")
    print("  ⚠️ 实例是 `named` 生成的合成实例，只对齐 (a, b)；不是原文真实数据。")
    print("  ⚠️ 归一化基准 = 该实例上 NHDRL ∪ RN ∪ 27 CDR 的全部解；参考点 (1,1)。")


if __name__ == "__main__":
    main()
