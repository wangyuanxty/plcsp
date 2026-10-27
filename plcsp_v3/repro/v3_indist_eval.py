# -*- coding: utf-8 -*-
"""**同分布留出集上的评测**：训练到底有没有把策略变好？

══ 为什么必须有这一步 ══

此前所有 eval 都是 **MK 零样本**。它同时混了两件事：

| 现象 | 同分布 | MK |
|---|---|---|
| **过拟合** | 变好 | 变差 |
| **优化失败** | 也变差 | 变差 |

不评同分布集就分不开。

══ 口径 ══

`gen_mk_like(seed=5000+i)` · layout 500+i · tau=2.0 · due_range=0.6
· n_agv=3 · **保养关** · 场景 1 · `T` 条平局流。

⚠️ §38.18 的 255.3 是**保养开**的读数，**不能**与本脚本比。本脚本自带 CDR 基线。

══ 用法 ══

`D:/anaconda/python.exe -m plcsp_v3.repro.v3_indist_eval`
环境变量：`V3_IE_T`（默认 5）· `V3_IE_CKPT`（逗号分隔，可选）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from plcsp_v3.env.constraints import ConstraintConfig        # noqa: E402
from plcsp_v3.env.instances import gen_mk_like               # noqa: E402
from plcsp_v3.env.reward import objective_vector              # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy            # noqa: E402
from plcsp_v3.hrl.cdr import cdr_pairs, run_cdr               # noqa: E402
from plcsp_v3.q1_flat import TIE_STREAM_BASE, build, evaluate  # noqa: E402

T = int(os.environ.get("V3_IE_T", "5"))
N_VAL = 10
VAL_SEED = 5000
LAYOUT_BASE = 500
N_AGV = 3
DT = 20
SC0 = 1

DEFAULT = {
    "init": "v3_mafix200_seed0.pt",
    "trained200": "v3_exact_fix_200.pt",
    "frozen100": "v3_exact_frozen_100.pt",
    "scratch300": "v3_exact_scratch_300.pt",
    "term100": "v3_exact_term_100.pt",
}
_raw = os.environ.get("V3_IE_CKPT", "").strip()
if _raw:
    DEFAULT = {os.path.splitext(os.path.basename(p))[0]: p
               for p in _raw.split(",") if p.strip()}
TEMP = os.environ.get("TEMP", "")


def _specs():
    cons = ConstraintConfig(maintenance=False)
    out = []
    for i in range(N_VAL):
        inst = gen_mk_like(seed=VAL_SEED + i)
        sp, _ = build(seed=0, inst_obj=inst, tau=2.0, pm_interval=30,
                      target_failures=4, n_agv=N_AGV, finite_buffer=True,
                      layout_seed=LAYOUT_BASE + i, reward_norm="minmax",
                      sim_overrides={"due_range": 0.6}, constraints=cons)
        out.append(sp)
    return out


def main() -> None:
    specs = _specs()
    print(f"**同分布留出集** · {N_VAL} 实例 · layout 500+i · 场景 {SC0}"
          f" · 保养关 · CDR {T} 条平局流\n", flush=True)

    pairs = cdr_pairs(12)
    rows = np.zeros((len(pairs), len(specs), 3))
    for k, (pa, la) in enumerate(pairs):
        for i, sp in enumerate(specs):
            vals = [objective_vector(run_cdr(sp.world(), pa, la, seed_chain=SC0,
                                             tie_seed=TIE_STREAM_BASE + j))
                    for j in range(T)]
            rows[k, i] = np.mean(vals, axis=0)
    best = int(np.argmin(rows[:, :, 0].mean(axis=1)))
    print(f"  最好单条 CDR：{pairs[best][0]}+{pairs[best][1]}  "
          f"MS {rows[best, :, 0].mean():.1f}  E {rows[best, :, 1].mean():.2f}"
          f"  TWT {rows[best, :, 2].mean():.1f}", flush=True)
    oracle = np.array([rows[int(np.argmin(rows[:, i, 0])), i]
                       for i in range(len(specs))]).mean(axis=0)
    print(f"  CDR 每实例 oracle：MS {oracle[0]:.1f}", flush=True)

    print(f"\n  {'ckpt':<28}{'MS':>9}{'E':>8}{'TWT':>9}{'vs best':>10}",
          flush=True)
    print("  " + "-" * 64, flush=True)
    for name, fn in DEFAULT.items():
        path = fn if os.path.isabs(fn) else os.path.join(TEMP, fn)
        if not os.path.exists(path):
            print(f"  {name:<28} ⚠️ 缺文件", flush=True)
            continue
        obj = torch.load(path, map_location="cpu", weights_only=False)
        net = HierarchicalPolicy(seq_ma=bool(obj.get("args", {}).get("seq_ma")))
        net.load_state_dict(obj["state_dict"])
        net.eval()
        ms = []
        for sp in specs:
            r = evaluate(net, sp, n=1, seed=SC0, dt_events=DT,
                         fixed_rules={"RA": 0}, pa_limit=9,
                         exact_rule=True, tie_streams=T)
            ms.append((r["makespan"], r["energy"], r["twt"]))
        m = np.asarray(ms).mean(axis=0)
        print(f"  {name:<28}{m[0]:>9.1f}{m[1]:>8.2f}{m[2]:>9.1f}"
              f"{m[0] - rows[best, :, 0].mean():>+10.1f}", flush=True)


if __name__ == "__main__":
    main()
