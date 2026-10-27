# -*- coding: utf-8 -*-
"""同分布留出验证：span=1 vs span=3。

10 个**留出的随机实例**（`gen_mk_like(seed=5000+i)`，layout 500+i），
与训练同分布，但不参与训练；场景 1–2；greedy。

用途：区分
- span=3 在训练分布上更好、在 MK 上更差 ⟹ 零样本分布偏移；
- span=3 在两边都更差 ⟹ 目标/训练不对齐。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from plcsp_v3.env.instances import gen_mk_like  # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy  # noqa: E402
from plcsp_v3.q1_flat import build, evaluate  # noqa: E402

CKPTS = {
    "span1": os.path.join(os.environ.get("TEMP", ""), "v3_mafix200_seed0.pt"),
    "span3": os.path.join(os.environ.get("TEMP", ""), "v3_span3_200_seed0.pt"),
}
N_VAL = 10
VAL_SEED = 5000
LAYOUT_BASE = 500
N_AGV = 3
DT = 20
N_SC = 2


def _specs():
    out = []
    for i in range(N_VAL):
        inst = gen_mk_like(seed=VAL_SEED + i)
        sp, _ = build(seed=0, inst_obj=inst, tau=2.0, pm_interval=30,
                      target_failures=4, n_agv=N_AGV, finite_buffer=True,
                      layout_seed=LAYOUT_BASE + i, reward_norm="minmax",
                      sim_overrides={"due_range": 0.6})
        out.append((f"val{i}", sp))
    return out


def _eval(ckpt, specs):
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = HierarchicalPolicy()
    net.load_state_dict(obj["state_dict"])
    net.eval()
    rows = []
    for name, sp in specs:
        r = evaluate(net, sp, n=N_SC, seed=1, dt_events=DT)
        rows.append((name, sp.inst.n_jobs, sp.inst.n_machines,
                     r["makespan"], r["energy"], r["twt"]))
    return rows


def main() -> None:
    specs = _specs()
    print(f"**同分布留出验证** · {N_VAL} 个随机实例 "
          f"(`gen_mk_like` seed {VAL_SEED}+i · layout {LAYOUT_BASE}+i) · "
          f"场景 1–{N_SC}\n", flush=True)
    res = {}
    for tag, ck in CKPTS.items():
        if not os.path.exists(ck):
            print(f"  ⚠️ 缺 ckpt {ck}", flush=True)
            continue
        res[tag] = _eval(ck, specs)
        print(f"{tag}: {ck}", flush=True)

    if len(res) < 2:
        return
    a = np.asarray([[r[3], r[4], r[5]] for r in res["span1"]])
    b = np.asarray([[r[3], r[4], r[5]] for r in res["span3"]])
    print(f"\n  {'instance':<9}{'jobs':>5}{'mach':>5}"
          f"{'s1 MS':>9}{'s3 MS':>9}{'s1 E':>8}{'s3 E':>8}"
          f"{'s1 TWT':>9}{'s3 TWT':>9}", flush=True)
    print("  " + "-" * 73, flush=True)
    r1 = res["span1"]
    r3 = res["span3"]
    for i in range(len(r1)):
        n, nj, nm = r1[i][0], r1[i][1], r1[i][2]
        print(f"  {n:<9}{nj:>5}{nm:>5}"
              f"{r1[i][3]:>9.1f}{r3[i][3]:>9.1f}"
              f"{r1[i][4]:>8.2f}{r3[i][4]:>8.2f}"
              f"{r1[i][5]:>9.1f}{r3[i][5]:>9.1f}", flush=True)
    print("  " + "-" * 73, flush=True)
    m1, m3 = a.mean(0), b.mean(0)
    print(f"  {'mean':<9}{'':>10}{m1[0]:>9.1f}{m3[0]:>9.1f}"
          f"{m1[1]:>8.2f}{m3[1]:>8.2f}{m1[2]:>9.1f}{m3[2]:>9.1f}", flush=True)
    print(f"\n  span3 − span1：MS {m3[0] - m1[0]:+.1f}（{(m3[0] / m1[0] - 1):+.1%}）"
          f"  E {m3[1] - m1[1]:+.2f}（{(m3[1] / m1[1] - 1):+.1%}）"
          f"  TWT {m3[2] - m1[2]:+.1f}（{(m3[2] / m1[2] - 1):+.1%}）",
          flush=True)
    win = int((b[:, 0] < a[:, 0]).sum())
    print(f"  span3 在 MS 上更好的实例：{win}/{len(a)}", flush=True)


if __name__ == "__main__":
    main()
