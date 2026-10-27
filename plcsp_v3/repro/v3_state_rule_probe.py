# -*- coding: utf-8 -*-
"""状态-规则对齐探针：同一顶层点上，108 组规则的段回报谁最好？

对每个顶层点 `Z_j`，跑全部 `12 PA × 3 LA × 3 RA = 108` 组规则，
比较段回报的三种口径：

- `scalar`：MA 训练实际用的标量（`train._scalar`）；
- `d_ms`：段内 MS 增量（越小越好）；
- `twt`：段内拖期（越小越好）。

若最优 PA 规则**随 j 变** ⟹ 状态-规则映射存在，MA 需要探索/熵。
若最优 PA 规则**恒定** ⟹ MA 的常数 argmax 是训练信号下的正确解，
零样本差距来自分布偏移。

只用评测，不训练。用修复后的 pilot ckpt（`v3_mafix200_seed0.pt`）。
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
from plcsp_v3.hrl.pool import SegmentPool  # noqa: E402
from plcsp_v3.hrl.replay import segment_return  # noqa: E402
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, PA_RULE_NAMES,  # noqa: E402
                                PM_RULE_NAMES)
from plcsp_v3.hrl.train import (_run_all_segments, _scalar, base_chain)  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

CKPT = os.environ.get(
    "V3_DIAG_CKPT",
    os.path.join(os.environ.get("TEMP", ""), "v3_mafix200_seed0.pt"))
DT = 20
N_WORKERS = int(os.environ.get("PROBE_WORKERS", "8"))
SPAN = int(os.environ.get("PROBE_SPAN", "1"))
N_AGV = 3


def _train_spec():
    inst = gen_mk_like(seed=0)
    sp, _ = build(seed=0, inst_obj=inst, tau=2.0, pm_interval=30,
                  target_failures=4, n_agv=N_AGV, finite_buffer=True,
                  layout_seed=100, reward_norm="minmax",
                  sim_overrides={"due_range": 0.6})
    return "train0", sp


def _mk_spec(name: str, idx: int):
    sp, _ = build(seed=0, instance=name, tau=None, pm_interval=30,
                  target_failures=4, n_agv=N_AGV, finite_buffer=True,
                  layout_seed=200 + idx, reward_norm="minmax")
    return name, sp


def _best(rules, vals, *, minimize: bool = False):
    i = int(np.argmin(vals) if minimize else np.argmax(vals))
    return rules[i], float(vals[i])


def _probe(tag, spec, net):
    base = base_chain(net, spec, greedy=True, seed=0, dt_events=DT,
                      seed_chain=100_001)
    n_top = len(base.ma_counters)
    rules = [(rp, rl, rr) for rp in range(len(PA_RULE_NAMES))
             for rl in range(len(LA_RULE_NAMES))
             for rr in range(len(PM_RULE_NAMES))]
    plan = [(j, [(r, 0, True) for r in rules]) for j in range(n_top)]
    pool = SegmentPool(net, spec, n_workers=N_WORKERS, dt_events=DT)
    try:
        all_segs = _run_all_segments(net, spec, base, plan, dt_events=DT,
                                     pool=pool, span=SPAN)
    finally:
        pool.close()

    print(f"\n══ {tag} · {spec.inst.n_jobs}×{spec.inst.n_machines} · "
          f"n_top={n_top} · 108 规则/点 · span={SPAN} ══", flush=True)
    print(f"  {'j':>3}{'best scalar PA':>22}{'scalar':>10}"
          f"{'best MS PA':>22}{'d_ms':>8}{'best TWT PA':>22}{'twt':>8}",
          flush=True)
    best_scalar_pa, best_ms_pa, best_twt_pa = [], [], []
    gaps = []
    gaps_pa, gaps_la, gaps_ra = [], [], []
    for j in range(n_top):
        segs = [segment_return(base.ma_counters[j], br.counters)
                for br in all_segs[j]]
        scal = np.asarray([_scalar(sg, spec.w) for sg in segs])
        dms = np.asarray([sg.d_ms for sg in segs])
        twt = np.asarray([sg.twt for sg in segs])
        r_s, v_s = _best(rules, scal)
        r_m, v_m = _best(rules, dms, minimize=True)
        r_t, v_t = _best(rules, twt, minimize=True)
        best_scalar_pa.append(r_s[0])
        best_ms_pa.append(r_m[0])
        best_twt_pa.append(r_t[0])
        order = np.sort(scal)[::-1]
        gaps.append(float(order[0] - order[1]) if len(order) > 1 else 0.0)
        # ⚠️ 分头看：全 108 的 top-2 可能只是同一 PA、不同 LA/RA ⟹ 假平局。
        S = scal.reshape(len(PA_RULE_NAMES), len(LA_RULE_NAMES),
                         len(PM_RULE_NAMES))
        pa_max = S.max(axis=(1, 2))
        la_max = S.max(axis=(0, 2))
        ra_max = S.max(axis=(0, 1))
        gaps_pa.append(float(np.sort(pa_max)[-1] - np.sort(pa_max)[-2]))
        gaps_la.append(float(np.sort(la_max)[-1] - np.sort(la_max)[-2]))
        gaps_ra.append(float(np.sort(ra_max)[-1] - np.sort(ra_max)[-2]))
        print(f"  {j:>3}{PA_RULE_NAMES[r_s[0]]:>22}{v_s:>10.4f}"
              f"{PA_RULE_NAMES[r_m[0]]:>22}{v_m:>8.1f}"
              f"{PA_RULE_NAMES[r_t[0]]:>22}{v_t:>8.1f}", flush=True)

    def _summary(name, arr):
        cnt = {PA_RULE_NAMES[k]: arr.count(k) for k in sorted(set(arr))}
        print(f"  {name}: " + "  ".join(f"{k}×{v}" for k, v in cnt.items()),
              flush=True)
        non = sum(1 for k in arr if PA_RULE_NAMES[k] != "EAJ+EAM")
        print(f"    非 EAJ+EAM 占比：{non}/{len(arr)} = {non / len(arr):.0%}",
              flush=True)

    _summary("best scalar PA", best_scalar_pa)
    _summary("best d_ms PA", best_ms_pa)
    _summary("best twt PA", best_twt_pa)
    print(f"  段回报 best−second 的均值：{np.mean(gaps):.4f}", flush=True)
    print(f"  分头 best−second 均值：PA {np.mean(gaps_pa):.4f}  "
          f"LA {np.mean(gaps_la):.4f}  RA {np.mean(gaps_ra):.4f}", flush=True)


def main() -> None:
    if not os.path.exists(CKPT):
        raise SystemExit(f"⚠️ 找不到 ckpt {CKPT}")
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = HierarchicalPolicy()
    net.load_state_dict(obj["state_dict"])
    net.eval()
    print(f"ckpt = {CKPT}", flush=True)
    for tag, sp in (_train_spec(), _mk_spec("mk01", 0)):
        _probe(tag, sp, net)


if __name__ == "__main__":
    main()
