# -*- coding: utf-8 -*-
"""Pilot 策略 vs RN vs 36 CDR —— 同一组 MK01–10、同一布局、同一场景。

口径与 `a_multi.py` 的评测一致：

- MK01–10，`layout_seed=200+i`；
- `evaluate(..., n=2, seed=1, dt_events=20)` ⟹ 场景 `seed_chain=1,2`；
- 三目标 `(makespan, energy, tardy_twt)`。

RN = **全 agent 随机动作**：PA 随机池成员 + 随机候选机台；LA 随机车；RA 随机 NOW/DEFER。
CDR = **12 PA × 3 LA = 36 条固定规则**；RA 走规则驱动（不给 `policy_m`）。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from plcsp_v3.env.reward import objective_vector  # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy  # noqa: E402
from plcsp_v3.hrl.cdr import cdr_pairs, run_cdr  # noqa: E402
from plcsp_v3.q1_flat import build, evaluate  # noqa: E402

MK_NAMES = tuple(f"mk{i:02d}" for i in range(1, 11))
N_SC = 2
SC0 = 1
N_AGV = 3
DT = 20
N_RN = 10
CKPTS = {
    "final": os.path.join(os.environ.get("TEMP", ""), "v3_pilot500_seed0.pt"),
    "peak375": os.path.join(os.environ.get("TEMP", ""),
                            "v3_pilot500_seed0_e0375.pt"),
}
# ⚠️ 关保养的 CDR 基线（与 `--no-maintenance` 训练档对齐）
_NO_MAINT = os.environ.get("V3_BASE_NO_MAINT", "") not in ("", "0")
# ⚠️ 只算 36 条 CDR（跳过 policy / RN）
_CDR_ONLY = os.environ.get("V3_BASE_CDR_ONLY", "") not in ("", "0")


def _specs():
    out = []
    cons = None
    if _NO_MAINT:
        from plcsp_v3.env.constraints import ConstraintConfig
        cons = ConstraintConfig(maintenance=False)
    for i, name in enumerate(MK_NAMES):
        sp, _ = build(seed=0, instance=name, tau=None, pm_interval=30,
                      target_failures=4, n_agv=N_AGV, finite_buffer=True,
                      layout_seed=200 + i, reward_norm="minmax",
                      constraints=cons)
        out.append((name, sp))
    return out


def _mean(rows):
    a = np.asarray(rows, dtype=float)
    return a.mean(axis=0)


def _policy(ckpt, specs):
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = HierarchicalPolicy()
    net.load_state_dict(obj["state_dict"])
    net.eval()
    rows = []
    for name, sp in specs:
        r = evaluate(net, sp, n=N_SC, seed=SC0, dt_events=DT)
        rows.append((r["makespan"], r["energy"], r["twt"]))
    return _mean(rows)


def _rn_policy(rng):
    def pa(view, pool, tie_rng=None):
        pi = int(rng.integers(len(pool)))
        alts = view.jobs[pool[pi][0]].alts
        m = int(alts[int(rng.integers(len(alts)))][0])
        return pi, m

    def pl(view, cand_v):
        return int(cand_v[int(rng.integers(len(cand_v)))])

    def pm(snap, mach, cands):
        return int(cands[int(rng.integers(len(cands)))])

    return pa, pl, pm


def _rn(specs):
    per_mk = []
    for name, sp in specs:
        vals = []
        for si in range(N_RN):
            pa, pl, pm = _rn_policy(np.random.default_rng([777, si]))
            for k in range(N_SC):
                m = sp.world().run_gated(
                    seed_chain=SC0 + k, policy_pa=pa, policy_l=pl,
                    policy_m=pm)
                vals.append(objective_vector(m))
        per_mk.append((name, _mean(vals)))
    return per_mk


def _cdr(specs):
    rows = []
    for pa, la in cdr_pairs(12):                 # 12 × 3 = 36
        per_mk = []
        for _name, sp in specs:
            vals = [objective_vector(run_cdr(sp.world(), pa, la,
                                             seed_chain=SC0 + k))
                    for k in range(N_SC)]
            per_mk.append(np.mean(vals, axis=0))
        per_mk = np.asarray(per_mk, dtype=float)   # (10, 3)
        rows.append((pa, la, *per_mk.mean(0), per_mk))
    return rows


def main() -> None:
    t0 = time.perf_counter()
    specs = _specs()
    print(f"**pilot baselines** · MK01–10 · layout 200+i · 场景 {SC0}..{SC0 + N_SC - 1}"
          f" · RN {N_RN} 随机种子\n", flush=True)

    for tag, ck in CKPTS.items():
        if _CDR_ONLY:
            break
        if not os.path.exists(ck):
            print(f"  ⚠️ 缺 ckpt {ck}", flush=True)
            continue
        v = _policy(ck, specs)
        print(f"  policy {tag:<8} MS={v[0]:7.1f}  E={v[1]:6.2f}  TWT={v[2]:8.1f}",
              flush=True)

    if not _CDR_ONLY:
        rn = _rn(specs)
        v = np.mean([x for _n, x in rn], axis=0)
        print(f"  RN             MS={v[0]:7.1f}  E={v[1]:6.2f}  TWT={v[2]:8.1f}",
              flush=True)

    cdr = _cdr(specs)
    cdr_sorted = sorted(cdr, key=lambda r: r[2])
    best = cdr_sorted[0]
    print(f"\n  CDR 最好单条（按 MS 均值）: {best[0]}+{best[1]}  "
          f"MS={best[2]:.1f}  E={best[3]:.2f}  TWT={best[4]:.1f}", flush=True)
    # 每实例 oracle：每个 MK 从 36 条里取 MS 最小的那条
    oracle = []
    for mi in range(len(MK_NAMES)):
        cand = [r[5][mi] for r in cdr]
        oracle.append(min(cand, key=lambda x: x[0]))
    oracle = np.asarray(oracle, dtype=float).mean(axis=0)
    print(f"  CDR 每实例最好（oracle）: MS={oracle[0]:.1f}  "
          f"E={oracle[1]:.2f}  TWT={oracle[2]:.1f}", flush=True)
    print(f"  CDR 36 条：MS [{cdr_sorted[0][2]:.1f}, {cdr_sorted[-1][2]:.1f}]  "
          f"均值 {np.mean([r[2] for r in cdr]):.1f}", flush=True)

    print(f"\n  {'PA':<12}{'LA':<20}{'MS':>8}{'E':>8}{'TWT':>10}")
    print("  " + "-" * 58)
    for pa, la, ms, en, tw, _per_mk in cdr_sorted:
        print(f"  {pa:<12}{la:<20}{ms:>8.1f}{en:>8.2f}{tw:>10.1f}", flush=True)
    print(f"\n  总用时 {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
