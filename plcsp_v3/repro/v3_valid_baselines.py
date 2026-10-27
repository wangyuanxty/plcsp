# -*- coding: utf-8 -*-
"""同分布留出集：span1 / span3 vs 36 CDR vs RN。

10 个留出随机实例（`gen_mk_like(seed=5000+i)`，layout 500+i），场景 1–2。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from plcsp_v3.env.instances import gen_mk_like  # noqa: E402
from plcsp_v3.env.reward import objective_vector  # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy  # noqa: E402
from plcsp_v3.hrl.cdr import cdr_pairs, run_cdr  # noqa: E402
from plcsp_v3.q1_flat import build, evaluate  # noqa: E402

CKPTS = {
    "span1": os.path.join(os.environ.get("TEMP", ""), "v3_mafix200_seed0.pt"),
    "span3": os.path.join(os.environ.get("TEMP", ""), "v3_span3_200_seed0.pt"),
}
# ⚠️ 诊断任意 ckpt 时用：`V3_VAL_CKPT=<path>`。
_CUSTOM = os.environ.get("V3_VAL_CKPT", "")
if _CUSTOM:
    CKPTS = {"custom": _CUSTOM}
# ⚠️ 只算策略、不算 RN/CDR 时置 1（CDR 扫 36 条 × 10 实例，慢）。
_SKIP_BASE = os.environ.get("V3_VAL_SKIP_BASELINES", "") not in ("", "0")
# ⚠️ 单训 PA/LA 诊断：固定另一个头、限制 PA 规则数、关保养。
_PA_LIMIT = int(os.environ.get("V3_VAL_PA_LIMIT", "0")) or None
_NO_MAINT = os.environ.get("V3_VAL_NO_MAINT", "") not in ("", "0")
_FIXED_RULES: dict[str, int] = {}
if os.environ.get("V3_VAL_FIXED_LA"):
    from plcsp_v3.hrl.rules import LA_RULE_NAMES
    _FIXED_RULES["LA"] = LA_RULE_NAMES.index(os.environ["V3_VAL_FIXED_LA"])
    _FIXED_RULES["RA"] = 0
if os.environ.get("V3_VAL_FIXED_PA"):
    from plcsp_v3.hrl.rules import PA_RULE_NAMES
    _FIXED_RULES["PA"] = PA_RULE_NAMES.index(os.environ["V3_VAL_FIXED_PA"])
    _FIXED_RULES["RA"] = 0
N_VAL = 10
VAL_SEED = 5000
LAYOUT_BASE = 500
N_AGV = 3
DT = 20
N_SC = 2
N_RN = 10


def _specs():
    out = []
    cons = None
    if _NO_MAINT:
        from plcsp_v3.env.constraints import ConstraintConfig
        cons = ConstraintConfig(maintenance=False)
    for i in range(N_VAL):
        inst = gen_mk_like(seed=VAL_SEED + i)
        sp, _ = build(seed=0, inst_obj=inst, tau=2.0, pm_interval=30,
                      target_failures=4, n_agv=N_AGV, finite_buffer=True,
                      layout_seed=LAYOUT_BASE + i, reward_norm="minmax",
                      sim_overrides={"due_range": 0.6}, constraints=cons)
        out.append((f"val{i}", sp))
    return out


def _policy(ck, specs):
    obj = torch.load(ck, map_location="cpu", weights_only=False)
    net = HierarchicalPolicy()
    net.load_state_dict(obj["state_dict"])
    net.eval()
    f = []
    for _n, sp in specs:
        r = evaluate(net, sp, n=N_SC, seed=1, dt_events=DT,
                     fixed_rules=(_FIXED_RULES or None), pa_limit=_PA_LIMIT)
        f.append((r["makespan"], r["energy"], r["twt"]))
    return np.asarray(f, dtype=float)


def _rn_policy(rng):
    def pa(view, pool, tie_rng=None):
        pi = int(rng.integers(len(pool)))
        alts = view.jobs[pool[pi][0]].alts
        return pi, int(alts[int(rng.integers(len(alts)))][0])

    def pl(view, cand_v):
        return int(cand_v[int(rng.integers(len(cand_v)))])

    def pm(snap, mach, cands):
        return int(cands[int(rng.integers(len(cands)))])

    return pa, pl, pm


def _rn(specs):
    out = []
    for _n, sp in specs:
        vals = []
        for si in range(N_RN):
            pa, pl, pm = _rn_policy(np.random.default_rng([777, si]))
            for k in range(N_SC):
                m = sp.world().run_gated(seed_chain=1 + k, policy_pa=pa,
                                         policy_l=pl, policy_m=pm)
                vals.append(objective_vector(m))
        out.append(np.mean(vals, axis=0))
    return np.asarray(out, dtype=float)


def _cdr(specs):
    pairs = cdr_pairs(12)
    agg = []
    per_mk = []
    for pa, la in pairs:
        rows = []
        for _n, sp in specs:
            vals = [objective_vector(run_cdr(sp.world(), pa, la, seed_chain=1 + k))
                    for k in range(N_SC)]
            rows.append(np.mean(vals, axis=0))
        rows = np.asarray(rows, dtype=float)
        per_mk.append(rows)
        agg.append((pa, la, *rows.mean(0)))
    return pairs, np.asarray(per_mk, dtype=float), agg


def main() -> None:
    specs = _specs()
    print(f"**同分布留出对照** · {N_VAL} 个随机实例 · 场景 1–{N_SC}\n", flush=True)
    if _SKIP_BASE:
        print(f"  {'method':<26}{'MS':>9}{'E':>8}{'TWT':>10}", flush=True)
        print("  " + "-" * 54, flush=True)
        for tag, ck in CKPTS.items():
            if not os.path.exists(ck):
                print(f"  ⚠️ 缺 ckpt {ck}", flush=True)
                continue
            m = _policy(ck, specs).mean(0)
            print(f"  {tag:<26}{m[0]:>9.1f}{m[1]:>8.2f}{m[2]:>10.1f}",
                  flush=True)
        return
    pol = {}
    for tag, ck in CKPTS.items():
        if os.path.exists(ck):
            pol[tag] = _policy(ck, specs)
    rn = _rn(specs)
    _pairs, cdr_pm, cdr_agg = _cdr(specs)       # cdr_pm: (36, 10, 3)
    best_i = int(np.argmin(cdr_pm[:, :, 0].mean(axis=1)))
    # per-instance oracle: 取 MS 最小的那条 CDR 的三个目标
    oracle_rows = []
    for mi in range(len(specs)):
        k = int(np.argmin(cdr_pm[:, mi, 0]))
        oracle_rows.append(cdr_pm[k, mi])
    oracle = np.asarray(oracle_rows, dtype=float)

    print(f"  {'method':<26}{'MS':>9}{'E':>8}{'TWT':>10}", flush=True)
    print("  " + "-" * 54, flush=True)
    for tag in ("span1", "span3"):
        if tag in pol:
            m = pol[tag].mean(0)
            print(f"  {tag:<26}{m[0]:>9.1f}{m[1]:>8.2f}{m[2]:>10.1f}",
                  flush=True)
    m = rn.mean(0)
    print(f"  {'RN':<26}{m[0]:>9.1f}{m[1]:>8.2f}{m[2]:>10.1f}", flush=True)
    pa, la, *v = cdr_agg[best_i]
    print(f"  {'best CDR: ' + pa + '+' + la:<26}{v[0]:>9.1f}{v[1]:>8.2f}"
          f"{v[2]:>10.1f}", flush=True)
    m = oracle.mean(0)
    print(f"  {'CDR per-instance oracle':<26}{m[0]:>9.1f}{m[1]:>8.2f}"
          f"{m[2]:>10.1f}", flush=True)

    if "span3" in pol:
        s3 = pol["span3"]
        print(f"\n  span3 vs best CDR: MS {s3[:, 0].mean() - cdr_pm[best_i, :, 0].mean():+.1f}"
              f"  E {s3[:, 1].mean() - cdr_pm[best_i, :, 1].mean():+.2f}"
              f"  TWT {s3[:, 2].mean() - cdr_pm[best_i, :, 2].mean():+.1f}",
              flush=True)
        win = int((s3[:, 0] < cdr_pm[best_i, :, 0]).sum())
        print(f"  span3 在 MS 上赢 best CDR 的实例：{win}/{len(specs)}", flush=True)
        win_o = int((s3[:, 0] < oracle[:, 0]).sum())
        print(f"  span3 在 MS 上赢 per-instance oracle 的实例：{win_o}/{len(specs)}",
              flush=True)


if __name__ == "__main__":
    main()
