# -*- coding: utf-8 -*-
"""Pilot 规则分布诊断：MA / PA / LA / RA 各选了什么。

同一组 MK01–10、`layout_seed=200+i`、场景 1–2、greedy、`dt_events=20`。
对比 final（500 步）与 peak（375 步）两个 ckpt。
"""
from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch  # noqa: E402

from plcsp_v3.hrl.agents import HierarchicalPolicy  # noqa: E402
from plcsp_v3.hrl.rollout import NetPolicy, gated_by_constraints  # noqa: E402
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, PA_RULE_NAMES,  # noqa: E402
                                PM_RULE_NAMES)
from plcsp_v3.q1_flat import build  # noqa: E402

MK_NAMES = tuple(f"mk{i:02d}" for i in range(1, 11))
N_SC = 2
SC0 = 1
N_AGV = 3
DT = 20
_CUSTOM = os.environ.get("V3_DIAG_CKPT", "")
# ⚠️ 诊断当前 `--force-top1` 训练档时置 1；默认 0 = 用打分头（历史口径）。
_FORCE_TOP1 = os.environ.get("V3_FORCE_TOP1", "") not in ("", "0")
# ⚠️ 9 条 PA 规则档：只在前 9 条里采样（与 `--pa-limit 9` 训练档一致）。
_PA_LIMIT = int(os.environ.get("V3_DIAG_PA_LIMIT", "0")) or None
# ⚠️ 顺序条件 MA 的 ckpt：置 1 时按 `seq_ma=True` 建网。
_SEQ_MA = os.environ.get("V3_DIAG_SEQ_MA", "") not in ("", "0")
# ⚠️ 单训 PA/LA 诊断：固定另一个头、关保养。
_NO_MAINT = os.environ.get("V3_DIAG_NO_MAINT", "") not in ("", "0")
_FIXED_RULES: dict[str, int] = {}
if os.environ.get("V3_DIAG_FIXED_LA"):
    _FIXED_RULES["LA"] = LA_RULE_NAMES.index(os.environ["V3_DIAG_FIXED_LA"])
    _FIXED_RULES["RA"] = 0
if os.environ.get("V3_DIAG_FIXED_PA"):
    _FIXED_RULES["PA"] = PA_RULE_NAMES.index(os.environ["V3_DIAG_FIXED_PA"])
    _FIXED_RULES["RA"] = 0
CKPTS = ({"custom": _CUSTOM} if _CUSTOM else {
    "final": os.path.join(os.environ.get("TEMP", ""), "v3_pilot500_seed0.pt"),
    "peak375": os.path.join(os.environ.get("TEMP", ""),
                            "v3_pilot500_seed0_e0375.pt"),
})
EDD_START = PA_RULE_NAMES.index("EDD+SMPT")


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


def _count(net, specs):
    c = {k: Counter() for k in ("ma_pa", "ma_la", "ma_ra",
                                "pa", "la", "ra", "combo")}
    per_mk_edd = {}
    for name, sp in specs:
        for k in range(N_SC):
            p = NetPolicy(net, sp.inst, sp.layout, sp.ctx, greedy=True,
                          seed=0, dt_events=DT, seed_chain=SC0 + k,
                          force_top1=_FORCE_TOP1,
                          fixed_rules=(_FIXED_RULES or None),
                          pa_limit=_PA_LIMIT)
            sp.world().run_gated(
                seed_chain=SC0 + k, policy_pa=p.policy_pa,
                policy_l=p.policy_l, policy_ma=p.policy_ma, dt_events=DT,
                **gated_by_constraints(p, sp.constraints))
            for d in p.rec.decisions:
                if d.kind == "MA":
                    c["ma_pa"][int(d.rule_pa)] += 1
                    c["ma_la"][int(d.rule_la)] += 1
                    c["ma_ra"][int(d.rule_ra)] += 1
                    c["combo"][(int(d.rule_pa), int(d.rule_la),
                                int(d.rule_ra))] += 1
                elif d.kind == "PA":
                    c["pa"][int(d.rule_pa)] += 1
                elif d.kind == "LA":
                    c["la"][int(d.rule_la)] += 1
                elif d.kind == "M":
                    c["ra"][int(d.rule_ra)] += 1
            edd = sum(1 for d in p.rec.decisions
                      if d.kind == "PA" and int(d.rule_pa) >= EDD_START)
            tot = sum(1 for d in p.rec.decisions if d.kind == "PA")
            per_mk_edd[name + f"/sc{k}"] = (edd, tot)
    return c, per_mk_edd


def _show(title, counter, names):
    tot = sum(counter.values())
    print(f"\n  {title}（{tot} 个决策）", flush=True)
    if not tot:
        print("    —— 无", flush=True)
        return
    for i, n in counter.most_common():
        print(f"    {names[i]:<22}{n:>7}{n / tot:>9.1%}", flush=True)


def main() -> None:
    specs = _specs()
    print("**规则分布** · MK01–10 · layout 200+i · 场景 1–2 · greedy", flush=True)
    for tag, ck in CKPTS.items():
        if not os.path.exists(ck):
            print(f"\n  ⚠️ 缺 ckpt {ck}", flush=True)
            continue
        obj = torch.load(ck, map_location="cpu", weights_only=False)
        net = HierarchicalPolicy(seq_ma=_SEQ_MA)
        net.load_state_dict(obj["state_dict"])
        net.eval()
        c, per_mk = _count(net, specs)
        print(f"\n══ {tag} · {ck} ══", flush=True)
        _show("MA 选的 PA 规则", c["ma_pa"], PA_RULE_NAMES)
        _show("MA 选的 LA 规则", c["ma_la"], LA_RULE_NAMES)
        _show("MA 选的 RA 规则", c["ma_ra"], PM_RULE_NAMES)
        _show("PA 实际执行的规则", c["pa"], PA_RULE_NAMES)
        _show("LA 实际执行的规则", c["la"], LA_RULE_NAMES)
        _show("RA 实际执行的规则", c["ra"], PM_RULE_NAMES)
        pa_tot = sum(c["pa"].values())
        edd = sum(n for i, n in c["pa"].items() if i >= EDD_START)
        ma_tot = sum(c["ma_pa"].values())
        edd_ma = sum(n for i, n in c["ma_pa"].items() if i >= EDD_START)
        print(f"\n  EDD 占比：PA 决策 {edd}/{pa_tot} = {edd / max(pa_tot, 1):.1%}；"
              f"MA 选择 {edd_ma}/{ma_tot} = {edd_ma / max(ma_tot, 1):.1%}",
              flush=True)
        print("\n  MA 最常见组合（PA, LA, RA）：", flush=True)
        for (k, top) in enumerate(c["combo"].most_common(8), start=1):
            (rp, rl, rr), n = top
            print(f"    {k}. {PA_RULE_NAMES[rp]} + {LA_RULE_NAMES[rl]} + "
                  f"{PM_RULE_NAMES[rr]}  {n} 次", flush=True)
        zero = [name for name, (e, t) in per_mk.items() if e == 0 and t > 0]
        print(f"\n  完全没有 EDD 的 MK/场景：{len(zero)}/{len(per_mk)}", flush=True)


if __name__ == "__main__":
    main()
