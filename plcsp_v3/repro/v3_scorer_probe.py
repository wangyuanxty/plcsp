# -*- coding: utf-8 -*-
"""**打分头到底学到了什么**：逐决策点，它选的是不是规则的 argmin？

══ 为什么要测 ══

§38.23 的差距分解把 MK 上的剩余差距压到 **9.1 min**，全部落在打分头。
但"打分头差"还有两种完全不同的解释：

| 解释 | 表现 | 修法 |
|---|---|---|
| **没学会** | 选中的候选在规则序里近乎随机 | 阶段 B 的梯度/信号 |
| **学会了但目标不同** | 选中的候选有规则，但与 makespan 不一致 | 不是 bug，是奖励口径 |

本脚本只回答第一个：**逐 PA/LA 决策点，打分的 argmax 是否落在规则的并列最优集里**。
另外记两个辅助量：并列集大小（决定"选对"有多难）、选中者的规则名次。

⚠️ **本脚本不判断"选得对不对"**（那要看最终 makespan），只判断"是否跟规则一致"。
一致率低 ≠ 策略差：规则本身不是最优。但一致率**接近随机** ⟹ 打分头没学到东西。

══ 口径 ══

MK01–10 · layout 200+i · 场景 1–2 · **保养关** · greedy。
ckpt 由 `V3_PROBE_CKPT` 指定（默认主模型 span1）。
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from plcsp_v3.env.constraints import ConstraintConfig          # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy              # noqa: E402
from plcsp_v3.hrl.rollout import NetPolicy, gated_by_constraints  # noqa: E402
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, PA_JOB_RULES,     # noqa: E402
                                PA_MACHINE_RULES, la_score,
                                pa_job_score, pa_machine_score)
from plcsp_v3.q1_flat import build                               # noqa: E402

MK_NAMES = tuple(f"mk{i:02d}" for i in range(1, 11))
SC0 = 1
N_SC = 2
DT = 20
N_AGV = 3

CKPTS = {
    "span1": os.path.join(os.environ.get("TEMP", ""), "v3_mafix200_seed0.pt"),
    "pilot500": os.path.join(os.environ.get("TEMP", ""), "v3_pilot500_seed0.pt"),
    "pa9la3": os.path.join(os.environ.get("TEMP", ""), "v3_fact_pa9_la3_100.pt"),
}
_CUSTOM = os.environ.get("V3_PROBE_CKPT", "")
if _CUSTOM:
    CKPTS = {"custom": _CUSTOM}


def _specs():
    cons = ConstraintConfig(maintenance=False)
    out = []
    for i, name in enumerate(MK_NAMES):
        sp, _ = build(seed=0, instance=name, tau=None, pm_interval=30,
                      target_failures=4, n_agv=N_AGV, finite_buffer=True,
                      layout_seed=200 + i, reward_norm="minmax",
                      constraints=cons)
        out.append((name, sp))
    return out


def _pa_optimal(view, rule: int, pool) -> set[tuple[int, int]]:
    """`apply_pa_rule` 的**并列最优集**：先工件分最小，再该工件的机台分最小。"""
    n_mr = len(PA_MACHINE_RULES)
    jr = PA_JOB_RULES[int(rule) // n_mr]
    mr = PA_MACHINE_RULES[int(rule) % n_mr]
    js = {int(j): float(pa_job_score(jr, view, int(j))) for j, _oi in pool}
    best_js = min(js.values())
    best_jobs = [j for j, s in js.items() if s == best_js]
    ms: dict[tuple[int, int], float] = {}
    for j in best_jobs:
        for alt in view.jobs[j].alts:
            ms[(j, int(alt[0]))] = float(pa_machine_score(mr, view, alt))
    if not ms:
        return set()
    best_ms = min(ms.values())
    return {k for k, s in ms.items() if s == best_ms}


def _la_optimal(view, rule: int, cand_v) -> set[int]:
    name = LA_RULE_NAMES[int(rule)]
    s = {int(a): la_score(name, view, int(a)) for a in cand_v}
    best = min(s.values())
    return {a for a, v in s.items() if v == best}


class ProbePolicy(NetPolicy):
    """在正常策略外面套一层：每次 PA/LA 决策后记"是否规则最优"。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.pa_rows: list[tuple[int, bool, int]] = []   # (并列集大小, 是否最优, 候选数)
        self.la_rows: list[tuple[int, bool, int]] = []
        self.pa_prob: list[float] = []
        self.la_prob: list[float] = []

    def policy_pa(self, view, pool, tie_rng=None):
        n_cand = len(pool)
        pi, m = super().policy_pa(view, pool, tie_rng)
        job = int(pool[int(pi)][0])
        opt = _pa_optimal(view, self.cur["PA"], pool)
        self.pa_rows.append((len(opt), (job, int(m)) in opt, n_cand))
        return pi, m

    def policy_l(self, view, cand_v):
        a = super().policy_l(view, cand_v)
        opt = _la_optimal(view, self.cur["LA"], cand_v)
        self.la_rows.append((len(opt), int(a) in opt, len(cand_v)))
        return a


def _rollout(net, sp, seed_chain: int) -> ProbePolicy:
    p = ProbePolicy(net, sp.inst, sp.layout, sp.ctx, greedy=True, seed=0,
                    dt_events=DT, seed_chain=seed_chain)
    p.world = sp.world()                                    # type: ignore[attr-defined]
    p.metrics = p.world.run_gated(                           # type: ignore[attr-defined]
        seed_chain=seed_chain, policy_pa=p.policy_pa, policy_l=p.policy_l,
        policy_ma=p.policy_ma, dt_events=DT,
        **gated_by_constraints(p, sp.constraints))
    return p


def _report(tag: str, rows: list[tuple[int, bool, int]], kind: str) -> None:
    if not rows:
        print(f"  {tag:<22}{kind}: 没有决策点", flush=True)
        return
    ties = np.asarray([r[0] for r in rows])
    hit = np.asarray([r[1] for r in rows])
    nc = np.asarray([r[2] for r in rows])
    print(f"  {tag:<22}{kind}: 决策 {len(rows):>5}　"
          f"规则最优率 {hit.mean():>6.1%}　"
          f"并列集 均值 {ties.mean():>4.1f}（=1 的占 "
          f"{(ties == 1).mean():>5.1%}）　"
          f"候选数 均值 {nc.mean():>4.1f}", flush=True)


def main() -> None:
    specs = _specs()
    print("**打分头 vs 规则 argmin** · MK01–10 · layout 200+i · 场景 1–2"
          " · 保养关 · greedy\n", flush=True)
    for tag, ck in CKPTS.items():
        if not os.path.exists(ck):
            print(f"  ⚠️ 缺 ckpt {ck}", flush=True)
            continue
        obj = torch.load(ck, map_location="cpu", weights_only=False)
        net = HierarchicalPolicy(seq_ma=bool(obj.get("args", {}).get("seq_ma")))
        net.load_state_dict(obj["state_dict"])
        net.eval()
        print(f"\n  ── {tag} · {os.path.basename(ck)} ──", flush=True)
        print(f"  {'实例':<8}{'MS':>9}　PA　　LA", flush=True)
        all_pa: list[tuple[int, bool, int]] = []
        all_la: list[tuple[int, bool, int]] = []
        per_rule: Counter = Counter()
        for name, sp in specs:
            for k in range(N_SC):
                p = _rollout(net, sp, SC0 + k)
                all_pa += p.pa_rows
                all_la += p.la_rows
                for d in p.rec.decisions:
                    if d.kind == "MA":
                        per_rule[d.rule_pa] += 1
                if k == 0:
                    print(f"  {name:<8}{p.metrics['makespan']:>9.1f}",
                          end="", flush=True)
            pa_hit = np.mean([r[1] for r in p.pa_rows]) if p.pa_rows else float("nan")
            la_hit = np.mean([r[1] for r in p.la_rows]) if p.la_rows else float("nan")
            print(f"　{pa_hit:>5.1%}　{la_hit:>5.1%}", flush=True)
        print(flush=True)
        _report(tag, all_pa, "PA")
        _report(tag, all_la, "LA")
        print("    MA 选过的 PA 规则（次数前 6）："
              + "　".join(f"{PA_JOB_RULES[i // 3]}+{PA_MACHINE_RULES[i % 3]}×{c}"
                          for i, c in per_rule.most_common(6)), flush=True)


if __name__ == "__main__":
    main()
