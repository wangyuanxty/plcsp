# -*- coding: utf-8 -*-
"""**训练目标对不对**：段回报 `d_ms` 能不能预测终局 makespan？

══ 为什么 ══

阶段 A 的组优势来自 `d_ms`（`SegmentReturn` 的第一分量）。
`d_ms` = `ms_committed` 的段增量，而 `ms_committed` **只统计已分配的活**
（`des.py:1582-1605`：机台已排定 + AGV 已派任务）。

⟹ **一个“少派、缓派”的规则在这一段里看起来更好**（计划完工时刻小），
但它把活推到后面，终局更差。

如果这是真的，训练会稳定地推向**近视规则**。这能解释：
训练从 `EAJ+EAM`（254.6）漂到 `SPT+SMPT`（330.1）。

══ 测法 ══

同一个顶层点，对 9 条 PA 规则各算两个量：

1. **段回报** `d_ms`（`span=1`，训练真正用的那个）；
2. **终局 MS**（从该点起把规则固定到局末）。

然后看**同一点上两者的排序相关性**。

- 正相关 ⟹ 段回报可用；
- **零/负相关 ⟹ 训练目标与真目标不一致 —— 根因。**

══ 用法 ══

`D:/anaconda/python.exe -m plcsp_v3.repro.v3_reward_alignment`
环境变量：`V3_RA_NINST`（默认 3）· `V3_RA_STRIDE`（默认 4）· `V3_RA_MAXPTS`（默认 6）。
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
from plcsp_v3.hrl.agents import HierarchicalPolicy            # noqa: E402
from plcsp_v3.hrl.replay import run_branch                    # noqa: E402
from plcsp_v3.hrl.rollout import NetPolicy                    # noqa: E402
from plcsp_v3.hrl.rules import LA_RULE_NAMES, PA_RULE_NAMES    # noqa: E402
from plcsp_v3.hrl.train import (_prefix_to, _run_all_segments,  # noqa: E402
                                base_chain, rules_dict,
                                segment_return)
from plcsp_v3.q1_flat import build                            # noqa: E402

NINST = int(os.environ.get("V3_RA_NINST", "3"))
STRIDE = int(os.environ.get("V3_RA_STRIDE", "4"))
MAXPTS = int(os.environ.get("V3_RA_MAXPTS", "6"))
NPA = 9
DT = 20
LA = LA_RULE_NAMES.index("LATEST_IN_TIME")


def _spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den else 0.0


def _terminal_ms(net, sp, base, j, rule) -> float:
    prefix, _ = _prefix_to(base.rec.decisions, j)
    inner = NetPolicy(net, sp.inst, sp.layout, sp.ctx, rules=rules_dict(rule),
                      greedy=True, seed=0, dt_events=DT, freeze_rules=True,
                      seed_chain=base.seed_chain, exact_rule=True)
    m, _ = run_branch(sp.world(), prefix, inner, switch_after_ma=j,
                      stop_after_ma=0)
    return float(m["makespan"])


def main() -> None:
    torch.manual_seed(0)
    net = HierarchicalPolicy().eval()      # 权重无关：exact_rule
    cons = ConstraintConfig(maintenance=False)
    corrs, segbest, termbest = [], [], []
    print(f"**段回报 vs 终局** · {NINST} 实例 · PA 前 {NPA} 条"
          f" · LA={LA_RULE_NAMES[LA]}\n", flush=True)
    print(f"  {'inst.j':<9}{'spearman':>10}  段最优 / 终局最优", flush=True)
    print("  " + "-" * 58, flush=True)
    for i in range(NINST):
        sp, _ = build(seed=0, inst_obj=gen_mk_like(seed=1000 + i), tau=2.0,
                      pm_interval=30, target_failures=4, n_agv=3,
                      finite_buffer=True, layout_seed=100 + i,
                      reward_norm="minmax", sim_overrides={"due_range": 0.6},
                      constraints=cons)
        base = base_chain(net, sp, greedy=True, seed=0, dt_events=DT,
                          seed_chain=1, fixed_rules={"PA": 3, "LA": LA},
                          exact_rule=True)
        ma = [d for d in base.rec.decisions if d.kind == "MA"]
        js = list(range(0, len(ma), max(1, STRIDE)))[:MAXPTS]
        for j in js:
            rules = [(k, LA, int(ma[j].rule_ra)) for k in range(NPA)]
            runs = _run_all_segments(net, sp, base, [(j, [(r, 0, True) for r in rules])],
                                     dt_events=DT, pool=None, span=1,
                                     exact_rule=True)[0]
            before = base.ma_counters[j]
            dms = np.array([segment_return(before, br.counters).d_ms
                            for br in runs])
            term = np.array([_terminal_ms(net, sp, base, j, r) for r in rules])
            c = _spearman(dms, term)
            corrs.append(c)
            sb, tb = int(np.argmin(dms)), int(np.argmin(term))
            segbest.append(sb)
            termbest.append(tb)
            print(f"  {i}.{j:<7}{c:>+10.3f}  {PA_RULE_NAMES[sb]:<10} / "
                  f"{PA_RULE_NAMES[tb]}", flush=True)
    c = np.asarray(corrs)
    print(f"\n  **排序相关性**：均值 {c.mean():+.3f}　中位 {np.median(c):+.3f}"
          f"　正相关的点 {int((c > 0).sum())}/{len(c)}", flush=True)
    same = float(np.mean([a == b for a, b in zip(segbest, termbest)]))
    print(f"  段回报选的规则 == 终局最优规则：**{same:.1%}**", flush=True)
    print("\n  判据：", flush=True)
    print("    · 相关性 ≈ 0 或为负 ⟹ **段回报不能预测终局，训练目标错了**", flush=True)
    print("    · 相关性明显为正 ⟹ 段回报可用，问题在别处", flush=True)


if __name__ == "__main__":
    main()
