# -*- coding: utf-8 -*-
"""**v3 侧的 `d_ms` 信号** —— 与复现侧（§37.4）做同一个测量，看共性。

`StepCounters.ms_committed` = `now + max_m(remaining_min + backlog_min)`。
**复现侧的 `cm_of` 是逐字同一个公式**（`v3_nhdrl.py`）。两边共用的就是它。

v3 的段回报第一分量 = `replay.SegmentReturn.d_ms` = 相邻顶层点的 `ms_committed` 差。
`StepCounters` **只在顶层点算**（`des.StepCounters` 的注释：每局 ~3–4 次）。

判据（与复现侧对齐）：
- 每局几个段；
- 段 `d_ms` 里 **0 的占比**（复现侧是 71.5%）；
- **不同取值数**；
- `corr(Σd_ms, 终局 makespan)`（复现侧 Σd 是 **+0.951**）；
- **9 条规则之间 `Σd_ms` 的极差**（有没有分辨力）。
"""
import os
import sys
from collections import Counter

sys.path.insert(0, r"D:\research\DeepReinforcementLearningScheduling")

import numpy as np  # noqa: E402

from plcsp_v3.env.instances import load_mk  # noqa: E402
from plcsp_v3.env.des import SimConfig, SimWorld  # noqa: E402
from plcsp_v3.hrl.cdr import cdr_policies  # noqa: E402
from plcsp_v3.hrl.rules import PA_RULE_NAMES  # noqa: E402


def _world():
    from plcsp_v3.env.corridors import build_corridor_graph, dock_distance_matrix
    from plcsp_v3.env.layout import sample_layout
    inst = load_mk("mk01")
    cfg = SimConfig()
    lay = sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    return SimWorld(inst, lay, dock_distance_matrix(g), cfg, graph=g)


def one(pa_name: str, la_name: str, sc: int, jmax: int = 6):
    """一条 CDR → `(每个 PA 决策点上的 ms_committed, 终局 makespan)`。

    ⚠️ **不在顶层点截**（顶层点要 `dt` 定时器驱动，本夹具没有 ⟹ 全挤在 t≈0，读数恒 0）。
    改为在**每个 PA 派工点**上直接调 `self._step_counters(stats)` ——
    **这才与复现侧的 `cm_of`（每个决策点算一次）逐点可比。**
    """
    from plcsp_v3.env.des import SimWorld
    pol_pa, pol_l = cdr_policies(pa_name, la_name)
    vals: list[float] = []
    orig = SimWorld._pa_dispatch

    def probe(self, plans, policy_pa, stats):
        vals.append(float(self._step_counters(stats).ms_committed))
        return orig(self, plans, policy_pa, stats)

    SimWorld._pa_dispatch = probe
    try:
        m = _world().run_gated(seed_chain=sc, policy_pa=pol_pa, policy_l=pol_l)
    finally:
        SimWorld._pa_dispatch = orig
    return vals, float(m["makespan"]), []


def main() -> None:
    n_sc = int(os.environ.get("V3_N_SC", "4"))
    scs = [100_000 + 13 * i for i in range(n_sc)]
    names = [PA_RULE_NAMES[k] for k in (0, 3, 6)]      # SPT+SMPT / EAJ+SMPT / SRPT+SMPT

    print("══ 一局里顶层点有几个、都是什么触发 ══")
    ms0, mk0, trg0 = one(names[0], "MINCOST", scs[0])
    print(f"  顶层点数 {len(ms0)}　触发序列 {trg0}")
    print(f"  ms_committed 序列 {[round(x, 1) for x in ms0]}")

    print("\n══ 段 d_ms 的分布 ══")
    segs, pairs, per_rule = [], [], {}
    for name in names:
        tot, mks = [], []
        for sc in scs:
            ms, mk, _t = one(name, "MINCOST", sc)
            d = np.diff(np.asarray(ms, dtype=float))
            segs += d.tolist()
            pairs.append((float(d.sum()), mk))
            tot.append(float(d.sum()))
            mks.append(mk)
        per_rule[name] = (float(np.mean(tot)), float(np.mean(mks)))
        print(f"  {name:<11} 决策点/局 {len(ms)}　**Σd_ms 均值 {np.mean(tot):7.2f}**"
              f"　**makespan 均值 {np.mean(mks):7.2f}**")

    # ⚠️ **两个量必须在同一个实例上比** —— 否则比的是实例不是量程
    sd = np.array([v[0] for v in per_rule.values()], dtype=float)
    sm = np.array([v[1] for v in per_rule.values()], dtype=float)
    print(f"\n  规则间极差（**同实例 MK01**）：Σd_ms {np.ptp(sd):.2f}"
          f"（占均值 {np.ptp(sd) / sd.mean():.1%}）"
          f"　makespan {np.ptp(sm):.2f}（占均值 {np.ptp(sm) / sm.mean():.1%}）")
    print(f"  ⟹ **压缩比 = {np.ptp(sm) / sm.mean() / (np.ptp(sd) / sd.mean()):.2f} 倍**"
          f"（复现侧同量见 §37.4）")

    S = np.asarray(segs, dtype=float)
    P = np.asarray(pairs, dtype=float)
    print(f"\n  段 d_ms：n={S.size}　**0 的占比 {float(np.mean(np.abs(S) < 1e-9)):.1%}**"
          f"　不同取值 {len(np.unique(np.round(S, 6)))}")
    print(f"           min {S.min():+.3f}　max {S.max():+.3f}　均值 {S.mean():+.3f}")
    nz = S[np.abs(S) > 1e-9]
    if nz.size:
        print(f"  非零部分 均值 {nz.mean():+.3f}　min {nz.min():+.3f}　max {nz.max():+.3f}")
    print(f"\n  **corr(Σd_ms, 终局 makespan) = {np.corrcoef(P[:, 0], P[:, 1])[0, 1]:+.4f}**"
          f"　（复现侧同量 = **+0.9510**）")
    print(f"  Σd_ms 极差 {P[:, 0].min():.1f} – {P[:, 0].max():.1f}"
          f"　极差/均值 = {np.ptp(P[:, 0]) / abs(P[:, 0].mean()):.4f}")
    d = Counter(np.round(S, 6).tolist())
    print("\n  取值分布（前 6）：")
    for k, c in d.most_common(6):
        print(f"      {k:+.6f}　{c:>5} 次　{c / S.size:6.1%}")


if __name__ == "__main__":
    main()
