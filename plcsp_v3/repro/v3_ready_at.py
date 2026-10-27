# -*- coding: utf-8 -*-
"""**`ready_at` 的实际分辨力** —— EAJ 这一族规则到底在不在看状态。

══ 为什么查它 ══

`rule_views.build_pa_view` 的口径（`rule_views.py:176-183`）：

    ready = snap.now
    if oi > 0 and oi-1 >= done:            # 前道未完成
        ready += m_prev.remaining_min + m_prev.backlog_min

两条**已知偏差**（docstring 自己写了）：

1. **在途件按 `now` 计** —— 工件在车上时 `ready_at` 仍报"现在就能开工"。
   `JobState` 里**有** `in_transit` / `on_agv` 可反推 ETA，**被丢掉**（§72.1 登记未修）。
2. **完工后要运到下一台机，运输时间没算** —— 前道一完工就报 `now`。

**但"偏差存在"不等于"它是病根"。** 关键量是**分辨力**：

- 若池里各作业的 `ready_at` **大面积等于 `now`** ⟹ **EAJ 退化成平局** ⟹
  基线里最好的那两条（CDR-10/12，都是 `EAJ+*`）**实际上不是 EAJ 在起作用**，
  而是池序 + 机台规则 ⟹ 它们拿到的优势与状态无关。
- 若池里 `ready_at` 有真实散布，那 EAJ 有信息，偏差只是噪声。
"""
import os
import sys
from collections import Counter

sys.path.insert(0, r"D:\research\DeepReinforcementLearningScheduling")

import numpy as np  # noqa: E402

from plcsp_v3.env.shi_instances import N_AGV_SHI, named  # noqa: E402
from plcsp_v3.hrl.rules import apply_pa_rule, la_rule_policy  # noqa: E402
from plcsp_v3.q1_flat import build  # noqa: E402

TOL = 1e-6


class Capture:
    def __init__(self, rule: int = 0) -> None:
        self.rule, self.caps = rule, []

    def policy_pa(self, view, pool, tie_rng=None):
        self.caps.append((view, pool))
        return apply_pa_rule(self.rule, view, pool, None)

    def policy_l(self, view, cand_v):
        return int(la_rule_policy(0, (0.0,))(view, cand_v))


def main() -> None:
    a, b = int(os.environ.get("V3_A", "16")), int(os.environ.get("V3_B", "140"))
    n_sc = int(os.environ.get("V3_N_SC", "6"))
    inst = named(b, a)
    spec, _ = build(seed=0, pm_interval=30, target_failures=4, n_agv=N_AGV_SHI,
                    finite_buffer=True, layout_seed=0, reward_norm="minmax",
                    tau=2.70, sim_overrides={"due_range": 0.6}, inst_obj=inst)

    rows = []
    for i in range(n_sc):
        c = Capture(0)
        spec.world().run_gated(seed_chain=100_000 + 13 * i, policy_pa=c.policy_pa,
                               policy_l=c.policy_l, dt_events=0)
        rows += c.caps

    n = len(rows)
    eq_now, spread, nuniq, size = [], [], [], []
    pool0_is_min, pool0_is_eaj, n_min_tie = [], [], []
    transit_in_pool, transit_eq_now = [], []
    prev_unfinished = []
    for view, pool in rows:
        r = np.array([view.jobs[j].ready_at for j, _oi in pool], dtype=float)
        size.append(len(pool))
        eq_now.append(float(np.mean(np.abs(r - view.now) < TOL)))
        spread.append(float(r.max() - r.min()))
        nuniq.append(len(np.unique(np.round(r, 6))))
        # EAJ 会挑 `ready_at` 最小的那个；看它在池里是不是第一个、有没有平局
        rmin = r.min()
        at_min = np.flatnonzero(np.abs(r - rmin) < TOL)
        n_min_tie.append(len(at_min))
        pool0_is_min.append(float(0 in at_min))
        # 池序第一个：EAJ 只有在"池序第一恰好也是最早"时才等价于取 pool[0]
        snap = view.snap
        tr = [j for j, _oi in pool if snap is not None and snap.jobs[j].in_transit]
        transit_in_pool.append(len(tr) / max(len(pool), 1))
        transit_eq_now.append(
            np.mean([abs(view.jobs[j].ready_at - view.now) < TOL for j in tr]) if tr else np.nan)
        prev_unfinished.append(
            float(np.mean([(oi > 0 and oi - 1 >= int(snap.jobs[j].done_ops))
                           for j, oi in pool])) if snap is not None else np.nan)
        # EAJ（规则 3 是 EAJ+SMPT；工件那一半 = EAJ）挑的池下标
        pi, _m = apply_pa_rule(3, view, pool, None)
        pool0_is_eaj.append(float(int(pi) == 0))

    def _s(x, fmt="{:.3f}"):
        v = np.asarray(x, dtype=float)
        v = v[np.isfinite(v)]
        return fmt.format(float(v.mean())) if v.size else "n/a"

    print(f"Exp({a}/{b}) · {n_sc} 个场景 · **{n} 个决策点**\n")
    print(f"  池大小                          均值 {_s(size, '{:.2f}')}")
    print(f"  `ready_at == now` 的池内占比     均值 **{_s(eq_now, '{:.1%}')}**"
          f"　（=1 表示池里全是「现在就能开」）")
    print(f"  `ready_at` 极差（max−min）        均值 {_s(spread, '{:.3f}')} min")
    print(f"  池内不同 `ready_at` 值个数        均值 {_s(nuniq, '{:.2f}')}")
    print(f"  取到最小值的池员个数（平局数）     均值 {_s(n_min_tie, '{:.2f}')}")
    print()
    print(f"  ⚠️ **池序第一个恰好是最早的**       均值 {_s(pool0_is_min, '{:.1%}')}")
    print(f"  ⚠️ **EAJ 挑的就是池序第一个**       均值 **{_s(pool0_is_eaj, '{:.1%}')}**"
          f"　（高 ⟹ EAJ 退化成「取池首」）")
    print()
    print(f"  池内**在途件**占比                均值 {_s(transit_in_pool, '{:.1%}')}")
    print(f"  在途件里 `ready_at == now` 的占比  均值 {_s(transit_eq_now, '{:.1%}')}"
          f"　（≈1 ⟹ 在途件全被当成「已就绪」）")
    print(f"  池内**前道未完成**的占比           均值 {_s(prev_unfinished, '{:.1%}')}"
          f"　（=0 ⟹ 上面那个 `ready +=` 分支从不触发）")

    d = Counter(int(round(x)) for x in nuniq)
    print("\n  池内不同 `ready_at` 值个数 的分布：")
    for k in sorted(d):
        print(f"      {k} 个　{d[k]:>5} 点　{d[k] / n:6.1%}")


if __name__ == "__main__":
    main()
