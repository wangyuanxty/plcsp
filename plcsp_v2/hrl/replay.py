# -*- coding: utf-8 -*-
"""**重放前缀 → 分段**（步 5b 的机制层）——设计 §4.2 的"G 个目标从同一个 `Z_t` 出发"靠它落地。

══ 为什么是重放，不是分叉 ══

设计 §13 阻塞项 #1 已核：`des.py` 无 `restore`/`deepcopy`，**SimPy 的生成器帧不可复制**。
⚠️ **但那说的是"不能 `deepcopy`"，不是"不能在 `Z_t` 分支"。**

> **帧不可复制 ≠ 不能重跑。**

**仿真在动作序列固定时是确定性的** ⟹ 从 t=0 起**照已记录的动作**重跑一遍，就**必然**回到
同一个 `Z_t`。到那儿再换成新目标接下去 —— 这就实现了 §4.2 要的"同一个 `Z_t` 出发"，
**`env/des.py` 一行都不用改**（只加了两个停表/读数的小接口）。

══ 一跑的形状 ══

```
分支"g 在 Z_j 上"：
  ① 前缀（t=0 → Z_j）：**照 base 链的动作序列回放**（`ScriptedPolicy`，不调策略、不做前向）
  ② 接续（Z_j 起的 ΔT）：换成 `net` + **固定 g**（`BranchPolicy` 在第 j+1 个顶层点切换）
  ③ 停表：第 (j+1) 个顶层点一到就停（`run_gated(stop_after_ma=…)`）
```

⚠️ **停表只用来读读数**（`StepCounters` 的累计值），**不用来续跑**（停了不留状态）。
**段与段之间靠"各自从头重放"**，不靠"接上一段的尾巴"。

**段回报 = 两条累计读数相减**：`R = counters(该分支在第 j+1 点的累计) − counters(base 在第 j 点的累计)`。
**前缀一模一样 ⟹ 第 j 点的累计读数共享** ✓（这正是重放能成立的理由）。
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from ..env.des import StepCounters, SimWorld
from ..env.rule_views import LAView, PAView
from ..env.snapshot import Snapshot
from .rollout import NetPolicy
from .rules import apply_pa_rule, la_rule_policy


class ScriptedPolicy:
    """照**已记录的动作序列**回放 —— 不调策略、不做任何前向。

    ⚠️ **一条游标跨种类共用**：仿真的调用次序由动作序列决定，动作一样 ⟹ 次序一样。
    **不按 kind 分桶**（分桶会在种类交错时错位）。
    """

    def __init__(self, actions: Sequence[int]):
        self.actions = [int(a) for a in actions]
        self.k = 0

    def _next(self) -> int:
        if self.k >= len(self.actions):
            raise IndexError(
                f"回放的动作序列用完了（第 {self.k + 1} 次调用）——"
                "序列与仿真的调用次序不一致，或前缀长度算错")
        a = self.actions[self.k]
        self.k += 1
        return a

    def policy_ma(self, snap: Snapshot, reason: str, counters: StepCounters):
        """MA 的返回值**不进仿真**（`g` 活在策略对象里）⟹ 这里只消费游标。"""
        self._next()

    def policy_pa(self, view: PAView, pool, tie_rng=None):
        return apply_pa_rule(self._next(), view, pool, tie_rng)

    def policy_l(self, view: LAView, cand_v):
        return int(la_rule_policy(self._next())(view, cand_v))

    def policy_c(self, snap: Snapshot, aid: int, cands):
        return self._next()

    def policy_m(self, snap: Snapshot, mach: int, cands):
        return self._next()


class BranchPolicy:
    """**前 `switch_after_ma` 个顶层点照脚本回放，之后换成 `net`（`g` 已固定）**。

    ⚠️ 切换判据是 **MA 的调用次数**（= 顶层点序号），**不是决策总次数** ——
    分支点就是顶层点，而两层顶之间的下层决策条数会随动作变。
    """

    def __init__(self, script: ScriptedPolicy, switch_after_ma: int, inner: NetPolicy):
        self.script, self.inner = script, inner
        self.switch_after_ma = int(switch_after_ma)
        self.ma_calls = 0

    def _live(self) -> bool:
        return self.ma_calls > self.switch_after_ma

    def policy_ma(self, snap, reason, counters):
        self.ma_calls += 1
        return (self.inner.policy_ma(snap, reason, counters) if self._live()
                else self.script.policy_ma(snap, reason, counters))

    def policy_pa(self, view, pool, tie_rng=None):
        return (self.inner.policy_pa(view, pool, tie_rng) if self._live()
                else self.script.policy_pa(view, pool, tie_rng))

    def policy_l(self, view, cand_v):
        return (self.inner.policy_l(view, cand_v) if self._live()
                else self.script.policy_l(view, cand_v))

    def policy_c(self, snap, aid, cands):
        return (self.inner.policy_c(snap, aid, cands) if self._live()
                else self.script.policy_c(snap, aid, cands))

    def policy_m(self, snap, mach, cands):
        return (self.inner.policy_m(snap, mach, cands) if self._live()
                else self.script.policy_m(snap, mach, cands))


class SegmentReturn(NamedTuple):
    """一段的**增量**回报（设计 §10③：段内完成工件数 + 段内能耗 + 段内拖期增量）。"""

    done_jobs: int
    done_ops: int
    energy_kwh: float
    twt: float

    def as_vector(self) -> tuple[float, float, float]:
        """→ `(完工工件数, 能耗, 拖期)`，**与 `reward.objective_vector` 同序`**（都越小越好）。

        ⚠️ **完工数取负**：它是"越多越好"，而 `reward` 的约定是"越小越好"。
        """
        return (-float(self.done_jobs), float(self.energy_kwh), float(self.twt))


def run_branch(world: SimWorld, prefix_actions: Sequence[int], net: NetPolicy,
               *, switch_after_ma: int, stop_after_ma: int) -> tuple[dict, StepCounters]:
    """**从第 `switch_after_ma` 个顶层点起**换成 `net`（`g` 已设好），跑到第 `stop_after_ma` 个停。

    返回 `(metrics, 停表处的累计读数)`。

    `prefix_actions` = base 链**覆盖到第 `switch_after_ma` 个顶层点（含）为止**的动作序列。
    （切换发生在**下一次** MA 调用上 ⟹ 第 `switch_after_ma` 次 MA 仍走脚本。）

    ⚠️ **比段回报一律用第二个返回值（`StepCounters`），不要用 `metrics["energy"]`** ——
    后者按 `_energy_report(stats, makespan)` 算，而**停表处 `makespan` 是"已完工作业里最晚的
    那个时刻"，不是 `env.now`** ⟹ 与 `_step_counters` 的口径**不同**（实测同一点上 0.945 vs 1.282）。
    **一个点半场只完工了 1 件时差距最大**。累计读数（`StepCounters`）才是两处同口径的那个。
    """
    w = SimWorld(world.inst, world.layout, world.m_dm, world.cfg,
                 graph=world.g, constraints=world.constraints)
    pol = BranchPolicy(ScriptedPolicy(prefix_actions), switch_after_ma, net)
    m = w.run_gated(seed_chain=0, policy_pa=pol.policy_pa, policy_l=pol.policy_l,
                    policy_c=pol.policy_c, policy_m=pol.policy_m,
                    policy_ma=pol.policy_ma, dt_events=net.dt_events,
                    stop_after_ma=stop_after_ma)
    if not net.ma_counters:
        raise RuntimeError("停表处没有读到累计读数 —— `switch_after_ma` 与 `stop_after_ma` 配错")
    return m, net.ma_counters[-1]


def segment_return(before: StepCounters, after: StepCounters) -> SegmentReturn:
    """两条累计读数相减 → 段增量。`before` = 分支点的读数（= base 链同点的读数）。"""
    return SegmentReturn(done_jobs=int(after.done_jobs - before.done_jobs),
                         done_ops=int(after.done_ops - before.done_ops),
                         energy_kwh=float(after.energy_kwh - before.energy_kwh),
                         twt=float(after.twt - before.twt))
