"""PA / LA 规则集与派工流（步 3 + 第二稿）的测试。

分两层：

1. **纯规则层**——手搓 `PAView`/`LAView`，逐条核对规则选了什么。不碰仿真，快且能覆盖平局、
   非候选这些边界；
2. **接线层**——真跑 MK01，核对"派工器真的被调用""池不退化""12 条规则彼此可辨"这些
   只有端到端才看得见的性质。

⚠️ **2026-10-08 第二稿改了流**：决策点从"完工事件"移到"派工那一刻"（原文口径，见
`docs/progress-log.md` §74）。后果：**离线路径（`policy_pa=None`）与 PA 路径不再是同一个
模型**，两条路给不出同一个数——旧的"离线 ≡ SPT+SMPT"断言已按事实删除。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from plcsp_v3.env.corridors import build_corridor_graph, dock_distance_matrix
from plcsp_v3.env.des import SimConfig, SimWorld
from plcsp_v3.env.instances import load_mk
from plcsp_v3.env.layout import sample_layout
from plcsp_v3.env.rule_views import (INF, LAVehicle, LAView, PAJob, PAMachine,
                                     PAView)
from plcsp_v3.env.transport import TransportCaliber
from plcsp_v3.hrl.rules import (LA_RULE_NAMES, N_AEI2025_RULES, N_LA_RULES,
                                N_PA_RULES, PA_JOB_RULES, PA_MACHINE_RULES,
                                PA_RULE_NAMES, apply_pa_rule, la_rule_policy,
                                la_score, pa_job_score, pa_rule_policy)

# ══════════════════ 夹具 ══════════════════

def _view(jobs, machines, now=10.0) -> PAView:
    """手搓一份纯规则视图（`snap=None`——规则不读快照，见 `PAView.snap` 的注释）。"""
    return PAView(now=now, jobs=tuple(jobs), machines=tuple(machines))


def _pajob(oi, alts, apt, rem_apt, ready_at, due, done_oi=0) -> PAJob:
    """`PAJob` 的简写（`done_oi` 默认 0——纯规则层不用它）。"""
    return PAJob(oi=oi, done_oi=done_oi, alts=alts, apt=apt, rem_apt=rem_apt,
                 ready_at=ready_at, due=due)


def _world(name: str = "mk01") -> SimWorld:
    inst = load_mk(name)
    cfg = SimConfig()
    lay = sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    return SimWorld(inst, lay, dock_distance_matrix(g), cfg, graph=g)


def _det(rule):
    """**确定性平局**的规则回调（`tie_rng=None`）——给需要钉死具体行为的测试用。

    ⚠️ 生产路径（`run` / `run_gated`）不给这个，走 AEI2025 的随机平局。
    """
    return lambda view, pool, tie_rng: apply_pa_rule(rule, view, pool, None)


# ══════════════════ 1. 规则表的形状 ══════════════════

@pytest.mark.unit
def test_pa_rule_table_shape_and_aei2025_order():
    """12 条 = 4 工件规则 × 3 机台规则；**前 9 条与 AEI2025 的 Rule 1–9 同名同序**。

    顺序不是审美问题：论文里要写"我们的动作空间含他们的 9 条"，那就得能在表上逐行对上。
    """
    assert N_PA_RULES == 12 == len(PA_RULE_NAMES)
    assert PA_JOB_RULES == ("SPT", "EAJ", "SRPT", "EDD")
    assert PA_MACHINE_RULES == ("SMPT", "MINU", "EAM")
    # AEI2025 的 Table 3，逐条抄录
    theirs = ("SPT+SMPT", "SPT+MINU", "SPT+EAM",
              "EAJ+SMPT", "EAJ+MINU", "EAJ+EAM",
              "SRPT+SMPT", "SRPT+MINU", "SRPT+EAM")
    assert N_AEI2025_RULES == len(theirs) == 9
    assert PA_RULE_NAMES[:N_AEI2025_RULES] == theirs
    # 我们的增量：EDD 一块，接在他们那 9 条之后
    assert PA_RULE_NAMES[N_AEI2025_RULES:] == ("EDD+SMPT", "EDD+MINU", "EDD+EAM")


@pytest.mark.unit
def test_la_rule_table_shape():
    assert N_LA_RULES == 3 == len(LA_RULE_NAMES)


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["NOPE", "SPT", "SPT+SMPT ", ""])
def test_pa_rule_name_errors_are_explicit(bad: str):
    with pytest.raises(ValueError, match="PA 规则名|PA 规则下标"):
        apply_pa_rule(bad, _view([], []), [(0, 0)])


@pytest.mark.unit
@pytest.mark.parametrize("bad", [-1, 12, 99])
def test_pa_rule_index_errors_are_explicit(bad: int):
    with pytest.raises(ValueError, match="越界"):
        apply_pa_rule(bad, _view([], []), [(0, 0)])


# ══════════════════ 2. 工件规则 ══════════════════

def _four_job_view() -> PAView:
    """4 个作业，四条工件规则**各挑一个不同的**——任何一条选错都看得出来。

    | 作业 | apt | rem_apt | ready | due | 候选机台 |
    |---|---|---|---|---|---|
    | 0 | **5**（最小） | 20 | 10 | 300 | (0, 5), (1, 9) |
    | 1 | 7 | 7（**最小**） | 10 | 200 | (0, 7), (1, 7) |
    | 2 | 9 | 30 | **40**（最晚） | 100 | (0, 9) |
    | 3 | 11 | 40 | 10 | **50**（最早） | (1, 11) |

    `EAJ` 的自变量还含机台空闲：机台 0 空闲 0 min、机台 1 空闲 3 min。
    作业 0/1/3 的 `max(ready, min_k avail_k)`：作业 0 候选 {0,1} ⟹ min(0, 3) = 0 ⟹ 10；
    作业 1 同 ⟹ 10；作业 3 只有机台 1 ⟹ 10 + 3 = 13。
    ⟹ `EAJ` 在 {0, 1} 之间平局，取池序最小 = 作业 0。

    ⚠️ 于是 `SPT` 与 `EAJ` **都挑作业 0** —— 这是刻意留的（顺带覆盖"平局取池序最小"）。
    """
    jobs = (
        _pajob(0, ((0, 5.0), (1, 9.0)), 5.0, 20.0, 10.0, 300.0),
        _pajob(0, ((0, 7.0), (1, 7.0)), 7.0, 7.0, 10.0, 200.0),
        _pajob(0, ((0, 9.0),), 9.0, 30.0, 40.0, 100.0),
        _pajob(0, ((1, 11.0),), 11.0, 40.0, 10.0, 50.0),
    )
    machines = (PAMachine(avail_at=10.0, assigned_min=0.0),      # 空闲 0
                PAMachine(avail_at=13.0, assigned_min=0.0))      # 空闲 3
    return _view(jobs, machines)


@pytest.mark.unit
@pytest.mark.parametrize("job_rule,expected_job", [
    ("SPT", 0),        # apt 最小
    ("EAJ", 0),        # 平局（见夹具注释）⟹ 池序最小
    ("SRPT", 1),       # rem_apt 最小
    ("EDD", 3),        # due 最早
])
def test_job_rules_pick_the_expected_job(job_rule: str, expected_job: int):
    view = _four_job_view()
    pool = ((0, 0), (1, 0), (2, 0), (3, 0))
    pi, _m = apply_pa_rule(f"{job_rule}+SMPT", view, pool)
    assert pool[pi][0] == expected_job


# ══════════════════ 3. 机台规则 ══════════════════

def _machine_race_view() -> PAView:
    """一个作业、三台候选机台——三条机台规则**各挑一台不同的**。

    | 机台 | 加工时长 | 最早空闲 | 已派工时（= 分子） |
    |---|---|---|---|
    | 0 | **4**（最短） | 5 | 90 |
    | 1 | 8 | 20 | **10**（最闲） |
    | 2 | 6 | **2**（最早） | 50 |
    """
    job = _pajob(0, ((0, 4.0), (1, 8.0), (2, 6.0)), 6.0, 6.0, 0.0, INF)
    machines = (PAMachine(avail_at=5.0, assigned_min=90.0),
                PAMachine(avail_at=20.0, assigned_min=10.0),
                PAMachine(avail_at=2.0, assigned_min=50.0))
    return _view((job,), machines, now=100.0)


@pytest.mark.unit
@pytest.mark.parametrize("mach_rule,expected_machine", [
    ("SMPT", 0),      # 4 min 最短
    ("MINU", 1),      # 10/20 最小
    ("EAM", 2),       # avail 2 最早
])
def test_machine_rules_pick_the_expected_machine(mach_rule: str, expected_machine: int):
    _pi, m = apply_pa_rule(f"SPT+{mach_rule}", _machine_race_view(), ((0, 0),))
    assert m == expected_machine


@pytest.mark.unit
def test_ties_are_deterministic_only_when_no_stream_is_given():
    """`tie_rng=None` ⟹ 确定性（工件取池序最小、机台取机台号最小）。

    ⚠️ **这是显式选择，不是默认路径**：生产路径恒给 `tie_rng`（AEI2025 的随机平局）。
    """
    job = _pajob(0, ((4, 7.0), (2, 7.0), (3, 7.0)), 7.0, 7.0, 0.0, 50.0)
    view = _view((job, job, job), (PAMachine(0.0, 0.0), PAMachine(0.0, 0.0),
                                   PAMachine(0.0, 0.0), PAMachine(0.0, 0.0),
                                   PAMachine(0.0, 0.0)))
    pi, m = apply_pa_rule("EDD+SMPT", view, ((2, 0), (0, 0), (1, 0)), tie_rng=None)
    assert pi == 0, "工件平局应取**池序**最小"
    assert m == 2, "机台平局应取**机台号**最小"


@pytest.mark.unit
def test_random_tie_break_covers_all_tied_candidates_and_is_seed_reproducible():
    """给了 `tie_rng` ⟹ **照 AEI2025 在并列最优里等概率抽**，且同一 seed 可复现。

    ⚠️ 车队的流是 `[seed_chain, 1000+a]`、故障是 `default_rng(seed_chain)`；
    PA 的平局流是 `[seed_chain, PA_TIE_SEED]` —— **三条互不干扰**。
    """
    job = _pajob(0, ((4, 7.0), (2, 7.0), (3, 7.0)), 7.0, 7.0, 0.0, 50.0)
    view = _view((job,), (PAMachine(0.0, 0.0), PAMachine(0.0, 0.0),
                          PAMachine(0.0, 0.0), PAMachine(0.0, 0.0),
                          PAMachine(0.0, 0.0)))
    same = apply_pa_rule("SPT+SMPT", view, ((0, 0),), np.random.default_rng([0, 2000]))[1]
    assert apply_pa_rule("SPT+SMPT", view, ((0, 0),),
                         np.random.default_rng([0, 2000]))[1] == same, "同一 seed 必须同选择"
    draws = {apply_pa_rule("SPT+SMPT", view, ((0, 0),),
                           np.random.default_rng([s, 2000]))[1] for s in range(40)}
    assert draws == {2, 3, 4}, f"并列的三台都该有机会被抽到，实得 {draws}"


@pytest.mark.unit
def test_minu_at_time_zero_uses_the_floor():
    """`t = 0`：分母走地板，退化为"已派工时绝对值最小"，不抛 `ZeroDivisionError`。"""
    job = _pajob(0, ((0, 1.0), (1, 1.0)), 1.0, 1.0, 0.0, INF)
    view = _view((job,), (PAMachine(0.0, 0.0), PAMachine(0.0, 5.0)), now=0.0)
    _pi, m = apply_pa_rule("SPT+MINU", view, ((0, 0),))
    assert m == 0


# ══════════════════ 4. 纯规则层的非法槽位 ══════════════════

@pytest.mark.unit
def test_apply_pa_rule_rejects_empty_pool_and_void_job():
    with pytest.raises(ValueError, match="空池"):
        apply_pa_rule("SPT+SMPT", _machine_race_view(), ())
    done = _pajob(3, (), 0.0, 0.0, 0.0, INF)
    with pytest.raises(ValueError, match="无候选机台"):
        apply_pa_rule("SPT+SMPT", _view((done,), (PAMachine(0.0, 0.0),)), ((0, 3),))


# ══════════════════ 5. 接线层：真跑 MK01 ══════════════════

@pytest.mark.unit
def test_pool_is_no_longer_degenerate():
    """⚠️⚠️ **本步修的那个缺陷**：池的口径从 `done_ops` 改成"最低**未分配**下标"。

    改前实测：`{0:9, 1:45, 10:1}`，**82% 的决策只有 1 个候选**（§73.1）。
    改后：分布重心在 9–10，**均值 > 5**。

    ⚠️ 步 2 第一稿的注释写"正在加工的工件也在池里"，**而代码用 `done_ops` 恰恰把它滤掉了**
    ——注释写的是意图，代码做的是反面。这条断言就是钉住那个回归的。
    """
    from collections import Counter

    from plcsp_v3.env.rule_views import build_pa_view
    orig = SimWorld._pa_dispatch
    sizes: list[int] = []

    def probe(self, plans, policy_pa, stats):
        snap = self.snapshot()
        v = build_pa_view(snap, self.inst, plans, [0.0] * self.inst.n_machines,
                          [0.0] * self.inst.n_machines)
        sizes.append(sum(1 for j, jv in enumerate(v.jobs)
                         if jv.oi < len(self.inst.jobs[j])))
        return orig(self, plans, policy_pa, stats)

    SimWorld._pa_dispatch = probe
    try:
        _world().run(seed_chain=0, policy_pa=pa_rule_policy("SPT+EAM"))
    finally:
        SimWorld._pa_dispatch = orig
    dist = Counter(sizes)
    assert len(sizes) == 65, f"决策点 = 10 投放 + 55 完工，实得 {len(sizes)}"
    mean = sum(sizes) / len(sizes)
    assert mean > 4.0, f"池仍退化：均值 {mean:.2f}，分布 {dict(sorted(dist.items()))}"
    # 改前 **82%** 的决策池只有 1 个候选；现在只剩极小一撮（收尾段，作业快做完了，
    # 池自然小）。这条是"退化是否被真正修掉"的判据。
    share1 = dist[1] / len(sizes)
    assert share1 <= 0.15, (
        f"池=1 的占比 {share1:.0%} 仍偏高：{dict(sorted(dist.items()))}")


@pytest.mark.unit
def test_pa_is_actually_called_once_per_operation():
    """每道工序恰好派一次 ⟹ `pa_assigns == ops_done == 55`（MK01：10 作业 / 55 道工序）。

    ⚠️ 步 2 曾实测 `pa_assigns == 0`（`plans` 被离线路径预填满、池恒空）。**硬证据，不靠读代码。**
    """
    m = _world().run(seed_chain=0, policy_pa=pa_rule_policy("SPT+SMPT"))
    assert m["pa_assigns"] == m["ops_done"] == 55
    assert m["jobs_done"] == 10 and not m["horizon_hit"]


@pytest.mark.unit
def test_offline_path_is_bit_identical_to_the_v2_golden_value():
    """离线路径（`policy_pa=None`）**逐位不变**：73.010 / task_flow 同形。

    ⚠️ **这是统一事件流之后必须守住的性质**：第二稿把"派工点即决策点"的新流与离线路径
    **合并成同一段代码**（只换"谁定机台"），故这条断言同时证明"合并没有改动离线行为"。

    ⚠️ **不要再断言"离线 ≡ SPT+SMPT"**：第二稿起 PA 路径与离线路径**不是同一个模型**
    （PA 一次只派一件、没派到的工件要等），两者给不出同一个数——旧断言已按事实删除。

    ⚠️ **2026-10-10：黄金值 `75.870` → `73.00999999999999`** —— ② 有限缓冲的默认值由
    `False` 翻成 `True`（用户裁定「在报告档打开」）。**本测试守的性质（离线路径逐位可比）不变**，
    变的是它所处的动力学。
    ⚠️ **本处是精确相等**（非 `approx`），正因如此它同时是"② 的翻转有没有引入随机性"的
    守卫 —— 若翻默认值顺带动了随机流，这里会**先炸**。

    ⚠️⚠️ **写全 17 位是有意的，不要"化简"成 `73.010`**：`73.00999999999999` 与 `73.01`
    是**两个不同的双精度数**（前者才是本次算出来的那个）。旧值 `75.870` 能写成短字面量
    是**碰巧**它恰好往返 —— 那让"精确相等"这条判据看起来比它实际的样子更随意。
    化简字面量会让本断言**从"逐位"退化成"近似"**，而它的名字就叫 `bit_identical`。
    """
    m = _world().run(seed_chain=0)
    assert m["makespan"] == 73.009999999999990
    assert m["pa_assigns"] == 0                        # 离线路径：PA 一次都不该被调用
    assert len(m["task_flow"]) == 65                   # 55 道工序流转 + 10 次投放
    assert m["jobs_done"] == 10


@pytest.mark.unit
def test_all_twelve_rules_run_and_are_distinguishable():
    """验收（设计 §14.2 步 3）：**规则驱动能跑；规则间有可辨差异**。

    第二稿起**工件规则真的改变结果**（EAJ+EAM ≈ 61 vs SRPT+EAM ≈ 95）——这才是
    "PA 的动作是真正的 (工件, 机台) 对"。旧流下 EAM 那一列四条挤在 50.3–50.7，
    因为池退化成 1、工件规则没得选。
    """
    makespans = {}
    for name in PA_RULE_NAMES:
        m = _world().run(seed_chain=0, policy_pa=pa_rule_policy(name))
        assert m["jobs_done"] == 10 and not m["horizon_hit"], f"{name} 没跑完"
        makespans[name] = m["makespan"]
    assert len(set(makespans.values())) >= 8, f"12 条规则的差异太少：{makespans}"
    for mach in ("SMPT", "MINU", "EAM"):
        vals = {v for k, v in makespans.items() if k.endswith("+" + mach)}
        assert len(vals) > 1, f"{mach} 列对工件规则不敏感——工件规则没起作用：{vals}"


@pytest.mark.unit
def test_tie_stream_is_a_separate_generator():
    """⚠️ **回归守卫**：PA 的平局流**不得**复用机台故障那条共享 `rng`。

    ⚠️ **这条断言我第一版写错了**：当时比的是"确定性平局 vs 随机平局两条运行的
    `fail_events` 相等"。**那证明不了任何事** —— 两条运行是**不同的轨迹**，
    机台故障数本来就该不同（改一次 `avail_at` 的口径它就红了）。

    要守的性质是**构造上的**：两个 `_pa_tie_rng` 与机台 `rng` 必须是**两个不同的
    `Generator` 对象**。不同对象不可能共享状态 ⟹ 平局流的消费推不动故障流。
    `Generator` 之间的独立性由 numpy 保证，本测试只钉住"我们确实建了两条"。
    """
    from plcsp_v3.env.des import PA_TIE_SEED

    w = _world()
    w.run(seed_chain=0, policy_pa=pa_rule_policy("EAJ+EAM"))
    assert PA_TIE_SEED != 0
    assert w._pa_tie_rng is not None, "PA 路径必须建平局流"
    assert w._pa_tie_rng is not w.machines[0].rng, (
        "平局流与机台故障流是**同一个** Generator —— 平局会把故障抽样整体移位")
    assert w._pa_tie_rng is not w.agvs[0].rng, "平局流与车队流也不得共用"


@pytest.mark.unit
def test_pa_ledger_makes_inflight_reservations_visible():
    """⚠️ **步 3 第一稿修的那个洞**：账本要看得见"已预留、尚未入目标机台缓冲"的件。

    判据两条：
    1. 该情形在本仓**真的存在**（否则这个洞只是理论上的）；
    2. 账本项**确实进入了视图**（`avail_at` 取 `max(账本, 物理)`）。
    """
    from plcsp_v3.env.rule_views import build_pa_view

    # (2) 视图层：账本 > 物理清空时刻 ⟹ `avail_at` 取账本
    w = _world()
    snap = w.snapshot()
    m = w.inst.n_machines
    led = [0.0] * m
    led[0] = snap.now + 999.0
    v = build_pa_view(snap, w.inst,
                      {j: [None] * len(ops) for j, ops in enumerate(w.inst.jobs)},
                      led, [0.0] * m)
    phys0 = (snap.now + snap.machines[0].remaining_min + snap.machines[0].backlog_min)
    assert v.machines[0].avail_at == pytest.approx(snap.now + 999.0)
    assert v.machines[0].avail_at > phys0
    assert v.machines[1].avail_at < snap.now + 999.0     # 别的机台不受影响

    # (1) 真实运行里确实有"在途预留"
    stat = {"calls": 0, "inflight": 0}
    orig = SimWorld._pa_dispatch

    def probe(self, plans, policy_pa, stats):
        snap = self.snapshot()
        stat["calls"] += 1
        riding = sum(1 for j in range(self.inst.n_jobs)
                     if int(snap.jobs[j].on_agv) >= 0)
        if riding:
            stat["inflight"] += 1
        return orig(self, plans, policy_pa, stats)

    SimWorld._pa_dispatch = probe
    try:
        _world().run(seed_chain=0, policy_pa=pa_rule_policy("SPT+EAM"))
    finally:
        SimWorld._pa_dispatch = orig
    assert stat["calls"] == 65
    assert stat["inflight"] > 0, (
        "本仓不存在在途预留 ⟹ 账本口径的动机不成立；若流的形状变了，应更新 §71.1 的记录")


@pytest.mark.unit
def test_policy_pa_contract_is_validated():
    """池下标越界 / 机台非候选 ⟹ **显式报错**（同 `policy_s` / `policy_l` 的处置）。"""
    with pytest.raises(ValueError, match="不存在的池下标"):
        _world().run(seed_chain=0, policy_pa=lambda view, pool, tie_rng: (999, 0))
    with pytest.raises(ValueError, match="非候选机台"):
        _world().run(seed_chain=0, policy_pa=lambda view, pool, tie_rng: (0, 999))


@pytest.mark.unit
def test_online_s_and_policy_pa_are_mutually_exclusive():
    with pytest.raises(ValueError, match="二选一"):
        _world().run_gated(seed_chain=0, online_s=True,
                           policy_s=lambda snap, job, oi, cand: cand[0],
                           policy_pa=pa_rule_policy(0))


# ══════════════════ 6. LA ══════════════════

def _la_view(now=0.0, ready_at=0.0) -> LAView:
    return LAView(now=now, job=0, oi=1, frm_node=100, to_node=200, ready_at=ready_at,
                  vehicles=(LAVehicle(node=100, busy_min=10.0, free_at=now),   # 已在取货点
                            LAVehicle(node=101, busy_min=1.0, free_at=now),    # 最近+最闲
                            LAVehicle(node=105, busy_min=50.0, free_at=now)),
                  leg_min=lambda u, v: float(abs(u - v)))


@pytest.mark.unit
@pytest.mark.parametrize("rule,expected", [("MINCOST", 0), ("MINU", 1), ("LATEST_IN_TIME", 0)])
def test_la_rules_pick_the_expected_vehicle(rule: str, expected: int):
    view = _la_view()
    best = min(range(3), key=lambda a: (la_score(rule, view, a), a))
    assert best == expected


@pytest.mark.unit
def test_la_unstarted_vehicle_counts_as_at_the_pickup_point():
    """`node == -1`（尚未出车）按"**已在取货点**"处置——与 `AgvSim.run` 的空载段判据同源。"""
    view = LAView(now=0.0, job=0, oi=1, frm_node=100, to_node=200, ready_at=0.0,
                  vehicles=(LAVehicle(node=-1, busy_min=0.0, free_at=0.0),
                            LAVehicle(node=105, busy_min=0.0, free_at=0.0)),
                  leg_min=lambda u, v: float(abs(u - v)))
    assert view.empty_leg(0) == 0.0
    assert la_score("MINCOST", view, 0) < la_score("MINCOST", view, 1)


@pytest.mark.unit
def test_latest_in_time_is_no_longer_degenerate_once_ready_at_is_in_the_future():
    """⚠️ **§70.8 那处塌陷已修**，这条钉住它。

    原文 Rule 3：*"select the AGV that can complete the no-load stage **at the latest before
    the earliest transportable time** of assigned operation"*。
    第一稿把 LA 接到**完工事件**上 ⟹ `ready_at ≡ now` ⟹ 只剩"空载段 == 0"一种合格情形
    ⟹ 与 `MINCOST` **恒等**（实测两列同为 74.200）。

    第二稿把 LA 移回**派工那一刻**（`ready_at` 是未来时刻）⟹ 三条规则两两可辨。
    """
    now = _la_view(now=0.0, ready_at=0.0)
    fut = _la_view(now=0.0, ready_at=30.0)          # 作业 30 min 后才可运输
    # `ready_at = now`：只有"已在取货点"的车赶得上 ⟹ 与 MINCOST 同选
    assert min(range(3), key=lambda a: (la_score("LATEST_IN_TIME", now, a), a)) == 0
    # `ready_at` 在未来：最远的车（105 号，10 min 到）**也赶得上**，且它是"最晚到达"的
    # ⟹ Rule 3 转向它，而 MINCOST 仍取最近的 0 号车。**两条规则分家了。**
    assert min(range(3), key=lambda a: (la_score("LATEST_IN_TIME", fut, a), a)) == 2
    assert min(range(3), key=lambda a: (la_score("MINCOST", fut, a), a)) == 0


@pytest.mark.unit
def test_la_minu_differs_from_mincost():
    """端到端：新的流下 LA 三条规则**互不相同**。"""
    def run(rule):
        w = _world()
        return w.run_gated(seed_chain=0,
                           policy_pa=pa_rule_policy("EAJ+EAM"),
                           policy_l=la_rule_policy(rule))["makespan"]

    got = {n: run(n) for n in LA_RULE_NAMES}
    assert len(set(got.values())) == 3, f"LA 三条规则应两两不同，实得 {got}"


# ══════════════════ 7. 行程时长口径 + 逐车工时 ══════════════════

@pytest.mark.unit
def test_leg_min_matrix_caliber_does_not_consult_the_removed_hetero_field():
    """**回归守卫**：⑩ 删除时漏删了 `_leg_min` 里的 `self.speed_ratio` 使用。

    矩阵口径**一查表就 AttributeError**；MK 实例走几何口径，故一直没暴露。
    """
    w = _world()
    m = w.inst.n_machines
    cal = TransportCaliber.from_matrix(np.full((m + 1, m + 1), 3.0), w.layout)
    # ⚠️ 口径对象在 `_build_entities` 里**逐车复制引用**，故换口径要连车一起换。
    w.transport = cal
    for agv in w.agvs:
        agv.transport = cal
    u, v = int(w.layout.machines[0].dock_node), int(w.layout.machines[1].dock_node)
    assert w.leg_min(u, v) == pytest.approx(3.0)
    assert w.leg_min(int(w.layout.lu.node), u) == pytest.approx(3.0)


@pytest.mark.unit
def test_vehicle_busy_min_is_tracked_per_vehicle():
    """LA 的 `MINU` 分子来自快照的逐车行驶工时——车队合计与逐车之和必须对得上。"""
    m = _world().run(seed_chain=0)
    w2 = _world()
    w2.run(seed_chain=0)
    busy = [v.busy_min for v in w2.snapshot().vehicles]
    assert len(busy) == w2.cfg.n_agv
    assert sum(busy) == pytest.approx(m["travel_time_total"])
    assert all(b >= 0.0 for b in busy)


# ══════════════════ 8. CDR 基线跑器 ══════════════════

@pytest.mark.unit
def test_cdr_pairs_are_the_two_published_sizes():
    """**27 = 他们的（9×3）；36 = 我们的（12×3）**。两个数不可混为一谈。"""
    from plcsp_v3.hrl.cdr import (N_AEI2025_RULES, N_PA_RULES, cdr_pairs,
                                  sweep_cdr)

    theirs = cdr_pairs(N_AEI2025_RULES)
    ours = cdr_pairs(N_PA_RULES)
    assert len(theirs) == 27 and len(ours) == 36
    assert set(theirs) < set(ours)                       # 他们的 27 是我们 36 的子集
    assert {p for p, _ in ours} - {p for p, _ in theirs} == {
        "EDD+SMPT", "EDD+MINU", "EDD+EAM"}               # 多出来的只有 EDD 那一块
    with pytest.raises(ValueError, match="须在"):
        cdr_pairs(0)
    with pytest.raises(ValueError, match="须在"):
        cdr_pairs(13)
    with pytest.raises(ValueError, match="只接受"):
        sweep_cdr(_world, seed_chain=0, n_pa_rules=10)


@pytest.mark.unit
def test_cdr_sweep_runs_and_ranks():
    """真跑一遍**他们的 27 条**并出一张按 makespan 排的表。"""
    from plcsp_v3.hrl.cdr import run_cdr, summarise, sweep_cdr

    rows = sweep_cdr(_world, seed_chain=0, n_pa_rules=9)
    assert len(rows) == 27
    assert all(m["jobs_done"] == 10 for m in rows.values())
    table = summarise(rows)
    assert len(table) == 27 and table[0][2] <= table[-1][2]
    # 单跑一条应与扫描里那条一致（世界工厂一局一用 ⟹ 可复现）
    again = run_cdr(_world(), "EAJ+EAM", "MINCOST", seed_chain=0)
    assert again["makespan"] == rows[("EAJ+EAM", "MINCOST")]["makespan"]


@pytest.mark.unit
def test_scaled_agv_mtbf_activates_constraint_9():
    """⚠️ **⑨ 在默认 MTBF 下对小实例等于不存在**（MK01：10 seed 只触发 1 次）。

    `scaled_agv_mtbf` 按"每局期望故障次数"反推 MTBF —— 目标是让 ⑨ **真的被测到**
    （AEI2025 的扰动参数是每实例 1–8 次）。
    """
    from plcsp_v3.env.des import scaled_agv_mtbf

    inst = load_mk("mk01")
    mtbf = scaled_agv_mtbf(inst, target_failures=4.0)
    assert 0.0 < mtbf < SimConfig().agv_mtbf, "小实例的 MTBF 必须比默认 480 短"
    cfg = SimConfig(agv_mtbf=mtbf)
    lay = sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    dm = dock_distance_matrix(g)
    hits = sum(SimWorld(inst, lay, dm, cfg, graph=g).run(seed_chain=s)["dist_events"]
               for s in range(6))
    assert hits >= 6, f"6 个 seed 只触发 {hits} 次 —— ⑨ 仍未被测到"


# ══════════════════ 9. v3 超参数化（作业侧）══════════════════

def _mixview() -> PAView:
    """给 v3 混合用的视图：**故意让四个基本量互相冲突**。

    ⚠️ 不冲突的话混合永远选同一个，测不出"参数化有效"（见
    `test_lambda_can_change_the_pick`）。
    """
    return _view(
        jobs=[_pajob(0, ((0, 5.0), (1, 5.0)), apt=5.0, rem_apt=90.0, ready_at=10.0, due=20.0),
              _pajob(1, ((0, 8.0), (2, 8.0)), apt=8.0, rem_apt=12.0, ready_at=5.0, due=200.0),
              _pajob(2, ((1, 2.0), (2, 2.0)), apt=2.0, rem_apt=30.0, ready_at=1.0, due=60.0),
              _pajob(3, ((0, 3.0), (1, 3.0)), apt=3.0, rem_apt=3.0, ready_at=9.0, due=15.0)],
        machines=[PAMachine(avail_at=12.0, assigned_min=30.0),
                  PAMachine(avail_at=10.0, assigned_min=0.0),
                  PAMachine(avail_at=30.0, assigned_min=100.0)],
        now=10.0)


def _original_pick(rule: str, view: PAView, pool) -> int:
    """v2 原规则的打法（**在测试里独立重写一遍**，不与实现共用代码）。"""
    def score(j: int) -> float:
        jv = view.jobs[j]
        if rule == "SPT":
            return jv.apt
        if rule == "SRPT":
            return jv.rem_apt
        if rule == "EDD":
            return jv.due
        if not jv.alts:
            return INF
        return max(jv.ready_at, min(view.machines[m].avail_at for m, _t in jv.alts))
    return min(pool, key=lambda j: (score(j), j))


@pytest.mark.unit
def test_lambda_zero_keeps_the_original_job_choice():
    """**设计 §13.2 断言 1**：`λ = (0, 0)` ⟹ 与原规则**选中同一个工件**。

    ⚠️ 断言的是**选择**（`argmin`），不是"逐位同值" —— 因为
    `EDD` 的原量 `due` 可能是 `INF`（v3 限幅到 `DUE_MARGIN_CAP`），
    而 `EAJ` 的原量是**时刻**、v3 的主量是**时长**（差一个常数 `now`）。
    **两者都不改变 `argmin`** —— 那才是"对既有读数不变"的真正含义。
    """
    view = _mixview()
    pool = list(range(len(view.jobs)))
    for rule in PA_JOB_RULES:
        a = _original_pick(rule, view, pool)
        b = min(pool, key=lambda j: (pa_job_score(rule, view, j, (0.0, 0.0)), j))
        assert a == b, f"{rule}: λ=0 改变了选择 {a} → {b}"


@pytest.mark.unit
def test_lambda_out_of_range_is_rejected():
    """`λ` 必须**有界**（设计 §4.0 ②）—— 越界**显式报错**，不静默夹取。"""
    view = _mixview()
    for bad in ((-0.1, 0.0), (0.0, 1.1)):
        with pytest.raises(ValueError, match="越界"):
            pa_job_score("SPT", view, 0, bad)


@pytest.mark.unit
def test_lambda_can_change_the_pick():
    """**参数化必须真的改变选择** —— 否则它是装饰（设计 §4.0 ①）。

    这是"幂律被否掉"那件事的**回归测试**：单调变换（`x^α`）不改 `argmin`，
    **混合才改**。
    """
    view = _mixview()
    base = min(range(len(view.jobs)), key=lambda j: pa_job_score("SPT", view, j))
    moved = False
    for lam in ((0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.5, 0.5)):
        pick = min(range(len(view.jobs)),
                   key=lambda j: pa_job_score("SPT", view, j, lam))
        moved |= pick != base
    assert moved, "λ 调遍了都没改变选择 —— 参数化是装饰"


@pytest.mark.unit
def test_infinite_due_never_becomes_nan():
    """⚠️ 无交期的工件（`due = INF`）**不得产生 `nan`**。

    `0 * inf = nan` —— 这是 `DUE_MARGIN_CAP` 存在的**唯一**理由。
    ⚠️ `λ = 0` 走快路，**必须原样返回 `inf`**（那才是 v2 的行为）；
    非零时才限幅。
    """
    view = _view(jobs=[_pajob(0, ((0, 5.0),), apt=5.0, rem_apt=5.0,
                              ready_at=10.0, due=INF)],
                 machines=[PAMachine(avail_at=10.0, assigned_min=0.0)], now=10.0)
    assert math.isinf(pa_job_score("EDD", view, 0)),         "λ=0 必须逐位等于 v2 —— 即返回 inf，不是限幅值"
    for lam in ((0.0, 1.0), (1.0, 0.0), (1.0, 1.0)):
        s = pa_job_score("EDD", view, 0, lam)
        assert not math.isnan(s), f"λ={lam} 产生了 nan"


@pytest.mark.unit
def test_mix_sets_differ_across_rules():
    """**各规则的辅量集合要各不相同**（设计 §4.0 ③）。

    若所有规则混同一组辅量，它们就成了同一流形的不同参数化 ⟹ 上层选规则变多余。
    """
    from plcsp_v3.hrl.rules import PA_JOB_MIX
    assert set(PA_JOB_MIX) == set(PA_JOB_RULES), "PA_JOB_MIX 没覆盖全部作业规则"
    for rule, (main, aux) in PA_JOB_MIX.items():
        assert len(aux) == 2, f"{rule} 的辅量不是 2 个"
        assert main not in aux, f"{rule} 的主量混进了辅量"
        missing = {"apt", "rtj", "wait", "due", "ready_delay"} - {main, *aux}
        assert missing, f"{rule} 什么都没漏 —— 会与其它规则趋同"
