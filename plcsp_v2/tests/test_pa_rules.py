"""PA / LA 规则集（步 3）的测试。

分两层：

1. **纯规则层**——手搓 `PAView`/`LAView`，逐条核对规则选了什么。不碰仿真，快且能覆盖平局、
   非候选、批内投影这些边界；
2. **接线层**——真跑 MK01，核对"PA 真的被调用""离线逐位不变""12 条规则彼此可辨"这些
   只有端到端才看得见的性质。
"""
from __future__ import annotations

import numpy as np
import pytest

from plcsp_v2.env.corridors import build_corridor_graph, dock_distance_matrix
from plcsp_v2.env.des import SimConfig, SimWorld
from plcsp_v2.env.instances import load_mk
from plcsp_v2.env.layout import sample_layout
from plcsp_v2.env.rule_views import (INF, LATask, LAVehicle, LAView, PAJob,
                                     PAMachine, PAView)
from plcsp_v2.env.transport import TransportCaliber
from plcsp_v2.hrl.rules import (LA_RULE_NAMES, N_AEI2025_RULES, N_LA_RULES,
                                N_PA_RULES, PA_JOB_RULES, PA_MACHINE_RULES,
                                PA_RULE_NAMES, apply_pa_rule, la_rule_policy,
                                la_score, pa_rule_policy)

# ══════════════════ 夹具 ══════════════════

def _view(jobs, machines, now=10.0) -> PAView:
    """手搓一份纯规则视图（`snap=None`——规则不读快照，见 `PAView.snap` 的注释）。"""
    return PAView(now=now, jobs=tuple(jobs), machines=tuple(machines))


def _world(name: str = "mk01") -> SimWorld:
    inst = load_mk(name)
    cfg = SimConfig()
    lay = sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    return SimWorld(inst, lay, dock_distance_matrix(g), cfg, graph=g)


# ══════════════════ 1. 规则表的形状 ══════════════════

@pytest.mark.unit
def test_pa_rule_table_shape_and_aei2025_order():
    """12 条 = 4 工件规则 × 3 机台规则；**前 9 条与 AEI2025 的 Rule 1–9 同名同序**。

    顺序不是审美问题：论文里要写"我们的动作空间含他们的 9 条"，那就得能在表上逐行对上。
    """
    assert N_PA_RULES == 12 == len(PA_RULE_NAMES)
    assert PA_JOB_RULES == ("SPT", "EAJ", "SRPT", "EDD")
    assert PA_MACHINE_RULES == ("SMPT", "MINU", "EAM")
    # AEI2025 卡（docs/citation-cards.md）的 Rule 1–9，逐条抄录
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

    ⚠️ 于是 `SPT` 与 `EAJ` **都挑作业 0** —— 这是刻意留的（顺带覆盖"平局取池序最小"），
    下面的断言对 EAJ 写明了这一点，不是含糊过去。
    """
    jobs = (
        PAJob(oi=0, alts=((0, 5.0), (1, 9.0)), apt=5.0, rem_apt=20.0,
              ready_at=10.0, due=300.0),
        PAJob(oi=0, alts=((0, 7.0), (1, 7.0)), apt=7.0, rem_apt=7.0,
              ready_at=10.0, due=200.0),
        PAJob(oi=0, alts=((0, 9.0),), apt=9.0, rem_apt=30.0, ready_at=40.0, due=100.0),
        PAJob(oi=0, alts=((1, 11.0),), apt=11.0, rem_apt=40.0, ready_at=10.0, due=50.0),
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

    | 机台 | 加工时长 | 最早空闲 | 已记工时（= 分子） |
    |---|---|---|---|
    | 0 | **4**（最短） | 5 | 90 |
    | 1 | 8 | 20 | **10**（最闲） |
    | 2 | 6 | **2**（最早） | 50 |
    """
    job = PAJob(oi=0, alts=((0, 4.0), (1, 8.0), (2, 6.0)), apt=6.0, rem_apt=6.0,
                ready_at=0.0, due=INF)
    machines = (PAMachine(avail_at=5.0, assigned_min=90.0),
                PAMachine(avail_at=20.0, assigned_min=10.0),
                PAMachine(avail_at=2.0, assigned_min=50.0))
    return _view((job,), machines, now=100.0)


@pytest.mark.unit
@pytest.mark.parametrize("mach_rule,expected_machine", [
    ("SMPT", 0),      # 4 min 最短
    ("MINU", 1),      # 10/100 最小
    ("EAM", 2),       # avail 2 最早
])
def test_machine_rules_pick_the_expected_machine(mach_rule: str, expected_machine: int):
    _pi, m = apply_pa_rule(f"SPT+{mach_rule}", _machine_race_view(), ((0, 0),))
    assert m == expected_machine


@pytest.mark.unit
def test_ties_are_deterministic_only_when_no_stream_is_given():
    """`tie_rng=None` ⟹ 确定性（工件取池序最小、机台取机台号最小）。

    ⚠️ **这是显式选择，不是默认路径**：生产路径（`run` / `run_gated`）恒给 `tie_rng`，
    即 AEI2025 的**随机平局**。确定性留给"要钉死具体选择"的单测与消融。
    """
    job = PAJob(oi=0, alts=((4, 7.0), (2, 7.0), (3, 7.0)), apt=7.0, rem_apt=7.0,
                ready_at=0.0, due=50.0)
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
    本测试同时核对"换 seed 会变"（= 真的在抽）与"同 seed 不变"（= 可复现）。
    """
    job = PAJob(oi=0, alts=((4, 7.0), (2, 7.0), (3, 7.0)), apt=7.0, rem_apt=7.0,
                ready_at=0.0, due=50.0)
    view = _view((job,), (PAMachine(0.0, 0.0), PAMachine(0.0, 0.0),
                          PAMachine(0.0, 0.0), PAMachine(0.0, 0.0),
                          PAMachine(0.0, 0.0)))
    picks = {apply_pa_rule("SPT+SMPT", view, ((0, 0),),
                           np.random.default_rng([0, 2000]))[1] for _ in range(1)}
    same = apply_pa_rule("SPT+SMPT", view, ((0, 0),), np.random.default_rng([0, 2000]))[1]
    assert picks == {same}, "同一 seed 必须给同一个选择"
    draws = {apply_pa_rule("SPT+SMPT", view, ((0, 0),),
                           np.random.default_rng([s, 2000]))[1] for s in range(40)}
    assert draws == {2, 3, 4}, f"并列的三台都该有机会被抽到，实得 {draws}"


@pytest.mark.unit
def test_minu_at_time_zero_uses_the_floor():
    """`t = 0`：分母走地板，退化为"已记工时绝对值最小"，不抛 `ZeroDivisionError`。"""
    job = PAJob(oi=0, alts=((0, 1.0), (1, 1.0)), apt=1.0, rem_apt=1.0,
                ready_at=0.0, due=INF)
    view = _view((job,), (PAMachine(0.0, 0.0), PAMachine(0.0, 5.0)), now=0.0)
    _pi, m = apply_pa_rule("SPT+MINU", view, ((0, 0),))
    assert m == 0


# ══════════════════ 4. 批内投影 ══════════════════

@pytest.mark.unit
def test_reserved_projection_moves_eam_and_minu_off_the_reserved_machine():
    """**批内投影的核心断言**：同一批里连派两件，第二件不能再去第一件刚占的那台。

    没有这个投影，`EAM`/`MINU` 在**同一批**里会给每件都挑同一台机台（状态没变），
    整批堆到一台机器上——那不是规则的本意。见 `env/rule_views.py` 的模块 docstring。
    """
    job = PAJob(oi=0, alts=((0, 3.0), (1, 9.0)), apt=6.0, rem_apt=6.0,
                ready_at=0.0, due=INF)
    view = _view((job, job), (PAMachine(avail_at=0.0, assigned_min=0.0),
                              PAMachine(avail_at=0.0, assigned_min=0.0)), now=0.0)

    _pi, m1 = apply_pa_rule("SPT+EAM", view, ((0, 0), (1, 0)))
    assert m1 == 0, "第一件：两台同样空闲 ⟹ 取机台号最小"
    view2 = view.reserved(m1, 3.0)
    _pi, m2 = apply_pa_rule("SPT+EAM", view2, ((1, 0),))
    assert m2 == 1, "第二件必须**换一台**——机台 1 现在才是最早可用的"

    # MINU 同理：分子变大了，利用率就上去了
    _pi, n1 = apply_pa_rule("SPT+MINU", view, ((0, 0), (1, 0)))
    view3 = view.reserved(n1, 3.0)
    _pi, n2 = apply_pa_rule("SPT+MINU", view3, ((1, 0),))
    assert (n1, n2) == (0, 1)


@pytest.mark.unit
def test_reserved_is_pure():
    """`reserved` 返回**新视图**，原视图不动（不可变约定）。"""
    view = _machine_race_view()
    before = view.machines[0]
    _ = view.reserved(0, 7.0)
    assert view.machines[0] == before


# ══════════════════ 5. 纯规则层的非法槽位 ══════════════════

@pytest.mark.unit
def test_apply_pa_rule_rejects_empty_pool_and_void_job():
    with pytest.raises(ValueError, match="空池"):
        apply_pa_rule("SPT+SMPT", _machine_race_view(), ())
    done = PAJob(oi=3, alts=(), apt=0.0, rem_apt=0.0, ready_at=0.0, due=INF)
    with pytest.raises(ValueError, match="无候选机台"):
        apply_pa_rule("SPT+SMPT", _view((done,), (PAMachine(0.0, 0.0),)), ((0, 3),))


# ══════════════════ 6. 接线层：真跑 MK01 ══════════════════

def _det(rule):
    """**确定性平局**的规则回调（`tie_rng=None`）——给需要钉死具体行为的测试用。

    ⚠️ 生产路径（`run` / `run_gated`）不给这个，走 AEI2025 的随机平局。
    """
    return lambda view, pool, tie_rng: apply_pa_rule(rule, view, pool, None)


@pytest.mark.unit
def test_pa_is_actually_called_once_per_operation():
    """PA 的池 = 「未完工工件的最早待加工工序」⟹ **每道工序恰好派一次**。

    ⚠️ 步 2 曾实测 `pa_assigns == 0`（`plans` 被离线路径预填满、池恒空）。这个断言就是
    防那个回归的：**"派工器真的被调用了"必须有硬证据，不能靠读代码相信。**
    """
    m = _world().run(seed_chain=0, policy_pa=pa_rule_policy("SPT+SMPT"))
    assert m["pa_assigns"] == m["ops_done"] == 55        # MK01：10 作业 / 55 道工序
    assert m["jobs_done"] == 10 and not m["horizon_hit"]
    assert m["horizon_hit"] is False


@pytest.mark.unit
def test_offline_run_is_bit_identical_to_deterministic_pool_head_smpt():
    """`policy_pa=None`（离线预填计划）与**确定性平局**的 `SPT+SMPT` 逐位相同。

    为什么该等：`SMPT` 的机台选择只看 `(工件, 工序)`，**与派工顺序无关**；而离线计划就是
    "每道工序取最短候选"——两者是同一个映射。这条同时钉住"离线路径没被步 3 改动"。

    ⚠️ **必须用确定性平局**：随机平局下同一个 `(工件, 工序)` 会被抽到不同的并列机台，
    与离线计划不再是同一个映射（实测 75.870 vs 57.970）。这不是缺陷，是随机平局的定义。
    """
    off = _world().run(seed_chain=0)
    on = _world().run(seed_chain=0, policy_pa=_det("SPT+SMPT"))
    assert off["makespan"] == on["makespan"] == 75.870
    assert off["task_flow"] == on["task_flow"]
    assert off["pa_assigns"] == 0                        # 离线路径：PA 一次都不该被调用


@pytest.mark.unit
def test_all_twelve_rules_run_and_are_distinguishable():
    """验收（设计 §14.2 步 3）：**规则驱动能跑；规则间有可辨差异**。"""
    makespans = {}
    for name in PA_RULE_NAMES:
        m = _world().run(seed_chain=0, policy_pa=pa_rule_policy(name))
        assert m["jobs_done"] == 10 and not m["horizon_hit"], f"{name} 没跑完"
        makespans[name] = m["makespan"]
    assert len(set(makespans.values())) >= 6, f"12 条规则的差异太少：{makespans}"
    # EAM / MINU 与派工顺序有关 ⟹ 工件规则**会**改变结果（= 顺序真的起作用了）。
    # ⚠️ SMPT 列在**确定性平局**下才与工件规则无关；随机平局下抽签次序本身依赖派工
    # 次序，故不再断言（见 `test_offline_run_is_bit_identical_to_...` 的注释）。
    for mach in ("MINU", "EAM"):
        with_tie = {makespans[n] for n in PA_RULE_NAMES if n.endswith("+" + mach)}
        det = {_world().run(seed_chain=0, policy_pa=_det(n))["makespan"]
               for n in PA_RULE_NAMES if n.endswith("+" + mach)}
        assert len(with_tie) > 1, f"{mach} 列（随机平局）对工件规则不敏感"
        assert len(det) > 1, f"{mach} 列（确定性平局）对工件规则不敏感"


@pytest.mark.unit
def test_deterministic_smpt_column_equals_the_offline_plan():
    """**确定性平局**下 `SMPT` 一列四条同值，且 = 离线计划。

    这条是"`SMPT` 与派工顺序无关"的形式化——随机平局下它不成立，故单独钉住。
    """
    vals = {_world().run(seed_chain=0, policy_pa=_det(n))["makespan"]
            for n in PA_RULE_NAMES if n.endswith("+SMPT")}
    assert vals == {75.870}, f"SMPT 列应与工件规则无关，实得 {vals}"


@pytest.mark.unit
def test_tie_stream_is_independent_of_the_failure_stream():
    """⚠️ **回归守卫（本步的教训）**：PA 的平局流**不得**复用机台故障那条共享 `rng`。

    复用会把故障抽样整体移位："只改了平局"变成"连扰动流都换了"，消融不再干净。
    判据：同一 `seed_chain` 下，`tie_rng` 关（确定性）与开（随机）两条运行的
    **故障事件数**应落在同一量级——更直接地，此处核对两条运行的 `fail_events` 相同，
    因为平局流是独立的、不会错开故障抽签。
    """
    a = _world().run(seed_chain=0, policy_pa=_det("EDD+EAM"))
    b = _world().run(seed_chain=0, policy_pa=pa_rule_policy("EDD+EAM"))
    assert a["fail_events"] == b["fail_events"], (
        "平局随机源动了故障流——检查 `PA_TIE_SEED` 是否与 `default_rng(seed_chain)` 分开")


@pytest.mark.unit
def test_pa_ledger_makes_inflight_reservations_visible():
    """⚠️ **本步修的那个洞**：账本要看得见"已预留、尚未入目标机台缓冲"的件。

    判据两条：
    1. 该情形在本仓**真的存在**（否则这个洞只是理论上的）；
    2. 账本项**确实进入了视图**（`avail_at` 取 `max(账本, 物理)`）。
    """
    from plcsp_v2.env.rule_views import build_pa_view

    # (2) 视图层：账本 > 物理清空时刻 ⟹ `avail_at` 取账本
    w = _world()
    snap = w.snapshot()
    m = w.inst.n_machines
    led = [0.0] * m
    led[0] = snap.now + 999.0
    v = build_pa_view(snap, w.inst, {j: [None] * len(ops) for j, ops in enumerate(w.inst.jobs)},
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
    assert stat["calls"] == 55
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


# ══════════════════ 7. LA ══════════════════

def _la_view(now=0.0) -> LAView:
    return LAView(now=now, task=LATask(job=0, oi=1, frm_node=100, to_node=200),
                  vehicles=(LAVehicle(node=100, busy_min=10.0),     # 已在取货点
                            LAVehicle(node=101, busy_min=1.0),      # 最近+最闲
                            LAVehicle(node=105, busy_min=50.0)),
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
    view = LAView(now=0.0, task=LATask(job=0, oi=1, frm_node=100, to_node=200),
                  vehicles=(LAVehicle(node=-1, busy_min=0.0),
                            LAVehicle(node=105, busy_min=0.0)),
                  leg_min=lambda u, v: float(abs(u - v)))
    assert view.empty_leg(0) == 0.0
    assert la_score("MINCOST", view, 0) < la_score("MINCOST", view, 1)


@pytest.mark.unit
def test_la_mincost_collapses_to_latest_in_time_and_minu_does_not():
    """⚠️ **记录一处口径塌陷，并钉住它**。

    `LA` 的触发点在本仓就是"工序可运输那一刻"（transporter 收到机台完工事件），故
    AEI2025 Rule 3 的"最早可运输时刻" ≡ `now`——"不晚于它"只剩 `空载段 == 0` 一种情形，
    而那也正是 `MINCOST`（成本式里负载段对所有车同值 ⟹ 等价于就近派车）会挑的。
    **两条规则在端到端上给出同一个数**，这不是 bug，是本仓的流使然（登记进进展记录）。

    真跑三条，把塌陷与"没塌的那条"一起钉住：若哪天 Rule 3 与 Rule 1 分家了，
    说明触发点改了——那时应当**主动**更新这条断言与进展记录，而不是让它悄悄变绿。
    """
    def run(policy_l):
        w = _world()
        return w.run_gated(seed_chain=0, policy_l=policy_l)["makespan"]

    w = _world()
    lay, leg = w.layout, w.leg_min
    got = {n: run(la_rule_policy(n, lay, leg)) for n in LA_RULE_NAMES}
    assert got["MINCOST"] == got["LATEST_IN_TIME"], f"塌陷论证失效：{got}"
    assert got["MINU"] != got["MINCOST"], f"MINU 应与就近派车不同：{got}"


# ══════════════════ 8. 行程时长口径 + 逐车工时 ══════════════════

@pytest.mark.unit
def test_leg_min_matrix_caliber_does_not_consult_the_removed_hetero_field():
    """**回归守卫**：⑩ 删除时漏删了 `_leg_min` 里的 `self.speed_ratio` 使用。

    矩阵口径**一查表就 AttributeError**；MK 实例走几何口径，故一直没暴露。
    这里手工把世界换成矩阵口径，覆盖那条路径。
    """
    w = _world()
    m = w.inst.n_machines
    cal = TransportCaliber.from_matrix(np.full((m + 1, m + 1), 3.0), w.layout)
    # ⚠️ 口径对象在 `_build_entities` 里**逐车复制引用**，故换口径要连车一起换——
    # 只改 `w.transport` 只影响 `SimWorld` 自己那份，车的 `_leg_min` 还是旧口径。
    w.transport = cal
    for agv in w.agvs:
        agv.transport = cal
    u, v = int(w.layout.machines[0].dock_node), int(w.layout.machines[1].dock_node)
    assert w.leg_min(u, v) == pytest.approx(3.0)
    assert w.leg_min(int(w.layout.lu.node), u) == pytest.approx(3.0)


@pytest.mark.unit
def test_vehicle_busy_min_is_tracked_per_vehicle():
    """LA 的 `MINU` 分母来自快照的逐车行驶工时——车队合计与逐车之和必须对得上。"""
    w = _world()
    m = w.run(seed_chain=0)
    w2 = _world()
    w2.run(seed_chain=0)
    busy = [v.busy_min for v in w2.snapshot().vehicles]
    assert len(busy) == w2.cfg.n_agv
    assert sum(busy) == pytest.approx(m["travel_time_total"])
    assert all(b >= 0.0 for b in busy)
