"""在线 S 层的测试（P2 Task 5，spec §5.3.2）。

⚠️ **对拍口径（控制方 pre-flight 裁定 T5-1）**：`rollout()` 走 `run()`——**共享** `tasks_in`
队列（单 Store）+ 轮询兜底；`run_gated()` 走绑定路径——**每车一队列**（`list[Store]`）+
`bound=True`。两条路径本就不同，拿它们对拍**必然假红**。故"关掉在线开关 = 旧行为"的判据
放在 **gated 路径内部**：同一 `policy_l` 行为、同一 `seed_chain`，`online_s=True`（配
"返回首个最短候选"的 `policy_s`）对 `online_s=False`（`_pick_machine` 读预计算的 `plans`，
而 `plans` 就是贪婪最短候选）——两者唯一差别是**决策时机**，应逐位相同。
"""
from __future__ import annotations

import itertools

import pytest

from plcsp.env.des import SimConfig, SimWorld
from plcsp.env.instances import load_mk
from plcsp.env.layout import sample_layout
from plcsp.env.corridors import build_corridor_graph, dock_distance_matrix


def _world(name: str = "mk01") -> SimWorld:
    """同 seed=0 布局 + 同实例的一个 SimWorld（两次调用给出**逐位相同**的世界骨架）。"""
    inst = load_mk(name)
    cfg = SimConfig()
    lay = sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    return SimWorld(inst, lay, dock_distance_matrix(g), cfg, graph=g)


def _greedy_policy_s(inst):
    """与静态计划**同口径**的在线策略：返回首个最短候选（= `np.argmin` 的选择）。

    ⚠️ 不能简单 `return 0`：候选机台列表与机台号无固定关系（MK01 job1 首工序只有机台 1
    一个候选），返回非候选会被 `_pick_machine` 显式拒绝（这是设计，不是偶然）。
    """
    def policy_s(snap, job, oi, cand):
        alts = inst.jobs[job][oi]
        k = min(range(len(alts)), key=lambda i: alts[i][1])      # 同 argmin：取首个最小
        return alts[k][0]
    return policy_s


def _rr_policy_l(n_agv: int):
    """确定性 L 策略（F1 契约：快照 + 任务身份 + 候选车号）：轮转派车。

    每条链各自一个计数器 ⇒ 两条链的派车序列逐位相同（不依赖任何入参）。
    """
    it = itertools.count()
    return lambda snap, job, frm, to, oi, cand: next(it) % n_agv


@pytest.mark.unit
def test_online_off_reproduces_static_trajectory():
    """⚠️ Review Focus #3：关掉在线开关 → **逐位复现旧行为**（对拍口径见模块头）。"""
    inst = load_mk("mk01")
    n_agv = SimConfig().n_agv
    off = _world().run_gated(seed_chain=1, policy_l=_rr_policy_l(n_agv), online_s=False)
    on = _world().run_gated(seed_chain=1, policy_l=_rr_policy_l(n_agv), online_s=True,
                            policy_s=_greedy_policy_s(inst))
    assert not off["horizon_hit"] and not on["horizon_hit"], "掐表截断——对拍无意义"
    assert on["makespan"] == off["makespan"]
    assert on["completes"] == off["completes"]          # 不仅总时长，逐作业完工表也要一致


@pytest.mark.unit
def test_online_s_is_called_once_per_operation():
    """在线模式下，决策次数 == 总工序数（首工序在投放点决策，其余在派工点决策）。"""
    inst = load_mk("mk01")
    calls = []

    def policy_s(snap, job, oi, cand):
        calls.append((job, oi, tuple(cand)))
        return cand[0]                                  # 候选内任选（此处取首候选）

    r = _world().run_gated(seed_chain=1, online_s=True, policy_s=policy_s)
    n_ops = sum(len(j) for j in inst.jobs)
    assert len(calls) == n_ops, f"应为每道工序一次决策，实得 {len(calls)}/{n_ops}"
    # 计数相等 + 键集合相等 ⇒ 每个 (job, oi) **恰好**决策一次（不多不少）
    assert {(j, oi) for (j, oi, _c) in calls} == {(j, oi) for j, job in enumerate(inst.jobs)
                                                  for oi in range(len(job))}
    assert r["jobs_done"] == inst.n_jobs and not r["horizon_hit"]


@pytest.mark.unit
def test_online_s_sees_live_machine_state():
    """⚠️ 在线 S 的意义：决策时看到的是**当时的**机台状态，而不只是初始状态。

    断言：全部决策里至少若干次看到**非零积压**——静态计划（以及"缓存初始快照"的实现）
    下这个数恒为 0（见报告里的变异验证）。
    """
    inst = load_mk("mk01")
    backlogs = []

    def policy_s(snap, job, oi, cand):
        backlogs.append(max(m.backlog_min for m in snap.machines))
        return cand[0]

    _world().run_gated(seed_chain=1, online_s=True, policy_s=policy_s)
    assert len(backlogs) == sum(len(j) for j in inst.jobs), "决策数不对——断言失去意义"
    seen = sum(1 for b in backlogs if b > 0.0)
    assert seen >= 5, (f"在线 S 只在 {seen}/{len(backlogs)} 次决策看到非零积压——"
                       "疑似看的还是初始快照（或只对首工序取活快照）")


@pytest.mark.unit
def test_policy_l_sees_live_snapshot_and_task_identity():
    """F1（Task 5 裁定）：L 派工点的契约与 `policy_s` 对称——
    `policy_l(snap, job, frm, to, oi, cand) -> 车号`，其中 `snap` 必须是**当时的**活状态。

    断言两件事：①快照是活的（若干次决策看到"已有作业完成"的世界——传缓存/初始快照的
    实现下该数恒为 0）；②回调拿到的**任务身份**与仿真记录的 `task_flow` 逐条一致
    （`(job, frm, to, oi)` 四项，防参数错位）。

    ⚠️ `job` 由 P2 Task 7 补入：L 头的任务特征含"该机是否需换型"，需要作业号；
    `(frm, to, oi)` 反查作业多义（MK01 51/112、MK10 538/985 键有歧义）。断言随之**加强**
    （原来只核三项，现在连作业号一并核对）。
    """
    n_agv = SimConfig().n_agv
    rec = []

    def policy_l(snap, job, frm, to, oi, cand):
        rec.append((snap.now, snap.n_done, job, frm, to, oi, tuple(cand)))
        return cand[0]

    r = _world().run_gated(seed_chain=1, policy_l=policy_l)
    flow = r["task_flow"]                       # (job, oi, frm, to)，按任务生成序
    assert len(rec) >= 10, f"L 决策太少（{len(rec)}）——断言失去意义"
    assert [e[0] for e in rec] == sorted(e[0] for e in rec), "L 决策顺序与时间顺序不符"
    assert max(e[0] for e in rec) > 0.0, "L 决策全程 now 都是 0——看的还是初始快照"
    done_seen = sum(1 for e in rec if e[1] > 0)
    assert done_seen >= 5, (f"只有 {done_seen}/{len(rec)} 次 L 决策看到已完成作业——"
                            "疑似看的是初始快照")
    assert len(rec) == len(flow)
    for e, (j_f, oi_f, frm_f, to_f) in zip(rec, flow):
        assert (e[2], e[3], e[4], e[5]) == (j_f, frm_f, to_f, oi_f), \
            "回调拿到的任务身份与 task_flow 不符"
        assert e[6] == tuple(range(n_agv)), f"L 候选应为全车队，实得 {e[6]}"
