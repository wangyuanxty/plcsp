# -*- coding: utf-8 -*-
"""⑩ multi-drop 行程模型（`SimConfig.multi_drop`）——**模型变更**，不是加一个头。

**改了什么**：一趟 = **一个取货点 + 多个卸货点**。AGV 在取货点装走若干件（同取货点、
不同卸货点），沿途逐站卸下；载货段 = **一串腿**（总时长 = 各腿之和，区段仍**逐腿**申请/释放）。
**为什么改**：`SimConfig.max_agv_capacity`（默认 3）在单卸货点模型下是**死参数**——一次取货
只有一个卸货点，容量永远用不上（`des.AgvSim._collect` 的 `while len(batch) < self.capacity`
因此几乎从不进第二圈）。

**判据**（本文件逐条钉）：
1. `multi_drop=False`（默认）⟹ 与改造前**逐位相同**（黄金摘要在下）；
2. 一趟真的能装 >1 件、逐站卸下，且**在车件数 ≤ capacity**；
3. 卸货序 = **任务到达序**（v1 规则，`des.drop_stops`）；
4. 记账逐件正确：`track.job_agv` / `agv_load_n` / `Snapshot.in_flight` 在多件在车时一致；
5. 队列内容进 `Snapshot`（`queued_tasks`），**两种队列形状**都接对，身份口径 = `_transporter`
   的 `task` 元组；
6. **存在性**：默认档（`n_agv=3`）真的出现 >1 件的批次（本批的成败判据）。

⚠️ 本文件不测 ⑩ 的**决策**（拼批头）——那是 `test_batch_head.py`。
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import simpy

import plcsp.env.des as des_mod
from plcsp.env.constraints import ConstraintConfig
from plcsp.env.corridors import build_corridor_graph, dock_distance_matrix
from plcsp.env.des import (AgvSim, SimConfig, SimTrack, SimWorld, ZoneManager, batch_cands,
                           build_zone_map, drop_stops, rollout)
from plcsp.env.instances import load_mk
from plcsp.env.layout import AgvSpec, sample_layout
from plcsp.env.transport import TransportCaliber

# ── 1. 默认档逐位不变：黄金摘要（**改造前**在 MK01/MK07 上捕获，本机 CPU 档） ──
# 摘要口径 = `rollout(..., seed_chain=1)` 的 (makespan, energy, deliveries, trips,
# travel_time_total, moves, tasks_get) + `dbg`。任何默认路径的行为改动都会翻红。
GOLDEN = {
    ("mk01", 3): "f293ff28f6bb0f1ac1603a185f7c46b49afeb619c2309a62216ca6211c455a0d",
    ("mk01", 1): "b7dc6081e5ec06f5337af7a01b3ed775773bc09e96c02d44160ef659c8600043",
    ("mk07", 3): "89d472f8cc659a6ccdcae888a9cfe9b95413e226cda169666c9ef08198c08832",
}


def _digest(met: dict) -> str:
    h = hashlib.sha256()
    for k in ("makespan", "energy", "deliveries", "trips", "travel_time_total",
              "moves", "tasks_get"):
        h.update(repr(met.get(k)).encode())
    h.update(repr(met["dbg"]).encode())
    return h.hexdigest()


def _no_charge_cons():
    """⑪ 关（本文件的判据都在"车不会半路去充电"的干净档上；⑪ 的读数见 test_charge_head）。"""
    return ConstraintConfig().with_off("charging")


@pytest.mark.unit
@pytest.mark.parametrize("name,n_agv", [("mk01", 3), ("mk01", 1), ("mk07", 3)])
def test_multi_drop_off_is_bit_identical_to_baseline(name, n_agv):
    """判据 1：`multi_drop=False`（默认）⟹ 逐位等于改造前——黄金摘要钉死。"""
    cfg = SimConfig(n_agv=n_agv)
    assert cfg.multi_drop is False, "默认值被改了——既有读数全部作废"
    met = rollout(load_mk(name), seed_chain=1, cfg=cfg)
    assert _digest(met) == GOLDEN[(name, n_agv)], \
        f"{name} n_agv={n_agv} 的默认档链路变了——既有读数不再成立"


@pytest.mark.unit
def test_batch_cands_are_prefixes_up_to_capacity():
    """候选动作码 = 追加件数 s ∈ {0..min(capacity−1, 同取货点任务数)}——**动作语义的唯一真相**。"""
    assert batch_cands(0, 3) == (0,)                    # 队列里没有同取货点任务：只有"不拼"
    assert batch_cands(5, 3) == (0, 1, 2)               # 容量封顶（头件已占 1 位）
    assert batch_cands(1, 3) == (0, 1)
    assert batch_cands(5, 1) == (0,)                    # 容量 1 = 退化（⑩ 关档）


@pytest.mark.unit
def test_drop_stops_keep_first_appearance_order():
    """卸货序 = 任务到达序（首次出现）；同站多件归到同一站（不折返）。"""
    b = [(0, 2, ("j", 0, None, False), None), (0, 1, ("k", 0, None, False), None),
         (0, 2, ("m", 0, None, False), None)]
    stops = drop_stops(b)
    assert [[t[2][0] for t in s] for s in stops] == [["j", "m"], ["k"]]


# ── 2. 直接测 `_collect_multi`（不绕端到端） ──

class _RecordingTrack(SimTrack):
    """记下每一次 `set_load(aid, n)` 的调用序（逐件记账判据的原料）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loads: list[tuple[int, int]] = []

    def set_load(self, aid: int, n: int) -> None:
        self.loads.append((int(aid), int(n)))
        super().set_load(aid, n)


def _bare_agv(capacity: int, n_machines: int = 4, *, multi_drop: bool = True):
    """一台**不跑仿真**的 AgvSim，只用来直接调收集/投递函数（算术判据不绕端到端）。"""
    inst = load_mk("mk01")
    cfg = SimConfig(n_agv=1, multi_drop=multi_drop)
    lay = sample_layout(n_machines, seed=0, n_agv=1)
    g = build_corridor_graph(lay)
    zof, nz = build_zone_map(lay, cfg.zone_granularity)
    env = simpy.Environment()
    spec = AgvSpec(id=0, speed_factor=1.0, capacity=capacity, battery_kwh=3.0)
    track = _RecordingTrack(8, n_machines, 1)   # n_jobs=8：够本文件手搓的 job 号用
    stats = {"tasks_get": 0, "trips": 0, "deliveries": 0, "requeue": 0,
             "agv_del": [0], "agv_pos": [-1], "travel_time": 0.0, "moves": 0,
             "agv_empty_min": 0.0, "agv_loaded_min": 0.0}
    agv = AgvSim(env, 0, dock_distance_matrix(g), TransportCaliber.for_instance(inst, lay), cfg,
                 stats, simpy.Store(env), [], g,
                 ZoneManager(env, zof, nz, cfg.zone_wait_limit),
                 ConstraintConfig().with_off("charging"), spec,
                 np.random.default_rng(0), track, chargers=lay.chargers)
    return agv, env


def _task(job: int, frm: int, to: int, oi: int = 0) -> tuple:
    return (frm, to, (job, oi, None, False), None)


@pytest.mark.unit
def test_collect_multi_takes_whole_queue_same_frm_up_to_capacity():
    """规则档：**全队列**找同取货点任务（不只看队首连续段），最多 `capacity` 件。"""
    agv, env = _bare_agv(capacity=3)
    q = simpy.Store(env)
    # 队列序：同取货点(1) 的 0 号、别的取货点(2) 的、同取货点(1) 的 2 号
    for t in (_task(0, 1, 3), _task(1, 2, 3), _task(2, 1, 3)):
        q.put(t)
    batch = agv._collect_multi(q, 1, _task(9, 1, 3))
    assert [t[2][0] for t in batch] == [9, 0, 2], "应取走全部同取货点任务（保队列序）"
    assert len(q.items) == 1 and q.items[0][2][0] == 1, "别的取货点的任务必须留在队列里"
    assert agv.stats["batch_items"] == 3 and agv.stats["batch_ge2"] == 1


@pytest.mark.unit
def test_collect_multi_respects_capacity_one():
    """`capacity == 1`（⑩ 异构车队关）⟹ 一件都不追加，与"该约束从未存在"逐位相同。"""
    agv, env = _bare_agv(capacity=1)
    q = simpy.Store(env)
    q.put(_task(0, 1, 3))
    batch = agv._collect_multi(q, 1, _task(9, 1, 3))
    assert [t[2][0] for t in batch] == [9]
    assert len(q.items) == 1, "容量 1 不得动队列"


# ── 3. `_deliver_multi`：多站停靠 + 逐件记账 ──

class _StubMachine:
    """只给 `_deliver_multi` 需要的最小形状：`pad.dock_node` + 有界 `in_q`。"""

    def __init__(self, env, node: int, cap: float = 99):
        self.pad = type("P", (), {"dock_node": node})()
        self.in_q = simpy.Store(env, capacity=cap)


@pytest.mark.unit
def test_deliver_multi_stops_once_per_drop_point_in_arrival_order():
    """一趟停多个点、每站卸下该站的件；**卸货序 = 任务到达序**；逐件递减在车件数。

    ⚠️ 区段管制关（① 关）⟹ 时长 = 各腿几何时长之和，可**逐位**对拍，不靠间接推断。
    """
    agv, env = _bare_agv(capacity=3)
    agv.congestion = False
    agv.machines = [_StubMachine(env, n) for n in (0, 1, 2, 3)]
    q = simpy.Store(env)
    # 批 = 任务序 [j0→2, j1→1, j2→2]：卸货点序列应为 2, 1（各停一次）
    batch = [_task(0, 0, 2), _task(1, 0, 1), _task(2, 0, 2)]
    agv.track.set_load(agv.aid, len(batch))     # 装车：3

    def run():
        yield from agv._deliver_multi(q, batch)

    env.process(run())
    env.run()
    # 逐件递减：装车 3 → 卸第一件 2 → 1 → 0（两件去 2 号、一件去 1 号；卸货序 = 2, 1）
    assert [n for _a, n in agv.track.loads] == [3, 2, 1, 0]
    assert [it[0] for it in agv.machines[2].in_q.items] == [0, 2]
    assert [it[0] for it in agv.machines[1].in_q.items] == [1]
    assert agv.machines[3].in_q.items == []
    assert agv.stats["deliveries"] == 3 and agv.stats["trips"] == 1
    assert agv.track.agv_load_n[0] == 0 and agv.track.agv_loaded[0] is False
    assert all(v == -1 for v in agv.track.job_agv)
    # 行程 = 腿 0→2 与 2→1 之和（① 关：一次到底，不拆区段）
    exp = agv._seg_min(0, 2) + agv._seg_min(2, 1)
    assert agv.stats["travel_time"] == pytest.approx(exp, rel=1e-12)
    assert agv.pos_node == 1, "车应停在最后一个卸货点"


@pytest.mark.unit
def test_deliver_multi_requeues_only_the_undelivered_part():
    """某一腿失败 ⟹ **仍在车上**的件退回队列（已卸下的不动），在车件数清零。"""
    agv, env = _bare_agv(capacity=3)
    agv.congestion = False
    agv.machines = [_StubMachine(env, n) for n in (0, 1, 2)]
    q = simpy.Store(env)
    batch = [_task(0, 0, 1), _task(1, 0, 2)]
    agv.track.set_load(agv.aid, 2)
    calls = {"n": 0}
    orig_drive = agv._drive

    def flaky(src, dst, leg):
        calls["n"] += 1
        if calls["n"] == 2:                     # 第二条腿失败
            return False, src
        ok, end = yield from orig_drive(src, dst, leg)
        return ok, end

    agv._drive = flaky

    def run():
        yield from agv._deliver_multi(q, batch)

    env.process(run())
    env.run()
    assert agv.stats["deliveries"] == 1, "第一站已卸下的一件不得回滚"
    assert [t[2][0] for t in q.items] == [1], "只有未交付的那件退回队列"
    assert agv.track.agv_load_n[0] == 0
    assert agv.track.job_agv[0] == -1 and agv.track.job_agv[1] == -1
    assert agv.stats["requeue"] == 1


# ── 4. 端到端：记账 + 快照队列内容（两种形状） ──

def _spy_world(name: str, *, multi_drop: bool, n_agv: int, agv_phi):
    """跑一局并**在校验点上**收集 (Snapshot, 活队列) 对——用来钉 `queued_tasks` 与 `in_flight`。

    ⚠️ 探针只读（不快照之外的额外仿真事件），故对轨迹零扰动。
    """
    inst = load_mk(name)
    lay = sample_layout(inst.n_machines, seed=0, n_agv=n_agv)
    g = build_corridor_graph(lay)
    dm = dock_distance_matrix(g)
    cfg = SimConfig(n_agv=n_agv, multi_drop=multi_drop)
    w = SimWorld(inst, lay, dm, cfg, graph=g, constraints=_no_charge_cons())
    rec: list[tuple] = []
    orig_drive = des_mod.AgvSim._drive

    def spy_drive(self, src, dst, leg):
        if w.track is self.track:               # 只记本 episode（参考运行不混进来）
            snap = w.snapshot()
            live = []
            stores = (list(enumerate(self.tasks_in)) if isinstance(self.tasks_in, list)
                      else [(-1, self.tasks_in)])
            for a, store in stores:
                for t in store.items:
                    live.append((a, t[2][0], t[0], t[1], t[2][1]))
            rec.append((snap, live, sum(self.track.agv_load_n)))
        return (yield from orig_drive(self, src, dst, leg))

    des_mod.AgvSim._drive = spy_drive
    try:
        met = w.run(seed_chain=1, agv_phi=agv_phi)
    finally:
        des_mod.AgvSim._drive = orig_drive
    return met, rec


@pytest.mark.unit
@pytest.mark.parametrize("multi_drop,agv_phi", [(True, []), (True, None)])
def test_snapshot_queue_contents_match_the_live_stores_in_both_shapes(multi_drop, agv_phi):
    """判据 5：`Snapshot.queued_tasks` = 活队列的**逐件身份**（队列序），两种形状都接对。

    `agv_phi=[]` ⟹ bound（每车一 Store，`veh` = 车号）；`agv_phi=None` ⟹ FIFO（共享 Store，
    `veh` = −1）。身份口径 = `_transporter` 的 `task = (frm, to, item, _path)`、`item = (job, oi, …)`。
    """
    met, rec = _spy_world("mk01", multi_drop=multi_drop, n_agv=3, agv_phi=agv_phi)
    assert rec, "校验点为空——探针没接上"
    for snap, live, onboard in rec:
        got = [(q.veh, q.job, q.frm, q.to, q.oi) for q in snap.queued_tasks]
        assert got == live, "快照的队列内容与活队列不符（身份或顺序错）"
        if agv_phi is None:
            assert snap.queued_tasks == () or all(q.veh == -1 for q in snap.queued_tasks)
        assert snap.in_flight == len(live) + onboard
    assert met["jobs_done"] == load_mk("mk01").n_jobs


@pytest.mark.unit
def test_multi_drop_delivers_every_task_exactly_once():
    """判据 4：multi-drop 档下每件任务仍**恰好送达一次**，收尾时车上无货。"""
    inst = load_mk("mk01")
    cfg = SimConfig(multi_drop=True)
    met = rollout(inst, seed_chain=1, cfg=cfg, constraints=_no_charge_cons())
    assert met["deliveries"] == len(met["task_flow"]), "送达件数 ≠ 任务数（漏送或重复送）"
    assert met["jobs_done"] == inst.n_jobs and not met["horizon_hit"]
    assert met["batch_items"] >= met["deliveries"], "装走的件数不该少于送达件数"
    assert met["batch_trips"] > 0


# ── 5. 存在性判据（本批成败） ──

@pytest.mark.unit
@pytest.mark.parametrize("name,min_ge2", [("mk01", 1), ("mk07", 1)])
def test_default_config_really_produces_multi_item_batches(name, min_ge2):
    """**判据 6（本批成败）**：默认档（`n_agv=3`、`SimConfig()`）真的出现 >1 件的批次。

    ⚠️ 口径 = §45.4 的同一条（bound 路径 = 训练路径，`seed_chain=1`）：数**取货时刻**的批大小。
    数字见 `progress-log.md` §47（MK01 的余量很小——如实记录，不粉饰）。
    """
    met = rollout(load_mk(name), seed_chain=1, cfg=SimConfig(multi_drop=True),
                  agv_phi=[], constraints=_no_charge_cons())
    assert met["batch_ge2"] >= min_ge2, \
        f"{name}：默认档没有出现 >1 件的批次（batch_ge2={met['batch_ge2']}）"


@pytest.mark.unit
def test_multi_drop_makes_the_capacity_parameter_bite():
    """**本批的目的**：`max_agv_capacity` 在单卸货点模型下是死参数，multi-drop 下真的起作用。

    同一实例、同一 seed：容量上限 1 与 3 的批大小分布**必须不同**——且上限 1 时恒无拼批
    （唯一能证明"容量进了动力学"的判据）。
    """
    inst = load_mk("mk01")
    cons = _no_charge_cons()
    # 容量上限 1：`sample_layout` 给出全队 capacity=1 ⟹ 规则档一件都拼不起来
    lay1 = sample_layout(inst.n_machines, seed=0, n_agv=3, max_agv_capacity=1)
    assert all(a.capacity == 1 for a in lay1.agvs)
    dm1 = dock_distance_matrix(build_corridor_graph(lay1))
    w1 = SimWorld(inst, lay1, dm1, SimConfig(n_agv=3, multi_drop=True, max_agv_capacity=1),
                  graph=build_corridor_graph(lay1), constraints=cons)
    m1 = w1.run(seed_chain=1, agv_phi=[])
    assert m1["batch_ge2"] == 0 and m1["batch_items"] == m1["batch_trips"]

    met3 = rollout(inst, seed_chain=1, cfg=SimConfig(multi_drop=True), agv_phi=[], constraints=cons)
    assert met3["batch_items"] > met3["batch_trips"], "容量 3 档必须有拼批发生"
