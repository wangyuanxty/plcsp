"""仿真状态快照的测试（P2 Task 1）。快照是特征层的唯一原料，且必须是只读的。"""
from __future__ import annotations

import pytest

import plcsp.env.des as des_mod
from plcsp.env.des import SimWorld, SimConfig
from plcsp.env.instances import load_mk
from plcsp.env.layout import sample_layout
from plcsp.env.corridors import build_corridor_graph, dock_distance_matrix


def _world(name: str = "mk01", cfg: SimConfig | None = None):
    inst = load_mk(name)
    lay = sample_layout(inst.n_machines, seed=0, n_agv=3)
    dm = dock_distance_matrix(build_corridor_graph(lay))
    return inst, SimWorld(inst, lay, dm, cfg or SimConfig(), graph=build_corridor_graph(lay))


def _spy_agv(monkeypatch, rec: dict, w) -> None:
    """观察 AGV 派车环（测试专用）。**不新增任何仿真事件 ⇒ 对轨迹零扰动**。

    - `rec["entries"]`：每次行驶开始前 `(now, aid, leg, 本车在手的工件)`；
    - `rec["requeues"]`：每次**退回队列** `(now, aid, leg, 本车在手的工件)`；
    - `rec["after_requeue"]`：每次退回之后**本车的第一个采样**
      `(now, aid, leg, 退回时在手的工件, 本车负载标志, job_agv)`；
    - `rec["inflight"]`：每次让出控制权时 `(now, Snapshot.in_flight, Σ 车上在运件数)`。

    ⚠️ 复位效果只看"退回后的第一个采样"——退回与复位之间没有让出点，故这一采样必是复位后
    的状态；改看"同一时刻的最后一个采样"会把"同一时刻又重新装上货"的合法状态误判成滞留。

    ⚠️ 只记**本 episode**（`w.track is self.track`）：`run()` 末尾算交期时会跑一次
    `reference_makespan` **嵌套 episode**，那也是 AgvSim，不滤掉会混进别的 world 的样本。
    """
    orig_run, orig_drive = des_mod.AgvSim.run, des_mod.AgvSim._drive
    for k in ("entries", "requeues", "after_requeue", "inflight"):
        rec.setdefault(k, [])
    pending: dict[int, tuple] = {}

    def spy_run(self):
        # ⚠️ 必须**透明转发 send**：SimPy 靠 `process.send(值)` 把事件值送回生成器；
        # 用 `for ev in gen` 转发会把值吞成 None（`q.get()` 解包立刻 TypeError）。
        gen, sent = orig_run(self), None
        while True:
            try:
                ev = gen.send(sent)
            except StopIteration:
                return
            if w.track is self.track:
                if self.aid in pending:
                    _t, _aid, leg, held = pending.pop(self.aid)
                    rec["after_requeue"].append((self.env.now, self.aid, leg, held,
                                                 bool(self.track.agv_loaded[self.aid]),
                                                 tuple(self.track.job_agv)))
                rec["inflight"].append((self.env.now, w.snapshot().in_flight,
                                        sum(self.track.agv_load_n)))
            sent = yield ev

    def in_hand(car) -> tuple:
        """本车此刻**在手**的工件：登记在本车上、且不在本车队列里排队的那些。

        ⚠️ 绑定模式下车队队列里可能还压着**别的**已派任务（transporter 派车即登记），
        那些不算"在手"——否则会把"排队等车的合法在途"误判成"退回未复位"。
        """
        queued = {t[2][0] for t in (car.tasks_in[car.aid].items if car.bound else [])}
        return tuple(j for j, v in enumerate(car.track.job_agv)
                     if v == car.aid and j not in queued)

    def spy_drive(self, src, dst, leg):
        held = in_hand(self)
        if w.track is self.track:
            rec["entries"].append((self.env.now, self.aid, leg, held))
        ok, end = yield from orig_drive(self, src, dst, leg)
        if not ok and w.track is self.track:
            rec["requeues"].append((self.env.now, self.aid, leg, held))
            pending[self.aid] = (self.env.now, self.aid, leg, held)
        return ok, end

    monkeypatch.setattr(des_mod.AgvSim, "run", spy_run)
    monkeypatch.setattr(des_mod.AgvSim, "_drive", spy_drive)


@pytest.mark.unit
def test_snapshot_shapes_match_instance():
    """快照的三段长度必须等于实例的机台数/作业数/车队数。"""
    inst, w = _world()
    snap = w.snapshot()
    assert len(snap.machines) == inst.n_machines
    assert len(snap.jobs) == inst.n_jobs
    assert len(snap.vehicles) == SimConfig().n_agv
    assert snap.now == 0.0, "未跑仿真时 now 应为 0"


@pytest.mark.unit
def test_snapshot_is_read_only():
    """取快照不得改变仿真状态——对拍两次必须逐位相同。"""
    inst, w = _world()
    a, b = w.snapshot(), w.snapshot()
    assert a == b


@pytest.mark.unit
def test_job_state_tracks_progress_and_location():
    """跑一段后：已完成作业的 done_ops == total_ops；未开始的 at_machine == -1。"""
    inst, w = _world()
    w.run(seed_chain=1)
    snap = w.snapshot()
    for j, js in enumerate(snap.jobs):
        assert js.total_ops == len(inst.jobs[j])
        assert 0 <= js.done_ops <= js.total_ops
        if js.done_ops == 0:
            assert js.at_machine == -1 and not js.in_transit
        if js.finished:
            assert js.done_ops == js.total_ops


@pytest.mark.unit
def test_machine_backlog_equals_sum_of_queued_op_times():
    """机台积压 = 输入队列内各工序的加工时长之和（这是最强的一条特征，来不得含糊）。"""
    inst, w = _world()
    snap = w.snapshot()
    for m, ms in enumerate(snap.machines):
        q = w.machines[m].in_q.items
        assert ms.in_q_len == len(q)
        assert ms.backlog_min == pytest.approx(sum(op.time for (_j, _oi, op, _l) in q))
        assert ms.backlog_min == 0.0, "初始队列应为空"


@pytest.mark.unit
def test_vehicle_status_is_one_of_four_values():
    inst, w = _world()
    snap = w.snapshot()
    assert all(v.status in (0, 1, 2, 3) for v in snap.vehicles)
    assert all(0.0 <= v.battery_frac <= 1.0 for v in snap.vehicles)
    assert all(v.capacity >= 1 for v in snap.vehicles)


# ══ 以下五项是评审后补的**回归钉**（F1/F2/F3/F5）：每条都对准一个具体缺陷，
#    去掉对应修复即变红（已逐条做变异验证）。══════════════════════════════


@pytest.mark.unit
def test_machine_remaining_min_decreases_while_processing():
    """F1：同一道工序内 `remaining_min` 必须**递减**，不是恒等于工序时长的常数。

    样本取自 `run_gated` 的决策回调（MK01 种子 1 有 50 对"同工件、时间已推进"的连续样本，
    余量充足）；时间未推进的样本对不参与断言，触底为 0 的样本只要求保持 0。
    """
    inst, w = _world()
    prev: dict[int, tuple[int, float, float]] = {}      # 机台 → (在制工件, now, 剩余)
    stat = {"strict": 0, "flat": 0}

    def policy_l(feat):
        snap = w.snapshot()
        for m, ms in enumerate(snap.machines):
            cur = w._cur_op[m]                          # (工件, 工序时长, 上机时刻)
            if cur is None:                             # 机台空 → 无在制
                prev.pop(m, None)
                continue
            p = prev.get(m)
            if p is not None and p[0] == cur[0] and snap.now > p[1]:
                if p[2] > 0.0:
                    if ms.remaining_min < p[2]:
                        stat["strict"] += 1
                    else:
                        stat["flat"] += 1
            prev[m] = (cur[0], snap.now, ms.remaining_min)
        return 0

    w.run_gated(seed_chain=1, policy_l=policy_l)
    assert stat["flat"] == 0, "同工序内 remaining_min 没有随时间递减（仍是常数）"
    assert stat["strict"] >= 5, f"同工序连续样本太少（{stat['strict']}），断言没被真正验证"


@pytest.mark.unit
def test_fifo_path_marks_jobs_in_transit(monkeypatch):
    """F2：旧 FIFO 路径（`run()` 不传 `agv_phi`，无派车分支）也必须写 `_job_agv`。

    否则 `JobState.in_transit` / `on_agv` 在这条路径上恒为 False / -1——车只有在
    **取到任务之后**才知道是哪台，故写法在 `AgvSim` 而不是只在 transporter 的派车处。
    """
    rec: dict = {}
    inst, w = _world()
    _spy_agv(monkeypatch, rec, w)
    r = w.run(seed_chain=1)                             # FIFO：不传 agv_phi
    assert r["deliveries"] > 0 and not r["horizon_hit"]
    marks = [held for (_t, _a, leg, held) in rec["entries"] if leg == "loaded" and held]
    assert len(marks) >= 1, "FIFO 路径上从没有工件被登记到车上——in_transit 会恒为 False"


@pytest.mark.unit
def test_requeue_resets_loaded_flag_and_transit(monkeypatch):
    """F3：区段争用**退回队列**时，车与工件必须当场复位（不是等下次投递洗掉）。

    场景：极粗区段（`row`）+ 极短等待上限（0.01 min = 0.6 s）+ 派车绑定（`agv_phi`），
    MK01 种子 0 实测 113 次退回，其中 5 次发生在**负载段**（目标分支）。
    断言：每次退回后的 `zone_hold` 窗口内——① 本车负载标志已复位；② 本车在手的工件已离车；
    ③ 收尾后全线无在途工件、无车带负载标志。
    """
    rec: dict = {}
    inst, w = _world(cfg=SimConfig(zone_granularity="row", zone_wait_limit=0.01))
    _spy_agv(monkeypatch, rec, w)
    r = w.run(seed_chain=0, agv_phi=[0, 1, 2] * 200)
    assert not r["horizon_hit"] and r["jobs_done"] == inst.n_jobs
    assert r["dbg"]["requeue"] > 0, "该配置没逼出 requeue，目标分支没被验到"
    rq = rec["requeues"]
    assert any(leg == "loaded" for (_t, _a, leg, _held) in rq), \
        "没有一次退回发生在负载段——目标分支没被验到"
    obs = rec["after_requeue"]
    assert len(obs) == len(rq), f"退回 {len(rq)} 次却只观察到 {len(obs)} 次复位结果"
    for (ts, aid, _leg, held, loaded, marks) in obs:
        assert not loaded, f"t={ts} 车 {aid} 退回后仍带负载标志（未复位）"
        assert all(marks[j] == -1 for j in held), f"t={ts} 车 {aid} 退回后工件仍记在车上"
    assert not any(v >= 0 for v in w._job_agv), "收尾后仍有工件被记为在途"
    assert not any(w._agv_loaded)
    assert all(not js.in_transit for js in w.snapshot().jobs)


@pytest.mark.unit
def test_in_flight_positive_in_fifo_path(monkeypatch):
    """F5：FIFO 路径（无每车队列）下 `in_flight` 必须**出现过 > 0**。

    只数每车队列会让这条路径恒为 0（`rollout()` 走的正是这条），标定脚本拿到的这一维
    永远是 0——正是 P2 要消灭的"恒零特征维"。MK01 种子 1 实测 371 个采样里 358 个 > 0。
    """
    rec: dict = {}
    inst, w = _world()
    _spy_agv(monkeypatch, rec, w)
    r = w.run(seed_chain=1)                             # FIFO：不传 agv_phi
    assert not r["horizon_hit"] and r["deliveries"] > 0
    rows = rec["inflight"]
    assert rows, "探针没夹到采样（观察失效）"
    assert max(f for (_t, f, _n) in rows) > 0, "FIFO 路径上 in_flight 恒为 0"
    # 车全空之外的差值只能来自**共享队列里的待取任务**（FIFO 形状没被吃就会全为相等）
    assert any(f > n for (_t, f, n) in rows), \
        "FIFO 的共享队列没被计入 in_flight（in_flight 只跟着车上的件数走）"


@pytest.mark.unit
def test_in_flight_counts_onboard_batch_in_bound_path(monkeypatch):
    """F5：`in_flight` 必须**含车上在运的批次**——`in_flight ≥ Σ 车上在运件数` 恒成立。

    已装车的批次**已离开队列**，漏掉"车上"那部分时，车满载而队列空的一刻 `in_flight`
    会掉到 0（MK01 种子 1 绑定路径实测 163 个采样会出现这种倒挂）。
    """
    rec: dict = {}
    inst, w = _world()
    _spy_agv(monkeypatch, rec, w)
    r = w.run(seed_chain=1, agv_phi=[0, 1, 2] * 200)    # 绑定/派车路径
    assert not r["horizon_hit"]
    rows = rec["inflight"]
    assert any(n > 0 for (_t, _f, n) in rows), "车上从没装过货，断言没被真正验证"
    bad = [(t, f, n) for (t, f, n) in rows if f < n]
    assert not bad, f"in_flight 漏计车上批次（in_flight < 在运件数）：{bad[:3]}"
