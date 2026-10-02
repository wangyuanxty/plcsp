"""仿真状态快照的测试（P2 Task 1）。快照是特征层的唯一原料，且必须是只读的。"""
from __future__ import annotations

import pytest

from plcsp.env.des import SimWorld, SimConfig
from plcsp.env.instances import load_mk
from plcsp.env.layout import sample_layout
from plcsp.env.corridors import build_corridor_graph, dock_distance_matrix


def _world(name: str = "mk01"):
    inst = load_mk(name)
    lay = sample_layout(inst.n_machines, seed=0, n_agv=3)
    dm = dock_distance_matrix(build_corridor_graph(lay))
    return inst, SimWorld(inst, lay, dm, SimConfig(), graph=build_corridor_graph(lay))


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
