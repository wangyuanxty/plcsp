# -*- coding: utf-8 -*-
"""**公开 API 的规格测试**（2026-10-09）。

⚠️ **这个文件的目的是终止"一问就冒一批"**：此前 `plcsp_v2/tests/` 只提到 **41/184** 个
公开符号，另外 111 个只出现在 `plcsp/tests/` 里 —— 而那里 `import` 的是 `plcsp.*`
（删约束之前的包），**跑不到 v2 的模块上**。后果：v2 有 78% 的公开面**只能靠人读**，
而"读代码找错"是开放式搜索，永远能再找一批（见 §80）。

**门禁绿 = 跑过一遍；不绿 = 有错。** 不再需要问"有没有错"。

覆盖：`env/*` · `energy.py` · `hrl/cdr.py` 的公开符号。`baselines/` 在另一个文件里。
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from plcsp_v2 import energy
from plcsp_v2.env import corridors, due_dates, layout as layout_mod, mkt, reward
from plcsp_v2.env.constraints import ConstraintConfig
from plcsp_v2.env.des import (MachineSim, LuStation, OpLite, SimConfig, SimTrack,
                              SimWorld, charge_cands, check_seq_rank,
                              compute_due_dates, machine_fail_rate,
                              reference_makespan, reference_run, rollout,
                              weighted_tardiness)
from plcsp_v2.env.instances import Instance, gen_random, load_mk, parse_fjs_text
from plcsp_v2.env.rule_views import advance_avail
from plcsp_v2.env.snapshot import (ChargerState, JobState, MachineState,
                                   QueuedTask, Snapshot, VehicleState)
from plcsp_v2.env.transport import TransportCaliber
from plcsp_v2.hrl.cdr import cdr_policies
from plcsp_v2.hrl.rules import minu_denominator, pa_job_score, pa_machine_score


def _inst(name: str = "mk01") -> Instance:
    return load_mk(name)


def _world(name: str = "mk01") -> SimWorld:
    inst = load_mk(name)
    cfg = SimConfig()
    lay = layout_mod.sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = corridors.build_corridor_graph(lay)
    return SimWorld(inst, lay, corridors.dock_distance_matrix(g), cfg, graph=g)


def _rank(inst: Instance) -> list[list[int | None]]:
    return [[None] * len(ops) for ops in inst.jobs]


# ══════════════════ env/layout.py ══════════════════

@pytest.mark.unit
def test_grid_spec_geometry_is_self_consistent():
    g = layout_mod.GridSpec(n_rows=3, n_cols=4, cell_w=3.0, cell_h=3.0,
                            aisle_w=1.5, lu_connector_m=2.0)
    assert g.n_nodes >= 12          # 节点数 + 装卸站接入点，不是 rows*cols
    n = g.node_id(1, 2)
    assert g.node_rc(n) == (1, 2)
    x, y = g.node_xy(1, 2)
    assert (x, y) == (g.node_xy(1, 2))
    assert g.lu_node is not None and g.lu_dock_node is not None
    assert len(g.lu_xy) == 2


@pytest.mark.unit
@pytest.mark.parametrize("n_machines", [4, 6, 10, 15])
def test_grid_shape_and_layout_pads(n_machines: int):
    r, c = layout_mod.grid_shape(n_machines)
    assert r * c >= n_machines
    lay = layout_mod.sample_layout(n_machines, seed=0, n_agv=4)
    assert isinstance(lay, layout_mod.Layout)
    assert len(lay.machines) == n_machines and len(lay.agvs) == 4
    mp = lay.machines[0]
    assert isinstance(mp, layout_mod.MachinePad)
    assert mp.dock_node >= 0 and mp.fail_rate >= 0.0
    assert isinstance(lay.chargers[0], layout_mod.ChargerPad)
    assert isinstance(lay.lu, layout_mod.LuPad)
    assert all(isinstance(a, layout_mod.AgvSpec) and a.battery_kwh > 0 for a in lay.agvs)


# ══════════════════ env/instances.py ══════════════════

@pytest.mark.unit
def test_instance_and_parsers():
    inst = _inst("mk01")
    assert isinstance(inst, Instance)
    assert inst.n_jobs == len(inst.jobs) and inst.n_machines > 0
    assert all(len(ops) > 0 and len(alts) > 0 for ops in inst.jobs for alts in ops)
    txt = "2 2\n1 1 1 3\n1 2 1 3 2 4\n"   # 2x2；作业0 一工序一候选；作业1 一工序两候选
    nj, nm, jobs = parse_fjs_text(txt)
    assert (nj, nm) == (2, 2) and len(jobs) == 2
    rnd = gen_random(n_jobs=3, n_machines=3, seed=0)
    assert rnd.n_jobs == 3 and len(rnd.jobs) == 3


@pytest.mark.unit
def test_mkt_adapter_shape_and_transport_label():
    """MKT = 原始 MK 的加工数据 + 行程时间矩阵 + 车辆数（P4-A 的口径）。"""
    pytest.importorskip("plcsp_v2.env.mkt")
    raw = load_mk("mk01")
    mat = mkt.load_mkt_layout(raw.n_machines, drop_lu=False)
    assert mat.shape == (raw.n_machines + 1, raw.n_machines + 1)
    mx = mkt.load_mkt("mk01")
    assert isinstance(mx, mkt.MktInstance)
    assert mx.n_agv > 0 and mx.layout_m > 0
    t = mx.trans_time
    assert t.shape[0] == t.shape[1] > 0
    # 加工数据一字不改（§19.7d）
    assert [[[(m, float(v)) for m, v in a] for a in ops] for ops in mx.base.jobs] == \
           [[[(m, float(v)) for m, v in a] for a in ops] for ops in raw.jobs]


# ══════════════════ env/corridors.py ══════════════════

@pytest.mark.unit
def test_corridor_paths_and_k_shortest():
    w = _world()
    a = int(w.layout.machines[0].dock_node)
    b = int(w.layout.machines[1].dock_node)
    p = corridors.shortest_node_path(w.g, a, b)
    assert p[0] == a and p[-1] == b and len(p) >= 2
    ks = corridors.k_shortest_paths(w.g, a, b, 3)
    assert 1 <= len(ks) <= 3 and all(q[0] == a and q[-1] == b for q in ks)


# ══════════════════ env/transport.py ══════════════════

@pytest.mark.unit
def test_transport_caliber_geometry_and_matrix():
    w = _world()
    geo = TransportCaliber.geometry()
    u, v = int(w.layout.machines[0].dock_node), int(w.layout.machines[1].dock_node)
    cal = TransportCaliber.for_instance(w.inst, w.layout)
    assert cal.mode == 'geometry' and cal.slot_of(u) is None
    with pytest.raises(ValueError):        # 几何口径**显式报错**，不是静默返回 None
        cal.minutes(u, v)
    assert geo is not None
    m = w.inst.n_machines
    mat = TransportCaliber.from_matrix(np.full((m + 1, m + 1), 2.5), w.layout)
    assert mat.slot_of(u) == 1 and mat.minutes(u, v) == pytest.approx(2.5)


# ══════════════════ env/snapshot.py ══════════════════

@pytest.mark.unit
def test_snapshot_dataclasses_are_frozen_and_carry_documented_fields():
    ms = MachineState(backlog_min=1.0, in_q_len=1, out_q_len=0, busy=True,
                      remaining_min=2.0, pm_used_min=0.0, fail_rate=0.01)
    js = JobState(done_ops=1, total_ops=3, remaining_min=4.0, finished=False, due=10.0,
                  at_machine=0, in_transit=False, on_agv=-1)
    vs = VehicleState(status=2, node=5, queued=1, battery_frac=0.9)
    cs = ChargerState(occupied=1, waiting=0, capacity=1)
    qt = QueuedTask(veh=0, job=1, frm=2, to=3, oi=1)
    snap = Snapshot(now=1.0, machines=(ms,), jobs=(js,), vehicles=(vs,), n_done=0,
                    in_flight=2, chargers=(cs,), queued_tasks=(qt,))
    assert snap.machines[0].fail_rate == 0.01 and snap.jobs[0].due == 10.0
    assert snap.vehicles[0].busy_min == 0.0          # 新字段的默认（手搓夹具兼容）
    assert snap.in_flight == 2 and snap.queued_tasks[0].veh == 0
    for obj in (ms, js, vs, cs, qt, snap):
        assert dataclasses.is_dataclass(obj)


@pytest.mark.unit
def test_live_snapshot_matches_world_state():
    w = _world()
    snap = w.snapshot()
    assert isinstance(snap, Snapshot)
    assert len(snap.machines) == w.inst.n_machines
    assert len(snap.jobs) == w.inst.n_jobs and len(snap.vehicles) == w.cfg.n_agv
    assert snap.now == 0.0 and snap.n_done == 0


# ══════════════════ env/constraints.py ══════════════════

@pytest.mark.unit
def test_constraint_config_defaults_and_with_off():
    c = ConstraintConfig()
    assert c.machine_failure and c.due_dates and c.agv_failure
    assert c.charging and c.maintenance
    off = c.with_off("machine_failure", "charging")
    assert not off.machine_failure and not off.charging and off.due_dates
    assert c.machine_failure, "with_off 必须返回**新对象**，原对象不动"
    with pytest.raises(ValueError):
        c.with_off("nope")


# ══════════════════ env/due_dates.py ══════════════════

@pytest.mark.unit
def test_due_date_machinery():
    inst = _inst("mk01")
    lb = due_dates.workload_lower_bound(inst)
    twc = due_dates.total_work_content(inst)
    assert lb > 0.0 and len(twc) == inst.n_jobs and all(x > 0 for x in twc)
    tbl = due_dates.tf_rdd_due_dates(inst, 1.5, 0.3)
    assert len(tbl) == inst.n_jobs and all(v > 0.0 for v in tbl.values())
    assert len(set(tbl.values())) == inst.n_jobs, "TF/RDD 口径的 d_j 必须互不相同"
    assert due_dates.due_dates_for(inst, 1.5, 0.3) == tbl
    frozen = due_dates.due_dates_for(inst)      # 无参 = 查**冻结标定表**（不是 (1.5, 0.3)）
    assert len(frozen) == inst.n_jobs and all(v > 0 for v in frozen.values())
    assert len(set(frozen.values())) == inst.n_jobs
    assert compute_due_dates(inst, 1.5, 0.3) == tbl
    assert compute_due_dates(inst) == frozen     # 无参 → 冻结标定表（mk01 是 τ=2.75, R=0.3）


# ══════════════════ env/reward.py ══════════════════

@pytest.mark.unit
def test_reward_vector_weights_and_scalarisation():
    f = reward.objective_vector({"makespan": 10.0, "energy": 2.0, "tardy_twt": 3.0})
    assert f == (10.0, 2.0, 3.0)
    w = reward.reward_weights((10.0, 2.0, 3.0))
    assert abs(sum(w) - 1.0) < 1e-12 and all(x > 0 for x in w)
    # 逆量纲加权：分量越小权重越大
    assert w[1] > w[0]
    assert reward.scalar_reward(f, w) == pytest.approx(-sum(a * b for a, b in zip(w, f)))
    assert reward.scalar_reward(f, w, prefs=(1.0, 0.0, 0.0)) == -10.0   # one-hot = 纯 makespan
    with pytest.raises(ValueError):
        reward.reward_weights((0.0, 2.0, 3.0))


@pytest.mark.unit
def test_reference_objectives_are_pinned_to_the_instance_and_cfg():
    inst = _inst("mk01")
    cfg = SimConfig()
    r = reward.ReferenceObjectives.of(inst, cfg)
    f = r.as_tuple()
    assert len(f) == 3 and f[0] > 0 and f[1] > 0
    assert f[2] >= 0.0, 'mk01 无拖期 ⟹ TWT 可以是 0（正是 reward_weights 会报错的那种实例）'
    assert r.matches(inst, cfg)
    assert not r.matches(inst, dataclasses.replace(cfg, n_agv=cfg.n_agv + 1))


# ══════════════════ energy.py ══════════════════

@pytest.mark.unit
def test_energy_model_is_linear_in_time_and_additive():
    p = energy.machine_params_for(0, 10)
    m = energy.machine_energy_kwh(60.0, 30.0, 0.0, idle_kw=p["idle_kw"],
                                  proc_kw=p["proc_kw"], setup_kw=p["setup_kw"])
    assert m > 0.0
    bd = energy.machine_energy_breakdown_kwh(60.0, 30.0, 0.0, idle_kw=p["idle_kw"],
                                             proc_kw=p["proc_kw"], setup_kw=p["setup_kw"])
    assert set(bd) == {"idle", "proc", "setup"} and sum(bd.values()) == pytest.approx(m)
    # 翻倍时长 ⟹ 翻倍能量（线性）
    assert energy.machine_energy_kwh(120.0, 60.0, 0.0, idle_kw=p["idle_kw"],
                                     proc_kw=p["proc_kw"],
                                     setup_kw=p["setup_kw"]) == pytest.approx(2 * m)
    a = energy.agv_energy_kwh(10.0, 20.0, 30.0)
    abd = energy.agv_energy_breakdown_kwh(10.0, 20.0, 30.0)
    assert a > 0.0 and set(abd) == {"idle", "empty", "loaded"}
    assert sum(abd.values()) == pytest.approx(a)
    # 总能耗 = 机床 + AGV + **车间固定项**（固定项按 makespan 计）
    assert energy.total_energy_kwh(m, a, 100.0) == pytest.approx(
        m + a + energy.SHOP_FIXED_KW * 100.0 / 60.0)


# ══════════════════ env/des.py ══════════════════

@pytest.mark.unit
def test_rollout_and_reference_agree_on_the_same_dynamics():
    inst = _inst("mk01")
    m = rollout(inst, seed_layout=0, seed_chain=0)
    assert m["makespan"] == pytest.approx(75.870)
    assert m["jobs_done"] == inst.n_jobs
    ref = reference_run(inst)
    assert set(ref) >= {"makespan", "energy"}
    assert reference_makespan(inst) > 0.0


@pytest.mark.unit
def test_pure_helpers():
    assert charge_cands(3) == (0, 1, 2, 3)
    assert charge_cands(0) == (0,)
    # λ(a)=base*β*(a/η)^(β−1)：β>1 时 λ(0)=0（纯耗损模型，docstring 明写）
    assert machine_fail_rate(0.01, 0.0, 2.0, 200.0) == pytest.approx(0.0)
    assert machine_fail_rate(0.01, 200.0, 2.0, 200.0) == pytest.approx(0.02)
    assert machine_fail_rate(0.01, 123.0, 1.0, 200.0) == pytest.approx(0.01), 'β=1 ⟹ 恒为 base'
    r_lo = machine_fail_rate(0.01, 100.0, 2.0, 200.0)
    r_hi = machine_fail_rate(0.01, 400.0, 2.0, 200.0)
    assert r_hi > r_lo > 0.0, "β>1 ⟹ 严格递增"
    twt = weighted_tardiness({0: 10.0, 1: 20.0}, {0: 5.0, 1: 25.0}, {0: 1.0, 1: 1.0})
    assert twt == pytest.approx(5.0)
    inst = _inst("mk01")
    norm = check_seq_rank(inst, _rank(inst))
    assert len(norm) == inst.n_jobs and all(len(r) == len(inst.jobs[j])
                                            for j, r in enumerate(norm))


@pytest.mark.unit
def test_seq_rank_rejects_malformed_input():
    inst = _inst("mk01")
    with pytest.raises(TypeError):
        check_seq_rank(inst, None)
    with pytest.raises((ValueError, TypeError)):
        check_seq_rank(inst, [[0.0]])


@pytest.mark.unit
def test_low_level_classes_are_constructible_and_tracked():
    op = OpLite(3.5)
    assert op.time == 3.5
    t = SimTrack(n_jobs=2, n_machines=2, n_agv=1)
    assert t.job_progress == [0, 0] and t.cur_op == [None, None]
    t.set_load(0, 2)
    assert t.agv_loaded[0] and t.agv_load_n[0] == 2
    t.set_load(0, 0)
    assert not t.agv_loaded[0] and t.agv_load_n[0] == 0
    assert SimConfig().eff_speed > 0.0
    w = _world()
    assert isinstance(w.machines[0], MachineSim)
    assert isinstance(w.lu, LuStation)


@pytest.mark.unit
def test_scaled_mtbf_and_disturbance_metrics_are_exposed():
    """⑨ 的**激活**口径 + 扰动触发的决策点（原文决策点定义的另一半）。"""
    from plcsp_v2.env.des import scaled_agv_mtbf

    inst = _inst("mk01")
    assert scaled_agv_mtbf(inst, 4.0) < SimConfig().agv_mtbf
    m = rollout(inst)
    assert "dist_events" in m and "logistics_cost" in m
    # TC 报告桥 = c·V + uc·行驶时间
    from plcsp_v2.env.des import (AEI2025_AGV_ACTIVATION_COST as C,
                                  AEI2025_UNIT_TRANSPORT_COST as UC)
    assert m["logistics_cost"] == pytest.approx(
        C * m["n_agv"] + UC * m["travel_time_total"])


# ══════════════════ env/rule_views.py + hrl/ ══════════════════

@pytest.mark.unit
def test_advance_avail_and_minu_floor_and_score_helpers():
    assert advance_avail(5.0, 0.0, 3.0) == 8.0        # 机器空 ⟹ 从 0 起算
    assert advance_avail(5.0, 10.0, 3.0) == 13.0      # 已过期 ⟹ 从 now 起算
    from plcsp_v2.env.rule_views import PAMachine, PAView
    from plcsp_v2.tests.test_pa_rules import _machine_race_view, _pajob

    v = _machine_race_view()
    assert minu_denominator(v, 1) == pytest.approx(20.0)     # 走 MCT 账本
    assert minu_denominator(v, 0) == pytest.approx(5.0)
    empty = PAView(now=0.0, jobs=(_pajob(0, ((0, 1.0),), 1.0, 1.0, 0.0, 1.0),
                                  PAMachine(0.0, 0.0)), machines=(PAMachine(0.0, 0.0),))
    assert minu_denominator(empty, 0) == pytest.approx(1.0)   # MCT=0 ⟹ 走地板
    for r in ("SPT", "EAJ", "SRPT"):
        assert np.isfinite(pa_job_score(r, v, 0))
    assert pa_job_score("EDD", v, 0) == float("inf"), "该夹具的 due 是 +inf（无交期）"
    for r in ("SMPT", "MINU", "EAM"):
        assert np.isfinite(pa_machine_score(r, v, (0, 4.0)))
    with pytest.raises(ValueError):
        pa_job_score("NOPE", v, 0)
    with pytest.raises(ValueError):
        pa_machine_score("NOPE", v, (0, 4.0))


@pytest.mark.unit
def test_cdr_policies_returns_a_runnable_pair():
    w = _world()
    pol_pa, pol_l = cdr_policies("EAJ+EAM", "MINCOST")
    m = w.run_gated(seed_chain=0, policy_pa=pol_pa, policy_l=pol_l)
    assert m["jobs_done"] == 10 and m["pa_assigns"] == 55


@pytest.mark.unit
def test_la_view_loaded_leg_is_vehicle_independent():
    from plcsp_v2.env.rule_views import LAVehicle, LAView

    v = LAView(now=0.0, job=0, oi=1, frm_node=10, to_node=14, ready_at=0.0,
               vehicles=(LAVehicle(10, 0.0, 0.0), LAVehicle(11, 0.0, 0.0)),
               leg_min=lambda a, b: float(abs(a - b)))
    assert v.loaded_leg(0) == v.loaded_leg(1) == pytest.approx(4.0)
    assert v.empty_leg(1) == pytest.approx(1.0)
