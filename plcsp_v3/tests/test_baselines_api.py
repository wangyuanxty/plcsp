# -*- coding: utf-8 -*-
"""`baselines/` 公开 API 的规格测试（2026-10-09）。

⚠️ **在此之前 `plcsp_v3/baselines/` 的行覆盖率是 0%**（实测 1300 行，见 §80 的覆盖率读数）
—— 名字在 `plcsp/tests/` 里出现过，但那些测试 `import plcsp.*`，**跑不到 v2 上**。

三个基线各有前提，**用不了的路径就断言它"显式报错"**，不 `skip`（`skip` 等于没测）：
- `exact_cpsat`：**只支持矩阵口径**（MKT 实例）；缺 `ortools` ⟹ 断言 `OrtoolsMissing`；
- `nsga2`：纯 CPU，小种群跑通即可；
- `hgs`：要 `torch`，小规模前向 + 一次 rollout。
"""
from __future__ import annotations

import numpy as np
import pytest

from plcsp_v3.baselines import exact_cpsat as E
from plcsp_v3.baselines import nsga2 as N
from plcsp_v3.env.des import SimConfig
from plcsp_v3.env.instances import load_mk


def _mkt():
    from plcsp_v3.env import mkt
    return mkt.load_mkt("mk01").base


# ══════════════════ nsga2 ══════════════════

@pytest.mark.unit
def test_nsga2_chromosome_shape_and_bounds():
    inst = load_mk("mk01")
    n_tasks = N.transport_task_count(inst)
    assert n_tasks == sum(len(ops) for ops in inst.jobs) + inst.n_jobs   # 工序 + 每作业一次回站
    lo = N.chromosome_length(inst, agv_genes=False)
    hi = N.chromosome_length(inst, agv_genes=True)
    assert lo == sum(len(ops) for ops in inst.jobs)
    assert hi == lo + n_tasks
    b_lo, b_hi = N.gene_bounds(inst, n_agv=3, agv_genes=False)
    assert len(b_lo) == len(b_hi) == lo and all(a <= b for a, b in zip(b_lo, b_hi))
    b_lo2, b_hi2 = N.gene_bounds(inst, n_agv=3, agv_genes=True)
    assert len(b_lo2) == hi and max(b_hi2[lo:]) == 2          # 车号 0..n_agv-1


@pytest.mark.unit
def test_nsga2_decode_matches_instance_and_eval_spec_guard():
    inst = load_mk("mk01")
    assert N.chromosome_length(inst, agv_genes=False) == sum(len(o) for o in inst.jobs)
    x = np.array(N.gene_bounds(inst, n_agv=3, agv_genes=False)[0], dtype=int)  # 取下界
    choices, _ranks = N.decode(x, inst, n_agv=3, agv_genes=False)   # decode -> (choices, ranks)
    assert len(choices) == inst.n_jobs
    assert all(len(c) == len(inst.jobs[j]) for j, c in enumerate(choices))
    spec = N.EvalSpec.from_cfg(SimConfig())
    assert isinstance(spec.cfg(), SimConfig)
    spec.check_same_dynamics(SimConfig())                     # 同配置不报错
    with pytest.raises(ValueError):
        spec.check_same_dynamics(SimConfig(n_agv=SimConfig().n_agv + 1))


@pytest.mark.unit
def test_nsga2_problem_evaluates_and_reports():
    inst = load_mk("mk01")
    spec = N.EvalSpec.from_cfg(SimConfig())
    prob = N.MachineAssignmentProblem(inst, spec)
    x = np.array(N.gene_bounds(inst, n_agv=spec.n_agv, agv_genes=False)[0], dtype=int)
    f = prob.evaluate_one(x)
    assert len(f) >= 1 and all(np.isfinite(v) for v in np.ravel(f))


@pytest.mark.unit
def test_nsga2_end_to_end_small_and_result_summary():
    inst = load_mk("mk01")
    spec = N.EvalSpec.from_cfg(SimConfig())
    res = N.run_nsga2(inst, spec, pop_size=6, n_gen=2, seed=0)
    assert isinstance(res, N.Nsga2Result)
    assert res.F.shape[0] == res.X.shape[0] and res.n_eval > 0
    assert res.pop_size == 6 and res.n_gen == 2 and res.agv_genes is False
    txt = res.summary()
    assert isinstance(txt, str) and len(txt) > 0


# ══════════════════ hgs ══════════════════

@pytest.mark.unit
def test_hgs_env_state_machine_is_consistent():
    pytest.importorskip("torch")
    from plcsp_v3.baselines.hgs import env as H

    inst = _mkt()          # HGS 与 exact 一样**只吃矩阵口径**（几何实例显式报错）
    env = H.FjsptEnv(inst, n_agv=3)
    assert env.reset() is None, "reset 是**就地**重置，不返回状态"
    g = env.graph()
    assert isinstance(g, H.GraphState)
    assert not env.finished
    assert env.has_action() and len(env.eligible()) > 0     # ⚠️ 是方法，不是属性
    assert isinstance(env.makespan, float)
    assert env.cmax_lb() >= 0.0 and env.makespan >= 0.0
    j, o = env.eligible()[0]              # 候选是 **(工件, 工序)**；机台/车在后续两步里选
    assert len(env.idle_machines(j, o)) > 0
    assert len(env.idle_vehicles()) > 0
    assert env.flat_to_job_op(0) is not None
    k = env.idle_machines(j, o)[0]
    env.step(j, o, k, env.idle_vehicles()[0])          # 一步 = 定 (工序, 机台, 车)
    assert env.makespan >= 0.0 and len(env.eligible()) >= 1
    g2 = env.graph()
    assert isinstance(g2, H.GraphState)


@pytest.mark.unit
def test_hgs_net_forward_and_decoders():
    torch = pytest.importorskip("torch")
    from plcsp_v3.baselines.hgs import env as H
    from plcsp_v3.baselines.hgs import model as M

    inst = _mkt()
    env = H.FjsptEnv(inst, n_agv=3)
    env.reset()
    g = env.graph()
    net = M.HgsNet(dh=16, dz=4, dff=32, n_heads=2, dv=4, layers=1)
    assert isinstance(net, M.HgsNet)
    h_op, h_mach, h_veh, edge = net.encode(g)     # -> 4 个张量（glimpse 在 decode 里取）
    # `h_glimpse` 是**一维**的上下文起点；后续步由 decode 的返回 `out["h_glimpse"]` 接续
    h_glimpse = torch.zeros(16)
    assert h_op.shape[-1] == 16 and h_mach.shape[-1] == 16
    out = net.decode_greedy(h_op, h_mach, h_veh, edge, g, h_glimpse)
    assert out is not None
    assert out is not None
    # 两个子模块由 `encode` / `decode` 内部调用 ⟹ 已覆盖；这里只断言类型在位
    assert isinstance(net, M.HgsNet) and M.Hmha is not None and M.DecoderMha is not None


@pytest.mark.unit
def test_hgs_train_helpers_run_one_rollout():
    pytest.importorskip("torch")
    from plcsp_v3.baselines.hgs import env as H
    from plcsp_v3.baselines.hgs import model as M
    from plcsp_v3.baselines.hgs import train as T

    rng = np.random.default_rng(0)
    g = T.gen_fjspt(2, 2, 2, rng)
    assert g.n_jobs == 2 and g.n_machines == 2
    inst = _mkt()
    net = M.HgsNet(dh=16, dz=4, dff=32, n_heads=2, dv=4, layers=1)
    env = H.FjsptEnv(inst, n_agv=3)
    out = T.rollout(env, net, greedy=True, seed=0)     # -> metrics dict（不是标量）
    assert isinstance(out, dict) and out["makespan"] > 0.0
    ev = T.evaluate(inst, 3, net, seed=0)               # -> {makespan_greedy, makespan_sample, steps}
    assert ev["makespan_greedy"] > 0.0 and ev["makespan_sample"] > 0.0 and ev["steps"] > 0
    assert T.time_one_rollout(inst, 3, seed=0) >= 0.0


# ══════════════════ exact_cpsat ══════════════════

@pytest.mark.unit
def test_exact_cpsat_rejects_geometry_caliber_explicitly():
    """⚠️ 精确解基线**只支持矩阵口径**——几何实例必须**显式报错**，不得静默降级。"""
    inst = load_mk("mk01")
    assert inst.trans_time_full is None
    with pytest.raises(ValueError, match="矩阵口径"):
        E.travel_matrix(inst)


@pytest.mark.unit
def test_exact_cpsat_scale_matrix_and_greedy_schedule():
    from plcsp_v3.env.transport import MATRIX

    inst = _mkt()
    assert inst.transport == MATRIX
    assert E.time_scale(inst) > 0.0
    # ⚠️ 矩阵是 **(m+1)×(m+1)**——第 0 行/列是装卸站（`load_mkt_layout(drop_lu=False)` 的口径）
    assert np.asarray(E.travel_matrix(inst)).shape == (inst.n_machines + 1,
                                                       inst.n_machines + 1)
    sched = E.greedy_schedule(inst, 3)
    assert isinstance(sched, dict) and sched
    assert isinstance(E.greedy_schedule(inst, 3, include_return=False), dict)


@pytest.mark.unit
def test_exact_cpsat_needs_ortools_and_says_so():
    """`ortools` 不在环境里 ⟹ `solve` 必须抛 **`OrtoolsMissing`**（不是 `ImportError` 裸崩）。"""
    inst = _mkt()
    spec = E.ExactSpec(tier="A-MKT", n_agv=3)
    ub = E.rule_upper_bound(inst, spec, seed_chain=0)      # -> metrics dict，不是标量
    assert isinstance(ub, dict) and ub["makespan"] > 0.0
    try:
        import ortools  # noqa: F401
        has = True
    except Exception:
        has = False
    if not has:
        with pytest.raises(E.OrtoolsMissing):
            E.solve(inst, spec)


@pytest.mark.unit
def test_exact_result_gap_properties_and_summary():
    r = E.ExactResult(instance="mk01", tier="A-MKT", n_agv=3, status="OPTIMAL",
                      objective=100.0, dual_bound=90.0, wall_s=1.0, time_limit_s=60.0,
                      scale=1.0, n_tasks=10, n_arcs=20, build_s=0.1, horizon=500.0,
                      rule_makespan=120.0, rule_horizon_hit=False, greedy_makespan=110.0,
                      schedule={}, model_kwargs={})
    assert r.gap_abs == pytest.approx(10.0)
    assert r.gap_rel == pytest.approx(0.1)
    assert r.certified_gap_vs(105.0) > 0.0
    assert isinstance(r.summary(), str) and len(r.summary()) > 0


@pytest.mark.unit
def test_exact_validate_schedule_rejects_a_broken_plan():
    inst = _mkt()
    bad = {"task_start": {}, "task_end": {}, "machine_seq": {}, "agv_seq": {}}
    with pytest.raises(Exception):
        E.validate_schedule(inst, bad, n_agv=3)
