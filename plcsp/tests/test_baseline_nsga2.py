"""NSGA-II 对位基线的测试（spec §6.1 的 ③ 层）。

钉住四件事：

1. **评价口径**——走 `rollout`（FIFO 规则派车）、默认不传 `agv_phi`、档位取 `REPORT_TIERS`；
2. **三目标**——返回 (makespan, energy, TWT)，且档 A 的 TWT 恒 0（⑧ 关，口径使然）；
3. **可复现**——同 seed 两轮跑出逐位相同的目标值；
4. **代价记账**——实测仿真次数 ≤ pop×(gen+1) 的理论上界，且随结果带出。
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_transport_task_count_matches_the_simulator():
    """车号块的基因数 = 仿真实际生成的运输任务数（65 for MK01）。

    这是 `agv_genes=True` 的**长度假设**：`_transporter` 每生成一个任务就 ++task_i。
    若某天任务结构变了（如多卸货点改了回站口径），此测试先红——否则基因会静默错位。
    """
    from plcsp.baselines.nsga2 import transport_task_count
    from plcsp.env.constraints import REPORT_TIERS
    from plcsp.env.des import SimConfig, rollout
    from plcsp.env.mkt import load_mkt

    mkt = load_mkt("mk01")
    n_tasks = transport_task_count(mkt.base)
    r = rollout(mkt.base, seed_chain=0,
                cfg=SimConfig(n_agv=mkt.n_agv, transport_unmapped="geometry"),
                constraints=REPORT_TIERS["A-MKT"])
    assert n_tasks == r["dbg"]["tasks_put"] == 65, (
        f"任务数假设不符：静态计数={n_tasks}，仿真生成={r['dbg']['tasks_put']}")


@pytest.mark.unit
def test_decode_respects_bounds_and_clips():
    """基因 → `op_choices`：序号必须落在该工序的候选表内。浮点/越界值裁剪。"""
    from plcsp.baselines.nsga2 import chromosome_length, decode
    from plcsp.env.instances import gen_random

    inst = gen_random(n_jobs=3, n_machines=4, seed=0)
    n_ops = sum(len(job) for job in inst.jobs)
    x = np.full(chromosome_length(inst), 10.0)          # 越界（候选最多 4 个）
    ms, phi = decode(x, inst, agv_genes=False)
    assert phi is None
    assert sum(len(row) for row in ms) == n_ops
    for job, row in zip(inst.jobs, ms):
        for alts, idx in zip(job, row):
            assert 0 <= idx < len(alts), "解码越界——候选序号会静默指向别的机台"

    x_agv = np.full(chromosome_length(inst, agv_genes=True), -3.0)   # 下越界
    ms2, phi2 = decode(x_agv, inst, n_agv=3, agv_genes=True)
    assert len(phi2) == chromosome_length(inst, agv_genes=True) - n_ops
    assert set(phi2) == {0}, "负值未裁到下界"


@pytest.mark.unit
def test_evaluation_uses_the_fifo_rollout_entry(monkeypatch):
    """⚠️ 口径绊线：评价必须走 `rollout`（规则派车），且默认**不传** `agv_phi`。

    改走 `run_gated`（在线策略入口）或默认开 bound 派车，本测试当场红。
    """
    from plcsp.baselines import nsga2 as mod
    from plcsp.env.instances import gen_random

    seen: list[dict] = []
    real = mod.rollout

    def spy(inst, **kw):
        seen.append(kw)
        return real(inst, **kw)

    monkeypatch.setattr(mod, "rollout", spy)
    inst = gen_random(n_jobs=3, n_machines=4, seed=1)
    prob = mod.MachineAssignmentProblem(inst, mod.EvalSpec(tier="A-MKT", n_agv=2))
    prob.evaluate_one(np.zeros(prob.n_var))
    assert seen and seen[-1]["op_choices"] is not None, "评价没把机台选择交给仿真"
    assert seen[-1]["agv_phi"] is None, (
        "默认档传了 agv_phi ⟹ 派车从共享队列 FIFO 变成了 bound 逐车队列，口径变了")
    assert seen[-1]["constraints"] is mod.REPORT_TIERS["A-MKT"]


@pytest.mark.unit
def test_small_budget_run_on_mk01_returns_three_objectives():
    """小预算真跑（MK01 MKT，pop=6 × gen=1）：三目标、前沿非空、代价记账。"""
    from plcsp.baselines.nsga2 import EvalSpec, run_nsga2
    from plcsp.env.mkt import load_mkt

    mkt = load_mkt("mk01")
    res = run_nsga2(mkt.base, EvalSpec(tier="A-MKT", seed_chain=0, n_agv=mkt.n_agv),
                    pop_size=6, n_gen=1, seed=0)
    assert res.F.shape[1] == 3, "目标数不是三种"
    assert res.F.shape[0] >= 1 and res.X.shape[0] == res.F.shape[0]
    assert np.all(res.F[:, 2] == 0.0), (
        "档 A 关着 ⑧ 交期 ⟹ TWT 必须恒 0；非零说明档位/交期接线变了")
    assert np.all(res.F[:, 1] > 0.0), "能耗必须为正（M2 模型）"
    assert 6 <= res.n_eval <= 6 * 2, f"实测仿真次数 {res.n_eval} 超出 [pop, pop×(gen+1)]"
    assert res.sim_s > 0.0 and res.wall_s >= res.sim_s


@pytest.mark.unit
def test_run_is_reproducible_bitwise():
    """同 seed + 同仿真链 ⟹ 目标值逐位相同（pymoo 随机流与仿真都确定性）。"""
    from plcsp.baselines.nsga2 import EvalSpec, run_nsga2
    from plcsp.env.instances import gen_random

    inst = gen_random(n_jobs=4, n_machines=3, seed=2)
    spec = EvalSpec(tier="A-MKT", seed_chain=0, n_agv=2)
    a = run_nsga2(inst, spec, pop_size=6, n_gen=1, seed=3)
    b = run_nsga2(inst, spec, pop_size=6, n_gen=1, seed=3)
    assert np.array_equal(a.F, b.F), "同 seed 两轮结果不同——不可复现"
    assert np.array_equal(a.X, b.X)


@pytest.mark.unit
def test_front_points_are_mutually_nondominated():
    """返回的前沿点两两不支配（报告里贴的就是这些点）。"""
    from plcsp.baselines.nsga2 import EvalSpec, run_nsga2
    from plcsp.env.instances import gen_random

    inst = gen_random(n_jobs=4, n_machines=3, seed=4)
    res = run_nsga2(inst, EvalSpec(tier="A-MKT", seed_chain=0, n_agv=2),
                    pop_size=8, n_gen=2, seed=5)
    F = res.F
    for i in range(len(F)):
        for j in range(len(F)):
            if i == j:
                continue
            dominates = np.all(F[j] <= F[i]) and np.any(F[j] < F[i])
            assert not dominates, f"点 {j} 支配点 {i}——前沿里混进了被支配点"


@pytest.mark.unit
def test_tier_flags_document_the_two_calibers():
    """两档的机制开关必须与 `REPORT_TIERS` 同源：档 A 关 ⑧（TWT 死）、档 B 开 ⑧。"""
    from plcsp.env.constraints import REPORT_TIERS

    assert REPORT_TIERS["A-MKT"].due_dates is False
    assert REPORT_TIERS["B-Full"].due_dates is True
