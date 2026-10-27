"""A-MKT 精确解基线（`plcsp.baselines.exact_cpsat`）的测试。

钉住四件事：

1. **下界性**——精确解的最优值/对偶界**永不超过**同实例同口徑的规则基线 makespan
   （`rollout`，A-MKT 档）。这是整个模块的立身之本：策略的可行调度 ⊂ 模型可行集。
2. **解的可行性**——抽出的调度经**独立校验器**逐条复算（机台不重叠、作业先后、
   行程 = 矩阵查表、每车空载、完工 = 回站到达），且**用本仓仿真器**复核。
3. **口径守卫**——只支持 A-MKT（B-8 含随机机制，没有单点最优解可证）；
   只支持矩阵口径；整数缩放不引入四舍五入。
4. **手算小例**——两道上界可笔算的实例，钉住"模型不是碰巧对"。

⚠️ 本文件在**门禁解释器（无 ortools）**下整支跳过（`importorskip`）。要真跑：
`D:/anaconda/python.exe -m pip install ortools`（会改门禁环境，需先征得同意），
或临时 `PYTHONPATH=<含 ortools 的目录>`。
"""
from __future__ import annotations

import numpy as np
import pytest

from plcsp.env.instances import Instance


def _ortools_or_skip():
    pytest.importorskip("ortools", reason="CP-SAT 未安装——精确解基线的求解测试整支跳过")


def _tiny(n_machines: int, jobs, mat, name: str) -> Instance:
    """手搓的矩阵口径实例（第 0 行/列 = 装卸站 LU，机台 i ↦ 下标 i+1）。"""
    return Instance(n_jobs=len(jobs), n_machines=n_machines, jobs=jobs, source=name,
                    transport="matrix", trans_time_full=np.asarray(mat, dtype=float))


def _mk01_sub(n_jobs: int, lo: int, hi: int) -> Instance:
    """MK01 前若干作业、机台号 [lo,hi) 的**真实数据**子实例（矩阵取对应的子块）。"""
    from plcsp.env.mkt import load_mkt
    inst = load_mkt("mk01").base
    jobs = [[[(mm - lo, t) for mm, t in alts if lo <= mm < hi] for alts in job]
            for job in inst.jobs[:n_jobs]]
    assert all(alts for job in jobs for alts in job), "子实例里有工序没有剩余候选机台"
    m = hi - lo
    mat = np.zeros((m + 1, m + 1))
    for u in range(m + 1):
        for v in range(m + 1):
            mat[u, v] = inst.trans_time_full[0 if u == 0 else u + lo, 0 if v == 0 else v + lo]
    return Instance(n_jobs=n_jobs, n_machines=m, jobs=jobs, source=f"mk01-sub{n_jobs}",
                    transport="matrix", trans_time_full=mat)


# ══════════════════ 1. 纯逻辑（不需要 ortools）══════════════════

@pytest.mark.unit
def test_time_scale_is_one_for_integer_data_and_two_for_half_minutes():
    """缩放只取**公共分母**、不四舍五入——整数数据（MK01）⟹ 1；MK06 的 `x.5` 矩阵 ⟹ 2。

    ⚠️ 四舍五入会破坏下界性（把时长改小 = 放松 = 下界更低但仍"合法"；改大则**非法**）。
    故本模块只做精确缩放：分母 > 上限时显式报错，不静默取整。
    """
    from plcsp.baselines.exact_cpsat import time_scale
    assert time_scale(_mk01_sub(3, 0, 3)) == 1
    assert time_scale(_tiny(1, [[[(0, 3.0)]]], [[0, 2.5], [2.5, 0]], "half")) == 2


@pytest.mark.unit
def test_travel_matrix_zeroes_the_diagonal_even_when_the_data_does_not():
    """`travel_matrix` 把对角线**置 0**：`AgvSim._drive(src == dst)` 直接返回、不查表。

    上游 MKT 数据里 `m=17` 的对角第 6 项是 4（README §2 已记）。不置 0 会给"同点往返"
    凭空加 4 分钟 ⟹ 模型比 DES 紧 ⟹ 下界性失效。此处用**合成**的对角非零矩阵钉住该行为。
    """
    from plcsp.baselines.exact_cpsat import travel_matrix
    inst = _tiny(2, [[[(0, 1.0)]]], [[0, 4, 7], [4, 0, 9], [7, 9, 0]], "diag")
    T = travel_matrix(inst)
    assert [T[i][i] for i in range(3)] == [0, 0, 0]
    assert (T[0][1], T[1][0], T[1][2]) == (4, 4, 9)
    odd = _tiny(2, [[[(0, 1.0)]]], [[0, 4, 7], [4, 4, 9], [7, 9, 0]], "diag#17 型异常")
    assert travel_matrix(odd)[1][1] == 0


@pytest.mark.unit
def test_task_table_matches_the_simulator_task_count():
    """任务表规模 = `nsga2.transport_task_count`（Σ每作业(工序数+1)）——与仿真同构。

    MK01 = 55 + 10 = 65；去掉回站段后 = 55。
    """
    from plcsp.baselines.exact_cpsat import _task_table
    from plcsp.baselines.nsga2 import transport_task_count
    from plcsp.env.mkt import load_mkt
    inst = load_mkt("mk01").base
    assert len(_task_table(inst, True)) == transport_task_count(inst) == 65
    assert len(_task_table(inst, False)) == 55


@pytest.mark.unit
def test_unsupported_tier_and_caliber_are_rejected():
    """B-8（随机机制）与几何口径**显式报错**，不静默给一个不成立的"最优解"。"""
    from plcsp.baselines.exact_cpsat import ExactSpec, build_model
    from plcsp.env.instances import load_mk
    with pytest.raises(ValueError, match="B-8|只支持 A-MKT"):
        build_model(_mk01_sub(2, 0, 3), ExactSpec(tier="B-8", n_agv=1))
    with pytest.raises(ValueError, match="矩阵口径"):
        build_model(load_mk("mk01"), ExactSpec(n_agv=1))


# ══════════════════ 2. 手算小例（需要 ortools）══════════════════

@pytest.mark.unit
def test_tiny_single_machine_optimum_is_hand_checkable():
    """1 机台 2 作业、p=10、LU↔M0=4 min ⟹ **J\\* = 4 + 20 + 4 = 28**。

    下界三段：首趟必须从装卸站出发（4 min）、机台串行做 20 min、末件必须回站（4 min）。
    上界：两件各占一台车同时投放，机台串行 → 恰好 28。
    """
    _ortools_or_skip()
    from plcsp.baselines.exact_cpsat import ExactSpec, solve, validate_schedule
    inst = _tiny(1, [[[(0, 10.0)]], [[(0, 10.0)]]], [[0, 4], [4, 0]], "tiny-1m")
    res = solve(inst, ExactSpec(n_agv=2, time_limit_s=30, workers=2, upper_bound=200),
                name="tiny-1m")
    assert res.status == "OPTIMAL"
    assert res.objective == 28.0 and res.dual_bound == 28.0
    assert validate_schedule(inst, res.schedule, 2) == 28.0
    # 去掉回站段 ⟹ 只剩 4 + 20 = 24（钉住"回站段确实是模型的一部分"）
    no_ret = solve(inst, ExactSpec(n_agv=2, time_limit_s=30, workers=2, upper_bound=200,
                                   include_return=False), name="tiny-1m")
    assert no_ret.objective == 24.0


@pytest.mark.unit
def test_tiny_agv_count_binds_and_deadhead_is_charged():
    """两台相距 100 min 的机台、各一件活：**2 台车 12 min，1 台车 16 min**。

    笔算（车数=1，必须避开 100 min 的 M0→M1 空载）：A 投放 0–1（到 M0）→ B 投放
    空载 M0→LU 1–2 + 负载 2–3（到 M1）→ A 加工 1–11 → B 加工 3–13 →
    B 回站（在 M1，空载 0）13–14 → A 回站（空载 LU→M0 1）15–16 ⟹ 16。
    车数=2 时两件并行 ⟹ 12。**车少反而更慢** ⟹ 空载时间确实进了模型。
    """
    _ortools_or_skip()
    from plcsp.baselines.exact_cpsat import ExactSpec, solve, validate_schedule
    inst = _tiny(2, [[[(0, 10.0)]], [[(1, 10.0)]]],
                 [[0, 1, 1], [1, 0, 100], [1, 100, 0]], "tiny-far")
    r2 = solve(inst, ExactSpec(n_agv=2, time_limit_s=30, workers=2, upper_bound=300),
               name="tiny-far")
    r1 = solve(inst, ExactSpec(n_agv=1, time_limit_s=30, workers=2, upper_bound=300),
               name="tiny-far")
    assert (r2.status, r2.objective) == ("OPTIMAL", 12.0)
    assert (r1.status, r1.objective) == ("OPTIMAL", 16.0)
    assert validate_schedule(inst, r2.schedule, 2) == 12.0
    assert validate_schedule(inst, r1.schedule, 1) == 16.0


# ══════════════════ 3. 真实实例：可行 + 下界 ══════════════════

@pytest.mark.unit
def test_real_instance_optimum_is_a_valid_lower_bound_than_the_rule_baseline():
    """MK01 真实数据（前 3 作业 × 机台 0–2）：模型**证明最优**，且最优值 ≤ 规则基线。

    ⚠️ 这是"精确解是尺子"的最小可证版本：`J* ≤ J_rule` 必须成立，否则模型不是 DES 的松弛。
    """
    _ortools_or_skip()
    from plcsp.baselines.exact_cpsat import ExactSpec, rule_upper_bound, solve
    from plcsp.baselines.exact_cpsat import validate_schedule
    inst = _mk01_sub(3, 0, 3)
    spec = ExactSpec(n_agv=2, time_limit_s=60, workers=4)
    rule = rule_upper_bound(inst, spec)
    assert not rule["horizon_hit"], "规则基线掐表 ⟹ 它的 makespan 不能当上界"
    res = solve(inst, spec, name="mk01-sub3", rule_bound=False)
    assert res.status == "OPTIMAL", f"该子实例应能证明最优，实得 {res.status}"
    assert res.objective <= rule["makespan"] + 1e-9, (
        f"模型最优 {res.objective} > 规则基线 {rule['makespan']} ⟹ 模型不是 DES 的松弛")
    assert abs(res.dual_bound - res.objective) < 1e-9        # 证最优 ⟹ 界 = 解
    assert validate_schedule(inst, res.schedule, 2) == res.objective


@pytest.mark.unit
def test_full_mk01_lower_bound_never_exceeds_the_rule_baseline():
    """满规模 MK01：**对偶界**（认证的下界）≤ 规则基线 makespan。

    ⚠️ 本测试**不要求**求出可行解——MK01 满规模可能 3600 s 也证不出最优（见模块 docstring）。
    但**对偶界一定拿得到**，而"下界 ≤ 规则基线"正是认证间隙成立的那一半。
    """
    _ortools_or_skip()
    from plcsp.baselines.exact_cpsat import ExactSpec, solve
    from plcsp.env.mkt import load_mkt
    inst = load_mkt("mk01").base
    spec = ExactSpec(n_agv=1, time_limit_s=20, workers=4)
    res = solve(inst, spec, name="mk01", rule_bound=True)
    assert res.rule_makespan is not None and not res.rule_horizon_hit
    assert res.dual_bound is not None, "对偶界在 UNKNOWN 下也必须拿得到"
    assert res.dual_bound <= res.rule_makespan + 1e-9
    if res.objective is not None:                            # 若拿到可行解，它也是模型的合法解
        assert res.dual_bound <= res.objective + 1e-9


@pytest.mark.unit
def test_constructor_solution_is_accepted_by_the_model():
    """**构造解必须能被模型逐值固定**（并判 OPTIMAL）——钉住 bug#16 那一类。

    构造解（`greedy_schedule`）走的是 DES 语义；模型是 DES 的**松弛** ⟹ 它必须**收得下**这份解。
    实测教训：若 (28)(29) 不挂存在性，不存在的运输任务会被强加 `ms ≥ E_prev`，构造解与模型
    就对不上——warm start 被整支否掉（只在满规模上暴露，小实例看不出来）。
    """
    _ortools_or_skip()
    from plcsp.baselines.exact_cpsat import ExactSpec, build_model, greedy_schedule
    from plcsp.baselines.exact_cpsat import _ortools
    cp_model = _ortools()
    inst = _mk01_sub(3, 0, 3)
    v = 2
    g = greedy_schedule(inst, v)
    spec = ExactSpec(n_agv=v, time_limit_s=60, workers=4, sym_break=False,
                     upper_bound=float(g["makespan"]))
    b = build_model(inst, spec)
    for j, job in enumerate(inst.jobs):
        for i, alts in enumerate(job):
            mm = int(g["ops"][j][i][0])
            for k, y in enumerate(b.op_mach[j][i]):
                b.model.add(y == (1 if int(alts[k][0]) == mm else 0))
            b.model.add(b.op_start[j][i] == int(round(g["ops"][j][i][1])))
            b.model.add(b.op_end[j][i] == int(round(g["ops"][j][i][2])))
    for t in g["tasks"]:
        b.model.add(b.present[t["idx"]] == int(t["present"]))
        b.model.add(b.mv_start[t["idx"]] == int(round(t["start"])))
        b.model.add(b.mv_end[t["idx"]] == int(round(t["end"])))
        for k in range(v):
            b.model.add(b.serve[k][t["idx"]] ==
                        (1 if (t["present"] == 1 and t["agv"] == k) else 0))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    assert solver.solve(b.model) == cp_model.OPTIMAL
    assert solver.value(b.cmax) == int(round(g["makespan"]))


@pytest.mark.unit
def test_validator_rejects_a_broken_schedule():
    """**负对照**：校验器必须抓得住"空载不足"这类错误——否则它不是校验器，是橡皮图章。"""
    _ortools_or_skip()
    from plcsp.baselines.exact_cpsat import ExactSpec, solve, validate_schedule
    inst = _tiny(2, [[[(0, 10.0)]], [[(1, 10.0)]]],
                 [[0, 1, 1], [1, 0, 100], [1, 100, 0]], "tiny-far")
    res = solve(inst, ExactSpec(n_agv=1, time_limit_s=30, workers=2, upper_bound=300),
                name="tiny-far")
    sched = res.schedule
    validate_schedule(inst, sched, 1)                        # 原样通过
    bad = {"ops": sched["ops"], "tasks": [dict(t) for t in sched["tasks"]],
           "completes": list(sched["completes"]), "makespan": sched["makespan"]}
    for t in bad["tasks"]:                                   # 把一台车的空载时间抹掉
        if t["agv"] == 0 and t["kind"] == "ret":
            t["start"] -= 90.0
            break
    with pytest.raises(AssertionError, match="空载不足|早于|晚于"):
        validate_schedule(inst, bad, 1)


@pytest.mark.unit
def test_same_machine_transport_variant_stays_a_relaxation():
    """`same_machine_transport=True`（DES 逐字口径）也解得出来，且解 ≥ NCP2 口径的解。

    同机台相邻工序在 NCP2 里不生成运输任务（更松）⟹ NCP2 的最优值 ≤ DES 逐字口径的最优值。
    """
    _ortools_or_skip()
    from plcsp.baselines.exact_cpsat import ExactSpec, solve, validate_schedule
    inst = _tiny(1, [[[(0, 6.0)], [(0, 4.0)]], [[(0, 8.0)]]], [[0, 3], [3, 0]], "tiny-same")
    a = solve(inst, ExactSpec(n_agv=1, time_limit_s=30, workers=2, upper_bound=200),
              name="ncp2")
    b = solve(inst, ExactSpec(n_agv=1, time_limit_s=30, workers=2, upper_bound=200,
                              same_machine_transport=True), name="des")
    assert a.status == "OPTIMAL" and b.status == "OPTIMAL"
    assert a.objective <= b.objective + 1e-9
    assert validate_schedule(inst, b.schedule, 1, same_machine_transport=True) == b.objective
