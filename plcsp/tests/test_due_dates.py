"""TF/RDD 交期口径的测试（⑧ 重设计 Task 1）。"""
from __future__ import annotations

import numpy as np
import pytest

from plcsp.env.instances import Instance, gen_random, load_mk


@pytest.mark.unit
def test_total_work_content_is_min_over_alternatives():
    """W_j = 该作业各工序**最短候选工时**之和（纯实例数据，与任何调度无关）。"""
    from plcsp.env.due_dates import total_work_content

    inst = Instance(n_jobs=1, n_machines=2, jobs=[[[(0, 5.0), (1, 9.0)], [(1, 3.0)]]])
    assert total_work_content(inst) == [8.0]


@pytest.mark.unit
def test_workload_lower_bound_is_a_lower_bound():
    """LB 是 makespan 的下界：不得大于参考调度（否则口径自相矛盾）。"""
    from plcsp.env.des import SimConfig, reference_makespan
    from plcsp.env.due_dates import workload_lower_bound

    for name in ("mk01", "mk07", "mk10"):
        inst = load_mk(name)
        assert workload_lower_bound(inst) <= reference_makespan(inst, SimConfig())


@pytest.mark.unit
def test_due_dates_are_per_job_not_common():
    """逐作业交期：不同作业的 d_j **必须不同**（共同交期是旧口径的退化形式）。"""
    from plcsp.env.due_dates import tf_rdd_due_dates

    inst = load_mk("mk01")
    d = tf_rdd_due_dates(inst, tau=2.5, due_range=0.8)
    assert len(set(round(v, 6) for v in d.values())) > 1, "还是共同交期"


@pytest.mark.unit
def test_due_dates_are_exogenous():
    """⚠️ Review Focus 的核心：**交期不得随我们的调度变**。

    旧口径 `d_j = τ·M_ref` 锚在自己的参考调度上——换车队规模就换 M_ref（实测 mk01：
    n_agv=3 → 103.42，n_agv=1 → 109.95），交期跟着动。新口径只读实例数据。
    """
    from plcsp.env.due_dates import due_dates_for

    inst = load_mk("mk01")
    a = due_dates_for(inst)
    b = due_dates_for(inst)
    assert a == b, "同实例两次调用不同值"
    # 与 cfg 无关——签名里根本没有 cfg
    import inspect
    assert "cfg" not in inspect.signature(due_dates_for).parameters
