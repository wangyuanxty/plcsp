"""消融链路单种子打通的测试（P4-A Task 4）。

⚠️ 本测试**只跑 2 组**（不是 5 组）以保住分钟级墙钟；完整的 5 组 × 3 实例由脚本承担。
⚠️ 用**原始 MK**（非 MKT）——用户裁定：原始 MK 跑我们自己的指标与消融，MKT 用于与已发表数字比较。
"""
from __future__ import annotations

import pytest

from plcsp.env.constraints import ABLATION_GROUPS


@pytest.mark.unit
def test_ablation_groups_are_five():
    """spec §6.2 定的是 5 组（Full / −物流 / −生产 / −信息 / None）。"""
    assert set(ABLATION_GROUPS) == {"Full", "-物流", "-生产", "-信息", "None"}


@pytest.mark.unit
def test_two_groups_run_and_differ():
    """⚠️ Review Focus：两组必须**跑得出且不同**——若相同，说明 constraints 没接进训练。

    （`§19.7` 的 F1 已把 constraints 接进 `joint_chain_step`；本测试是端到端的再确认。）
    """
    from plcsp.env.instances import load_mk
    from plcsp.m15_ablation_smoke import run_group

    inst = load_mk("mk01")
    full = run_group(inst, ABLATION_GROUPS["Full"], seed=0, steps=3, G=2)
    none = run_group(inst, ABLATION_GROUPS["None"], seed=0, steps=3, G=2)
    assert full["jobs_done"] == inst.n_jobs and none["jobs_done"] == inst.n_jobs
    assert full["makespan"] != none["makespan"], "两组跑出同一 makespan——constraints 没生效"


@pytest.mark.unit
def test_train_and_eval_seed_streams_do_not_overlap():
    """⚠️ 评估流必须与训练流**不相交**——这是本项目栽过两次的坑（评审 F1 / I-2）。

    训练第 s 步的链种子 = `(seed*1000+s)*1000 + g`（`group_rel.SEED_STRIDE=1000`），
    评估若直接用 `seed*1000+steps`，在 `steps` 取到 1000 的整数倍附近就**正好撞进训练块**，
    而症状只是"评估数字略好"——不会报错。故评估流另起一个显式基址，且有守卫。
    """
    from plcsp.env.instances import load_mk
    from plcsp.m15_ablation_smoke import (
        EVAL_SEED_BASE, _eval_seed, _train_step_seed, assert_streams_disjoint)
    from plcsp.algo.group_rel import SEED_STRIDE

    # 训练流上界 < 评估基址（正常参数下）
    assert_streams_disjoint(seed=0, steps=30, G=8)
    assert_streams_disjoint(seed=7, steps=100, G=8)

    # 用真的跑一遍会用到的那两个函数交叉验证：训练块 ≠ 评估种子
    train_seeds = {_train_step_seed(0, s) * SEED_STRIDE + g
                   for s in range(30) for g in range(8)}
    assert _eval_seed(0, 30) not in train_seeds, "评估种子落进了训练块"

    # 越界必须**显式报错**，不得静默跑（m13_train_a 的同款守卫）
    with pytest.raises(ValueError):
        assert_streams_disjoint(seed=0, steps=EVAL_SEED_BASE, G=8)
