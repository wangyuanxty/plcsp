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
def test_same_seed_is_reproducible_across_calls():
    """⚠️ **同 seed 必须逐位可复现**——网络初始化也是种子流之一，漏了它整张消融表都不可复现。

    本项目中招记录：`m13_train_a.py` 的 `main()` 入口就有 `torch.manual_seed(args.seed)`
    （其 docstring 明列三条流：网络初始化 / 仿真扰动 / 动作采样）。`m15` 起初只锁了后两条，
    `_policy_for()` 每次从**全局未播种**的 torch RNG 取初始化 ⇒ 同参数两次调用读数不同，
    实测（`Full`、mk01、seed=0、steps=0、G=1，三个独立进程）：**129.17 / 234.00 / 89.28**。

    ⚠️ 后果不止"不可复现"：跨组的 Cmax 差异会被**初始化噪声**污染——而 init 噪声的幅度
    （上例 89↔234）远大于组间差异，于是"五组读数互不相同"根本不能推出"constraints 接通了"。
    """
    from plcsp.env.instances import load_mk
    from plcsp.m15_ablation_smoke import run_group

    inst = load_mk("mk01")
    a = run_group(inst, ABLATION_GROUPS["Full"], seed=0, steps=2, G=2)
    b = run_group(inst, ABLATION_GROUPS["Full"], seed=0, steps=2, G=2)
    assert a == b, f"同 seed 两次调用读数不同（{a} vs {b}）——网络初始化没播种"


@pytest.mark.unit
def test_two_groups_run_and_differ():
    """端到端冒烟：两组都能跑完，且（同初始化下）读数不同。

    ⚠️ **这个断言单独证明不了"constraints 接进了仿真"**——`ctx` 也随 constraints 变，
    故即便 `constraints` 没透传给 `SimWorld`，特征侧仍会让两组跑出不同读数。
    "约束关掉真的改变仿真"由 `test_constraints*.py` 直接钉（比较同一 `op_choices` 下的
    仿真读数）。此处只保证**链路没断**，别再把它当成约束生效的证据。
    （这正是本项目"测试名声称的属性 > 实际验证的内容"那一类，故在此写明边界。）
    """
    from plcsp.env.instances import load_mk
    from plcsp.m15_ablation_smoke import run_group

    inst = load_mk("mk01")
    full = run_group(inst, ABLATION_GROUPS["Full"], seed=0, steps=3, G=2)
    none = run_group(inst, ABLATION_GROUPS["None"], seed=0, steps=3, G=2)
    assert full["jobs_done"] == inst.n_jobs and none["jobs_done"] == inst.n_jobs
    assert full["makespan"] != none["makespan"], "两组跑出同一 makespan——链路断了"


@pytest.mark.unit
def test_train_and_eval_seed_streams_do_not_overlap():
    """⚠️ 评估流必须与训练流**不相交**——这是本项目栽过两次的坑（评审 F1 / I-2）。

    训练第 s 步的链种子 = `(seed*1000+s)*1000 + g`（`group_rel.SEED_STRIDE=1000`），
    评估若直接用 `seed*1000+steps`，在 `steps` 取到 1000 的整数倍附近就**正好撞进训练块**，
    而症状只是"评估数字略好"——不会报错。故评估流另起一个显式基址，且有守卫。
    """
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
