# -*- coding: utf-8 -*-
"""`algo.setup` 环境构造入口的测试（评审 F4：五份拷贝收敛成一份）。"""
from __future__ import annotations

import pytest

from plcsp.algo.setup import build_setup
from plcsp.env.des import SimConfig, reference_makespan
from plcsp.env.instances import load_mk


@pytest.mark.unit
def test_build_setup_binds_layout_to_cfg():
    """⚠️ F4 根因：`sample_layout` 必须收到 cfg 的**全部**车队/几何参数。

    漏传 `aisle_w` ⇒ `SimConfig(aisle_width=1.0)` 时布局几何仍按 1.5 m 生成，而 `eff_speed`
    按 1.0 降速——**距离来自一个布局、速度按另一个布局**，几何与动力学错配且零报错。
    漏传 `max_agv_capacity` ⇒ 布局车队载量上限与 `SimConfig` 不符（同类静默错配）。
    """
    inst = load_mk("mk01")
    cfg = SimConfig(aisle_width=1.0, n_agv=2, max_agv_capacity=1)
    lay, dm, ctx = build_setup(inst, cfg)
    assert lay.grid.aisle_w == cfg.aisle_width, "布局几何没按 cfg.aisle_width 生成"
    assert len(lay.agvs) == cfg.n_agv == 2
    assert all(a.capacity == 1 for a in lay.agvs), "max_agv_capacity 没传进 sample_layout"
    assert ctx.max_capacity == 1
    assert ctx.m_ref == pytest.approx(reference_makespan(inst, cfg)), \
        "m_ref 必须是真实参考 makespan（不是 100.0 之类的占位）"


@pytest.mark.unit
def test_build_training_setup_binds_constraints_to_ctx():
    """⚠️ R2：`build_training_setup` 必须把 constraints **原样返回**，且 ctx 记的就是那一份。

    旧状：形参存在但 `main` 从不传（钩子邀请"ctx 按 A 组约束建、训练跑全开"的静默错配）。
    返回 constraints 使调用方拿到的就是"ctx 是按哪组约束建的"那一份——不需要凭记忆配对。
    """
    from plcsp.env.constraints import ConstraintConfig
    from plcsp.m13_train_a import build_training_setup

    off = ConstraintConfig().with_off("machine_failure", "due_dates")
    inst, lay, dm, cfg, ctx, _pol, cons = build_training_setup("mk01", constraints=off)
    assert cons == off, "build_training_setup 没把 constraints 原样返回"
    assert ctx.constraints == off, "ctx 记的约束不是建它时用的那一份"


@pytest.mark.unit
def test_main_threads_constraints_to_setup_step_and_eval(monkeypatch):
    """⚠️ R2：`main` 必须把 constraints 一路送进 `step_kwargs` 与评估回调。

    判据 = **三处同源**：`build_training_setup` 返回的那一份、`step_kwargs["constraints"]`、
    评估 `roll_chain` 收到的 `constraints`（评估跑的是训练后的策略，动力学口径必须一致）。
    用 `run_training` 换桩驱动 `main`——只测接线，不真跑训练。
    """
    import sys

    import plcsp.m13_train_a as m13
    from plcsp.env.constraints import ConstraintConfig

    cap: dict = {}
    roll_calls: list = []

    def _fake_run(pol, inst, **kw):
        cap.update(kw)
        return {"step": 0, "best": 0.0, "r_last": 0.0}

    def _fake_roll(*a, **k):
        roll_calls.append(k)
        # 完成度两键：`_make_eval_fn` 自 2026-10-05（T3 批）起一并报它们。
        # ⚠️ 自 2026-10-05（期① 消融批）起评估还读 `energy` 与 `completes`（另两项目标
        # 的逐种子原始值）——替身必须凑齐**全部**被读的键，否则是替身失配、不是被测代码错。
        return [], {"makespan": 1.0, "energy": 0.0, "completes": {},
                    "jobs_done": 1, "horizon_hit": False}

    monkeypatch.setattr(m13, "run_training", _fake_run)
    monkeypatch.setattr(m13, "roll_chain", _fake_roll)
    monkeypatch.setattr(sys, "argv", ["m13_train_a", "--inst", "mk01", "--steps", "1",
                                      "--eval-every", "1", "--eval-seeds", "1"])
    m13.main()

    kw = cap["step_kwargs"]
    assert "constraints" in kw, ("main 没把 constraints 传进 step_kwargs——ctx 与训练的约束"
                                 "会静默错配（P4 调用方踩的正是这个）")
    cons = kw["constraints"]
    assert cons == ConstraintConfig()
    assert cons == kw["ctx"].constraints, "step_kwargs 的 constraints 与 ctx 不同源"
    cap["eval_fn"](None)                              # 评估回调：同约束口径
    assert roll_calls and "constraints" in roll_calls[0], "评估没收到 constraints"
    assert roll_calls[0]["constraints"] == cons, "评估的约束与训练不同源"
