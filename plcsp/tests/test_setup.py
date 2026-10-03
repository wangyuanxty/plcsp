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
