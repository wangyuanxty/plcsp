"""⑧ 接线的测试（⑧ 重设计 Task 3）。"""
from __future__ import annotations

import pytest

from plcsp.env.instances import load_mk


@pytest.mark.unit
def test_due_dates_do_not_depend_on_fleet_size():
    """⚠️ Review Focus 的核心：**换车队规模不得改变交期**。

    旧口径下 mk01 的 M_ref 随 n_agv 变（n_agv=3 → 103.42，n_agv=1 → 109.95），交期跟着变——
    等于"交期随我们的配置漂"。新口径只读实例数据。
    """
    from plcsp.env.des import SimConfig
    from plcsp.env.due_dates import due_dates_for

    inst = load_mk("mk01")
    assert due_dates_for(inst) == due_dates_for(inst, tau=None, due_range=None)
    # 通过 SimWorld 的 _due_map 再验一次（真正被仿真用的那条路）
    from plcsp.env.des import SimWorld
    from plcsp.algo.setup import build_setup
    d1 = SimWorld(inst, *build_setup(inst, SimConfig(n_agv=1))[:2],
                  SimConfig(n_agv=1))._due_map()
    d3 = SimWorld(inst, *build_setup(inst, SimConfig(n_agv=3))[:2],
                  SimConfig(n_agv=3))._due_map()
    assert d1 == d3, "交期随车队规模变了——锚又回到内生量上了"


@pytest.mark.unit
def test_makespan_is_unchanged_by_due_date_caliber():
    """⚠️ Review Focus #1：⑧ 是 metric，**不得改变任何事件时序**。"""
    from plcsp.env.des import SimConfig, rollout

    inst = load_mk("mk01")
    a = rollout(inst, seed_chain=1, cfg=SimConfig())
    b = rollout(inst, seed_chain=1, cfg=SimConfig(tau=2.0, due_range=1.0))
    c = rollout(inst, seed_chain=1, cfg=SimConfig(tau=9.0, due_range=0.0))
    assert a["makespan"] == b["makespan"] == c["makespan"]
    assert a["tardy_twt"] != c["tardy_twt"], "换 τ 读数没变——τ 根本没接上"


@pytest.mark.unit
def test_reference_objectives_twt_uses_the_same_due_dates():
    """⚠️ Review Focus #5：`ReferenceObjectives` 的 TWT 必须与仿真**同源**。

    ⚠️ **改写说明**（原测试不可能通过）：参考运行自身的 `tardy_twt` **结构性为 0**——
    它跑在自己的 `_MREF_BUSY` 短路里（`_due_map()` 返回 `{}`），这正是 `reward.of` 要事后
    重算 TWT 的原因（见该处 docstring）。故判据不能取 `reference_run()` 的 `tardy_twt`，
    而应与**同一次运行**（同 seed_layout/seed_chain/cfg，`reference_run` 就是这么跑的）的
    仿真读数对拍：那条路径上 ⑧ 是开着的，`tardy_twt` 由 `run()` 用同一个交期算出。
    """
    from plcsp.env.des import SimConfig, reference_run, rollout
    from plcsp.env.reward import ReferenceObjectives

    inst = load_mk("mk01")
    cfg = SimConfig()
    ref = ReferenceObjectives.of(inst, cfg)
    r = rollout(inst, seed_chain=0, cfg=cfg)          # 与参考运行同一次（默认 seed_layout=0）
    assert r["makespan"] == reference_run(inst, cfg)["makespan"], "不是同一次运行"
    assert ref.twt == pytest.approx(r["tardy_twt"]), "ref 的 TWT 与仿真不同源"


@pytest.mark.unit
def test_due_margin_is_no_longer_redundant():
    """⚠️ Review Focus #4：**有跨度**（R ≥ 0.2）的交期下 `due_margin` 维**不再**是冗余维。

    旧口径（共同交期 `d_j = τ·M_ref`）下 `due_margin = τ − time_progress`，对所有作业 token
    同值 ⟹ 冗余维。新口径 `d_j = LB·τ·(1+R(2ρ_j−1))` 逐作业取不同值 ⟹ 有区分度。
    ⚠️ **改写说明**：标定网格的 R 下限一度是 0，mk01/mk04 因此落到 **R=0**（共同交期）——
    本测试当时只能退到 mk08。R 下限改为 0.20 后**全部 10 个实例的 d_j 互不相同**，
    故判据回到计划点名的 **mk01**（= P4 的训练实例），并保留 mk08（跨度最宽）作第二判据。
    """
    from plcsp.env.due_dates import due_dates_for

    for name in ("mk01", "mk08"):
        d = due_dates_for(load_mk(name))
        assert len(set(round(v, 6) for v in d.values())) > 1, \
            f"{name} 仍给出共同交期——逐作业口径没接通"
