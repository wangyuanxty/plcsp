"""约束配置框架的单元测试（P1b Task 1）。

开关共 **10 个**（spec §3.3 十约束）。此前 11 个——**⑥ 模糊加工已砍**（2026-10-03，用户指令）。
"""
from __future__ import annotations

import dataclasses

import pytest

from plcsp.env.constraints import ABLATION_GROUPS, ConstraintConfig

ALL_FLAGS = ("congestion", "finite_buffer", "machine_failure",
             "rework", "setup_time", "due_dates",
             "agv_failure", "heterogeneous_fleet", "charging", "maintenance")


@pytest.mark.unit
def test_default_config_has_all_constraints_on():
    cfg = ConstraintConfig()
    assert all(getattr(cfg, f) is True for f in ALL_FLAGS)


@pytest.mark.unit
def test_config_is_frozen():
    cfg = ConstraintConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.rework = False          # frozen dataclass 应拒绝赋值


@pytest.mark.unit
def test_ablation_groups_are_five_and_well_formed():
    assert set(ABLATION_GROUPS) == {"Full", "-物流", "-生产", "-信息", "None"}
    assert all(getattr(ABLATION_GROUPS["Full"], f) for f in ALL_FLAGS)
    assert not any(getattr(ABLATION_GROUPS["None"], f) for f in ALL_FLAGS)


@pytest.mark.unit
def test_ablation_groups_partition_the_flags():
    """三组减法必须互不重叠、且并集 = 除 Full 外的全部约束（否则消融表有洞）。"""
    on = {f for f in ALL_FLAGS if getattr(ABLATION_GROUPS["Full"], f)}
    off_sets = [{f for f in ALL_FLAGS if not getattr(ABLATION_GROUPS[g], f)}
                for g in ("-物流", "-生产", "-信息")]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not (off_sets[i] & off_sets[j]), "两组减法有重叠"
    assert set().union(*off_sets) == on, "三组的并集未覆盖全部约束"


@pytest.mark.unit
def test_with_off_returns_new_object_and_turns_flags_off():
    """`with_off` 是不可变风格：返回新对象、原对象不变。"""
    base = ConstraintConfig()
    off = base.with_off("rework", "setup_time")
    assert off is not base
    assert off.rework is False and off.setup_time is False
    assert base.rework is True and base.setup_time is True, "原对象被就地修改了"
