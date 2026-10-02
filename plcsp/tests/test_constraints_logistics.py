"""物流侧约束（⑨ AGV 故障 · ⑩ 异构多载量 · ⑪ 充电）的测试（P1b Task 5）。

## Review Focus #4 先定语义（写代码前必须定死）

⑩ 的"多载量"实现为 **同向拼车**：同一 (取货点, 卸货点) 的多件合成**一趟**。
**索引口径**（P2 的 L 层训练必须知道）：
- `agv_phi[i]` 仍按**运输任务**索引（不是按行程）——`i` 是 transporter 生成序，
  与 `moves`/`trips` 无关。一次行程消费多个 `task_i`。
- 故返回 dict 里 **`deliveries` = 件数**（= 任务数），**`trips` = 行程数**，两者在多载量下不等。
  混淆这两个数会让"多载量减少了行程"的结论反过来。

## 参数（**全部 assumed**，见 spec §9）
车队 速度 ∈ [0.8, 1.2]×基准 / 载量 ∈ {1,2,3} / 电量 ∈ [2,4] kWh；
AGV MTBF 480 min、MTTR 10 min；充电阈值 20%、充到 80%、充电功率 `charge_kw`。
"""
from __future__ import annotations

import pytest

from plcsp.env.constraints import ConstraintConfig
from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import load_mk
from plcsp.env.layout import sample_layout

SEEDS = (0, 1, 2)


@pytest.mark.unit
def test_heterogeneous_fleet_has_varied_specs():
    """⑩ 异构：车队的速度/载量/电量不得全部相同。"""
    lay = sample_layout(6, seed=0, n_agv=4)
    specs = [(a.speed_factor, a.capacity, a.battery_kwh) for a in lay.agvs]
    assert len(lay.agvs) == 4, "车队规模没传进布局——布局的车队与仿真的车队会不一致（静默错误）"
    assert len(set(specs)) > 1, "异构车队却给出全同规格"
    assert all(1 <= a.capacity <= 3 for a in lay.agvs)
    assert all(0.8 <= a.speed_factor <= 1.2 for a in lay.agvs)
    assert all(a.battery_kwh > 0 for a in lay.agvs)


@pytest.mark.unit
def test_layout_fleet_size_follows_cfg():
    """⚠️ 计划点名的静默错误：`sample_layout` 的车队必须随 `cfg.n_agv` 走。"""
    inst = load_mk("mk01")
    r = rollout(inst, seed_chain=1, cfg=SimConfig(n_agv=5))
    assert r["n_agv"] == 5
    assert r["fleet_size"] == 5, "布局的车队规模与仿真不一致"


@pytest.mark.unit
def test_multiload_delivers_more_items_per_trip():
    """⑩ 多载量：件数不变，行程数不得增加；容量 >1 时应真的少跑。"""
    inst = load_mk("mk01")
    tot = {}
    for tag, cap in (("single", 1), ("multi", 3)):
        trips = deliveries = 0
        for s in SEEDS:
            r = rollout(inst, seed_chain=s, cfg=SimConfig(max_agv_capacity=cap),
                        constraints=ConstraintConfig())
            assert not r["horizon_hit"] and r["jobs_done"] == inst.n_jobs
            trips += r["trips"]
            deliveries += r["deliveries"]
        tot[tag] = (trips, deliveries)
    assert tot["multi"][1] == tot["single"][1], "件数应不受载量影响"
    assert tot["multi"][0] <= tot["single"][0], "多载量下行程数反而变多"


@pytest.mark.unit
def test_charging_never_drives_battery_negative():
    """⑪ 充电：电量不得为负；开启且低于阈值时必须真的去过充电桩。"""
    inst = load_mk("mk01")
    lo = 0.0
    events = 0
    for s in SEEDS:
        r = rollout(inst, seed_chain=s, cfg=SimConfig(), constraints=ConstraintConfig())
        lo = min(lo, r["battery_min_kwh"])
        events += r["charge_events"]
    assert lo >= 0.0, f"电量出现负值（{lo}）→ 充电模型有误"
    assert events >= 0


@pytest.mark.unit
def test_switches_off_equals_no_constraint():
    """Review Focus #2：三个开关全关时，不得出现充电/故障事件，且规格退化为同构单载。"""
    inst = load_mk("mk01")
    off = ConstraintConfig(charging=False, agv_failure=False, heterogeneous_fleet=False)
    for s in SEEDS:
        r = rollout(inst, seed_chain=s, cfg=SimConfig(), constraints=off)
        assert r["charge_events"] == 0
        assert r["agv_fail_events"] == 0
        assert not r["horizon_hit"] and r["jobs_done"] == inst.n_jobs


@pytest.mark.unit
def test_agv_failure_off_shifts_no_streams():
    """Review Focus #2/#3：⑨ 关掉时不得消耗随机数（否则机台故障流被移位，消融不干净）。"""
    inst = load_mk("mk01")
    a = rollout(inst, seed_chain=1, cfg=SimConfig(),
                constraints=ConstraintConfig(agv_failure=False))
    b = rollout(inst, seed_chain=1, cfg=SimConfig(),
                constraints=ConstraintConfig(agv_failure=False))
    assert a["makespan"] == b["makespan"] and a["fail_events"] == b["fail_events"]


@pytest.mark.unit
def test_three_logistics_switches_independently_toggleable():
    """Review Focus #3：2³ 种组合都不崩、都跑完。"""
    inst = load_mk("mk01")
    for ch in (False, True):
        for af in (False, True):
            for hf in (False, True):
                r = rollout(inst, seed_chain=1, cfg=SimConfig(),
                            constraints=ConstraintConfig(charging=ch, agv_failure=af,
                                                         heterogeneous_fleet=hf))
                assert r["jobs_done"] == inst.n_jobs and not r["horizon_hit"], \
                    f"组合 charging={ch} agv_failure={af} hetero={hf} 未跑完"


@pytest.mark.unit
def test_agv_phi_still_indexes_tasks_not_trips():
    """Review Focus #4 的核心断言：多载量下一趟送多件，但 `agv_phi` 仍按**任务**索引。

    若 `agv_phi` 误按行程索引，P2 的 L 层训练会**静默错配**（第 i 个决策对应错的车）。
    这里给定与件数等长的 `agv_phi`，必须跑通且每件都送到。
    """
    inst = load_mk("mk01")
    r0 = rollout(inst, seed_chain=1, cfg=SimConfig(max_agv_capacity=3))
    n_tasks = len(r0["task_flow"])
    phi = [i % SimConfig().n_agv for i in range(n_tasks)]      # 按**任务**索引，长度 = 件数
    r = rollout(inst, seed_chain=1, cfg=SimConfig(max_agv_capacity=3), agv_phi=phi)
    assert r["jobs_done"] == inst.n_jobs, "按任务索引的 agv_phi 应能跑通"
    assert r["deliveries"] == n_tasks, "deliveries 必须是**件数**（= 任务数），不是行程数"
    assert r["trips"] <= r["deliveries"], "行程数不得超过件数"
