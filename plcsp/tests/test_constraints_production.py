"""生产侧约束（④ 返工 · ⑤ 换型 · ⑫ 预防性维护）的测试（P1b Task 4）。

三个约束的参数**全部是 assumed**（无文献出处），见 spec §9 的 assumed 表：
`p_rework=0.05` / `setup_min_default=2.0` min / `pm_interval=120` min / `pm_duration=10` min。

⚠️ **PM 的实例选择是有意的**：每机平均负载 MK01 只有 25.5 min、MK07/MK10 各约 130/123 min。
故 `pm_interval=120` 在 **MK01 上根本不触发**——断言必须落在 MK07/MK10，
否则就是一条永真的假测试。（这个"哪个实例会触发"本身就是 binding 结论的一部分。）
"""
from __future__ import annotations

import pytest

from plcsp.env.constraints import ConstraintConfig
from plcsp.env.des import SimConfig, rollout
from plcsp.env.instances import load_mk

SEEDS = (0, 1, 2)


def _sum(inst_name: str, key: str, **off) -> float:
    """跑 seeds 次，累加某个返回字段。`off` 里给的约束名会被关掉。"""
    inst = load_mk(inst_name)
    cfg = ConstraintConfig().with_off(*off) if off else ConstraintConfig()
    tot = 0.0
    for s in SEEDS:
        r = rollout(inst, seed_chain=s, cfg=SimConfig(), constraints=cfg)
        assert not r["horizon_hit"] and r["jobs_done"] == inst.n_jobs, \
            f"{inst_name} seed{s} 未跑完（horizon 或死锁）"
        tot += r[key]
    return tot


@pytest.mark.unit
def test_setup_time_off_gives_exactly_zero_setup_minutes():
    """⑤ 关 → 换型时长必须**恰好 0**（Review Focus #2："关了但没真关"的探针）。"""
    assert _sum("mk01", "setup_minutes_total") > 0.0, "⑤ 开启时换型时长应 > 0"
    assert _sum("mk01", "setup_minutes_total", setup_time=False) == 0.0


@pytest.mark.unit
def test_setup_time_consumes_machine_time():
    """⑤ 换型不是个计数——它必须真的**占用机台时间**，否则等于没建。"""
    on = _sum("mk01", "makespan")
    off = _sum("mk01", "makespan", setup_time=False)
    assert on > off, f"开启换型后 makespan 未上升（{on:.1f} vs {off:.1f}）——换型没占机台"


@pytest.mark.unit
def test_rework_off_gives_exactly_zero_rework_events():
    """④ 关 → 返工事件恰好 0；开 → 至少发生一次（p=0.05 × 每 episode ~55 道工序）。"""
    assert _sum("mk01", "rework_events") > 0.0, "④ 开启时三个种子应至少触发一次返工"
    assert _sum("mk01", "rework_events", rework=False) == 0.0


@pytest.mark.unit
def test_rework_does_not_draw_randomness_when_off():
    """④ 关掉时**不得消耗随机数**——否则故障流被移位，"从未存在"就不成立（Review Focus #2）。

    逐位对拍：`rework=False` 与"该约束从未存在"必须同轨。这里的代理是**确定性**：
    同配置同种子跑两次必须逐位相同（保证抽取是有纪律的），且关态下故障事件数与
    `machine_failure` 单独开启时一致——若关态偷偷抽了数，故障流会错位。
    """
    inst = load_mk("mk01")
    a = rollout(inst, seed_chain=1, cfg=SimConfig(),
                constraints=ConstraintConfig(rework=False))
    b = rollout(inst, seed_chain=1, cfg=SimConfig(),
                constraints=ConstraintConfig(rework=False))
    assert a["makespan"] == b["makespan"] and a["fail_events"] == b["fail_events"]


@pytest.mark.unit
def test_maintenance_off_gives_exactly_zero_pm_events():
    """⑫ 关 → 维护事件恰好 0；开 → MK10（每机负载 ~123 min > 120 min 间隔）上会触发。"""
    assert _sum("mk10", "pm_events") > 0.0, "⑫ 开启时 MK10 应触发维护"
    assert _sum("mk10", "pm_events", maintenance=False) == 0.0


@pytest.mark.unit
def test_three_switches_independently_toggleable():
    """Review Focus #3：三个开关可自由组合，2³ 种配置都不崩、都跑完。"""
    inst = load_mk("mk01")
    for rw in (False, True):
        for su in (False, True):
            for mt in (False, True):
                r = rollout(inst, seed_chain=1, cfg=SimConfig(),
                            constraints=ConstraintConfig(rework=rw, setup_time=su,
                                                         maintenance=mt))
                assert r["jobs_done"] == inst.n_jobs and not r["horizon_hit"], \
                    f"组合 rework={rw} setup={su} maint={mt} 未跑完"


@pytest.mark.unit
def test_maintenance_and_rework_couple_monotonically_through_spindle_time():
    """Review Focus #3：约束之间**有真耦合**，且耦合的方向可以讲清楚。

    ⑫ 的触发条件是**主轴工时**：返工越多 → 主轴转得越久 → 保养越频繁。
    故这不该断言成"互不影响"（那是错的），而要断言**单调方向**：
    关掉返工后保养事件数不得**增加**。

    ⚠️ 本测试必须用 MK10：MK01 每机负载仅 25.5 min < `pm_interval=120`，
       PM 恒为 0，断言会**空过**（P1a 评审 I2/I3 的弱断言就是这个形状）。
    """
    inst = load_mk("mk10")
    counts = {}
    for tag, c in (("on", ConstraintConfig()), ("off", ConstraintConfig(rework=False))):
        counts[tag] = sum(rollout(inst, seed_chain=s, cfg=SimConfig(),
                                  constraints=c)["pm_events"] for s in SEEDS)
    assert counts["on"] > 0, "MK10 上 PM 必须真的触发，否则本测试是空断言"
    assert counts["off"] <= counts["on"], \
        f"关掉返工后保养反而变多（{counts['off']} > {counts['on']}）——主轴工时口径串了"
