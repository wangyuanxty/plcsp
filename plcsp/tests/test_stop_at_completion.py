# -*- coding: utf-8 -*-
"""停表口径：episode 在**末件到站**时结束，`horizon` 退居真·死锁护栏（2026-10-07 修复）。

**缺陷**：`env.run(until=horizon)` 总把仿真时间推到护栏（6×工时+500）才停。报告档 B
（矩阵口径）实测 MK01 makespan 711 min / horizon 1418 min ⟹ 完工后还空转 ~707 min。
空转期间 `AgvSim._failures` 照抽照记（每车一条独立泊松流）⟹ `agv_fail_events` 按**护栏**
而不是按 episode 累积：n_agv=6 时实测 25 次，真 episode 的期望只有 6×711/485 ≈ 8.8 次。
`fail_events` / `rework_events` / `pm_events` 只在加工中触发，末件回站前全部结束 ⟹ 不受影响。

**修法**：末件到站（`LuStation.run` 记 `completes` 处）成功 `done`，停表条件 =
`done | env.timeout(horizon)`。三条判据（对应修复的三条验收）：

1. **停表在 makespan**：`stop_time == makespan < horizon`，`run` 与 `run_gated` 两条入口都成立；
2. **真死锁仍掐表**：恒不充 ⟹ 车队永久不可用 ⟹ `stop_time == horizon` 且 `horizon_hit=True`
   （奖励的完成度守卫继续拿到它要的旗标与上界）；
3. **计数按 episode 累积**：高频故障档下 `agv_fail_events` 与本 episode 的期望同量级
   （旧口径的读数落在**护栏**期望那一侧，本判据在旧代码上必红）。
"""
from __future__ import annotations

import pytest

from plcsp.algo.setup import build_layout_and_dm
from plcsp.env.des import CHARGE_CAND_SKIP, SimConfig, SimWorld
from plcsp.env.instances import load_mk
from plcsp.env.layout import AgvSpec
from plcsp.env.reward import objective_vector

# ⑪ 耗尽验证档的小电池（同 `test_reward._small_battery_world`：0.15 kWh ⟹ 部分完工而非全丢）。
SMALL_BATTERY_KWH = 0.15


# ────────────────────────── 1. 停表在 makespan ──────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("entry", ["run", "run_gated"])
def test_the_run_stops_at_the_makespan_not_at_the_guard(entry):
    """两条入口都必须停在末件到站时刻——`stop_time` 是"仿真跑到哪儿"的可观测读数。

    ⚠️ `horizon` 仍是**护栏值**（工序最短候选之和 ×6 + 500），不是停表时刻——本判据同时
    钉住这一点：完工时 `stop_time < horizon`，护栏没被取到。
    """
    inst = load_mk("mk01")
    cfg = SimConfig()
    lay, dm = build_layout_and_dm(inst, cfg)
    w = SimWorld(inst, lay, dm, cfg)
    met = w.run(seed_chain=1) if entry == "run" else w.run_gated(seed_chain=1)
    assert met["jobs_done"] == inst.n_jobs and met["horizon_hit"] is False, \
        f"{entry} 没跑完——本判据的前提不成立"
    assert met["stop_time"] == met["makespan"], \
        (f"{entry} 的停表时刻 {met['stop_time']} ≠ makespan {met['makespan']}——"
         "停表条件不是末件到站")
    assert met["stop_time"] < met["horizon"], \
        (f"{entry} 停表在护栏 {met['horizon']}——完工后仍在空转，计数会继续虚增")


# ────────────────────────── 2. 真死锁仍掐表 ──────────────────────────

@pytest.mark.unit
def test_a_genuinely_stuck_run_still_hits_the_guard():
    """恒不充 ⟹ 车跑干后永久不可用（⑪ 的既定语义）⟹ 真·跑不完，护栏必须兜住。

    停表修复**只**把"完工"从停表条件里换掉，护栏那一支不动：`stop_time == horizon`、
    `horizon_hit=True`，奖励的完成度守卫照旧拿到 `horizon + 1` 的上界三元组。
    """
    inst = load_mk("mk01")
    cfg = SimConfig(battery_low=0.0)           # 规则配置到 0 才补电（同 §33.4 口径）
    lay, dm = build_layout_and_dm(inst, cfg)
    for i, a in enumerate(lay.agvs):           # 小电池验证档
        lay.agvs[i] = AgvSpec(id=a.id, speed_factor=a.speed_factor,
                              capacity=a.capacity, battery_kwh=SMALL_BATTERY_KWH)
    pick_first = lambda snap, job, frm, to, oi, cand: cand[0]        # noqa: E731
    met = SimWorld(inst, lay, dm, cfg).run_gated(
        seed_chain=3, policy_l=pick_first,
        policy_c=lambda snap, aid, cands: CHARGE_CAND_SKIP)
    assert met["horizon_hit"] is True, "恒不充竟然跑完了——本判据的前提不成立"
    assert 0 < met["jobs_done"] < inst.n_jobs, "部分完工的形态变了（本判据要的是掐表而非全丢）"
    assert met["stop_time"] == met["horizon"], \
        "掐表运行没停在护栏上——停表条件写反了？"
    # 完成度守卫仍可用：上界的第一个分量 = horizon + 1（`reward._incomplete_objectives` 的契约）
    assert objective_vector(met)[0] == met["horizon"] + 1.0, \
        "掐表运行的奖励上界不再以 horizon 为基准——守卫的前提被停表修复动了"


# ────────────────────────── 3. 计数按 episode 累积 ──────────────────────────

@pytest.mark.unit
def test_the_agv_failure_count_is_no_longer_inflated_by_the_guard():
    """高频故障档（mtbf=12 ≪ episode）：计数只与本 episode 的停表时刻相称。

    期望量级 = `n_agv × stop_time ÷ (mtbf + mttr)`（每车一条"抽故障 → 停机 mttr"的更新流）。
    护栏口径的期望是它的 ~12 倍——**旧代码的实测值落在护栏那一侧**（本配置实测 277，
    bound/FIFO 两条路径同值；修复后 30 / 28），故本条同时是"修复真的生效"的判据，
    而不只是"数字变小了"。
    """
    inst = load_mk("mk01")
    cfg = SimConfig(agv_mtbf=12.0, agv_mttr=5.0)
    lay, dm = build_layout_and_dm(inst, cfg)
    met = SimWorld(inst, lay, dm, cfg).run(seed_chain=0)
    assert met["jobs_done"] == inst.n_jobs and met["horizon_hit"] is False, "本判据的前提不成立"
    tau = cfg.agv_mtbf + cfg.agv_mttr
    ep = met["n_agv"] * met["stop_time"] / tau        # 按 episode 的期望
    guard = met["n_agv"] * met["horizon"] / tau       # 按护栏的期望（旧口径落在这一侧）
    assert 0 < met["agv_fail_events"] <= 4.0 * (ep + 1.0), \
        (f"agv_fail_events={met['agv_fail_events']} 与 episode 期望 {ep:.1f} 不符——"
         "计数仍在按别的时间窗累积")
    assert met["agv_fail_events"] < 0.5 * guard, \
        (f"agv_fail_events={met['agv_fail_events']} 仍在护栏期望 {guard:.1f} 那一侧——"
         "空转尾巴又回来了")
