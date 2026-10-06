# -*- coding: utf-8 -*-
"""③ 机器故障率随役龄上升（`SimConfig.machine_age_failure`，默认关）。

设计（`docs/mechanism-designs.md` 的「③ 机器故障率随役龄上升」节，开工前定死）：

    λ(a) = base · β · (a / η)^(β−1)      a = `pm_clock`（主轴工时累计、保养归零）
                                         η = `pm_interval`，β = `machine_age_beta`（assumed）

- β = 1 ⟹ λ ≡ base（与常数故障率同式）；β > 1 ⟹ 严格递增；[0, η] 上均值 = base。
- 自变量**只能是 `pm_clock`**（当前状态里已有的量）——不许引入历史量（用户裁定）。
- ⚠️ **改动力学**：故障的"何时"变了；保养（⑫）因此多一重收益（把故障率打回 0）。
- 只对 ③ 成立，**不推广**到 ④⑨。

摘要值是**改造前**（2026-10-05，HEAD=9fc2487）在 MK01 上捕获的（三种配置）。
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from plcsp.algo.group_rel import roll_chain
from plcsp.algo.policy import PolicyNet
from plcsp.algo.setup import build_setup
from plcsp.env.constraints import ConstraintConfig
from plcsp.env.des import MachineSim, OpLite, SimConfig, SimWorld, machine_fail_rate
from plcsp.env.instances import load_mk
from plcsp.env.layout import MachinePad
from plcsp.nn.encoder import LayoutEncoder

# 改造前捕获的三条摘要（failover 与全部既有开关默认关；高故障配置把布局机台的 fail_rate
# 抬到 0.05/min，让故障真的发生）。
# ⚠️ 2026-10-06 重捕：机床待机功率由 P^u 改为 Table 9 的 Standby Power（见 energy.py 模块 docstring）。
# 已实证决策 / makespan / travel / deliveries 逐位不变，只有 met[energy] 变。
DIGESTS = {
    "default": "69528db4a63bdc84b6f02e611bcf2a6f7b09f720c1876a93b6898199caa09406",
    "pm20": "0b4a8cb1dd30ecd261148fdb8a6d8a0ebd3d96e510e23c305cbeba317991de59",
    "hotfail": "6b1e985ead6a21e256156239efbbd364d4b83a02e8b3d05171be80b6001b8886",
}


def _digest(met: dict) -> str:
    h = hashlib.sha256()
    for k in ("makespan", "energy", "deliveries", "travel_time_total", "moves", "requeue",
              "trips", "agv_fail_events", "jobs_done", "horizon_hit", "fail_events",
              "pm_events", "rework_events", "setup_minutes_total"):
        h.update(f"{k}={met.get(k)!r};".encode())
    h.update(repr(sorted(met.get("completes", {}).items())).encode())
    return h.hexdigest()


def _run(cfg, hot_beta=None, hot_fail=False):
    inst = load_mk("mk01")
    lay, dm, ctx = build_setup(inst, cfg)
    if hot_fail:
        for m in lay.machines:
            m.fail_rate = 0.05
    import torch
    torch.manual_seed(20261005)
    pol = PolicyNet(enc=LayoutEncoder())
    dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    return dec, met


# ────────────────────────── 1. 关态逐位不变 ──────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("label,cfg,hot", [
    ("default", SimConfig(), False),
    ("pm20", SimConfig(pm_interval=20.0), False),
])
def test_age_failure_off_is_bit_identical_to_baseline(label, cfg, hot):
    """判据 1（低故障配置）：关态时链路与读数逐位等于改造前。"""
    _dec, met = _run(cfg, hot_fail=hot)
    assert _digest(met) == DIGESTS[label], f"{label} 配置链路变了——既有读数不再成立"


@pytest.mark.unit
def test_age_failure_off_with_failures_is_bit_identical():
    """判据 1（高故障配置）：`fail_rate=0.05` 让故障真的发生（fail_events=12），仍逐位不变。"""
    cfg = SimConfig(pm_interval=20.0)
    _dec, met = _run(cfg, hot_fail=True)
    assert met["fail_events"] == 12, "前提：本配置必须有故障，否则判据盖不住故障路径"
    assert _digest(met) == DIGESTS["hotfail"], "高故障配置链路变了——既有读数不再成立"


# ────────────────────────── 2. 曲线的性质 ──────────────────────────

@pytest.mark.unit
def test_weibull_rate_shape_and_mean_matching():
    """判据 2/4（纯函数）：β=1 恒为 base；β>1 严格递增且 [0,η] 均值 = base；β<1 报错。"""
    base, eta = 0.05, 20.0
    assert machine_fail_rate(base, 7.0, 1.0, eta) == pytest.approx(base), \
        "β=1 必须退化回常数率（与关态同式的自检点）"
    ages = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0]
    vals = [machine_fail_rate(base, a, 2.0, eta) for a in ages]
    assert vals[0] == pytest.approx(0.0), "β>1 的 Weibull 风险在役龄 0 处为 0"
    assert all(b > a for a, b in zip(vals, vals[1:])), f"风险必须严格递增：{vals}"
    mean = float(np.mean([machine_fail_rate(base, a, 2.0, eta)
                          for a in np.linspace(0.0, eta, 20001)]))
    assert mean == pytest.approx(base, rel=1e-3), \
        f"[0,η] 上的均值应 = base（保养周期均值匹配）：{mean} vs {base}"
    # β 的敏感性方向（均值匹配下的形状效应）：η 之前 β 越大率越低，η 之后 β 越大率越高
    assert machine_fail_rate(base, eta / 2, 3.0, eta) < machine_fail_rate(base, eta / 2, 2.0, eta)
    assert machine_fail_rate(base, 2 * eta, 3.0, eta) > machine_fail_rate(base, 2 * eta, 2.0, eta)
    with pytest.raises(ValueError, match="beta"):
        machine_fail_rate(base, 10.0, 0.5, eta)
    with pytest.raises(ValueError, match="eta"):
        machine_fail_rate(base, 10.0, 2.0, 0.0)


# ────────────────────────── 3. 役龄↑ ⟹ 故障率↑；保养 ⟹ 回落 ──────────────────────────

class _RecordingRng:
    """确定性伪随机流：`exponential(scale)` 直接把 scale 当抽样值返回并记录。

    `scale = 1/λ`——所以记录下来的序列就是"故障率倒数"的**逐次读数**：
    役龄越长 scale 越小（率越高）；保养归零后 scale 跳回上限（λ=0 ⟹ max(λ,1e-9)）。
    `random()` 恒 1.0（④ 返工不触发）。
    """

    def __init__(self):
        self.scales: list[tuple[float, float]] = []      # (当时的 pm_clock, scale)
        self.machine = None

    def exponential(self, scale: float) -> float:
        age = float(self.machine.pm_clock) if self.machine is not None else -1.0
        self.scales.append((age, float(scale)))
        return float(scale)

    def random(self) -> float:
        return 1.0


def _drive(cfg, op_times, fail_rate=0.05):
    """直接驱动一台 `MachineSim._process`（不经 in_q/out_q）——返回 (机器, 记录流, stats)。"""
    import simpy
    env = simpy.Environment()
    pad = MachinePad(id=0, x=0.0, y=0.0, w=1.0, h=1.0, dock_node=0, fail_rate=fail_rate,
                     in_cap=99, out_cap=99)
    rng = _RecordingRng()
    stats = {"fail_events": 0, "setup_min": [0.0], "process_time": 0.0, "proc_min": [0.0],
             "pm_events": 0}
    cons = ConstraintConfig(finite_buffer=False)
    track = __import__("plcsp.env.des", fromlist=["SimTrack"]).SimTrack(1, 1, 1)
    m = MachineSim(env, pad, rng, cfg, stats, {}, simpy.Store(env), cons, track, pm=None)
    rng.machine = m
    for k, t in enumerate(op_times):
        env.process(m._process(k % 2, OpLite(t)))       # 同作业/异作业交替不影响本判据
    env.run()
    return m, rng.scales, stats


@pytest.mark.unit
def test_failure_rate_rises_with_age_within_a_cycle():
    """判据 2（受控积分）：同一保养周期内，役龄越大 ⟹ 指数尺度 `1/λ` 越小（故障越频）。

    用伪 rng 把"率"逐次读出来：λ(0) = 0 ⟹ 第一次尺度 = 1/1e-9（上限，即"刚保养完不易坏"）；
    之后随 `pm_clock` 累积严格下降。
    """
    cfg = SimConfig(pm_interval=10_000.0, machine_age_failure=True, machine_age_beta=2.0,
                    repair_time=0.0)
    _m, scales, _stats = _drive(cfg, [100.0, 100.0, 100.0])
    assert scales, "没有抽到任何故障间隔——判据失去意义"
    assert scales[0][1] >= 1e8, f"役龄 0 处 λ=0 ⟹ 尺度应为上限：{scales[0]}"
    # 只比较**役龄不同**的抽样：同一年龄段内因修复重抽，尺度会重复（同 λ）
    by_age: dict[float, float] = {}
    for age, scale in scales:
        by_age[age] = scale
    ages = sorted(by_age)
    assert len(ages) >= 2, f"役龄没有增长（{ages}）——判据失去意义"
    vals = [by_age[a] for a in ages]
    assert all(b < a for a, b in zip(vals, vals[1:])), \
        f"役龄越大尺度应越小（率越大）：{list(zip(ages, vals))}"


@pytest.mark.unit
def test_maintenance_resets_the_failure_rate():
    """判据 3：保养（规则配置到点强制）把 `pm_clock` 归零 ⟹ 下一次抽样的尺度跳回上限。

    `pm_interval=150`、每道工序 100 主轴分钟：第 2 道工序结束时逾期 ⟹ 强制保养；保养后
    第一抽样的 λ(0)=0，尺度 = 1e-9 的倒数（≈1e9）——与保养前的小尺度形成断崖。
    """
    cfg = SimConfig(pm_interval=150.0, machine_age_failure=True, machine_age_beta=2.0,
                    repair_time=0.0)
    m, scales, stats = _drive(cfg, [100.0, 100.0, 100.0])
    assert stats["pm_events"] == 1, f"前提：本配置必须恰好触发一次保养，实得 {stats['pm_events']}"
    assert m.pm_clock < cfg.pm_interval
    zeros = [s for a, s in scales if a == 0.0]
    before = [s for a, s in scales if a > 0.0]
    assert before, f"缺保养前的抽样：{scales}"
    # 保养前的抽样：役龄 100 ⟹ λ=0.0667 ⟹ 尺度 15 min（远小于上限）
    assert max(before) < 1e-3 * max(zeros), \
        f"保养未让故障率回落：高役龄尺度 {max(before)} vs 归零后 {max(zeros)}"
    # 役龄 0 的抽样应恰好两次：开工初始 + 保养归零后（两者都 λ(0)=0 ⟹ 尺度=上限）
    assert len(zeros) == 2, f"役龄 0 的抽样应恰两次（初始 + 保养后）：{scales}"
    assert all(s > 1e8 for s in zeros), f"λ(0)=0 ⟹ 尺度应为上限：{zeros}"


# ────────────────────────── 4. β=1 退化与前置守卫 ──────────────────────────

@pytest.mark.unit
def test_beta_one_degenerates_to_the_baseline_chain():
    """判据 4：β=1 时 λ ≡ base，故链路必须与关态**逐位相同**（内置退化为自检点）。"""
    cfg_hot = SimConfig(pm_interval=20.0)
    _d0, met_off = _run(cfg_hot, hot_fail=True)
    cfg_on = SimConfig(pm_interval=20.0, machine_age_failure=True, machine_age_beta=1.0)
    _d1, met_on = _run(cfg_on, hot_fail=True)
    assert _digest(met_on) == _digest(met_off), \
        "β=1 的役龄模型没有退化回常数率——曲线或接线写错了"


@pytest.mark.unit
def test_age_failure_requires_failure_and_maintenance_on():
    """判据 5：③ 关或 ⑫ 关时打开役龄开关必须**显式报错**（否则是死开关/无自变量）。"""
    inst = load_mk("mk01")
    cfg = SimConfig(machine_age_failure=True)
    lay, dm, ctx = build_setup(inst, cfg)
    for name, cons, match in (
            ("machine_failure", ConstraintConfig(machine_failure=False), "machine_failure"),
            ("maintenance", ConstraintConfig(maintenance=False), "maintenance")):
        with pytest.raises(ValueError, match=match):
            SimWorld(inst, lay, dm, cfg, constraints=cons)


@pytest.mark.unit
def test_age_failure_changes_dynamics_and_sensitivity():
    """判据 2（系统级）+ 要报的数：高故障配置下打开后的 fail_events/makespan 与关态不同，
    且形状参数 β 影响读数（敏感性 1.5/2/3）。"""
    cfg_off = SimConfig(pm_interval=20.0)
    _d, met_off = _run(cfg_off, hot_fail=True)
    res = {}
    for beta in (1.5, 2.0, 3.0):
        cfg = SimConfig(pm_interval=20.0, machine_age_failure=True, machine_age_beta=beta)
        _d, met = _run(cfg, hot_fail=True)
        res[beta] = met
        assert met["fail_events"] != met_off["fail_events"], \
            f"β={beta} 的故障数与关态相同——曲线没影响动力学"
    assert len({_digest(m) for m in res.values()}) == 3, "三个 β 的读数应互不相同（敏感性可见）"
