# -*- coding: utf-8 -*-
"""⑨ AGV 故障：在途任务退回队列（`SimConfig.agv_failover`，默认关）。

设计（`docs/mechanism-designs.md` 的「⑨ AGV 故障：在途任务退回队列」节，开工前定死）：

- **加决策点？否**——退回是规则（故障期间退回/转交），不是策略动作。
- **时点不变**：故障仍只在**段间**（不持任何区段锁）生效——持锁停机 = 死锁。
  退回发生在三个段间检查点：循环顶（转交本车队列）、`q.get()` 之后、`_collect` 之后。
- **退到哪**：`bound`（每车一 Store）转交**未停机的**其它车；FIFO（单个共享 Store）
  放回共享队列队尾。
- ⚠️ **这是改动力学**：打开后 makespan 会变（⑨ 的代价口径从"仅停机"改为
  "停机 + 重新派车 + 队列重排"）。

本文件的判据即设计节的验收判据 1–5。摘要值是**改造前**（2026-10-05，HEAD=2c7fe87）
在 MK01 上捕获的（默认配置与故障高频验证档 `agv_mtbf=12, agv_mttr=5`；bound 与 FIFO 两条路径）。
"""
from __future__ import annotations

import hashlib

import pytest
import torch

from plcsp.algo.group_rel import roll_chain
from plcsp.algo.policy import PolicyNet
from plcsp.algo.setup import build_setup
from plcsp.env.des import SimConfig, SimWorld
from plcsp.env.instances import load_mk
from plcsp.nn.encoder import LayoutEncoder

# 故障高频验证档：MTBF 12 min ≪ episode，MTTR 5 min——桩桩故障都落在 episode 内。
HOT = dict(agv_mtbf=12.0, agv_mttr=5.0)
# 改造前捕获的四个摘要（bound/fifo × 默认/高频，failover 关）。
# ⚠️ 2026-10-06 重捕：机床待机功率由 P^u 改为 Table 9 的 Standby Power（见 energy.py 模块 docstring）。
# 已实证决策 / makespan / travel / deliveries 逐位不变，只有 met[energy] 变。
# ⚠️ 2026-10-07 重捕（progress-log §64）：episode 停表由**死锁护栏**改为**末件到站**
# （`des.SimWorld.run`：`until=done | timeout(horizon)`，`horizon` 只兜真死锁）。
# 停表窗口缩短 ⟹ `agv_fail_events`（护栏窗口里照抽照记的 Poisson 计数）变小，其余字段不动。
# 已逐位实证：把**旧计数**换回新读数即命中全部 4 条旧 pin（旧 17/17/277/277 → 新 1/1/30/28），
# 决策链指纹与 n_dec 亦逐位不变 ⟹ 动的是**量窗**，不是轨迹。
DIGESTS = {
    ("bound", "default"): "eec6607bee8fbfda49816a521e130746cf14706739e79523dd446ea25cc192b0",
    ("fifo", "default"): "83567fb39ca361f1bc7b1c717cd8bbe1b72be40f5944816bb5c600a1f4c037a4",
    ("bound", "hot"): "61e4a4d7895506ad12b69492c226ddaeb586499e4ba0919e580d8d7cf81fe03d",
    ("fifo", "hot"): "3de7358a5a5ea09b33332e20f85d90a82fd090072439bfc3132932e0c3cf1fab",
}


def _digest(met: dict) -> str:
    """指标摘要——含 makespan/energy/deliveries/travel/moves/requeue/trips/故障数/
    完工表全表。任何"多抽一个随机数 / 时序变了 / 任务丢了"的漂移都会翻红。"""
    h = hashlib.sha256()
    for k in ("makespan", "energy", "deliveries", "travel_time_total", "moves",
              "requeue", "trips", "agv_fail_events", "jobs_done", "horizon_hit"):
        h.update(f"{k}={met.get(k)!r};".encode())
    h.update(repr(sorted(met.get("completes", {}).items())).encode())
    return h.hexdigest()


def _setup(cfg):
    inst = load_mk("mk01")
    lay, dm, ctx = build_setup(inst, cfg)
    return inst, lay, dm, cfg, ctx


def _expected_tasks(inst) -> int:
    """本实例的搬运任务总数 = 每作业（投放 + 工序流转 + 回站）= 工序数 + 1。"""
    return sum(len(job) + 1 for job in inst.jobs)


# ────────────────────────── 1. 关态逐位不变 ──────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("label,hot", [("default", False), ("hot", True)])
def test_failover_off_is_bit_identical_to_baseline(label, hot):
    """判据 1：`agv_failover=False`（默认）⟹ bound 与 FIFO 两条路径都逐位等于改造前。

    高频验证档（每小时约 5 次故障）把"退回逻辑若被误触发"的窗口放大——默认配置故障少，
    单靠它盖不住静默漂移。
    """
    cfg = SimConfig(**HOT) if hot else SimConfig()
    inst, lay, dm, cfg, ctx = _setup(cfg)
    torch.manual_seed(20261005)
    pol = PolicyNet(enc=LayoutEncoder())
    dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    assert met["agv_failover_tasks"] == 0, "关态不得发生任何退回/转交"
    assert _digest(met) == DIGESTS[("bound", label)], \
        "failover 关态的 bound 路径变了——既有读数不再成立"
    m2 = SimWorld(inst, lay, dm, cfg).run(seed_chain=0)
    assert m2["agv_failover_tasks"] == 0, "关态不得发生任何退回/转交"
    assert _digest(m2) == DIGESTS[("fifo", label)], \
        "failover 关态的 FIFO 路径变了——既有读数不再成立"


# ────────────────────────── 2. 打开：退回真的发生、动力学真的变 ──────────────────────────

@pytest.mark.unit
def test_failover_requeues_and_changes_makespan():
    """判据 2：高频验证档下 `agv_failover_tasks > 0`，且 makespan 与关态不同。

    关态读数（改造前捕获）：bound makespan = 139.089103；FIFO = 118.714584。
    打开后任务不再"车趴多久卡多久"，时序必然变——这是改动力学的**预期**后果。
    """
    inst, lay, dm, _, ctx = _setup(SimConfig(**HOT))
    cfg_off = SimConfig(**HOT)
    torch.manual_seed(20261005)
    pol = PolicyNet(enc=LayoutEncoder())
    _dec, met_off = roll_chain(inst, lay, dm, cfg_off, pol, seed=0, ctx=ctx)
    # 同一初始化/同一策略：打开后另建一份策略（同 seed ⟹ 同参数）
    cfg_on = SimConfig(**HOT, agv_failover=True)
    lay_on, dm_on, ctx_on = build_setup(inst, cfg_on)
    torch.manual_seed(20261005)
    pol_on = PolicyNet(enc=LayoutEncoder())
    _dec2, met_on = roll_chain(inst, lay_on, dm_on, cfg_on, pol_on, seed=0, ctx=ctx_on)
    assert met_on["agv_failover_tasks"] > 0, \
        "高频验证档下一次退回都没发生——failover 没接上（任务仍卡在停机车手里）"
    assert met_on["horizon_hit"] is False and met_on["jobs_done"] == inst.n_jobs, \
        "failover 配置跑不完（死锁？）"
    assert met_on["makespan"] != met_off["makespan"], \
        "打开 failover 后 makespan 一位没变——退回没有改变动力学"
    assert met_on["makespan"] < met_off["makespan"], \
        "退回任务后反而更慢——检查是否在转交时打乱了队列/重复执行"


# ────────────────────────── 3. 任务守恒（两种队列形状） ──────────────────────────

@pytest.mark.unit
def test_failover_conserves_tasks_bound_and_fifo():
    """判据 3/4：两种队列形状都跑完，且**每件搬运恰好送达一次**
    （`deliveries == 任务总数`；多了 = 重复执行，少了 = 丢任务）。"""
    cfg = SimConfig(**HOT, agv_failover=True)
    inst, lay, dm, cfg, ctx = _setup(cfg)
    n_tasks = _expected_tasks(inst)
    torch.manual_seed(7)
    pol = PolicyNet(enc=LayoutEncoder())
    _dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    assert met["horizon_hit"] is False and met["jobs_done"] == inst.n_jobs
    assert met["agv_failover_tasks"] > 0, "前提：本配置必须真的发生过退回"
    assert met["deliveries"] == n_tasks, \
        f"bound 配置送达 {met['deliveries']} 件 ≠ 任务总数 {n_tasks}——任务丢了或重复执行"
    m2 = SimWorld(inst, lay, dm, cfg).run(seed_chain=0)
    assert m2["horizon_hit"] is False and m2["jobs_done"] == inst.n_jobs
    assert m2["deliveries"] == n_tasks, \
        f"FIFO 配置送达 {m2['deliveries']} 件 ≠ 任务总数 {n_tasks}——任务丢了或重复执行"


@pytest.mark.unit
def test_failover_clears_in_transit_tracking():
    """判据 3（快照口径）：跑完后不得有任何作业仍挂在车上（`on_agv == -1`）。

    退回若只搬任务、不清 `track.job_agv`，快照的 `in_transit` / `on_agv` 会滞留成假信号
    （"这批活还在车上"而车早已空）。
    """
    cfg = SimConfig(**HOT, agv_failover=True)
    inst, lay, dm, cfg, ctx = _setup(cfg)
    w = SimWorld(inst, lay, dm, cfg)
    met = w.run(seed_chain=0)
    assert met["jobs_done"] == inst.n_jobs
    snap = w.snapshot()
    lingering = [j for j, js in enumerate(snap.jobs) if js.in_transit or js.on_agv != -1]
    assert not lingering, f"跑完后仍有作业挂在车上：{lingering}——job_agv 没清对"
    assert all(v.node >= 0 or v.status == 0 for v in snap.vehicles)
    assert snap.in_flight == 0, f"跑完后 in_flight={snap.in_flight} 应为 0"


# ────────────────────────── 4. 不死锁（持锁停机的守卫） ──────────────────────────

@pytest.mark.unit
def test_failover_never_deadlocks_under_brutal_failures():
    """判据 5：更狠的故障配置（MTBF 6 ≪ MTTR 8，车队近半时间在停机）也必须跑完。

    若实现让车**在持区段锁时**停机（或退回任务时没释放锁），同区段的车会永久等待 ⟹
    仿真推进到 horizon 仍未完工 ⟹ `horizon_hit=True` 当场翻红。这是"时点不许改"的
    端到端守卫（结构性保证 + 判据双保险）。
    """
    cfg = SimConfig(agv_mtbf=6.0, agv_mttr=8.0, agv_failover=True)
    inst, lay, dm, cfg, ctx = _setup(cfg)
    torch.manual_seed(3)
    pol = PolicyNet(enc=LayoutEncoder())
    _dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    assert met["horizon_hit"] is False, "重故障配置掐表（疑似死锁：车在持锁/在途时被处理？）"
    assert met["jobs_done"] == inst.n_jobs
    assert met["agv_failover_tasks"] > 0
    # 退回次数不应爆炸：转交给"停机车"会在两辆停机车之间乒乓（任务原地打转、计数虚增）。
    # 实测重故障配置 ≈ 1×任务数；5× 是防乒乓的宽上界，不是性能判据。
    assert met["agv_failover_tasks"] <= 5 * _expected_tasks(inst), \
        f"退回次数 {met['agv_failover_tasks']} 异常多——疑似在停机车之间乒乓"
    assert met["deliveries"] == _expected_tasks(inst), "重故障配置下任务不守恒"


@pytest.mark.unit
def test_failover_is_a_noop_when_agv_failure_is_off():
    """⑨ 关（`agv_failure=False`）时 failover 开关**无副作用**：没有故障可退。"""
    from plcsp.env.constraints import ConstraintConfig
    cfg = SimConfig(**HOT, agv_failover=True)
    inst = load_mk("mk01")
    cons = ConstraintConfig().with_off("agv_failure")
    lay, dm, ctx = build_setup(inst, cfg, constraints=cons)
    torch.manual_seed(5)
    pol = PolicyNet(enc=LayoutEncoder())
    _dec, met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx, constraints=cons)
    assert met["agv_fail_events"] == 0 and met["agv_failover_tasks"] == 0
    assert met["horizon_hit"] is False and met["jobs_done"] == inst.n_jobs
