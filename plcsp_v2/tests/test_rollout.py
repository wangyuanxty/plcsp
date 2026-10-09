# -*- coding: utf-8 -*-
"""**网络 ↔ 仿真的桥**（步 5a）：`NetPolicy` 的契约。

验收：一局由网络全程驱动跑完；**决策链**条数与仿真事件对得上；**同 seed 逐位复现**
（重算是训练的前提）；顶层（MA）的触发理由与设计 §10⑤ 一致。
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from plcsp_v2.env import layout as layout_mod                        # noqa: E402
from plcsp_v2.env.corridors import build_corridor_graph, dock_distance_matrix  # noqa: E402
from plcsp_v2.env.des import SimConfig, SimWorld, reference_makespan  # noqa: E402
from plcsp_v2.env.instances import load_mk                            # noqa: E402
from plcsp_v2.hrl.agents import HierarchicalPolicy            # noqa: E402
from plcsp_v2.hrl.rollout import NetPolicy, Recorder                  # noqa: E402
from plcsp_v2.nn.features import norm_context                         # noqa: E402


def _mk(name: str = "mk01"):
    """建一套（实例 · 布局 · 图 · 距离 · 配置 · 归一化上下文 · 网络）。

    ⚠️ **网络权重要固定**：`HierarchicalPolicy` 默认**随机初始化**，不播种的话两次调用
    拿到的是**两个不同的网络** —— 那样"同 seed 复现"测的就不是仿真了。
    """
    torch.manual_seed(0)
    inst = load_mk(name)
    cfg = SimConfig()
    lay = layout_mod.sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    ctx = norm_context(inst, lay, reference_makespan(inst), cfg)
    net = HierarchicalPolicy(n_chargers=len(lay.chargers)).eval()
    return inst, lay, g, dock_distance_matrix(g), cfg, ctx, net


def _drive(name="mk01", *, greedy=True, seed=0, dt_events=20, g_bins=None):
    inst, lay, g, dm, cfg, ctx, net = _mk(name)
    w = SimWorld(inst, lay, dm, cfg, graph=g)
    p = NetPolicy(net, inst, lay, ctx, g_bins=g_bins, greedy=greedy, seed=seed)
    m = w.run_gated(seed_chain=0, policy_pa=p.policy_pa, policy_l=p.policy_l,
                    policy_c=p.policy_c, policy_m=p.policy_m,
                    policy_ma=None if g_bins is not None else p.policy_ma,
                    dt_events=dt_events)
    return m, p


@pytest.mark.unit
def test_network_drives_a_full_episode_and_the_chain_matches_the_sim_events():
    """一局跑完；**决策链条数与仿真事件一一对上**。

    | 头 | 应有多少条 | 依据 |
    |---|---|---|
    | **PA** | `ops_done`（每道工序恰派一次） | §74 |
    | **LA** | `pa_assigns`（派工那一刻定车）+ 回站任务数 | §74.5 |
    | **M** | `ops_done`（每道工序后一个维护决策点） | `_pm_after_op` |
    """
    m, p = _drive()
    kinds = p.rec.kinds()
    assert m["jobs_done"] == 10 and not m["horizon_hit"]
    assert kinds["PA"] == m["ops_done"] == 55, kinds
    assert kinds["LA"] == m["pa_assigns"] + 10, ("派工点 55 + 回站 10", kinds)
    assert kinds["M"] == m["ops_done"], kinds
    assert kinds["C"] > 0 and kinds["MA"] >= 1, kinds
    assert len(p.rec) == sum(kinds.values())


@pytest.mark.unit
def test_ma_trigger_is_timer_plus_disturbance():
    """设计 §10⑤：顶层触发 = **定时器 + 机障 ③ + AGV 故障 ⑨**。"""
    m, p = _drive(dt_events=20)
    reasons = set(p.ma_reasons)
    assert "timer" in reasons, p.ma_reasons
    assert p.ma_reasons.count("timer") == m["ops_done"] // 20, (p.ma_reasons, m["ops_done"])
    assert p.ma_reasons.count("disturbance") == m["dist_events"], (p.ma_reasons,
                                                                  m["dist_events"])
    assert m["ma_decisions"] == len(p.ma_reasons)
    # `dt_events=0` ⟹ 定时器关，只剩扰动触发
    _, p0 = _drive(dt_events=0)
    assert "timer" not in p0.ma_reasons


@pytest.mark.unit
def test_chain_is_reproducible_under_the_same_seed():
    """**同 seed ⟹ 整条决策链逐位相同**（动作序列 + logp）。这是重算的前提。"""
    m1, p1 = _drive(greedy=False, seed=7)
    m2, p2 = _drive(greedy=False, seed=7)
    assert m1["makespan"] == m2["makespan"]
    a = [(d.kind, d.action) for d in p1.rec.decisions]
    b = [(d.kind, d.action) for d in p2.rec.decisions]
    assert a == b
    assert [d.logp for d in p1.rec.decisions] == [d.logp for d in p2.rec.decisions]


@pytest.mark.unit
def test_recorded_decisions_reproduce_their_own_logits():
    """**重算是训练的前提**：拿记录下来的 `(tok, seg, g, la_ctx)` 再前向一次，
    该动作的 logits 必须逐位相同 —— 否则 `ratio ≠ 1` 是假象（v1 的教训）。

    存的必须是**当时的**上下文：状态变了用新快照重算 = 把策略输入换成了另一个状态。
    """
    _, p = _drive(greedy=True)
    net = _mk()[-1]
    net.load_state_dict(p.net.state_dict())
    checked = 0
    for d in p.rec.decisions:
        if d.kind == "MA":
            continue
        t = torch.as_tensor(d.tok).unsqueeze(0)
        gb = torch.as_tensor([list(d.g_bins)])
        lc = None if d.la_ctx is None else torch.as_tensor(d.la_ctx)[None]
        with torch.no_grad():
            out = net(t, d.seg, gb, la_ctx=lc, hard=True)
        head = {"PA": out.pa, "LA": out.la, "C": out.ra_charge, "M": out.ra_maint}[d.kind]
        assert head is not None, d.kind
        row = head[0]
        prob = torch.softmax(row, dim=-1)
        assert float(torch.log(prob[d.action].clamp_min(1e-12))) == pytest.approx(
            d.logp, abs=1e-5), f"{d.kind} 的动作 {d.action} 重算 logp 不一致"
        checked += 1
        if checked >= 40:
            break
    assert checked > 0


@pytest.mark.unit
def test_fixed_g_conditions_the_lower_heads_and_skips_ma():
    """给定 `g_bins` ⟹ **条件在固定目标上**（阶段 B / 执行）：MA 不再被触发。"""
    m, p = _drive(g_bins=(3, 0, 3))
    assert p.ma_reasons == [] and m["ma_decisions"] == 0
    assert p.g_bins == (3, 0, 3)
    assert all(d.g_bins == (3, 0, 3) for d in p.rec.decisions if d.kind != "MA")
    assert not any(d.kind == "MA" for d in p.rec.decisions)


@pytest.mark.unit
def test_recorder_is_optional_and_counts_by_kind():
    inst, lay, g, dm, cfg, ctx, net = _mk()
    r = Recorder()
    assert len(r) == 0 and r.kinds() == {}
    w = SimWorld(inst, lay, dm, cfg, graph=g)
    p = NetPolicy(net, inst, lay, ctx, greedy=True, seed=0, record=False)
    w.run_gated(seed_chain=0, policy_pa=p.policy_pa, policy_l=p.policy_l,
                policy_c=p.policy_c, policy_m=p.policy_m, dt_events=0)
    assert p.rec is None, "record=False 时不该留决策链（省内存）"
