# -*- coding: utf-8 -*-
"""**网络 ↔ 仿真的桥**（v3 top-K 架构）：`NetPolicy` 的契约。

验收：一局由网络全程驱动跑完；**决策链**条数与仿真事件对得上；**同 seed 逐位复现**
（重算是训练的前提）；候选集与 logp 在 K 内归一化。
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from plcsp_v3.env import layout as layout_mod                        # noqa: E402
from plcsp_v3.env.corridors import build_corridor_graph, dock_distance_matrix  # noqa: E402
from plcsp_v3.env.des import SimConfig, SimWorld, reference_makespan  # noqa: E402
from plcsp_v3.env.instances import load_mk                            # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy                    # noqa: E402
from plcsp_v3.hrl.rollout import NetPolicy, Recorder                  # noqa: E402
from plcsp_v3.hrl.train import (RULE_NAMES, _candidate_logits,        # noqa: E402
                                _one_logp, rules_dict)
from plcsp_v3.nn.features import norm_context                          # noqa: E402


def _mk(name: str = "mk01"):
    """建一套（实例 · 布局 · 图 · 距离 · 配置 · 归一化上下文 · 网络）。"""
    torch.manual_seed(0)
    inst = load_mk(name)
    cfg = SimConfig()
    lay = layout_mod.sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    ctx = norm_context(inst, lay, reference_makespan(inst, cfg), cfg)
    net = HierarchicalPolicy().eval()
    return inst, lay, g, dock_distance_matrix(g), cfg, ctx, net


def _drive(name="mk01", *, greedy=True, seed=0, dt_events=20, rules=None):
    inst, lay, g, dm, cfg, ctx, net = _mk(name)
    w = SimWorld(inst, lay, dm, cfg, graph=g)
    p = NetPolicy(net, inst, lay, ctx, rules=rules, greedy=greedy, seed=seed,
                  dt_events=dt_events, freeze_rules=rules is not None)
    m = w.run_gated(seed_chain=0, policy_pa=p.policy_pa, policy_l=p.policy_l,
                    policy_m=p.policy_m,
                    policy_ma=p.policy_ma, dt_events=dt_events)
    return m, p


def _rules_row(d) -> dict:
    """`Decision` 的三个规则号 → `net(..., rules=…)` 要的**逐行**张量（这里只有一行）。"""
    return {k: torch.as_tensor([int(getattr(d, "rule_" + k.lower()))])
            for k in RULE_NAMES}


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
    # ⚠️ **同机台相邻工序跳过搬运**（A 组，2026-10-17）⟹ 不再恒为 `pa_assigns + 10`。
    # 每次 LA 决策对应一次实际运输 ⟹ 与 `deliveries` 同数（含回站段）。
    assert kinds["LA"] == m["deliveries"], ("每次 LA 决策对应一次运输", kinds, m["deliveries"])
    assert kinds["M"] == m["ops_done"], kinds
    assert kinds["MA"] >= 1, kinds
    assert len(p.rec) == sum(kinds.values())
    for d in p.rec.decisions:
        if d.kind == "MA":
            assert d.cand is None and d.act is None, "MA 不产候选/动作"
        else:
            assert d.cand is not None, f"{d.kind} 缺候选集"
            assert d.act is not None, f"{d.kind} 缺具体动作"
            assert 0 <= d.cand.act < len(d.cand.mask)


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
    _, p0 = _drive(dt_events=0)
    assert "timer" not in p0.ma_reasons


@pytest.mark.unit
def test_chain_is_reproducible_under_the_same_seed():
    """**同 seed ⟹ 整条决策链逐位相同**（规则 + 候选 + 动作 + logp）。这是重算的前提。"""
    m1, p1 = _drive(greedy=False, seed=7)
    m2, p2 = _drive(greedy=False, seed=7)
    assert m1["makespan"] == m2["makespan"]
    a = [(d.kind, d.rule_pa, d.rule_la, d.rule_ra, d.act) for d in p1.rec.decisions]
    b = [(d.kind, d.rule_pa, d.rule_la, d.rule_ra, d.act) for d in p2.rec.decisions]
    assert a == b
    assert [d.logp for d in p1.rec.decisions] == [d.logp for d in p2.rec.decisions]


@pytest.mark.unit
def test_recorded_decisions_reproduce_their_own_logits():
    """**重算是训练的前提**：拿记录下来的候选集再前向一次，该动作的 logp 必须相同。"""
    _, p = _drive(greedy=True)
    net = p.net
    checked = 0
    for d in p.rec.decisions:
        t = torch.as_tensor(d.tok).unsqueeze(0)
        with torch.no_grad():
            out = net(t, d.seg, rules=_rules_row(d), hard=True)
            lp = float(_one_logp(net, out, d, 0).detach())
        assert lp == pytest.approx(d.logp, abs=1e-5), \
            f"{d.kind} 的重算 logp 不一致：{lp} vs {d.logp}"
        checked += 1
        if checked >= 40:
            break
    assert checked > 0


@pytest.mark.unit
def test_frozen_rules_condition_the_lower_heads_and_skip_ma():
    """给定 `rules` ⟹ **条件在固定规则上**（阶段 B / 执行）：

    MA **照常被触发**（`ma_decisions` 照增 —— 停表靠的就是这个计数），
    但**不再重新采规则**，且**自己不进决策链**。
    """
    m, p = _drive(rules={"PA": 3, "LA": 1, "RA": 0})
    assert m["ma_decisions"] > 0, "顶层**照常触发**（冻结的是规则，不是触发）"
    assert p.cur == {"PA": 3, "LA": 1, "RA": 0}, "冻结 ⟹ 生效规则全程不变"
    assert all((d.rule_pa, d.rule_la, d.rule_ra) == (3, 1, 0)
               for d in p.rec.decisions if d.kind != "MA")
    assert not any(d.kind == "MA" for d in p.rec.decisions), \
        "冻结 ⟹ 顶层不产决策（但**触发计数照走**，见上）"


@pytest.mark.unit
def test_recorder_is_optional_and_counts_by_kind():
    inst, lay, g, dm, cfg, ctx, net = _mk()
    r = Recorder()
    assert len(r) == 0 and r.kinds() == {}
    w = SimWorld(inst, lay, dm, cfg, graph=g)
    p = NetPolicy(net, inst, lay, ctx, greedy=True, seed=0, record=False)
    w.run_gated(seed_chain=0, policy_pa=p.policy_pa, policy_l=p.policy_l,
                policy_m=p.policy_m, dt_events=0)
    assert p.rec is None, "record=False 时不该留决策链（省内存）"


@pytest.mark.unit
def test_greedy_takes_the_argmax_candidate():
    """greedy ⟹ 在候选集内取 argmax；logp 是 K 内 `log_softmax`。"""
    _, p = _drive(greedy=True)
    checked = 0
    for d in p.rec.decisions:
        if d.kind == "MA":
            continue
        t = torch.as_tensor(d.tok).unsqueeze(0)
        with torch.no_grad():
            out = p.net(t, d.seg, rules=_rules_row(d), hard=True)
            logits = _candidate_logits(p.net, out, d, 0)
        if int(np.count_nonzero(d.cand.mask)) <= 1:
            assert d.logp == 0.0
            continue
        best = float(torch.max(logits))
        tied = torch.nonzero(logits == best, as_tuple=False).flatten().tolist()
        if len(tied) == 1:                       # 并列时走逐决策 rng，这里不测
            assert int(d.cand.act) == int(tied[0])
        expect = float(torch.log_softmax(logits, dim=-1)[int(d.cand.act)])
        assert d.logp == pytest.approx(expect, abs=1e-5)
        checked += 1
        if checked >= 30:
            break
    assert checked > 0


@pytest.mark.unit
def test_forced_single_candidate_does_not_advance_the_sampling_stream():
    """⚠️ `K_eff == 1` 的强制决策**不抽签、不推进 `self.gen`**。

    否则"多一个强制决策"会把后续采样流整体移位 —— 同 seed 的两条链在不同状态下会分岔。
    判据：先过一个强制行、再过一个双候选行；与直接过双候选行抽到同一个动作。
    """
    inst, lay, g, dm, cfg, ctx, net = _mk()
    p = NetPolicy(net, inst, lay, ctx, greedy=False, seed=0)
    row1 = torch.tensor([0.0])                    # 只有一个有限值 ⟹ 强制
    row2 = torch.tensor([0.0, 1.0])
    rng = np.random.default_rng([0, 0])
    p.gen.manual_seed(11)
    a0, lp0 = p._pick_candidate(row1, rng)
    a, lp = p._pick_candidate(row2, rng)
    p.gen.manual_seed(11)
    b, lp_b = p._pick_candidate(row2, rng)
    assert a0 == 0 and lp0 == 0.0
    assert (a, lp) == (b, lp_b), "强制决策推进了采样流"


@pytest.mark.unit
def test_sampled_actions_are_seeded():
    """采样档同 seed 可复现；不同 seed 可不同。"""
    _, p1 = _drive(greedy=False, seed=3)
    _, p2 = _drive(greedy=False, seed=3)
    acts1 = [d.act for d in p1.rec.decisions]
    acts2 = [d.act for d in p2.rec.decisions]
    assert acts1 == acts2, "同 seed 的采样档不可复现 ⟹ `torch.Generator` 没接上"


# ══════════════════ 步 5b 的机制层：重放前缀 ══════════════════

@pytest.mark.unit
def test_top_level_points_record_cumulative_counters():
    """顶层每触发一次就记一条**累计读数**（§10③ 的段回报靠它取增量）。"""
    m, p = _drive(greedy=True, dt_events=20)
    assert len(p.ma_counters) == len(p.ma_reasons) == m["ma_decisions"]
    assert p.ma_reasons[0] == "initial", "第一个顶层点必须在 t=0（否则开局那 ΔT 没目标管）"
    c0, c1 = p.ma_counters[0], p.ma_counters[-1]
    assert c0.now == 0.0 and c0.done_ops == 0 and c0.done_jobs == 0
    assert c1.done_ops > c0.done_ops, "累计读数必须单调"
    assert all(b.done_ops >= a.done_ops for a, b in zip(p.ma_counters, p.ma_counters[1:]))


@pytest.mark.unit
def test_replaying_the_prefix_lands_on_the_same_state():
    """⚠️ **本步的核心性质**：从 t=0 **照已记录的决策回放**，必然回到同一个 `Z_t`。

    判据用**累计读数逐位相同**。
    """
    from plcsp_v3.hrl.replay import run_branch

    m, p = _drive(greedy=True, dt_events=20)
    ma_idx = [i for i, d in enumerate(p.rec.decisions) if d.kind == "MA"]
    assert len(ma_idx) >= 2, "本实例至少要两个顶层点才测得到"
    prefix = list(p.rec.decisions[:ma_idx[1]])
    d1 = p.rec.decisions[ma_idx[1]]
    rules_at_z1 = (d1.rule_pa, d1.rule_la, d1.rule_ra)

    inst, lay, g, dm, cfg, ctx, net = _mk()
    torch.manual_seed(0)
    net2 = HierarchicalPolicy().eval()
    net2.load_state_dict(net.state_dict())
    inner = NetPolicy(net2, inst, lay, ctx, rules=rules_dict(rules_at_z1),
                      greedy=True, seed=0, dt_events=20, freeze_rules=True)
    metrics, at_z1 = run_branch(_world(inst, lay, g, dm, cfg), prefix, inner,
                                switch_after_ma=1, stop_after_ma=2)

    base_at_z1 = p.ma_counters[1]
    assert metrics["ma_decisions"] == 2
    assert at_z1.now == pytest.approx(base_at_z1.now)
    assert at_z1.done_ops == base_at_z1.done_ops
    assert at_z1.done_jobs == base_at_z1.done_jobs
    assert at_z1.energy_kwh == pytest.approx(base_at_z1.energy_kwh)
    assert at_z1.twt == pytest.approx(base_at_z1.twt)


def _world(inst, lay, g, dm, cfg):
    return SimWorld(inst, lay, dm, cfg, graph=g)


def _spec_from(inst, lay, g, dm, cfg, ctx, constraints=None):
    """`_mk()` 的那套对象 → `Spec`（`base_chain` 要）。"""
    from plcsp_v3.env.reward import EQUAL_WEIGHTS
    from plcsp_v3.hrl.train import Spec
    return Spec(inst, lay, g, dm, cfg, ctx, EQUAL_WEIGHTS,
                constraints=constraints)


@pytest.mark.unit
def test_exact_rule_reproduces_the_cdr_baseline():
    """⚠️ **`exact_rule` 必须与 CDR 基线逐位相同**（2026-10-20）。

    此前固定规则对照表里的"策略 top-1"走的是 `pa_candidates` 排序后取第 0 个 ——
    **那不是规则的 argmin**（候选集截断、确定性平局）。本测试钉住修复：
    `exact_rule=True` 直接调 `rules.apply_pa_rule` + 环境 `tie_rng`，
    必须与 `cdr.run_cdr` 给出**同一个** makespan / 能耗 / 拖期。
    """
    from plcsp_v3.env.constraints import ConstraintConfig
    from plcsp_v3.hrl.cdr import run_cdr
    from plcsp_v3.hrl.rules import LA_RULE_NAMES, PA_RULE_NAMES
    from plcsp_v3.hrl.train import base_chain

    inst, lay, g, dm, cfg, ctx, net = _mk()
    cons = ConstraintConfig(maintenance=False)      # 与 CDR 对照档一致（RA 不参与）
    pa, la = "EAJ+EAM", "LATEST_IN_TIME"
    rules = {"PA": PA_RULE_NAMES.index(pa), "LA": LA_RULE_NAMES.index(la),
             "RA": 0}
    m_cdr = run_cdr(_world(inst, lay, g, dm, cfg), pa, la, seed_chain=0)
    # ⚠️ 必须走 `base_chain`：它负责"三个头全固定 ⟹ 不挂 `policy_ma`"。
    #    直接调 `run_gated(policy_ma=...)` 会打开机台扰动事件流，换掉世界。
    spec_p = base_chain(net, _spec_from(inst, lay, g, dm, cfg, ctx, cons),
                        greedy=True, seed=0, dt_events=20,
                        fixed_rules=rules, exact_rule=True)
    m_ex = spec_p.metrics
    for k in ("makespan", "energy", "tardy_twt"):
        assert m_ex[k] == pytest.approx(m_cdr[k], abs=1e-9), \
            f"exact_rule 与 CDR 在 {k} 上不同：{m_ex[k]} vs {m_cdr[k]}"


@pytest.mark.unit
def test_tie_seed_defaults_to_seed_chain_and_variants_differ():
    """⚠️ **平局流与场景流分开**（2026-10-20）。

    评测精度要求：同一场景跑多条平局流。`tie_seed=None` 必须**逐位等于**旧口径
    （即 `tie_seed=seed_chain`）；不同的 `tie_seed` 必须给出不同的读数。
    """
    from plcsp_v3.env.constraints import ConstraintConfig
    from plcsp_v3.hrl.rules import LA_RULE_NAMES, PA_RULE_NAMES
    from plcsp_v3.hrl.train import base_chain

    inst, lay, g, dm, cfg, ctx, net = _mk()
    cons = ConstraintConfig(maintenance=False)
    pa, la = "EAJ+EAM", "LATEST_IN_TIME"
    rules = {"PA": PA_RULE_NAMES.index(pa), "LA": LA_RULE_NAMES.index(la),
             "RA": 0}
    sp = _spec_from(inst, lay, g, dm, cfg, ctx, cons)
    kw = dict(greedy=True, seed=0, dt_events=20, fixed_rules=rules,
              exact_rule=True, seed_chain=3)
    a = base_chain(net, sp, **kw).metrics["makespan"]
    b = base_chain(net, sp, **kw, tie_seed=3).metrics["makespan"]
    assert a == pytest.approx(b, abs=1e-12), \
        "tie_seed=None 必须等于 tie_seed=seed_chain（逐位兼容）"
    vals = {round(float(base_chain(net, sp, **kw, tie_seed=10_000 + j)
                        .metrics["makespan"]), 6) for j in range(4)}
    assert len(vals) > 1, "4 条平局流给出同一个读数 ⟹ tie_seed 没接上"
