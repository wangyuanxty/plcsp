# -*- coding: utf-8 -*-
"""**网络 ↔ 仿真的桥**（步 5a）：`NetPolicy` 的契约（**v3 口径**）。

验收：一局由网络全程驱动跑完；**决策链**条数与仿真事件对得上；**同 seed 逐位复现**
（重算是训练的前提）；顶层（MA）的触发理由与设计 §10⑤ 一致。

⚠️ **2026-10-10 改到 v3 口径**（`docs/v3/progress-log.md` §9.5）：

| v2 | **v3** |
|---|---|
| `PolicyOut.g_logits/g_bins/g/pa/la/ra_maint` | **`rule_logits / rule_sel / param_mu / value`** |
| `Decision.action`（**一个整数**） | **`(rule_pa, rule_la, rule_ra)` + `param_u`（一个向量）** |
| 一次决策 = "从固定几条规则里挑一条" | **"(规则, 参数)"两样 —— 一个整数装不下** |

**三条必须钉住的 v3 口径**（设计 §6.1 / §13.3）：

1. **`logp` 算在 `u` 上**（无界动作），**不是**算在 `param` 上；
2. **`greedy=True` ⟹ 连续头取 `μ`**（确定性模式）—— 不是取 argmax，也不是照旧采样；
3. **规则号 ↔ head 不错位**（`split_pa_rule` 与 `apply_pa_rule` 同源）。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from plcsp_v3.env import layout as layout_mod                        # noqa: E402
from plcsp_v3.env.corridors import build_corridor_graph, dock_distance_matrix  # noqa: E402
from plcsp_v3.env.des import SimConfig, SimWorld, reference_makespan  # noqa: E402
from plcsp_v3.env.instances import load_mk                            # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy                    # noqa: E402
from plcsp_v3.hrl.rollout import NetPolicy, Recorder                  # noqa: E402
from plcsp_v3.hrl.rules import SIGMA_DEFAULT                          # noqa: E402
from plcsp_v3.hrl.train import RULE_NAMES, _one_logp, rules_dict      # noqa: E402
from plcsp_v3.nn.features import norm_context                         # noqa: E402


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
    net = HierarchicalPolicy().eval()
    return inst, lay, g, dock_distance_matrix(g), cfg, ctx, net


def _drive(name="mk01", *, greedy=True, seed=0, dt_events=20, rules=None):
    inst, lay, g, dm, cfg, ctx, net = _mk(name)
    w = SimWorld(inst, lay, dm, cfg, graph=g)
    # ⚠️ `rules` 给定时用 **`freeze_rules`**（顶层**照常触发**、但不重新采规则）——
    # 不是"把 `policy_ma` 传 None"：那样顶层一次都不发生，`stop_after_ma` 也就永远不触发。
    p = NetPolicy(net, inst, lay, ctx, rules=rules, greedy=greedy, seed=seed,
                  dt_events=dt_events, freeze_rules=rules is not None)
    m = w.run_gated(seed_chain=0, policy_pa=p.policy_pa, policy_l=p.policy_l,
                    policy_m=p.policy_m,
                    policy_ma=p.policy_ma, dt_events=dt_events)
    return m, p


def _rules_row(d) -> dict:
    """`Decision` 的三个规则号 → `net(..., rules=…)` 要的**逐行**张量（这里只有一行）。

    ⚠️ 字段名从 `train.RULE_NAMES` 派生 —— **不另抄一遍**（设计 §13.3 ③ 的同一纪律）。
    """
    return {k: torch.as_tensor([int(getattr(d, "rule_" + k.lower()))])
            for k in RULE_NAMES}


def _us(p) -> list:
    """整条链的 `param_u`（MA 记 `None`）—— 逐位比较用。"""
    return [None if d.param_u is None
            else tuple(np.asarray(d.param_u, dtype=np.float32).tolist())
            for d in p.rec.decisions]


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
    assert kinds["MA"] >= 1, kinds
    assert len(p.rec) == sum(kinds.values())
    # ⚠️ **v3 新增**：一次决策是"(规则, 参数)"两样 —— 下层每一条都必须带 `param_u`
    # （少了它，事后带梯度重算 logp 就无从谈起）。
    for d in p.rec.decisions:
        if d.kind == "MA":
            assert d.param_u is None, "MA 不产参数（v3 里它产**规则**）"
        else:
            assert d.param_u is not None and d.param_u.size > 0, d.kind


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
    """**同 seed ⟹ 整条决策链逐位相同**（规则 + 参数 + logp）。这是重算的前提。"""
    m1, p1 = _drive(greedy=False, seed=7)
    m2, p2 = _drive(greedy=False, seed=7)
    assert m1["makespan"] == m2["makespan"]
    a = [(d.kind, d.rule_pa, d.rule_la, d.rule_ra) for d in p1.rec.decisions]
    b = [(d.kind, d.rule_pa, d.rule_la, d.rule_ra) for d in p2.rec.decisions]
    assert a == b
    assert [d.logp for d in p1.rec.decisions] == [d.logp for d in p2.rec.decisions]
    assert _us(p1) == _us(p2), "连续头的 `u` 不同 ⟹ 采样器没接在同一个 `torch.Generator` 上"


@pytest.mark.unit
def test_recorded_decisions_reproduce_their_own_logits():
    """**重算是训练的前提**：拿记录下来的 `(tok, seg, rules, la_ctx, param_u)` 再前向一次，
    该动作的 logp 必须逐位相同 —— 否则 `ratio ≠ 1` 是假象（v1 的教训）。

    存的必须是**当时的**上下文：状态变了用新快照重算 = 把策略输入换成了另一个状态。

    ⚠️ **v3 的判据统一走 `train._one_logp`** —— 那是训练器实际用的那一个。
    在这里另抄一遍公式，测的就是"两处抄得一样不一样"，不是"记录对不对"。
    """
    _, p = _drive(greedy=True)
    net = _mk()[-1]
    net.load_state_dict(p.net.state_dict())
    checked = 0
    for d in p.rec.decisions:
        t = torch.as_tensor(d.tok).unsqueeze(0)
        lc = None if d.la_ctx is None else torch.as_tensor(d.la_ctx)[None]
        with torch.no_grad():
            out = net(t, d.seg, rules=_rules_row(d), la_ctx=lc, hard=True)
            lp = float(_one_logp(out, d, 0).detach())
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

    ⚠️ **v2 对应的是"冻结 `g`"**；v3 里 `g` 没了，冻结的是**规则**（设计 §6 阶段 B）。
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
def test_greedy_takes_mu_and_the_logp_is_computed_on_u():
    """⚠️ **设计 §13.3 ①②**（v3 新加，这两条是连续动作最容易做错的地方）。

    **① `greedy=True` ⟹ 连续头取 `μ`**（确定性模式）—— 不是 argmax、也不是照旧采样。
    判据：记录的 `param_u` 与网络重算出的 `param_mu` **逐位相同**。

    **② 而由此可以**反推** `logp` 算在 `u` 上**：`u = μ` 时高斯密度取峰值 ⟹
    `logp = −½·k·log(2πσ²)`（`k` = 参数个数，**与 μ 无关**）。
    **若 `logp` 被算在 `param` 上**（少乘一个 Jacobian），这个值就不成立。
    """
    _, p = _drive(greedy=True)
    net = p.net
    checked = 0
    for d in p.rec.decisions:
        if d.kind == "MA":
            continue
        t = torch.as_tensor(d.tok).unsqueeze(0)
        lc = None if d.la_ctx is None else torch.as_tensor(d.la_ctx)[None]
        with torch.no_grad():
            out = net(t, d.seg, rules=_rules_row(d), la_ctx=lc, hard=True)
        key = {"PA": "PA", "LA": "LA", "M": "RA"}[d.kind]
        mu = out.param_mu[key][0].numpy()
        u = np.asarray(d.param_u, dtype=np.float32)
        assert u.shape == mu.shape, (d.kind, u.shape, mu.shape)
        assert np.array_equal(u, mu), \
            f"{d.kind}：greedy 档下 `u` 应当**逐位等于** `μ`（实测 {u} vs {mu}）"
        k = int(u.size)
        expect = -0.5 * k * math.log(2.0 * math.pi * SIGMA_DEFAULT ** 2)
        assert d.logp == pytest.approx(expect, abs=1e-5), \
            f"{d.kind}：`u = μ` 时的 logp 应为 {expect}，实得 {d.logp}（logp 算错变量了？）"
        checked += 1
        if checked >= 30:
            break
    assert checked > 0


@pytest.mark.unit
def test_param_mu_is_routed_by_the_rules_in_force_not_by_argmax():
    """⚠️ **规则号 ↔ 参数 head 必须对得上**（设计 §13.3 ③）—— 最隐蔽的一类错位。

    `param_mu` 按**哪条规则**的 head 算，决定了那几个 `λ` 的含义。若它按"网络自己 argmax
    出来的规则"算，而 `apply_pa_rule` 用的是 `self.cur`（**采样的**），参数就配到了
    **另一条规则**上 —— 不报错，只是学错。

    ⚠️ **这类错位有两个天然盲区**：
    ① `greedy=True` 时 `self.cur` **恰好就是** argmax ⟹ 两条路一致；
    ② 采样档下错位是**确定性**的 ⟹ "同 seed 复现"两边错到同一处。
    ⟹ 判据只能**直接比 `μ`**：拿冻结的规则重算，与记录的 `param_u` 逐位对上。
    """
    rules = {"PA": 11, "LA": 2, "RA": 2}
    other = {"PA": 3, "LA": 0, "RA": 1}
    _, p = _drive(rules=rules, greedy=True)
    assert p.cur == rules
    d = next(x for x in p.rec.decisions if x.kind == "PA")
    t = torch.as_tensor(d.tok).unsqueeze(0)
    with torch.no_grad():
        mu_r = p.net(t, d.seg, rules={k: torch.tensor([v]) for k, v in rules.items()},
                     hard=True).param_mu["PA"][0]
        mu_o = p.net(t, d.seg, rules={k: torch.tensor([v]) for k, v in other.items()},
                     hard=True).param_mu["PA"][0]
        sel_default = p.net(t, d.seg, hard=True).rule_sel["PA"]
    # 夹具有效性：argmax 不能恰好等于 `rules`（否则两条路本来就是同一条，测不出来）
    assert int(sel_default) != rules["PA"], "本夹具下 argmax 恰好等于 rules ⟹ 换一组规则"
    assert not torch.allclose(mu_r, mu_o), "两组规则算出同一个 μ ⟹ 换一组规则"
    u = torch.as_tensor(np.asarray(d.param_u, dtype=np.float32))
    assert torch.allclose(u, mu_r), \
        "`param_u` 不是按**生效的规则**路由的 —— 按 argmax 路由就是参数与规则错位"


@pytest.mark.unit
def test_sampled_params_actually_carry_sigma():
    """⚠️ **`σ` 要真的接上**：采样档的 `u` 必须**不等于** `μ`，且同 seed 可复现。

    若 `σ` 被写成 0（或忘了加噪声），**探索就整条没了**，而曲线看上去照跑。
    """
    _, pg = _drive(greedy=True)
    _, ps = _drive(greedy=False, seed=3)
    _, ps2 = _drive(greedy=False, seed=3)
    ug = [u for u in _us(pg) if u is not None]
    us = [u for u in _us(ps) if u is not None]
    assert ug and len(ug) == len(us), "两档的决策条数不同 ⟹ 不能用这条比"
    assert all(a != b for a, b in zip(ug, us)), "采样档与 `μ` 逐位相同 ⟹ σ 没接上"
    assert _us(ps2) == _us(ps), "同 seed 的采样档不可复现 ⟹ `torch.Generator` 没接上"


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

    这是"用重放代替分叉"（§13 #1）能成立的**唯一理由**。判据用**累计读数逐位相同** ——
    它把"机台队列 · 工件位置 · 车辆状态 · 计划表"等一切压缩成一个可比的数。

    ⚠️ 必须 `greedy=True`：**采样**下分支自己的随机流与 base 不同，动作会分岔，
    那测的就不是"回放"而是"另一条链"。

    ⚠️ **v3：回放吃的是 `Decision` 序列**（一次决策 = "(规则, 参数)"，一个整数装不下）——
    与 `train._prefix_to` **同源**，不另建一套编码。
    """
    from plcsp_v3.hrl.replay import run_branch

    m, p = _drive(greedy=True, dt_events=20)
    ma_idx = [i for i, d in enumerate(p.rec.decisions) if d.kind == "MA"]
    assert len(ma_idx) >= 2, "本实例至少要两个顶层点才测得到"
    # 前缀 = base 从 0 到**第 2 个顶层点之前**的全部决策（含第 1 个 MA 决策）
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
    # ⚠️ 比的是**累计读数**（两处同口径），不是 `metrics["energy"]`（那个按 makespan 算）
    assert at_z1.now == pytest.approx(base_at_z1.now)
    assert at_z1.done_ops == base_at_z1.done_ops
    assert at_z1.done_jobs == base_at_z1.done_jobs
    assert at_z1.energy_kwh == pytest.approx(base_at_z1.energy_kwh)
    assert at_z1.twt == pytest.approx(base_at_z1.twt)


def _world(inst, lay, g, dm, cfg):
    return SimWorld(inst, lay, dm, cfg, graph=g)
