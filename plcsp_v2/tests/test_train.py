# -*- coding: utf-8 -*-
"""两阶段交替训练（步 5b）的机制验收。

⚠️ 这一步**慢**（重放前缀 × 组大小）—— 测试用 `dt_events=50` 把顶层点压到 2–3 个，
**不是**设计值 §10② 的 20。设计值的代价见模块末的实测记录。
"""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from plcsp_v2.env import layout as layout_mod                        # noqa: E402
from plcsp_v2.env.corridors import build_corridor_graph, dock_distance_matrix  # noqa: E402
from plcsp_v2.env.des import SimConfig, reference_makespan            # noqa: E402
from plcsp_v2.env.instances import load_mk                            # noqa: E402
from plcsp_v2.env.reward import ReferenceObjectives, reward_weights   # noqa: E402
from plcsp_v2.hrl.agents import G_BINS, G_DIM, HierarchicalPolicy     # noqa: E402
from plcsp_v2.hrl.pool import SegmentPool                             # noqa: E402
from plcsp_v2.hrl.train import (Spec, base_chain, group_advantage,    # noqa: E402
                                ma_logp_at, segment_from, train_step)
from plcsp_v2.nn.features import norm_context                         # noqa: E402

DT = 50          # ⚠️ 只为测试提速；设计值见 §10②（20）


def _spec(tau: float = 1.0):
    """⚠️ **必须给一个收紧的 `tau`**：MK01 默认参考解 **TWT=0**，而 `reward_weights`
    对 0 分量**显式报错**（"TWT=0 的实例须先调 τ"）。不静默补假权重。"""
    torch.manual_seed(0)
    inst = load_mk("mk01")
    cfg = SimConfig(tau=tau)
    lay = layout_mod.sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    dm = dock_distance_matrix(g)
    ctx = norm_context(inst, lay, reference_makespan(inst, cfg), cfg)
    w = reward_weights(ReferenceObjectives.of(inst, cfg).as_tuple())
    net = HierarchicalPolicy(n_chargers=len(lay.chargers))
    return Spec(inst, lay, g, dm, cfg, ctx, w), net


@pytest.mark.unit
def test_reward_weights_refuse_a_zero_component():
    """⚠️ MK01 的**默认**参考解 TWT=0 ⟹ `reward_weights` 必须**显式报错**。

    **这不是障碍，是纪律**：0 分量会让逆量纲权重变成 `1/0`。语料里写明的处置是
    "先调 τ"，**不是**给一个假权重。
    """
    torch.manual_seed(0)
    inst = load_mk("mk01")
    cfg = SimConfig()                                   # 默认 τ = 冻结标定表
    ref = ReferenceObjectives.of(inst, cfg)
    assert ref.as_tuple()[2] == 0.0, "MK01 默认参考解应无拖期（本测试的前提）"
    with pytest.raises(ValueError, match="TWT=0"):
        reward_weights(ref.as_tuple())


@pytest.mark.unit
def test_group_advantage_sums_to_zero_and_is_scale_invariant():
    """组内 z 化（§4.1）：**和恒为 0**（这是"组内只比相对好坏"的代数形式）、对整体平移不变。"""
    a = group_advantage([1.0, 2.0, 3.0, 4.0])
    assert abs(float(a.sum())) < 1e-9
    b = group_advantage([101.0, 102.0, 103.0, 104.0])
    assert a == pytest.approx(b)
    assert group_advantage([5.0, 5.0, 5.0]).tolist() == [0.0, 0.0, 0.0]   # σ=0 ⟹ 全 0


@pytest.mark.slow
def test_gradients_reach_only_the_layer_being_trained():
    """⚠️ **本步的关键验收**：**阶段 B 只动下层、阶段 A 只动顶层**（设计 §4.2/§4.3）。

    而且——**实测会看到阶段 A 在 `B=0` 时梯度恒 0**（`σ=0`，见 §89.2）。
    **这不是 bug**：`B` 零初始化 ⟹ `g` 不影响行为 ⟹ G 个目标的段回报全相同。
    **阶段 A 要有信号，得先等阶段 B 把 `B` 学非零。**
    """
    spec, net = _spec()
    out = train_step(net, spec, None, M=2, G=2, seed=0, dt_events=DT)
    b, a = out["B"]["grad_norm"], out["A"]["grad_norm"]
    assert b["upper"] == 0.0, f"阶段 B 动了顶层（应为 0）：{b}"
    assert b["lower"] > 0.0, f"阶段 B 没动下层：{b}"
    assert out["B"]["loss"] != 0.0
    # 阶段 A：`B=0` ⟹ σ=0 ⟹ Â≡0 ⟹ loss 与梯度都为 0（**预期的空转**）
    assert all(s == 0.0 for s in out["A"]["seg_std"]), out["A"]["seg_std"]
    assert out["A"]["loss"] == 0.0 and a["upper"] == 0.0


@pytest.mark.slow
def test_segment_pool_matches_the_sequential_path():
    """⚠️ **并行跑段 = 只改怎么算，不改算什么**（设计 §105 的优化 #1）。

    `train_step` 每步要做 `n_top × (M+G)` 条段（mk01 96 条、mk10 400 条），**互相独立**。
    把它们放进进程池是**唯一不动设计参数**就能拿量级提升的一条 ——
    但必须证明它**不改读数**，否则所有既有结论作废。

    ⚠️ 容差不是逐位：并行档每进程 **1 线程**，浮点归约次序与父进程不同 ⟹ 末位可差。
    """
    spec, net = _spec()
    torch.manual_seed(0)
    base_net = HierarchicalPolicy(n_chargers=len(spec.layout.chargers))
    n_seq = copy.deepcopy(base_net)
    seq = train_step(n_seq, spec, None, M=2, G=2, seed=0, dt_events=DT)
    n_par = copy.deepcopy(base_net)
    with SegmentPool(n_par, spec, n_workers=3, dt_events=DT) as pool:
        par = train_step(n_par, spec, None, M=2, G=2, seed=0, dt_events=DT, pool=pool)
    for ph in ("B", "A"):
        assert seq[ph]["n_top"] == par[ph]["n_top"], "顶层点数不同 ⟹ 段的结构就不同了"
        assert seq[ph]["loss"] == pytest.approx(par[ph]["loss"], rel=1e-4, abs=1e-4), \
            f"阶段 {ph} 的 loss 不同：{seq[ph]['loss']} vs {par[ph]['loss']}"


@pytest.mark.unit
def test_segment_pool_refuses_one_worker():
    """⚠️ **先查后跑**：1 个进程没有意义，直接报错。"""
    spec, _ = _spec()
    net = HierarchicalPolicy(n_chargers=len(spec.layout.chargers))
    with pytest.raises(ValueError, match="n_workers"):
        SegmentPool(net, spec, n_workers=1, dt_events=DT)


@pytest.mark.slow
def test_ma_logp_is_per_goal_not_a_64_way_softmax():
    """⚠️ **顶层是 3 个独立分档，不是一个 64 路类别**（§10④）。

    同一顶层点上，`ma_logp_at` 给出的 64 个数必须等于**三维 log-softmax 之和**；
    且**各不相同**（否则"每个成员配自己的动作"就退化成同一个数）。
    """
    spec, net = _spec()
    base = base_chain(net, spec, greedy=True, seed=0, dt_events=DT)
    d = next(x for x in base.rec.decisions if x.kind == "MA")
    lp = ma_logp_at(net, d)
    assert lp.shape == (G_BINS ** G_DIM,) == (64,)
    assert len(set(lp.tolist())) > 1, "64 个 logp 全相同 ⟹ 分档没起作用"
    # 手算第 0 个：b=(0,0,0)
    t = torch.as_tensor(d.tok).unsqueeze(0)
    gb = torch.as_tensor([list(d.g_bins)])
    with torch.no_grad():
        o = net(t, d.seg, gb, la_ctx=None, hard=True)
    manual = sum(torch.log_softmax(o.g_logits[0, i], dim=-1)[0] for i in range(G_DIM))
    assert float(lp[0].detach()) == pytest.approx(float(manual.detach()), abs=1e-6)


@pytest.mark.slow
def test_segment_from_returns_the_branchs_own_decision_chain():
    """⚠️ **分支的 loss 必须用分支自己的决策链**，不是 base 链的。

    目标变了 ⟹ 动作变 ⟹ 状态变 ⟹ 决策条数与内容都不同。用 base 的链算 loss 是**错配**。
    """
    spec, net = _spec()
    base = base_chain(net, spec, greedy=True, seed=0, dt_events=DT)
    goals = [(0, 0, 0), (G_BINS - 1,) * G_DIM]
    br0 = segment_from(net, spec, base, 0, goal=goals[0], seed=0, greedy=True, dt_events=DT)
    br1 = segment_from(net, spec, base, 0, goal=goals[1], seed=0, greedy=True, dt_events=DT)
    assert br0.decisions, "分支必须记下自己的决策链"
    assert all(d.kind != "MA" for d in br0.decisions), \
        "分支里 `g` 被冻结 ⟹ 不该有顶层决策进链（前缀由脚本回放，也不进链）"
    # 目标不同的两条分支，决策链**可以**相同（`B=0` 时必然相同）——这一点如实钉住
    same = [(d.kind, d.action) for d in br0.decisions] == [(d.kind, d.action) for d in br1.decisions]
    assert same, ("`B=0` 时两条分支应逐位相同（`g` 还不影响行为）——"
                  "若这里变了，说明 `B` 已经不是零初始了")
