# -*- coding: utf-8 -*-
"""**MAPPO 基线**的机制验收：PPO 更新真的落到四个头与 critic 上。"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from plcsp_v3.env import layout as layout_mod                        # noqa: E402
from plcsp_v3.env.corridors import build_corridor_graph, dock_distance_matrix  # noqa: E402
from plcsp_v3.env.des import SimConfig, reference_makespan            # noqa: E402
from plcsp_v3.env.instances import load_mk                            # noqa: E402
from plcsp_v3.env.reward import EQUAL_WEIGHTS   # noqa: E402
from plcsp_v3.hrl.agents import HierarchicalPolicy                    # noqa: E402
from plcsp_v3.hrl.flat import ALL_KINDS                              # noqa: E402
from plcsp_v3.hrl.mappo import MAPPOConfig, mappo_step                # noqa: E402
from plcsp_v3.hrl.train import (Spec, base_chain,                     # noqa: E402
                                batched_terms_entropy)
from plcsp_v3.nn.features import norm_context                         # noqa: E402

DT = 50          # ⚠️ 只为测试提速；设计值见 §10②（20）


def _spec(tau: float = 1.0) -> Spec:
    """小规模测试档（与 `test_flat._spec` 同口径）。"""
    torch.manual_seed(0)
    inst = load_mk("mk01")
    cfg = SimConfig(tau=tau)
    lay = layout_mod.sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    dm = dock_distance_matrix(g)
    ctx = norm_context(inst, lay, reference_makespan(inst, cfg), cfg)
    w = EQUAL_WEIGHTS
    bounds = ((60.0, 4.0, 0.0), (140.0, 8.0, 300.0))
    return Spec(inst, lay, g, dm, cfg, ctx, w, bounds=bounds)


@pytest.mark.unit
def test_mappo_refuses_a_net_without_a_critic_head():
    """⚠️ **先查后跑**：缺 critic 头当场报错，不先跑仿真。"""
    spec = _spec()
    with pytest.raises(ValueError, match="critic"):
        mappo_step(HierarchicalPolicy(), spec, n=1, seed=0, dt_events=DT)


@pytest.mark.slow
def test_mappo_step_produces_finite_metrics_and_updates_all_heads():
    """一步 MAPPO：四个头、encoder、critic 都拿到梯度；读数有限。"""
    spec = _spec()
    torch.manual_seed(0)
    net = HierarchicalPolicy(with_value=True)
    # ⚠️ **minibatch 必须覆盖全部决策**：否则最后一个 minibatch 可能恰好没有
    #    MA 决策 ⟹ `optim.zero_grad` 之后 `select` 的 `.grad` 是 None（不是没更新过）。
    out = mappo_step(net, spec, n=2, seed=0, dt_events=DT,
                     cfg=MAPPOConfig(epochs=1, minibatch=100_000))
    assert out["n_decisions"] > 0 and out["n_chains"] == 2
    for key in ("loss", "policy_loss", "value_loss", "entropy", "grad_norm",
                "r_mean", "r_std"):
        assert np.isfinite(out[key]), f"{key} 不是有限值：{out[key]}"
    assert out["value_loss"] > 0.0, "critic 从 0 附近起步，value loss 应大于 0"
    assert out["entropy"] >= 0.0
    assert out["grad_norm"] > 0.0
    # ⚠️ **四个头都要有梯度**：一个头的 grad 为 None ⟹ 它根本没进 PPO 的图。
    for head in (net.select, net.scorer_pa, net.scorer_la, net.scorer_ra,
                 net.value_head, net.enc):
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        assert grads, f"{type(head).__name__} 一个梯度都没有"
        assert sum(float(g.abs().sum()) for g in grads) > 0.0


@pytest.mark.slow
def test_entropy_terms_align_with_logp_and_value():
    """`batched_terms_entropy` 三个向量**同序同长**，熵非负。"""
    spec = _spec()
    torch.manual_seed(0)
    net = HierarchicalPolicy(with_value=True)
    base = base_chain(net, spec, greedy=True, seed=0, dt_events=DT)
    lp, v, ent = batched_terms_entropy(net, base.rec.decisions, ALL_KINDS)
    n = len(base.rec.decisions)
    assert lp.shape == v.shape == ent.shape == (n,)
    assert bool((ent >= -1e-6).all()), "熵不该为负"
