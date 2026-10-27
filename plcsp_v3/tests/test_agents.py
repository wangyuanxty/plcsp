# -*- coding: utf-8 -*-
"""两层策略的网络（v3 top-K 架构）：前向形状 · 掩码 · 候选打分。

⚠️ **2026-10-18 改到 top-K 口径**（`docs/v3/topk-rerank-spec.md`）。
旧架构的连续参数头（`RuleParamHeads` / `param_mu`）已删；接替的是三个打分头
`score_pa` / `score_la` / `score_ra`。
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from plcsp_v3.env import layout as layout_mod                     # noqa: E402
from plcsp_v3.env.corridors import build_corridor_graph, dock_distance_matrix  # noqa: E402
from plcsp_v3.env.des import SimConfig, SimWorld                  # noqa: E402
from plcsp_v3.env.instances import load_mk                        # noqa: E402
from plcsp_v3.hrl.agents import (LA_CTX_DIM, HierarchicalPolicy)  # noqa: E402
from plcsp_v3.hrl.rules import (N_LA_RULES, N_PA_RULES,  # noqa: E402
                                N_PM_RULES)
from plcsp_v3.nn.features import norm_context              # noqa: E402
from plcsp_v3.nn.state_emb import build_tok                       # noqa: E402


def _fwd(net, tok, seg, rules=None):
    """统一前向入口（top-K 起下层不在 `forward` 里算）。"""
    return net(tok, seg, rules=rules)


def _tok_seg(name: str = "mk01"):
    inst = load_mk(name)
    cfg = SimConfig()
    lay = layout_mod.sample_layout(inst.n_machines, seed=0, n_agv=cfg.n_agv)
    g = build_corridor_graph(lay)
    w = SimWorld(inst, lay, dock_distance_matrix(g), cfg, graph=g)
    ctx = norm_context(inst, lay, 75.87, cfg)
    arr, seg = build_tok(w.snapshot(), inst, lay, ctx)
    return torch.as_tensor(arr).unsqueeze(0), seg


@pytest.mark.unit
def test_forward_shapes_and_masks():
    """上层三个独立分类头 + 编码器 `z`；下层打分头在 `forward` 之外。"""
    tok, seg = _tok_seg()
    net = HierarchicalPolicy()
    b = tok.shape[0]
    out = _fwd(net, tok, seg)
    assert set(out.rule_logits) == {"PA", "LA", "RA"}
    assert out.rule_logits["PA"].shape == (b, N_PA_RULES) == (b, 12)
    assert out.rule_logits["LA"].shape == (b, N_LA_RULES) == (b, 3)
    assert out.rule_logits["RA"].shape == (b, N_PM_RULES) == (b, 3)
    for k, lg in out.rule_logits.items():
        assert out.rule_sel[k].shape == (b,), k
        assert 0 <= int(out.rule_sel[k].min()) and int(out.rule_sel[k].max()) < lg.shape[-1]
    assert out.z.shape == (b, sum(seg), net.enc.d_model)
    assert out.value is None
    mk = HierarchicalPolicy.masks(seg)
    assert mk["PA"] == (0, seg[0] + seg[1]) and mk["RA_maint"] == (0, seg[0])
    assert mk["LA"] == (seg[0] + seg[1], seg[0] + seg[1] + seg[2])
    assert len(seg) == 4 and seg[3] == 1


@pytest.mark.unit
def test_the_selected_rule_conditions_the_score_head():
    """⚠️ **top-K 架构的核心接缝**：同一候选集、不同规则号 ⟹ 不同 logits。

    规则既要筛候选集（`candidates.py`），也要条件化打分头（规则 embedding）。
    """
    tok, seg = _tok_seg()
    net = HierarchicalPolicy().eval()
    with torch.no_grad():
        z = _fwd(net, tok, seg).z
        rows_a = torch.tensor([[seg[0] + 0, seg[0] + 1, seg[0] + 2]])
        rows_b = torch.tensor([[0, 1, 2]])
        feats = torch.zeros(1, 3, 9)
        a = net.score_pa(z, rows_a, rows_b, feats, torch.tensor([0]))
        b = net.score_pa(z, rows_a, rows_b, feats, torch.tensor([1]))
    assert a.shape == b.shape == (1, 3)
    assert not torch.allclose(a, b), "换规则号没改变打分 ⟹ 规则 embedding 没接上"


@pytest.mark.unit
def test_frozen_rules_are_used_verbatim():
    """给定 `rules` ⟹ `rule_sel` **逐位等于**给的号（阶段 B / 执行档）。"""
    tok, seg = _tok_seg()
    net = HierarchicalPolicy()
    rules = {"PA": torch.tensor([7]), "LA": torch.tensor([2]), "RA": torch.tensor([1])}
    out = _fwd(net, tok, seg, rules)
    for k, want in rules.items():
        assert int(out.rule_sel[k]) == int(want), k


@pytest.mark.unit
def test_rule_head_sizes_match_the_rule_space():
    """上层类数 = 规则空间（12 / 3 / 3）；下层打分头每个 agent 一个。"""
    net = HierarchicalPolicy()
    assert net.select.heads["PA"].out_features == N_PA_RULES == 12
    assert net.select.heads["LA"].out_features == N_LA_RULES == 3
    assert net.select.heads["RA"].out_features == N_PM_RULES == 3
    assert net.scorer_pa.trunk[-1].out_features == 1
    assert net.scorer_la.trunk[-1].out_features == 1
    assert net.scorer_ra.trunk[-1].out_features == 2


@pytest.mark.unit
def test_score_pa_normalizes_within_the_candidate_set():
    """组内 z 化：K>=3 时 logits 的均值 ≈ 0、标准差 ≈ 1。"""
    tok, seg = _tok_seg()
    net = HierarchicalPolicy().eval()
    with torch.no_grad():
        z = _fwd(net, tok, seg).z
        rows_a = torch.tensor([[seg[0] + 0, seg[0] + 1, seg[0] + 2, seg[0] + 3]])
        rows_b = torch.tensor([[0, 1, 2, 3]])
        feats = torch.randn(1, 4, 9)
        s = net.score_pa(z, rows_a, rows_b, feats, torch.tensor([5]))
    assert s.shape == (1, 4)
    assert float(s.mean()) == pytest.approx(0.0, abs=1e-5)
    # ⚠️ eps 让 std 略小于 1（`sqrt(var + eps) + eps`）；只钉"量级被拉回 1"。
    assert 0.9 <= float(s.std(unbiased=False)) <= 1.001


@pytest.mark.unit
def test_score_la_requires_task_context():
    """⚠️ **没给 `la_ctx` ⟹ 显式报错**，不是补 0 再算。"""
    tok, seg = _tok_seg()
    net = HierarchicalPolicy()
    with torch.no_grad():
        z = _fwd(net, tok, seg).z
        rows = torch.tensor([[seg[0] + seg[1] + 0, seg[0] + seg[1] + 1]])
        feats = torch.zeros(1, 2, 4)
        with pytest.raises(ValueError, match="la_ctx"):
            net.score_la(z, rows, feats, torch.tensor([0]), None)
        out = net.score_la(z, rows, feats, torch.tensor([0]),
                           torch.zeros(1, LA_CTX_DIM))
    assert out.shape == (1, 2)


@pytest.mark.unit
def test_score_ra_has_two_actions_and_masks_disallowed():
    """RA：每个决策出 2 个 logit；允许集外的置 `-inf`。"""
    tok, seg = _tok_seg()
    net = HierarchicalPolicy().eval()
    with torch.no_grad():
        z = _fwd(net, tok, seg).z
        row = torch.tensor([[0]])
        feats = torch.zeros(1, 1, 5)
        mask = torch.tensor([[False, True]])
        s = net.score_ra(z, row, feats, torch.tensor([0]), mask)
    assert s.shape == (1, 2)
    assert np.isneginf(float(s[0, 0])) and np.isfinite(float(s[0, 1]))


@pytest.mark.unit
def test_empty_mask_is_an_explicit_error():
    """掩码段为空必须**显式报错**——静默给 `nan` 会让"这个 agent 没接上"变成无声的坏策略。"""
    net = HierarchicalPolicy()
    z = torch.zeros(1, 4, 128)
    with pytest.raises(ValueError, match="掩码段"):
        net._pool(z, (2, 2), "PA")
    with pytest.raises(ValueError, match="掩码段"):
        net._pool(z, (3, 1), "LA")


@pytest.mark.unit
def test_forward_is_deterministic_and_batch_rows_are_independent():
    tok, seg = _tok_seg()
    net = HierarchicalPolicy()
    a = _fwd(net, tok, seg).z
    b = _fwd(net, tok, seg).z
    assert torch.allclose(a, b), "同输入必须同输出（无 dropout / 无随机）"
    both = torch.cat([tok, tok], dim=0)
    c = _fwd(net, both, seg).z
    assert torch.allclose(c[0], c[1], atol=1e-5), "批内各行独立（无跨 batch 注意力）"
    assert all(np.isfinite(v) for v in c.flatten().tolist())
