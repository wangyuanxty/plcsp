# -*- coding: utf-8 -*-
"""v3 的头模块 —— `RuleSelectHeads`（上层）· `PairScoreHead` / `SingleScoreHead`（下层）。

设计依据：`docs/v3/topk-rerank-spec.md` §3。

⚠️ **2026-10-18：`RuleParamHeads` 已删**（top-K 架构把连续超参数 `λ` / `A` 退出策略路径）。
原文件里它的 6 条测试一并删除；接替它们的是三个打分头的形状 / 规则条件 / 上下文 / 批独立。
"""
from __future__ import annotations

import torch
import pytest

from plcsp_v3.hrl.agents import (PairScoreHead, RuleSelectHeads,
                                 SingleScoreHead)

D = 128
B, K = 3, 5


# ══════════════════ 上层：三个独立分类头 ══════════════════

@pytest.mark.unit
def test_select_heads_shape_and_independence():
    """三个头**各自**出 logits（形状 = 各组的类数），**不是一个联合分类**。"""
    sel = RuleSelectHeads(D, {"PA": 12, "LA": 3, "RA": 3})
    out = sel(torch.randn(5, D))
    assert set(out) == {"PA", "LA", "RA"}
    assert tuple(out["PA"].shape) == (5, 12)
    assert tuple(out["LA"].shape) == (5, 3)
    assert tuple(out["RA"].shape) == (5, 3)


@pytest.mark.unit
def test_select_heads_empty_is_rejected():
    with pytest.raises(ValueError, match="不能是空的"):
        RuleSelectHeads(D, {})


# ══════════════════ 下层：每个 agent 一个打分头 ══════════════════

@pytest.mark.unit
def test_pair_score_head_shape():
    """PA 打分头：两个实体向量 + 候选特征 + 规则 embedding → 每候选 1 个 logit。"""
    head = PairScoreHead(D, cand_dim=9, n_rules=12)
    z = torch.randn(B, K, D)
    feats = torch.randn(B, K, 9)
    rule = torch.tensor([0, 7, 11])
    out = head(z, z, feats, rule)
    assert tuple(out.shape) == (B, K)


@pytest.mark.unit
def test_single_score_head_shapes():
    """LA：每候选 1 个 logit；RA：2 个动作 logit（不经过候选维的 squeeze）。"""
    la = SingleScoreHead(D, cand_dim=4, n_rules=3, ctx_dim=4)
    z = torch.randn(B, K, D)
    feats = torch.randn(B, K, 4)
    ctx = torch.randn(B, 4)
    rule = torch.tensor([0, 1, 2])
    assert tuple(la(z, feats, rule, ctx).shape) == (B, K)
    ra = SingleScoreHead(D, cand_dim=5, n_rules=3, n_out=2)
    feats1 = torch.randn(B, 1, 5)
    assert tuple(ra(z[:, :1], feats1, rule).shape) == (B, 2)


@pytest.mark.unit
def test_rule_embedding_changes_the_score():
    """⚠️ **规则身份必须进打分头**：同一候选集、不同规则号 ⟹ 不同 logits。

    这是 top-K 架构里"MA 不可替代"的直接证据：候选集由规则决定，打分头也按规则条件化。
    """
    head = PairScoreHead(D, cand_dim=9, n_rules=12)
    head.eval()
    z = torch.randn(1, K, D)
    feats = torch.randn(1, K, 9)
    with torch.no_grad():
        a = head(z, z, feats, torch.tensor([0]))
        b = head(z, z, feats, torch.tensor([1]))
    assert not torch.allclose(a, b), "换规则号没改变打分 ⟹ 规则 embedding 没接上"


@pytest.mark.unit
def test_single_score_head_ctx_is_required_when_declared():
    """声明了 `ctx_dim` 就**必须**给上下文 —— 静默补 0 会让"头没接上"变成无声的坏策略。"""
    la = SingleScoreHead(D, cand_dim=4, n_rules=3, ctx_dim=4)
    z = torch.randn(B, K, D)
    feats = torch.randn(B, K, 4)
    rule = torch.tensor([0, 1, 2])
    with pytest.raises(ValueError, match="逐决策上下文"):
        la(z, feats, rule)
    assert tuple(la(z, feats, rule, torch.randn(B, 4)).shape) == (B, K)


@pytest.mark.unit
def test_score_heads_batch_independent():
    """**批内各行互相独立**（与编码器同一纪律：没有跨 batch 的运算）。"""
    head = PairScoreHead(D, cand_dim=9, n_rules=12)
    head.eval()
    z = torch.randn(4, K, D)
    feats = torch.randn(4, K, 9)
    rule = torch.tensor([0, 1, 2, 3])
    with torch.no_grad():
        full = head(z, z, feats, rule)
        one = head(z[:2], z[:2], feats[:2], rule[:2])
    assert torch.allclose(full[:2], one, atol=1e-6), "批内不独立"
