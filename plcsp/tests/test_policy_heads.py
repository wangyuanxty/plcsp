"""两个头对称性的测试（P2 Task 4）。"""
from __future__ import annotations

import torch
import pytest

from plcsp.algo.policy import PolicyNet, v_token_index
from plcsp.nn.encoder import LayoutEncoder
from plcsp.nn.features import F_MAX, SEG_SLICE

# L 头任务特征维数（起送机台 / 目标机台 / 工序序号 / 该机是否需换型）= `PolicyNet.n_feat_task`。
# ⚠️ 是 **4** 不是 3：3 维版本（frm, to, oi）是旧扁平路径 `_l_feat` 的形状，已废。
F_TASK = 4

# 逐候选特征维数 = `PolicyNet.n_feat_cand`（S 头：换型代价；L 头：该车到取货点的行驶时长）。
F_CAND = 1


def _cand_feat(n_cand):
    """逐候选特征 `(1, n_cand, F_CAND)`——全 0（打分是否变化由各测试自己决定）。"""
    return torch.zeros(1, n_cand, F_CAND)


def _enc_inputs(n_m=6, n_b=10, n_v=3):
    """按补齐规则造一张 (1, N, F_MAX)：各段只填自己的有效列。

    ⚠️ 不能图省事用 `torch.randn(1, N, F_MAX)` 直接造——补零列非 0 会被编码器
    `_require_zero_padding` 当场拒绝（spec §5.3.1，与 `test_encoder_segments.py` 同一约束）。
    """
    t = torch.zeros(1, n_m + n_b + n_v + 1, F_MAX)
    for start, n, key in ((0, n_m, "M"), (n_m, n_b, "B"), (n_m + n_b, n_v, "V"),
                          (n_m + n_b + n_v, 1, "G")):
        t[0, start:start + n, SEG_SLICE[key]] = torch.randn(n, SEG_SLICE[key].stop)
    return t, (n_m, n_b, n_v, 1)


@pytest.mark.unit
def test_both_heads_read_the_same_encoder():
    """⚠️ Review Focus #5：L 头必须走编码器——这是'联合链'的前提。"""
    pol = PolicyNet(enc=LayoutEncoder())
    tok_feat, seg = _enc_inputs()
    tok, _ = pol.forward_enc(tok_feat, seg)
    idx = v_token_index(seg)
    logits = pol.agv_logits_emb(tok, torch.zeros(1, 1, F_TASK), _cand_feat(len(idx)),
                                torch.tensor(idx))
    assert logits.shape == (1, 1, 3)


@pytest.mark.unit
def test_l_head_gradient_reaches_encoder():
    """⚠️ Review Focus #5：从 L 头反传，编码器参数必须有非零梯度（否则'联合'是假的）。"""
    pol = PolicyNet(enc=LayoutEncoder())
    tok_feat, seg = _enc_inputs()
    tok, _ = pol.forward_enc(tok_feat, seg)
    idx = v_token_index(seg)
    out = pol.agv_logits_emb(tok, torch.zeros(1, 1, F_TASK), _cand_feat(len(idx)),
                             torch.tensor(idx)).sum()
    out.backward()
    g = pol.enc.embed.weight.grad            # 编码器输入投影
    assert g is not None and g.abs().sum() > 0, "L 头梯度没到编码器"


@pytest.mark.unit
def test_candidate_features_change_the_score():
    """⚠️ F1：`feat_cand` 槽必须是**活的**——只改它，两个头的打分都必须变（否则该槽是死的）。

    候选特征是本仓"消灭死维"纪律下的关键通路：⑤ 换型代价（S 头，`(机台,作业)` 的交互量）与
    "派最近的车"（L 头）都只能走这里（spec §5.3.1②）；恒零的死维等于该通路不存在。
    """
    pol = PolicyNet(enc=LayoutEncoder())
    tok_feat, seg = _enc_inputs()
    tok, _ = pol.forward_enc(tok_feat, seg)
    idx = torch.tensor(v_token_index(seg))
    for name, head, dec, cand in (("S", pol.mach_logits_emb, 3, torch.tensor([0, 2, 5])),
                                  ("L", pol.agv_logits_emb, F_TASK, idx)):
        zeros = head(tok, torch.zeros(1, 1, dec), _cand_feat(len(cand)), cand)
        ones = head(tok, torch.zeros(1, 1, dec), torch.ones(1, len(cand), F_CAND), cand)
        assert not torch.allclose(zeros, ones), f"{name} 头的 feat_cand 改了打分却不变——该槽是死的"


@pytest.mark.unit
def test_v_token_index_matches_seg():
    assert v_token_index((6, 10, 3, 1)) == [16, 17, 18]
    assert v_token_index((15, 20, 4, 1)) == [35, 36, 37, 38]
