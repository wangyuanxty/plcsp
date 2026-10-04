"""两个头对称性的测试（P2 Task 4）。

2026-10-04 追加**生产路径**的 L 头下标守卫：`roll_chain._act` 曾把 L 的候选下标
当成 S 的下标（车号 = 序列位置），使 L 头读的是 M 段 token——见文末测试的 docstring。
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import pytest

from plcsp.algo.group_rel import decisions_logp, roll_chain
from plcsp.algo.policy import PolicyNet, v_token_index
from plcsp.algo.setup import build_setup
from plcsp.env.des import SimConfig
from plcsp.env.instances import load_mk
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
    # ⚠️ 2026-10-04 分段投影：L 头读 V 段 token ⟹ 梯度落在 `proj[2]`（不是共享的 `embed`）。
    g = pol.enc.proj[2].weight.grad
    assert g is not None and g.abs().sum() > 0, "L 头梯度没到编码器（V 段投影）"


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


class _NoMixEncoder(torch.nn.Module):
    """逐 token **恒等**嵌入（无跨 token 注意力）——只为隔离"L 头索引了哪一行"这一件事。

    真编码器（`LayoutEncoder`）是全连接注意力：扰动 V 段的行会经注意力改变 M 段 token 的
    **上下文**嵌入，于是"读错行的头"也跟着变——判据被掩盖（本仓实测：缺陷在位时，真编码器
    下扰动 V 段，L 的 logp 照样从 -1.0927 变到 -1.0873）。逐 token 恒等嵌入切断这条泄漏后，
    "分数是否依赖被索引的那一行"才是可判的量，且 `_act` / 重算路径仍走生产代码。
    `d_model = F_MAX` 使 `PolicyNet` 按 11 维 token 建头；`forward` 即 `forward_enc` 期望的
    `enc(x, seg) -> (tok, global)` 契约。
    """
    d_model = F_MAX

    def forward(self, x, seg):
        return x, x.mean(dim=1)


@pytest.mark.unit
def test_l_head_reads_v_tokens_on_the_production_path():
    """⚠️ L 头的候选下标在**生产路径**上必须指向 V 段 token——不得把车号当序列位置。

    缺陷形态（2026-10-04 修复）：`roll_chain._act` 里 S/L 共用一个分支 `tok_idx = cand`。
    对 S 正确（机台号 = 序列位置，M 段在最前），对 L 则把**车号**（0..n_agv-1）当序列位置，
    取到 M 段（机台）token。后果：L 头看不到任何车辆特征（电量/载量/速度/趴窝/位置/等待），
    只剩逐候选特征（行驶时长）区分车辆——"派车头"实际在"选机台"。`v_token_index` 于是成了
    **只在测试里用**的函数：手搓 `agv_logits_emb(..., torch.tensor(v_token_index(seg)))` 的测试
    自己把正确下标喂给了头，全绿；生产路径却错。故本测试必须走 `roll_chain`（`_act` 真正
    记录下标）+ `decisions_logp`（训练真正用的重算路径），**不自己提供下标**。

    判据：只扰动 L 决策记录里 **V 段的行**，重算 logp 必须变。缺陷在位时 L 头读 M 段，
    V 段怎么改都与它的分数无关。

    ⚠️ 逐行扰动必须**不同**（此处 1.0..3.0）：同一列加同一个常数只是给所有候选同样的平移，
    `log_softmax` 对其不变——那样修好后的头也测不出变化，判据变成恒真。
    """
    torch.manual_seed(7)
    pol = PolicyNet(enc=_NoMixEncoder())
    inst = load_mk("mk01")
    cfg = SimConfig()
    lay, dm, ctx = build_setup(inst, cfg)
    dec, _met = roll_chain(inst, lay, dm, cfg, pol, seed=0, ctx=ctx)
    l_dec = [d for d in dec if d.kind == "L" and len(d.cand) >= 2]
    assert l_dec, "链路里没有多候选的 L 决策——判据失去意义"
    d = l_dec[0]
    base = decisions_logp([d], pol).detach().clone()
    v_rows = v_token_index(d.seg)
    d2 = copy.deepcopy(d)
    d2.tok[v_rows] += np.linspace(1.0, 3.0, len(v_rows), dtype=np.float32)[:, None]
    after = decisions_logp([d2], pol).detach()
    assert not torch.equal(base, after), (
        f"只扰动 V 段 token 后 L 决策的 logp 不变（{base.tolist()} vs {after.tolist()}）"
        f"——L 头读的不是 V 段（记录下标 {d.tok_idx.tolist()}，V 段下标应为 {v_rows}）")
    # 记录的下标必须逐候选指向**该候选车的 V token**（S 才是机台号 = 序列位置）
    assert np.array_equal(d.tok_idx, np.asarray(v_rows, dtype=np.int64)[list(d.cand)]), (
        f"L 决策记录的 token 下标 {d.tok_idx.tolist()} 未按车号映射到 V 段 {v_rows}")
