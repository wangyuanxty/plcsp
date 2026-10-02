"""统一宽度投影 + 类型嵌入的测试（P2 Task 3）。"""
from __future__ import annotations

import torch
import pytest

from plcsp.nn.encoder import LayoutEncoder
from plcsp.nn.features import F_B, F_G, F_M, F_MAX


def _tok(n_m=6, n_b=10, n_v=3, fill=1.0):
    """按补齐规则造一张 (1, N, F_MAX)：各段只填自己的有效列，其余为 0。"""
    from plcsp.nn.features import F_MAX, SEG_SLICE
    t = torch.zeros(1, n_m + n_b + n_v + 1, F_MAX)
    for start, n, key in ((0, n_m, "M"), (n_m, n_b, "B"), (n_m + n_b, n_v, "V"),
                          (n_m + n_b + n_v, 1, "G")):
        t[0, start:start + n, SEG_SLICE[key]] = fill
    return t, (n_m, n_b, n_v, 1)


@pytest.mark.unit
def test_forward_shapes():
    enc = LayoutEncoder()
    tok, ctx = enc(*_tok())
    assert tok.shape == (1, 6 + 10 + 3 + 1, enc.d_model)
    assert ctx.shape == (1, enc.d_model)


@pytest.mark.unit
def test_single_linear_projection_not_per_segment():
    """输入是**单张**张量、**单个** Linear（spec §5.3.1 的统一宽度方案）。"""
    enc = LayoutEncoder()
    assert hasattr(enc, "embed") and isinstance(enc.embed, torch.nn.Linear)
    assert enc.embed.in_features == F_MAX
    assert not hasattr(enc, "proj"), "仍存在分段投影——与统一宽度方案不符"


@pytest.mark.unit
def test_type_embedding_distinguishes_segments():
    """⚠️ Review Focus #2：三类 token 即使特征值相同，输出也必须不同——否则网络分不出类型。

    旧版正是这样坏的：单一 Linear + 无类型嵌入，`mask='full'` 下 seg 完全不被使用。
    """
    enc = LayoutEncoder().eval()
    tok, _ = enc(*_tok(fill=0.0))
    # 机台段首 token 与车辆段首 token 的嵌入必须不同
    assert not torch.allclose(tok[0, 0], tok[0, 16], atol=1e-6)
    assert not torch.allclose(tok[0, 0], tok[0, 19], atol=1e-6)


@pytest.mark.unit
def test_padded_columns_do_not_affect_embedding():
    """补零列不得影响嵌入——把它们改成任意值，输出必须**逐位不变**。

    这是"统一宽度 + 单 Linear"方案的正确性前提：补零列乘的是权重列，
    若某处列偏移写错、把有效列当成了补零列，嵌入就会变。本测试把该风险钉死。
    """
    enc = LayoutEncoder().eval()
    a, seg = _tok(fill=1.0)
    b = a.clone()
    n_m, n_b, _, _ = seg
    b[0, :n_m, F_M:] = 99.0                      # M 段补零列
    b[0, n_m:n_m + n_b, F_B:] = 99.0             # B 段补零列
    b[0, -1, F_G:] = 99.0                        # G 段补零列
    with torch.no_grad():
        assert torch.allclose(enc(a, seg)[0], enc(b, seg)[0], atol=1e-6)


@pytest.mark.unit
def test_attention_is_full_not_masked():
    """spec §5.1 定的是**全连接**注意力（轴掩码已砍）——不得再有 block_mask 分支。"""
    import inspect
    src = inspect.getsource(LayoutEncoder.forward)
    assert "block_mask" not in src
