"""统一宽度投影 + 类型嵌入的测试（P2 Task 3）。"""
from __future__ import annotations

import torch
import pytest

from plcsp.nn.encoder import LayoutEncoder
from plcsp.nn.features import F_B, F_G, F_M, F_MAX, F_V


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
def test_nonzero_in_padded_columns_is_rejected():
    """补零列非 0 必须**报错**（快速失败），而不是被静默吞掉。

    单 Linear 会照吃补零列的权重（`W[:, 7:]`），故"数据落进补零列"（上游列偏移写错）
    必须当场炸出来；静默清零会把它变成看不见的 bug。补零列本应恒 0，见 spec §5.3.1。
    """
    enc = LayoutEncoder().eval()
    a, seg = _tok(fill=1.0)
    b = a.clone()
    n_m, n_b, _, _ = seg
    b[0, :n_m, F_M:] = 99.0                      # M 段补零列
    b[0, n_m:n_m + n_b, F_B:] = 99.0             # B 段补零列
    b[0, -1, F_G:] = 99.0                        # G 段补零列
    with pytest.raises(ValueError):
        enc(b, seg)


@pytest.mark.unit
def test_seg_slice_matches_declared_widths():
    """`SEG_SLICE` 必须从 0 起、恰好覆盖该段的**声明宽度**——直接守列偏移。

    若某段被收窄一位（如 `M: slice(0, 6)` 而 `F_M=7`），补零列断言照样通过
    （`build_tok` 不往那列写，它本来就是 0），列的**语义**却已错位——本断言把该盲区补上。
    """
    from plcsp.nn.features import SEG_SLICE
    for key, width in (("M", F_M), ("B", F_B), ("V", F_V), ("G", F_G)):
        sl = SEG_SLICE[key]
        assert sl.start == 0, f"{key} 段列区间未从 0 起：{sl}"
        assert sl.stop == width, f"{key} 段列区间 {sl} 与声明宽度 {width} 不符"


@pytest.mark.unit
def test_attention_is_full_not_masked():
    """spec §5.1 定的是**全连接**注意力（轴掩码已砍）——不得再有 block_mask 分支。"""
    import inspect
    src = inspect.getsource(LayoutEncoder.forward)
    assert "block_mask" not in src
