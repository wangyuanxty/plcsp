"""重算的**分段 + 梯度检查点**（2026-10-05，mk10 显存闸）。

## 为什么需要它

mk10 全开配置（`route_k=2` + R2 区段 token）会把 8 GB 显存打满——**不是模型大**
（1.08M 参数 = 4.3 MB），是**批的激活大**：批 B = Σn_g（mk10 ≈ 8 链 × 440 决策 ≈ 3520）
× token N（含区段 ≈ 74）× 8 层，每层的 qkv / 注意力矩阵（O(N²)）/ FFN 中间量**都要留到反向**。

**⚠️ 单分段不降峰值**：`torch.cat([f(x₁), f(x₂), …])` 保留的图与一次大前向完全相同。
**只有 `torch.utils.checkpoint` 真降**——它不存段内中间量，反向时按段重算。

## 本文件钉什么

1. **`chunk=0`（默认）走原路**——逐位不变的前提；
2. **分段 vs 整批**：数值 ≤1e-5、**梯度也要对得上**（只对前向会让训练悄悄跑偏）；
3. **`chunk ≥ B` 等于没分段**（不许多算一遍重算）；
4. **检查点不吞梯度**——反向必须真的流到参数上。
"""
from __future__ import annotations

import torch

from plcsp.algo.group_rel import _encoder_forward_batched
from plcsp.algo.policy import PolicyNet
from plcsp.nn.encoder import LayoutEncoder, _SEG_DIMS
from plcsp.nn.features import F_MAX

SEG = (4, 3, 2, 1)          # 小实例：4 机 + 3 作业 + 2 车 + 1 全局 = 10 token


def _tok(B: int) -> torch.Tensor:
    """合法的 `(B, N, F_MAX)`——**每段只填自己那几列**（补零列必须恒 0，否则守卫拒绝）。"""
    N = sum(SEG)
    x = torch.zeros(B, N, F_MAX)
    r = 0
    for n, d in zip(SEG, _SEG_DIMS):
        x[:, r:r + n, :d] = torch.randn(B, n, d)
        r += n
    return x


def _policy() -> PolicyNet:
    torch.manual_seed(0)
    return PolicyNet(enc=LayoutEncoder())


def test_chunk_zero_and_oversized_are_the_original_path(monkeypatch):
    """`chunk=0`（默认）与 `chunk ≥ B` 都必须**恰好调一次** `forward_enc`。

    这是"默认关 ⟹ 逐位不变"的实现层证据：走原路就是走原路，不是"结果碰巧一样"。
    """
    pol = _policy()
    x = _tok(6)
    calls = {"n": 0}
    orig = pol.forward_enc

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(pol, "forward_enc", counting)

    _encoder_forward_batched(pol, x, SEG, None, 0)
    assert calls["n"] == 1, "chunk=0 必须走原路（一次前向）"

    calls["n"] = 0
    _encoder_forward_batched(pol, x, SEG, None, 6)      # chunk == B
    assert calls["n"] == 1, "chunk ≥ B 必须走原路"

    calls["n"] = 0
    _encoder_forward_batched(pol, x, SEG, None, 99)     # chunk > B
    assert calls["n"] == 1, "chunk > B 必须走原路"


def test_chunked_forward_matches_unchunked_within_tolerance():
    """分段 vs 整批：**≤1e-5**（分段改 GEMM 分块 ⟹ 末位会差，与既有批量化同量级）。"""
    pol = _policy()
    x = _tok(10)
    with torch.no_grad():
        whole = _encoder_forward_batched(pol, x, SEG, None, 0)
        for chunk in (1, 3, 4, 7):
            got = _encoder_forward_batched(pol, x, SEG, None, chunk)
            d = (got - whole).abs().max().item()
            assert d < 1e-5, f"chunk={chunk} 与整批差 {d:.3e}（应为末位量级）"


def test_chunked_gradients_match_unchunked():
    """**梯度也要对得上**——只对前向会让训练悄悄跑偏，而且不报错。

    这是检查点最容易坏的地方：它必须让反向真的流过重算的那一段。
    """
    x = _tok(8)

    def grads(chunk: int) -> dict[str, torch.Tensor]:
        pol = _policy()
        out = _encoder_forward_batched(pol, x, SEG, None, chunk)
        out.pow(2).mean().backward()
        return {k: v.detach().clone() for k, v in pol.named_parameters()
                if v.grad is not None}

    g0, g3 = grads(0), grads(3)
    assert set(g0) == set(g3), "两条路径的梯度覆盖面不同（有参数没收到梯度）"
    assert g0, "反向没有产生任何梯度——检查点把梯度吞了"
    for k in g0:
        d = (g0[k] - g3[k]).abs().max().item()
        scale = max(g0[k].abs().max().item(), 1e-12)
        assert d <= 1e-5 * max(scale, 1.0), f"{k} 的梯度差 {d:.3e}（相对 {scale:.3e}）"


def test_chunk_actually_reduces_saved_bytes():
    """**分段 + 检查点必须真的少留东西**——否则这个功能毫无意义。

    ⚠️ **判据是"反传要保留多少字节"，不是"计算图有多少节点"。**
    检查点（`use_reentrant=False`）**照旧建图**——它省的是**存下来的中间张量**
    （段内的不存，反向时按段重算）。数图节点会得出相反的结论
    （分段后每段各建一份图 ⟹ 节点更多），那是**错的判据**。本测试第一版就栽在这。

    用 `saved_tensors_hooks` 直接量前向期间被保存下来的张量总字节数。
    """
    x = _tok(8)

    def saved_bytes(chunk: int) -> int:
        pol = _policy()
        total = [0]

        def pack(t: torch.Tensor) -> torch.Tensor:
            total[0] += t.numel() * t.element_size()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            _encoder_forward_batched(pol, x, SEG, None, chunk)
        return total[0]

    whole, chunked = saved_bytes(0), saved_bytes(1)
    assert chunked < whole, (
        f"分段+检查点没有减少反传要留的字节（整批 {whole} B vs 分段 {chunked} B）——"
        "要么检查点没生效，要么分段只是把前向切开了（那样峰值不降，功能等于没做）"
    )
    # 不是"少一点"——检查点省的是段内**全部**中间量，应当少一个量级
    assert chunked * 4 <= whole, (
        f"省得太少（整批 {whole} B → 分段 {chunked} B）：检查点应当省掉段内全部中间量")
