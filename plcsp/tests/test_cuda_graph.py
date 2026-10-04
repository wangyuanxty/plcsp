"""`CudaGraphForward` 的正确性与边界。

⚠️ **本机默认测试环境是 CPU-only torch**（`D:/anaconda/python.exe`），故这一整个文件在
默认门禁下**整片 skip**。要真跑它，用带 CUDA 的解释器：

    D:/anaconda/envs/py312/python.exe -m pytest plcsp/tests/test_cuda_graph.py -v

这是**如实记账**：图快路的正确性**不在默认门禁的覆盖范围内**。故这里把边界钉死，
并让"跳过"是显式的，而不是让人以为它绿了。
"""
from __future__ import annotations

import pytest
import torch

from plcsp.nn.cuda_graph import CudaGraphForward

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                               reason="需要 CUDA（默认环境是 CPU-only torch）")


class _Enc(torch.nn.Module):
    """最小前向：一个 Linear。用来把"图算得对不对"与编码器解耦。"""

    def __init__(self, din: int = 3, dout: int = 4):
        super().__init__()
        self.lin = torch.nn.Linear(din, dout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def test_cpu_tensor_is_rejected():
    """CPU 张量必须当场拒——静默退回 eager 会让"图没接上"变成看不见的性能回归。"""
    enc = _Enc()
    x = torch.randn(1, 5, 3)
    with pytest.raises(ValueError, match="CUDA"):
        CudaGraphForward(lambda t: enc(t), x)


def test_zero_warmup_is_rejected():
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")
    enc = _Enc().cuda()
    x = torch.randn(1, 5, 3, device="cuda")
    with pytest.raises(ValueError, match="warmup"):
        CudaGraphForward(lambda t: enc(t), x, warmup=0)


@cuda_only
def test_replay_matches_eager_on_the_same_input():
    enc = _Enc().cuda()
    x = torch.randn(1, 5, 3, device="cuda")
    g = CudaGraphForward(lambda t: enc(t), x.clone())
    got = g.replay(x).clone()
    with torch.no_grad():
        want = enc(x)
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()


@cuda_only
def test_replay_follows_a_changed_input():
    """⚠️ **本文件最重要的一条**：换输入必须换输出。

    图重放最容易犯的错是"静态输入没拷进去，重放的是旧值"——那会让策略拿旧状态做决策，
    **不报错、只是错**。逐位比 eager 抓不住它（第一条测试就过了），必须比**两次不同输入**。
    """
    enc = _Enc().cuda()
    x1 = torch.randn(1, 5, 3, device="cuda")
    x2 = x1 + 1.0
    g = CudaGraphForward(lambda t: enc(t), x1.clone())
    r1 = g.replay(x1).clone()          # ⚠️ 当场 clone：缓冲会被下一次重放原地覆盖
    r2 = g.replay(x2).clone()
    with torch.no_grad():
        w1, w2 = enc(x1), enc(x2)
    assert torch.allclose(r1, w1, atol=1e-6)
    assert torch.allclose(r2, w2, atol=1e-6)
    assert not torch.allclose(r1, r2), "输入变了输出没变 ⟹ 重放吃的是静态缓冲的旧值"


@cuda_only
def test_output_buffer_is_overwritten_by_the_next_replay():
    """把"输出是图自己的缓冲"这条**钉成可执行的边界**，而不是只写在 docstring 里。"""
    enc = _Enc().cuda()
    x1 = torch.randn(1, 5, 3, device="cuda")
    x2 = x1 + 1.0
    g = CudaGraphForward(lambda t: enc(t), x1.clone())
    r1 = g.replay(x1)                  # ⚠️ 故意不 clone
    before = r1.clone()
    g.replay(x2)
    assert not torch.allclose(r1, before), \
        "上一次的返回值应被原地覆盖——若这条不成立，docstring 第 2 条就要改写"


@cuda_only
def test_shape_and_device_mismatch_are_rejected():
    enc = _Enc().cuda()
    x = torch.randn(1, 5, 3, device="cuda")
    g = CudaGraphForward(lambda t: enc(t), x.clone())
    with pytest.raises(ValueError, match="形状"):
        g.replay(torch.randn(2, 5, 3, device="cuda"))
    with pytest.raises(ValueError, match="设备"):
        g.replay(torch.randn(1, 5, 3))


@cuda_only
def test_graph_goes_stale_after_parameter_reassignment():
    """⚠️ **已知边界，如实钉死**：图存的是张量**地址**。

    参数被**原地更新**（Adam 的 `param.add_`）不影响图——地址没变。
    但参数被**重建**（`.to()`、赋一个新 `Parameter`）后，图**仍算旧权重且不报错**。
    这不是本模块能自动发现的，故在 `LayoutEncoder._apply` 里清缓存。
    本测试把"边界存在"这件事固定下来，免得日后有人以为图会自动跟随。
    """
    enc = _Enc().cuda()
    x = torch.randn(1, 5, 3, device="cuda")
    g = CudaGraphForward(lambda t: enc(t), x.clone())
    with torch.no_grad():
        enc.lin.weight.mul_(2.0)                 # 原地改 ⟹ 图应跟随
        assert torch.allclose(g.replay(x).clone(), enc(x), atol=1e-6)
        old = enc.lin
        enc.lin = torch.nn.Linear(3, 4).cuda()   # 重建 ⟹ 图**不**跟随（已知边界）
        assert not torch.allclose(g.replay(x).clone(), enc(x), atol=1e-6)
        enc.lin = old
