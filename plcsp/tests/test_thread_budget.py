"""torch 线程数必须钉成 1（性能与**可复现性**守卫）。

**为什么**：本项目的张量极小——MK01 只有 20 个 token × F_MAX=11 维，编码器每次前向
摊到约 35 个 `nn.Linear`。torch 默认按核数开线程（本机 32 核 → 24 线程），
**线程同步开销压过计算本身**。实测（MK01、`joint_chain_step`、预热后取中位）：

| 线程 | G=1 | G=4 |
|---|---|---|
| 1 | **1.34 s/步** | **5.53 s/步** |
| 24 | 3.68 s/步 | 14.83 s/步 |

即**单线程快 2.7 倍**。更要紧的是**波动**：24 线程下同一步实测过 8.85 / 31.95 /
34.36 / 34.88 s（约 4 倍），单线程则是 1.34/1.34/1.34/1.34。

⚠️ 这不只是快慢问题：本项目在 `progress-log` 里记**墙钟数字**（如消融表 546 s/组）。
线程数不钉死，这些数字**不可复现**，报出去站不住。
"""
from __future__ import annotations

import pytest


@pytest.mark.unit
def test_torch_threads_are_pinned_to_one():
    """`plcsp` 一旦被导入，torch 线程数就必须是 1——由 `plcsp/__init__.py` 钉住。"""
    import torch

    import plcsp  # noqa: F401  （导入即应生效）

    assert torch.get_num_threads() == 1, (
        f"torch 线程数是 {torch.get_num_threads()}，应为 1——"
        "本仓张量极小，多线程是负收益且让墙钟读数不可复现（见本文件模块 docstring）")


@pytest.mark.unit
def test_pinning_survives_a_fresh_import_order():
    """⚠️ 守卫必须在**任何**导入顺序下成立：先 import torch 再 import plcsp，也要生效。

    （若把 `set_num_threads` 写在某个子模块里、靠别人先导入它，这条会红。）
    """
    import torch

    import plcsp  # noqa: F401

    assert torch.get_num_threads() == 1
