"""单条前向的 CUDA 图快路——**只服务在线采样（batch=1）**。

## 为什么需要它

`progress-log.md` §36.3 实测（MK01 `seg=(6,10,3,1)`、batch 1、200 次取中位）：

| | 中位 | vs CPU |
|---|---|---|
| CPU（钉 1 线程） | 2.889 ms | 1.00× |
| GPU **eager** | 4.032 ms | **0.72×（更慢）** |
| **GPU + 本模块** | **0.387 ms** | **7.46×** |
| **GPU + 本模块 + 把结果拉回 CPU** | **0.438 ms** | **6.60×** |

eager 的交叉点落在 **batch 1 与 8 之间**（batch 8 起 GPU 就反超）⟹ batch-1 上 GPU 亏的是
**kernel 启动开销**（约 80 个 kernel × 约 40 µs ≈ 3.2 ms，与实测 3.72 ms 同量级），
**不是算力**。把整条前向录成一张图，启动开销只剩一次重放，GPU 的真实算力才显出来。

## 为什么只对 batch=1

- **B>1 用不着**：重算路径是 batch=数百到数千，eager 在那一档本来就快（§36.1）。
- **B>1 用不了**：重算**要梯度**，而 CUDA 图**不支持 autograd**。
  故本模块的前向一律在 `torch.no_grad()` 下录制与重放。

## ⚠️ 三条硬约束

1. **形状固定**。图与输入形状绑定。换形状要另建一张图（本仓里形状由 `seg` 决定，
   而 `seg` 是实例级常量 ⟹ **每实例一张图**）。
2. **输出是图自己的缓冲，会被下一次 `replay` 原地覆盖**。调用方必须当场用完，
   **不得跨 replay 持有**。
3. **参数被重建（`.to()` / `.cuda()` / `.float()`）后图失效**——图里存的是当时那些张量的
   地址。**优化器的原地更新（Adam）不影响**（地址不变）。`LayoutEncoder._apply` 负责清缓存。

## ⚠️ 捕获期间**不允许任何设备同步**

图里出现 `.item()` / `bool()` / `.cpu()` 会当场抛错。故调用方必须把守卫/检查放在
**捕获之外**——本仓的补零列守卫（`_require_zero_padding`）就落在 `forward` 里、进图之前，
**守卫本身没有削弱**。
"""
from __future__ import annotations

from typing import Callable

import torch


class CudaGraphForward:
    """把一个形状固定的 `encode(x)` 录成 CUDA 图，之后按 `replay(x)` 重放。

    `encode` 只收一个张量参数（`seg` 之类的常量由调用方闭包绑住）。
    """

    def __init__(self, encode: Callable[[torch.Tensor], object], x: torch.Tensor,
                 *, warmup: int = 3):
        if not x.is_cuda:
            raise ValueError(f"CudaGraphForward 只接受 CUDA 张量，收到 device={x.device}")
        if warmup < 1:
            raise ValueError(f"warmup 至少要 1 次（torch 要求捕获前先跑热），收到 {warmup}")
        self._encode = encode
        # 静态输入：地址固定，重放前把新输入 `copy_` 进来
        self._static_in = x.detach().clone()
        self.n_replay = 0
        # torch 要求：捕获前在**旁路流**上预热，否则捕获会挂在未初始化的 cuBLAS 句柄上。
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(warmup):
                encode(self._static_in)
        torch.cuda.current_stream().wait_stream(side)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph), torch.no_grad():
            # ⚠️ 这里录进去的**不是**这次算出的值，而是"以后每次重放要执行的那串 kernel"。
            self._out = encode(self._static_in)

    @property
    def shape(self) -> torch.Size:
        return self._static_in.shape

    def replay(self, x: torch.Tensor):
        """把 `x` 拷进静态输入、重放图，返回输出。

        ⚠️ 返回的张量**每调用一次就被覆盖一次**（见模块 docstring 第 2 条）。
        """
        if x.shape != self._static_in.shape:
            raise ValueError(
                f"图与形状绑定：录制时 {tuple(self._static_in.shape)}，收到 {tuple(x.shape)}。"
                "换形状要另建一张图，不得静默复用。")
        if x.device != self._static_in.device:
            raise ValueError(
                f"图与设备绑定：录制时 {self._static_in.device}，收到 {x.device}。")
        self._static_in.copy_(x)
        self._graph.replay()
        self.n_replay += 1
        return self._out
