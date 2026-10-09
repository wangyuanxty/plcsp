# -*- coding: utf-8 -*-
"""**并行跑链**（把 v1 `ChainWorkerPool` 的账还回来）。

══ 为什么需要 ══

一趟训练要采 `n` 条链（设计 §12.2 的 `G=8`）。**这 8 条链互相独立** ——
每条只用自己的 `torch.Generator`（`NetPolicy.gen`）与自己的仿真世界。
**但 v2 一直是单进程顺序跑**，32 个逻辑核闲着 31 个。

**v1 有 `ChainWorkerPool`（`algo/chain_pool.py`）—— v2 重写时丢了。**

══ ⚠️ 每进程 **1 线程**（这条有实测支撑）══

暖机后同一条链、5 次取中位（MK01 · `dt_events=20` · 249 决策）：

| 线程数 | 1 | 4 | 8 | 16 | 24 |
|---|---|---|---|---|---|
| 单链 | **0.60 s** | 0.64 | 0.81 | 1.08 | **1.61 s** |

**线程越多越慢** —— token 只有 ~22 行、128 维，**线程调度开销盖过了并行收益**。
⟹ **并行档每进程 1 线程**：既是最快的单链，又不能与"多进程"叠加（否则 8×8 线程互抢）。

══ ⚠️ 调用方**必须**加 `if __name__ == "__main__":` 守卫 ══

Windows 用 `spawn` ⟹ **子进程会重新 import `__main__`**。裸脚本（顶层直接建池）会
**无限递归建池**，报出来的是 `BrokenProcessPool`，看不出根因。**实测踩过。**

```python
if __name__ == "__main__":          # ← 缺了这行，Windows 上必炸
    with ChainPool(net, spec, n_workers=8, dt_events=20) as pool:
        ...
```

（`plcsp_v2/q1_flat.py` 已经有守卫；pytest 里不需要，因为 `__main__` 是 pytest 本身。）

══ ⚠️ 并行档与单进程档**不是逐位相同** ══

线程数会改浮点归约的结合次序 ⟹ 末位可能不同。**两臂必须同档**，
**报出的数字要注明跑在哪个档**。（CPU 默认档 ⟹ 一条链 24 线程；本模块 ⟹ 每进程 1 线程。）
"""
from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple

import torch

from ..env.reward import objective_vector, scalar_reward
from .agents import HierarchicalPolicy
from .rollout import Decision
from .train import Spec, base_chain

_W: dict = {}


class Chain(NamedTuple):
    """一条**采完的链**的产物 —— 顺序档与并行档**产出同一种东西**。"""

    R: float                                  # 标量化回报（越大越好）
    f: tuple[float, float, float]             # (makespan, energy, TWT)，越小越好
    decisions: list[Decision]                 # **这条链自己的**决策链（重算 logp 用）


def _worker(job):
    """子进程入口：一条链。`job = (state_dict, spec, with_value, i, seed, dt_events)`。

    ⚠️ **`set_num_threads(1)` 放在这里**（不是父进程）—— 见模块 docstring。
    ⚠️ 每任务都 `load_state_dict`：参数每步都在变，缓存网络结构、**不缓存权重**。
    （600k 参数的加载约 5 ms，相对 0.6 s 的链可忽略。）
    """
    state, spec, with_value, i, seed, dt_events = job
    net = _W.get("net")
    if net is None:
        torch.set_num_threads(1)
        net = HierarchicalPolicy(n_chargers=len(spec.layout.chargers), with_value=with_value)
        _W["net"] = net
    net.load_state_dict(state)
    p = base_chain(net, spec, greedy=False, seed=seed * 1000 + i, dt_events=dt_events)
    f = objective_vector(p.metrics)                     # type: ignore[attr-defined]
    return scalar_reward(f, spec.w), tuple(f), p.rec.decisions


class ChainPool:
    """进程池：`rollout(n, seed)` 一次跑完 `n` 条链。

    用法（`flat_step` 的 `pool=` 参数就是它）：

    ```python
    with ChainPool(net, spec, n_workers=8, dt_events=20) as pool:
        ...  # 每一步把 pool 传给 flat_step
    ```

    ⚠️ **池是"每步"还是"整轮"**：本类**整轮复用**（进程启动一次），
    但**权重每步都从父进程重发**（`net` 变了）。故 `net` 是**引用**，不是副本。
    """

    def __init__(self, net: HierarchicalPolicy, spec: Spec, *, n_workers: int,
                 dt_events: int = 20):
        if n_workers < 2:
            raise ValueError(f"n_workers={n_workers} 少于 2 ⟹ 请直接用顺序档（`flat_rollout`）")
        self.net, self.spec = net, spec
        self.n_workers = int(n_workers)
        self.dt_events = int(dt_events)
        # ⚠️ `spawn`（Windows 默认）⟹ 子进程重新 import 一遍；**不要在子进程里建 CUDA 上下文**
        # （一个 GPU 上开 8 个上下文只会互抢）。并行档**只跑 CPU**。
        self._ctx = mp.get_context("spawn")
        self.ex = ProcessPoolExecutor(max_workers=self.n_workers, mp_context=self._ctx)

    def rollout(self, n: int, seed: int) -> list[Chain]:
        """采 `n` 条链（并行）。**返回值与顺序档同形**（`Chain` 列表，按下标序）。"""
        net = self.net
        if next(net.parameters()).type() != "torch.FloatTensor":
            raise ValueError("并行档只跑 CPU —— 先把 `net` 留在 CPU，或改用顺序档 + `--device cuda`")
        state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        with_value = net.value_head is not None
        jobs = [(state, self.spec, with_value, i, seed, self.dt_events) for i in range(n)]
        out = list(self.ex.map(_worker, jobs))
        return [Chain(float(R), f, dec) for R, f, dec in out]

    def close(self) -> None:
        self.ex.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "ChainPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
