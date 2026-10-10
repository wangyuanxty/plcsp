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

（`plcsp_v3/q1_flat.py` 已经有守卫；pytest 里不需要，因为 `__main__` 是 pytest 本身。）

══ ⚠️ 并行档与单进程档**不是逐位相同** ══

线程数会改浮点归约的结合次序 ⟹ 末位可能不同。**两臂必须同档**，
**报出的数字要注明跑在哪个档**。（CPU 默认档 ⟹ 一条链 24 线程；本模块 ⟹ 每进程 1 线程。）
"""
from __future__ import annotations

import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple

import torch

from ..env.reward import objective_vector, scalar_reward
from .agents import HierarchicalPolicy
from .replay import run_branch
from .rollout import Decision, NetPolicy
from .train import Spec, base_chain, rules_dict

_W: dict = {}


def _get_net(spec: Spec, with_value: bool):
    """子进程里的网络缓存 —— 按 `with_value` 键。

    ⚠️ **2026-10-10：键由 `(n_chargers, with_value)` 简化为 `with_value`** ——
    充电头删了之后，本网络**不再随布局变宽度**（`HierarchicalPolicy()` 无必需参数），
    故同一个进程里的网络结构只由 `with_value` 决定。`spec` 参数保留（调用点按位传参）。

    ⚠️ **缓存结构、不缓存权重**：参数每步都在变（父进程 `optim.step()`），
    故每次任务都 `load_state_dict`（≈5 ms，相对一条 0.15–0.6 s 的链/段可忽略）。
    """
    key = bool(with_value)
    net = _W.get(key)
    if net is None:
        torch.set_num_threads(1)          # ⚠️ 每进程 1 线程（见模块 docstring）
        net = HierarchicalPolicy(with_value=key)
        _W[key] = net
    return net


def _shutdown(ex: ProcessPoolExecutor) -> None:
    """关池 —— ⚠️ **失败只报不抛**。

    为什么：实测（§110）一次长跑在 `shutdown` 里挂掉，**把刚跑完的一整轮（200 步）拖没了**
    ——死因在栈里，但结果是丢的。关池是**收尾**动作，不该有权否决已经算出来的结果。

    ⚠️ **仍然大声报**（打到 stderr，带异常类型与消息）—— 不静默。调用方若想升级成致命，
    自己接 `OSError`/`BrokenProcessPool`。
    """
    try:
        ex.shutdown(wait=True, cancel_futures=True)
    except Exception as ex_:
        print(f"⚠️ 关池失败（已忽略，不影响已算出的结果）："
              f"{type(ex_ ).__name__}: {ex_}", file=sys.stderr)


class Chain(NamedTuple):
    """一条**采完的链**的产物 —— 顺序档与并行档**产出同一种东西**。"""

    R: float                                  # 标量化回报（越大越好）
    f: tuple[float, float, float]             # (makespan, energy, TWT)，越小越好
    decisions: list[Decision]                 # **这条链自己的**决策链（重算 logp 用）


def _worker(job):
    """子进程入口：一条**链**。`job = (state_dict, spec, with_value, i, seed, dt_events)`。

    ⚠️ 每任务都 `load_state_dict`：参数每步都在变，缓存网络结构、**不缓存权重**。
    """
    state, spec, with_value, i, seed, dt_events, seed_chain = job
    net = _get_net(spec, with_value)
    net.load_state_dict(state)
    p = base_chain(net, spec, greedy=False, seed=seed * 1000 + i,
                   dt_events=dt_events, seed_chain=seed_chain)
    f = objective_vector(p.metrics)                     # type: ignore[attr-defined]
    return scalar_reward(f, spec.w), tuple(f), p.rec.decisions


def _seg_worker(chunk):
    """子进程入口：**一批段**（`chunk = (state, spec, with_value, jobs)`）。

    `jobs = [(prefix, j, rule, seed, greedy, dt_events, seed_chain), ...]` —— 与
    `train.segment_from` 同一件事，只是不在这里算 loss（那要梯度，回父进程做）。

    ⚠️ **`prefix` 是 `Decision` 序列、`rule` 是三元组**（v3：一次决策 = "(规则, 参数)"两样）。

    ⚠️ **按批送**（不是一个任务一条段）：`state_dict` 约 2.4 MB，逐条送的话
    400 条段要传近 1 GB；按 `n_workers` 批送只要 ~40 MB。
    """
    state, spec, with_value, jobs = chunk
    net = _get_net(spec, with_value)
    net.load_state_dict(state)
    out = []
    for prefix, j, rule, seed, greedy, dt_events, seed_chain in jobs:
        inner = NetPolicy(net, spec.inst, spec.layout, spec.ctx, rules=rules_dict(rule),
                          greedy=greedy, seed=seed, dt_events=dt_events,
                          freeze_rules=True, seed_chain=seed_chain)
        _, counters = run_branch(spec.world(), prefix, inner,
                                 switch_after_ma=j, stop_after_ma=j + 2)
        out.append((counters, inner.rec.decisions))
    return out


def _chunk_round_robin(jobs: list, n_chunks: int) -> list[tuple[list[int], list]]:
    """**轮转**分块 → `[(原下标, [job, ...]), ...]`。

    为什么轮转：**段的成本随 `j` 涨**（前缀越长）—— 连续切块会让最后一块独占大半时间。
    带上原下标，回填时按它复位（**不能靠比对 job 内容** —— 两条段可以逐字段相同）。
    """
    buckets: list[list[tuple[int, object]]] = [[] for _ in range(n_chunks)]
    for i, job in enumerate(jobs):
        buckets[i % n_chunks].append((i, job))
    return [([i for i, _ in b], [j for _, j in b]) for b in buckets if b]


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

    def rollout(self, n: int, seed: int, seed_chain: int = 0) -> list[Chain]:
        """采 `n` 条链（并行）。**返回值与顺序档同形**（`Chain` 列表，按下标序）。

        ⚠️ `seed_chain` = **场景**那条随机流（机障 ③ / AGV 故障 ⑨），`n` 条链**共用**它。
        """
        net = self.net
        if next(net.parameters()).type() != "torch.FloatTensor":
            raise ValueError("并行档只跑 CPU —— 先把 `net` 留在 CPU，或改用顺序档 + `--device cuda`")
        state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        with_value = net.value_head is not None
        jobs = [(state, self.spec, with_value, i, seed, self.dt_events, seed_chain)
                for i in range(n)]
        out = list(self.ex.map(_worker, jobs))
        return [Chain(float(R), f, dec) for R, f, dec in out]

    def close(self) -> None:
        _shutdown(self.ex)

    def __enter__(self) -> "ChainPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SegmentPool:
    """进程池：**并行跑"段"**（两层训练器 `train_step` 的 `M+G` 段）。

    ⚠️ **为什么这是最大的一条优化**（§104.5 / §105）：`train_step` 每步要做
    `n_top × (M+G)` 条段 —— **96 条（mk01）到 400 条（mk10）**，而它们**互相独立**
    （每条都是"从 t=0 回放到 `Z_j`、再换目标接续"）。
    实测单段 145–298 ms，**全部串行** ⟹ mk10 一步 119 s。

    ⚠️ **与 `ChainPool` 同一套模式**，只是任务粒度为"段"。
    ⚠️ **每进程 1 线程**，见模块 docstring。
    ⚠️ **`net` 是引用**（整轮复用），权重每步重发。
    """

    def __init__(self, net: HierarchicalPolicy, spec: Spec, *, n_workers: int,
                 dt_events: int = 20):
        if n_workers < 2:
            raise ValueError(f"n_workers={n_workers} 少于 2 ⟹ 请直接用顺序档（`pool=None`）")
        self.net, self.spec = net, spec
        self.n_workers = int(n_workers)
        self.dt_events = int(dt_events)
        self._ctx = mp.get_context("spawn")
        self.ex = ProcessPoolExecutor(max_workers=self.n_workers, mp_context=self._ctx)

    def segments(self, jobs: list[tuple]) -> list[tuple]:
        """`jobs = [(prefix, j, rule, seed, greedy, seed_chain), ...]`
        → `[(StepCounters, decisions), ...]`。

        **返回顺序与 `jobs` 一致**（分块是轮转的，回填时按原下标复位）。
        """
        net = self.net
        if next(net.parameters()).type() != "torch.FloatTensor":
            raise ValueError("并行档只跑 CPU —— 先把 `net` 留在 CPU，或改用顺序档")
        if not jobs:
            return []
        state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        with_value = net.value_head is not None
        full = [(p, j, g, s, gr, self.dt_events, sc) for (p, j, g, s, gr, sc) in jobs]
        chunks = _chunk_round_robin(full, self.n_workers)
        payload = [(state, self.spec, with_value, c) for _idx, c in chunks]
        res = list(self.ex.map(_seg_worker, payload))
        # ⚠️ **按记录的原下标复位**（轮转分块打乱了顺序，直接拼接会错位）
        out: list = [None] * len(jobs)
        for ci, (idxs, _c) in enumerate(chunks):
            for k, orig in enumerate(idxs):
                out[orig] = res[ci][k]
        if any(o is None for o in out):
            raise RuntimeError("段回填错位 —— 并行档的结果对不上任务列表")
        return out

    def close(self) -> None:
        _shutdown(self.ex)

    def __enter__(self) -> "SegmentPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
