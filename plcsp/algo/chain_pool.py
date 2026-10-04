"""链级多进程池（2026-10-04 并行批次）：把 G 条独立链铺到 worker 进程上跑。

**为什么**：MK01、G=8、route_k=1 实测（`docs/progress-log.md` §36–§38）——CUDA 档整步
1.78 s 里约 **1.4 s（79%）是每条链各自的在线部分**：SimPy 事件循环 660 ms + 在线打分头
330 ms + 采样前向 300 ms + build_tok 80 ms + snapshot 53 ms。这些量与其它链无关
（第 g 条链的扰动种子 = `seed*SEED_STRIDE + g`），却挤在 `joint_chain_step` 的串行循环里。
只有**重算前向、反向、优化器步**是全局的，必须留在主进程。

**分工**
- **worker 进程**（`_worker_run_chain`）：跑 `roll_chain` 的整段 episode——仿真 + 在线策略
  前向 + 决策记录；返回决策列表与该链的 metrics。
- **主进程**：`_advantages` + 重算（`chains_logp` / `all_decisions_logp`）+ 反向 + 优化器步。

**四条硬口径**
1. **worker 的设备可选**（`worker_device`，默认 `"cpu"`）。
   - `"cpu"`（默认，逐位等于上一批）：worker 用一个 **CPU 策略镜像**——`share_memory_()`
     之后的共享内存模块；主进程每步用 `mirror.load_state_dict(policy.state_dict())`
     **原地**刷新（`copy_`，不换 storage），worker 直接读同一块内存 ⟹ **每步零参数 IPC**
     （不是每步复制 4.3 MB × G）。
   - `"cuda"`（新增可选档）：worker **在自己的进程里**建 CUDA 上下文与 GPU 副本（spawn 的
     子进程不继承父进程的 CUDA 上下文），每步把共享镜像 H2D 刷进 GPU 副本；在线前向走
     `LayoutEncoder` 的 CUDA 图快路（batch=1、无梯度）。采样流随之用 CUDA generator
     ——**与 CPU 档不是同一条流，数值会变**（如实记录；默认档不受影响）。
     worker 建上下文 / 显存不足 / 建图失败一律**显式报错**，绝不悄悄退回 CPU worker。
2. **不用 `fork`。** 用 `torch.multiprocessing` 的 **spawn** 上下文（Windows 默认即 spawn；
   torch 线程 + fork 不安全）。
3. **池常驻复用。** 每步只提交任务，不新建进程（spawn 8 个进程约 1–2 s，比省下的还多）。
   清理路径 = `close()`（幂等）/ `terminate()`；`with` 语句与 `run_training` 的 `finally`
   都会走到。
4. **不静默退回串行。** worker 起不来 / 数量不对 / 任务抛异常 / 超时 → `RuntimeError`，
   且池被显式终止，绝不改跑串行——那会让"并行没接上"变成看不见的性能回归。

⚠️ **采样流的口径变了**（本批的数值代价，**第九次读数作废**）：串行档 G 条链**共用一条**
`torch.Generator`、按链序消费；并行档每条链必须有自己的流（否则消费次序不确定），种子取
`torch.Generator().manual_seed(seed*SEED_STRIDE + g)`——逐链独立且确定，但**与串行档同
seed 的数值不同**。`joint_chain_step(parallel=False)`（默认）走原串行路径，**逐位不变**。

⚠️ **`worker_device="cuda"` 时另有一层设备差**：主进程在 CPU 时，并行档的在线前向浮点路径
与采样流设备仍与串行 CPU 档不同（CPU 与 CUDA 的 generator 是两条流，§36.9）。跨档读数
不可逐位互比；**同档同 seed 同 worker 数逐位可复现**（由测试钉住）。
"""
from __future__ import annotations

import copy
import os
import pickle
import time
import traceback
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as tmp

from ..env.constraints import ConstraintConfig
from ..env.des import SimConfig
from ..env.instances import Instance
from ..env.layout import Layout
from ..nn.features import NormContext
from .group_rel import SEED_STRIDE, Decision, roll_chain
from .policy import PolicyNet

# worker 进程的全局状态——由 `_worker_init` 写入。spawn 的子进程不继承父进程内存，
# 这里存的就是**本进程**从共享内存映射进来的策略镜像（CPU 档）或它在本进程 GPU 上的副本
# （CUDA 档；CUDA 上下文必须在 worker 进程内建，见模块 docstring 硬口径 1）。
_WORKER: dict[str, Any] = {}

# 允许的 worker 设备——**白名单**：写错名字当场报错，不静默按 CPU 跑。
_WORKER_DEVICES = ("cpu", "cuda")

# 一个任务 = (链号, 该链种子, 环境八件套)。环境逐任务传：口径与调用方**当次**传进来的
# 对象完全一致（不存在"池建好时的旧环境"静默错配），代价只是每步 G 份的小对象 pickle。
_Task = tuple[Any, ...]


def _worker_fail(exc: BaseException, failed: Any, errors: Any) -> None:
    """记录一次 worker 初始化失败——父进程据此**显式报错**（绝不静默退回 CPU worker）。

    先把消息塞进共享队列再让异常冒出去；队列本身出错也不得掩盖原始异常，故吞掉。
    """
    msg = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    if errors is not None:
        try:
            errors.put(msg)
        except Exception:       # noqa: BLE001 —— 记录失败不得掩盖原始异常
            pass
    if failed is not None:
        with failed.get_lock():
            failed.value += 1


def _worker_init(mirror: PolicyNet, started: Any = None, failed: Any = None,
                 errors: Any = None, worker_device: str = "cpu",
                 version: Any = None, slots: Any = None,
                 sync_times: Any = None) -> None:
    """worker 进程入口：绑定策略（CPU 镜像或本进程的 GPU 副本），并把 torch 线程数钉成 1。

    ⚠️ 线程数**必须**在这里再钉一次：`torch.set_num_threads` 是进程级设置，spawn 的子进程
    不继承父进程的它（`plcsp/__init__.py` 的包级钉法在子进程 import 时也会生效，这里显式
    重申，免得将来有人把 `import plcsp` 换成子模块直导而静默失去钉法）。

    ⚠️ **CUDA 上下文在本函数内建**（硬要求）：spawn 的子进程不继承父进程的 CUDA 上下文，
    故 `.to("cuda")` 与后续的建图都必须发生在 worker 进程内——父进程只传 **CPU** 共享镜像。
    初始化失败（无 CUDA / 建上下文失败 / 显存不足）→ 计入 `failed` 并把 traceback 塞进
    `errors`，再让异常冒出去；父进程据此**显式报错**，**不退回 CPU worker**。

    `started` = 父进程持有的共享计数器：每个 worker 初始化成功就 +1。父进程靠它核对
    **数量**（否则"只起来 1 个 worker、其余全死"会被任务队列悄悄掩盖——池只会重拉进程，
    调用方看不见）。计数器**只在初始化成功后**加：初始化抛异常的 worker 不加，父进程改看
    `failed`。
    """
    torch.set_num_threads(1)
    bad = [n for n, p in mirror.named_parameters() if p.device.type != "cpu"]
    if bad:
        raise RuntimeError(
            f"worker 的策略镜像必须整体在 CPU，实得非 CPU 参数 {bad[:3]}…——"
            "父进程只传 CPU 共享镜像；CUDA 副本由 worker 在本进程内自建。")
    try:
        if worker_device == "cpu":
            # 默认档：直接读同一块共享内存，每步**零参数 IPC**（与上一批逐位相同）。
            _WORKER["policy"] = mirror
            _WORKER["mirror"] = None
        elif worker_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "worker_device='cuda' 但本进程的 torch 报告 cuda.is_available()=False。"
                    "显式报错，不退回 CPU worker。")
            # ⚠️ 深拷贝**在本进程内**做：`to('cuda')` 才会在此进程建上下文。用 CPU 镜像的
            #    数据（父进程通过 spawn 的 initargs 传进来的是 CPU 共享张量）。
            gpu = copy.deepcopy(mirror)
            gpu.optim = None
            gpu.to(torch.device("cuda"))
            gpu.eval()
            _WORKER["policy"] = gpu
            _WORKER["mirror"] = mirror          # 每步的 H2D 参数源（仍在共享内存）
            _WORKER["version_seen"] = -1
            _WORKER["version"] = version
            _WORKER["sync_times"] = sync_times
        else:
            raise ValueError(
                f"worker_device={worker_device!r} 非法：只接受 {_WORKER_DEVICES}。")
        if slots is not None:
            with slots.get_lock():             # 领取本 worker 的诊断槽位（同步耗时用）
                _WORKER["slot"] = int(slots.value)
                slots.value += 1
    except BaseException as e:                 # noqa: BLE001 —— 任何初始化失败都显式记账
        _worker_fail(e, failed, errors)
        raise
    if started is not None:
        with started.get_lock():
            started.value += 1


def _worker_ping(_: int) -> tuple[int, str]:
    """启动探针：返回 (本 worker 的 pid, 策略设备)——父进程据此核对池真的能执行任务，
    且**设备真的是请求的那一档**（CUDA 档若静默落回 CPU，这里当场露馅）。"""
    pol = _WORKER.get("policy")
    if pol is None:
        raise RuntimeError("worker 未初始化（`_worker_init` 没跑）——这是实现错误，"
                           "不是可以回退串行的状态。")
    return os.getpid(), str(pol.device)


def _sync_worker_policy() -> None:
    """CUDA worker：把共享内存镜像刷进本进程的 GPU 副本（H2D），耗时记进共享数组。

    版本号（`sync_policy` 每步 +1）没变就跳过——同一训练步里一个 worker 若连跑多条链，
    第二条不必再刷。CPU worker（`mirror is None`）直接读同一块共享内存，**零同步**。
    """
    mirror = _WORKER.get("mirror")
    if mirror is None:
        return
    ver = _WORKER.get("version")
    seen = _WORKER["version_seen"]
    if ver is not None and int(ver.value) == seen:
        return
    t0 = time.perf_counter()
    # `load_state_dict` 内部是 `param.copy_`（原地，不换 storage）⟹ **CUDA 图不失效**
    # （图存的是张量地址，Adam 的原地更新同理）。
    _WORKER["policy"].load_state_dict(mirror.state_dict())
    dt = time.perf_counter() - t0
    if ver is not None:
        _WORKER["version_seen"] = int(ver.value)
    arr, slot = _WORKER.get("sync_times"), _WORKER.get("slot")
    if arr is not None and slot is not None:
        with arr.get_lock():
            arr[slot] += dt


def _worker_run_chain(task: _Task) -> tuple[list[Decision], dict]:
    """worker 任务：跑**一条**链（`roll_chain` 原样调用，语义一位不改），返回决策 + 指标。

    ⚠️ 采样流**逐链独立**：`seed_chain = seed*SEED_STRIDE + g`（与串行档的仿真扰动种子
    同一个数）。串行档是 G 条链共用一条流、按链序消费——并行档做不到（消费次序不确定），
    故口径改为逐链独立。这是本批**有意的数值变化**，见模块 docstring。
    ⚠️ 采样流的**设备跟随 worker 的策略设备**（CUDA 上 `torch.multinomial` 不接受 CPU
    generator）——故 `worker_device="cuda"` 档用的是 CUDA generator，**与 CPU 档不是同一
    条流**，数值会变（默认档不受影响）。
    """
    (_g, seed_chain, inst, layout, dm, cfg, ctx, constraints,
     route_k, route_zones, geom_bias, pm_head, charge_head) = task
    policy = _WORKER.get("policy")
    if policy is None:
        raise RuntimeError("worker 未初始化（`_worker_init` 没跑）——这是实现错误，"
                           "不是可以回退串行的状态。")
    _sync_worker_policy()
    gen = torch.Generator(device=policy.device.type).manual_seed(seed_chain)
    dec, met = roll_chain(inst, layout, dm, cfg, policy, seed_chain, ctx,
                          sample=True, generator=gen, constraints=constraints,
                          route_k=route_k, route_zones=route_zones, geom_bias=geom_bias,
                          pm_head=pm_head, charge_head=charge_head)
    return dec, met


class ChainWorkerPool:
    """常驻的链级 worker 进程池（spawn）。一次建池、多步复用；`close()` 释放。

    用法::

        pool = ChainWorkerPool(policy, n_workers=8)
        try:
            r, diag = joint_chain_step(policy, ..., parallel=True, pool=pool)
        finally:
            pool.close()          # 或 with ChainWorkerPool(...) as pool:
    """

    def __init__(self, policy: PolicyNet, n_workers: int, *,
                 worker_device: str = "cpu",
                 start_timeout_s: float = 180.0, task_timeout_s: float = 900.0,
                 measure_ipc: bool = False) -> None:
        if policy.enc is None:
            raise ValueError("并行档要求策略带编码器（enc）——无编码器的策略连打分头都没有，"
                             "joint_chain_step 本就走不通。")
        if int(n_workers) < 1:
            raise ValueError(f"n_workers={n_workers} 非法：至少 1 个 worker。")
        if worker_device not in _WORKER_DEVICES:
            raise ValueError(f"worker_device={worker_device!r} 非法：只接受 {_WORKER_DEVICES}。")
        self._worker_device = str(worker_device)
        # ⚠️ 先清编码器缓存再深拷贝：CUDA 档的 `enc._graphs` 里是 `torch.cuda.CUDAGraph`
        #    对象（不可深拷贝、不可 pickle）。它们只是**缓存**，清掉与 `LayoutEncoder._apply`
        #    的既有清法同义，下一次前向按需重建——不清则"CUDA 串行档之后转并行"会当场报错。
        if policy.enc is not None:
            policy.enc._graphs.clear()
            policy.enc._tid_cache.clear()
            policy.enc._pad_cache.clear()
        # ⚠️ 镜像 = 结构副本 + 共享内存张量。`optim` 置空：镜像只做推理，不训练。
        mirror = copy.deepcopy(policy)
        mirror.optim = None
        mirror.to("cpu").eval()
        mirror.share_memory()           # 参数/缓冲的 storage 移入共享内存（worker 直接映射）
        self._mirror = mirror
        self._n_workers = int(n_workers)
        self._task_timeout_s = float(task_timeout_s)
        self._measure_ipc = bool(measure_ipc)
        self._pool: Any = None
        self._closed = False
        # 诊断读数（基准脚本用；生产路径 measure_ipc=False 时零开销）
        self.startup_s: float | None = None
        self.last_return_bytes: int | None = None
        self.last_task_bytes: int | None = None
        self.last_sync_s: float | None = None       # 上一轮 run_chains 里单 worker 的最大同步耗时
        # 上一轮里每个 worker 的健康度：ping 回来的 (pid, 设备) 里设备是否等于请求档
        self.worker_devices: list[str] = []
        ctx = tmp.get_context("spawn")          # ⚠️ spawn，不是 fork（硬要求 2）
        t0 = time.perf_counter()
        started = ctx.Value("i", 0)             # 初始化成功的 worker 计数（核对数量用）
        failed = ctx.Value("i", 0)              # 初始化**失败**的 worker 计数（显式报错用）
        errors = ctx.Queue()                    # 失败 traceback（逐个 worker 一条）
        version = ctx.Value("l", 0)             # 参数版本号（CUDA worker 的 H2D 同步按它去重）
        slots = ctx.Value("i", 0)               # 诊断槽位分配（给 `sync_times` 用）
        sync_times = ctx.Array("d", self._n_workers)
        self._version = version
        self._sync_times = sync_times
        self._pool = ctx.Pool(processes=self._n_workers, initializer=_worker_init,
                              initargs=(mirror, started, failed, errors,
                                        self._worker_device, version, slots, sync_times))
        # 启动核对：① 所有 worker 都跑完初始化（计数器到 n）② 池真的能执行任务（往返探针）。
        # ⚠️ 不查"pid 去重数 = n"：一个空闲 worker 可以连拿两个任务，去重会误报（实测踩过）。
        # ⚠️ 任一 worker 初始化失败（无 CUDA / 建上下文失败 / 显存不足）→ **立刻显式报错**，
        #    不等超时、不退回 CPU worker。失败检查放在**循环之后**：`started + failed` 一到
        #    数量就跳出循环，那时正是要看 failed 的时刻（放在循环体里会漏掉最后一次计数）。
        deadline = time.perf_counter() + start_timeout_s
        while started.value + failed.value < self._n_workers:
            if time.perf_counter() > deadline:
                self.terminate()
                raise RuntimeError(
                    f"worker 起不来：请求 {self._n_workers} 个，{start_timeout_s:.0f} s 内只有 "
                    f"{started.value} 个完成初始化——显式报错，不退回串行。")
            time.sleep(0.05)
        if failed.value:
            n_bad = int(failed.value)
            msgs = self._drain_errors(errors, n=min(n_bad, 3))
            self.terminate()
            raise RuntimeError(
                f"worker 初始化失败：请求 {self._n_workers} 个（worker_device="
                f"{self._worker_device!r}），有 {n_bad} 次初始化报错——**显式报错，"
                f"不退回 CPU worker**。首个错误：\n{msgs}")
        ar = self._pool.map_async(_worker_ping, list(range(self._n_workers)), chunksize=1)
        try:
            pings = ar.get(timeout=max(start_timeout_s - (time.perf_counter() - t0), 1.0))
        except Exception as e:      # noqa: BLE001 —— 任何启动失败都转成显式 RuntimeError
            self.terminate()
            raise RuntimeError(
                f"worker 进程起不来（请求 {self._n_workers} 个，探针 {start_timeout_s:.0f} s "
                f"内没有全部回话）：{e!r}——显式报错，不退回串行。") from e
        if len(pings) != self._n_workers:
            self.terminate()
            raise RuntimeError(
                f"worker 数量不对：请求 {self._n_workers} 个，探针只回了 {len(pings)} 条"
                "——显式报错，不退回串行。")
        self.worker_devices = [str(dev) for _pid, dev in pings]
        # ⚠️ 设备核对：CUDA 档若哪个 worker 静默落回 CPU，这里当场报错（不静默收下）。
        wrong = [i for i, dev in enumerate(self.worker_devices)
                 if not dev.startswith(self._worker_device)]
        if wrong:
            self.terminate()
            raise RuntimeError(
                f"worker 设备与请求不符：worker_device={self._worker_device!r}，但 worker "
                f"{wrong[:3]} 报的设备是 {[self.worker_devices[i] for i in wrong[:3]]}"
                "——显式报错，不静默按 CPU 跑。")
        self.startup_s = time.perf_counter() - t0

    @staticmethod
    def _drain_errors(errors: Any, n: int = 1, timeout_s: float = 0.5) -> str:
        """把 worker 塞进队列的失败 traceback 取回来（取到几条算几条，不阻塞太久）。"""
        out: list[str] = []
        for _ in range(max(int(n), 1)):
            try:
                out.append(str(errors.get(timeout=timeout_s)))
            except Exception:       # noqa: BLE001 —— 队列空了/坏了都不该盖住原始报错
                break
        return "\n".join(out) if out else "(worker 未回传 traceback)"

    # ── 生命周期 ─────────────────────────────────────────────────────────────

    @property
    def n_workers(self) -> int:
        return self._n_workers

    @property
    def mirror(self) -> PolicyNet:
        """共享内存镜像（只读用途：worker 读它、主进程原地刷新它）。"""
        return self._mirror

    @property
    def worker_device(self) -> str:
        """worker 的策略设备档（`"cpu"` / `"cuda"`）——建池时请求的那一档。"""
        return self._worker_device

    def _require_open(self, what: str) -> None:
        if self._closed or self._pool is None:
            raise RuntimeError(
                f"进程池已关闭/终止，不能再调用 {what}——并行档不得静默回退串行"
                "（请新建 ChainWorkerPool）。")

    def sync_policy(self, policy: PolicyNet) -> None:
        """把主进程策略参数刷进共享内存镜像（**原地** `copy_`，不换 storage），并 +1 版本号。

        `load_state_dict(strict=True)`：键或形状不符**当场报错**（不静默丢参数）。
        跨设备 `copy_`（CUDA→CPU）合法，故主进程可以继续用 GPU 训练。
        ⚠️ 版本号让 CUDA worker 只在参数**真的变过**时才做 H2D（同一训练步里一个 worker
        连跑多条链时不重复搬）；CPU worker 不使用它（直接读共享内存，零同步）。
        """
        self._require_open("sync_policy")
        self._mirror.load_state_dict(policy.state_dict())
        with self._version.get_lock():
            self._version.value += 1

    def run_chains(self, seed: int, G: int, *, inst: Instance, layout: Layout,
                   dm: np.ndarray, cfg: SimConfig, ctx: NormContext,
                   constraints: ConstraintConfig | None = None,
                   route_k: int = 1, route_zones: bool = False, geom_bias: bool = False,
                   pm_head: bool = False, charge_head: bool = False,
                   timeout_s: float | None = None
                   ) -> list[tuple[list[Decision], dict]]:
        """跑 G 条链（每条一个 worker 任务），按链号 g = 0..G-1 返回 `[(决策, 指标), …]`。

        ⚠️ 失败模式一律**显式报错 + 终止进程池**：worker 抛异常、任务超时、结果条数不对、
        决策字段类型不对（pickle 丢字段的形态）。绝不退回串行。
        """
        self._require_open("run_chains")
        if int(G) < 1:
            raise ValueError(f"G={G} 非法：至少 1 条链。")
        cons = constraints or ConstraintConfig()
        tasks = [(g, seed * SEED_STRIDE + g, inst, layout, dm, cfg, ctx, cons,
                  route_k, route_zones, geom_bias, pm_head, charge_head) for g in range(int(G))]
        limit = self._task_timeout_s if timeout_s is None else float(timeout_s)
        with self._sync_times.get_lock():       # 重置诊断槽位（CUDA worker 写回本轮的同步耗时）
            for i in range(self._n_workers):
                self._sync_times[i] = 0.0
        ar = self._pool.map_async(_worker_run_chain, tasks, chunksize=1)
        try:
            res = ar.get(timeout=limit)
        except Exception as e:      # noqa: BLE001 —— 任何 worker 失败都转成显式 RuntimeError
            self.terminate()
            raise RuntimeError(
                f"并行链任务失败（G={G}，超时上限 {limit:.0f} s，或 worker 抛异常）：{e!r}"
                "——已终止进程池；显式报错，不退回串行。") from e
        if len(res) != int(G):
            self.terminate()
            raise RuntimeError(
                f"并行链返回条数不对：请求 G={G}，实得 {len(res)}——已终止进程池，不退回串行。")
        for g, item in enumerate(res):
            if not (isinstance(item, tuple) and len(item) == 2):
                self.terminate()
                raise RuntimeError(f"链 {g} 的返回值形状不对（应为 (决策, 指标)）：{type(item)}")
            dec, met = item
            if not isinstance(dec, list) or not isinstance(met, dict):
                self.terminate()
                raise RuntimeError(f"链 {g} 的返回类型不对：决策 {type(dec)}、指标 {type(met)}")
            if any(not isinstance(d, Decision) for d in dec):
                self.terminate()
                raise RuntimeError(f"链 {g} 的决策里混进了非 Decision 对象——pickle 丢字段/"
                                   "类型漂移；显式报错，不静默丢弃。")
            for key in ("makespan", "energy", "tardy_twt"):
                if key not in met:
                    self.terminate()
                    raise RuntimeError(f"链 {g} 的指标缺 {key}（objective_vector 需要）——"
                                       "不静默继续。")
        if self._measure_ipc:
            # 诊断用（基准脚本）：父→子 = G 份任务（含环境八件套），子→父 = 全部决策 + 指标。
            # 注意：measure_ipc=True 时这两次 pickle 计入当步墙钟（约几 ms）。
            self.last_task_bytes = int(G) * len(pickle.dumps(tasks[0], protocol=pickle.HIGHEST_PROTOCOL))
            self.last_return_bytes = len(pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL))
        # 上一轮的每 worker 参数同步耗时（CUDA 档才有；CPU 档恒 0）。取**最大**值 = 该步
        # 关键路径上多付的同步钱（8 个 worker 各付各的，看最大值比看均值更贴近天花板）。
        with self._sync_times.get_lock():
            self.last_sync_s = max(self._sync_times[i] for i in range(self._n_workers))
        return list(res)

    def terminate(self) -> None:
        """异常路径：立即杀 worker（池不可再用）。幂等。"""
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
        self._closed = True

    def close(self) -> None:
        """正常路径：关池并等 worker 退出（幂等）。"""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        self._closed = True

    def __enter__(self) -> "ChainWorkerPool":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        self.close()
        return False
