# -*- coding: utf-8 -*-
"""**实验 A：主干验证** —— "策略能不能稳定赢过固定规则"（`docs/v3/experiment-plan.md` §A）。

| 臂 | 训练 | 每步的优势从哪来 |
|---|---|---|
| **`flat`** | `flat_step(arm="B")` + `ChainPool` | **组内 z 化**（整条链一个优势），**无 critic** |
| **`hrl`** | `train_step`（**A 先 B 后**）+ `SegmentPool` | 两个阶段各做一次**组内 z 化** |

══ ⚠️ 扁平臂为什么用 `B`，不用 `A`/`A2` ══

**两臂只该差"分层不分层"这一个变量。**
`B` 的优势是**组内 z 化**，与两层两个阶段用的 `train.group_advantage` **同源**；
用 `A`（critic）会引入**第二个**变量（有没有 critic）⟹ 读出的差异分不清是哪个起的作用。
（同 `hrl/flat.py` 的口径 ②："两臂的更新规则逐字相同"。）

══ ⚠️ 两臂的成本**不对等**（设计 §6.0 ①）══

每步：
- `flat` = **`n` 条完整链**
- `hrl`  = **1 条 base 链 + `n_top × (G + M)` 条段**

一条完整链 ≈ `n_top` 条段 ⟹ **`hrl` 约为 `flat` 的 2 倍**。
**这是分层的固有成本**（交替训练把 `G×M` 压到 `G+M`），**不是可调的口径**。

══ 判据（§A）══

**峰值**赢规则 ⟹ 能力够；**最终值**赢规则 ⟹ 收敛稳。**两个分开报**
（v2 的教训：只看最终值会把"学到又丢掉"误读成"学不会"）。

══ ⚠️ 还要记"空转"══

阶段 B 的组内优势**可能全为 0** ⟹ 那个顶层点不产生梯度（`docs/v3/progress-log.md` §12.3）。
**成因不是"链全同"** —— 实测 `种链 = M/M`；是"M 组不同的参数算出**同一个决策序列**"。
**⟹ 每步记 `idle_top` / `n_top`**，让"阶段 B 有一半在空转"当场可见。

用法（⚠️ **必须带 `-m`**，多进程池要 `__main__` 守卫）：

```
python -m plcsp_v3.a_main --arms flat hrl --seeds 0 1 2 --steps 200 \
    --dt-events 20 --n 8 --workers 8 --layout-seed 0 \
    --pm-interval 30 --target-failures 4 --save-dir <目录> --out <jsonl>
```
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .env.constraints import ConstraintConfig          # noqa: F401  （转出给调用方）
from .hrl.agents import HierarchicalPolicy
from .hrl.flat import flat_step
from .hrl.pool import ChainPool, SegmentPool
from .hrl.train import Spec, train_step
from .q1_flat import DEFAULT_OUT, build, constraint_counts, evaluate

ARMS: tuple[str, ...] = ("flat", "hrl")
"""`flat` = 不分层（一条链一个组）· `hrl` = 两层（按层切两个组、交替更新）。"""

FLAT_ARM_INTERNAL = "B"
"""`flat` 臂内部用的优势口径 —— **`B`（组内 z 化）**，理由见模块 docstring。"""


def stage_b_idle(segments: list[dict]) -> tuple[int, int]:
    """阶段 B 的**空转顶层点** → `(空转数, 顶层点数)`。

    空转 = 该顶层点的 M 条分支**优势全为 0**（= 组内段回报全相同）⟹ `Â ≡ 0` ⟹ 不产生梯度。
    """
    by_j: dict[int, list[float]] = {}
    for row in segments:
        by_j.setdefault(int(row["j"]), []).append(float(row["adv"]))
    idle = sum(1 for advs in by_j.values() if all(a == 0.0 for a in advs))
    return idle, len(by_j)


def _make_pool(arm: str, net: HierarchicalPolicy, spec: Spec, workers: int,
               dt_events: int):
    """按臂建池（`workers < 2` ⟹ 顺序档，返回 `None`）。"""
    if workers < 2:
        return None
    if arm == "flat":
        return ChainPool(net, spec, n_workers=workers, dt_events=dt_events)
    return SegmentPool(net, spec, n_workers=workers, dt_events=dt_events)


def run_one(arm: str, seed: int, *, steps: int, n: int, dt_events: int,
            eval_every: int, eval_n: int, rec, workers: int = 0,
            save_dir: str = "", env: dict | None = None) -> None:
    """一条 `(臂, 种子)` 的完整训练 + 评测。`rec` = 写一行的回调。

    ⚠️ `workers > 1` ⟹ 多进程（**只支持 CPU**）。并行档每进程 **1 线程** ⟹
    浮点归约次序与顺序档不同 ⟹ **同一次对比里两臂必须同档**，报出的数字要注明档位。

    ⚠️ **两臂的 `train` 行字段不完全相同**（`loss` 的口径不同，见各自的 `extra`）——
    **能逐格比的是 `grad_upper` / `grad_lower` / `idle_top` / `elapsed_s`**，
    以及所有 `eval` 行。
    """
    if arm not in ARMS:
        raise ValueError(f"arm={arm!r} 未知：只接受 {ARMS}")
    env = dict(env or {})
    if workers > 1 and str(env.get("device", "cpu")) != "cpu":
        raise ValueError("并行档只跑 CPU —— `--workers` 与 `--device cuda` 二选一")
    spec, net = build(seed, with_value=False, **env)
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    rec({"kind": "start", "arm": arm, "seed": seed, "steps": steps, "n": n,
         "dt_events": dt_events, "workers": workers,
         "n_params": sum(p.numel() for p in net.parameters()),
         "reward_norm": env.get("reward_norm", "ref"),
         "constraints": constraint_counts(spec)})
    rec({"kind": "eval", "arm": arm, "seed": seed, "step": 0,
         **evaluate(net, spec, n=eval_n, seed=0, dt_events=dt_events)})

    pool = _make_pool(arm, net, spec, workers, dt_events)
    t0 = time.perf_counter()
    try:
        for step in range(1, steps + 1):
            t1 = time.perf_counter()
            if arm == "flat":
                out = flat_step(net, spec, optim, arm=FLAT_ARM_INTERNAL, n=n,
                                seed=seed * 10000 + step, dt_events=dt_events,
                                pool=pool)
                gn = out["grad_norm"]
                row = {"loss": out["loss"] / n,          # ⚠️ `flat_step` 给的是**整组之和**
                       "r_mean": float(np.mean([r["R"] for r in out["runs"]])),
                       "grad_upper": 0.0,                # 结构上没有上层
                       "grad_lower": gn["heads"], "enc_grad": gn["encoder"],
                       "idle_top": None, "n_top": None}  # 不分层 ⟹ 无此概念（**不是 0**）
            else:
                out = train_step(net, spec, optim, M=n, G=n,
                                 seed=seed * 10000 + step, dt_events=dt_events,
                                 pool=pool)
                a, b = out["A"], out["B"]
                idle, n_top = stage_b_idle(b["segments"])
                row = {"loss_a": a["loss"], "loss_b": b["loss"],
                       "seg_std_max": max(a["seg_std"]) if a["seg_std"] else 0.0,
                       "base_makespan": out["base_makespan"],
                       "grad_upper": a["grad_norm"]["upper"],
                       "grad_lower": b["grad_norm"]["lower"],
                       "enc_grad": None,                 # `train._grad_norm` 不拆 `enc`
                       "idle_top": idle, "n_top": n_top}
            rec({"kind": "train", "arm": arm, "seed": seed, "step": step, **row,
                 "elapsed_s": time.perf_counter() - t1})
            if eval_every and step % eval_every == 0:
                rec({"kind": "eval", "arm": arm, "seed": seed, "step": step,
                     **evaluate(net, spec, n=eval_n, seed=0, dt_events=dt_events)})
    finally:
        if pool is not None:
            pool.close()

    rec({"kind": "final", "arm": arm, "seed": seed,
         "total_s": time.perf_counter() - t0,
         "clean": evaluate(net, spec, n=eval_n, seed=1, dt_events=dt_events),
         # ⚠️ **规则档**下的约束触发数（§115.3）—— 让"跑通了但约束没触发"当场可见
         "constraints": constraint_counts(spec)})

    if save_dir:
        # ⚠️ **必须存盘**（§103.5 的教训）：不存 ⟹ 后面的 H 敏感性 / 同策略重评都做不了。
        # 连同**重建网络要的全部元信息**一起存。
        p = Path(save_dir) / f"{arm}_s{seed}.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                    "arm": arm, "seed": seed, "steps": steps, "n": n,
                    "dt_events": dt_events, "with_value": False,
                    "env": dict(env), "tau": spec.cfg.tau}, p)
        rec({"kind": "saved", "arm": arm, "seed": seed, "path": str(p)})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="实验 A：主干验证（experiment-plan §A）")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS),
                    help="flat = 不分层 · hrl = 两层（A 先 B 后）")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="§A 要求 ≥3")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n", type=int, default=8, help="组大小（两层的 `G = M = n`）")
    ap.add_argument("--dt-events", type=int, default=20, help="顶层定时器步长（§10②）")
    ap.add_argument("--eval-every", type=int, default=25, help="0 = 不中途评测")
    ap.add_argument("--eval-n", type=int, default=3, help="评测链数（greedy）")
    ap.add_argument("--device", default="cpu", help="cpu / cuda")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch 线程数（默认 1）。⚠️ 实测 §99：24 线程比 1 线程慢 2.7×")
    ap.add_argument("--workers", type=int, default=0,
                    help="多进程的进程数（0 = 顺序档）。⚠️ **两层臂不上池跑不完**："
                         "每步 `n_top×(G+M)` 条段，MK01/`dt_events=20`/`n=8` ≈ 224 条")
    ap.add_argument("--save-dir", default="", help="存训练好的策略（.pt）；空 = 不存")
    ap.add_argument("--layout-seed", type=int, default=0,
                    help="⭐ §2 的口径是**固定 0**（默认已改 0，不是 None）")
    ap.add_argument("--reward-norm", default="ref", choices=("ref", "minmax"))
    ap.add_argument("--instance", default="mk01", help="mk01–mk10")
    ap.add_argument("--tau", type=float, default=None, help="交期 τ；None = 查标定表")
    ap.add_argument("--pm-interval", type=float, default=30, help="⑫ 保养间隔（§2 = 30）")
    ap.add_argument("--no-finite-buffer", dest="finite_buffer", action="store_false",
                    help="② **关掉**有限缓冲（**默认开着**）")
    ap.add_argument("--agv-n-failures", type=int, default=0)
    ap.add_argument("--target-failures", type=float, default=4, help="⑨ 每局期望故障数（§2 = 4）")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    if a.threads:
        # ⚠️ **只在跑数器里设**，不进库（见 `q1_flat.main` 的同款说明）。
        torch.set_num_threads(a.threads)
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"✗ 这个解释器没有 CUDA（torch {torch.__version__}）。"
              "GPU 请用 D:/anaconda/envs/py312/python.exe", file=sys.stderr)
        return 2
    if a.workers < 2:
        print("⚠️ 顺序档：两层臂每步要跑 `n_top×(G+M)` 条段，MK01/`dt_events=20`/`n=8` "
              "约 224 条 ⟹ **这个档跑不完 200 步**。请给 `--workers`。", file=sys.stderr)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # ⚠️ **断点续跑**：跳过已经写出 `final` 的 `(臂, 种子)`（同 `q1_flat`）。
    done: set[tuple[str, int]] = set()
    if out.exists():
        for ln in out.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if d.get("kind") == "final":
                done.add((str(d["arm"]), int(d["seed"])))
        if done:
            print(f"续跑：已完成 {sorted(done)}，跳过", file=sys.stderr)

    env = {"instance": a.instance, "layout_seed": a.layout_seed, "tau": a.tau,
           "reward_norm": a.reward_norm, "pm_interval": a.pm_interval,
           "target_failures": a.target_failures, "agv_n_failures": a.agv_n_failures,
           "finite_buffer": a.finite_buffer, "device": a.device}
    with out.open("a", encoding="utf-8") as fh:
        def rec(d: dict) -> None:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            fh.flush()
            print(json.dumps(d, ensure_ascii=False), flush=True)

        for seed in a.seeds:
            for arm in a.arms:
                if (arm, seed) in done:
                    continue
                run_one(arm, seed, steps=a.steps, n=a.n, dt_events=a.dt_events,
                        eval_every=a.eval_every, eval_n=a.eval_n, rec=rec,
                        workers=a.workers, save_dir=a.save_dir, env=env)
    print(f"\n写出：{out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
