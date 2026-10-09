# -*- coding: utf-8 -*-
"""**D5 跑数器**：扁平链 vs 两层（v2 实验计划 D5 / 设计 §12.3）。

```
python -m plcsp_v2.d5_nested --arms flat two --seeds 0 1 2 --steps 200
```

**问的是**：**分层（我们的两层 + 交替）值不值那个代价？**

| 臂 | 怎么训 | 组 | 更新 |
|---|---|---|---|
| **`flat`** | `flat_step(arm="B")` —— 一条链的**全部**决策算**一个**组 | 共享 | 同步 |
| **`two`** | `train_step` —— 按层切成两个组 | 分层 | 交替 |

**两臂用同一套网络、同一个仿真、同一个 `n=8`、同 `dt_events`。差别只在"组怎么切、怎么更新"。**

⚠️ **D5 与 Q2（D2）几乎是同一件事。** 设计 §12.0 已核：**「分层组 + 同步」不可行**
（同步要付 `N×M`）⟹ 能比的只有「扁平分同步」vs「分层交替」。**两个臂都已有代码，所以先做 D5。**

⚠️ **两臂的每步成本不同**（设计 §12.3 说分层是 1× vs 2×）。**同一 `--steps` 下两臂花的机时不一样，
报数时必须一并报**——否则"谁更好"分不清是不是"谁跑得多"。

══ 输出 ══

与 `q1_flat.py` 同形的 JSONL（`start / train / eval / final / saved`），便于两处一起读。

⚠️ 默认写到**系统临时目录**（项目硬规则 1）；正式结果用 `--out checkpoints/d5/d5.jsonl`。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from .hrl.flat import flat_step
from .hrl.pool import ChainPool, SegmentPool
from .hrl.train import train_step
from .q1_flat import DEFAULT_OUT, build, constraint_counts, evaluate

ARMS = ("flat", "two")
"""`flat` = 扁平链（一个组 · 同步）· `two` = 两层（按层切组 · 交替）。"""


def run_one(arm: str, seed: int, *, steps: int, n: int, dt_events: int,
            eval_every: int, eval_n: int, rec, workers: int = 0,
            save_dir: str = "", lr: float = 1e-4, env: dict | None = None) -> None:
    """一条 `(臂, 种子)`。**两臂共用 `build` / `evaluate`** ⟹ 实例/布局/编码器/约束档全同。"""
    env = dict(env or {})
    spec, net = build(seed, **env)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    rec({"kind": "start", "arm": arm, "seed": seed, "steps": steps, "n": n,
         "dt_events": dt_events, "workers": workers,
         "n_params": sum(p.numel() for p in net.parameters())})
    rec({"kind": "eval", "arm": arm, "seed": seed, "step": 0,
         **evaluate(net, spec, n=eval_n, seed=0, dt_events=dt_events)})

    # ⚠️ 两臂的池**不是同一个类**：扁平采"链"，两层采"段"。
    pool = None
    if workers > 1:
        pool = (ChainPool(net, spec, n_workers=workers, dt_events=dt_events) if arm == "flat"
                else SegmentPool(net, spec, n_workers=workers, dt_events=dt_events))
    t0 = time.perf_counter()
    try:
        for step in range(1, steps + 1):
            if arm == "flat":
                out = flat_step(net, spec, optim, arm="B", n=n,
                                seed=seed * 10000 + step, dt_events=dt_events, pool=pool)
                # ⚠️ **两臂的返回不同构**（`flat_step` 给 `loss`/`r_mean`；
                # `train_step` 给 `{"base_makespan", "B": {...}, "A": {...}}`）——
                # 这里**按各自有的字段记**，不用零顶替（那会让"没这个量"看起来像"这个量是 0"）。
                row = {"kind": "train", "arm": arm, "seed": seed, "step": step,
                       "elapsed_s": time.perf_counter() - t0,
                       "loss": float(out["loss"]), "r_mean": float(out["r_mean"])}
            else:
                out = train_step(net, spec, optim, M=n, G=n,
                                 seed=seed * 10000 + step, dt_events=dt_events, pool=pool)
                row = {"kind": "train", "arm": arm, "seed": seed, "step": step,
                       "elapsed_s": time.perf_counter() - t0,
                       "base_makespan": out["base_makespan"],
                       "loss_B": out["B"]["loss"], "loss_A": out["A"]["loss"],
                       "n_top": out["B"]["n_top"],
                       "seg_std_A": out["A"]["seg_std"]}
            rec(row)
            if eval_every and step % eval_every == 0:
                rec({"kind": "eval", "arm": arm, "seed": seed, "step": step,
                     **evaluate(net, spec, n=eval_n, seed=0, dt_events=dt_events)})

        # ⚠️ **结果先落盘、再关池**（§110）：实测一次长跑在 `shutdown` 里挂掉，
        # 把刚跑完的 200 步整轮拖没了。收尾动作不该有权否决已经算出来的结果。
        dspec, _ = build(seed, disturb=True, **env)
        rec({"kind": "final", "arm": arm, "seed": seed, "total_s": time.perf_counter() - t0,
             "clean": evaluate(net, spec, n=eval_n, seed=1, dt_events=dt_events),
             "disturbed": evaluate(net, dspec, n=eval_n, seed=1, dt_events=dt_events),
             "agv_mtbf": dspec.cfg.agv_mtbf,
             # ⚠️ **规则档**下的五条约束触发数（§115.3）—— 让"跑通了但约束没触发"当场可见
             "constraints": constraint_counts(spec)})
        if save_dir:
            p = Path(save_dir) / f"{arm}_s{seed}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                        "arm": arm, "seed": seed, "steps": steps, "n": n,
                        "dt_events": dt_events, "with_value": False, "tau": spec.cfg.tau}, p)
            rec({"kind": "saved", "arm": arm, "seed": seed, "path": str(p)})
    finally:
        if pool is not None:
            pool.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D5：扁平链 vs 两层（v2 实验计划 D5）")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n", type=int, default=8, help="组大小（两臂共用）")
    ap.add_argument("--dt-events", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=25, help="0 = 不中途评测")
    ap.add_argument("--eval-n", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1, help="父进程基线线程数（§107：段/链要 1）")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--instance", default="mk01", help="mk01–mk10")
    ap.add_argument("--tau", type=float, default=None, help="交期 τ；None = 查标定表")
    ap.add_argument("--pm-interval", type=float, default=None, help="⑫ 保养间隔（dev 档 30）")
    ap.add_argument("--finite-buffer", action="store_true",
                    help="② 开有限缓冲（默认关）。⚠️ 开时改动力学（见 progress-log §118）")
    ap.add_argument("--agv-n-failures", type=int, default=0,
                    help="⑨ 按次数注入（0 = 关，用 agv_mtbf 泊松流）。照 Shi 2025 口径")
    ap.add_argument("--target-failures", type=float, default=None,
                    help="⑨ 按每局期望故障次数反推 agv_mtbf")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
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
                        workers=a.workers, save_dir=a.save_dir, lr=a.lr,
                        env={"instance": a.instance, "tau": a.tau,
                             "pm_interval": a.pm_interval,
                             "target_failures": a.target_failures,
                             "agv_n_failures": a.agv_n_failures,
                             "finite_buffer": a.finite_buffer})
    print(f"\n写出：{out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
