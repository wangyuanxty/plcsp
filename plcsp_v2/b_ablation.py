# -*- coding: utf-8 -*-
"""**B · 约束组消融跑数器**（`docs/v2/experiment-plan.md` §B）—— 支撑"五条约束各自要紧"。

```
python -m plcsp_v2.b_ablation --arms Full -物流 -生产 -信息 None --seeds 0 1 2 --steps 200
```

**问的是**：把某一条约束关掉之后，学出来的策略差多少。
**非加性效应**（去掉 X 之后 Y 的贡献变了多少）也在读数列里 —— **非加性 = 机制**。

══ 与 `d5_nested.py` / `q1_flat.py` 的关系 ══

**同一个 `build` / `evaluate` / 扁平训练器**，只是把 `env["constraints"]` 换成五个组之一。
**⟹ 五个臂之间只差约束档这一个变量**，别的（实例 / 布局 / 编码器 / 超参 / 种子）全同。

══ ⚠️ 参考解**共享**，这是刻意的 ══

五个臂的 `cfg` **相同**（只有约束档不同）⟹ `ReferenceObjectives.of(inst, cfg)` 与
`reference_makespan(inst, cfg)` 给出的**是同一份参考解** ⟹ `reward_weights` 的 `w`
与 `norm_context` 的 `m_ref` **五个臂共用**。

**这是对比实验该有的样子**：同一把尺子量所有臂，差才归因得到"约束"头上。
⚠️ `reference_run` **不接受 `constraints` 参数**（它恒用默认档）——
**在本跑数器里这恰好是我们要的**，故不改它。**若将来要"每臂用各自的参考解，须显式改。**

══ ⚠️ 本跑数器**不测**的事 ══

- **不测**"约束在真实车间里有多要紧" —— 那只由 `ABLATION_GROUPS` 的**开关**定义，
  与"该约束的物理强度"是两回事（后者看 `constraint_counts` 的激活量）。
- 每个 `final` 记录都带**规则驱动**的五条约束触发数（§115.3 的口径）——
  **让"这一臂里被关掉的那条真的没触发"当场可见。**

⚠️ 默认写到系统临时目录（项目硬规则 1：`*.jsonl` 不进项目根）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .env.constraints import ABLATION_GROUPS
from .hrl.flat import flat_step
from .hrl.pool import ChainPool
from .q1_flat import build, constraint_counts, evaluate

DEFAULT_OUT = os.path.join(tempfile.gettempdir(), "b_ablation.jsonl")

# 顺序照 `experiment-plan.md` §B 的表：先 Full，再四组"关一类"，最后全关。
ARMS: tuple[str, ...] = ("Full", "-物流", "-生产", "-信息", "None")
"""五个消融组 —— **名字必须与 `ABLATION_GROUPS` 的键逐字相同**（下面有断言守着）。"""


def _norm(name: str) -> str:
    """命令行里的臂名 → `ABLATION_GROUPS` 的键。

    ⚠️ **为什么需要它**：四个组名以 `-` 开头（`-物流` / `-生产` / `-信息`），
    而 **argparse 会把 `-物流` 当成选项**（`--arms -物流` 直接报
    `unrecognized arguments`；`--arms=-物流` 能收，但**重复的 `--arms=` 只保留最后一个**）。
    ⟹ **允许省掉前导 `-`**：`--arms Full 物流 生产 信息 None`。
    两种写法都收（原本就带 `-` 的原样通过）。
    """
    if name in ABLATION_GROUPS:
        return name
    return "-" + name if ("-" + name) in ABLATION_GROUPS else name


def run_one(arm: str, seed: int, *, steps: int, n: int, dt_events: int,
            eval_every: int, eval_n: int, rec, workers: int = 0,
            save_dir: str = "", lr: float = 1e-4, env: dict | None = None) -> None:
    """一条 `(臂, 种子)`。`arm` 是 `ABLATION_GROUPS` 的键。"""
    env = dict(env or {})
    cons = ABLATION_GROUPS[arm]
    spec, net = build(seed, constraints=cons, **env)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    rec({"kind": "start", "arm": arm, "seed": seed, "steps": steps, "n": n,
         "dt_events": dt_events, "workers": workers,
         # ⚠️ 把**这一臂到底开了哪些约束**记全 —— 名字（"-生产"）与内容是两回事，
         # 将来 `ABLATION_GROUPS` 改了而名字没改，读数的人必须看得见。
         "constraints": {k: bool(v) for k, v in cons.__dict__.items()},
         "n_params": sum(p.numel() for p in net.parameters())})
    rec({"kind": "eval", "arm": arm, "seed": seed, "step": 0,
         **evaluate(net, spec, n=eval_n, seed=0, dt_events=dt_events)})

    pool = None
    if workers > 1:
        pool = ChainPool(net, spec, n_workers=workers, dt_events=dt_events)
    t0 = time.perf_counter()
    try:
        for step in range(1, steps + 1):
            out = flat_step(net, spec, optim, arm="B", n=n,
                            seed=seed * 10000 + step, dt_events=dt_events, pool=pool)
            rec({"kind": "train", "arm": arm, "seed": seed, "step": step,
                 "elapsed_s": time.perf_counter() - t0,
                 "r_mean": float(out["r_mean"]),
                 "loss": float(out["loss"]),
                 "makespan": float(np.mean([r["makespan"] for r in out["runs"]]))})
            if eval_every and step % eval_every == 0:
                rec({"kind": "eval", "arm": arm, "seed": seed, "step": step,
                     **evaluate(net, spec, n=eval_n, seed=0, dt_events=dt_events)})

        # ⚠️ **结果先落盘、再关池**（§110）：收尾动作不该有权否决已算出的结果。
        dspec, _ = build(seed, disturb=True, constraints=cons, **env)
        rec({"kind": "final", "arm": arm, "seed": seed,
             "total_s": time.perf_counter() - t0,
             "clean": evaluate(net, spec, n=eval_n, seed=1, dt_events=dt_events),
             "disturbed": evaluate(net, dspec, n=eval_n, seed=1, dt_events=dt_events),
             "agv_mtbf": dspec.cfg.agv_mtbf,
             "constraints": constraint_counts(spec)})
        if save_dir:
            p = Path(save_dir) / f"{arm.replace('-', 'm')}_s{seed}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                        "arm": arm, "seed": seed, "steps": steps, "n": n,
                        "dt_events": dt_events, "with_value": False,
                        "constraints": {k: bool(v) for k, v in cons.__dict__.items()},
                        # ⚠️ **必须存环境**（2026-10-10 加）—— 见 `q1_flat.resolve_train_env`
                        "env": dict(env),
                        "tau": spec.cfg.tau}, p)
            rec({"kind": "saved", "arm": arm, "seed": seed, "path": str(p)})
    finally:
        if pool is not None:
            pool.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="B：约束组消融（设计 §B）")
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    help="消融组名。⚠️ 四个组名以 `-` 开头，**argparse 会把它当选项** "
                         "⟹ 可省掉前导 `-`：`--arms Full 物流 生产 信息 None`")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n", type=int, default=8, help="组大小")
    ap.add_argument("--dt-events", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=25, help="0 = 不中途评测")
    ap.add_argument("--eval-n", type=int, default=5)
    ap.add_argument("--workers", type=int, default=0, help="并行采链的进程数（0 = 顺序）")
    ap.add_argument("--threads", type=int, default=1, help="父进程 torch 线程数")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--instance", default="mk01", help="mk01–mk10")
    ap.add_argument("--tau", type=float, default=None, help="交期 τ；None = 查标定表")
    ap.add_argument("--pm-interval", type=float, default=None, help="⑫ 保养间隔（dev 档 30）")
    ap.add_argument("--agv-n-failures", type=int, default=0,
                    help="⑨ 按次数注入（0 = 关）。照 Shi 2025 口径")
    ap.add_argument("--target-failures", type=float, default=None,
                    help="⑨ 按每局期望故障次数反推 agv_mtbf")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    if a.threads:
        torch.set_num_threads(a.threads)

    # ⚠️ **先规范化、再查后跑**：四个组名以 `-` 开头，命令行里写不出来（见 `_norm`）。
    arms = [_norm(x) for x in a.arms]
    missing = [x for x in arms if x not in ABLATION_GROUPS]
    if missing:
        print(f"✗ 这些臂不在 ABLATION_GROUPS 里：{missing}；"
              f"可用的={list(ABLATION_GROUPS)}", file=sys.stderr)
        return 2

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 断点续跑：跳过已经写出 `final` 的 `(臂, 种子)`（同 `q1_flat` 的口径）
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
            for arm in arms:
                if (arm, seed) in done:
                    continue
                run_one(arm, seed, steps=a.steps, n=a.n, dt_events=a.dt_events,
                        eval_every=a.eval_every, eval_n=a.eval_n, rec=rec,
                        workers=a.workers, save_dir=a.save_dir, lr=a.lr,
                        env={"instance": a.instance, "tau": a.tau,
                             "pm_interval": a.pm_interval,
                             "target_failures": a.target_failures,
                             "agv_n_failures": a.agv_n_failures})
    print(f"\n写出：{out}", file=sys.stderr)
    return 0


if __name__ == "__main__":      # ⚠️ Windows `spawn` 需要（见 `hrl/pool.py` 的 docstring）
    raise SystemExit(main())
