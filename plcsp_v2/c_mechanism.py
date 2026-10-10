# -*- coding: utf-8 -*-
"""**C · 机制消融跑数器**（`docs/v2/experiment-plan.md` §C）—— 支撑"每条机制有贡献"。

```
python -m plcsp_v2.c_mechanism --arms base no_failover age_on --seeds 0 1 2 --steps 200
```

**问的是**：把某一条机制拿掉（或加上）之后，学出来的策略差多少。

══ ⚠️ 本跑数器**只做"纯配置"的臂** ══

按 `docs/v2/experiment-plan.md` §C 的表，机制清单里有**两类**：

| 类 | 怎么关 | 本跑数器 |
|---|---|---|
| **纯配置**（`SimConfig` 的开关） | `agv_failover` / `machine_age_failure` / `dt_events` | ✅ **做** |
| **改动作空间**（PA 12 条 → 1 条、LA 3 条 → 1 条、保养头 → 规则驱动） | 要在打分层加**规则掩码** | ⛔ **未做** |

**⛔ 后一类未做的原因（如实登记，不遮掩）**：加掩码要动 `hrl/rollout.py` 的
`NetPolicy.policy_pa` / `policy_l`，而它**正是正在跑的 D5 / B 两个跑数器用的模块** ——
`spawn` 池的 worker 会重新 import。**在长跑途中改它们有污染读数的风险**，故**推迟**。
（见 `docs/v2/progress-log.md` §121.4。）

══ 与 `b_ablation.py` 的关系 ══

同一个 `build` / `evaluate` / 扁平训练器。**差别只在"改什么"**：
- B 改**约束档**（`ABLATION_GROUPS`，五个整组）；
- C 改**单条机制**（一个 `SimConfig` 开关，或 `dt_events`）。

⚠️ 两者的 `base` / `Full` 臂**应当给出逐位相同的数**（同 seed 同参数）——
**这是免费的交叉验证**，每次跑完都该看一眼。

⚠️ 默认写到系统临时目录（项目硬规则 1）。
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

from .env.constraints import ConstraintConfig
from .hrl.flat import flat_step
from .hrl.pool import ChainPool
from .q1_flat import build, constraint_counts, evaluate

DEFAULT_OUT = os.path.join(tempfile.gettempdir(), "c_mechanism.jsonl")

ARMS: dict[str, dict] = {
    "base": {},
    "failover_on": {"sim": {"agv_failover": True}},
    "age_on": {"sim": {"machine_age_failure": True}},
    "no_buffer": {"constraints": {"finite_buffer": False}},
    "dt10": {"dt": 10},
    "dt40": {"dt": 40},
}
"""逐臂的**改动**（空 dict = 与 `base` 同）。

- `sim`：**并进 `SimConfig`** —— 与参考解/奖励权重同源（见 `q1_flat.build` 的 `sim_overrides`）。
- `constraints`：**只覆盖列出的那几个开关**（其余照默认）。
- `dt`：覆盖顶层的 `dt_events`（段回报的 ΔT）。

⚠️⚠️ **`failover_on` / `age_on` 是"加上"不是"拿掉"** —— 这两个开关
（`agv_failover` / `machine_age_failure`）**默认都是 `False`**。
**⟹ 本跑数器测的是"打开它们会怎样"，方向与 `no_buffer` 相反。**

**实测踩过**：本跑数器第一版的臂名照抄了 `experiment-plan.md` §C 的表，
写成 `no_failover`（`{"agv_failover": False}`）—— 那是**空操作**，
读数与 `base` **逐位相同**（97.41 = 97.41），差点当成"这条机制没用"。
**⟹ 那张表对这两个机制写的是"怎么关"，但它们的默认本来就是关的。已改名为 `*_on`。**

**读表时别把 `*_on` 当成 leave-one-out 的一员。**
"""


def run_one(arm: str, seed: int, *, steps: int, n: int, dt_events: int,
            eval_every: int, eval_n: int, rec, workers: int = 0,
            save_dir: str = "", lr: float = 1e-4, env: dict | None = None) -> None:
    """一条 `(臂, 种子)`。"""
    env = dict(env or {})
    spec_arm = ARMS[arm]
    dt = int(spec_arm.get("dt", dt_events))
    sim = dict(spec_arm.get("sim", {}))
    cons_default = ConstraintConfig()
    cons = (ConstraintConfig(**{**cons_default.__dict__, **spec_arm["constraints"]})
            if "constraints" in spec_arm else cons_default)

    spec, net = build(seed, constraints=cons, sim_overrides=sim or None, **env)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    rec({"kind": "start", "arm": arm, "seed": seed, "steps": steps, "n": n,
         "dt_events": dt, "workers": workers,
         # ⚠️ 把**这一臂到底改了什么**记全 —— 臂名（`no_failover`）与内容是两回事。
         "sim_overrides": sim,
         "constraints": {k: bool(v) for k, v in cons.__dict__.items()},
         "n_params": sum(p.numel() for p in net.parameters())})
    rec({"kind": "eval", "arm": arm, "seed": seed, "step": 0,
         **evaluate(net, spec, n=eval_n, seed=0, dt_events=dt)})

    pool = None
    if workers > 1:
        pool = ChainPool(net, spec, n_workers=workers, dt_events=dt)
    t0 = time.perf_counter()
    try:
        for step in range(1, steps + 1):
            out = flat_step(net, spec, optim, arm="B", n=n,
                            seed=seed * 10000 + step, dt_events=dt, pool=pool)
            rec({"kind": "train", "arm": arm, "seed": seed, "step": step,
                 "elapsed_s": time.perf_counter() - t0,
                 "r_mean": float(out["r_mean"]), "loss": float(out["loss"]),
                 "makespan": float(np.mean([r["makespan"] for r in out["runs"]]))})
            if eval_every and step % eval_every == 0:
                rec({"kind": "eval", "arm": arm, "seed": seed, "step": step,
                     **evaluate(net, spec, n=eval_n, seed=0, dt_events=dt)})

        dspec, _ = build(seed, disturb=True, constraints=cons,
                         sim_overrides=sim or None, **env)
        rec({"kind": "final", "arm": arm, "seed": seed,
             "total_s": time.perf_counter() - t0,
             "clean": evaluate(net, spec, n=eval_n, seed=1, dt_events=dt),
             "disturbed": evaluate(net, dspec, n=eval_n, seed=1, dt_events=dt),
             "agv_mtbf": dspec.cfg.agv_mtbf,
             "constraints": constraint_counts(spec)})
        if save_dir:
            p = Path(save_dir) / f"{arm}_s{seed}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                        "arm": arm, "seed": seed, "steps": steps, "n": n,
                        "dt_events": dt, "with_value": False,
                        "sim_overrides": sim,
                        "constraints": {k: bool(v) for k, v in cons.__dict__.items()},
                        # ⚠️ **必须存环境**（2026-10-10 加）—— 见 `q1_flat.resolve_train_env`
                        "env": dict(env),
                        "tau": spec.cfg.tau}, p)
            rec({"kind": "saved", "arm": arm, "seed": seed, "path": str(p)})
    finally:
        if pool is not None:
            pool.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="C：机制消融（设计 §C）")
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    help=f"可选：{' '.join(ARMS)}")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--dt-events", type=int, default=20, help="基准 ΔT（逐臂可覆盖）")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-n", type=int, default=5)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--layout-seed", type=int, default=None,
                    help="⭐ 布局种子。**消融必须固定它**（§130.3）")
    ap.add_argument("--instance", default="mk01")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--pm-interval", type=float, default=None)
    ap.add_argument("--agv-n-failures", type=int, default=0)
    ap.add_argument("--target-failures", type=float, default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    if a.threads:
        torch.set_num_threads(a.threads)

    # ⚠️ **先查后跑**（同 `b_ablation`）：名字不在表里当场报错，不静默 KeyError。
    missing = [x for x in a.arms if x not in ARMS]
    if missing:
        print(f"✗ 这些臂不在表里：{missing}；可用的={list(ARMS)}", file=sys.stderr)
        return 2

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
                        env={"instance": a.instance, "layout_seed": a.layout_seed, "tau": a.tau,
                             "pm_interval": a.pm_interval,
                             "target_failures": a.target_failures,
                             "agv_n_failures": a.agv_n_failures})
    print(f"\n写出：{out}", file=sys.stderr)
    return 0


if __name__ == "__main__":      # ⚠️ Windows `spawn` 需要
    raise SystemExit(main())
