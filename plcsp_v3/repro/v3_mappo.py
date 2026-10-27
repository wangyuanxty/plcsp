# -*- coding: utf-8 -*-
"""**MAPPO 基线跑数器**：一个模型 · 随机实例训练 · MK01–10 零样本。

协议与 `a_multi.py` 相同（同一批随机实例、同一批 MK 布局种子、同一奖励口径），
只把训练算法换成 MAPPO（集中式 critic + PPO clip）：

```
python -m plcsp_v3.repro.v3_mappo --n-train 20 --steps 200 --n 8 \
    --dt-events 20 --eval-n 2 --eval-every 25 --seed 0 --workers 8 \
    --ckpt %TEMP%/v3_mappo_200.pt
```

⚠️ `--init-ckpt` 可以载入 v3 主线的 ckpt（此时 `value_head` 是新初始化的）。
⚠️ 本基线与主线**动作空间、状态特征、编码器、奖励完全相同**；差异只在训练算法。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch  # noqa: E402

from plcsp_v3.env.instances import gen_mk_like  # noqa: E402
from plcsp_v3.hrl.mappo import MAPPOConfig, mappo_step  # noqa: E402
from plcsp_v3.q1_flat import build, evaluate  # noqa: E402

MK_NAMES = tuple(f"mk{i:02d}" for i in range(1, 11))
CHAIN_SEED_BASE = 100_000


def _fit(seed: int, insts, layout_base: int, tau: float, n_agv: int):
    """预建 `Spec`（带 critic 的网络只建一次）。"""
    specs, net = [], None
    for i, inst in enumerate(insts):
        sp, nn = build(seed=0, inst_obj=inst, tau=tau, pm_interval=30,
                       target_failures=4, n_agv=n_agv, finite_buffer=True,
                       layout_seed=layout_base + i, reward_norm="minmax",
                       sim_overrides={"due_range": 0.6}, with_value=True)
        specs.append(sp)
        if net is None:
            net = nn
    return specs, net


def _eval_mk(net, mk_specs, eval_n: int, dt_events: int):
    rows = []
    for name, sp in zip(MK_NAMES, mk_specs):
        r = evaluate(net, sp, n=eval_n, seed=1, dt_events=dt_events)
        rows.append((name, sp.inst.n_jobs, sp.inst.n_machines, r))
    n = max(len(rows), 1)
    summary = {
        "makespan": sum(r["makespan"] for *_x, r in rows) / n,
        "energy": sum(r["energy"] for *_x, r in rows) / n,
        "twt": sum(r["twt"] for *_x, r in rows) / n,
    }
    return summary, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MAPPO 基线 · 随机实例训练 · MK 零样本")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n", type=int, default=8, help="每步采几条完整链")
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--n-agv", type=int, default=3)
    ap.add_argument("--dt-events", type=int, default=20)
    ap.add_argument("--eval-n", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0,
                    help="ChainPool 进程数（0/1 = 顺序档；只支持 CPU）")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--ms-only", action="store_true",
                    help="奖励只保留 MS（w=(1,0,0)）")
    ap.add_argument("--init-ckpt", default="")
    ap.add_argument("--ckpt", default="", help="权重保存路径（正式训练必须给）")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    insts = [gen_mk_like(seed=args.seed * 1000 + i)
             for i in range(args.n_train)]
    specs, net = _fit(args.seed, insts, 100, args.tau, args.n_agv)
    if args.init_ckpt:
        obj = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if unexpected:
            raise ValueError(f"init-ckpt 有未知键：{sorted(unexpected)}")
        print(f"  初始化权重：{args.init_ckpt}（missing={len(missing)}）",
              flush=True)
    if args.ms_only:
        specs = [sp._replace(w=(1.0, 0.0, 0.0)) for sp in specs]
        print("  奖励：单目标 MS（w=(1,0,0)）", flush=True)
    torch.manual_seed(args.seed * 1000)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    cfg = MAPPOConfig(clip=args.clip, epochs=args.epochs,
                      minibatch=args.minibatch, ent_coef=args.ent_coef,
                      vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm)

    pool = None
    if args.workers > 1:
        from plcsp_v3.hrl.pool import ChainPool
        pool = ChainPool(net, specs[0], n_workers=args.workers,
                         dt_events=args.dt_events)
        print(f"  并行链池：{args.workers} 进程（每进程 1 线程）", flush=True)

    print(f"**MAPPO 基线** · seed={args.seed} · {args.n_train} 个随机实例 · "
          f"{args.steps} 步 · n={args.n} · 参数 {sum(p.numel() for p in net.parameters())}\n",
          flush=True)
    for i, sp in enumerate(specs):
        print(f"  train[{i}] {sp.inst.n_jobs}×{sp.inst.n_machines} · "
              f"{sum(len(o) for o in sp.inst.jobs)} 工序", flush=True)

    mk_specs = None
    if not args.no_eval:
        mk_specs = []
        for i, name in enumerate(MK_NAMES):
            sp, _ = build(seed=0, instance=name, tau=None, pm_interval=30,
                          target_failures=4, n_agv=args.n_agv,
                          finite_buffer=True, layout_seed=200 + i,
                          reward_norm="minmax")
            mk_specs.append(sp)
        print(f"  MK 评测实例：{len(mk_specs)} 个（layout_seed 200+i）", flush=True)

    last_eval = None
    try:
        for step in range(1, args.steps + 1):
            sp = specs[(step - 1) % len(specs)]
            if pool is not None:
                pool.spec = sp
            t1 = time.perf_counter()
            out = mappo_step(net, sp, optim, n=args.n,
                             seed=args.seed * 10000 + step,
                             dt_events=args.dt_events, pool=pool,
                             chain_seed=CHAIN_SEED_BASE + args.seed * 1000 + step,
                             cfg=cfg)
            if step == 1 or step % max(1, args.steps // 10) == 0:
                print(f"  step {step:>3}/{args.steps}  "
                      f"{time.perf_counter() - t1:7.2f}s  "
                      f"实例 {sp.inst.n_jobs}×{sp.inst.n_machines}  "
                      f"loss={out['loss']:.4f}  R={out['r_mean']:.4f}  "
                      f"V={out['value_loss']:.4f}  H={out['entropy']:.3f}",
                      flush=True)
            if (mk_specs is not None and args.eval_every > 0
                    and step % args.eval_every == 0):
                te = time.perf_counter()
                summary, eval_rows = _eval_mk(net, mk_specs, args.eval_n,
                                             args.dt_events)
                last_eval = (step, summary, eval_rows)
                print(f"  eval {step:>4}  MS={summary['makespan']:.1f}  "
                      f"E={summary['energy']:.2f}  TWT={summary['twt']:.1f}  "
                      f"({time.perf_counter() - te:.1f}s)", flush=True)
                if args.ckpt:
                    base, ext = os.path.splitext(args.ckpt)
                    torch.save({"state_dict": net.state_dict(),
                                "args": vars(args), "step": step},
                               f"{base}_e{step:04d}{ext}")
    finally:
        if pool is not None:
            pool.close()

    print(f"\n  训练用时 {time.perf_counter() - t0:.1f}s\n", flush=True)
    if args.ckpt:
        torch.save({"state_dict": net.state_dict(), "args": vars(args),
                    "mappo": cfg.__dict__}, args.ckpt)
        print(f"  ckpt 已存 {args.ckpt}\n", flush=True)
    else:
        print("  ⚠️ 未给 `--ckpt`：本次训练没有保存权重。", flush=True)
    if args.no_eval:
        print("  （--no-eval：跳过 MK01–10 评测）")
        return 0
    if last_eval is not None and last_eval[0] == args.steps:
        _, summary, rows = last_eval
    else:
        summary, rows = _eval_mk(net, mk_specs, args.eval_n, args.dt_events)
    print("  ── MK01–10 零样本（未参与训练）──")
    print(f"  {'实例':<6}{'工件':>4}{'机台':>4}{'MS':>9}{'能耗':>9}{'拖期':>9}")
    print("  " + "-" * 45)
    for name, nj, nm, r in rows:
        print(f"  {name:<6}{nj:>4}{nm:>4}{r['makespan']:>9.1f}"
              f"{r['energy']:>9.2f}{r['twt']:>9.1f}", flush=True)
    print("  " + "-" * 45)
    print(f"  {'均值':<6}{'':>8}{summary['makespan']:>9.1f}"
          f"{summary['energy']:>9.2f}{summary['twt']:>9.1f}")
    print(f"\n  总用时 {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
