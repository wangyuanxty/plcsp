# -*- coding: utf-8 -*-
"""**一个模型 · 跨实例训练 · MK01–10 零样本评测**（2026-10-18）。

协议：

```
训练：随机实例（MK 范围）
      工件 10–20 · 机台 4–15 · 工序 3–15 · 候选 1–6 · 工时 1–19
      每实例固定 layout_seed
测试：MK01–MK10（完全没见过的实例）
      每实例用没见过的 layout_seed
```

⚠️ 这是**多实例/零样本**实验，不是逐实例训 10 个模型。
⚠️ 训练/测试的布局种子分开（100+i / 200+i）。
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from .env.instances import gen_mk_like
from .hrl.flat import flat_step
from .hrl.train import train_step
from .q1_flat import build, evaluate

MK_NAMES = tuple(f"mk{i:02d}" for i in range(1, 11))
CHAIN_SEED_BASE = 100_000


def _fit(seed: int, insts, layout_base: int, tau: float, n_agv: int,
         constraints=None, seq_ma: bool = False):
    """预建 `Spec`；网络只取第一个，优化器在全部建完后由调用方建。"""
    specs, net = [], None
    for i, inst in enumerate(insts):
        sp, nn = build(seed=0, inst_obj=inst, tau=tau, pm_interval=30,
                       target_failures=4, n_agv=n_agv, finite_buffer=True,
                       layout_seed=layout_base + i, reward_norm="minmax",
                       sim_overrides={"due_range": 0.6},
                       constraints=constraints, seq_ma=seq_ma)
        specs.append(sp)
        if net is None:
            net = nn
    return specs, net


def _eval_mk(net, mk_specs, args, fixed_rules=None, pa_limit=None):
    """greedy 评测 MK01–10 → `(summary, rows)`。**同一个评测 seed**，曲线可比。"""
    rows = []
    for name, sp in zip(MK_NAMES, mk_specs):
        r = evaluate(net, sp, n=args.eval_n, seed=1,
                     dt_events=args.dt_events, force_top1=args.force_top1,
                     fixed_rules=fixed_rules, pa_limit=pa_limit,
                     exact_rule=args.exact_rule, tie_streams=args.tie_streams)
        rows.append((name, sp.inst.n_jobs, sp.inst.n_machines, r))
    n = max(len(rows), 1)
    summary = {
        "makespan": sum(r["makespan"] for *_x, r in rows) / n,
        "energy": sum(r["energy"] for *_x, r in rows) / n,
        "twt": sum(r["twt"] for *_x, r in rows) / n,
    }
    return summary, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="一个模型 · 随机实例训练 · MK01–10 零样本")
    ap.add_argument("--arm", choices=("flat", "hrl"), default="hrl")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--n", type=int, default=4, help="组大小（HRL 的 G=M=n）")
    ap.add_argument("--dt-events", type=int, default=20)
    ap.add_argument("--eval-n", type=int, default=2)
    ap.add_argument("--tie-streams", type=int, default=1,
                    help="每个评测场景的平局流条数（默认 1 = 旧口径）。"
                         "实测平局噪声 std≈17.8 min ⟹ 测 8 min 需要 ≈20 条")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="每 N 步做一次 MK01–10 greedy 评测；0 = 只在结束时评测")
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=2.0,
                    help="随机实例的交期松紧（MK 评测仍用逐实例标定表）")
    ap.add_argument("--n-agv", type=int, default=3)
    ap.add_argument("--workers", type=int, default=0,
                    help="HRL 段池进程数；0/1 = 顺序档（默认）。只对 --arm hrl 生效")
    ap.add_argument("--span", type=int, default=1,
                    help="MA 的段跨几个 ΔT；默认 1。只对 --arm hrl 生效")
    ap.add_argument("--terminal-ma", action="store_true",
                    help="phase A（MA 规则比较）跑到局末；phase B 仍用 span。"
                         "只对 --arm hrl 生效")
    ap.add_argument("--per-head", action="store_true",
                    help="阶段 A 逐头分组：每次只变一个头（PA 12 / LA 3 / RA 3），"
                         "另外两个头钉死。只对 --arm hrl 生效")
    ap.add_argument("--no-maintenance", action="store_true",
                    help="关掉保养约束（无 RA 决策点）。必须与 --per-head 同用")
    ap.add_argument("--pa-only", action="store_true",
                    help="只训 MA 的 PA 规则头；LA 固定为 --fixed-la；RA 关")
    ap.add_argument("--la-only", action="store_true",
                    help="只训 MA 的 LA 规则头；PA 固定为 --fixed-pa；RA 关")
    ap.add_argument("--heads", nargs="+", choices=("PA", "LA", "RA"), default=[],
                    help="只训这些规则头（逐头分组）；未列出的头固定。"
                         "例：--heads PA LA")
    ap.add_argument("--train-encoder", action="store_true",
                    help="与 --heads/--pa-only/--la-only 同用：把 encoder 也解冻")
    ap.add_argument("--train-trunk", action="store_true",
                    help="与 --heads/--pa-only/--la-only 同用：把 select trunk 也解冻")
    ap.add_argument("--seq-ma", action="store_true",
                    help="MA 改成顺序条件：PA → LA（LA 输入拼 PA embedding）；先不加 RA")
    ap.add_argument("--factorial", action="store_true",
                    help="PA×LA 全因子组（诊断）：需 --per-head --heads PA LA "
                         "--no-maintenance")
    ap.add_argument("--pa-limit", type=int, default=12,
                    help="PA 可选规则数（9 = 去掉 EDD 三条；默认 12）")
    ap.add_argument("--fixed-la", default="LATEST_IN_TIME",
                    help="--pa-only 时固定的 LA 规则（默认 LATEST_IN_TIME）")
    ap.add_argument("--fixed-pa", default="EAJ+SMPT",
                    help="--la-only 时固定的 PA 规则（默认 EAJ+SMPT）")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="学习率（默认 1e-4）")
    ap.add_argument("--enc-lr-scale", type=float, default=1.0,
                    help="encoder 学习率倍率（默认 1.0 = 同 LR）。"
                         "只训 MA 时 encoder 可能被 MA 目标带偏；0.1 是折中档")
    ap.add_argument("--init-ckpt", default="",
                    help="训练前加载权重（如只训 MA 时加载已有下层）")
    ap.add_argument("--force-top1", action="store_true",
                    help="下层取规则 top-1，绕过打分头（诊断 / 只训 MA）")
    ap.add_argument("--exact-rule", action="store_true",
                    help="PA/LA 直接调 rules.apply_pa_rule + 环境 tie_rng（诊断）")
    ap.add_argument("--adv-norm", choices=("z", "demean"), default="z",
                    help="组内优势口径：z = (R−μ)/σ（默认）；"
                         "demean = R−μ（保留幅度，见 progress-log §38.26）")
    ap.add_argument("--ms-only", action="store_true",
                    help="奖励只保留 MS（w=(1,0,0)）")
    ap.add_argument("--w", nargs=3, type=float, default=None,
                    metavar=("W_MS", "W_E", "W_TWT"),
                    help="自定义三目标权重（如 `--w 1 1 0` = MS+Energy）。"
                         "⚠️ 训练用的是**原始量纲**：MS ~20、Energy ~0.5 ⟹ "
                         "w=(1,1,0) 实际仍由 MS 主导（见 progress-log §38.24）")
    ap.add_argument("--ma-only", action="store_true",
                    help="只训 MA 的三个 select 头 + encoder；下层打分头冻结；跳过 phase B")
    ap.add_argument("--reset-select", action="store_true",
                    help="加载 init-ckpt 后重置三个 select 头（不从塌缩的 MA 继续）")
    ap.add_argument("--ckpt", default="",
                    help="权重保存路径；空 = 不保存（⚠️ 正式训练必须给）")
    ap.add_argument("--no-eval", action="store_true",
                    help="跳过 MK01–10 零样本评测（只测训练速度/正确性）")
    args = ap.parse_args(argv)

    from .hrl.rules import (LA_RULE_NAMES, N_PA_RULES, PA_RULE_NAMES)
    if args.pa_only and args.la_only:
        raise ValueError("--pa-only 与 --la-only 只能二选一")
    if args.heads and (args.pa_only or args.la_only):
        raise ValueError("--heads 与 --pa-only/--la-only 不能同用")
    if args.factorial:
        if (not args.per_head or tuple(args.heads) != ("PA", "LA")
                or not args.no_maintenance):
            raise ValueError("--factorial 需要 --per-head --heads PA LA "
                             "--no-maintenance")
    if not 1 <= args.pa_limit <= N_PA_RULES:
        raise ValueError(f"--pa-limit 应在 1..{N_PA_RULES}，收到 {args.pa_limit}")
    if args.heads:
        only_heads = tuple(args.heads)
    elif args.pa_only:
        only_heads = ("PA",)
    elif args.la_only:
        only_heads = ("LA",)
    else:
        only_heads = None
    fixed_rules = None
    if only_heads is not None:
        if args.arm != "hrl" or (not args.per_head and not args.seq_ma):
            raise ValueError("--pa-only / --la-only / --heads 需要 "
                             "--arm hrl --per-head（或用 --seq-ma 联合组）")
        if "RA" in only_heads and args.no_maintenance:
            raise ValueError("RA 在训练头里时不能 --no-maintenance")
        if "RA" not in only_heads and not args.no_maintenance:
            raise ValueError("未训练 RA 时必须 --no-maintenance")
        fixed_rules = {}
        if "PA" not in only_heads:
            fixed_rules["PA"] = PA_RULE_NAMES.index(args.fixed_pa)
        if "LA" not in only_heads:
            fixed_rules["LA"] = LA_RULE_NAMES.index(args.fixed_la)
        if "RA" not in only_heads:
            fixed_rules["RA"] = 0
        if not fixed_rules:
            fixed_rules = None
        print(f"  只训 {only_heads} 规则头；固定 {fixed_rules}", flush=True)
    only_ma = args.ma_only or only_heads is not None
    if only_ma and args.arm != "hrl":
        raise ValueError("--ma-only / --pa-only / --la-only / --heads "
                         "只支持 --arm hrl")

    if (args.no_maintenance and not args.per_head and not args.seq_ma):
        # ⚠️ **快速失败**：联合组会把 RA 的 logp 也带进 loss，而 RA 决策点不存在
        # ⟹ RA 头拿一个恒 0 的优势，白更新。逐头分组 / 顺序条件组会跳过 RA。
        raise ValueError("--no-maintenance 必须与 --per-head 同用（或用 --seq-ma）")
    cons = None
    if args.no_maintenance:
        from .env.constraints import ConstraintConfig
        cons = ConstraintConfig(maintenance=False)
        print("  保养 RA：关（无 RA 决策点）", flush=True)

    t0 = time.perf_counter()
    insts = [gen_mk_like(seed=args.seed * 1000 + i)
             for i in range(args.n_train)]
    specs, net = _fit(args.seed, insts, 100, args.tau, args.n_agv,
                      constraints=cons, seq_ma=args.seq_ma)
    if args.init_ckpt:
        obj = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
        if args.seq_ma:
            from .hrl.agents import HierarchicalPolicy
            old = HierarchicalPolicy()
            old.load_state_dict(sd)
            net.enc.load_state_dict(old.enc.state_dict())
            net.scorer_pa.load_state_dict(old.scorer_pa.state_dict())
            net.scorer_la.load_state_dict(old.scorer_la.state_dict())
            net.scorer_ra.load_state_dict(old.scorer_ra.state_dict())
            net.select.trunk.load_state_dict(old.select.trunk.state_dict())
            net.select.heads["PA"].load_state_dict(old.select.heads["PA"].state_dict())
            h = int(old.select.trunk[0].out_features)
            with torch.no_grad():
                net.select.heads["LA"].weight[:, :h].copy_(
                    old.select.heads["LA"].weight)
                net.select.heads["LA"].bias.copy_(old.select.heads["LA"].bias)
                net.select.heads["LA"].weight[:, h:].zero_()
                net.select.heads["RA"].load_state_dict(
                    old.select.heads["RA"].state_dict())
            print("  初始化：encoder/lower + PA 头；LA 条件列置零", flush=True)
        else:
            net.load_state_dict(sd)
        print(f"  初始化权重：{args.init_ckpt}", flush=True)
    if args.reset_select:
        if args.seq_ma:
            raise ValueError("--seq-ma 暂不支持 --reset-select")
        from .hrl.agents import RuleSelectHeads
        from .hrl.rules import N_LA_RULES, N_PA_RULES, N_PM_RULES
        net.select = RuleSelectHeads(
            net.enc.d_model,
            {"PA": N_PA_RULES, "LA": N_LA_RULES, "RA": N_PM_RULES},
            in_dim=4 * net.enc.d_model)
        print("  重置 select 头（不从塌缩的 MA 继续）", flush=True)
    if args.ms_only:
        specs = [sp._replace(w=(1.0, 0.0, 0.0)) for sp in specs]
        print("  奖励：单目标 MS（w=(1,0,0)）", flush=True)
    if args.w is not None:
        if args.ms_only:
            raise ValueError("--w 与 --ms-only 不能同用")
        specs = [sp._replace(w=tuple(args.w)) for sp in specs]
        print(f"  奖励：自定义权重 w={tuple(args.w)}"
              "（⚠️ 原始量纲，MS 仍主导）", flush=True)
    if only_ma:
        for p in net.parameters():
            p.requires_grad_(False)
        if only_heads is not None:
            # ⚠️ **只训目标规则头**：encoder / select trunk / 其余头 / 下层全冻结。
            for h in only_heads:
                for p in net.select.heads[h].parameters():
                    p.requires_grad_(True)
            if args.train_encoder:
                for p in net.enc.parameters():
                    p.requires_grad_(True)
            if args.train_trunk:
                for p in net.select.trunk.parameters():
                    p.requires_grad_(True)
            if args.seq_ma:
                # 顺序条件的 PA/LA 规则 embedding 也要训。
                for p in net.select.rule_emb_pa.parameters():
                    p.requires_grad_(True)
                for p in net.select.rule_emb_la.parameters():
                    p.requires_grad_(True)
            extra = (" + encoder" if args.train_encoder else "") + (
                " + trunk" if args.train_trunk else "")
            if args.seq_ma:
                extra += " + rule_emb"
            print(f"  只训 {only_heads} 规则头{extra}（其余全冻结）", flush=True)
        else:
            # ⚠️ **encoder 属于 MA 的表征**：只训 MA 时它必须可训。
            #    冻结的只是下层的三个打分头（`scorer_*`）。
            for p in net.enc.parameters():
                p.requires_grad_(True)
            for p in net.select.parameters():
                p.requires_grad_(True)
            print("  只训 MA：encoder + select；下层打分头冻结", flush=True)
    if args.per_head:
        if args.arm != "hrl":
            raise ValueError("--per-head 只支持 --arm hrl")
        print("  阶段 A：逐头分组", flush=True)
    if args.factorial:
        print("  阶段 A：PA×LA 全因子组（27 条/顶层点）", flush=True)
    torch.manual_seed(args.seed * 1000)
    if only_heads is not None:
        params = [p for p in net.parameters() if p.requires_grad]
    elif args.ma_only:
        params = list(net.enc.parameters()) + list(net.select.parameters())
    else:
        params = list(net.parameters())
    if args.enc_lr_scale <= 0.0:
        raise ValueError(f"--enc-lr-scale 必须 > 0，收到 {args.enc_lr_scale}")
    if args.enc_lr_scale == 1.0:
        optim = torch.optim.Adam(params, lr=args.lr)
    else:
        # ⚠️ **只降 encoder 的 LR**：单训 MA 时 encoder 只被 MA 目标更新，
        # 可能把全模型学到的表征带偏（实测：冻结 encoder 的 414.2 vs
        # 全 LR encoder 的 521.3）。0.1 是折中档，不是冻结。
        enc_ids = {id(p) for p in net.enc.parameters()}
        enc = [p for p in params if id(p) in enc_ids]
        rest = [p for p in params if id(p) not in enc_ids]
        groups = ([{"params": enc, "lr": args.lr * args.enc_lr_scale}] if enc else [])
        if rest:
            groups.append({"params": rest, "lr": args.lr})
        optim = torch.optim.Adam(groups, lr=args.lr)
        print(f"  encoder LR 倍率：{args.enc_lr_scale}", flush=True)

    pool = None
    if args.workers > 1:
        if args.arm == "flat":
            print("  ⚠️ `--workers` 只对 `--arm hrl` 生效；flat 仍走顺序档",
                  flush=True)
        else:
            from .hrl.pool import SegmentPool
            pool = SegmentPool(net, specs[0], n_workers=args.workers,
                               dt_events=args.dt_events)
            print(f"  并行段池：{args.workers} 进程（每进程 1 线程）", flush=True)

    print(f"**跨实例训练** · arm={args.arm} · seed={args.seed} · "
          f"{args.n_train} 个随机实例 · {args.steps} 步 · n={args.n} · "
          f"参数 {sum(p.numel() for p in net.parameters())}\n", flush=True)
    for i, sp in enumerate(specs):
        print(f"  train[{i}] {sp.inst.n_jobs}×{sp.inst.n_machines} · "
              f"{sum(len(o) for o in sp.inst.jobs)} 工序", flush=True)

    # MK 评测实例**预建一次**：per-eval 重 build 10 个实例会白花 ~10 s/次。
    mk_specs = None
    if not args.no_eval:
        mk_specs = []
        for i, name in enumerate(MK_NAMES):
            sp, _ = build(seed=0, instance=name, tau=None, pm_interval=30,
                          target_failures=4, n_agv=args.n_agv,
                          finite_buffer=True, layout_seed=200 + i,
                          reward_norm="minmax", constraints=cons,
                          seq_ma=args.seq_ma)
            mk_specs.append(sp)
        print(f"  MK 评测实例：{len(mk_specs)} 个（layout_seed 200+i）",
              flush=True)
    last_eval = None
    if mk_specs is not None and args.eval_every > 0:
        print(f"\n  ── eval curve（greedy, MK01–10, eval_n={args.eval_n}, "
              f"every {args.eval_every} steps）──", flush=True)

    try:
        for step in range(1, args.steps + 1):
            sp = specs[(step - 1) % len(specs)]
            if pool is not None:
                pool.spec = sp          # ⚠️ 池的 spec 必须与本步一致
            t1 = time.perf_counter()
            cs = CHAIN_SEED_BASE + args.seed * 1000 + step
            if args.arm == "flat":
                out = flat_step(net, sp, optim, arm="B", n=args.n,
                                seed=args.seed * 10000 + step,
                                dt_events=args.dt_events, chain_seed=cs)
                row = f"loss={out['loss'] / args.n:.4f}"
            else:
                out = train_step(net, sp, optim, M=args.n, G=args.n,
                                 seed=args.seed * 10000 + step,
                                 dt_events=args.dt_events, chain_seed=cs,
                                 pool=pool, span=args.span,
                                 terminal_ma=args.terminal_ma,
                                 force_top1=args.force_top1,
                                 ma_only=only_ma,
                                 per_head=args.per_head,
                                 only_head=only_heads,
                                 fixed_rules=fixed_rules,
                                 pa_limit=args.pa_limit,
                                 factorial=args.factorial,
                                 exact_rule=args.exact_rule,
                                 adv_scale=(args.adv_norm == "z"))
                if only_ma:
                    row = f"lossA={out['A']['loss']:.4f}"
                else:
                    row = (f"lossA={out['A']['loss']:.4f} "
                           f"lossB={out['B']['loss']:.4f}")
            if step == 1 or step % max(1, args.steps // 10) == 0:
                print(f"  step {step:>3}/{args.steps}  "
                      f"{time.perf_counter() - t1:7.2f}s  "
                      f"实例 {sp.inst.n_jobs}×{sp.inst.n_machines}  {row}",
                      flush=True)
            if (mk_specs is not None and args.eval_every > 0
                    and step % args.eval_every == 0):
                te = time.perf_counter()
                summary, eval_rows = _eval_mk(net, mk_specs, args,
                                              fixed_rules=fixed_rules,
                                              pa_limit=args.pa_limit)
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
        torch.save({"state_dict": net.state_dict(), "args": vars(args)},
                   args.ckpt)
        print(f"  ckpt 已存 {args.ckpt}\n", flush=True)
    else:
        print("  ⚠️ 未给 `--ckpt`：本次训练没有保存权重。"
              "正式训练必须给路径。\n", flush=True)
    if args.no_eval:
        print("  （--no-eval：跳过 MK01–10 评测）")
        return 0
    if last_eval is not None and last_eval[0] == args.steps:
        _, summary, rows = last_eval
    else:
        summary, rows = _eval_mk(net, mk_specs, args,
                                 fixed_rules=fixed_rules, pa_limit=args.pa_limit)
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
