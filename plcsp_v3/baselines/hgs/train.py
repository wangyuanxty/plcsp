# -*- coding: utf-8 -*-
"""HGS 训练与评测（原文 Alg.1：REINFORCE + greedy rollout baseline）。

**原文的超参（逐字，§V-A2）**：Adam `lr = 2e-4`、batch `B = 50`、
每 epoch `E' = 1000` 回合、共 `E = 1000` epoch；每 20 epoch 重生成一批 B 个实例。

⚠️ **代价**：`1000 × 1000 = 1,000,000` 回合/实例；每个回合还要跑一次贪心 rollout 做 baseline
⟹ 约 **2,000,000 次图编码**。本机 CPU 单回合实测见 `--help` 打印；
**本仓的机时预算装不下**（如实报，不偷改方法）。

CLI 默认值是**小预算自测**，不是论文协议。跑论文协议要显式给 `--epochs 1000 --episodes 1000
--batch 50`——先看打印的代价外推，再决定。
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from ...env.instances import Instance
from .env import FjsptEnv
from .model import HgsNet

# 原文 §V-A2 的协议常数（照抄，供代价外推；不是 CLI 默认值）
PAPER = {"epochs": 1000, "episodes_per_epoch": 1000, "batch": 50, "regenerate_every": 20,
         "lr": 2e-4}


def gen_fjspt(n_jobs: int, n_machines: int, n_agv: int, rng: np.random.Generator) -> Instance:
    """原文 §V-A1 的合成实例生成器。

    `n_i ~ U(0.8m, 1.2m)`（取整）、`|M_ij| ~ U(1, m)`、`T̄p_ij ~ U(1,30)`、
    `Tp_ijk ~ U(0.8 T̄p, 1.2 T̄p)`、`T̄t_kk' ~ U(1,20)`、`Tt_kk' ~ U(0.8 T̄t, 1.2 T̄t)`。
    行程时间矩阵含装卸站（下标 0），其行/列同样按上式采样（原文未区分 LU）。
    """
    jobs = []
    for _ in range(n_jobs):
        n_ops = int(round(rng.uniform(0.8 * n_machines, 1.2 * n_machines)))
        job = []
        for _ in range(max(n_ops, 1)):
            n_alt = int(rng.integers(1, n_machines + 1))
            machs = rng.choice(n_machines, size=n_alt, replace=False)
            t_bar = float(rng.uniform(1.0, 30.0))
            job.append([(int(k), float(rng.uniform(0.8 * t_bar, 1.2 * t_bar)))
                        for k in sorted(machs)])
        jobs.append(job)
    size = n_machines + 1
    trans = np.zeros((size, size))
    for a in range(size):
        for b in range(size):
            if a != b:
                t_bar = float(rng.uniform(1.0, 20.0))
                trans[a, b] = float(rng.uniform(0.8 * t_bar, 1.2 * t_bar))
    return Instance(n_jobs=n_jobs, n_machines=n_machines, jobs=jobs,
                    source=f"synthetic n={n_jobs} m={n_machines} v={n_agv}",
                    transport="matrix", trans_time_full=trans)


def rollout(env: FjsptEnv, net: HgsNet, *, greedy: bool, seed: int | None = None) -> dict:
    """一回合。`greedy=True` 走 argmax（baseline）；否则按分布采样。

    ⚠️ **不加 `no_grad` 装饰器**：训练要采样回合的 logp 带梯度。贪心/eval 的调用方自己包
    `torch.no_grad()`（见 `train` 与 `evaluate`）。
    """
    if seed is not None:
        torch.manual_seed(seed)
    h_glimpse = torch.zeros(net.cfg["dh"])
    logp_sum = torch.zeros(())
    ent_sum = torch.zeros(())
    returns = 0.0
    steps = 0
    while not env.finished:
        if not env.has_action():
            env.advance()
            continue
        g = env.graph()
        h_op, h_mach, h_veh, edge = net.encode(g)
        if greedy:
            act = net.decode_greedy(h_op, h_mach, h_veh, edge, g, h_glimpse)
        else:
            act = net.decode(h_op, h_mach, h_veh, edge, g, h_glimpse)
            logp_sum = logp_sum + act["logp_op"] + act["logp_mach"] + act["logp_veh"]
            ent_sum = ent_sum + act["ent_op"] + act["ent_mach"] + act["ent_veh"]
        h_glimpse = act["h_glimpse"]
        j, o = env.flat_to_job_op(act["op_i"])
        returns += env.step(j, o, act["mach_k"], act["veh_u"])
        steps += 1
    return {"makespan": env.makespan, "return": float(returns), "steps": steps,
            "logp_sum": logp_sum, "ent_sum": ent_sum}


def train(instances: list[Instance], n_agv: int, *, epochs: int, episodes_per_epoch: int,
          batch: int, lr: float, seed: int, log_every: int = 1, log=print) -> dict:
    """原文 Alg.1 的训练循环。`instances` = 当前这一代 B 个实例（每 20 epoch 重生成）。"""
    torch.manual_seed(seed)
    net = HgsNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n_updates = 0
    t0 = time.perf_counter()
    last = {"makespan": float("nan"), "return": float("nan")}
    for epoch in range(1, epochs + 1):
        epi = 0
        while epi < episodes_per_epoch:
            loss = torch.zeros(())
            ms = []
            for b in range(batch):
                env = FjsptEnv(instances[b % len(instances)], n_agv)
                out = rollout(env, net, greedy=False)             # 采样（带梯度）
                with torch.no_grad():
                    base = rollout(FjsptEnv(instances[b % len(instances)], n_agv), net,
                                   greedy=True)                   # greedy rollout baseline
                adv = out["return"] - base["return"]              # 原文 Eq.27
                loss = loss - adv * out["logp_sum"]
                ms.append(out["makespan"])
                last = {"makespan": float(np.mean(ms)), "return": out["return"]}
            loss = loss / batch
            opt.zero_grad()
            loss.backward()
            opt.step()
            n_updates += 1
            epi += batch
        if epoch % log_every == 0:
            wall = time.perf_counter() - t0
            log(f"epoch {epoch}/{epochs}｜update {n_updates}｜训练 makespan {last['makespan']:.1f}"
                f"｜{wall:.1f}s（{wall / n_updates:.1f}s/update）")
    return {"net": net, "n_updates": n_updates, "wall_s": time.perf_counter() - t0}


def evaluate(inst: Instance, n_agv: int, net: HgsNet, *, seed: int = 0) -> dict:
    """贪心与采样各跑一回合（评测用贪心；采样给方差参考）。"""
    with torch.no_grad():
        g = rollout(FjsptEnv(inst, n_agv), net, greedy=True)
        s = rollout(FjsptEnv(inst, n_agv), net, greedy=False, seed=seed)
    return {"makespan_greedy": g["makespan"], "makespan_sample": s["makespan"],
            "steps": g["steps"]}


def time_one_rollout(inst: Instance, n_agv: int, seed: int = 0) -> float:
    """单回合墙钟（未训练网络）——代价外推用。"""
    net = HgsNet()
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    with torch.no_grad():
        rollout(FjsptEnv(inst, n_agv), net, greedy=True)
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description="HGS 对位基线（原文 Alg.1）")
    ap.add_argument("--inst", default="mk01", help="评测实例（MKT 口径）")
    ap.add_argument("--n-agv", type=int, default=None, help="默认 = 该实例的 v=m")
    ap.add_argument("--train-inst", default="synthetic",
                    help='"synthetic" = 按原文 §V-A1 生成；也可给 MK 名')
    ap.add_argument("--epochs", type=int, default=1, help=f"论文协议 = {PAPER['epochs']}")
    ap.add_argument("--episodes", type=int, default=5,
                    help=f"每 epoch 回合数（论文协议 = {PAPER['episodes_per_epoch']}）")
    ap.add_argument("--batch", type=int, default=2, help=f"论文协议 = {PAPER['batch']}")
    ap.add_argument("--lr", type=float, default=PAPER["lr"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ...env.mkt import load_mkt
    mkt = load_mkt(args.inst)
    inst = mkt.base
    v = mkt.n_agv if args.n_agv is None else args.n_agv

    per_roll = time_one_rollout(inst, v, seed=args.seed)
    n_ep = PAPER["epochs"] * PAPER["episodes_per_epoch"]
    print(f"⚠️ 原文协议 = {PAPER['epochs']} epoch × {PAPER['episodes_per_epoch']} 回合 "
          f"= {n_ep:,} 回合/规模；每个梯度步还要 B={PAPER['batch']} 次贪心 rollout 做 baseline。")
    print(f"   本机单回合（MK01，{inst.n_jobs} 作业 / {inst.n_machines} 机台 / v={v}，未训练）"
          f"≈ {per_roll:.2f} s ⟹ 原文协议 ≈ {n_ep * 2 * per_roll / 86400:.1f} 天/实例。")

    if args.train_inst == "synthetic":
        rng = np.random.default_rng(args.seed)
        n_syn = max(args.batch, 1)
        pool = [gen_fjspt(inst.n_jobs, inst.n_machines, v, rng) for _ in range(n_syn)]
        print(f"训练实例 = 合成（原文 §V-A1），n={inst.n_jobs} m={inst.n_machines} v={v}，"
              f"池 {n_syn} 个")
    else:
        pool = [load_mkt(args.train_inst).base]
        print(f"训练实例 = {args.train_inst}")

    print(f"本轮预算：{args.epochs} epoch × {args.episodes} 回合，B={args.batch}，"
          f"lr={args.lr}（CLI 小预算，不是论文协议）")
    out = train(pool, v, epochs=args.epochs, episodes_per_epoch=args.episodes,
                batch=args.batch, lr=args.lr, seed=args.seed)
    ev = evaluate(inst, v, out["net"], seed=args.seed)
    print(f"评测（{args.inst}，MKT，v={v}）：贪心 makespan = {ev['makespan_greedy']:.1f}｜"
          f"采样 makespan = {ev['makespan_sample']:.1f}")
    print(f"训练墙钟 = {out['wall_s']:.1f} s（{out['n_updates']} 个梯度步，"
          f"{out['wall_s'] / max(out['n_updates'], 1):.1f} s/步）")


if __name__ == "__main__":
    main()
