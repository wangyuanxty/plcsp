# -*- coding: utf-8 -*-
"""A 阶段（加权标量化）的正式训练脚本（P2 Task 8）——MK01 上的 300 步级运行**由本脚本承担**。

    PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m13_train_a --inst mk01 --steps 300 --G 8

**为什么是脚本而不是测试**：MK01 上 `joint_chain_step(G=8)` 实测 **18.2 s/步**（Task 7）
⇒ 300 步 ≈ 1.5 h，**不能进单元测试**。验收测试只跑 30 步 × G=4 的小预算
（`plcsp/tests/test_end_to_end_a.py`）。本脚本要能**后台跑**、**断点续**（`--resume`，
ckpt 每 `--save-every` 步落盘）。

⚠️ **固定 seed 跨进程不可复现**（Task 7 实测记录，**论文口径必须照此声明**）：
`roll_chain` / `joint_chain_step` 的**动作采样**走**全局 torch RNG**（`torch.multinomial`），
**不是**种子流——`--seed` 只决定 `joint_chain_step(seed=...)` 里传给仿真的 `seed_chain`
（SimPy 的扰动流），**不锁定动作采样**。故：

- 同一份 ckpt + 同一 `--seed` **跨进程**跑出来的曲线**不会逐位相同**；
- 论文里"同 seed 复跑"的对照（A/B、消融、种子矩阵）必须在**同一进程、同一调用序列**下做
  ——本仓的正例是 `plcsp/tests/test_end_to_end_a.py` 的模块级 fixture（训练前
  `torch.manual_seed(0)`，同一序列一次跑完）；
- 跨进程只保证**统计可比**（多 seed 取均值/分布），不保证逐位复现；
- **出路**（要跨进程逐位复现时）：在训练开始前**自己钉** `torch.manual_seed(...)`——钉完之后
  整条序列（网络初始化 + 每步的动作采样）都被确定，跨进程即可复现。本脚本**故意不钉**：
  `--seed` 只表达"仿真扰动流"这一个语义，把它同时当动作种子会让两种随机来源纠缠在一起。

输出**一律落 `run_dir`（默认 = **仓库根**的 `checkpoints/a_<inst>`，锚 `__file__` 而非 CWD，
故在包目录里执行也不会建出包内 `checkpoints/`；该目录已被 `.gitignore` 排除），不落包目录**：

- `run_dir/metrics.ndjson`——每步一行 `{"step","r","t",…诊断}`（增量追加，可断外部分析）；
  `--eval-every` 打开时另有 `{"step","eval":{"makespan_mean","makespan_std","rule_makespan"}}`；
- `run_dir/ckpt.pt`——每 `--save-every` 步（含最后一步）覆写。

⚠️ `--lr` 只在**第一次** `joint_chain_step` 时生效（Adam 由该函数惰性创建，之后忽略新 lr）；
`--resume` 不恢复优化器状态（ckpt 只存模型权重与步号），续跑会以**当次 `--lr`** 重建 Adam。

⚠️ `--resume` 只在 `run_dir/ckpt.pt` **已存在**时生效（`runner.resume_training` 的既有语义）：
没有 ckpt 就**从头跑**，而 `metrics.ndjson` 是**追加**打开的——此时文件里会出现**重复的 step 号**
（前一段是废弃的尝试）。按 step 取最新一行即可；要干净曲线就换 `--run-dir` 或先删 run_dir。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, pstdev

from .algo.group_rel import roll_chain
from .algo.policy import PolicyNet
from .algo.runner import run_training
from .env.corridors import build_corridor_graph, dock_distance_matrix
from .env.des import SimConfig, reference_makespan, rollout
from .env.instances import load_mk
from .env.layout import sample_layout
from .env.reward import ReferenceObjectives, reward_weights
from .nn.encoder import LayoutEncoder
from .nn.state_emb import norm_context

# 评估种子起点。⚠️ **必须避开训练用过的扰动流**（评审 F1 修的就是这里）：`joint_chain_step`
# 第 s 步第 g 条链用 `seed_chain = s*1000 + g`（`group_rel.py`），故训练流落在
# `[seed0*1000, (seed0+steps)*1000)`。**旧值 10_000 与"第 10 步"的训练流正面相撞**
# （G=8 时 10000..10007 全被第 10 步用过 ⇒ 5 个评估流全是训练流），评估集与训练集不分离会把
# 论文 setup 节那句话写错。取 **10**6**：`steps < 1000` 时恒安全。
EVAL_SEED_BASE = 10 ** 6


def build_setup(inst_name: str, cfg: SimConfig | None = None):
    """(inst, layout, dm, cfg, ctx, policy) —— 布局 seed 固定 0（`joint_chain_step` 的守卫前提）。

    ⚠️ 布局 seed 必须为 0：奖励权重取自 `ReferenceObjectives.of`（固定用 seed_layout=0 的参考
    运行），而 `SimWorld._due_map` 用**该布局**的 seed 取 `M_ref`——两者同源才有一致的口径
    （非 0 种子会被 `joint_chain_step` 入口拒绝）。

    `ctx` 的归一化标度 `m_ref` 取**真实参考 makespan**（`reference_makespan`）而非占位值：
    spec §5.3.1③ 要求归一化用实例静态量，且这个数与交期 `d_j = τ·M_ref` 的 `M_ref` **是同一个**
    ——特征归一化与交期同源（有缓存，不额外付参考运行的代价）。
    """
    inst = load_mk(inst_name)
    c = cfg or SimConfig()
    lay = sample_layout(inst.n_machines, seed=0, n_agv=c.n_agv)
    g = build_corridor_graph(lay)
    pol = PolicyNet(enc=LayoutEncoder(), n_agv=c.n_agv)
    ctx = norm_context(inst, lay, m_ref=reference_makespan(inst, c))
    return inst, lay, dock_distance_matrix(g), c, ctx, pol


def _make_eval_fn(inst, lay, dm, cfg, ctx, seeds: int, rule: float):
    """评估回调：argmax 策略在 `seeds` 个扰动种子上的 makespan（spec §5.3.4 约定 3：J>1 只评估）。"""
    def eval_fn(policy) -> dict:
        ms = [float(roll_chain(inst, lay, dm, cfg, policy, seed=EVAL_SEED_BASE + s,
                               ctx=ctx, sample=False)[1]["makespan"]) for s in range(seeds)]
        return {"makespan_mean": mean(ms), "makespan_std": pstdev(ms) if len(ms) > 1 else 0.0,
                "rule_makespan": rule}
    return eval_fn


def main() -> None:
    ap = argparse.ArgumentParser(description="A（加权标量化）联合链 GRPO 训练")
    ap.add_argument("--inst", default="mk01", help="实例名（mk01..mk10）")
    ap.add_argument("--steps", type=int, default=300, help="训练步数（MK01/G=8 约 18 s/步）")
    ap.add_argument("--G", type=int, default=8, help="组大小（J=1，预算全给 G）")
    ap.add_argument("--lr", type=float, default=3e-4, help="Adam 学习率（仅首步生效）")
    ap.add_argument("--seed", type=int, default=0, help="仿真扰动流种子基（不锁动作采样）")
    ap.add_argument("--run-dir", default=None,
                    help="默认 <仓库根>/checkpoints/a_<inst>（锚 __file__，不是 CWD）")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=0, help="每 N 步评估一次；0 = 不评估")
    ap.add_argument("--eval-seeds", type=int, default=5, help="评估的扰动种子数")
    ap.add_argument("--resume", action="store_true", help="从 run_dir/ckpt.pt 续跑")
    args = ap.parse_args()

    # 默认落**仓库根**的 checkpoints/（锚 `__file__`，不是 CWD）——否则在包目录里执行会建出
    # `plcsp/checkpoints/`，正好破坏本文件 docstring 里"不落包目录"的保证（评审 Minor）。
    run_dir = Path(args.run_dir) if args.run_dir else (
        Path(__file__).resolve().parents[1] / "checkpoints" / f"a_{args.inst}")
    if args.resume and not (run_dir / "ckpt.pt").exists():
        print(f"[m13] ⚠️ --resume 但 {run_dir / 'ckpt.pt'} 不存在：将从 step 0 重跑，且 "
              "metrics.ndjson 是**追加**模式 ⇒ 会出现重复 step 号（按 step 取最新一行）。",
              flush=True)
    inst, lay, dm, cfg, ctx, pol = build_setup(args.inst)
    w = reward_weights(ReferenceObjectives.of(inst, cfg).as_tuple())
    rule = float(rollout(inst, seed_chain=0, cfg=cfg)["makespan"])   # = M_ref（同一运行）
    print(f"[m13] inst={args.inst} 作业{inst.n_jobs}×机台{inst.n_machines} "
          f"车队{cfg.n_agv}｜steps={args.steps} G={args.G} lr={args.lr} seed={args.seed}")
    print(f"[m13] 权重 w={tuple(round(x, 4) for x in w)}（f^ref={ReferenceObjectives.of(inst, cfg).as_tuple()}）")
    print(f"[m13] 规则基线 makespan={rule:.1f}｜run_dir={run_dir.resolve()}")
    print("[m13] ⚠️ 动作采样走全局 torch RNG：跨进程不可逐位复现（见模块 docstring）", flush=True)

    run_training(pol, inst, steps=args.steps,
                 step_kwargs=dict(layout=lay, dm=dm, cfg=cfg, ctx=ctx, w=w, G=args.G, lr=args.lr),
                 seed0=args.seed, run_dir=str(run_dir), save_every=args.save_every,
                 resume=args.resume,
                 eval_fn=(_make_eval_fn(inst, lay, dm, cfg, ctx, args.eval_seeds, rule)
                          if args.eval_every > 0 else None),
                 eval_every=(args.eval_every or None))
    print(f"[m13] 完成：{run_dir / 'metrics.ndjson'}（每步一行）｜{run_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
