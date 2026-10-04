# -*- coding: utf-8 -*-
"""A 阶段（加权标量化）的正式训练脚本（P2 Task 8）——MK01 上的 300 步级运行**由本脚本承担**。

    PYTHONIOENCODING=utf-8 D:/anaconda/python.exe -m plcsp.m13_train_a --inst mk01 --steps 300 --G 8

**为什么是脚本而不是测试**：MK01 上 `joint_chain_step(G=8)` 实测 **18.2 s/步**（Task 7）
⇒ 300 步 ≈ 1.5 h，**不能进单元测试**。验收测试只跑 30 步 × G=4 的小预算
（`plcsp/tests/test_end_to_end_a.py`）。本脚本要能**后台跑**、**断点续**（`--resume`，
ckpt 每 `--save-every` 步落盘）。

⚠️ **`--seed` 的语义（评审 I-3 修后：可复现）**——`--seed` 同时锁定三条随机来源：

1. **网络初始化**：`main()` 入口 `torch.manual_seed(args.seed)`（在 `build_training_setup`
   构造 `PolicyNet` **之前**）；
2. **仿真扰动流**：第 s 步第 g 条链的 `seed_chain = (args.seed + s) * SEED_STRIDE + g`
   （`runner.py` 传 `seed0+s`；步长常量在 `group_rel.py`）；
3. **动作采样**：`joint_chain_step` 用 `torch.Generator(device=策略设备).manual_seed(seed0+s)`
   采样并透传给 `roll_chain`——**不再走全局 torch RNG**。旧实现（Task 7）走
   `torch.multinomial` 的全局流，`--seed` 锁不住动作 ⇒ 多进程各跑各的、且**不可复现**，
   "同 seed 对照 / 种子矩阵"两件事都不成立。

故**同 seed 同调用序列 ⇒ 逐位可复现**，跨进程亦然（CPU 上 torch 的 RNG 由 seed 完全确定；
SimPy + 每条链独立的 numpy 流亦然）。**两个例外**：
① `--resume` 续跑——ckpt 不存优化器状态、`--lr` 以当次为准，故"续跑段"与"一次跑完"不同
（种子对照请用同一起点、同一预算）；
② **跨设备**（`--device cpu` vs `cuda`）——动作采样流跟随策略设备，CPU 与 CUDA 的
`torch.Generator` 是两条不同的流 ⟹ 同 seed 采出**不同**的链。种子对照必须在**同一设备**上做。

⚠️ **`--device`（2026-10-04，GPU 批次）默认 `cpu`**：CUDA 只影响**算在哪**，不影响任何公式。
本仓测试环境是 CPU-only torch（`cuda.is_available()=False`）；GPU 训练用
`D:/anaconda/envs/py312/python.exe`（torch 2.13.0+cu126，RTX 4060 8 GB）。本机实测（MK01、
`route_k=2`、G=1/G=4、预热后中位 3 次）：批重算的批量是 Σn_g（数百到数千），GPU 在这一档
有优势；但**在线前向是 batch 1，GPU eager 在那一档更慢**（kernel 启动开销，见
`progress-log` §36.3）⟹ "整策略上 GPU" 只与 CPU 批量化打平（G=1 1.68 vs 1.55 s/步），
**不是最优**；把两段分设备的"分工"档（roll 在 CPU、重算在 GPU）实测约 **2.6×**（G=1 1.17、
G=4 5.00 s/步，基准 3.00/12.75），**本批未实现为开关**（需两模型或逐步搬模型的机制，
如实声明）。本节数字与 `progress-log` §36 的三段成本表可互相对照。

输出**一律落 `run_dir`（默认 = **仓库根**的 `checkpoints/a_<inst>`，锚 `__file__` 而非 CWD，
故在包目录里执行也不会建出包内 `checkpoints/`；该目录已被 `.gitignore` 排除），不落包目录**：

- `run_dir/metrics.ndjson`——每步一行 `{"step","r","t",…诊断}`（增量追加，可断外部分析）；
  `--eval-every` 打开时另有 `{"step","eval":{"makespan_mean","makespan_std","rule_makespan"}}`；
- `run_dir/ckpt.pt`——每 `--save-every` 步（含最后一步）覆写。

⚠️ `--lr` 只在**第一次** `joint_chain_step` 时生效（Adam 由该函数惰性创建，之后忽略新 lr）；
`--resume` 不恢复优化器状态（ckpt 只存模型权重与步号），续跑会以**当次 `--lr`** 重建 Adam。

⚠️ **`--route-k`（路线头 R，2026-10-04 恢复）默认 1 = 关闭**：`_drive` 恒走最短路，训练步与
既有读数**逐位相同**（黄金摘要见 `tests/test_route_choice.py`）。传 2 启用——每次行驶在 2 条
候选路径里由策略选（① 拥堵必须开，否则入口显式报错）。它同时透传进**评估**：训练开、
评估关 = 用另一个策略评估，且**静默**（同 `constraints` 的 R2 理由）。

⚠️ `--resume` 只在 `run_dir/ckpt.pt` **已存在**时生效（`runner.resume_training` 的既有语义）：
没有 ckpt 就**从头跑**，而 `metrics.ndjson` 是**追加**打开的——此时文件里会出现**重复的 step 号**
（前一段是废弃的尝试）。按 step 取最新一行即可；要干净曲线就换 `--run-dir` 或先删 run_dir。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, pstdev

import torch

from .algo.group_rel import SEED_STRIDE, roll_chain
from .algo.policy import PolicyNet
from .algo.runner import run_training
from .algo.setup import build_setup
from .env.constraints import ConstraintConfig
from .env.des import SimConfig, rollout
from .env.instances import load_mk
from .env.reward import ReferenceObjectives, reward_weights
from .nn.encoder import LayoutEncoder

# 评估种子起点。⚠️ **必须避开训练用过的扰动流**（评审 F1 修的就是这里）：`joint_chain_step`
# 第 s 步第 g 条链用 `seed_chain = (seed0+s)*SEED_STRIDE + g`（`runner.py` 传 `seed0+s`），
# 故训练流落在 `[seed0*1000, (seed0+steps)*1000)`。**旧值 10_000 与"第 10 步"的训练流正面
# 相撞**（G=8 时 10000..10007 全被第 10 步用过 ⇒ 5 个评估流全是训练流），评估集与训练集不
# 分离会把论文 setup 节那句话写错。取 **10**6**（本常量唯一定义处，测试从本模块导入）。
# ⚠️ 安全条件是 **`seed0 + steps ≤ 1000`**，**不是** `steps < 1000`（评审 I-2：训练流随
# `--seed` 整体平移；`--seed 700 --steps 400` 这种组合旧条件说"安全"（400<1000），实际
# 训练流最高到 1099000+G-1，正面撞上评估流——同一个 bug 在 seed 维度复发）。
# 该条件由 `assert_eval_seed_isolated()` 在入口**硬查**，不静默。
EVAL_SEED_BASE = 10 ** 6


def assert_eval_seed_isolated(seed0: int, steps: int) -> None:
    """训练扰动流与评估扰动流不得相交（="评估集与训练集分离"成立的前提）。

    训练流 = `(seed0+s)*SEED_STRIDE + g`（s < steps，g < G）⇒ 上界 `< (seed0+steps)*1000`；
    评估流自 `EVAL_SEED_BASE = 10**6` 起。条件：`(seed0 + steps) * SEED_STRIDE ≤ EVAL_SEED_BASE`
    （G ≤ SEED_STRIDE 时充分）。不满足则**显式报错**——`--seed 700 --steps 301` 正是旧注释
    `steps < 1000` 放过的反例（700+300 = 1000 恰在边界上，再多一步就越界）。
    """
    if (seed0 + steps) * SEED_STRIDE > EVAL_SEED_BASE:
        raise ValueError(
            f"--seed {seed0} 配 --steps {steps}：训练扰动流最高到 "
            f"{(seed0 + steps) * SEED_STRIDE - 1}，会与评估流 [{EVAL_SEED_BASE}, +∞) 相交"
            f"——评估集与训练集不再分离。请满足 seed0 + steps ≤ "
            f"{EVAL_SEED_BASE // SEED_STRIDE}（减小 --seed 或 --steps）。")


def build_training_setup(inst_name: str, cfg: SimConfig | None = None,
                         constraints: ConstraintConfig | None = None):
    """(inst, layout, dm, cfg, ctx, policy, constraints)——在共享 `algo.setup.build_setup`
    之上再补两件事：加载实例、构造策略网络。环境三件套的口径（布局 seed=0 / 真实参考
    makespan / `cfg` 的几何+车队参数 / `constraints` 同源）**全部**由 `build_setup` 定死
    （评审 F4 收敛）。

    ⚠️ **`constraints` 原样返回**（R2）：`ctx` 是按它建的（③/⑧ 的特征静默读 `ctx.constraints`），
    调用方拿到的就是"ctx 是按哪组约束建的"那一份，**不需要凭记忆把同一份再传给
    `joint_chain_step`**——`joint_chain_step` 入口还会校验两者同源，不同源直接报错。
    旧状（R2 前）：形参存在但 `main` 从不传、也不进 `step_kwargs` ⟹ 钩子在邀请
    "ctx 按 A 组约束建、训练跑全开"的静默错配（F2 要消灭的正是这一类）。`None` = 十约束全开。

    ⚠️ 布局 seed 必须为 0：奖励权重取自 `ReferenceObjectives.of`（固定用 seed_layout=0 的参考
    运行），而特征归一化的 `m_ref` 按**该布局**的 seed 取——两者同源才有一致的归一化刻度
    （非 0 种子会被 `joint_chain_step` 入口拒绝，`build_setup` 的默认值即 0）。
    """
    inst = load_mk(inst_name)
    c = cfg or SimConfig()
    cons = constraints or ConstraintConfig()    # None = 十约束全开（与 SimWorld 的语义一致）
    lay, dm, ctx = build_setup(inst, c, constraints=cons)
    pol = PolicyNet(enc=LayoutEncoder())        # 车队规模由 seg 定，网络无 n_agv 形参（M-4）
    return inst, lay, dm, c, ctx, pol, cons


def _make_eval_fn(inst, lay, dm, cfg, ctx, seeds: int, rule: float,
                  constraints: ConstraintConfig | None = None, route_k: int = 1,
                  pm_head: bool = False):
    """评估回调：argmax 策略在 `seeds` 个扰动种子上的 makespan（spec §5.3.4 约定 3：J>1 只评估）。

    ⚠️ `constraints` 必须与训练同一份（R2）：评估跑的是训练后的策略，动力学口径不一致
    （如训练 ③ 关、评估全开）会让读数对不上训练环境，且**静默**。
    ⚠️ `route_k` 同理（R 头恢复后）：训练开路线头（`route_k=2`）而评估关着 = 用**另一个策略**
    评估（恒走最短路的那一个），读数与训练不对应，且**静默**——故与 `constraints` 一样必须透传。
    ⚠️ `pm_head`（⑫ 维护头）同型：训练开、评估关 = 用规则保养的策略评估一个学出来的策略。
    """
    def eval_fn(policy) -> dict:
        ms = [float(roll_chain(inst, lay, dm, cfg, policy, seed=EVAL_SEED_BASE + s,
                               ctx=ctx, sample=False, constraints=constraints,
                               route_k=route_k, pm_head=pm_head)[1]["makespan"])
              for s in range(seeds)]
        return {"makespan_mean": mean(ms), "makespan_std": pstdev(ms) if len(ms) > 1 else 0.0,
                "rule_makespan": rule}
    return eval_fn


def main() -> None:
    ap = argparse.ArgumentParser(description="A（加权标量化）联合链 GRPO 训练")
    ap.add_argument("--inst", default="mk01", help="实例名（mk01..mk10）")
    ap.add_argument("--steps", type=int, default=300, help="训练步数（MK01/G=8 约 18 s/步）")
    ap.add_argument("--G", type=int, default=8, help="组大小（J=1，预算全给 G）")
    ap.add_argument("--lr", type=float, default=3e-4, help="Adam 学习率（仅首步生效）")
    ap.add_argument("--seed", type=int, default=0,
                    help="随机种子：同时锁定网络初始化/仿真扰动流/动作采样（见模块 docstring）")
    ap.add_argument("--run-dir", default=None,
                    help="默认 <仓库根>/checkpoints/a_<inst>（锚 __file__，不是 CWD）")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=0, help="每 N 步评估一次；0 = 不评估")
    ap.add_argument("--eval-seeds", type=int, default=5, help="评估的扰动种子数")
    ap.add_argument("--resume", action="store_true", help="从 run_dir/ckpt.pt 续跑")
    ap.add_argument("--route-k", type=int, default=1,
                    help="路线头（R）候选条数：1 = 关闭（默认，逐位等于既有读数）；"
                         "2 = 每次行驶在 2 条候选里选（① 拥堵必须开）")
    ap.add_argument("--pm-head", action="store_true",
                    help="⑫ 维护头（M）：默认关（规则自动保养，逐位等于既有读数）；"
                         "开启后机台在两件之间由策略选 {现在保养, 不保养}（要求 ⑫ 维护开启；"
                         "MK01 默认 pm_interval=120 从不逾期，机制验证请配短间隔档）")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                    help="策略所在设备：默认 cpu（本仓测试环境是 CPU-only torch）。"
                         "cuda = 整步（在线前向 + 批重算）都在 GPU 上——重算的批大小是 "
                         "Σn_g（数百到数千），GPU 在这一档对 CPU 有优势；但**在线前向是 "
                         "batch 1**，GPU eager 在那一档反而慢（kernel 启动开销，"
                         "见 progress-log §36.3）⟹ 本开关是「整策略」粒度，"
                         "不是最优分工；分工方案（roll 在 CPU、重算在 GPU）见 §36")
    ap.add_argument("--parallel", action="store_true",
                    help="链级多进程（2026-10-04 并行批次）：G 条链铺到 worker 进程，worker "
                         "只跑 CPU（仿真 + 在线前向），主进程继续用 --device 做重算/反向。"
                         "默认关（原串行路径，读数逐位不变）。⚠️ 开启后采样流改为**逐链"
                         "独立**（否则消费次序不确定）⟹ 与串行档同 seed 的数值不同"
                         "（第九次读数作废，见 progress-log §39）")
    ap.add_argument("--workers", type=int, default=None,
                    help="并行档的 worker 进程数（默认 min(核数, G)）")
    args = ap.parse_args()

    # 默认落**仓库根**的 checkpoints/（锚 `__file__`，不是 CWD）——否则在包目录里执行会建出
    # `plcsp/checkpoints/`，正好破坏本文件 docstring 里"不落包目录"的保证（评审 Minor）。
    run_dir = Path(args.run_dir) if args.run_dir else (
        Path(__file__).resolve().parents[1] / "checkpoints" / f"a_{args.inst}")
    if args.resume and not (run_dir / "ckpt.pt").exists():
        print(f"[m13] ⚠️ --resume 但 {run_dir / 'ckpt.pt'} 不存在：将从 step 0 重跑，且 "
              "metrics.ndjson 是**追加**模式 ⇒ 会出现重复 step 号（按 step 取最新一行）。",
              flush=True)
    if args.eval_every > 0:                  # 只有真会用评估流时才查（不开评估 = 该条件空真）
        assert_eval_seed_isolated(args.seed, args.steps)
    # ⚠️ 必须在 build_training_setup **之前**：`PolicyNet` 的初始化吃全局 torch RNG。
    #    动作采样自评审 I-3 起由 `seed0+s` 派生的 `torch.Generator` 负责，与本流互不干扰。
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "[m13] --device cuda 但当前解释器的 torch 不可用 CUDA（cuda.is_available()=False）。"
            "本仓测试环境（D:/anaconda/python.exe）是 CPU-only 构建；GPU 训练请用 "
            "D:/anaconda/envs/py312/python.exe（torch 2.13.0+cu126）。")
    inst, lay, dm, cfg, ctx, pol, constraints = build_training_setup(args.inst)
    # ⚠️ 设备在**建好策略之后**统一搬（`--device cuda` 时整步在 GPU 上：在线前向经
    #    `forward_enc`、批重算经 `_decision_logp_terms`、动作采样流经 `policy.device` 三处
    #    全部跟随参数设备，没有任何一处硬编码 cpu）。
    pol = pol.to(args.device)
    ref = ReferenceObjectives.of(inst, cfg)     # ref 进训练入口（评审 F5：w 由 ref 派生）
    w = reward_weights(ref.as_tuple())          # 仅为日志打印
    # ⚠️ rule 与 f^ref 都锚在**全约束**参考运行（`reference_run` 的既定语义，des.py），
    #    不随 constraints 走；随 constraints 走的是训练/评估的动力学（ctx + step_kwargs + eval_fn）。
    rule = float(rollout(inst, seed_chain=0, cfg=cfg)["makespan"])   # = M_ref（同一运行）
    print(f"[m13] inst={args.inst} 作业{inst.n_jobs}×机台{inst.n_machines} "
          f"车队{cfg.n_agv}｜steps={args.steps} G={args.G} lr={args.lr} seed={args.seed}"
          f"｜route_k={args.route_k}｜pm_head={args.pm_head}｜device={pol.device}"
          f"｜parallel={args.parallel}(workers={args.workers or 'auto'})")
    print(f"[m13] 权重 w={tuple(round(x, 4) for x in w)}（f^ref={ref.as_tuple()}）")
    print(f"[m13] 规则基线 makespan={rule:.1f}｜run_dir={run_dir.resolve()}")
    print(f"[m13] seed={args.seed} 锁定「初始化 + 仿真流 + 动作采样」：同 seed 可逐位复现"
          "（同设备跨进程亦然；例外：--resume 续跑，以及跨设备——CPU/CUDA 的动作采样流不同，"
          "见模块 docstring）", flush=True)

    run_training(pol, inst, steps=args.steps,
                 step_kwargs=dict(layout=lay, dm=dm, cfg=cfg, ctx=ctx, ref=ref,
                                  constraints=constraints,   # R2：与 ctx 同一份（入口校验同源）
                                  G=args.G, lr=args.lr, route_k=args.route_k,
                                  pm_head=args.pm_head),
                 seed0=args.seed, run_dir=str(run_dir), save_every=args.save_every,
                 resume=args.resume,
                 parallel=args.parallel, n_workers=args.workers,
                 eval_fn=(_make_eval_fn(inst, lay, dm, cfg, ctx, args.eval_seeds, rule,
                                        constraints=constraints, route_k=args.route_k,
                                        pm_head=args.pm_head)
                          if args.eval_every > 0 else None),
                 eval_every=(args.eval_every or None))
    print(f"[m13] 完成：{run_dir / 'metrics.ndjson'}（每步一行）｜{run_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
