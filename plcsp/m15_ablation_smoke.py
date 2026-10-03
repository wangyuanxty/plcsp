# -*- coding: utf-8 -*-
"""消融链路单种子打通（P4-A Task 4）。

⚠️ 用**原始 MK**（非 MKT）——用户裁定：原始 MK 跑我们自己的指标与消融，MKT 用于与已发表数字比较。
⚠️ 墙钟：`joint_chain_step(G=8)` 实测 MK01 ≈ 18.2 s/步、MK10 ≈ 86–94 s/步（P2）。
   **5 组 × 3 实例 × 100 步不可能分钟级跑完**——故测试只跑 2 组 × 3 步，本脚本是长跑。
⚠️ 术语：描述本脚本做的事用平实说法（「消融链路打通」），**不自造术语**（用户明令）。

## 种子流（⚠️ 本项目栽过两次的坑：评审 F1 / I-2）

- **训练流**：第 s 步第 g 条链的仿真扰动种子 = `(seed*TRAIN_STEP_STRIDE + s) * SEED_STRIDE + g`，
  `TRAIN_STEP_STRIDE = SEED_STRIDE = 1000`。上界 = `(seed*1000 + steps-1)*1000 + G-1`。
- **评估流**：自 `EVAL_SEED_BASE = 10**9` 起——**显式**与训练块分离。
- 入口 `assert_streams_disjoint` **显式报错**（不静默跑），与 `m13_train_a` 的同款守卫一致。
  旧写法（评估直接用 `seed*1000+steps`）在 `steps` 取到 1000 的整数倍附近会**撞进训练块**，
  症状只是"评估数字略好"，不会报错。
"""
from __future__ import annotations

import argparse
import time

from .algo.group_rel import SEED_STRIDE, joint_chain_step, roll_chain
from .algo.setup import build_setup
from .env.constraints import ABLATION_GROUPS, ConstraintConfig
from .env.des import SimConfig
from .env.instances import Instance, load_mk
from .env.reward import ReferenceObjectives

# 每步在链种子空间里跨 SEED_STRIDE，组内用 g（0..G-1）填充——故 G 必须 ≤ SEED_STRIDE。
TRAIN_STEP_STRIDE = SEED_STRIDE
# 评估流基址：**显式**高于任何合法训练块（见 `assert_streams_disjoint`）。
EVAL_SEED_BASE = 10 ** 9


def _train_step_seed(seed: int, step: int) -> int:
    """第 `step` 步传给 `joint_chain_step` 的 seed（组内再乘 `SEED_STRIDE` 加 g）。"""
    return seed * TRAIN_STEP_STRIDE + step


def _eval_seed(seed: int, steps: int) -> int:
    """评估链的种子——**直接**传给 `roll_chain`（它自己不再乘 `SEED_STRIDE`）。"""
    return EVAL_SEED_BASE + seed * TRAIN_STEP_STRIDE + steps


def assert_streams_disjoint(seed: int, steps: int, G: int) -> None:
    """训练块的上界必须低于 `EVAL_SEED_BASE`——否则评估会读到训练见过的仿真流。

    ⚠️ 越界**显式报错**：这个 bug 的症状是"评估数字略好"，**不会**自己暴露（评审 F1/I-2）。
    """
    if G > SEED_STRIDE:
        raise ValueError(f"G={G} 超过 SEED_STRIDE={SEED_STRIDE}——组内链会与下一步的流相交")
    hi = _train_step_seed(seed, steps - 1) * SEED_STRIDE + (G - 1) if steps > 0 else -1
    if hi >= EVAL_SEED_BASE:
        raise ValueError(
            f"训练流上界 {hi} 已达/超过评估流基址 {EVAL_SEED_BASE}——两者会相交。"
            f"请减小 --steps（当前 {steps}）或 --seed（当前 {seed}）。")


def _policy_for() -> object:
    """建一个带编码器的 `PolicyNet`（两个打分头只在 `enc is not None` 时存在）。"""
    from .algo.policy import PolicyNet
    from .nn.encoder import LayoutEncoder
    return PolicyNet(enc=LayoutEncoder())


def run_group(inst: Instance, constraints: ConstraintConfig, *, seed: int,
              steps: int, G: int, cfg: SimConfig | None = None) -> dict:
    """跑一组消融 `steps` 步，返回**末步**的指标（由 `roll_chain` 的 metrics 带出）。

    ⚠️ `constraints` 必须**同时**进 `build_setup`（决定 `NormContext`）与两个训练/评估入口
       ——③/⑧ 的特征静默读 `ctx.constraints`，不同源会给出"不会坏/无交期"的假信号（评审 R2）。
    ⚠️ 返回的是 **argmax**（`sample=False`）评估：无采样噪声，同 seed 逐位可复现。
    """
    cfg = cfg or SimConfig()
    assert_streams_disjoint(seed=seed, steps=steps, G=G)
    lay, dm, ctx = build_setup(inst, cfg, constraints=constraints)
    ref = ReferenceObjectives.of(inst, cfg)
    pol = _policy_for()
    for s in range(steps):
        joint_chain_step(pol, inst, lay, dm, cfg, ctx, ref,
                         seed=_train_step_seed(seed, s), G=G, constraints=constraints)
    _, met = roll_chain(inst, lay, dm, cfg, pol, seed=_eval_seed(seed, steps), ctx=ctx,
                        constraints=constraints, sample=False)
    return {"makespan": met["makespan"], "energy": met["energy"],
            "tardy_twt": met["tardy_twt"], "jobs_done": met["jobs_done"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="mk01,mk07,mk10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30,
                    help="默认 30：P2 实测 MK01 上 ~25 步即饱和（progress-log §19.2）")
    ap.add_argument("--G", type=int, default=8)
    args = ap.parse_args()
    print(f"基准 = MK（原始，非 MKT）｜种子 {args.seed}｜{args.steps} 步｜G={args.G}"
          f"｜评估流基址 {EVAL_SEED_BASE}（与训练流显式分离）")
    print(f"{'组':<8}{'实例':<7}{'Cmax':>10}{'energy':>10}{'TWT':>10}{'秒':>8}")
    for name in args.instances.split(","):
        inst = load_mk(name)
        for gname, cons in ABLATION_GROUPS.items():
            t0 = time.time()
            r = run_group(inst, cons, seed=args.seed, steps=args.steps, G=args.G)
            print(f"{gname:<8}{name:<7}{r['makespan']:>10.1f}{r['energy']:>10.2f}"
                  f"{r['tardy_twt']:>10.1f}{time.time() - t0:>8.1f}", flush=True)


if __name__ == "__main__":
    main()
