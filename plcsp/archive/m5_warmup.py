"""M5 预热：MK01 上 GRPO vs SA-GRPO 短跑训练对比（信号验证，非论文实验）。

预热口径（进展记录将注明，与论文实验的差异）：
- 实例：官方 MK01（load_mk）；布局 line seed=1 固定（论文实验 = 多布局 × 多拓扑）
- GRPO    = train_step(G=8, J=1, mode='z', epochs=1)  —— 纯组内相对（M3a）
- SA-GRPO = tree_step(G=8, J=1, cap_opts=[10,5,2])     —— B 层 3 模式 × S 层 × 分层归因（M3b-①）
- 均 MLP 特征路径（编码器路径 = 另一消融维度 ②，论文实验再做）
- 每 10 步贪心评估：greedy_plan（argmax，确定性）× 扰动种子 [1,11,21] → 均值 makespan
- 步数 40/方法（预热带；论文 = 200+ 步 × 多种子 × G/J 扫描）
输出：plcsp/m5_warmup.jsonl（{step, method, eval_mean} 逐评估点一行）。

运行：python -m plcsp.m5_warmup（后台推荐——含 SimPy rollout，约 30-45 分钟）
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from .env.instances import load_mk
from .env.des import rollout_evaluate
from .algo.policy import PolicyNet
from .algo.group_rel import train_step, tree_step, greedy_plan

STEPS, EVAL_EVERY = 500, 50
SEEDS_EVAL = (1, 11, 21)          # 扰动流种子（评估均值的随机化）

METHODS = [
    ("GRPO", train_step, dict(G=8, J=1, mode="z")),
    ("SA-GRPO", tree_step, dict(G=8, J=1, cap_opts=[10, 5, 2])),
]


def evaluate(policy: PolicyNet, inst, t_max: float = 10.0) -> float:
    ms = []
    for s in SEEDS_EVAL:
        pl = greedy_plan(inst, policy, t_max=t_max)
        ms.append(-rollout_evaluate(inst, pl, seed=s))       # -makespan → 正 makespan
    return float(np.mean(ms))


def main(out: str = "plcsp/m5_warmup_500.jsonl") -> None:
    inst = load_mk("mk01")
    t0 = time.time()
    rows = []
    with open(out, "a", encoding="utf-8") as f:
        for tag, fn, kw in METHODS:
            torch.manual_seed(0)                             # 策略初始化一致（同一起点才可比）
            pol = PolicyNet()
            for s in range(STEPS):
                fn(pol, inst, seed=s, **kw)
                if (s + 1) % EVAL_EVERY == 0:
                    mu = evaluate(pol, inst)
                    row = {"method": tag, "step": s + 1, "eval_mean": mu,
                           "elapsed": round(time.time() - t0, 1)}
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
    print(f"[warmup] done {len(rows)} eval points -> {out}")


if __name__ == "__main__":
    main()
