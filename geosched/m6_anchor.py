# -*- coding: utf-8 -*-
"""M6 同口径锚：L-组内-GRPO，n_agv=2、z 模式、J=16、800 步 —— 与标准树 M6（372.5）对齐。

背景（诚实）：此前"锚 372.7"来自 m5_l_gated_verify.py（N_AGV=4、median 基线、3 种子评估），
与标准树（n_agv=2、z、J_L=8、30 种子）**不可直接比**。本脚本生成同口径对照：
- 训练：l_seq_step_gated(policy, inst, seed=s, J=16, mode="z", lr=1e-4, n_agv=2)（纯 L 轴，plan 重采随机）
- 评估（每 100 步 3 种子 + 终局 30 种子）：与 m6_std_tree.eval_fn 同协议（sample_plan + L=argmax）
- 判定：标准树 372.5 vs 本锚 M → 若 M ≈ 372.5 → "SA-GRPO ≈ 单轴 GRPO"（如实写）；
  若 M > 372.5 → 标准树有真增益；若 M < 372.5 → 树反而劣化（诚实记录）。
输出：geosched/m6_anchor.jsonl。运行：python -m geosched.m6_anchor ~1.5h。
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from .env.instances import load_mk
from .env.layout import sample_layout
from .env.des import SimWorld, SimConfig
from .env.corridors import build_corridor_graph, dock_distance_matrix
from .algo.policy import PolicyNet
from .algo.group_rel import sample_plan, l_seq_step_gated

N_AGV, J, LR = 2, 16, 1e-4
STEPS = 800
OUT = "geosched/m6_anchor.jsonl"


def make_world(inst):
    lay = sample_layout(inst.n_machines, "line", 1)
    dm = dock_distance_matrix(build_corridor_graph(lay))
    return SimWorld(inst, lay, dm, SimConfig(n_agv=N_AGV))


def eval_fn(policy, inst, n_seeds=30) -> dict:
    """同 M6 协议：sample_plan（无 enc=随机计划——L-GRPO 不学 plan）+ L=argmax。"""
    pl = sample_plan(inst, policy, 10.0)

    def greedy(feat_np):
        lg = policy.agv_logits(torch.tensor(feat_np[None, None, :]))[0, 0]
        return int(torch.argmax(lg).item())

    mss = []
    for se in (1, 11, 21) if n_seeds == 3 else range(1, 31):
        out = make_world(inst).run_gated(seed_chain=se, op_choices=pl, policy_l=greedy)
        mss.append(out["makespan"])
    return {"mean": round(float(np.mean(mss)), 2),
            "median": round(float(np.median(mss)), 2),
            "min": round(float(np.min(mss)), 2), "max": round(float(np.max(mss)), 2),
            "n_seeds": n_seeds}


def main() -> None:
    inst = load_mk("mk01")
    torch.manual_seed(0)
    pol = PolicyNet(n_agv=N_AGV, n_feat_l=6 + N_AGV + 2)
    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as f:
        for s in range(STEPS):
            l_seq_step_gated(pol, inst, seed=s, J=J, mode="z", lr=LR, n_agv=N_AGV)
            if (s + 1) % 100 == 0:
                ev = eval_fn(pol, inst, n_seeds=3)
                row = {"step": s + 1, "eval": ev, "elapsed": round(time.time() - t0, 1)}
                print(json.dumps(row, ensure_ascii=False), flush=True)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    ev30 = eval_fn(pol, inst, n_seeds=30)
    row = {"step": STEPS, "eval_30seeds": ev30, "elapsed": round(time.time() - t0, 1)}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("[m6anchor] done")


if __name__ == "__main__":
    main()
