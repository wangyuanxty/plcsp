"""M6-规模扫描（用户修正版）：工厂规模 × L 层可学性 —— 机器+车按真实比例整体放大。

设计（比例守恒：车 ≈ 0.67× 机）：
  M: gen_random(20, 12, seed=0) × 8 车   (1.0×0.67)
  L: gen_random(30, 18, seed=1) × 12 车  (1.0×0.67)
每档流程：固定参考计划（本策略 sample_plan，seed=0，30 种子）→
  基线（轮询 / 随机 L 30 种子）→ L-组内-GRPO 训练 300 步（J=16, z, lr=1e-4）→
  终值（学习后 argmax L / 采样 L 30 种子）。
判定：学习后 argmax < 轮询（同计划同种子）→ 该规模 L 层可学（正例：规模-可学性曲线）。
输出：geosched/m6_agv.jsonl + ckpt checkpoints/m6_agv_{M,L}.pt。
运行：python -m geosched.m6_agv（后台 ~1.5-2h）。小规模（MK01 6机×2-4车）已有数据作 S 档。
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from .env.instances import gen_random
from .env.layout import sample_layout
from .env.des import SimWorld, SimConfig
from .env.corridors import build_corridor_graph, dock_distance_matrix
from .algo.policy import PolicyNet
from .algo.group_rel import sample_plan, l_seq_step_gated

CONFIGS = [
    ("M20x12", gen_random(20, 12, seed=0), 8),
    ("L30x18", gen_random(30, 18, seed=1), 12),
]
TRAIN_STEPS = 300
OUT = "geosched/m6_agv.jsonl"


def main() -> None:
    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as f:
        for tag, inst, k in CONFIGS:
            w = SimWorld(inst, sample_layout(inst.n_machines, "line", 1),
                         dock_distance_matrix(build_corridor_graph(
                             sample_layout(inst.n_machines, "line", 1))),
                         SimConfig(n_agv=k))
            torch.manual_seed(0)
            pol = PolicyNet(n_agv=k, n_feat_l=6 + k + 2)
            plan = sample_plan(inst, pol, 10.0)                     # 固定参考计划（同档内一致）

            # —— 1) 基线：轮询 vs 随机 L（30 种子，同计划）——
            rr, rd = [], []
            for se in range(1, 31):
                rr.append(w.run_gated(seed_chain=se, op_choices=plan)["makespan"])
                rng_t = np.random.default_rng(se)
                rd.append(w.run_gated(seed_chain=se, op_choices=plan,
                                      policy_l=lambda x, r=rng_t: int(r.integers(0, k)))["makespan"])
            row = {"cfg": tag, "k": k, "rr_mean": round(float(np.mean(rr)), 2),
                   "rand_mean": round(float(np.mean(rd)), 2),
                   "elapsed": round(time.time() - t0, 1)}

            # —— 2) L-组内-GRPO 训练（300 步）——
            for s in range(TRAIN_STEPS):
                l_seq_step_gated(pol, inst, seed=s, J=16, mode="z", lr=1e-4, n_agv=k)

            # —— 3) 终值：argmax L / 采样 L（30 种子，同计划）——
            def mk_argmax(p):

                def lx(feat_np):
                    lg = p.agv_logits(torch.tensor(feat_np[None, None, :]))[0, 0]
                    return int(torch.argmax(lg).item())
                return lx

            am, sp = [], []
            for se in range(1, 31):
                am.append(w.run_gated(seed_chain=se, op_choices=plan,
                                      policy_l=mk_argmax(pol))["makespan"])
                sp.append(w.run_gated(seed_chain=se, op_choices=plan,
                                      policy_l=mk_argmax(pol))["makespan"])
            row.update({"argmax_mean": round(float(np.mean(am)), 2),
                        "sample_mean": round(float(np.mean(sp)), 2)})
            torch.save({"cfg": tag, "k": k, "model": pol.state_dict()},
                       f"checkpoints/m6_agv_{tag}.pt")
            print(json.dumps(row, ensure_ascii=False), flush=True)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    print("[m6agv] done")


if __name__ == "__main__":
    main()
