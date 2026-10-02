"""L 层完整版验证：真·门控事件驱动（l_seq_step_gated）800 步 × 3 种子。

评估 = run_gated(argmax π_L)（闭环部署口径）；对照锚：轮询 347.9 / 全0 481.2 / FIFO 357.3。
输出：geosched/m5_l_gated.jsonl。运行：python -m geosched.m5_l_gated_verify（后台 ~1.5h）
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
from .algo.group_rel import l_seq_step_gated, sample_plan

STEPS, EVAL_EVERY, N_AGV, J = 800, 100, 4, 16
LR = 1e-4
SEEDS_TRAIN = (0, 1, 2)
OUT = "geosched/m5_l_gated.jsonl"


def make_world(inst):
    lay = sample_layout(inst.n_machines, "line", 1)
    dm = dock_distance_matrix(build_corridor_graph(lay))
    return SimWorld(inst, lay, dm, SimConfig(n_agv=N_AGV))


def main() -> None:
    inst = load_mk("mk01")
    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as f:
        for seed in SEEDS_TRAIN:
            torch.manual_seed(seed)
            pol = PolicyNet(n_agv=N_AGV, n_feat_l=6 + N_AGV)
            for s in range(STEPS):
                l_seq_step_gated(pol, inst, seed=s, J=J, mode="z", lr=LR, n_agv=N_AGV)
                if (s + 1) % EVAL_EVERY == 0:
                    pl = sample_plan(inst, pol, 10.0)

                    def greedy(feat_np):
                        lg = pol.agv_logits(torch.tensor(feat_np[None, None, :]))[0, 0]
                        return int(torch.argmax(lg).item())

                    mss = []
                    for se in (1, 11, 21):
                        out = make_world(inst).run_gated(seed_chain=se, op_choices=pl,
                                                         policy_l=greedy)
                        mss.append(out["makespan"])
                    row = {"seed": seed, "step": s + 1,
                           "greedy_mean": round(float(np.mean(mss)), 2),
                           "elapsed": round(time.time() - t0, 1)}
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
    print("[m5lg] done")


if __name__ == "__main__":
    main()
