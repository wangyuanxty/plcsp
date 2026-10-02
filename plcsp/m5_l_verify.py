"""L 层验证（救活计划 step 1）：local_l_step 训练 + 贪心 AGV 序列 vs 固定规则锚。

方法：200 步 L 层局部树训练（J=2，MK01×line）；每 20 步评估：
  greedy_agv_seq = 按预演流逐任务 argmax(π_L) → rollout ×3 扰动种子 → makespan 均值。
对照锚（同期固定规则）：轮询 ~347.9 / 全 AGV0 ~481.2 / FIFO ~357.3（本数据同实例同计划）。
输出：plcsp/m5_l_verify.jsonl + 终端行。运行：python -m plcsp.m5_l_verify（后台 ~10min）
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from .env.instances import load_mk
from .env.des import rollout
from .algo.policy import PolicyNet
from .algo.group_rel import l_seq_step, _l_feat, sample_plan

STEPS, EVAL_EVERY, N_AGV, J = 800, 100, 2, 4
SEEDS_EVAL = (1, 11, 21)
SEEDS_TRAIN = (0, 1, 2)
OUT = "plcsp/m5_l_verify.jsonl"


def greedy_agv_seq(policy, flow, loads, n_m: int, n_agv: int) -> list[int]:
    out = []
    for i, (_, oi, frm, to) in enumerate(flow):
        if i < len(loads):
            (d0, d1), (p0, p1) = loads[i]
            ft = _l_feat(frm, to, oi, n_m, load_d=float(d0 - d1), pos0=int(p0), pos1=int(p1))
        else:
            ft = _l_feat(frm, to, oi, n_m)
        out.append(int(torch.argmax(policy.agv_logits(ft)).item()))
    return out


def main() -> None:
    inst = load_mk("mk01")
    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as f:
        for seed in SEEDS_TRAIN:
            torch.manual_seed(seed)
            pol = PolicyNet(n_agv=N_AGV)
            for s in range(STEPS):
                l_seq_step(pol, inst, seed=s, J=J, mode="z")
                if (s + 1) % EVAL_EVERY == 0:
                    pl = sample_plan(inst, pol, 10.0)
                    prev = rollout(inst, op_choices=pl, seed_chain=s,
                                   agv_phi=[i % N_AGV for i in range(64)])
                    flow = prev["task_flow"]
                    loads = prev.get("agv_load", [])
                    seq = greedy_agv_seq(pol, flow, loads, inst.n_machines, N_AGV)
                    ms = [-rollout(inst, op_choices=pl, seed_chain=se, agv_phi=seq)["makespan"]
                          for se in SEEDS_EVAL]
                    row = {"seed": seed, "step": s + 1,
                           "greedy_mean": round(float(np.mean(ms)), 2),
                           "flow_len": len(flow), "n_seq": sum(a == 1 for a in seq),
                           "elapsed": round(time.time() - t0, 1)}
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
    print("[m5l] done")


if __name__ == "__main__":
    main()
