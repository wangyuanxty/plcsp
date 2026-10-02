"""M6：标准 SA-GRPO（状态克隆树）实现验证 —— 玩具 sanity → 大问题判定实验。

协议（与 m5_l_gated_verify 同口径，判定锚可比较）：
- 训练：standard_tree_step(policy, inst, seed=s, G=4, J=2, lr=1e-4, n_agv=2)；
  G·J²=16 叶/步（= L 组内 GRPO 的 J=16 序列/步预算）。
- 评估（每 EVAL_EVERY 步）：eval 协议 = sample_plan（与 372.7 锚同协议）+
  L=argmax 门控 run_gated，seeds (1,11,21) → mean makespan（m5_l_gated_verify 同款）。
- 判定（记录 §四）：<372.7（超 L 组内 GRPO）且 <457.1（超规则 load_min）= 成功。
- 玩具 sanity：gen_random(2,2,seed=0)（2job×2m 表行同款）——纯机器/梯度覆盖检查（不评判定）。

用法： python -m plcsp.m6_std_tree sanity | big
输出： sanity: 每步一行 diag（40 步）；big: plcsp/m6_std_tree.jsonl（200 步 + eval 行）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .env.instances import load_mk, gen_random
from .env.layout import sample_layout
from .env.des import SimWorld, SimConfig
from .env.corridors import build_corridor_graph, dock_distance_matrix
from .nn.state_emb import encode_state
from .nn.encoder import LayoutEncoder
from .algo.policy import PolicyNet
from .algo.group_rel import sample_plan
from .algo.standard_tree import standard_tree_step

G, J, J_L, N_AGV, LR, ENT = 4, 2, 8, 2, 1e-4, 0.03
FEAT_L = 6 + N_AGV + 2                      # run_gated 任务特征：6 任务+车状态 + n_agv 队列 + 2 序号
BIG_OUT = "plcsp/m6_std_tree.jsonl"


def make_world(inst):
    lay = sample_layout(inst.n_machines, "line", 1)
    dm = dock_distance_matrix(build_corridor_graph(lay))
    return SimWorld(inst, lay, dm, SimConfig(n_agv=N_AGV))


def make_policy(inst) -> tuple[PolicyNet, object]:
    """M3c 编码器版策略 + 状态嵌入（x-条件化的 S 层——理论要求 x=s_t 特征编码；MLP 无状态=x）。"""
    es = encode_state(inst, sample_layout(inst.n_machines, "line", 1), n_agv=N_AGV)
    return PolicyNet(enc=LayoutEncoder(feat_dim=es.tok_feat.shape[-1]),
                     n_agv=N_AGV, n_feat_l=FEAT_L), es


def eval_fn(policy, inst, es) -> dict:
    """部署口径评估：采样计划（tok 版）+ L=argmax（m5_l_gated_verify 同协议）。"""
    tok = policy.encode_state(es)
    pl = sample_plan(inst, policy, 10.0, tok_emb=tok)

    def greedy(feat_np):
        lg = policy.agv_logits(torch.tensor(feat_np[None, None, :]))[0, 0]
        return int(torch.argmax(lg).item())

    mss = []
    for se in (1, 11, 21):
        out = make_world(inst).run_gated(seed_chain=se, op_choices=pl, policy_l=greedy)
        mss.append(out["makespan"])
    return {"mean": float(np.mean(mss)), "seeds": [round(float(x), 1) for x in mss]}


def sanity() -> None:
    """玩具 2×2：40 步，检查机器健康 + 三头梯度覆盖（防假平台）。"""
    inst = gen_random(2, 2, seed=0)
    pol, es = make_policy(inst)
    t0 = time.time()
    standard_tree_step(pol, inst, seed=0, G=G, J=J, lr=LR, n_agv=N_AGV,
                       enc_state=es)   # 首步：先暴露形状错误
    for s in range(40):
        r, d = standard_tree_step(pol, inst, seed=s, G=G, J=J, J_L=J_L, lr=LR,
                                  n_agv=N_AGV, enc_state=es, ent_beta=ENT)
        assert np.isfinite(r) and np.isfinite(d["loss"]), f"step {s}: NaN ({d})"
        if (s + 1) % 5 == 0:
            print(json.dumps({"step": s + 1, "r": round(float(r), 2),
                              "loss": round(float(d["loss"]), 3),
                              "g_b": d["g_b"], "g_s": d["g_s"], "g_l": d["g_l"],
                              "ratio_S": d["ratio_S"], "sig_L": d["sig_L"],
                              "dec_len": d["dec_len"], "t": round(time.time() - t0, 1)},
                             ensure_ascii=False), flush=True)
    print(f"SANITY-OK 40 steps, elapsed={time.time()-t0:.1f}s")


def big() -> None:
    """MK01（10job×6m）：续训至 800 步（ckpt 恢复）；判定锚 372.7 / 457.1。"""
    inst = load_mk("mk01")
    ckpt = Path("checkpoints") / "m6_std_tree.pt"
    pol, es = make_policy(inst)
    start = 0
    if ckpt.exists():
        pol.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
        start = int(torch.load(ckpt, map_location="cpu", weights_only=False)["step"])
        print(f"[m6] resume from step {start}", flush=True)
    else:
        torch.manual_seed(0)
    t0 = time.time()
    with open(BIG_OUT, "a", encoding="utf-8") as f:
        for s in range(start, 800):
            r, d = standard_tree_step(pol, inst, seed=s, G=G, J=J, J_L=J_L, lr=LR,
                                      n_agv=N_AGV, enc_state=es, ent_beta=ENT)
            assert np.isfinite(r) and np.isfinite(d["loss"]), f"step {s}: NaN"
            if (s + 1) % 20 == 0:
                ev = eval_fn(pol, inst, es)
                row = {"step": s + 1, "r": round(float(r), 2),
                       "ratio_S": d["ratio_S"], "ratio_B": d["ratio_B"],
                       "g_b": d["g_b"], "g_stok": d.get("g_stok", 0.0), "g_l": d["g_l"],
                       "eval": ev, "elapsed": round(time.time() - t0, 1)}
                print(json.dumps(row, ensure_ascii=False), flush=True)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
        torch.save({"step": s + 1, "model": pol.state_dict()}, ckpt)
    print("[m6] done")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "sanity":
        sanity()
    elif sys.argv[1] == "big":
        big()
