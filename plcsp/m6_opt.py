"""M6-opt：无树场景化 GRPO 变体 A/B（对照 flat=371.05 / 树=372.5）。

A（channel-grouping）：4 通道（makespan/tardy/moves/energy）各自组内 z 后等权求和
  —— MO-GRPO(arXiv 2509.22047)/Multi-GRPO(2512.00743) reward-grouping 同构：
  方差对齐、防奖励黑客、免手调权重（z 后各通道 N(0,1) → 等权=均匀贡献）。
B（A + GRPO-λ）：L 段 logp 按时效折扣 w_t = λ^(T-1-t)（近期决策权重高；
  GRPO-λ 2510.00194 / eligibility traces 2605.05965 同族）。
其余机制与 m6_flat 完全一致（G=4,J=2,J_L=8=64叶/步、编码器、pert_shared、ent=0.03、
norm="flat" 仅 A 构造差异——本实验"无树"=没争议的基线域）。
评估：同协议（sample_plan + L=argmax；每 100 步 3 种子 + 终局 30 种子）。
输出：plcsp/m6_opt_a.jsonl / m6_opt_b.jsonl；ckpt: checkpoints/m6_opt_a.pt / m6_opt_b.pt。
运行：python -m plcsp.m6_opt（~3h，A 在前 B 在后）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .env.instances import load_mk
from .env.layout import sample_layout
from .env.des import SimWorld, SimConfig
from .env.corridors import build_corridor_graph, dock_distance_matrix
from .nn.state_emb import encode_state
from .nn.encoder import LayoutEncoder
from .algo.policy import PolicyNet
from .algo.group_rel import sample_plan
from .algo.standard_tree import standard_tree_step

G, J, J_L, N_AGV, LR, ENT = 4, 2, 8, 2, 1e-4, 0.03
FEAT_L = 6 + N_AGV + 2


def make_world(inst):
    lay = sample_layout(inst.n_machines, "line", 1)
    dm = dock_distance_matrix(build_corridor_graph(lay))
    return SimWorld(inst, lay, dm, SimConfig(n_agv=N_AGV))


def final_eval(policy, inst, es) -> dict:
    tok = policy.encode_state(es)
    pl = sample_plan(inst, policy, 10.0, tok_emb=tok)

    def lg(feat_np):
        lx = policy.agv_logits(torch.tensor(feat_np[None, None, :]))[0, 0]
        return int(torch.argmax(lx).item())

    mss = [make_world(inst).run_gated(seed_chain=se, op_choices=pl, policy_l=lg)["makespan"]
           for se in range(1, 31)]
    return {"mean": round(float(np.mean(mss)), 2), "median": round(float(np.median(mss)), 2),
            "min": round(float(np.min(mss)), 2), "max": round(float(np.max(mss)), 2)}


def run_variant(tag: str, reward_mode: str, lam: float | None) -> None:
    inst = load_mk("mk01")
    es = encode_state(inst, sample_layout(inst.n_machines, "line", 1), n_agv=N_AGV)
    pol = PolicyNet(enc=LayoutEncoder(feat_dim=es.tok_feat.shape[-1]),
                    n_agv=N_AGV, n_feat_l=FEAT_L)
    ckpt = Path("checkpoints") / f"m6_opt_{tag}.pt"
    start = 0
    if ckpt.exists():                     # 中途恢复（防 3h 断点全丢——m6_opt 首跑教训）
        pol.load_state_dict(torch.load(ckpt, weights_only=False)["model"])
        start = int(torch.load(ckpt, weights_only=False)["step"])
        print(f"[m6opt-{tag}] resume from step {start}", flush=True)
    else:
        torch.manual_seed(0)
    out = f"plcsp/m6_opt_{tag}.jsonl"
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for s in range(start, 800):
            r, d = standard_tree_step(pol, inst, seed=s, G=G, J=J, J_L=J_L, lr=LR,
                                      n_agv=N_AGV, enc_state=es, ent_beta=ENT,
                                      norm="flat", reward_mode=reward_mode, lam=lam)
            assert np.isfinite(r) and np.isfinite(d["loss"]), f"{tag} step {s}: NaN"
            if (s + 1) % 100 == 0:
                tok = pol.encode_state(es)
                pl = sample_plan(inst, pol, 10.0, tok_emb=tok)

                def lg(feat_np):
                    lx = pol.agv_logits(torch.tensor(feat_np[None, None, :]))[0, 0]
                    return int(torch.argmax(lx).item())

                mss = [make_world(inst).run_gated(seed_chain=se, op_choices=pl, policy_l=lg)["makespan"]
                       for se in (1, 11, 21)]
                row = {"variant": tag, "step": s + 1, "r": round(float(r), 2),
                       "g_stok": d.get("g_stok", 0.0), "g_l": d["g_l"],
                       "eval_mean3": round(float(np.mean(mss)), 2),
                       "elapsed": round(time.time() - t0, 1)}
                print(json.dumps(row, ensure_ascii=False), flush=True)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            if (s + 1) % 400 == 0 and s + 1 < 800:      # 中途 ckpt（防长跑断点）
                torch.save({"step": s + 1, "model": pol.state_dict()}, ckpt)
        ev = final_eval(pol, inst, es)
        torch.save({"step": 800, "model": pol.state_dict()},
                   Path("checkpoints") / f"m6_opt_{tag}.pt")
        row = {"variant": tag, "step": 800, "eval_30seeds": ev,
               "elapsed": round(time.time() - t0, 1)}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[m6opt-{tag}] done")


def main() -> None:
    run_variant("a", reward_mode="channels", lam=None)     # A：4 通道组内 z 等权
    run_variant("b", reward_mode="channels", lam=0.9)      # B：A + GRPO-λ(L 段时序折扣)


if __name__ == "__main__":
    main()
