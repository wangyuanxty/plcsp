"""M6-扁平对照：标准 GRPO（优势=全局 z，无树/无分层域）—— 与标准树（372.5）的唯一差异实验。

机制保真（与 m6_std_tree 完全一致，仅 A 的域不同）：
- 采样结构：π_B·π_S·π_L + 扰动克隆（pert_shared=True）+ 编码器 + J_L=8 → G=4, J=2, J_L=8 = 64 叶/步
- 链 logp = logp_B + logp_S(平均) + logp_L(平均)；损失 = -(A·链logp).mean()（与树同为三段之和）
- 优势：norm="flat" → A_B=A_S=A_L = 全部 64 叶全局 z（标准 GRPO 本质：同 prompt 组全局 z）
- 树与 flat 的差异 = 唯一机制变量；判定：flat vs 372.5 → <372.5 树劣化 / ≈372.5 无增益 /
  >372.5 树有真增益（理想）。
评估：同 m6_std_tree.eval_fn（sample_plan + L=argmax；终局 30 种子）。
输出：plcsp/m6_flat.jsonl。运行：python -m plcsp.m6_flat（后台 ~85min）。
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
OUT = "plcsp/m6_flat.jsonl"


def make_world(inst):
    lay = sample_layout(inst.n_machines, "line", 1)
    dm = dock_distance_matrix(build_corridor_graph(lay))
    return SimWorld(inst, lay, dm, SimConfig(n_agv=N_AGV))


def main() -> None:
    inst = load_mk("mk01")
    es = encode_state(inst, sample_layout(inst.n_machines, "line", 1), n_agv=N_AGV)
    pol = PolicyNet(enc=LayoutEncoder(feat_dim=es.tok_feat.shape[-1]),
                    n_agv=N_AGV, n_feat_l=FEAT_L)
    torch.manual_seed(0)
    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as f:
        for s in range(800):
            r, d = standard_tree_step(pol, inst, seed=s, G=G, J=J, J_L=J_L, lr=LR,
                                      n_agv=N_AGV, enc_state=es, ent_beta=ENT,
                                      norm="flat")
            assert np.isfinite(r) and np.isfinite(d["loss"]), f"step {s}: NaN"
            if (s + 1) % 100 == 0:
                tok = pol.encode_state(es)
                pl = sample_plan(inst, pol, 10.0, tok_emb=tok)

                def lg(feat_np):
                    lx = pol.agv_logits(torch.tensor(feat_np[None, None, :]))[0, 0]
                    return int(torch.argmax(lx).item())

                mss = [make_world(inst).run_gated(seed_chain=se, op_choices=pl, policy_l=lg)["makespan"]
                       for se in (1, 11, 21)]
                ev = {"mean3": round(float(np.mean(mss)), 2)}
                row = {"step": s + 1, "r": round(float(r), 2),
                       "ratio_S": d["ratio_S"], "ratio_B": d["ratio_B"],
                       "g_b": d["g_b"], "g_stok": d.get("g_stok", 0.0), "g_l": d["g_l"],
                       "eval": ev, "elapsed": round(time.time() - t0, 1)}
                print(json.dumps(row, ensure_ascii=False), flush=True)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
        # 终局 30 种子（同 m6_std_tree 协议）
        tok = pol.encode_state(es)
        pl = sample_plan(inst, pol, 10.0, tok_emb=tok)

        def lg(feat_np):
            lx = pol.agv_logits(torch.tensor(feat_np[None, None, :]))[0, 0]
            return int(torch.argmax(lx).item())

        mss = [make_world(inst).run_gated(seed_chain=se, op_choices=pl, policy_l=lg)["makespan"]
               for se in range(1, 31)]
        ev30 = {"mean": round(float(np.mean(mss)), 2), "median": round(float(np.median(mss)), 2),
                "min": round(float(np.min(mss)), 2), "max": round(float(np.max(mss)), 2)}
        torch.save({"step": 800, "model": pol.state_dict()},
                   Path("checkpoints") / "m6_flat.pt")
        print(json.dumps({"step": 800, "eval_30seeds": ev30,
                          "elapsed": round(time.time() - t0, 1)}, ensure_ascii=False), flush=True)
    print("[m6flat] done")


if __name__ == "__main__":
    main()
