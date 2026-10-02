"""M7-2×2 判据实验：机制（度量感知） × 训练数据（line1 / DR 随机布局） —— 岛式分水岭。

引擎 = 论文正式引擎（flat GRPO：standard_tree_step norm="flat"，64 叶/步、S+L 联合、
编码器）——与性能表（371.05/-18.5%）完全同构；机制差异仅 via enc_args（bias/noBias）。
目标：判定"跨拓扑泛化"来自机制还是数据增强（DR）—— 防"这不就是 DR 吗"审稿质疑。
矩阵：A1 bias×line1 / A2 noBias×line1 / A3 bias×DR / A4 noBias×DR × 3 测试域（line2/U1/岛式）
      × 30 种子（greedy 计划 + L=argmax 同 M6 协议）。
分水岭 = 岛式列：A1 > A4 → 机制创新成立；A4 ≥ A1 → 降级"机制+DR 组合"。
DR 训练 = 每步布局轮换 (line/U/island, seed) → 每步重建 enc_state。
输出：geosched/m7_mechanism.jsonl。运行：python -m geosched.m7_mechanism（后台 ~80min）。
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
from .nn.state_emb import encode_state
from .nn.encoder import LayoutEncoder
from .algo.policy import PolicyNet
from .algo.group_rel import greedy_plan
from .algo.standard_tree import standard_tree_step

SEEDS_EVAL = range(1, 31)
OUT = "geosched/m7_mechanism.jsonl"
CFGS = [("bias", dict(mask="axial")), ("noBias", dict(mask="axial", w_d=0.0, w_c=0.0))]
G, J, JL, LR, ENT = 4, 2, 8, 1e-4, 0.03


def eval_domain(pol, inst, lt, sl, t_max: float = 10.0) -> dict:
    lay = sample_layout(inst.n_machines, lt, sl)
    es_x = encode_state(inst, lay, n_agv=2)
    tok = pol.encode_state(es_x)
    pl = greedy_plan(inst, pol, t_max=t_max, tok_emb=tok)
    w = SimWorld(inst, lay, dock_distance_matrix(build_corridor_graph(lay)),
                 SimConfig(n_agv=2))

    def lg(feat_np):
        lx = pol.agv_logits(torch.tensor(feat_np[None, None, :]))[0, 0]
        return int(torch.argmax(lx).item())

    ms = [w.run_gated(seed_chain=s, op_choices=pl, policy_l=lg)["makespan"]
          for s in SEEDS_EVAL]
    return {"mean": round(float(np.mean(ms)), 2), "median": round(float(np.median(ms)), 2),
            "min": round(float(np.min(ms)), 2), "max": round(float(np.max(ms)), 2)}


ENGINE_TAG = "flat"   # 引擎标记（防混跑教训：旧 train_step 行曾被 resume 误判为已完成）


def _done_pairs(out_path: str) -> set[tuple[str, str]]:
    """已完成的 (train, cfg) 组合（3 域全齐且引擎匹配才计入）——中断续跑。"""
    from collections import defaultdict
    import os
    cnt: dict[tuple[str, str], int] = defaultdict(int)
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("engine") != ENGINE_TAG:
                    continue
                cnt[(r.get("train"), r.get("cfg"))] += 1
    return {k for k, v in cnt.items() if v >= 3}


def main() -> None:
    inst = load_mk("mk01")
    es1 = encode_state(inst, sample_layout(inst.n_machines, "line", 1), n_agv=2)
    feat_dim = es1.tok_feat.shape[-1]
    done = _done_pairs(OUT)
    if done:
        print(f"[m7] resume: skip {sorted(done)}", flush=True)
    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as f:
        for train_mode in ("line1", "dr"):
            for cfg_name, enc_args in CFGS:
                if (train_mode, cfg_name) in done:
                    continue
                torch.manual_seed(0)
                pol = PolicyNet(enc=LayoutEncoder(feat_dim=feat_dim, **enc_args),
                                n_agv=2, n_feat_l=10)
                for s in range(200):
                    if train_mode == "dr":                      # DR：每步布局轮换 → 每步重建 es
                        lt = ("line", "U", "island")[s % 3]
                        sl = int((s // 3) % 10) + 1
                        es_s = encode_state(inst, sample_layout(inst.n_machines, lt, sl),
                                            n_agv=2)
                    else:
                        es_s = es1
                    standard_tree_step(pol, inst, seed=s, G=G, J=J, J_L=JL, lr=LR,
                                       n_agv=2, enc_state=es_s, ent_beta=ENT, norm="flat")
                for lt, sl in (("line", 2), ("U", 1), ("island", 1)):
                    ev = eval_domain(pol, inst, lt, sl)
                    row = {"train": train_mode, "cfg": cfg_name, "domain": f"{lt}{sl}",
                           "step": 200, "engine": ENGINE_TAG, **ev,
                           "elapsed": round(time.time() - t0, 1)}
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
    print("[m7] done")


if __name__ == "__main__":
    main()
