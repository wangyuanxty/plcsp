"""度量感知几何注意力 · 证据子实验（设计 A：固定权重 + 对称偏置验证）。

实验 A（数据效率）  ：MK01×line(1)，{bias, noBias} × 200 步 GRPO（train_step G=8/J=1/z），
                      每 50 步贪心评估 —— 期望：短训练下偏置先验加速（bias 更早下降）。
实验 B（跨布局先验）：训练 200 步（line1）→ 评估 {line2, U1, island1}（几何偏置=不变
                      先验 → 泛化更稳；noBias 对照）。
评估 = greedy_plan(argmax, 编码器路径) × 3 扰动种子均值；两配置共用几何特征（仅偏置开关差）。
运行：python -m plcsp.m5_bias_verify（后台 ~15-20min）
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from .env.instances import load_mk
from .env.layout import sample_layout
from .env.des import rollout_evaluate
from .nn.state_emb import encode_state
from .nn.encoder import LayoutEncoder
from .algo.policy import PolicyNet
from .algo.group_rel import train_step, greedy_plan

SEEDS_EVAL = (1, 11, 21)
OUT = "plcsp/m5_bias.jsonl"
CFGS = [("bias", dict(mask="axial")), ("noBias", dict(mask="axial", w_d=0.0, w_c=0.0))]


def eval_on(pol, inst, es, lt, sl, t_max: float = 10.0) -> float:
    tok = pol.encode_state(es)
    ms = []
    for s in SEEDS_EVAL:
        pl = greedy_plan(inst, pol, t_max=t_max, tok_emb=tok)
        ms.append(-rollout_evaluate(inst, pl, seed=s, layout_type=lt, seed_layout=sl))
    return float(np.mean(ms))


def main() -> None:
    inst = load_mk("mk01")
    es1 = encode_state(inst, sample_layout(inst.n_machines, "line", 1), n_agv=2)
    feat_dim = es1.tok_feat.shape[-1]
    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as f:
        for cfg_name, enc_args in CFGS:
            torch.manual_seed(0)
            pol = PolicyNet(enc=LayoutEncoder(feat_dim=feat_dim, **enc_args))
            for s in range(200):
                train_step(pol, inst, seed=s, G=8, J=1, mode="z", enc_state=es1)
                if (s + 1) % 50 == 0:
                    row = {"exp": "A", "cfg": cfg_name, "step": s + 1,
                           "greedy_mean": round(eval_on(pol, inst, es1, "line", 1), 2),
                           "elapsed": round(time.time() - t0, 1)}
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
            for lt, sl in (("line", 2), ("U", 1), ("island", 1)):
                es_x = encode_state(inst, sample_layout(inst.n_machines, lt, sl), n_agv=2)
                row = {"exp": "B", "cfg": cfg_name, "step": 200, "layout": f"{lt}{sl}",
                       "greedy_mean": round(eval_on(pol, inst, es_x, lt, sl), 2),
                       "elapsed": round(time.time() - t0, 1)}
                print(json.dumps(row, ensure_ascii=False), flush=True)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    print("[bias] done")


if __name__ == "__main__":
    main()
