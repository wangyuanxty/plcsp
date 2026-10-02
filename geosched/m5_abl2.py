"""M5 消融② v0：编码器结构对照 —— 同特征系统（几何特征）下 {axial | full | noBias | mlp}。

设计（控制变量）：算法层统一 GRPO（train_step G=8/J=1/z）；同一 seed0 下随机流一致
（策略初始化相同、采样序列相同）→ 曲线差异 = 结构差异：
- axial  ：双轴解耦掩码 + 几何偏置（默认 LayoutEncoder）—— 论文主配置
- full   ：全注意力掩码 + 几何偏置（检验"轴解耦" vs "全连接"）
- noBias ：轴向掩码 + 偏置 w_d=w_c=0（检验"几何偏置本身"贡献——设计 §2.3 核心机制）
- mlp    ：无编码器（时长特征下界；MLP 平台 373.5 先验）
每结构 × 3 种子 × 800 步（覆盖 310→292 翻盘区间），每 100 步双口径评估（greedy + 32 采样
mean/min）。输出：checkpoints/abl2/{tag}-s{seed}/metrics.ndjson（runner 格式：r 行 + eval 行）。
运行：python -m geosched.m5_abl2（后台 ~2h）
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .env.instances import load_mk
from .env.layout import sample_layout
from .env.des import rollout_evaluate
from .nn.state_emb import encode_state
from .nn.encoder import LayoutEncoder
from .algo.policy import PolicyNet
from .algo.group_rel import train_step, greedy_plan, sample_plan
from .algo.runner import run_training

STEPS, EVAL_EVERY = 800, 100
SEEDS = (0, 1, 2)
SAMPLE_K = 32

STRUCTS = [
    ("axial", dict(mask="axial")),
    ("full", dict(mask="full")),
    ("noBias", dict(mask="axial", w_d=0.0, w_c=0.0)),
    ("mlp", None),                       # enc=None → MLP 特征路径
]


def make_policy(enc_args: dict | None, feat_dim: int):
    return PolicyNet(enc=(LayoutEncoder(feat_dim=feat_dim, **enc_args) if enc_args else None))


def main() -> None:
    inst = load_mk("mk01")
    es = encode_state(inst, sample_layout(inst.n_machines, "line", seed=1), n_agv=2)
    feat_dim = es.tok_feat.shape[-1]
    t0 = time.time()
    for tag, enc_args in STRUCTS:
        for seed in SEEDS:
            torch.manual_seed(seed)                     # 同 seed 跨结构同起点/采样流
            pol = make_policy(enc_args, feat_dim)

            def make_eval(pol=pol):
                tok = pol.encode_state(es)
                ms_g = [-rollout_evaluate(inst, greedy_plan(inst, pol, tok_emb=tok), seed=s)
                        for s in (1, 11, 21)]
                ms_s = [-rollout_evaluate(inst, sample_plan(inst, pol, 10.0, tok_emb=tok),
                                          seed=1) for _ in range(SAMPLE_K)]
                return {"greedy_mean": float(np.mean(ms_g)),
                        "sample_mean": float(np.mean(ms_s)),
                        "sample_min": float(np.min(ms_s))}

            run_training(pol, inst, steps=STEPS, step_fn=train_step, seed0=seed * 1000,
                         run_dir=f"checkpoints/abl2/{tag}-s{seed}",
                         step_kwargs=dict(G=8, J=1, mode="z", enc_state=es),
                         save_every=50, resume=False,
                         eval_fn=make_eval, eval_every=EVAL_EVERY)
            print(f"[abl2] {tag} s{seed} done @{time.time()-t0:.0f}s", flush=True)
    print("[abl2] done")


if __name__ == "__main__":
    main()
