"""M5 正式 v1：编码器路径长跑（runner 版）—— GRPO-ENC vs SA-GRPO-ENC（MK01 × line）。

依据：预热 + 修复版 v1（tok_emb 漏传 bug 修复后）显示编码器路径真实学习
（310.87 → 292.46 @step 900）；延至 2000 步观察是否继续下降/平台，并验证 runner 续训。

口径：MK01 官方实例；布局 line seed=1；PolicyNet(enc=LayoutEncoder(128/4/8))；
两方法共用特征系统，仅训练器不同（SA-GRPO vs GRPO 的干净对照）；
runner：checkpoints/m5v1/{method}/ckpt.pt（每 50 步）→ 中断可续（resume=True）；
每 100 步贪心评估 × 3 扰动种子 → {"mean","min"}（写入 metrics.jsonl 的 eval 行）。
运行：python -m geosched.m5_v1（后台；~2h，可中断续）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .env.instances import load_mk
from .env.layout import sample_layout
from .env.des import rollout_evaluate
from .nn.state_emb import encode_state
from .nn.encoder import LayoutEncoder
from .algo.policy import PolicyNet
from .algo.group_rel import train_step, tree_step, greedy_plan, sample_plan
from .algo.runner import run_training, resume_training

STEPS, EVAL_EVERY, SAVE_EVERY = 1000, 100, 50
SEEDS_EVAL = (1, 11, 21)
SAMPLE_K = 32                   # 采样评估口径（分布质量：mean/min）

METHODS = [
    ("GRPO-ENC", train_step, dict(G=8, J=1, mode="z")),
    ("SA-GRPO-ENC", tree_step, dict(G=8, J=1, cap_opts=[10, 5, 2])),
]


def _load_or_new(tag: str, feat_dim: int):
    path = f"checkpoints/m5v1/{tag}"
    if Path(path, "ckpt.pt").exists():
        pol, ck, st = resume_training(path)
        print(f"[m5v1] resume {tag} from step {st}", flush=True)
        return pol
    torch.manual_seed(0)                                # 同一起点才可比
    return PolicyNet(enc=LayoutEncoder(feat_dim=feat_dim))


def main() -> None:
    inst = load_mk("mk01")
    es = encode_state(inst, sample_layout(inst.n_machines, "line", seed=1), n_agv=2)
    feat_dim = es.tok_feat.shape[-1]

    def make_eval(policy):
        tok = policy.encode_state(es)
        ms_g = [-rollout_evaluate(inst, greedy_plan(inst, policy, tok_emb=tok), seed=s)
                for s in SEEDS_EVAL]
        ms_s = [-rollout_evaluate(inst, sample_plan(inst, policy, 10.0, tok_emb=tok), seed=1)
                for _ in range(SAMPLE_K)]
        return {"greedy_mean": float(np.mean(ms_g)),
                "sample_mean": float(np.mean(ms_s)), "sample_min": float(np.min(ms_s))}

    for tag, fn, kw in METHODS:
        pol = _load_or_new(tag, feat_dim)
        run_training(pol, inst, steps=STEPS, step_fn=fn, seed0=0,
                     run_dir=f"checkpoints/m5v1/{tag}",
                     step_kwargs={**kw, "enc_state": es},
                     save_every=SAVE_EVERY, resume=True,
                     eval_fn=make_eval, eval_every=EVAL_EVERY)
    print("[m5v1] done")


if __name__ == "__main__":
    main()
