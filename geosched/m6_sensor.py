"""M6-传感器：1-步"耦合温度计"—— 判定 SA-GRPO 树式分层在任务上的理论收益空间。

原理：LoTV 收益空间 ∝ 层级条件结构解释的方差 = ratio_S/ratio_B = σ(层节点)/σ(叶)：
  ratio < 0.6 → 分层吸收显著方差 → 树 vs flat 的 800 步实验值得跑（SA-GRPO 有救的证据）
  ratio ≈ 1.0 → 层级不解释方差 → 树=flat 是任务特性（结论坐实，无需再烧 3h）
配置（各跑 5 步取 ratio 均值，防单步噪声）：
  c1_single        : 现状单目标（对照，预期≈1）
  c2_axis_weighted : 层级-目标对齐多目标（w=(mks1,tardy2,moves1,energy0.5)；轴控目标→层级有可归因方差）
  c3_congestion1   : 强拥塞单车（n_agv=1；L 轴信号增强+依赖 S）
输出：geosched/m6_sensor.jsonl。运行：python -m geosched.m6_sensor（~3min）。
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from .env.instances import load_mk
from .env.layout import sample_layout
from .nn.state_emb import encode_state
from .nn.encoder import LayoutEncoder
from .algo.policy import PolicyNet
from .algo.standard_tree import standard_tree_step

STEPS_PER_CFG = 5
OUT = "geosched/m6_sensor.jsonl"

CONFIGS = [
    ("c1_single", dict(reward_mode="mks", n_agv=2)),
    ("c4_topo_random", dict(reward_mode="mks", n_agv=2, topo_random=True, feat_b=7)),
    ("c2_axis_weighted", dict(reward_mode="channels", w_channels=(1.0, 2.0, 1.0, 0.5), n_agv=2)),
    ("c3_congestion1", dict(reward_mode="mks", n_agv=1)),
]


def main() -> None:
    inst = load_mk("mk01")
    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as f:
        for tag, kw in CONFIGS:
            es = encode_state(inst, sample_layout(inst.n_machines, "line", 1),
                              n_agv=kw["n_agv"])
            pol = PolicyNet(enc=LayoutEncoder(feat_dim=es.tok_feat.shape[-1]),
                            n_agv=kw["n_agv"], n_feat_l=6 + kw["n_agv"] + 2,
                            feat_b=kw.get("feat_b", 4))          # c4: B 头 7 维（实例4+布局3）
            rs, rb = [], []
            step_kw = {k: v for k, v in kw.items()
                       if k in ("reward_mode", "w_channels", "topo_random")}
            for s in range(STEPS_PER_CFG):
                _, d = standard_tree_step(pol, inst, seed=s, G=4, J=2, J_L=8,
                                          lr=1e-4, n_agv=kw["n_agv"], enc_state=es,
                                          ent_beta=0.03, norm="within", **step_kw)
                rs.append(d["ratio_S"]); rb.append(d["ratio_B"])
            row = {"config": tag, **{k: v for k, v in kw.items()
                                     if not isinstance(v, tuple)},
                   "ratio_S_mean": round(float(np.mean(rs)), 3),
                   "ratio_B_mean": round(float(np.mean(rb)), 3),
                   "ratio_S_range": [round(float(min(rs)), 3), round(float(max(rs)), 3)],
                   "ratio_B_range": [round(float(min(rb)), 3), round(float(max(rb)), 3)],
                   "elapsed": round(time.time() - t0, 1)}
            print(json.dumps(row, ensure_ascii=False), flush=True)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    print("[m6sensor] done\n判读: ratio<0.6=>树收益空间大(跑完整实验) | ~1.0=>任务无耦合(结论坐实)")


if __name__ == "__main__":
    main()
