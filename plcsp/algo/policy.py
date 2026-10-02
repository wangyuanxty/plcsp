"""策略网络——机台选择（S 层）+ AGV 派车（L 层）两个头。

**无 critic**：组内相对优势用组内基线（见 `group_rel.py`），不需要价值网络。
2026-10-02：分批（B 层）头已删（分批环节砍除）；2026-10-03：critic 头 `v_head` 随 PPO 变体一并删除。
2026-10-03（P2 Task 4）：**两个头一律走编码器 token 嵌入**——L 头不再吃 `des.py` 手搓的
扁平向量（旧路径 `l_head` / `agv_logits` 与 S 头的 MLP 回退 `s_head` / `mach_logits` 一并删除）。
此前 L 头**不走编码器**，对生产侧结构性失明（看不到机台状态/计划/布局，`n_agv≥3` 时看不到
2 号以后的车）——这也解释了"L 在随机计划下无信号"的实测（见 spec §5.3.1 #4）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..nn.encoder import LayoutEncoder


def v_token_index(seg: tuple[int, int, int, int]) -> list[int]:
    """V 段 token 在序列中的位置（= M 段 + B 段之后）。有序，第 i 个 = 第 i 台车。"""
    n_m, n_b, n_v, _ = seg
    return list(range(n_m + n_b, n_m + n_b + n_v))


class PolicyNet(nn.Module):
    """π = π_S(机台候选) · π_L(AGV 派车)；无 critic。

    两个头读**同一份** token 嵌入（spec §5.3.1）：S 头 `mach_logits_emb` / L 头 `agv_logits_emb`，
    嵌入均由 `forward_enc` 产出。⚠️ `enc=None` 时**没有可用的头**——旧的扁平特征 MLP 回退
    路径（`s_head` / `mach_logits`）已在 P2 Task 4 删除，不存在第二条打分通路。

    两个头的打分输入同构：`token 嵌入 ⊕ 决策特征 ⊕ **候选特征**`。
    - **决策特征**（`feat_op` / `feat_task`，形参 `n_feat_op` / `n_feat_task`）与候选**无关**，
      broadcast 给所有候选；
    - **候选特征**（`feat_cand`，形参 `n_feat_cand`）**逐候选**——S 头放换型代价
      `setup(prev_job_of_m, j)`、L 头放"该车到取货点的预计行驶时长"。spec §5.3.1②：换型是
      `(机台, 作业)` 的**交互量**，塞不进 M token，只能走这个槽（旧 MLP 路径本有 `feat_cand`，
      重写时不可丢）。
    """
    def __init__(self, n_feat_op: int = 3,
                 hidden: int = 64, enc: LayoutEncoder | None = None, n_agv: int = 2,
                 n_feat_task: int = 4, n_feat_cand: int = 1):
        super().__init__()
        self.enc = enc
        self.n_agv = n_agv                       # 仅留痕：候选台数现由 tok/seg 定（`v_token_index`）
        self.optim: torch.optim.Optimizer | None = None   # 由训练器在首步惰性创建（Adam）
        if enc is not None:
            self.s_head_tok = nn.Sequential(          # 编码器 token 打分头（候选机台）
                nn.Linear(enc.d_model + n_feat_op + n_feat_cand, hidden), nn.GELU(),
                nn.Linear(hidden, 1))
            # 编码器 token 打分头（候选车辆）：任务特征 = 起送机台/目标机台/工序序号/该机是否需换型
            self.l_head_tok = nn.Sequential(
                nn.Linear(enc.d_model + n_feat_task + n_feat_cand, hidden), nn.GELU(),
                nn.Linear(hidden, 1))

    def forward_enc(self, tok_feat: torch.Tensor | np.ndarray,
                    seg: tuple[int, int, int, int]
                    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """编码器前向（两个头共用）——**唯一的 numpy→torch 转换点**；无编码器返回 (None, None)。

        `build_tok`（Task 2）产 numpy `(N, F_MAX)`，编码器要 torch `(1, N, F_MAX)`：
        numpy/列表 → float32 张量，2 维 → 补 batch 维，在此**一处**统一（其余调用方只传 torch）。
        """
        if self.enc is None:
            return None, None
        x = (tok_feat.float() if torch.is_tensor(tok_feat)
             else torch.tensor(np.asarray(tok_feat, dtype=np.float32)))  # 复制：不共享上游 numpy 内存
        if x.dim() == 2:                                    # (N, F_MAX) → (1, N, F_MAX)
            x = x.unsqueeze(0)
        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"forward_enc 要 (N,F) 或 (1,N,F) 的 token 特征，"
                             f"收到 shape={tuple(x.shape)}")
        return self.enc(x, seg)

    def mach_logits_emb(self, tok: torch.Tensor, feat_op: torch.Tensor,
                        feat_cand: torch.Tensor, cand_idx: torch.Tensor) -> torch.Tensor:
        """(1,N,d) × (1,1,F_op) × (1,n_cand,F_cand) × (n_cand,) → (1,1,n_cand) 候选分数（M3c）。

        M 段位于序列前部：cand_idx = 机台编号即序列位置（B/V 段追加于后）。
        `feat_op` 与候选**无关**（broadcast 给所有候选）；`feat_cand` **逐候选**
        （也收 `(n_cand, F_cand)`）——⑤ 换型代价是 `(机台, 作业)` 的交互量，只能走这个槽
        （spec §5.3.1②），塞不进 M token。
        """
        tok_c = tok[0, cand_idx.long()]                      # (n_cand, d)
        op = feat_op.expand(1, tok_c.shape[0], -1)[0]        # (n_cand, F_op)
        cand = feat_cand[0] if feat_cand.dim() == 3 else feat_cand     # (n_cand, F_cand)
        return self.s_head_tok(torch.cat([tok_c, op, cand], dim=-1)).squeeze(-1).unsqueeze(0).unsqueeze(1)

    def agv_logits_emb(self, tok: torch.Tensor, feat_task: torch.Tensor,
                       feat_cand: torch.Tensor, cand_idx: torch.Tensor) -> torch.Tensor:
        """(1,N,d) × (1,1,F_task) × (1,n_cand,F_cand) × (n_cand,) → (1,1,n_cand)。

        与 `mach_logits_emb` **对称**：候选对象的 token 嵌入 ⊕ 决策特征 ⊕ **候选特征** → 打分。
        `feat_cand` = "该车到取货点的预计行驶时长"（spec §5.3.1 V 段坐标维的既定用途：
        派最近的车）——没有这个槽，L 头落不了该语义。
        2026-10-03 新增：此前 L 头吃 `des.py` 手搓的 11 维扁平向量、**不走编码器**，
        导致它对生产侧结构性失明（看不到机台状态/计划/布局，n_agv≥3 时看不到 2 号以后的车）。
        """
        tok_c = tok[0, cand_idx.long()]                       # (n_cand, d)
        ft = feat_task.expand(1, tok_c.shape[0], -1)[0]       # (n_cand, F_task)
        cand = feat_cand[0] if feat_cand.dim() == 3 else feat_cand     # (n_cand, F_cand)
        return self.l_head_tok(torch.cat([tok_c, ft, cand], dim=-1)).squeeze(-1).unsqueeze(0).unsqueeze(1)
