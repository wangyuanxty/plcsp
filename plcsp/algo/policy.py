"""策略网络——机台选择（S 层）+ AGV 派车（L 层）+ **路线选择（R 层）**三个头。

**无 critic**：组内相对优势用组内基线（见 `group_rel.py`），不需要价值网络。
2026-10-02：分批（B 层）头已删（分批环节砍除）；2026-10-03：critic 头 `v_head` 随 PPO 变体一并删除。
2026-10-03（P2 Task 4）：**两个头一律走编码器 token 嵌入**——L 头不再吃 `des.py` 手搓的
扁平向量（旧路径 `l_head` / `agv_logits` 与 S 头的 MLP 回退 `s_head` / `mach_logits` 一并删除）。
此前 L 头**不走编码器**，对生产侧结构性失明（看不到机台状态/计划/布局，`n_agv≥3` 时看不到
2 号以后的车）——这也解释了"L 在随机计划下无信号"的实测（见 spec §5.3.1 #4）。
2026-10-04：**R 头（`route_logits_emb`）恢复**——`route_logits` 当初随 ① 拥堵一并被砍
（"无拥堵时选远路严格更差"，spec §5.3），但 ① 后来在 `eb1d1da` 恢复而路线头漏恢复
（`docs/progress-log.md` §27.3/§28）。R 头是那次遗漏的补建，不是新发明。
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
    """π = π_S(机台候选) · π_L(AGV 派车) · π_R(路线候选)；无 critic。

    三个头读**同一份** token 嵌入（spec §5.3.1）：S 头 `mach_logits_emb` / L 头
    `agv_logits_emb` / R 头 `route_logits_emb`，嵌入均由 `forward_enc` 产出。
    ⚠️ `enc=None` 时**没有可用的头**——旧的扁平特征 MLP 回退路径（`s_head` / `mach_logits`）
    已在 P2 Task 4 删除，不存在第二条打分通路。

    三个头的打分输入同构：`token 嵌入 ⊕ 决策特征 ⊕ **候选特征**`。
    - **决策特征**（`feat_op` / `feat_task` / `feat_route`，形参 `n_feat_op` / `n_feat_task` /
      `n_feat_route`）与候选**无关**，broadcast 给所有候选；
    - **候选特征**（`feat_cand`，形参 `n_feat_cand` / `n_feat_route_cand`）**逐候选**——S 头放
      换型代价 `setup(prev_job_of_m, j)`、L 头放"该车到取货点的预计行驶时长"、R 头放
      "该路径的长度比 / 区段数 / 当前争用"。spec §5.3.1②：换型是 `(机台, 作业)` 的**交互量**，
      塞不进 M token，只能走这个槽（旧 MLP 路径本有 `feat_cand`，重写时不可丢）；R 头同理，
      候选是路径，在序列里没有 token（见 `route_logits_emb`）。

    ⚠️ **无 `n_agv` 形参**（评审 M-4 删）：车队规模由 `seg` 的 V 段长度定（`v_token_index`），
    网络结构里没有任何一处随车队规模变——旧的 `n_agv` 形参与其 `self.n_agv` 属性**全仓零
    读取方**，只会给读者"车队规模进网络"的错觉（要理解车队规模如何进网，看 V 段 token）。
    """
    def __init__(self, n_feat_op: int = 3,
                 hidden: int = 64, enc: LayoutEncoder | None = None,
                 n_feat_task: int = 4, n_feat_cand: int = 1,
                 n_feat_route: int = 4, n_feat_route_cand: int = 3):
        super().__init__()
        self.enc = enc
        self.optim: torch.optim.Optimizer | None = None   # 由训练器在首步惰性创建（Adam）
        if enc is not None:
            self.s_head_tok = nn.Sequential(          # 编码器 token 打分头（候选机台）
                nn.Linear(enc.d_model + n_feat_op + n_feat_cand, hidden), nn.GELU(),
                nn.Linear(hidden, 1))
            # 编码器 token 打分头（候选车辆）：任务特征 = 起送机台/目标机台/工序序号/该机是否需换型
            self.l_head_tok = nn.Sequential(
                nn.Linear(enc.d_model + n_feat_task + n_feat_cand, hidden), nn.GELU(),
                nn.Linear(hidden, 1))
            # 编码器打分头（候选路径，R）：行驶特征 = 起点/终点节点/是否负载/本车号；
            # 逐候选特征 = 长度比/区段数/争用（见 `route_logits_emb` 的"候选没有 token"说明）
            self.r_head_tok = nn.Sequential(
                nn.Linear(enc.d_model + n_feat_route + n_feat_route_cand, hidden), nn.GELU(),
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

    def route_logits_emb(self, tok: torch.Tensor, feat_drive: torch.Tensor,
                         feat_cand: torch.Tensor, cand_idx: torch.Tensor) -> torch.Tensor:
        """(1,N,d) × (1,1,F_route) × (1,k,F_cand) × (k,) → (1,1,k) 候选**路径**分数（R 头）。

        ⚠️ **与 S/L 两头的关键差别——路线候选在 token 序列里没有自己的 token**：
        序列只有 M（机台）/B（作业）/V（车辆）/G（全局）四段，候选是**路径/区段实体**，
        不是这三种实体中的任何一种（给区段加 token 是另一条机制 R2，不在本次恢复范围）。
        故本头的 `cand_idx` 不逐候选区分，它取**本车自己的 V token 下标**（k 个候选传同一个值）：
        - 语义 = "这辆车在做什么决定"，与 L 头（车辆实体）同源；
        - 作用 = 上下文 + 到编码器的梯度通路（R 头不是脱离编码器的第二条打分通路）；
        - **候选之间的分数差只能来自 `feat_cand`**（逐候选的长度比/区段数/当前争用）——
          `tok` 与 `feat_drive` 对 k 个候选是同一份输入，不携带候选间差异。
        ⚠️ **已知限度**（如实记下，不许含糊）：本头的"偏好哪条路"因此基本由逐候选特征决定，
        上下文只经 GELU 的非线性调节对各维的敏感度；要表达"同一辆车在不同状态下偏好不同路"，
        须等 R2 给区段/路径加 token（`progress-log.md` §27.2 的配对项）。
        """
        cand = feat_cand[0] if feat_cand.dim() == 3 else feat_cand     # (k, F_cand)
        tok_c = tok[0, cand_idx.long()]                                # (k, d)
        fd = feat_drive.expand(1, tok_c.shape[0], -1)[0]               # (k, F_route)
        return self.r_head_tok(torch.cat([tok_c, fd, cand], dim=-1)).squeeze(-1).unsqueeze(0).unsqueeze(1)
