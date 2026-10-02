"""策略网络——机台选择（S 层）+ AGV 派车（L 层）两个头。无 critic（组内相对优势见 group_rel.py）。

⚠️ 2026-10-02：**分批（B 层）头已删除**——分批环节整条砍除（`progress-log.md` §12.5）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..nn.encoder import LayoutEncoder


class PolicyNet(nn.Module):
    """π = π_S(机台候选) · π_L(AGV 派车)；无 critic。

    可选 enc=LayoutEncoder → s_head_tok 路由（mach_logits_emb）；
    不传 enc 时走特征 MLP 路径。
    """
    def __init__(self, feat_op: int = 3,
                 hidden: int = 64, enc: LayoutEncoder | None = None, n_agv: int = 2,
                 n_feat_l: int = 8):
        super().__init__()
        self.enc = enc
        self.s_head = nn.Sequential(nn.Linear(feat_op * 2, hidden), nn.GELU(),
                                    nn.Linear(hidden, 1))
        if enc is not None:
            self.s_head_tok = nn.Sequential(          # 编码器 token 打分头（候选机台）
                nn.Linear(enc.d_model + feat_op, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.l_head = nn.Sequential(nn.Linear(n_feat_l, hidden), nn.GELU(),
                                    nn.Linear(hidden, n_agv))  # L 层：任务特征(+车状态)→AGV 候选
        self.v_head = nn.Sequential(nn.Linear(n_feat_l, hidden), nn.GELU(),
                                    nn.Linear(hidden, 1))      # critic：任务特征→终局价值基线

    def v(self, feat_l: torch.Tensor) -> torch.Tensor:
        """(1,1,F_l) → (1,) 终局价值（critic 基线；PPO/基线策略用）。"""
        return self.v_head(feat_l).squeeze()

    def mach_logits(self, feat_op: torch.Tensor, feat_cand: torch.Tensor) -> torch.Tensor:
        """(B,1,F_op) × (B,Ncand,F_cand) → (B,1,Ncand) 候选分数。"""
        op = feat_op.expand(-1, feat_cand.shape[1], -1)                    # (B,Ncand,F)
        return self.s_head(torch.cat([op, feat_cand], dim=-1)).squeeze(-1).unsqueeze(1)

    def mach_logits_emb(self, tok: torch.Tensor, feat_op: torch.Tensor,
                        cand_idx: torch.Tensor) -> torch.Tensor:
        """(1,N,d) 编码器全序列嵌入 × (1,1,F_op) × (n_cand,) → (1,1,n_cand) 候选分数（M3c）。

        M 段位于序列前部：cand_idx = 机台编号即序列位置（B/V 段追加于后）。
        """
        tok_c = tok[0, cand_idx.long()]                      # (n_cand, d)
        op = feat_op.expand(1, tok_c.shape[0], -1)[0]        # (n_cand, F_op)
        return self.s_head_tok(torch.cat([tok_c, op], dim=-1)).squeeze(-1).unsqueeze(0).unsqueeze(1)

    def agv_logits(self, feat_l: torch.Tensor) -> torch.Tensor:
        """(1,1,F_l) 任务特征 → (1,1,n_agv) AGV 候选分数（L 层决策头）。"""
        return self.l_head(feat_l)

    def encode_state(self, enc_state) -> torch.Tensor | None:
        """enc_state 特征 → 全序列 token 嵌入 (1,N,d)（带图供更新）；无编码器返回 None。

        2026-10-02：几何删除后 `EncState` 不再含 dist/conf，故只传 (tok_feat, seg)。
        """
        if self.enc is None:
            return None
        tok, _ = self.enc(enc_state.tok_feat, enc_state.seg)
        return tok
