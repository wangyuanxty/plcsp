"""Stage-0 AGV actor: a Set Transformer over AGV tokens plus a WAIT token.

Action space: ``0 .. n_agv-1`` select an AGV, ``n_agv`` means WAIT.
The WAIT token is always available, so the masked softmax never sees an
all ``-inf`` row when every AGV is busy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # running from the project root
    from plcsp_v5.model.set_transformer import SetTransformerEncoder
except ImportError:  # running from inside plcsp_v5
    from model.set_transformer import SetTransformerEncoder


class AgvActor_SetTransformer(nn.Module):
    def __init__(self, feat_dim: int = 11, ctx_dim: int = 6,
                 hidden: int = 128, num_layers: int = 2,
                 num_heads: int = 4):
        super().__init__()
        self.feat_dim = feat_dim
        self.ctx_dim = ctx_dim
        self.input_dim = feat_dim + ctx_dim
        self.encoder = SetTransformerEncoder(
            input_dim=self.input_dim,
            hidden_dim=hidden,
            num_layers=num_layers,
            num_heads=num_heads,
            num_seeds=1,
        )
        self.wait_token = nn.Parameter(torch.zeros(1, 1, self.input_dim))
        nn.init.normal_(self.wait_token, std=0.02)
        self.score = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, agv_feat: torch.Tensor, ctx: torch.Tensor,
                avail_mask: torch.Tensor):
        """Return ``(prob, value)``.

        agv_feat : (B, N, feat_dim)
        ctx      : (B, ctx_dim)
        avail_mask: (B, N) bool, True = selectable
        prob     : (B, N+1)  last entry is WAIT
        value    : (B, 2)    local critic for (makespan, idle time)
        """
        batch, n_agv, _ = agv_feat.shape
        ctx_rep = ctx.unsqueeze(1).expand(batch, n_agv, -1)
        tokens = torch.cat([agv_feat, ctx_rep], dim=-1)

        wait = self.wait_token.expand(batch, 1, -1)
        tokens = torch.cat([tokens, wait], dim=1)          # (B, N+1, D)

        h, pooled = self.encoder(tokens)                    # (B, N+1, H), (B, H)
        pooled_rep = pooled.unsqueeze(1).expand_as(h)
        scores = self.score(torch.cat([h, pooled_rep], dim=-1)).squeeze(-1)

        wait_mask = torch.zeros(batch, 1, dtype=torch.bool,
                                device=agv_feat.device)
        mask = torch.cat([~avail_mask.bool(), wait_mask], dim=1)
        scores = scores.masked_fill(mask, float("-inf"))
        prob = F.softmax(scores, dim=-1)
        value = self.critic(pooled)
        return prob, value

    @staticmethod
    def wait_index(n_agv: int) -> int:
        return n_agv
