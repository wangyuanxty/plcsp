"""Stage-0 AGV actor: a Set Transformer over AGV tokens.

Action space: ``0 .. n_agv-1`` selects an AGV. Busy AGVs stay selectable;
choosing one pays the wait in the environment. The mask only filters AGVs
from other workshops. Callers must leave at least one AGV unmasked.
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
                avail_mask: torch.Tensor,
                logit_bias: torch.Tensor | None = None):
        """Return ``(prob, value)``.

        agv_feat : (B, N, feat_dim)
        ctx      : (B, ctx_dim)
        avail_mask: (B, N) bool, True = selectable
        logit_bias: optional (B, N) additive rule prior
        prob     : (B, N)
        value    : (B, 2)    local critic for (makespan, idle time)
        """
        batch, n_agv, _ = agv_feat.shape
        ctx_rep = ctx.unsqueeze(1).expand(batch, n_agv, -1)
        tokens = torch.cat([agv_feat, ctx_rep], dim=-1)    # (B, N, D)

        h, pooled = self.encoder(tokens)                    # (B, N, H), (B, H)
        pooled_rep = pooled.unsqueeze(1).expand_as(h)
        scores = self.score(torch.cat([h, pooled_rep], dim=-1)).squeeze(-1)
        if logit_bias is not None:
            scores = scores + logit_bias.to(
                device=scores.device, dtype=scores.dtype)

        mask = ~avail_mask.bool()                           # True = masked
        all_masked = mask.all(dim=1, keepdim=True)
        mask = mask & ~all_masked
        scores = scores.masked_fill(mask, float("-inf"))
        prob = F.softmax(scores, dim=-1)
        value = self.critic(pooled)
        return prob, value
