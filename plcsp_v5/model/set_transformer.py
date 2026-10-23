"""Minimal Set Transformer blocks (Lee et al., ICML 2019).

Used by ``model/actor_critic_st.py`` as a drop-in replacement for the
GIN operation encoder and the 2-node machine GAT.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MAB(nn.Module):
    """Multihead Attention Block between two sets."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.att = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.ln_ff = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None,
                attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, N, D) queries; y: (B, M, D) keys/values.
        # attn_bias: optional additive bias, shape (B*num_heads, N, M).
        h = self.att(
            self.ln_q(x), self.ln_kv(y), self.ln_kv(y),
            key_padding_mask=key_padding_mask,
            attn_mask=attn_bias,
            need_weights=False)[0]
        x = x + h
        x = x + self.ff(self.ln_ff(x))
        return x


class SAB(nn.Module):
    """Self-Attention Block: MAB(X, X)."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mab = MAB(dim, num_heads, dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None,
                attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        return self.mab(x, x, key_padding_mask=key_padding_mask,
                        attn_bias=attn_bias)


class PMA(nn.Module):
    """Pooling by Multihead Attention with learnable seed vectors."""

    def __init__(self, dim: int, num_seeds: int = 1,
                 num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.seeds = nn.Parameter(torch.zeros(1, num_seeds, dim))
        nn.init.normal_(self.seeds, std=0.02)
        self.mab = MAB(dim, num_heads, dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        seeds = self.seeds.expand(x.size(0), -1, -1)
        return self.mab(seeds, x, key_padding_mask=key_padding_mask)


class SetTransformerEncoder(nn.Module):
    """Encode a set of tokens; return per-token embeddings and a pooled vector."""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, num_heads: int = 4,
                 num_seeds: int = 1, dropout: float = 0.0,
                 project_input: bool = True):
        super().__init__()
        if project_input:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.input_proj = nn.Identity()
        self.sabs = nn.ModuleList([
            SAB(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.pma = PMA(hidden_dim, num_seeds, num_heads, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None,
                attn_bias: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, N, input_dim); mask: (B, N) True = valid.
        # attn_bias: optional additive graph bias, (B*num_heads, N, N).
        h = self.input_proj(x)
        kpm = None if mask is None else ~mask.bool()
        for sab in self.sabs:
            h = sab(h, key_padding_mask=kpm, attn_bias=attn_bias)
        pooled = self.pma(h, key_padding_mask=kpm).mean(dim=1)
        return self.norm(h), pooled
