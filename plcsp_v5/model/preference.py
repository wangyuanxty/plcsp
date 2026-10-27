"""Preference embedding for the E2E multi-objective scheduler.

The raw three reward weights are replaced by a learned vector that is
injected into the job head, the machine head, and the global critic.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PreferenceEmbedding(nn.Module):
    def __init__(self, in_dim: int = 3, hidden_dim: int = 32,
                 out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, lam: torch.Tensor) -> torch.Tensor:
        lam = lam / lam.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return self.net(lam)
