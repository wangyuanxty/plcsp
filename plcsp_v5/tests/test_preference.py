"""Tests for the preference embedding."""
from __future__ import annotations

import torch

from plcsp_v5.model.preference import PreferenceEmbedding


def test_preference_embedding_shape():
    emb = PreferenceEmbedding(3, hidden_dim=8, out_dim=4)
    out = emb(torch.tensor([[0.4, 0.4, 0.2], [1.0, 0.0, 0.0]]))
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()


def test_preference_embedding_scale_invariance():
    emb = PreferenceEmbedding(3, hidden_dim=8, out_dim=4)
    a = emb(torch.tensor([[0.4, 0.4, 0.2]]))
    b = emb(torch.tensor([[4.0, 4.0, 2.0]]))
    assert torch.allclose(a, b, atol=1e-6)
