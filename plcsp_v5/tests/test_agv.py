"""Unit tests for the Stage-0 AGV agent."""
from __future__ import annotations

import numpy as np
import torch

from plcsp_v5.agv.actor import AgvActor_SetTransformer
from plcsp_v5.agv.state import (AgvState, AgvStatus, agv_features,
                                available_mask, context_features,
                                split_cross_edge)


def _travel(_src: int, _dst: int) -> float:
    return 2.0


def test_assign_timing():
    agv = AgvState(idx=0, edge=0, location=0)
    out = agv.assign(now=0.0, pickup=1, dropoff=2, travel=_travel)

    assert out["arrive_pickup"] == 2.0
    assert out["depart_pickup"] == 2.0
    assert out["arrive_dropoff"] == 4.0
    assert agv.free_time == 4.0
    assert agv.location == 2
    assert agv.status == AgvStatus.IDLE


def test_assign_waits_for_free_time():
    agv = AgvState(idx=0, edge=0, location=0, free_time=5.0)
    out = agv.assign(now=0.0, pickup=1, dropoff=2, travel=_travel)

    assert out["arrive_pickup"] == 7.0
    assert agv.free_time == 9.0


def test_features_shapes():
    agvs = [AgvState(idx=i, edge=0, location=i) for i in range(3)]
    feats = agv_features(agvs, pickup=0, dropoff=2, travel=_travel, now=1.0)
    ctx = context_features(pickup=0, dropoff=2, pickup_edge=0,
                           dropoff_edge=1, travel=_travel, now=1.0,
                           n_available=3, n_agv=3)
    mask = available_mask(agvs, now=1.0)

    assert feats.shape == (3, 11)
    assert ctx.shape == (6,)
    assert mask.tolist() == [True, True, True]


def test_busy_agv_is_not_available():
    agvs = [AgvState(idx=0, edge=0, location=0),
            AgvState(idx=1, edge=0, location=0, free_time=5.0)]

    mask = available_mask(agvs, now=1.0)

    assert mask.tolist() == [True, False]


def test_available_mask_filters_home_edge():
    agvs = [AgvState(idx=0, edge=0, location=0),
            AgvState(idx=1, edge=1, location=3)]

    mask = available_mask(agvs, now=1.0, edge=0)

    assert mask.tolist() == [True, False]


def test_split_cross_edge_normal():
    src, delay, dst = split_cross_edge(total=13.0, cross_delay=5.0,
                                       leg_ratio=0.5)

    assert np.isclose(src, 4.0)
    assert np.isclose(delay, 5.0)
    assert np.isclose(dst, 4.0)


def test_split_cross_edge_clamps_delay():
    src, delay, dst = split_cross_edge(total=3.0, cross_delay=5.0,
                                       leg_ratio=0.5)

    assert np.isclose(src, 0.0)
    assert np.isclose(delay, 3.0)
    assert np.isclose(dst, 0.0)


def test_actor_wait_when_all_busy():
    model = AgvActor_SetTransformer(feat_dim=11, ctx_dim=6)
    agv_feat = torch.randn(2, 4, 11)
    ctx = torch.randn(2, 6)
    mask = torch.zeros(2, 4, dtype=torch.bool)  # none available

    prob, value = model(agv_feat, ctx, mask)

    assert prob.shape == (2, 5)
    assert value.shape == (2, 2)
    assert torch.isfinite(prob).all()
    assert torch.allclose(prob[:, -1], torch.ones(2), atol=1e-5)


def test_actor_masks_unavailable_agvs():
    model = AgvActor_SetTransformer(feat_dim=11, ctx_dim=6)
    agv_feat = torch.randn(2, 4, 11)
    ctx = torch.randn(2, 6)
    mask = torch.tensor([[True, False, False, False],
                         [True, True, False, False]])

    prob, _ = model(agv_feat, ctx, mask)

    assert torch.all(prob[0, 1:4] == 0)
    assert torch.all(prob[1, 2:4] == 0)
    assert torch.isfinite(prob).all()
    assert np.allclose(prob.sum(dim=-1).detach().numpy(), 1.0, atol=1e-5)
