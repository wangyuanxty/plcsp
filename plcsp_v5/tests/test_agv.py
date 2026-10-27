"""Unit tests for the Stage-0 AGV agent."""
from __future__ import annotations

import numpy as np
import torch

from plcsp_v5.agv.actor import AgvActor_SetTransformer
from plcsp_v5.agv.state import (AgvState, AgvStatus, agv_features,
                                available_mask, context_features,
                                earliest_free_agv, split_cross_edge,
                                transport_timing)


def _travel(_src: int, _dst: int) -> float:
    return 2.0


def test_earliest_free_agv_respects_mask():
    agvs = [
        AgvState(idx=0, edge=0, location=0, free_time=5.0),
        AgvState(idx=1, edge=0, location=0, free_time=1.0),
        AgvState(idx=2, edge=1, location=0, free_time=0.0),
    ]
    assert earliest_free_agv(agvs, [True, True, False]) == 1
    assert earliest_free_agv(agvs, [False, False, True]) == 2
    assert earliest_free_agv(agvs) == 2
    assert earliest_free_agv(agvs, [False, False, False]) == 0


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


def test_transport_timing_same_edge_charges_deadhead():
    travel = lambda a, b: 2.0 if a != b else 0.0
    src = AgvState(idx=0, edge=0, location=3)

    out = transport_timing(src, pickup=1, dropoff=2, ready=0.0,
                           total=5.0, travel=travel)

    assert out["dead_src"] == 2.0
    assert out["wait"] == 2.0
    assert out["loaded"] == 5.0
    assert out["src_free"] == 7.0
    assert out["src_location"] == 2
    assert out["src_travelled"] == 7.0
    assert out["dst_free"] is None


def test_transport_timing_same_edge_waits_for_free_time():
    travel = lambda a, b: 2.0 if a != b else 0.0
    src = AgvState(idx=0, edge=0, location=3, free_time=5.0)

    out = transport_timing(src, pickup=1, dropoff=2, ready=0.0,
                           total=5.0, travel=travel)

    assert out["wait"] == 7.0      # 5 wait + 2 deadhead
    assert out["src_free"] == 12.0


def test_transport_timing_cross_edge_charges_both_deadheads():
    def travel(a, b):
        if a < 0 or b < 0:
            return 0.0
        return {(3, 1): 1.0, (4, 5): 2.0}.get((int(a), int(b)), 1.0)

    src = AgvState(idx=0, edge=0, location=3)
    dst = AgvState(idx=1, edge=1, location=4)
    out = transport_timing(
        src, pickup=1, dropoff=2, ready=0.0, total=10.0, travel=travel,
        dst=dst, cross_delay=2.0, leg_ratio=0.5, boundary_machine=5)

    assert out["dead_src"] == 1.0
    assert out["dead_dst"] == 2.0
    assert out["t_src"] == 4.0
    assert out["t_dst"] == 4.0
    assert out["src_free"] == 5.0          # 0 + dead 1 + src leg 4
    assert out["src_location"] == -1
    assert out["dst_free"] == 13.0         # boundary 7 + dead 2 + dst leg 4
    assert out["dst_location"] == 2
    assert out["wait"] == 5.0              # 1 + 2 + 0 + 2
    assert out["loaded"] == 8.0


def test_transport_timing_cross_edge_waits_at_boundary():
    def travel(a, b):
        if a < 0 or b < 0:
            return 0.0
        return {(3, 1): 1.0, (4, 5): 2.0}.get((int(a), int(b)), 1.0)

    src = AgvState(idx=0, edge=0, location=3)
    dst = AgvState(idx=1, edge=1, location=4, free_time=10.0)
    out = transport_timing(
        src, pickup=1, dropoff=2, ready=0.0, total=10.0, travel=travel,
        dst=dst, cross_delay=2.0, leg_ratio=0.5, boundary_machine=5)

    assert out["dst_free"] == 16.0         # 10 + dead 2 + dst leg 4
    assert out["wait"] == 8.0              # 1 + 2 + 3 + 2


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


def test_busy_agv_is_selectable():
    agvs = [AgvState(idx=0, edge=0, location=0),
            AgvState(idx=1, edge=0, location=0, free_time=5.0)]

    mask = available_mask(agvs, now=1.0)

    assert mask.tolist() == [True, True]


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


def test_actor_handles_all_masked_rows():
    model = AgvActor_SetTransformer(feat_dim=11, ctx_dim=6)
    agv_feat = torch.randn(2, 4, 11)
    ctx = torch.randn(2, 6)
    mask = torch.ones(2, 4, dtype=torch.bool)  # safety: no all -inf

    prob, value = model(agv_feat, ctx, mask)

    assert prob.shape == (2, 4)
    assert value.shape == (2, 2)
    assert torch.isfinite(prob).all()
    assert np.allclose(prob.sum(dim=-1).detach().numpy(), 1.0, atol=1e-5)


def test_actor_masks_other_edge_agvs():
    model = AgvActor_SetTransformer(feat_dim=11, ctx_dim=6)
    agv_feat = torch.randn(2, 4, 11)
    ctx = torch.randn(2, 6)
    mask = torch.tensor([[True, False, False, False],
                         [True, True, False, False]])

    prob, _ = model(agv_feat, ctx, mask)

    assert prob.shape == (2, 4)
    assert torch.all(prob[0, 1:4] == 0)
    assert torch.all(prob[1, 2:4] == 0)
    assert torch.isfinite(prob).all()
    assert np.allclose(prob.sum(dim=-1).detach().numpy(), 1.0, atol=1e-5)


def _dst_harness():
    """Minimal harness for get_agv_dst_decision_batch (2 workshops)."""
    import networkx as nx
    from types import SimpleNamespace

    from plcsp_v5.trainer.parallel_env import Parallel_env

    env = Parallel_env.__new__(Parallel_env)
    env.batch_size = 1
    env.nmachines = 2
    env.nedges = 2
    env.n_agv = 4
    env.cross_delay = 1.0
    env.cross_leg_ratio = 0.5

    graph = nx.DiGraph()
    graph.add_node(1, machine=0, finish_time=10.0, scheduled=True)
    env.paral_env_DG = [SimpleNamespace(G=graph)]

    trans = np.full((4, 4), 4.0, dtype=float)
    np.fill_diagonal(trans, 0.0)
    edge_info = [[0, 1], [2, 3]]
    env.ability_instance = [[None, None, trans, edge_info]]
    env.agv_states = [[
        AgvState(idx=0, edge=0, location=0, free_time=0.0),
        AgvState(idx=1, edge=0, location=1, free_time=5.0),
        AgvState(idx=2, edge=1, location=2, free_time=0.0),
        AgvState(idx=3, edge=1, location=3, free_time=2.0),
    ]]
    return env


def test_dst_decision_batch_cross_edge():
    env = _dst_harness()

    feat, ctx, mask, valid = env.get_agv_dst_decision_batch(
        np.asarray([1]), np.asarray([3]), np.asarray([0]))

    assert valid.tolist() == [True]
    assert mask.tolist() == [[False, False, True, True]]
    # boundary = max(10, 0) + travel(0, 0) + t_src(1.5) + delay(1.0)
    assert np.isclose(ctx[0, 4], 12.5)
    # destination deadhead: AGV 2 is at the boundary, AGV 3 is one hop away
    assert np.isclose(feat[0, 2, 7], 0.0)
    assert np.isclose(feat[0, 3, 7], 4.0)
    # both destination AGVs are already free at the boundary
    assert np.isclose(feat[0, 2, 9], 0.0)
    assert np.isclose(feat[0, 3, 9], 0.0)


def test_dst_decision_batch_same_edge_is_invalid():
    env = _dst_harness()

    _, _, mask, valid = env.get_agv_dst_decision_batch(
        np.asarray([1]), np.asarray([1]), np.asarray([0]))

    assert valid.tolist() == [False]
    assert mask.tolist() == [[True, True, True, True]]
