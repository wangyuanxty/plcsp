"""Tests for the design-A top-level rule policy."""
from __future__ import annotations

import numpy as np
import torch

from plcsp_v5.hierarchy.rules import (N_AGV_RULES, N_JOB_RULES,
                                     N_MACHINE_RULES, agv_rule_bias,
                                     job_rule_bias, machine_rule_bias,
                                     select_agv_by_rule, top_state_features)
from plcsp_v5.hierarchy.top_ppo import TopLevelController


def _task_features(values):
    """Build (N, 12) task features with f0/f1/f2 set per row."""
    feat = np.zeros((len(values), 12), dtype=np.float32)
    for i, (est, eft, pt) in enumerate(values):
        feat[i, 0] = est
        feat[i, 1] = eft
        feat[i, 2] = pt
    return feat


def test_job_rule_prefers_min_est():
    feat = _task_features([(5.0, 8.0, 3.0), (10.0, 12.0, 2.0)])
    candidate = np.array([[0, 1]])
    mask = torch.zeros((1, 2), dtype=torch.bool)
    # job rule 0 = est
    bias = job_rule_bias(feat, candidate, mask, [0], 2, alpha=1.0)
    assert bias.shape == (1, 2)
    assert bias[0, 0] > bias[0, 1]
    assert abs(float(bias.sum())) < 1e-6


def test_job_rule_respects_mask():
    feat = _task_features([(5.0, 8.0, 3.0), (10.0, 12.0, 2.0)])
    candidate = np.array([[0, 1]])
    mask = torch.tensor([[False, True]])
    bias = job_rule_bias(feat, candidate, mask, [0], 2, alpha=1.0)
    assert float(bias[0, 1]) == 0.0
    assert abs(float(bias.sum())) < 1e-6


def test_mwko_prefers_more_remaining_operations():
    # Candidate 2 is the last operation of job 0; candidate 3 is job 1's first.
    feat = _task_features([(0.0, 0.0, 0.0)] * 6)
    candidate = np.array([[2, 3]])
    mask = torch.zeros((1, 2), dtype=torch.bool)
    # job rule 3 = mwko
    bias = job_rule_bias(feat, candidate, mask, [3], 3, alpha=1.0)
    assert bias[0, 1] > bias[0, 0]


def test_machine_rule_prefers_its_feature():
    fea = np.zeros((1, 3, 6), dtype=np.float32)
    fea[0, :, 0] = [9.0, 2.0, 5.0]   # t
    fea[0, :, 1] = [1.0, 8.0, 4.0]   # pt
    fea[0, :, 2] = [7.0, 3.0, 1.0]   # transport
    mask = torch.zeros((1, 1, 3), dtype=torch.bool)

    min_t = machine_rule_bias(fea, mask, [0], alpha=1.0)
    min_pt = machine_rule_bias(fea, mask, [1], alpha=1.0)
    min_trans = machine_rule_bias(fea, mask, [2], alpha=1.0)
    assert int(torch.argmax(min_t)) == 1
    assert int(torch.argmax(min_pt)) == 0
    assert int(torch.argmax(min_trans)) == 2
    for bias in (min_t, min_pt, min_trans):
        assert abs(float(bias.sum())) < 1e-6


def test_top_state_features_shape_and_finite():
    feat = _task_features([(2.0, 5.0, 3.0), (1.0, 4.0, 3.0)])
    candidate = np.array([[0, 1]])
    mask = torch.zeros((1, 2), dtype=torch.bool)
    m2 = np.zeros((1, 2, 8), dtype=np.float32)
    state = top_state_features(feat, candidate, mask, m2, 0, 2, 2, 2)
    assert state.shape == (1, 18)
    assert torch.isfinite(state).all()
    assert torch.equal(state[0, 6:10], torch.zeros(4))
    assert float(state[0, 10]) == 0.0
    assert torch.equal(state[0, 12:15], torch.zeros(3))
    assert torch.allclose(state[0, 15:18], torch.tensor([0.4, 0.4, 0.2]))


def test_top_ppo_update_runs_and_clears_buffer():
    ctl = TopLevelController(
        input_dim=18, head_sizes=(N_JOB_RULES, N_MACHINE_RULES, N_AGV_RULES),
        hidden_dim=16, lr=1e-3, epochs=2, device="cpu")
    before = [p.detach().clone() for p in ctl.model.parameters()]
    for _ in range(3):
        state = torch.randn(4, 18)
        action, logprob, value = ctl.act(state, store=True)
        assert action.shape == (4, 3)
        assert logprob.shape == (4,)
        assert value.shape == (4,)
        ctl.close_segment(torch.randn(4))
    stats = ctl.finish_episode(torch.full((4,), 100.0))
    assert stats is not None
    assert np.isfinite(stats["top_actor_loss"])
    assert np.isfinite(stats["top_critic_loss"])
    assert not ctl.actions
    assert not ctl.rewards
    after = [p.detach().clone() for p in ctl.model.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_top_ppo_two_heads_without_agv():
    ctl = TopLevelController(
        input_dim=18, head_sizes=(N_JOB_RULES, N_MACHINE_RULES),
        hidden_dim=16, device="cpu")
    action, logprob, value = ctl.act(torch.randn(2, 18), store=False)
    assert action.shape == (2, 2)
    assert logprob.shape == (2,)
    assert value.shape == (2,)


def test_top_ppo_greedy_does_not_store():
    ctl = TopLevelController(
        input_dim=18, head_sizes=(N_JOB_RULES, N_MACHINE_RULES),
        hidden_dim=16, device="cpu")
    ctl.act(torch.randn(2, 18), greedy=True, store=False)
    assert not ctl.actions


def test_top_ppo_local_credit_update_runs():
    ctl = TopLevelController(
        input_dim=18, head_sizes=(N_JOB_RULES, N_MACHINE_RULES,
                                  N_AGV_RULES),
        hidden_dim=16, lr=1e-3, epochs=2, credit="local", device="cpu")
    before = [p.detach().clone() for p in ctl.model.parameters()]
    for _ in range(3):
        state = torch.randn(4, 18)
        action, logprob, value = ctl.act(state, store=True)
        assert action.shape == (4, 3)
        assert logprob.shape == (4,)
        assert value.shape == (4,)
        ctl.close_segment(torch.rand(4, 3))
    stats = ctl.finish_episode(
        torch.full((4,), 100.0), final_local=torch.full((4, 3), 100.0))
    assert stats is not None
    assert np.isfinite(stats["top_actor_loss"])
    assert np.isfinite(stats["top_critic_loss"])
    assert not ctl.actions
    assert not ctl.rewards
    after = [p.detach().clone() for p in ctl.model.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_top_ppo_local_credit_requires_per_head_reward():
    ctl = TopLevelController(
        input_dim=18, head_sizes=(N_JOB_RULES, N_MACHINE_RULES,
                                  N_AGV_RULES),
        hidden_dim=16, credit="local", device="cpu")
    ctl.act(torch.randn(4, 18), store=True)
    try:
        ctl.close_segment(torch.randn(4))
    except ValueError:
        pass
    else:
        raise AssertionError("local credit accepted a scalar reward")


def test_top_ppo_coma_update_runs():
    ctl = TopLevelController(
        input_dim=18, head_sizes=(N_JOB_RULES, N_MACHINE_RULES,
                                  N_AGV_RULES),
        hidden_dim=16, lr=1e-3, epochs=2, credit="coma", device="cpu")
    before = [p.detach().clone() for p in ctl.model.parameters()]
    for _ in range(3):
        state = torch.randn(4, 18)
        action, logprob, value = ctl.act(state, store=True)
        assert action.shape == (4, 3)
        assert logprob.shape == (4,)
        assert value.shape == (4,)
        ctl.close_segment(torch.randn(4))
    stats = ctl.finish_episode(torch.full((4,), 100.0))
    assert stats is not None
    assert np.isfinite(stats["top_actor_loss"])
    assert np.isfinite(stats["top_q_loss"])
    assert not ctl.actions
    assert not ctl.rewards
    after = [p.detach().clone() for p in ctl.model.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_top_ppo_rejects_unknown_credit():
    try:
        TopLevelController(
            input_dim=18, head_sizes=(N_JOB_RULES, N_MACHINE_RULES),
            hidden_dim=16, credit="unknown", device="cpu")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown credit mode was accepted")


def test_top_ppo_coma_q_replay_caps():
    ctl = TopLevelController(
        input_dim=18, head_sizes=(N_JOB_RULES, N_MACHINE_RULES,
                                  N_AGV_RULES),
        hidden_dim=16, lr=1e-3, epochs=1, credit="coma", device="cpu")
    ctl.q_replay_max = 2
    for _ in range(4):
        ctl.act(torch.randn(2, 18), store=True)
        ctl.close_segment(torch.randn(2))
        stats = ctl.finish_episode(torch.full((2,), 100.0))
        assert np.isfinite(stats["top_q_loss"])
    assert len(ctl.q_replay) == 2
    assert all(torch.isfinite(item[0]).all() for item in ctl.q_replay)
    assert all(torch.isfinite(item[1]).all() for item in ctl.q_replay)


def test_top_ppo_close_segment_rejects_wrong_batch():
    ctl = TopLevelController(
        input_dim=18, head_sizes=(N_JOB_RULES, N_MACHINE_RULES),
        hidden_dim=16, device="cpu")
    ctl.act(torch.randn(4, 18), store=True)
    try:
        ctl.close_segment(torch.tensor(1.0))
    except ValueError:
        pass
    else:
        raise AssertionError("scalar reward was accepted for B=4")


def test_top_ppo_final_cost_rejects_wrong_batch():
    ctl = TopLevelController(
        input_dim=18, head_sizes=(N_JOB_RULES, N_MACHINE_RULES),
        hidden_dim=16, device="cpu")
    for _ in range(2):
        ctl.act(torch.randn(4, 18), store=True)
        ctl.close_segment(torch.randn(4))
    try:
        ctl.finish_episode(torch.full((1,), 100.0))
    except ValueError:
        pass
    else:
        raise AssertionError("final_cost with one entry was accepted for B=4")


def test_top_ppo_single_sample_update_is_finite():
    ctl = TopLevelController(
        input_dim=18, head_sizes=(N_JOB_RULES, N_MACHINE_RULES),
        hidden_dim=16, epochs=2, device="cpu")
    ctl.act(torch.randn(1, 18), store=True)
    ctl.close_segment(torch.tensor([1.0]))
    stats = ctl.finish_episode(torch.tensor([100.0]))
    assert stats is not None
    assert np.isfinite(stats["top_actor_loss"])
    assert np.isfinite(stats["top_critic_loss"])


def test_top_ppo_local_single_head_vector():
    ctl = TopLevelController(
        input_dim=18, head_sizes=(N_JOB_RULES,),
        hidden_dim=16, epochs=2, credit="local", device="cpu")
    ctl.act(torch.randn(3, 18), store=True)
    ctl.close_segment(torch.randn(3))
    stats = ctl.finish_episode(
        torch.full((3,), 100.0), final_local=torch.full((3,), 100.0))
    assert stats is not None
    assert np.isfinite(stats["top_actor_loss"])


def test_top_ppo_coma_requires_final_cost():
    ctl = TopLevelController(
        input_dim=18, head_sizes=(N_JOB_RULES, N_MACHINE_RULES),
        hidden_dim=16, epochs=1, credit="coma", device="cpu")
    ctl.act(torch.randn(2, 18), store=True)
    ctl.close_segment(torch.randn(2))
    try:
        ctl.finish_episode()
    except ValueError:
        pass
    else:
        raise AssertionError("coma accepted an unnormalised final cost")


def test_top_ppo_hybrid_update_runs():
    ctl = TopLevelController(
        input_dim=18, head_sizes=(N_JOB_RULES, N_MACHINE_RULES,
                                  N_AGV_RULES),
        hidden_dim=16, lr=1e-3, epochs=2, credit="hybrid", device="cpu")
    before = [p.detach().clone() for p in ctl.model.parameters()]
    for _ in range(3):
        state = torch.randn(4, 18)
        action, logprob, value = ctl.act(state, store=True)
        assert action.shape == (4, 3)
        assert logprob.shape == (4,)
        assert value.shape == (4,)
        ctl.close_segment(torch.rand(4, 3))
    stats = ctl.finish_episode(
        torch.full((4,), 100.0), final_local=torch.full((4, 3), 100.0))
    assert stats is not None
    assert np.isfinite(stats["top_actor_loss"])
    assert np.isfinite(stats["top_q_loss"])
    assert len(ctl.q_replay) == 1
    after = [p.detach().clone() for p in ctl.model.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_top_ppo_hybrid_requires_final_local():
    ctl = TopLevelController(
        input_dim=18, head_sizes=(N_JOB_RULES, N_MACHINE_RULES,
                                  N_AGV_RULES),
        hidden_dim=16, epochs=1, credit="hybrid", device="cpu")
    ctl.act(torch.randn(2, 18), store=True)
    ctl.close_segment(torch.rand(2, 3))
    try:
        ctl.finish_episode(torch.full((2,), 100.0))
    except ValueError:
        pass
    else:
        raise AssertionError("hybrid accepted a missing final_local")


def _agv_features():
    feat = np.zeros((1, 3, 11), dtype=np.float32)
    feat[0, :, 10] = [5.0, 1.0, 3.0]   # free time
    feat[0, :, 7] = [1.0, 9.0, 2.0]    # deadhead
    return feat


def test_agv_rule_bias_earliest_free():
    feat = _agv_features()
    mask = np.ones((1, 3), dtype=bool)
    bias = agv_rule_bias(feat, mask, [0], alpha=1.0)
    assert int(torch.argmax(bias)) == 1
    assert abs(float(bias.sum())) < 1e-6


def test_agv_rule_bias_nearest():
    feat = _agv_features()
    mask = np.ones((1, 3), dtype=bool)
    bias = agv_rule_bias(feat, mask, [1], alpha=1.0)
    assert int(torch.argmax(bias)) == 0
    assert abs(float(bias.sum())) < 1e-6


def test_agv_rule_bias_random_is_zero():
    feat = _agv_features()
    mask = np.ones((1, 3), dtype=bool)
    bias = agv_rule_bias(feat, mask, [2], alpha=1.0)
    assert torch.equal(bias, torch.zeros_like(bias))


def test_select_agv_by_rule():
    feat = _agv_features()
    mask = np.ones((1, 3), dtype=bool)
    assert int(select_agv_by_rule(feat, mask, [0])[0]) == 1
    assert int(select_agv_by_rule(feat, mask, [1])[0]) == 0
    rand = int(select_agv_by_rule(feat, mask, [2])[0])
    assert 0 <= rand < 3
