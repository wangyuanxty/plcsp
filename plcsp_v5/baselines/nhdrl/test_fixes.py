# -*- coding: utf-8 -*-
"""Tests for the three implementation fixes and the idle-mode alignment.

Run from the repository root::

    D:/anaconda/python.exe -m pytest plcsp_v5/baselines/nhdrl/test_fixes.py -q
"""
from __future__ import annotations

import argparse
import math
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
BASELINES = ROOT / "plcsp_v5" / "baselines"
for _path in (str(BASELINES), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from nhdrl.cli import _reward  # noqa: E402
from nhdrl.ppo import (  # noqa: E402
    Agent,
    PPOConfig,
    RunningMinMax,
    RunningStandardizer,
    normalise_branch_reward,
)
from nhdrl.sim import EARLIEST, NestedSim, StepOutcome  # noqa: E402

TEST_PKL = ROOT / "plcsp_v5" / "instance" / "test_Instance_J6M6E2.pkl"


def _agent(n_actions: int = 3) -> Agent:
    agent = Agent("t", 4, n_actions, PPOConfig(), torch.device("cpu"))
    # Non-uniform logits so the actor probability differs from 1 / |A|.
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.zero_()
        agent.actor.net[-1].bias.copy_(torch.arange(n_actions,
                                                    dtype=torch.float32) * 0.7)
    return agent


def _reward_args() -> argparse.Namespace:
    return argparse.Namespace(w1=0.8, w2=0.2, pref_ms=5.0, pref_tc=5.0,
                              maxp=10.0)


def _outcome(ms_gain: float, cost_delta: float) -> StepOutcome:
    return StepOutcome(task=0, machine=0, job=0, ms_before=ms_gain,
                       ms_after=0.0, cost_before=0.0,
                       cost_after=cost_delta, transport=False)


def test_behaviour_logprob_is_the_epsilon_mixture() -> None:
    agent = _agent()
    state = np.zeros(4, dtype=np.float32)
    action, logprob, _value = agent.act(
        state, epsilon=1.0, greedy=False, rng=random.Random(0),
        behaviour_logprob=True)
    assert logprob == pytest.approx(math.log(1.0 / agent.n_actions))

    action, logprob, _value = agent.act(
        state, epsilon=0.0, greedy=False, rng=random.Random(0),
        behaviour_logprob=True)
    with torch.no_grad():
        logits = agent.actor(torch.as_tensor(state).unsqueeze(0))
        expected = float(torch.log_softmax(logits, dim=-1)[0, action])
    assert logprob == pytest.approx(expected, abs=1e-6)


def test_old_logprob_path_is_unchanged() -> None:
    agent = _agent()
    state = np.zeros(4, dtype=np.float32)
    action, logprob, _value = agent.act(
        state, epsilon=1.0, greedy=False, rng=random.Random(0),
        behaviour_logprob=False)
    with torch.no_grad():
        logits = agent.actor(torch.as_tensor(state).unsqueeze(0))
        expected = float(torch.log_softmax(logits, dim=-1)[0, action])
    assert logprob == pytest.approx(expected, abs=1e-6)


def test_running_minmax_window_forgets_old_extremes() -> None:
    stats = RunningMinMax(-1.0, 0.0, window=3)
    stats.update(100.0)
    for value in (1.0, 2.0, 3.0):
        stats.update(value)
    assert (stats.min_value, stats.max_value) == (1.0, 3.0)

    old = RunningMinMax(-1.0, 0.0, window=0)
    old.update(100.0)
    for value in (1.0, 2.0, 3.0):
        old.update(value)
    assert old.max_value == 100.0


def test_common_scale_maps_equal_deviations_to_equal_rewards() -> None:
    scale = RunningStandardizer()
    for value in (1.0, 2.0, 3.0, 4.0, 5.0):
        scale.update(value)
    mean = 3.0
    std = float(np.std([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert scale.z(mean) == pytest.approx(0.0, abs=1e-9)
    assert normalise_branch_reward(scale, mean) == pytest.approx(0.5)
    assert normalise_branch_reward(scale, mean + 3.0 * std) == pytest.approx(0.0)
    assert normalise_branch_reward(scale, mean - 3.0 * std) == pytest.approx(1.0)


def test_common_scale_equalises_the_two_branch_levels() -> None:
    ms_scale = RunningStandardizer(window=50)
    tc_scale = RunningStandardizer(window=50)
    ms_rewards: list[float] = []
    tc_rewards: list[float] = []
    rng = np.random.default_rng(0)
    for _ in range(200):
        ms_badness = float(abs(rng.normal(20.0, 5.0)))
        tc_badness = float(abs(rng.normal(20.0, 5.0)))
        ms_scale.update(ms_badness)
        tc_scale.update(tc_badness)
        ms_rewards.append(normalise_branch_reward(ms_scale, ms_badness))
        tc_rewards.append(normalise_branch_reward(tc_scale, tc_badness))
    assert abs(np.mean(ms_rewards) - np.mean(tc_rewards)) < 0.05
    assert abs(np.std(ms_rewards) - np.std(tc_rewards)) < 0.05


def test_common_scale_keeps_the_reward_ordering() -> None:
    scale = RunningStandardizer()
    for value in (1.0, 2.0, 3.0, 4.0, 5.0):
        scale.update(value)
    good = normalise_branch_reward(scale, 1.0)
    neutral = normalise_branch_reward(scale, 3.0)
    bad = normalise_branch_reward(scale, 5.0)
    assert good > neutral > bad

    args = _reward_args()
    outcome = _outcome(ms_gain=-10.0, cost_delta=10.0)
    scales = (RunningStandardizer(), RunningStandardizer())
    ms_reward = _reward(0, outcome, RunningMinMax(-1.0, 0.0),
                        RunningMinMax(0.0, 1.0), args, scales)
    tc_reward = _reward(1, outcome, RunningMinMax(-1.0, 0.0),
                        RunningMinMax(0.0, 1.0), args, scales)
    # The first sample sets both means to the observed badness, so both
    # branches get the neutral reward and neither dominates.
    assert ms_reward == pytest.approx(0.45)
    assert tc_reward == pytest.approx(0.45)


def _instance() -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    with open(TEST_PKL, "rb") as handle:
        raw = pickle.load(handle)
    return (np.asarray(raw[0][0]), np.asarray(raw[1][0]),
            np.asarray(raw[2][0]), raw[3][0])


def _run(idle_mode: str) -> NestedSim:
    t, p, tt, edge = _instance()
    sim = NestedSim(t, p, tt, edge, n_agv=6, idle_mode=idle_mode)
    while not sim.done:
        _job, task, machine = sim.choose_pa(0)
        sim.schedule(task, machine, EARLIEST)
    return sim


def test_idle_mode_defaults_to_trailing_and_matches_the_definition() -> None:
    sim = _run("trailing")
    result = sim.result()
    expected = sum(result["mk"] - proc for proc in sim.machine_proc)
    assert sim.idle_mode == "trailing"
    assert result["idleT"] == pytest.approx(expected, abs=1e-9)
    assert result["idleT"] != pytest.approx(sim.gap_idle(), abs=1e-6)


def test_gap_idle_mode_keeps_the_old_reading() -> None:
    sim = _run("gap")
    assert sim.result()["idleT"] == pytest.approx(sim.gap_idle(), abs=1e-9)
