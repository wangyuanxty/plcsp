# -*- coding: utf-8 -*-
"""Tests for the adapted NHDRL simulator and rule layer.

Run from the repository root::

    D:/anaconda/python.exe -m pytest plcsp_v5/baselines/nhdrl/test_sim.py -q

The central check replays every PA rule with the earliest-free AGV rule and
compares the stepwise simulator with the existing v5 decoder.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
BASELINES = ROOT / "plcsp_v5" / "baselines"
for _path in (str(BASELINES), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agv_baselines import decode  # noqa: E402
from nhdrl.sim import (  # noqa: E402
    AGV_RULES,
    EARLIEST,
    N_LA,
    N_OA,
    N_PA,
    NestedSim,
)

TEST_PKL = ROOT / "plcsp_v5" / "instance" / "test_Instance_J6M6E2.pkl"


def _instance() -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    with open(TEST_PKL, "rb") as handle:
        raw = pickle.load(handle)
    return (np.asarray(raw[0][0]), np.asarray(raw[1][0]),
            np.asarray(raw[2][0]), raw[3][0])


def _run(pa_action: int, agv_rule: str) -> NestedSim:
    t, p, tt, edge = _instance()
    sim = NestedSim(t, p, tt, edge, n_agv=6)
    while not sim.done:
        _job, task, machine = sim.choose_pa(pa_action)
        sim.schedule(task, machine, agv_rule)
    return sim


def test_feature_and_action_dimensions() -> None:
    t, p, tt, edge = _instance()
    sim = NestedSim(t, p, tt, edge, n_agv=6)
    assert sim.oa_features().shape == (N_OA,)
    assert sim.pa_features(0).shape == (N_PA,)
    assert sim.la_features(0, 0, 0).shape == (N_LA,)
    assert (N_OA, N_PA, N_LA) == (31, 24, 30)


@pytest.mark.parametrize("pa_action", range(9))
def test_stepwise_sim_matches_decode(pa_action: int) -> None:
    t, p, tt, edge = _instance()
    sim = _run(pa_action, EARLIEST)
    dec = decode(t, p, tt, edge, sim.os_seq, np.asarray(sim.ms, dtype=int),
                 agv=True, n_agv=6)
    result = sim.result()
    for key in ("mk", "pt", "transT", "idleT", "obj"):
        assert abs(float(result[key]) - float(dec[key])) < 1e-6, key
    np.testing.assert_allclose(sim.start, np.asarray(dec["start"]),
                               atol=1e-6)
    np.testing.assert_allclose(sim.finish, np.asarray(dec["finish"]),
                               atol=1e-6)


def test_agv_rules_stay_in_the_requested_workshop() -> None:
    t, p, tt, edge = _instance()
    sim = NestedSim(t, p, tt, edge, n_agv=6)
    for edge_id in range(sim.n_edge):
        for rule in AGV_RULES:
            agv = sim.pick_agv(rule, edge_id, 0, 1, 0.0, 1.0)
            assert agv.edge == edge_id


def test_objective_uses_the_v5_weights() -> None:
    sim = _run(0, EARLIEST)
    result = sim.result()
    expected = (0.4 * result["mk"]
                + 0.4 * (result["pt"] / sim.n_task + result["idleT"])
                + 0.2 * result["transT"])
    assert result["obj"] == pytest.approx(expected, abs=1e-9)
    assert result["mk"] > 0.0
    assert result["obj"] > 0.0
