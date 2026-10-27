"""Tests for Stage-0 failure event models."""
from __future__ import annotations

import numpy as np

from plcsp_v5.constraints import (buffer_mask, failure_probability,
                                  sample_repair)


def test_failure_probability_bounds():
    assert failure_probability(0.0, 100.0) == 0.0
    assert failure_probability(10.0, 0.0) == 0.0
    p = failure_probability(10.0, 100.0)
    assert 0.0 < p < 1.0


def test_sample_repair_disabled_and_zero_active():
    rng = np.random.default_rng(0)
    assert sample_repair(rng, 0.0, 100.0, 10.0) == 0.0
    assert sample_repair(rng, 10.0, 0.0, 10.0) == 0.0
    assert sample_repair(rng, 10.0, 100.0, 0.0) == 0.0


def test_sample_repair_is_seeded():
    a = sample_repair(np.random.default_rng(7), 1e6, 1.0, 5.0)
    b = sample_repair(np.random.default_rng(7), 1e6, 1.0, 5.0)
    assert a == b
    assert a > 0.0


def test_buffer_mask():
    routes = {0: [1, 2], 1: [3]}
    start = {1: 0.0, 2: 10.0, 3: 0.0}
    assert buffer_mask(routes, start, 5.0, 1, 2).tolist() == [True, False]
    assert buffer_mask(routes, start, 5.0, 2, 2).tolist() == [False, False]
    assert buffer_mask(routes, start, 5.0, 0, 2).tolist() == [False, False]
