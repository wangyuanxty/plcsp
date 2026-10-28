"""Tests for the M2 (AGV-aware) MIP baseline.

The MIP needs gurobipy, which the gate interpreter does not ship, so the
whole module skips when the import fails.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gurobipy")

from plcsp_v5.baselines.agv_baselines import pdr_schedule
from plcsp_v5.baselines.mip_e2e import model_m2
from plcsp_v5.baselines.mip_e2e.verify_m2 import (accept_check, verify_m2,
                                                  verify_reference)

POOL = (Path(__file__).resolve().parents[1] / "instance"
        / "test_Instance_J6M6E2_lu.pkl")
TOL = 1e-6


@pytest.fixture(scope="module")
def inst() -> model_m2.InstanceM2:
    return model_m2.load_instance_m2(str(POOL), 0)


@pytest.fixture(scope="module")
def assignment(inst: model_m2.InstanceM2) -> np.ndarray:
    _, ms = pdr_schedule(inst.t, inst.power, "mor", "spt")
    return ms


def test_legs_cover_every_job(inst: model_m2.InstanceM2,
                              assignment: np.ndarray) -> None:
    """Every job has one entry leg; the rest follow the machine changes."""
    legs = model_m2.build_legs(inst, assignment)
    entry_tasks = {leg.task for leg in legs if leg.kind == "entry"}
    assert entry_tasks == {i * inst.n_op for i in range(inst.n_job)}
    for leg in legs:
        assert leg.task % inst.n_op > 0 or leg.kind == "entry"
        assert leg.kind in ("entry", "single", "src", "dst")
        if leg.kind == "dst":
            src = legs[leg.src_leg]
            assert src.kind == "src" and src.task == leg.task
            assert leg.edge != src.edge


def test_model_fits_the_size_limited_licence(
        inst: model_m2.InstanceM2) -> None:
    """Every 6x6 model must stay inside 2000 variables and 2000 rows."""
    for idx in range(4):
        sample = model_m2.load_instance_m2(str(POOL), idx)
        machines, _, _ = model_m2.choose_assignment(sample, rga_seeds=0)
        n_vars, n_constrs = model_m2.model_size_m2(sample, machines)
        assert n_vars <= model_m2.LIMIT_VARS
        assert n_constrs <= model_m2.LIMIT_CONSTR


def test_accepts_the_decoder_schedule(inst: model_m2.InstanceM2,
                                      assignment: np.ndarray) -> None:
    """The model must accept decode_m2's schedule at the same objective."""
    os_seq, _ = pdr_schedule(inst.t, inst.power, "mor", "spt")
    res = accept_check(inst, assignment, os_seq)
    assert res.feasible
    assert res.variance <= TOL


def test_solution_is_feasible_and_matches_the_reference(
        inst: model_m2.InstanceM2, assignment: np.ndarray) -> None:
    """The MIP solution survives both replays, at the same objective."""
    sol = model_m2.solve_m2(inst, assignment, time_limit=30.0)
    assert sol.ok and not sol.message
    legs = model_m2.build_legs(inst, assignment)
    per_edge: dict = {}
    for chain in sol.chains:
        edge = legs[chain[0]].edge
        per_edge[edge] = per_edge.get(edge, 0) + 1
        assert all(legs[i].edge == edge for i in chain)
    fleet = model_m2.agv_per_workshop(inst.n_edge, 6)
    for edge, count in per_edge.items():
        assert count <= fleet[edge]
    local = verify_m2(inst, assignment, sol.starts, sol.chains)
    ref = verify_reference(inst, assignment, sol.starts, sol.chains)
    assert local["feasible"] and ref["feasible"]
    assert abs(local["obj"] - sol.objective) <= TOL
    assert abs(ref["obj"] - sol.objective) <= TOL
