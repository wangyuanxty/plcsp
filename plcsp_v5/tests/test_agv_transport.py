"""Tests for the AGV transport accounting split.

AGV wait is a scheduling delay (counted once, in the idle/SEC term).
AGV travel is the transport time (TEC and graph edge weights).
"""
from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np

from plcsp_v5.trainer.DGenv_func import (find_max_arrivaTime_for_currentNode,
                                         find_transportT)

TRAVEL = np.asarray([
    [0, 3, 5],
    [7, 0, 11],
    [13, 17, 0],
], dtype=float)


def _graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for node in range(1, 7):
        g.add_node(node, machine=-2, duration=1.0, scheduled=False)
    g.nodes[1]["machine"] = 0
    g.nodes[2]["machine"] = 1
    g.nodes[3]["machine"] = 2
    g.nodes[1]["finish_time"] = 10.0
    g.nodes[1]["scheduled"] = True
    g.add_edge(1, 2)
    return g


def _configs() -> SimpleNamespace:
    return SimpleNamespace(n_job=2, n_machine=3)


def test_transport_override_is_pair_scoped_travel():
    g = _graph()
    g.graph["transport_override"] = {(1, 2): 3.0}

    assert find_transportT(g, 1, 2, TRAVEL, _configs()) == 3.0
    # A different job edge keeps the raw transport time.
    assert find_transportT(g, 2, 3, TRAVEL, _configs()) == 11


def test_wait_is_added_to_start_time_only():
    g = _graph()
    g.graph["transport_override"] = {(1, 2): 3.0}
    g.graph["transport_wait_override"] = {(1, 2): 4.0}

    assert find_max_arrivaTime_for_currentNode(g, 2, TRAVEL, _configs()) == 17.0
    # TEC and edge weights must not see the wait.
    assert find_transportT(g, 1, 2, TRAVEL, _configs()) == 3.0


def test_scheduled_node_keeps_its_paid_wait():
    g = _graph()
    g.nodes[2]["scheduled"] = True
    g.nodes[2]["start_time"] = 17.0
    g.graph["transport_wait_override"] = {(1, 2): 4.0}

    assert find_max_arrivaTime_for_currentNode(g, 2, TRAVEL, _configs()) == 17.0


def test_no_override_keeps_upstream_behaviour():
    g = _graph()

    assert find_transportT(g, 1, 2, TRAVEL, _configs()) == 3
    assert find_max_arrivaTime_for_currentNode(g, 2, TRAVEL, _configs()) == 13.0
