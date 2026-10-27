"""Dump the machine schedule produced by a trained top-level checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plcsp_v5"))
sys.path.insert(0, str(ROOT / "plcsp_v5" / "graph-jsp-env" / "src"))

from graph_jsp_env.disjunctive_graph_jsp_env_singlestep import (
    DisjunctiveGraphJspEnv_singleStep)  # noqa: E402
from baselines.diag_rs_actions import run_arm  # noqa: E402
from instance.generate_allsize_mofjsp_dataset import Instance_Dataset  # noqa: E402

HOLDER = {"nodes": [], "n": 0}
_ORIG = DisjunctiveGraphJspEnv_singleStep.step


def _wrapped(self, *args, **kwargs):
    out = _ORIG(self, *args, **kwargs)
    nodes = []
    for i in range(1, 37):
        node = self.G.nodes[i]
        if node.get("scheduled"):
            nodes.append((i, dict(node)))
    if len(nodes) >= HOLDER["n"]:
        HOLDER["nodes"] = nodes
        HOLDER["n"] = len(nodes)
    return out


DisjunctiveGraphJspEnv_singleStep.step = _wrapped

AGV_ROWS = []
try:
    from agv import state as agv_state  # noqa: E402
except ImportError:
    from plcsp_v5.agv import state as agv_state  # noqa: E402

_ORIG_TT = agv_state.transport_timing


def _tt(src, pickup, dropoff, ready, total, travel, **kw):
    out = _ORIG_TT(src, pickup, dropoff, ready, total, travel, **kw)
    AGV_ROWS.append({
        "agv": int(src.idx),
        "start": max(float(ready), float(src.free_time)),
        "end": float(out["src_free"]),
    })
    dst = kw.get("dst")
    if dst is not None:
        AGV_ROWS.append({
            "agv": int(dst.idx),
            "start": float(out["dst_free"]) - float(out["dst_travelled"]),
            "end": float(out["dst_free"]),
        })
    return out


agv_state.transport_timing = _tt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--run-tag", default="plancredit_local_E4b_J6M6E2_s1_remote")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = Instance_Dataset(
        generate_true=0,
        dataset_pth=str(ROOT / "plcsp_v5" / "instance" /
                        "test_Instance_J6M6E2.pkl"))
    run_arm("E4b", args.seed, data, 1, args.run_tag, credit="local")

    rows = []
    for i, node in HOLDER["nodes"]:
        rows.append({
            "task": i - 1,
            "start": float(node["start_time"]),
            "finish": float(node["finish_time"]),
            "machine": int(node["machine"]),
        })
    Path(args.out).write_text(json.dumps({"rows": rows, "agv_rows": AGV_ROWS}))
    print("wrote", args.out, "scheduled", len(rows), "agv legs", len(AGV_ROWS))


if __name__ == "__main__":
    main()
