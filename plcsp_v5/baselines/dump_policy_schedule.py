# -*- coding: utf-8 -*-
"""Dump the schedule the trained controller builds on one test instance.

The evaluation loop does not keep the decisions, so this script hooks the
environment step and the transport timing, records the operation, the machine
and the AGV of every low-level step, and writes them to JSON.  The record is
enough to replay the same schedule through ``agv_sim.simulate``.

Usage::

    python -m baselines.dump_policy_schedule --run-tag m3_agv6_ours_s1 \
        --instances 0 1 2 --out rows.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "graph-jsp-env", "src"))

# ``parameters`` parses sys.argv when it is imported, so the script parses its
# own flags first and hides them from that parser.
_OWN = argparse.ArgumentParser()
_OWN.add_argument("--run-tag", required=True)
_OWN.add_argument("--instances", type=int, nargs="+", required=True)
_OWN.add_argument("--out", required=True)
_OWN.add_argument("--seed", type=int, default=1)
OWN = _OWN.parse_args()
sys.argv = sys.argv[:1]

# The vendored environment calls matplotlib.cm.get_cmap, which matplotlib 3.9
# removed.  Shim it for this process only.
import matplotlib  # noqa: E402
import matplotlib.cm  # noqa: E402
if not hasattr(matplotlib.cm, "get_cmap"):
    matplotlib.cm.get_cmap = matplotlib.colormaps.get_cmap

from parameters import args  # noqa: E402

RECORD = []
TRANS = {}
TRANSPORTS = []
CURRENT = {"step": None}


def _install_hooks() -> None:
    from graph_jsp_env.disjunctive_graph_jsp_env_singlestep import (
        DisjunctiveGraphJspEnv_singleStep as Env)

    orig_step = Env.step

    def step(self, joint_action=None, **kw):
        CURRENT["step"] = {"task": int(joint_action[0]),
                           "machine": int(joint_action[1]),
                           "agv": -1, "dst": -1}
        out = orig_step(self, joint_action=joint_action, **kw)
        RECORD.append(CURRENT["step"])
        CURRENT["step"] = None
        return out

    Env.step = step

    # ``validate.py`` imports the timing helpers as ``agv.state``, which is a
    # different module object from ``plcsp_v5.agv.state`` for the same file.
    # Patch both so whichever alias the caller bound is hooked.
    import plcsp_v5.agv.state as state_a
    try:
        import agv.state as state_b
    except ImportError:
        state_b = None
    for mod in [m for m in (state_a, state_b) if m is not None]:
        for name in ("transport_timing", "transport_timing_m2"):
            orig = getattr(mod, name, None)
            if orig is None or getattr(orig, "_hooked", False):
                continue

            def make(orig=orig):
                def call(src, *a, **kw):
                    out = orig(src, *a, **kw)
                    dst = kw.get("dst")
                    TRANSPORTS.append((int(src.idx),
                                       int(dst.idx) if dst is not None else -1))
                    return out
                call._hooked = True
                return call

            setattr(mod, name, make())


def main() -> int:
    args.encoder_type = "settrans"
    args.eval_seed = 3
    args.use_agv = 1
    args.hier = 1
    args.lu_station = 1
    args.agv_dst_actor = 1
    args.machine_intra_attn = 1
    args.run_tag = OWN.run_tag
    args.seed = OWN.seed
    args.train_seed = OWN.seed

    random.seed(args.test_seed)
    np.random.seed(args.test_seed)
    torch.manual_seed(args.test_seed)

    from algorithm.ppo_algorithm import PPOAlgorithm
    from instance.generate_allsize_mofjsp_dataset import Instance_Dataset
    from trainer.train_device import device
    from trainer.validate import validate_cost_gcn_jointActor_GAT
    from hierarchy.rules import N_AGV_RULES, N_JOB_RULES, N_MACHINE_RULES
    from hierarchy.top_ppo import TopLevelController

    _install_hooks()

    variant = vars(args)
    ppo = PPOAlgorithm(args=variant, load_pretrained=False)
    id_name = f"J{args.n_job}M{args.n_machine}E{args.n_edge}"
    model_dir = os.path.join(REPO, "trained_model", OWN.run_tag)
    modules = [("job_actor", ppo.job_actor),
               ("machine_actor", ppo.machine_actor_gcn),
               ("agv_actor", ppo.agv_actor)]
    for name, module in modules:
        path = os.path.join(model_dir, f"PPO_{name}_{id_name}_top1.pth")
        module.load_state_dict(torch.load(path, map_location=device))

    top_ctl = TopLevelController(
        input_dim=18, head_sizes=(N_JOB_RULES, N_MACHINE_RULES, N_AGV_RULES),
        credit=getattr(args, "top_credit", "team"), device=device)
    top_ctl.load_state_dict(torch.load(
        os.path.join(model_dir, f"PPO_hier_top_{id_name}_top1.pth"),
        map_location=device))

    data = Instance_Dataset(
        generate_true=0,
        dataset_pth=os.path.join(REPO, "instance",
                                 f"test_Instance_{id_name}_lu.pkl"))

    rows = []
    for i in OWN.instances:
        RECORD.clear()
        TRANS.clear()
        TRANSPORTS.clear()
        _, _, obj = validate_cost_gcn_jointActor_GAT(
            ppo=ppo, gantt_flag=False, data=data, data_index=i,
            data_type="random", greedy=True, args=variant, top_ctl=top_ctl)
        rows.append({"instance": i, "objective": float(obj),
                     "steps": [dict(s) for s in RECORD],
                     "transports": [list(t) for t in TRANSPORTS]})
        print(f"instance {i}: obj {obj:.3f}  steps {len(RECORD)}", flush=True)

    with open(OWN.out, "w", encoding="utf-8") as fh:
        json.dump({"run_tag": OWN.run_tag, "rows": rows}, fh)
    print("wrote", OWN.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())