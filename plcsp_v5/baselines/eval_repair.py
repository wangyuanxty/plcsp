# -*- coding: utf-8 -*-
"""Controller evaluated when one machine needs a repair mid-episode.

At step ``--after`` the processing times of every remaining operation on the
chosen machine are raised by ``--repair``.  The machine stays selectable, the
state features read the same matrix, so the policy sees it as expensive and can
pick another machine; if it still picks it, that operation pays the repair.
No environment change, no retraining.
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

import matplotlib  # noqa: E402
import matplotlib.cm  # noqa: E402
if not hasattr(matplotlib.cm, "get_cmap"):
    matplotlib.cm.get_cmap = matplotlib.colormaps.get_cmap

_OWN = argparse.ArgumentParser()
_OWN.add_argument("--run-tag", required=True)
_OWN.add_argument("--instances", type=int, nargs="+", required=True)
_OWN.add_argument("--machine", type=int, default=-1)
_OWN.add_argument("--after", type=int, default=14)
_OWN.add_argument("--repair", type=float, default=100.0)
_OWN.add_argument("--out", required=True)
OWN = _OWN.parse_args()
sys.argv = sys.argv[:1]

from parameters import args  # noqa: E402

STEPS = {"n": 0, "machines": [], "on": False, "agv_done": False}


def _install_hooks() -> None:
    from graph_jsp_env.disjunctive_graph_jsp_env_singlestep import (
        DisjunctiveGraphJspEnv_singleStep as Env)

    orig_step = Env.step

    def step(self, joint_action=None, **kw):
        STEPS["machines"].append(int(joint_action[1]))
        out = orig_step(self, joint_action=joint_action, **kw)
        STEPS["n"] += 1
        if STEPS["on"] and STEPS["n"] == OWN.after:
            m = OWN.machine
            if m < 0:  # first machine the policy actually used
                m = STEPS["machines"][0]
            arr = self.jsp_instance[0]
            arr[:, m] += OWN.repair
        return out

    Env.step = step

    # AGV repair: the vehicle that just finished a trip has its free time
    # pushed by the repair, so the actor sees it as unavailable for a while.
    import plcsp_v5.agv.state as state_a
    try:
        import agv.state as state_b
    except ImportError:
        state_b = None
    for mod in [m for m in (state_a, state_b) if m is not None]:
        for name in ("transport_timing", "transport_timing_m2"):
            orig = getattr(mod, name, None)
            if orig is None or getattr(orig, "_rep_hooked", False):
                continue

            def make(orig=orig):
                def call(src, *a, **kw):
                    out = orig(src, *a, **kw)
                    if (STEPS["on"] and not STEPS["agv_done"]
                            and STEPS["n"] >= OWN.after - 1):
                        STEPS["agv_done"] = True
                        print(f"  AGV {src.idx} repaired at step {STEPS['n']}",
                              flush=True)
                        src.free_time += OWN.repair
                        out = dict(out)
                        out["src_free"] = float(src.free_time)
                    return out
                call._rep_hooked = True
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
    for name, module in (("job_actor", ppo.job_actor),
                         ("machine_actor", ppo.machine_actor_gcn),
                         ("agv_actor", ppo.agv_actor)):
        module.load_state_dict(torch.load(
            os.path.join(model_dir, f"PPO_{name}_{id_name}_top1.pth"),
            map_location=device))
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
        STEPS["n"], STEPS["machines"] = 0, []
        _, _, clean = validate_cost_gcn_jointActor_GAT(
            ppo=ppo, gantt_flag=False, data=data, data_index=i,
            data_type="random", greedy=True, args=variant, top_ctl=top_ctl)
        STEPS["n"], STEPS["machines"], STEPS["on"] = 0, [], True
        STEPS["agv_done"] = False
        _, _, hit = validate_cost_gcn_jointActor_GAT(
            ppo=ppo, gantt_flag=False, data=data, data_index=i,
            data_type="random", greedy=True, args=variant, top_ctl=top_ctl)
        STEPS["on"] = False
        rows.append({"instance": i, "clean": float(clean), "hit": float(hit)})
        print(f"instance {i}: {clean:.2f} -> {hit:.2f}", flush=True)

    with open(OWN.out, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh)
    a = np.array([[r["clean"], r["hit"]] for r in rows])
    print("mean %.2f -> %.2f  (%+.1f%%)" %
          (a[:, 0].mean(), a[:, 1].mean(),
           100 * (a[:, 1].mean() - a[:, 0].mean()) / a[:, 0].mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
