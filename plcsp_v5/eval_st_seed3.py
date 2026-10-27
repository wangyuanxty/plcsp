# -*- coding: utf-8 -*-
"""Evaluate the Set Transformer checkpoint on the official seed3 test set.

Loads ``PPO_*_J6M6E2_top1.pth`` (selected on seed1 validation), then runs the
100 unseen test instances with greedy actions. No training, no model selection.
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from parameters import args  # noqa: E402

args.encoder_type = os.environ.get("EVAL_ENCODER", "settrans")
args.eval_seed = 3
# Keep --use_agv from the command line (default 0 = no-AGV baseline).
# EVAL_GREEDY=0 evaluates the sampled policy instead of the greedy policy.
_EVAL_GREEDY = os.environ.get("EVAL_GREEDY", "1") != "0"
_EVAL_SUFFIX = os.environ.get("EVAL_OUTPUT_SUFFIX", "")
random.seed(args.test_seed)
np.random.seed(args.test_seed)
torch.manual_seed(args.test_seed)

from algorithm.ppo_algorithm import PPOAlgorithm  # noqa: E402
from instance.generate_allsize_mofjsp_dataset import Instance_Dataset  # noqa: E402
from trainer.train_device import device  # noqa: E402
from trainer.validate import validate_cost_gcn_jointActor_GAT  # noqa: E402


def main() -> int:
    variant = vars(args)
    ppo = PPOAlgorithm(args=variant, load_pretrained=False)
    id_name = f"J{args.n_job}M{args.n_machine}E{args.n_edge}"

    model_dir = os.path.join(REPO, "trained_model")
    if getattr(args, "run_tag", ""):
        model_dir = os.path.join(model_dir, args.run_tag)
    modules = [("job_actor", ppo.job_actor),
               ("machine_actor", ppo.machine_actor_gcn)]
    if getattr(args, "use_agv", 0):
        modules.append(("agv_actor", ppo.agv_actor))
    for name, module in modules:
        path = os.path.join(model_dir, f"PPO_{name}_{id_name}_top1.pth")
        module.load_state_dict(torch.load(path, map_location=device))
        print(f"loaded {path}", flush=True)

    top_ctl = None
    if getattr(args, "hier", 0):
        from hierarchy.rules import (N_AGV_RULES, N_JOB_RULES,
                                     N_MACHINE_RULES)
        from hierarchy.top_ppo import TopLevelController
        head_sizes = [N_JOB_RULES, N_MACHINE_RULES]
        if getattr(args, "use_agv", 0):
            head_sizes.append(N_AGV_RULES)
        top_ctl = TopLevelController(
            input_dim=18, head_sizes=tuple(head_sizes),
            credit=getattr(args, "top_credit", "team"), device=device)
        path = os.path.join(model_dir, f"PPO_hier_top_{id_name}_top1.pth")
        top_ctl.load_state_dict(torch.load(path, map_location=device))
        print(f"loaded {path}", flush=True)

    test_instance = Instance_Dataset(
        generate_true=0,
        dataset_pth=os.path.join(REPO, "instance",
                                 f"test_Instance_{id_name}.pkl"))

    objs, cost4 = [], []
    for i in range(100):
        _, cost, obj = validate_cost_gcn_jointActor_GAT(
            ppo=ppo, gantt_flag=False, data=test_instance, data_index=i,
            data_type="random", greedy=_EVAL_GREEDY, args=variant,
            top_ctl=top_ctl)
        objs.append(float(obj))
        cost4.append([float(x) for x in cost])
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/100", flush=True)

    objs = np.asarray(objs)
    cost4 = np.asarray(cost4)
    print(f"\nmode={'greedy' if _EVAL_GREEDY else 'sampling'}")
    print(f"obj mean={objs.mean():.2f} std={objs.std():.2f}")
    for j, name in enumerate(("mk", "pt", "transT", "idleT")):
        print(f"{name} mean={cost4[:, j].mean():.2f}")
    out = os.path.join(model_dir, f"eval_st_seed3{_EVAL_SUFFIX}.npz")
    np.savez(out, obj=objs, cost4=cost4)
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
