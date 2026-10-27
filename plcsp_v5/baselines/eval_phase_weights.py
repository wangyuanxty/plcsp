# -*- coding: utf-8 -*-
"""Evaluate fixed phase-wise objective weights on a trained low-level model.

No retraining.  The low-level actors are conditioned on the phase weight
vector; the reported Objective always uses the evaluation weights
0.4 / 0.4 / 0.2.

Run on the remote host::

    /root/venv/bin/python plcsp_v5/baselines/eval_phase_weights.py \
        --run-tag plan_E0_J6M6E2_s1_remote --limit 100
"""
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

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from parameters import args as base_args  # noqa: E402
sys.argv = _argv

from algorithm.ppo_algorithm import PPOAlgorithm  # noqa: E402
from instance.generate_allsize_mofjsp_dataset import Instance_Dataset  # noqa: E402
from trainer.validate import validate_cost_gcn_jointActor_GAT  # noqa: E402

N_JOB, N_MACHINE, N_EDGE = 6, 6, 2
EVAL_W = [0.4, 0.4, 0.2]


def schedules():
    return {
        "identity": [EVAL_W, EVAL_W, EVAL_W],
        "uniform": [[1/3, 1/3, 1/3]] * 3,
        "mk_early": [[0.6, 0.3, 0.1], [0.4, 0.4, 0.2], [0.2, 0.6, 0.2]],
        "ec_early": [[0.2, 0.6, 0.2], [0.4, 0.4, 0.2], [0.6, 0.3, 0.1]],
        "mk_sharp": [[0.8, 0.1, 0.1]] * 3,
        "ec_sharp": [[0.1, 0.8, 0.1]] * 3,
        "trans_early": [[0.3, 0.3, 0.4], [0.4, 0.4, 0.2],
                        [0.5, 0.4, 0.1]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default="plan_E0_J6M6E2_s1_remote")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--only", default="",
                    help="comma-separated schedule names")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = dict(vars(base_args))
    cfg.update({
        "n_job": N_JOB,
        "n_machine": N_MACHINE,
        "n_edge": N_EDGE,
        "encoder_type": "settrans",
        "machine_intra_attn": 1,
        "st_token_context": 1,
        "st_task_features": 1,
        "st_edge_features": 0,
        "use_agv": 0,
        "eval_seed": 3,
        "test_seed": 3,
        "log_to_wandb": 0,
    })
    ppo = PPOAlgorithm(args=cfg, load_pretrained=False)
    model_dir = (ROOT / "plcsp_v5" / "trained_model" / args.run_tag)
    id_name = f"J{N_JOB}M{N_MACHINE}E{N_EDGE}"
    for name, module in (("job_actor", ppo.job_actor),
                         ("machine_actor", ppo.machine_actor_gcn)):
        path = model_dir / f"PPO_{name}_{id_name}_top1.pth"
        module.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"loaded {path}", flush=True)

    data = Instance_Dataset(
        generate_true=0,
        dataset_pth=str(ROOT / "plcsp_v5" / "instance" /
                        f"test_Instance_{id_name}.pkl"))

    out = {}
    scheds = schedules()
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        scheds = {k: scheds[k] for k in keys}
    for name, sched in scheds.items():
        vals = []
        for i in range(int(args.limit)):
            _, _, obj = validate_cost_gcn_jointActor_GAT(
                ppo=ppo, gantt_flag=False, data=data, data_index=i,
                data_type="random", greedy=True, args=cfg,
                phase_weights=sched)
            vals.append(float(obj))
        arr = np.asarray(vals)
        out[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "values": [float(v) for v in vals],
            "schedule": sched,
        }
        print(f"{name:12s} mean={arr.mean():.2f} std={arr.std():.2f}",
              flush=True)

    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1),
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
