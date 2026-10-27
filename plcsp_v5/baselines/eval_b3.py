# -*- coding: utf-8 -*-
"""B3-AGV baseline: Lei et al. 2022 job/machine policy + AGV rule.

The pretrained job/machine actors are the ones shipped with this repository
(``tester/ESWA_MPPO``).  They were trained without AGV.  The adapted
baseline runs the same actors and dispatches transport with the earliest-free
AGV rule (``--agv_rule 1``).

Run on the remote GPU host::

    /root/venv/bin/python plcsp_v5/baselines/eval_b3.py --agv 1 \
        --out results/b3_agv_100.json
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

# parameters.py calls parse_args() at import time; hide this script's argv.
_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from parameters import args as base_args  # noqa: E402
sys.argv = _argv

from algorithm.ppo_algorithm import PPOAlgorithm  # noqa: E402
from instance.generate_allsize_mofjsp_dataset import Instance_Dataset  # noqa: E402
from trainer.validate import esa_validate_cost_gcn_jointActor_GAT  # noqa: E402

W = (0.4, 0.4, 0.2)


def _load_state_padded(model, path, device):
    """Load a checkpoint, padding a widened first layer with zeros.

    The B3 machine actor was trained on 6 machine features.  The AGV
    environment exposes 10.  Zero-padding the first linear layer keeps the
    learned weights and lets the actor ignore the four AGV-only features.
    """
    ckpt = torch.load(path, map_location=device)
    target = model.state_dict()
    out = {}
    for key, value in ckpt.items():
        if key not in target:
            raise RuntimeError(f"{path}: unexpected key {key}")
        if value.shape == target[key].shape:
            out[key] = value
            continue
        if (value.dim() == target[key].dim()
                and all(a <= b for a, b in zip(value.shape,
                                               target[key].shape))):
            padded = torch.zeros_like(target[key])
            slices = tuple(slice(0, int(s)) for s in value.shape)
            padded[slices] = value
            out[key] = padded
            print(f"padded {key}: {tuple(value.shape)} -> "
                  f"{tuple(target[key].shape)}", flush=True)
        else:
            raise RuntimeError(
                f"{path}: shape mismatch for {key}: "
                f"{tuple(value.shape)} vs {tuple(target[key].shape)}")
    model.load_state_dict(out, strict=True)


def _load_weights(ppo, n_job, n_machine, n_edge, iters):
    pth = ROOT / "plcsp_v5" / "tester" / "ESWA_MPPO"
    job = pth / f"esa_PPO_operation_actor_J{n_job}M{n_machine}E{n_edge}_{iters}.pth"
    mch = pth / f"esa_PPO_machine_actor_J{n_job}M{n_machine}E{n_edge}_{iters}.pth"
    device = next(ppo.esa_job_actor.parameters()).device
    _load_state_padded(ppo.esa_job_actor, job, device)
    _load_state_padded(ppo.esa_machine_actor_gcn, mch, device)
    return job, mch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agv", type=int, default=1)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--n-job", type=int, default=6)
    ap.add_argument("--n-machine", type=int, default=6)
    ap.add_argument("--n-edge", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    n_job, n_machine, n_edge = args.n_job, args.n_machine, args.n_edge
    cfg = dict(vars(base_args))
    cfg.update({
        "n_job": n_job,
        "n_machine": n_machine,
        "n_edge": n_edge,
        "use_agv": int(args.agv),
        "n_agv": 6,
        "cross_delay": 5.0,
        "cross_leg_ratio": 0.5,
        "agv_rule": 1,
        "agv_dst_actor": 0,
        "eval_sample": int(args.limit),
        "test_seed": 3,
        "log_to_wandb": 0,
    })

    ppo = PPOAlgorithm(args=cfg, load_pretrained=False)
    job, mch = _load_weights(ppo, n_job, n_machine, n_edge, args.iters)
    print(f"loaded {job.name} / {mch.name}", flush=True)

    data_path = (ROOT / "plcsp_v5" / "instance" /
                 f"test_Instance_J{n_job}M{n_machine}E{n_edge}.pkl")
    data = Instance_Dataset(generate_true=0, dataset_pth=str(data_path))

    records = []
    for i in range(int(args.limit)):
        _, real_4r, _ = esa_validate_cost_gcn_jointActor_GAT(
            ppo=ppo, gantt_flag=False, data=data, data_index=i,
            data_type="random", greedy=True, args=cfg)
        mk, pt, trans_t, idle_t = [float(x) for x in real_4r]
        obj = (W[0] * mk + W[1] * (pt / (n_job * n_machine) + idle_t)
               + W[2] * trans_t)
        records.append({"instance": i, "mk": mk, "pt": pt,
                        "transT": trans_t, "idleT": idle_t, "obj": obj})
        print(f"{i + 1}/{args.limit} obj={obj:.2f}", flush=True)

    objs = np.asarray([r["obj"] for r in records], dtype=float)
    summary = {
        "mean": float(objs.mean()),
        "std": float(objs.std(ddof=0)),
        "min": float(objs.min()),
        "max": float(objs.max()),
    }
    payload = {
        "mode": "B3-AGV" if args.agv else "B3-noAGV",
        "source": "Lei et al. 2022, ESWA 205:117796; adapted",
        "weights": [str(job), str(mch)],
        "n_job": n_job, "n_machine": n_machine, "n_edge": n_edge,
        "agv": bool(args.agv),
        "n": len(records),
        "summary": summary,
        "records": records,
    }
    print(json.dumps(payload, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=1),
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
