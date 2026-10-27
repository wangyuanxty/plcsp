# -*- coding: utf-8 -*-
"""Record the top-level RS actions on the 6x6 test instances.

For each RS arm, run the greedy top-level policy and print the per-head
action histogram and the entropy of the marginal action distribution.
A marginal entropy near 0 means one action dominates across states.
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
from trainer.train_device import device  # noqa: E402
from trainer.validate import validate_cost_gcn_jointActor_GAT  # noqa: E402
from hierarchy.rules import (N_AGV_RULES, N_JOB_RULES,  # noqa: E402
                             N_MACHINE_RULES)
from hierarchy.top_ppo import TopLevelController  # noqa: E402

N_JOB, N_MACHINE, N_EDGE = 6, 6, 2
ARMS = {
    "E1": {"use_agv": 0, "heads": (N_JOB_RULES, N_MACHINE_RULES)},
    "E4a": {"use_agv": 1, "heads": (N_JOB_RULES, N_MACHINE_RULES,
                                    N_AGV_RULES),
            "agv_top_rule": 1, "agv_dst_actor": 0},
    "E4b": {"use_agv": 1, "heads": (N_JOB_RULES, N_MACHINE_RULES,
                                    N_AGV_RULES),
            "agv_top_rule": 0, "agv_dst_actor": 1},
}


def _marginal_entropy(counts):
    p = counts / max(1, counts.sum())
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def run_arm(arm: str, seed: int, data, limit: int, run_tag: str,
            credit: str = "team", dump_probs: str = ""):
    spec = ARMS[arm]
    cfg = dict(vars(base_args))
    cfg.update({
        "n_job": N_JOB, "n_machine": N_MACHINE, "n_edge": N_EDGE,
        "encoder_type": "settrans", "machine_intra_attn": 1,
        "st_token_context": 1, "st_task_features": 1,
        "st_edge_features": 0,
        "use_agv": int(spec["use_agv"]),
        "n_agv": 6, "cross_delay": 5.0, "cross_leg_ratio": 0.5,
        "agv_rule": 0,
        "agv_top_rule": int(spec.get("agv_top_rule", 0)),
        "agv_dst_actor": int(spec.get("agv_dst_actor", 1)),
        "hier": 1, "hier_interval": 8, "hier_alpha": 0.5,
        "top_credit": credit,
        "eval_seed": 3, "test_seed": 3, "log_to_wandb": 0,
    })
    ppo = PPOAlgorithm(args=cfg, load_pretrained=False)
    id_name = f"J{N_JOB}M{N_MACHINE}E{N_EDGE}"
    model_dir = ROOT / "plcsp_v5" / "trained_model" / run_tag
    modules = [("job_actor", ppo.job_actor),
               ("machine_actor", ppo.machine_actor_gcn)]
    if cfg["use_agv"]:
        modules.append(("agv_actor", ppo.agv_actor))
    for name, module in modules:
        path = model_dir / f"PPO_{name}_{id_name}_top1.pth"
        if path.exists():
            module.load_state_dict(torch.load(path, map_location=device))
    ctl = TopLevelController(input_dim=18, head_sizes=spec["heads"],
                             credit=credit, device=device)
    top_path = model_dir / f"PPO_hier_top_{id_name}_top1.pth"
    ctl.load_state_dict(torch.load(top_path, map_location=device))

    records = []
    decision_idx = {}
    ctx = {"inst": 0}
    orig_act = ctl.act

    def record_act(state, greedy=False, store=False):
        action, logprob, value = orig_act(state, greedy=greedy,
                                          store=store)
        with torch.no_grad():
            logits_list, _ = ctl.model(state)
            probs = [torch.softmax(l, dim=-1).detach().cpu()
                     .numpy().reshape(-1).tolist() for l in logits_list]
        d = decision_idx.get(ctx["inst"], 0)
        decision_idx[ctx["inst"]] = d + 1
        records.append((ctx["inst"], d,
                        action.detach().cpu().numpy().reshape(-1), probs))
        return action, logprob, value

    ctl.act = record_act
    for i in range(int(limit)):
        ctx["inst"] = i
        validate_cost_gcn_jointActor_GAT(
            ppo=ppo, gantt_flag=False, data=data, data_index=i,
            data_type="random", greedy=True, args=cfg, top_ctl=ctl)

    acts = np.stack([a for _, _, a, _ in records], axis=0)
    first = {}
    for inst, _, a, _ in records:
        first.setdefault(inst, a)
    first = np.stack(list(first.values()), axis=0)
    out = {"arm": arm, "credit": credit, "run_tag": run_tag,
           "n_instances": int(limit),
           "n_decisions": len(records), "heads": []}
    for h in range(acts.shape[1]):
        n = spec["heads"][h]
        counts = np.bincount(acts[:, h], minlength=n)
        fcounts = np.bincount(first[:, h], minlength=n)
        out["heads"].append({
            "head": h,
            "counts": counts.tolist(),
            "marginal_entropy": _marginal_entropy(counts),
            "top_share": float(counts.max() / max(1, counts.sum())),
            "first_counts": fcounts.tolist(),
        })
    if dump_probs:
        Path(dump_probs).write_text(json.dumps([
            {"inst": int(i), "decision": int(d), "probs": p}
            for i, d, _, p in records]), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arm", default="", choices=[""] + list(ARMS),
                    help="run one arm only (default: all arms)")
    ap.add_argument("--credit", default="team",
                    choices=["team", "local", "coma", "hybrid"])
    ap.add_argument("--run-tag", default="",
                    help="checkpoint subdirectory; requires --arm")
    ap.add_argument("--out", default="")
    ap.add_argument("--dump-probs", default="")
    args = ap.parse_args()
    data = Instance_Dataset(
        generate_true=0,
        dataset_pth=str(ROOT / "plcsp_v5" / "instance" /
                        f"test_Instance_J{N_JOB}M{N_MACHINE}E{N_EDGE}.pkl"))
    results = []
    arms = ([args.arm] if args.arm else list(ARMS))
    for arm in arms:
        tag = (args.run_tag if args.run_tag
               else f"planheads3_{arm}_J6M6E2_s{args.seed}_remote")
        res = run_arm(arm, args.seed, data, args.limit, tag,
                      credit=args.credit, dump_probs=args.dump_probs)
        results.append(res)
        print(json.dumps(res, indent=1), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1),
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
