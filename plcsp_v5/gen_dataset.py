"""Generate train / eval / test pkl files for one instance size.

The CSV output is skipped by pointing ``csv_pth`` at a missing directory.
Existing pkl files are left untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

V5_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, V5_DIR)

from instance.generate_allsize_mofjsp_dataset import Instance_Dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_job", type=int, required=True)
    parser.add_argument("--n_machine", type=int, required=True)
    parser.add_argument("--n_edge", type=int, required=True)
    parser.add_argument("--train_samples", type=int, default=12800)
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=1)
    parser.add_argument("--test_seed", type=int, default=3)
    args = parser.parse_args()

    with open(os.path.join(V5_DIR, "instance", "config_ins.json"),
              "r", encoding="utf-8") as f:
        conf = json.load(f)
    scope = conf["ability_scope"]
    pth = os.path.join(V5_DIR, "instance")
    skip_csv = os.path.join(pth, "__skip__", "x.csv")

    tag = f"J{args.n_job}M{args.n_machine}E{args.n_edge}"
    specs = [
        ("train", args.train_samples, args.train_seed,
         os.path.join(pth, f"Instance_{tag}.pkl")),
        ("eval", args.eval_samples, args.eval_seed,
         os.path.join(pth, f"eval_Instance_{tag}.pkl")),
        ("test", args.eval_samples, args.test_seed,
         os.path.join(pth, f"test_Instance_{tag}.pkl")),
    ]
    for split, samples, seed, pkl_path in specs:
        if os.path.exists(pkl_path):
            print(f"[gen] exists: {pkl_path}", flush=True)
            continue
        print(f"[gen] {split}: samples={samples} seed={seed} -> {pkl_path}",
              flush=True)
        Instance_Dataset(
            samples=samples,
            n_job=args.n_job,
            n_machine=args.n_machine,
            n_edge=args.n_edge,
            ability_scope=scope,
            use_PT=0,
            seed=seed,
            generate_true=1,
            csv_pth=skip_csv,
            pkl_pth=pkl_path)
    print("[gen] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
