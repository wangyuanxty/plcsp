# -*- coding: utf-8 -*-
"""Check one RGA best schedule against the v5 environment and the MIP CSV.

Run from the repo root (or /root on the remote host)::

    /root/venv/bin/python plcsp_v5/baselines/check_rga_best.py --instance 0
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plcsp_v5"))
sys.path.insert(0, str(ROOT / "plcsp_v5" / "graph-jsp-env" / "src"))

from baselines.agv_baselines import decode, rga_instance  # noqa: E402
from baselines.check_agv_env import run_env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances",
                    default="plcsp_v5/instance/test_Instance_J6M6E2.pkl")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--gen", type=int, default=100)
    args = ap.parse_args()

    data = pickle.load(open(ROOT / args.instances, "rb"))
    t, p, tt, edge = [x[args.instance] for x in data]
    keys = ("mk", "pt", "transT", "idleT", "obj")

    for agv in (0, 1):
        r = rga_instance(t, p, tt, edge, seed=args.seed, pop=args.pop,
                         gen=args.gen, agv=bool(agv),
                         return_schedule=True)
        dec = decode(t, p, tt, edge, r["os_seq"], r["ms"],
                     agv=bool(agv))
        env = run_env(t, p, tt, edge, r["os_seq"], r["ms"],
                      bool(agv))
        print(f"=== instance {args.instance}, agv={agv} ===")
        print("rga ", {k: round(r[k], 4) for k in keys})
        print("dec ", {k: round(dec[k], 4) for k in keys})
        print("env ", {k: round(env[k], 4) for k in keys})
        print("diff", {k: round(abs(dec[k] - env[k]), 6) for k in keys})

    csv_path = (ROOT / "plcsp_v5" / "tester" / "Solver_seed3" /
                "MO_FJSP_MIP_result_(J6_M6_seed3_sample100_w442).csv")
    row = list(csv.DictReader(open(csv_path)))[args.instance]
    mk = float(row["Makespan"])
    ec = float(row["MachineEC"])
    it = float(row["MachineIdleT"])
    tt_mip = float(row["TransEC"])
    obj = 0.4 * mk + 0.4 * (ec + it) + 0.2 * tt_mip
    print("=== MIP CSV row ===")
    print(row)
    print("MIP recomputed:", {"mk": mk, "pt": ec, "transT": tt_mip,
                              "idleT": it, "obj": obj})


if __name__ == "__main__":
    main()
