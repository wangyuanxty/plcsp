"""Run the Wang et al. (IEEE IoT-J 2025) MIP baseline on the v5 pools.

Each instance is solved with a time limit, then the MIP schedule is
scored a second time with the v5 decoder
(``plcsp_v5.baselines.agv_baselines.decode``, ``agv=False``), so the
two objectives are on the same scale::

    their objective: 0.4*MKT + 0.4*(PEC/n_task + SEC) + 0.2*TEC
    our objective:   decode(...)["obj"]

Run from the repository root::

    D:/anaconda/envs/py312/python.exe -m plcsp_v5.baselines.mip_e2e.run \
        --out plcsp_v5/tester/mip_e2e/mo_fjsp_mip_j6m6.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..agv_baselines import decode
from .model import (LIMIT_CONSTR, LIMIT_VARS, Instance, Solution,
                    load_instance, solve)


def _score(inst: Instance, sol: Solution) -> Dict[str, float]:
    """Re-score the MIP schedule with the v5 decoder."""
    out = decode(inst.t, inst.power, inst.travel, inst.edges,
                 sol.os_seq, sol.machines, agv=False)
    return {"obj": float(out["obj"]), "mk": float(out["mk"]),
            "idle": float(out["idleT"]), "trans": float(out["transT"]),
            "pec_mean": float(out["pt"]) / inst.n_task}


def _record(inst: Instance, sol: Solution) -> Dict[str, object]:
    rec: Dict[str, object] = {
        "instance": inst.index,
        "status": sol.status,
        "ok": bool(sol.ok),
        "certified": bool(sol.certified),
        "rounds": int(sol.rounds),
        "n_vars": int(sol.n_vars),
        "n_constrs": int(sol.n_constrs),
        "runtime": float(sol.runtime),
        "gap": float(sol.gap) if sol.ok else None,
        "message": sol.message,
    }
    if sol.ok:
        rec.update({
            "mip_obj": float(sol.objective),
            "mip_mk": float(sol.mkt),
            "mip_pec_mean": float(sol.pec) / inst.n_task,
            "mip_idle": float(sol.idle),
            "mip_idle_eq5": float(sol.idle_eq5),
            "mip_idle_csv": float(sol.idle_csv),
            "mip_trans": float(sol.trans),
        })
        rec.update({"dec_" + k: v for k, v in _score(inst, sol).items()})
    return rec


def _stat(values: Sequence[float]) -> Dict[str, float]:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return {"n": 0, "mean": None, "std": None}
    return {"n": int(a.size), "mean": float(a.mean()),
            "std": float(a.std(ddof=0)), "min": float(a.min()),
            "max": float(a.max())}


def _summary(records: List[Dict[str, object]]) -> Dict[str, object]:
    ok = [r for r in records if r["ok"]]
    key = lambda name: [r.get(name) for r in ok]  # noqa: E731
    return {
        "n_instances": len(records),
        "n_solved": len(ok),
        "n_failed": len(records) - len(ok),
        "failed_instances": [r["instance"] for r in records if not r["ok"]],
        "n_uncertified": sum(1 for r in ok if not r["certified"]),
        "n_licence_limit": sum(1 for r in records
                               if r["status"] == "LICENSE_LIMIT"),
        "n_no_incumbent": sum(1 for r in records
                              if r["status"] != "LICENSE_LIMIT"
                              and not r["ok"]),
        "their_obj": _stat(key("mip_obj")),
        "their_mk": _stat(key("mip_mk")),
        "their_pec_mean": _stat(key("mip_pec_mean")),
        "their_idle": _stat(key("mip_idle")),
        "their_idle_eq5": _stat(key("mip_idle_eq5")),
        "their_idle_csv": _stat(key("mip_idle_csv")),
        "their_trans": _stat(key("mip_trans")),
        "our_obj": _stat(key("dec_obj")),
        "our_mk": _stat(key("dec_mk")),
        "our_pec_mean": _stat(key("dec_pec_mean")),
        "our_idle": _stat(key("dec_idle")),
        "our_trans": _stat(key("dec_trans")),
        "runtime_s": _stat(key("runtime")),
        "gap": _stat(key("gap")),
        "n_vars": _stat(key("n_vars")),
        "n_constrs": _stat(key("n_constrs")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances",
                    default="plcsp_v5/instance/test_Instance_J6M6E2.pkl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--time_limit", type=float, default=60.0)
    ap.add_argument("--formulation", choices=("compact", "paper"),
                    default="compact")
    ap.add_argument("--sec", choices=("csv", "eq5"), default="csv",
                    help="standby-energy reading: csv = sum_k (MKT - P_k), "
                         "eq5 = busy span minus processing")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with open(args.instances, "rb") as fh:
        pool = pickle.load(fh)
    n = len(pool[0]) if args.limit <= 0 else min(args.limit, len(pool[0]))

    records: List[Dict[str, object]] = []
    t0 = time.time()
    for idx in range(n):
        inst = load_instance(args.instances, idx)
        sol = solve(inst, time_limit=args.time_limit,
                    formulation=args.formulation, sec=args.sec,
                    threads=args.threads, verbose=args.verbose)
        rec = _record(inst, sol)
        records.append(rec)
        if sol.ok:
            print(f"[{idx:3d}] {sol.status:10s} cert={sol.certified} "
                  f"their={sol.objective:8.2f} our={rec['dec_obj']:8.2f} "
                  f"t={sol.runtime:6.1f}s", flush=True)
        else:
            print(f"[{idx:3d}] {sol.status:10s} {sol.message}", flush=True)

    wall = time.time() - t0
    summary = _summary(records)
    summary["wall_s"] = wall
    payload = {
        "source": "Wang et al., IEEE IoT-J 2025, eqs (3)-(8)",
        "instances": args.instances,
        "n_instances": n,
        "time_limit": args.time_limit,
        "formulation": args.formulation,
        "sec": args.sec,
        "licence": {"max_vars": LIMIT_VARS, "max_constrs": LIMIT_CONSTR},
        "summary": summary,
        "records": records,
    }
    print(json.dumps({k: summary[k] for k in
                      ("n_instances", "n_solved", "n_failed",
                       "n_uncertified", "their_obj", "our_obj",
                       "their_mk", "their_idle", "their_trans",
                       "their_idle_eq5", "their_idle_csv",
                       "runtime_s", "wall_s")},
                     indent=2))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
