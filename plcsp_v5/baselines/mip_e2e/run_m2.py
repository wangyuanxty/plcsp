"""Run the M2 (AGV-aware) MIP baseline on the v5 six-array pools.

For every instance the runner

1. picks a machine assignment from the PDR rules and a short RGA run
   (:func:`model_m2.choose_assignment`; the MIP fixes the assignment,
   see the module docstring for why the licence forces this),
2. builds and solves the fixed-assignment M2 MIP with a time limit,
3. re-verifies the schedule with an independent replay of the AGV
   chains (:func:`model_m2.verify_m2`).

Every record is scored with our objective, so the numbers are
comparable with the other v5 baselines::

    0.4 * Cmax + 0.4 * (PEC / n_task + idle) + 0.2 * (loaded + 0.5 * empty)

Run from the repository root::

    D:/anaconda/envs/py312/python.exe -m plcsp_v5.baselines.mip_e2e.run_m2 \
        --instances plcsp_v5/instance/test_Instance_J6M6E2_lu.pkl \
        --limit 10 --time_limit 60 \
        --out plcsp_v5/tester/mip_m2/m2_mip_j6m6.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .model_m2 import (LIMIT_CONSTR, LIMIT_VARS, InstanceM2, SolutionM2,
                       build_legs, choose_assignment, load_instance_m2,
                       solve_m2)
from .verify_m2 import accept_check, verify_reference

SUMMARY_KEYS = ("n_instances", "n_solved", "n_feasible", "n_ref_feasible",
                "n_licence_limit", "objective", "mk", "idle", "loaded",
                "empty", "trans", "decoded_obj", "ref_diff", "runtime_s",
                "gap", "n_vars", "n_constrs", "n_legs", "wall_s")


def _stat(values: Sequence[float]) -> Dict[str, object]:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {"n": int(a.size), "mean": float(a.mean()),
            "std": float(a.std(ddof=0)), "min": float(a.min()),
            "max": float(a.max())}


def _record(inst: InstanceM2, sol: SolutionM2,
            assign_obj: float, assign_src: str, n_legs: int,
            select_s: float) -> Dict[str, object]:
    rec: Dict[str, object] = {
        "instance": inst.index,
        "status": sol.status,
        "ok": bool(sol.ok),
        "feasible": bool(sol.ok and not sol.message),
        "message": sol.message,
        "assignment_source": assign_src,
        "assignment_obj": float(assign_obj),
        "assignment_s": float(select_s),
        "n_vars": int(sol.n_vars),
        "n_constrs": int(sol.n_constrs),
        "n_legs": int(n_legs),
        "n_chains": len(sol.chains),
        "runtime": float(sol.runtime),
    }
    if sol.ok:
        ref = verify_reference(inst, sol.machines, sol.starts, sol.chains)
        rec.update({
            "objective": float(sol.objective),
            "lower_bound": float(sol.lower_bound),
            "gap": float(sol.gap),
            "mk": float(sol.mk),
            "pec": float(sol.pec),
            "idle": float(sol.idle),
            "loaded": float(sol.loaded),
            "empty": float(sol.empty),
            "trans": float(sol.trans),
            "ref_feasible": bool(ref["feasible"]),
            "ref_obj": float(ref["obj"]),
            "ref_diff": abs(float(ref["obj"]) - float(sol.objective)),
        })
    return rec


def _summary(records: List[Dict[str, object]]) -> Dict[str, object]:
    ok = [r for r in records if r["ok"]]
    feas = [r for r in records if r["feasible"]]

    def col(name: str, rows: Sequence[Dict[str, object]]) -> List[float]:
        return [r[name] for r in rows if r.get(name) is not None]

    return {
        "n_instances": len(records),
        "n_solved": len(ok),
        "n_feasible": len(feas),
        "n_ref_feasible": sum(1 for r in ok if r.get("ref_feasible")),
        "n_licence_limit": sum(1 for r in records
                               if r["status"] == "LICENSE_LIMIT"),
        "n_unverified": sum(1 for r in ok if not r["feasible"]),
        "failed_instances": [r["instance"] for r in records if not r["ok"]],
        "objective": _stat(col("objective", feas)),
        "mk": _stat(col("mk", feas)),
        "idle": _stat(col("idle", feas)),
        "loaded": _stat(col("loaded", feas)),
        "empty": _stat(col("empty", feas)),
        "trans": _stat(col("trans", feas)),
        "decoded_obj": _stat(col("assignment_obj", feas)),
        "ref_diff": _stat(col("ref_diff", feas)),
        "runtime_s": _stat(col("runtime", feas)),
        "gap": _stat(col("gap", feas)),
        "n_vars": _stat(col("n_vars", feas)),
        "n_constrs": _stat(col("n_constrs", feas)),
        "n_legs": _stat(col("n_legs", feas)),
    }


def _base_payload(args, n: int) -> Dict[str, object]:
    return {
        "source": "M2 (AGV-aware) MIP, v5 objective",
        "instances": args.instances,
        "n_instances": n,
        "time_limit": args.time_limit,
        "licence": {"max_vars": LIMIT_VARS, "max_constrs": LIMIT_CONSTR},
        "formulation": "fixed machine assignment + exact AGV chain cover",
        "assignment_search": {"rga_seeds": args.rga_seeds,
                              "rga_pop": args.rga_pop,
                              "rga_gen": args.rga_gen,
                              "pdr_rules": 12,
                              "note": "best decode_m2 value over the candidates"},
    }


def run(args) -> Dict[str, object]:
    records: List[Dict[str, object]] = []
    t0 = time.time()
    for idx in range(args.limit):
        inst = load_instance_m2(args.instances, idx)
        ts = time.time()
        machines, assign_obj, src = choose_assignment(
            inst, rga_seeds=args.rga_seeds, rga_pop=args.rga_pop,
            rga_gen=args.rga_gen, seed=args.seed)
        select_s = time.time() - ts
        sol = solve_m2(inst, machines, n_agv=args.n_agv,
                       time_limit=args.time_limit, mip_gap=args.mip_gap,
                       threads=args.threads, verbose=args.verbose)
        n_legs = len(build_legs(inst, machines))
        rec = _record(inst, sol, assign_obj, src, n_legs, select_s)
        records.append(rec)
        if sol.ok:
            print(f"[{idx:3d}] {sol.status:12s} obj={rec['objective']:9.3f} "
                  f"lb={rec['lower_bound']:9.3f} gap={rec['gap']:.4f} "
                  f"dec={assign_obj:9.3f} t={sol.runtime:5.1f}s "
                  f"vars={sol.n_vars:4d} cons={sol.n_constrs:5d} "
                  f"legs={n_legs:3d} ok={rec['feasible']} "
                  f"ref={rec['ref_feasible']} d={rec['ref_diff']:.1e}",
                  flush=True)
        else:
            print(f"[{idx:3d}] {sol.status:12s} {sol.message} "
                  f"(vars={sol.n_vars} cons={sol.n_constrs})", flush=True)

    summary = _summary(records)
    summary["wall_s"] = time.time() - t0
    payload = _base_payload(args, args.limit)
    payload["summary"] = summary
    payload["records"] = records
    print(json.dumps({k: summary[k] for k in SUMMARY_KEYS}, indent=2))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"wrote {path}")
    return payload


def run_check(args) -> Dict[str, object]:
    """Acceptance check on every instance: the MIP accepts the decoder."""
    from ..agv_baselines import pdr_schedule

    out = []
    worst = 0.0
    for idx in range(args.limit):
        inst = load_instance_m2(args.instances, idx)
        machines, _, _ = choose_assignment(
            inst, rga_seeds=0, seed=args.seed)
        os_seq, _ = pdr_schedule(inst.t, inst.power, "mor", "spt")
        res = accept_check(inst, machines, os_seq, time_limit=args.check_time)
        worst = max(worst, res.variance)
        out.append({"instance": idx, "feasible": res.feasible,
                    "model_obj": res.model_obj, "decoded_obj": res.decoded_obj,
                    "variance": res.variance, "n_vars": res.n_vars,
                    "n_constrs": res.n_constrs})
        print(f"[{idx:3d}] accept={res.feasible} "
              f"model={res.model_obj:.6f} dec={res.decoded_obj:.6f} "
              f"|diff|={res.variance:.2e}", flush=True)
    ok = all(r["feasible"] for r in out)
    payload = {"mode": "accept", "instances": args.instances,
               "n_instances": args.limit, "all_feasible": ok,
               "max_abs_diff": worst, "records": out}
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"wrote {path}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances",
                    default="plcsp_v5/instance/test_Instance_J6M6E2_lu.pkl")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--time_limit", type=float, default=60.0)
    ap.add_argument("--mip_gap", type=float, default=1e-4)
    ap.add_argument("--n_agv", type=int, default=6)
    ap.add_argument("--rga_seeds", type=int, default=2)
    ap.add_argument("--rga_pop", type=int, default=50)
    ap.add_argument("--rga_gen", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--check", action="store_true",
                    help="run the fixed-schedule acceptance check instead")
    ap.add_argument("--check_time", type=float, default=30.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.check:
        run_check(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
