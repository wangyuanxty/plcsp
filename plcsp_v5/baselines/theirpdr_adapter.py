# -*- coding: utf-8 -*-
"""Run Wang et al.'s upstream PDR code on a v5 instance pool.

The upstream implementation lives in ``plcsp_v5/tester/pdrs.py`` and uses
the disjunctive-graph environment in ``plcsp_v5/graph-jsp-env``.  This
module is a read-only adapter: it builds their ``args`` dict and their
four-array data object, calls
``run_Rules_jointActions_withMinus_1217`` for the 12 rule pairs, and
records the four real cost terms that the function returns at ``done``.

Two idle readings are run, because the upstream environment has both:

* ``trailing`` is the current v5 default, ``sum_k (MKT - P_k)``;
* ``gap`` is the original upstream code path, leading idle plus the gaps
  between consecutive operations on a machine.

The objective is the one used by both sides::

    0.4 * MK + 0.4 * (PEC / n_task + idleT) + 0.2 * transT

For every run this adapter also decodes the *same* operation sequence and
machine assignment with our decoder, in two modes: gap insertion (the v5
decoder) and append-only.  The append-only result must reproduce the
upstream environment; the gap result isolates the effect of left-shifting.

Published 6x6 values are Wang et al.'s Table V (P-G 732.5 is the trained
model and is not part of this table).

Usage::

    python plcsp_v5/baselines/theirpdr_adapter.py \
        --instances plcsp_v5/instance/test_Instance_J6M6E2.pkl \
        --workers 8 --out <path>.json

    # dump operation / machine orders for one instance
    python plcsp_v5/baselines/theirpdr_adapter.py --dump-orders --order-index 0
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V5 = ROOT / "plcsp_v5"
ENV_SRC = V5 / "graph-jsp-env" / "src"


def _bootstrap() -> None:
    """Make ``trainer``, ``graph_jsp_env`` and ``agv`` importable."""
    for path in (str(ROOT), str(V5), str(ENV_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


_bootstrap()

from baselines.agv_baselines import decode, pdr_schedule  # noqa: E402

W_MK = 0.4
W_EC = 0.4
W_TT = 0.2

O_RULES = [
    ("fifo", 0),
    ("mor", 1),
    ("lwkr_t", 2),
    ("lwkr_pt", 3),
    ("mwkr_t", 4),
    ("mwkr_pt", 5),
]
M_RULES = [("spt", 0), ("seyc", 1)]

# Wang et al., Table V, 6x6 (the P-G 732.5 entry is the trained model and
# is not a PDR value).
PUBLISHED_J6M6 = {
    ("mor", "spt"): 1045.6,
    ("mor", "seyc"): 1073.1,
    ("mwkr_t", "spt"): 1189.3,
    ("mwkr_t", "seyc"): 1147.2,
    ("mwkr_pt", "spt"): 1348.7,
    ("mwkr_pt", "seyc"): 1357.1,
    ("lwkr_t", "spt"): 2816.0,
    ("lwkr_t", "seyc"): 2790.4,
    ("lwkr_pt", "spt"): 2834.7,
    ("lwkr_pt", "seyc"): 2829.9,
    ("fifo", "spt"): 2894.5,
    ("fifo", "seyc"): 2927.1,
}

_PDRS = None


def load_pdrs():
    """Import the upstream ``tester/pdrs.py`` without touching it."""
    global _PDRS
    if _PDRS is None:
        path = V5 / "tester" / "pdrs.py"
        spec = importlib.util.spec_from_file_location("their_pdrs_upstream",
                                                      path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PDRS = module
    return _PDRS


def load_pool(path: str) -> tuple[np.ndarray, ...]:
    """Load a four-array pool ``(t, p, transT, edge)``."""
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    if len(data) != 4:
        raise SystemExit(f"{path} has {len(data)} arrays; this adapter "
                         f"needs the four-array no-AGV pool")
    return tuple(np.asarray(a) for a in data)


def make_args(n_job: int, n_machine: int, n_edge: int,
              idle_mode: str) -> dict:
    """Build the dict that upstream ``run_Rules...`` expects.

    Upstream indexes it by key (``args['n_job']``); the environment wraps
    the same dict as ``Variant(**configs)`` and uses attributes.
    """
    return {
        "n_job": int(n_job),
        "n_machine": int(n_machine),
        "n_edge": int(n_edge),
        "weight_mk": W_MK,
        "weight_ec": W_EC,
        "weight_tt": W_TT,
        "reward_scaling": {"scaling_divisor": 1},
        "idle_mode": idle_mode,
    }


def _rule_orders(pdrs, o_rule: int, m_rule: int, t: np.ndarray,
                 p: np.ndarray, seed: int):
    """Rebuild upstream's operation and machine lists for one instance.

    Only the MOR rule draws random numbers.  ``run_Rules`` seeds nothing,
    so reseeding here (and before the call in ``run_one``) reproduces the
    same order that the environment executed.
    """
    myrules = pdrs.FJSP_Rules(int(t.shape[0] // t.shape[1]), int(t.shape[1]))
    random.seed(seed)
    if o_rule == 0:
        operation_lst = myrules.FIFO_o()
    elif o_rule == 1:
        operation_lst = myrules.MOR_o()
    elif o_rule == 2:
        operation_lst = myrules.LWKR_T_o_jointActor(
            t, "mean", rule_type="least")
    elif o_rule == 3:
        operation_lst = myrules.LWKR_PT_o_jointActor(
            t, p, "mean", rule_type="least")
    elif o_rule == 4:
        operation_lst = myrules.LWKR_T_o_jointActor(
            t, "mean", rule_type="most")
    elif o_rule == 5:
        operation_lst = myrules.LWKR_PT_o_jointActor(
            t, p, "mean", rule_type="most")
    else:
        raise ValueError(o_rule)

    if m_rule == 0:
        machine_lst = myrules.SPT_m(t)
    elif m_rule == 1:
        machine_lst = myrules.SEC_m(t, p)
    else:
        raise ValueError(m_rule)
    return [int(x) for x in operation_lst], [int(x) for x in machine_lst]


def _objective(terms: dict) -> float:
    return (W_MK * terms["mk"]
            + W_EC * (terms["pt"] / terms["n_task"] + terms["idleT"])
            + W_TT * terms["transT"])


def _decode_terms(t, p, tt, edge, operation_lst, machine_lst,
                  insert_gaps: bool) -> dict:
    n_task, n_machine = t.shape
    os_seq = [(int(task_id) - 1) // n_machine for task_id in operation_lst]
    ms = np.asarray(machine_lst, dtype=int)
    out = decode(t, p, tt, edge, os_seq, ms, agv=False,
                 idle_mode="trailing", insert_gaps=insert_gaps)
    terms = {"mk": float(out["mk"]), "pt": float(out["pt"]),
             "transT": float(out["transT"]), "idleT": float(out["idleT"]),
             "n_task": int(n_task)}
    terms["obj"] = _objective(terms)
    return terms


def run_one(payload: dict) -> dict:
    """Run one (instance, rule pair, idle mode) through upstream code."""
    pdrs = load_pdrs()
    t, p, tt, edge = payload["t"], payload["p"], payload["tt"], payload["edge"]
    n_task, n_machine = t.shape
    n_edge = len(edge)
    args = make_args(n_task // n_machine, n_machine, n_edge,
                     payload["idle_mode"])

    # Upstream indexes data.t[data_index]; give it a one-instance pool.
    data = SimpleNamespace(
        t=t[np.newaxis, ...], p=p[np.newaxis, ...],
        transT=tt[np.newaxis, ...], edge=edge[np.newaxis, ...])

    random.seed(payload["seed"])
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, until_now = pdrs.run_Rules_jointActions_withMinus_1217(
            args=args, o_rule=payload["o_rule"],
            m_rule=payload["m_rule"], data=data, data_index=0,
            data_type="mean", MLWKR_type="max")
    mk, pt_avg, trans_t, idle_t = (float(x) for x in until_now)

    terms = {"mk": mk, "pt": pt_avg * n_task, "transT": trans_t,
             "idleT": idle_t, "n_task": int(n_task)}
    terms["obj"] = _objective(terms)

    # Same order, our decoder, two scheduling modes.
    operation_lst, machine_lst = _rule_orders(
        pdrs, payload["o_rule"], payload["m_rule"], t, p,
        payload["seed"])
    our_gap = _decode_terms(t, p, tt, edge, operation_lst, machine_lst,
                            insert_gaps=True)
    our_append = _decode_terms(t, p, tt, edge, operation_lst, machine_lst,
                               insert_gaps=False)

    # Our own PDR (sequence built by our job/machine rules).
    os_seq, ms = pdr_schedule(t, p, payload["job_rule"],
                              payload["machine_rule"])
    our_pdr = decode(t, p, tt, edge, os_seq, ms, agv=False,
                     idle_mode="trailing")
    our_pdr_terms = {
        "mk": float(our_pdr["mk"]), "pt": float(our_pdr["pt"]),
        "transT": float(our_pdr["transT"]),
        "idleT": float(our_pdr["idleT"]), "n_task": int(n_task)}
    our_pdr_terms["obj"] = _objective(our_pdr_terms)

    return {
        "instance": int(payload["instance"]),
        "job_rule": payload["job_rule"],
        "machine_rule": payload["machine_rule"],
        "o_rule": int(payload["o_rule"]),
        "m_rule": int(payload["m_rule"]),
        "idle_mode": payload["idle_mode"],
        "their": terms,
        "our_gap": our_gap,
        "our_append": our_append,
        "our_pdr": our_pdr_terms,
        "operation_lst": operation_lst,
        "machine_lst": machine_lst,
    }


def _stat(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)),
            "min": float(a.min()), "max": float(a.max())}


def _summarize(records: list[dict]) -> list[dict]:
    pairs = []
    for o_name, o_rule in O_RULES:
        for m_name, m_rule in M_RULES:
            rows = [r for r in records
                    if (r["o_rule"], r["m_rule"]) == (o_rule, m_rule)]
            if not rows:
                continue
            entry = {
                "job_rule": o_name,
                "machine_rule": m_name,
                "published": PUBLISHED_J6M6.get((o_name, m_name)),
                "n": len({r["instance"] for r in rows}),
            }
            for mode in ("trailing", "gap"):
                vals = [r["their"]["obj"] for r in rows
                        if r["idle_mode"] == mode]
                if not vals:
                    continue
                entry[f"their_{mode}"] = _stat(vals)
                entry[f"their_{mode}_components"] = {
                    k: float(np.mean([r["their"][k] for r in rows
                                      if r["idle_mode"] == mode]))
                    for k in ("mk", "pt", "transT", "idleT")}
            entry["our_gap"] = _stat([r["our_gap"]["obj"] for r in rows])
            entry["our_append"] = _stat([r["our_append"]["obj"] for r in rows])
            entry["our_pdr"] = _stat([r["our_pdr"]["obj"] for r in rows])
            pairs.append(entry)
    return pairs


def _print_table(pairs: list[dict]) -> None:
    head = ("rule pair", "their trailing", "their gap", "our decoder",
            "our append-only", "published")
    print("{:<18}{:>15}{:>12}{:>14}{:>16}{:>12}".format(*head))
    for e in pairs:
        name = f'{e["job_rule"]}+{e["machine_rule"]}'
        print("{:<18}{:>15.2f}{:>12.2f}{:>14.2f}{:>16.2f}{:>12.1f}".format(
            name,
            e.get("their_trailing", {}).get("mean", float("nan")),
            e.get("their_gap", {}).get("mean", float("nan")),
            e["our_gap"]["mean"], e["our_append"]["mean"],
            e["published"] or float("nan")))


def _dump_orders(args) -> None:
    t_all, p_all, tt_all, edge_all = load_pool(args.instances)
    i = args.order_index
    t, p = t_all[i], p_all[i]
    pdrs = load_pdrs()
    out = {"instance": i, "pairs": []}
    for o_name, o_rule in O_RULES:
        for m_name, m_rule in M_RULES:
            op, ml = _rule_orders(pdrs, o_rule, m_rule, t, p, args.seed)
            os_seq, ms = pdr_schedule(t, p, o_name, m_name)
            out["pairs"].append({
                "pair": f"{o_name}+{m_name}",
                "their_operation_lst": op,
                "their_machine_lst": ml,
                "their_job_order": [(x - 1) // t.shape[1] for x in op],
                "our_job_order": [int(x) for x in os_seq],
                "our_machine_lst": [int(x) for x in ms],
            })
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances",
                    default=str(V5 / "instance" / "test_Instance_J6M6E2.pkl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for upstream's random MOR tie-breaking")
    ap.add_argument("--idle_modes", default="trailing,gap")
    ap.add_argument("--pairs", default="",
                    help="optional comma list, e.g. 'fifo+spt,lwkr_t+spt'")
    ap.add_argument("--out", default="")
    ap.add_argument("--dump-orders", action="store_true")
    ap.add_argument("--order-index", type=int, default=0)
    args = ap.parse_args()

    if args.dump_orders:
        _dump_orders(args)
        return

    t_all, p_all, tt_all, edge_all = load_pool(args.instances)
    n = len(t_all) if args.limit <= 0 else min(args.limit, len(t_all))
    modes = [m.strip() for m in args.idle_modes.split(",") if m.strip()]
    wanted = {x.strip() for x in args.pairs.split(",") if x.strip()}

    jobs = []
    for i in range(n):
        for o_name, o_rule in O_RULES:
            for m_name, m_rule in M_RULES:
                if wanted and f"{o_name}+{m_name}" not in wanted:
                    continue
                for mode in modes:
                    jobs.append({
                        "instance": i,
                        "t": t_all[i], "p": p_all[i],
                        "tt": tt_all[i], "edge": edge_all[i],
                        "job_rule": o_name, "o_rule": o_rule,
                        "machine_rule": m_name, "m_rule": m_rule,
                        "idle_mode": mode, "seed": args.seed,
                    })

    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            records = list(ex.map(run_one, jobs))
    else:
        records = [run_one(j) for j in jobs]
    wall = time.time() - t0

    pairs = _summarize(records)
    _print_table(pairs)

    if args.out:
        payload = {
            "instances": args.instances,
            "n_instances": n,
            "seed": args.seed,
            "idle_modes": modes,
            "objective": "0.4*mk + 0.4*(pt/n_task + idleT) + 0.2*transT",
            "published_source": "Wang et al. Table V (6x6)",
            "wall_s": wall,
            "summary": pairs,
            "records": records,
        }
        Path(args.out).write_text(json.dumps(payload, indent=1),
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
