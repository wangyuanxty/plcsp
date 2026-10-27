# -*- coding: utf-8 -*-
"""AGV-aware PDR and RGA baselines for the v5 MT-FJSP.

The decoder mirrors the v5 Stage-0 transport accounting:

* same workshop: earliest-free source AGV, deadhead + loaded leg;
* cross workshop: source AGV + fixed delay + target AGV, boundary is the
  first machine of the target workshop;
* makespan / processing energy / loaded transport time / machine idle
  energy use the same definitions as the v5 ``_weighted_cost``.

Objective (v5 reporting weights)::

    0.4 * mk + 0.4 * (pt / n_task + idleT) + 0.2 * transT

where ``idleT`` is machine idle *energy* (idle power is 1.0 in v5) and
``transT`` is loaded travel time.

Both baselines support ``--agv 0`` (Table A code check) and ``--agv 1``
(Table B, the paper's AGV table).
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

N_AGV = 6
CROSS_DELAY = 5.0
LEG_RATIO = 0.5
W_MK = 0.4
W_EC = 0.4
W_TT = 0.2


def edge_of(edge_info, m):
    for e, members in enumerate(edge_info):
        if int(m) in [int(x) for x in members]:
            return e
    return 0


def split_cross(total, delay=CROSS_DELAY, ratio=LEG_RATIO):
    total = max(0.0, float(total))
    d = min(max(0.0, float(delay)), total)
    rem = total - d
    src = rem * min(max(float(ratio), 0.0), 1.0)
    return src, d, rem - src


def decode(t, p, tt, edge_info, os_seq, ms, agv=True, n_agv=N_AGV):
    """Decode an operation sequence + machine assignment into v5 costs."""
    n_task, n_machine = t.shape
    n_job = n_task // n_machine
    n_ops = n_machine
    n_edge = len(edge_info)

    def travel(a, b):
        if a < 0 or b < 0:
            return 0.0
        return float(tt[int(a)][int(b)])

    agvs = []
    if agv:
        for k in range(n_agv):
            e = k % n_edge
            agvs.append([e, int(edge_info[e][0]), 0.0])

    op_count = [0] * n_job
    finish = [0.0] * n_task
    start = [0.0] * n_task
    routes = {m: [] for m in range(n_machine)}
    proc_energy = 0.0
    trans_total = 0.0

    for j in os_seq:
        j = int(j)
        oi = op_count[j]
        task = j * n_ops + oi
        m = int(ms[task])
        dur = float(t[task][m])
        if dur < 0:
            raise ValueError(f"task {task} cannot use machine {m}")
        op_count[j] += 1

        if oi == 0:
            ready = 0.0
        else:
            prev_task = task - 1
            prev_m = int(ms[prev_task])
            prev_ft = finish[prev_task]
            if prev_m == m:
                ready = prev_ft
            elif not agv:
                tr = travel(prev_m, m)
                trans_total += tr
                ready = prev_ft + tr
            else:
                pe = edge_of(edge_info, prev_m)
                de = edge_of(edge_info, m)
                src_idx = [k for k, a in enumerate(agvs) if a[0] == pe]
                if not src_idx:
                    raise ValueError(f"workshop {pe} has no AGV")
                src = min(src_idx, key=lambda k: (agvs[k][2], k))
                _, sloc, sfree = agvs[src]
                start_src = max(prev_ft, sfree)
                dead_src = travel(sloc, prev_m)
                total = travel(prev_m, m)
                if pe == de:
                    loaded = total
                    finish_src = start_src + dead_src + loaded
                    agvs[src][2] = finish_src
                    agvs[src][1] = m
                    trans_total += loaded
                    ready = finish_src
                else:
                    t_src, delay, t_dst = split_cross(total)
                    src_arrival = start_src + dead_src + t_src
                    boundary = src_arrival + delay
                    boundary_m = int(edge_info[de][0])
                    dst_idx = [k for k, a in enumerate(agvs) if a[0] == de]
                    if not dst_idx:
                        raise ValueError(f"workshop {de} has no AGV")
                    dst = min(dst_idx, key=lambda k: (agvs[k][2], k))
                    _, dloc, dfree = agvs[dst]
                    start_dst = max(boundary, dfree)
                    dead_dst = travel(dloc, boundary_m)
                    finish_dst = start_dst + dead_dst + t_dst
                    agvs[src][2] = src_arrival
                    agvs[src][1] = -1
                    agvs[dst][2] = finish_dst
                    agvs[dst][1] = m
                    trans_total += t_src + t_dst
                    ready = finish_dst

        route = routes[m]
        pos = len(route)
        for k in range(len(route) + 1):
            prev_ft = route[k - 1][1] if k > 0 else 0.0
            next_st = route[k][0] if k < len(route) else math.inf
            st = max(ready, prev_ft)
            if st + dur <= next_st + 1e-9:
                pos = k
                break
        route.insert(pos, (st, st + dur, task))
        start[task] = st
        finish[task] = st + dur
        proc_energy += float(t[task][m]) * float(p[task][m])

    mk = max(finish)
    idle = 0.0
    for m, route in routes.items():
        if not route:
            continue
        idle += route[0][0]
        for k in range(len(route) - 1):
            idle += route[k + 1][0] - route[k][1]
    obj = W_MK * mk + W_EC * (proc_energy / n_task + idle) + W_TT * trans_total
    return {"mk": mk, "pt": proc_energy, "transT": trans_total,
            "idleT": idle, "obj": obj,
            "start": start, "finish": finish}


def _remaining(t, p, j, oi, n_ops, n_machine):
    rw_t = 0.0
    rw_pt = 0.0
    for o in range(oi, n_ops):
        task = j * n_ops + o
        avail = np.where(t[task] >= 0)[0]
        rw_t += float(np.mean(t[task][avail]))
        rw_pt += float(np.mean(t[task][avail] * p[task][avail]))
    return rw_t, rw_pt


def pdr_schedule(t, p, job_rule, machine_rule):
    """Greedy PDR: choose a ready job by a job rule, then a machine by rule."""
    n_task, n_machine = t.shape
    n_job = n_task // n_machine
    n_ops = n_machine
    op_count = [0] * n_job
    os_seq = []
    ms = np.zeros(n_task, dtype=int)
    while len(os_seq) < n_task:
        best = None
        best_val = None
        for j in range(n_job):
            oi = op_count[j]
            if oi >= n_ops:
                continue
            task = j * n_ops + oi
            rw_t, rw_pt = _remaining(t, p, j, oi, n_ops, n_machine)
            if job_rule == "fifo":
                val = j * n_ops + oi
                better = best_val is None or val < best_val
            elif job_rule == "mor":
                val = n_ops - oi
                better = best_val is None or val > best_val
            elif job_rule == "lwkr_t":
                val = rw_t
                better = best_val is None or val < best_val
            elif job_rule == "lwkr_pt":
                val = rw_pt
                better = best_val is None or val < best_val
            elif job_rule == "mwkr_t":
                val = rw_t
                better = best_val is None or val > best_val
            elif job_rule == "mwkr_pt":
                val = rw_pt
                better = best_val is None or val > best_val
            else:
                raise ValueError(job_rule)
            if better:
                best = (j, task)
                best_val = val
        j, task = best
        avail = np.where(t[task] >= 0)[0]
        if machine_rule == "spt":
            m = int(avail[np.argmin(t[task][avail])])
        elif machine_rule == "seyc":
            m = int(avail[np.argmin(t[task][avail] * p[task][avail])])
        else:
            raise ValueError(machine_rule)
        ms[task] = m
        os_seq.append(j)
        op_count[j] += 1
    return os_seq, ms


def ra_schedule(t, rng):
    n_task, n_machine = t.shape
    n_job = n_task // n_machine
    n_ops = n_machine
    os_seq = [j for j in range(n_job) for _ in range(n_ops)]
    rng.shuffle(os_seq)
    ms = np.zeros(n_task, dtype=int)
    for task in range(n_task):
        avail = np.where(t[task] >= 0)[0]
        ms[task] = int(rng.choice(avail))
    return os_seq, ms


def _random_ms(t, rng):
    n_task, n_machine = t.shape
    ms = np.zeros(n_task, dtype=int)
    for task in range(n_task):
        avail = np.where(t[task] >= 0)[0]
        ms[task] = int(rng.choice(avail))
    return ms


def _pox(a, b, rng):
    n_job = max(a) + 1
    cut = set(rng.sample(range(n_job), rng.randint(1, max(1, n_job - 1))))
    child = [None] * len(a)
    for i, j in enumerate(a):
        if j in cut:
            child[i] = j
    fill = [j for j in b if j not in cut]
    k = 0
    for i in range(len(child)):
        if child[i] is None:
            child[i] = fill[k]
            k += 1
    return child


def rga_instance(t, p, tt, edge_info, seed=1, pop=50, gen=100, agv=True,
                 return_schedule=False):
    rng = random.Random(seed)
    n_task, n_machine = t.shape
    n_job = n_task // n_machine
    n_ops = n_machine
    base_os = [j for j in range(n_job) for _ in range(n_ops)]

    def cost(os_seq, ms):
        return decode(t, p, tt, edge_info, os_seq, ms, agv=agv)["obj"]

    population = []
    for _ in range(pop):
        os_seq = list(base_os)
        rng.shuffle(os_seq)
        ms = _random_ms(t, rng)
        population.append((os_seq, ms, cost(os_seq, ms)))
    population.sort(key=lambda x: x[2])
    n_eval = pop
    for _ in range(gen):
        new = population[:2]
        while len(new) < pop:
            p1 = min(rng.sample(population, 2), key=lambda x: x[2])
            p2 = min(rng.sample(population, 2), key=lambda x: x[2])
            c_os = _pox(p1[0], p2[0], rng)
            c_ms = p1[1].copy()
            for task in range(n_task):
                if rng.random() < 0.5:
                    c_ms[task] = p2[1][task]
            if rng.random() < 0.8:
                i, j = rng.sample(range(len(c_os)), 2)
                c_os[i], c_os[j] = c_os[j], c_os[i]
            if rng.random() < 0.8:
                task = rng.randrange(n_task)
                avail = np.where(t[task] >= 0)[0]
                c_ms[task] = int(rng.choice(avail))
            new.append((c_os, c_ms, cost(c_os, c_ms)))
            n_eval += 1
        new.sort(key=lambda x: x[2])
        population = new
    best = population[0]
    out = decode(t, p, tt, edge_info, best[0], best[1], agv=agv)
    result = {"obj": out["obj"], "mk": out["mk"], "pt": out["pt"],
              "transT": out["transT"], "idleT": out["idleT"],
              "n_eval": n_eval}
    if return_schedule:
        result["os_seq"] = [int(j) for j in best[0]]
        result["ms"] = [int(m) for m in best[1]]
    return result


def _stat(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)),
            "min": float(a.min()), "max": float(a.max())}


def _cost_fields(out):
    return {k: float(out[k]) for k in ("mk", "pt", "transT", "idleT", "obj")}


def _worker(args):
    kind, data, params = args
    t, p, tt, edge_info = data
    if kind == "pdr":
        inst, jr, mr, agv = params
        os_seq, ms = pdr_schedule(t, p, jr, mr)
        out = decode(t, p, tt, edge_info, os_seq, ms, agv=agv)
        return {"instance": inst, "job_rule": jr, "machine_rule": mr,
                **_cost_fields(out)}
    if kind == "ra":
        inst, seed, agv = params
        rng = random.Random(seed)
        os_seq, ms = ra_schedule(t, rng)
        out = decode(t, p, tt, edge_info, os_seq, ms, agv=agv)
        return {"instance": inst, "seed": seed, **_cost_fields(out)}
    if kind == "rga":
        p_params = dict(params)
        inst = p_params.pop("instance")
        seed = int(p_params["seed"])
        out = rga_instance(t, p, tt, edge_info, **p_params)
        return {"instance": inst, "seed": seed, **_cost_fields(out),
                "n_eval": int(out["n_eval"])}
    raise ValueError(kind)


def _pdr_summary(records, wall):
    keys = sorted({(r["job_rule"], r["machine_rule"]) for r in records})
    per_rule = []
    for jr, mr in keys:
        vals = [r["obj"] for r in records
                if r["job_rule"] == jr and r["machine_rule"] == mr]
        per_rule.append({"job_rule": jr, "machine_rule": mr, **_stat(vals)})
    per_rule.sort(key=lambda x: x["mean"])
    return {"overall": _stat([r["obj"] for r in records]),
            "per_rule": per_rule, "wall_s": wall}


def _rga_summary(records, n_instances, seeds, wall):
    per_seed = []
    for s in range(1, seeds + 1):
        vals = [r["obj"] for r in records if r["seed"] == s]
        per_seed.append({"seed": s, **_stat(vals)})
    inst_std = []
    for i in range(n_instances):
        vals = [r["obj"] for r in records if r["instance"] == i]
        if len(vals) > 1:
            inst_std.append(float(np.std(vals, ddof=1)))
    return {"overall": _stat([r["obj"] for r in records]),
            "per_seed": per_seed,
            "instance_std_mean": (float(np.mean(inst_std))
                                  if inst_std else 0.0),
            "eval_mean": float(np.mean([r["n_eval"] for r in records])),
            "wall_s": wall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="plcsp_v5/instance/test_Instance_J6M6E2.pkl")
    ap.add_argument("--mode", choices=["pdr", "ra", "rga"], default="pdr")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--gen", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--agv", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data = pickle.load(open(args.instances, "rb"))
    t_all, p_all, tt_all, edge_all = data
    n = len(t_all) if args.limit <= 0 else min(args.limit, len(t_all))
    agv = bool(args.agv)
    jobs = []

    if args.mode == "pdr":
        jr = ["fifo", "mor", "lwkr_t", "lwkr_pt", "mwkr_t", "mwkr_pt"]
        mr = ["spt", "seyc"]
        for i in range(n):
            inst = (t_all[i], p_all[i], tt_all[i], edge_all[i])
            for a in jr:
                for b in mr:
                    jobs.append(("pdr", inst, (i, a, b, agv)))
    elif args.mode == "ra":
        for i in range(n):
            inst = (t_all[i], p_all[i], tt_all[i], edge_all[i])
            jobs.append(("ra", inst, (i, i, agv)))
    else:
        for i in range(n):
            for s in range(args.seeds):
                inst = (t_all[i], p_all[i], tt_all[i], edge_all[i])
                jobs.append(("rga", inst,
                             dict(instance=i, seed=s + 1, pop=args.pop,
                                  gen=args.gen, agv=agv)))

    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            records = list(ex.map(_worker, jobs))
    else:
        records = [_worker(j) for j in jobs]
    wall = time.time() - t0

    if args.mode == "rga":
        summary = _rga_summary(records, n, args.seeds, wall)
    elif args.mode == "pdr":
        summary = _pdr_summary(records, wall)
    else:
        summary = {"overall": _stat([r["obj"] for r in records]),
                   "wall_s": wall}

    payload = {
        "mode": args.mode,
        "agv": agv,
        "instances": args.instances,
        "n_instances": n,
        "n_runs": len(records),
        "seeds": args.seeds if args.mode == "rga" else 1,
        "pop": args.pop if args.mode == "rga" else None,
        "gen": args.gen if args.mode == "rga" else None,
        "summary": summary,
        "records": records,
    }
    print(json.dumps({"mode": args.mode, "agv": agv,
                      "n_instances": n, "n_runs": len(records),
                      "summary": summary}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=1),
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
