# -*- coding: utf-8 -*-
"""Check the standalone v5 decoder against the v5 environment.

The check fixes one PDR schedule (operation sequence + machine assignment),
runs the same schedule through ``Parallel_env`` with the earliest-free AGV
rule, and compares the four cost components.

Run from the repo root::

    D:/anaconda/python.exe plcsp_v5/baselines/check_agv_env.py --limit 3
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plcsp_v5"))
sys.path.insert(0, str(ROOT / "plcsp_v5" / "graph-jsp-env" / "src"))


def _install_gym_shim():
    """The CPU interpreter has no gym; the env only needs the base classes."""
    try:
        import gym  # noqa: F401
        return
    except ImportError:
        pass
    import types

    gym = types.ModuleType("gym")

    class Env:
        def __init__(self, *args, **kwargs):
            pass

    class Discrete:
        def __init__(self, n, *args, **kwargs):
            self.n = n

    class Box:
        def __init__(self, low=0, high=1, shape=None, dtype=None,
                     *args, **kwargs):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class Dict(dict):
        pass

    spaces = types.ModuleType("gym.spaces")
    spaces.Discrete = Discrete
    spaces.Box = Box
    spaces.Dict = Dict
    gym.Env = Env
    gym.spaces = spaces
    sys.modules["gym"] = gym
    sys.modules["gym.spaces"] = spaces


_install_gym_shim()

import matplotlib  # noqa: E402

if not hasattr(matplotlib.cm, "get_cmap"):
    matplotlib.cm.get_cmap = matplotlib.colormaps.get_cmap

from baselines.agv_baselines import decode, pdr_schedule  # noqa: E402
from trainer.parallel_env import Parallel_env  # noqa: E402

W = (0.4, 0.4, 0.2)


def _variant(t, n_edge, use_agv, n_agv=6):
    n_task, n_machine = t.shape
    return {
        "n_job": n_task // n_machine,
        "n_machine": n_machine,
        "n_edge": n_edge,
        "env_batch": 1,
        "m_scaling": 1,
        "reward_scaling": {"scaling_divisor": 1},
        "random_weight_type": "eval",
        "use_agv": int(use_agv),
        "n_agv": n_agv,
        "cross_delay": 5.0,
        "cross_leg_ratio": 0.5,
        "agv_failure": 0,
        "agv_mtbf": 100.0,
        "agv_mttr": 10.0,
        "failure_seed": 0,
        "machine_failure": 0,
        "machine_mtbf": 100.0,
        "machine_mttr": 10.0,
        "GAMMA": 0.99,
        "weight_mk": W[0],
        "weight_ec": W[1],
        "weight_tt": W[2],
    }


def _actions(os_seq, ms, n_ops):
    op_count = {}
    actions = []
    for j in os_seq:
        oi = op_count.get(int(j), 0)
        task = int(j) * n_ops + oi
        op_count[int(j)] = oi + 1
        actions.append([task, int(ms[task])])
    return actions


def run_env(t, p, tt, edge, os_seq, ms, use_agv, n_agv=6):
    variant = _variant(t, len(edge), use_agv, n_agv)
    pe = Parallel_env(args=variant)
    pe.get_batch({
        "t": torch.from_numpy(t[None, ...]).float(),
        "p": torch.from_numpy(p[None, ...]).float(),
        "transT": torch.from_numpy(tt[None, ...]).float(),
        "edge": torch.from_numpy(edge[None, ...]).float(),
    })
    pe.init_RewardScaling_sameBATCH(shape=4)
    pe.init_DGFJSPEnv_state0()

    n_task, n_machine = t.shape
    actions = _actions(os_seq, ms, n_machine)
    for action in actions:
        pe.DGFJSPEnv_paral_step([action])

    env0 = pe.paral_env_DG[0]
    mk = float(env0.makespan_previous_step)
    pt = float(env0.total_e1_previous_step)
    trans = float(env0.trans_t_previous_step)
    idle = float(env0.idle_t_previous_step)
    obj = W[0] * mk + W[1] * (pt / n_task + idle) + W[2] * trans
    starts = [float(env0.G.nodes[i + 1]["start_time"])
              for i in range(n_task)]
    finishes = [float(env0.G.nodes[i + 1]["finish_time"])
                for i in range(n_task)]
    return {"mk": mk, "pt": pt, "transT": trans, "idleT": idle,
            "obj": obj, "starts": starts, "finishes": finishes}


def _trace_env_timing():
    import agv.state as agv_state
    orig = agv_state.transport_timing

    def wrapped(src, pickup, dropoff, ready, total, travel, **kw):
        dst = kw.get("dst")
        out = orig(src, pickup, dropoff, ready, total, travel, **kw)
        print("ENV_AGV", {
            "pickup": int(pickup), "dropoff": int(dropoff),
            "ready": round(float(ready), 4),
            "total": round(float(total), 4),
            "cross": dst is not None,
            "src_loc0": int(src.location),
            "src_free0": round(float(src.free_time), 4),
            "dst_loc0": int(dst.location) if dst is not None else None,
            "dst_free0": (round(float(dst.free_time), 4)
                          if dst is not None else None),
            "wait": round(float(out["wait"]), 4),
            "loaded": round(float(out["loaded"]), 4),
            "src_free": round(float(out["src_free"]), 4),
            "dst_free": (round(float(out["dst_free"]), 4)
                         if out["dst_free"] is not None else None),
        })
        return out

    agv_state.transport_timing = wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances",
                    default="plcsp_v5/instance/test_Instance_J6M6E2.pkl")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--job-rule", default="mor")
    ap.add_argument("--machine-rule", default="spt")
    ap.add_argument("--n-agv", type=int, default=6)
    ap.add_argument("--debug", type=int, default=0)
    args = ap.parse_args()

    if args.debug:
        _trace_env_timing()

    data = pickle.load(open(args.instances, "rb"))
    t_all, p_all, tt_all, edge_all = data
    n = min(args.limit, len(t_all))
    keys = ("mk", "pt", "transT", "idleT", "obj")
    worst = {k: 0.0 for k in keys}

    for use_agv in (0, 1):
        print(f"=== use_agv={use_agv} ===")
        for i in range(n):
            t, p = t_all[i], p_all[i]
            tt, edge = tt_all[i], edge_all[i]
            os_seq, ms = pdr_schedule(t, p, args.job_rule,
                                      args.machine_rule)
            dec = decode(t, p, tt, edge, os_seq, ms, agv=bool(use_agv),
                         n_agv=args.n_agv)
            env = run_env(t, p, tt, edge, os_seq, ms, use_agv,
                          args.n_agv)
            diffs = {k: abs(dec[k] - env[k]) for k in keys}
            for k in keys:
                worst[k] = max(worst[k], diffs[k])
            print(f"inst={i} dec={ {k: round(dec[k], 4) for k in keys} }")
            print(f"inst={i} env={ {k: round(env[k], 4) for k in keys} }")
            print(f"inst={i} abs_diff={ {k: round(diffs[k], 6) for k in keys} }")
            if args.debug and use_agv:
                for task in range(len(env["starts"])):
                    d = abs(dec["start"][task] - env["starts"][task])
                    if d > 1e-3:
                        print(f"START_DIFF task={task} "
                              f"dec={dec['start'][task]:.4f} "
                              f"env={env['starts'][task]:.4f} "
                              f"d={d:.4f}")

    print("=== worst abs diff ===")
    print({k: round(v, 8) for k, v in worst.items()})
    if max(worst.values()) > 1e-3:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
