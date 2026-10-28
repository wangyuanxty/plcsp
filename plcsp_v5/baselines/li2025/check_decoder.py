# -*- coding: utf-8 -*-
"""Verify the adapted decoder against the v5 decoder and the v5 environment.

The adapted Li et al. baseline must not change the v5 cost accounting.  This
script runs the same PDR schedule through three implementations:

1. ``agv_baselines.decode`` (the reference standalone decoder);
2. ``li2025.env_model.ScheduleSession`` (this baseline's decoder);
3. ``check_agv_env.run_env`` (the v5 ``Parallel_env`` replay), optional.

Run from the repository root::

    D:/anaconda/python.exe plcsp_v5/baselines/li2025/check_decoder.py --limit 3
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plcsp_v5"))
sys.path.insert(0, str(ROOT / "plcsp_v5" / "baselines"))

from agv_baselines import decode, pdr_schedule  # noqa: E402
from li2025.env_model import Instance, ScheduleSession  # noqa: E402

KEYS = ("mk", "pt", "transT", "idleT", "obj")


def _run_ours(t, p, tt, edge, os_seq, ms):
    inst = Instance.from_arrays(t, p, tt, edge)
    sess = ScheduleSession(inst)
    for j in os_seq:
        task = sess.next_task(int(j))
        sess.apply(int(j), int(ms[task]))
    return sess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="plcsp_v5/instance/"
                                        "test_Instance_J6M6E2.pkl")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--env-check", type=int, default=0)
    args = ap.parse_args()

    data = pickle.load(open(args.instances, "rb"))
    t_all, p_all, tt_all, edge_all = data
    rules = (("fifo", "spt"), ("mor", "seyc"), ("lwkr_pt", "spt"))
    worst_cost = 0.0
    worst_time = 0.0
    for i in range(min(args.limit, len(t_all))):
        t, p = t_all[i], p_all[i]
        tt, edge = tt_all[i], edge_all[i]
        for job_rule, machine_rule in rules:
            os_seq, ms = pdr_schedule(t, p, job_rule, machine_rule)
            ref = decode(t, p, tt, edge, os_seq, ms, agv=True)
            sess = _run_ours(t, p, tt, edge, os_seq, ms)
            ours = sess.finalize()
            diffs = {k: abs(float(ours[k]) - float(ref[k])) for k in KEYS}
            worst_cost = max(worst_cost, max(diffs.values()))
            for task in range(len(os_seq)):
                worst_time = max(
                    worst_time,
                    abs(float(sess.start[task]) - float(ref["start"][task])),
                    abs(float(sess.finish[task]) - float(ref["finish"][task])),
                )
            print(f"inst={i} {job_rule}+{machine_rule} "
                  f"obj ours={ours['obj']:.6f} ref={ref['obj']:.6f} "
                  f"max_cost_diff={max(diffs.values()):.3e}")

        if args.env_check:
            from check_agv_env import run_env
            os_seq, ms = pdr_schedule(t, p, "fifo", "spt")
            env = run_env(t, p, tt, edge, os_seq, ms, use_agv=1)
            sess = _run_ours(t, p, tt, edge, os_seq, ms)
            env_cost = {k: abs(float(env[k]) - float(sess.finalize()[k]))
                        for k in KEYS}
            # The environment stores float32 tensors; the existing
            # check_agv_env.py accepts a 1e-3 tolerance for the same reason.
            print(f"inst={i} env check max_cost_diff="
                  f"{max(env_cost.values()):.3e}")
            worst_cost = max(worst_cost, max(env_cost.values()))

    print(f"worst cost diff={worst_cost:.3e} worst time diff={worst_time:.3e}")
    tolerance = 1e-3 if args.env_check else 1e-6
    if worst_cost > tolerance or worst_time > 1e-6:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
