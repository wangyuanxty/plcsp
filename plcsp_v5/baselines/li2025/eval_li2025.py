# -*- coding: utf-8 -*-
"""Adapted baseline: Li, Li & Gao, Advanced Engineering Informatics 65 (2025).

Paper
-----
Yuxin Li, Xinyu Li, Liang Gao. "Real-time scheduling for production-logistics
collaborative environment using multi-agent deep reinforcement learning."
Advanced Engineering Informatics, 65 (2025) 103216.
DOI: 10.1016/j.aei.2025.103216.

The authors released no code (searched GitHub and the web on 2026-10-27; the
paper has no repository link and no code-availability statement).  This module
is therefore written from the paper.  It is an *adapted baseline*, never a
reproduction.

What the paper does
-------------------
JSP with fixed machines, dynamic new-job arrivals, one global AGV fleet, and
total weighted tardiness.  Three agents (job filter, job selection, AGV
selection) output tiny rule-weight/proto-action vectors; hand-written PDR
priority vectors make the actual choice.  Machines use the FUFS rule.  MAPPO
with three actors, three critics, GAE, PopArt, clipped PPO and a clipped Huber
critic trains the agents for 1500 episodes.

What this baseline does
-----------------------
The v5 shop: flexible job shop, two workshops, per-workshop AGV fleets,
static instances, and the v5 energy-aware objective::

    0.4 * makespan + 0.4 * (processing_energy / n_task + machine_idle_energy)
    + 0.2 * loaded_transport_time

Every deviation is listed below.  The decoder is verified bit-exact against
``plcsp_v5/baselines/agv_baselines.decode`` and within float32 tolerance of
the v5 ``Parallel_env`` (see ``check_decoder.py``).

Adaptations (explicit)
----------------------
1. Problem: fixed-machine JSP -> FJSP with machine flexibility, two workshops,
   per-workshop AGV fleets, static instances, energy-aware objective.
2. Added machine selection: the paper has no machine choice.  A fourth
   rule-weight agent selects SPT / SEYEC / EFT.  The paper's FUFS machine
   running logic is not implemented; the serial insertion decoder determines
   the processing order.
3. Objective: TWT -> the v5 weighted objective above.  The paper's reported
   numbers are not comparable and are not reproduced.
4. ETC and due-date rules: v5 instances have no due dates and no job weights.
   ``ETC`` is redefined as the estimated remaining v5 objective contribution.
   The paper's CR and EDD rules are replaced by MEC (remaining processing
   energy) and MTT (remaining transport).  No due date is synthesised.
5. Reward: potential-based, ``r_t = (C_t - C_{t+1}) / max(1, |C_0|)``, shared
   by all agents.  ``C_t`` is the estimated remaining objective and equals the
   realised objective at the terminal state, so the episode return is
   ``(C_0 - objective) / scale``, the analogue of the paper's Eq. (18).
6. Static vs dynamic arrivals: no new-job arrivals are simulated.  The
   paper's logistics-task release moment is mapped to serial schedule
   generation: one operation per job is eligible at a time, so each job has
   at most one pending logistics task.
7. Rule-weight action space: job filter 4 weights (sigmoid) + SEAT
   (omega=50); job selection 4-dimensional proto-action (tanh) with Euclidean
   distance + SEAT; AGV 2 raw weights + SCPT (omega=5); machine 3 weights
   (sigmoid).  The job-selection distance is negated: the paper writes
   ``Norm(||a - attr||)`` and then "highest priority", which is only
   consistent if the distance is negated.
8. AGV model: an AGV never crosses a workshop; cross-workshop trips are a
   source leg + fixed delay + target leg, with deadhead time.  The destination
   AGV is chosen deterministically by SCT at the boundary; the learned agent
   chooses the source AGV.  ``CTA`` is ``max(AGV free time, job ready time)``.
9. MAPPO: four actors + four critics (the paper has three of each), FC
   128-64-64, centralised critic input ``concat(o_jf, Gaussian noise)`` as in
   Eq. (20), GAE (gamma=0.99, lambda=0.95), clip 0.2, entropy 0.01, clipped
   Huber critic.  PopArt is approximated by per-batch advantage
   normalisation.  The Gaussian is diagonal; the paper does not specify the
   covariance.
10. Training budget: 100 episodes on the v5 training pool (paper: 1500).
    Update buffer 512 (paper: 6400), minibatch 128 (paper: 512), 4 epochs
    (paper: 8), lr 1e-3 (paper: 1e-4 decaying to 1e-5).  These reductions are
    forced by the requested budget.  The actors are also rule-initialised
    (job-selection bias at the MWKR corner, machine bias at SPT); the paper
    starts from random weights.  ``--warm-start 0`` gives the cold start.
11. The paper's 20-machine / 10-AGV scale, 120-instance dynamic protocol, real
    case study, and machine FUFS logic are not reproduced.

CLI
---
Run from the repository root on the CPU interpreter::

    D:/anaconda/python.exe plcsp_v5/baselines/li2025/eval_li2025.py \
        --n-job 6 --n-machine 6 --n-edge 2 --limit 3 \
        --train-episodes 100 --out results/li2025_J6M6E2.json

The script trains a small policy on the matching v5 training pool (the 6x6
pickle, or freshly generated instances for other sizes), caches the
checkpoint under ``checkpoints/``, and evaluates the deterministic policy on
the requested test pickle.  ``--reuse-ckpt 1`` (default) loads an existing
checkpoint and skips training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plcsp_v5"))
sys.path.insert(0, str(ROOT / "plcsp_v5" / "baselines"))

from li2025.features import state_dims  # noqa: E402
from li2025.policy import (PolicyBundle, PPOConfig, load_policy,  # noqa: E402
                           save_policy)
from li2025.training import (INSTANCE_DIR, evaluate,  # noqa: E402
                             load_pickle_instances,
                             resolve_train_instances, train)

ACT_DIMS = {"jf": 4, "js": 4, "ms": 3, "as": 2}
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

ADAPTATIONS = [
    "Problem: fixed-machine JSP -> v5 FJSP with two workshops, per-workshop "
    "AGV fleets, static instances, and an energy-aware objective.",
    "Added machine selection: a fourth rule-weight agent over SPT/SEYEC/EFT. "
    "The paper's FUFS machine running logic is not implemented.",
    "Objective: TWT -> 0.4*makespan + 0.4*(processing_energy/n_task + "
    "machine_idle_energy) + 0.2*loaded_transport_time.",
    "ETC is redefined as the estimated remaining v5 objective contribution. "
    "The paper's CR/EDD rules are replaced by MEC/MTT because v5 instances "
    "have no due dates or job weights; no due date is synthesised.",
    "Potential-based shared reward r_t = (C_t - C_{t+1}) / max(1, |C_0|); "
    "the terminal potential is the realised objective (analogue of Eq. 18).",
    "Static arrivals only. The paper's release moment maps to serial "
    "schedule generation: one pending logistics task per job at a time.",
    "Rule-weight actions: job filter 4 sigmoid weights + SEAT(50); job "
    "selection 4-D tanh proto-action + Euclidean distance + SEAT(50); AGV 2 "
    "raw weights + SCPT(5); machine 3 sigmoid weights.",
    "The job-selection distance is negated: the paper's 'highest priority' "
    "wording conflicts with a distance that is smaller-is-better.",
    "AGV model: per-workshop fleets, deadhead, source leg + fixed delay + "
    "target leg. The destination AGV is deterministic SCT; the learned agent "
    "chooses the source AGV. CTA = max(AGV free time, job ready time).",
    "MAPPO: four actors + four critics, FC 128-64-64, centralised critic "
    "concat(o_jf, noise), GAE, PPO clip 0.2, entropy 0.01, clipped Huber. "
    "PopArt is approximated by per-batch advantage normalisation.",
    "Training budget: 100 episodes (paper 1500), buffer 512 (paper 6400), "
    "minibatch 128 (paper 512), 4 epochs (paper 8), lr 1e-3 (paper 1e-4).",
    "Rule-informed initialisation: job-selection bias at the MWKR corner and "
    "machine bias at SPT. The paper starts from random weights; "
    "--warm-start 0 gives the cold start.",
    "The paper's 20-machine/10-AGV scale, 120-instance dynamic protocol, real "
    "case study, and machine FUFS logic are not reproduced.",
]

REMAINING_WORK = [
    "Dynamic new-job arrivals and the paper's event-driven release moment.",
    "Due dates and job weights to restore CR/EDD/ETC exactly.",
    "A 1500-episode training budget or a validation split for hyperparameter "
    "search; full PopArt.",
    "The paper's 20-machine/10-AGV scale and 120-instance protocol.",
    "A distributed pretrained checkpoint (the authors released none).",
    "Machine FUFS logic and the paper's real case study.",
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Adapted Li et al. (AEI 2025) baseline for v5.")
    ap.add_argument("--n-job", type=int, default=6)
    ap.add_argument("--n-machine", type=int, default=6)
    ap.add_argument("--n-edge", type=int, default=2)
    ap.add_argument("--n-agv", type=int, default=6)
    ap.add_argument("--lu-station", type=int, default=0,
                    help="1 = M2 transport model with load/unload stations")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--instances", default="")
    ap.add_argument("--train-instances", default="")
    ap.add_argument("--train-episodes", type=int, default=100)
    ap.add_argument("--train-pool", type=int, default=512)
    ap.add_argument("--train-seed", type=int, default=17)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--reuse-ckpt", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--update-every", type=int, default=512)
    ap.add_argument("--noise-dim", type=int, default=10)
    ap.add_argument("--noise-sigma", type=float, default=0.1)
    ap.add_argument("--warm-start", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    return ap.parse_args()


def _check_meta(meta: dict, args: argparse.Namespace,
                obs_dims: dict[str, int]) -> None:
    expected = {
        "n_job": int(args.n_job), "n_machine": int(args.n_machine),
        "n_edge": int(args.n_edge), "n_agv": int(args.n_agv),
        "noise_dim": int(args.noise_dim),
    }
    for key, value in expected.items():
        if int(meta.get(key, -1)) != value:
            raise SystemExit(
                f"checkpoint {key}={meta.get(key)} does not match "
                f"requested {value}; use --reuse-ckpt 0 or another --ckpt")
    if {k: int(v) for k, v in meta["obs_dims"].items()} != obs_dims:
        raise SystemExit("checkpoint observation dimensions do not match")
    if {k: int(v) for k, v in meta["act_dims"].items()} != ACT_DIMS:
        raise SystemExit("checkpoint action dimensions do not match")
    if "warm_start" in meta and bool(meta["warm_start"]) != bool(
            args.warm_start):
        raise SystemExit(
            f"checkpoint warm_start={meta['warm_start']} does not match "
            f"requested {bool(args.warm_start)}; use --reuse-ckpt 0")
    if "lu_station" in meta and bool(meta["lu_station"]) != bool(
            args.lu_station):
        raise SystemExit(
            f"checkpoint lu_station={meta['lu_station']} does not match "
            f"requested {bool(args.lu_station)}; use --reuse-ckpt 0 or "
            f"another --ckpt")


def _m2_out_path(path: Path) -> Path:
    """Never overwrite an M1 result JSON with an M2 run."""
    if "m2" in path.name.lower():
        return path
    return path.with_name("m2_" + path.name)


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(max(1, int(args.threads)))
    n_job, n_machine, n_edge = args.n_job, args.n_machine, args.n_edge
    n_agv = args.n_agv
    lu_station = bool(args.lu_station)
    suffix = "_lu" if lu_station else ""
    obs_dims = state_dims(n_agv)
    test_path = (Path(args.instances) if args.instances else
                 INSTANCE_DIR / f"test_Instance_J{n_job}M{n_machine}"
                                f"E{n_edge}{suffix}.pkl")
    if not test_path.exists():
        raise SystemExit(f"test instances not found: {test_path}")
    test_instances = load_pickle_instances(test_path,
                                           lu_station=lu_station)

    ckpt_prefix = "m2_li2025" if lu_station else "li2025"
    ckpt_path = (Path(args.ckpt) if args.ckpt else
                 CHECKPOINT_DIR / f"{ckpt_prefix}_J{n_job}M{n_machine}"
                                  f"E{n_edge}{suffix}"
                                  f"_ep{args.train_episodes}.pt")
    training_info: dict
    if args.reuse_ckpt and ckpt_path.exists():
        policy, cfg, meta = load_policy(ckpt_path)
        _check_meta(meta, args, obs_dims)
        training_info = dict(meta.get("training", {}))
        training_info["loaded_from"] = str(ckpt_path)
        train_source = meta.get("train_source", "checkpoint")
        n_train = int(meta.get("n_train", 0))
        print(f"loaded checkpoint {ckpt_path}", flush=True)
    else:
        train_instances, train_source = resolve_train_instances(
            n_job, n_machine, n_edge, args.train_instances,
            args.train_pool, args.train_seed, lu_station=lu_station)
        n_train = len(train_instances)
        policy = PolicyBundle.create(obs_dims, ACT_DIMS,
                                     noise_dim=args.noise_dim,
                                     seed=args.seed,
                                     warm_start=bool(args.warm_start))
        cfg = PPOConfig(lr=args.lr, update_every=args.update_every,
                        noise_dim=args.noise_dim,
                        noise_sigma=args.noise_sigma)
        if args.train_episodes > 0:
            training_info = train(policy, train_instances, cfg,
                                  args.train_episodes, args.seed,
                                  log_every=args.log_every)
            training_info["train_source"] = train_source
            training_info["n_train_pool"] = n_train
            meta = {
                "n_job": n_job, "n_machine": n_machine, "n_edge": n_edge,
                "n_agv": n_agv, "noise_dim": args.noise_dim,
                "warm_start": bool(args.warm_start),
                "lu_station": lu_station,
                "obs_dims": obs_dims, "act_dims": ACT_DIMS,
                "training": training_info, "train_source": train_source,
                "n_train": n_train, "adaptations": ADAPTATIONS,
                "remaining_work": REMAINING_WORK,
            }
            save_policy(ckpt_path, policy, cfg, meta)
            print(f"saved checkpoint {ckpt_path}", flush=True)
        else:
            training_info = {"episodes": 0, "note": "untrained policy"}

    records, summary = evaluate(policy, test_instances, cfg, args.limit,
                                n_agv=n_agv)
    payload = {
        "mode": "Li2025-adapted",
        "source": ("Li, Li & Gao, Advanced Engineering Informatics 65 "
                   "(2025) 103216; adapted, not a reproduction"),
        "cpu_only": bool(not torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
        "n_job": n_job, "n_machine": n_machine, "n_edge": n_edge,
        "n_agv": n_agv,
        "lu_station": lu_station,
        "instances": str(test_path),
        "n": len(records),
        "objective": ("0.4*mk + 0.4*(pt/n_task + idleT) + 0.2*transT"),
        "train_source": train_source,
        "train_instances": n_train,
        "training": training_info,
        "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "ckpt": str(ckpt_path) if ckpt_path.exists() else "",
        "adaptations": ADAPTATIONS,
        "remaining_work": REMAINING_WORK,
        "summary": summary,
        "records": records,
    }
    print(json.dumps({
        "mode": payload["mode"], "cpu_only": payload["cpu_only"],
        "n": payload["n"], "train_source": payload["train_source"],
        "training": training_info, "summary": summary,
    }, indent=2, ensure_ascii=False))
    if args.out:
        out_path = Path(args.out)
        if lu_station:
            out_path = _m2_out_path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                            encoding="utf-8")
        print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
