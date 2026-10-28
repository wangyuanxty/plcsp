# -*- coding: utf-8 -*-
"""CLI, training loop, and evaluation for the adapted NHDRL baseline.

Run ``python plcsp_v5/baselines/nhdrl_baseline.py --help`` for the options.
The module docstring of ``plcsp_v5/baselines/nhdrl/__init__.py`` lists every
deviation from the source paper.  This file only adds the run mechanics.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pickle
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from .ppo import Agent, PPOConfig, RunningMinMax, compute_gae
from .sim import (
    AGV_RULES,
    EARLIEST,
    FEATURE_NAMES,
    N_LA,
    N_LA_ACTIONS,
    N_OA,
    N_OA_ACTIONS,
    N_PA,
    N_PA_ACTIONS,
    NestedSim,
)

SOURCE = (
    "J. Shi, F. Qiao, J. Liu, Y. Ma, D. Wang, C. Ding. "
    "Production-logistics collaborative scheduling in dynamic flexible "
    "job shops using nested-hierarchical deep reinforcement learning. "
    "Advanced Engineering Informatics 65 (2025) 103195. "
    "doi:10.1016/j.aei.2025.103195"
)

OFFICIAL_CODE = (
    "none found on 2026-10-27 (GitHub repository/code search and web search); "
    "the paper states that data are available on request"
)

DEVIATIONS = [
    "Adapted baseline, not a reproduction; no official code was found.",
    "Static v5 instances; no AGV breakdown or failure-triggered decisions.",
    "Two workshops and per-workshop AGV fleets; cross-workshop transport "
    "uses a source AGV and a destination AGV selected by the same LA rule.",
    "No L/U station and no return trip; first operations and same-machine "
    "moves need no transport, so LA does not act there.",
    "Second objective is the non-makespan part of the v5 objective "
    "C = 0.4*(processing energy/n_task + idle energy) + 0.2*transport time; "
    "J = 0.4*makespan + C. The paper's TC is c*sum(y_v)+uc*transport time.",
    "Reward uses partial-makespan gain (before - after, <=0) and non-MS cost "
    "change (after - before, may be negative); both are normalised by "
    "running min/max and the cost signal is clipped to [0,1].",
    "State features keep the 31/24/30 dimensions but adapt definitions: no "
    "dummy machine/L-U station, MCT is machine completion time, f9 counts "
    "used AGVs, f17 is cumulative C, and times are scaled by the mean "
    "compatible processing time.",
    "PA rules: SPT/SRPT use APT; EAJ ignores AGV deadhead/wait; EAM uses the "
    "earliest feasible insertion start; MINU uses processing time/MCT.",
    "Ties are broken by the lowest index; the paper selects randomly.",
    "Actors use linear categorical logits and critics a linear value; the "
    "paper's Table 4 labels the output Tanh.",
    "Independent PPO per agent with a shared reward; no centralised joint "
    "critic or mixing network.",
    "Default learning rate 1e-3 for OA/PA/LA (paper level 2 is "
    "1e-4/1e-4/1e-3) because the training budget is 100 episodes.",
    "PPO epochs = 20; the paper does not report this value.",
    "OA/PA update on the whole current episode (cap 1000); LA updates on "
    "all transport decisions of the episode, at episode end.",
    "A terminal reward -terminal_coef*J/(t_scale*n_task) is added to the "
    "last step, default coefficient 1.0; the paper uses per-step rewards.",
    "Epsilon-greedy exploration instead of noise on the action distribution.",
    "Training budget 100 episodes; the default policy is trained on 6x6 and "
    "evaluated zero-shot on 10x6 and 10x10.  --train-mix 6x6,10x6,10x10 "
    "generates a mixed training pool.",
    "One deterministic greedy schedule per instance and the v5 weighted "
    "objective; no 20-run Pareto/HV/SP protocol.",
    "The stepwise simulator is checked against the existing v5 decoder; see "
    "--check.",
]


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
@dataclass
class InstancePool:
    """Lazy view over a v5 instance pickle.

    The pool holds either four arrays (M1) or six arrays that add
    ``lu_machine`` and ``lu_edge`` (M2).
    """

    tag: str
    t: Any
    p: Any
    tt: Any
    edge: Any
    lu_machine: Any = None
    lu_edge: Any = None

    def __len__(self) -> int:
        return len(self.t)

    def get(self, index: int) -> tuple[np.ndarray, np.ndarray,
                                       np.ndarray, Sequence[Sequence[int]],
                                       Any, Any]:
        return (np.asarray(self.t[index], dtype=np.float64),
                np.asarray(self.p[index], dtype=np.float64),
                np.asarray(self.tt[index], dtype=np.float64),
                self.edge[index],
                (None if self.lu_machine is None
                 else np.asarray(self.lu_machine[index], dtype=np.float64)),
                (None if self.lu_edge is None
                 else np.asarray(self.lu_edge[index], dtype=np.float64)))


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def _v5_dir() -> Path:
    return _root_dir() / "plcsp_v5"


def load_instance_pool(path: str | os.PathLike[str], tag: str) -> InstancePool:
    with open(path, "rb") as handle:
        raw = pickle.load(handle)
    if (not isinstance(raw, (list, tuple))
            or len(raw) not in (4, 6)):
        raise ValueError(f"{path}: expected a four- or six-element "
                         f"instance list")
    return InstancePool(tag=tag, t=raw[0], p=raw[1], tt=raw[2], edge=raw[3],
                        lu_machine=raw[4] if len(raw) >= 6 else None,
                        lu_edge=raw[5] if len(raw) >= 6 else None)


def _generate_pool(n_job: int, n_machine: int, samples: int,
                   seed: int) -> InstancePool:
    """Generate a temporary v5 training pool with the repository generator."""
    v5 = _v5_dir()
    sys.path.insert(0, str(v5))
    from instance.generate_allsize_mofjsp_dataset import Instance_Dataset  # noqa: E402

    with open(v5 / "instance" / "config_ins.json", "r",
              encoding="utf-8") as handle:
        conf = json.load(handle)["ability_scope"]

    tag = f"J{n_job}M{n_machine}E2"
    with tempfile.TemporaryDirectory(prefix="nhdrl_train_") as tmp:
        pkl_path = os.path.join(tmp, f"train_{tag}.pkl")
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            Instance_Dataset(
                samples=int(samples),
                n_job=int(n_job),
                n_machine=int(n_machine),
                n_edge=2,
                ability_scope=conf,
                use_PT=0,
                seed=int(seed),
                generate_true=1,
                csv_pth=os.path.join(tmp, "__skip__", "x.csv"),
                pkl_pth=pkl_path,
            )
        with open(pkl_path, "rb") as handle:
            raw = pickle.load(handle)
    return InstancePool(tag=tag, t=raw[0], p=raw[1], tt=raw[2], edge=raw[3],
                        lu_machine=raw[4] if len(raw) >= 6 else None,
                        lu_edge=raw[5] if len(raw) >= 6 else None)


def load_train_pools(args: argparse.Namespace) -> list[InstancePool]:
    pools: list[InstancePool] = []
    suffix = "_lu" if bool(args.lu_station) else ""
    for raw_tag in str(args.train_mix).split(","):
        tag = raw_tag.strip()
        if not tag:
            continue
        match = re.fullmatch(r"(\d+)x(\d+)", tag)
        if match is None:
            raise ValueError(f"bad --train-mix entry: {tag}")
        n_job, n_machine = int(match.group(1)), int(match.group(2))
        label = f"J{n_job}M{n_machine}E2"
        if (n_job, n_machine) == (6, 6):
            path = (_v5_dir() / "instance"
                    / f"Instance_J6M6E2{suffix}.pkl")
            if path.exists():
                pools.append(load_instance_pool(path, label))
                continue
        seed = int(args.seed) + 1000 + 10 * n_job + n_machine
        pool = _generate_pool(n_job, n_machine,
                              int(args.train_samples), seed)
        if args.lu_station and pool.lu_machine is None:
            raise ValueError("the generator produced no lu_machine/lu_edge; "
                             "M2 training needs the six-array pool")
        pools.append(pool)
    if not pools:
        raise ValueError("the training mix is empty")
    return pools


# ----------------------------------------------------------------------
# Trajectories
# ----------------------------------------------------------------------
class Trajectory:
    def __init__(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.logprobs: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []

    def add(self, state: np.ndarray, action: int, logprob: float,
            value: float, reward: float) -> None:
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.logprobs.append(float(logprob))
        self.values.append(float(value))
        self.rewards.append(float(reward))

    def __len__(self) -> int:
        return len(self.rewards)


def _update_agent(agent: Agent, traj: Trajectory, cap: int) -> dict[str, float]:
    if len(traj) < 2:
        return {"n": 0}
    adv, ret = compute_gae(traj.rewards, traj.values,
                           agent.config.gamma, agent.config.gae_lambda)
    start = max(0, len(traj) - int(cap))
    return agent.update(
        np.asarray(traj.states[start:], dtype=np.float32),
        np.asarray(traj.actions[start:], dtype=np.int64),
        np.asarray(traj.logprobs[start:], dtype=np.float32),
        adv[start:],
        ret[start:],
    )


# ----------------------------------------------------------------------
# Reward
# ----------------------------------------------------------------------
def _reward(oa_action: int, outcome, ms_stats: RunningMinMax,
            tc_stats: RunningMinMax, args: argparse.Namespace) -> float:
    ms_stats.update(outcome.ms_gain)
    tc_stats.update(outcome.cost_delta)
    if int(oa_action) == 0:
        norm = ms_stats.high(outcome.ms_gain)
        pref = float(args.pref_ms)
    else:
        norm = tc_stats.low(outcome.cost_delta)
        pref = float(args.pref_tc)
    value = (float(args.w1) * norm
             + float(args.w2) * (pref / float(args.maxp)) * norm)
    return float(value)


# ----------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------
def build_agents(args: argparse.Namespace,
                 device: torch.device) -> dict[str, Agent]:
    def cfg(lr: float) -> PPOConfig:
        return PPOConfig(
            lr=float(lr), gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda), clip=float(args.clip),
            ppo_epochs=int(args.ppo_epochs),
            minibatch=int(args.minibatch),
            entropy_coef=float(args.entropy_coef),
            value_coef=float(args.value_coef),
            max_grad_norm=float(args.max_grad_norm),
        )

    return {
        "oa": Agent("oa", N_OA, N_OA_ACTIONS, cfg(args.lr_oa), device),
        "pa": Agent("pa", N_PA, N_PA_ACTIONS, cfg(args.lr_pa), device),
        "la": Agent("la", N_LA, N_LA_ACTIONS, cfg(args.lr_la), device),
    }


def train(args: argparse.Namespace, pools: list[InstancePool],
          agents: dict[str, Agent], device: torch.device,
          rng: random.Random, ms_stats: RunningMinMax,
          tc_stats: RunningMinMax) -> dict[str, Any]:
    eps = float(args.eps_start)
    history: list[dict[str, Any]] = []
    t0 = time.time()
    updates = {"oa": 0, "pa": 0, "la": 0}

    for episode in range(int(args.epochs)):
        pool = pools[episode % len(pools)]
        index = rng.randrange(len(pool))
        t, p, tt, edge, lu_m, lu_e = pool.get(index)
        sim = NestedSim(t, p, tt, edge, n_agv=int(args.n_agv),
                        lu_station=bool(args.lu_station),
                        lu_machine=lu_m, lu_edge=lu_e)
        oa_traj, pa_traj, la_traj = Trajectory(), Trajectory(), Trajectory()
        episode_return = 0.0

        while not sim.done:
            oa_vec = sim.oa_features(float(args.pref_ms),
                                     float(args.pref_tc),
                                     float(args.maxp))
            a_oa, lp_oa, v_oa = agents["oa"].act(oa_vec, eps, False, rng)

            pa_vec = sim.pa_features(a_oa, oa_vec)
            a_pa, lp_pa, v_pa = agents["pa"].act(pa_vec, eps, False, rng)
            _job, task, machine = sim.choose_pa(a_pa)

            need = sim.transport_info(task, machine) is not None
            if need:
                la_vec = sim.la_features(task, machine, a_oa, oa_vec)
                a_la, lp_la, v_la = agents["la"].act(la_vec, eps, False, rng)
            else:
                a_la = None

            outcome = sim.schedule(task, machine,
                                   AGV_RULES[a_la] if need else None)
            reward = _reward(a_oa, outcome, ms_stats, tc_stats, args)
            episode_return += reward

            oa_traj.add(oa_vec, a_oa, lp_oa, v_oa, reward)
            pa_traj.add(pa_vec, a_pa, lp_pa, v_pa, reward)
            if need:
                la_traj.add(la_vec, a_la, lp_la, v_la, reward)

        final_obj = float(sim.result()["obj"])
        if float(args.terminal_coef) > 0.0:
            term = (-float(args.terminal_coef) * final_obj
                    / (sim.t_scale * sim.n_task))
            episode_return += term
            if oa_traj.rewards:
                oa_traj.rewards[-1] += term
            if pa_traj.rewards:
                pa_traj.rewards[-1] += term
            if la_traj.rewards:
                la_traj.rewards[-1] += term

        stats_oa = _update_agent(agents["oa"], oa_traj, int(args.cap_oa))
        stats_pa = _update_agent(agents["pa"], pa_traj, int(args.cap_pa))
        stats_la = _update_agent(agents["la"], la_traj, int(args.cap_la))
        if stats_oa.get("n", 0):
            updates["oa"] += 1
        if stats_pa.get("n", 0):
            updates["pa"] += 1
        if stats_la.get("n", 0):
            updates["la"] += 1

        history.append({
            "episode": episode,
            "pool": pool.tag,
            "instance": index,
            "return": float(episode_return),
            "obj": final_obj,
            "eps": float(eps),
            "la_steps": len(la_traj),
        })
        eps = max(eps * float(args.eps_decay), float(args.eps_min))

        if (episode + 1) % max(1, int(args.log_every)) == 0 or episode == 0:
            print(f"[train] ep={episode + 1}/{args.epochs} "
                  f"pool={pool.tag} idx={index} "
                  f"return={episode_return:.3f} obj={final_obj:.2f} "
                  f"eps={eps:.3f} la_steps={len(la_traj)}", flush=True)

    wall = time.time() - t0
    tail = history[-max(1, min(10, len(history))):]
    return {
        "ms_stats": ms_stats,
        "tc_stats": tc_stats,
        "history": history,
        "wall_s": wall,
        "updates": updates,
        "last10_return": float(np.mean([h["return"] for h in tail])),
        "last10_obj": float(np.mean([h["obj"] for h in tail])),
        "eps_end": float(eps),
    }


def evaluate(args: argparse.Namespace, agents: dict[str, Agent],
             pool: InstancePool, device: torch.device) -> dict[str, Any]:
    rng = random.Random(int(args.seed) + 7919)
    records: list[dict[str, Any]] = []
    n = len(pool) if int(args.limit) <= 0 else min(int(args.limit), len(pool))
    t0 = time.time()

    for index in range(n):
        t, p, tt, edge, lu_m, lu_e = pool.get(index)
        sim = NestedSim(t, p, tt, edge, n_agv=int(args.n_agv),
                        lu_station=bool(args.lu_station),
                        lu_machine=lu_m, lu_edge=lu_e)
        oa_counts = [0] * N_OA_ACTIONS
        pa_counts = [0] * N_PA_ACTIONS
        la_counts = [0] * N_LA_ACTIONS
        la_steps = 0

        while not sim.done:
            oa_vec = sim.oa_features(float(args.pref_ms),
                                     float(args.pref_tc),
                                     float(args.maxp))
            a_oa = agents["oa"].act(oa_vec, 0.0, True, rng)[0]
            oa_counts[a_oa] += 1

            pa_vec = sim.pa_features(a_oa, oa_vec)
            a_pa = agents["pa"].act(pa_vec, 0.0, True, rng)[0]
            pa_counts[a_pa] += 1
            _job, task, machine = sim.choose_pa(a_pa)

            need = sim.transport_info(task, machine) is not None
            if need:
                la_vec = sim.la_features(task, machine, a_oa, oa_vec)
                a_la = agents["la"].act(la_vec, 0.0, True, rng)[0]
                la_counts[a_la] += 1
                la_steps += 1
            else:
                a_la = None

            sim.schedule(task, machine,
                         AGV_RULES[a_la] if need else None)

        result = sim.result()
        records.append({
            "instance": index,
            "mk": result["mk"],
            "pt": result["pt"],
            "transT": result["transT"],
            "idleT": result["idleT"],
            "obj": result["obj"],
            "cost_non_ms": result["cost_non_ms"],
            "steps": int(sim.n_task),
            "la_steps": la_steps,
            "oa_actions": oa_counts,
            "pa_actions": pa_counts,
            "la_actions": la_counts,
        })
        if (index + 1) % max(1, int(args.log_every)) == 0 or index == 0:
            print(f"[eval] {index + 1}/{n} obj={result['obj']:.2f} "
                  f"mk={result['mk']:.2f}", flush=True)

    wall = time.time() - t0
    objs = np.asarray([r["obj"] for r in records], dtype=float)
    summary: dict[str, Any] = {
        "mean": float(objs.mean()),
        "std": float(objs.std(ddof=0)),
        "min": float(objs.min()),
        "max": float(objs.max()),
        "components": {
            key: float(np.mean([r[key] for r in records]))
            for key in ("mk", "pt", "transT", "idleT", "cost_non_ms")
        },
        "la_steps_mean": float(np.mean([r["la_steps"] for r in records])),
        "wall_s": wall,
    }
    return {"records": records, "summary": summary, "wall_s": wall}


# ----------------------------------------------------------------------
# Validation against the existing v5 decoder
# ----------------------------------------------------------------------
def check_decode(args: argparse.Namespace) -> int:
    baselines_dir = _v5_dir() / "baselines"
    sys.path.insert(0, str(baselines_dir))
    from agv_baselines import decode, decode_m2  # noqa: E402

    pool = load_instance_pool(args.instances, "check")
    n = min(int(args.check_limit), len(pool))
    worst = {key: 0.0 for key in ("mk", "pt", "transT", "idleT", "obj",
                                  "start", "finish")}
    for index in range(n):
        t, p, tt, edge, lu_m, lu_e = pool.get(index)
        for pa_action in range(N_PA_ACTIONS):
            sim = NestedSim(t, p, tt, edge, n_agv=int(args.n_agv),
                            lu_station=bool(args.lu_station),
                            lu_machine=lu_m, lu_edge=lu_e)
            while not sim.done:
                _job, task, machine = sim.choose_pa(pa_action)
                sim.schedule(task, machine, EARLIEST)
            ms = np.asarray(sim.ms, dtype=int)
            if args.lu_station:
                dec = decode_m2(t, p, tt, edge, lu_m, lu_e, sim.os_seq, ms,
                                n_agv=int(args.n_agv))
            else:
                dec = decode(t, p, tt, edge, sim.os_seq, ms, agv=True,
                             n_agv=int(args.n_agv))
            res = sim.result()
            for key in ("mk", "pt", "transT", "idleT", "obj"):
                worst[key] = max(worst[key], abs(float(res[key])
                                                 - float(dec[key])))
            start_diff = float(np.max(np.abs(sim.start
                                             - np.asarray(dec["start"]))))
            finish_diff = float(np.max(np.abs(sim.finish
                                              - np.asarray(dec["finish"]))))
            worst["start"] = max(worst["start"], start_diff)
            worst["finish"] = max(worst["finish"], finish_diff)
        print(f"[check] instance {index}: worst so far "
              f"{ {k: round(v, 6) for k, v in worst.items()} }", flush=True)

    print("[check] worst absolute difference: "
          f"{ {k: round(v, 8) for k, v in worst.items()} }", flush=True)
    if max(worst.values()) > 1e-3:
        print("[check] FAILED", flush=True)
        return 1
    print("[check] OK", flush=True)
    return 0


# ----------------------------------------------------------------------
# Weights and CLI
# ----------------------------------------------------------------------
def _n_params(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _save_weights(path: Path, agents: dict[str, Agent],
                  ms_stats: RunningMinMax, tc_stats: RunningMinMax,
                  args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "nhdrl-adapted-v1",
        "source": SOURCE,
        "adapted": True,
        "feature_dims": {"oa": N_OA, "pa": N_PA, "la": N_LA},
        "action_dims": {"oa": N_OA_ACTIONS, "pa": N_PA_ACTIONS,
                        "la": N_LA_ACTIONS},
        "reward_stats": {
            "ms": ms_stats.state_dict(),
            "tc": tc_stats.state_dict(),
        },
        "args": {key: value for key, value in vars(args).items()},
        "agents": {
            name: {
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
            }
            for name, agent in agents.items()
        },
    }
    torch.save(payload, path)


def _load_weights(path: Path, agents: dict[str, Agent],
                  ms_stats: RunningMinMax, tc_stats: RunningMinMax,
                  device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    if ckpt.get("format") != "nhdrl-adapted-v1":
        raise ValueError(f"{path}: not an adapted NHDRL checkpoint")
    expected_dims = {"oa": N_OA, "pa": N_PA, "la": N_LA}
    expected_actions = {"oa": N_OA_ACTIONS, "pa": N_PA_ACTIONS,
                        "la": N_LA_ACTIONS}
    if ckpt.get("feature_dims") != expected_dims:
        raise ValueError(f"{path}: feature dimensions do not match")
    if ckpt.get("action_dims") != expected_actions:
        raise ValueError(f"{path}: action dimensions do not match")
    for name, agent in agents.items():
        agent.actor.load_state_dict(ckpt["agents"][name]["actor"])
        agent.critic.load_state_dict(ckpt["agents"][name]["critic"])
    ms_stats.min_value = float(ckpt["reward_stats"]["ms"]["min"])
    ms_stats.max_value = float(ckpt["reward_stats"]["ms"]["max"])
    tc_stats.min_value = float(ckpt["reward_stats"]["tc"]["min"])
    tc_stats.max_value = float(ckpt["reward_stats"]["tc"]["max"])
    return ckpt


def build_parser() -> argparse.ArgumentParser:
    v5 = _v5_dir()
    parser = argparse.ArgumentParser(
        description="Adapted NHDRL baseline (Shi et al., AEI 2025 103195).")
    parser.add_argument(
        "--instances",
        default=str(v5 / "instance" / "test_Instance_J6M6E2.pkl"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100,
                        help="training episodes (paper: 1000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-agv", type=int, default=6)
    parser.add_argument("--lu-station", type=int, default=0,
                        help="1 = M2 transport model with load/unload "
                             "stations (default: M1)")
    parser.add_argument("--train-mix", default="6x6",
                        help="comma-separated JxM sizes for training; "
                             "use 6x6,10x6,10x10 for a mixed pool")
    parser.add_argument("--train-samples", type=int, default=200,
                        help="generated training instances per non-6x6 size")
    parser.add_argument("--lr-oa", type=float, default=1e-3)
    parser.add_argument("--lr-pa", type=float, default=1e-3)
    parser.add_argument("--lr-la", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=20)
    parser.add_argument("--minibatch", type=int, default=32)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--cap-oa", type=int, default=1000)
    parser.add_argument("--cap-pa", type=int, default=1000)
    parser.add_argument("--cap-la", type=int, default=500)
    parser.add_argument("--eps-start", type=float, default=0.9)
    parser.add_argument("--eps-min", type=float, default=0.1)
    parser.add_argument("--eps-decay", type=float, default=0.99)
    parser.add_argument("--w1", type=float, default=0.8,
                        help="weight of the objective-optimisation signal")
    parser.add_argument("--w2", type=float, default=0.2,
                        help="weight of the preference-satisfaction signal")
    parser.add_argument("--pref-ms", type=float, default=5.0)
    parser.add_argument("--pref-tc", type=float, default=5.0)
    parser.add_argument("--maxp", type=float, default=10.0)
    parser.add_argument("--terminal-coef", type=float, default=1.0,
                        help="terminal reward coefficient (0 disables it)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="",
                        help="JSON output path")
    parser.add_argument("--weights-out", default="",
                        help="checkpoint path (default: next to --out)")
    parser.add_argument("--load-weights", default="",
                        help="initialize from an adapted NHDRL checkpoint")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training; requires --load-weights")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--check", action="store_true",
                        help="compare the simulator with the v5 decoder")
    parser.add_argument("--check-limit", type=int, default=3)
    return parser


def _resolve_device(name: str) -> torch.device:
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA is not available; using CPU", flush=True)
        return torch.device("cpu")
    return torch.device(name)


def _m2_out_path(path: Path) -> Path:
    """Never overwrite an M1 result file with an M2 run."""
    if "m2" in path.name.lower():
        return path
    return path.with_name("m2_" + path.name)


def _default_test_instances(lu_station: int) -> str:
    suffix = "_lu" if int(lu_station) else ""
    return str(_v5_dir() / "instance"
               / f"test_Instance_J6M6E2{suffix}.pkl")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if str(args.instances) == _default_test_instances(0):
        # The user kept the default path; follow the selected model.
        args.instances = _default_test_instances(int(args.lu_station))
    if args.lu_station:
        if args.out:
            args.out = str(_m2_out_path(Path(args.out)))
        if args.weights_out:
            args.weights_out = str(_m2_out_path(Path(args.weights_out)))
    device = _resolve_device(args.device)
    if args.check:
        return check_decode(args)

    total_start = time.time()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    rng = random.Random(int(args.seed))

    agents = build_agents(args, device)
    ms_stats = RunningMinMax(-1.0, 0.0)
    tc_stats = RunningMinMax(0.0, 1.0)
    loaded_ckpt: Optional[dict[str, Any]] = None
    if args.load_weights:
        print(f"[load] {args.load_weights}", flush=True)
        loaded_ckpt = _load_weights(Path(args.load_weights), agents,
                                    ms_stats, tc_stats, device)
        saved_lu = bool((loaded_ckpt.get("args") or {}).get("lu_station", 0))
        if saved_lu != bool(args.lu_station):
            print(f"[warn] checkpoint lu_station={int(saved_lu)} but this run "
                  f"uses lu_station={int(bool(args.lu_station))}; the network "
                  f"is unchanged, only the transport model differs",
                  flush=True)

    if args.eval_only:
        if not args.load_weights:
            print("--eval-only requires --load-weights", file=sys.stderr)
            return 2
        pools = []
        saved_args = (loaded_ckpt or {}).get("args", {})
        train_mix_label = str(saved_args.get("train_mix", args.train_mix))
        trained_episodes = int(saved_args.get("epochs", 0))
        train_info = {
            "ms_stats": ms_stats,
            "tc_stats": tc_stats,
            "history": [],
            "wall_s": 0.0,
            "updates": {"oa": 0, "pa": 0, "la": 0},
            "last10_return": 0.0,
            "last10_obj": 0.0,
            "eps_end": 0.0,
        }
    else:
        print(f"[data] loading training pools: {args.train_mix}", flush=True)
        pools = load_train_pools(args)
        for pool in pools:
            print(f"[data] pool {pool.tag}: {len(pool)} instances",
                  flush=True)
        train_info = train(args, pools, agents, device, rng,
                           ms_stats, tc_stats)
        ms_stats = train_info["ms_stats"]
        tc_stats = train_info["tc_stats"]
        train_mix_label = str(args.train_mix)
        trained_episodes = int(args.epochs)

    print(f"[data] loading test pool: {args.instances}", flush=True)
    test_pool = load_instance_pool(args.instances, "test")
    eval_info = evaluate(args, agents, test_pool, device)
    total_wall = time.time() - total_start

    weights_path: Path
    if args.weights_out:
        weights_path = Path(args.weights_out)
    elif args.eval_only and args.load_weights:
        weights_path = Path(args.load_weights)
    elif args.out:
        weights_path = Path(args.out).with_suffix(".pt")
    else:
        weights_path = (Path(tempfile.gettempdir())
                        / f"nhdrl_adapted_seed{args.seed}.pt")
    if not (args.eval_only and not args.weights_out):
        _save_weights(weights_path, agents, ms_stats, tc_stats, args)

    payload = {
        "mode": "NHDRL-adapted",
        "adapted": True,
        "source": SOURCE,
        "official_code": OFFICIAL_CODE,
        "deviations": DEVIATIONS,
        "lu_station": bool(args.lu_station),
        "instances": str(args.instances),
        "n": len(eval_info["records"]),
        "summary": eval_info["summary"],
        "records": eval_info["records"],
        "training": {
            "episodes": int(trained_episodes),
            "eval_only": bool(args.eval_only),
            "loaded_weights": str(args.load_weights),
            "train_mix": train_mix_label,
            "pool_sizes": {pool.tag: len(pool) for pool in pools},
            "train_samples_per_generated_size": int(
                (loaded_ckpt or {}).get("args", {}).get(
                    "train_samples", args.train_samples)),
            "lr": {"oa": float(args.lr_oa), "pa": float(args.lr_pa),
                   "la": float(args.lr_la)},
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "clip": float(args.clip),
            "ppo_epochs": int(args.ppo_epochs),
            "minibatch": int(args.minibatch),
            "caps": {"oa": int(args.cap_oa), "pa": int(args.cap_pa),
                     "la": int(args.cap_la)},
            "eps": {"start": float(args.eps_start),
                    "min": float(args.eps_min),
                    "decay": float(args.eps_decay),
                    "end": float(train_info["eps_end"])},
            "reward": {
                "w1": float(args.w1), "w2": float(args.w2),
                "pref_ms": float(args.pref_ms),
                "pref_tc": float(args.pref_tc),
                "maxp": float(args.maxp),
                "terminal_coef": float(args.terminal_coef),
                "ms_stats": ms_stats.state_dict(),
                "tc_stats": tc_stats.state_dict(),
            },
            "updates": train_info["updates"],
            "last10_return": train_info["last10_return"],
            "last10_obj": train_info["last10_obj"],
            "wall_s": train_info["wall_s"],
            "history": train_info["history"],
        },
        "feature_dims": {"oa": N_OA, "pa": N_PA, "la": N_LA},
        "feature_names": FEATURE_NAMES,
        "action_dims": {"oa": N_OA_ACTIONS, "pa": N_PA_ACTIONS,
                        "la": N_LA_ACTIONS},
        "action_labels": {
            "oa": ["MS", "cost"],
            "pa": [f"{j}+{m}" for j in ("SPT", "EAJ", "SRPT")
                   for m in ("SMPT", "MINU", "EAM")],
            "la": list(AGV_RULES),
        },
        "model_params": {
            name: {"actor": _n_params(agent.actor),
                   "critic": _n_params(agent.critic)}
            for name, agent in agents.items()
        },
        "runtime": {
            "device": str(device),
            "cpu_only": bool(device.type == "cpu"),
            "torch_threads": int(torch.get_num_threads()),
            "train_wall_s": float(train_info["wall_s"]),
            "eval_wall_s": float(eval_info["wall_s"]),
            "total_wall_s": float(total_wall),
        },
        "weights": str(weights_path),
    }

    print(json.dumps({
        "mode": payload["mode"],
        "n": payload["n"],
        "summary": payload["summary"],
        "training": {
            "episodes": payload["training"]["episodes"],
            "last10_obj": payload["training"]["last10_obj"],
            "wall_s": payload["training"]["wall_s"],
        },
        "runtime": payload["runtime"],
    }, indent=2), flush=True)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=1),
                            encoding="utf-8")
        print(f"[out] {out_path}", flush=True)
    print(f"[weights] {weights_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
