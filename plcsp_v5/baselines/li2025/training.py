# -*- coding: utf-8 -*-
"""Training and evaluation loops for the adapted Li et al. baseline."""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from li2025.env_model import Instance
from li2025.policy import AGENTS, PolicyBundle, PPOConfig, update_agent
from li2025.runner import EpisodeResult, run_episode

ROOT = Path(__file__).resolve().parents[3]
INSTANCE_DIR = ROOT / "plcsp_v5" / "instance"


def load_pickle_instances(path: Path, limit: int = 0,
                          lu_station: bool = False) -> list[Instance]:
    """Load a v5 instance pickle into ``Instance`` objects.

    The pool holds either four arrays ``[t, p, tt, edge]`` (M1) or six arrays
    that add ``lu_machine`` and ``lu_edge`` (M2).
    """
    import pickle

    with open(path, "rb") as f:
        raw = pickle.load(f)
        t_all, p_all, tt_all, edge_all = raw[:4]
        lu_machine = raw[4] if len(raw) >= 6 else None
        lu_edge = raw[5] if len(raw) >= 6 else None
    n = len(t_all) if limit <= 0 else min(int(limit), len(t_all))
    return [Instance.from_arrays(
        t_all[i], p_all[i], tt_all[i], edge_all[i],
        lu_machine=None if lu_machine is None else lu_machine[i],
        lu_edge=None if lu_edge is None else lu_edge[i],
        lu_station=lu_station) for i in range(n)]


def generate_instances(n_job: int, n_machine: int, n_edge: int,
                       samples: int, seed: int,
                       lu_station: bool = False) -> list[Instance]:
    """Generate a training pool with the v5 instance generator.

    The seed is independent of the test seed (3), so no test instance leaks
    into training.  The generator prints progress; its output is suppressed.
    """
    from instance.generate_allsize_mofjsp_dataset import Instance_Dataset

    with open(INSTANCE_DIR / "config_ins.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    buffer = io.StringIO()
    # The v5 generator always writes its CSV/PKL products.  Keep them in the
    # system temp directory and delete them after the arrays are copied.
    tmp_dir = Path(tempfile.mkdtemp(prefix="li2025_gen_"))
    tmp_str = str(tmp_dir).replace("\\", "/")
    csv_pth = f"{tmp_str}/ability.csv"
    pkl_pth = f"{tmp_str}/instances.pkl"
    try:
        with contextlib.redirect_stdout(buffer):
            Instance_Dataset(
                samples=int(samples), n_job=int(n_job),
                n_machine=int(n_machine), n_edge=int(n_edge),
                ability_scope=config["ability_scope"],
                use_PT=0, seed=int(seed), generate_true=1,
                csv_pth=csv_pth, pkl_pth=pkl_pth,
            )
        # The generate branch of Instance_Dataset keeps its arrays local and
        # writes them to ``pkl_pth``; load that product.
        import pickle
        with open(pkl_pth, "rb") as f:
            raw = pickle.load(f)
        t_all, p_all, tt_all, edge_all = raw[:4]
        lu_machine = raw[4] if len(raw) >= 6 else None
        lu_edge = raw[5] if len(raw) >= 6 else None
        return [Instance.from_arrays(
            t_all[i], p_all[i], tt_all[i], edge_all[i],
            lu_machine=None if lu_machine is None else lu_machine[i],
            lu_edge=None if lu_edge is None else lu_edge[i],
            lu_station=lu_station)
            for i in range(len(t_all))]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def resolve_train_instances(n_job: int, n_machine: int, n_edge: int,
                            requested: str, pool_size: int,
                            seed: int,
                            lu_station: bool = False
                            ) -> tuple[list[Instance], str]:
    """Load the matching training pickle or generate a fresh pool."""
    if requested:
        path = Path(requested)
        return load_pickle_instances(path, lu_station=lu_station), str(path)
    suffix = "_lu" if lu_station else ""
    candidate = (INSTANCE_DIR
                 / f"Instance_J{n_job}M{n_machine}E{n_edge}{suffix}.pkl")
    if candidate.exists():
        return (load_pickle_instances(candidate, lu_station=lu_station),
                str(candidate))
    instances = generate_instances(n_job, n_machine, n_edge, pool_size, seed,
                                   lu_station=lu_station)
    return instances, f"generated(seed={seed}, n={len(instances)})"


def train(policy: PolicyBundle, instances: list[Instance], cfg: PPOConfig,
          episodes: int, seed: int, log_every: int = 10) -> dict:
    """Train the four actors/critics for ``episodes`` episodes."""
    if not instances:
        raise ValueError("empty training pool")
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    optimizers = {
        agent: torch.optim.Adam(
            list(policy.actors[agent].parameters())
            + list(policy.critics[agent].parameters()),
            lr=cfg.lr,
        )
        for agent in AGENTS
    }
    buffers: dict[str, list] = {agent: [] for agent in AGENTS}
    obj_history: list[float] = []
    update_history: list[dict] = []
    last_stats: dict[str, dict] = {}
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    for episode in range(int(episodes)):
        index = int(rng.integers(len(instances)))
        result = run_episode(instances[index], policy, cfg, rng,
                             training=True)
        obj_history.append(float(result.cost["obj"]))
        for agent in AGENTS:
            buffers[agent].extend(result.transitions[agent])
        total = sum(len(buffers[agent]) for agent in AGENTS)
        if total >= cfg.update_every:
            stats: dict[str, dict] = {}
            for agent in AGENTS:
                out = update_agent(agent, buffers[agent], policy,
                                   optimizers[agent], cfg)
                if out:
                    stats[agent] = out
                buffers[agent].clear()
            last_stats = stats
            update_history.append({"episode": episode + 1, "n": total})
        if log_every and (episode + 1) % int(log_every) == 0:
            recent = obj_history[-int(log_every):]
            print(f"train {episode + 1}/{episodes} "
                  f"obj_mean={float(np.mean(recent)):.3f} "
                  f"updates={len(update_history)}", flush=True)
    if any(buffers[agent] for agent in AGENTS):
        stats = {}
        for agent in AGENTS:
            if buffers[agent]:
                out = update_agent(agent, buffers[agent], policy,
                                   optimizers[agent], cfg)
                if out:
                    stats[agent] = out
            buffers[agent].clear()
        if stats:
            last_stats = stats
        update_history.append({"episode": int(episodes), "n": -1})
    wall = time.perf_counter() - t0
    cpu = time.process_time() - cpu0
    return {
        "episodes": int(episodes),
        "wall_s": float(wall),
        "cpu_s": float(cpu),
        "updates": len(update_history),
        "update_history": update_history,
        "last_stats": last_stats,
        "obj_history": obj_history,
        "obj_last10_mean": (float(np.mean(obj_history[-10:]))
                            if obj_history else 0.0),
        "obj_last10_std": (float(np.std(obj_history[-10:], ddof=0))
                           if obj_history else 0.0),
        "train_instance_wall_s": (float(wall) / max(1, int(episodes))),
    }


def _stat(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0)),
            "min": float(array.min()), "max": float(array.max())}


def evaluate(policy: PolicyBundle, instances: list[Instance], cfg: PPOConfig,
             limit: int, n_agv: int = 6) -> tuple[list[dict], dict]:
    """Run the deterministic policy on the test instances."""
    n = len(instances) if limit <= 0 else min(int(limit), len(instances))
    if n == 0:
        raise ValueError("no evaluation instances")
    rng = np.random.default_rng(0)
    records: list[dict] = []
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    for i in range(n):
        start = time.perf_counter()
        cpu_start = time.process_time()
        result: EpisodeResult = run_episode(
            instances[i], policy, cfg, rng, training=False, n_agv=n_agv)
        wall = time.perf_counter() - start
        cpu = time.process_time() - cpu_start
        record = {"instance": i, **{k: float(v)
                                    for k, v in result.cost.items()},
                  "steps": int(result.steps),
                  "n_transport": int(result.n_transport),
                  "wall_s": float(wall), "cpu_s": float(cpu)}
        records.append(record)
        print(f"eval {i + 1}/{n} obj={record['obj']:.3f} "
              f"wall={wall:.3f}s", flush=True)
    wall_total = time.perf_counter() - t0
    cpu_total = time.process_time() - cpu0
    summary = {
        "n": n,
        "obj": _stat([r["obj"] for r in records]),
        "mk": _stat([r["mk"] for r in records]),
        "pt": _stat([r["pt"] for r in records]),
        "transT": _stat([r["transT"] for r in records]),
        "idleT": _stat([r["idleT"] for r in records]),
        "steps_mean": float(np.mean([r["steps"] for r in records])),
        "transport_mean": float(np.mean([r["n_transport"]
                                         for r in records])),
        "wall_s": float(wall_total),
        "cpu_s": float(cpu_total),
        "wall_per_instance_s": float(wall_total) / max(1, n),
        "cpu_per_instance_s": float(cpu_total) / max(1, n),
    }
    return records, summary
