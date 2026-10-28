# -*- coding: utf-8 -*-
"""Tiny MAPPO-style policy used by the adapted Li et al. baseline.

The paper trains three actors and three critics (fully connected 128-64-64)
with MAPPO: GAE, PopArt, clipped PPO actor loss, clipped Huber critic loss,
Multivariate Gaussian action sampling, and a deterministic mean at test time.
This module keeps the architecture and the loss shapes, and makes three
implementation-level deviations that are documented in ``eval_li2025.py``:

* diagonal Gaussian instead of an unspecified multivariate covariance;
* one independent critic per actor, all conditioned on the same global state
  ``concat(o_jf, noise)`` exactly as in Eq. (20), but PopArt is approximated by
  per-batch advantage normalisation;
* smaller update buffers because the total training budget is 100 episodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

AGENTS = ("jf", "js", "ms", "as")
MEAN_GAIN = 0.1
LOG_STD_INIT = -0.5


class RunningMeanStd:
    """Online mean/variance used for the paper's Z-score state normalisation."""

    def __init__(self, shape: int, count: float = 1e-4) -> None:
        self.mean = np.zeros(int(shape), dtype=np.float64)
        self.var = np.ones(int(shape), dtype=np.float64)
        self.count = float(count)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0) if x.ndim > 1 else x
        batch_var = (np.var(x, axis=0) if x.ndim > 1
                     else np.zeros_like(batch_mean))
        batch_count = x.shape[0] if x.ndim > 1 else 1
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = (self.var * self.count + batch_var * batch_count
              + delta ** 2 * self.count * batch_count / total)
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        out = ((np.asarray(x, dtype=np.float64) - self.mean)
               / np.sqrt(self.var + 1e-8))
        return out.astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(),
                "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class Actor(nn.Module):
    """Fully connected actor with a state-independent diagonal Gaussian."""

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: tuple[int, ...] = (128, 64, 64),
                 bias: Iterable[float] | None = None) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = int(obs_dim)
        for width in hidden:
            linear = nn.Linear(last, int(width))
            nn.init.orthogonal_(linear.weight, gain=float(np.sqrt(2.0)))
            nn.init.zeros_(linear.bias)
            layers.extend([linear, nn.Tanh()])
            last = int(width)
        self.body = nn.Sequential(*layers)
        self.mean = nn.Linear(last, int(act_dim))
        # The output gain is larger than the usual 0.01 because the
        # 100-episode budget must move the action mean away from its init.
        nn.init.orthogonal_(self.mean.weight, gain=MEAN_GAIN)
        nn.init.zeros_(self.mean.bias)
        if bias is not None:
            with torch.no_grad():
                self.mean.bias.copy_(torch.as_tensor(
                    list(bias), dtype=torch.float32))
        self.log_std = nn.Parameter(
            torch.full((int(act_dim),), float(LOG_STD_INIT)))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean(self.body(obs))
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)


class Critic(nn.Module):
    """Fully connected centralised critic (one per actor, as in the paper)."""

    def __init__(self, global_dim: int,
                 hidden: tuple[int, ...] = (128, 64, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = int(global_dim)
        for width in hidden:
            linear = nn.Linear(last, int(width))
            nn.init.orthogonal_(linear.weight, gain=float(np.sqrt(2.0)))
            nn.init.zeros_(linear.bias)
            layers.extend([linear, nn.Tanh()])
            last = int(width)
        self.body = nn.Sequential(*layers)
        self.value = nn.Linear(last, 1)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.value(self.body(global_state)).squeeze(-1)


@dataclass(frozen=True)
class Step:
    """One actor step: normalised observation, raw action, log-prob, value."""

    obs: np.ndarray
    action: np.ndarray
    log_prob: float
    value: float


@dataclass
class Transition:
    """One stored transition for PPO."""

    obs: np.ndarray
    action: np.ndarray
    log_prob: float
    value: float
    reward: float
    done: bool
    global_state: np.ndarray
    advantage: float = 0.0
    ret: float = 0.0


@dataclass
class PPOConfig:
    """Training hyperparameters.  Defaults follow the paper where feasible."""

    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    epochs: int = 4
    minibatch: int = 128
    lr: float = 1e-3
    grad_clip: float = 0.5
    update_every: int = 512
    critic_coef: float = 0.5
    noise_dim: int = 10
    noise_sigma: float = 0.1


@dataclass
class PolicyBundle:
    """Actors, critics and state normalisers of the four agents."""

    actors: dict[str, Actor]
    critics: dict[str, Critic]
    normalizers: dict[str, RunningMeanStd]
    noise_dim: int

    @classmethod
    def create(cls, obs_dims: dict[str, int], act_dims: dict[str, int],
               noise_dim: int = 10, seed: int = 0,
               warm_start: bool = True) -> "PolicyBundle":
        torch.manual_seed(int(seed))
        # Rule-informed initialisation (budget adaptation).  Without it, the
        # 100-episode budget leaves the action mean near its random init.
        # ``warm_start=False`` reproduces the paper's random initialisation.
        if warm_start:
            biases = {
                "js": (3.0, 0.0, 0.0, 0.0),   # MWKR corner of the proto-action
                "ms": (3.0, 0.0, 0.0),        # SPT machine rule
            }
        else:
            biases = {"ms": (0.5, 0.5, 0.5)}
        actors: dict[str, Actor] = {}
        for agent in AGENTS:
            actors[agent] = Actor(obs_dims[agent], act_dims[agent],
                                  bias=biases.get(agent))
        global_dim = int(obs_dims["jf"]) + int(noise_dim)
        critics = {agent: Critic(global_dim) for agent in AGENTS}
        normalizers = {agent: RunningMeanStd(obs_dims[agent])
                       for agent in AGENTS}
        return cls(actors=actors, critics=critics,
                   normalizers=normalizers, noise_dim=int(noise_dim))

    def normalize(self, agent: str, raw_obs: np.ndarray,
                  update: bool) -> np.ndarray:
        normalizer = self.normalizers[agent]
        if update:
            normalizer.update(raw_obs)
        return normalizer.normalize(raw_obs)

    def act(self, agent: str, obs: np.ndarray, global_state: np.ndarray,
            deterministic: bool) -> Step:
        """Act from an already normalised observation."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        global_t = torch.as_tensor(global_state,
                                   dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = self.actors[agent].distribution(obs_t)
            raw = dist.mean.squeeze(0) if deterministic else dist.sample()[0]
            log_prob = dist.log_prob(raw.unsqueeze(0)).sum(-1)
            value = self.critics[agent](global_t)
        return Step(obs=np.asarray(obs, dtype=np.float32),
                    action=raw.numpy().astype(np.float32),
                    log_prob=float(log_prob.item()),
                    value=float(value.item()))


def compute_gae(transitions: list[Transition], gamma: float,
                lam: float) -> None:
    """In-place GAE for one agent's episode transitions."""
    if not transitions:
        return
    values = np.asarray([t.value for t in transitions], dtype=np.float64)
    gae = 0.0
    for t in range(len(transitions) - 1, -1, -1):
        next_value = 0.0 if transitions[t].done else values[t + 1]
        delta = transitions[t].reward + gamma * next_value - values[t]
        gae = delta + gamma * lam * (0.0 if transitions[t].done else 1.0) * gae
        transitions[t].advantage = float(gae)
        transitions[t].ret = float(gae + values[t])


def update_agent(agent: str, transitions: list[Transition],
                 bundle: PolicyBundle, optimizer: torch.optim.Optimizer,
                 cfg: PPOConfig) -> dict[str, float]:
    """PPO update of one actor/critic pair.  Returns mean losses."""
    if len(transitions) < 2:
        return {}
    obs = torch.as_tensor(np.stack([t.obs for t in transitions]),
                          dtype=torch.float32)
    actions = torch.as_tensor(np.stack([t.action for t in transitions]),
                              dtype=torch.float32)
    old_log_prob = torch.as_tensor([t.log_prob for t in transitions],
                                   dtype=torch.float32)
    old_values = torch.as_tensor([t.value for t in transitions],
                                 dtype=torch.float32)
    global_state = torch.as_tensor(
        np.stack([t.global_state for t in transitions]), dtype=torch.float32)
    advantage = torch.as_tensor([t.advantage for t in transitions],
                                dtype=torch.float32)
    returns = torch.as_tensor([t.ret for t in transitions],
                              dtype=torch.float32)
    advantage = ((advantage - advantage.mean())
                 / (advantage.std(unbiased=False) + 1e-8))
    actor = bundle.actors[agent]
    critic = bundle.critics[agent]
    n = len(transitions)
    stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
    updates = 0
    for _epoch in range(int(cfg.epochs)):
        perm = torch.randperm(n)
        for start in range(0, n, int(cfg.minibatch)):
            idx = perm[start:start + int(cfg.minibatch)]
            dist = actor.distribution(obs[idx])
            new_log_prob = dist.log_prob(actions[idx]).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
            ratio = torch.exp(new_log_prob - old_log_prob[idx])
            surrogate = ratio * advantage[idx]
            clipped = torch.clamp(ratio, 1.0 - cfg.clip,
                                  1.0 + cfg.clip) * advantage[idx]
            actor_loss = -(torch.min(surrogate, clipped).mean()
                           + cfg.entropy_coef * entropy)
            value = critic(global_state[idx])
            huber = F.smooth_l1_loss(value, returns[idx], beta=1.0,
                                     reduction="none")
            value_clipped = torch.clamp(value,
                                        old_values[idx] - cfg.clip,
                                        old_values[idx] + cfg.clip)
            huber_clipped = F.smooth_l1_loss(
                value_clipped, returns[idx], beta=1.0, reduction="none")
            critic_loss = torch.max(huber, huber_clipped).mean()
            loss = actor_loss + cfg.critic_coef * critic_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()),
                cfg.grad_clip)
            optimizer.step()
            stats["actor_loss"] += float(actor_loss.item())
            stats["critic_loss"] += float(critic_loss.item())
            stats["entropy"] += float(entropy.item())
            updates += 1
    for key in stats:
        stats[key] /= max(1, updates)
    stats["n"] = float(n)
    return stats


def save_policy(path: Path, bundle: PolicyBundle, cfg: PPOConfig,
                meta: dict) -> None:
    """Save actors, critics, normalisers, config and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "actors": {a: m.state_dict() for a, m in bundle.actors.items()},
        "critics": {a: m.state_dict() for a, m in bundle.critics.items()},
        "normalizers": {a: n.state_dict()
                        for a, n in bundle.normalizers.items()},
        "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "meta": meta,
    }
    torch.save(payload, path)


def load_policy(path: Path) -> tuple[PolicyBundle, PPOConfig, dict]:
    """Load a checkpoint created by :func:`save_policy`."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    obs_dims = {a: int(v) for a, v in meta["obs_dims"].items()}
    act_dims = {a: int(v) for a, v in meta["act_dims"].items()}
    bundle = PolicyBundle.create(obs_dims, act_dims,
                                 noise_dim=int(meta["noise_dim"]))
    for agent in AGENTS:
        bundle.actors[agent].load_state_dict(payload["actors"][agent])
        bundle.critics[agent].load_state_dict(payload["critics"][agent])
        bundle.normalizers[agent].load_state_dict(
            payload["normalizers"][agent])
    cfg = PPOConfig(**payload["config"])
    return bundle, cfg, meta
