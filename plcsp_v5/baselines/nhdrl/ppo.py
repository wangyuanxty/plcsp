# -*- coding: utf-8 -*-
"""Independent PPO agents for the adapted NHDRL baseline.

The paper (Shi et al., Advanced Engineering Informatics 65 (2025) 103195)
uses three agents (OA, PA, LA), each with an actor and a critic, trained by
MAPPO with a shared reward.  Table 4 of the paper gives the actor 256 hidden
units and the critic 128 hidden units.  This module keeps that shape.

Deviation: the paper's output layer is labelled Tanh.  A Tanh output cannot
produce categorical action probabilities, so the actors here emit linear
logits and use a categorical distribution.  The critics emit a linear value.
The training is independent PPO per agent with a shared reward; the paper's
critics also take each agent's own state, so no centralized joint critic or
mixing network is used.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class MLPActor(nn.Module):
    """Actor: input -> 256 ReLU -> action logits."""

    def __init__(self, in_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPCritic(nn.Module):
    """Critic: input -> 128 ReLU -> scalar value."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class PPOConfig:
    lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ppo_epochs: int = 4
    minibatch: int = 32
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


class Agent:
    """One actor-critic pair with its own optimizer."""

    def __init__(self, name: str, state_dim: int, n_actions: int,
                 config: PPOConfig, device: torch.device):
        self.name = name
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.config = config
        self.device = device
        self.actor = MLPActor(state_dim, n_actions).to(device)
        self.critic = MLPCritic(state_dim).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.lr,
        )

    def act(self, state: np.ndarray, epsilon: float, greedy: bool,
            rng: random.Random,
            behaviour_logprob: bool = True) -> tuple[int, float, float]:
        """Return (action, log probability, state value).

        Training uses an epsilon-greedy behaviour policy
        ``(1 - eps) * pi + eps / |A|``.  PPO divides the new policy by the
        policy that generated the action, so the stored log probability must
        be the behaviour-policy one.  With ``behaviour_logprob=False`` the
        old reading (the actor's own log probability) is kept.
        """
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            logits = self.actor(s)
            value = float(self.critic(s).item())
            dist = Categorical(logits=logits)
            if greedy:
                action = int(torch.argmax(logits, dim=-1).item())
                logprob = float(
                    dist.log_prob(torch.tensor([action],
                                               device=self.device)).item())
            else:
                if rng.random() < float(epsilon):
                    action = int(rng.randrange(self.n_actions))
                else:
                    action = int(dist.sample().item())
                if behaviour_logprob:
                    probs = torch.softmax(logits, dim=-1)
                    mixed = ((1.0 - float(epsilon)) * probs[0, action]
                             + float(epsilon) / self.n_actions)
                    logprob = float(torch.log(
                        mixed.clamp_min(1e-12)).item())
                else:
                    logprob = float(
                        dist.log_prob(torch.tensor([action],
                                                   device=self.device)).item())
        return action, logprob, value

    def update(self, states: np.ndarray, actions: np.ndarray,
               old_logprobs: np.ndarray, advantages: np.ndarray,
               returns: np.ndarray) -> dict[str, float]:
        """Run PPO epochs on one batch of transitions."""
        cfg = self.config
        states_t = torch.as_tensor(states, dtype=torch.float32,
                                   device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long,
                                    device=self.device)
        old_logp_t = torch.as_tensor(old_logprobs, dtype=torch.float32,
                                     device=self.device)
        adv = np.asarray(advantages, dtype=np.float32)
        if adv.size > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32,
                                device=self.device)

        n = int(states_t.shape[0])
        if n == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0,
                    "entropy": 0.0, "n": 0}
        stats = {"policy_loss": 0.0, "value_loss": 0.0,
                 "entropy": 0.0, "n": 0}
        batch_size = min(max(1, int(cfg.minibatch)), n)
        for _ in range(int(cfg.ppo_epochs)):
            order = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                logits = self.actor(states_t[idx])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(actions_t[idx])
                entropy = dist.entropy().mean()
                # Numerical guard: a random epsilon-greedy action can have a
                # very small probability under a confident policy.  A raw
                # exp(new-old) can then overflow to inf and turn the loss
                # into NaN.  Clamp the log ratio and the ratio.
                log_ratio = torch.clamp(new_logp - old_logp_t[idx],
                                        -20.0, 20.0)
                ratio = torch.clamp(torch.exp(log_ratio), 0.0, 10.0)
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip,
                                    1.0 + cfg.clip) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                values = self.critic(states_t[idx])
                value_loss = F.mse_loss(values, ret_t[idx])
                loss = (policy_loss + cfg.value_coef * value_loss
                        - cfg.entropy_coef * entropy)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters())
                    + list(self.critic.parameters()),
                    cfg.max_grad_norm)
                self.optimizer.step()
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.item())
                stats["n"] += 1
        if stats["n"]:
            for key in ("policy_loss", "value_loss", "entropy"):
                stats[key] /= stats["n"]
        return stats


class RunningMinMax:
    """Running min/max used by the paper's reward normalisation.

    ``window=0`` keeps the old behaviour: the range only widens, so a few
    early extreme values set it for the whole run.  ``window>0`` keeps the
    last ``window`` observations and recomputes the range from them, so the
    normalised reward tracks the current signal and does not saturate.
    """

    def __init__(self, min_value: float, max_value: float,
                 window: int = 0):
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.window = int(window)
        self._recent: Optional[deque[float]] = (
            deque(maxlen=self.window) if self.window > 0 else None)

    def update(self, value: float) -> None:
        value = float(value)
        if self._recent is not None:
            self._recent.append(value)
            self.min_value = min(self._recent)
            self.max_value = max(self._recent)
            return
        if value < self.min_value:
            self.min_value = value
        if value > self.max_value:
            self.max_value = value

    def high(self, value: float) -> float:
        """Map a high value to a high normalised reward."""
        rng = self.max_value - self.min_value
        if rng <= 1e-9:
            return 1.0
        return float(np.clip((float(value) - self.min_value) / rng,
                            0.0, 1.0))

    def low(self, value: float) -> float:
        """Map a low value to a high normalised reward."""
        rng = self.max_value - self.min_value
        if rng <= 1e-9:
            return 1.0
        return float(np.clip((self.max_value - float(value)) / rng,
                            0.0, 1.0))

    def state_dict(self) -> dict[str, float]:
        return {"min": self.min_value, "max": self.max_value}


class RunningStandardizer:
    """Running mean and standard deviation over a recent window.

    Used to put the two OA reward branches on one scale: each branch's
    "badness" signal is divided by its own running standard deviation, so
    one standard deviation of either branch maps to the same reward change.
    ``window=0`` uses every observation (Welford's online algorithm).
    """

    def __init__(self, window: int = 0):
        self.window = int(window)
        self._recent: Optional[deque[float]] = (
            deque(maxlen=self.window) if self.window > 0 else None)
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._var = 0.0

    def update(self, value: float) -> None:
        value = float(value)
        if self._recent is not None:
            self._recent.append(value)
            values = np.fromiter(self._recent, dtype=np.float64,
                                 count=len(self._recent))
            self._mean = float(values.mean())
            self._var = float(values.var())
            self._count = int(values.size)
            return
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        self._m2 += delta * (value - self._mean)
        self._var = self._m2 / self._count

    def z(self, value: float) -> float:
        std = float(np.sqrt(max(self._var, 0.0)))
        if std <= 1e-9:
            return 0.0
        return (float(value) - self._mean) / std

    def state_dict(self) -> dict[str, float]:
        return {"mean": self._mean, "var": self._var,
                "count": float(self._count)}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self._mean = float(state["mean"])
        self._var = float(state["var"])
        self._count = int(state["count"])
        self._m2 = self._var * self._count
        if self._recent is not None:
            self._recent.clear()


def normalise_branch_reward(standardizer: RunningStandardizer,
                            badness: float) -> float:
    """Map an oriented "badness" signal to [0, 1] on a common scale.

    ``badness`` is positive when the objective component grows.  Zero maps
    to 1 only when it is below the running mean; three standard deviations
    below and above the mean map to 1 and 0.  Both OA branches use the same
    transform, so their reward levels and sensitivities are comparable.
    """
    z = standardizer.z(badness)
    return float(np.clip(0.5 - z / 6.0, 0.0, 1.0))


def compute_gae(rewards: list[float], values: list[float],
                gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation for one terminal trajectory."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last = 0.0
    for t in range(n - 1, -1, -1):
        next_value = values[t + 1] if t + 1 < n else 0.0
        delta = float(rewards[t]) + gamma * next_value - float(values[t])
        last = delta + gamma * lam * last
        advantages[t] = last
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns
