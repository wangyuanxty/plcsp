"""PPO controller for the top-level rule policy (three heads).

The controller selects one job rule, one machine rule, and (when AGV is
enabled) one AGV rule.  The heads are sampled independently.  The action
log-probability and entropy are the sums over the active heads.  The selected
rules are held for ``interval`` low-level decisions, and the controller is
updated from the weighted-cost change of each segment.  Low-level actors are
not modified.

Three credit-assignment variants are supported:

``team``
    One shared team reward.  All heads use the summed log-probability and the
    scalar critic, exactly as before.
``local``
    Each head receives the change of its own local cost.  Each head has its
    own value baseline and its own PPO advantage.
``coma``
    One shared team reward.  A joint-action Q-critic gives each head a
    counterfactual advantage (Foerster et al., AAAI 2018)::

        A_i = Q(s, a) - sum_{a_i'} pi_i(a_i'|s) Q(s, (a_i', a_-i))

"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

CREDIT_MODES = ("team", "local", "coma", "hybrid")


class TopLevelActorCritic(nn.Module):
    def __init__(self, input_dim: int, head_sizes: Sequence[int],
                 hidden_dim: int = 64, credit: str = "team"):
        super().__init__()
        if credit not in CREDIT_MODES:
            raise ValueError(f"unknown credit mode {credit!r}")
        self.credit = credit
        self.head_sizes = tuple(int(n) for n in head_sizes)
        if not self.head_sizes or any(n <= 0 for n in self.head_sizes):
            raise ValueError(f"bad head sizes {self.head_sizes}")
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, n) for n in self.head_sizes])
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        if credit == "local":
            self.local_critic = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, len(self.head_sizes)),
            )
        if credit in ("coma", "hybrid"):
            joint_dim = int(input_dim) + sum(self.head_sizes)
            q_out = len(self.head_sizes) if credit == "hybrid" else 1
            self.q_critic = nn.Sequential(
                nn.Linear(joint_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, q_out),
            )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2 ** 0.5)
                nn.init.constant_(module.bias, 0.0)
        # Near-uniform initial rule policy; the rule bias then moves it.
        for head in self.heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        if self.credit == "local":
            nn.init.orthogonal_(self.local_critic[-1].weight, gain=1.0)
        if self.credit in ("coma", "hybrid"):
            nn.init.orthogonal_(self.q_critic[-1].weight, gain=1.0)

    def forward(self, state: torch.Tensor):
        hidden = self.trunk(state)
        logits = [head(hidden) for head in self.heads]
        value = self.critic(state).squeeze(-1)
        return logits, value

    def joint_encode(self, states: torch.Tensor,
                     actions: torch.Tensor) -> torch.Tensor:
        """State plus one-hot joint action, for the COMA Q-critic."""
        parts = [states]
        for i, n in enumerate(self.head_sizes):
            parts.append(F.one_hot(actions[:, i], num_classes=n).float())
        return torch.cat(parts, dim=-1)


class TopLevelController:
    """Rollout buffer + PPO update for one episode of segment decisions."""

    def __init__(self, input_dim: int = 18,
                 head_sizes: Optional[Sequence[int]] = None,
                 n_actions: Optional[int] = None,
                 hidden_dim: int = 64, lr: float = 1e-3,
                 gamma: float = 0.99, lam: float = 0.95,
                 clip: float = 0.2, epochs: int = 4,
                 entropy_coef: float = 0.01, grad_clip: float = 0.5,
                 credit: str = "team", device="cpu"):
        if head_sizes is None:
            head_sizes = (int(n_actions) if n_actions else 12,)
        if credit not in CREDIT_MODES:
            raise ValueError(f"unknown credit mode {credit!r}")
        self.device = torch.device(device)
        self.credit = credit
        self.head_sizes = tuple(int(n) for n in head_sizes)
        self.n_heads = len(self.head_sizes)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.clip = float(clip)
        self.epochs = max(1, int(epochs))
        self.entropy_coef = float(entropy_coef)
        self.grad_clip = float(grad_clip)
        self.model = TopLevelActorCritic(
            input_dim, self.head_sizes, hidden_dim,
            credit=credit).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # COMA keeps a small cross-episode Q replay.  A top-level episode
        # has only 4-7 decisions per environment, so one episode alone is
        # too noisy to train a joint-action Q-critic.
        self.q_replay: List[tuple] = []
        self.q_replay_max = 32
        self.reset()

    def reset(self) -> None:
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        # Per-head log-probabilities, shape (B, H), so each head can receive
        # its own advantage in the local and COMA modes.
        self.logprobs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []

    @torch.no_grad()
    def act(self, state: torch.Tensor, greedy: bool = False,
            store: bool = True):
        state = state.to(self.device, dtype=torch.float32)
        self.model.eval()
        logits_list, value = self.model(state)
        batch = state.size(0)
        logprob_vec = torch.zeros(
            (batch, self.n_heads), device=self.device)
        actions = []
        for i, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)
            action = (torch.argmax(logits, dim=-1) if greedy
                      else dist.sample())
            actions.append(action)
            logprob_vec[:, i] = dist.log_prob(action)
        action = torch.stack(actions, dim=-1)
        logprob = logprob_vec.sum(dim=-1)
        if self.credit == "local":
            local_values = self.model.local_critic(state)
            value_out = local_values.mean(dim=-1)
            q_values = None
        elif self.credit in ("coma", "hybrid"):
            q_values = self.model.q_critic(
                self.model.joint_encode(state, action))
            local_values = None
            if self.credit == "hybrid":
                value_out = q_values.mean(dim=-1)
            else:
                value_out = q_values.squeeze(-1)
        else:
            local_values = None
            q_values = None
            value_out = value
        if store:
            self.states.append(state.detach())
            self.actions.append(action.detach())
            self.logprobs.append(logprob_vec.detach())
            if self.credit == "local":
                self.values.append(local_values.detach())
            elif self.credit == "hybrid":
                self.values.append(q_values.detach())
            else:
                self.values.append(value_out.detach())
        return action, logprob, value_out

    def _as_batch_vector(self, value, batch: int, name: str) -> torch.Tensor:
        """Validate a per-environment scalar vector, shape (B,)."""
        v = value if torch.is_tensor(value) else torch.as_tensor(value)
        v = v.detach().to(self.device, dtype=torch.float32)
        if v.dim() == 0:
            if batch != 1:
                raise ValueError(
                    f"{name} is scalar but batch size is {batch}")
            return v.view(1)
        v = v.reshape(-1)
        if v.numel() != batch:
            raise ValueError(
                f"{name} must have {batch} entries, got {v.numel()}")
        return v

    def _as_local_matrix(self, value, batch: int,
                         name: str) -> torch.Tensor:
        """Validate a per-head local value, shape (B, H).

        Accepts the natural 1-D forms when they are unambiguous: (H,) for
        B=1 and (B,) for H=1.
        """
        r = value if torch.is_tensor(value) else torch.as_tensor(value)
        r = r.detach().to(self.device, dtype=torch.float32)
        if r.dim() == 0:
            r = r.view(1, 1)
        elif r.dim() == 1:
            if r.numel() == self.n_heads and batch == 1:
                r = r.unsqueeze(0)
            elif self.n_heads == 1 and r.numel() == batch:
                r = r.unsqueeze(-1)
            else:
                raise ValueError(
                    f"{name} must have shape (B, {self.n_heads}), "
                    f"got {tuple(r.shape)} with B={batch}")
        if r.dim() != 2 or tuple(r.shape) != (batch, self.n_heads):
            raise ValueError(
                f"{name} must have shape (B, {self.n_heads}), "
                f"got {tuple(r.shape)} with B={batch}")
        return r

    def close_segment(self, reward) -> None:
        """Attach the reward of the segment that just ended.

        ``team`` and ``coma`` expect a scalar per environment, shape (B,).
        ``local`` expects one value per head, shape (B, H).
        """
        if not self.actions:
            raise RuntimeError("close_segment called before act")
        if len(self.rewards) >= len(self.actions):
            raise RuntimeError(
                "close_segment called twice for the same top-level action")
        batch = self.actions[-1].size(0)
        if self.credit in ("local", "hybrid"):
            r = self._as_local_matrix(reward, batch, "local reward")
        else:
            r = self._as_batch_vector(reward, batch, "segment reward")
        self.rewards.append(r)

    def finish_episode(self, final_cost: Optional[torch.Tensor] = None,
                       final_local: Optional[torch.Tensor] = None
                       ) -> Optional[Dict[str, float]]:
        """Normalise segment rewards, run the credit update, clear buffer."""
        if not self.actions:
            self.reset()
            return None
        if len(self.rewards) != len(self.actions):
            raise RuntimeError(
                f"top-level buffer mismatch: {len(self.actions)} actions, "
                f"{len(self.rewards)} rewards")

        states = torch.stack(self.states, dim=0)       # (T, B, D)
        actions = torch.stack(self.actions, dim=0)     # (T, B, H)
        old_logp = torch.stack(self.logprobs, dim=0)   # (T, B, H)
        rewards = torch.stack(self.rewards, dim=0)     # (T, B) or (T, B, H)

        if self.credit == "local":
            stats = self._finish_local(
                states, actions, old_logp, rewards, final_local)
        elif self.credit == "hybrid":
            stats = self._finish_hybrid(
                states, actions, old_logp, rewards, final_local)
        elif self.credit == "coma":
            stats = self._finish_coma(
                states, actions, old_logp, rewards, final_cost)
        else:
            values = torch.stack(self.values, dim=0)   # (T, B)
            if final_cost is not None:
                cost = self._as_batch_vector(
                    final_cost, states.size(1), "final_cost")
                rewards = rewards / cost.clamp_min(1.0).unsqueeze(0)
            adv, returns = self._gae(rewards, values)
            stats = self._update(
                states, actions, old_logp.sum(dim=-1), adv, returns)
        self.reset()
        return stats

    def _finish_local(self, states, actions, old_logp, rewards,
                      final_local) -> Dict[str, float]:
        if final_local is None:
            raise ValueError("local credit needs final_local")
        values = torch.stack(self.values, dim=0)       # (T, B, H)
        scale = self._as_local_matrix(
            final_local, states.size(1), "final_local").clamp_min(1.0)
        rewards = rewards / scale.unsqueeze(0)
        adv, returns = self._gae_heads(rewards, values)
        return self._update_local(
            states, actions, old_logp, adv, returns)

    def _finish_hybrid(self, states, actions, old_logp, rewards,
                       final_local) -> Dict[str, float]:
        """COMA with per-head local returns."""
        if final_local is None:
            raise ValueError("hybrid credit needs final_local")
        scale = self._as_local_matrix(
            final_local, states.size(1), "final_local").clamp_min(1.0)
        rewards = rewards / scale.unsqueeze(0)
        returns = self._mc_returns_heads(rewards)      # (T, B, H)
        q_loss = self._update_q_local(states, actions, returns)
        adv = self._hybrid_advantages(states, actions)  # (N, H)
        stats = self._update_actor_only(
            states, actions, old_logp, adv)
        stats["top_q_loss"] = q_loss
        stats["top_critic_loss"] = 0.0
        return stats

    def _finish_coma(self, states, actions, old_logp, rewards,
                     final_cost) -> Dict[str, float]:
        if final_cost is None:
            raise ValueError("coma credit needs final_cost")
        cost = self._as_batch_vector(
            final_cost, states.size(1), "final_cost")
        rewards = rewards / cost.clamp_min(1.0).unsqueeze(0)
        returns = self._mc_returns(rewards)            # (T, B)
        q_loss = self._update_q(states, actions, returns)
        adv = self._coma_advantages(states, actions)   # (N, H)
        stats = self._update_actor_only(
            states, actions, old_logp, adv)
        stats["top_q_loss"] = q_loss
        stats["top_critic_loss"] = 0.0
        return stats

    def _gae(self, rewards: torch.Tensor, values: torch.Tensor):
        steps, batch = rewards.shape
        adv = torch.zeros_like(rewards)
        last = torch.zeros(batch, device=self.device)
        for t in reversed(range(steps)):
            next_value = last if t == steps - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value - values[t]
            last = delta + self.gamma * self.lam * last
            adv[t] = last
        returns = adv + values
        flat = adv.reshape(-1)
        std = flat.std(unbiased=False)
        if float(std) > 1e-8:
            adv = (adv - flat.mean()) / (std + 1e-8)
        return adv, returns

    def _gae_heads(self, rewards: torch.Tensor, values: torch.Tensor):
        steps, batch, heads = rewards.shape
        adv = torch.zeros_like(rewards)
        last = torch.zeros((batch, heads), device=self.device)
        for t in reversed(range(steps)):
            next_value = last if t == steps - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value - values[t]
            last = delta + self.gamma * self.lam * last
            adv[t] = last
        returns = adv + values
        for h in range(heads):
            flat = adv[:, :, h].reshape(-1)
            std = flat.std(unbiased=False)
            if float(std) > 1e-8:
                adv[:, :, h] = ((adv[:, :, h] - flat.mean())
                                / (std + 1e-8))
        return adv, returns

    def _mc_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        steps, batch = rewards.shape
        returns = torch.zeros_like(rewards)
        running = torch.zeros(batch, device=self.device)
        for t in reversed(range(steps)):
            running = rewards[t] + self.gamma * running
            returns[t] = running
        return returns

    def _mc_returns_heads(self, rewards: torch.Tensor) -> torch.Tensor:
        steps, batch, heads = rewards.shape
        returns = torch.zeros_like(rewards)
        running = torch.zeros((batch, heads), device=self.device)
        for t in reversed(range(steps)):
            running = rewards[t] + self.gamma * running
            returns[t] = running
        return returns

    def _update(self, states: torch.Tensor, actions: torch.Tensor,
                old_logp: torch.Tensor, adv: torch.Tensor,
                returns: torch.Tensor) -> Dict[str, float]:
        n_heads = len(self.head_sizes)
        states = states.reshape(-1, states.size(-1))
        actions = actions.reshape(-1, n_heads)
        old_logp = old_logp.reshape(-1)
        adv = adv.reshape(-1)
        returns = returns.reshape(-1)

        self.model.train()
        last = {}
        for _ in range(self.epochs):
            logits_list, values = self.model(states)
            new_logp = torch.zeros_like(old_logp)
            entropy = torch.zeros_like(old_logp)
            for i, logits in enumerate(logits_list):
                dist = Categorical(logits=logits)
                new_logp = new_logp + dist.log_prob(actions[:, i])
                entropy = entropy + dist.entropy()
            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip,
                                1.0 + self.clip) * adv
            actor_loss = (-torch.min(surr1, surr2).mean()
                          - self.entropy_coef * entropy.mean())
            critic_loss = F.mse_loss(values, returns)
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            last = {
                "top_actor_loss": float(actor_loss.detach().cpu()),
                "top_critic_loss": float(critic_loss.detach().cpu()),
                "top_entropy": float(entropy.mean().detach().cpu()),
            }
        self.model.eval()
        return last

    def _update_local(self, states: torch.Tensor, actions: torch.Tensor,
                      old_logp: torch.Tensor, adv: torch.Tensor,
                      returns: torch.Tensor) -> Dict[str, float]:
        n_heads = self.n_heads
        states = states.reshape(-1, states.size(-1))
        actions = actions.reshape(-1, n_heads)
        old_logp = old_logp.reshape(-1, n_heads)
        adv = adv.reshape(-1, n_heads)
        returns = returns.reshape(-1, n_heads)

        self.model.train()
        last = {}
        for _ in range(self.epochs):
            logits_list, _ = self.model(states)
            new_logp = []
            entropy = []
            for i, logits in enumerate(logits_list):
                dist = Categorical(logits=logits)
                new_logp.append(dist.log_prob(actions[:, i]))
                entropy.append(dist.entropy())
            new_logp = torch.stack(new_logp, dim=1)
            entropy = torch.stack(entropy, dim=1)
            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip,
                                1.0 + self.clip) * adv
            actor_loss = (-torch.min(surr1, surr2).mean()
                          - self.entropy_coef
                          * entropy.sum(dim=1).mean())
            values = self.model.local_critic(states)
            critic_loss = F.mse_loss(values, returns)
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            last = {
                "top_actor_loss": float(actor_loss.detach().cpu()),
                "top_critic_loss": float(critic_loss.detach().cpu()),
                "top_entropy": float(entropy.mean().detach().cpu()),
            }
        self.model.eval()
        return last

    def _update_q(self, states: torch.Tensor, actions: torch.Tensor,
                  returns: torch.Tensor) -> float:
        n_states = states.reshape(-1, states.size(-1))
        n_actions = actions.reshape(-1, self.n_heads)
        n_returns = returns.reshape(-1)
        encoded = self.model.joint_encode(n_states, n_actions)
        self.q_replay.append((encoded.detach(), n_returns.detach()))
        if len(self.q_replay) > self.q_replay_max:
            self.q_replay = self.q_replay[-self.q_replay_max:]
        all_encoded = torch.cat([item[0] for item in self.q_replay], dim=0)
        all_returns = torch.cat([item[1] for item in self.q_replay], dim=0)
        self.model.train()
        last = 0.0
        for _ in range(max(1, self.epochs)):
            q_values = self.model.q_critic(all_encoded).squeeze(-1)
            loss = F.mse_loss(q_values, all_returns)
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            last = float(loss.detach().cpu())
        self.model.eval()
        return last

    def _update_q_local(self, states: torch.Tensor,
                        actions: torch.Tensor,
                        returns: torch.Tensor) -> float:
        """Per-head Q targets for the hybrid credit mode."""
        n_states = states.reshape(-1, states.size(-1))
        n_actions = actions.reshape(-1, self.n_heads)
        n_returns = returns.reshape(-1, self.n_heads)
        encoded = self.model.joint_encode(n_states, n_actions)
        self.q_replay.append((encoded.detach(), n_returns.detach()))
        if len(self.q_replay) > self.q_replay_max:
            self.q_replay = self.q_replay[-self.q_replay_max:]
        all_encoded = torch.cat([item[0] for item in self.q_replay], dim=0)
        all_returns = torch.cat([item[1] for item in self.q_replay], dim=0)
        self.model.train()
        last = 0.0
        for _ in range(max(1, self.epochs)):
            q_values = self.model.q_critic(all_encoded)   # (N, H)
            loss = F.mse_loss(q_values, all_returns)
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            last = float(loss.detach().cpu())
        self.model.eval()
        return last

    @torch.no_grad()
    def _coma_advantages(self, states: torch.Tensor,
                         actions: torch.Tensor) -> torch.Tensor:
        n_states = states.reshape(-1, states.size(-1))
        n_actions = actions.reshape(-1, self.n_heads)
        self.model.eval()
        logits_list, _ = self.model(n_states)
        q_actual = self.model.q_critic(
            self.model.joint_encode(n_states, n_actions)).squeeze(-1)
        head_adv = []
        for i, logits in enumerate(logits_list):
            probs = torch.softmax(logits, dim=-1)
            q_alts = []
            for a in range(self.head_sizes[i]):
                alt = n_actions.clone()
                alt[:, i] = a
                q_alts.append(self.model.q_critic(
                    self.model.joint_encode(n_states, alt)).squeeze(-1))
            q_alts = torch.stack(q_alts, dim=1)        # (N, n_i)
            baseline = (probs * q_alts).sum(dim=1)
            head_adv.append(q_actual - baseline)
        adv = torch.stack(head_adv, dim=1)             # (N, H)
        for h in range(self.n_heads):
            col = adv[:, h]
            std = col.std(unbiased=False)
            if float(std) > 1e-8:
                adv[:, h] = ((col - col.mean()) / (std + 1e-8))
        return adv

    @torch.no_grad()
    def _hybrid_advantages(self, states: torch.Tensor,
                           actions: torch.Tensor) -> torch.Tensor:
        """Counterfactual advantages from per-head local Q values."""
        n_states = states.reshape(-1, states.size(-1))
        n_actions = actions.reshape(-1, self.n_heads)
        self.model.eval()
        logits_list, _ = self.model(n_states)
        q_actual = self.model.q_critic(
            self.model.joint_encode(n_states, n_actions))  # (N, H)
        head_adv = []
        for i, logits in enumerate(logits_list):
            probs = torch.softmax(logits, dim=-1)
            q_alts = []
            for a in range(self.head_sizes[i]):
                alt = n_actions.clone()
                alt[:, i] = a
                q_alts.append(self.model.q_critic(
                    self.model.joint_encode(
                        n_states, alt))[:, i])         # (N,)
            q_alts = torch.stack(q_alts, dim=1)        # (N, n_i)
            baseline = (probs * q_alts).sum(dim=1)
            head_adv.append(q_actual[:, i] - baseline)
        adv = torch.stack(head_adv, dim=1)             # (N, H)
        for h in range(self.n_heads):
            col = adv[:, h]
            std = col.std(unbiased=False)
            if float(std) > 1e-8:
                adv[:, h] = ((col - col.mean()) / (std + 1e-8))
        return adv

    def _update_actor_only(self, states: torch.Tensor, actions: torch.Tensor,
                           old_logp: torch.Tensor,
                           adv: torch.Tensor) -> Dict[str, float]:
        n_heads = self.n_heads
        states = states.reshape(-1, states.size(-1))
        actions = actions.reshape(-1, n_heads)
        old_logp = old_logp.reshape(-1, n_heads)
        adv = adv.reshape(-1, n_heads)

        self.model.train()
        last = {}
        for _ in range(self.epochs):
            logits_list, _ = self.model(states)
            new_logp = []
            entropy = []
            for i, logits in enumerate(logits_list):
                dist = Categorical(logits=logits)
                new_logp.append(dist.log_prob(actions[:, i]))
                entropy.append(dist.entropy())
            new_logp = torch.stack(new_logp, dim=1)
            entropy = torch.stack(entropy, dim=1)
            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip,
                                1.0 + self.clip) * adv
            actor_loss = (-torch.min(surr1, surr2).mean()
                          - self.entropy_coef
                          * entropy.sum(dim=1).mean())
            loss = actor_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            last = {
                "top_actor_loss": float(actor_loss.detach().cpu()),
                "top_critic_loss": 0.0,
                "top_entropy": float(entropy.mean().detach().cpu()),
            }
        self.model.eval()
        return last

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.model.load_state_dict(state_dict)

    def optimizer_state_dict(self):
        return self.optimizer.state_dict()

    def load_optimizer_state_dict(self, state_dict) -> None:
        self.optimizer.load_state_dict(state_dict)
