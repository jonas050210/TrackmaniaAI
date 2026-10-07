"""Soft Actor-Critic: the default training algorithm.

Why SAC and not PPO for this project (the requirement was to justify the choice, not to
assume it) -- see ``docs/ALGORITHM.md`` for the full argument. In short:

* **Samples are the scarce resource.** Transitions come from one real game instance running
  in (at best) accelerated real time. PPO is on-policy and discards every transition after a
  handful of epochs; SAC is off-policy and reuses each transition many times. For
  slow-sampling continuous control this is typically one to two orders of magnitude fewer
  environment steps to the same performance.
* **The action space is continuous and bounded** (analog steer/gas), which is exactly the
  setting SAC's squashed-Gaussian policy is designed for. PPO's Gaussian policy ignores the
  bounds and relies on clipping.
* **Maximum entropy gives principled exploration** from scratch, without a hand-tuned
  epsilon schedule, and keeps the policy from collapsing onto a single line early.

The algorithm owns losses, target networks, optimisers and the entropy temperature. The
network (:class:`tmai.models.networks.ActorCriticNetwork`) owns representation only.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from tmai.agents.base import Batch
from tmai.models.networks import ActorCriticNetwork, NetworkConfig

logger = logging.getLogger(__name__)


@dataclass
class SACConfig:
    """SAC hyper-parameters."""

    gamma: float = 0.99
    #: Polyak coefficient for the target critic.
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4
    #: Initial entropy temperature.
    initial_temperature: float = 0.2
    learnable_temperature: bool = True
    #: ``None`` uses the common ``-dim(A)`` heuristic.
    target_entropy: float | None = None
    #: Clip gradient global norm; ``0`` disables.
    max_grad_norm: float = 0.0
    network: NetworkConfig = field(default_factory=NetworkConfig)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "network": self.network.to_dict()}


class SACLearner:
    """Soft Actor-Critic with twin critics and automatic entropy tuning.

    Args:
        observation_dim: width of the observation vector.
        action_dim: width of the action vector.
        config: hyper-parameters.
        action_low / action_high: action bounds; defaults to ``[-1, 1]`` per dimension.
        device: ``"cpu"``, ``"cuda"`` or ``None`` for auto.
        seed: RNG seed for weight initialisation and action sampling.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: SACConfig | None = None,
        *,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        device: str | torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.config = config or SACConfig()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        low = np.full(action_dim, -1.0, dtype=np.float32) if action_low is None else np.asarray(
            action_low, dtype=np.float32
        )
        high = np.full(action_dim, 1.0, dtype=np.float32) if action_high is None else np.asarray(
            action_high, dtype=np.float32
        )

        self.network = ActorCriticNetwork(
            self.observation_dim,
            self.action_dim,
            self.config.network,
            action_low=low,
            action_high=high,
        ).to(self.device)

        actor_params, critic_params = self.network.parameter_groups()
        self.actor_optimizer = optim.Adam(actor_params, lr=self.config.actor_lr)
        self.critic_optimizer = optim.Adam(critic_params, lr=self.config.critic_lr)

        self.target_entropy = float(
            self.config.target_entropy
            if self.config.target_entropy is not None
            else -float(self.action_dim)
        )
        self.log_temperature = torch.tensor(
            float(np.log(max(self.config.initial_temperature, 1e-6))),
            device=self.device,
            requires_grad=self.config.learnable_temperature,
        )
        self.temperature_optimizer = (
            optim.Adam([self.log_temperature], lr=self.config.temperature_lr)
            if self.config.learnable_temperature
            else None
        )

        self._gradient_steps = 0
        self._action_low = low
        self._action_high = high

    # -- inference ------------------------------------------------------------------

    @property
    def temperature(self) -> float:
        return float(self.log_temperature.detach().exp().item())

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Compute an action for one observation ``(dim,)`` or a batch ``(B, dim)``."""
        obs = np.asarray(observation, dtype=np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[None, :]
        if obs.shape[-1] != self.observation_dim:
            raise ValueError(
                f"observation has {obs.shape[-1]} features, learner expects {self.observation_dim}"
            )
        self.network.eval()
        action, _ = self.network.act(
            torch.as_tensor(obs, device=self.device), deterministic=deterministic
        )
        self.network.train()
        out = action.detach().cpu().numpy().astype(np.float32)
        out = np.clip(out, self._action_low, self._action_high)
        return out[0] if single else out

    # -- learning -------------------------------------------------------------------

    def update(self, batch: Batch) -> dict[str, float]:
        """One SAC gradient step over ``batch``; returns scalar metrics."""
        cfg = self.config
        obs = torch.as_tensor(batch.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)[:, None]
        next_obs = torch.as_tensor(
            batch.next_observations, dtype=torch.float32, device=self.device
        )
        terminated = torch.as_tensor(
            batch.terminated, dtype=torch.float32, device=self.device
        )[:, None]

        alpha = self.log_temperature.detach().exp()

        # ---- critic ----
        with torch.no_grad():
            next_actions, next_log_probs = self.network.policy.sample(next_obs)
            next_q1, next_q2 = self.network.target_q_values(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2)
            # Bootstrap through truncations (time limit) but not through true terminals.
            target = rewards + cfg.gamma * (1.0 - terminated) * (next_q - alpha * next_log_probs)

        q1, q2 = self.network.q_values(obs, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = self._clip(self.network.critic.parameters())
        self.critic_optimizer.step()

        # ---- actor ----
        new_actions, log_probs = self.network.policy.sample(obs)
        q1_new, q2_new = self.network.q_values(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = self._clip(self.network.policy.parameters())
        self.actor_optimizer.step()

        # ---- entropy temperature ----
        temperature_loss_value = 0.0
        if self.temperature_optimizer is not None:
            temperature_loss = -(
                self.log_temperature * (log_probs.detach() + self.target_entropy)
            ).mean()
            self.temperature_optimizer.zero_grad(set_to_none=True)
            temperature_loss.backward()
            self.temperature_optimizer.step()
            temperature_loss_value = float(temperature_loss.item())

        self.network.update_targets(cfg.tau)
        self._gradient_steps += 1

        with torch.no_grad():
            return {
                "sac/critic_loss": float(critic_loss.item()),
                "sac/actor_loss": float(actor_loss.item()),
                "sac/temperature_loss": temperature_loss_value,
                "sac/temperature": float(alpha.item()),
                "sac/q1_mean": float(q1.mean().item()),
                "sac/q2_mean": float(q2.mean().item()),
                "sac/q_target_mean": float(target.mean().item()),
                "sac/log_prob_mean": float(log_probs.mean().item()),
                "sac/action_std_mean": float(
                    self.network.policy.forward(obs)[1].exp().mean().item()
                ),
                "sac/critic_grad_norm": critic_grad_norm,
                "sac/actor_grad_norm": actor_grad_norm,
                "sac/gradient_steps": float(self._gradient_steps),
            }

    def _clip(self, parameters) -> float:
        """Optional global-norm gradient clipping; returns the pre-clip norm."""
        params = [p for p in parameters if p.grad is not None]
        if not params:
            return 0.0
        norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm=1e12).item())
        if self.config.max_grad_norm and self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.config.max_grad_norm)
        return norm

    # -- serialisation --------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "network": self.network.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "temperature_optimizer": (
                self.temperature_optimizer.state_dict() if self.temperature_optimizer else None
            ),
            "log_temperature": float(self.log_temperature.detach().item()),
            "gradient_steps": self._gradient_steps,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "config": self.config.to_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("observation_dim", self.observation_dim)) != self.observation_dim:
            raise ValueError(
                f"checkpoint observation_dim={state.get('observation_dim')} does not match "
                f"this learner's {self.observation_dim}"
            )
        if int(state.get("action_dim", self.action_dim)) != self.action_dim:
            raise ValueError(
                f"checkpoint action_dim={state.get('action_dim')} does not match "
                f"this learner's {self.action_dim}"
            )
        self.network.load_state_dict(state["network"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if self.temperature_optimizer is not None and state.get("temperature_optimizer"):
            self.temperature_optimizer.load_state_dict(state["temperature_optimizer"])
        with torch.no_grad():
            self.log_temperature.fill_(float(state.get("log_temperature", 0.0)))
        self._gradient_steps = int(state.get("gradient_steps", 0))
        logger.info("restored learner after %d gradient steps", self._gradient_steps)

    @property
    def gradient_steps(self) -> int:
        return self._gradient_steps

    def describe(self) -> dict[str, Any]:
        return {
            "algorithm": "sac",
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "device": str(self.device),
            "target_entropy": self.target_entropy,
            "gradient_steps": self._gradient_steps,
            "config": self.config.to_dict(),
            "model": self.network.state_snapshot(),
        }


__all__ = ["SACConfig", "SACLearner"]
