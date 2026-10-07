"""The neural network policy/value model.

This module is deliberately **algorithm-agnostic**: it knows how to represent a stochastic
policy and value functions, but it contains no RL update rule. The learner
(:class:`tmai.agents.sac.SACLearner`) owns the loss functions, target networks and
optimisers. That separation is the architectural requirement "a neural network serves as the
policy/value model while the RL algorithm is responsible for training it" made concrete, and
it is what lets a different algorithm (PPO, TD3, REDQ) reuse the same model.

The policy is a squashed Gaussian: ``a = tanh(mu(s) + sigma(s) * eps)``. Squashing matters
here because the action space is bounded (steer in ``[-1, 1]``, throttle/brake in ``[0, 1]``)
and the log-probability correction is required for a correct SAC objective.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0
EPSILON = 1e-6


@dataclass
class NetworkConfig:
    """Shape and regularisation of the policy/value network."""

    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    #: LayerNorm before each activation. Helps a lot when observation scales drift.
    layer_norm: bool = False
    #: Whether the critics get their own encoder instead of sharing the actor's.
    separate_encoders: bool = True
    dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _activation(name: str) -> nn.Module:
    table = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"unknown activation {name!r}; expected one of {sorted(table)}")
    return table[key]()


def _mlp(
    in_features: int,
    hidden_sizes: Sequence[int],
    out_features: int,
    *,
    activation: str = "relu",
    layer_norm: bool = False,
    dropout: float = 0.0,
    output_activation: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_features
    for width in hidden_sizes:
        layers.append(nn.Linear(prev, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev = width
    layers.append(nn.Linear(prev, out_features))
    if output_activation:
        layers.append(_activation(activation))
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """State-dependent Gaussian policy with tanh squashing into ``[lo, hi]``.

    The action bounds are applied by an affine map after ``tanh``, so the same network works
    for steer (``[-1, 1]``) and throttle/brake (``[0, 1]``) without special casing.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: NetworkConfig | None = None,
        *,
        action_low: Sequence[float] | None = None,
        action_high: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        self.action_dim = action_dim

        low = torch.tensor(
            [-1.0] * action_dim if action_low is None else list(action_low), dtype=torch.float32
        )
        high = torch.tensor(
            [1.0] * action_dim if action_high is None else list(action_high), dtype=torch.float32
        )
        if torch.any(high <= low):
            raise ValueError("action_high must be strictly greater than action_low")
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_offset", (high + low) / 2.0)

        self.trunk = _mlp(
            obs_dim,
            self.config.hidden_sizes,
            2 * action_dim,
            activation=self.config.activation,
            layer_norm=self.config.layer_norm,
            dropout=self.config.dropout,
        )
        # A slightly optimistic initial policy: zero mean, moderate exploration.
        nn.init.zeros_(self.trunk[-1].bias)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        out = self.trunk(obs)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        """Sample an action; returns ``(action, log_prob)`` in the *bounded* action space."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        normal = torch.zeros_like(mean) if deterministic else torch.randn_like(mean)

        pre_tanh = mean + std * normal
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_offset

        if deterministic:
            # log-prob of the mean under the tanh transform; used for logging only.
            log_prob = torch.zeros(obs.shape[0], 1, device=obs.device, dtype=obs.dtype)
        else:
            log_prob = _gaussian_log_prob(normal, std) - torch.log(
                1.0 - squashed.pow(2) + EPSILON
            )
            # Correction for the affine rescaling into [low, high].
            log_prob = log_prob - torch.log(self.action_scale + EPSILON)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def log_prob(self, obs: Tensor, action: Tensor) -> Tensor:
        """Log-probability of an already-bounded action (used for entropy targets)."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        unit = (action - self.action_offset) / self.action_scale
        unit = torch.clamp(unit, -1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atanh(unit)
        log_prob = _gaussian_log_prob((pre_tanh - mean) / std, std) - torch.log(
            1.0 - unit.pow(2) + EPSILON
        )
        log_prob = log_prob - torch.log(self.action_scale + EPSILON)
        return log_prob.sum(dim=-1, keepdim=True)


def _gaussian_log_prob(normal: Tensor, std: Tensor) -> Tensor:
    """Per-dimension log-density of ``normal * std`` under ``N(0, std^2)``."""
    return -0.5 * (normal.pow(2) + 2.0 * std.log() + math.log(2.0 * math.pi))


class TwinCritic(nn.Module):
    """Two independent Q-functions, as required by clipped double-Q learning."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: NetworkConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        self.q1 = _mlp(
            obs_dim + action_dim,
            self.config.hidden_sizes,
            1,
            activation=self.config.activation,
            layer_norm=self.config.layer_norm,
            dropout=self.config.dropout,
        )
        self.q2 = _mlp(
            obs_dim + action_dim,
            self.config.hidden_sizes,
            1,
            activation=self.config.activation,
            layer_norm=self.config.layer_norm,
            dropout=self.config.dropout,
        )

    def forward(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, obs: Tensor, action: Tensor) -> Tensor:
        return self.q1(torch.cat([obs, action], dim=-1))


class ActorCriticNetwork(nn.Module):
    """The complete policy/value model used by the learner.

    Attributes:
        policy: the :class:`GaussianPolicy`.
        critic: the online :class:`TwinCritic`.
        critic_target: the slowly-updated target copy. Owned by the model so that
            checkpointing captures everything needed to resume training exactly.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: NetworkConfig | None = None,
        *,
        action_low: Sequence[float] | None = None,
        action_high: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        self.config = config or NetworkConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.policy = GaussianPolicy(
            obs_dim,
            action_dim,
            self.config,
            action_low=action_low,
            action_high=action_high,
        )
        self.critic = TwinCritic(obs_dim, action_dim, self.config)
        self.critic_target = TwinCritic(obs_dim, action_dim, self.config)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad_(False)

    # -- inference ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        action, log_prob = self.policy.sample(obs, deterministic=deterministic)
        return action, log_prob

    def q_values(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        return self.critic(obs, action)

    def target_q_values(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        return self.critic_target(obs, action)

    # -- maintenance ----------------------------------------------------------------

    @torch.no_grad()
    def update_targets(self, tau: float) -> None:
        """Polyak-average the online critic into the target critic."""
        if not 0.0 < tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {tau}")
        for target, online in zip(
            self.critic_target.parameters(), self.critic.parameters(), strict=True
        ):
            target.data.mul_(1.0 - tau).add_(tau * online.data)

    def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """``(actor_params, critic_params)`` for the learner's two optimisers."""
        return list(self.policy.parameters()), list(self.critic.parameters())

    def state_snapshot(self) -> dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "config": self.config.to_dict(),
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }


__all__ = [
    "EPSILON",
    "LOG_STD_MAX",
    "LOG_STD_MIN",
    "ActorCriticNetwork",
    "GaussianPolicy",
    "NetworkConfig",
    "TwinCritic",
]
