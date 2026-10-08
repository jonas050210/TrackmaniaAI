"""Off-policy experience storage.

Sample efficiency is the binding constraint of this project: transitions come from a real
game running in real time (even accelerated), so every transition is expensive and must be
reused many times. A circular buffer of preallocated numpy arrays is the right structure --
no per-insert allocation, O(1) add, and a single vectorised gather per minibatch.

Memory is allocated up front because a buffer that grows during a multi-day run turns into a
fragmentation problem at exactly the wrong moment. Inputs are fully validated before the
cursor advances, so one malformed transition cannot partially corrupt a live slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tmai.agents.base import Batch, Transition


@dataclass
class ReplayBufferConfig:
    capacity: int = 1_000_000
    #: Seed for reproducible sampling. ``None`` uses fresh entropy.
    seed: int | None = None


def _vector(value: Any, *, name: str, expected_dim: int) -> np.ndarray:
    """Convert one transition field after enforcing its vector shape and finite storage."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector with {expected_dim} features") from exc
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {raw.shape}")
    if raw.shape[0] != expected_dim:
        raise ValueError(
            f"{name} has {raw.shape[0]} features, buffer expects {expected_dim}"
        )
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            result = raw.astype(np.float32, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite float32 values")
    return result


def _flag(value: Any, *, name: str) -> bool:
    """Accept boolean or explicit 0/1 scalar flags, rejecting ambiguous arrays/values."""
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar boolean flag")
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    if isinstance(scalar, (int, float, np.integer, np.floating)) and scalar in (0, 1):
        return bool(scalar)
    raise ValueError(f"{name} must be boolean or 0/1, got {scalar!r}")


class ReplayBuffer:
    """Fixed-capacity circular buffer of transitions."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: ReplayBufferConfig | None = None,
    ) -> None:
        self.config = config or ReplayBufferConfig()
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be positive, got {observation_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if self.config.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.config.capacity}")

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        capacity = self.config.capacity

        self._observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self._terminated = np.zeros((capacity,), dtype=np.float32)
        self._truncated = np.zeros((capacity,), dtype=np.float32)
        self._discount_exponents = np.ones((capacity,), dtype=np.float32)

        self._cursor = 0
        self._size = 0
        self._total_added = 0
        self._rng = np.random.default_rng(self.config.seed)

    # -- writing --------------------------------------------------------------------

    def add(self, transition: Transition) -> None:
        """Validate a complete transition, then atomically add it to the ring."""
        obs = _vector(
            transition.observation,
            name="observation",
            expected_dim=self.observation_dim,
        )
        action = _vector(transition.action, name="action", expected_dim=self.action_dim)
        next_obs = _vector(
            transition.next_observation,
            name="next_observation",
            expected_dim=self.observation_dim,
        )
        try:
            reward = float(transition.reward)
            discount_exponent = float(transition.discount_exponent)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("reward and discount_exponent must be numeric scalars") from exc
        if not np.isfinite(reward) or abs(reward) > np.finfo(np.float32).max:
            raise ValueError(f"reward must be finite and representable as float32, got {reward}")
        if (
            not np.isfinite(discount_exponent)
            or discount_exponent < 0.0
            or discount_exponent > np.finfo(np.float32).max
        ):
            raise ValueError(
                "discount_exponent must be finite, non-negative and representable as float32, "
                f"got {discount_exponent}"
            )
        terminated = _flag(transition.terminated, name="terminated")
        truncated = _flag(transition.truncated, name="truncated")

        i = self._cursor
        self._observations[i] = obs
        self._actions[i] = action
        self._rewards[i] = np.float32(reward)
        self._next_observations[i] = next_obs
        self._terminated[i] = float(terminated)
        self._truncated[i] = float(truncated)
        self._discount_exponents[i] = np.float32(discount_exponent)

        self._cursor = (i + 1) % self.config.capacity
        self._size = min(self._size + 1, self.config.capacity)
        self._total_added += 1

    def extend(self, transitions: list[Transition]) -> None:
        for transition in transitions:
            self.add(transition)

    # -- reading --------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    @property
    def total_added(self) -> int:
        """Lifetime insertions, including overwritten ones."""
        return self._total_added

    @property
    def is_full(self) -> bool:
        return self._size >= self.config.capacity

    def sample(self, batch_size: int) -> Batch:
        """Draw a uniform random minibatch."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > self._size:
            raise ValueError(
                f"cannot sample {batch_size} transitions from a buffer holding {self._size}"
            )
        idx = self._rng.integers(0, self._size, size=batch_size)
        return Batch(
            observations=self._observations[idx],
            actions=self._actions[idx],
            rewards=self._rewards[idx],
            next_observations=self._next_observations[idx],
            terminated=self._terminated[idx],
            truncated=self._truncated[idx],
            discount_exponents=self._discount_exponents[idx],
        )

    def snapshot(self) -> Batch:
        """All stored transitions in insertion order (oldest first).

        Deterministic, unlike :meth:`sample`. Intended for inspection, debugging and tests;
        do not use it for training minibatches.
        """
        if self._size == 0:
            empty = lambda shape: np.zeros((0, *shape), dtype=np.float32)  # noqa: E731
            return Batch(
                observations=empty((self.observation_dim,)),
                actions=empty((self.action_dim,)),
                rewards=empty(()),
                next_observations=empty((self.observation_dim,)),
                terminated=empty(()),
                truncated=empty(()),
                discount_exponents=empty(()),
            )
        if self._size < self.config.capacity:
            order = np.arange(self._size)
        else:
            # The buffer has wrapped: the oldest entry is at the cursor.
            order = np.concatenate(
                [np.arange(self._cursor, self.config.capacity), np.arange(self._cursor)]
            )
        return Batch(
            observations=self._observations[order],
            actions=self._actions[order],
            rewards=self._rewards[order],
            next_observations=self._next_observations[order],
            terminated=self._terminated[order],
            truncated=self._truncated[order],
            discount_exponents=self._discount_exponents[order],
        )

    def statistics(self) -> dict[str, float]:
        """Cheap diagnostics; computed over the live region only."""
        if self._size == 0:
            return {"buffer/size": 0.0, "buffer/mean_reward": 0.0, "buffer/terminal_rate": 0.0}
        rewards = self._rewards[: self._size]
        return {
            "buffer/size": float(self._size),
            "buffer/mean_reward": float(rewards.mean()),
            "buffer/terminal_rate": float(self._terminated[: self._size].mean()),
        }


__all__ = ["ReplayBuffer", "ReplayBufferConfig"]
