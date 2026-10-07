"""Off-policy experience storage.

Sample efficiency is the binding constraint of this project: transitions come from a real
game running in real time (even accelerated), so every transition is expensive and must be
reused many times. A circular buffer of preallocated numpy arrays is the right structure --
no per-insert allocation, O(1) add, and a single vectorised gather per minibatch.

Memory is allocated up front because a buffer that grows during a multi-day run turns into a
fragmentation problem at exactly the wrong moment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tmai.agents.base import Batch, Transition


@dataclass
class ReplayBufferConfig:
    capacity: int = 1_000_000
    #: Seed for reproducible sampling. ``None`` uses fresh entropy.
    seed: int | None = None


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

        self._cursor = 0
        self._size = 0
        self._total_added = 0
        self._rng = np.random.default_rng(self.config.seed)

    # -- writing --------------------------------------------------------------------

    def add(self, transition: Transition) -> None:
        obs = np.asarray(transition.observation, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.observation_dim:
            raise ValueError(
                f"observation has {obs.shape[0]} features, buffer expects {self.observation_dim}"
            )
        action = np.asarray(transition.action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(
                f"action has {action.shape[0]} features, buffer expects {self.action_dim}"
            )

        i = self._cursor
        self._observations[i] = obs
        self._actions[i] = action
        self._rewards[i] = np.float32(transition.reward)
        self._next_observations[i] = np.asarray(
            transition.next_observation, dtype=np.float32
        ).reshape(-1)
        self._terminated[i] = 1.0 if transition.terminated else 0.0
        self._truncated[i] = 1.0 if transition.truncated else 0.0

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
