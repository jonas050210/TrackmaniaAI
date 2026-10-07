"""The algorithm-facing contract, kept independent of any specific algorithm.

``tmai.agents`` answers "how do we train the model in ``tmai.models``". The trainer only
ever sees :class:`Learner`, which is why swapping SAC for PPO, TD3 or REDQ is an additive
change rather than a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class Transition:
    """One environment interaction, in the layout the replay buffer stores."""

    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    #: True when the task genuinely ended (finished, crashed). Value bootstrapping stops.
    terminated: bool
    #: True when we stopped the episode (time limit). Value bootstrapping continues.
    truncated: bool
    info: dict[str, Any] | None = None


@dataclass
class Batch:
    """A minibatch of transitions, as plain numpy arrays.

    Conversion to tensors happens inside the learner so that the buffer stays
    framework-agnostic and can be shared by a non-torch algorithm.
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray  # float32, 1.0 = terminal
    truncated: np.ndarray

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    @property
    def size(self) -> int:
        return len(self)


@runtime_checkable
class Learner(Protocol):
    """What the trainer needs from any RL algorithm."""

    observation_dim: int
    action_dim: int

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Return an action for a single observation (batch dim optional)."""

    def update(self, batch: Batch) -> dict[str, float]:
        """Perform one gradient update; return scalar metrics for logging."""

    def state_dict(self) -> dict[str, Any]:
        """Everything needed to resume training exactly (weights + optimiser + counters)."""

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`state_dict`."""

    def describe(self) -> dict[str, Any]:
        """Serialisable description for the run manifest."""


__all__ = ["Batch", "Learner", "Transition"]
