"""Wrappers that adapt any :class:`~tmai.agents.base.Learner` without changing it.

These exist so that cross-cutting concerns (observation normalisation, action noise
schedules) are additive rather than baked into SAC. A future PPO implementation gets the
same treatment for free by wrapping it, which is the concrete payoff of the ``Learner``
protocol being the only thing the trainer depends on.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tmai.agents.base import Batch, Learner
from tmai.env.normalization import RunningNormalizer


class NormalizingLearner:
    """Normalises observations on the way into a wrapped learner.

    The environment and the replay buffer continue to see *raw* observations; the normaliser
    is applied only at the two points where an observation meets the network:

    * :meth:`act`, so behaviour uses the same representation the network was trained on.
    * :meth:`update`, for both the observations and the bootstrap targets.

    Doing it here rather than in the environment is what keeps stored transitions valid as
    the statistics improve.

    The wrapped learner's ``state_dict`` is stored alongside the normaliser statistics, so a
    checkpoint of this object restores both and a resumed run does not silently restart with
    unit statistics.
    """

    def __init__(
        self,
        inner: Learner,
        normalizer: RunningNormalizer | None = None,
    ) -> None:
        self.inner = inner
        self.normalizer = normalizer or RunningNormalizer(int(inner.observation_dim))
        if self.normalizer.dim != int(inner.observation_dim):
            raise ValueError(
                f"normalizer dim {self.normalizer.dim} does not match learner observation_dim "
                f"{inner.observation_dim}"
            )

    # -- pass-through attributes ----------------------------------------------------

    @property
    def observation_dim(self) -> int:
        return int(self.inner.observation_dim)

    @property
    def action_dim(self) -> int:
        return int(self.inner.action_dim)

    @property
    def gradient_steps(self) -> int:
        return int(getattr(self.inner, "gradient_steps", 0))

    @property
    def temperature(self) -> float:
        return float(getattr(self.inner, "temperature", 0.0))

    @property
    def device(self) -> Any:
        return getattr(self.inner, "device", None)

    # -- normalisation --------------------------------------------------------------

    def _normalize(self, observations: np.ndarray) -> np.ndarray:
        return self.normalizer.normalize(observations)

    # -- Learner protocol -----------------------------------------------------------

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.inner.act(self._normalize(observation), deterministic=deterministic)

    def update(self, batch: Batch) -> dict[str, float]:
        # Update the statistics from the batch actually being learned from, then normalise
        # both sides of the transition. Rewards and actions are untouched.
        self.normalizer.update(batch.observations)
        normalised = Batch(
            observations=self._normalize(batch.observations),
            actions=batch.actions,
            rewards=batch.rewards,
            next_observations=self._normalize(batch.next_observations),
            terminated=batch.terminated,
            truncated=batch.truncated,
        )
        metrics = self.inner.update(normalised)
        metrics.update(self.normalizer.statistics())
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "normalizer": self.normalizer.state_dict(),
            "inner": self.inner.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "normalizer" in state:
            self.normalizer.load_state_dict(state["normalizer"])
        # Tolerate a raw inner state so checkpoints from an unwrapped learner still load.
        self.inner.load_state_dict(state.get("inner", state))

    def describe(self) -> dict[str, Any]:
        description = self.inner.describe()
        description["observation_normalization"] = {
            "enabled": True,
            "count": self.normalizer.count,
            "clip": self.normalizer.clip,
            "warmed_up": self.normalizer.is_warmed_up,
        }
        return description


__all__ = ["NormalizingLearner"]
