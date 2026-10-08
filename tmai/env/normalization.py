"""Online observation normalisation.

Why this exists: the policy is trained across *several* tracks with quite different geometry,
and the fixed ``ObservationScales`` are a reasoned guess rather than a measurement. Real
Trackmania values are not documented either. A network trained on features whose true range
is five times the assumed scale sees saturated inputs and learns slowly or not at all.

Design decision -- **normalise at use time, not at collection time.** The replay buffer keeps
raw observations. Statistics are applied when an observation is fed to the network, both for
action selection and inside ``update()``. If normalisation were baked into the stored data
instead, every improvement in the running statistics would silently invalidate the millions
of transitions already in the buffer. This is the same reasoning behind normalising inside a
``VecNormalize``-style wrapper rather than in the environment.

Statistics use Welford's online algorithm so a multi-day run cannot accumulate catastrophic
cancellation, and they are part of the learner's checkpoint so a resumed run continues with
the statistics it had rather than restarting from zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np

NORMALIZER_VERSION = 1


class RunningNormalizer:
    """Per-feature running mean and standard deviation.

    Args:
        dim: number of features.
        epsilon: added to the standard deviation before dividing, so a constant feature
            normalises to ~0 instead of exploding.
        clip: symmetric clip applied after normalisation. Clipping matters here because a
            single outlier frame (a respawn teleport) would otherwise produce an enormous
            input that dominates a gradient step.
        warmup_steps: until this many updates have been seen, ``std`` is treated as 1. The
            first handful of batches are not representative, and dividing by a near-zero
            provisional standard deviation would inject large transient inputs.
    """

    def __init__(
        self,
        dim: int,
        *,
        epsilon: float = 1e-4,
        clip: float = 10.0,
        warmup_steps: int = 100,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if clip <= 0:
            raise ValueError(f"clip must be positive, got {clip}")
        self.dim = int(dim)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.warmup_steps = int(warmup_steps)

        self._count = 0
        self._mean = np.zeros(dim, dtype=np.float64)
        self._m2 = np.zeros(dim, dtype=np.float64)

    # -- updating -------------------------------------------------------------------

    def update(self, observations: np.ndarray) -> None:
        """Fold a batch of observations into the running statistics.

        The batch is merged in one step with Chan et al.'s parallel update, which gives the same
        mean and variance as folding the rows one at a time (Welford) but without a Python loop
        per row. That loop cost about a millisecond per gradient step on a 256-row batch.
        """
        obs = np.asarray(observations, dtype=np.float64)
        if obs.ndim == 1:
            obs = obs[None, :]
        if obs.shape[-1] != self.dim:
            raise ValueError(f"expected {self.dim} features, got {obs.shape[-1]}")
        if not np.all(np.isfinite(obs)):
            # Never let a NaN into the statistics: it would poison the mean forever.
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        batch = obs.reshape(-1, self.dim)
        batch_count = batch.shape[0]
        if batch_count == 0:
            return

        batch_mean = batch.mean(axis=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(axis=0)
        total = self._count + batch_count
        delta = batch_mean - self._mean
        self._mean = self._mean + delta * (batch_count / total)
        self._m2 = self._m2 + batch_m2 + delta**2 * (self._count * batch_count / total)
        self._count = total

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def std(self) -> np.ndarray:
        """Population standard deviation; 1.0 for features with no observed variance."""
        if self._count < 2:
            return np.ones(self.dim, dtype=np.float64)
        variance = self._m2 / self._count
        return np.sqrt(np.maximum(variance, 0.0))

    @property
    def is_warmed_up(self) -> bool:
        return self._count >= self.warmup_steps

    # -- applying -------------------------------------------------------------------

    def normalize(self, observations: np.ndarray) -> np.ndarray:
        """Normalise one observation or a batch, preserving shape and dtype."""
        obs = np.asarray(observations)
        original_shape = obs.shape
        flat = obs.reshape(-1, self.dim).astype(np.float64)

        # Until warmup completes the provisional std is not trusted, so only centre.
        out = (
            flat - self._mean
            if not self.is_warmed_up
            else (flat - self._mean) / (self.std + self.epsilon)
        )
        out = np.clip(out, -self.clip, self.clip)

        return out.reshape(original_shape).astype(np.float32)

    # -- persistence ----------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": NORMALIZER_VERSION,
            "dim": self.dim,
            "count": self._count,
            "mean": self._mean.tolist(),
            "m2": self._m2.tolist(),
            "epsilon": self.epsilon,
            "clip": self.clip,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        version = int(state.get("version", 0))
        if version != NORMALIZER_VERSION:
            raise ValueError(
                f"normalizer state version {version} is not compatible with {NORMALIZER_VERSION}"
            )
        dim = int(state["dim"])
        if dim != self.dim:
            raise ValueError(f"normalizer state has dim {dim}, expected {self.dim}")
        self._count = int(state["count"])
        self._mean = np.asarray(state["mean"], dtype=np.float64)
        self._m2 = np.asarray(state["m2"], dtype=np.float64)

    def statistics(self) -> dict[str, float]:
        """Cheap diagnostics for the metrics stream."""
        std = self.std
        return {
            "normalizer/count": float(self._count),
            "normalizer/std_mean": float(std.mean()),
            "normalizer/std_max": float(std.max()),
            "normalizer/mean_abs_max": float(np.abs(self._mean).max()),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"RunningNormalizer(dim={self.dim}, count={self._count}, "
            f"warmed_up={self.is_warmed_up})"
        )


__all__ = ["NORMALIZER_VERSION", "RunningNormalizer"]
