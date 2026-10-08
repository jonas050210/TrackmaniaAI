"""Explainable, failure-focused training direction over the training-track split.

The Training Director is deliberately a small meta-controller, not another learned policy. It
observes completed *training* episodes and increases sampling frequency for maps on which the
current policy has been unreliable. Every training map keeps a non-zero share, held-out maps
are never considered, and the controller state is checkpointed so a resumed run does not
forget what it has learned about the suite.

The score is episode coverage (the fraction of the full lap covered in that episode), not raw
reward. That keeps the decision tied to driving performance rather than reward-weight tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass
class TrainingDirectorSpec:
    """Settings for adaptive, training-only track focus."""

    enabled: bool = False
    #: EMA weight assigned to each newly completed episode.
    ema_alpha: float = 0.25
    #: Episodes per track before its weakness meaningfully affects sampling.
    warmup_episodes_per_track: int = 3
    #: Extra probability multiplier for a track with zero episode coverage.
    focus_strength: float = 2.0
    #: Hard cap on any one track's relative sampling weight.
    max_weight: float = 3.0
    #: Coverage deduction for an unfinished episode; zero disables the extra penalty.
    failure_penalty: float = 0.15

    def validate(self) -> list[str]:
        if not self.enabled:
            return []
        problems: list[str] = []
        if not isfinite(self.ema_alpha) or not 0.0 < self.ema_alpha <= 1.0:
            problems.append(f"director.ema_alpha must be finite and in (0, 1], got {self.ema_alpha}")
        if self.warmup_episodes_per_track < 1:
            problems.append(
                "director.warmup_episodes_per_track must be >= 1, "
                f"got {self.warmup_episodes_per_track}"
            )
        if not isfinite(self.focus_strength) or self.focus_strength < 0.0:
            problems.append(
                f"director.focus_strength must be finite and non-negative, got {self.focus_strength}"
            )
        if not isfinite(self.max_weight) or self.max_weight < 1.0:
            problems.append(f"director.max_weight must be finite and >= 1, got {self.max_weight}")
        if not isfinite(self.failure_penalty) or not 0.0 <= self.failure_penalty <= 1.0:
            problems.append(
                "director.failure_penalty must be finite and in [0, 1], "
                f"got {self.failure_penalty}"
            )
        return problems


@dataclass
class TrackPerformance:
    """Exponential moving estimate of one training track's episode coverage."""

    episodes: int = 0
    mean_coverage: float = 0.0
    mean_score: float = 0.0
    last_coverage: float = 0.0
    failures: int = 0
    successes: int = 0
    last_end_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "mean_coverage": round(self.mean_coverage, 4),
            "mean_score": round(self.mean_score, 4),
            "last_coverage": round(self.last_coverage, 4),
            "failures": self.failures,
            "successes": self.successes,
            "last_end_reason": self.last_end_reason,
        }


class TrainingDirector:
    """Turn recent training outcomes into bounded sampling priorities.

    This remains completely deterministic for a fixed sequence of episode outcomes. Random
    selection itself is owned by ``MultiTrackEnv`` and uses the environment's seeded RNG.
    """

    STATE_VERSION = 1

    def __init__(self, spec: TrainingDirectorSpec, track_names: list[str]) -> None:
        if not spec.enabled:
            raise ValueError("TrainingDirector requires director.enabled=true")
        self.spec = spec
        self.track_names = tuple(dict.fromkeys(str(name) for name in track_names))
        if not self.track_names:
            raise ValueError("TrainingDirector needs at least one training track")
        self._performance = {name: TrackPerformance() for name in self.track_names}

    def observe_episode(
        self,
        *,
        track: str,
        progress_fraction: float,
        finished: bool,
        end_reason: str,
    ) -> dict[str, float]:
        """Update one track's estimate and return current sampling weights."""
        if track not in self._performance:
            return self.weights()
        coverage = float(progress_fraction)
        if not isfinite(coverage):
            coverage = 0.0
        coverage = min(1.0, max(0.0, coverage))
        if finished:
            coverage = max(coverage, 1.0)
        failed = not finished and str(end_reason) not in {"", "running"}
        score = max(0.0, coverage - (self.spec.failure_penalty if failed else 0.0))
        row = self._performance[track]
        if row.episodes == 0:
            row.mean_coverage = coverage
            row.mean_score = score
        else:
            row.mean_coverage = (
                self.spec.ema_alpha * coverage
                + (1.0 - self.spec.ema_alpha) * row.mean_coverage
            )
            row.mean_score = (
                self.spec.ema_alpha * score
                + (1.0 - self.spec.ema_alpha) * row.mean_score
            )
        row.episodes += 1
        row.last_coverage = coverage
        row.failures += int(failed)
        row.successes += int(finished)
        row.last_end_reason = str(end_reason)
        return self.weights()

    def weights(self) -> dict[str, float]:
        """Prioritize weakly-driven maps without ever removing an easy map from training."""
        result: dict[str, float] = {}
        for name, row in self._performance.items():
            confidence = min(1.0, row.episodes / self.spec.warmup_episodes_per_track)
            weakness = 1.0 - row.mean_score if row.episodes else 0.0
            result[name] = min(
                self.spec.max_weight,
                1.0 + self.spec.focus_strength * confidence * weakness,
            )
        return result

    def summary(self) -> dict[str, float | str]:
        observed = [row for row in self._performance.values() if row.episodes]
        if not observed:
            return {
                "director/observed_tracks": 0.0,
                "director/weakest_score": 0.0,
                "director/failure_rate": 0.0,
                "director/mean_weight": 1.0,
            }
        weakest = min(observed, key=lambda row: row.mean_score)
        total_episodes = sum(row.episodes for row in observed)
        failures = sum(row.failures for row in observed)
        return {
            "director/observed_tracks": float(len(observed)),
            "director/weakest_score": float(weakest.mean_score),
            "director/failure_rate": failures / max(1, total_episodes),
            "director/mean_weight": float(sum(self.weights().values()) / len(self.track_names)),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "strategy": "ema_failure_focused_track_sampling",
            "config": {
                "ema_alpha": self.spec.ema_alpha,
                "warmup_episodes_per_track": self.spec.warmup_episodes_per_track,
                "focus_strength": self.spec.focus_strength,
                "max_weight": self.spec.max_weight,
                "failure_penalty": self.spec.failure_penalty,
            },
            "tracks": {
                name: row.as_dict() for name, row in self._performance.items()
            },
            "sampling_weights": self.weights(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "track_names": list(self.track_names),
            "tracks": {
                name: row.as_dict() for name, row in self._performance.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        version = int(state.get("version", 0))
        if version != self.STATE_VERSION:
            raise ValueError(
                f"training director state version {version} is not compatible with "
                f"{self.STATE_VERSION}"
            )
        saved_names = tuple(str(name) for name in state.get("track_names", []))
        if saved_names != self.track_names:
            raise ValueError(
                "training director track set changed since the checkpoint; "
                f"saved={saved_names}, current={self.track_names}"
            )
        records = state.get("tracks", {})
        for name, row in self._performance.items():
            saved = records.get(name, {})
            row.episodes = max(0, int(saved.get("episodes", 0)))
            coverage = float(saved.get("mean_coverage", 0.0))
            row.mean_coverage = min(1.0, max(0.0, coverage)) if isfinite(coverage) else 0.0
            score = float(saved.get("mean_score", row.mean_coverage))
            row.mean_score = min(1.0, max(0.0, score)) if isfinite(score) else 0.0
            last = float(saved.get("last_coverage", 0.0))
            row.last_coverage = min(1.0, max(0.0, last)) if isfinite(last) else 0.0
            row.failures = max(0, int(saved.get("failures", 0)))
            row.successes = max(0, int(saved.get("successes", 0)))
            row.last_end_reason = str(saved.get("last_end_reason", ""))


__all__ = ["TrackPerformance", "TrainingDirector", "TrainingDirectorSpec"]
