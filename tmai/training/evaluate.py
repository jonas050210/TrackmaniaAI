"""Deterministic policy evaluation.

Evaluation is what a long run is actually judged by, so it is kept strictly separate from
training: the policy runs deterministically (no exploration noise), no gradient steps happen,
and the reported numbers are the ones a human cares about -- did it finish, how far did it
get, and how fast.

``score`` is defined as completed track fraction, with finish time as a tie-break, so that
``best`` checkpoint selection cannot be gamed by an agent that finishes slowly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tmai.agents.base import Learner
from tmai.env.tm_env import TrackmaniaEnv

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    """Outcome of one evaluation episode."""

    steps: int = 0
    total_reward: float = 0.0
    progress: float = 0.0
    progress_fraction: float = 0.0
    finished: bool = False
    race_time: float = 0.0
    end_reason: str = ""
    mean_speed: float = 0.0
    max_speed: float = 0.0
    wall_seconds: float = 0.0
    invalid_finish: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "total_reward": round(self.total_reward, 4),
            "progress": round(self.progress, 2),
            "progress_fraction": round(self.progress_fraction, 4),
            "finished": self.finished,
            "race_time": round(self.race_time, 3),
            "end_reason": self.end_reason,
            "mean_speed": round(self.mean_speed, 3),
            "max_speed": round(self.max_speed, 3),
            "wall_seconds": round(self.wall_seconds, 3),
            "invalid_finish": self.invalid_finish,
        }


@dataclass
class EvaluationReport:
    """Aggregate over several evaluation episodes."""

    episodes: list[EpisodeResult] = field(default_factory=list)
    deterministic: bool = True

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def finish_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.finished) / len(self.episodes)

    @property
    def mean_progress_fraction(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.progress_fraction for e in self.episodes]))

    @property
    def best_race_time(self) -> float | None:
        times = [e.race_time for e in self.episodes if e.finished and e.race_time > 0]
        return min(times) if times else None

    @property
    def mean_race_time(self) -> float | None:
        times = [e.race_time for e in self.episodes if e.finished and e.race_time > 0]
        return float(np.mean(times)) if times else None

    @property
    def score(self) -> float:
        """Track fraction completed, with a bonus for finishing quickly.

        Bounded in ``[0, 2)`` so a single value can drive best-checkpoint selection:
        finishing always beats not finishing, and among finishes the faster one wins.
        """
        if not self.episodes:
            return 0.0
        base = self.mean_progress_fraction
        best = self.best_race_time
        if best is not None and best > 0:
            # Up to +1.0 for a fast finish, decaying with time.
            base += 1.0 / (1.0 + best / 60.0)
        return float(base)

    def metrics(self) -> dict[str, float]:
        out = {
            "eval/episodes": float(self.num_episodes),
            "eval/finish_rate": self.finish_rate,
            "eval/mean_progress_fraction": self.mean_progress_fraction,
            "eval/score": self.score,
            "eval/mean_steps": float(np.mean([e.steps for e in self.episodes])) if self.episodes else 0.0,
            "eval/mean_speed": float(np.mean([e.mean_speed for e in self.episodes])) if self.episodes else 0.0,
        }
        best = self.best_race_time
        if best is not None:
            out["eval/best_race_time"] = best
        mean_time = self.mean_race_time
        if mean_time is not None:
            out["eval/mean_race_time"] = mean_time
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "deterministic": self.deterministic,
            "finish_rate": round(self.finish_rate, 4),
            "mean_progress_fraction": round(self.mean_progress_fraction, 4),
            "best_race_time": self.best_race_time,
            "mean_race_time": self.mean_race_time,
            "score": round(self.score, 4),
            "episodes": [e.as_dict() for e in self.episodes],
        }

    def summary(self) -> str:
        best = self.best_race_time
        return (
            f"finish {self.finish_rate * 100:.0f}% | "
            f"progress {self.mean_progress_fraction * 100:.1f}% | "
            f"best lap {best:.2f}s" if best is not None else
            f"finish {self.finish_rate * 100:.0f}% | "
            f"progress {self.mean_progress_fraction * 100:.1f}% | no finish yet"
        )


def run_episode(
    env: TrackmaniaEnv,
    learner: Learner,
    *,
    max_steps: int,
    deterministic: bool = True,
    seed: int | None = None,
) -> EpisodeResult:
    """Run one evaluation episode and summarise it."""
    observation, info = env.reset(seed=seed)
    result = EpisodeResult()
    speeds: list[float] = []
    started = time.monotonic()

    for _ in range(max_steps):
        action = learner.act(observation, deterministic=deterministic)
        observation, reward, terminated, truncated, info = env.step(action)
        result.steps += 1
        result.total_reward += float(reward)
        speeds.append(float(info.get("speed_forward", 0.0)))
        if terminated or truncated:
            break

    result.progress = float(info.get("progress", 0.0))
    result.progress_fraction = float(info.get("progress_fraction", 0.0))
    result.finished = bool(info.get("finished", False))
    result.race_time = float(info.get("race_time", 0.0))
    result.end_reason = str(info.get("end_reason", ""))
    result.mean_speed = float(np.mean(speeds)) if speeds else 0.0
    result.max_speed = float(np.max(speeds)) if speeds else 0.0
    result.wall_seconds = time.monotonic() - started

    # A "finish" that did not collect the map's checkpoints is not a valid lap.
    total_cp = int(info.get("checkpoint_total", 0) or 0)
    index_cp = int(info.get("checkpoint_index", 0) or 0)
    result.invalid_finish = bool(result.finished and total_cp > 0 and index_cp < total_cp)
    if result.invalid_finish:
        logger.warning(
            "episode reported finished with %d/%d checkpoints -- treating as invalid",
            index_cp,
            total_cp,
        )
    return result


def evaluate_policy(
    env: TrackmaniaEnv,
    learner: Learner,
    *,
    episodes: int = 3,
    max_steps: int = 2000,
    deterministic: bool = True,
    seed: int | None = None,
) -> EvaluationReport:
    """Evaluate ``learner`` over several episodes."""
    report = EvaluationReport(deterministic=deterministic)
    for i in range(episodes):
        episode_seed = None if seed is None else seed + 1000 + i
        episode = run_episode(
            env,
            learner,
            max_steps=max_steps,
            deterministic=deterministic,
            seed=episode_seed,
        )
        report.episodes.append(episode)
        logger.info(
            "eval episode %d/%d: %s",
            i + 1,
            episodes,
            {k: v for k, v in episode.as_dict().items() if k != "steps"},
        )
    return report


__all__ = ["EpisodeResult", "EvaluationReport", "evaluate_policy", "run_episode"]
