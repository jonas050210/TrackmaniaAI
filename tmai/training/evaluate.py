"""Evaluation across tracks, with machine-readable output.

Evaluation is what a long run is judged by, so it is kept strictly separate from training:
the policy runs deterministically, no gradient steps happen, and the reported numbers are the
ones a person actually cares about -- did it finish, how far did it get, how fast, and how
reliably.

Three things this module is careful about:

**Train versus unseen tracks are reported separately.** A single blended number hides the only
result that matters for generalisation. Every report carries a per-track breakdown and a
per-split summary, so a policy that is excellent on training maps and useless elsewhere is
obvious rather than averaged away.

**Consistency is measured, not assumed.** Mean progress over three episodes where one finished
and two crashed is a misleading headline. Reports therefore carry the standard deviation, the
worst episode, and the distribution of end reasons.

**A "finish" is only a finish if the map's checkpoints were collected.** The game's own
checkpoint counter is authoritative; an episode that crosses the line without them is flagged
``invalid_finish`` and excluded from lap-time statistics.

Output is JSON-serialisable throughout so results can be diffed between runs, ingested by a
dashboard, or compared by ``tmai compare``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tmai.agents.base import Learner
from tmai.env.tm_env import TrackmaniaEnv

logger = logging.getLogger(__name__)

EVAL_SCHEMA_VERSION = 2


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
    #: Track the episode ran on; empty for single-track evaluation.
    track: str = ""
    #: Number of times the episode left the drivable corridor, as a crash proxy.
    off_track_events: int = 0
    #: Metres of corridor excursion accumulated over the episode.
    off_track_metres: float = 0.0

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
            "track": self.track,
            "off_track_events": self.off_track_events,
            "off_track_metres": round(self.off_track_metres, 3),
        }


@dataclass
class TrackResult:
    """Aggregate over the episodes run on one track."""

    track: str
    split: str = ""
    episodes: list[EpisodeResult] = field(default_factory=list)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def finish_rate(self) -> float:
        valid = [e for e in self.episodes if not e.invalid_finish]
        if not valid:
            return 0.0
        return sum(1 for e in valid if e.finished) / len(valid)

    @property
    def mean_progress_fraction(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.progress_fraction for e in self.episodes]))

    @property
    def std_progress_fraction(self) -> float:
        """Spread across episodes. High values mean the policy is not reliable here."""
        if len(self.episodes) < 2:
            return 0.0
        return float(np.std([e.progress_fraction for e in self.episodes]))

    @property
    def worst_progress_fraction(self) -> float:
        if not self.episodes:
            return 0.0
        return float(min(e.progress_fraction for e in self.episodes))

    @property
    def best_race_time(self) -> float | None:
        times = [
            e.race_time
            for e in self.episodes
            if e.finished and not e.invalid_finish and e.race_time > 0
        ]
        return min(times) if times else None

    @property
    def mean_race_time(self) -> float | None:
        times = [
            e.race_time
            for e in self.episodes
            if e.finished and not e.invalid_finish and e.race_time > 0
        ]
        return float(np.mean(times)) if times else None

    @property
    def crash_rate(self) -> float:
        """Fraction of episodes that ended in something other than finishing or time limit.

        A proxy for "drove badly": off-track, stalled, airborne. Truncation by time limit is
        not a crash, it just means the lap is long.
        """
        if not self.episodes:
            return 0.0
        bad = {"off_track", "stalled", "no_ground_contact", "game_error"}
        return sum(1 for e in self.episodes if e.end_reason in bad) / len(self.episodes)

    def as_dict(self) -> dict[str, Any]:
        best = self.best_race_time
        return {
            "track": self.track,
            "split": self.split,
            "episodes": self.num_episodes,
            "finish_rate": round(self.finish_rate, 4),
            "mean_progress_fraction": round(self.mean_progress_fraction, 4),
            "std_progress_fraction": round(self.std_progress_fraction, 4),
            "worst_progress_fraction": round(self.worst_progress_fraction, 4),
            "crash_rate": round(self.crash_rate, 4),
            "best_race_time": round(best, 3) if best is not None else None,
            "mean_race_time": round(self.mean_race_time, 3) if self.mean_race_time else None,
            "episodes_detail": [e.as_dict() for e in self.episodes],
        }


@dataclass
class EvaluationReport:
    """Aggregate over every evaluated track and episode."""

    #: Per-track results, in evaluation order.
    tracks: list[TrackResult] = field(default_factory=list)
    deterministic: bool = True
    #: Label for whatever was evaluated (a checkpoint path, a run name).
    label: str = ""
    #: Environment step the policy was at, when known.
    step: int | None = None

    # -- construction ---------------------------------------------------------------

    @staticmethod
    def from_episodes(
        episodes: list[EpisodeResult],
        *,
        deterministic: bool = True,
        label: str = "",
        step: int | None = None,
    ) -> EvaluationReport:
        """Build a report from a flat list of episodes, grouping by track.

        Episodes without a track name land in a single ``"unknown"`` bucket, which keeps
        single-track usage (and tests) simple without giving up the per-track breakdown.
        """
        buckets: dict[str, TrackResult] = {}
        for episode in episodes:
            name = episode.track or "unknown"
            buckets.setdefault(name, TrackResult(track=name)).episodes.append(episode)
        return EvaluationReport(
            tracks=list(buckets.values()),
            deterministic=deterministic,
            label=label,
            step=step,
        )

    # -- flattening -----------------------------------------------------------------

    @property
    def episodes(self) -> list[EpisodeResult]:
        return [e for track in self.tracks for e in track.episodes]

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def num_tracks(self) -> int:
        return len(self.tracks)

    # -- aggregates -----------------------------------------------------------------

    @property
    def finish_rate(self) -> float:
        episodes = [e for e in self.episodes if not e.invalid_finish]
        if not episodes:
            return 0.0
        return sum(1 for e in episodes if e.finished) / len(episodes)

    @property
    def mean_progress_fraction(self) -> float:
        """Mean of per-track means, so every track counts equally regardless of episode count."""
        if not self.tracks:
            return 0.0
        return float(np.mean([t.mean_progress_fraction for t in self.tracks]))

    @property
    def crash_rate(self) -> float:
        if not self.tracks:
            return 0.0
        return float(np.mean([t.crash_rate for t in self.tracks]))

    @property
    def consistency(self) -> float:
        """Mean within-track standard deviation of progress. Lower is more reliable."""
        if not self.tracks:
            return 0.0
        return float(np.mean([t.std_progress_fraction for t in self.tracks]))

    @property
    def best_race_time(self) -> float | None:
        times = [t.best_race_time for t in self.tracks if t.best_race_time is not None]
        return min(times) if times else None

    @property
    def mean_race_time(self) -> float | None:
        times = [t.mean_race_time for t in self.tracks if t.mean_race_time is not None]
        return float(np.mean(times)) if times else None

    def by_split(self) -> dict[str, dict[str, Any]]:
        """Per-split summary -- the train-versus-unseen comparison in one place."""
        out: dict[str, dict[str, Any]] = {}
        for track in self.tracks:
            split = track.split or "unspecified"
            bucket = out.setdefault(
                split,
                {"tracks": 0, "episodes": 0, "finish_rate": [], "progress": [], "crash": []},
            )
            bucket["tracks"] += 1
            bucket["episodes"] += track.num_episodes
            bucket["finish_rate"].append(track.finish_rate)
            bucket["progress"].append(track.mean_progress_fraction)
            bucket["crash"].append(track.crash_rate)
        return {
            split: {
                "tracks": data["tracks"],
                "episodes": data["episodes"],
                "finish_rate": round(float(np.mean(data["finish_rate"])), 4),
                "mean_progress_fraction": round(float(np.mean(data["progress"])), 4),
                "crash_rate": round(float(np.mean(data["crash"])), 4),
            }
            for split, data in out.items()
        }

    @property
    def generalization_gap(self) -> float | None:
        """Train progress minus held-out progress. Positive means overfitting to seen maps.

        ``None`` when either side is missing, which is the honest answer rather than 0.
        """
        splits = self.by_split()
        if "train" not in splits:
            return None
        held = [s for s in ("validation", "test") if s in splits]
        if not held:
            return None
        train = splits["train"]["mean_progress_fraction"]
        unseen = float(np.mean([splits[s]["mean_progress_fraction"] for s in held]))
        return round(train - unseen, 4)

    @property
    def score(self) -> float:
        """Single scalar for best-checkpoint selection, bounded in ``[0, 2)``.

        Completed track fraction, plus up to 1.0 for finishing quickly. Finishing always beats
        not finishing, and among finishes the faster one wins.
        """
        if not self.tracks:
            return 0.0
        base = self.mean_progress_fraction
        best = self.best_race_time
        if best is not None and best > 0:
            base += 1.0 / (1.0 + best / 60.0)
        return float(base)

    # -- serialisation --------------------------------------------------------------

    def metrics(self, *, prefix: str = "eval") -> dict[str, float]:
        """Flat scalar metrics for the training metrics stream."""
        out = {
            f"{prefix}/tracks": float(self.num_tracks),
            f"{prefix}/episodes": float(self.num_episodes),
            f"{prefix}/finish_rate": self.finish_rate,
            f"{prefix}/mean_progress_fraction": self.mean_progress_fraction,
            f"{prefix}/crash_rate": self.crash_rate,
            f"{prefix}/consistency_std": self.consistency,
            f"{prefix}/score": self.score,
        }
        best = self.best_race_time
        if best is not None:
            out[f"{prefix}/best_race_time"] = best
        mean_time = self.mean_race_time
        if mean_time is not None:
            out[f"{prefix}/mean_race_time"] = mean_time
        gap = self.generalization_gap
        if gap is not None:
            out[f"{prefix}/generalization_gap"] = gap
        for split, data in self.by_split().items():
            out[f"{prefix}/{split}_progress"] = data["mean_progress_fraction"]
            out[f"{prefix}/{split}_finish_rate"] = data["finish_rate"]
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVAL_SCHEMA_VERSION,
            "label": self.label,
            "step": self.step,
            "deterministic": self.deterministic,
            "num_tracks": self.num_tracks,
            "num_episodes": self.num_episodes,
            "finish_rate": round(self.finish_rate, 4),
            "mean_progress_fraction": round(self.mean_progress_fraction, 4),
            "crash_rate": round(self.crash_rate, 4),
            "consistency_std": round(self.consistency, 4),
            "best_race_time": self.best_race_time,
            "mean_race_time": self.mean_race_time,
            "score": round(self.score, 4),
            "generalization_gap": self.generalization_gap,
            "by_split": self.by_split(),
            "tracks": [t.as_dict() for t in self.tracks],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, default=str)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    def summary(self) -> str:
        best = self.best_race_time
        lap = f"best lap {best:.2f}s" if best is not None else "no finish yet"
        gap = self.generalization_gap
        gap_text = f" | gap {gap:+.3f}" if gap is not None else ""
        return (
            f"{self.num_tracks} tracks, {self.num_episodes} episodes | "
            f"finish {self.finish_rate * 100:.0f}% | "
            f"progress {self.mean_progress_fraction * 100:.1f}% | "
            f"crash {self.crash_rate * 100:.0f}% | {lap}{gap_text}"
        )

    def table(self) -> str:
        """Human-readable per-track table."""
        if not self.tracks:
            return "(no episodes)"
        header = (
            f"{'track':<22} {'split':<11} {'fin%':>5} {'prog%':>6} {'std':>6} "
            f"{'worst%':>7} {'crash%':>7} {'best lap':>9}"
        )
        lines = [header, "-" * len(header)]
        for track in self.tracks:
            best = track.best_race_time
            lines.append(
                f"{track.track[:22]:<22} {(track.split or '-'):<11} "
                f"{track.finish_rate * 100:>5.0f} "
                f"{track.mean_progress_fraction * 100:>6.1f} "
                f"{track.std_progress_fraction * 100:>6.1f} "
                f"{track.worst_progress_fraction * 100:>7.1f} "
                f"{track.crash_rate * 100:>7.0f} "
                f"{(f'{best:.2f}s' if best is not None else '--'):>9}"
            )
        lines.append("-" * len(header))
        lines.append(f"{'TOTAL':<22} {'':<11} {self.finish_rate * 100:>5.0f} "
                     f"{self.mean_progress_fraction * 100:>6.1f} "
                     f"{self.consistency * 100:>6.1f} {'':>7} "
                     f"{self.crash_rate * 100:>7.0f}")
        return "\n".join(lines)


# -- running episodes ---------------------------------------------------------------


def run_episode(
    env: TrackmaniaEnv,
    learner: Learner,
    *,
    max_steps: int,
    deterministic: bool = True,
    seed: int | None = None,
    track_name: str = "",
    split: str = "",
) -> EpisodeResult:
    """Run one evaluation episode and summarise it."""
    observation, info = env.reset(seed=seed)
    result = EpisodeResult(track=track_name or str(info.get("track", "")))
    speeds: list[float] = []
    started = time.monotonic()

    was_off_track = False
    for _ in range(max_steps):
        action = learner.act(observation, deterministic=deterministic)
        observation, reward, terminated, truncated, info = env.step(action)
        result.steps += 1
        result.total_reward += float(reward)
        speeds.append(float(info.get("speed_forward", 0.0)))

        # Count off-track *excursions*, not steps: consecutive off-track steps are one event.
        off_track = float(info.get("reward/off_track", 0.0)) < 0.0
        if off_track:
            result.off_track_metres += abs(float(info.get("lateral_offset", 0.0)))
            if not was_off_track:
                result.off_track_events += 1
        was_off_track = off_track

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
    label: str = "",
    step: int | None = None,
    track_name: str = "",
    split: str = "",
) -> EvaluationReport:
    """Evaluate ``learner`` over several episodes on one environment.

    This is the single-track path, and also what the trainer calls mid-run. Multi-track
    evaluation goes through :func:`evaluate_tracks`, which reuses this function per track so
    there is exactly one implementation of "run an episode".
    """
    track = TrackResult(
        track=track_name or getattr(env.track, "name", "unknown"), split=split
    )
    for i in range(episodes):
        episode_seed = None if seed is None else seed + 1000 + i
        episode = run_episode(
            env,
            learner,
            max_steps=max_steps,
            deterministic=deterministic,
            seed=episode_seed,
            track_name=track.track,
            split=split,
        )
        track.episodes.append(episode)
        logger.info(
            "eval episode %d/%d on %s: progress %.1f%%, %s",
            i + 1,
            episodes,
            track.track,
            episode.progress_fraction * 100,
            episode.end_reason,
        )
    return EvaluationReport(
        tracks=[track], deterministic=deterministic, label=label, step=step
    )


def evaluate_tracks(
    env,
    learner: Learner,
    *,
    tracks: list[tuple[str, str]] | None = None,
    episodes_per_track: int = 2,
    max_steps: int = 2000,
    deterministic: bool = True,
    seed: int | None = None,
    label: str = "",
    step: int | None = None,
) -> EvaluationReport:
    """Evaluate across many tracks using a :class:`~tmai.env.multi_track.MultiTrackEnv`.

    Args:
        env: a ``MultiTrackEnv``.
        tracks: explicit ``(track_name, split)`` pairs to evaluate exhaustively, in order.
            When ``None``, episodes are drawn from the environment's sampler instead, which is
            what cheap mid-training evaluation wants.
        episodes_per_track: episodes per track.

    Track selection uses :meth:`MultiTrackEnv.select_track`, so there is exactly one code path
    that runs an episode (:func:`run_episode`) whether or not tracks were pinned.
    """
    report = EvaluationReport(deterministic=deterministic, label=label, step=step)

    if tracks is None:
        buckets: dict[str, TrackResult] = {}
        for i in range(max(1, episodes_per_track)):
            episode_seed = None if seed is None else seed + 5000 + i
            episode = run_episode(
                env,
                learner,
                max_steps=max_steps,
                deterministic=deterministic,
                seed=episode_seed,
            )
            bucket = buckets.setdefault(
                episode.track, TrackResult(track=episode.track, split="sampled")
            )
            bucket.episodes.append(episode)
        report.tracks = list(buckets.values())
        return report

    for name, split in tracks:
        bucket = TrackResult(track=name, split=split)
        for i in range(episodes_per_track):
            episode_seed = None if seed is None else seed + 7000 + i
            env.select_track(name)
            bucket.episodes.append(
                run_episode(
                    env,
                    learner,
                    max_steps=max_steps,
                    deterministic=deterministic,
                    seed=episode_seed,
                    track_name=name,
                    split=split,
                )
            )
        report.tracks.append(bucket)
    return report


__all__ = [
    "EVAL_SCHEMA_VERSION",
    "EpisodeResult",
    "EvaluationReport",
    "TrackResult",
    "evaluate_policy",
    "evaluate_tracks",
    "run_episode",
]
