"""Benchmarking: evaluate several models across splits and compare them fairly.

``tmai eval`` answers "how good is this checkpoint?". A benchmark answers "which of these
models is best, and where?" -- the same protocol for every model: the same splits, the
same episode count, the same determinism, so differences are the models, not the
measurement.

Every model is evaluated on every requested split with :func:`evaluate_tracks` over the
library, and the report carries both per-split numbers and a ranking by a single score,
so "model A wins on validation but loses on test" is visible rather than averaged away.
"""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tmai.agents.base import Learner
from tmai.agents.curvature import CurvaturePilot
from tmai.config import RunConfig
from tmai.training.checkpoint import latest_checkpoint, load_checkpoint
from tmai.training.evaluate import (
    EpisodeResult,
    EvaluationReport,
    TrackResult,
    evaluate_tracks,
)

logger = logging.getLogger(__name__)

BENCHMARK_SCHEMA_VERSION = 3
_BOOTSTRAP_RESAMPLES = 2_000


@dataclass
class BenchmarkModel:
    """One model taking part in a benchmark."""

    label: str
    checkpoint: str
    #: ``curvature`` for the named heuristic baseline, otherwise ``None``.
    baseline_kind: str | None = None
    #: Resolved after evaluation.
    reports: dict[str, EvaluationReport] = field(default_factory=dict)
    #: Approximate 95% bootstrap intervals per split/metric.
    confidence_intervals: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Mean evaluation score across splits (higher is better)."""
        if not self.reports:
            return 0.0
        return float(sum(r.score for r in self.reports.values()) / len(self.reports))

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "checkpoint": self.checkpoint,
            "baseline_kind": self.baseline_kind,
            "score": round(self.score, 4),
            "splits": {split: report.as_dict() for split, report in self.reports.items()},
            "confidence_intervals": self.confidence_intervals,
        }


@dataclass
class PairedComparison:
    """Head-to-head result for identical track / episode-index pairs."""

    split: str
    model_a: str
    model_b: str
    tracks: int = 0
    episodes: int = 0
    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0
    mean_progress_delta: float = 0.0
    confidence_intervals: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def win_rate_a(self) -> float:
        """Paired win share for model A, giving ties half a win."""
        if not self.episodes:
            return 0.0
        return (self.wins_a + 0.5 * self.ties) / self.episodes

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "tracks": self.tracks,
            "episodes": self.episodes,
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "ties": self.ties,
            "win_rate_a": round(self.win_rate_a, 6),
            "mean_progress_delta": round(self.mean_progress_delta, 6),
            "confidence_intervals": self.confidence_intervals,
        }


@dataclass
class BenchmarkReport:
    """The outcome of one benchmark run."""

    name: str
    created_utc: str = ""
    splits: list[str] = field(default_factory=list)
    episodes_per_track: int = 0
    seed_repeats: int = 1
    evaluation_seeds: list[int] = field(default_factory=list)
    models: list[BenchmarkModel] = field(default_factory=list)
    config_summary: dict[str, Any] = field(default_factory=dict)
    head_to_head: list[PairedComparison] = field(default_factory=list)

    @property
    def ranking(self) -> list[str]:
        """Model labels, best first."""
        return [m.label for m in sorted(self.models, key=lambda m: m.score, reverse=True)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "name": self.name,
            "created_utc": self.created_utc,
            "splits": list(self.splits),
            "episodes_per_track": self.episodes_per_track,
            "seed_repeats": self.seed_repeats,
            "evaluation_seeds": list(self.evaluation_seeds),
            "ranking": self.ranking,
            "config": self.config_summary,
            "models": [m.as_dict() for m in self.models],
            "head_to_head": [comparison.as_dict() for comparison in self.head_to_head],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, default=str), encoding="utf-8")
        return path

    def table(self) -> str:
        """Human-readable comparison table."""
        if not self.models:
            return "(no models)"
        header = f"{'model':<24} {'score':>7}"
        for split in self.splits:
            header += (
                f" {split + ' fin%':>10} {split + ' fin CI95':>17}"
                f" {split + ' prog%':>11} {split + ' prog CI95':>17}"
            )
        lines = [header, "-" * len(header)]
        for model in sorted(self.models, key=lambda m: m.score, reverse=True):
            row = f"{model.label[:24]:<24} {model.score:>7.3f}"
            for split in self.splits:
                report = model.reports.get(split)
                intervals = model.confidence_intervals.get(split, {})
                finish_ci = intervals.get("finish_rate", {}).get("ci95")
                progress_ci = intervals.get("mean_progress_fraction", {}).get("ci95")
                finish_text = f"[{finish_ci[0] * 100:.0f},{finish_ci[1] * 100:.0f}]" if finish_ci else "--"
                progress_text = (
                    f"[{progress_ci[0] * 100:.1f},{progress_ci[1] * 100:.1f}]" if progress_ci else "--"
                )
                if report is None:
                    row += f" {'--':>10} {finish_text:>17} {'--':>11} {progress_text:>17}"
                else:
                    row += (
                        f" {report.finish_rate * 100:>10.0f} {finish_text:>17}"
                        f" {report.mean_progress_fraction * 100:>11.1f} {progress_text:>17}"
                    )
            lines.append(row)
        if self.head_to_head:
            lines.extend(
                [
                    "",
                    "paired head-to-head (same track and episode index)",
                    f"{'split':<12} {'model A':<20} {'model B':<20} {'W-L-T':>9} "
                    f"{'pairs':>7} {'A win% [CI95]':>21} {'Δprog% [CI95]':>21}",
                ]
            )
            for comparison in self.head_to_head:
                win_ci = comparison.confidence_intervals.get("win_rate_a", {}).get("ci95")
                progress_ci = comparison.confidence_intervals.get("mean_progress_delta", {}).get("ci95")
                win_text = (
                    f"{comparison.win_rate_a * 100:.0f}% [{win_ci[0] * 100:.0f},{win_ci[1] * 100:.0f}]"
                    if win_ci
                    else f"{comparison.win_rate_a * 100:.0f}% [--]"
                )
                progress_text = (
                    f"{comparison.mean_progress_delta * 100:+.1f}pp "
                    f"[{progress_ci[0] * 100:+.1f},{progress_ci[1] * 100:+.1f}]"
                    if progress_ci
                    else f"{comparison.mean_progress_delta * 100:+.1f}pp [--]"
                )
                row = (
                    f"{comparison.split:<12} {comparison.model_a[:20]:<20} "
                    f"{comparison.model_b[:20]:<20} "
                    f"{f'{comparison.wins_a}-{comparison.wins_b}-{comparison.ties}':>9} "
                    f"{comparison.episodes:>7} {win_text:>21} {progress_text:>21}"
                )
                lines.append(row)
        return "\n".join(lines)

    def summary(self) -> str:
        if not self.models:
            return "empty benchmark"
        best = self.ranking[0]
        return (
            f"{len(self.models)} models x {len(self.splits)} splits x "
            f"{self.seed_repeats} seed repeats; best: {best}"
        )


def _episode_winner(first: EpisodeResult, second: EpisodeResult) -> int:
    """Compare a paired episode: +1 first, -1 second, 0 tie.

    A valid finish always beats a non-finish. Among finishes, lower race time wins; among
    non-finishes, more lap coverage wins, with off-track excursions as a conservative final
    tie-breaker. Tiny numeric differences are treated as ties.
    """
    first_finished = first.finished and not first.invalid_finish
    second_finished = second.finished and not second.invalid_finish
    if first_finished != second_finished:
        return 1 if first_finished else -1
    if first_finished and second_finished:
        time_delta = first.race_time - second.race_time
        if abs(time_delta) > 0.02:
            return 1 if time_delta < 0.0 else -1
    progress_delta = first.progress_fraction - second.progress_fraction
    if abs(progress_delta) > 1e-4:
        return 1 if progress_delta > 0.0 else -1
    if first.off_track_events != second.off_track_events:
        return 1 if first.off_track_events < second.off_track_events else -1
    return 0


def _stable_seed(seed: int, *parts: str) -> int:
    """Derive a process-independent random seed for a bootstrap statistic."""
    payload = "\0".join([str(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _track_cluster_id(track: TrackResult) -> str:
    if track.family:
        family = " ".join(unicodedata.normalize("NFKC", track.family).split()).casefold()
        return f"family:{family}"
    return f"track:{track.track}"


def _bootstrap_summary(
    groups: Sequence[Sequence[float]],
    *,
    seed: int,
    cluster_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return a percentile 95% interval with cluster-aware resampling.

    Each input group contains episodes from one track. When family IDs are supplied, tracks
    from the same family are resampled together. The cluster bootstrap is weighted by the
    number of tracks in a family, so its estimate remains the track-weighted mean. A single
    independent cluster can only provide an episode-level interval; one observation has none.
    """
    if cluster_ids is not None and len(cluster_ids) != len(groups):
        raise ValueError("cluster_ids must have one entry per metric group")

    clusters: dict[str, list[np.ndarray]] = {}
    total_episodes = 0
    for index, group in enumerate(groups):
        values = np.asarray(group, dtype=np.float64).reshape(-1)
        if not len(values) or not np.all(np.isfinite(values)):
            continue
        key = cluster_ids[index] if cluster_ids is not None else f"track:{index}"
        clusters.setdefault(key, []).append(values)
        total_episodes += len(values)

    if len(clusters) >= 2:
        cluster_keys = list(clusters)
        cluster_sizes = np.asarray([len(clusters[key]) for key in cluster_keys], dtype=np.float64)
        cluster_means = np.asarray(
            [np.mean([float(track.mean()) for track in clusters[key]]) for key in cluster_keys],
            dtype=np.float64,
        )
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(cluster_keys), size=(_BOOTSTRAP_RESAMPLES, len(cluster_keys)))
        estimates = np.empty(_BOOTSTRAP_RESAMPLES, dtype=np.float64)
        for row_index, row in enumerate(draws):
            counts = np.bincount(row, minlength=len(cluster_keys))
            weights = counts * cluster_sizes
            estimates[row_index] = float(np.average(cluster_means, weights=weights))
        lower, upper = np.quantile(estimates, [0.025, 0.975])
        unit = (
            "families"
            if any(len(clusters[key]) > 1 or key.startswith("family:") for key in cluster_keys)
            else "tracks"
        )
        return {
            "ci95": [round(float(lower), 6), round(float(upper), 6)],
            "resampling_unit": unit,
            "sample_count": len(cluster_keys),
        }

    if len(clusters) == 1:
        key, track_groups = next(iter(clusters.items()))
        samples = np.concatenate(track_groups)
        if len(samples) >= 2:
            rng = np.random.default_rng(seed)
            draws = rng.choice(samples, size=(_BOOTSTRAP_RESAMPLES, len(samples)), replace=True)
            lower, upper = np.quantile(draws.mean(axis=1), [0.025, 0.975])
            unit = "episodes_within_family" if key.startswith("family:") else "episodes"
            return {
                "ci95": [round(float(lower), 6), round(float(upper), 6)],
                "resampling_unit": unit,
                "sample_count": int(len(samples)),
            }

    return {
        "ci95": None,
        "resampling_unit": None,
        "sample_count": total_episodes,
    }


def _evaluation_confidence_intervals(
    report: EvaluationReport,
    *,
    seed: int,
    model_label: str,
    split: str,
) -> dict[str, dict[str, Any]]:
    crash_reasons = {
        "crash",
        "out_of_bounds",
        "fell_off",
        "off_track",
        "stalled",
        "no_ground_contact",
        "wrong_way",
        "game_error",
    }
    populated_tracks = [track for track in report.tracks if track.episodes]
    cluster_ids = [_track_cluster_id(track) for track in populated_tracks]
    progress = [[episode.progress_fraction for episode in track.episodes] for track in populated_tracks]
    finish = []
    crash = []
    for track in populated_tracks:
        valid = [episode for episode in track.episodes if not episode.invalid_finish]
        # Match TrackResult.finish_rate: a track with only invalid finishes contributes zero,
        # rather than disappearing from the held-out population.
        finish.append([float(episode.finished) for episode in valid] if valid else [0.0])
        crash.append([float(episode.end_reason in crash_reasons) for episode in track.episodes])

    metrics = {
        "finish_rate": finish,
        "mean_progress_fraction": progress,
        "crash_rate": crash,
    }
    return {
        metric: _bootstrap_summary(
            groups,
            seed=_stable_seed(seed, model_label, split, metric),
            cluster_ids=cluster_ids,
        )
        for metric, groups in metrics.items()
    }


def _merge_evaluations(
    reports: Sequence[EvaluationReport], *, families: Mapping[str, str | None] | None = None
) -> EvaluationReport:
    """Combine seed-repeat reports while retaining a per-track episode grouping."""
    if not reports:
        raise ValueError("cannot merge an empty sequence of evaluation reports")
    tracks: dict[tuple[str, str], TrackResult] = {}
    for report in reports:
        for source in report.tracks:
            key = (source.track, source.split)
            target = tracks.setdefault(
                key,
                TrackResult(
                    track=source.track,
                    split=source.split,
                    family=(families or {}).get(source.track, source.family),
                ),
            )
            target.episodes.extend(source.episodes)
    first = reports[0]
    return EvaluationReport(
        tracks=list(tracks.values()),
        deterministic=all(report.deterministic for report in reports),
        label=first.label,
        step=first.step,
    )


def _paired_comparisons(models: Sequence[BenchmarkModel], splits: Sequence[str]) -> list[PairedComparison]:
    """Compare matching track/episode pairs, preserving the controlled seed pairing."""
    comparisons: list[PairedComparison] = []
    for split in splits:
        for first_index, first_model in enumerate(models):
            for second_model in models[first_index + 1 :]:
                first_report = first_model.reports.get(split)
                second_report = second_model.reports.get(split)
                if first_report is None or second_report is None:
                    continue
                second_tracks = {track.track: track for track in second_report.tracks}
                comparison = PairedComparison(
                    split=split,
                    model_a=first_model.label,
                    model_b=second_model.label,
                )
                progress_deltas: list[float] = []
                win_outcomes_by_track: list[list[float]] = []
                progress_by_track: list[list[float]] = []
                paired_cluster_ids: list[str] = []
                for first_track in first_report.tracks:
                    second_track = second_tracks.get(first_track.track)
                    if second_track is None:
                        continue
                    episode_pairs = list(zip(first_track.episodes, second_track.episodes, strict=False))
                    if not episode_pairs:
                        continue
                    comparison.tracks += 1
                    paired_cluster_ids.append(_track_cluster_id(first_track))
                    track_outcomes: list[float] = []
                    track_progress: list[float] = []
                    for first_episode, second_episode in episode_pairs:
                        comparison.episodes += 1
                        delta = first_episode.progress_fraction - second_episode.progress_fraction
                        progress_deltas.append(delta)
                        track_progress.append(delta)
                        winner = _episode_winner(first_episode, second_episode)
                        if winner > 0:
                            comparison.wins_a += 1
                            track_outcomes.append(1.0)
                        elif winner < 0:
                            comparison.wins_b += 1
                            track_outcomes.append(0.0)
                        else:
                            comparison.ties += 1
                            track_outcomes.append(0.5)
                    win_outcomes_by_track.append(track_outcomes)
                    progress_by_track.append(track_progress)
                if comparison.episodes:
                    comparison.mean_progress_delta = float(np.mean(progress_deltas))
                    seed = _stable_seed(0, split, comparison.model_a, comparison.model_b)
                    comparison.confidence_intervals = {
                        "win_rate_a": _bootstrap_summary(
                            win_outcomes_by_track,
                            seed=seed,
                            cluster_ids=paired_cluster_ids,
                        ),
                        "mean_progress_delta": _bootstrap_summary(
                            progress_by_track,
                            seed=seed ^ 0x5A17,
                            cluster_ids=paired_cluster_ids,
                        ),
                    }
                    comparisons.append(comparison)
    return comparisons


def _resolve_checkpoint(target: str | Path) -> Path:
    """A checkpoint file, or the latest checkpoint of a run directory."""
    path = Path(target)
    if path.is_file():
        return path
    if path.is_dir():
        latest = latest_checkpoint(path)
        if latest is not None:
            return latest
        best = path / "best.pt"
        if best.is_file():
            return best
        raise FileNotFoundError(f"no checkpoint found in run directory {path}")
    raise FileNotFoundError(f"checkpoint not found: {path}")


def run_benchmark(
    config: RunConfig,
    models: Sequence[tuple[str, str | Path]],
    *,
    splits: Sequence[str],
    episodes_per_track: int = 3,
    name: str = "benchmark",
    max_steps: int | None = None,
    deterministic: bool = True,
    seed_repeats: int = 3,
) -> BenchmarkReport:
    """Evaluate every model on every split and build a comparison report.

    Args:
        config: the run configuration (track library, env, learner shape).
        models: ``(label, checkpoint_or_run_dir)`` pairs.
        splits: library splits to evaluate (``"train"``, ``"validation"``, ``"test"``).
        episodes_per_track: episodes per track, per split and per seed repeat.
        name: benchmark name, recorded in the report.
        seed_repeats: paired evaluation seeds; 1 is a quick smoke benchmark, while 3 or more
            is a more useful comparison when the environment randomises start conditions.
    """
    from tmai.training.factory import build_learner, build_library, build_multi_track_env

    config.validate_or_raise()
    if not models:
        raise ValueError("benchmark needs at least one model")
    normalized_models = [(label.strip(), target) for label, target in models]
    labels = [label for label, _ in normalized_models]
    if any(not label for label in labels):
        raise ValueError("benchmark model labels must be non-empty")
    if len(set(labels)) != len(labels):
        raise ValueError("benchmark model labels must be unique")
    if not splits:
        raise ValueError("benchmark needs at least one split")
    if len(set(splits)) != len(splits):
        raise ValueError("benchmark splits must be unique")
    if episodes_per_track < 1:
        raise ValueError(f"episodes_per_track must be >= 1, got {episodes_per_track}")
    if not 1 <= seed_repeats <= 100:
        raise ValueError(f"seed_repeats must be in [1, 100], got {seed_repeats}")
    seed_stride = max(10_000, episodes_per_track + 1)
    evaluation_seeds = [int(config.train.seed) + repeat * seed_stride for repeat in range(seed_repeats)]
    library = build_library(config)
    max_steps = config.env.termination.max_steps if max_steps is None else max_steps
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")

    report = BenchmarkReport(
        name=name,
        created_utc=datetime.now(timezone.utc).isoformat(),
        splits=list(splits),
        episodes_per_track=episodes_per_track,
        seed_repeats=seed_repeats,
        evaluation_seeds=evaluation_seeds,
        config_summary={
            "driver": config.driver.kind,
            "track_source": config.track.directory
            or config.track.path
            or config.track.synthetic
            or "synthetic_suite",
            "seed": config.train.seed,
            "evaluation_seeds": evaluation_seeds,
            "seed_repeats": seed_repeats,
            "random_start_station": config.multi.random_start_station,
            "start_lateral_std": config.multi.start_lateral_std,
        },
    )

    # One environment per split, reused across models: the protocol is the same for every
    # model, so the environments must be too. Construct inside the try so a later split
    # failing to build still closes any earlier real-game connections.
    envs: dict[str, Any] = {}
    try:
        for split in splits:
            envs[split] = build_multi_track_env(config, library, split=split, seed=config.train.seed)
        for label, target in normalized_models:
            target_text = str(target)
            baseline_kind: str | None = None
            payload: dict[str, Any]
            if target_text.startswith("baseline:"):
                baseline_kind = target_text.partition(":")[2].strip().lower()
                if baseline_kind != "curvature":
                    raise ValueError(
                        f"unknown benchmark baseline {target_text!r}; available: baseline:curvature"
                    )
                checkpoint_text = target_text
                payload = {"step": 0}
                logger.info("benchmark model %r: heuristic %s", label, baseline_kind)
            else:
                checkpoint = _resolve_checkpoint(target)
                checkpoint_text = str(checkpoint)
                payload = load_checkpoint(checkpoint)
                logger.info("benchmark model %r: %s", label, checkpoint)
            model = BenchmarkModel(
                label=label,
                checkpoint=checkpoint_text,
                baseline_kind=baseline_kind,
            )

            for split, env in envs.items():
                learner: Learner
                if baseline_kind == "curvature":
                    learner = CurvaturePilot(
                        config.env.observation,
                        action_dim=int(env.action_space.low.size),
                    )
                else:
                    learner = build_learner(env, config)
                    learner.load_state_dict(payload["learner"])
                split_entries = library.by_split(split)
                tracks = [(entry.track.name, split) for entry in split_entries]
                families = {entry.track.name: entry.family for entry in split_entries}
                repeat_reports = [
                    evaluate_tracks(
                        env,
                        learner,
                        tracks=tracks,
                        episodes_per_track=episodes_per_track,
                        max_steps=max_steps,
                        deterministic=deterministic,
                        seed=evaluation_seed,
                        label=f"benchmark:{label}:{split}:seed={evaluation_seed}",
                        step=int(payload.get("step", 0)),
                    )
                    for evaluation_seed in evaluation_seeds
                ]
                evaluation = _merge_evaluations(repeat_reports, families=families)
                model.reports[split] = evaluation
                model.confidence_intervals[split] = _evaluation_confidence_intervals(
                    evaluation,
                    seed=int(config.train.seed),
                    model_label=label,
                    split=split,
                )
                logger.info(
                    "  %s on %s (%d seed repeats): %s",
                    label,
                    split,
                    seed_repeats,
                    evaluation.summary(),
                )
            report.models.append(model)
    finally:
        for env in envs.values():
            env.close()

    report.head_to_head = _paired_comparisons(report.models, report.splits)
    logger.info("benchmark %s: %s", name, report.summary())
    return report


__all__ = ["BenchmarkModel", "BenchmarkReport", "PairedComparison", "run_benchmark"]
