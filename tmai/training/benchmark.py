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

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tmai.config import RunConfig
from tmai.training.checkpoint import latest_checkpoint, load_checkpoint
from tmai.training.evaluate import EvaluationReport, evaluate_tracks

logger = logging.getLogger(__name__)

BENCHMARK_SCHEMA_VERSION = 1


@dataclass
class BenchmarkModel:
    """One model taking part in a benchmark."""

    label: str
    checkpoint: str
    #: Resolved after evaluation.
    reports: dict[str, EvaluationReport] = field(default_factory=dict)

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
            "score": round(self.score, 4),
            "splits": {
                split: report.as_dict() for split, report in self.reports.items()
            },
        }


@dataclass
class BenchmarkReport:
    """The outcome of one benchmark run."""

    name: str
    created_utc: str = ""
    splits: list[str] = field(default_factory=list)
    episodes_per_track: int = 0
    models: list[BenchmarkModel] = field(default_factory=list)
    config_summary: dict[str, Any] = field(default_factory=dict)

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
            "ranking": self.ranking,
            "config": self.config_summary,
            "models": [m.as_dict() for m in self.models],
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
            header += f" {split + ' fin%':>10} {split + ' prog%':>11}"
        lines = [header, "-" * len(header)]
        for model in sorted(self.models, key=lambda m: m.score, reverse=True):
            row = f"{model.label[:24]:<24} {model.score:>7.3f}"
            for split in self.splits:
                report = model.reports.get(split)
                if report is None:
                    row += f" {'--':>10} {'--':>11}"
                else:
                    row += f" {report.finish_rate * 100:>10.0f} {report.mean_progress_fraction * 100:>11.1f}"
            lines.append(row)
        return "\n".join(lines)

    def summary(self) -> str:
        if not self.models:
            return "empty benchmark"
        best = self.ranking[0]
        return f"{len(self.models)} models x {len(self.splits)} splits; best: {best}"


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
    models: list[tuple[str, str | Path]],
    *,
    splits: list[str],
    episodes_per_track: int = 3,
    name: str = "benchmark",
    max_steps: int | None = None,
    deterministic: bool = True,
) -> BenchmarkReport:
    """Evaluate every model on every split and build a comparison report.

    Args:
        config: the run configuration (track library, env, learner shape).
        models: ``(label, checkpoint_or_run_dir)`` pairs.
        splits: library splits to evaluate (``"train"``, ``"validation"``, ``"test"``).
        episodes_per_track: episodes per track per split.
        name: benchmark name, recorded in the report.
    """
    from tmai.training.factory import build_learner, build_library, build_multi_track_env

    config.validate_or_raise()
    library = build_library(config)
    max_steps = max_steps or config.env.termination.max_steps

    report = BenchmarkReport(
        name=name,
        created_utc=datetime.now(timezone.utc).isoformat(),
        splits=list(splits),
        episodes_per_track=episodes_per_track,
        config_summary={
            "driver": config.driver.kind,
            "track_source": config.track.directory or config.track.path
            or config.track.synthetic or "synthetic_suite",
            "seed": config.train.seed,
        },
    )

    # One environment per split, reused across models: the protocol is the same for every
    # model, so the environments must be too.
    envs = {
        split: build_multi_track_env(config, library, split=split, seed=config.train.seed)
        for split in splits
    }
    try:
        for label, target in models:
            checkpoint = _resolve_checkpoint(target)
            logger.info("benchmark model %r: %s", label, checkpoint)
            model = BenchmarkModel(label=label, checkpoint=str(checkpoint))
            payload = load_checkpoint(checkpoint)

            for split, env in envs.items():
                learner = build_learner(env, config)
                learner.load_state_dict(payload["learner"])
                tracks = [(e.track.name, split) for e in library.by_split(split)]
                evaluation = evaluate_tracks(
                    env,
                    learner,
                    tracks=tracks,
                    episodes_per_track=episodes_per_track,
                    max_steps=max_steps,
                    deterministic=deterministic,
                    seed=config.train.seed,
                    label=f"benchmark:{label}:{split}",
                    step=int(payload.get("step", 0)),
                )
                model.reports[split] = evaluation
                logger.info(
                    "  %s on %s: %s", label, split, evaluation.summary()
                )
            report.models.append(model)
    finally:
        for env in envs.values():
            env.close()

    logger.info("benchmark %s: %s", name, report.summary())
    return report


__all__ = ["BenchmarkModel", "BenchmarkReport", "run_benchmark"]
