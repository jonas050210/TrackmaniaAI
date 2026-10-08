"""Read-only status and history API for a run directory.

This is the data layer a future GUI binds to. It is deliberately *not* a GUI: no rendering,
no framework, no server. It reads the artefacts a training run already writes
(``manifest.json``, ``metrics.jsonl``, ``events.jsonl``, checkpoints) and returns plain
dataclasses that serialise to JSON.

Keeping this separate is what makes the GUI cheap to build later and impossible to get
wrong in a way that corrupts training: the dashboard reads the same files a human would
``tail``, so it can be developed, tested and shipped independently of both the trainer and
the game.

The API surface a dashboard needs:

* :func:`run_status` -- where a run is right now: step, throughput, latest metrics, whether
  it looks stalled or dead.
* :func:`run_history` -- downsampled curves for plotting, with the metric names grouped.
* :func:`run_episodes` -- per-episode outcomes, for a table or a scatter.
* :func:`run_evaluations` -- every evaluation, with train-versus-held-out separated.
* :func:`run_checkpoints` -- what can be resumed or compared.

Everything is JSON-serialisable and free of any GUI or plotting dependency.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tmai.runlog import (
    CONFIG_NAME,
    EVENTS_NAME,
    LOG_NAME,
    MANIFEST_NAME,
    METRICS_NAME,
    read_manifest,
    read_metrics,
)
from tmai.training.checkpoint import list_checkpoints, read_meta

logger = logging.getLogger(__name__)

STATUS_SCHEMA_VERSION = 1

#: Metrics a dashboard shows as headline numbers, in display order.
HEADLINE_METRICS: tuple[str, ...] = (
    "env/progress_fraction",
    "env/speed_forward",
    "reward/total",
    "eval/mean_progress_fraction",
    "eval/finish_rate",
    "heldout/mean_progress_fraction",
    "director/weakest_score",
    "director/failure_rate",
    "director/mean_weight",
    "throughput/env_steps_per_second",
    "buffer/size",
    "learner/temperature",
)

#: Metric prefixes, for grouping curves into panels.
METRIC_GROUPS: tuple[tuple[str, str], ...] = (
    ("env/", "Environment"),
    ("reward/", "Reward"),
    ("sac/", "Learner"),
    ("buffer/", "Replay buffer"),
    ("eval/", "Evaluation"),
    ("heldout/", "Held-out evaluation"),
    ("throughput/", "Throughput"),
    ("normalizer/", "Normalisation"),
    ("curriculum/", "Curriculum"),
    ("director/", "Training Director"),
    ("system/", "System resources"),
)


@dataclass
class RunStatus:
    """Where a run is right now."""

    run_dir: str
    run_name: str = ""
    exists: bool = False
    #: Total configured steps, when the manifest records them.
    total_steps: int | None = None
    #: Last step written to metrics.jsonl.
    step: int = 0
    #: Seconds since the last metrics record. ``None`` when there are none.
    seconds_since_update: float | None = None
    #: Heuristic: no metrics written for this long suggests the run is dead or stuck.
    stale: bool = False
    #: Best-effort completion fraction in ``[0, 1]``.
    progress_fraction: float | None = None
    #: Latest value of each headline metric.
    latest: dict[str, float] = field(default_factory=dict)
    #: Records written so far.
    metrics_records: int = 0
    episodes: int = 0
    evaluations: int = 0
    #: Set when the events log records a terminal event.
    ended: bool = False
    end_reason: str | None = None
    #: Driver name, so a dashboard can label a run that used the toy model as such.
    driver: str | None = None
    #: True when the run used the simulated driver; dashboards should make this loud.
    simulated: bool = False
    checkpoints: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": STATUS_SCHEMA_VERSION, **asdict(self)}


@dataclass
class Series:
    """One plottable curve, optionally downsampled."""

    name: str
    group: str = ""
    steps: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunHistory:
    """Downsampled metric curves for a dashboard."""

    run_dir: str
    #: Maximum points per curve; long runs are decimated, not truncated.
    max_points: int = 800
    series: list[Series] = field(default_factory=list)
    #: Metric names present in the log, grouped by prefix.
    available: dict[str, list[str]] = field(default_factory=dict)
    total_records: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "max_points": self.max_points,
            "total_records": self.total_records,
            "available": self.available,
            "series": [s.as_dict() for s in self.series],
        }

    def get(self, name: str) -> Series | None:
        for curve in self.series:
            if curve.name == name:
                return curve
        return None


@dataclass
class EpisodeRecord:
    """One episode, as recorded by the trainer."""

    episode: int = 0
    step: int = 0
    steps: int = 0
    reward: float = 0.0
    progress: float = 0.0
    max_speed: float = 0.0
    end_reason: str = ""
    finished: bool = False
    game_finished: bool = False
    invalid_finish: bool = False
    race_time: float = 0.0
    track: str = ""
    steps_per_second: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CheckpointRecord:
    """One resumable checkpoint."""

    path: str
    name: str
    step: int = 0
    gradient_steps: int = 0
    best_score: float | None = None
    created_utc: str = ""
    is_best: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# -- reading ------------------------------------------------------------------------


def _read_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / EVENTS_NAME
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            logger.debug("skipping malformed event line in %s", path)
    return out


def _group(metric: str) -> str:
    for prefix, label in METRIC_GROUPS:
        if metric.startswith(prefix):
            return label
    return "Other"


def run_status(run_dir: str | Path, *, stale_after_s: float = 180.0) -> RunStatus:
    """Summarise the current state of a run directory.

    Args:
        run_dir: the run directory.
        stale_after_s: seconds without a metrics record before the run is flagged ``stale``.
            A real game at 1x with a slow policy can legitimately pause for a while, so this
            is a hint for a dashboard, not a verdict.
    """
    run_dir = Path(run_dir)
    status = RunStatus(run_dir=str(run_dir), exists=run_dir.is_dir())
    if not status.exists:
        return status

    metrics = read_metrics(run_dir)
    events = _read_events(run_dir)
    status.metrics_records = len(metrics)

    if metrics:
        status.step = int(metrics[-1].get("step", 0))
        # Latest value of each headline metric, searched backwards: the final record is often
        # an evaluation record carrying only eval keys, so reading just that one would report
        # almost nothing.
        for key in HEADLINE_METRICS:
            for record in reversed(metrics):
                if key in record:
                    with contextlib.suppress(TypeError, ValueError):
                        status.latest[key] = float(record[key])
                    break
        # Staleness comes from the file's mtime, not from the record's own timestamp: a
        # record cannot report how long ago it was written.
        try:
            modified = (run_dir / METRICS_NAME).stat().st_mtime
            status.seconds_since_update = max(0.0, time.time() - modified)
            status.stale = status.seconds_since_update > stale_after_s
        except OSError:  # pragma: no cover - filesystem dependent
            status.seconds_since_update = None

    for event in events:
        name = event.get("event")
        if name == "episode_end":
            status.episodes = max(status.episodes, int(event.get("episode", 0)))
        elif name in ("evaluation", "held_out_evaluation"):
            status.evaluations += 1
        elif name == "training_end":
            status.ended = True
            status.end_reason = "interrupted" if event.get("interrupted") else (
                event.get("failure") or "completed"
            )
        elif name == "training_start":
            status.total_steps = event.get("total_steps")

    if status.total_steps:
        status.progress_fraction = round(min(1.0, status.step / status.total_steps), 4)

    try:
        manifest = read_manifest(run_dir)
    except FileNotFoundError:
        logger.debug("no manifest in %s", run_dir)
    else:
        status.run_name = str(manifest.get("run_name", ""))
        driver = (manifest.get("environment") or {}).get("driver") or {}
        status.driver = driver.get("driver") if isinstance(driver, dict) else None
        config = manifest.get("config") or {}
        status.driver = status.driver or (config.get("driver") or {}).get("kind")
        status.simulated = status.driver == "simulated"
        if status.total_steps is None:
            status.total_steps = (config.get("train") or {}).get("total_steps")

    status.checkpoints = len(list_checkpoints(run_dir))
    return status


def run_history(
    run_dir: str | Path,
    *,
    max_points: int = 800,
    metrics_filter: list[str] | None = None,
) -> RunHistory:
    """Return downsampled curves for every metric in the log.

    Downsampling uses largest-triangle-style bucketing by taking the last value in each
    bucket, which preserves spikes better than averaging and keeps the payload small enough
    to send over a wire on every refresh.
    """
    run_dir = Path(run_dir)
    history = RunHistory(run_dir=str(run_dir), max_points=max_points)
    metrics = read_metrics(run_dir)
    history.total_records = len(metrics)
    if not metrics:
        return history

    # Metric names in first-seen order, excluding the per-record bookkeeping keys.
    names: list[str] = []
    for record in metrics:
        for key in record:
            if key not in ("step", "t") and key not in names:
                names.append(key)
    if metrics_filter:
        wanted = set(metrics_filter)
        names = [n for n in names if n in wanted]

    available: dict[str, list[str]] = {}
    for name in names:
        available.setdefault(_group(name), []).append(name)
    history.available = available

    for name in names:
        pairs = [
            (int(r["step"]), float(r[name]))
            for r in metrics
            if "step" in r and name in r and isinstance(r[name], (int, float))
        ]
        if not pairs:
            continue
        if len(pairs) > max_points:
            last_pair = pairs[-1]
            stride = len(pairs) / max_points
            pairs = [pairs[int(i * stride)] for i in range(max_points)]
            # The last metrics record may contain a different group (e.g. evaluation
            # rather than environment metrics). Append this series' last known point.
            if pairs[-1] != last_pair:
                pairs.append(last_pair)
        history.series.append(
            Series(
                name=name,
                group=_group(name),
                steps=[p[0] for p in pairs],
                values=[round(p[1], 6) for p in pairs],
            )
        )
    return history


def run_episodes(run_dir: str | Path, *, limit: int | None = None) -> list[EpisodeRecord]:
    """Per-episode outcomes from the events log, oldest first."""
    events = _read_events(Path(run_dir))
    out: list[EpisodeRecord] = []
    for event in events:
        if event.get("event") != "episode_end":
            continue
        out.append(
            EpisodeRecord(
                episode=int(event.get("episode", 0)),
                step=int(event.get("step", 0)),
                steps=int(event.get("steps", 0)),
                reward=float(event.get("reward", 0.0)),
                progress=float(event.get("progress", 0.0)),
                max_speed=float(event.get("max_speed", 0.0)),
                end_reason=str(event.get("end_reason", "")),
                finished=bool(event.get("finished", False)),
                game_finished=bool(event.get("game_finished", event.get("finished", False))),
                invalid_finish=bool(event.get("invalid_finish", False)),
                race_time=float(event.get("race_time", 0.0)),
                track=str(event.get("track", "")),
                steps_per_second=float(event.get("steps_per_second", 0.0)),
            )
        )
    return out[-limit:] if limit else out


def run_evaluations(run_dir: str | Path) -> list[dict[str, Any]]:
    """Every evaluation recorded, with training and held-out runs kept distinct."""
    events = _read_events(Path(run_dir))
    out: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") not in ("evaluation", "held_out_evaluation"):
            continue
        report = event.get("report")
        if not isinstance(report, dict):
            # Older runs stored the report as a string; skip rather than misrepresent it.
            continue
        out.append(
            {
                "kind": "held_out" if event["event"] == "held_out_evaluation" else "training",
                "step": int(event.get("step", 0)),
                "report": report,
            }
        )
    return out


def run_checkpoints(run_dir: str | Path) -> list[CheckpointRecord]:
    """Checkpoints available for resume or comparison, newest last."""
    run_dir = Path(run_dir)
    out: list[CheckpointRecord] = []
    for path in list_checkpoints(run_dir):
        try:
            meta = read_meta(path)
        except Exception:  # noqa: BLE001 - a corrupt checkpoint must not break the listing
            logger.warning("could not read checkpoint %s", path)
            continue
        out.append(
            CheckpointRecord(
                path=str(path),
                name=path.name,
                step=int(getattr(meta, "step", 0)),
                gradient_steps=int(getattr(meta, "gradient_steps", 0)),
                best_score=getattr(meta, "best_score", None),
                created_utc=str(getattr(meta, "created_utc", "")),
            )
        )
    best = run_dir / "best.pt"
    if best.is_file():
        out.append(
            CheckpointRecord(
                path=str(best), name=best.name, step=-1, is_best=True
            )
        )
    return out


def run_snapshot(
    run_dir: str | Path, *, max_points: int = 400, episodes: int = 50
) -> dict[str, Any]:
    """One JSON payload containing everything a dashboard needs to render a run page.

    This is the single call a GUI should make on a refresh tick.
    """
    run_dir = Path(run_dir)
    manifest: dict[str, Any] = {}
    with contextlib.suppress(FileNotFoundError):
        manifest = read_manifest(run_dir)
    config_path = run_dir / CONFIG_NAME
    config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else None

    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "status": run_status(run_dir).as_dict(),
        "history": run_history(run_dir, max_points=max_points).as_dict(),
        "episodes": [e.as_dict() for e in run_episodes(run_dir, limit=episodes)],
        "evaluations": run_evaluations(run_dir),
        "checkpoints": [c.as_dict() for c in run_checkpoints(run_dir)],
        "manifest": manifest,
        "config_yaml": config_text,
        "log_tail": _log_tail(run_dir),
    }


def _log_tail(run_dir: Path, *, lines: int = 40) -> list[str]:
    path = run_dir / LOG_NAME
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8").splitlines()[-lines:]
    except OSError:  # pragma: no cover - filesystem dependent
        return []


def list_runs(output_dir: str | Path) -> list[dict[str, Any]]:
    """Summarise every run in a directory, newest first. For a run-picker UI.

    Each row carries both identifiers: ``name`` is the *directory* name (the canonical id
    used by every other run endpoint), while ``run_name`` is the human label from the
    manifest, which can differ when a run was started with ``--run-name``.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []
    runs = []
    for child in sorted(output_dir.iterdir(), reverse=True):
        if child.is_dir() and not child.is_symlink() and (child / MANIFEST_NAME).exists():
            status = run_status(child)
            runs.append(
                {
                    "name": child.name,
                    "run_dir": str(child),
                    "run_name": status.run_name,
                    "step": status.step,
                    "total_steps": status.total_steps,
                    "progress_fraction": status.progress_fraction,
                    "ended": status.ended,
                    "simulated": status.simulated,
                    "driver": status.driver,
                    "checkpoints": status.checkpoints,
                }
            )
    return runs


__all__ = [
    "HEADLINE_METRICS",
    "METRIC_GROUPS",
    "STATUS_SCHEMA_VERSION",
    "CheckpointRecord",
    "EpisodeRecord",
    "RunHistory",
    "RunStatus",
    "Series",
    "list_runs",
    "run_checkpoints",
    "run_episodes",
    "run_evaluations",
    "run_history",
    "run_snapshot",
    "run_status",
]
