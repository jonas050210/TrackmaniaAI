"""Episode replays and ghost comparison.

A *replay* is a decimated trajectory of one episode: positions, speeds, actions, rewards
and arc-length progress, plus the outcome (finish, crash, ...). The trainer records them
during training (``train.record_replays``) so any episode can be inspected, visualised and
compared after the fact -- the raw material for "why did the car do that?".

A *ghost* is the same structure recorded from a human (a demonstration lap, or any
recorded trajectory with timestamps). Comparing an AI replay against a ghost answers the
racing question directly: where is the AI losing time?

Everything here is plain data and JSON -- no game, no torch -- so replay analysis works on
any machine that can read the run directory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tmai.tracks.centerline import CenterlineTrack

logger = logging.getLogger(__name__)

REPLAY_SCHEMA_VERSION = 1
REPLAY_DIR_NAME = "replays"


@dataclass
class EpisodeReplay:
    """One episode's decimated trajectory."""

    episode: int
    step: int
    track: str = ""
    split: str = ""
    end_reason: str = ""
    finished: bool = False
    race_time: float = 0.0
    total_reward: float = 0.0
    progress_fraction: float = 0.0
    #: World positions, (N, 3), metres.
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    #: Forward speeds, (N,), m/s.
    speeds: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Actions, (N, 3).
    actions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    #: Per-sample rewards, (N,).
    rewards: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Arc-length progress at each sample, (N,), metres.
    progress: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Race time at each sample, (N,), seconds.
    race_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: ``"training"``, ``"evaluation"`` or ``"human"`` (a ghost).
    source: str = "training"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float64).reshape(-1, 3)
        self.speeds = np.asarray(self.speeds, dtype=np.float64).reshape(-1)
        self.actions = np.asarray(self.actions, dtype=np.float64).reshape(-1, 3)
        self.rewards = np.asarray(self.rewards, dtype=np.float64).reshape(-1)
        self.progress = np.asarray(self.progress, dtype=np.float64).reshape(-1)
        self.race_times = np.asarray(self.race_times, dtype=np.float64).reshape(-1)
        n = len(self.positions)
        for name in ("speeds", "actions", "rewards", "progress", "race_times"):
            if getattr(self, name).shape[0] not in (0, n):
                raise ValueError(
                    f"{name} has {getattr(self, name).shape[0]} entries, positions has {n}"
                )

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_samples(self) -> int:
        return len(self)

    @property
    def duration(self) -> float:
        return float(self.race_times[-1] - self.race_times[0]) if len(self) > 1 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "episode": self.episode,
            "step": self.step,
            "track": self.track,
            "split": self.split,
            "end_reason": self.end_reason,
            "finished": self.finished,
            "race_time": round(self.race_time, 4),
            "total_reward": round(self.total_reward, 4),
            "progress_fraction": round(self.progress_fraction, 4),
            "source": self.source,
            "num_samples": self.num_samples,
            "positions": self.positions.tolist(),
            "speeds": self.speeds.tolist(),
            "actions": self.actions.tolist(),
            "rewards": self.rewards.tolist(),
            "progress": self.progress.tolist(),
            "race_times": self.race_times.tolist(),
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict()), encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path) -> EpisodeReplay:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"replay not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        version = int(data.get("schema_version", 0))
        if version != REPLAY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported replay schema version {version}; this build reads "
                f"{REPLAY_SCHEMA_VERSION}"
            )
        return EpisodeReplay(
            episode=int(data.get("episode", 0)),
            step=int(data.get("step", 0)),
            track=str(data.get("track", "")),
            split=str(data.get("split", "")),
            end_reason=str(data.get("end_reason", "")),
            finished=bool(data.get("finished", False)),
            race_time=float(data.get("race_time", 0.0)),
            total_reward=float(data.get("total_reward", 0.0)),
            progress_fraction=float(data.get("progress_fraction", 0.0)),
            positions=np.asarray(data.get("positions", []), dtype=np.float64),
            speeds=np.asarray(data.get("speeds", []), dtype=np.float64),
            actions=np.asarray(data.get("actions", []), dtype=np.float64),
            rewards=np.asarray(data.get("rewards", []), dtype=np.float64),
            progress=np.asarray(data.get("progress", []), dtype=np.float64),
            race_times=np.asarray(data.get("race_times", []), dtype=np.float64),
            source=str(data.get("source", "training")),
            metadata=dict(data.get("metadata") or {}),
        )

    @staticmethod
    def from_demonstration(demo, *, track: str = "", source: str = "human") -> EpisodeReplay:
        """Turn a recorded human demonstration into a ghost replay."""
        from tmai.training.demos import Demonstration

        if not isinstance(demo, Demonstration):
            raise TypeError(f"expected a Demonstration, got {type(demo).__name__}")
        positions = (
            demo.positions
            if demo.positions is not None
            else np.zeros((len(demo), 3), dtype=np.float64)
        )
        return EpisodeReplay(
            episode=0,
            step=0,
            track=track or str(demo.metadata.get("track", "")),
            end_reason=str(demo.metadata.get("end_reason", "")),
            finished=bool(demo.metadata.get("finished", False)),
            race_time=float(demo.race_times[-1]) if demo.race_times is not None and len(demo) else 0.0,
            total_reward=float(demo.rewards.sum()) if demo.rewards is not None else 0.0,
            positions=positions,
            speeds=demo.speeds if demo.speeds is not None else np.zeros(len(demo)),
            actions=demo.actions,
            rewards=demo.rewards if demo.rewards is not None else np.zeros(len(demo)),
            race_times=demo.race_times if demo.race_times is not None else np.zeros(len(demo)),
            source=source,
            metadata=dict(demo.metadata),
        )


class ReplayRecorder:
    """Accumulates one episode's per-step data; decimates it into an EpisodeReplay."""

    def __init__(self, *, decimation: int = 1) -> None:
        if decimation < 1:
            raise ValueError(f"decimation must be >= 1, got {decimation}")
        self.decimation = int(decimation)
        self._positions: list[np.ndarray] = []
        self._speeds: list[float] = []
        self._actions: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._progress: list[float] = []
        self._race_times: list[float] = []
        self._steps = 0

    def reset(self) -> None:
        self._positions.clear()
        self._speeds.clear()
        self._actions.clear()
        self._rewards.clear()
        self._progress.clear()
        self._race_times.clear()
        self._steps = 0

    @property
    def steps(self) -> int:
        return self._steps

    def record(
        self,
        *,
        position: np.ndarray,
        speed: float,
        action: np.ndarray,
        reward: float,
        progress: float,
        race_time: float,
    ) -> None:
        self._steps += 1
        if self._steps % self.decimation != 0 and self._steps > 1:
            return
        self._positions.append(np.asarray(position, dtype=np.float64).reshape(3).copy())
        self._speeds.append(float(speed))
        self._actions.append(np.asarray(action, dtype=np.float64).reshape(-1)[:3].copy())
        self._rewards.append(float(reward))
        self._progress.append(float(progress))
        self._race_times.append(float(race_time))

    def build(
        self,
        *,
        episode: int,
        step: int,
        track: str = "",
        split: str = "",
        end_reason: str = "",
        finished: bool = False,
        race_time: float = 0.0,
        total_reward: float = 0.0,
        progress_fraction: float = 0.0,
        source: str = "training",
        metadata: dict[str, Any] | None = None,
    ) -> EpisodeReplay:
        return EpisodeReplay(
            episode=episode,
            step=step,
            track=track,
            split=split,
            end_reason=end_reason,
            finished=finished,
            race_time=race_time,
            total_reward=total_reward,
            progress_fraction=progress_fraction,
            positions=np.asarray(self._positions, dtype=np.float64).reshape(-1, 3),
            speeds=np.asarray(self._speeds, dtype=np.float64),
            actions=np.asarray(self._actions, dtype=np.float64).reshape(-1, 3),
            rewards=np.asarray(self._rewards, dtype=np.float64),
            progress=np.asarray(self._progress, dtype=np.float64),
            race_times=np.asarray(self._race_times, dtype=np.float64),
            source=source,
            metadata=dict(metadata or {}),
        )


class ReplayStore:
    """A directory of replay files: save with a cap, list, load."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    @property
    def path(self) -> Path:
        return self.directory

    def _ensure(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, replay: EpisodeReplay, *, max_replays: int = 0) -> Path:
        """Write ``episode_<n>.json``; prune the oldest when ``max_replays`` is exceeded."""
        self._ensure()
        path = self.directory / f"episode_{replay.episode:06d}.json"
        replay.save(path)
        if max_replays > 0:
            files = sorted(self.directory.glob("episode_*.json"))
            for old in files[:-max_replays]:
                try:
                    old.unlink()
                except OSError:  # pragma: no cover - best effort
                    logger.warning("could not prune replay %s", old)
        return path

    def list(self) -> list[dict[str, Any]]:
        """Metadata for every replay in the directory, oldest first."""
        if not self.directory.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for path in sorted(self.directory.glob("episode_*.json")):
            try:
                replay = EpisodeReplay.load(path)
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning("skipping unreadable replay %s: %s", path, exc)
                continue
            out.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "episode": replay.episode,
                    "step": replay.step,
                    "track": replay.track,
                    "split": replay.split,
                    "end_reason": replay.end_reason,
                    "finished": replay.finished,
                    "race_time": replay.race_time,
                    "total_reward": replay.total_reward,
                    "progress_fraction": replay.progress_fraction,
                    "source": replay.source,
                    "num_samples": replay.num_samples,
                }
            )
        return out

    def load(self, name_or_path: str | Path) -> EpisodeReplay:
        path = Path(name_or_path)
        if not path.is_absolute() and not path.exists():
            path = self.directory / path.name
        return EpisodeReplay.load(path)

    def latest(self) -> EpisodeReplay | None:
        files = sorted(self.directory.glob("episode_*.json")) if self.directory.is_dir() else []
        return EpisodeReplay.load(files[-1]) if files else None


# -- ghost comparison ----------------------------------------------------------------


@dataclass
class ReplayComparison:
    """AI replay versus ghost replay, compared station by station along the track.

    Two gap series are reported:

    * ``gaps`` -- the *station* gap: how much race time each replay had spent when it
      reached the station. This is the racing-ghost view, but it assumes both replays
      started at the same place, so with random start stations it is only meaningful over
      the overlap.
    * ``segment_gaps`` -- the *segment* gap: the difference in time each replay took to
      drive from one station to the next. This is start-station invariant, so it is the
      honest "where is the AI losing time?" answer even for a partial lap.

    The headline statistics (``mean_gap`` and friends) use the segment gaps.
    """

    #: Arc-length stations compared, metres.
    stations: np.ndarray
    #: AI race time at each station, seconds.
    ai_times: np.ndarray
    #: Ghost race time at each station, seconds.
    ghost_times: np.ndarray
    #: ``ai_time - ghost_time`` per station; positive means the AI is behind.
    gaps: np.ndarray
    #: Per-segment time difference, ``(ai_dt - ghost_dt)`` between consecutive stations.
    segment_gaps: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ai_finished: bool = False
    ghost_finished: bool = False
    ai_race_time: float = 0.0
    ghost_race_time: float = 0.0
    track: str = ""

    @property
    def mean_gap(self) -> float:
        gaps = self.segment_gaps if len(self.segment_gaps) else self.gaps
        return float(np.nanmean(gaps)) if len(gaps) else 0.0

    @property
    def max_gap(self) -> float:
        gaps = self.segment_gaps if len(self.segment_gaps) else self.gaps
        return float(np.nanmax(gaps)) if len(gaps) else 0.0

    @property
    def min_gap(self) -> float:
        gaps = self.segment_gaps if len(self.segment_gaps) else self.gaps
        return float(np.nanmin(gaps)) if len(gaps) else 0.0

    @property
    def race_time_delta(self) -> float | None:
        """AI race time minus ghost race time; positive means the AI is slower."""
        if not (self.ai_finished and self.ghost_finished):
            return None
        return float(self.ai_race_time - self.ghost_race_time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "track": self.track,
            "stations": self.stations.tolist(),
            "ai_times": self.ai_times.tolist(),
            "ghost_times": self.ghost_times.tolist(),
            "gaps": self.gaps.tolist(),
            "segment_gaps": self.segment_gaps.tolist(),
            "mean_gap": round(self.mean_gap, 4),
            "max_gap": round(self.max_gap, 4),
            "min_gap": round(self.min_gap, 4),
            "ai_finished": self.ai_finished,
            "ghost_finished": self.ghost_finished,
            "ai_race_time": round(self.ai_race_time, 4),
            "ghost_race_time": round(self.ghost_race_time, 4),
            "race_time_delta": (
                round(self.race_time_delta, 4)
                if self.race_time_delta is not None
                else None
            ),
        }

    def summary(self) -> str:
        delta = self.race_time_delta
        delta_text = f"{delta:+.3f}s" if delta is not None else "no finish comparison"
        return (
            f"{len(self.stations)} stations | mean segment gap {self.mean_gap:+.3f}s | "
            f"worst {self.max_gap:+.3f}s | best {self.min_gap:+.3f}s | {delta_text}"
        )


def _time_at_stations(
    progress: np.ndarray, times: np.ndarray, stations: np.ndarray
) -> np.ndarray:
    """Interpolate a replay's race time at each station (progress must be increasing)."""
    progress = np.asarray(progress, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if len(progress) < 2 or len(times) != len(progress):
        return np.full(len(stations), np.nan)
    order = np.argsort(progress, kind="stable")
    progress = progress[order]
    times = times[order]
    return np.interp(stations, progress, times, left=np.nan, right=np.nan)


def compare_replays(
    ai: EpisodeReplay,
    ghost: EpisodeReplay,
    *,
    track: CenterlineTrack | None = None,
    station_spacing: float = 5.0,
) -> ReplayComparison:
    """Compare an AI replay against a ghost, station by station.

    The comparison is over arc length: for a grid of stations along the track, how much
    race time had each replay spent when it *reached* that station? The station difference
    is where the AI sits on the clock; the *segment* difference (time per station pair)
    is where the AI actually loses or gains time, and is invariant to where each replay
    started. Only the arc length both replays drove is compared, so an early crash
    shortens the comparison instead of faking gaps over the rest of the track.

    Args:
        ai: the AI's replay.
        ghost: the reference replay (a human demonstration, another model, ...).
        track: when the replays carry no progress array, project their positions onto
            this track to obtain it.
        station_spacing: metres between compared stations.
    """
    def _progress_of(replay: EpisodeReplay) -> np.ndarray:
        progress = np.asarray(replay.progress, dtype=np.float64)
        if len(progress) >= 2 or track is None:
            return progress
        # No usable progress array: project the trajectory onto the track instead. This is
        # what makes a human demonstration (positions + times, no progress) comparable.
        return np.array(
            [track.project(p).progress for p in replay.positions], dtype=np.float64
        )

    ai_progress = _progress_of(ai)
    ghost_progress = _progress_of(ghost)

    if len(ai_progress) < 2 or len(ghost_progress) < 2:
        raise ValueError(
            "cannot compare replays without arc-length progress; pass the track so it "
            "can be projected from positions"
        )

    start = max(float(ai_progress.min()), float(ghost_progress.min()))
    end = min(float(ai_progress.max()), float(ghost_progress.max()))
    if end <= start:
        raise ValueError(
            "the replays do not overlap in arc length; they did not drive the same track"
        )
    stations = np.arange(start, end, station_spacing, dtype=np.float64)
    # Always include the final station of the overlap, so the comparison covers the whole
    # shared arc and not just up to the last grid point before it.
    if len(stations) == 0 or stations[-1] < end:
        stations = np.append(stations, end)

    ai_times = _time_at_stations(ai_progress, ai.race_times, stations)
    ghost_times = _time_at_stations(ghost_progress, ghost.race_times, stations)
    with np.errstate(invalid="ignore"):
        gaps = ai_times - ghost_times
        # Segment gaps: time each replay took between consecutive stations. Unlike the
        # station gap this is invariant to where each replay started, so a partial lap
        # (random start station) still answers "where is time lost?" honestly.
        segment_gaps = np.diff(ai_times) - np.diff(ghost_times)

    return ReplayComparison(
        stations=stations,
        ai_times=ai_times,
        ghost_times=ghost_times,
        gaps=gaps,
        segment_gaps=segment_gaps,
        ai_finished=ai.finished,
        ghost_finished=ghost.finished,
        ai_race_time=ai.race_time,
        ghost_race_time=ghost.race_time,
        track=ai.track or ghost.track,
    )


__all__ = [
    "REPLAY_DIR_NAME",
    "REPLAY_SCHEMA_VERSION",
    "EpisodeReplay",
    "ReplayComparison",
    "ReplayRecorder",
    "ReplayStore",
    "compare_replays",
]
