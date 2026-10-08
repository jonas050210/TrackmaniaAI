"""Record a track centreline by driving the real map once.

This is how phase 1 obtains map knowledge without a ``.Map.Gbx`` parser: a human (or a
ghost replay) drives the racing line, the driver streams telemetry, and we keep a
distance-decimated copy of the trajectory as the centreline.

Limitations, stated plainly:

* The recorded centreline is only as good as the lap that produced it. A sloppy lap gives a
  sloppy reference line, which biases the reward. Recording several laps and averaging, or
  recording from the map author's medal ghost, is the obvious improvement (see ROADMAP).
* The corridor half-width is a single constant here. Real maps vary in width; per-point
  width needs block geometry.

Usage::

    tmai record-track --driver tminterface --out data/tracks/my_map.json

Then drive one clean lap. Press Ctrl-C when done; the partial recording is still saved.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tmai.game.protocol import Action, GameDriver, GameFrame
from tmai.tracks.centerline import CenterlineTrack

logger = logging.getLogger(__name__)


@dataclass
class CenterlineRecorder:
    """Collects car positions into a decimated centreline.

    Args:
        min_spacing: minimum metres between two kept samples. Decimating by distance rather
            than by time is what makes the centreline spacing independent of how fast the
            recording lap was driven.
        corridor_half_width: metres of drivable corridor either side of the centreline.
        smoothing_window: odd sample count for a moving-average pass over the raw points,
            which removes the jitter of a 100 Hz telemetry stream. ``1`` disables it.
    """

    min_spacing: float = 1.0
    corridor_half_width: float = 5.0
    smoothing_window: int = 1
    points: list[np.ndarray] = field(default_factory=list)

    def add(self, position: np.ndarray) -> bool:
        """Add a position; returns whether it was kept (i.e. far enough from the last one)."""
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(pos)):
            logger.warning("ignoring non-finite position while recording")
            return False
        if self.points and float(np.linalg.norm(pos - self.points[-1])) < self.min_spacing:
            return False
        self.points.append(pos.copy())
        return True

    def add_frame(self, frame: GameFrame) -> bool:
        return self.add(frame.vehicle.position)

    def __len__(self) -> int:
        return len(self.points)

    def _smooth(self, pts: np.ndarray, *, closed: bool = False) -> np.ndarray:
        window = int(self.smoothing_window)
        if window <= 1 or len(pts) < window:
            return pts
        if window % 2 == 0:
            raise ValueError(f"smoothing_window must be odd, got {window}")
        pad = window // 2
        if closed:
            padded = np.concatenate([pts[-pad:], pts, pts[:pad]], axis=0)
        else:
            padded = np.concatenate(
                [np.repeat(pts[:1], pad, axis=0), pts, np.repeat(pts[-1:], pad, axis=0)],
                axis=0,
            )
        kernel = np.ones(window) / window
        return np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(3)], axis=1)

    def build_track(
        self,
        *,
        name: str,
        uid: str | None = None,
        metadata: dict | None = None,
        closed: bool = False,
    ) -> CenterlineTrack:
        """Turn the collected points into a :class:`CenterlineTrack`."""
        if len(self.points) < 2:
            raise ValueError(f"need at least 2 recorded points to build a track, got {len(self.points)}")
        raw_points = np.asarray(self.points, dtype=np.float64)
        if closed and len(raw_points) > 2 and float(np.linalg.norm(raw_points[-1] - raw_points[0])) < 1e-3:
            # Remove a duplicate finish-line sample before circular smoothing so the seam is
            # treated as a true ring instead of smoothing an artificial zero-length segment.
            raw_points = raw_points[:-1]
        pts = self._smooth(raw_points, closed=closed)
        pts = _deduplicate(pts, eps=1e-3)
        if len(pts) < 2:
            raise ValueError("smoothing collapsed the recording to fewer than 2 distinct points")
        meta = {
            "source": "recorded_telemetry",
            "raw_samples": len(self.points),
            "min_spacing": self.min_spacing,
            "smoothing_window": self.smoothing_window,
            **(metadata or {}),
        }
        return CenterlineTrack(
            pts,
            name=name,
            uid=uid,
            corridor_half_width=self.corridor_half_width,
            closed=closed,
            metadata=meta,
        )


def _deduplicate(pts: np.ndarray, eps: float) -> np.ndarray:
    """Drop consecutive points closer than ``eps`` (zero-length segments are rejected later)."""
    keep = [pts[0]]
    for p in pts[1:]:
        if float(np.linalg.norm(p - keep[-1])) > eps:
            keep.append(p)
    return np.asarray(keep, dtype=np.float64)


def record_track(
    driver: GameDriver,
    *,
    out_path: str,
    name: str,
    min_spacing: float = 1.0,
    corridor_half_width: float = 5.0,
    smoothing_window: int = 5,
    max_points: int = 20000,
    should_stop: Callable[[], bool] | None = None,
    action_provider: Callable[[GameFrame], Action | None] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CenterlineTrack:
    """Record a centreline from the car's telemetry.

    With the real driver, the default ``None`` action leaves game input untouched, so the
    player can drive with their normal keyboard/controller. Supply an ``action_provider`` to
    automate a recording (or to drive the simulated test driver).

    Args:
        driver: an opened :class:`~tmai.game.protocol.GameDriver`.
        out_path: where to write the track JSON.
        name: track name.
        max_points: safety cap on the recording length.
        should_stop: polled each frame; return True to stop early (Ctrl-C friendly).
        action_provider: optional control source. If omitted, advances without injecting an
            action (human controls remain live on TMInterface).
        metadata: additional provenance, including an optional shared ``family`` label for
            related maps that must stay in the same data split.

    Returns:
        The recorded :class:`CenterlineTrack`, also saved to ``out_path``.
    """
    recorder = CenterlineRecorder(
        min_spacing=min_spacing,
        corridor_half_width=corridor_half_width,
        smoothing_window=smoothing_window,
    )
    frame = driver.reset()
    finished = False
    recorder.add_frame(frame)
    logger.info("recording started for track %r; drive one clean lap", name)

    try:
        while True:
            if should_stop is not None and should_stop():
                logger.info("recording stopped by operator after %d points", len(recorder))
                break
            action = action_provider(frame) if action_provider is not None else None
            frame = driver.step(action)
            if frame.race.finished:
                finished = True
                logger.info("finish line reached at %.3fs; recording complete", frame.race.race_time)
                recorder.add_frame(frame)
                break
            recorder.add_frame(frame)
            if len(recorder) >= max_points:
                logger.warning("hit max_points=%d; stopping the recording", max_points)
                break
    except KeyboardInterrupt:  # pragma: no cover - interactive path
        logger.info("interrupted; saving the %d points recorded so far", len(recorder))

    track = recorder.build_track(
        name=name,
        metadata={
            **dict(metadata or {}),
            "race_time_s": frame.race.race_time,
            "finished_lap": finished,
        },
        closed=finished,
    )
    saved = track.save(out_path)
    logger.info("saved centreline: %d points, %.1f m -> %s", track.num_points, track.length, saved)
    return track


__all__ = ["CenterlineRecorder", "record_track"]
