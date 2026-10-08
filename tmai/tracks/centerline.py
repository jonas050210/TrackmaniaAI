"""Track representation: an arc-length parameterised centreline with a corridor.

Why this exists: to generalise to unseen maps the agent must never see absolute world
coordinates. Everything the policy consumes is expressed **relative to the track** --
progress along the centreline, signed lateral offset, heading error, upcoming curvature and
distance to the track edges. Those quantities are defined here.

The centreline is the *only* piece of map knowledge phase 1 needs. It is recorded from real
game telemetry by driving a lap once (``tmai record-track``), which is what makes the
representation available for any map without writing a ``.Map.Gbx`` parser. Block-level
geometry (walls, ramps, obstacles) is a documented next step; see ``docs/ROADMAP.md``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrackProjection:
    """Where a world-space point sits relative to the track."""

    #: Arc-length coordinate along the centreline, metres.
    progress: float
    #: Signed lateral offset, metres. Positive is to the right of the direction of travel.
    lateral_offset: float
    #: Height above/below the centreline plane at that station, metres.
    vertical_offset: float
    #: Index of the nearest centreline sample.
    index: int
    #: Euclidean distance to the centreline, metres.
    distance: float
    #: Centreline point at ``progress``.
    centre: np.ndarray
    #: Unit tangent at ``progress``.
    tangent: np.ndarray

    @property
    def on_track(self) -> bool:
        """Whether the point is inside the corridor (needs a corridor to be meaningful)."""
        return True


class CenterlineTrack:
    """A 3-D polyline with arc-length parameterisation and an optional corridor width.

    Args:
        points: ``(N, 3)`` ordered centreline samples in metres, ``N >= 2``.
        name: human-readable track identifier.
        uid: stable map identifier (the Trackmania map UID when known).
        corridor_half_width: metres from the centreline to the drivable edge. Either a
            scalar or an ``(N,)`` array for maps whose width varies.
        closed: whether the last sample connects back to the first (a circuit).
        metadata: free-form serialisable data (map name, recording date, driver, ...).
    """

    def __init__(
        self,
        points: np.ndarray | Sequence[Sequence[float]],
        *,
        name: str = "track",
        uid: str | None = None,
        corridor_half_width: float | Sequence[float] | np.ndarray = 5.0,
        closed: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
        if pts.shape[0] < 2:
            raise ValueError("a track needs at least 2 centreline points")
        if closed and pts.shape[0] < 3:
            raise ValueError("a closed track needs at least 3 centreline points")
        if not np.all(np.isfinite(pts)):
            raise ValueError("track points contain non-finite values")

        self.points = pts
        self.name = name
        self.uid = uid
        self.closed = bool(closed)
        self.metadata: dict[str, Any] = dict(metadata or {})

        if isinstance(corridor_half_width, (int, float, np.integer, np.floating)):
            scalar_width = float(corridor_half_width)
            if scalar_width <= 0:
                raise ValueError(f"corridor_half_width must be positive, got {scalar_width}")
            width_values = np.full(pts.shape[0], scalar_width, dtype=np.float64)
        elif isinstance(corridor_half_width, np.ndarray) and corridor_half_width.ndim == 0:
            scalar_width = float(corridor_half_width.item())
            if scalar_width <= 0:
                raise ValueError(f"corridor_half_width must be positive, got {scalar_width}")
            width_values = np.full(pts.shape[0], scalar_width, dtype=np.float64)
        else:
            width_values = np.asarray(corridor_half_width, dtype=np.float64).reshape(-1)
            if width_values.shape[0] != pts.shape[0]:
                raise ValueError("corridor_half_width must have one entry per point")
            if np.any(width_values <= 0):
                raise ValueError("corridor_half_width entries must be positive")
        self.corridor_half_width = width_values

        if self.closed:
            self._segment_vectors = np.roll(self.points, -1, axis=0) - self.points
        else:
            self._segment_vectors = np.diff(self.points, axis=0)
        self._segment_lengths = np.linalg.norm(self._segment_vectors, axis=1)
        if np.any(self._segment_lengths < 1e-6):
            raise ValueError(
                "track has duplicate consecutive points or a zero-length segment at the closing seam"
            )
        self._cum_lengths = np.concatenate([[0.0], np.cumsum(self._segment_lengths)])
        self._length = float(self._cum_lengths[-1])
        self._tangents = self._segment_vectors / self._segment_lengths[:, None]
        self._curvature = self._compute_curvature()

    # -- basic accessors ------------------------------------------------------------

    @property
    def length(self) -> float:
        """Total centreline length in metres."""
        return self._length

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    def index_at(self, s: float) -> int:
        """Index of the segment containing arc-length ``s``.

        Open tracks clamp out-of-range values. Closed tracks wrap, so the closing segment
        is the last segment and ``s == length`` is the start of the next lap.
        """
        station = self._station(s)
        return int(np.clip(np.searchsorted(self._cum_lengths, station, side="right") - 1,
                           0, len(self._segment_lengths) - 1))

    def _station(self, s: float) -> float:
        """Convert a possibly unwrapped coordinate to this track's local station."""
        value = float(s)
        if not np.isfinite(value):
            raise ValueError(f"track station must be finite, got {value}")
        if self.closed:
            return value % self._length
        return float(np.clip(value, 0.0, self._length))

    def point_at(self, s: float) -> np.ndarray:
        """Centreline position at arc-length ``s`` (wrapped on a closed circuit).

        ``_tangents`` are *unit* vectors, so the offset is in metres and multiplies the unit
        tangent directly. The final segment of a circuit runs from its last sample back to
        the first; callers do not need to duplicate the start point in the input data.
        """
        station = self._station(s)
        i = self.index_at(station)
        offset = station - self._cum_lengths[i]
        return self.points[i] + self._tangents[i] * offset

    def heading_at(self, s: float) -> np.ndarray:
        """Unit tangent (direction of travel) at arc-length ``s``."""
        return self._tangents[self.index_at(s)].copy()

    def curvature_at(self, s: float) -> float:
        """Signed curvature (1/m) at arc-length ``s``; positive turns right."""
        return float(self._curvature[self.index_at(s)])

    def corridor_half_width_at(self, s: float) -> float:
        """Corridor half width at arc-length ``s``, linearly interpolated between samples.

        The array holds one width per *point*, while most lookups are by arc length inside a
        *segment*. Interpolating keeps the two consistent and means the final sample is
        actually used instead of being silently ignored.
        """
        station = self._station(s)
        i = self.index_at(station)
        j = (i + 1) % len(self.corridor_half_width) if self.closed else min(
            i + 1, len(self.corridor_half_width) - 1
        )
        fraction = (station - self._cum_lengths[i]) / self._segment_lengths[i]
        return float(
            self.corridor_half_width[i] * (1.0 - fraction) + self.corridor_half_width[j] * fraction
        )

    def sample_lookahead(self, s: float, distances: Iterable[float]) -> np.ndarray:
        """Signed curvature at several lookahead distances, shape ``(len(distances),)``."""
        return np.array([self.curvature_at(s + d) for d in distances], dtype=np.float64)

    # -- projection -----------------------------------------------------------------

    def project(
        self,
        position: np.ndarray | Sequence[float],
        *,
        hint_s: float | None = None,
        search_window: float = 120.0,
    ) -> TrackProjection:
        """Project a world position onto the centreline.

        Args:
            position: world position, metres.
            hint_s: previous progress. When given, only the neighbourhood is searched,
                which makes per-step projection O(window) instead of O(N).
            search_window: metres of arc length searched around ``hint_s``. A ``None`` hint
                (episode start) always searches the whole track.

        Returns:
            The :class:`TrackProjection`. Progress is clamped to ``[0, length]`` on open
            tracks. On a closed circuit, a hinted projection is unwrapped to the lap nearest
            ``hint_s`` so progress stays continuous across the start/finish seam.
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(pos)):
            raise ValueError(f"cannot project non-finite position {pos!r}")

        segment_count = len(self._segment_lengths)
        if hint_s is not None and not np.isfinite(float(hint_s)):
            raise ValueError(f"hint_s must be finite, got {hint_s}")
        window = float(search_window)
        if not np.isfinite(window) or window < 0:
            raise ValueError(f"search_window must be finite and non-negative, got {window}")
        if hint_s is None:
            idx = np.arange(segment_count)
        elif self.closed:
            if 2.0 * window >= self._length:
                idx = np.arange(segment_count)
            else:
                station = float(hint_s) % self._length
                lo = station - window
                hi = station + window
                if lo < 0.0:
                    first = np.arange(self.index_at(self._length + lo), segment_count)
                    second = np.arange(0, self.index_at(hi) + 1)
                    idx = np.concatenate((first, second))
                elif hi >= self._length:
                    first = np.arange(self.index_at(lo), segment_count)
                    second = np.arange(0, self.index_at(hi - self._length) + 1)
                    idx = np.concatenate((first, second))
                else:
                    idx = np.arange(self.index_at(lo), self.index_at(hi) + 1)
                if idx.size == 0:
                    idx = np.array([self.index_at(station)], dtype=np.int64)
        else:
            lo = self.index_at(float(hint_s) - window)
            hi = min(self.index_at(float(hint_s) + window) + 1, segment_count)
            idx = np.arange(lo, hi)

        a = self.points[idx]
        ab = self._segment_vectors[idx]
        lengths = self._segment_lengths[idx]

        ap = pos[None, :] - a
        # Parameter of the closest point on each segment, in [0, 1].
        denom = np.maximum(lengths**2, 1e-12)
        t = np.clip((ap * ab).sum(axis=1) / denom, 0.0, 1.0)
        closest = a + ab * t[:, None]
        delta = pos[None, :] - closest
        dist_sq = (delta**2).sum(axis=1)

        best = int(np.argmin(dist_sq))
        seg = int(idx[best])
        t_best = float(t[best])
        distance = float(np.sqrt(dist_sq[best]))

        # `lengths` is the windowed slice, `seg` is a global segment index: index the full
        # array. Mixing the two corrupts the arc length whenever a hint is supplied.
        segment_length = float(self._segment_lengths[seg])
        local_progress = float(self._cum_lengths[seg] + t_best * segment_length)
        if self.closed and hint_s is not None:
            # Keep the station continuous across the finish/start seam. This lets reward,
            # wrong-way detection and episode progress use ordinary signed deltas over laps.
            hint = float(hint_s)
            progress = local_progress + round((hint - local_progress) / self._length) * self._length
        else:
            progress = float(np.clip(local_progress, 0.0, self._length))
        tangent = self._tangents[seg]
        centre = a[best] + ab[best] * t_best

        # Signed lateral offset: positive to the right of the direction of travel.
        #
        # Trackmania's world frame is LEFT-HANDED with +y up (+x east, +z south). In a
        # left-handed frame the right-hand side of a heading is cross(up, tangent), not
        # cross(tangent, up) as it would be in the right-handed convention most 3-D maths
        # libraries assume. Getting this backwards flips every steering correction the
        # policy learns, so it is spelled out here and covered by tests.
        right = np.cross(np.array([0.0, 1.0, 0.0]), tangent)
        right_norm = float(np.linalg.norm(right))
        right = np.zeros(3) if right_norm < 1e-9 else right / right_norm
        lateral = float(np.dot(pos - centre, right))
        vertical = float(pos[1] - centre[1])

        return TrackProjection(
            progress=progress,
            lateral_offset=lateral,
            vertical_offset=vertical,
            index=seg,
            distance=distance,
            centre=centre,
            tangent=tangent,
        )

    def is_on_track(self, projection: TrackProjection) -> bool:
        """Whether a projection lies inside the drivable corridor."""
        return abs(projection.lateral_offset) <= self.corridor_half_width_at(projection.progress)

    def edge_distances(self, projection: TrackProjection) -> tuple[float, float]:
        """``(distance_to_left_edge, distance_to_right_edge)`` in metres, along the lateral axis.

        This is the cheapest useful proxy for "how much room do I have" that does not
        require map block geometry, and it is enough to learn to stay on the road.
        """
        half = self.corridor_half_width_at(projection.progress)
        left = half + projection.lateral_offset
        right = half - projection.lateral_offset
        return max(0.0, left), max(0.0, right)

    def heading_error(self, projection: TrackProjection, yaw: float) -> float:
        """Signed angle (rad) between the car's heading and the centreline tangent."""
        tangent = projection.tangent
        track_yaw = float(np.arctan2(tangent[0], tangent[2]))
        err = yaw - track_yaw
        return float(np.arctan2(np.sin(err), np.cos(err)))

    # -- curvature ------------------------------------------------------------------

    def _compute_curvature(self) -> np.ndarray:
        """Signed curvature per segment, including the seam on a closed circuit."""
        if self.closed:
            t0 = self._tangents
            t1 = np.roll(self._tangents, -1, axis=0)
            lengths = self._segment_lengths
        else:
            t0 = self._tangents[:-1]
            t1 = self._tangents[1:]
            lengths = self._segment_lengths[:-1]

        # Angle around the world up axis, signed so that a right turn is positive.
        cross_y = t0[:, 0] * t1[:, 2] - t0[:, 2] * t1[:, 0]
        dot = np.clip((t0 * t1).sum(axis=1), -1.0, 1.0)
        dtheta = np.arctan2(-cross_y, dot)
        per_segment = dtheta / np.maximum(lengths, 1e-9)

        if self.closed:
            return per_segment.astype(np.float64, copy=False)

        # Assign the final open segment the last observed turn, as before.
        curvature = np.zeros(len(self._segment_lengths), dtype=np.float64)
        curvature[:-1] = per_segment
        curvature[-1] = per_segment[-1] if len(per_segment) else 0.0
        return curvature

    # -- serialisation --------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "centerline_track",
            "name": self.name,
            "uid": self.uid,
            "closed": self.closed,
            "length": self.length,
            "points": self.points.tolist(),
            "corridor_half_width": self.corridor_half_width.tolist(),
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> CenterlineTrack:
        version = int(data.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported track schema version {version}; this build understands {SCHEMA_VERSION}"
            )
        if data.get("kind") != "centerline_track":
            raise ValueError(f"not a centerline track document: kind={data.get('kind')!r}")
        return CenterlineTrack(
            np.asarray(data["points"], dtype=np.float64),
            name=str(data.get("name", "track")),
            uid=data.get("uid"),
            corridor_half_width=np.asarray(
                data.get("corridor_half_width", 5.0), dtype=np.float64
            ),
            closed=bool(data.get("closed", False)),
            metadata=dict(data.get("metadata") or {}),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path) -> CenterlineTrack:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"track file not found: {path}")
        return CenterlineTrack.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"CenterlineTrack(name={self.name!r}, points={self.num_points}, "
            f"length={self.length:.1f}m, closed={self.closed})"
        )


__all__ = ["SCHEMA_VERSION", "CenterlineTrack", "TrackProjection"]
