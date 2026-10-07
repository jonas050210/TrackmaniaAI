"""Descriptive statistics for a track.

These exist to answer one question that generalisation work cannot avoid: **are the tracks I
train on actually representative of the tracks I test on?** A suite that trains on ovals and
tests on a technical track is not measuring generalisation, it is measuring a distribution
shift nobody accounted for.

So every track gets a cheap, deterministic fingerprint computed from its centreline geometry
alone -- no map file, no game, nothing platform-specific. The suite can then report whether
the train and held-out splits cover comparable geometry, and the numbers go into the run
manifest so a later reader can see what a run was actually exposed to.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tmai.tracks.centerline import CenterlineTrack


@dataclass(frozen=True)
class TrackStats:
    """Geometry fingerprint of one track. All lengths in metres, angles in radians."""

    #: Total centreline length.
    length: float
    num_points: int
    closed: bool
    #: Mean and peak absolute curvature, 1/m. ``1/radius``; a 20 m radius corner is 0.05.
    curvature_mean: float
    curvature_max: float
    #: Total absolute turning angle around the track, radians. ~2*pi for a single circuit.
    total_turning: float
    #: Radius (m) of the tightest corner; ``inf`` for a perfectly straight track.
    min_corner_radius: float
    #: Fraction of the track that is effectively straight (|curvature| below threshold).
    straight_fraction: float
    #: Number of distinct corner groups (runs of consecutive turning segments).
    corner_count: int
    #: Corridor width statistics, metres.
    corridor_mean: float
    corridor_min: float
    corridor_max: float
    #: Vertical extent of the centreline, metres. 0 for a flat track.
    elevation_range: float
    #: Chord length between the first and last point, metres. Small for a closed circuit.
    span: float
    #: Mean spacing between consecutive samples, metres.
    sample_spacing: float

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        # JSON cannot represent inf; the serialisation must round-trip.
        if not np.isfinite(out["min_corner_radius"]):
            out["min_corner_radius"] = None
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}

    def summary(self) -> str:
        radius = self.min_corner_radius
        radius_text = "straight" if not np.isfinite(radius) else f"{radius:.0f}m"
        return (
            f"{self.length:.0f}m, {self.corner_count} corners, tightest {radius_text}, "
            f"{self.straight_fraction * 100:.0f}% straight"
        )


def compute_stats(
    track: CenterlineTrack,
    *,
    straight_curvature_threshold: float = 0.005,
) -> TrackStats:
    """Compute the geometry fingerprint of ``track``.

    Curvature is taken from the track's own per-segment estimate, so this adds no new
    geometry code that could disagree with the curvature the observation uses.
    """
    curvature = np.abs(track._curvature)  # noqa: SLF001 - same module family, avoids a copy
    turning = np.abs(track._curvature) * np.maximum(track._segment_lengths, 1e-9)  # noqa: SLF001

    finite = curvature[np.isfinite(curvature) & (curvature > 0)]
    max_curvature = float(finite.max()) if finite.size else 0.0
    min_radius = float(1.0 / max_curvature) if max_curvature > 0 else float("inf")

    straight_mask = curvature <= straight_curvature_threshold
    lengths = track._segment_lengths  # noqa: SLF001
    total_length = float(lengths.sum()) or 1.0
    straight_fraction = float(lengths[straight_mask].sum() / total_length) if lengths.size else 0.0

    # A "corner" is a maximal run of consecutive turning segments.
    turning_mask = ~straight_mask
    corner_count = 0
    in_corner = False
    for is_turning in turning_mask:
        if is_turning and not in_corner:
            corner_count += 1
            in_corner = True
        elif not is_turning:
            in_corner = False

    widths = track.corridor_half_width * 2.0
    points = track.points
    span = float(np.linalg.norm(points[-1] - points[0]))

    return TrackStats(
        length=track.length,
        num_points=track.num_points,
        closed=track.closed,
        curvature_mean=float(curvature.mean()) if curvature.size else 0.0,
        curvature_max=max_curvature,
        total_turning=float(turning.sum()),
        min_corner_radius=min_radius,
        straight_fraction=straight_fraction,
        corner_count=corner_count,
        corridor_mean=float(widths.mean()),
        corridor_min=float(widths.min()),
        corridor_max=float(widths.max()),
        elevation_range=float(points[:, 1].max() - points[:, 1].min()),
        span=span,
        sample_spacing=float(lengths.mean()) if lengths.size else 0.0,
    )


__all__ = ["TrackStats", "compute_stats"]
