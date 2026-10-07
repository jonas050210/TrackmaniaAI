"""Track-relative observations.

Design rule: **the policy never sees absolute world coordinates.** Every feature is either
an ego quantity (speed, rpm, gear, sliding) or a quantity expressed in the track frame
(progress rate, lateral offset, heading error, upcoming curvature, distance to the corridor
edges) plus the previous action. That is what makes a policy trained on one map have any
chance of transferring to another, and it keeps the input dimension small and fixed.

The layout is explicit and ordered so it can be logged, diffed between runs and validated
against a checkpoint. ``ObservationSpec.names()`` returns the human-readable layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tmai.game.protocol import Action, GameFrame
from tmai.tracks.centerline import CenterlineTrack, TrackProjection

#: Version of the observation layout. Bump when the vector layout changes: a checkpoint
#: records it and refuses to load into a differently-shaped network.
OBSERVATION_VERSION = 1


@dataclass
class ObservationScales:
    """Normalisation constants. Values are in metres / m/s / rad unless noted.

    They are *fixed* rather than learned so that the meaning of an observation never drifts
    during training; a separate running normaliser can still be applied on top (see
    :class:`tmai.env.wrappers.RunningNormWrapper`).
    """

    speed: float = 70.0
    lateral: float = 8.0
    curvature: float = 0.05
    yaw_rate: float = 2.0
    rpm: float = 10000.0
    gear: float = 6.0
    progress_delta: float = 3.5


@dataclass
class ObservationSpec:
    """Which features go into the observation vector, and in what order."""

    #: Metres of lookahead at which curvature is sampled.
    curvature_lookahead: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0, 80.0)
    include_speed: bool = True
    include_drivetrain: bool = True
    include_sliding: bool = True
    include_lateral: bool = True
    include_heading_error: bool = True
    include_progress_rate: bool = True
    include_curvature: bool = True
    include_edge_distances: bool = True
    include_last_action: bool = True
    include_checkpoint_progress: bool = True
    scales: ObservationScales = field(default_factory=ObservationScales)

    def feature_layout(self) -> list[tuple[str, int]]:
        """Ordered ``(name, width)`` pairs describing the vector."""
        layout: list[tuple[str, int]] = []
        if self.include_speed:
            layout += [("speed_forward", 1), ("speed_sideward", 1)]
        if self.include_drivetrain:
            layout += [("rpm", 1), ("gear", 1)]
        if self.include_sliding:
            layout += [("is_sliding", 1)]
        layout += [("yaw_rate", 1)]
        if self.include_lateral:
            layout += [("lateral_offset", 1)]
        if self.include_heading_error:
            layout += [("heading_error", 1)]
        if self.include_progress_rate:
            layout += [("progress_rate", 1)]
        if self.include_curvature:
            layout += [(f"curvature_at_{d:g}m", 1) for d in self.curvature_lookahead]
        if self.include_edge_distances:
            layout += [("edge_distance_left", 1), ("edge_distance_right", 1)]
        if self.include_checkpoint_progress:
            layout += [("checkpoint_progress", 1)]
        if self.include_last_action:
            layout += [("last_steer", 1), ("last_throttle", 1), ("last_brake", 1)]
        return layout

    @property
    def dim(self) -> int:
        return sum(width for _, width in self.feature_layout())

    def names(self) -> list[str]:
        out: list[str] = []
        for name, width in self.feature_layout():
            out.extend([name] if width == 1 else [f"{name}[{i}]" for i in range(width)])
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_version": OBSERVATION_VERSION,
            "dim": self.dim,
            "names": self.names(),
            "curvature_lookahead": list(self.curvature_lookahead),
            "scales": {
                "speed": self.scales.speed,
                "lateral": self.scales.lateral,
                "curvature": self.scales.curvature,
                "yaw_rate": self.scales.yaw_rate,
                "rpm": self.scales.rpm,
                "gear": self.scales.gear,
                "progress_delta": self.scales.progress_delta,
            },
        }


@dataclass
class ObservationInputs:
    """Everything needed to build one observation vector."""

    frame: GameFrame
    projection: TrackProjection
    prev_projection: TrackProjection | None
    prev_yaw: float | None
    dt: float
    last_action: Action
    yaw_rate: float


class ObservationEncoder:
    """Turns a :class:`GameFrame` plus track geometry into a flat observation vector."""

    def __init__(self, track: CenterlineTrack, spec: ObservationSpec | None = None) -> None:
        self.track = track
        self.spec = spec or ObservationSpec()
        self._dim = self.spec.dim
        if self._dim <= 0:
            raise ValueError("observation spec produced an empty vector")

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, inputs: ObservationInputs) -> np.ndarray:
        spec = self.spec
        scales = spec.scales
        frame = inputs.frame
        vehicle = frame.vehicle
        projection = inputs.projection

        parts: list[np.ndarray] = []

        if spec.include_speed:
            parts.append(
                np.array(
                    [
                        vehicle.speed_forward / scales.speed,
                        vehicle.speed_sideward / scales.speed,
                    ],
                    dtype=np.float64,
                )
            )
        if spec.include_drivetrain:
            parts.append(
                np.array(
                    [vehicle.rpm / scales.rpm, vehicle.gear / max(1.0, scales.gear)],
                    dtype=np.float64,
                )
            )
        if spec.include_sliding:
            parts.append(np.array([1.0 if vehicle.is_sliding else 0.0], dtype=np.float64))

        parts.append(np.array([inputs.yaw_rate / scales.yaw_rate], dtype=np.float64))

        if spec.include_lateral:
            parts.append(
                np.array([projection.lateral_offset / scales.lateral], dtype=np.float64)
            )
        if spec.include_heading_error:
            yaw = vehicle.yaw()
            heading_error = self.track.heading_error(projection, yaw)
            parts.append(np.array([heading_error], dtype=np.float64))
        if spec.include_progress_rate:
            if inputs.prev_projection is None or inputs.dt <= 0:
                rate = 0.0
            else:
                rate = (projection.progress - inputs.prev_projection.progress) / inputs.dt
            parts.append(np.array([rate / max(scales.progress_delta, 1e-6)], dtype=np.float64))
        if spec.include_curvature:
            curvatures = self.track.sample_lookahead(
                projection.progress, spec.curvature_lookahead
            )
            parts.append(curvatures / scales.curvature)
        if spec.include_edge_distances:
            left, right = self.track.edge_distances(projection)
            parts.append(
                np.array([left / scales.lateral, right / scales.lateral], dtype=np.float64)
            )
        if spec.include_checkpoint_progress:
            parts.append(np.array([frame.race.checkpoint_progress], dtype=np.float64))
        if spec.include_last_action:
            parts.append(
                np.array(
                    [inputs.last_action.steer, inputs.last_action.throttle, inputs.last_action.brake],
                    dtype=np.float64,
                )
            )

        obs = np.concatenate(parts).astype(np.float32)
        if obs.shape[0] != self._dim:
            raise AssertionError(
                f"observation encoder produced {obs.shape[0]} values, spec says {self._dim}"
            )
        if not np.all(np.isfinite(obs)):
            # Never feed NaNs to the learner; clip instead and let the caller notice via
            # the diagnostics counter.
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)
        return obs


__all__ = [
    "OBSERVATION_VERSION",
    "ObservationEncoder",
    "ObservationInputs",
    "ObservationScales",
    "ObservationSpec",
]
