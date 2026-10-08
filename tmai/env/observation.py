"""Track-relative observations.

Design rule: **the policy never sees absolute world coordinates.** Every feature is either
an ego quantity (speed, rpm, gear, sliding) or a quantity expressed in the track frame
(progress rate, lateral offset, heading error, upcoming curvature, distance to the corridor
edges) plus the previous action. That is what makes a policy trained on one map have any
chance of transferring to another, and it keeps the input dimension small and fixed.

The layout is explicit and ordered so it can be logged, diffed between runs and validated
against a checkpoint. ``ObservationSpec.names()`` returns the human-readable layout.

Temporal stacking
-----------------
A single frame is a partially observable snapshot: it carries no explicit acceleration or
speed trend. ``ObservationSpec.history_length`` stacks the last N encoded frames into one
observation vector (oldest first), which gives the policy an implicit temporal window --
the standard frame-stacking remedy -- without changing the encoder or the track model.
The default of 1 is the single-frame observation; raising it multiplies the observation
dimension, which checkpoints detect and refuse to load across (the learner guards on
``observation_dim``).
"""

from __future__ import annotations

from collections import deque
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
    #: How many consecutive encoded frames are stacked into one observation. 1 is the
    #: single-frame observation; N > 1 gives the policy an implicit temporal window.
    history_length: int = 1
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
        """Width of one encoded frame (before temporal stacking)."""
        return sum(width for _, width in self.feature_layout())

    @property
    def stacked_dim(self) -> int:
        """Width of the observation the policy actually sees (after stacking)."""
        return self.dim * self.history_length

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.history_length < 1:
            problems.append(
                f"history_length must be >= 1, got {self.history_length}"
            )
        if not self.curvature_lookahead:
            problems.append("curvature_lookahead must not be empty when include_curvature")
        elif any(d <= 0 for d in self.curvature_lookahead):
            problems.append("curvature_lookahead distances must be positive")
        return problems

    def names(self) -> list[str]:
        """Names of the stacked observation, oldest frame first.

        With ``history_length == 1`` these are the plain feature names. With N > 1 each
        frame's names get a ``[k]`` suffix, ``k=0`` being the oldest frame in the window.
        """
        base: list[str] = []
        for name, width in self.feature_layout():
            base.extend([name] if width == 1 else [f"{name}[{i}]" for i in range(width)])
        if self.history_length == 1:
            return base
        out: list[str] = []
        for k in range(self.history_length):
            out.extend(f"{name}[{k}]" for name in base)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_version": OBSERVATION_VERSION,
            "dim": self.dim,
            "history_length": self.history_length,
            "stacked_dim": self.stacked_dim,
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


class ObservationStacker:
    """Stacks the last ``length`` encoded frames into one observation vector.

    The stacker is owned by the environment, not the encoder: the encoder stays a pure
    single-frame function, and the replay buffer stores whatever the policy consumed, so
    stacked and unstacked runs are both just data. At episode start the stack is filled
    with the first frame (the standard frame-stack initialisation), so the policy never
    sees a partially zero-padded window mid-episode.
    """

    def __init__(self, dim: int, length: int) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")
        self.dim = int(dim)
        self.length = int(length)
        self._frames: deque[np.ndarray] = deque(maxlen=self.length)

    @property
    def stacked_dim(self) -> int:
        return self.dim * self.length

    def reset(self, first: np.ndarray) -> np.ndarray:
        """Start a new episode: the stack is the first frame repeated ``length`` times."""
        frame = np.asarray(first, dtype=np.float32).reshape(-1)
        if frame.shape[0] != self.dim:
            raise ValueError(f"frame has {frame.shape[0]} features, stacker expects {self.dim}")
        self._frames.clear()
        for _ in range(self.length):
            self._frames.append(frame.copy())
        return self._stack()

    def push(self, frame: np.ndarray) -> np.ndarray:
        """Add one frame and return the stacked observation (oldest first)."""
        arr = np.asarray(frame, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.dim:
            raise ValueError(f"frame has {arr.shape[0]} features, stacker expects {self.dim}")
        self._frames.append(arr)
        return self._stack()

    def _stack(self) -> np.ndarray:
        return np.concatenate(list(self._frames)).astype(np.float32)


__all__ = [
    "OBSERVATION_VERSION",
    "ObservationEncoder",
    "ObservationInputs",
    "ObservationScales",
    "ObservationSpec",
    "ObservationStacker",
]
