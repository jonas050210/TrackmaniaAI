"""A geometry-grounded controller used as an honest evaluation baseline.

``CurvaturePilot`` is a small, explainable heuristic: it combines upcoming signed track
curvature (feed-forward steering), heading error and lateral offset (feedback), then reduces
target speed for tighter corners. It consumes the same track-relative observation vector as
the policy and does not use world coordinates or per-track memorisation.

This is a benchmark baseline, not a trained policy. Its gains and speed model are deliberately
simple and have **not** been validated in a live Windows Trackmania/TMInterface session. A
simulated benchmark can test plumbing and relative behavior; it cannot establish real-game
handling, timing, or safety.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from typing import Any

import numpy as np

from tmai.env.observation import ObservationSpec


@dataclass
class CurvaturePilotConfig:
    """Conservative parameters for the transparent track-relative baseline."""

    speed_limit: float = 70.0
    lateral_acceleration: float = 20.0
    lookahead_seconds: float = 0.45
    min_lookahead: float = 5.0
    max_lookahead: float = 40.0
    minimum_curvature: float = 0.002
    steering_gain: float = 0.72
    heading_gain: float = 1.05
    lateral_gain: float = 0.10
    max_target_heading: float = 1.1

    def validate(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if not isfinite(value):
                raise ValueError(f"curvature pilot {name} must be finite, got {value}")
        if self.speed_limit <= 0 or self.lateral_acceleration <= 0:
            raise ValueError("speed_limit and lateral_acceleration must be positive")
        if self.lookahead_seconds <= 0 or self.min_lookahead <= 0:
            raise ValueError("lookahead_seconds and min_lookahead must be positive")
        if self.max_lookahead < self.min_lookahead:
            raise ValueError("max_lookahead must be >= min_lookahead")
        if self.minimum_curvature <= 0:
            raise ValueError("minimum_curvature must be positive")
        if self.max_target_heading <= 0:
            raise ValueError("max_target_heading must be positive")


class CurvaturePilot:
    """Track-relative heuristic implementing the evaluation learner interface."""

    STATE_VERSION = 1

    def __init__(
        self,
        observation_spec: ObservationSpec,
        *,
        action_dim: int = 3,
        config: CurvaturePilotConfig | None = None,
    ) -> None:
        if action_dim != 3:
            raise ValueError(f"CurvaturePilot requires the 3-control action layout, got {action_dim}")
        self.observation_spec = observation_spec
        self.config = config or CurvaturePilotConfig()
        self.config.validate()
        self._observation_dim = observation_spec.stacked_dim
        self._action_dim = action_dim
        self._feature_indices: dict[str, int] = {}
        offset = 0
        for name, width in observation_spec.feature_layout():
            if width == 1:
                self._feature_indices[name] = offset
            else:
                for component in range(width):
                    self._feature_indices[f"{name}[{component}]"] = offset + component
            offset += width
        if offset != observation_spec.dim:
            raise ValueError(
                f"observation layout has {offset} features, spec reports {observation_spec.dim}"
            )

    @property
    def observation_dim(self) -> int:
        return self._observation_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _feature(self, frame: np.ndarray, name: str, default: float = 0.0) -> float:
        index = self._feature_indices.get(name)
        return float(frame[index]) if index is not None else default

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Return ``[steer, throttle, brake]`` from the most recent stacked frame."""
        del deterministic  # the controller is deterministic by construction
        vector = np.asarray(observation, dtype=np.float64).reshape(-1)
        if vector.size != self.observation_dim:
            raise ValueError(
                f"observation has {vector.size} values, expected {self.observation_dim}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("CurvaturePilot received a non-finite observation")
        frame = vector[-self.observation_spec.dim :]
        scales = self.observation_spec.scales

        speed = max(0.0, self._feature(frame, "speed_forward") * scales.speed)
        lateral = self._feature(frame, "lateral_offset") * scales.lateral
        heading_error = self._feature(frame, "heading_error")
        lookahead = float(
            np.clip(
                max(self.config.min_lookahead, speed * self.config.lookahead_seconds),
                self.config.min_lookahead,
                self.config.max_lookahead,
            )
        )

        curvature_samples: list[tuple[float, float]] = []
        if self.observation_spec.include_curvature:
            for distance in self.observation_spec.curvature_lookahead:
                name = f"curvature_at_{distance:g}m"
                index = self._feature_indices.get(name)
                if index is not None:
                    curvature_samples.append(
                        (float(distance), float(frame[index]) * scales.curvature)
                    )
        if curvature_samples:
            curvature_samples.sort(key=lambda item: item[0])
            distances = np.asarray([item[0] for item in curvature_samples], dtype=np.float64)
            curvatures = np.asarray([item[1] for item in curvature_samples], dtype=np.float64)
            curvature = float(np.interp(lookahead, distances, curvatures))
        else:
            curvature = 0.0

        # The observation convention is positive-right heading/lateral error and positive-
        # right curvature. Feed-forward follows the upcoming turn; feedback steers back
        # toward the centreline when the car is pointed or displaced away from it.
        target_heading = float(
            np.clip(
                curvature * lookahead,
                -self.config.max_target_heading,
                self.config.max_target_heading,
            )
        )
        steer = (
            self.config.steering_gain * target_heading
            - self.config.heading_gain * heading_error
            - self.config.lateral_gain * lateral
        )
        steer = float(np.clip(steer, -1.0, 1.0))

        curve_speed = sqrt(
            self.config.lateral_acceleration
            / max(abs(curvature), self.config.minimum_curvature)
        )
        target_speed = min(self.config.speed_limit, curve_speed)
        throttle = float(np.clip((target_speed - speed + 5.0) / 10.0, 0.0, 1.0))
        brake = float(np.clip((speed - target_speed - 2.0) / 10.0, 0.0, 0.9))
        if brake > 0.0:
            throttle = 0.0
        return np.asarray([steer, throttle, brake], dtype=np.float32)

    def update(self, batch: Any) -> dict[str, float]:
        del batch
        raise RuntimeError("CurvaturePilot is a fixed heuristic and cannot be trained")

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "config": asdict(self.config),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("version", 0)) != self.STATE_VERSION:
            raise ValueError("incompatible CurvaturePilot checkpoint state")
        if int(state.get("observation_dim", -1)) != self.observation_dim:
            raise ValueError("CurvaturePilot observation dimension does not match saved state")
        if int(state.get("action_dim", -1)) != self.action_dim:
            raise ValueError("CurvaturePilot action dimension does not match saved state")
        if state.get("config") != asdict(self.config):
            raise ValueError("CurvaturePilot configuration does not match saved state")

    def describe(self) -> dict[str, Any]:
        return {
            "algorithm": "curvature_pilot",
            "kind": "heuristic_baseline",
            "real_game_validated": False,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "config": asdict(self.config),
        }


__all__ = ["CurvaturePilot", "CurvaturePilotConfig"]
