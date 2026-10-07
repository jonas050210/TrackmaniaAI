"""A clearly-labelled stand-in for the game, used only to exercise the RL stack.

.. warning::
   **This is not Trackmania and must never be presented as a result about Trackmania.**

   :class:`SimulatedGameDriver` is a ~40-line kinematic bicycle model. It exists so that
   the environment, reward, learner, trainer, checkpointing and logging can be developed,
   unit-tested and CI-tested on a machine with no Windows and no game. A policy trained
   here has learned nothing about real Trackmania physics, tyre behaviour, drifting or map
   geometry, and will not transfer.

   The trainer refuses to use it unless the operator passes ``--allow-simulated-driver``,
   every run records ``driver="simulated"`` in its manifest, and the CLI prints a banner.

It is registered under the name ``"simulated"``; the real driver is ``"tminterface"``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from tmai.game.protocol import (
    Action,
    DriverCapabilities,
    GameFrame,
    RacePhase,
    RaceState,
    VehicleState,
)
from tmai.tracks.centerline import CenterlineTrack

SIMULATED_DRIVER_BANNER = (
    "!!! SIMULATED DRIVER IN USE !!!\n"
    "!!! This is a toy kinematic model, NOT Trackmania. Results say nothing about the real game. !!!"
)


@dataclass
class SimulatedDriverConfig:
    """Tunables of the toy model. Defaults roughly resemble an arcade racer."""

    dt: float = 0.05  # control tick, s
    max_accel: float = 14.0  # m/s^2 at full throttle
    max_brake: float = 22.0  # m/s^2 at full brake
    drag: float = 0.0035  # quadratic drag coefficient
    rolling: float = 0.6  # linear rolling resistance, 1/s
    wheelbase: float = 2.4  # m
    max_steer_angle: float = 0.55  # rad at |steer| = 1
    off_track_drag: float = 8.0  # extra linear damping off the racing surface
    lane_half_width: float = 5.0  # m; outside this the car is "off track"
    checkpoint_every: float = 60.0  # m of arc length between synthetic checkpoints
    seed: int | None = None


class SimulatedGameDriver:
    """Kinematic bicycle model constrained to a :class:`CenterlineTrack`.

    The driver reproduces the *interface* semantics of the real integration (one frame per
    ``step``, race phases, checkpoints, finish) so that the layers above it cannot tell the
    difference, while remaining obviously fake to any human reading a config or a log.
    """

    name = "simulated"

    def __init__(
        self,
        track: CenterlineTrack,
        config: SimulatedDriverConfig | None = None,
    ) -> None:
        self.track = track
        self.config = config or SimulatedDriverConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._opened = False
        self._speed_ratio = 1.0
        self._reset_state()

    # -- lifecycle ------------------------------------------------------------------

    @property
    def capabilities(self) -> DriverCapabilities:
        return DriverCapabilities(
            analog_control=True,
            game_speed_control=True,
            deterministic_reset=True,
            reports_checkpoints=True,
            reports_finish=True,
            reports_sliding=False,
            headless_capable=True,
            max_speed_ratio=1000.0,
            notes=(
                "TOY MODEL - not Trackmania. Used for pipeline tests and CI only.",
                "Kinematic bicycle model; no tyres, no drift, no collisions.",
            ),
        )

    def open(self) -> None:
        self._opened = True

    def close(self) -> None:
        self._opened = False

    def is_connected(self) -> bool:
        return self._opened

    def set_speed_ratio(self, ratio: float) -> float:
        if ratio <= 0:
            raise ValueError(f"speed ratio must be positive, got {ratio}")
        self._speed_ratio = float(ratio)
        return self._speed_ratio

    def describe(self) -> dict[str, Any]:
        return {
            "driver": self.name,
            "connected": self._opened,
            "track": self.track.name,
            "speed_ratio": self._speed_ratio,
            "model": "kinematic bicycle (toy)",
            "capabilities": self.capabilities.describe(),
        }

    # -- episode --------------------------------------------------------------------

    def reset(self) -> GameFrame:
        self._require_open()
        self._reset_state()
        return self._frame()

    def step(self, action: Action) -> GameFrame:
        self._require_open()
        action = action.clipped()
        cfg = self.config
        dt = cfg.dt

        projection = self.track.project(self._position)
        on_track = abs(projection.lateral_offset) <= cfg.lane_half_width

        accel = action.throttle * cfg.max_accel
        decel = action.brake * cfg.max_brake
        damping = cfg.rolling + (cfg.off_track_drag if not on_track else 0.0)

        dv = (accel - decel * np.sign(self._speed) - cfg.drag * self._speed**2
              - damping * self._speed) * dt
        self._speed = float(np.clip(self._speed + dv, -cfg.max_brake * dt, 90.0))

        steer_angle = float(action.steer) * cfg.max_steer_angle
        yaw_rate = (
            self._speed / cfg.wheelbase * math.tan(steer_angle)
            if abs(self._speed) > 1e-6
            else 0.0
        )
        self._yaw += yaw_rate * dt
        self._yaw_rate = yaw_rate

        forward = np.array([math.sin(self._yaw), 0.0, math.cos(self._yaw)])
        self._position = self._position + forward * (self._speed * dt)

        self._time += dt
        self._update_progress()
        return self._frame()

    # -- internals ------------------------------------------------------------------

    def _reset_state(self) -> None:
        start = self.track.point_at(0.0)
        heading = self.track.heading_at(0.0)
        self._position = np.array([start[0], start[1], start[2]], dtype=np.float64)
        self._yaw = float(math.atan2(heading[0], heading[2]))
        self._speed = 0.0
        self._yaw_rate = 0.0
        self._time = 0.0
        self._progress = 0.0
        self._checkpoint_index = 0
        self._finished = False
        self._respawns = 0

    def _update_progress(self) -> None:
        projection = self.track.project(self._position)
        self._progress = projection.progress
        expected = int(self._progress // self.config.checkpoint_every)
        if expected > self._checkpoint_index and not self._finished:
            self._checkpoint_index = min(expected, self._checkpoint_total())
        if self._progress >= self.track.length and not self._finished:
            self._finished = True
            self._checkpoint_index = self._checkpoint_total()

    def _checkpoint_total(self) -> int:
        return max(1, int(self.track.length // self.config.checkpoint_every))

    def _frame(self) -> GameFrame:
        yaw = self._yaw
        # VehicleState.yaw() reads the forward vector as column 0 of the rotation matrix and
        # defines yaw = atan2(forward_x, forward_z). This matrix is the proper rotation about
        # the world up axis that satisfies that (det = +1), so heading_error starts at 0.
        rotation = np.array(
            [
                [math.sin(yaw), 0.0, -math.cos(yaw)],
                [0.0, 1.0, 0.0],
                [math.cos(yaw), 0.0, math.sin(yaw)],
            ],
            dtype=np.float64,
        )
        velocity = np.array(
            [math.sin(yaw) * self._speed, 0.0, math.cos(yaw) * self._speed],
            dtype=np.float64,
        )
        vehicle = VehicleState(
            position=self._position.copy(),
            velocity=velocity,
            rotation=rotation,
            speed_forward=self._speed,
            speed_sideward=0.0,
            rpm=abs(self._speed) * 90.0 + 900.0,
            gear=int(min(6, 1 + abs(self._speed) // 15.0)),
            is_sliding=False,
            has_ground_contact=True,
        )
        phase = RacePhase.FINISHED if self._finished else RacePhase.RUNNING
        race = RaceState(
            race_time=self._time,
            phase=phase,
            checkpoint_index=self._checkpoint_index,
            checkpoint_total=self._checkpoint_total(),
            finished=self._finished,
            respawn_count=self._respawns,
        )
        return GameFrame(vehicle=vehicle, race=race, wall_time=time.monotonic())

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("SimulatedGameDriver.open() has not been called")


__all__ = ["SIMULATED_DRIVER_BANNER", "SimulatedDriverConfig", "SimulatedGameDriver"]
