"""Backend-agnostic contract between the RL stack and the Trackmania game.

Everything above this module (environment, reward, learner, trainer) talks *only* to the
:class:`GameDriver` protocol and the plain data objects defined here. That is what keeps
the real game integration swappable and unit-testable: the concrete drivers live in
``tmai.game.tminterface`` (real game) and ``tmai.game.simulated`` (explicit test double).

Units and conventions (they are part of the contract, not an implementation detail):

* Positions, velocities and distances are in **metres**, the unit Trackmania itself uses
  for map/vehicle coordinates.
* Speeds are **m/s**. ``VehicleState.speed_forward`` is the longitudinal component in the
  vehicle frame (positive = forward), which is what the game reports directly.
* ``rotation`` is the vehicle-to-world 3x3 rotation matrix as reported by the game.
* Time is seconds since the start of the current race attempt.
* ``Action.steer`` is in ``[-1, 1]`` where negative is left, positive is right, matching
  the sign convention of the game's analog steer input.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np


class RacePhase(str, Enum):
    """Coarse state of the current race attempt, as reported by the game."""

    UNKNOWN = "unknown"
    NOT_RACING = "not_racing"  # in a menu, map loading, or before the countdown finished
    RUNNING = "running"  # the car is live and controllable
    FINISHED = "finished"  # the finish line was passed
    GAVE_UP = "gave_up"  # the player/AI gave up (respawn-to-start of a fresh attempt)


@dataclass(frozen=True)
class Action:
    """A single continuous control command.

    Attributes:
        steer: ``[-1, 1]``, negative left / positive right.
        throttle: ``[0, 1]`` analog gas.
        brake: ``[0, 1]`` analog brake.

    Trackmania also has handbrake/horn/respawn, which are deliberately not part of the
    learning action space in phase 1: they are issued as driver-level commands instead so
    that the policy cannot accidentally terminate its own episode.
    """

    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

    def clipped(self) -> Action:
        """Return an equivalent action with every component inside its legal range."""
        return Action(
            steer=float(np.clip(self.steer, -1.0, 1.0)),
            throttle=float(np.clip(self.throttle, 0.0, 1.0)),
            brake=float(np.clip(self.brake, 0.0, 1.0)),
        )

    def as_array(self) -> np.ndarray:
        return np.array([self.steer, self.throttle, self.brake], dtype=np.float32)

    @staticmethod
    def from_array(values: np.ndarray | Sequence[float]) -> Action:  # type: ignore[name-defined]
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size != 3:
            raise ValueError(f"Action.from_array expects 3 values, got {arr.size}")
        return Action(float(arr[0]), float(arr[1]), float(arr[2]))

    def is_neutral(self) -> bool:
        return self.steer == 0.0 and self.throttle == 0.0 and self.brake == 0.0


@dataclass(frozen=True)
class VehicleState:
    """Kinematic/dynamic state of the player's car at one physics instant."""

    position: np.ndarray  # (3,) world position, metres
    velocity: np.ndarray  # (3,) world velocity, m/s
    rotation: np.ndarray  # (3, 3) vehicle-to-world rotation matrix
    speed_forward: float  # longitudinal speed in vehicle frame, m/s
    speed_sideward: float  # lateral speed in vehicle frame, m/s
    rpm: float = 0.0
    gear: int = 0
    is_sliding: bool = False
    has_ground_contact: bool = True

    def yaw(self) -> float:
        """Heading around the world up axis, radians, from the vehicle's forward vector.

        Trackmania's world frame is left-handed with ``+y`` up; ``+x`` is "east" and ``+z``
        is "south". We define yaw as the compass-style angle of the forward vector
        projected on the ground plane so that it is independent of the sign conventions of
        the game's matrix layout and therefore stable for the environment to consume.
        """
        fwd = self.forward_vector()
        return float(np.arctan2(fwd[0], fwd[2]))

    def forward_vector(self) -> np.ndarray:
        """Unit forward vector of the car in world space."""
        fwd = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)[:, 0]
        norm = float(np.linalg.norm(fwd))
        if norm < 1e-9:
            return np.array([0.0, 0.0, 1.0])
        return fwd / norm


@dataclass(frozen=True)
class RaceState:
    """Progress/lap state of the current race attempt, as reported by the game."""

    race_time: float = 0.0
    phase: RacePhase = RacePhase.UNKNOWN
    checkpoint_index: int = 0
    checkpoint_total: int = 0
    finished: bool = False
    respawn_count: int = 0

    @property
    def checkpoint_progress(self) -> float:
        """Fraction of checkpoints collected, ``0`` when the map has none."""
        if self.checkpoint_total <= 0:
            return 0.0
        return float(np.clip(self.checkpoint_index / self.checkpoint_total, 0.0, 1.0))


@dataclass(frozen=True)
class GameFrame:
    """Everything the environment gets back from one driver step."""

    vehicle: VehicleState
    race: RaceState
    wall_time: float = 0.0  # host clock, seconds; used for throughput diagnostics only

    def info(self) -> dict[str, Any]:
        return {
            "race_time": self.race.race_time,
            "phase": self.race.phase.value,
            "checkpoint_index": self.race.checkpoint_index,
            "checkpoint_total": self.race.checkpoint_total,
            "finished": self.race.finished,
            "speed_forward": self.vehicle.speed_forward,
            "is_sliding": self.vehicle.is_sliding,
            "gear": self.vehicle.gear,
        }


@dataclass(frozen=True)
class DriverCapabilities:
    """What a concrete driver can actually do.

    The environment degrades gracefully instead of assuming features: for instance,
    game-speed manipulation is what makes long training runs affordable, but a driver that
    cannot do it is still usable at 1x.
    """

    analog_control: bool = False
    game_speed_control: bool = False
    deterministic_reset: bool = False
    reports_checkpoints: bool = False
    reports_finish: bool = False
    reports_sliding: bool = False
    headless_capable: bool = False
    #: Whether the driver can place the car at an arbitrary point on the track rather than
    #: only at the start line. The real game cannot do this without recorded checkpoint
    #: states, so start-position randomisation is honestly unavailable against Trackmania
    #: until checkpoint states are recorded (see docs/LIMITATIONS.md). Callers must check
    #: this rather than assume: silently ignoring the request would make a run report
    #: "randomised starts" that never happened.
    supports_start_repositioning: bool = False
    max_speed_ratio: float = 1.0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> dict[str, Any]:
        return {
            "analog_control": self.analog_control,
            "game_speed_control": self.game_speed_control,
            "deterministic_reset": self.deterministic_reset,
            "reports_checkpoints": self.reports_checkpoints,
            "reports_finish": self.reports_finish,
            "reports_sliding": self.reports_sliding,
            "headless_capable": self.headless_capable,
            "supports_start_repositioning": self.supports_start_repositioning,
            "max_speed_ratio": self.max_speed_ratio,
            "notes": list(self.notes),
        }


@runtime_checkable
class GameDriver(Protocol):
    """The one and only interface the RL stack uses to talk to Trackmania.

    Lifecycle::

        driver.open()
        frame = driver.reset()
        while not done:
            frame = driver.step(action)
        driver.close()

    Implementations must be explicit about failure: raising
    :class:`tmai.game.errors.GameConnectionError` rather than returning silently wrong
    data. A driver must never fabricate telemetry.
    """

    #: Short, stable, lowercase identifier used in configuration (e.g. ``"tminterface"``).
    name: str

    capabilities: DriverCapabilities

    def open(self) -> None:
        """Establish the connection to the game. Raises on failure."""

    def close(self) -> None:
        """Release the game. Must be safe to call twice and after a failed ``open``."""

    def is_connected(self) -> bool:
        """Whether the game connection is currently usable."""

    def reset(self) -> GameFrame:
        """Start a fresh race attempt and return its first frame."""

    def reposition(self, station: float, lateral: float = 0.0) -> GameFrame:
        """Place the car at ``station`` metres along the track, ``lateral`` metres off centre.

        Optional: only drivers reporting ``supports_start_repositioning`` implement this
        meaningfully. Everyone else raises :class:`tmai.game.errors.UnsupportedFeatureError`.
        The capability flag exists so callers can check *before* relying on it, instead of
        discovering at runtime that their "randomised start" was silently the start line.
        """

    def step(self, action: Action) -> GameFrame:
        """Apply ``action`` and advance the game by one control tick."""

    def set_speed_ratio(self, ratio: float) -> float:
        """Ask the game to run at ``ratio`` times real time; return the ratio applied."""

    def describe(self) -> dict[str, Any]:
        """Serialisable description used in run manifests and by ``tmai doctor``."""


def neutral_action() -> Action:
    return Action()


__all__ = [
    "Action",
    "DriverCapabilities",
    "GameDriver",
    "GameFrame",
    "RacePhase",
    "RaceState",
    "VehicleState",
    "neutral_action",
]
