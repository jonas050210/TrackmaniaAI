"""Translation from TMInterface's raw simulation state into :mod:`tmai.game.protocol` types.

This module is intentionally **pure** and free of any ``tminterface`` import. It consumes
the attributes that ``tminterface.structs.SimStateData`` exposes (``position``,
``velocity``, ``rotation_matrix``, ``scene_mobil.sync_vehicle_state.speed_forward``, ...)
and produces our own frozen dataclasses.

Keeping it pure is what makes the real integration unit-testable on a machine that has
neither Windows nor the game: tests can hand it a stub with the same attribute names, or a
genuine ``SimStateData`` decoded from a byte buffer, and check the mapping is correct.

Attribute names used below were taken from ``tminterface`` 1.0.2 ``structs.py``:

* ``SimStateData.position`` / ``.velocity`` -- lists of 3 floats (zero when the
  ``SIM_HAS_DYNA`` flag is unset).
* ``SimStateData.rotation_matrix`` -- 3x3 matrix (``[[0,0,0]]*3`` when unset).
* ``SimStateData.flags`` -- bitmask of ``SIM_HAS_*`` flags.
* ``SimStateData.scene_mobil`` -- ``SceneVehicleCar`` with ``.sync_vehicle_state``
  (``speed_forward``, ``speed_sideward``, ``rpm``, ``gearbox_state``, ``input_steer``,
  ``input_gas``, ``input_brake``), ``.engine`` (``gear``, ``max_rpm``), ``.is_sliding``
  and ``.has_any_lateral_contact`` (the game's own wall-contact flag).
* ``SimStateData.simulation_wheels`` -- four ``SimulationWheel`` structs whose
  ``.real_time_state.has_ground_contact`` reports per-wheel ground contact.
* ``SimStateData.player_info`` -- ``PlayerInfoStruct`` with ``race_time`` (ms),
  ``race_finished``, ``cur_cp_count``, ``display_speed``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Protocol

import numpy as np

from tmai.game.errors import GameProtocolError
from tmai.game.protocol import RacePhase, RaceState, VehicleState

logger = logging.getLogger(__name__)

# SIM_HAS_* flags from tminterface.constants, duplicated here so that this module has no
# dependency on the (Windows-only) tminterface package.
SIM_HAS_TIMERS = 0x1
SIM_HAS_DYNA = 0x2
SIM_HAS_SIMULATION_WHEELS = 0x8
SIM_HAS_PLAYER_INFO = 0x80


class _CarStateLike(Protocol):
    speed_forward: float
    speed_sideward: float
    rpm: float
    gearbox_state: int
    input_steer: float
    input_gas: float
    input_brake: float


class _EngineLike(Protocol):
    gear: int
    max_rpm: float


class _RealTimeStateLike(Protocol):
    has_ground_contact: bool


class _WheelLike(Protocol):
    real_time_state: _RealTimeStateLike


class _CarLike(Protocol):
    sync_vehicle_state: _CarStateLike
    engine: _EngineLike
    is_sliding: bool
    has_any_lateral_contact: bool


class _PlayerInfoLike(Protocol):
    race_time: int
    race_finished: bool
    cur_cp_count: int


class SimStateLike(Protocol):
    """The subset of ``tminterface.structs.SimStateData`` that this module relies on."""

    flags: int
    position: Any
    velocity: Any
    rotation_matrix: Any
    scene_mobil: _CarLike
    player_info: _PlayerInfoLike
    race_time: Any
    num_respawns: int
    simulation_wheels: Any


def has_dynamics(sim_state: SimStateLike) -> bool:
    """Whether the state actually carries vehicle dynamics (position/velocity/rotation)."""
    return bool(int(getattr(sim_state, "flags", 0)) & SIM_HAS_DYNA)


def has_player_info(sim_state: SimStateLike) -> bool:
    return bool(int(getattr(sim_state, "flags", 0)) & SIM_HAS_PLAYER_INFO)


def _as_vec3(value: Any) -> np.ndarray:
    """Coerce a game-supplied position/velocity into a finite ``(3,)`` float64 array."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise GameProtocolError(f"expected a 3-vector from the game, got {arr.size} values")
    arr = arr[:3]
    if not np.all(np.isfinite(arr)):
        raise GameProtocolError(f"non-finite vector from the game: {arr!r}")
    return arr


def _as_mat3(value: Any) -> np.ndarray:
    """Coerce a game-supplied rotation into a ``(3, 3)`` float64 matrix.

    A zero matrix (what ``SimStateData`` yields when the dynamics region is not present)
    is replaced by the identity so that downstream code never divides by a zero norm.
    """
    mat = np.asarray(value, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(mat)) or float(np.abs(mat).max()) < 1e-9:
        return np.eye(3)
    return mat


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _wheel_ground_contacts(sim_state: SimStateLike) -> int | None:
    """Count wheels reporting ground contact, or ``None`` when the state lacks the data.

    ``SimStateData.simulation_wheels`` holds four ``SimulationWheel`` structs, each with a
    ``real_time_state.has_ground_contact`` flag. The lookup is defensive: a stub or an older
    struct without the field yields ``None``, and the caller then falls back to the
    conservative default rather than guessing.
    """
    # SimStateData always exposes a four-wheel array in its memory layout, even when the
    # server did not populate that region. Trust it only when the upstream validity flag is
    # set; otherwise four zero-filled structs falsely look like four airborne wheels.
    flags = int(getattr(sim_state, "flags", 0))
    if not flags & SIM_HAS_SIMULATION_WHEELS:
        return None
    wheels = getattr(sim_state, "simulation_wheels", None)
    if wheels is None:
        return None
    count = 0
    seen = 0
    for wheel in wheels:
        real_time = getattr(wheel, "real_time_state", None)
        contact = getattr(real_time, "has_ground_contact", None)
        if contact is None:
            return None
        seen += 1
        if bool(contact):
            count += 1
    return count if seen else None


def vehicle_state_from_sim_state(
    sim_state: SimStateLike,
    *,
    position_scale: float = 1.0,
    forward_axis: int = 0,
    forward_sign: float = 1.0,
) -> VehicleState:
    """Build a :class:`VehicleState` from a TMInterface simulation state.

    Args:
        sim_state: a ``SimStateData`` (or anything exposing the same attributes).
        position_scale: multiplier converting game position units to metres. Trackmania's
            internal unit is not documented by Nadeo, so this is a configuration knob that
            ``tmai doctor --measure-units`` calibrates on real hardware. Velocities are
            scaled by the same factor so that speed and position stay consistent.

    Raises:
        GameProtocolError: if the state is malformed or carries non-finite values.
    """
    if not has_dynamics(sim_state):
        raise GameProtocolError(
            "TMInterface returned a simulation state without the dynamics region "
            "(SIM_HAS_DYNA unset); the car is probably not loaded yet"
        )

    car = sim_state.scene_mobil
    car_state = car.sync_vehicle_state

    speed_forward = _finite(car_state.speed_forward)
    speed_sideward = _finite(car_state.speed_sideward)

    # Ground contact comes from the game's per-wheel flags when the struct carries them.
    # The previous behaviour (hard-coded True) meant "fell off the map" could never be
    # detected from real telemetry; a stub without the wheels region still gets the
    # conservative default.
    wheel_contacts = _wheel_ground_contacts(sim_state)
    has_ground_contact = True if wheel_contacts is None else wheel_contacts > 0

    return VehicleState(
        position=_as_vec3(sim_state.position) * float(position_scale),
        velocity=_as_vec3(sim_state.velocity) * float(position_scale),
        rotation=_as_mat3(sim_state.rotation_matrix),
        speed_forward=speed_forward * float(position_scale),
        speed_sideward=speed_sideward * float(position_scale),
        rpm=_finite(car_state.rpm),
        gear=int(_finite(getattr(car.engine, "gear", 0))),
        is_sliding=bool(getattr(car, "is_sliding", False)),
        has_ground_contact=has_ground_contact,
        # The game's own lateral-contact flag: trusted over any derived signal by the
        # crash detector (tmai.env.termination).
        has_lateral_contact=bool(getattr(car, "has_any_lateral_contact", False)),
        num_wheels_ground_contact=4 if wheel_contacts is None else int(wheel_contacts),
        input_steer=_finite(getattr(car_state, "input_steer", 0.0)),
        input_gas=_finite(getattr(car_state, "input_gas", 0.0)),
        input_brake=_finite(getattr(car_state, "input_brake", 0.0)),
        forward_axis=int(forward_axis),
        forward_sign=float(forward_sign),
    )


def race_state_from_sim_state(
    sim_state: SimStateLike,
    *,
    race_time_ms: int | None = None,
    checkpoint_total: int = 0,
    time_scale: float = 1.0,
) -> RaceState:
    """Build a :class:`RaceState` from a TMInterface simulation state.

    Args:
        sim_state: a ``SimStateData``.
        race_time_ms: race time override in milliseconds. ``SimStateData.race_time`` is
            already in ms, but the driver also receives the tick time from
            ``on_run_step`` and may prefer it.
        checkpoint_total: total number of checkpoints on the map, supplied by the driver
            (TMInterface reports it through ``on_checkpoint_count_changed``).
        time_scale: multiplier applied to the race time (1.0 normally).
    """
    player_info = getattr(sim_state, "player_info", None)

    if race_time_ms is None:
        race_time_ms = int(_finite(getattr(sim_state, "race_time", 0)))
    race_time = max(0, int(race_time_ms)) / 1000.0 * float(time_scale)

    finished = bool(getattr(player_info, "race_finished", False)) if player_info else False
    checkpoint_index = int(_finite(getattr(player_info, "cur_cp_count", 0))) if player_info else 0

    # Trackmania sets race_time to -1 while the car is waiting for the countdown; the game
    # reports that as a negative player_info.race_time.
    raw_race_time = int(_finite(getattr(player_info, "race_time", 0))) if player_info else 0
    if finished:
        phase = RacePhase.FINISHED
    elif raw_race_time < 0 or not has_player_info(sim_state):
        phase = RacePhase.NOT_RACING
    else:
        phase = RacePhase.RUNNING

    return RaceState(
        race_time=race_time,
        phase=phase,
        checkpoint_index=max(0, checkpoint_index),
        checkpoint_total=max(0, int(checkpoint_total)),
        finished=finished,
        respawn_count=int(_finite(getattr(sim_state, "num_respawns", 0))),
    )


def frame_from_sim_state(
    sim_state: SimStateLike,
    *,
    position_scale: float = 1.0,
    forward_axis: int = 0,
    forward_sign: float = 1.0,
    race_time_ms: int | None = None,
    checkpoint_total: int = 0,
    wall_time: float = 0.0,
):
    """Convenience helper returning a full :class:`tmai.game.protocol.GameFrame`."""
    from tmai.game.protocol import GameFrame

    return GameFrame(
        vehicle=vehicle_state_from_sim_state(
            sim_state,
            position_scale=position_scale,
            forward_axis=forward_axis,
            forward_sign=forward_sign,
        ),
        race=race_state_from_sim_state(
            sim_state, race_time_ms=race_time_ms, checkpoint_total=checkpoint_total
        ),
        wall_time=wall_time,
    )


__all__ = [
    "SimStateLike",
    "frame_from_sim_state",
    "has_dynamics",
    "has_player_info",
    "race_state_from_sim_state",
    "vehicle_state_from_sim_state",
]
