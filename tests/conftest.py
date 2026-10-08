"""Shared fixtures.

The suite is designed to run anywhere: no Windows, no Trackmania, no display. Anything that
genuinely needs the game is marked ``game`` and skipped.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from tmai.game.protocol import Action, GameFrame, RacePhase, RaceState, VehicleState
from tmai.tracks.synthetic import oval, s_curve, straight

HAS_TMINTERFACE = importlib.util.find_spec("tminterface") is not None


def make_rotation(yaw: float) -> np.ndarray:
    """Vehicle-to-world rotation whose column 0 is the forward vector ``[sin, 0, cos]``.

    This matches the convention documented on :meth:`VehicleState.yaw`, and is the convention
    the simulated driver emits.
    """
    return np.array(
        [
            [math.sin(yaw), 0.0, -math.cos(yaw)],
            [0.0, 1.0, 0.0],
            [math.cos(yaw), 0.0, math.sin(yaw)],
        ],
        dtype=np.float64,
    )


def make_vehicle(
    position=(0.0, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0),
    yaw: float = 0.0,
    speed_forward: float = 0.0,
    speed_sideward: float = 0.0,
    rpm: float = 1000.0,
    gear: int = 1,
    is_sliding: bool = False,
    has_ground_contact: bool = True,
    has_lateral_contact: bool = False,
    num_wheels_ground_contact: int = 4,
) -> VehicleState:
    return VehicleState(
        position=np.asarray(position, dtype=np.float64),
        velocity=np.asarray(velocity, dtype=np.float64),
        rotation=make_rotation(yaw),
        speed_forward=speed_forward,
        speed_sideward=speed_sideward,
        rpm=rpm,
        gear=gear,
        is_sliding=is_sliding,
        has_ground_contact=has_ground_contact,
        has_lateral_contact=has_lateral_contact,
        num_wheels_ground_contact=num_wheels_ground_contact,
    )


def make_frame(
    position=(0.0, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0),
    yaw: float = 0.0,
    speed_forward: float = 0.0,
    *,
    race_time: float = 0.0,
    phase: RacePhase = RacePhase.RUNNING,
    checkpoint_index: int = 0,
    checkpoint_total: int = 0,
    finished: bool = False,
    wall_time: float = 0.0,
    **vehicle_kwargs,
) -> GameFrame:
    return GameFrame(
        vehicle=make_vehicle(position, velocity, yaw, speed_forward=speed_forward, **vehicle_kwargs),
        race=RaceState(
            race_time=race_time,
            phase=phase,
            checkpoint_index=checkpoint_index,
            checkpoint_total=checkpoint_total,
            finished=finished,
        ),
        wall_time=wall_time,
    )


class ScriptedTickSource:
    """A :class:`~tmai.game.ticksource.TickSource` driven by a test script.

    It reproduces the *semantics* the real session guarantees -- one frame per tick, control
    operations applied at the top of a tick, ``is_alive`` flipping on shutdown -- so the driver
    logic under test is the same code that runs against the game.
    """

    def __init__(self, frames, *, alive: bool = True, checkpoint_total: int = 4,
                 repeat_last: bool = False):
        self._frames = list(frames)
        self._index = 0
        self.actions: list[Action] = []
        self.ops: list = []
        self.alive = alive
        self._checkpoint_total = checkpoint_total
        self._speed_ratio = 1.0
        self.started = False
        self.stopped = False
        self._repeat_last = repeat_last

    @property
    def is_alive(self) -> bool:
        return self.alive

    @property
    def checkpoint_total(self) -> int:
        return self._checkpoint_total

    @property
    def speed_ratio(self) -> float:
        return self._speed_ratio

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        self.alive = False

    def request_op(self, op) -> None:
        from tmai.game.tminterface.ops import SetSpeedOp

        self.ops.append(op)
        if isinstance(op, SetSpeedOp):
            self._speed_ratio = op.ratio

    def push_action(self, action: Action) -> None:
        self.actions.append(action)

    def next_frame(self, timeout: float | None = None) -> GameFrame:
        if self._index >= len(self._frames):
            if self._repeat_last:
                # Emulates a game that keeps ticking in the same state forever, so tests can
                # exercise driver deadlines without enumerating thousands of frames.
                self._index += 1
                return self._frames[-1]
            raise AssertionError(
                f"ScriptedTickSource ran out of frames after {len(self._frames)}"
            )
        frame = self._frames[self._index]
        self._index += 1
        return frame

    @property
    def consumed(self) -> int:
        return self._index


@pytest.fixture
def straight_track():
    return straight(length=200.0)


@pytest.fixture
def s_curve_track():
    return s_curve(length=300.0)


@pytest.fixture
def oval_track():
    return oval()


@pytest.fixture
def neutral_action():
    return Action()
