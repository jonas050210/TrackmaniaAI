"""Tests for the TMInterface telemetry mapping.

The important part of this file is :class:`TestAgainstRealStructs`: it builds genuine
``tminterface.structs.SimStateData`` objects from raw byte buffers and runs our mapper over
them. That validates the mapping against the *actual* upstream memory layout and field
offsets, not against a hand-written stand-in -- which is the strongest verification possible
without the game itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import HAS_TMINTERFACE
from tmai.game.errors import GameProtocolError
from tmai.game.protocol import RacePhase
from tmai.game.tminterface.telemetry import (
    frame_from_sim_state,
    has_dynamics,
    has_player_info,
    race_state_from_sim_state,
    vehicle_state_from_sim_state,
)

SIM_HAS_TIMERS = 0x1
SIM_HAS_DYNA = 0x2
SIM_HAS_PLAYER_INFO = 0x80


class _StubCarState:
    def __init__(self, speed_forward=0.0, speed_sideward=0.0, rpm=0.0, gearbox_state=0):
        self.speed_forward = speed_forward
        self.speed_sideward = speed_sideward
        self.rpm = rpm
        self.gearbox_state = gearbox_state


class _StubEngine:
    def __init__(self, gear=0, max_rpm=10000.0):
        self.gear = gear
        self.max_rpm = max_rpm


class _StubCar:
    def __init__(self, **kwargs):
        gear = kwargs.pop("gear", 0)
        is_sliding = kwargs.pop("is_sliding", False)
        self.sync_vehicle_state = _StubCarState(**kwargs)
        self.engine = _StubEngine(gear=gear)
        self.is_sliding = is_sliding


class _StubPlayerInfo:
    def __init__(self, race_time=0, race_finished=False, cur_cp_count=0):
        self.race_time = race_time
        self.race_finished = race_finished
        self.cur_cp_count = cur_cp_count


class StubSimState:
    """Mimics the attribute surface of ``tminterface.structs.SimStateData``."""

    def __init__(
        self,
        position=(1.0, 2.0, 3.0),
        velocity=(4.0, 0.0, 5.0),
        rotation=None,
        flags=SIM_HAS_TIMERS | SIM_HAS_DYNA | SIM_HAS_PLAYER_INFO,
        race_time=1500,
        num_respawns=0,
        **car_kwargs,
    ):
        self.flags = flags
        self.position = list(position)
        self.velocity = list(velocity)
        self.rotation_matrix = np.eye(3) if rotation is None else rotation
        player_kwargs = car_kwargs.pop("player", {})
        self.scene_mobil = _StubCar(**car_kwargs)
        self.player_info = _StubPlayerInfo(race_time=race_time, **player_kwargs)
        self.race_time = race_time
        self.num_respawns = num_respawns


class TestVehicleStateMapping:
    def test_maps_all_fields(self):
        state = StubSimState(
            position=(1.0, 2.0, 3.0),
            velocity=(10.0, 0.0, 0.0),
            speed_forward=12.5,
            speed_sideward=-1.5,
            rpm=7200.0,
            gear=3,
            is_sliding=True,
        )
        vehicle = vehicle_state_from_sim_state(state)
        np.testing.assert_allclose(vehicle.position, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(vehicle.velocity, [10.0, 0.0, 0.0])
        assert vehicle.speed_forward == 12.5
        assert vehicle.speed_sideward == -1.5
        assert vehicle.rpm == 7200.0
        assert vehicle.gear == 3
        assert vehicle.is_sliding is True

    def test_position_scale_applies_to_position_velocity_and_speed(self):
        state = StubSimState(position=(10.0, 0.0, 0.0), velocity=(2.0, 0.0, 0.0), speed_forward=2.0)
        vehicle = vehicle_state_from_sim_state(state, position_scale=0.1)
        np.testing.assert_allclose(vehicle.position, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(vehicle.velocity, [0.2, 0.0, 0.0])
        assert vehicle.speed_forward == pytest.approx(0.2)

    def test_zero_rotation_matrix_becomes_identity(self):
        state = StubSimState(rotation=np.zeros((3, 3)))
        vehicle = vehicle_state_from_sim_state(state)
        np.testing.assert_allclose(vehicle.rotation, np.eye(3))

    def test_missing_dynamics_flag_raises(self):
        state = StubSimState(flags=SIM_HAS_TIMERS | SIM_HAS_PLAYER_INFO)
        with pytest.raises(GameProtocolError, match="SIM_HAS_DYNA"):
            vehicle_state_from_sim_state(state)

    def test_non_finite_position_raises(self):
        state = StubSimState(position=(float("nan"), 0.0, 0.0))
        with pytest.raises(GameProtocolError, match="non-finite"):
            vehicle_state_from_sim_state(state)

    def test_short_position_vector_raises(self):
        state = StubSimState()
        state.position = [1.0, 2.0]
        with pytest.raises(GameProtocolError, match="3-vector"):
            vehicle_state_from_sim_state(state)

    def test_non_finite_speed_falls_back_to_zero(self):
        state = StubSimState(speed_forward=float("inf"))
        vehicle = vehicle_state_from_sim_state(state)
        assert vehicle.speed_forward == 0.0


class TestRaceStateMapping:
    def test_running_phase_and_units(self):
        state = StubSimState(race_time=12_345)
        race = race_state_from_sim_state(state, checkpoint_total=5)
        assert race.phase is RacePhase.RUNNING
        assert race.race_time == pytest.approx(12.345)
        assert race.checkpoint_total == 5

    def test_negative_race_time_means_not_racing(self):
        state = StubSimState(race_time=-1)
        race = race_state_from_sim_state(state)
        assert race.phase is RacePhase.NOT_RACING
        # A negative race time must never leak out as a negative duration.
        assert race.race_time == 0.0

    def test_finished_flag_wins(self):
        state = StubSimState(race_time=90_000)
        state.player_info.race_finished = True
        race = race_state_from_sim_state(state)
        assert race.phase is RacePhase.FINISHED
        assert race.finished is True

    def test_missing_player_info_flag_means_not_racing(self):
        state = StubSimState(flags=SIM_HAS_TIMERS | SIM_HAS_DYNA)
        assert has_player_info(state) is False
        race = race_state_from_sim_state(state)
        assert race.phase is RacePhase.NOT_RACING

    def test_checkpoint_count_and_respawns(self):
        state = StubSimState(race_time=1000, num_respawns=3)
        state.player_info.cur_cp_count = 2
        race = race_state_from_sim_state(state, checkpoint_total=4)
        assert race.checkpoint_index == 2
        assert race.checkpoint_total == 4
        assert race.checkpoint_progress == pytest.approx(0.5)
        assert race.respawn_count == 3

    def test_race_time_override_wins(self):
        state = StubSimState(race_time=1000)
        race = race_state_from_sim_state(state, race_time_ms=2500)
        assert race.race_time == pytest.approx(2.5)


class TestFrameHelper:
    def test_frame_from_sim_state(self):
        state = StubSimState(race_time=2000, speed_forward=5.0)
        frame = frame_from_sim_state(state, checkpoint_total=3, wall_time=123.0)
        assert frame.wall_time == 123.0
        assert frame.vehicle.speed_forward == 5.0
        assert frame.race.checkpoint_total == 3
        info = frame.info()
        assert info["phase"] == "running"
        assert info["speed_forward"] == 5.0


@pytest.mark.skipif(not HAS_TMINTERFACE, reason="tminterface package not installed")
class TestAgainstRealStructs:
    """Decode real ``SimStateData`` objects and check the mapping end to end."""

    @staticmethod
    def _make(flags=SIM_HAS_TIMERS | SIM_HAS_DYNA | SIM_HAS_PLAYER_INFO):
        from tminterface.structs import SimStateData

        state = SimStateData(bytearray(SimStateData.min_size))
        state.flags = flags
        return state

    def test_min_size_is_nonzero(self):
        from tminterface.structs import SimStateData

        # Guards against an upstream layout change silently breaking the mapping.
        assert SimStateData.min_size > 0

    def test_position_roundtrip_through_real_struct(self):
        state = self._make()
        state.position = [11.0, 22.0, 33.0]
        vehicle = vehicle_state_from_sim_state(state)
        np.testing.assert_allclose(vehicle.position, [11.0, 22.0, 33.0], rtol=1e-6)

    def test_speed_and_gear_roundtrip_through_real_struct(self):
        state = self._make()
        state.scene_mobil.sync_vehicle_state.speed_forward = 41.25
        state.scene_mobil.sync_vehicle_state.speed_sideward = -3.5
        state.scene_mobil.engine.gear = 4
        state.scene_mobil.is_sliding = True
        vehicle = vehicle_state_from_sim_state(state)
        assert vehicle.speed_forward == pytest.approx(41.25, rel=1e-6)
        assert vehicle.speed_sideward == pytest.approx(-3.5, rel=1e-6)
        assert vehicle.gear == 4
        assert vehicle.is_sliding is True

    def test_player_info_roundtrip_through_real_struct(self):
        state = self._make()
        state.player_info.race_time = 65_432
        state.player_info.cur_cp_count = 3
        race = race_state_from_sim_state(state, checkpoint_total=6)
        assert race.race_time == pytest.approx(65.432, rel=1e-6)
        assert race.checkpoint_index == 3
        assert race.phase is RacePhase.RUNNING

    def test_rotation_matrix_shape_from_real_struct(self):
        state = self._make()
        vehicle = vehicle_state_from_sim_state(state)
        assert vehicle.rotation.shape == (3, 3)
        # A zeroed buffer yields a zero matrix, which the mapper must replace with identity.
        np.testing.assert_allclose(vehicle.rotation, np.eye(3))

    def test_frame_helper_over_real_struct(self):
        state = self._make()
        state.position = [1.0, 2.0, 3.0]
        frame = frame_from_sim_state(state, checkpoint_total=2, wall_time=1.0)
        np.testing.assert_allclose(frame.vehicle.position, [1.0, 2.0, 3.0], rtol=1e-6)
        assert frame.race.checkpoint_total == 2
        assert has_dynamics(state) is True

    def test_missing_dynamics_flag_detected_on_real_struct(self):
        state = self._make(flags=SIM_HAS_TIMERS | SIM_HAS_PLAYER_INFO)
        assert has_dynamics(state) is False
        with pytest.raises(GameProtocolError):
            vehicle_state_from_sim_state(state)
