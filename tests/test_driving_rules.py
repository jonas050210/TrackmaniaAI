"""Tests for the driving rules: crash, out-of-bounds, fell-off and wrong-way handling.

The required behaviours, and where each is pinned:

* a significant wall collision causes an immediate respawn (episode ends, ``crash``);
* falling off the map or a confirmed out-of-bounds state causes an immediate respawn;
* those events produce a meaningful negative reward (a one-off terminal penalty);
* normal contact (a scrape, parking against a wall) is distinguished from a crash;
* the game's own contact state is trusted when the driver reports it;
* failure reasons are machine-readable and recorded.
"""

from __future__ import annotations

import numpy as np
import pytest

from tmai.env.reward import ProgressReward, RewardConfig
from tmai.env.termination import (
    EndReason,
    TerminationConfig,
    TerminationResult,
    TerminationTracker,
)
from tmai.game.protocol import Action, GameFrame, RaceState, VehicleState
from tmai.game.simulated import SimulatedDriverConfig, SimulatedGameDriver
from tmai.tracks.synthetic import straight

from .conftest import make_frame


def _projection(track, lateral: float = 0.0, station: float = 50.0, height: float = 0.0):
    """A projection of a point ``lateral`` metres off the centreline at ``station``."""
    centre = track.point_at(station)
    tangent = track.heading_at(station)
    right = np.cross(np.array([0.0, 1.0, 0.0]), tangent)
    right = right / np.linalg.norm(right)
    position = centre + right * lateral
    position = position + np.array([0.0, height, 0.0])
    return track.project(position), position


class TestCrashDetection:
    """The crash detector: impact and sustained-contact confirmation, scrape exclusion."""

    def test_impact_with_contact_terminates_immediately(self, straight_track):
        tracker = TerminationTracker(straight_track)
        projection, _ = _projection(straight_track)
        # Fast approach, then a collision that removes 10 m/s in one step.
        tracker.update(frame=make_frame(speed_forward=30.0), projection=projection,
                       progress_delta=1.5)
        result = tracker.update(
            frame=make_frame(speed_forward=20.0, has_lateral_contact=True),
            projection=projection,
            progress_delta=1.0,
        )
        assert result.terminated and result.reason is EndReason.CRASH

    def test_impact_without_contact_flag_still_terminates(self, straight_track):
        """Drivers that do not report contact still get crash detection from kinematics."""
        tracker = TerminationTracker(straight_track)
        projection, _ = _projection(straight_track)
        tracker.update(frame=make_frame(speed_forward=30.0), projection=projection,
                       progress_delta=1.5)
        result = tracker.update(
            frame=make_frame(speed_forward=15.0),  # 15 m/s lost, no contact reported
            projection=projection,
            progress_delta=0.75,
        )
        assert result.terminated and result.reason is EndReason.CRASH

    def test_scrape_is_not_a_crash(self, straight_track):
        """Brief contact at maintained speed is normal driving, not a crash."""
        tracker = TerminationTracker(straight_track)
        projection, _ = _projection(straight_track)
        tracker.update(frame=make_frame(speed_forward=30.0), projection=projection,
                       progress_delta=1.5)
        for _ in range(2):  # shorter than crash_contact_steps, speed maintained
            result = tracker.update(
                frame=make_frame(speed_forward=29.5, has_lateral_contact=True),
                projection=projection,
                progress_delta=1.4,
            )
            assert not result.terminated
            assert result.lateral_contact is True

    def test_sustained_contact_with_speed_loss_is_a_crash(self, straight_track):
        tracker = TerminationTracker(straight_track)
        projection, _ = _projection(straight_track)
        tracker.update(frame=make_frame(speed_forward=30.0), projection=projection,
                       progress_delta=1.5)
        # Contact begins at 30 m/s; the car is held by the wall and slows below 60%.
        speeds = [29.0, 25.0, 17.0]
        result = None
        for speed in speeds:
            result = tracker.update(
                frame=make_frame(speed_forward=speed, has_lateral_contact=True),
                projection=projection,
                progress_delta=0.5,
            )
        assert result is not None
        assert result.terminated and result.reason is EndReason.CRASH

    def test_parking_against_a_wall_is_not_a_crash(self, straight_track):
        """Contact entered below min_speed_for_crash is parking, not a crash."""
        tracker = TerminationTracker(straight_track)
        projection, _ = _projection(straight_track)
        tracker.update(frame=make_frame(speed_forward=0.5), projection=projection,
                       progress_delta=0.0)
        for _ in range(10):
            result = tracker.update(
                frame=make_frame(speed_forward=0.4, has_lateral_contact=True),
                projection=projection,
                progress_delta=0.0,
            )
            assert result.reason is not EndReason.CRASH

    def test_hard_braking_is_not_a_crash(self, straight_track):
        """A 22 m/s^2 stop loses ~1 m/s per control step -- far below the impact threshold."""
        tracker = TerminationTracker(straight_track)
        projection, _ = _projection(straight_track)
        speed = 30.0
        for _ in range(10):
            tracker.update(frame=make_frame(speed_forward=speed), projection=projection,
                           progress_delta=speed * 0.05)
            speed = max(0.0, speed - 1.1)
        result = tracker.update(frame=make_frame(speed_forward=speed),
                                projection=projection, progress_delta=0.0)
        assert result.reason is not EndReason.CRASH


class TestOutOfBoundsAndFalling:
    def test_sustained_out_of_bounds_terminates(self, straight_track):
        tracker = TerminationTracker(straight_track)
        projection, _ = _projection(straight_track, lateral=20.0)  # corridor 6 + margin 6
        result = None
        for _ in range(3):  # out_of_bounds_steps
            result = tracker.update(frame=make_frame(speed_forward=20.0),
                                    projection=projection, progress_delta=1.0)
        assert result is not None
        assert result.terminated and result.reason is EndReason.OUT_OF_BOUNDS

    def test_single_step_excursion_is_not_out_of_bounds(self, straight_track):
        tracker = TerminationTracker(straight_track)
        projection, _ = _projection(straight_track, lateral=20.0)
        result = tracker.update(frame=make_frame(speed_forward=20.0),
                                projection=projection, progress_delta=1.0)
        assert not result.terminated
        # Back inside the corridor resets the streak.
        inside, _ = _projection(straight_track, lateral=0.0)
        result = tracker.update(frame=make_frame(speed_forward=20.0),
                                projection=inside, progress_delta=1.0)
        assert not result.terminated

    def test_falling_below_the_track_terminates(self, straight_track):
        tracker = TerminationTracker(straight_track)
        projection, _ = _projection(straight_track, lateral=2.0, height=-3.0)
        frame = make_frame(speed_forward=10.0, has_ground_contact=False)
        result = None
        for _ in range(3):  # fall_steps
            result = tracker.update(frame=frame, projection=projection,
                                    progress_delta=0.5)
        assert result is not None
        assert result.terminated and result.reason is EndReason.FELL_OFF

    def test_jump_is_not_falling_off(self, straight_track):
        """Airborne *above* the track plane is a jump, not a fall."""
        tracker = TerminationTracker(straight_track)
        projection, _ = _projection(straight_track, lateral=0.0, height=+4.0)
        frame = make_frame(speed_forward=25.0, has_ground_contact=False)
        for _ in range(10):
            result = tracker.update(frame=frame, projection=projection,
                                    progress_delta=1.25)
            assert result.reason is not EndReason.FELL_OFF


class TestWrongWay:
    def test_sustained_reverse_progress_terminates(self, straight_track):
        # Isolated config: stall detection disabled so only the wrong-way rule can fire.
        config = TerminationConfig(wrong_way_steps=10, min_steps_before_stall=10**9,
                                   stall_limit=10**9)
        tracker = TerminationTracker(straight_track, config)
        projection, _ = _projection(straight_track, station=100.0)
        # Drive forward to establish a best progress of ~100 m.
        tracker.update(frame=make_frame(speed_forward=20.0), projection=projection,
                       progress_delta=1.0)
        # Now drive backwards, more than wrong_way_progress behind the best.
        result = None
        for i in range(40):
            back, _ = _projection(straight_track, station=100.0 - 0.5 * (i + 1))
            result = tracker.update(frame=make_frame(speed_forward=-10.0),
                                    projection=back, progress_delta=-0.5)
        assert result is not None
        assert result.terminated and result.reason is EndReason.WRONG_WAY

    def test_brief_backward_blip_is_not_wrong_way(self, straight_track):
        config = TerminationConfig(wrong_way_steps=10, min_steps_before_stall=10**9,
                                   stall_limit=10**9)
        tracker = TerminationTracker(straight_track, config)
        projection, _ = _projection(straight_track, station=100.0)
        tracker.update(frame=make_frame(speed_forward=20.0), projection=projection,
                       progress_delta=1.0)
        for _ in range(5):  # behind, but fewer than wrong_way_steps
            back, _ = _projection(straight_track, station=95.0)
            result = tracker.update(frame=make_frame(speed_forward=-5.0),
                                    projection=back, progress_delta=-0.25)
            assert result.reason is not EndReason.WRONG_WAY


class TestPenalties:
    def test_terminal_penalties_are_meaningful_and_negative(self, straight_track):
        reward = ProgressReward(straight_track)
        for reason in (EndReason.CRASH, EndReason.OUT_OF_BOUNDS, EndReason.FELL_OFF,
                       EndReason.WRONG_WAY):
            penalty = reward.terminal_penalty(reason)
            assert penalty < 0.0, f"{reason} must carry a negative reward"
            assert penalty <= -10.0, f"{reason} penalty must be meaningful, got {penalty}"

    def test_non_failure_reasons_have_no_terminal_penalty(self, straight_track):
        reward = ProgressReward(straight_track)
        for reason in (EndReason.FINISHED, EndReason.TIME_LIMIT, EndReason.STALLED,
                       EndReason.OFF_TRACK, EndReason.RUNNING):
            assert reward.terminal_penalty(reason) == 0.0

    def test_contact_penalty_charged_per_second(self, straight_track):
        reward = ProgressReward(straight_track)
        projection, _ = _projection(straight_track)
        frame = make_frame(speed_forward=20.0, has_lateral_contact=True)
        breakdown = reward.compute(frame=frame, projection=projection, prev_progress=0.0,
                                   finished=False, dt=0.05)
        assert breakdown.contact < 0.0
        quiet = reward.compute(frame=make_frame(speed_forward=20.0), projection=projection,
                               prev_progress=0.0, finished=False, dt=0.05)
        assert quiet.contact == 0.0

    def test_invalid_penalty_config_rejected(self, straight_track):
        with pytest.raises(ValueError, match="crash_penalty"):
            ProgressReward(straight_track, RewardConfig(crash_penalty=-1.0))

    def test_penalty_reason_property(self):
        assert TerminationResult(True, False, EndReason.CRASH).penalty_reason is EndReason.CRASH
        assert TerminationResult(True, False, EndReason.FINISHED).penalty_reason is None
        assert TerminationResult(False, True, EndReason.TIME_LIMIT).penalty_reason is None

    def test_termination_config_validation(self, straight_track):
        bad = TerminationConfig(crash_speed_loss=0.0)
        assert any("crash_speed_loss" in p for p in bad.validate())
        bad = TerminationConfig(wrong_way_steps=0)
        assert any("wrong_way_steps" in p for p in bad.validate())
        assert TerminationConfig().validate() == []


class ReverseDriver:
    """A minimal driver that drives the car backwards along a straight track.

    The simulated driver cannot reverse (its brakes stop at zero), so wrong-way is
    exercised end-to-end through this stub, which honours the GameDriver contract.
    """

    name = "reverse-stub"

    def __init__(self, track, speed: float = 10.0):
        from tmai.game.protocol import DriverCapabilities

        self.track = track
        self._speed = speed
        self._station = track.length * 0.8
        self.capabilities = DriverCapabilities(analog_control=True)
        self._opened = False

    def open(self) -> None:
        self._opened = True

    def close(self) -> None:
        self._opened = False

    def is_connected(self) -> bool:
        return self._opened

    def reset(self) -> GameFrame:
        self._station = self.track.length * 0.8
        return self._frame()

    def reposition(self, station: float, lateral: float = 0.0) -> GameFrame:
        self._station = station
        return self._frame()

    def step(self, action: Action) -> GameFrame:
        self._station -= self._speed * 0.05
        return self._frame()

    def set_speed_ratio(self, ratio: float) -> float:
        return float(ratio)

    def describe(self) -> dict:
        return {"driver": self.name}

    def _frame(self) -> GameFrame:
        position = self.track.point_at(max(0.0, self._station))
        yaw = float(np.arctan2(self.track.heading_at(self._station)[0],
                               self.track.heading_at(self._station)[2]))
        rotation = np.array(
            [[np.sin(yaw), 0.0, -np.cos(yaw)],
             [0.0, 1.0, 0.0],
             [np.cos(yaw), 0.0, np.sin(yaw)]],
            dtype=np.float64,
        )
        vehicle = VehicleState(
            position=position,
            velocity=np.array([0.0, 0.0, -self._speed]),
            rotation=rotation,
            speed_forward=-self._speed,
            speed_sideward=0.0,
        )
        race = RaceState(race_time=0.0)
        return GameFrame(vehicle=vehicle, race=race)


class TestEnvIntegration:
    """End-to-end through TrackmaniaEnv: the driving rules as the trainer sees them."""

    def _env(self, track, **driver_kwargs):
        from tmai.env.tm_env import EnvConfig, TrackmaniaEnv

        driver = SimulatedGameDriver(track, SimulatedDriverConfig(**driver_kwargs))
        driver.open()
        return TrackmaniaEnv(driver, track, EnvConfig()), driver

    def test_wall_collision_respawns_immediately_with_penalty(self):
        track = straight(length=400.0)
        # Low off-track drag so the car reaches the wall at speed; the default toy drag
        # (8/s) stops the car before the wall, which is parking, not a crash.
        env, driver = self._env(track, off_track_drag=1.5)
        try:
            env.reset(seed=0)
            # Steer off the lane at speed: the toy model's wall sits just past the lane
            # edge and absorbs most of the car's speed on impact.
            action = np.array([0.35, 1.0, 0.0], dtype=np.float32)
            terminated = False
            end_reason = ""
            reward_on_end = 0.0
            for _ in range(400):
                _, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    end_reason = info["end_reason"]
                    reward_on_end = reward
                    break
            assert terminated, "a significant wall collision must end the episode"
            assert end_reason == "crash", f"expected crash, got {end_reason}"
            assert reward_on_end < 0.0, "the crash step must carry a negative reward"
            assert info["reward/terminal_penalty"] <= -10.0
            assert info["has_lateral_contact"] is True
        finally:
            env.close()

    def test_leaving_the_map_respawns_with_penalty(self):
        track = straight(length=400.0)
        env, driver = self._env(track, off_track_drag=1.5)
        try:
            env.reset(seed=0)
            action = np.array([0.35, 1.0, 0.0], dtype=np.float32)
            end_reason = ""
            for _ in range(600):
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    end_reason = info["end_reason"]
                    break
            assert end_reason in ("crash", "out_of_bounds", "fell_off"), (
                f"leaving the track must respawn, got {end_reason}"
            )
        finally:
            env.close()

    def test_wrong_way_respawns_with_penalty(self):
        from tmai.env.tm_env import EnvConfig, TrackmaniaEnv

        track = straight(length=400.0)
        driver = ReverseDriver(track)
        driver.open()
        env = TrackmaniaEnv(driver, track, EnvConfig())
        try:
            env.reset(seed=0)
            terminated = False
            for _ in range(400):
                _, reward, terminated, truncated, info = env.step(
                    np.zeros(3, dtype=np.float32)
                )
                if terminated or truncated:
                    break
            assert terminated
            assert info["end_reason"] == "wrong_way"
            assert reward < 0.0
        finally:
            env.close()

    def test_crash_penalty_cannot_be_farmed(self):
        """Crashing ends the episode; the respawn teleport must not pay out progress."""
        track = straight(length=400.0)
        env, driver = self._env(track, off_track_drag=1.5)
        try:
            env.reset(seed=0)
            action = np.array([0.35, 1.0, 0.0], dtype=np.float32)
            crashed_at = None
            for step in range(400):
                _, reward, terminated, truncated, info = env.step(action)
                if terminated:
                    crashed_at = step
                    break
            assert crashed_at is not None
            assert info["end_reason"] == "crash"
            # The crash penalty plus the loss of future progress must dominate: the
            # episode return is negative even though the car drove for a while.
            assert info["episode_return"] < 0.0
            # And progress credit never exceeded what the car could physically drive:
            # the speed-derived cut clamp means the respawn teleport pays nothing.
            assert info["reward/raw_progress_metres"] <= 95.0 * 0.05 * 1.15 + 1e-6
        finally:
            env.close()

    def test_failure_reason_recorded_in_episode_info(self):
        track = straight(length=400.0)
        env, driver = self._env(track)
        try:
            env.reset(seed=0)
            action = np.array([1.0, 1.0, 0.0], dtype=np.float32)
            reasons = set()
            for _ in range(400):
                _, _, terminated, truncated, info = env.step(action)
                reasons.add(info["end_reason"])
                if terminated or truncated:
                    break
            assert info["end_reason"] in {r.value for r in EndReason}
        finally:
            env.close()

    def test_simulated_driver_reports_contact_capability(self):
        track = straight(length=100.0)
        driver = SimulatedGameDriver(track)
        assert driver.capabilities.reports_contact is True


class TestTelemetryContactMapping:
    """The real driver's telemetry maps the game's own contact fields."""

    def test_lateral_contact_and_wheels_mapped(self):
        from tmai.game.tminterface.telemetry import vehicle_state_from_sim_state

        class Wheel:
            def __init__(self, contact):
                self.real_time_state = type("RTS", (), {"has_ground_contact": contact})()

        class CarState:
            speed_forward = 10.0
            speed_sideward = 0.0
            rpm = 5000.0
            gearbox_state = 0
            input_steer = 0.25
            input_gas = 0.5
            input_brake = 0.0

        class Car:
            sync_vehicle_state = CarState()
            engine = type("Engine", (), {"gear": 3, "max_rpm": 8000.0})()
            is_sliding = False
            has_any_lateral_contact = True

        class SimState:
            flags = 0x2 | 0x80  # SIM_HAS_DYNA | SIM_HAS_PLAYER_INFO
            position = [1.0, 2.0, 3.0]
            velocity = [0.0, 0.0, 10.0]
            rotation_matrix = [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
            scene_mobil = Car()
            player_info = type("PI", (), {"race_time": 1000, "race_finished": False,
                                          "cur_cp_count": 0})()
            race_time = 1000
            num_respawns = 0
            simulation_wheels = [Wheel(True), Wheel(True), Wheel(False), Wheel(True)]

        state = vehicle_state_from_sim_state(SimState())
        assert state.has_lateral_contact is True
        assert state.num_wheels_ground_contact == 3
        assert state.has_ground_contact is True
        assert state.input_steer == 0.25

    def test_missing_wheels_region_falls_back(self):
        from tmai.game.tminterface.telemetry import vehicle_state_from_sim_state

        class CarState:
            speed_forward = 10.0
            speed_sideward = 0.0
            rpm = 5000.0
            gearbox_state = 0

        class Car:
            sync_vehicle_state = CarState()
            engine = type("Engine", (), {"gear": 3, "max_rpm": 8000.0})()
            is_sliding = False
            has_any_lateral_contact = False

        class SimState:
            flags = 0x2 | 0x80
            position = [0.0, 0.0, 0.0]
            velocity = [0.0, 0.0, 10.0]
            rotation_matrix = np.eye(3)
            scene_mobil = Car()
            player_info = type("PI", (), {"race_time": 0, "race_finished": False,
                                          "cur_cp_count": 0})()
            race_time = 0
            num_respawns = 0
            # no simulation_wheels attribute at all

        state = vehicle_state_from_sim_state(SimState())
        assert state.has_ground_contact is True  # conservative default
        assert state.num_wheels_ground_contact == 4
        assert state.has_lateral_contact is False

    def test_all_wheels_airborne_means_no_ground_contact(self):
        from tmai.game.tminterface.telemetry import vehicle_state_from_sim_state

        class Wheel:
            real_time_state = type("RTS", (), {"has_ground_contact": False})()

        class CarState:
            speed_forward = 10.0
            speed_sideward = 0.0
            rpm = 5000.0
            gearbox_state = 0

        class Car:
            sync_vehicle_state = CarState()
            engine = type("Engine", (), {"gear": 3, "max_rpm": 8000.0})()
            is_sliding = False
            has_any_lateral_contact = False

        class SimState:
            flags = 0x2 | 0x80
            position = [0.0, 0.0, 0.0]
            velocity = [0.0, 0.0, 10.0]
            rotation_matrix = np.eye(3)
            scene_mobil = Car()
            player_info = type("PI", (), {"race_time": 0, "race_finished": False,
                                          "cur_cp_count": 0})()
            race_time = 0
            num_respawns = 0
            simulation_wheels = [Wheel(), Wheel(), Wheel(), Wheel()]

        state = vehicle_state_from_sim_state(SimState())
        assert state.has_ground_contact is False
        assert state.num_wheels_ground_contact == 0
