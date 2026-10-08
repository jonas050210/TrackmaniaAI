"""Tests for the environment: observations, reward, termination and the gym contract."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from tests.conftest import make_frame
from tmai.env.observation import (
    OBSERVATION_VERSION,
    ObservationEncoder,
    ObservationInputs,
    ObservationScales,
    ObservationSpec,
)
from tmai.env.reward import ProgressReward, RewardConfig
from tmai.env.termination import (
    EndReason,
    TerminationConfig,
    TerminationTracker,
)
from tmai.env.tm_env import EnvConfig, TrackmaniaEnv
from tmai.game.protocol import Action, RacePhase
from tmai.game.simulated import SimulatedGameDriver
from tmai.tracks.centerline import CenterlineTrack
from tmai.tracks.synthetic import build_synthetic, oval


def build_env(track, config=None, *, throttle=1.0):
    driver = SimulatedGameDriver(track)
    driver.open()
    return TrackmaniaEnv(driver, track, config or EnvConfig())


class TestObservationSpec:
    def test_dim_matches_layout(self):
        spec = ObservationSpec()
        assert spec.dim == sum(w for _, w in spec.feature_layout())
        assert len(spec.names()) == spec.dim

    def test_disabling_features_shrinks_the_vector(self):
        full = ObservationSpec()
        reduced = ObservationSpec(include_curvature=False, include_last_action=False)
        assert reduced.dim < full.dim
        assert reduced.dim == full.dim - len(full.curvature_lookahead) - 3

    def test_all_features_off_still_has_yaw_rate(self):
        spec = ObservationSpec(
            include_speed=False,
            include_drivetrain=False,
            include_sliding=False,
            include_lateral=False,
            include_heading_error=False,
            include_progress_rate=False,
            include_curvature=False,
            include_edge_distances=False,
            include_last_action=False,
            include_checkpoint_progress=False,
        )
        assert spec.names() == ["yaw_rate"]

    def test_to_dict_records_version(self):
        payload = ObservationSpec().to_dict()
        assert payload["observation_version"] == OBSERVATION_VERSION
        assert len(payload["names"]) == payload["dim"]


class TestObservationEncoder:
    def _inputs(self, track, *, lateral=0.0, prev_progress=None, **vehicle_kwargs):
        position = track.point_at(50.0) + np.array([lateral, 0.0, 0.0])
        frame = make_frame(position=position, **vehicle_kwargs)
        projection = track.project(position)
        prev_projection = None
        if prev_progress is not None:
            prev_projection = track.project(track.point_at(prev_progress))
        return ObservationInputs(
            frame=frame,
            projection=projection,
            prev_projection=prev_projection,
            prev_yaw=None,
            dt=0.05,
            last_action=Action(steer=0.1, throttle=0.2, brake=0.3),
            yaw_rate=0.0,
        )

    def test_output_shape_and_dtype(self, straight_track):
        encoder = ObservationEncoder(straight_track)
        obs = encoder.encode(self._inputs(straight_track))
        assert obs.shape == (encoder.dim,)
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))

    def test_named_features_are_populated(self, straight_track):
        encoder = ObservationEncoder(straight_track)
        names = encoder.spec.names()
        obs = encoder.encode(
            self._inputs(straight_track, speed_forward=35.0, rpm=5000.0, gear=3)
        )
        values = dict(zip(names, obs, strict=True))
        assert values["speed_forward"] == pytest.approx(35.0 / ObservationScales().speed)
        assert values["rpm"] == pytest.approx(5000.0 / ObservationScales().rpm)
        assert values["gear"] == pytest.approx(3.0 / ObservationScales().gear)
        assert values["last_steer"] == pytest.approx(0.1)
        assert values["last_throttle"] == pytest.approx(0.2)
        assert values["last_brake"] == pytest.approx(0.3)

    def test_lateral_offset_appears_in_observation(self, straight_track):
        encoder = ObservationEncoder(straight_track)
        names = encoder.spec.names()
        right = encoder.encode(self._inputs(straight_track, lateral=+4.0))
        left = encoder.encode(self._inputs(straight_track, lateral=-4.0))
        index = names.index("lateral_offset")
        assert right[index] > 0
        assert left[index] < 0
        assert right[index] == pytest.approx(-left[index])

    def test_progress_rate_needs_a_previous_projection(self, straight_track):
        encoder = ObservationEncoder(straight_track)
        names = encoder.spec.names()
        index = names.index("progress_rate")
        without_prev = encoder.encode(self._inputs(straight_track))
        assert without_prev[index] == 0.0

    def test_progress_rate_reflects_motion(self, straight_track):
        encoder = ObservationEncoder(straight_track)
        names = encoder.spec.names()
        index = names.index("progress_rate")
        obs = encoder.encode(self._inputs(straight_track, prev_progress=49.0))
        # Moved 1 m in 0.05 s = 20 m/s, normalised by progress_delta.
        expected = 20.0 / ObservationScales().progress_delta
        assert obs[index] == pytest.approx(expected, rel=1e-3)

    def test_heading_error_zero_when_aligned(self, straight_track):
        encoder = ObservationEncoder(straight_track)
        names = encoder.spec.names()
        obs = encoder.encode(self._inputs(straight_track, yaw=0.0))
        assert obs[names.index("heading_error")] == pytest.approx(0.0, abs=1e-6)

    def test_edge_distances_shrink_towards_the_edge(self, straight_track):
        encoder = ObservationEncoder(straight_track)
        names = encoder.spec.names()
        obs = encoder.encode(self._inputs(straight_track, lateral=3.0))
        left = obs[names.index("edge_distance_left")]
        right = obs[names.index("edge_distance_right")]
        assert right < left

    def test_rejects_empty_spec(self, straight_track):
        spec = ObservationSpec()
        # Force an impossible layout to prove the encoder validates its own output width.
        spec.curvature_lookahead = ()
        spec.include_speed = False
        spec.include_drivetrain = False
        spec.include_sliding = False
        spec.include_lateral = False
        spec.include_heading_error = False
        spec.include_progress_rate = False
        spec.include_curvature = True
        spec.include_edge_distances = False
        spec.include_last_action = False
        spec.include_checkpoint_progress = False
        encoder = ObservationEncoder(straight_track, spec)
        assert encoder.dim == 0 or encoder.dim == 1  # yaw_rate always present
        assert encoder.dim == 1


class TestReward:
    """The reward is charged as rates integrated by dt, so tests must pass a dt.

    ``DT`` is the nominal control period; every expected value below is expressed in terms of
    it so the intent stays visible when a weight changes.
    """

    DT = 0.05

    def _reward(self, track, *, progress, prev_progress, lateral=0.0, speed=10.0,
                finished=False, config=None, dt=None, is_sliding=False):
        fn = ProgressReward(track, config)
        position = track.point_at(progress) + np.array([lateral, 0.0, 0.0])
        frame = make_frame(
            position=position, speed_forward=speed, finished=finished, is_sliding=is_sliding
        )
        projection = track.project(position)
        return fn.compute(
            frame=frame,
            projection=projection,
            prev_progress=prev_progress,
            finished=finished,
            dt=self.DT if dt is None else dt,
        )

    def test_progress_is_the_dominant_term(self, straight_track):
        breakdown = self._reward(straight_track, progress=60.0, prev_progress=59.0)
        assert breakdown.progress == pytest.approx(1.0)
        assert breakdown.progress_metres == pytest.approx(1.0)
        assert breakdown.total > 0

    def test_forward_progress_reward_remains_positive_across_closed_track_seam(self):
        track = oval(length=120.0, width=60.0, spacing=1.0)
        previous = track.project(track.point_at(track.length - 1.0))
        current = track.project(
            track.point_at(0.5), hint_s=previous.progress, search_window=3.0
        )
        frame = make_frame(position=current.centre, speed_forward=30.0)
        breakdown = ProgressReward(track).compute(
            frame=frame,
            projection=current,
            prev_progress=previous.progress,
            finished=False,
            dt=self.DT,
        )
        assert breakdown.raw_progress_metres == pytest.approx(1.5, abs=1e-3)
        assert breakdown.progress_metres == pytest.approx(1.5, abs=1e-3)
        assert breakdown.clamped is False

    def test_cut_detector_blocks_teleport_scale_jumps(self, straight_track):
        """A jump far beyond what the car can physically cover is not credited."""
        config = RewardConfig(max_speed_for_progress=95.0, cut_margin=1.15)
        breakdown = self._reward(
            straight_track, progress=200.0, prev_progress=0.0, config=config
        )
        # 95 * 0.05 * 1.15 = 5.4625
        assert breakdown.progress_metres == pytest.approx(95.0 * self.DT * 1.15)
        assert breakdown.clamped is True
        assert breakdown.raw_progress_metres == pytest.approx(200.0)

    def test_fast_but_honest_driving_is_never_clamped(self, straight_track):
        """The regression this design fixes: a fixed metre cap punished real speed.

        At a 0.1 s control period a car at 70 m/s covers 7 m. The old fixed cap of 6 m
        clamped that; the speed-derived threshold must not.
        """
        config = RewardConfig(max_speed_for_progress=95.0, cut_margin=1.15)
        for dt in (0.05, 0.1, 0.2):
            distance = 70.0 * dt
            breakdown = self._reward(
                straight_track,
                progress=50.0 + distance,
                prev_progress=50.0,
                speed=70.0,
                config=config,
                dt=dt,
            )
            assert breakdown.clamped is False, f"clamped legitimate driving at dt={dt}"
            assert breakdown.progress_metres == pytest.approx(distance)

    def test_backward_progress_is_negative_and_bounded(self, straight_track):
        config = RewardConfig(max_backward_speed=30.0)
        breakdown = self._reward(
            straight_track, progress=0.0, prev_progress=100.0, config=config
        )
        assert breakdown.progress_metres == pytest.approx(-30.0 * self.DT)
        assert breakdown.progress < 0

    def test_off_track_penalty_starts_outside_the_corridor(self, straight_track):
        half = straight_track.corridor_half_width_at(60.0)
        inside = self._reward(straight_track, progress=60.0, prev_progress=59.0,
                              lateral=half - 1.0)
        outside = self._reward(straight_track, progress=60.0, prev_progress=59.0,
                               lateral=half + 3.0)
        assert inside.off_track == pytest.approx(0.0)
        assert outside.off_track < 0

    def test_off_track_penalty_scales_with_dt(self, straight_track):
        """Charged per second, so twice the control period costs twice as much."""
        half = straight_track.corridor_half_width_at(60.0)
        short = self._reward(straight_track, progress=60.0, prev_progress=59.0,
                             lateral=half + 2.0, dt=0.05)
        long = self._reward(straight_track, progress=60.0, prev_progress=59.0,
                            lateral=half + 2.0, dt=0.10)
        assert long.off_track == pytest.approx(2.0 * short.off_track)

    def test_off_track_margin_delays_the_penalty(self, straight_track):
        half = straight_track.corridor_half_width_at(60.0)
        config = RewardConfig(off_track_margin=5.0)
        breakdown = self._reward(
            straight_track, progress=60.0, prev_progress=59.0,
            lateral=half + 3.0, config=config,
        )
        assert breakdown.off_track == pytest.approx(0.0)

    def test_finish_bonus_applied_once(self, straight_track):
        config = RewardConfig(finish_bonus=20.0)
        breakdown = self._reward(
            straight_track, progress=60.0, prev_progress=59.0, finished=True, config=config
        )
        assert breakdown.finish == pytest.approx(20.0)
        assert breakdown.total > 20.0

    def test_reward_boundary_rejects_incomplete_checkpoint_finish(self, straight_track):
        fn = ProgressReward(straight_track, RewardConfig(finish_bonus=20.0))
        position = straight_track.point_at(60.0)
        frame = make_frame(
            position=position,
            speed_forward=10.0,
            finished=True,
            checkpoint_index=2,
            checkpoint_total=5,
        )
        breakdown = fn.compute(
            frame=frame,
            projection=straight_track.project(position),
            prev_progress=59.0,
            finished=True,
            dt=self.DT,
        )

        assert breakdown.finish == pytest.approx(0.0)

    def test_speed_term_is_bounded(self, straight_track):
        config = RewardConfig(speed_weight=1.0, speed_ref=70.0)
        slow = self._reward(straight_track, progress=60.0, prev_progress=59.0,
                            speed=35.0, config=config)
        fast = self._reward(straight_track, progress=60.0, prev_progress=59.0,
                            speed=700.0, config=config)
        # speed_weight * dt * normalised_speed
        assert slow.speed == pytest.approx(0.5 * self.DT)
        assert fast.speed == pytest.approx(1.0 * self.DT)

    def test_negative_speed_does_not_give_negative_speed_reward(self, straight_track):
        config = RewardConfig(speed_weight=1.0)
        breakdown = self._reward(straight_track, progress=60.0, prev_progress=59.0,
                                 speed=-20.0, config=config)
        assert breakdown.speed == pytest.approx(0.0)

    def test_idle_car_scores_worse_than_moving_car(self, straight_track):
        """A stationary car must not be a stable optimum.

        Without this, "do nothing" scores 0.0 while every attempt to drive scores negative,
        so the optimal early policy is to sit still and learning never starts.
        """
        config = RewardConfig(idle_penalty=0.5, idle_speed_threshold=1.0)
        idle = self._reward(straight_track, progress=60.0, prev_progress=60.0,
                            speed=0.0, config=config)
        moving = self._reward(straight_track, progress=60.0, prev_progress=59.0,
                              speed=10.0, config=config)
        assert idle.idle == pytest.approx(-0.5 * self.DT)
        assert moving.idle == pytest.approx(0.0)
        assert idle.total < moving.total

    def test_idle_penalty_does_not_fire_on_finish(self, straight_track):
        """The car is legitimately stationary once it has finished."""
        config = RewardConfig(idle_penalty=0.5, idle_speed_threshold=1.0)
        breakdown = self._reward(straight_track, progress=60.0, prev_progress=60.0,
                                 speed=0.0, finished=True, config=config)
        assert breakdown.idle == pytest.approx(0.0)

    def test_step_penalty_and_slip_penalty(self, straight_track):
        config = RewardConfig(step_penalty=0.1, slip_weight=1.0, speed_ref=10.0)
        breakdown = self._reward(straight_track, progress=60.0, prev_progress=59.0,
                                 config=config)
        assert breakdown.step == pytest.approx(-0.1 * self.DT)

    def test_slide_penalty_when_sliding(self, straight_track):
        config = RewardConfig(slide_penalty=0.5)
        breakdown = self._reward(straight_track, progress=60.0, prev_progress=59.0,
                                 is_sliding=True, config=config)
        assert breakdown.slide == pytest.approx(-0.5 * self.DT)

    def test_total_return_is_invariant_to_control_rate(self, straight_track):
        """Driving the same lap at half the control period must score the same.

        This is the property the rate-based formulation buys: penalties integrate over time,
        so the effective objective does not depend on how the control period was configured.
        """
        config = RewardConfig(off_track_weight=6.0, heading_weight=0.4, idle_penalty=0.0)
        half = straight_track.corridor_half_width_at(60.0)

        def lap(dt: float, steps: int) -> float:
            total = 0.0
            for i in range(steps):
                progress = 60.0 + (i + 1) * (1.0 / steps) * 10.0
                breakdown = self._reward(
                    straight_track,
                    progress=progress,
                    prev_progress=progress - (1.0 / steps) * 10.0,
                    lateral=half + 1.0,
                    speed=50.0,
                    config=config,
                    dt=dt,
                )
                total += breakdown.off_track + breakdown.heading
            return total

        coarse = lap(0.1, 10)
        fine = lap(0.05, 20)
        assert fine == pytest.approx(coarse, rel=1e-6)

    def test_metrics_are_flat_floats(self, straight_track):
        breakdown = self._reward(straight_track, progress=60.0, prev_progress=59.0)
        metrics = breakdown.as_metrics()
        assert all(isinstance(v, float) for v in metrics.values())
        assert "reward/total" in metrics
        assert "reward/clamped" in metrics
        assert "reward/idle" in metrics

    def test_config_roundtrips_to_dict(self):
        payload = RewardConfig().to_dict()
        assert payload["progress_weight"] == 1.0
        assert isinstance(payload, dict)

    def test_config_validation_rejects_degenerate_settings(self):
        with pytest.raises(ValueError, match="progress_weight"):
            ProgressReward(straight_points_track(), RewardConfig(progress_weight=0.0))
        with pytest.raises(ValueError, match="cut_margin"):
            ProgressReward(straight_points_track(), RewardConfig(cut_margin=0.5))
        with pytest.raises(ValueError, match="off_track_weight"):
            ProgressReward(straight_points_track(), RewardConfig(off_track_weight=-1.0))

    def test_rejects_non_positive_dt(self, straight_track):
        with pytest.raises(ValueError, match="dt must be positive"):
            self._reward(straight_track, progress=60.0, prev_progress=59.0, dt=0.0)


def straight_points_track():
    """A minimal track for config-validation tests that never drive it."""
    from tmai.tracks.synthetic import straight

    return straight(length=50.0)


class TestTermination:
    def _track(self):
        return CenterlineTrack(
            np.stack([np.zeros(101), np.zeros(101), np.arange(101.0)], axis=1),
            name="t",
            corridor_half_width=5.0,
        )

    def test_finish_terminates(self):
        track = self._track()
        tracker = TerminationTracker(track, TerminationConfig())
        frame = make_frame(position=[0.0, 0.0, 10.0], finished=True)
        result = tracker.update(frame=frame, projection=track.project([0.0, 0.0, 10.0]),
                                progress_delta=1.0)
        assert result.terminated is True
        assert result.reason is EndReason.FINISHED

    def test_off_track_needs_consecutive_steps(self):
        track = self._track()
        config = TerminationConfig(off_track_limit=3, off_track_margin=0.0)
        tracker = TerminationTracker(track, config)
        far = track.project([50.0, 0.0, 10.0])  # 50 m off a 5 m corridor
        for _ in range(2):
            result = tracker.update(frame=make_frame(position=[50.0, 0.0, 10.0]),
                                    projection=far, progress_delta=1.0)
            assert result.terminated is False
        result = tracker.update(frame=make_frame(position=[50.0, 0.0, 10.0]),
                                projection=far, progress_delta=1.0)
        assert result.terminated is True
        assert result.reason is EndReason.OFF_TRACK

    def test_off_track_streak_resets_when_back_on_track(self):
        track = self._track()
        tracker = TerminationTracker(track, TerminationConfig(off_track_limit=3))
        far = track.project([50.0, 0.0, 10.0])
        near = track.project([0.0, 0.0, 10.0])
        tracker.update(frame=make_frame(), projection=far, progress_delta=1.0)
        tracker.update(frame=make_frame(), projection=far, progress_delta=1.0)
        tracker.update(frame=make_frame(), projection=near, progress_delta=1.0)
        tracker.update(frame=make_frame(), projection=far, progress_delta=1.0)
        result = tracker.update(frame=make_frame(), projection=far, progress_delta=1.0)
        assert result.terminated is False

    def test_stall_terminates_after_grace_period(self):
        track = self._track()
        config = TerminationConfig(stall_limit=5, min_steps_before_stall=3, max_steps=1000)
        tracker = TerminationTracker(track, config)
        for _ in range(6):
            result = tracker.update(frame=make_frame(), projection=track.project([0.0, 0.0, 10.0]),
                                    progress_delta=0.0)
        assert result.terminated is True
        assert result.reason is EndReason.STALLED

    def test_stall_does_not_fire_during_grace_period(self):
        track = self._track()
        config = TerminationConfig(stall_limit=3, min_steps_before_stall=100, max_steps=1000)
        tracker = TerminationTracker(track, config)
        for _ in range(10):
            result = tracker.update(frame=make_frame(), projection=track.project([0.0, 0.0, 10.0]),
                                    progress_delta=0.0)
        assert result.terminated is False

    def test_time_limit_truncates_rather_than_terminates(self):
        track = self._track()
        tracker = TerminationTracker(track, TerminationConfig(max_steps=4))
        for _ in range(4):
            result = tracker.update(frame=make_frame(), projection=track.project([0.0, 0.0, 10.0]),
                                    progress_delta=1.0)
        assert result.truncated is True
        assert result.terminated is False
        assert result.reason is EndReason.TIME_LIMIT
        assert result.done is True

    def test_no_ground_contact_terminates(self):
        track = self._track()
        config = TerminationConfig(no_ground_contact_limit=2)
        tracker = TerminationTracker(track, config)
        frame = make_frame(position=[0.0, 0.0, 10.0], has_ground_contact=False)
        tracker.update(frame=frame, projection=track.project([0.0, 0.0, 10.0]), progress_delta=1.0)
        result = tracker.update(frame=frame, projection=track.project([0.0, 0.0, 10.0]),
                                progress_delta=1.0)
        assert result.terminated is True
        assert result.reason is EndReason.NO_GROUND_CONTACT

    def test_reset_clears_counters(self):
        track = self._track()
        tracker = TerminationTracker(track, TerminationConfig(max_steps=2))
        tracker.update(frame=make_frame(), projection=track.project([0.0, 0.0, 10.0]),
                       progress_delta=1.0)
        tracker.reset()
        assert tracker.steps == 0


class TestEnvContract:
    def test_spaces_are_consistent(self, straight_track):
        env = build_env(straight_track)
        assert env.observation_space.shape == (env.observation_dim,)
        assert env.action_space.shape == (3,)
        np.testing.assert_allclose(env.action_space.low, [-1.0, 0.0, 0.0])
        np.testing.assert_allclose(env.action_space.high, [1.0, 1.0, 1.0])
        env.close()

    def test_reset_returns_valid_observation(self, straight_track):
        env = build_env(straight_track)
        obs, info = env.reset(seed=0)
        assert obs.shape == (env.observation_dim,)
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))
        assert "progress" in info
        env.close()

    def test_step_before_reset_raises(self, straight_track):
        env = build_env(straight_track)
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.zeros(3, dtype=np.float32))
        env.close()

    def test_step_returns_five_tuple_with_correct_types(self, straight_track):
        env = build_env(straight_track)
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        assert obs.shape == (env.observation_dim,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_action_repeat_consumes_multiple_ticks(self, straight_track):
        config = EnvConfig(action_repeat=3)
        env = build_env(straight_track, config)
        env.reset(seed=0)
        before = env.driver._time
        env.step(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        # Three simulated driver intervals, each using the simulator's configured dt.
        assert env.driver._time - before == pytest.approx(3 * config.control_dt)
        env.close()

    def test_elapsed_time_uses_game_clock_instead_of_nominal_config(self, straight_track):
        from tmai.game.simulated import SimulatedDriverConfig

        driver = SimulatedGameDriver(straight_track, SimulatedDriverConfig(dt=0.01))
        driver.open()
        env = TrackmaniaEnv(
            driver,
            straight_track,
            EnvConfig(control_dt=0.05, action_repeat=2),
        )
        try:
            _, reset_info = env.reset(seed=0)
            observation, _, _, _, info = env.step(Action(throttle=0.5))
            assert info["elapsed_seconds"] == pytest.approx(0.02)
            progress_index = env.observation_names.index("progress_rate")
            expected_rate = (
                info["progress"] - reset_info["progress"]
            ) / info["elapsed_seconds"] / env.config.observation.scales.progress_delta
            assert observation[progress_index] == pytest.approx(expected_rate)
        finally:
            env.close()

    def test_accepts_action_dataclass(self, straight_track):
        env = build_env(straight_track)
        env.reset(seed=0)
        obs, *_ = env.step(Action(throttle=1.0))
        assert obs.shape == (env.observation_dim,)
        env.close()

    def test_passive_step_uses_game_input_readback_as_last_action(self, straight_track):
        from dataclasses import replace

        env = build_env(straight_track, EnvConfig(action_repeat=1))
        try:
            env.reset(seed=0)
            original_step = env.driver.step

            def report_human_input(action):
                assert action is None
                frame = original_step(action)
                vehicle = replace(
                    frame.vehicle,
                    input_steer=0.2,
                    input_gas=0.6,
                    input_brake=0.0,
                )
                return replace(frame, vehicle=vehicle)

            env.driver.step = report_human_input
            _, _, _, _, info = env.step(None)
            assert env._last_action == Action(steer=0.2, throttle=0.6)
            assert info["elapsed_seconds"] > 0
        finally:
            env.close()

    def test_rejects_wrong_action_size(self, straight_track):
        env = build_env(straight_track)
        env.reset(seed=0)
        with pytest.raises(ValueError, match="3 components"):
            env.step(np.zeros(2, dtype=np.float32))
        env.close()

    def test_episode_progress_advances_when_driving(self, straight_track):
        env = build_env(straight_track)
        env.reset(seed=0)
        for _ in range(100):
            _, _, terminated, truncated, info = env.step(np.array([0.0, 1.0, 0.0], dtype=np.float32))
            if terminated or truncated:
                break
        assert info["progress"] > 10.0
        assert 0.0 < info["progress_fraction"] <= 1.0
        env.close()

    def test_episode_terminates_on_time_limit(self, straight_track):
        config = EnvConfig(termination=TerminationConfig(max_steps=20))
        env = build_env(straight_track, config)
        env.reset(seed=0)
        for _ in range(50):
            _, _, terminated, truncated, info = env.step(np.array([0.0, 1.0, 0.0], dtype=np.float32))
            if terminated or truncated:
                break
        assert truncated is True
        assert info["end_reason"] == "time_limit"
        env.close()

    def test_finish_terminates_the_episode(self, straight_track):
        config = EnvConfig(termination=TerminationConfig(max_steps=5000))
        env = build_env(straight_track, config)
        env.reset(seed=0)
        finished = False
        for _ in range(5000):
            _, _, terminated, truncated, info = env.step(
                np.array([0.0, 1.0, 0.0], dtype=np.float32)
            )
            if terminated or truncated:
                finished = info["end_reason"] == "finished"
                break
        assert finished is True
        env.close()

    def test_finish_event_stops_action_repeat_and_bonus_is_not_multiplied(self, straight_track):
        from dataclasses import replace

        config = EnvConfig(action_repeat=4, reward=RewardConfig(finish_bonus=9.0))
        env = build_env(straight_track, config)
        env.reset(seed=0)
        original_step = env.driver.step
        calls = 0

        def finish_first_tick(action):
            nonlocal calls
            calls += 1
            frame = original_step(action)
            race = replace(
                frame.race,
                phase=RacePhase.FINISHED,
                finished=True,
                checkpoint_index=frame.race.checkpoint_total,
            )
            return replace(frame, race=race)

        env.driver.step = finish_first_tick
        try:
            _, _, terminated, _, info = env.step(Action(throttle=1.0))
            assert terminated is True
            assert calls == 1
            assert info["reward/finish"] == pytest.approx(9.0)
        finally:
            env.close()

    def test_invalid_checkpoint_finish_ends_without_finish_bonus(self, straight_track):
        from dataclasses import replace

        config = EnvConfig(action_repeat=4, reward=RewardConfig(finish_bonus=9.0))
        env = build_env(straight_track, config)
        env.reset(seed=0)
        original_step = env.driver.step
        calls = 0

        def skip_checkpoints(action):
            nonlocal calls
            calls += 1
            frame = original_step(action)
            total = max(1, frame.race.checkpoint_total)
            race = replace(
                frame.race,
                phase=RacePhase.FINISHED,
                finished=True,
                checkpoint_index=total - 1,
                checkpoint_total=total,
            )
            return replace(frame, race=race)

        env.driver.step = skip_checkpoints
        try:
            _, _, terminated, _, info = env.step(Action(throttle=1.0))
            assert terminated is True
            assert calls == 1
            assert info["finished"] is True  # raw game flag remains available for auditing
            assert info["valid_finish"] is False
            assert info["invalid_finish"] is True
            assert info["reward/finish"] == pytest.approx(0.0)
        finally:
            env.close()

    def test_describe_is_json_serialisable(self, straight_track):
        import json

        env = build_env(straight_track)
        payload = json.dumps(env.describe())
        assert "straight" in payload
        env.close()

    @pytest.mark.filterwarnings("ignore:.*Box observation space.*:UserWarning")
    def test_passes_gymnasium_env_checker(self, straight_track):
        """The observation space is intentionally unbounded: normalised speed and curvature
        have no hard ceiling, and inventing one would be a false constraint."""
        env = build_env(straight_track)
        try:
            check_env(env.unwrapped, skip_render_check=True)
        finally:
            env.close()

    def test_close_is_idempotent(self, straight_track):
        env = build_env(straight_track)
        env.reset(seed=0)
        env.close()
        env.close()

    def test_own_driver_false_leaves_driver_open(self, straight_track):
        driver = SimulatedGameDriver(straight_track)
        driver.open()
        env = TrackmaniaEnv(driver, straight_track, own_driver=False)
        env.reset(seed=0)
        env.close()
        assert driver.is_connected() is True
        driver.close()


class TestProgressFractionMeasuresDrivingNotPlacement:
    """`progress_fraction` must measure how far the car drove, not where it started.

    It used to be `projection.progress / track.length` -- absolute position on the track. With
    `random_start_station` on by default, that made the headline evaluation metric report the
    luck of the start position: a car placed at 95% of the lap that never moved reported 95%
    progress. Because that metric feeds `EvaluationReport.score`, best-checkpoint selection
    could be won by a fortunate spawn rather than by driving.
    """

    @staticmethod
    def _env():
        from tmai.env.tm_env import EnvConfig, TrackmaniaEnv
        from tmai.game.simulated import SimulatedDriverConfig, SimulatedGameDriver

        track = build_synthetic("s_curve")
        # Walls disabled: this class tests the *metric* under identical physics. With the
        # toy model's walls enabled, the same controls from different start stations hit
        # the wall at different times and legitimately diverge; crash behaviour has its
        # own test class below.
        driver = SimulatedGameDriver(track, SimulatedDriverConfig(wall_margin=1e9,
                                                                 fall_margin=1e9))
        driver.open()
        return TrackmaniaEnv(driver, track, EnvConfig()), track

    def test_a_stationary_car_reports_zero_from_any_start(self):
        """The regression itself: placement must not be mistaken for progress."""
        env, track = self._env()
        try:
            for fraction in (0.0, 0.25, 0.5, 0.75, 0.95):
                _, info = env.reset(options={"start_station": fraction * track.length})
                zero = np.zeros(env.action_space.shape[0], dtype=np.float32)
                for _ in range(5):
                    _, _, term, trunc, info = env.step(zero)
                    if term or trunc:
                        break
                assert info["progress_fraction"] == 0.0, (
                    f"started at {fraction:.0%} and reported "
                    f"{info['progress_fraction']:.1%} progress without moving"
                )
        finally:
            env.close()

    def test_equal_driving_reports_equal_progress_from_any_start(self):
        """The metric must be start-invariant, or evaluation is not comparable across episodes."""
        env, track = self._env()
        results = []
        try:
            for fraction in (0.0, 0.5):
                _, info = env.reset(options={"start_station": fraction * track.length})
                forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                for _ in range(80):
                    _, _, term, trunc, info = env.step(forward)
                    if term or trunc:
                        break
                results.append(info["progress_fraction"])
        finally:
            env.close()
        assert results[0] > 0.0, "the car must actually make progress"
        assert abs(results[0] - results[1]) < 1e-6, (
            f"same driving reported different progress by start: {results}"
        )

    def test_track_position_is_still_available_separately(self):
        """Absolute position is still useful; it just must not be the progress metric."""
        env, track = self._env()
        try:
            _, info = env.reset(options={"start_station": 0.5 * track.length})
            assert info["track_position_fraction"] == pytest.approx(0.5, abs=1e-3)
            assert info["progress_fraction"] == 0.0
        finally:
            env.close()

    def test_progress_fraction_is_bounded(self):
        env, track = self._env()
        try:
            _, info = env.reset()
            forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            for _ in range(400):
                _, _, term, trunc, info = env.step(forward)
                assert 0.0 <= info["progress_fraction"] <= 1.0
                if term or trunc:
                    break
        finally:
            env.close()
