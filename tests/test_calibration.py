"""Tests for telemetry calibration.

The point of the calibrator is to *catch* wrong assumptions about the real game, so the tests
feed it deliberately wrong conventions and require a failure. A calibrator that passed
everything would be worthless.
"""

from __future__ import annotations

import numpy as np
import pytest

from tmai.game.calibration import (
    CalibrationReport,
    TelemetryCalibrator,
    calibrate_driver,
    collect_calibration_sample,
)
from tmai.game.protocol import Action, GameFrame, RacePhase, RaceState, VehicleState
from tmai.game.simulated import SimulatedGameDriver
from tmai.tracks.synthetic import straight


def make_driving_frames(
    count: int = 40,
    *,
    dt: float = 0.05,
    speed: float = 20.0,
    forward_axis: int = 0,
    forward_sign: float = 1.0,
    position_scale: float = 1.0,
    speed_forward_matches: bool = True,
):
    """Synthesise a straight-line driving sample with controllable conventions.

    The car drives along +z at a constant speed, which is the condition every check assumes.
    """
    frames = []
    for i in range(count):
        t = i * dt
        z = speed * t * position_scale
        velocity = np.array([0.0, 0.0, speed])
        position = np.array([0.0, 0.0, z])

        # Build a rotation whose column `forward_axis` (times sign) points along +z.
        rotation = np.eye(3)
        if forward_axis == 0:
            # Column 0 = [sin, 0, cos] convention: yaw 0 gives forward = +z.
            rotation = np.array(
                [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
            )
        elif forward_axis == 2:
            rotation = np.array(
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
            )
        elif forward_axis == 1:
            rotation = np.array(
                [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float64
            )
        rotation = rotation * np.array(
            [[forward_sign, 1, 1], [1, forward_sign, 1], [1, 1, forward_sign]], dtype=np.float64
        )

        speed_forward = speed if speed_forward_matches else speed * 0.5

        frames.append(
            GameFrame(
                vehicle=VehicleState(
                    position=position,
                    velocity=velocity,
                    rotation=rotation,
                    speed_forward=speed_forward,
                    speed_sideward=0.0,
                    rpm=5000.0,
                    gear=3,
                ),
                race=RaceState(race_time=t, phase=RacePhase.RUNNING),
                wall_time=t,
            )
        )
    return frames


class TestForwardAxisCheck:
    def test_passes_with_the_documented_convention(self):
        report = TelemetryCalibrator().analyse(make_driving_frames(forward_axis=0))
        check = next(c for c in report.checks if c.name == "forward_axis")
        assert check.passed is True
        assert report.recommended_forward_axis == 0

    def test_detects_a_wrong_forward_axis(self):
        # If column 2 were the forward axis, column 0 would be perpendicular to velocity.
        report = TelemetryCalibrator().analyse(make_driving_frames(forward_axis=2))
        check = next(c for c in report.checks if c.name == "forward_axis")
        # The best candidate should be column 2 (with a flipped sign), not the assumed column 0.
        assert report.recommended_forward_axis == 2
        assert report.recommended_forward_sign == -1.0
        assert check.value == pytest.approx(1.0, abs=1e-6)

    def test_skips_frames_below_min_speed(self):
        frames = make_driving_frames(count=10, speed=1.0)
        report = TelemetryCalibrator(min_speed=5.0).analyse(frames)
        assert report.checks == []
        assert any("min_speed" in n for n in report.notes)

    def test_needs_enough_frames(self):
        report = TelemetryCalibrator().analyse(make_driving_frames(count=2))
        assert report.ok is False
        assert any("at least 3" in n for n in report.notes)


class TestSpeedForwardCheck:
    def test_passes_when_consistent(self):
        report = TelemetryCalibrator().analyse(make_driving_frames())
        check = next(c for c in report.checks if c.name == "speed_forward_consistency")
        assert check.passed is True

    def test_fails_when_speed_forward_disagrees(self):
        report = TelemetryCalibrator().analyse(
            make_driving_frames(speed_forward_matches=False)
        )
        check = next(c for c in report.checks if c.name == "speed_forward_consistency")
        assert check.passed is False
        assert check.value > 0.15


class TestPositionScaleCheck:
    def test_passes_when_units_are_consistent(self):
        report = TelemetryCalibrator().analyse(make_driving_frames(position_scale=1.0))
        check = next(c for c in report.checks if c.name == "position_scale")
        assert check.passed is True
        assert report.recommended_position_scale == pytest.approx(1.0, rel=0.05)

    def test_detects_a_unit_mismatch_and_recommends_a_scale(self):
        # Positions reported in centimetres while velocity is in m/s: ratio ~100.
        report = TelemetryCalibrator().analyse(make_driving_frames(position_scale=100.0))
        check = next(c for c in report.checks if c.name == "position_scale")
        assert check.passed is False
        assert report.recommended_position_scale == pytest.approx(0.01, rel=0.05)


class TestTickPeriodCheck:
    def test_passes_when_the_period_matches(self):
        report = TelemetryCalibrator(assumed_control_dt=0.05).analyse(make_driving_frames())
        check = next(c for c in report.checks if c.name == "tick_period")
        assert check.passed is True
        assert report.measured_tick_period == pytest.approx(0.05, rel=0.05)

    def test_fails_when_the_period_differs(self):
        report = TelemetryCalibrator(assumed_control_dt=0.05).analyse(
            make_driving_frames(dt=0.01)
        )
        check = next(c for c in report.checks if c.name == "tick_period")
        assert check.passed is False


class TestReport:
    def test_ok_requires_all_checks_to_pass(self):
        report = TelemetryCalibrator().analyse(make_driving_frames())
        assert report.ok is True
        assert all(c.passed for c in report.checks)

    def test_not_ok_with_a_failure(self):
        report = TelemetryCalibrator().analyse(make_driving_frames(position_scale=100.0))
        assert report.ok is False

    def test_format_is_readable(self):
        report = TelemetryCalibrator().analyse(make_driving_frames())
        text = report.format()
        assert "calibration:" in text
        assert "[PASS]" in text
        assert "forward_axis" in text

    def test_as_dict_is_json_serialisable(self):
        import json

        report = TelemetryCalibrator().analyse(make_driving_frames())
        assert json.dumps(report.as_dict())

    def test_empty_report_is_not_ok(self):
        assert CalibrationReport().ok is False


class TestCollectionAgainstDriver:
    """The collection path is exercised against the simulated driver."""

    def test_collects_the_requested_number_of_frames(self):
        driver = SimulatedGameDriver(straight(length=500.0))
        driver.open()
        frames = collect_calibration_sample(driver, steps=30)
        driver.close()
        assert len(frames) == 30

    def test_custom_action_is_used(self):
        driver = SimulatedGameDriver(straight(length=500.0))
        driver.open()
        frames = collect_calibration_sample(
            driver, steps=20, action=Action(throttle=0.0, brake=1.0)
        )
        driver.close()
        # Braking from rest must not produce forward motion.
        assert max(f.vehicle.speed_forward for f in frames) <= 1e-6

    def test_full_calibration_of_the_simulated_driver(self):
        """The simulated driver emits the documented conventions, so the *convention* checks
        must pass clean. This is the end-to-end check that the calibrator's expectations match
        what VehicleState actually promises.

        ``tick_period`` is expected to FAIL here: the simulated driver does not pace itself in
        real time (it runs as fast as the CPU allows), so its frame period is nowhere near the
        assumed control dt. That is the check working, not a bug -- which is exactly why it is
        asserted explicitly instead of being folded into an overall pass.
        """
        driver = SimulatedGameDriver(straight(length=1000.0))
        driver.open()
        report = calibrate_driver(driver, steps=60, assumed_control_dt=0.05)
        driver.close()
        assert report.samples == 60
        by_name = {c.name: c for c in report.checks}
        assert by_name["forward_axis"].passed is True, by_name["forward_axis"].detail
        assert by_name["speed_forward_consistency"].passed is True
        assert by_name["position_scale"].passed is True
        assert report.recommended_forward_axis == 0
        assert report.recommended_position_scale == pytest.approx(1.0, rel=0.05)

        assert by_name["tick_period"].passed is False
        assert report.measured_tick_period < 0.05
        assert report.ok is False
