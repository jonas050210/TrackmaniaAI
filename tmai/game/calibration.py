"""Calibrate the conventions of real game telemetry instead of assuming them.

TMInterface exposes raw game structures whose unit and axis conventions Nadeo does not
document: which column of the rotation matrix is the car's forward axis, whether positions
and velocities share a unit, and the exact physics tick period. Guessing wrong here silently
poisons every observation and the whole reward.

So instead of hard-coding them, this module measures them from a short real driving sample
and reports pass/fail with numbers. Run it once per game version / setup::

    tmai doctor --calibrate

Each check is a physical invariant that must hold regardless of the game's internal
conventions, which is what makes the report trustworthy:

===============================  ==========================================================
check                            invariant
===============================  ==========================================================
``forward_axis``                 the forward column of the rotation matrix is parallel to
                                 the velocity vector while driving forwards
``speed_forward_consistency``    ``speed_forward`` equals ``dot(velocity, forward)``
``position_scale``               ``|dPosition|`` equals ``|velocity| * dt`` in the same unit
``tick_period``                  the observed frame period matches the assumed control dt
===============================  ==========================================================
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tmai.game.protocol import Action, GameDriver, GameFrame

logger = logging.getLogger(__name__)

#: Columns of the rotation matrix to try as the vehicle forward axis (and their negations).
_CANDIDATE_AXES = ((0, 1.0), (1, 1.0), (2, 1.0), (0, -1.0), (1, -1.0), (2, -1.0))


@dataclass
class CheckResult:
    """One calibration check."""

    name: str
    passed: bool
    value: float
    reference: float
    tolerance: float
    detail: str
    #: Machine-readable recommendations produced by this check (e.g. ``{"axis": 2}``). Kept
    #: as data rather than parsed back out of ``detail``, which would be fragile.
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "reference": self.reference,
            "tolerance": self.tolerance,
            "detail": self.detail,
            "extras": dict(self.extras),
        }


@dataclass
class CalibrationReport:
    """Outcome of a calibration run."""

    checks: list[CheckResult] = field(default_factory=list)
    recommended_forward_axis: int = 0
    recommended_forward_sign: float = 1.0
    recommended_position_scale: float = 1.0
    measured_tick_period: float = 0.0
    samples: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "samples": self.samples,
            "recommended_forward_axis": self.recommended_forward_axis,
            "recommended_forward_sign": self.recommended_forward_sign,
            "recommended_position_scale": self.recommended_position_scale,
            "measured_tick_period": self.measured_tick_period,
            "checks": [c.as_dict() for c in self.checks],
            "notes": self.notes,
        }

    def format(self) -> str:
        lines = [
            f"calibration: {'OK' if self.ok else 'PROBLEMS FOUND'} "
            f"({self.samples} samples, tick period {self.measured_tick_period * 1000:.2f} ms)"
        ]
        for check in self.checks:
            lines.append(
                f"  [{check.status}] {check.name}: {check.value:.4f} "
                f"(expected {check.reference:.4f} +/- {check.tolerance:.4f}) -- {check.detail}"
            )
        lines.append(
            f"  recommended: forward_axis={self.recommended_forward_axis} "
            f"sign={self.recommended_forward_sign:+.0f} "
            f"position_scale={self.recommended_position_scale:.4f}"
        )
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


class TelemetryCalibrator:
    """Analyses a sample of frames and reports which conventions hold."""

    def __init__(
        self,
        *,
        assumed_control_dt: float = 0.05,
        min_speed: float = 5.0,
        axis_tolerance: float = 0.15,
        scale_tolerance: float = 0.15,
        tick_tolerance: float = 0.35,
    ) -> None:
        self.assumed_control_dt = assumed_control_dt
        self.min_speed = min_speed
        self.axis_tolerance = axis_tolerance
        self.scale_tolerance = scale_tolerance
        self.tick_tolerance = tick_tolerance

    def analyse(self, frames: Sequence[GameFrame]) -> CalibrationReport:
        """Run every check over ``frames`` (consecutive frames from one driving sample)."""
        report = CalibrationReport(samples=len(frames))
        if len(frames) < 3:
            report.notes.append("need at least 3 frames to calibrate")
            return report

        usable = [
            i
            for i, f in enumerate(frames)
            if float(np.linalg.norm(f.vehicle.velocity)) >= self.min_speed
        ]
        if len(usable) < 2:
            report.notes.append(
                f"no frames above min_speed={self.min_speed} m/s; drive faster during calibration"
            )
            return report

        report.checks.append(self._check_forward_axis(frames, usable))
        report.checks.append(self._check_speed_forward(frames, usable))
        report.checks.append(self._check_position_scale(frames, usable))
        report.checks.append(self._check_tick_period(frames))

        # Aggregate the machine-readable recommendations. Doing this here (rather than in
        # calibrate_driver) means the report is complete no matter how it was produced.
        for check in report.checks:
            extras = check.extras
            if check.name == "forward_axis" and "axis" in extras:
                report.recommended_forward_axis = int(extras["axis"])
                report.recommended_forward_sign = float(extras.get("sign", 1.0))
            elif check.name == "position_scale" and "scale" in extras:
                report.recommended_position_scale = float(extras["scale"])
            elif check.name == "tick_period" and "period" in extras:
                report.measured_tick_period = float(extras["period"])
        return report

    # -- individual checks ----------------------------------------------------------

    def _check_forward_axis(
        self, frames: Sequence[GameFrame], usable: list[int]
    ) -> CheckResult:
        """Find the rotation column that is parallel to the velocity while moving forward."""
        best_axis, best_sign, best_score = 0, 1.0, -2.0
        for axis, sign in _CANDIDATE_AXES:
            scores = []
            for i in usable:
                v = frames[i].vehicle.velocity
                speed = float(np.linalg.norm(v))
                if speed < 1e-6:
                    continue
                column = frames[i].vehicle.rotation[:, axis] * sign
                norm = float(np.linalg.norm(column))
                if norm < 1e-9:
                    continue
                scores.append(float(np.dot(column / norm, v / speed)))
            if scores:
                mean = float(np.mean(scores))
                if mean > best_score:
                    best_axis, best_sign, best_score = axis, sign, mean
        passed = best_score >= 1.0 - self.axis_tolerance
        return CheckResult(
            name="forward_axis",
            passed=passed,
            value=best_score,
            reference=1.0,
            tolerance=self.axis_tolerance,
            detail=(
                f"best forward candidate is rotation column {best_axis} "
                f"(sign {best_sign:+.0f}); cosine with velocity = {best_score:.3f}"
            ),
            extras={"axis": float(best_axis), "sign": float(best_sign)},
        )

    def _check_speed_forward(
        self, frames: Sequence[GameFrame], usable: list[int]
    ) -> CheckResult:
        """``speed_forward`` must equal the longitudinal component of world velocity."""
        errors = []
        for i in usable:
            vehicle = frames[i].vehicle
            speed = float(np.linalg.norm(vehicle.velocity))
            if speed < 1e-6:
                continue
            # Use the same convention the environment uses so the check is meaningful.
            forward = vehicle.forward_vector()
            projected = float(np.dot(vehicle.velocity, forward))
            errors.append(abs(projected - vehicle.speed_forward) / max(speed, 1e-6))
        value = float(np.mean(errors)) if errors else float("nan")
        passed = bool(errors) and value <= self.axis_tolerance
        return CheckResult(
            name="speed_forward_consistency",
            passed=passed,
            value=value,
            reference=0.0,
            tolerance=self.axis_tolerance,
            detail=f"mean relative mismatch between speed_forward and dot(velocity, forward) = {value:.3f}",
        )

    def _check_position_scale(
        self, frames: Sequence[GameFrame], usable: list[int]
    ) -> CheckResult:
        """``|dPosition|`` must equal ``|velocity| * dt``; the ratio calibrates the unit."""
        ratios = []
        for prev, curr in zip(usable[:-1], usable[1:], strict=False):
            if curr - prev < 1:
                continue
            dt = self.assumed_control_dt * (curr - prev)
            delta = float(np.linalg.norm(frames[curr].vehicle.position - frames[prev].vehicle.position))
            speed = float(np.linalg.norm(frames[prev].vehicle.velocity))
            if speed * dt < 1e-6:
                continue
            ratios.append(delta / (speed * dt))
        value = float(np.median(ratios)) if ratios else float("nan")
        passed = bool(ratios) and abs(value - 1.0) <= self.scale_tolerance
        scale = 1.0 / value if bool(ratios) and value > 0 else 1.0
        return CheckResult(
            name="position_scale",
            passed=passed,
            value=value,
            reference=1.0,
            tolerance=self.scale_tolerance,
            detail=(
                f"|dPosition| / (|velocity| * dt) = {value:.3f}; "
                f"set position_scale={scale:.4f} if this is not 1"
                if ratios
                else "not enough consecutive samples"
            ),
            extras={"scale": float(scale)} if ratios else {},
        )

    def _check_tick_period(self, frames: Sequence[GameFrame]) -> CheckResult:
        """Compare the observed frame period against the assumed control dt."""
        times = np.array([f.wall_time for f in frames if f.wall_time > 0], dtype=np.float64)
        if len(times) < 3:
            return CheckResult(
                name="tick_period",
                passed=False,
                value=0.0,
                reference=self.assumed_control_dt,
                tolerance=self.tick_tolerance,
                detail="frames carry no wall-clock timestamps",
            )
        deltas = np.diff(times)
        # Ignore outliers caused by the host stalling; the median is what the loop runs at.
        measured = float(np.median(deltas))
        ratio = measured / self.assumed_control_dt if self.assumed_control_dt > 0 else float("nan")
        passed = abs(ratio - 1.0) <= self.tick_tolerance
        return CheckResult(
            name="tick_period",
            passed=passed,
            value=measured,
            reference=self.assumed_control_dt,
            tolerance=self.tick_tolerance * self.assumed_control_dt,
            detail=(
                f"median frame period {measured * 1000:.2f} ms vs assumed "
                f"{self.assumed_control_dt * 1000:.2f} ms"
            ),
            extras={"period": measured},
        )


def collect_calibration_sample(
    driver: GameDriver,
    *,
    steps: int = 300,
    action: Action | None = None,
) -> list[GameFrame]:
    """Drive the game with a constant input and collect frames.

    The default input is full throttle with no steering, which on a straight stretch of the
    map produces exactly the condition the checks need (forward motion, no slip).
    """
    command = action or Action(steer=0.0, throttle=1.0, brake=0.0)
    frames: list[GameFrame] = []
    driver.reset()
    started = time.monotonic()
    for _ in range(steps):
        frame = driver.step(command)
        frames.append(frame)
    logger.info("collected %d calibration frames in %.1fs", len(frames), time.monotonic() - started)
    return frames


def calibrate_driver(
    driver: GameDriver,
    *,
    steps: int = 300,
    assumed_control_dt: float = 0.05,
    action: Action | None = None,
) -> CalibrationReport:
    """Collect a driving sample from the real game and analyse it.

    ``TelemetryCalibrator.analyse`` fills in every recommendation, so this is nothing but
    "drive, then analyse".
    """
    frames = collect_calibration_sample(driver, steps=steps, action=action)
    return TelemetryCalibrator(assumed_control_dt=assumed_control_dt).analyse(frames)


__all__ = [
    "CalibrationReport",
    "CheckResult",
    "TelemetryCalibrator",
    "calibrate_driver",
    "collect_calibration_sample",
]
