"""Unit tests for the deterministic geometry-grounded benchmark heuristic."""

from __future__ import annotations

import numpy as np
import pytest

from tmai.agents.curvature import CurvaturePilot
from tmai.env.observation import ObservationSpec


def _observation(
    spec: ObservationSpec,
    *,
    speed: float = 0.0,
    lateral: float = 0.0,
    heading: float = 0.0,
    curvature: float = 0.0,
) -> np.ndarray:
    frame = np.zeros(spec.dim, dtype=np.float32)
    offset = 0
    for name, width in spec.feature_layout():
        if name == "speed_forward":
            frame[offset] = speed / spec.scales.speed
        elif name == "lateral_offset":
            frame[offset] = lateral / spec.scales.lateral
        elif name == "heading_error":
            frame[offset] = heading
        elif name.startswith("curvature_at_"):
            frame[offset] = curvature / spec.scales.curvature
        offset += width
    return np.tile(frame, spec.history_length)


class TestCurvaturePilot:
    def test_steers_into_upcoming_positive_right_curvature(self):
        spec = ObservationSpec()
        pilot = CurvaturePilot(spec)
        action = pilot.act(_observation(spec, speed=10.0, curvature=0.04))
        assert action.shape == (3,)
        assert action[0] > 0.0
        assert action[1] >= 0.0
        assert action[2] == 0.0

    def test_lateral_and_heading_errors_correct_back_toward_centerline(self):
        spec = ObservationSpec()
        pilot = CurvaturePilot(spec)
        lateral_action = pilot.act(_observation(spec, speed=20.0, lateral=2.0))
        heading_action = pilot.act(_observation(spec, speed=20.0, heading=0.3))
        assert lateral_action[0] < 0.0
        assert heading_action[0] < 0.0

    def test_tight_curvature_reduces_target_speed_and_brakes(self):
        spec = ObservationSpec()
        pilot = CurvaturePilot(spec)
        action = pilot.act(_observation(spec, speed=40.0, curvature=0.25))
        assert action[1] == 0.0
        assert action[2] > 0.0
        assert action[2] <= 0.9
        assert np.all(np.isfinite(action))

    def test_uses_latest_frame_in_a_temporal_stack(self):
        spec = ObservationSpec(history_length=3)
        pilot = CurvaturePilot(spec)
        observation = _observation(spec, speed=20.0)
        older_frame = _observation(spec, speed=20.0, curvature=0.05)[-spec.dim :]
        observation[: spec.dim] = older_frame
        action = pilot.act(observation)
        assert action[0] == pytest.approx(0.0)

    def test_rejects_malformed_or_nonfinite_observation(self):
        pilot = CurvaturePilot(ObservationSpec())
        with pytest.raises(ValueError, match="expected"):
            pilot.act(np.zeros(pilot.observation_dim - 1, dtype=np.float32))
        observation = np.zeros(pilot.observation_dim, dtype=np.float32)
        observation[0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            pilot.act(observation)

    def test_baseline_is_explicitly_not_real_game_validated(self):
        description = CurvaturePilot(ObservationSpec()).describe()
        assert description["kind"] == "heuristic_baseline"
        assert description["real_game_validated"] is False
