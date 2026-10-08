"""Tests for temporal observations (frame stacking)."""

from __future__ import annotations

import numpy as np
import pytest

from tmai.env.observation import ObservationSpec, ObservationStacker
from tmai.env.tm_env import EnvConfig, TrackmaniaEnv
from tmai.game.simulated import SimulatedGameDriver
from tmai.tracks.synthetic import straight


def _env(track, history_length: int):
    config = EnvConfig()
    config.observation.history_length = history_length
    driver = SimulatedGameDriver(track)
    driver.open()
    return TrackmaniaEnv(driver, track, config)


class TestObservationStacker:
    def test_stacks_oldest_first(self):
        stacker = ObservationStacker(dim=2, length=3)
        out = stacker.reset(np.array([1.0, 1.0]))
        assert out.shape == (6,)
        assert np.allclose(out, [1, 1, 1, 1, 1, 1])
        out = stacker.push(np.array([2.0, 2.0]))
        assert np.allclose(out, [1, 1, 1, 1, 2, 2])
        out = stacker.push(np.array([3.0, 3.0]))
        assert np.allclose(out, [1, 1, 2, 2, 3, 3])
        # The window slides: the oldest frame drops out.
        out = stacker.push(np.array([4.0, 4.0]))
        assert np.allclose(out, [2, 2, 3, 3, 4, 4])

    def test_reset_refills_with_first_frame(self):
        stacker = ObservationStacker(dim=1, length=2)
        stacker.push(np.array([9.0]))
        out = stacker.reset(np.array([5.0]))
        assert np.allclose(out, [5.0, 5.0])

    def test_rejects_wrong_dim_and_length(self):
        with pytest.raises(ValueError, match="dim"):
            ObservationStacker(dim=0, length=1)
        with pytest.raises(ValueError, match="length"):
            ObservationStacker(dim=1, length=0)
        stacker = ObservationStacker(dim=2, length=1)
        with pytest.raises(ValueError, match="features"):
            stacker.push(np.array([1.0, 2.0, 3.0]))


class TestSpec:
    def test_default_is_single_frame(self):
        spec = ObservationSpec()
        assert spec.history_length == 1
        assert spec.stacked_dim == spec.dim
        names = spec.names()
        assert "speed_forward" in names
        assert not any(n.endswith("[0]") for n in names)

    def test_stacked_names_carry_frame_index(self):
        spec = ObservationSpec(history_length=3)
        assert spec.stacked_dim == 3 * spec.dim
        names = spec.names()
        assert "speed_forward[0]" in names  # oldest frame
        assert "speed_forward[2]" in names  # newest frame
        assert len(names) == spec.stacked_dim

    def test_validation(self):
        assert ObservationSpec().validate() == []
        assert any("history_length" in p for p in ObservationSpec(history_length=0).validate())


class TestEnvIntegration:
    def test_observation_dim_multiplies(self, straight_track):
        env = _env(straight_track, history_length=4)
        try:
            single = EnvConfig().observation.dim
            assert env.observation_dim == 4 * single
            assert env.observation_space.shape == (4 * single,)
            assert len(env.observation_names) == 4 * single
        finally:
            env.close()

    def test_history_gives_the_policy_a_temporal_window(self, straight_track):
        env = _env(straight_track, history_length=3)
        try:
            obs, _ = env.reset(seed=0)
            single = env.observation_dim // 3
            # At reset the stack is the first frame repeated: all three windows equal.
            assert np.allclose(obs[:single], obs[single:2 * single])
            assert np.allclose(obs[single:2 * single], obs[2 * single:])
            # After driving, the windows differ: the car accelerated.
            action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            for _ in range(20):
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            assert not np.allclose(obs[:single], obs[2 * single:])
        finally:
            env.close()

    def test_reset_clears_the_history(self, straight_track):
        env = _env(straight_track, history_length=2)
        try:
            obs, _ = env.reset(seed=0)
            action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            for _ in range(10):
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            obs2, _ = env.reset(seed=0)
            single = env.observation_dim // 2
            # After a fresh reset the two windows are the same first frame again.
            assert np.allclose(obs2[:single], obs2[single:])
        finally:
            env.close()

    def test_translation_invariance_survives_stacking(self):
        """The generalisation invariant must hold for stacked observations too."""
        from tmai.tracks.centerline import CenterlineTrack

        base = straight(length=120.0)
        shifted = CenterlineTrack(
            base.points + np.array([750.0, 0.0, -300.0]),
            name=base.name,
            uid=base.uid,
            corridor_half_width=base.corridor_half_width,
            closed=base.closed,
        )
        env_a = _env(base, history_length=2)
        env_b = _env(shifted, history_length=2)
        try:
            obs_a, _ = env_a.reset(seed=0)
            obs_b, _ = env_b.reset(seed=0)
            assert np.array_equal(obs_a, obs_b)
        finally:
            env_a.close()
            env_b.close()

    def test_stacked_checkpoint_dim_guard(self, straight_track):
        """A stacked checkpoint must not load into an unstacked learner."""
        from tmai.agents.sac import SACLearner

        env = _env(straight_track, history_length=2)
        try:
            learner = SACLearner(env.observation_dim, env.action_space.shape[0], seed=0)
            state = learner.state_dict()
            other = SACLearner(env.observation_dim // 2, env.action_space.shape[0], seed=0)
            with pytest.raises(ValueError, match="observation_dim"):
                other.load_state_dict(state)
        finally:
            env.close()

    def test_config_validation_rejects_zero_history(self):
        from tmai.config import RunConfig

        config = RunConfig()
        config.env.observation.history_length = 0
        assert any("history_length" in p for p in config.validate())

    def test_to_dict_records_history(self):
        spec = ObservationSpec(history_length=2)
        data = spec.to_dict()
        assert data["history_length"] == 2
        assert data["stacked_dim"] == 2 * data["dim"]
