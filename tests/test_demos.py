"""Demonstration recording: the data layer for behaviour cloning."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tmai.env.tm_env import EnvConfig, TrackmaniaEnv
from tmai.game.protocol import Action
from tmai.game.simulated import SimulatedDriverConfig, SimulatedGameDriver
from tmai.tracks.synthetic import build_synthetic
from tmai.training.demos import (
    Demonstration,
    DemonstrationRecorder,
    load_demonstrations,
    record_demonstration,
)

from .conftest import make_frame


def _obs(dim: int, value: float) -> np.ndarray:
    return np.full(dim, value, dtype=np.float32)


class TestDemonstration:
    def test_round_trip(self, tmp_path):
        demo = Demonstration(
            observations=np.arange(12, dtype=np.float32).reshape(4, 3),
            actions=np.tile([0.1, 0.9, 0.0], (4, 1)).astype(np.float32),
            positions=np.tile([1.0, 2.0, 3.0], (4, 1)),
            speeds=np.arange(4, dtype=np.float64),
            rewards=np.arange(4, dtype=np.float64) * 0.5,
            race_times=np.arange(4, dtype=np.float64) * 0.05,
            metadata={"track": "oval", "finished": True},
        )
        path = demo.save(tmp_path / "lap.jsonl")
        loaded = Demonstration.load(path)
        assert len(loaded) == 4
        assert loaded.observation_dim == 3
        assert loaded.action_dim == 3
        np.testing.assert_allclose(loaded.observations, demo.observations)
        np.testing.assert_allclose(loaded.actions, demo.actions)
        np.testing.assert_allclose(loaded.positions, demo.positions)
        np.testing.assert_allclose(loaded.speeds, demo.speeds)
        np.testing.assert_allclose(loaded.rewards, demo.rewards)
        np.testing.assert_allclose(loaded.race_times, demo.race_times)
        assert loaded.metadata["track"] == "oval"

    def test_header_line_carries_metadata(self, tmp_path):
        demo = Demonstration(
            observations=_obs(2, 1.0)[None, :],
            actions=np.zeros((1, 3), dtype=np.float32),
            metadata={"track": "straight"},
        )
        path = demo.save(tmp_path / "lap.jsonl")
        header = json.loads(path.read_text().splitlines()[0])
        assert header["header"]["track"] == "straight"
        assert header["observation_dim"] == 2
        assert header["action_dim"] == 3

    def test_malformed_trailing_line_is_skipped(self, tmp_path):
        demo = Demonstration(
            observations=_obs(2, 1.0)[None, :],
            actions=np.zeros((1, 3), dtype=np.float32),
        )
        path = demo.save(tmp_path / "lap.jsonl")
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"observation": [1.0, 2.0], "act')  # torn write
        loaded = Demonstration.load(path)
        assert len(loaded) == 1

    def test_dimension_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            Demonstration(
                observations=np.zeros((3, 2), dtype=np.float32),
                actions=np.zeros((4, 3), dtype=np.float32),
            )
        with pytest.raises(ValueError, match="observations must be"):
            Demonstration(
                observations=np.zeros(3, dtype=np.float32),
                actions=np.zeros((3, 3), dtype=np.float32),
            )

    def test_empty_file_rejected(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="no transitions"):
            Demonstration.load(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Demonstration.load(tmp_path / "nope.jsonl")


class TestLoadDemonstrations:
    def test_concatenates_files(self, tmp_path):
        for i in range(2):
            Demonstration(
                observations=_obs(2, float(i))[None, :],
                actions=np.zeros((1, 3), dtype=np.float32),
            ).save(tmp_path / f"lap{i}.jsonl")
        demos = load_demonstrations([tmp_path / "lap0.jsonl", tmp_path / "lap1.jsonl"])
        assert len(demos) == 2
        assert demos.observation_dim == 2
        assert demos.metadata["num_files"] == 2

    def test_dimension_mismatch_rejected(self, tmp_path):
        Demonstration(
            observations=_obs(2, 0.0)[None, :], actions=np.zeros((1, 3), dtype=np.float32)
        ).save(tmp_path / "a.jsonl")
        Demonstration(
            observations=_obs(5, 0.0)[None, :], actions=np.zeros((1, 3), dtype=np.float32)
        ).save(tmp_path / "b.jsonl")
        with pytest.raises(ValueError, match="observation_dim 5, expected 2"):
            load_demonstrations(
                [tmp_path / "a.jsonl", tmp_path / "b.jsonl"], observation_dim=2
            )

    def test_no_files_rejected(self):
        with pytest.raises(ValueError, match="no demonstration files"):
            load_demonstrations([])


class TestDemonstrationRecorder:
    def test_records_game_reported_inputs_not_ai_action(self):
        """The whole point: a human's inputs are what the game reports, not what we sent."""
        import dataclasses

        recorder = DemonstrationRecorder()
        frame = make_frame(position=(1.0, 0.0, 2.0), speed_forward=12.0)
        frame = dataclasses.replace(
            frame,
            vehicle=dataclasses.replace(
                frame.vehicle, input_steer=-0.4, input_gas=0.8, input_brake=0.0
            ),
        )
        recorder.record_step(_obs(3, 0.5), frame, reward=0.25)
        demo = recorder.build(metadata={"track": "oval"})
        assert len(demo) == 1
        np.testing.assert_allclose(demo.actions[0], [-0.4, 0.8, 0.0])
        np.testing.assert_allclose(demo.positions[0], [1.0, 0.0, 2.0])
        assert demo.speeds[0] == pytest.approx(12.0)
        assert demo.rewards[0] == pytest.approx(0.25)
        assert demo.race_times[0] == pytest.approx(0.0)

    def test_build_without_steps_rejected(self):
        with pytest.raises(ValueError, match="nothing recorded"):
            DemonstrationRecorder().build()


class TestRecordDemonstration:
    def _env(self):
        track = build_synthetic("straight", length=120.0)
        driver = SimulatedGameDriver(
            track, SimulatedDriverConfig(wall_margin=1e9, fall_margin=1e9)
        )
        driver.open()
        return TrackmaniaEnv(driver, track, EnvConfig()), track

    def test_records_a_lap(self, tmp_path):
        env, track = self._env()
        try:
            demo = record_demonstration(
                env,
                out_path=tmp_path / "lap.jsonl",
                max_steps=60,
                action_provider=lambda frame: Action(0.0, 0.7, 0.0),
                metadata={"track": track.name},
            )
        finally:
            env.close()
        assert len(demo) > 10
        assert demo.metadata["track"] == "straight"
        # The simulated driver echoes the injected action into the input read-back, so the
        # recorded actions are exactly what was driven. (The first sample is the reset
        # frame, recorded before any action was applied, so it reports neutral inputs.)
        np.testing.assert_allclose(demo.actions[1:, 1], 0.7, atol=1e-6)
        assert demo.positions.shape == (len(demo), 3)
        assert demo.race_times[-1] > demo.race_times[0]
        # Saved and loadable.
        loaded = Demonstration.load(tmp_path / "lap.jsonl")
        assert len(loaded) == len(demo)

    def test_stops_at_episode_end(self, tmp_path):
        env, track = self._env()
        try:
            demo = record_demonstration(
                env,
                out_path=tmp_path / "lap.jsonl",
                max_steps=10_000,
                action_provider=lambda frame: Action(0.0, 1.0, 0.0),
            )
        finally:
            env.close()
        # A 120 m straight at full throttle finishes well inside the step cap.
        assert demo.metadata["finished"] is True
        assert len(demo) < 10_000

    def test_should_stop_ends_recording(self, tmp_path):
        env, _ = self._env()
        calls = {"n": 0}

        def stop_after_three():
            calls["n"] += 1
            return calls["n"] > 3

        try:
            demo = record_demonstration(
                env,
                out_path=tmp_path / "lap.jsonl",
                max_steps=10_000,
                action_provider=lambda frame: Action(0.0, 0.5, 0.0),
                should_stop=stop_after_three,
            )
        finally:
            env.close()
        assert len(demo) <= 6  # initial record + up to 3 steps + the stopping step
