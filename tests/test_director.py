"""Tests for failure-focused adaptive sampling, state recovery, and isolation."""

from __future__ import annotations

from collections import Counter

import pytest

from tmai.config import RunConfig
from tmai.env.multi_track import MultiTrackConfig, MultiTrackEnv
from tmai.env.tm_env import EnvConfig
from tmai.game.simulated import SimulatedGameDriver
from tmai.tracks.library import TrackLibrary
from tmai.tracks.synthetic import oval, s_curve, straight
from tmai.training.director import TrainingDirector, TrainingDirectorSpec


def _library() -> TrackLibrary:
    library = TrackLibrary(split_weights={"train": 1.0, "validation": 0.0, "test": 0.0})
    for track in (straight(length=160.0), oval(), s_curve(length=220.0)):
        library.add(track, split="train")
    return library


def _env(library: TrackLibrary, seed: int = 0) -> MultiTrackEnv:
    return MultiTrackEnv(
        library.sampler("train", seed=seed),
        lambda track: SimulatedGameDriver(track),
        EnvConfig(),
        MultiTrackConfig(random_start_station=False, start_lateral_std=0.0),
        seed=seed,
        split="train",
    )


class TestTrainingDirector:
    def test_weak_tracks_get_bounded_higher_priority(self):
        director = TrainingDirector(
            TrainingDirectorSpec(
                enabled=True,
                ema_alpha=1.0,
                warmup_episodes_per_track=1,
                focus_strength=3.0,
                max_weight=2.25,
                failure_penalty=0.2,
            ),
            ["easy", "hard"],
        )

        director.observe_episode(
            track="easy",
            progress_fraction=1.0,
            finished=True,
            end_reason="finished",
        )
        weights = director.observe_episode(
            track="hard",
            progress_fraction=0.3,
            finished=False,
            end_reason="crash",
        )

        assert weights["easy"] == pytest.approx(1.0)
        assert weights["hard"] == pytest.approx(2.25)
        assert director.summary()["director/failure_rate"] == pytest.approx(0.5)
        assert director.describe()["tracks"]["hard"]["failures"] == 1

    def test_held_out_names_cannot_enter_training_state(self):
        director = TrainingDirector(TrainingDirectorSpec(enabled=True), ["train-only"])
        initial = director.weights()
        assert director.observe_episode(
            track="held-out", progress_fraction=0.0, finished=False, end_reason="crash"
        ) == initial
        assert director.state_dict()["track_names"] == ["train-only"]

    def test_state_roundtrip_preserves_ema_failures_and_priorities(self):
        spec = TrainingDirectorSpec(enabled=True, ema_alpha=0.5, warmup_episodes_per_track=2)
        first = TrainingDirector(spec, ["a", "b"])
        first.observe_episode(
            track="a", progress_fraction=0.8, finished=False, end_reason="time_limit"
        )
        first.observe_episode(
            track="a", progress_fraction=1.0, finished=True, end_reason="finished"
        )
        first.observe_episode(
            track="b", progress_fraction=0.1, finished=False, end_reason="crash"
        )

        restored = TrainingDirector(spec, ["a", "b"])
        restored.load_state_dict(first.state_dict())
        assert restored.describe() == first.describe()
        assert restored.weights() == first.weights()

    def test_state_rejects_track_set_change(self):
        source = TrainingDirector(TrainingDirectorSpec(enabled=True), ["a", "b"])
        target = TrainingDirector(TrainingDirectorSpec(enabled=True), ["a", "c"])
        with pytest.raises(ValueError, match="track set changed"):
            target.load_state_dict(source.state_dict())

    @pytest.mark.parametrize(
        "field,value",
        [
            ("ema_alpha", 0.0),
            ("ema_alpha", float("nan")),
            ("warmup_episodes_per_track", 0),
            ("focus_strength", float("inf")),
            ("max_weight", 0.5),
            ("failure_penalty", 1.1),
        ],
    )
    def test_invalid_director_settings_are_reported(self, field, value):
        assert TrainingDirectorSpec(enabled=True, **{field: value}).validate()

    def test_run_config_roundtrips_director_options(self):
        config = RunConfig()
        config.director = TrainingDirectorSpec(enabled=True, max_weight=2.5)
        restored = RunConfig.from_dict(config.to_dict())
        assert restored.director.enabled is True
        assert restored.director.max_weight == pytest.approx(2.5)


class TestAdaptiveMultiTrackSampling:
    def test_adaptive_sampling_can_be_disabled_after_checkpoint_restore(self):
        library = _library()
        source = _env(library, seed=5)
        restored = _env(library, seed=5)
        try:
            source.set_sampling_weights({"straight": 1.0, "oval": 2.0, "s_curve": 1.0})
            source.reset(seed=5)
            restored.load_state_dict(source.state_dict())
            assert restored.describe()["multi_track"]["sampling_weights"]

            restored.clear_sampling_weights()
            assert restored.describe()["multi_track"]["sampling_weights"] == {}
            assert restored.state_dict()["weighted_order"] == []
        finally:
            source.close()
            restored.close()

    def test_weighted_blocks_keep_every_training_track_in_rotation(self):
        library = _library()
        env = _env(library, seed=13)
        try:
            env.set_sampling_weights({"straight": 1.0, "oval": 4.0, "s_curve": 1.0})
            counts: Counter[str] = Counter()
            for _ in range(60):
                _, info = env.reset()
                counts[info["track"]] += 1
            assert set(counts) == {"straight", "oval", "s_curve"}
            assert counts["oval"] > counts["straight"]
            assert env.describe()["multi_track"]["sampling_weights"]["oval"] == 4.0
        finally:
            env.close()

    def test_sampling_rng_and_pending_schedule_resume_exactly(self):
        library = _library()
        first = _env(library, seed=23)
        second = _env(library, seed=23)
        try:
            priorities = {"straight": 1.0, "oval": 3.0, "s_curve": 1.5}
            first.set_sampling_weights(priorities)
            for _ in range(5):
                first.reset()
            state = first.state_dict()
            expected = []
            for _ in range(12):
                _, info = first.reset()
                expected.append((info["track"], info["start_station"]))

            second.load_state_dict(state)
            actual = []
            for _ in range(12):
                _, info = second.reset()
                actual.append((info["track"], info["start_station"]))
            assert actual == expected
        finally:
            first.close()
            second.close()

    def test_held_out_track_weight_is_rejected(self):
        env = _env(_library())
        try:
            with pytest.raises(ValueError, match="unknown tracks"):
                env.set_sampling_weights({"validation-map": 3.0})
        finally:
            env.close()
