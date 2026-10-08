"""Tests for curriculum learning: stage resolution, track reveal, episode caps, wiring."""

from __future__ import annotations

import json

import pytest

from tmai.config import RunConfig
from tmai.env.multi_track import MultiTrackEnv
from tmai.env.termination import TerminationConfig
from tmai.env.tm_env import TrackmaniaEnv
from tmai.game.simulated import SimulatedGameDriver
from tmai.tracks.library import TrackLibrary
from tmai.tracks.synthetic import figure_eight, oval, s_curve, straight
from tmai.training.curriculum import (
    Curriculum,
    CurriculumSpec,
    CurriculumStage,
    difficulty_score,
)
from tmai.training.trainer import train_from_config


def _library() -> TrackLibrary:
    library = TrackLibrary(split_weights={"train": 1.0, "validation": 0.0, "test": 0.0})
    for track in (straight(length=200.0), oval(), s_curve(length=240.0), figure_eight()):
        library.add(track)
    return library


def _spec(stages) -> CurriculumSpec:
    return CurriculumSpec(enabled=True, stages=stages)


class TestSpecValidation:
    def test_disabled_needs_no_stages(self):
        assert CurriculumSpec(enabled=False).validate() == []

    def test_enabled_without_stages_is_a_problem(self):
        assert any("no stages" in p for p in CurriculumSpec(enabled=True).validate())

    def test_stages_must_increase(self):
        spec = _spec([CurriculumStage(until_step=100), CurriculumStage(until_step=50)])
        assert any("greater than the previous" in p for p in spec.validate())

    def test_stage_fields_validated(self):
        spec = _spec([CurriculumStage(until_step=0)])
        assert any("until_step" in p for p in spec.validate())
        spec = _spec([CurriculumStage(until_step=10, reveal_tracks=0)])
        assert any("reveal_tracks" in p for p in spec.validate())
        spec = _spec([CurriculumStage(until_step=10, episode_length_fraction=0.0)])
        assert any("episode_length_fraction" in p for p in spec.validate())

    def test_config_validation_includes_curriculum(self):
        config = RunConfig()
        config.curriculum = CurriculumSpec(enabled=True)
        assert any("curriculum" in p for p in config.validate())


class TestDifficultyOrder:
    def test_straight_is_easiest(self):
        library = _library()
        curriculum = Curriculum(_spec([CurriculumStage(until_step=100)]), library.entries)
        assert curriculum.difficulty_order()[0] == "straight"

    def test_score_is_curvature_dominated(self):
        """The score is mean curvature plus a small corner-density tie-break."""
        library = _library()
        by_name = {e.track.name: e for e in library.entries}
        for entry in library.entries:
            stats = entry.stats
            expected = stats.curvature_mean + 0.01 * stats.corner_count / stats.length
            assert difficulty_score(entry) == pytest.approx(expected)
        # A straight track has no curvature at all, so it is always easiest.
        assert difficulty_score(by_name["straight"]) == pytest.approx(0.0)

    def test_order_is_deterministic(self):
        library = _library()
        a = Curriculum(_spec([CurriculumStage(until_step=100)]), library.entries)
        b = Curriculum(_spec([CurriculumStage(until_step=100)]), library.entries)
        assert a.difficulty_order() == b.difficulty_order()


class TestStageResolution:
    def test_stages_apply_in_order(self):
        library = _library()
        curriculum = Curriculum(
            _spec([CurriculumStage(until_step=100, reveal_tracks=1),
                   CurriculumStage(until_step=200, reveal_tracks=2)]),
            library.entries,
        )
        assert curriculum.stage_index_at(1) == 0
        assert curriculum.stage_index_at(100) == 0
        assert curriculum.stage_index_at(101) == 1
        assert curriculum.stage_index_at(200) == 1

    def test_beyond_last_stage_is_the_final_stage(self):
        library = _library()
        curriculum = Curriculum(_spec([CurriculumStage(until_step=100)]), library.entries)
        assert curriculum.stage_index_at(10**9) == curriculum.num_stages - 1
        final = curriculum.stage_at(10**9)
        assert final.reveal_tracks is None
        assert final.episode_length_fraction == 1.0
        assert curriculum.active_track_names(10**9) is None

    def test_reveal_tracks_returns_easiest_first(self):
        library = _library()
        curriculum = Curriculum(
            _spec([CurriculumStage(until_step=100, reveal_tracks=2)]), library.entries
        )
        names = curriculum.active_track_names(50)
        assert names == curriculum.difficulty_order()[:2]

    def test_reveal_tracks_clamped_to_library_size(self):
        library = _library()
        curriculum = Curriculum(
            _spec([CurriculumStage(until_step=100, reveal_tracks=99)]), library.entries
        )
        assert len(curriculum.active_track_names(50)) == len(library)

    def test_episode_max_steps_fraction(self):
        library = _library()
        curriculum = Curriculum(
            _spec([CurriculumStage(until_step=100, episode_length_fraction=0.5)]),
            library.entries,
        )
        assert curriculum.episode_max_steps(50, base_max_steps=200) == 100
        # A fraction of 1.0 means no cap at all.
        assert curriculum.episode_max_steps(10**9, base_max_steps=200) is None

    def test_describe_records_the_plan(self):
        library = _library()
        curriculum = Curriculum(
            _spec([CurriculumStage(until_step=100, reveal_tracks=1)]), library.entries
        )
        description = curriculum.describe()
        assert description["num_stages"] == 2
        assert description["difficulty_order"][0] == "straight"
        assert description["num_train_tracks"] == 4


class TestEnvIntegration:
    def _env(self, library, seed=0):
        sampler = library.sampler("train", seed=seed)
        driver_factory = lambda track: SimulatedGameDriver(track)  # noqa: E731
        env = MultiTrackEnv(
            sampler,
            driver_factory,
            split="train",
            track_identities={e.track.name: e.identity for e in library.entries},
        )
        return env

    def test_curriculum_limits_the_tracks_sampled(self):
        library = _library()
        env = self._env(library)
        try:
            curriculum = Curriculum(
                _spec([CurriculumStage(until_step=100, reveal_tracks=1)]), library.entries
            )
            env.attach_curriculum(curriculum)
            env.set_curriculum_step(50)
            seen = set()
            for _ in range(6):
                env.reset(seed=0)
                seen.add(env.episode_context.track_name)
            assert seen == {curriculum.difficulty_order()[0]}
        finally:
            env.close()

    def test_curriculum_widens_as_steps_advance(self):
        library = _library()
        env = self._env(library)
        try:
            curriculum = Curriculum(
                _spec([CurriculumStage(until_step=100, reveal_tracks=1),
                       CurriculumStage(until_step=200, reveal_tracks=2)]),
                library.entries,
            )
            env.attach_curriculum(curriculum)
            env.set_curriculum_step(50)
            assert env.curriculum_stage == 0
            first = set()
            for _ in range(4):
                env.reset(seed=0)
                first.add(env.episode_context.track_name)
            assert first == {curriculum.difficulty_order()[0]}
            env.set_curriculum_step(150)
            assert env.curriculum_stage == 1
            second = set()
            for _ in range(8):
                env.reset(seed=0)
                second.add(env.episode_context.track_name)
            assert second == set(curriculum.difficulty_order()[:2])
        finally:
            env.close()

    def test_curriculum_caps_episode_length(self):
        library = _library()
        env = self._env(library)
        try:
            env.config.termination.max_steps = 200
            curriculum = Curriculum(
                _spec([CurriculumStage(until_step=100, episode_length_fraction=0.25)]),
                library.entries,
            )
            env.attach_curriculum(curriculum)
            env.set_curriculum_step(50)
            env.reset(seed=0)
            inner = env._inner_env
            assert inner._termination.max_steps_override == 50
            env.set_curriculum_step(10**9)
            env.reset(seed=0)
            assert inner._termination.max_steps_override is None
        finally:
            env.close()

    def test_no_curriculum_samples_everything(self):
        library = _library()
        env = self._env(library)
        try:
            assert env.curriculum_stage is None
            seen = set()
            for _ in range(12):
                env.reset(seed=0)
                seen.add(env.episode_context.track_name)
            assert len(seen) > 1
        finally:
            env.close()

    def test_pinned_track_overrides_curriculum(self):
        library = _library()
        env = self._env(library)
        try:
            curriculum = Curriculum(
                _spec([CurriculumStage(until_step=100, reveal_tracks=1)]), library.entries
            )
            env.attach_curriculum(curriculum)
            env.set_curriculum_step(50)
            hardest = curriculum.difficulty_order()[-1]
            env.select_track(hardest)
            env.reset(seed=0)
            assert env.episode_context.track_name == hardest
        finally:
            env.close()

    def test_single_track_env_applies_step_cap(self):
        track = straight(length=200.0)
        driver = SimulatedGameDriver(track)
        driver.open()
        env = TrackmaniaEnv(driver, track)
        try:
            env.config.termination.max_steps = 200
            library = TrackLibrary()
            entry = library.add(track)
            curriculum = Curriculum(
                _spec([CurriculumStage(until_step=100, episode_length_fraction=0.5)]),
                [entry],
            )
            env.attach_curriculum(curriculum)
            env.set_curriculum_step(50)
            env.reset(seed=0)
            assert env._termination.max_steps_override == 100
        finally:
            env.close()


class TestTrainerIntegration:
    def _config(self, tmp_path, **curriculum_kwargs) -> RunConfig:
        config = RunConfig()
        config.driver.kind = "simulated"
        config.driver.allow_simulated = True
        config.track.synthetic_suite = [
            {"name": "straight", "kwargs": {"length": 120.0}},
            {"name": "oval"},
            {"name": "s_curve", "kwargs": {"length": 200.0}},
        ]
        config.track.split_weights = {"train": 1.0, "validation": 0.0, "test": 0.0}
        config.env.termination = TerminationConfig(max_steps=30)
        config.sac.network.hidden_sizes = (16, 16)
        config.replay.capacity = 2_000
        config.train.total_steps = 150
        config.train.warmup_steps = 20
        config.train.batch_size = 16
        config.train.log_interval = 25
        config.train.eval_interval = 0
        config.train.eval_episodes = 0
        config.train.checkpoint_interval = 0
        config.train.output_dir = str(tmp_path / "runs")
        config.train.run_name = "curriculum-test"
        config.curriculum = CurriculumSpec(
            enabled=True,
            stages=[
                CurriculumStage(until_step=60, reveal_tracks=1, episode_length_fraction=0.5),
                CurriculumStage(until_step=120, reveal_tracks=2),
            ],
        )
        return config

    def test_training_runs_with_curriculum_and_logs_stages(self, tmp_path):
        config = self._config(tmp_path)
        result = train_from_config(config)
        assert not result.failure

        events = [
            json.loads(line)
            for line in (result.run_dir / "events.jsonl").read_text().splitlines()
        ]
        stage_events = [e for e in events if e["event"] == "curriculum_stage"]
        assert stage_events, "curriculum stage transitions must be logged"
        stages = [e["stage"] for e in stage_events]
        assert stages == sorted(stages)
        assert 0 in stages and result.run_dir is not None

        metrics = [
            json.loads(line)
            for line in (result.run_dir / "metrics.jsonl").read_text().splitlines()
        ]
        assert any("curriculum/stage" in m for m in metrics)

        manifest = json.loads((result.run_dir / "manifest.json").read_text())
        assert manifest["curriculum"]["enabled"] is True
        assert manifest["curriculum"]["difficulty_order"][0] == "straight"

    def test_episodes_respect_the_stage_cap(self, tmp_path):
        config = self._config(tmp_path)
        # Force every episode to be cut by the cap: fraction 0.5 of max_steps=30 -> 15.
        config.curriculum.stages = [
            CurriculumStage(until_step=10**9, reveal_tracks=1, episode_length_fraction=0.5)
        ]
        result = train_from_config(config)
        events = [
            json.loads(line)
            for line in (result.run_dir / "events.jsonl").read_text().splitlines()
        ]
        episodes = [e for e in events if e["event"] == "episode_end"]
        assert episodes
        assert all(e["steps"] <= 15 for e in episodes)

    def test_disabled_curriculum_changes_nothing(self, tmp_path):
        config = self._config(tmp_path)
        config.curriculum = CurriculumSpec(enabled=False)
        result = train_from_config(config)
        events = [
            json.loads(line)
            for line in (result.run_dir / "events.jsonl").read_text().splitlines()
        ]
        assert not [e for e in events if e["event"] == "curriculum_stage"]
        manifest = json.loads((result.run_dir / "manifest.json").read_text())
        assert "curriculum" not in manifest or not manifest["curriculum"]["enabled"]
