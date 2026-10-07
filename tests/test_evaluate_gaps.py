"""Tests for evaluation paths that coverage showed were never executed.

Three gaps, all found by measuring coverage rather than by reading code:

* ``EvaluationReport.generalization_gap`` — the headline overfitting metric.
* ``evaluate_tracks(..., tracks=None)`` — the sampler-driven mid-training path.
* ``run_episode``'s off-track accounting — the crash proxy.
"""

from __future__ import annotations

import numpy as np

from tmai.agents.base import Batch
from tmai.env.multi_track import MultiTrackConfig, MultiTrackEnv
from tmai.env.tm_env import EnvConfig
from tmai.game.simulated import SimulatedGameDriver
from tmai.tracks.library import TrackLibrary
from tmai.tracks.synthetic import build_synthetic
from tmai.training.evaluate import (
    EpisodeResult,
    EvaluationReport,
    TrackResult,
    evaluate_tracks,
    run_episode,
)


def _track_result(name: str, split: str, *fractions: float) -> TrackResult:
    """A TrackResult whose episodes all report the given progress fractions."""
    result = TrackResult(track=name, split=split)
    for fraction in fractions:
        result.episodes.append(EpisodeResult(progress_fraction=fraction))
    return result


# -- generalization_gap --------------------------------------------------------------


class TestGeneralizationGap:
    def test_positive_gap_means_overfitting(self):
        report = EvaluationReport(
            tracks=[
                _track_result("a", "train", 0.9),
                _track_result("b", "validation", 0.4),
            ]
        )
        # Train 0.9 versus held-out 0.4: the policy has memorised the training map.
        assert report.generalization_gap == 0.5

    def test_negative_gap_means_held_out_is_better(self):
        report = EvaluationReport(
            tracks=[
                _track_result("a", "train", 0.3),
                _track_result("b", "validation", 0.7),
            ]
        )
        # A negative gap is legitimate (the unseen maps can be easier) and must be reported
        # as negative rather than clamped to zero.
        assert report.generalization_gap == -0.4

    def test_none_when_only_train_is_present(self):
        report = EvaluationReport(tracks=[_track_result("a", "train", 0.9)])
        # Returning 0.0 here would silently claim "no overfitting", which is a lie.
        assert report.generalization_gap is None

    def test_none_when_only_held_out_is_present(self):
        report = EvaluationReport(tracks=[_track_result("b", "validation", 0.4)])
        assert report.generalization_gap is None

    def test_none_for_an_empty_report(self):
        assert EvaluationReport().generalization_gap is None

    def test_averages_validation_and_test_together(self):
        report = EvaluationReport(
            tracks=[
                _track_result("a", "train", 1.0),
                _track_result("b", "validation", 0.4),
                _track_result("c", "test", 0.2),
            ]
        )
        # Held-out side is the mean of validation and test: 1.0 - 0.3 = 0.7.
        assert report.generalization_gap == 0.7

    def test_averages_multiple_tracks_per_split(self):
        report = EvaluationReport(
            tracks=[
                _track_result("a", "train", 0.8),
                _track_result("b", "train", 0.6),
                _track_result("c", "validation", 0.2),
                _track_result("d", "validation", 0.4),
            ]
        )
        # Train mean 0.7, held-out mean 0.3.
        assert report.generalization_gap == 0.4

    def test_unlabelled_splits_do_not_produce_a_gap(self):
        # from_episodes() leaves splits empty, so no train/held-out distinction exists.
        report = EvaluationReport.from_episodes(
            [EpisodeResult(progress_fraction=0.5, track="a")]
        )
        assert report.generalization_gap is None

    def test_gap_appears_in_metrics_and_dict_only_when_known(self):
        report = EvaluationReport(tracks=[_track_result("a", "train", 0.9)])
        assert "eval/generalization_gap" not in report.metrics()
        assert report.as_dict()["generalization_gap"] is None

        both = EvaluationReport(
            tracks=[
                _track_result("a", "train", 0.9),
                _track_result("b", "validation", 0.4),
            ]
        )
        assert both.metrics()["eval/generalization_gap"] == 0.5
        assert both.as_dict()["generalization_gap"] == 0.5

    def test_summary_omits_the_gap_when_unknown(self):
        report = EvaluationReport(tracks=[_track_result("a", "train", 0.9)])
        assert "gap" not in report.summary()

        both = EvaluationReport(
            tracks=[
                _track_result("a", "train", 0.9),
                _track_result("b", "validation", 0.4),
            ]
        )
        assert "gap +0.500" in both.summary()


# -- evaluate_tracks sampling mode ---------------------------------------------------


class _StubLearner:
    """A learner that returns one fixed action, so episodes are predictable."""

    observation_dim = 20
    action_dim = 3

    def __init__(self, action=(0.0, 1.0, 0.0)):
        self._action = np.asarray(action, dtype=np.float32)

    def act(self, observation, deterministic=True):
        return self._action

    def update(self, batch: Batch) -> dict[str, float]:
        return {}

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None

    def describe(self) -> dict:
        return {"kind": "stub"}


def _multi_env(split: str = "train", seed: int = 0) -> MultiTrackEnv:
    library = TrackLibrary()
    for name in ("straight", "oval"):
        library.add(build_synthetic(name), split=split)
    return MultiTrackEnv(
        library.sampler(split, seed=seed),
        lambda track: SimulatedGameDriver(track),
        EnvConfig(),
        MultiTrackConfig(),
        seed=seed,
        split=split,
    )


class TestEvaluateTracksSamplingMode:
    def test_runs_at_least_one_episode(self):
        env = _multi_env()
        try:
            report = evaluate_tracks(env, _StubLearner(), episodes_per_track=1, max_steps=30)
        finally:
            env.close()
        assert report.num_episodes >= 1

    def test_episodes_are_bucketed_by_the_track_they_ran_on(self):
        env = _multi_env()
        try:
            # The sampler hands out a permutation block, so two distinct tracks appear.
            report = evaluate_tracks(env, _StubLearner(), episodes_per_track=4, max_steps=30)
        finally:
            env.close()
        names = [t.track for t in report.tracks]
        assert len(names) == len(set(names)), "each track must appear exactly once"
        assert set(names) == {"straight", "oval"}

    def test_sampled_tracks_are_labelled_sampled_not_a_real_split(self):
        env = _multi_env()
        try:
            report = evaluate_tracks(env, _StubLearner(), episodes_per_track=2, max_steps=30)
        finally:
            env.close()
        # "sampled" keeps these out of the train/validation/test generalization_gap maths,
        # which would otherwise treat mid-training samples as a held-out set.
        assert all(t.split == "sampled" for t in report.tracks)
        assert report.generalization_gap is None

    def test_total_episodes_equals_requested_count(self):
        env = _multi_env()
        try:
            report = evaluate_tracks(env, _StubLearner(), episodes_per_track=6, max_steps=30)
        finally:
            env.close()
        assert report.num_episodes == 6

    def test_zero_episodes_still_runs_one(self):
        env = _multi_env()
        try:
            report = evaluate_tracks(env, _StubLearner(), episodes_per_track=0, max_steps=30)
        finally:
            env.close()
        assert report.num_episodes == 1

    def test_label_and_step_are_carried_through(self):
        env = _multi_env()
        try:
            report = evaluate_tracks(
                env, _StubLearner(), episodes_per_track=1, max_steps=30,
                label="mid-training", step=1234,
            )
        finally:
            env.close()
        assert report.label == "mid-training"
        assert report.step == 1234

    def test_sampling_mode_is_reproducible_for_a_fixed_seed(self):
        tracks = []
        for _ in range(2):
            env = _multi_env(seed=7)
            try:
                report = evaluate_tracks(
                    env, _StubLearner(), episodes_per_track=4, max_steps=30, seed=7
                )
            finally:
                env.close()
            tracks.append([(t.track, t.num_episodes) for t in report.tracks])
        assert tracks[0] == tracks[1]

    def test_explicit_track_list_still_works(self):
        """The pinned path and the sampled path must not have drifted apart."""
        env = _multi_env()
        try:
            report = evaluate_tracks(
                env,
                _StubLearner(),
                tracks=[("oval", "train"), ("straight", "validation")],
                episodes_per_track=1,
                max_steps=30,
            )
        finally:
            env.close()
        assert [t.track for t in report.tracks] == ["oval", "straight"]
        assert [t.split for t in report.tracks] == ["train", "validation"]
        # Labels are honoured here, so the gap is computable.
        assert report.generalization_gap is not None


# -- off-track accounting ------------------------------------------------------------


class TestOffTrackAccounting:
    def _env(self):
        from tmai.env.termination import TerminationConfig
        from tmai.env.tm_env import TrackmaniaEnv

        track = build_synthetic("straight", length=200.0)
        driver = SimulatedGameDriver(track)
        driver.open()
        config = EnvConfig()
        # Let the car leave the corridor without ending the episode immediately, so the
        # excursion accounting has something to count.
        config.termination = TerminationConfig(
            max_steps=400, off_track_limit=10**6, stall_limit=10**6
        )
        return TrackmaniaEnv(driver, track, config)

    def test_off_track_driving_records_excursions_and_metres(self):
        env = self._env()
        try:
            # Full steering lock with throttle drives the car out of the corridor.
            learner = _StubLearner(action=(1.0, 1.0, 0.0))
            episode = run_episode(env, learner, max_steps=400, deterministic=True)
        finally:
            env.close()
        assert episode.off_track_events >= 1
        assert episode.off_track_metres > 0.0

    def test_a_clean_run_records_no_off_track(self):
        env = self._env()
        try:
            # Straight ahead down a straight track stays inside the corridor.
            learner = _StubLearner(action=(0.0, 1.0, 0.0))
            episode = run_episode(env, learner, max_steps=120, deterministic=True)
        finally:
            env.close()
        assert episode.off_track_events == 0
        assert episode.off_track_metres == 0.0

    def test_consecutive_off_track_steps_count_as_one_event(self):
        """Excursions, not steps: otherwise the metric just measures episode length."""
        env = self._env()
        try:
            learner = _StubLearner(action=(1.0, 1.0, 0.0))
            episode = run_episode(env, learner, max_steps=400, deterministic=True)
        finally:
            env.close()
        # Once out, the car stays out for many steps, so events must be far fewer than the
        # number of steps it spent off track.
        assert episode.steps > 20
        assert episode.off_track_events <= episode.steps // 2

    def test_off_track_fields_survive_serialisation(self):
        episode = EpisodeResult(off_track_events=3, off_track_metres=12.5)
        payload = episode.as_dict()
        assert payload["off_track_events"] == 3
        assert payload["off_track_metres"] == 12.5

    def test_off_track_episode_counts_as_a_crash(self):
        result = TrackResult(track="a", split="train")
        result.episodes.append(EpisodeResult(end_reason="off_track"))
        result.episodes.append(EpisodeResult(end_reason="finished", finished=True))
        assert result.crash_rate == 0.5
