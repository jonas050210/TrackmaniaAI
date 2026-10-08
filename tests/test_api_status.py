"""Tests for the read-only run status API that a future dashboard consumes.

These run a real (simulated-driver) training run and then read its artefacts back, so the API
is exercised against the files the trainer actually writes rather than against hand-made
fixtures that could drift from the real format.
"""

from __future__ import annotations

import json

import pytest

from tmai.api.status import (
    HEADLINE_METRICS,
    RunStatus,
    list_runs,
    run_checkpoints,
    run_episodes,
    run_evaluations,
    run_history,
    run_snapshot,
    run_status,
)
from tmai.config import RunConfig


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory) -> str:
    """One short real training run, shared by every test in this module."""
    from tmai.training.trainer import train_from_config

    output = tmp_path_factory.mktemp("runs")
    config = RunConfig()
    config.driver.kind = "simulated"
    config.driver.allow_simulated = True
    config.track.synthetic_suite = [
        {"name": "straight", "kwargs": {"length": 150.0}},
        {"name": "oval"},
        {"name": "s_curve"},
        {"name": "figure_eight"},
    ]
    config.track.split_weights = {"train": 0.5, "validation": 0.5, "test": 0.0}
    config.train.run_name = "api-run"
    config.train.output_dir = str(output)
    config.train.total_steps = 300
    config.train.warmup_steps = 40
    config.train.batch_size = 32
    config.train.log_interval = 100
    config.train.eval_interval = 150
    config.train.eval_episodes = 1
    config.train.held_out_eval_interval = 150
    config.train.held_out_eval_episodes = 1
    config.train.checkpoint_interval = 300
    config.env.termination.max_steps = 100
    config.sac.network.hidden_sizes = (16, 16)

    result = train_from_config(config)
    assert result.failure is None
    return str(result.run_dir)


class TestRunStatus:
    def test_reads_a_completed_run(self, completed_run):
        status = run_status(completed_run)
        assert status.exists
        assert status.step == 300
        assert status.total_steps == 300
        assert status.progress_fraction == pytest.approx(1.0)
        assert status.ended
        assert status.metrics_records > 0

    def test_reports_the_driver_and_flags_the_toy_model(self, completed_run):
        status = run_status(completed_run)
        assert status.driver == "simulated"
        assert status.simulated is True

    def test_exposes_headline_metrics(self, completed_run):
        latest = run_status(completed_run).latest
        assert latest, "no headline metrics reported"
        assert all(key in HEADLINE_METRICS for key in latest)
        assert all(isinstance(value, float) for value in latest.values())

    def test_latest_is_searched_backwards_across_records(self, completed_run):
        """The final record is often an eval record with only eval keys.

        Reading just that record would report almost nothing, so each headline metric is
        looked up from the most recent record that actually carries it.
        """
        latest = run_status(completed_run).latest
        # Progress and throughput are only written on log_interval records, never on the
        # final evaluation record.
        assert "env/progress_fraction" in latest
        assert "throughput/env_steps_per_second" in latest
        assert "eval/generalization_gap" in {
            curve.name for curve in run_history(completed_run).series
        }

    def test_counts_episodes_evaluations_and_checkpoints(self, completed_run):
        status = run_status(completed_run)
        assert status.episodes >= 1
        assert status.evaluations >= 2  # training + held-out
        assert status.checkpoints >= 1

    def test_missing_directory(self, tmp_path):
        status = run_status(tmp_path / "nope")
        assert status.exists is False
        assert status.step == 0

    def test_serialises_to_json(self, completed_run):
        payload = run_status(completed_run).as_dict()
        json.dumps(payload)
        assert payload["schema_version"] == 1

    def test_empty_run_status_defaults(self):
        status = RunStatus(run_dir="/nowhere")
        assert status.step == 0
        assert status.latest == {}


class TestRunHistory:
    def test_returns_series(self, completed_run):
        history = run_history(completed_run)
        assert history.total_records > 0
        assert history.series

    def test_groups_metrics_by_prefix(self, completed_run):
        history = run_history(completed_run)
        assert history.available
        assert any(group in ("Environment", "Reward", "Learner") for group in history.available)

    def test_downsampling_bounds_the_point_count(self, completed_run):
        history = run_history(completed_run, max_points=3)
        for curve in history.series:
            # One extra point is appended to guarantee the final value is present.
            assert len(curve.steps) <= 4

    def test_downsampling_keeps_last_point_of_sparse_series(self, tmp_path):
        from tmai.runlog import RunLogger

        with RunLogger(tmp_path, run_name="sparse") as logger:
            for step in range(10):
                logger.log_metrics(step, {"env/progress_fraction": step / 10})
            logger.log_metrics(10, {"eval/finish_rate": 0.5})

        history = run_history(tmp_path, max_points=3)
        curve = history.get("env/progress_fraction")
        assert curve is not None
        assert curve.steps[-1] == 9
        assert curve.values[-1] == pytest.approx(0.9)

    def test_steps_and_values_align(self, completed_run):
        for curve in run_history(completed_run).series:
            assert len(curve.steps) == len(curve.values)

    def test_filter_selects_metrics(self, completed_run):
        history = run_history(completed_run, metrics_filter=["env/progress_fraction"])
        assert [c.name for c in history.series] == ["env/progress_fraction"]

    def test_get_by_name(self, completed_run):
        history = run_history(completed_run)
        curve = history.get("env/progress_fraction")
        assert curve is not None
        assert history.get("does/not/exist") is None

    def test_empty_run(self, tmp_path):
        history = run_history(tmp_path)
        assert history.total_records == 0
        assert history.series == []

    def test_training_director_metrics_are_exposed_and_grouped(self, tmp_path):
        from tmai.runlog import RunLogger

        with RunLogger(tmp_path, run_name="director-status") as logger:
            logger.log_metrics(
                10,
                {
                    "director/weakest_score": 0.35,
                    "director/failure_rate": 0.2,
                    "director/mean_weight": 1.4,
                },
            )

        history = run_history(tmp_path)
        assert set(history.available["Training Director"]) == {
            "director/weakest_score",
            "director/failure_rate",
            "director/mean_weight",
        }
        status = run_status(tmp_path)
        assert status.latest["director/weakest_score"] == pytest.approx(0.35)
        assert status.latest["director/failure_rate"] == pytest.approx(0.2)

    def test_serialises(self, completed_run):
        json.dumps(run_history(completed_run).as_dict())


class TestRunEpisodes:
    def test_reads_episode_records(self, completed_run):
        episodes = run_episodes(completed_run)
        assert episodes
        assert all(e.episode > 0 for e in episodes)

    def test_records_which_track_each_episode_ran_on(self, completed_run):
        """Without this, per-track episode analysis is impossible for a multi-track run."""
        episodes = run_episodes(completed_run)
        assert any(e.track for e in episodes), "no episode recorded its track"

    def test_limit_returns_the_most_recent(self, completed_run):
        all_episodes = run_episodes(completed_run)
        limited = run_episodes(completed_run, limit=1)
        assert len(limited) == 1
        assert limited[0].episode == all_episodes[-1].episode

    def test_end_reasons_are_machine_readable(self, completed_run):
        reasons = {e.end_reason for e in run_episodes(completed_run)}
        assert reasons
        assert all(isinstance(r, str) and r for r in reasons)

    def test_serialises(self, completed_run):
        json.dumps([e.as_dict() for e in run_episodes(completed_run)])

    def test_invalid_finish_preserves_raw_game_outcome(self, tmp_path):
        from tmai.runlog import RunLogger

        with RunLogger(tmp_path, run_name="invalid-finish") as logger:
            logger.log_event(
                "episode_end",
                step=12,
                episode=1,
                steps=12,
                reward=3.0,
                progress=50.0,
                end_reason="invalid_finish",
                finished=False,
                game_finished=True,
                invalid_finish=True,
                race_time=4.0,
                track="cut-map",
            )

        [episode] = run_episodes(tmp_path)
        assert episode.finished is False
        assert episode.game_finished is True
        assert episode.invalid_finish is True
        assert episode.end_reason == "invalid_finish"

    def test_missing_directory(self, tmp_path):
        assert run_episodes(tmp_path) == []


class TestRunEvaluations:
    def test_separates_training_from_held_out(self, completed_run):
        kinds = {item["kind"] for item in run_evaluations(completed_run)}
        assert "training" in kinds
        assert "held_out" in kinds

    def test_reports_are_dicts_not_strings(self, completed_run):
        """A regression: nested payloads used to be stringified, breaking machine reads."""
        for item in run_evaluations(completed_run):
            assert isinstance(item["report"], dict)
            assert "mean_progress_fraction" in item["report"]

    def test_evaluations_carry_a_step(self, completed_run):
        assert all(item["step"] > 0 for item in run_evaluations(completed_run))

    def test_missing_directory(self, tmp_path):
        assert run_evaluations(tmp_path) == []


class TestRunCheckpoints:
    def test_lists_checkpoints_and_best(self, completed_run):
        names = {c.name for c in run_checkpoints(completed_run)}
        assert any(name.startswith("checkpoint_") for name in names)
        assert "best.pt" in names

    def test_best_is_flagged(self, completed_run):
        best = [c for c in run_checkpoints(completed_run) if c.is_best]
        assert len(best) == 1
        assert best[0].name == "best.pt"

    def test_metadata_is_populated(self, completed_run):
        regular = [c for c in run_checkpoints(completed_run) if not c.is_best]
        assert regular
        assert regular[0].step == 300
        assert regular[0].gradient_steps > 0

    def test_missing_directory(self, tmp_path):
        assert run_checkpoints(tmp_path) == []


class TestRunSnapshot:
    def test_contains_everything_a_dashboard_needs(self, completed_run):
        snapshot = run_snapshot(completed_run)
        for key in (
            "schema_version",
            "status",
            "history",
            "episodes",
            "evaluations",
            "checkpoints",
            "manifest",
            "config_yaml",
            "log_tail",
        ):
            assert key in snapshot

    def test_is_fully_json_serialisable(self, completed_run):
        blob = json.dumps(run_snapshot(completed_run), default=str)
        assert json.loads(blob)["status"]["step"] == 300

    def test_includes_the_config_so_a_run_is_self_describing(self, completed_run):
        snapshot = run_snapshot(completed_run)
        assert snapshot["config_yaml"]
        assert "synthetic_suite" in snapshot["config_yaml"] or "track" in snapshot["config_yaml"]

    def test_log_tail_is_bounded(self, completed_run):
        assert len(run_snapshot(completed_run)["log_tail"]) <= 40

    def test_missing_directory_does_not_raise(self, tmp_path):
        snapshot = run_snapshot(tmp_path / "nope")
        assert snapshot["status"]["exists"] is False


class TestListRuns:
    def test_finds_runs(self, completed_run):
        from pathlib import Path

        runs = list_runs(Path(completed_run).parent)
        assert len(runs) >= 1
        assert any(run["run_name"] == "api-run" for run in runs)

    def test_reports_progress_and_driver(self, completed_run):
        from pathlib import Path

        run = next(r for r in list_runs(Path(completed_run).parent) if r["run_name"] == "api-run")
        assert run["step"] == 300
        assert run["driver"] == "simulated"
        assert run["simulated"] is True

    def test_missing_directory(self, tmp_path):
        assert list_runs(tmp_path / "nope") == []
