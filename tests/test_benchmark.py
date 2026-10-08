"""Benchmarking: same protocol for every model, ranked output."""

from __future__ import annotations

import json

import pytest

from tmai.config import RunConfig
from tmai.training.benchmark import (
    BenchmarkModel,
    BenchmarkReport,
    _bootstrap_summary,
    _evaluation_confidence_intervals,
    _merge_evaluations,
    _paired_comparisons,
    run_benchmark,
)
from tmai.training.checkpoint import save_checkpoint
from tmai.training.evaluate import EpisodeResult, EvaluationReport, TrackResult


def _config(tmp_path) -> RunConfig:
    config = RunConfig.from_yaml("tmai/configs/multitrack_smoke.yaml")
    config.train.output_dir = str(tmp_path / "runs")
    config.env.termination.max_steps = 60
    config.train.eval_interval = 0
    config.train.held_out_eval_interval = 0
    return config


def _checkpoint(tmp_path, name: str, step: int):
    """A checkpoint whose policy has a distinctive (step-dependent) bias."""
    import torch

    from tmai.training.factory import build_learner, build_library, build_multi_track_env

    config = _config(tmp_path)
    library = build_library(config)
    env = build_multi_track_env(config, library, split="train", seed=0)
    try:
        learner = build_learner(env, config)
    finally:
        env.close()
    # Bias the actor's output so different checkpoints behave differently. The learner may
    # be normalisation-wrapped; the policy lives on the inner SAC learner.
    sac = getattr(learner, "inner", learner)
    with torch.no_grad():
        for param in sac.network.policy.trunk[-1].parameters():
            param.add_(float(step) * 0.01)
    saved = save_checkpoint(tmp_path / name, step=step, learner=learner, config=config.to_dict())
    path = tmp_path / name / saved.name
    return path


class TestRunBenchmark:
    def test_two_models_are_ranked(self, tmp_path):
        config = _config(tmp_path)
        first = _checkpoint(tmp_path, "first", step=10)
        second = _checkpoint(tmp_path, "second", step=20)

        report = run_benchmark(
            config,
            [("first", first), ("second", second)],
            splits=["validation"],
            episodes_per_track=1,
            name="test-bench",
        )
        assert isinstance(report, BenchmarkReport)
        assert report.name == "test-bench"
        assert report.seed_repeats == 3
        assert len(report.evaluation_seeds) == 3
        assert len(set(report.evaluation_seeds)) == 3
        assert [m.label for m in report.models] == ["first", "second"]
        assert len(report.ranking) == 2
        assert set(report.ranking) == {"first", "second"}
        for model in report.models:
            assert "validation" in model.reports
            assert model.reports["validation"].num_episodes >= 3
            intervals = model.confidence_intervals["validation"]
            assert set(intervals) == {"finish_rate", "mean_progress_fraction", "crash_rate"}
            progress_ci = intervals["mean_progress_fraction"]["ci95"]
            assert progress_ci is None or len(progress_ci) == 2
        # Deterministic given the same inputs.
        again = run_benchmark(
            config,
            [("first", first), ("second", second)],
            splits=["validation"],
            episodes_per_track=1,
        )
        assert report.ranking == again.ranking

    def test_curvature_heuristic_runs_in_benchmark_and_is_paired(self, tmp_path):
        config = _config(tmp_path)
        checkpoint = _checkpoint(tmp_path, "policy", step=10)
        report = run_benchmark(
            config,
            [("pilot", "baseline:curvature"), ("policy", checkpoint)],
            splits=["validation"],
            episodes_per_track=2,
        )

        pilot = next(model for model in report.models if model.label == "pilot")
        assert pilot.baseline_kind == "curvature"
        assert pilot.checkpoint == "baseline:curvature"
        assert pilot.reports["validation"].num_episodes >= 2
        assert len(report.head_to_head) == 1
        paired = report.head_to_head[0]
        assert paired.model_a == "pilot"
        assert paired.model_b == "policy"
        assert paired.episodes == paired.wins_a + paired.wins_b + paired.ties
        assert "paired head-to-head" in report.table()

    def test_bootstrap_resamples_track_clusters_deterministically(self):
        groups = [[0.2, 0.4], [0.8, 1.0]]
        first = _bootstrap_summary(groups, seed=7)
        second = _bootstrap_summary(groups, seed=7)
        assert first == second
        assert first["resampling_unit"] == "tracks"
        assert first["sample_count"] == 2
        assert first["ci95"][0] <= 0.6 <= first["ci95"][1]

    def test_bootstrap_reports_family_cluster_count(self):
        estimate = _bootstrap_summary(
            [[0.2, 0.4], [0.4, 0.6], [0.8]],
            seed=9,
            cluster_ids=["family:pack", "family:pack", "family:other"],
        )
        assert estimate["resampling_unit"] == "families"
        assert estimate["sample_count"] == 2
        assert estimate["ci95"][0] <= 0.433333 <= estimate["ci95"][1]

    def test_one_family_interval_is_explicitly_episode_scoped(self):
        estimate = _bootstrap_summary(
            [[0.2, 0.4], [0.4, 0.6]],
            seed=1,
            cluster_ids=["family:pack", "family:pack"],
        )
        assert estimate["resampling_unit"] == "episodes_within_family"
        assert estimate["sample_count"] == 4

    def test_bootstrap_reports_no_interval_for_one_observation(self):
        estimate = _bootstrap_summary([[0.5]], seed=0)
        assert estimate == {
            "ci95": None,
            "resampling_unit": None,
            "sample_count": 1,
        }

    def test_evaluation_intervals_keep_related_tracks_in_one_family_cluster(self):
        source = EvaluationReport(
            tracks=[
                TrackResult(
                    "variant-a",
                    "test",
                    episodes=[EpisodeResult(progress_fraction=0.2), EpisodeResult(progress_fraction=0.3)],
                ),
                TrackResult(
                    "variant-b",
                    "test",
                    episodes=[EpisodeResult(progress_fraction=0.4), EpisodeResult(progress_fraction=0.5)],
                ),
            ]
        )
        merged = _merge_evaluations(
            [source], families={"variant-a": "author-pack", "variant-b": "author-pack"}
        )
        assert [track.family for track in merged.tracks] == ["author-pack", "author-pack"]
        intervals = _evaluation_confidence_intervals(merged, seed=0, model_label="policy", split="test")
        assert intervals["mean_progress_fraction"]["resampling_unit"] == "episodes_within_family"

    def test_pairing_matches_track_and_episode_index_only(self):
        first = BenchmarkModel(
            "first",
            "first.pt",
            reports={
                "validation": EvaluationReport(
                    tracks=[
                        TrackResult(
                            "same-track",
                            "validation",
                            episodes=[
                                EpisodeResult(finished=True, race_time=12.0, progress_fraction=1.0),
                                EpisodeResult(progress_fraction=0.8),
                            ],
                        ),
                        TrackResult("unmatched", "validation", episodes=[EpisodeResult()]),
                    ]
                )
            },
        )
        second = BenchmarkModel(
            "second",
            "second.pt",
            reports={
                "validation": EvaluationReport(
                    tracks=[
                        TrackResult(
                            "same-track",
                            "validation",
                            episodes=[
                                EpisodeResult(progress_fraction=1.0),
                                EpisodeResult(progress_fraction=0.9),
                            ],
                        )
                    ]
                )
            },
        )

        [comparison] = _paired_comparisons([first, second], ["validation"])
        assert comparison.tracks == 1
        assert comparison.episodes == 2
        assert (comparison.wins_a, comparison.wins_b, comparison.ties) == (1, 1, 0)
        assert comparison.mean_progress_delta == pytest.approx(-0.05)
        assert comparison.win_rate_a == pytest.approx(0.5)
        assert comparison.confidence_intervals["win_rate_a"]["resampling_unit"] == "episodes"
        assert comparison.confidence_intervals["mean_progress_delta"]["ci95"] is not None

    def test_model_labels_must_be_unique_and_nonempty(self, tmp_path):
        config = _config(tmp_path)
        with pytest.raises(ValueError, match="labels must be unique"):
            run_benchmark(
                config,
                [("same", "baseline:curvature"), ("same", "baseline:curvature")],
                splits=["validation"],
            )
        with pytest.raises(ValueError, match="labels must be non-empty"):
            run_benchmark(
                config,
                [("   ", "baseline:curvature")],
                splits=["validation"],
            )

    def test_seed_repeats_are_bounded(self, tmp_path):
        for repeats in (0, 101):
            with pytest.raises(ValueError, match="seed_repeats must be in"):
                run_benchmark(
                    _config(tmp_path),
                    [("pilot", "baseline:curvature")],
                    splits=["validation"],
                    seed_repeats=repeats,
                )

    def test_unknown_baseline_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown benchmark baseline"):
            run_benchmark(
                _config(tmp_path),
                [("unknown", "baseline:unknown")],
                splits=["validation"],
                episodes_per_track=1,
            )

    def test_run_directory_resolves_to_latest_checkpoint(self, tmp_path):
        config = _config(tmp_path)
        checkpoint = _checkpoint(tmp_path, "ckpt", step=5)
        run_dir = checkpoint.parent
        report = run_benchmark(
            config,
            [("from-run", run_dir)],
            splits=["validation"],
            episodes_per_track=1,
        )
        assert report.models[0].checkpoint.endswith("checkpoint_000000005.pt")

    def test_missing_checkpoint_raises(self, tmp_path):
        config = _config(tmp_path)
        with pytest.raises(FileNotFoundError):
            run_benchmark(config, [("ghost", tmp_path / "nope.pt")], splits=["validation"])

    def test_empty_run_directory_raises(self, tmp_path):
        config = _config(tmp_path)
        empty = tmp_path / "empty-run"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="no checkpoint"):
            run_benchmark(config, [("x", empty)], splits=["validation"])

    def test_report_serialises_and_tables(self, tmp_path):
        config = _config(tmp_path)
        checkpoint = _checkpoint(tmp_path, "solo", step=5)
        report = run_benchmark(
            config,
            [("solo", checkpoint)],
            splits=["validation"],
            episodes_per_track=1,
        )
        out = report.save(tmp_path / "bench.json")
        data = json.loads(out.read_text())
        assert data["schema_version"] == 3
        assert data["seed_repeats"] == 3
        assert len(data["evaluation_seeds"]) == 3
        assert data["ranking"] == ["solo"]
        assert data["models"][0]["splits"]["validation"]["num_tracks"] >= 1
        assert data["head_to_head"] == []
        table = report.table()
        assert "solo" in table
        assert "validation" in table
        assert "best" in report.summary()

    def test_empty_split_is_rejected_not_faked(self, tmp_path):
        """A split with no tracks is a configuration error, not an empty report row."""
        from tmai.training.factory import ConfigError

        config = _config(tmp_path)
        checkpoint = _checkpoint(tmp_path, "solo", step=5)
        with pytest.raises(ConfigError, match="'test' split is empty"):
            run_benchmark(
                config,
                [("solo", checkpoint)],
                splits=["test"],  # multitrack_smoke assigns 0 tracks to test
                episodes_per_track=1,
            )
