"""Benchmarking: same protocol for every model, ranked output."""

from __future__ import annotations

import json

import pytest

from tmai.config import RunConfig
from tmai.training.benchmark import BenchmarkReport, run_benchmark
from tmai.training.checkpoint import save_checkpoint


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
    saved = save_checkpoint(
        tmp_path / name, step=step, learner=learner, config=config.to_dict()
    )
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
        assert [m.label for m in report.models] == ["first", "second"]
        assert len(report.ranking) == 2
        assert set(report.ranking) == {"first", "second"}
        for model in report.models:
            assert "validation" in model.reports
            assert model.reports["validation"].num_episodes >= 1
        # Deterministic given the same inputs.
        again = run_benchmark(
            config,
            [("first", first), ("second", second)],
            splits=["validation"],
            episodes_per_track=1,
        )
        assert report.ranking == again.ranking

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
            run_benchmark(
                config, [("ghost", tmp_path / "nope.pt")], splits=["validation"]
            )

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
        assert data["schema_version"] == 1
        assert data["ranking"] == ["solo"]
        assert data["models"][0]["splits"]["validation"]["num_tracks"] >= 1
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
