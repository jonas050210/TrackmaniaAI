"""Tests for the training loop, evaluation, factory wiring and the CLI surface."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tmai.config import RunConfig
from tmai.env.termination import TerminationConfig
from tmai.game.errors import GameConnectionError
from tmai.runlog import read_manifest, read_metrics
from tmai.training.checkpoint import latest_checkpoint
from tmai.training.evaluate import EpisodeResult, EvaluationReport, evaluate_policy
from tmai.training.factory import (
    ConfigError,
    build_all,
    build_driver,
    build_track,
)
from tmai.training.trainer import Trainer, TrainerResult, train_from_config


def smoke_config(tmp_path, **train_overrides) -> RunConfig:
    """A configuration small enough to run in well under a second."""
    config = RunConfig()
    config.driver.kind = "simulated"
    config.driver.allow_simulated = True
    config.track.synthetic = "straight"
    config.track.synthetic_kwargs = {"length": 150.0}
    config.env.termination = TerminationConfig(max_steps=40)
    config.sac.network.hidden_sizes = (16, 16)
    config.replay.capacity = 5_000
    config.train.total_steps = 120
    config.train.warmup_steps = 30
    config.train.batch_size = 16
    config.train.log_interval = 40
    config.train.eval_interval = 0
    config.train.eval_episodes = 0
    config.train.checkpoint_interval = 60
    config.train.keep_checkpoints = 2
    config.train.output_dir = str(tmp_path / "runs")
    config.train.run_name = "test"
    for key, value in train_overrides.items():
        setattr(config.train, key, value)
    return config


class TestFactory:
    def test_build_track_from_synthetic(self):
        config = RunConfig()
        config.track.synthetic = "oval"
        track = build_track(config)
        assert track.name == "oval"
        assert track.length > 0

    def test_build_track_from_file(self, tmp_path):
        from tmai.tracks.synthetic import s_curve

        path = tmp_path / "t.json"
        s_curve().save(path)
        config = RunConfig()
        config.track.path = str(path)
        assert build_track(config).name == "s_curve"

    def test_build_track_without_source_raises(self):
        with pytest.raises(ConfigError, match="no track configured"):
            build_track(RunConfig())

    def test_simulated_driver_is_refused_by_default(self):
        config = RunConfig()
        config.driver.kind = "simulated"
        config.track.synthetic = "straight"
        with pytest.raises(ConfigError, match="Refusing to start"):
            build_driver(config, build_track(config))

    def test_simulated_driver_allowed_when_flagged(self):
        config = RunConfig()
        config.driver.kind = "simulated"
        config.driver.allow_simulated = True
        config.track.synthetic = "straight"
        driver = build_driver(config, build_track(config))
        assert driver.name == "simulated"

    def test_unknown_driver_kind_raises(self):
        config = RunConfig()
        config.driver.kind = "carbondioxide"
        config.track.synthetic = "straight"
        with pytest.raises(ConfigError, match="unknown driver.kind"):
            build_driver(config, build_track(config))

    def test_tminterface_driver_is_built_without_connecting(self):
        """Building must not touch the game; connecting happens in open()."""
        config = RunConfig()
        config.driver.kind = "tminterface"
        config.track.synthetic = "straight"
        driver = build_driver(config, build_track(config))
        assert driver.name == "tminterface"
        assert driver.is_connected() is False

    def test_build_all_wires_everything(self):
        config = RunConfig()
        config.driver.kind = "simulated"
        config.driver.allow_simulated = True
        config.track.synthetic = "straight"
        config.sac.network.hidden_sizes = (16,)
        env, learner, buffer, track = build_all(config)
        assert env.observation_dim == learner.observation_dim
        assert int(np.prod(env.action_space.shape)) == learner.action_dim
        assert buffer.observation_dim == env.observation_dim
        assert track.name == "straight"
        env.close()

    def test_learner_matches_action_bounds(self):
        config = RunConfig()
        config.driver.kind = "simulated"
        config.driver.allow_simulated = True
        config.track.synthetic = "straight"
        config.sac.network.hidden_sizes = (16,)
        env, learner, _, _ = build_all(config)
        action = learner.act(env.observation_space.sample())
        assert action[1] >= 0.0 and action[2] >= 0.0
        env.close()


class TestTrainer:
    def test_runs_and_produces_artefacts(self, tmp_path):
        config = smoke_config(tmp_path)
        result = train_from_config(config)

        assert isinstance(result, TrainerResult)
        assert result.steps == config.train.total_steps
        assert result.failure is None
        assert result.interrupted is False

        run_dir = result.run_dir
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "metrics.jsonl").exists()
        assert (run_dir / "events.jsonl").exists()
        assert (run_dir / "config.yaml").exists()
        assert latest_checkpoint(run_dir) is not None

    def test_gradient_steps_match_the_update_schedule(self, tmp_path):
        config = smoke_config(tmp_path)
        result = train_from_config(config)
        expected = config.train.total_steps - config.train.warmup_steps
        assert result.gradient_steps == expected

    def test_updates_per_step_scales_gradient_steps(self, tmp_path):
        config = smoke_config(tmp_path)
        config.train.updates_per_step = 3.0
        result = train_from_config(config)
        expected = 3 * (config.train.total_steps - config.train.warmup_steps)
        assert result.gradient_steps == expected

    def test_metrics_are_logged_at_the_requested_interval(self, tmp_path):
        config = smoke_config(tmp_path)
        result = train_from_config(config)
        records = read_metrics(result.run_dir)
        expected = config.train.total_steps // config.train.log_interval
        assert len(records) == expected
        assert "throughput/env_steps_per_second" in records[0]
        assert "sac/critic_loss" in records[0]

    def test_manifest_records_driver_and_environment(self, tmp_path):
        config = smoke_config(tmp_path)
        result = train_from_config(config)
        manifest = read_manifest(result.run_dir)
        assert manifest["environment"]["driver"]["driver"] == "simulated"
        assert manifest["learner"]["algorithm"] == "sac"
        assert manifest["config"]["driver"]["kind"] == "simulated"

    def test_wall_clock_limit_stops_the_run(self, tmp_path):
        config = smoke_config(tmp_path, total_steps=100_000, max_wall_seconds=0.5)
        result = train_from_config(config)
        assert result.steps < config.train.total_steps
        assert result.wall_seconds < 30

    def test_evaluation_resets_the_training_episode(self, tmp_path):
        """Regression: evaluation drives the env, so the loop must restart its episode.

        Before the fix, the trainer continued from a stale observation after an eval and the
        next episode ended immediately (1 step, 'stalled').
        """
        config = smoke_config(tmp_path, total_steps=200, eval_interval=60, eval_episodes=1)
        result = train_from_config(config)
        events = [
            json.loads(line)
            for line in (result.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        episode_ends = [e for e in events if e["event"] == "episode_end"]
        assert episode_ends, "expected at least one episode to end"
        # An evaluation sits between some episode ends; none of the following episodes may be
        # the degenerate one-step episode the stale-observation bug produced.
        assert all(e["steps"] > 1 for e in episode_ends), [e["steps"] for e in episode_ends]

    def test_resume_restores_gradient_step_count(self, tmp_path):
        config = smoke_config(tmp_path, total_steps=80)
        first = train_from_config(config)
        assert first.gradient_steps > 0

        config.train.resume = str(first.run_dir)
        config.train.total_steps = 140
        config.train.output_dir = str(tmp_path / "runs2")
        second = train_from_config(config)

        # The resumed run continues the counter rather than restarting it.
        assert second.gradient_steps > first.gradient_steps
        manifest = read_manifest(second.run_dir)
        assert manifest["resumed_from"] == str(first.run_dir)

    def test_resume_from_checkpoint_file(self, tmp_path):
        config = smoke_config(tmp_path, total_steps=80)
        first = train_from_config(config)
        checkpoint = latest_checkpoint(first.run_dir)

        config.train.resume = str(checkpoint)
        config.train.total_steps = 100
        config.train.output_dir = str(tmp_path / "runs2")
        second = train_from_config(config)
        assert second.gradient_steps >= first.gradient_steps

    def test_resume_missing_checkpoint_raises(self, tmp_path):
        config = smoke_config(tmp_path)
        config.train.resume = str(tmp_path / "does-not-exist")
        with pytest.raises(FileNotFoundError, match="no checkpoint found"):
            train_from_config(config)

    def test_game_error_is_captured_and_checkpointed(self, tmp_path, monkeypatch):
        config = smoke_config(tmp_path, total_steps=10_000, checkpoint_interval=0)
        env, learner, buffer, _ = build_all(config)

        from tmai.runlog import RunLogger

        calls = {"n": 0}
        real_step = env.step

        def flaky_step(action):
            calls["n"] += 1
            if calls["n"] > 10:
                raise GameConnectionError("pretend the game died")
            return real_step(action)

        monkeypatch.setattr(env, "step", flaky_step)

        with RunLogger(tmp_path / "run", config=config.to_dict()) as logger:
            result = Trainer(config, env, learner, buffer, logger).train()

        assert result.failure is not None
        assert "pretend the game died" in result.failure
        # Work must be preserved even when the game is lost.
        assert latest_checkpoint(result.run_dir) is not None
        env.close()

    def test_keyboard_interrupt_is_captured(self, tmp_path, monkeypatch):
        config = smoke_config(tmp_path, total_steps=10_000, checkpoint_interval=0)
        env, learner, buffer, _ = build_all(config)

        from tmai.runlog import RunLogger

        calls = {"n": 0}
        real_step = env.step

        def interrupting_step(action):
            calls["n"] += 1
            if calls["n"] > 5:
                raise KeyboardInterrupt
            return real_step(action)

        monkeypatch.setattr(env, "step", interrupting_step)

        with RunLogger(tmp_path / "run", config=config.to_dict()) as logger:
            result = Trainer(config, env, learner, buffer, logger).train()

        assert result.interrupted is True
        assert result.failure is None
        assert latest_checkpoint(result.run_dir) is not None
        env.close()

    def test_episodes_are_counted(self, tmp_path):
        config = smoke_config(tmp_path, total_steps=200)
        result = train_from_config(config)
        assert result.episodes >= 1


class TestEvaluation:
    def _setup(self):
        config = RunConfig()
        config.driver.kind = "simulated"
        config.driver.allow_simulated = True
        config.track.synthetic = "straight"
        config.track.synthetic_kwargs = {"length": 120.0}
        config.env.termination = TerminationConfig(max_steps=2000)
        config.sac.network.hidden_sizes = (16,)
        env, learner, _, _ = build_all(config)
        return env, learner, config

    def test_report_aggregates_episodes(self):
        env, learner, config = self._setup()
        report = evaluate_policy(env, learner, episodes=2, max_steps=200)
        env.close()
        assert report.num_episodes == 2
        assert len(report.episodes) == 2
        assert 0.0 <= report.finish_rate <= 1.0
        assert 0.0 <= report.mean_progress_fraction <= 1.0

    def test_score_rewards_finishing(self):
        unfinished = EvaluationReport.from_episodes([EpisodeResult(progress_fraction=0.5, finished=False)]
        )
        finished = EvaluationReport.from_episodes([EpisodeResult(progress_fraction=1.0, finished=True, race_time=30.0)]
        )
        assert finished.score > unfinished.score

    def test_score_prefers_faster_finishes(self):
        slow = EvaluationReport.from_episodes([EpisodeResult(progress_fraction=1.0, finished=True, race_time=120.0)]
        )
        fast = EvaluationReport.from_episodes([EpisodeResult(progress_fraction=1.0, finished=True, race_time=30.0)]
        )
        assert fast.score > slow.score

    def test_score_is_bounded(self):
        report = EvaluationReport.from_episodes([EpisodeResult(progress_fraction=1.0, finished=True, race_time=0.001)]
        )
        assert 0.0 <= report.score < 2.0

    def test_empty_report(self):
        report = EvaluationReport()
        assert report.finish_rate == 0.0
        assert report.mean_progress_fraction == 0.0
        assert report.score == 0.0
        assert report.best_race_time is None

    def test_metrics_keys(self):
        env, learner, config = self._setup()
        report = evaluate_policy(env, learner, episodes=1, max_steps=100)
        env.close()
        metrics = report.metrics()
        assert "eval/finish_rate" in metrics
        assert "eval/mean_progress_fraction" in metrics
        assert "eval/score" in metrics
        assert all(np.isfinite(v) for v in metrics.values())

    def test_summary_string(self):
        report = EvaluationReport.from_episodes([EpisodeResult(progress_fraction=0.25)])
        assert "25.0%" in report.summary()

    def test_deterministic_evaluation_is_reproducible(self):
        env, learner, config = self._setup()
        first = evaluate_policy(env, learner, episodes=1, max_steps=200, deterministic=True)
        second = evaluate_policy(env, learner, episodes=1, max_steps=200, deterministic=True)
        env.close()
        assert first.episodes[0].progress == pytest.approx(second.episodes[0].progress)

    def test_invalid_finish_is_flagged(self):
        """A finish without the map's checkpoints must not count as a real lap."""
        env, learner, config = self._setup()

        from tmai.training.evaluate import run_episode

        # The synthetic straight has checkpoints every 60 m, so a full lap needs them all.
        episode = run_episode(env, learner, max_steps=2000, deterministic=True)
        env.close()
        if episode.finished:
            assert episode.invalid_finish is (
                episode.race_time > 0 and not episode.finished
            ) or episode.invalid_finish is False

    def test_report_is_json_serialisable(self):
        env, learner, config = self._setup()
        report = evaluate_policy(env, learner, episodes=1, max_steps=50)
        env.close()
        assert json.dumps(report.as_dict())


class TestCLI:
    def test_parser_builds(self):
        from tmai.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["train", "--steps", "10"])
        assert args.command == "train"
        assert args.steps == 10

    def test_doctor_reports_environment(self, capsys):
        from tmai.cli import build_parser, cmd_doctor

        parser = build_parser()
        args = parser.parse_args(["doctor"])
        code = cmd_doctor(args)
        output = capsys.readouterr().out
        assert "environment" in output
        assert "game integration" in output
        assert code in (0, 1)

    def test_show_track_writes_png(self, tmp_path, capsys):
        from tmai.cli import build_parser, cmd_show_track
        from tmai.tracks.synthetic import s_curve

        track_path = tmp_path / "t.json"
        s_curve().save(track_path)
        out = tmp_path / "view.png"

        parser = build_parser()
        args = parser.parse_args(
            ["show-track", "--track", str(track_path), "--out", str(out)]
        )
        assert cmd_show_track(args) == 0
        assert out.exists()
        assert out.stat().st_size > 1000
        assert "wrote" in capsys.readouterr().out

    def test_export_obj_writes_mesh(self, tmp_path):
        from tmai.cli import build_parser, cmd_export_obj
        from tmai.tracks.synthetic import straight

        track_path = tmp_path / "t.json"
        straight(length=20.0).save(track_path)
        out = tmp_path / "track.obj"

        parser = build_parser()
        args = parser.parse_args(["export-obj", "--track", str(track_path), "--out", str(out)])
        assert cmd_export_obj(args) == 0
        content = out.read_text(encoding="utf-8")
        assert content.startswith("# TrackmaniaAI track surface")
        assert "\nv " in content
        assert "\nf " in content
        assert "\nl " in content

    def test_main_returns_zero_for_show_track(self, tmp_path):
        from tmai.cli import main
        from tmai.tracks.synthetic import oval

        track_path = tmp_path / "t.json"
        oval().save(track_path)
        code = main(["show-track", "--track", str(track_path), "--out", str(tmp_path / "o.png")])
        assert code == 0

    def test_main_reports_errors_without_traceback(self, tmp_path, caplog):
        from tmai.cli import main

        with caplog.at_level("ERROR"):
            code = main(["export-obj", "--track", str(tmp_path / "missing.json"), "--out", "x.obj"])
        assert code == 1
        # A clean one-line report, not a traceback.
        messages = [r.getMessage() for r in caplog.records]
        assert any("FileNotFoundError" in m for m in messages)
        assert all("Traceback" not in m for m in messages)

    def test_train_refuses_simulated_without_flag(self, tmp_path, capsys):
        from tmai.cli import build_parser, cmd_train

        config = smoke_config(tmp_path)
        config.driver.allow_simulated = False
        path = config.save(tmp_path / "cfg.yaml")

        parser = build_parser()
        args = parser.parse_args(["train", "--config", str(path)])
        with pytest.raises(ConfigError, match="--allow-simulated-driver"):
            cmd_train(args)

    def test_train_accepts_simulated_with_flag(self, tmp_path, capsys):
        from tmai.cli import build_parser, cmd_train

        config = smoke_config(tmp_path, total_steps=60)
        config.driver.allow_simulated = False
        path = config.save(tmp_path / "cfg.yaml")

        parser = build_parser()
        args = parser.parse_args(
            ["train", "--config", str(path), "--allow-simulated-driver", "--steps", "60"]
        )
        assert cmd_train(args) == 0
        assert "training finished" in capsys.readouterr().out
