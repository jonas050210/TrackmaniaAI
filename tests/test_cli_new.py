"""Tests for the CLI commands added for generalisation, inspection and comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tmai.cli import build_parser, main
from tmai.tracks.synthetic import build_synthetic


@pytest.fixture(scope="module")
def track_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("tracks")
    for name in ("straight", "oval", "s_curve", "figure_eight"):
        build_synthetic(name).save(directory / f"{name}.json")
    return directory


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory) -> list[str]:
    """Two short training runs, so `compare` has something real to compare."""
    from tmai.config import RunConfig
    from tmai.training.trainer import train_from_config

    output = tmp_path_factory.mktemp("cmp-runs")
    runs = []
    for index, name in enumerate(("run-a", "run-b")):
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
        config.train.run_name = name
        config.train.output_dir = str(output)
        config.train.total_steps = 200
        config.train.warmup_steps = 40
        config.train.batch_size = 32
        config.train.log_interval = 100
        config.train.eval_interval = 200
        config.train.eval_episodes = 1
        config.train.held_out_eval_interval = 200
        config.train.held_out_eval_episodes = 1
        config.train.checkpoint_interval = 200
        config.train.seed = index
        config.env.termination.max_steps = 80
        config.sac.network.hidden_sizes = (16, 16)
        runs.append(str(train_from_config(config).run_dir))
    return runs


# -- validate-config ----------------------------------------------------------------


class TestValidateConfig:
    def test_real_game_preset_targets_one_recorded_map(self):
        from tmai.config import RunConfig

        config = RunConfig.from_yaml("tmai/configs/default.yaml")
        assert config.driver.kind == "tminterface"
        assert config.track.path == "data/tracks/my_map.json"
        assert config.track.directory is None
        assert config.multi.sample_tracks is False
        assert config.train.held_out_eval_interval == 0
        assert config.validate() == []

    def test_valid_config_exits_zero(self, tmp_path, capsys):
        path = tmp_path / "good.yaml"
        path.write_text(
            "driver:\n  kind: simulated\n  allow_simulated: true\n"
            "track:\n  synthetic: oval\n"
            "train:\n  total_steps: 100\n  warmup_steps: 10\n  eval_interval: 50\n"
        )
        assert main(["validate-config", "-c", str(path)]) == 0
        assert "configuration is valid" in capsys.readouterr().out

    def test_invalid_config_exits_nonzero_and_lists_problems(self, tmp_path, capsys):
        path = tmp_path / "bad.yaml"
        # Two track sources, warmup above total, and the toy driver without permission.
        path.write_text(
            "driver:\n  kind: simulated\n"
            "track:\n  synthetic: oval\n  path: also.json\n"
            "train:\n  total_steps: 50\n  warmup_steps: 100\n"
        )
        assert main(["validate-config", "-c", str(path)]) == 1
        err = capsys.readouterr().err
        assert "exactly one track source" in err
        assert "warmup_steps" in err

    def test_json_output(self, tmp_path, capsys):
        path = tmp_path / "good.yaml"
        path.write_text(
            "driver:\n  kind: simulated\n  allow_simulated: true\n"
            "track:\n  synthetic: oval\ntrain:\n  total_steps: 100\n  warmup_steps: 10\n"
            "  eval_interval: 50\n"
        )
        assert main(["validate-config", "-c", str(path), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["valid"] is True
        assert payload["problems"] == []

    def test_detects_the_real_game_start_randomisation_conflict(self, tmp_path, capsys):
        """The real game cannot reposition the car; the config must not ask it to."""
        path = tmp_path / "conflict.yaml"
        path.write_text(
            "driver:\n  kind: tminterface\n"
            "track:\n  synthetic: oval\n"
            "multi:\n  random_start_station: true\n  start_lateral_std: 1.5\n"
            "train:\n  total_steps: 100\n  warmup_steps: 10\n  eval_interval: 50\n"
        )
        assert main(["validate-config", "-c", str(path)]) == 1
        err = capsys.readouterr().err
        assert "can reposition the car" in err
        assert "random_start_station" in err

    def test_shipped_configs_are_all_valid(self):
        import tmai

        configs = sorted((Path(tmai.__file__).parent / "configs").glob("*.yaml"))
        assert configs, "no shipped configs found"
        for path in configs:
            from tmai.config import RunConfig

            assert RunConfig.from_yaml(path).validate() == [], f"{path.name} is not valid"


# -- list-tracks --------------------------------------------------------------------


class TestRacingAnalysis:
    def test_analyze_replay_directory_writes_machine_readable_report(self, tmp_path, capsys):
        import numpy as np

        from tmai.replay import EpisodeReplay, ReplayStore
        from tmai.tracks.centerline import CenterlineTrack

        track = CenterlineTrack([[0.0, 0.0, 0.0], [0.0, 0.0, 100.0]], name="analysis-track")
        track_path = track.save(tmp_path / "track.json")
        positions = np.column_stack([np.zeros(101), np.zeros(101), np.linspace(0.0, 100.0, 101)])
        replay = EpisodeReplay(
            episode=1,
            step=100,
            track=track.name,
            end_reason="crash",
            positions=positions,
            speeds=np.full(101, 20.0),
            progress=np.linspace(0.0, 100.0, 101),
            race_times=np.linspace(0.0, 5.0, 101),
        )
        store = ReplayStore(tmp_path / "run" / "replays")
        store.save(replay)
        report_path = tmp_path / "analysis.json"

        assert (
            main(
                [
                    "analyze",
                    str(tmp_path / "run"),
                    "--track",
                    str(track_path),
                    "--sectors",
                    "4",
                    "--json-out",
                    str(report_path),
                ]
            )
            == 0
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["num_replays"] == 1
        assert report["failure_reasons"] == {"crash": 1}
        assert "failure heatmap" in capsys.readouterr().out


class TestListTracks:
    def test_lists_tracks_with_geometry(self, track_dir, capsys):
        assert main(["list-tracks", str(track_dir)]) == 0
        out = capsys.readouterr().out
        for name in ("straight", "oval", "s_curve", "figure_eight"):
            assert name in out
        assert "corners" in out
        assert "straight%" in out

    def test_reports_split_composition(self, track_dir, capsys):
        main(["list-tracks", str(track_dir)])
        out = capsys.readouterr().out
        assert "tracks:" in out

    def test_reports_geometry_coverage_per_split(self, track_dir, capsys):
        main(["list-tracks", str(track_dir)])
        assert "geometry coverage by split" in capsys.readouterr().out

    def test_json_output(self, track_dir, capsys):
        assert main(["list-tracks", str(track_dir), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["num_tracks"] == 4
        assert all(entry["stats"] for entry in payload["tracks"])

    def test_empty_directory_fails_cleanly(self, tmp_path, caplog):
        assert main(["list-tracks", str(tmp_path)]) == 1

    def test_pattern_filter(self, track_dir, capsys):
        assert main(["list-tracks", str(track_dir), "--pattern", "oval.json"]) == 0
        out = capsys.readouterr().out
        assert "oval" in out
        assert "figure_eight" not in out


# -- status -------------------------------------------------------------------------


class TestStatusCommand:
    def test_status_of_a_run(self, two_runs, capsys):
        assert main(["status", "--run", two_runs[0]]) == 0
        out = capsys.readouterr().out
        assert "run status" in out
        assert "step" in out
        assert "SIMULATED" in out  # the toy driver must be flagged loudly

    def test_status_json_snapshot(self, two_runs, capsys):
        assert main(["status", "--run", two_runs[0], "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"]["step"] == 200
        assert payload["history"]["series"]
        assert payload["episodes"]

    def test_status_shows_evaluations(self, two_runs, capsys):
        main(["status", "--run", two_runs[0]])
        out = capsys.readouterr().out
        assert "evaluations" in out
        assert "held_out" in out

    def test_missing_run(self, tmp_path, capsys):
        assert main(["status", "--run", str(tmp_path / "nope")]) == 1

    def test_no_run_argument_is_a_usage_error(self, capsys):
        assert main(["status"]) == 2

    def test_list_runs(self, two_runs, capsys):
        parent = str(Path(two_runs[0]).parent)
        assert main(["status", "--list", "--runs-dir", parent]) == 0
        out = capsys.readouterr().out
        assert "run-a" in out
        assert "run-b" in out

    def test_list_runs_in_an_empty_directory(self, tmp_path, capsys):
        assert main(["status", "--list", "--runs-dir", str(tmp_path)]) == 1


# -- compare ------------------------------------------------------------------------


class TestCompareCommand:
    def test_compares_two_runs(self, two_runs, capsys):
        assert main(["compare", *two_runs]) == 0
        out = capsys.readouterr().out
        assert "run-a" in out
        assert "run-b" in out
        assert "prog%" in out

    def test_json_output(self, two_runs, capsys):
        assert main(["compare", *two_runs, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload) == 2
        assert {row["name"] for row in payload} == {"run-a", "run-b"}

    def test_gap_is_computed_across_the_two_reports(self, two_runs, capsys):
        """Training and held-out runs are logged as separate reports, each carrying only its own
        split, so neither can produce the gap alone. Reading it off the training report yielded
        None and `compare` silently printed `--` for the one column that shows overfitting.
        """
        assert main(["compare", *two_runs, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)

        row = payload[0]
        assert row["progress"] is not None, "precondition: a training evaluation exists"
        assert row["held_progress"] is not None, "precondition: a held-out evaluation exists"
        assert row["gap"] is not None, "the gap must be computed, not left null"
        assert row["gap"] == round(row["progress"] - row["held_progress"], 4)

    def test_gap_is_absent_when_there_is_no_held_out_side(self, tmp_path, capsys):
        """Without a held-out evaluation the honest answer is None, not 0."""
        from tmai.config import RunConfig
        from tmai.training.trainer import train_from_config

        config = RunConfig()
        config.driver.kind = "simulated"
        config.driver.allow_simulated = True
        config.track.synthetic = "straight"
        config.train.output_dir = str(tmp_path)
        config.train.run_name = "no-heldout"
        config.train.total_steps = 40
        config.train.warmup_steps = 10
        config.train.batch_size = 8
        config.train.log_interval = 20
        config.train.eval_interval = 40
        config.train.eval_episodes = 1
        config.train.held_out_eval_interval = 0
        config.train.checkpoint_interval = 40
        config.sac.network.hidden_sizes = (16,)
        run = train_from_config(config).run_dir

        assert main(["compare", str(run), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload[0]["held_progress"] is None
        assert payload[0]["gap"] is None

    def test_warns_about_the_simulated_driver(self, two_runs, capsys):
        main(["compare", *two_runs])
        assert "SIMULATED" in capsys.readouterr().out

    def test_skips_non_run_directories(self, two_runs, tmp_path, capsys):
        assert main(["compare", two_runs[0], str(tmp_path / "nope")]) == 0
        assert "run-a" in capsys.readouterr().out

    def test_nothing_to_compare(self, tmp_path, capsys):
        assert main(["compare", str(tmp_path / "nope")]) == 1


# -- parser -------------------------------------------------------------------------


class TestValidateConfigCLI:
    def test_reports_effective_dimension_after_temporal_stacking(self, capsys):
        assert main(["validate-config", "-c", "tmai/configs/pipeline_smoke.yaml"]) == 0
        assert "observation dim 60" in capsys.readouterr().out


class TestParser:
    def test_every_new_command_is_registered(self):
        parser = build_parser()
        choices = parser._subparsers._group_actions[0].choices  # noqa: SLF001
        for name in ("status", "list-tracks", "validate-config", "compare", "analyze", "play"):
            assert name in choices

    def test_every_command_has_a_handler(self):
        parser = build_parser()
        choices = parser._subparsers._group_actions[0].choices  # noqa: SLF001
        for name, sub in choices.items():
            args = sub.parse_args(_minimal_args(name))
            assert callable(getattr(args, "func", None)), f"{name} has no handler"

    def test_benchmark_seed_and_track_family_options_parse(self):
        parser = build_parser()
        benchmark = parser.parse_args(
            ["benchmark", "--model", "pilot=baseline:curvature", "--seed-repeats", "5"]
        )
        assert benchmark.seed_repeats == 5
        record = parser.parse_args(["record-track", "--out", "track.json", "--family", "author-pack"])
        assert record.family == "author-pack"


def _minimal_args(command: str) -> list[str]:
    """The fewest arguments each subcommand needs to parse."""
    required = {
        "record-track": ["--out", "x.json"],
        "list-tracks": ["some/dir"],
        "compare": ["some/run"],
        "record-demo": ["--out", "x.jsonl"],
        "pretrain": ["--demo", "x.jsonl", "--out", "x.pt"],
        "models": ["list"],
        "replay": ["list"],
        "analyze": ["some/run", "--track", "some/track.json"],
    }
    return required.get(command, [])


class TestEvalCoversTheLibrary:
    """A regression guard for `tmai eval` reporting a single track.

    Before this was fixed the command built one env from the first train track, so evaluating a
    multi-track run silently reported one map and threw away the per-track comparison that is the
    entire point of evaluation.
    """

    def test_eval_reports_every_track_in_a_split(self, tmp_path, capsys):
        out = tmp_path / "out"
        run = self._train(out)
        main(["eval", "--checkpoint", str(run), "--episodes", "1", "--split", "train"])
        text = capsys.readouterr().out
        # Both train-split tracks must appear by name in the table.
        assert "straight" in text
        assert "oval" in text
        assert "eval:train" in text
        # And it must not silently report only one track.
        assert "2 tracks" in text

    def test_eval_can_compare_train_and_validation(self, tmp_path, capsys):
        out = tmp_path / "out"
        run = self._train(out)
        main(
            [
                "eval",
                "--checkpoint",
                str(run),
                "--episodes",
                "1",
                "--split",
                "train,validation",
            ]
        )
        text = capsys.readouterr().out
        assert "eval:train" in text
        assert "eval:validation" in text
        assert "s_curve" in text

    def test_eval_skips_an_empty_split_with_a_message(self, tmp_path, capsys):
        out = tmp_path / "out"
        run = self._train(out)
        # "test" is not in the smoke suite's splits, so it must be reported, not crash.
        main(["eval", "--checkpoint", str(run), "--episodes", "1", "--split", "test,train"])
        captured = capsys.readouterr()
        assert "empty" in captured.err
        assert "eval:train" in captured.out

    def test_eval_json_contains_one_entry_per_split(self, tmp_path):
        out = tmp_path / "out"
        run = self._train(out)
        json_out = tmp_path / "eval.json"
        main(
            [
                "eval",
                "--checkpoint",
                str(run),
                "--episodes",
                "1",
                "--split",
                "train,validation",
                "--json-out",
                str(json_out),
            ]
        )
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        assert isinstance(payload, list)
        assert [p["label"] for p in payload] == ["eval:train", "eval:validation"]
        for entry in payload:
            assert entry["num_tracks"] == 2
            assert entry["tracks"]

    @staticmethod
    def _train(out):
        """Train the multitrack smoke config and return its run directory."""
        import tmai
        from tmai.config import RunConfig

        source = Path(tmai.__file__).parent / "configs" / "multitrack_smoke.yaml"
        cfg = RunConfig.from_yaml(str(source))
        cfg.train.output_dir = str(out)
        cfg.train.total_steps = 40
        cfg.train.warmup_steps = 5
        cfg.train.eval_interval = 0
        cfg.train.held_out_eval_interval = 0
        cfg.train.checkpoint_interval = 40
        cfg.train.log_interval = 20
        from tmai.training.trainer import train_from_config

        return train_from_config(cfg).run_dir


class TestConfigDiscoveryForResumeAndEval:
    """A run directory says exactly how it was configured, so pointing a command at one is also
    a request to use that configuration.

    `eval --checkpoint` did this from the start. `train --resume` did not, so the documented
    `tmai train --resume runs/<run>` fell back to default.yaml and failed with "no track
    configured" even though the run's own config.yaml was sitting right there.
    """

    def test_load_config_finds_the_saved_config_for_checkpoint(self, two_runs):
        from tmai.cli import _run_dir_config

        assert _run_dir_config(two_runs[0]) is not None

    def test_load_config_finds_the_saved_config_for_resume(self, two_runs):
        """The regression guard: `--resume` must discover the config just like `--checkpoint`."""
        import argparse

        from tmai.cli import _load_config

        # An args namespace carrying only `resume`, as `tmai train --resume <run>` produces.
        args = argparse.Namespace(config=None, resume=two_runs[0], set=None)
        config = _load_config(args)

        # The smoke runs configure a four-track synthetic suite; default.yaml does not, so a
        # non-empty suite here proves the run's config.yaml was loaded.
        assert config.track.synthetic_suite, "the run's own track configuration must be used"

    def test_resume_discovers_config_from_a_checkpoint_file_path(self, two_runs):
        """A `.pt` path lives inside the run directory, so discovery must walk up from it."""
        import argparse

        from tmai.cli import _load_config
        from tmai.training.checkpoint import latest_checkpoint

        ckpt = latest_checkpoint(two_runs[0])
        assert ckpt is not None, "precondition: the fixture run has a checkpoint"

        args = argparse.Namespace(config=None, resume=str(ckpt), set=None)
        assert _load_config(args).track.synthetic_suite

    def test_explicit_config_wins_over_discovery(self, two_runs, tmp_path):
        """An operator who passes -c means it; discovery must not override an explicit choice."""
        import argparse

        from tmai.cli import _load_config

        explicit = tmp_path / "explicit.yaml"
        explicit.write_text("track:\n  synthetic: oval\n", encoding="utf-8")

        args = argparse.Namespace(config=str(explicit), resume=two_runs[0], set=None)
        config = _load_config(args)
        assert config.track.synthetic == "oval"
        assert not config.track.synthetic_suite

    def test_no_run_and_no_config_falls_back_to_defaults(self):
        import argparse

        from tmai.cli import _load_config

        args = argparse.Namespace(config=None, resume=None, checkpoint=None, set=None)
        config = _load_config(args)
        assert not config.track.synthetic
        assert not config.track.synthetic_suite
        assert not config.track.path
        assert not config.track.directory

    def test_resume_actually_continues_a_run(self, tmp_path):
        """End to end: the documented command form must work, and must restore state."""
        import json

        from tmai.config import RunConfig
        from tmai.runlog import read_manifest
        from tmai.training.trainer import train_from_config

        config = RunConfig()
        config.driver.kind = "simulated"
        config.driver.allow_simulated = True
        config.track.synthetic_suite = [{"name": "straight"}, {"name": "oval"}]
        config.train.output_dir = str(tmp_path / "first")
        config.train.run_name = "resume-e2e"
        config.train.total_steps = 30
        config.train.warmup_steps = 5
        config.train.batch_size = 8
        config.train.log_interval = 15
        config.train.eval_interval = 0
        config.train.held_out_eval_interval = 0
        config.train.checkpoint_interval = 30
        config.sac.network.hidden_sizes = (16,)
        first = train_from_config(config).run_dir

        # The command under test: resume with no -c, exactly as documented.
        assert (
            main(
                [
                    "train",
                    "--resume",
                    str(first),
                    "--set",
                    "train.total_steps=60",
                    "--set",
                    "train.checkpoint_interval=30",
                    "--set",
                    f"train.output_dir={tmp_path / 'second'}",
                ]
            )
            == 0
        )

        second = sorted(p for p in (tmp_path / "second").iterdir() if p.is_dir())[0]
        manifest = read_manifest(second)
        assert manifest["resumed_from"] == str(first)

        events = [
            json.loads(line)
            for line in (second / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        resumed = [e for e in events if e.get("event") == "resumed"]
        assert resumed, "the resumed run must record a `resumed` event"
        # State really was restored rather than restarted from zero.
        assert resumed[0]["step"] == 30
        assert resumed[0]["gradient_steps"] > 0


class TestDoctorCalibrateIsNotSilent:
    """`--calibrate` needs the real game, so against the simulated driver it is skipped.

    Skipping is correct. Skipping *silently* was not: the operator asked for a measurement and
    got no output at all, with exit 0, which reads exactly like a calibration that found nothing
    wrong.
    """

    def test_skipped_calibration_says_so(self, capsys):
        import tmai

        smoke = Path(tmai.__file__).parent / "configs" / "smoke.yaml"
        assert main(["doctor", "-c", str(smoke), "--calibrate", "--calibrate-steps", "5"]) == 0
        err = capsys.readouterr().err
        assert "calibration skipped" in err
        assert "needs the real game" in err

    def test_without_the_flag_nothing_is_claimed(self, capsys):
        """Guard against the message being printed unconditionally."""
        import tmai

        smoke = Path(tmai.__file__).parent / "configs" / "smoke.yaml"
        assert main(["doctor", "-c", str(smoke)]) == 0
        captured = capsys.readouterr()
        assert "calibration skipped" not in captured.err
        assert "calibration skipped" not in captured.out

    def test_doctor_still_reports_the_simulated_driver_as_a_warning(self, capsys):
        import tmai

        smoke = Path(tmai.__file__).parent / "configs" / "smoke.yaml"
        assert main(["doctor", "-c", str(smoke)]) == 0
        out = capsys.readouterr().out
        assert "simulated (NOT the real game)" in out

    def test_calibration_can_run_before_a_centreline_is_recorded(self, tmp_path, monkeypatch, capsys):
        """Doctor must connect without track geometry so the operator can calibrate first."""
        from types import SimpleNamespace

        import tmai.game.calibration as calibration
        import tmai.training.factory as factory

        cfg = tmp_path / "unrecorded.yaml"
        cfg.write_text("driver:\n  kind: tminterface\ntrack:\n  path: missing.json\n")
        calls = []

        class FakeDriver:
            name = "mock-game"

            def open(self):
                calls.append("open")

            def describe(self):
                return {"checkpoint_total": 2}

            def close(self):
                calls.append("close")

        def fake_build_driver(config, track):
            assert track is None
            return FakeDriver()

        def fake_calibrate(driver, **kwargs):
            calls.append("calibrate")
            assert kwargs["steps"] == 5
            return SimpleNamespace(ok=True, format=lambda: "calibration ok")

        monkeypatch.setattr(factory, "build_driver", fake_build_driver)
        monkeypatch.setattr(calibration, "calibrate_driver", fake_calibrate)
        assert main(["doctor", "-c", str(cfg), "--calibrate", "--calibrate-steps", "5"]) == 0
        assert calls == ["open", "calibrate", "close"]
        assert "calibration ok" in capsys.readouterr().out

    def test_failed_driver_does_not_send_calibration_into_a_closed_driver(self, tmp_path, capsys):
        """A driver whose open() raised must be treated as absent, not as connected.

        On a non-Windows host TMInterface cannot open, so `doctor` reports the driver as a
        failure. The `--calibrate` path decides whether to skip by testing whether a driver was
        obtained; because the failed driver stayed bound, calibration ran against it and
        reported a secondary "open() has not been called" error over the real cause. The
        operator must see the actionable explanation, not the knock-on one.
        """
        import tmai

        smoke = Path(tmai.__file__).parent / "configs" / "smoke.yaml"
        cfg = tmp_path / "realgame.yaml"
        cfg.write_text(smoke.read_text().replace("kind: simulated", "kind: tminterface"))
        assert "kind: tminterface" in cfg.read_text()

        # Only meaningful on a host where TMInterface cannot connect; on the real game host this
        # would legitimately succeed, so the assertion is scoped to that case.
        if sys.platform == "win32":  # pragma: no cover - the real game host
            pytest.skip("TMInterface can connect here, so the failure path is unreachable")

        rc = main(["doctor", "-c", str(cfg), "--calibrate", "--calibrate-steps", "5"])
        captured = capsys.readouterr()
        assert rc == 1, "a driver that failed to open must not exit 0"
        assert "calibration skipped" in captured.err
        assert "open() has not been called" not in captured.err
        assert "open() has not been called" not in captured.out


class TestStatusGapColumnIsPopulated:
    """The `gap` column in `tmai status` used to be permanently blank.

    Training and held-out runs are logged as separate single-split reports, so neither carries a
    `generalization_gap` of its own -- reading it per row always yielded None, and the one column
    that shows whether a run is overfitting printed `--` forever.
    """

    def _run(self, tmp_path):
        from tmai.config import RunConfig
        from tmai.training.trainer import train_from_config

        config = RunConfig()
        config.driver.kind = "simulated"
        config.driver.allow_simulated = True
        config.track.synthetic_suite = [
            {"name": "straight"},
            {"name": "oval"},
            {"name": "s_curve"},
            {"name": "figure_eight"},
        ]
        config.track.split_weights = {"train": 0.5, "validation": 0.5, "test": 0.0}
        config.train.output_dir = str(tmp_path)
        config.train.run_name = "gap-status"
        config.train.total_steps = 60
        config.train.warmup_steps = 10
        config.train.batch_size = 8
        config.train.log_interval = 30
        config.train.eval_interval = 60
        config.train.eval_episodes = 1
        config.train.held_out_eval_interval = 60
        config.train.held_out_eval_episodes = 1
        config.train.checkpoint_interval = 60
        config.sac.network.hidden_sizes = (16,)
        return train_from_config(config).run_dir

    def test_gap_is_printed_for_the_training_row(self, tmp_path, capsys):
        run = self._run(tmp_path)
        assert main(["status", "--run", str(run)]) == 0
        out = capsys.readouterr().out

        table = out.split("evaluations")[-1]
        rows = [line for line in table.splitlines() if "training" in line]
        assert rows, "precondition: a training evaluation row exists"
        # The gap must be a signed number, not the '--' placeholder.
        assert any("--" not in row.split()[-1] for row in rows), f"gap column still blank: {rows}"

    def test_gap_is_not_duplicated_across_both_rows(self, tmp_path, capsys):
        """Printing the same gap on both rows would read as two independent measurements."""
        run = self._run(tmp_path)
        assert main(["status", "--run", str(run)]) == 0
        out = capsys.readouterr().out

        table = out.split("evaluations")[-1]
        held = [line for line in table.splitlines() if "held_out" in line]
        assert held
        for row in held:
            assert row.split()[-1] == "--", f"held-out row should not repeat the gap: {row}"


class TestRecordDemoCli:
    def test_simulated_pipeline_recording_resolves_library_track(self, tmp_path, capsys):
        from tmai.training.demos import Demonstration

        output = tmp_path / "demo.jsonl"
        assert (
            main(
                [
                    "record-demo",
                    "-c",
                    "tmai/configs/smoke.yaml",
                    "--out",
                    str(output),
                    "--max-steps",
                    "3",
                    "--allow-simulated-driver",
                ]
            )
            == 0
        )
        demo = Demonstration.load(output)
        # The initial reset frame plus three control transitions are recorded.
        assert len(demo) == 4
        assert demo.metadata["steps"] == 3
        assert demo.metadata["track"] == "s_curve"
        assert demo.metadata["not_human_driving"] is True
        assert "SIMULATED PIPELINE TEST" in capsys.readouterr().out


class TestReplayShowCli:
    def test_show_renders_replay_trajectory(self, tmp_path, capsys):
        import numpy as np

        from tmai.replay import EpisodeReplay, ReplayStore
        from tmai.tracks.synthetic import straight

        run_dir = tmp_path / "run"
        store = ReplayStore(run_dir / "replays")
        replay = EpisodeReplay(
            episode=1,
            step=5,
            track="straight",
            positions=np.array([straight().point_at(station) for station in (0.0, 10.0, 20.0)]),
        )
        store.save(replay)
        track_path = tmp_path / "straight.json"
        straight().save(track_path)
        output = tmp_path / "replay.png"

        assert (
            main(
                [
                    "replay",
                    "show",
                    "--run",
                    str(run_dir),
                    "--replay",
                    "episode_000001.json",
                    "--track",
                    str(track_path),
                    "--out",
                    str(output),
                ]
            )
            == 0
        )
        assert output.is_file()
        assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert "wrote" in capsys.readouterr().out


class TestReplayCompareCli:
    """`tmai replay compare` accepts a human demonstration (JSONL) as the ghost."""

    def _run_with_replay(self, tmp_path):

        from tmai.config import RunConfig
        from tmai.replay import ReplayStore
        from tmai.training.trainer import train_from_config

        config = RunConfig()
        config.driver.kind = "simulated"
        config.driver.allow_simulated = True
        config.track.synthetic = "straight"
        # Short episodes so the run finishes episodes (and thus records replays) quickly.
        config.env.termination.max_steps = 10
        config.train.output_dir = str(tmp_path / "runs")
        config.train.run_name = "replay-cli"
        config.train.total_steps = 30
        config.train.warmup_steps = 5
        config.train.batch_size = 8
        config.train.log_interval = 15
        config.train.eval_interval = 0
        config.train.held_out_eval_interval = 0
        config.train.checkpoint_interval = 0
        config.train.record_replays = True
        config.train.replay_decimation = 1
        config.train.max_replays = 5
        config.sac.network.hidden_sizes = (16,)
        run_dir = train_from_config(config).run_dir
        store = ReplayStore(run_dir / "replays")
        rows = store.list()
        assert rows, "the run must have recorded replays"
        return run_dir, rows[0]["name"]

    def test_compare_against_a_demonstration_ghost(self, tmp_path, capsys):
        import numpy as np

        from tmai.training.demos import Demonstration

        run_dir, replay_name = self._run_with_replay(tmp_path)
        demo_path = tmp_path / "human.jsonl"
        n = 30
        Demonstration(
            observations=np.zeros((n, 4), dtype=np.float32),
            actions=np.zeros((n, 3), dtype=np.float32),
            positions=np.stack([np.zeros(n), np.zeros(n), np.arange(n, dtype=float)], axis=1),
            speeds=np.full(n, 20.0),
            rewards=np.zeros(n),
            race_times=np.arange(n, dtype=float) * 0.05,
            metadata={"track": "straight", "finished": True},
        ).save(demo_path)

        # No --track: the track is resolved from the replay's own name (synthetic suite).
        assert (
            main(
                [
                    "replay",
                    "compare",
                    "--run",
                    str(run_dir),
                    "--replay",
                    replay_name,
                    "--other",
                    str(demo_path),
                    "--out",
                    str(tmp_path / "gaps.json"),
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "AI replay vs ghost" in out
        assert "mean segment gap" in out
        gaps = json.loads((tmp_path / "gaps.json").read_text())
        assert gaps["ghost_finished"] is True
        assert gaps["segment_gaps"]

    def test_compare_rejects_a_missing_ghost(self, tmp_path, capsys):
        run_dir, replay_name = self._run_with_replay(tmp_path)
        assert (
            main(
                [
                    "replay",
                    "compare",
                    "--run",
                    str(run_dir),
                    "--replay",
                    replay_name,
                    "--other",
                    str(tmp_path / "nope.jsonl"),
                ]
            )
            == 1
        )


class TestPretrainCli:
    """`tmai pretrain` produces a resumable checkpoint from demonstrations."""

    def test_pretrain_writes_a_checkpoint(self, tmp_path, capsys):
        import numpy as np

        from tmai.config import RunConfig
        from tmai.training.checkpoint import load_checkpoint
        from tmai.training.demos import Demonstration
        from tmai.training.factory import build_learner, build_library, build_multi_track_env

        config = RunConfig.from_yaml("tmai/configs/pipeline_smoke.yaml")
        library = build_library(config)
        env = build_multi_track_env(config, library, split="train", seed=0)
        try:
            dim = build_learner(env, config).observation_dim
        finally:
            env.close()

        demo_path = tmp_path / "demo.jsonl"
        Demonstration(
            observations=np.random.default_rng(0).normal(size=(64, dim)).astype(np.float32),
            actions=np.tile([0.1, 0.8, 0.0], (64, 1)).astype(np.float32),
        ).save(demo_path)

        out = tmp_path / "pretrained.pt"
        assert (
            main(
                [
                    "pretrain",
                    "-c",
                    "tmai/configs/pipeline_smoke.yaml",
                    "--demo",
                    str(demo_path),
                    "--out",
                    str(out),
                    "--epochs",
                    "2",
                ]
            )
            == 0
        )
        assert out.is_file()
        payload = load_checkpoint(out)
        assert payload["extra"]["reason"] == "bc_pretrain"
        assert payload["learner"]
        assert "bc/final_train_loss" in capsys.readouterr().out

    def test_pretrain_honours_the_device_flag(self, tmp_path):
        import numpy as np

        from tmai.config import RunConfig
        from tmai.training.checkpoint import load_checkpoint
        from tmai.training.demos import Demonstration

        demo_path = tmp_path / "demo.jsonl"
        # The observation layout comes from the config, so size the demo from the real learner.
        from tmai.training.factory import build_learner, build_library, build_multi_track_env

        config = RunConfig.from_yaml("tmai/configs/pipeline_smoke.yaml")
        env = build_multi_track_env(config, build_library(config), split="train", seed=0)
        try:
            dim = build_learner(env, config).observation_dim
        finally:
            env.close()
        Demonstration(
            observations=np.random.default_rng(0).normal(size=(64, dim)).astype(np.float32),
            actions=np.tile([0.1, 0.8, 0.0], (64, 1)).astype(np.float32),
        ).save(demo_path)

        out = tmp_path / "pretrained.pt"
        assert (
            main(
                [
                    "pretrain",
                    "-c",
                    "tmai/configs/pipeline_smoke.yaml",
                    "--demo",
                    str(demo_path),
                    "--out",
                    str(out),
                    "--epochs",
                    "1",
                    "--device",
                    "cpu",
                ]
            )
            == 0
        )
        # The override is what the checkpoint records as the device it was trained on.
        assert load_checkpoint(out)["config"]["train"]["device"] == "cpu"

    def test_pretrain_rejects_a_dimension_mismatch(self, tmp_path, capsys):
        import numpy as np

        from tmai.training.demos import Demonstration

        demo_path = tmp_path / "demo.jsonl"
        Demonstration(
            observations=np.zeros((16, 999), dtype=np.float32),
            actions=np.zeros((16, 3), dtype=np.float32),
        ).save(demo_path)
        assert (
            main(
                [
                    "pretrain",
                    "-c",
                    "tmai/configs/pipeline_smoke.yaml",
                    "--demo",
                    str(demo_path),
                    "--out",
                    str(tmp_path / "x.pt"),
                ]
            )
            == 1
        )


class TestPlayCli:
    """The direct-drive loop works offline and does not masquerade as real game control."""

    @staticmethod
    def _write_config(path: Path, *, simulated: bool = True) -> Path:
        from tmai.config import RunConfig

        config = RunConfig()
        config.driver.kind = "simulated" if simulated else "tminterface"
        config.driver.allow_simulated = False
        config.track.synthetic = "straight" if simulated else None
        config.track.synthetic_kwargs = {"length": 100.0}
        config.track.directory = None
        config.multi.random_start_station = False
        config.multi.start_lateral_std = 0.0
        config.env.termination.max_steps = 4
        config.train.output_dir = str(path.parent / "runs")
        path.write_text(config.to_yaml(), encoding="utf-8")
        return path

    def test_curvature_pilot_drives_simulator_and_records_replay(self, tmp_path, capsys):
        config_path = self._write_config(tmp_path / "play.yaml")
        assert (
            main(
                [
                    "play",
                    "-c",
                    str(config_path),
                    "--allow-simulated-driver",
                    "--record-replay",
                    "--log-interval",
                    "1",
                ]
            )
            == 0
        )
        output = capsys.readouterr().out
        assert "SIMULATED DRIVER" in output
        assert "CurvaturePilot heuristic (not a trained policy)" in output
        assert "Episode result:" in output
        assert "cmd=(" in output
        replays = list((tmp_path / "runs" / "play-replays").rglob("episode_*.json"))
        assert len(replays) == 1
        assert json.loads(replays[0].read_text(encoding="utf-8"))["source"] == "play"

    def test_simulated_driver_requires_explicit_opt_in(self, tmp_path, capsys):
        config_path = self._write_config(tmp_path / "play.yaml")
        assert main(["play", "-c", str(config_path)]) == 2
        assert "allow_simulated=true" in capsys.readouterr().err

    def test_real_direct_drive_refuses_a_multi_track_library(self, tmp_path, capsys):
        from tmai.tracks.synthetic import build_synthetic

        config_path = self._write_config(tmp_path / "play.yaml", simulated=False)
        tracks = tmp_path / "tracks"
        tracks.mkdir()
        for name in ("oval", "s_curve"):
            build_synthetic(name).save(tracks / f"{name}.json")
        from tmai.config import RunConfig

        config = RunConfig.from_yaml(config_path)
        config.track.path = None
        config.track.synthetic = None
        config.track.directory = str(tracks)
        config_path.write_text(config.to_yaml(), encoding="utf-8")

        assert main(["play", "-c", str(config_path)]) == 2
        assert "requires exactly one configured track" in capsys.readouterr().err

    def test_interrupt_closes_the_environment(self, tmp_path, monkeypatch, capsys):
        from tmai.env.tm_env import TrackmaniaEnv
        from tmai.training import evaluate

        config_path = self._write_config(tmp_path / "play.yaml")
        closed = []
        original_close = TrackmaniaEnv.close

        def tracked_close(env):
            closed.append(True)
            original_close(env)

        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(TrackmaniaEnv, "close", tracked_close)
        monkeypatch.setattr(evaluate, "run_episode", interrupt)
        assert (
            main(["play", "-c", str(config_path), "--allow-simulated-driver"]) == 130
        )
        assert closed == [True]
        assert "interrupted" in capsys.readouterr().err

    def test_checkpoint_drives_through_the_same_play_loop(self, tmp_path, capsys):
        from tmai.config import RunConfig
        from tmai.training.checkpoint import save_checkpoint
        from tmai.training.factory import build_learner, build_library, build_multi_track_env

        config_path = self._write_config(tmp_path / "play.yaml")
        config = RunConfig.from_yaml(config_path)
        config.driver.allow_simulated = True
        config_path.write_text(config.to_yaml(), encoding="utf-8")
        library = build_library(config)
        env = build_multi_track_env(config, library, split="train", seed=0)
        try:
            learner = build_learner(env, config)
            checkpoint_dir = tmp_path / "saved-run"
            checkpoint_dir.mkdir()
            checkpoint = save_checkpoint(
                checkpoint_dir,
                step=17,
                learner=learner,
                config=config.to_dict(),
                keep=1,
                rng_state=False,
            )
        finally:
            env.close()

        assert (
            main(
                [
                    "play",
                    "-c",
                    str(config_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--allow-simulated-driver",
                    "--max-steps",
                    "3",
                ]
            )
            == 0
        )
        output = capsys.readouterr().out
        assert "checkpoint" in output
        assert "step 17" in output
        assert "Episode result:" in output
