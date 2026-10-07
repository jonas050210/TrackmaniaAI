"""Tests for the CLI commands added for generalisation, inspection and comparison."""

from __future__ import annotations

import json
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


class TestParser:
    def test_every_new_command_is_registered(self):
        parser = build_parser()
        choices = parser._subparsers._group_actions[0].choices  # noqa: SLF001
        for name in ("status", "list-tracks", "validate-config", "compare"):
            assert name in choices

    def test_every_command_has_a_handler(self):
        parser = build_parser()
        choices = parser._subparsers._group_actions[0].choices  # noqa: SLF001
        for name, sub in choices.items():
            args = sub.parse_args(_minimal_args(name))
            assert callable(getattr(args, "func", None)), f"{name} has no handler"


def _minimal_args(command: str) -> list[str]:
    """The fewest arguments each subcommand needs to parse."""
    required = {
        "record-track": ["--out", "x.json"],
        "list-tracks": ["some/dir"],
        "compare": ["some/run"],
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
                "--checkpoint", str(run),
                "--episodes", "1",
                "--split", "train,validation",
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
                "--checkpoint", str(run),
                "--episodes", "1",
                "--split", "train,validation",
                "--json-out", str(json_out),
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
