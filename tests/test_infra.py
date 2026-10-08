"""Tests for configuration management, run logging and checkpointing."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from tmai.agents.sac import SACConfig, SACLearner
from tmai.config import DriverSpec, RunConfig, parse_overrides
from tmai.models.networks import NetworkConfig
from tmai.runlog import (
    RunLogger,
    make_run_dir,
    read_manifest,
    read_metrics,
)
from tmai.training.checkpoint import (
    CheckpointError,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    read_meta,
    restore_rng,
    save_best,
    save_checkpoint,
    write_index,
)

# -- config ---------------------------------------------------------------------------


class TestConfigRoundtrip:
    def test_yaml_roundtrip_preserves_values(self, tmp_path):
        config = RunConfig()
        config.train.total_steps = 12345
        config.sac.gamma = 0.97
        config.sac.network.hidden_sizes = (128, 64)
        config.env.reward.progress_weight = 2.5
        config.driver.position_scale = 0.125
        config.driver.forward_axis = 2
        config.driver.forward_sign = -1.0
        path = config.save(tmp_path / "cfg.yaml")
        loaded = RunConfig.from_yaml(path)
        assert loaded.train.total_steps == 12345
        assert loaded.sac.gamma == pytest.approx(0.97)
        assert tuple(loaded.sac.network.hidden_sizes) == (128, 64)
        assert loaded.env.reward.progress_weight == pytest.approx(2.5)
        assert loaded.driver.position_scale == pytest.approx(0.125)
        assert loaded.driver.forward_axis == 2
        assert loaded.driver.forward_sign == -1.0

    def test_shipped_default_config_loads(self):
        from pathlib import Path

        import tmai

        path = Path(tmai.__file__).parent / "configs" / "default.yaml"
        config = RunConfig.from_yaml(path)
        assert config.driver.kind == "tminterface"
        assert config.driver.allow_simulated is False
        assert config.train.total_steps > 0

    def test_shipped_smoke_config_loads(self):
        from pathlib import Path

        import tmai

        path = Path(tmai.__file__).parent / "configs" / "smoke.yaml"
        config = RunConfig.from_yaml(path)
        assert config.driver.kind == "simulated"
        assert config.driver.allow_simulated is True
        assert config.track.synthetic == "s_curve"

    def test_shipped_configs_are_distinct(self):
        from pathlib import Path

        import tmai

        base = Path(tmai.__file__).parent / "configs"
        default = RunConfig.from_yaml(base / "default.yaml")
        smoke = RunConfig.from_yaml(base / "smoke.yaml")
        assert default.to_dict() != smoke.to_dict()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RunConfig.from_yaml(tmp_path / "nope.yaml")

    def test_non_mapping_root_rejected(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            RunConfig.from_yaml(path)

    def test_unknown_keys_are_rejected_by_default(self, tmp_path):
        path = tmp_path / "cfg.yaml"
        path.write_text(
            yaml.safe_dump({"train": {"total_steps": 10, "not_a_real_field": True}}),
            encoding="utf-8",
        )
        # A silently ignored key is how a typo'd step count became a 100,000-step run.
        with pytest.raises(ValueError, match="train.not_a_real_field"):
            RunConfig.from_yaml(path)

    def test_unknown_keys_are_ignored_with_warning_when_lenient(self, tmp_path, caplog):
        path = tmp_path / "cfg.yaml"
        path.write_text(
            yaml.safe_dump({"train": {"total_steps": 10, "not_a_real_field": True}}),
            encoding="utf-8",
        )
        with caplog.at_level("WARNING"):
            config = RunConfig.from_yaml(path, strict=False)
        assert config.train.total_steps == 10
        assert "not_a_real_field" in caplog.text

    def test_nested_dataclasses_are_constructed(self):
        config = RunConfig.from_dict(
            {
                "driver": {"kind": "tminterface", "server_name": "TMInterface3"},
                "sac": {"gamma": 0.95, "network": {"hidden_sizes": [16]}},
            }
        )
        assert isinstance(config.driver, DriverSpec)
        assert config.driver.server_name == "TMInterface3"
        assert isinstance(config.sac, SACConfig)
        assert tuple(config.sac.network.hidden_sizes) == (16,)

    def test_to_dict_is_json_serialisable(self):
        payload = json.dumps(RunConfig().to_dict())
        assert "tminterface" in payload

    def test_invalid_forward_convention_is_reported(self):
        config = RunConfig()
        config.driver.forward_axis = 3
        config.driver.forward_sign = 0.0
        problems = config.validate()
        assert any("driver.forward_axis" in problem for problem in problems)
        assert any("driver.forward_sign" in problem for problem in problems)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("speed_ratio", float("nan")),
            ("position_scale", float("inf")),
            ("physics_hz", 0.0),
            ("connect_timeout_s", -1.0),
            ("frame_timeout_s", float("nan")),
        ],
    )
    def test_invalid_driver_timing_and_scale_values_are_reported(self, field, value):
        config = RunConfig()
        setattr(config.driver, field, value)
        assert any(f"driver.{field}" in problem for problem in config.validate())

    def test_zero_settle_ticks_is_rejected(self):
        config = RunConfig()
        config.driver.settle_ticks = 0
        assert any("driver.settle_ticks" in problem for problem in config.validate())

    @pytest.mark.parametrize(
        ("section", "field", "value", "problem_name"),
        [
            ("env", "control_dt", float("nan"), "env.control_dt"),
            ("sac", "gamma", float("nan"), "sac.gamma"),
            ("track", "split_weights", float("nan"), "track.split_weights"),
        ],
    )
    def test_nonfinite_control_and_training_values_are_rejected(
        self, section, field, value, problem_name
    ):
        config = RunConfig()
        if field == "split_weights":
            config.track.split_weights["train"] = value
        else:
            setattr(getattr(config, section), field, value)
        assert any(problem_name in problem for problem in config.validate())


class TestOverrides:
    def test_scalar_override(self):
        config = RunConfig().apply_overrides({"sac.gamma": 0.5})
        assert config.sac.gamma == pytest.approx(0.5)

    def test_original_is_not_mutated(self):
        original = RunConfig()
        before = original.sac.gamma
        original.apply_overrides({"sac.gamma": 0.1})
        assert original.sac.gamma == before

    def test_int_coercion_from_string(self):
        config = RunConfig().apply_overrides({"train.total_steps": "500"})
        assert config.train.total_steps == 500

    def test_float_coercion_from_string(self):
        config = RunConfig().apply_overrides({"sac.tau": "0.01"})
        assert config.sac.tau == pytest.approx(0.01)

    def test_bool_coercion_from_string(self):
        assert RunConfig().apply_overrides({"sac.learnable_temperature": "false"}).sac.learnable_temperature is False
        assert RunConfig().apply_overrides({"driver.allow_simulated": "true"}).driver.allow_simulated is True

    def test_none_override(self):
        config = RunConfig().apply_overrides({"train.resume": None})
        assert config.train.resume is None

    def test_unknown_path_raises(self):
        with pytest.raises(KeyError, match="unknown config path"):
            RunConfig().apply_overrides({"sac.not_a_field": 1})

    def test_unknown_section_raises(self):
        with pytest.raises(KeyError, match="unknown config path"):
            RunConfig().apply_overrides({"nope.gamma": 1})

    def test_bad_bool_string_raises(self):
        with pytest.raises(ValueError, match="boolean"):
            RunConfig().apply_overrides({"sac.learnable_temperature": "maybe"})

    def test_parse_overrides(self):
        parsed = parse_overrides(["sac.gamma=0.9", "train.total_steps=100", "driver.kind=simulated"])
        assert parsed == {"sac.gamma": 0.9, "train.total_steps": 100, "driver.kind": "simulated"}

    def test_parse_overrides_rejects_malformed(self):
        with pytest.raises(ValueError, match="key=value"):
            parse_overrides(["sac.gamma"])

    def test_parse_overrides_empty(self):
        assert parse_overrides(None) == {}
        assert parse_overrides([]) == {}


# -- run logging ----------------------------------------------------------------------


class TestRunLogger:
    def test_creates_expected_files(self, tmp_path):
        with RunLogger(tmp_path / "run", run_name="t", config={"a": 1}) as logger:
            logger.log_metrics(1, {"loss": 0.5})
        run = tmp_path / "run"
        assert (run / "manifest.json").exists()
        assert (run / "metrics.jsonl").exists()
        assert (run / "events.jsonl").exists()
        assert (run / "config.yaml").exists()
        assert (run / "run.log").exists()

    def test_manifest_records_config_and_environment(self, tmp_path):
        with RunLogger(tmp_path / "run", run_name="t", config={"a": 1}, seed=7) as logger:
            logger.log_metrics(1, {"x": 1.0})
        manifest = read_manifest(tmp_path / "run")
        assert manifest["run_name"] == "t"
        assert manifest["seed"] == 7
        assert manifest["config"] == {"a": 1}
        assert manifest["versions"]["python"]
        assert "hostname" in manifest["host"]

    def test_metrics_are_readable_back(self, tmp_path):
        with RunLogger(tmp_path / "run") as logger:
            logger.log_metrics(1, {"loss": 0.5})
            logger.log_metrics(2, {"loss": 0.25, "extra": "text"})
        records = read_metrics(tmp_path / "run")
        assert len(records) == 2
        assert records[0]["step"] == 1
        assert records[0]["loss"] == pytest.approx(0.5)
        assert records[1]["extra"] == "text"
        assert "t" in records[0]

    def test_numpy_values_are_coerced(self, tmp_path):
        with RunLogger(tmp_path / "run") as logger:
            logger.log_metrics(
                1,
                {
                    "np_float": np.float32(1.5),
                    "np_array": np.array([1.0, 2.0]),
                    "np_int": np.int64(3),
                },
            )
        record = read_metrics(tmp_path / "run")[0]
        assert record["np_float"] == pytest.approx(1.5)
        assert record["np_array"] == [1.0, 2.0]
        assert record["np_int"] == 3

    def test_metrics_are_flushed_before_close(self, tmp_path):
        logger = RunLogger(tmp_path / "run")
        logger.log_metrics(1, {"loss": 1.0})
        # Readable without close(), which is what makes a crash survivable.
        assert len(read_metrics(tmp_path / "run")) == 1
        logger.close()

    def test_events_are_recorded(self, tmp_path):
        with RunLogger(tmp_path / "run") as logger:
            logger.log_event("episode_end", step=10, reward=1.5)
        lines = (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0])["event"] == "episode_end"

    def test_update_manifest_rewrites_file(self, tmp_path):
        with RunLogger(tmp_path / "run", config={}) as logger:
            logger.update_manifest(learner={"algorithm": "sac"})
            manifest = read_manifest(tmp_path / "run")
            assert manifest["learner"]["algorithm"] == "sac"

    def test_close_is_idempotent(self, tmp_path):
        logger = RunLogger(tmp_path / "run")
        logger.close()
        logger.close()

    def test_read_metrics_missing_file(self, tmp_path):
        assert read_metrics(tmp_path / "nothing") == []

    def test_read_manifest_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_manifest(tmp_path / "nothing")

    def test_malformed_metrics_line_is_skipped(self, tmp_path):
        with RunLogger(tmp_path / "run") as logger:
            logger.log_metrics(1, {"loss": 1.0})
        path = tmp_path / "run" / "metrics.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "{broken json\n", encoding="utf-8")
        assert len(read_metrics(tmp_path / "run")) == 1


class TestMakeRunDir:
    def test_creates_timestamped_directory(self, tmp_path):
        path = make_run_dir(tmp_path, "my run")
        assert path.exists()
        assert "my_run" in path.name

    def test_sanitises_the_name(self, tmp_path):
        path = make_run_dir(tmp_path, "a/b c:d")
        assert "/" not in path.name
        assert path.exists()


# -- checkpoints ----------------------------------------------------------------------


@pytest.fixture
def learner():
    return SACLearner(
        4,
        3,
        SACConfig(network=__import__("tmai.models.networks", fromlist=["x"]).NetworkConfig(hidden_sizes=(16,))),
        action_low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        device="cpu",
        seed=0,
    )


class TestCheckpoints:
    def test_save_and_load_roundtrip(self, tmp_path, learner):
        from tmai.agents.base import Batch

        rng = np.random.default_rng(0)
        batch = Batch(
            observations=rng.normal(size=(16, 4)).astype(np.float32),
            actions=rng.uniform(
                low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                size=(16, 3),
            ).astype(np.float32),
            rewards=rng.normal(size=(16,)).astype(np.float32),
            next_observations=rng.normal(size=(16, 4)).astype(np.float32),
            terminated=np.zeros(16, dtype=np.float32),
            truncated=np.zeros(16, dtype=np.float32),
        )
        for _ in range(3):
            learner.update(batch)

        obs = np.zeros((8, 4), dtype=np.float32)
        expected = learner.act(obs, deterministic=True)

        path = save_checkpoint(tmp_path, step=42, learner=learner, episode=5, buffer_size=100)
        assert path.exists()

        payload = load_checkpoint(path)
        assert payload["step"] == 42
        assert payload["episode"] == 5
        assert payload["buffer_size"] == 100
        assert payload["gradient_steps"] == 3

        restored = SACLearner(
            4, 3, SACConfig(network=__import__("tmai.models.networks", fromlist=["x"]).NetworkConfig(hidden_sizes=(16,))),
            action_low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            action_high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            device="cpu", seed=1,
        )
        restored.load_state_dict(payload["learner"])
        np.testing.assert_allclose(restored.act(obs, deterministic=True), expected, rtol=1e-5, atol=1e-6)

    def test_pruning_keeps_the_newest(self, tmp_path, learner):
        for step in (10, 20, 30, 40, 50):
            save_checkpoint(tmp_path, step=step, learner=learner, keep=3)
        remaining = list_checkpoints(tmp_path)
        assert len(remaining) == 3
        assert [p.name for p in remaining] == [
            "checkpoint_000000030.pt",
            "checkpoint_000000040.pt",
            "checkpoint_000000050.pt",
        ]

    def test_latest_checkpoint_picks_highest_step(self, tmp_path, learner):
        for step in (5, 100, 50):
            save_checkpoint(tmp_path, step=step, learner=learner, keep=0)
        assert latest_checkpoint(tmp_path).name == "checkpoint_000000100.pt"

    def test_latest_checkpoint_none_when_empty(self, tmp_path):
        assert latest_checkpoint(tmp_path) is None
        assert latest_checkpoint(tmp_path / "missing") is None

    def test_best_checkpoint_is_overwritten(self, tmp_path, learner):
        first = save_best(tmp_path, step=1, learner=learner, score=0.5)
        second = save_best(tmp_path, step=2, learner=learner, score=0.9)
        assert first == second
        assert load_checkpoint(first)["best_score"] == pytest.approx(0.9)
        assert load_checkpoint(first)["extra"]["reason"] == "best_score"

    def test_read_meta(self, tmp_path, learner):
        path = save_checkpoint(tmp_path, step=7, learner=learner, episode=2, best_score=0.25)
        meta = read_meta(path)
        assert meta.step == 7
        assert meta.episode == 2
        assert meta.best_score == pytest.approx(0.25)
        assert meta.created_utc

    # -- a damaged file must be reported, not silently accepted or cryptically rejected ------

    def test_truncated_checkpoint_raises_actionable_error(self, tmp_path, learner):
        """A half-written file names itself, its size and a remedy.

        ``torch.load`` alone raises a bare ``OSError`` that points at neither the file nor a way
        forward, which is the wrong report for someone resuming a multi-day run.
        """
        path = save_checkpoint(tmp_path, step=11, learner=learner)
        path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])

        with pytest.raises(CheckpointError) as exc:
            load_checkpoint(path)
        msg = str(exc.value)
        assert path.name in msg
        assert "bytes" in msg
        assert "truncated or damaged" in msg

    def test_zero_byte_checkpoint_raises_actionable_error(self, tmp_path, learner):
        path = save_checkpoint(tmp_path, step=12, learner=learner)
        path.write_bytes(b"")
        with pytest.raises(CheckpointError, match="0 bytes"):
            load_checkpoint(path)

    def test_non_torch_bytes_raise_actionable_error(self, tmp_path, learner):
        path = save_checkpoint(tmp_path, step=13, learner=learner)
        path.write_bytes(b"this is not a torch checkpoint at all")
        with pytest.raises(CheckpointError, match="truncated or damaged"):
            load_checkpoint(path)

    def test_payload_that_is_not_a_mapping_is_rejected(self, tmp_path, learner):
        """A valid pickle of the wrong shape must not reach the resume path."""
        path = save_checkpoint(tmp_path, step=14, learner=learner)
        torch.save([1, 2, 3], path)
        with pytest.raises(CheckpointError, match="payload mapping"):
            load_checkpoint(path)

    def test_missing_checkpoint_still_reports_filenotfound(self, tmp_path):
        """Absence is a different problem from damage and keeps its distinct exception."""
        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            load_checkpoint(tmp_path / "checkpoint_000000001.pt")

    def test_rng_state_is_restored(self, tmp_path, learner):
        np.random.seed(123)
        torch.manual_seed(123)
        path = save_checkpoint(tmp_path, step=1, learner=learner)
        np.random.seed(999)
        torch.manual_seed(999)
        before_restore = np.random.random()

        restore_rng(load_checkpoint(path))
        after_restore = np.random.random()
        assert after_restore != before_restore

        # Restoring twice from the same payload yields the same stream.
        payload = load_checkpoint(path)
        restore_rng(payload)
        first = np.random.random()
        restore_rng(payload)
        second = np.random.random()
        assert first == pytest.approx(second)

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nope.pt")

    def test_write_index(self, tmp_path):
        path = write_index(tmp_path, [{"step": 1}, {"step": 2}])
        assert path.name == "checkpoints.json"
        assert len(json.loads(path.read_text(encoding="utf-8"))) == 2

    def test_atomic_save_leaves_no_temp_files(self, tmp_path, learner):
        save_checkpoint(tmp_path, step=1, learner=learner)
        assert not list(tmp_path.glob("*.tmp"))


# --------------------------------------------------------------------------- eval config
class TestRunConfigDiscovery:
    """`tmai eval --checkpoint <run>` must use the configuration that produced that run.

    A checkpoint is only meaningful with its own observation layout, track and normalisation.
    Falling back to `default.yaml` would silently load weights into a differently-shaped
    observation.
    """

    def _make_run(self, tmp_path: Path) -> tuple[Path, RunConfig]:
        """Create a run directory shaped exactly like the one the trainer produces."""
        run = tmp_path / "2026-01-01T00-00-00Z_run"
        cfg = RunConfig()
        cfg.train.run_name = "recorded"
        cfg.train.output_dir = str(tmp_path)
        cfg.train.eval_episodes = 1
        cfg.track.synthetic = "oval"
        cfg.driver.kind = "simulated"
        cfg.driver.allow_simulated = True
        # Every field is settled *before* config.yaml is written: the saved config must describe
        # the network the checkpoint below was actually trained with.
        cfg.sac.network.hidden_sizes = (8, 8)

        with RunLogger(run, run_name=cfg.train.run_name, config=cfg.to_dict()) as logger:
            logger.write_manifest()
        assert (run / "config.yaml").is_file(), "the trainer's run dir must carry its config"

        # Built through the factory, from a real environment, so the observation width and the
        # action bounds match exactly what `eval` will rebuild from this same config.
        from tmai.training.factory import build_driver, build_env, build_learner, build_track

        track = build_track(cfg)
        env = build_env(build_driver(cfg, track), track, cfg)
        try:
            learner = build_learner(env, cfg)
            for step in (100, 200):
                save_checkpoint(run, step=step, learner=learner, episode=0, config=cfg.to_dict())
            save_best(run, step=200, learner=learner, score=0.5, episode=0, config=cfg.to_dict())
        finally:
            env.close()
        return run, cfg

    def test_finds_config_beside_a_run_directory(self, tmp_path: Path):
        from tmai.cli import _run_dir_config

        run, _ = self._make_run(tmp_path)
        assert _run_dir_config(str(run)) == run / "config.yaml"

    def test_finds_config_from_a_checkpoint_file(self, tmp_path: Path):
        from tmai.cli import _run_dir_config

        run, _ = self._make_run(tmp_path)
        assert _run_dir_config(str(run / "checkpoint_000000100.pt")) == run / "config.yaml"
        assert _run_dir_config(str(run / "best.pt")) == run / "config.yaml"

    def test_none_when_there_is_no_saved_config(self, tmp_path: Path):
        from tmai.cli import _run_dir_config

        assert _run_dir_config(str(tmp_path / "missing.pt")) is None
        assert _run_dir_config(str(tmp_path)) is None
        assert _run_dir_config(None) is None

    def test_eval_picks_up_the_runs_own_track(self, tmp_path: Path, capsys, caplog):
        """End to end: no -c, no --set, and the run's own config is still used."""
        from tmai.cli import main

        run, _ = self._make_run(tmp_path)
        with caplog.at_level(logging.INFO):
            assert main(["eval", "--checkpoint", str(run), "--episodes", "1"]) == 0

        assert "=== evaluation" in capsys.readouterr().out
        assert "using the run's saved configuration" in caplog.text

    def test_mismatched_checkpoint_is_rejected(self, tmp_path: Path, caplog):
        """The guard that makes config discovery worth having: a wrong-shaped checkpoint must
        fail loudly rather than being silently loaded into a different observation."""
        from tmai.cli import main
        from tmai.training.checkpoint import save_checkpoint

        run, cfg = self._make_run(tmp_path)
        bogus = SACLearner(3, 3, SACConfig(network=NetworkConfig(hidden_sizes=(8, 8))))
        bad_dir = tmp_path / "wrong"
        bad_dir.mkdir()
        (bad_dir / "config.yaml").write_text(cfg.to_yaml())
        save_checkpoint(bad_dir, step=999, learner=bogus, episode=0, config=cfg.to_dict())

        with caplog.at_level(logging.ERROR):
            code = main(["eval", "--checkpoint", str(bad_dir / "checkpoint_000000999.pt"),
                         "--episodes", "1"])

        assert code == 1
        assert "observation_dim=3 does not match this learner's 20" in caplog.text

    def test_explicit_config_still_wins(self, tmp_path: Path, caplog):
        """An explicit -c must not be overridden by discovery."""
        import argparse

        from tmai.cli import _load_config

        run, _ = self._make_run(tmp_path)
        explicit = tmp_path / "explicit.yaml"
        cfg = RunConfig()
        cfg.train.run_name = "explicit"
        explicit.write_text(cfg.to_yaml())

        args = argparse.Namespace(
            config=str(explicit), checkpoint=str(run), set=None, allow_simulated_driver=False
        )
        with caplog.at_level(logging.INFO):
            loaded = _load_config(args)

        assert loaded.train.run_name == "explicit"
        assert "saved configuration" not in caplog.text


class TestStrictConfigKeys:
    """A misspelt key must fail loudly, not silently fall back to a default."""

    def test_typo_in_a_top_level_section_is_rejected(self):
        with pytest.raises(ValueError, match="train.total_step"):
            RunConfig.from_yaml_text("train:\n  total_step: 5000\n")

    def test_typo_in_a_nested_section_is_rejected(self):
        with pytest.raises(ValueError, match="sac.network.hidden_size"):
            RunConfig.from_yaml_text("sac:\n  network:\n    hidden_size: [8, 8]\n")

    def test_every_unknown_key_is_listed_at_once(self):
        with pytest.raises(ValueError) as caught:
            RunConfig.from_dict({"train": {"a": 1, "b": 2}, "bogus": 3})
        message = str(caught.value)
        assert "train.a" in message and "train.b" in message and "bogus" in message

    def test_lenient_loading_ignores_retired_keys_for_saved_runs(self):
        config = RunConfig.from_yaml_text("train:\n  retired_key: 1\n  total_steps: 777\n", strict=False)
        assert config.train.total_steps == 777

    def test_shipped_configurations_are_strict_clean(self):
        from pathlib import Path

        import tmai

        for path in sorted((Path(tmai.__file__).parent / "configs").glob("*.yaml")):
            RunConfig.from_yaml(path)  # raises if any shipped key is unknown

    def test_saved_configuration_round_trips_strictly(self):
        config = RunConfig()
        assert RunConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
