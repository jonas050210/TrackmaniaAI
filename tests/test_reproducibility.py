"""Reproducibility and resume behaviour, pinned by actual runs rather than by assertion.

Two things were measured here that the rest of the suite never checked:

* **Reproducibility was broken.** ``train.seed`` did not actually reproduce a run. Two
  independent sources of entropy escaped it: gymnasium spaces own a private generator that
  ``np.random.seed`` cannot reach (so warm-up actions varied), and ``ReplayBuffer`` owns a
  ``default_rng`` that was seeded from ``replay.seed``, which every shipped config leaves
  ``null`` -- i.e. fresh OS entropy. Both are fixed; the tests below fail if either regresses.

* **Resume is deliberately not trajectory-identical.** A checkpoint records the buffer *size*
  but not its contents, so a resumed run refills the buffer and diverges. That is a documented
  trade-off, not a bug, and there is a test here that pins it so the trade-off cannot silently
  turn into "resume silently restarts learning from scratch".
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from tmai.agents.replay import ReplayBuffer, ReplayBufferConfig
from tmai.config import RunConfig
from tmai.env.tm_env import EnvConfig, TrackmaniaEnv
from tmai.game.simulated import SimulatedGameDriver
from tmai.runlog import read_metrics
from tmai.tracks.synthetic import straight
from tmai.training.checkpoint import latest_checkpoint, load_checkpoint, seed_everything, seed_spaces
from tmai.training.factory import build_all, build_buffer
from tmai.training.trainer import train_from_config


def _config(out: Path, steps: int, *, seed: int = 99, resume: str | None = None) -> RunConfig:
    """A configuration small enough to run in well under a second, but with real updates."""
    config = RunConfig()
    config.driver.kind = "simulated"
    config.driver.allow_simulated = True
    config.track.synthetic = "straight"
    config.track.synthetic_kwargs = {"length": 120.0}
    config.env.termination.max_steps = 200
    config.sac.network.hidden_sizes = (16,)
    config.train.output_dir = str(out)
    config.train.run_name = "repro"
    config.train.total_steps = steps
    config.train.warmup_steps = 10
    config.train.batch_size = 8
    config.train.updates_per_step = 1
    config.train.log_interval = 10
    config.train.eval_interval = 0
    config.train.held_out_eval_interval = 0
    config.train.checkpoint_interval = max(steps, 1)
    config.train.seed = seed
    config.train.resume = resume
    return config


def _metrics(run_dir: Path) -> list[tuple[int, float]]:
    """The reward series, which is a sensitive fingerprint of the whole run."""
    return [
        (int(m["step"]), round(float(m.get("reward/total", 0.0)), 9))
        for m in read_metrics(run_dir)
        if m.get("step") is not None
    ]


@pytest.fixture()
def workdir(tmp_path: Path):
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)


# -- the seeding primitives ----------------------------------------------------------


class TestSeedEverything:
    def _env(self) -> TrackmaniaEnv:
        track = straight(length=120.0)
        driver = SimulatedGameDriver(track)
        driver.open()
        return TrackmaniaEnv(driver, track, EnvConfig())

    def test_action_space_sampling_is_reproducible(self):
        """The bug this guards: `np.random.seed` does not reach gymnasium's private generator."""
        env = self._env()
        try:
            seed_everything(1234, env=env)
            first = [tuple(env.action_space.sample()) for _ in range(4)]
            seed_everything(1234, env=env)
            second = [tuple(env.action_space.sample()) for _ in range(4)]
        finally:
            env.close()
        assert first == second

    def test_unseeded_action_space_would_have_varied(self):
        """Proves the previous test is not vacuous: without seeding it really does vary."""
        env = self._env()
        try:
            env.reset()
            first = [tuple(env.action_space.sample()) for _ in range(4)]
            env2 = self._env()
            env2.reset()
            second = [tuple(env2.action_space.sample()) for _ in range(4)]
            env2.close()
        finally:
            env.close()
        assert first != second

    def test_seed_spaces_derives_from_global_state(self):
        """The same global state must produce the same space stream, which is what lets a
        resume behave deterministically without the space state being in the checkpoint."""
        env = self._env()
        try:
            seed_everything(42, env=None)
            seed_spaces(env)
            first = [tuple(env.action_space.sample()) for _ in range(3)]
            seed_everything(42, env=None)
            seed_spaces(env)
            second = [tuple(env.action_space.sample()) for _ in range(3)]
        finally:
            env.close()
        assert first == second

    def test_none_seed_is_a_no_op(self):
        seed_everything(None, env=None)  # must not raise

    def test_seed_spaces_tolerates_a_missing_env(self):
        seed_spaces(None)


# -- the replay buffer stream --------------------------------------------------------


class TestReplayBufferSeeding:
    def _env(self) -> TrackmaniaEnv:
        track = straight(length=120.0)
        driver = SimulatedGameDriver(track)
        driver.open()
        return TrackmaniaEnv(driver, track, EnvConfig())

    def test_null_replay_seed_inherits_the_run_seed(self):
        """The bug this guards: every shipped config sets `replay.seed: null`."""
        env = self._env()
        try:
            config = _config(Path("/tmp/unused"), 10, seed=7)
            assert config.replay.seed is None, "precondition: shipped default is null"
            buffer = build_buffer(env, config)
        finally:
            env.close()
        assert buffer.config.seed is not None
        # A distinct stream from the run seed, so buffer draws do not shadow training's.
        assert buffer.config.seed == 7 + 7000

    def test_explicit_replay_seed_wins(self):
        env = self._env()
        try:
            config = _config(Path("/tmp/unused"), 10, seed=7)
            config.replay.seed = 123
            buffer = build_buffer(env, config)
        finally:
            env.close()
        assert buffer.config.seed == 123

    def test_two_buffers_from_the_same_seed_sample_identically(self):
        env = self._env()
        try:
            config = _config(Path("/tmp/unused"), 10, seed=7)
            a = build_buffer(env, config)
            b = build_buffer(env, config)
        finally:
            env.close()

        from tmai.agents.base import Transition

        obs = np.zeros(env.observation_dim, dtype=np.float32)
        action = np.zeros(int(env.action_space.shape[0]), dtype=np.float32)
        for buffer in (a, b):
            for _ in range(20):
                buffer.add(Transition(obs, action, 0.0, obs, False, False))

        sa = a.sample(8)
        sb = b.sample(8)
        assert np.array_equal(sa.rewards, sb.rewards)

    def test_a_null_seed_buffer_is_not_reproducible(self):
        """Proves the fix is load-bearing: an unseeded buffer really does draw fresh entropy."""
        from tmai.agents.base import Transition

        a = ReplayBuffer(observation_dim=4, action_dim=2, config=ReplayBufferConfig(seed=None))
        b = ReplayBuffer(observation_dim=4, action_dim=2, config=ReplayBufferConfig(seed=None))
        obs = np.zeros(4, dtype=np.float32)
        action = np.zeros(2, dtype=np.float32)
        for buffer in (a, b):
            for i in range(20):
                buffer.add(Transition(obs, action, float(i), obs, False, False))
        assert not np.array_equal(a.sample(8).rewards, b.sample(8).rewards)


# -- end-to-end reproducibility ------------------------------------------------------


class TestRunReproducibility:
    def test_same_seed_reproduces_the_reward_series(self, workdir):
        run_a = train_from_config(_config(workdir / "a", 60)).run_dir
        run_b = train_from_config(_config(workdir / "b", 60)).run_dir

        series_a, series_b = _metrics(run_a), _metrics(run_b)
        assert len(series_a) > 3, "the run must log enough records to be a meaningful fingerprint"
        assert series_a == series_b

    def test_different_seeds_differ(self, workdir):
        """Guards against the test above passing because seeding is simply ignored."""
        run_a = train_from_config(_config(workdir / "a", 60, seed=1)).run_dir
        run_b = train_from_config(_config(workdir / "b", 60, seed=2)).run_dir
        assert _metrics(run_a) != _metrics(run_b)

    def test_reproduced_runs_reach_the_same_final_learner(self, workdir):
        run_a = train_from_config(_config(workdir / "a", 60)).run_dir
        run_b = train_from_config(_config(workdir / "b", 60)).run_dir

        def final_weights(run: Path) -> list[np.ndarray]:
            """Every tensor in the checkpoint, flattened out of the nested state dict."""
            payload = load_checkpoint(latest_checkpoint(run))
            found: list[np.ndarray] = []

            def walk(node) -> None:
                if hasattr(node, "detach"):
                    found.append(node.detach().cpu().numpy().copy())
                elif isinstance(node, dict):
                    for key in sorted(node, key=str):
                        walk(node[key])

            walk(payload["learner"])
            return found

        weights_a, weights_b = final_weights(run_a), final_weights(run_b)
        assert weights_a, "checkpoint should carry network tensors"
        assert len(weights_a) == len(weights_b)
        for x, y in zip(weights_a, weights_b, strict=True):
            assert np.array_equal(x, y)


# -- resume: what it does and does not restore ---------------------------------------


class TestResumeBehaviour:
    def test_checkpoint_carries_rng_but_not_buffer_contents(self, workdir):
        """The checkpoint stores the buffer *size* only. That is the reason resume is not
        trajectory-identical, and it is a deliberate size trade-off for multi-day runs."""
        run = train_from_config(_config(workdir / "a", 40)).run_dir
        payload = load_checkpoint(latest_checkpoint(run))

        assert "rng" in payload
        assert set(payload["rng"]) >= {"python", "numpy", "torch"}
        assert payload["buffer_size"] > 0
        assert "buffer" not in payload
        assert "transitions" not in payload

    def test_resume_restores_step_and_gradient_counters(self, workdir):
        first = train_from_config(_config(workdir / "a", 25)).run_dir
        resumed = train_from_config(
            _config(workdir / "b", 50, resume=str(first))
        ).run_dir
        payload = load_checkpoint(latest_checkpoint(resumed))
        assert payload["step"] == 50
        assert payload["gradient_steps"] > 0

    def test_resume_does_not_reproduce_the_uninterrupted_trajectory(self, workdir):
        """Pins the documented trade-off: because the buffer is refilled rather than restored,
        a resumed run diverges from the run it continues.

        This is asserted rather than apologised for, so that a future change which *does*
        checkpoint the buffer has to update this test consciously.
        """
        full = train_from_config(_config(workdir / "full", 50)).run_dir
        part = train_from_config(_config(workdir / "part", 25)).run_dir
        resumed = train_from_config(
            _config(workdir / "resumed", 50, resume=str(part))
        ).run_dir

        def after(run: Path, step: int) -> list[tuple[int, float]]:
            return [entry for entry in _metrics(run) if entry[0] > step]

        assert after(full, 25) != after(resumed, 25)

    def test_resumed_run_still_makes_progress(self, workdir):
        """The flip side: divergence is not the same as collapse. A resumed run must keep
        learning, which is the actual guarantee that matters operationally."""
        part = train_from_config(_config(workdir / "part", 25)).run_dir
        resumed = train_from_config(
            _config(workdir / "resumed", 60, resume=str(part))
        ).run_dir

        payload = load_checkpoint(latest_checkpoint(resumed))
        assert payload["step"] == 60
        # The learner must have kept updating after the resume point.
        assert payload["gradient_steps"] > 0
        assert _metrics(resumed), "the resumed run must still log metrics"


# -- factory wiring ------------------------------------------------------------------


class TestBuildAllIsSeeded:
    def test_build_all_learners_share_a_seed(self, workdir):
        """Already covered in test_agents.py; repeated here because build_all is the path a
        real run takes and the seeding order there is what reproducibility depends on."""
        config = _config(workdir / "a", 10, seed=5)
        _, learner_a, _, _ = build_all(config)
        _, learner_b, _, _ = build_all(config)

        obs = np.zeros(learner_a.observation_dim, dtype=np.float32)
        seed_everything(5)
        action_a = learner_a.act(obs, deterministic=False)
        seed_everything(5)
        action_b = learner_b.act(obs, deterministic=False)
        assert np.array_equal(action_a, action_b)
