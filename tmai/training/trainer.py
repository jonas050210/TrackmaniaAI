"""The training loop.

Structure of the loop, and why:

* **Collect then learn.** The game is the bottleneck, so environment steps and gradient steps
  are decoupled by ``updates_per_step`` (the update-to-data ratio). With a real game that runs
  in real time a ratio above 1 is what makes the expensive samples pay for themselves; the
  ratio is configurable rather than fixed because too high a ratio over-fits the buffer.
* **Warm-up with random actions.** Gradients from an empty buffer are meaningless, so
  ``warmup_steps`` transitions are collected with uniform random actions first.
* **Correct bootstrapping.** ``terminated`` and ``truncated`` are tracked separately and only
  ``terminated`` stops bootstrapping, so time-limit truncations do not bias the value function.
* **Crash-resilient.** Metrics are flushed per record, checkpoints are atomic, and both
  ``KeyboardInterrupt`` and a lost game connection save a checkpoint before exiting. A
  multi-day run must never lose the last hours of work.
* **Interruptible by wall clock.** ``max_wall_seconds`` lets CI and scheduled runs bound
  themselves without a step-count hack.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tmai.agents.base import Learner, Transition
from tmai.agents.replay import ReplayBuffer
from tmai.config import RunConfig
from tmai.env.tm_env import TrackmaniaEnv
from tmai.game.errors import GameError
from tmai.runlog import RunLogger, make_run_dir
from tmai.training.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    restore_rng,
    save_best,
    save_checkpoint,
    seed_everything,
    seed_spaces,
)
from tmai.training.evaluate import EvaluationReport, evaluate_policy

logger = logging.getLogger(__name__)


@dataclass
class TrainerResult:
    """What a training run produced."""

    run_dir: Path
    steps: int = 0
    episodes: int = 0
    gradient_steps: int = 0
    best_score: float = 0.0
    wall_seconds: float = 0.0
    interrupted: bool = False
    failure: str | None = None
    final_evaluation: EvaluationReport | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "steps": self.steps,
            "episodes": self.episodes,
            "gradient_steps": self.gradient_steps,
            "best_score": round(self.best_score, 4),
            "wall_seconds": round(self.wall_seconds, 1),
            "interrupted": self.interrupted,
            "failure": self.failure,
            "final_evaluation": self.final_evaluation.as_dict() if self.final_evaluation else None,
        }


@dataclass
class EpisodeAccumulator:
    """Running totals for the current episode."""

    steps: int = 0
    reward: float = 0.0
    progress: float = 0.0
    max_speed: float = 0.0
    started: float = field(default_factory=time.monotonic)

    def reset(self) -> None:
        self.steps = 0
        self.reward = 0.0
        self.progress = 0.0
        self.max_speed = 0.0
        self.started = time.monotonic()


class Trainer:
    """Runs the collect/learn loop described in the module docstring.

    Args:
        config: the resolved run configuration.
        env: the environment (already wired to a driver).
        learner: the RL algorithm.
        buffer: the replay buffer.
        run_logger: where metrics and the manifest go.
    """

    def __init__(
        self,
        config: RunConfig,
        env: TrackmaniaEnv,
        learner: Learner,
        buffer: ReplayBuffer,
        run_logger: RunLogger,
        held_out_env: Any | None = None,
    ) -> None:
        self.config = config
        self.env = env
        self.learner = learner
        self.buffer = buffer
        self.log = run_logger
        #: Separate environment over tracks the policy never trains on. Optional: a run with
        #: no held-out split simply never reports a generalisation gap.
        self.held_out_env = held_out_env
        self._held_out_env = held_out_env
        self._episode = 0
        self._best_score = float("-inf")
        self._accumulator = EpisodeAccumulator()
        self._update_debt = 0.0
        self._last_eval_step = -1

    # -- public API -----------------------------------------------------------------

    def train(self) -> TrainerResult:
        cfg = self.config.train
        started = time.monotonic()
        result = TrainerResult(run_dir=self.log.run_dir)

        start_step = 0
        if cfg.resume:
            start_step = self._resume(cfg.resume)
            # The checkpoint restored the *global* RNG state, but gymnasium spaces keep their
            # own generators, so they are re-seeded from the state that was just restored.
            seed_spaces(self.env)
            seed_spaces(self.held_out_env)
        else:
            seed_everything(cfg.seed, env=self.env)
            if self.held_out_env is not None:
                # A distinct stream, so held-out exploration cannot shadow training's.
                seed_everything(None if cfg.seed is None else cfg.seed + 500, env=self.held_out_env)

        self.log.update_manifest(
            learner=self.learner.describe(),
            environment=self.env.describe(),
            resumed_from=cfg.resume,
        )

        observation, info = self._reset_env()
        self.log.log_event(
            "training_start",
            total_steps=cfg.total_steps,
            start_step=start_step,
            observation_dim=self.env.observation_dim,
            action_dim=int(np.prod(self.env.action_space.shape)),
            driver=self.env.driver.name,
        )

        try:
            for step in range(start_step + 1, cfg.total_steps + 1):
                if cfg.max_wall_seconds and time.monotonic() - started > cfg.max_wall_seconds:
                    logger.info("wall-clock limit reached after %.0fs", cfg.max_wall_seconds)
                    self.log.log_event("wall_clock_limit", step=step)
                    break

                action = self._select_action(observation, step)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                self.buffer.add(
                    Transition(
                        observation=observation,
                        action=np.asarray(action, dtype=np.float32),
                        reward=float(reward),
                        next_observation=next_observation,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                    )
                )

                self._accumulator.steps += 1
                self._accumulator.reward += float(reward)
                self._accumulator.progress = float(info.get("progress", 0.0))
                self._accumulator.max_speed = max(
                    self._accumulator.max_speed, float(info.get("speed_forward", 0.0))
                )

                metrics = self._maybe_update(step)

                if terminated or truncated:
                    self._on_episode_end(step, info)
                    observation, info = self._reset_env()
                else:
                    observation = next_observation

                if step % max(1, cfg.log_interval) == 0:
                    self._log_step(step, metrics, info, started)
                if cfg.eval_interval and step % cfg.eval_interval == 0:
                    # Evaluation drives the environment itself, so the training episode in
                    # progress is gone afterwards. Restart it rather than resuming from a
                    # stale observation.
                    self._run_evaluation(step)
                    self._last_eval_step = step
                    observation, info = self._reset_env()
                if cfg.checkpoint_interval and step % cfg.checkpoint_interval == 0:
                    self._save(step)

                result.steps = step
                result.gradient_steps = int(getattr(self.learner, "gradient_steps", 0))
                result.episodes = self._episode

        except KeyboardInterrupt:
            logger.warning("interrupted by operator; saving a checkpoint")
            self.log.log_event("interrupted", step=result.steps)
            result.interrupted = True
            self._save(result.steps, note="interrupted")
        except GameError as exc:
            # Losing the game is a deployment problem, not a bug in the loop. Preserve work.
            logger.error("game error: %s", exc)
            self.log.log_event("game_error", step=result.steps, error=str(exc))
            result.failure = str(exc)
            self._save(result.steps, note="game_error")

        result.wall_seconds = time.monotonic() - started
        result.best_score = 0.0 if self._best_score == float("-inf") else self._best_score

        if cfg.eval_episodes > 0 and result.steps != self._last_eval_step:
            result.final_evaluation = self._run_evaluation(result.steps, force=True)
        self._save(result.steps, note="final")
        self.log.log_event(
            "training_end",
            **{k: v for k, v in result.as_dict().items() if k != "final_evaluation"},
        )
        return result

    # -- internals ------------------------------------------------------------------

    def _reset_env(self) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset()
        self._accumulator.reset()
        return observation, info

    def _select_action(self, observation: np.ndarray, step: int) -> np.ndarray:
        if step <= self.config.train.warmup_steps:
            return np.asarray(self.env.action_space.sample(), dtype=np.float32)
        return self.learner.act(observation, deterministic=False)

    def _maybe_update(self, step: int) -> dict[str, float]:
        cfg = self.config.train
        if step <= cfg.warmup_steps:
            return {}
        if step % max(1, cfg.update_every) != 0:
            return {}
        if len(self.buffer) < cfg.batch_size:
            return {}

        self._update_debt += cfg.updates_per_step * cfg.update_every
        updates = int(self._update_debt)
        self._update_debt -= updates

        metrics: dict[str, float] = {}
        for _ in range(max(0, updates)):
            metrics = self.learner.update(self.buffer.sample(cfg.batch_size))
        return metrics

    def _on_episode_end(self, step: int, info: dict[str, Any]) -> None:
        self._episode += 1
        elapsed = max(1e-9, time.monotonic() - self._accumulator.started)
        self.log.log_event(
            "episode_end",
            step=step,
            episode=self._episode,
            steps=self._accumulator.steps,
            reward=round(self._accumulator.reward, 4),
            progress=round(self._accumulator.progress, 2),
            max_speed=round(self._accumulator.max_speed, 2),
            end_reason=str(info.get("end_reason", "")),
            finished=bool(info.get("finished", False)),
            race_time=float(info.get("race_time", 0.0)),
            steps_per_second=self._accumulator.steps / elapsed,
            driver=str(info.get("driver", "")),
            # Which map this episode ran on. Without it, per-track episode analysis is
            # impossible for a multi-track run, and a bad track looks like a bad policy.
            track=str(info.get("track", getattr(self.env, "track", None)
                                 and self.env.track.name or "")),
        )
        logger.info(
            "episode %d ended (%s): %d steps, return %.2f, progress %.1f m",
            self._episode,
            info.get("end_reason", "?"),
            self._accumulator.steps,
            self._accumulator.reward,
            self._accumulator.progress,
        )

    def _log_step(
        self,
        step: int,
        metrics: dict[str, float],
        info: dict[str, Any],
        started: float,
    ) -> None:
        elapsed = max(1e-9, time.monotonic() - started)
        record: dict[str, Any] = {
            "env/episode": self._episode,
            "env/progress": float(info.get("progress", 0.0)),
            "env/progress_fraction": float(info.get("progress_fraction", 0.0)),
            "env/speed_forward": float(info.get("speed_forward", 0.0)),
            "env/lateral_offset": float(info.get("lateral_offset", 0.0)),
            "env/race_time": float(info.get("race_time", 0.0)),
            "env/nonfinite_observations": int(info.get("nonfinite_observations", 0)),
            "throughput/env_steps_per_second": step / elapsed,
            "throughput/wall_seconds": elapsed,
        }
        for key in ("reward/total", "reward/progress", "reward/off_track", "reward/clamped"):
            if key in info:
                record[key] = float(info[key])
        record.update(metrics)
        record.update(self.buffer.statistics())
        record["learner/temperature"] = float(getattr(self.learner, "temperature", 0.0))
        self.log.log_metrics(step, record)
        logger.info(
            "step %d | ep %d | progress %.1f%% | %.1f steps/s | %s",
            step,
            self._episode,
            100.0 * float(info.get("progress_fraction", 0.0)),
            step / elapsed,
            ", ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in list(metrics.items())[:3]),
        )

    def _run_evaluation(self, step: int, *, force: bool = False) -> EvaluationReport | None:
        episodes = self.config.train.eval_episodes
        if episodes <= 0 and not force:
            return None
        report = evaluate_policy(
            self.env,
            self.learner,
            episodes=episodes,
            max_steps=self.config.env.termination.max_steps,
            deterministic=True,
            seed=self.config.train.seed,
            label="training",
            step=step,
            split="train",
        )
        self.log.log_metrics(step, report.metrics())
        # The report carries its own "step" field; nest it rather than colliding with the
        # event's step argument.
        self.log.log_event("evaluation", step=step, report=report.as_dict())
        logger.info("evaluation at step %d: %s", step, report.summary())

        self._maybe_evaluate_held_out(step)

        if report.score > self._best_score:
            self._best_score = report.score
            save_best(
                self.log.run_dir,
                step=step,
                learner=self.learner,
                score=report.score,
                episode=self._episode,
                config=self.config.to_dict(),
            )
            self.log.log_event("new_best", step=step, score=round(report.score, 4))
        return report

    def _maybe_evaluate_held_out(self, step: int) -> EvaluationReport | None:
        """Evaluate on tracks the policy never trained on.

        This is the number that says whether the agent is learning to *drive* rather than
        learning one map. It runs on a separate environment so the training episode and the
        training driver are untouched, and it is skipped entirely when there is no held-out
        split -- reported as such rather than silently scoring zero.
        """
        if self._held_out_env is None:
            return None
        cfg = self.config.train
        if cfg.held_out_eval_interval <= 0 or step % cfg.held_out_eval_interval != 0:
            return None
        try:
            report = evaluate_policy(
                self._held_out_env,
                self.learner,
                episodes=cfg.held_out_eval_episodes,
                max_steps=self.config.env.termination.max_steps,
                deterministic=True,
                seed=(cfg.seed or 0) + 90000,
                label="held_out",
                step=step,
                split="validation",
            )
        except Exception:  # noqa: BLE001 - held-out eval must never kill a long run
            logger.exception("held-out evaluation failed; continuing training")
            self.log.log_event("held_out_eval_error", step=step)
            return None

        self.log.log_metrics(step, report.metrics(prefix="heldout"))
        self.log.log_event("held_out_evaluation", step=step, report=report.as_dict())
        logger.info("HELD-OUT evaluation at step %d: %s", step, report.summary())
        return report

    def _save(self, step: int, *, note: str | None = None) -> None:
        save_checkpoint(
            self.log.run_dir,
            step=step,
            learner=self.learner,
            episode=self._episode,
            buffer_size=len(self.buffer),
            config=self.config.to_dict(),
            best_score=None if self._best_score == float("-inf") else self._best_score,
            extra={"note": note} if note else None,
            keep=self.config.train.keep_checkpoints,
        )

    def _resume(self, resume: str) -> int:
        """Restore learner, RNG and counters; returns the step to continue from."""
        source = Path(resume)
        path = source if source.is_file() else latest_checkpoint(source)
        if path is None:
            raise FileNotFoundError(f"no checkpoint found at {resume}")
        payload = load_checkpoint(path)
        self.learner.load_state_dict(payload["learner"])
        restore_rng(payload)
        self._episode = int(payload.get("episode", 0))
        self._best_score = float(payload.get("best_score") or float("-inf"))
        step = int(payload.get("step", 0))
        logger.info(
            "resumed from %s (step %d, gradient_steps %s, episode %d)",
            path.name,
            step,
            payload.get("gradient_steps"),
            self._episode,
        )
        self.log.log_event(
            "resumed",
            checkpoint=str(path),
            step=step,
            gradient_steps=payload.get("gradient_steps"),
            episode=self._episode,
        )
        return step


def train_from_config(config: RunConfig) -> TrainerResult:
    """Build everything from ``config`` and run training. The ``tmai train`` entry point."""
    from tmai.training.factory import (
        build_buffer,
        build_learner,
        build_library,
        build_multi_track_env,
    )

    config.validate_or_raise()

    library = build_library(config)
    env = build_multi_track_env(config, library, split="train", seed=config.train.seed)
    learner = build_learner(env, config)
    buffer = build_buffer(env, config)

    # A held-out environment is built only when there is something to hold out and the run
    # asked for it. Building one unconditionally would open game connections for nothing.
    held_out_env = None
    if config.train.held_out_eval_interval > 0 and library.by_split("validation"):
        held_out_env = build_multi_track_env(
            config,
            library,
            split="validation",
            seed=(config.train.seed or 0) + 500,
        )

    run_dir = make_run_dir(config.train.output_dir, config.train.run_name)
    with RunLogger(
        run_dir,
        run_name=config.train.run_name,
        config=config.to_dict(),
        seed=config.train.seed,
        extra_manifest={"tracks": library.report()},
    ) as run_logger:
        trainer = Trainer(config, env, learner, buffer, run_logger, held_out_env=held_out_env)
        try:
            return trainer.train()
        finally:
            env.close()
            if held_out_env is not None:
                held_out_env.close()


__all__ = ["EpisodeAccumulator", "Trainer", "TrainerResult", "train_from_config"]
