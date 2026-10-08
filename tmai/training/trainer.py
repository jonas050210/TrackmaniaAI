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
from tmai.replay import REPLAY_DIR_NAME, ReplayRecorder, ReplayStore
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
        curriculum: Any | None = None,
        director: Any | None = None,
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
        #: Progressive track/episode curriculum, attached to the training env only.
        self._curriculum = curriculum
        self._last_curriculum_stage = -1
        #: Failure-focused sampling, wired only to the training environment.
        self._director = director
        self._episode = 0
        self._best_score = float("-inf")
        self._accumulator = EpisodeAccumulator()
        self._update_debt = 0.0
        self._last_eval_step = -1
        # -- replay recording ------------------------------------------------------------
        #: Records the current episode's trajectory when ``train.record_replays`` is on.
        self._replay_recorder: ReplayRecorder | None = None
        self._replay_store: ReplayStore | None = None
        if config.train.record_replays:
            self._replay_recorder = ReplayRecorder(
                decimation=max(1, config.train.replay_decimation)
            )
            self._replay_store = ReplayStore(self.log.run_dir / REPLAY_DIR_NAME)

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
        if self._curriculum is not None:
            self.log.update_manifest(curriculum=self._curriculum.describe())
        if self._director is not None:
            self.log.update_manifest(training_director=self._director.describe())

        observation, info = self._reset_env()
        self.log.log_event(
            "training_start",
            total_steps=cfg.total_steps,
            start_step=start_step,
            observation_dim=self.env.observation_dim,
            action_dim=int(self.env.action_space.low.size),
            driver=self.env.driver.name,
        )

        try:
            for step in range(start_step + 1, cfg.total_steps + 1):
                if cfg.max_wall_seconds and time.monotonic() - started > cfg.max_wall_seconds:
                    logger.info("wall-clock limit reached after %.0fs", cfg.max_wall_seconds)
                    self.log.log_event("wall_clock_limit", step=step)
                    break

                self._advance_curriculum(step)

                action = self._select_action(observation, step)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                self._record_replay_step(action, reward, info)
                nominal_control_seconds = self.config.env.control_dt * max(
                    1, self.config.env.action_repeat
                )
                elapsed_seconds = float(info.get("elapsed_seconds", 0.0))
                discount_exponent = (
                    elapsed_seconds / nominal_control_seconds
                    if np.isfinite(elapsed_seconds) and elapsed_seconds > 0.0
                    else 1.0
                )
                self.buffer.add(
                    Transition(
                        observation=observation,
                        action=np.asarray(action, dtype=np.float32),
                        reward=float(reward),
                        next_observation=next_observation,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        discount_exponent=discount_exponent,
                    )
                )
                result.steps = step

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
        except Exception as exc:
            # Unexpected failures should remain visible to the caller, but not destroy the
            # last useful learner/director state. Checkpoint failure must never mask the
            # original traceback.
            logger.exception("unexpected trainer failure at step %d; saving state", result.steps)
            try:
                self.log.log_event(
                    "trainer_exception",
                    step=result.steps,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            except Exception:  # noqa: BLE001 - logging must not mask the trainer exception
                logger.exception("could not write trainer exception event")
            try:
                self._save(result.steps, note="unexpected_exception")
            except Exception:  # noqa: BLE001 - preserve the original exception
                logger.exception("could not checkpoint after trainer exception")
            raise

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

    def _advance_curriculum(self, step: int) -> None:
        """Push the current step into the env and log stage transitions."""
        if self._curriculum is None:
            return
        setter = getattr(self.env, "set_curriculum_step", None)
        if setter is None:
            return
        setter(step)
        stage = self._curriculum.stage_index_at(step)
        if stage != self._last_curriculum_stage:
            self._last_curriculum_stage = stage
            active = self._curriculum.active_track_names(step)
            logger.info(
                "curriculum stage %d at step %d: %s",
                stage,
                step,
                f"{len(active)} track(s)" if active is not None else "all tracks",
            )
            self.log.log_event(
                "curriculum_stage",
                step=step,
                stage=stage,
                num_stages=self._curriculum.num_stages,
                active_tracks=active,
                episode_max_steps=self._curriculum.episode_max_steps(
                    step, self.config.env.termination.max_steps
                ),
            )

    def _reset_env(self) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset()
        self._accumulator.reset()
        if self._replay_recorder is not None:
            self._replay_recorder.reset()
            frame = getattr(self.env, "last_frame", None)
            if frame is not None:
                self._replay_recorder.record(
                    position=frame.vehicle.position,
                    speed=frame.vehicle.speed_forward,
                    action=np.zeros(3, dtype=np.float32),
                    reward=0.0,
                    progress=float(info.get("progress", 0.0)),
                    race_time=float(info.get("race_time", 0.0)),
                )
        return observation, info

    # -- replay recording -------------------------------------------------------------

    def _record_replay_step(
        self, action: np.ndarray, reward: float, info: dict[str, Any]
    ) -> None:
        """Append one step to the current episode's replay, if recording is enabled."""
        if self._replay_recorder is None:
            return
        frame = getattr(self.env, "last_frame", None)
        if frame is None:
            return
        self._replay_recorder.record(
            position=frame.vehicle.position,
            speed=float(info.get("speed_forward", frame.vehicle.speed_forward)),
            action=np.asarray(action, dtype=np.float64).reshape(-1),
            reward=float(reward),
            progress=float(info.get("progress", 0.0)),
            race_time=float(info.get("race_time", frame.race.race_time)),
        )

    def _save_replay(self, step: int, info: dict[str, Any]) -> None:
        """Write the finished episode's replay to ``<run>/replays/``."""
        if self._replay_recorder is None or self._replay_store is None:
            return
        if self._replay_recorder.steps == 0:
            return
        replay = self._replay_recorder.build(
            episode=self._episode,
            step=step,
            track=str(info.get("track", "")),
            split=str(info.get("split", "train")),
            end_reason=str(info.get("end_reason", "")),
            finished=bool(info.get("finished", False)),
            race_time=float(info.get("race_time", 0.0)),
            total_reward=round(self._accumulator.reward, 4),
            progress_fraction=float(info.get("progress_fraction", 0.0)),
            source="training",
            metadata={
                "episode_steps": self._accumulator.steps,
                "game_end_reason": str(info.get("game_end_reason", info.get("end_reason", ""))),
                "game_finished": bool(info.get("game_finished", info.get("finished", False))),
                "invalid_finish": bool(info.get("invalid_finish", False)),
                "checkpoint_index": int(info.get("checkpoint_index", 0) or 0),
                "checkpoint_total": int(info.get("checkpoint_total", 0) or 0),
                "track_identity": str(info.get("track_identity", "")),
            },
        )
        path = self._replay_store.save(
            replay, max_replays=max(0, self.config.train.max_replays)
        )
        logger.info(
            "recorded replay %s (%d samples, %s)",
            path.name, replay.num_samples, replay.end_reason or "running",
        )

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
        track_name = str(info.get("track", getattr(self.env, "track", None)
                                 and self.env.track.name or ""))
        game_finished = bool(info.get("finished", False))
        checkpoint_total = int(info.get("checkpoint_total", 0) or 0)
        checkpoint_index = int(info.get("checkpoint_index", 0) or 0)
        checkpoint_invalid = (
            game_finished and checkpoint_total > 0 and checkpoint_index < checkpoint_total
        )
        invalid_finish = bool(info.get("invalid_finish", checkpoint_invalid))
        finished = bool(info.get("valid_finish", game_finished and not invalid_finish))
        game_end_reason = str(info.get("end_reason", ""))
        end_reason = "invalid_finish" if invalid_finish else game_end_reason

        replay_info = dict(info)
        replay_info.update(
            {
                "end_reason": end_reason,
                "game_end_reason": game_end_reason,
                "finished": finished,
                "game_finished": game_finished,
                "invalid_finish": invalid_finish,
                "checkpoint_index": checkpoint_index,
                "checkpoint_total": checkpoint_total,
            }
        )
        self._save_replay(step, replay_info)

        if self._director is not None:
            progress_fraction = float(info.get("progress_fraction", 0.0))
            weights = self._director.observe_episode(
                track=track_name,
                progress_fraction=progress_fraction,
                finished=finished,
                end_reason=end_reason,
            )
            setter = getattr(self.env, "set_sampling_weights", None)
            if callable(setter):
                setter(weights)
            self.log.log_event(
                "training_director_update",
                step=step,
                episode=self._episode,
                track=track_name,
                progress_fraction=progress_fraction,
                end_reason=end_reason,
                game_end_reason=game_end_reason,
                finished=finished,
                game_finished=game_finished,
                invalid_finish=invalid_finish,
                checkpoint_index=checkpoint_index,
                checkpoint_total=checkpoint_total,
                weights=weights,
                summary=self._director.summary(),
            )
        self.log.log_event(
            "episode_end",
            step=step,
            episode=self._episode,
            steps=self._accumulator.steps,
            reward=round(self._accumulator.reward, 4),
            progress=round(self._accumulator.progress, 2),
            max_speed=round(self._accumulator.max_speed, 2),
            end_reason=end_reason,
            game_end_reason=game_end_reason,
            finished=finished,
            game_finished=game_finished,
            invalid_finish=invalid_finish,
            checkpoint_index=checkpoint_index,
            checkpoint_total=checkpoint_total,
            race_time=float(info.get("race_time", 0.0)),
            steps_per_second=self._accumulator.steps / elapsed,
            driver=str(info.get("driver", "")),
            # Which map this episode ran on. Without it, per-track episode analysis is
            # impossible for a multi-track run, and a bad track looks like a bad policy.
            track=track_name,
        )
        logger.info(
            "episode %d ended (%s): %d steps, return %.2f, progress %.1f m",
            self._episode,
            end_reason or "?",
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
        if self._curriculum is not None:
            record["curriculum/stage"] = float(self._curriculum.stage_index_at(step))
        if self._director is not None:
            record.update(self._director.summary())
        # Resource usage rides along with the metrics so a dashboard can plot it against
        # reward without a second data source. Stdlib-only; missing counters are skipped.
        from tmai.monitoring import flat_system_metrics

        record.update(flat_system_metrics())
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
                extra=self._checkpoint_extra(note="best_score"),
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

    def _checkpoint_extra(self, *, note: str | None = None) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if note:
            extra["note"] = note
        if self._director is not None:
            extra["training_director"] = self._director.state_dict()
        state_getter = getattr(self.env, "state_dict", None)
        if callable(state_getter):
            extra["training_env"] = state_getter()
        return extra

    def _save(self, step: int, *, note: str | None = None) -> None:
        save_checkpoint(
            self.log.run_dir,
            step=step,
            learner=self.learner,
            episode=self._episode,
            buffer_size=len(self.buffer),
            config=self.config.to_dict(),
            best_score=None if self._best_score == float("-inf") else self._best_score,
            extra=self._checkpoint_extra(note=note),
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
        extra = payload.get("extra", {})
        if self._director is not None and "training_director" in extra:
            self._director.load_state_dict(extra["training_director"])
        state_loader = getattr(self.env, "load_state_dict", None)
        if callable(state_loader) and "training_env" in extra:
            state_loader(extra["training_env"])
        setter = getattr(self.env, "set_sampling_weights", None)
        if self._director is not None and callable(setter):
            setter(self._director.weights())
        elif self._director is None:
            clearer = getattr(self.env, "clear_sampling_weights", None)
            if callable(clearer):
                clearer()
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
    from tmai.training.curriculum import Curriculum
    from tmai.training.director import TrainingDirector
    from tmai.training.factory import (
        ConfigError,
        build_buffer,
        build_learner,
        build_library,
        build_multi_track_env,
    )

    config.validate_or_raise()

    library = build_library(config)
    if config.driver.kind == "tminterface" and len(library.entries) > 1:
        raise ConfigError(
            "the real TMInterface runtime currently supports one already-loaded map per run; "
            "it cannot switch or hold out multiple maps automatically. Use a one-map library, "
            "or use the explicitly simulated driver for multi-track tests."
        )
    env = build_multi_track_env(config, library, split="train", seed=config.train.seed)
    learner = build_learner(env, config)
    buffer = build_buffer(env, config)

    # The curriculum is resolved over the *training* tracks and attached to the training
    # environment only; held-out evaluation always sees every track at full length.
    curriculum = None
    if config.curriculum.enabled:
        entries = library.by_split("train")
        if not entries:
            raise ValueError("curriculum.enabled is true but the train split is empty")
        curriculum = Curriculum(config.curriculum, entries)
        attacher = getattr(env, "attach_curriculum", None)
        if attacher is not None:
            attacher(curriculum)
        logger.info(
            "curriculum: %d stage(s) over %d training track(s); reveal order %s",
            curriculum.num_stages,
            len(entries),
            curriculum.difficulty_order(),
        )

    # Adaptive track priorities are constructed exclusively from the training split. The
    # held-out environment below never receives the director or its weights.
    director = None
    if config.director.enabled:
        train_entries = library.by_split("train")
        director = TrainingDirector(config.director, [entry.name for entry in train_entries])
        weight_setter = getattr(env, "set_sampling_weights", None)
        if callable(weight_setter):
            weight_setter(director.weights())
        else:
            logger.info(
                "Training Director is recording outcomes, but adaptive track sampling has no "
                "effect with a single training track"
            )

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
        # Behaviour cloning: warm-start the policy from demonstrations before RL begins.
        # Skipped when resuming: the resumed checkpoint already carries whatever warm start
        # it was trained with, and re-running BC here would be discarded work.
        if config.bc.enabled and not config.train.resume:
            from tmai.agents.bc import pretrain_policy
            from tmai.training.demos import load_demonstrations

            demos = load_demonstrations(
                config.bc.demo_paths,
                observation_dim=learner.observation_dim,
                action_dim=int(env.action_space.shape[0]),
            )
            logger.info("behaviour cloning: %d demonstration steps from %d file(s)",
                        len(demos), len(config.bc.demo_paths))
            stats = pretrain_policy(
                learner,
                demos,
                epochs=config.bc.epochs,
                batch_size=config.bc.batch_size,
                lr=config.bc.lr,
                val_fraction=config.bc.val_fraction,
                shuffle=config.bc.shuffle,
                seed=config.train.seed,
                log_every=max(1, config.bc.epochs // 5),
            )
            run_logger.log_event("bc_pretrain", **stats)

        trainer = Trainer(
            config, env, learner, buffer, run_logger,
            held_out_env=held_out_env, curriculum=curriculum, director=director,
        )
        try:
            return trainer.train()
        finally:
            env.close()
            if held_out_env is not None:
                held_out_env.close()


__all__ = ["EpisodeAccumulator", "Trainer", "TrainerResult", "train_from_config"]
