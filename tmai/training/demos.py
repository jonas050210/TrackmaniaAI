"""Human demonstrations: record them, load them, and learn from them.

A demonstration is a lap of ``(observation, action)`` pairs -- what the car saw and what
actually drove it. Recording works against the real game: the human drives, and the
*game-reported* inputs (``SceneVehicleCarState.input_steer`` / ``input_gas`` /
``input_brake``, read back through :class:`~tmai.game.protocol.VehicleState`) are what get
stored, so a demonstration captures the human's real control, not whatever the AI happened
to output. Against the simulated driver the same fields echo the actions passed to
``step()``, which is what makes the whole path testable without the game.

The file format is JSON Lines -- one transition per line, flushed as it is recorded -- so
an interrupted recording still leaves a usable (partial) dataset behind.

Behaviour cloning itself lives in :mod:`tmai.agents.bc`; this module is the data layer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tmai.game.protocol import Action, GameFrame

logger = logging.getLogger(__name__)

DEMO_SCHEMA_VERSION = 1


@dataclass
class Demonstration:
    """One recorded lap: aligned observation/action arrays plus telemetry context."""

    observations: np.ndarray  # (N, obs_dim) float32
    actions: np.ndarray  # (N, action_dim) float32
    #: World positions, (N, 3), for replay/ghost visualisation.
    positions: np.ndarray | None = None
    #: Forward speeds, (N,), m/s.
    speeds: np.ndarray | None = None
    #: Per-step rewards, (N,).
    rewards: np.ndarray | None = None
    #: Race time at each step, (N,), seconds.
    race_times: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        if self.observations.ndim != 2:
            raise ValueError(f"observations must be (N, obs_dim), got {self.observations.shape}")
        if self.actions.ndim != 2:
            raise ValueError(f"actions must be (N, action_dim), got {self.actions.shape}")
        if self.observations.shape[0] != self.actions.shape[0]:
            raise ValueError(
                f"observations and actions must have the same length, got "
                f"{self.observations.shape[0]} and {self.actions.shape[0]}"
            )
        for name in ("positions", "speeds", "rewards", "race_times"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value, dtype=np.float64)
                if value.shape[0] != self.observations.shape[0]:
                    raise ValueError(
                        f"{name} must have one entry per step, got {value.shape[0]}"
                    )
                setattr(self, name, value)

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    @property
    def observation_dim(self) -> int:
        return int(self.observations.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DEMO_SCHEMA_VERSION,
            "num_steps": len(self),
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "observations": self.observations.tolist(),
            "actions": self.actions.tolist(),
            "positions": None if self.positions is None else self.positions.tolist(),
            "speeds": None if self.speeds is None else self.speeds.tolist(),
            "rewards": None if self.rewards is None else self.rewards.tolist(),
            "race_times": None if self.race_times is None else self.race_times.tolist(),
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        """Write as JSON Lines, one transition per line (crash-safe, append-friendly)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"header": self.as_dict()["metadata"],
                            "schema_version": DEMO_SCHEMA_VERSION,
                            "observation_dim": self.observation_dim,
                            "action_dim": self.action_dim}) + "\n"
            )
            for i in range(len(self)):
                row: dict[str, Any] = {
                    "observation": self.observations[i].tolist(),
                    "action": self.actions[i].tolist(),
                }
                if self.positions is not None:
                    row["position"] = self.positions[i].tolist()
                if self.speeds is not None:
                    row["speed"] = float(self.speeds[i])
                if self.rewards is not None:
                    row["reward"] = float(self.rewards[i])
                if self.race_times is not None:
                    row["race_time"] = float(self.race_times[i])
                handle.write(json.dumps(row) + "\n")
        logger.info("saved demonstration: %d steps -> %s", len(self), path)
        return path

    @staticmethod
    def load(path: str | Path) -> Demonstration:
        """Load a JSONL demonstration, skipping torn trailing lines."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"demonstration not found: {path}")
        observations: list[list[float]] = []
        actions: list[list[float]] = []
        positions: list[list[float]] = []
        speeds: list[float] = []
        rewards: list[float] = []
        race_times: list[float] = []
        metadata: dict[str, Any] = {}
        header_seen = False
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed demonstration line in %s", path)
                continue
            if "header" in row and not header_seen:
                header_seen = True
                metadata = dict(row.get("header") or {})
                continue
            observations.append([float(v) for v in row["observation"]])
            actions.append([float(v) for v in row["action"]])
            positions.append([float(v) for v in row.get("position", (0.0, 0.0, 0.0))])
            speeds.append(float(row.get("speed", 0.0)))
            rewards.append(float(row.get("reward", 0.0)))
            race_times.append(float(row.get("race_time", 0.0)))
        if not observations:
            raise ValueError(f"demonstration {path} contains no transitions")
        return Demonstration(
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            positions=np.asarray(positions, dtype=np.float64),
            speeds=np.asarray(speeds, dtype=np.float64),
            rewards=np.asarray(rewards, dtype=np.float64),
            race_times=np.asarray(race_times, dtype=np.float64),
            metadata=metadata,
        )


def load_demonstrations(
    paths: Iterable[str | Path],
    *,
    observation_dim: int | None = None,
    action_dim: int | None = None,
) -> Demonstration:
    """Load and concatenate several demonstration files into one dataset.

    Every file must agree with the expected dimensions (when given); a mismatch is a
    configuration error, not data to be silently truncated.
    """
    demos = [Demonstration.load(p) for p in paths]
    if not demos:
        raise ValueError("no demonstration files given")
    for demo in demos:
        if observation_dim is not None and demo.observation_dim != observation_dim:
            raise ValueError(
                f"demonstration has observation_dim {demo.observation_dim}, expected "
                f"{observation_dim}; the demo was recorded against a different observation "
                "layout"
            )
        if action_dim is not None and demo.action_dim != action_dim:
            raise ValueError(
                f"demonstration has action_dim {demo.action_dim}, expected {action_dim}"
            )
    return Demonstration(
        observations=np.concatenate([d.observations for d in demos], axis=0),
        actions=np.concatenate([d.actions for d in demos], axis=0),
        positions=(
            np.concatenate([d.positions for d in demos], axis=0)
            if all(d.positions is not None for d in demos)
            else None
        ),
        speeds=(
            np.concatenate([d.speeds for d in demos])
            if all(d.speeds is not None for d in demos)
            else None
        ),
        rewards=(
            np.concatenate([d.rewards for d in demos])
            if all(d.rewards is not None for d in demos)
            else None
        ),
        race_times=(
            np.concatenate([d.race_times for d in demos])
            if all(d.race_times is not None for d in demos)
            else None
        ),
        metadata={
            "files": [str(p) for p in paths],
            "num_files": len(demos),
            "steps_per_file": [len(d) for d in demos],
        },
    )


class DemonstrationRecorder:
    """Records one episode of (observation, game-reported action, telemetry) tuples.

    The action stored is the one the *game* reports (``VehicleState.input_*``), which is
    the human's real input when a human is driving the real game.
    """

    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.positions: list[np.ndarray] = []
        self.speeds: list[float] = []
        self.rewards: list[float] = []
        self.race_times: list[float] = []

    def record_step(
        self,
        observation: np.ndarray,
        frame: GameFrame,
        reward: float,
    ) -> None:
        vehicle = frame.vehicle
        action = Action(
            steer=float(vehicle.input_steer),
            throttle=float(vehicle.input_gas),
            brake=float(vehicle.input_brake),
        )
        self.observations.append(np.asarray(observation, dtype=np.float32).reshape(-1))
        self.actions.append(action.as_array())
        self.positions.append(np.asarray(vehicle.position, dtype=np.float64).copy())
        self.speeds.append(float(vehicle.speed_forward))
        self.rewards.append(float(reward))
        self.race_times.append(float(frame.race.race_time))

    def build(self, metadata: dict[str, Any] | None = None) -> Demonstration:
        if not self.observations:
            raise ValueError("nothing recorded: record at least one step")
        meta = {
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "source": "demonstration",
            **(metadata or {}),
        }
        return Demonstration(
            observations=np.asarray(self.observations, dtype=np.float32),
            actions=np.asarray(self.actions, dtype=np.float32),
            positions=np.asarray(self.positions, dtype=np.float64),
            speeds=np.asarray(self.speeds, dtype=np.float64),
            rewards=np.asarray(self.rewards, dtype=np.float64),
            race_times=np.asarray(self.race_times, dtype=np.float64),
            metadata=meta,
        )


def record_demonstration(
    env,
    *,
    out_path: str | Path,
    max_steps: int = 5000,
    action_provider=None,
    should_stop=None,
    metadata: dict[str, Any] | None = None,
) -> Demonstration:
    """Drive one lap and record it as a demonstration.

    Args:
        env: a :class:`~tmai.env.tm_env.TrackmaniaEnv` (or the multi-track wrapper). A
            human is expected to be at the wheel of the real game; ``action_provider``
            supplies the AI-side inputs (neutral by default -- the game reports what
            actually happened through the input read-back).
        out_path: where to write the JSONL dataset.
        max_steps: safety cap on the recording length.
        action_provider: called each step with the latest frame; defaults to no input.
        should_stop: polled each step; return True to stop early (Ctrl-C friendly).

    Returns:
        The recorded :class:`Demonstration`, also saved to ``out_path``.
    """
    from tmai.game.protocol import neutral_action

    provide = action_provider or (lambda frame: neutral_action())
    recorder = DemonstrationRecorder()

    observation, info = env.reset()
    frame = env.last_frame
    track_name = str(info.get("track", getattr(env, "track", None) and env.track.name or ""))
    recorder.record_step(observation, frame, 0.0)

    steps = 0
    try:
        while steps < max_steps:
            if should_stop is not None and should_stop():
                logger.info("recording stopped by operator after %d steps", steps)
                break
            action = provide(frame)
            observation, reward, terminated, truncated, info = env.step(action)
            frame = env.last_frame
            recorder.record_step(observation, frame, reward)
            steps += 1
            if terminated or truncated:
                logger.info(
                    "episode ended (%s) after %d steps; recording complete",
                    info.get("end_reason", "?"),
                    steps,
                )
                break
    except KeyboardInterrupt:  # pragma: no cover - interactive path
        logger.info("interrupted; saving the %d steps recorded so far", steps)

    demo = recorder.build(
        metadata={
            **(metadata or {}),
            "track": track_name,
            "end_reason": str(info.get("end_reason", "")),
            "finished": bool(info.get("finished", False)),
            "steps": steps,
        }
    )
    demo.save(out_path)
    return demo


__all__ = [
    "DEMO_SCHEMA_VERSION",
    "Demonstration",
    "DemonstrationRecorder",
    "load_demonstrations",
    "record_demonstration",
]
