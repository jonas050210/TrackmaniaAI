"""The Trackmania reinforcement-learning environment.

A thin, explicit gymnasium environment over a :class:`~tmai.game.protocol.GameDriver` and a
:class:`~tmai.tracks.centerline.CenterlineTrack`. It owns exactly three pieces of logic --
observation encoding, reward and termination -- each delegated to its own module so they can
be reasoned about and tested independently.

Timing: the real game advances at its own physics rate (100 Hz in Trackmania, possibly
multiplied by the game-speed factor). This environment therefore does **not** try to
regulate real time; it is lock-step with the driver. One ``step()`` equals
``action_repeat`` physics ticks. Real-time pacing, when it is wanted for evaluation against
a human, belongs in a wrapper on top.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tmai.env.observation import (
    OBSERVATION_VERSION,
    ObservationEncoder,
    ObservationInputs,
    ObservationSpec,
)
from tmai.env.reward import ProgressReward, RewardBreakdown, RewardConfig
from tmai.env.termination import (
    EndReason,
    TerminationConfig,
    TerminationTracker,
)
from tmai.game.errors import UnsupportedFeatureError
from tmai.game.protocol import Action, GameDriver, GameFrame
from tmai.tracks.centerline import CenterlineTrack, TrackProjection

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """Environment configuration."""

    #: Nominal control period, seconds. Used for the yaw-rate estimate and for reporting.
    control_dt: float = 0.05
    #: Physics ticks per environment step. Higher values cut the effective control rate,
    #: which is cheaper and often stabilises learning.
    action_repeat: int = 1
    observation: ObservationSpec = field(default_factory=ObservationSpec)
    reward: RewardConfig = field(default_factory=RewardConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    #: Metres of projection search window around the previous progress. Small windows keep
    #: projection O(1) but can lose the car if it teleports (respawn).
    projection_window: float = 120.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_version": OBSERVATION_VERSION,
            "control_dt": self.control_dt,
            "action_repeat": self.action_repeat,
            "projection_window": self.projection_window,
            "observation": self.observation.to_dict(),
            "reward": self.reward.to_dict(),
            "termination": {
                "max_steps": self.termination.max_steps,
                "off_track_limit": self.termination.off_track_limit,
                "off_track_margin": self.termination.off_track_margin,
                "stall_limit": self.termination.stall_limit,
                "min_progress_per_step": self.termination.min_progress_per_step,
                "min_steps_before_stall": self.termination.min_steps_before_stall,
                "no_ground_contact_limit": self.termination.no_ground_contact_limit,
            },
        }


class TrackmaniaEnv(gym.Env):
    """Gymnasium environment that drives Trackmania.

    Args:
        driver: an unopened or opened :class:`GameDriver`. The environment opens it lazily
            and closes it in :meth:`close` when it created it.
        track: centreline of the map the game is currently running.
        config: environment configuration.
        own_driver: close the driver in :meth:`close` (default True).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        driver: GameDriver,
        track: CenterlineTrack,
        config: EnvConfig | None = None,
        *,
        own_driver: bool = True,
    ) -> None:
        super().__init__()
        self.driver = driver
        self.track = track
        self.config = config or EnvConfig()
        self._own_driver = own_driver

        self._encoder = ObservationEncoder(track, self.config.observation)
        self._reward_fn = ProgressReward(track, self.config.reward)
        self._termination = TerminationTracker(track, self.config.termination)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._encoder.dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._frame: GameFrame | None = None
        self._projection: TrackProjection | None = None
        self._prev_projection: TrackProjection | None = None
        self._prev_yaw: float | None = None
        self._last_action = Action()
        self._episode_return = 0.0
        self._episode_steps = 0
        self._episode_progress = 0.0
        self._end_reason = EndReason.RUNNING
        self._nonfinite_observations = 0
        self._t_episode_start = 0.0
        self._reward_breakdown: RewardBreakdown = RewardBreakdown()

    # -- introspection --------------------------------------------------------------

    @property
    def observation_dim(self) -> int:
        return self._encoder.dim

    @property
    def observation_names(self) -> list[str]:
        return self._encoder.spec.names()

    def describe(self) -> dict[str, Any]:
        return {
            "track": {
                "name": self.track.name,
                "uid": self.track.uid,
                "length": self.track.length,
                "points": self.track.num_points,
                "closed": self.track.closed,
            },
            "driver": self.driver.describe(),
            "config": self.config.to_dict(),
        }

    # -- gymnasium API --------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._open_driver_if_needed()

        frame = self.driver.reset()
        frame = self._apply_start_options(frame, options)
        self._frame = frame
        projection = self.track.project(frame.vehicle.position)
        self._projection = projection
        self._prev_projection = None
        self._prev_yaw = frame.vehicle.yaw()
        self._last_action = Action()
        self._episode_return = 0.0
        self._episode_steps = 0
        self._episode_progress = 0.0
        self._end_reason = EndReason.RUNNING
        self._reward_breakdown = RewardBreakdown()
        self._termination.reset()
        self._t_episode_start = time.monotonic()

        obs = self._encode(frame, projection, yaw_rate=0.0)
        return obs, self._info(projection, RewardBreakdown(), EndReason.RUNNING)

    def step(
        self, action: np.ndarray | Action
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._frame is None or self._projection is None:
            raise RuntimeError("TrackmaniaEnv.reset() must be called before step()")

        act = self._coerce_action(action)
        self._last_action = act

        cfg = self.config
        total_reward = 0.0
        breakdown = RewardBreakdown()
        frame = self._frame
        projection = self._projection
        dt = cfg.control_dt * max(1, cfg.action_repeat)

        for _ in range(max(1, cfg.action_repeat)):
            frame = self.driver.step(act)
            prev_progress = projection.progress
            projection = self.track.project(
                frame.vehicle.position,
                hint_s=prev_progress,
                search_window=cfg.projection_window,
            )
            breakdown = self._reward_fn.compute(
                frame=frame,
                projection=projection,
                prev_progress=prev_progress,
                finished=frame.race.finished,
                dt=cfg.control_dt,
            )
            total_reward += breakdown.total

        yaw = frame.vehicle.yaw()
        yaw_rate = self._yaw_rate(self._prev_yaw, yaw, dt)
        self._prev_yaw = yaw

        result = self._termination.update(
            frame=frame,
            projection=projection,
            progress_delta=projection.progress - self._projection.progress,
        )

        self._frame = frame
        self._prev_projection = self._projection
        self._projection = projection
        self._episode_return += total_reward
        self._episode_steps += 1
        self._episode_progress += breakdown.progress_metres
        self._end_reason = result.reason
        self._reward_breakdown = breakdown

        obs = self._encode(frame, projection, yaw_rate=yaw_rate)
        return obs, float(total_reward), result.terminated, result.truncated, self._info(
            projection, breakdown, result.reason
        )

    def close(self) -> None:
        if self._own_driver:
            try:
                self.driver.close()
            except Exception:  # noqa: BLE001 - close must not raise
                logger.debug("error closing driver", exc_info=True)

    def render(self) -> Any:
        """Rendering is handled by :mod:`tmai.viz`, which reads ``env.last_frame``."""
        return None

    # -- internals ------------------------------------------------------------------

    @property
    def last_frame(self) -> GameFrame | None:
        return self._frame

    @property
    def last_projection(self) -> TrackProjection | None:
        return self._projection

    @property
    def end_reason(self) -> EndReason:
        return self._end_reason

    def _open_driver_if_needed(self) -> None:
        if not self.driver.is_connected():
            self.driver.open()

    def _apply_start_options(
        self, frame: GameFrame, options: dict[str, Any] | None
    ) -> GameFrame:
        """Honour ``start_station`` / ``start_lateral`` reset options, if the driver can.

        A driver that cannot reposition the car raises :class:`UnsupportedFeatureError`, and
        that is surfaced rather than swallowed: silently starting at the start line would
        make a run report randomised starts that never happened.
        """
        if not options:
            return frame
        station = options.get("start_station")
        lateral = float(options.get("start_lateral", 0.0))
        if station is None and not lateral:
            return frame
        if not self.driver.capabilities.supports_start_repositioning:
            raise UnsupportedFeatureError(
                f"driver {self.driver.name!r} cannot place the car at an arbitrary track "
                "station, but start randomisation was requested",
                remedy=(
                    "Set multi.random_start_station=false and multi.start_lateral_std=0, "
                    "or use a driver that supports start repositioning."
                ),
            )
        return self.driver.reposition(float(station or 0.0), lateral)

    def _coerce_action(self, action: np.ndarray | Action) -> Action:
        if isinstance(action, Action):
            return action.clipped()
        arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if arr.size != 3:
            raise ValueError(
                f"action must have 3 components (steer, throttle, brake), got {arr.size}"
            )
        return Action(float(arr[0]), float(arr[1]), float(arr[2])).clipped()

    @staticmethod
    def _yaw_rate(prev_yaw: float | None, yaw: float, dt: float) -> float:
        if prev_yaw is None or dt <= 0:
            return 0.0
        delta = yaw - prev_yaw
        delta = float(np.arctan2(np.sin(delta), np.cos(delta)))
        return delta / dt

    def _encode(
        self, frame: GameFrame, projection: TrackProjection, *, yaw_rate: float
    ) -> np.ndarray:
        obs = self._encoder.encode(
            ObservationInputs(
                frame=frame,
                projection=projection,
                prev_projection=self._prev_projection,
                prev_yaw=self._prev_yaw,
                dt=self.config.control_dt * max(1, self.config.action_repeat),
                last_action=self._last_action,
                yaw_rate=yaw_rate,
            )
        )
        if not np.all(np.isfinite(obs)):
            self._nonfinite_observations += 1
        return obs

    def _info(
        self,
        projection: TrackProjection,
        breakdown: RewardBreakdown,
        reason: EndReason,
    ) -> dict[str, Any]:
        """Per-step diagnostics.

        Everything here must be a deterministic function of the environment state: the
        gymnasium contract requires ``info`` to be reproducible for a given seed and action,
        and ``gymnasium.utils.env_checker.check_env`` asserts it. Wall-clock throughput
        therefore lives in the trainer's metrics, not here.
        """
        frame = self._frame
        assert frame is not None
        info: dict[str, Any] = {
            "progress": projection.progress,
            "progress_fraction": projection.progress / max(self.track.length, 1e-6),
            "lateral_offset": projection.lateral_offset,
            "distance_to_centerline": projection.distance,
            "speed_forward": frame.vehicle.speed_forward,
            "speed_sideward": frame.vehicle.speed_sideward,
            "rpm": frame.vehicle.rpm,
            "gear": frame.vehicle.gear,
            "is_sliding": frame.vehicle.is_sliding,
            "race_time": frame.race.race_time,
            "finished": frame.race.finished,
            "checkpoint_index": frame.race.checkpoint_index,
            "checkpoint_total": frame.race.checkpoint_total,
            "episode_return": self._episode_return,
            "episode_steps": self._episode_steps,
            "episode_progress": self._episode_progress,
            "end_reason": reason.value,
            "driver": self.driver.name,
            "nonfinite_observations": self._nonfinite_observations,
        }
        info.update(breakdown.as_metrics())
        return info


__all__ = ["EnvConfig", "TrackmaniaEnv"]
