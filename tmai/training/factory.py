"""Construct the runtime objects described by a :class:`~tmai.config.RunConfig`.

This is the only place where configuration becomes live objects, which keeps the trainer free
of construction details and makes the wiring itself testable.

It is also the single gate for the simulated driver: refusing to build it unless the operator
explicitly allowed it is what prevents a run against the toy model from being mistaken for a
run against the real game.
"""

from __future__ import annotations

import logging

import numpy as np

from tmai.agents.base import Learner
from tmai.agents.replay import ReplayBuffer
from tmai.config import RunConfig
from tmai.env.tm_env import TrackmaniaEnv
from tmai.game.protocol import GameDriver
from tmai.game.simulated import SIMULATED_DRIVER_BANNER, SimulatedGameDriver
from tmai.tracks.centerline import CenterlineTrack
from tmai.tracks.synthetic import build_synthetic

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """The configuration cannot be turned into working objects."""


def build_track(config: RunConfig) -> CenterlineTrack:
    """Load a recorded centreline, or generate a synthetic one if asked."""
    spec = config.track
    if spec.path:
        track = CenterlineTrack.load(spec.path)
        logger.info("loaded track %r (%d points, %.1f m)", track.name, track.num_points, track.length)
        return track
    if spec.synthetic:
        track = build_synthetic(spec.synthetic, **spec.synthetic_kwargs)
        logger.warning(
            "using SYNTHETIC track %r (%.1f m); this is not a real Trackmania map",
            track.name,
            track.length,
        )
        return track
    raise ConfigError(
        "no track configured: set track.path to a centreline recorded with "
        "'tmai record-track', or track.synthetic for a generated test track"
    )


def build_driver(config: RunConfig, track: CenterlineTrack) -> GameDriver:
    """Instantiate the configured game driver."""
    spec = config.driver
    kind = spec.kind.lower()

    if kind == "tminterface":
        from tmai.game.tminterface.driver import build_tminterface_driver

        logger.info(
            "using the REAL Trackmania driver (TMInterface server %r, speed x%.1f)",
            spec.server_name,
            spec.speed_ratio,
        )
        return build_tminterface_driver(
            server_name=spec.server_name,
            speed_ratio=spec.speed_ratio,
            position_scale=spec.position_scale,
            reset_strategy=spec.reset_strategy,
            reset_command=spec.reset_command,
            settle_ticks=spec.settle_ticks,
            frame_timeout_s=spec.frame_timeout_s,
            connect_timeout_s=spec.connect_timeout_s,
        )

    if kind == "simulated":
        if not spec.allow_simulated:
            raise ConfigError(
                "driver.kind='simulated' is a toy model, not Trackmania. Refusing to start. "
                "If you really mean it, set driver.allow_simulated=true "
                "(CLI: --allow-simulated-driver)."
            )
        logger.warning("%s", SIMULATED_DRIVER_BANNER)
        return SimulatedGameDriver(track)

    raise ConfigError(
        f"unknown driver.kind {spec.kind!r}; expected 'tminterface' or 'simulated'"
    )


def build_env(driver: GameDriver, track: CenterlineTrack, config: RunConfig) -> TrackmaniaEnv:
    return TrackmaniaEnv(driver, track, config.env)


def build_learner(env: TrackmaniaEnv, config: RunConfig) -> Learner:
    """Build the RL learner. SAC today; the config decides, not the trainer."""
    from tmai.agents.sac import SACLearner

    low, high = env.action_space.low, env.action_space.high
    learner = SACLearner(
        observation_dim=env.observation_dim,
        action_dim=int(env.action_space.shape[0]),
        config=config.sac,
        action_low=np.asarray(low, dtype=np.float32),
        action_high=np.asarray(high, dtype=np.float32),
        device=config.train.device,
        seed=config.train.seed,
    )
    logger.info(
        "learner: SAC, %d observations -> %d actions, %d parameters, device=%s",
        learner.observation_dim,
        learner.action_dim,
        learner.describe()["model"]["num_parameters"],
        learner.describe()["device"],
    )
    return learner


def build_buffer(env: TrackmaniaEnv, config: RunConfig) -> ReplayBuffer:
    return ReplayBuffer(
        observation_dim=env.observation_dim,
        action_dim=int(env.action_space.shape[0]),
        config=config.replay,
    )


def build_all(config: RunConfig) -> tuple[TrackmaniaEnv, Learner, ReplayBuffer, CenterlineTrack]:
    """Build the full runtime graph from configuration."""
    track = build_track(config)
    driver = build_driver(config, track)
    env = build_env(driver, track, config)
    learner = build_learner(env, config)
    buffer = build_buffer(env, config)
    return env, learner, buffer, track


__all__ = [
    "ConfigError",
    "build_all",
    "build_buffer",
    "build_driver",
    "build_env",
    "build_learner",
    "build_track",
]
