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
from tmai.tracks.library import TrackLibrary
from tmai.tracks.synthetic import build_synthetic

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """The configuration cannot be turned into working objects."""


def build_library(config: RunConfig) -> TrackLibrary:
    """Build the track library described by ``config.track``.

    Single-track configurations produce a one-track library rather than a special case, so
    everything downstream (sampling, evaluation, manifests) has exactly one code path.
    """
    spec = config.track
    weights = dict(spec.split_weights)

    if spec.directory:
        library = TrackLibrary.from_directory(
            spec.directory,
            split_weights=weights,
            pattern=spec.pattern,
            explicit_splits=spec.explicit_splits,
            recursive=spec.recursive,
        )
        logger.info("track library: %s", library.summary())
        return library

    library = TrackLibrary(split_weights=weights)
    if spec.path:
        track = CenterlineTrack.load(spec.path)
        logger.info(
            "loaded track %r (%d points, %.1f m)", track.name, track.num_points, track.length
        )
        library.add(track, split="train", source=str(spec.path))
        return library

    if spec.synthetic_suite:
        for entry in spec.synthetic_suite:
            name = entry.get("name") if isinstance(entry, dict) else str(entry)
            kwargs = entry.get("kwargs", {}) if isinstance(entry, dict) else {}
            track = build_synthetic(name, **kwargs)
            library.add(track)
        logger.warning(
            "using %d SYNTHETIC tracks; these are not real Trackmania maps", len(library)
        )
        return library

    if spec.synthetic:
        track = build_synthetic(spec.synthetic, **spec.synthetic_kwargs)
        logger.warning(
            "using SYNTHETIC track %r (%.1f m); this is not a real Trackmania map",
            track.name,
            track.length,
        )
        library.add(track, split="train")
        return library

    raise ConfigError(
        "no track configured: set track.path, track.directory, track.synthetic or "
        "track.synthetic_suite"
    )


def build_track(config: RunConfig) -> CenterlineTrack:
    """Load the single track a run is configured for.

    Kept for the single-track paths (``tmai eval``, ``tmai show-track``). Multi-track
    training goes through :func:`build_library`.
    """
    library = build_library(config)
    train = library.train
    if train:
        return train[0]
    # A library built purely from a directory can legitimately have no train split when the
    # weights put everything elsewhere; fall back to whatever is present.
    if library.entries:
        return library.entries[0].track
    raise ConfigError("the track library is empty")


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

    if config.normalize.enabled:
        from tmai.agents.normalize import NormalizingLearner
        from tmai.env.normalization import RunningNormalizer

        learner = NormalizingLearner(
            learner,
            RunningNormalizer(
                int(learner.observation_dim),
                clip=config.normalize.clip,
                epsilon=config.normalize.epsilon,
                warmup_steps=config.normalize.warmup_steps,
            ),
        )
        logger.info(
            "observation normalisation enabled (clip=%.1f, warmup=%d steps)",
            config.normalize.clip,
            config.normalize.warmup_steps,
        )
    return learner


def build_buffer(env: TrackmaniaEnv, config: RunConfig) -> ReplayBuffer:
    return ReplayBuffer(
        observation_dim=env.observation_dim,
        action_dim=int(env.action_space.shape[0]),
        config=config.replay,
    )


def build_multi_track_env(
    config: RunConfig,
    library: TrackLibrary,
    *,
    split: str = "train",
    seed: int | None = None,
):
    """Build a :class:`MultiTrackEnv` over one split of ``library``.

    Falls back to a plain :class:`TrackmaniaEnv` when the split holds a single track, so a
    one-map run does not pay for machinery it cannot use.
    """
    from tmai.env.multi_track import MultiTrackConfig, MultiTrackEnv

    entries = library.by_split(split)
    if not entries:
        raise ConfigError(
            f"the {split!r} split is empty (library has {library.counts()}); check "
            "track.split_weights and track.explicit_splits"
        )
    if len(entries) == 1:
        track = entries[0].track
        logger.info("single track in %r split: %s", split, track.name)
        return TrackmaniaEnv(build_driver(config, track), track, config.env)

    spec = config.multi
    return MultiTrackEnv(
        library.sampler(split, seed=seed),
        lambda track: build_driver(config, track),
        config.env,
        MultiTrackConfig(
            sample_tracks=spec.sample_tracks,
            random_start_station=spec.random_start_station,
            start_station_fraction=spec.start_station_fraction,
            start_lateral_std=spec.start_lateral_std,
            start_edge_margin=spec.start_edge_margin,
        ),
        seed=seed,
        split=split,
        track_identities={e.track.name: e.identity for e in entries},
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
    "build_library",
    "build_multi_track_env",
    "build_track",
]
