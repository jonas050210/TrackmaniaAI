"""Training across many tracks instead of one.

Generalisation cannot be retro-fitted. A policy trained on a single map learns that map,
including everything about it that has nothing to do with driving: where the corners are,
which curvature follows which. The only structural defence is to sample a different track at
each episode reset, from the very first step of training.

This wrapper owns that sampling. It keeps the single-track :class:`TrackmaniaEnv` untouched
-- the multi-track case is a *wrapper* rather than a rewrite, so every test and every piece
of logic written for one track applies unchanged.

Anti-memorisation mechanisms, and what each one actually defends against:

* **Track sampling per episode.** The policy cannot key on track identity, because identity
  is not in the observation and the track changes underneath it.
* **Random start station.** Episodes begin at a random arc-length position rather than
  always the start line. Without this, the start of the lap becomes a memorised cue and the
  policy never learns to drive the middle of a track it has not seen from the beginning.
* **Random lateral offset.** Starting slightly off the racing line forces recovery
  behaviour, which is most of what "driving a new track" actually consists of.
* **No absolute coordinates.** Enforced by the observation encoder, and covered by a test
  that translates a track and asserts the observation is unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from tmai.env.tm_env import EnvConfig, TrackmaniaEnv
from tmai.game.protocol import GameDriver
from tmai.tracks.centerline import CenterlineTrack
from tmai.tracks.library import TrackSampler

logger = logging.getLogger(__name__)


@dataclass
class MultiTrackConfig:
    """How tracks and start conditions are sampled between episodes."""

    #: Sample a new track on every reset. Turning this off makes the wrapper a plain
    #: single-track environment with start randomisation, which is a useful ablation.
    sample_tracks: bool = True
    #: Sample a random start station along the centreline.
    random_start_station: bool = True
    #: Fraction of the lap the random start may cover. ``1.0`` is anywhere on the track.
    start_station_fraction: float = 1.0
    #: Standard deviation of the random lateral start offset, metres.
    start_lateral_std: float = 1.5
    #: Metres of the corridor edge kept clear of the random start offset, so a randomised
    #: start never begins the episode already terminated.
    start_edge_margin: float = 1.0
    #: Raise on reset if a track cannot be prepared. When False, fall back to the start line.
    strict: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_tracks": self.sample_tracks,
            "random_start_station": self.random_start_station,
            "start_station_fraction": self.start_station_fraction,
            "start_lateral_std": self.start_lateral_std,
            "start_edge_margin": self.start_edge_margin,
            "strict": self.strict,
        }


@dataclass
class EpisodeContext:
    """Which track and start condition the current episode is using."""

    track_name: str = ""
    track_identity: str = ""
    split: str = ""
    start_station: float = 0.0
    start_lateral: float = 0.0
    track_length: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "track": self.track_name,
            "track_identity": self.track_identity,
            "split": self.split,
            "start_station": round(self.start_station, 2),
            "start_lateral": round(self.start_lateral, 3),
            "track_length": round(self.track_length, 2),
        }


class MultiTrackEnv(gym.Env):
    """Samples a track and a start condition per episode, then delegates to ``TrackmaniaEnv``.

    Args:
        sampler: yields the track for each episode.
        driver_factory: builds a driver for a given track. Called once per distinct track and
            the result is cached, because opening a real game connection is expensive.
        config: the single-track environment configuration.
        multi_config: track and start-condition sampling settings.
        seed: RNG seed for reproducible track and start sampling.
        split: label recorded in the episode context (``"train"``, ``"validation"``, ...).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        sampler: TrackSampler,
        driver_factory: Callable[[CenterlineTrack], GameDriver],
        config: EnvConfig | None = None,
        multi_config: MultiTrackConfig | None = None,
        *,
        seed: int | None = None,
        split: str = "train",
        track_identities: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.sampler = sampler
        self.driver_factory = driver_factory
        self.config = config or EnvConfig()
        self.multi_config = multi_config or MultiTrackConfig()
        self.split = split
        self._identities = dict(track_identities or {})
        self._rng = np.random.default_rng(seed)
        self._drivers: dict[int, GameDriver] = {}
        self._inner: TrackmaniaEnv | None = None
        self._context = EpisodeContext()
        #: Track pinned for the next reset by :meth:`select_track`; ``None`` means sample.
        self._pinned: CenterlineTrack | None = None
        # -- curriculum ---------------------------------------------------------------
        #: Attached curriculum (training only), or ``None`` for "always everything".
        self._curriculum = None
        #: Training step the curriculum is resolved at; updated by the trainer.
        self._curriculum_step = 0
        #: All tracks in this environment, in sampler order (the curriculum filter draws
        #: from this list with its own permutation, leaving the sampler untouched).
        self._all_tracks = self.sampler.peek_all()
        #: Permutation state for curriculum-filtered sampling.
        self._filtered_order: list[int] = []

        # Build one environment up front so the action/observation spaces are real rather
        # than guessed. The track it gets is replaced on the first reset.
        first = self.sampler.peek_all()[0]
        self._inner = TrackmaniaEnv(
            self._driver_for(first), first, self.config, own_driver=False
        )
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space

    # -- gymnasium plumbing ---------------------------------------------------------

    @property
    def observation_dim(self) -> int:
        return self._inner_env.observation_dim

    @property
    def observation_names(self) -> list[str]:
        return self._inner_env.observation_names

    @property
    def _inner_env(self) -> TrackmaniaEnv:
        if self._inner is None:
            raise RuntimeError("MultiTrackEnv has not been initialised")
        return self._inner

    @property
    def track(self) -> CenterlineTrack:
        """The track the current episode is running on."""
        return self._inner_env.track

    @property
    def episode_context(self) -> EpisodeContext:
        return self._context

    @property
    def driver(self) -> GameDriver:
        return self._inner_env.driver

    @property
    def end_reason(self) -> Any:
        return self._inner_env.end_reason

    @property
    def last_frame(self) -> Any:
        return self._inner_env.last_frame

    @property
    def last_projection(self) -> Any:
        return self._inner_env.last_projection

    def describe(self) -> dict[str, Any]:
        return {
            "multi_track": {
                "num_tracks": len(self.sampler),
                "split": self.split,
                "config": self.multi_config.to_dict(),
                "draws": self.sampler.draws,
            },
            "current": self._context.as_dict(),
            **self._inner_env.describe(),
        }

    def select_track(self, name: str) -> CenterlineTrack:
        """Pin the *next* reset to the track called ``name``.

        Used by exhaustive evaluation, which must visit specific tracks rather than whatever
        the sampler produces. The pin is consumed by the next :meth:`reset` and then cleared,
        so a pinned evaluation cannot silently leak into subsequent training episodes.
        """
        for track in self.sampler.peek_all():
            if track.name == name:
                self._pinned = track
                return track
        available = [t.name for t in self.sampler.peek_all()]
        raise KeyError(f"no track named {name!r} in this environment; available: {available}")

    @property
    def pinned_track(self) -> CenterlineTrack | None:
        return self._pinned

    # -- curriculum -----------------------------------------------------------------

    def attach_curriculum(self, curriculum) -> None:
        """Attach a :class:`~tmai.training.curriculum.Curriculum` (training only).

        Held-out evaluation must not be curriculum-filtered, so this is attached to the
        training environment only; the held-out environment keeps sampling everything.
        """
        self._curriculum = curriculum

    def set_curriculum_step(self, step: int) -> None:
        """Tell the environment the current training step, for curriculum resolution."""
        self._curriculum_step = int(step)

    @property
    def curriculum_stage(self) -> int | None:
        if self._curriculum is None:
            return None
        return self._curriculum.stage_index_at(self._curriculum_step)

    def _curriculum_filter(self) -> list[CenterlineTrack] | None:
        """The tracks the curriculum currently allows, or ``None`` for all of them."""
        if self._curriculum is None:
            return None
        names = self._curriculum.active_track_names(self._curriculum_step)
        if names is None:
            return None
        wanted = set(names)
        return [t for t in self._all_tracks if t.name in wanted]

    def _curriculum_step_cap(self) -> int | None:
        if self._curriculum is None:
            return None
        return self._curriculum.episode_max_steps(
            self._curriculum_step, self.config.termination.max_steps
        )

    def _pick_track(self) -> CenterlineTrack:
        """Pick the next track: pinned, curriculum-filtered, or the plain sampler."""
        if self._pinned is not None:
            track, self._pinned = self._pinned, None
            return track
        allowed = self._curriculum_filter()
        if allowed is None:
            if self.multi_config.sample_tracks:
                return self.sampler.next()
            return self.sampler.peek_all()[0]
        # Curriculum-filtered sampling: a shuffled block permutation over the allowed
        # tracks only, so every revealed track is still seen regularly.
        if not self.multi_config.sample_tracks:
            return allowed[0]
        if not self._filtered_order:
            self._filtered_order = list(self._rng.permutation(len(allowed)))
        index = self._filtered_order.pop()
        return allowed[index]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        track = self._pick_track()
        self._switch_to(track)

        station, lateral = self._sample_start(track)
        self._context = EpisodeContext(
            track_name=track.name,
            track_identity=self._identities.get(track.name, ""),
            split=self.split,
            start_station=station,
            start_lateral=lateral,
            track_length=track.length,
        )

        reset_options: dict[str, Any] = {"start_station": station, "start_lateral": lateral}
        cap = self._curriculum_step_cap()
        if cap is not None:
            reset_options["max_steps"] = cap
        observation, info = self._inner_env.reset(options=reset_options)
        info.update(self._context.as_dict())
        info["track"] = track.name
        if self._curriculum is not None:
            info["curriculum_stage"] = self._curriculum.stage_index_at(self._curriculum_step)
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self._inner_env.step(action)
        # The episode context must survive the whole episode, not just reset(): per-episode
        # logging and the evaluation report both need to know which track was being driven.
        info.update(self._context.as_dict())
        info["track"] = self._context.track_name
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        for driver in self._drivers.values():
            try:
                driver.close()
            except Exception:  # noqa: BLE001 - close must never raise
                logger.debug("error closing driver", exc_info=True)
        self._drivers.clear()

    def render(self) -> Any:
        return self._inner_env.render()

    # -- internals ------------------------------------------------------------------

    def _driver_for(self, track: CenterlineTrack) -> GameDriver:
        """Return a cached driver for ``track``, building it on first use."""
        key = id(track)
        driver = self._drivers.get(key)
        if driver is None:
            driver = self.driver_factory(track)
            self._drivers[key] = driver
        return driver

    def _switch_to(self, track: CenterlineTrack) -> None:
        """Point the inner environment at ``track`` without rebuilding it.

        The encoder, reward and termination tracker all hold a reference to the track, so all
        four have to be rebound together. Rebuilding the environment instead would be simpler
        but would silently discard the driver connection.
        """
        from tmai.env.observation import ObservationEncoder
        from tmai.env.reward import ProgressReward
        from tmai.env.termination import TerminationTracker

        inner = self._inner_env
        inner.track = track
        inner._encoder = ObservationEncoder(track, self.config.observation)  # noqa: SLF001
        inner._reward_fn = ProgressReward(track, self.config.reward)  # noqa: SLF001
        inner._termination = TerminationTracker(track, self.config.termination)  # noqa: SLF001
        inner.driver = self._driver_for(track)  # noqa: SLF001

    def _sample_start(self, track: CenterlineTrack) -> tuple[float, float]:
        """Choose ``(start_station, lateral_offset)`` for the next episode."""
        cfg = self.multi_config
        station = 0.0
        lateral = 0.0

        if cfg.random_start_station:
            span = track.length * float(np.clip(cfg.start_station_fraction, 0.0, 1.0))
            station = float(self._rng.uniform(0.0, max(0.0, span)))

        if cfg.start_lateral_std > 0:
            half_width = track.corridor_half_width_at(station)
            limit = max(0.0, half_width - cfg.start_edge_margin)
            lateral = float(np.clip(self._rng.normal(0.0, cfg.start_lateral_std), -limit, limit))

        return station, lateral


__all__ = ["EpisodeContext", "MultiTrackConfig", "MultiTrackEnv"]
