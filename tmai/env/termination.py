"""Episode termination logic.

Early termination matters for training throughput: an episode that spends 60 s stuck against
a wall produces no useful gradient and blocks the single game instance from collecting
anything else. Every rule here exists to cut such dead time, and every one of them reports a
machine-readable reason so a long run can be diagnosed from metrics.

``terminated`` means "the task ended" (finished, or unrecoverable); ``truncated`` means "we
stopped it" (time limit). The distinction is required for correct bootstrapping in
value-based RL: the learner must not bootstrap through a truncation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tmai.game.protocol import GameFrame
from tmai.tracks.centerline import CenterlineTrack, TrackProjection


class EndReason(str, Enum):
    RUNNING = "running"
    FINISHED = "finished"
    OFF_TRACK = "off_track"
    STALLED = "stalled"
    NO_GROUND_CONTACT = "no_ground_contact"
    TIME_LIMIT = "time_limit"
    GAME_ERROR = "game_error"


@dataclass
class TerminationConfig:
    """Termination thresholds, in control steps unless stated otherwise."""

    #: Hard episode length (truncation).
    max_steps: int = 2000
    #: Consecutive steps outside the corridor before giving up.
    off_track_limit: int = 40
    #: Metres beyond the corridor edge that count as "off track".
    off_track_margin: float = 1.5
    #: Consecutive steps below ``min_progress_per_step`` before declaring a stall.
    stall_limit: int = 60
    #: Minimum metres of progress per step to count as moving.
    min_progress_per_step: float = 0.05
    #: Grace period before the stall rule can fire (launch takes a moment).
    min_steps_before_stall: int = 40
    #: Consecutive steps with no ground contact before giving up.
    no_ground_contact_limit: int = 25


@dataclass
class TerminationResult:
    terminated: bool
    truncated: bool
    reason: EndReason

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class TerminationTracker:
    """Stateful per-episode termination bookkeeping. Call :meth:`reset` on episode start."""

    def __init__(self, track: CenterlineTrack, config: TerminationConfig | None = None) -> None:
        self.track = track
        self.config = config or TerminationConfig()
        self.reset()

    def reset(self) -> None:
        self._steps = 0
        self._off_track_streak = 0
        self._stall_streak = 0
        self._airborne_streak = 0

    @property
    def steps(self) -> int:
        return self._steps

    def update(
        self,
        *,
        frame: GameFrame,
        projection: TrackProjection,
        progress_delta: float,
    ) -> TerminationResult:
        """Advance one control step and decide whether the episode continues."""
        cfg = self.config
        self._steps += 1

        if frame.race.finished:
            return TerminationResult(True, False, EndReason.FINISHED)

        # Interpolated by arc length, matching the reward and the observation's edge
        # distances. Indexing by sample instead would make a car near a segment boundary
        # off-track for the reward but on-track for termination.
        half_width = self.track.corridor_half_width_at(projection.progress)
        outside = abs(projection.lateral_offset) > half_width + cfg.off_track_margin
        self._off_track_streak = self._off_track_streak + 1 if outside else 0
        if self._off_track_streak >= cfg.off_track_limit:
            return TerminationResult(True, False, EndReason.OFF_TRACK)

        if not frame.vehicle.has_ground_contact:
            self._airborne_streak += 1
        else:
            self._airborne_streak = 0
        if self._airborne_streak >= cfg.no_ground_contact_limit:
            return TerminationResult(True, False, EndReason.NO_GROUND_CONTACT)

        moving = progress_delta >= cfg.min_progress_per_step
        self._stall_streak = 0 if moving else self._stall_streak + 1
        if (
            self._steps >= cfg.min_steps_before_stall
            and self._stall_streak >= cfg.stall_limit
        ):
            return TerminationResult(True, False, EndReason.STALLED)

        if self._steps >= cfg.max_steps:
            return TerminationResult(False, True, EndReason.TIME_LIMIT)

        return TerminationResult(False, False, EndReason.RUNNING)


__all__ = ["EndReason", "TerminationConfig", "TerminationResult", "TerminationTracker"]
