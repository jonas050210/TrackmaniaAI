"""Episode termination logic.

Early termination matters for training throughput: an episode that spends 60 s stuck against
a wall produces no useful gradient and blocks the single game instance from collecting
anything else. Every rule here exists to cut such dead time, and every one of them reports a
machine-readable reason so a long run can be diagnosed from metrics.

``terminated`` means "the task ended" (finished, or unrecoverable); ``truncated`` means "we
stopped it" (time limit). The distinction is required for correct bootstrapping in
value-based RL: the learner must not bootstrap through a truncation.

Crash and failure detection
---------------------------
The required driving rules are: a significant wall collision must respawn the car
immediately, falling off the map or a confirmed out-of-bounds state must respawn it
immediately, and both must carry a meaningful negative reward (the reward side lives in
:mod:`tmai.env.reward`; this module decides *when*).

Detection trusts the game first. When the driver reports contact state
(``DriverCapabilities.reports_contact``), the game's own ``has_any_lateral_contact`` flag
and per-wheel ground contacts are used; kinematic inference (impact deceleration) is the
fallback and is also used to *confirm* an impact. The rules:

* **Crash (CRASH).** Confirmed when either
  (a) *impact*: the car loses more than ``crash_speed_loss`` m/s in a single control step
      while lateral contact is reported -- a scrape does not produce that, a collision does;
  or (b) *sustained contact with speed loss*: lateral contact for ``crash_contact_steps``
      consecutive steps while the car has fallen below ``crash_speed_fraction`` of the
      speed it had when the contact started. This separates "nudging a wall while parking"
      (contact without a prior fast approach, excluded by ``min_speed_for_crash``) and
      "brief scrape at speed" (contact for one or two ticks, speed maintained) from a
      genuine crash.
* **Out of bounds (OUT_OF_BOUNDS).** The car is more than ``out_of_bounds_margin`` metres
  beyond the drivable corridor edge for ``out_of_bounds_steps`` consecutive steps. A
  single-step excursion (telemetry jitter, a wide racing line) is not out of bounds; a
  sustained one is, and the car is respawned rather than driven back from the scenery.
* **Fell off the map (FELL_OFF).** No wheel reports ground contact *and* the car is more
  than ``fall_height`` below the centreline plane, for ``fall_steps`` consecutive steps.
  Airborne over a jump rises *above* the plane, so this does not fire on jumps; sinking
  below it means the car has left the track entirely.
* **Wrong way (WRONG_WAY).** The car sits more than ``wrong_way_progress`` metres behind
  its best progress this episode for ``wrong_way_steps`` consecutive steps, i.e. it is
  driving backwards for a sustained period rather than recovering from a spin.

All four end the episode immediately (``terminated=True``) so the trainer respawns the car,
and all four carry a one-off negative reward (see ``RewardConfig``). None of them can be
exploited for reward: the episode ends, the terminal penalty is large and negative, and
progress credit is already clamped by a speed-derived cut detector, so respawning cannot
manufacture progress.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    #: Confirmed wall collision (impact or sustained contact with speed loss).
    CRASH = "crash"
    #: Sustained excursion far beyond the drivable corridor.
    OUT_OF_BOUNDS = "out_of_bounds"
    #: Airborne and below the track plane: the car has fallen off the map.
    FELL_OFF = "fell_off"
    #: Sustained reverse progress: driving the track backwards.
    WRONG_WAY = "wrong_way"


#: Reasons that carry a one-off terminal penalty in the reward (see RewardConfig).
PENALTY_REASONS: frozenset[EndReason] = frozenset(
    {EndReason.CRASH, EndReason.OUT_OF_BOUNDS, EndReason.FELL_OFF, EndReason.WRONG_WAY}
)


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

    # -- crash detection ------------------------------------------------------------
    #: Speed lost in one control step, while in contact, that confirms an impact, m/s.
    crash_speed_loss: float = 8.0
    #: Consecutive steps of reported lateral contact that confirm a crash once the speed
    #: has fallen to ``crash_speed_fraction`` of the contact-entry speed.
    crash_contact_steps: int = 3
    #: Speed below this fraction of the contact-entry speed counts as "held by the wall".
    crash_speed_fraction: float = 0.6
    #: Contact entered below this speed is parking, not a crash, m/s.
    min_speed_for_crash: float = 2.0

    # -- out of bounds / fell off ----------------------------------------------------
    #: Metres beyond the corridor edge that count as out of bounds.
    out_of_bounds_margin: float = 6.0
    #: Consecutive out-of-bounds steps before the car is respawned.
    out_of_bounds_steps: int = 3
    #: Metres below the centreline plane (while airborne) that mean "fell off the map".
    fall_height: float = 1.5
    #: Consecutive falling steps before the car is respawned.
    fall_steps: int = 3

    # -- wrong way -------------------------------------------------------------------
    #: Metres behind the episode's best progress that count as wrong way.
    wrong_way_progress: float = 8.0
    #: Consecutive steps behind that mark before terminating.
    wrong_way_steps: int = 25

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.crash_speed_loss <= 0:
            problems.append(f"crash_speed_loss must be positive, got {self.crash_speed_loss}")
        if self.crash_contact_steps < 1:
            problems.append(
                f"crash_contact_steps must be >= 1, got {self.crash_contact_steps}"
            )
        if not 0.0 < self.crash_speed_fraction <= 1.0:
            problems.append(
                f"crash_speed_fraction must be in (0, 1], got {self.crash_speed_fraction}"
            )
        if self.min_speed_for_crash < 0:
            problems.append(
                f"min_speed_for_crash must be non-negative, got {self.min_speed_for_crash}"
            )
        if self.out_of_bounds_margin < 0:
            problems.append(
                f"out_of_bounds_margin must be non-negative, got {self.out_of_bounds_margin}"
            )
        if self.out_of_bounds_steps < 1:
            problems.append(
                f"out_of_bounds_steps must be >= 1, got {self.out_of_bounds_steps}"
            )
        if self.fall_height <= 0:
            problems.append(f"fall_height must be positive, got {self.fall_height}")
        if self.fall_steps < 1:
            problems.append(f"fall_steps must be >= 1, got {self.fall_steps}")
        if self.wrong_way_progress <= 0:
            problems.append(
                f"wrong_way_progress must be positive, got {self.wrong_way_progress}"
            )
        if self.wrong_way_steps < 1:
            problems.append(f"wrong_way_steps must be >= 1, got {self.wrong_way_steps}")
        return problems


@dataclass
class TerminationResult:
    terminated: bool
    truncated: bool
    reason: EndReason
    #: True when the car was in lateral contact on this step (a scrape, not necessarily a
    #: crash). Reported so the reward can charge a small per-second contact penalty.
    lateral_contact: bool = False

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def penalty_reason(self) -> EndReason | None:
        """The reason when this step's termination carries a terminal reward penalty."""
        if self.terminated and self.reason in PENALTY_REASONS:
            return self.reason
        return None


@dataclass
class TerminationTracker:
    """Stateful per-episode termination bookkeeping. Call :meth:`reset` on episode start."""

    track: CenterlineTrack
    config: TerminationConfig = field(default_factory=TerminationConfig)
    #: Per-episode override of ``max_steps`` (used by curriculum stage length caps).
    max_steps_override: int | None = None

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._steps = 0
        self._off_track_streak = 0
        self._stall_streak = 0
        self._airborne_streak = 0
        self._out_of_bounds_streak = 0
        self._fall_streak = 0
        self._wrong_way_streak = 0
        self._contact_streak = 0
        self._contact_entry_speed: float | None = None
        self._prev_speed: float | None = None
        self._max_progress = 0.0

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def max_steps(self) -> int:
        return self.max_steps_override or self.config.max_steps

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
        vehicle = frame.vehicle
        speed = abs(vehicle.speed_forward)

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

        # ---- crash: game-reported contact plus an impact signature ------------------
        crash = self._update_crash(vehicle.has_lateral_contact, speed, projection)
        if crash:
            return TerminationResult(
                True, False, EndReason.CRASH, lateral_contact=vehicle.has_lateral_contact
            )

        # ---- out of bounds: sustained excursion far beyond the corridor -------------
        out_of_bounds = (
            abs(projection.lateral_offset) > half_width + cfg.out_of_bounds_margin
        )
        self._out_of_bounds_streak = self._out_of_bounds_streak + 1 if out_of_bounds else 0
        if self._out_of_bounds_streak >= cfg.out_of_bounds_steps:
            return TerminationResult(True, False, EndReason.OUT_OF_BOUNDS)

        # ---- fell off the map: airborne and sinking below the track plane -----------
        falling = (
            not vehicle.has_ground_contact
            and projection.vertical_offset < -cfg.fall_height
        )
        self._fall_streak = self._fall_streak + 1 if falling else 0
        if self._fall_streak >= cfg.fall_steps:
            return TerminationResult(True, False, EndReason.FELL_OFF)

        if not vehicle.has_ground_contact:
            self._airborne_streak += 1
        else:
            self._airborne_streak = 0
        if self._airborne_streak >= cfg.no_ground_contact_limit:
            return TerminationResult(True, False, EndReason.NO_GROUND_CONTACT)

        # ---- wrong way: sustained reverse progress ------------------------------------
        self._max_progress = max(self._max_progress, projection.progress)
        behind = projection.progress < self._max_progress - cfg.wrong_way_progress
        self._wrong_way_streak = self._wrong_way_streak + 1 if behind else 0
        if self._wrong_way_streak >= cfg.wrong_way_steps:
            return TerminationResult(True, False, EndReason.WRONG_WAY)

        moving = progress_delta >= cfg.min_progress_per_step
        self._stall_streak = 0 if moving else self._stall_streak + 1
        if (
            self._steps >= cfg.min_steps_before_stall
            and self._stall_streak >= cfg.stall_limit
        ):
            return TerminationResult(True, False, EndReason.STALLED)

        if self._steps >= self.max_steps:
            return TerminationResult(False, True, EndReason.TIME_LIMIT)

        return TerminationResult(
            False, False, EndReason.RUNNING, lateral_contact=vehicle.has_lateral_contact
        )

    # -- internals ------------------------------------------------------------------

    def _update_crash(
        self, lateral_contact: bool, speed: float, projection: TrackProjection
    ) -> bool:
        """Update crash bookkeeping; return True when a crash is confirmed this step.

        Two independent confirmations, either of which is sufficient:

        * **Impact with contact** -- the game reports lateral contact *and* the car lost
          more than ``crash_speed_loss`` m/s in a single control step. A wall scrape at
          speed does not decelerate the car by 8 m/s in 50 ms; a collision does.
        * **Sustained contact with speed loss** -- the game reports lateral contact for
          ``crash_contact_steps`` consecutive steps while the car has fallen below
          ``crash_speed_fraction`` of the speed it entered the contact with, having
          entered it above ``min_speed_for_crash``. Parking against a wall and brief
          scrapes are excluded by construction.

        A third, deliberately conservative fallback covers drivers that do not report
        contact at all: a very large one-step speed loss *while inside the corridor*.
        Off-track deceleration (mud, grass drag) is excluded by the corridor gate, because
        surface drag is not a crash; a loss that large on the racing line is.
        """
        cfg = self.config

        if self._prev_speed is not None:
            speed_loss = self._prev_speed - speed
            if speed_loss > cfg.crash_speed_loss and lateral_contact:
                self._prev_speed = speed
                return True
            half_width = self.track.corridor_half_width_at(projection.progress)
            on_corridor = abs(projection.lateral_offset) <= half_width
            if speed_loss > cfg.crash_speed_loss * 1.5 and on_corridor:
                self._prev_speed = speed
                return True

        # Sustained contact with speed loss: needs the game's contact flag.
        if lateral_contact:
            if self._contact_streak == 0:
                self._contact_entry_speed = speed
            self._contact_streak += 1
            entry = self._contact_entry_speed or 0.0
            if (
                self._contact_streak >= cfg.crash_contact_steps
                and entry >= cfg.min_speed_for_crash
                and speed <= cfg.crash_speed_fraction * entry
            ):
                self._prev_speed = speed
                return True
        else:
            self._contact_streak = 0
            self._contact_entry_speed = None

        self._prev_speed = speed
        return False


__all__ = [
    "EndReason",
    "PENALTY_REASONS",
    "TerminationConfig",
    "TerminationResult",
    "TerminationTracker",
]
