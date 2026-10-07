"""Reward shaping for racing.

Objective
---------
Drive the centreline **fast and cleanly**. Because every control step has a fixed duration,
metres of centreline progress per second and lap time are the same objective up to a
constant, so dense progress is the primary term. It is available at every step, unlike a
sparse finish bonus that cannot guide an agent that has never completed a lap.

Why the terms are *rates*, integrated by ``dt``
-----------------------------------------------
The environment's control period is configurable (``control_dt * action_repeat``), and the
game-speed multiplier changes how much simulation time a step covers. If penalties are charged
per *step*, doubling the step rate doubles the penalty an identical lap receives, so the same
driving scores differently under different control rates -- and worse, the effective discount
horizon silently changes. Every shaping term here is therefore expressed **per second** and
multiplied by ``dt``. Total return over a lap is then invariant to the control rate, which is
what makes results comparable across configurations.

Progress is the exception, and deliberately so: metres of centreline covered is already a
time-integrated quantity, so it needs no ``dt`` factor. A lap is worth its own length.

Reward hacking
--------------
The real risk is not a wrong weight but a loophole, so each is closed explicitly:

* **Cuts.** Credited progress is capped by what is *physically achievable* in one step
  (``max_speed_for_progress * dt * cut_margin``) rather than by a fixed metre count. A fixed
  cap is a trap: at a 0.1 s control period it clamps legitimate driving above 60 m/s,
  punishing exactly the speed the reward is meant to encourage. Speed-aware capping detects a
  genuine cut (a teleport-scale jump) without ever binding on fast but honest driving.
* **Oscillation.** Backwards progress is credited negatively, bounded by a plausible reverse
  speed, so shuttling across a station cannot farm reward.
* **Standing still.** A stationary car must score *worse* than a moving one, otherwise "do
  nothing" is the optimal early policy and learning never starts. ``idle_penalty`` charges a
  rate while the car is below ``idle_speed_threshold``, which targets idling specifically
  without penalising genuinely slow cornering.
* **Off-track shortcuts.** Excursion beyond the corridor is charged per metre per second, so
  cutting across grass costs more the longer it lasts.
* **Invalid finishes.** The game's own checkpoint counter is authoritative: an episode that
  reports ``finished`` without collecting the map's checkpoints is flagged by the evaluation
  layer and never counted as a lap.

Every weight is configuration and every component is returned in :class:`RewardBreakdown`, so
a long run can be diagnosed from its metrics alone rather than by re-running it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tmai.game.protocol import GameFrame
from tmai.tracks.centerline import CenterlineTrack, TrackProjection


@dataclass
class RewardConfig:
    """Reward weights. See the module docstring for the rationale behind each."""

    #: Reward per metre of centreline progress. The dominant term.
    progress_weight: float = 1.0
    #: Reward for normalised forward speed, independent of path, per second. Kept small:
    #: progress already encodes speed along the line, and a large speed term makes the agent
    #: prefer going fast in the wrong direction.
    speed_weight: float = 1.0
    #: Reference speed used to normalise the speed and slip terms, m/s.
    speed_ref: float = 70.0

    # -- penalties, all charged per second and integrated by dt ----------------------

    #: Penalty per metre of corridor excursion, per second.
    off_track_weight: float = 6.0
    #: Penalty per radian of heading error, per second.
    heading_weight: float = 0.4
    #: Penalty per unit of normalised lateral speed, per second.
    slip_weight: float = 0.4
    #: Flat penalty per second while the car reports sliding.
    slide_penalty: float = 0.0
    #: Constant penalty per second. Discourages idling broadly; usually left at 0 because
    #: ``idle_penalty`` targets the same failure mode more precisely.
    step_penalty: float = 0.0
    #: Penalty per second while the car is below ``idle_speed_threshold``. This is what
    #: removes the "stand still and score zero" optimum.
    idle_penalty: float = 0.5
    #: Speed below which the car counts as idle, m/s.
    idle_speed_threshold: float = 1.0

    # -- episode end ----------------------------------------------------------------

    #: One-off bonus for crossing the finish line.
    finish_bonus: float = 20.0

    # -- anti-exploit limits, all speed-based rather than distance-based ------------

    #: Speed assumed to be the fastest the car can legitimately travel, m/s. Progress beyond
    #: what this allows in one step is treated as a cut or a teleport. Set it comfortably
    #: above the car's true top speed so it never binds on honest driving.
    max_speed_for_progress: float = 95.0
    #: Tolerance on the cut threshold, to absorb telemetry jitter and projection error.
    cut_margin: float = 1.15
    #: Maximum credited reverse speed, m/s.
    max_backward_speed: float = 30.0
    #: Corridor excursion (metres) tolerated before the off-track penalty starts.
    off_track_margin: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        """Reject configurations that would silently produce a degenerate reward."""
        problems: list[str] = []
        if self.progress_weight <= 0:
            problems.append("progress_weight must be positive: it is the only term that "
                            "rewards completing the track")
        if self.speed_ref <= 0:
            problems.append(f"speed_ref must be positive, got {self.speed_ref}")
        if self.max_speed_for_progress <= 0:
            problems.append(f"max_speed_for_progress must be positive, got {self.max_speed_for_progress}")
        if self.cut_margin < 1.0:
            problems.append(f"cut_margin must be >= 1.0, got {self.cut_margin}; below 1.0 the "
                            "cut detector clamps legitimate driving")
        if self.max_backward_speed < 0:
            problems.append(f"max_backward_speed must be non-negative, got {self.max_backward_speed}")
        if self.idle_speed_threshold < 0:
            problems.append(f"idle_speed_threshold must be non-negative, got {self.idle_speed_threshold}")
        for name in ("off_track_weight", "heading_weight", "slip_weight", "slide_penalty",
                     "step_penalty", "idle_penalty"):
            if getattr(self, name) < 0:
                problems.append(f"{name} must be non-negative, got {getattr(self, name)}")
        if problems:
            raise ValueError("invalid RewardConfig:\n  - " + "\n  - ".join(problems))


@dataclass
class RewardBreakdown:
    """Per-component reward for one step, for logging and tuning."""

    total: float = 0.0
    progress: float = 0.0
    speed: float = 0.0
    off_track: float = 0.0
    heading: float = 0.0
    slip: float = 0.0
    slide: float = 0.0
    step: float = 0.0
    idle: float = 0.0
    finish: float = 0.0
    #: Credited progress in metres after cut clamping (a key diagnostic).
    progress_metres: float = 0.0
    #: True when raw progress exceeded the speed-derived cut threshold.
    clamped: bool = False
    #: Raw (unclamped) progress in metres, so the size of a detected cut is visible.
    raw_progress_metres: float = 0.0

    def as_metrics(self) -> dict[str, float]:
        return {
            "reward/total": self.total,
            "reward/progress": self.progress,
            "reward/speed": self.speed,
            "reward/off_track": self.off_track,
            "reward/heading": self.heading,
            "reward/slip": self.slip,
            "reward/slide": self.slide,
            "reward/step": self.step,
            "reward/idle": self.idle,
            "reward/finish": self.finish,
            "reward/progress_metres": self.progress_metres,
            "reward/raw_progress_metres": self.raw_progress_metres,
            "reward/clamped": 1.0 if self.clamped else 0.0,
        }


class ProgressReward:
    """Computes the reward for one environment step.

    Args:
        track: the centreline the progress and corridor terms are measured against.
        config: reward weights. Validated on construction so a bad configuration fails at
            startup rather than producing silently wrong training signal.
    """

    def __init__(self, track: CenterlineTrack, config: RewardConfig | None = None) -> None:
        self.track = track
        self.config = config or RewardConfig()
        self.config.validate()

    def compute(
        self,
        *,
        frame: GameFrame,
        projection: TrackProjection,
        prev_progress: float,
        finished: bool,
        dt: float,
    ) -> RewardBreakdown:
        """Reward one control step of duration ``dt`` seconds."""
        cfg = self.config
        vehicle = frame.vehicle
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        # ---- progress, with a speed-derived cut detector ----
        raw_delta = projection.progress - prev_progress
        forward_limit = cfg.max_speed_for_progress * dt * cfg.cut_margin
        backward_limit = cfg.max_backward_speed * dt
        clipped_delta = float(np.clip(raw_delta, -backward_limit, forward_limit))
        clamped = abs(raw_delta - clipped_delta) > 1e-9
        progress_term = cfg.progress_weight * clipped_delta

        # ---- shaping terms, charged as rates over dt ----
        forward_speed = max(0.0, vehicle.speed_forward)
        speed_term = (
            cfg.speed_weight
            * dt
            * min(1.0, forward_speed / max(cfg.speed_ref, 1e-6))
        )

        # Interpolated by arc length rather than indexed by sample: the width array holds one
        # value per point while the car is generally mid-segment, and the two must agree with
        # what the observation reports for edge distances.
        half_width = self.track.corridor_half_width_at(projection.progress)
        excursion = max(0.0, abs(projection.lateral_offset) - half_width - cfg.off_track_margin)
        off_track_term = -cfg.off_track_weight * dt * excursion

        heading_term = (
            -cfg.heading_weight
            * dt
            * abs(self.track.heading_error(projection, vehicle.yaw()))
        )

        slip_term = (
            -cfg.slip_weight
            * dt
            * min(1.0, abs(vehicle.speed_sideward) / max(cfg.speed_ref, 1e-6))
        )

        slide_term = -cfg.slide_penalty * dt * (1.0 if vehicle.is_sliding else 0.0)
        step_term = -cfg.step_penalty * dt

        # Idling must score worse than driving, or a stationary car is a stable optimum.
        idle = abs(vehicle.speed_forward) < cfg.idle_speed_threshold and not finished
        idle_term = -cfg.idle_penalty * dt if idle else 0.0

        finish_term = cfg.finish_bonus if finished else 0.0

        total = (
            progress_term
            + speed_term
            + off_track_term
            + heading_term
            + slip_term
            + slide_term
            + step_term
            + idle_term
            + finish_term
        )

        return RewardBreakdown(
            total=float(total),
            progress=float(progress_term),
            speed=float(speed_term),
            off_track=float(off_track_term),
            heading=float(heading_term),
            slip=float(slip_term),
            slide=float(slide_term),
            step=float(step_term),
            idle=float(idle_term),
            finish=float(finish_term),
            progress_metres=clipped_delta,
            raw_progress_metres=float(raw_delta),
            clamped=clamped,
        )


__all__ = ["ProgressReward", "RewardBreakdown", "RewardConfig"]
