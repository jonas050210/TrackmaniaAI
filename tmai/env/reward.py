"""Reward shaping for racing.

Core idea
---------
Each control step lasts a fixed ``dt`` seconds, so **metres of centreline progress per step
and lap time are the same objective up to a constant**: maximising progress rate minimises
the time needed to cover the line. That makes dense progress the primary reward, which is
both time-optimal and available at every step (unlike a sparse finish reward, which cannot
guide an agent that has never completed a lap).

Reward hacking is the real risk, so the shaping is built to close the obvious holes:

* ``max_progress_per_step`` clamps a single step's credited progress. Without it, driving
  straight across the map and rejoining far ahead scores a huge reward; this is the same
  failure mode TMRL addresses with its ``CHECK_FORWARD`` cut limit.
* Backwards progress is credited negatively (bounded by ``max_backward_per_step``), so the
  agent cannot farm reward by oscillating across a station.
* Leaving the corridor costs per metre of excursion, so a shortcut across grass is not free.
* The game's own checkpoint counter is authoritative for *validity*: an episode that
  finishes without collecting checkpoints is logged as an invalid finish.

All weights are configuration, and every component is returned in
:class:`RewardBreakdown` so a long training run can be diagnosed from its metrics alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tmai.game.protocol import GameFrame
from tmai.tracks.centerline import CenterlineTrack, TrackProjection


@dataclass
class RewardConfig:
    """Reward weights. All terms are summed; see the module docstring for rationale."""

    #: Reward per metre of centreline progress. The dominant term.
    progress_weight: float = 1.0
    #: Reward for normalised forward speed, independent of path. Kept small: progress
    #: already encodes speed along the line, and a large speed term makes the agent prefer
    #: going fast in the wrong direction.
    speed_weight: float = 0.05
    #: Penalty per metre outside the drivable corridor.
    off_track_weight: float = 0.3
    #: Penalty per radian of heading error relative to the centreline.
    heading_weight: float = 0.02
    #: Penalty per normalised lateral speed (uncontrolled sliding).
    slip_weight: float = 0.02
    #: Flat penalty per control step while the car reports sliding.
    slide_penalty: float = 0.0
    #: Constant penalty per control step. Useful to discourage idling; usually left at 0
    #: because progress already rewards moving.
    step_penalty: float = 0.0
    #: One-off bonus for crossing the finish line.
    finish_bonus: float = 20.0
    #: Maximum credited forward progress per step, metres.
    max_progress_per_step: float = 6.0
    #: Maximum credited backward progress per step, metres.
    max_backward_per_step: float = 2.0
    #: Reference speed used to normalise the speed and slip terms, m/s.
    speed_ref: float = 70.0
    #: Corridor excursion (metres) tolerated before the off-track penalty starts.
    off_track_margin: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RewardBreakdown:
    """Per-component reward, for logging and tuning."""

    total: float = 0.0
    progress: float = 0.0
    speed: float = 0.0
    off_track: float = 0.0
    heading: float = 0.0
    slip: float = 0.0
    slide: float = 0.0
    step: float = 0.0
    finish: float = 0.0
    #: Credited progress in metres after clamping (a key diagnostic).
    progress_metres: float = 0.0
    #: True when the raw progress exceeded the clamp, i.e. a likely cut or a teleport.
    clamped: bool = False

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
            "reward/finish": self.finish,
            "reward/progress_metres": self.progress_metres,
            "reward/clamped": 1.0 if self.clamped else 0.0,
        }


class ProgressReward:
    """Computes the reward for one environment step."""

    def __init__(self, track: CenterlineTrack, config: RewardConfig | None = None) -> None:
        self.track = track
        self.config = config or RewardConfig()

    def compute(
        self,
        *,
        frame: GameFrame,
        projection: TrackProjection,
        prev_progress: float,
        finished: bool,
    ) -> RewardBreakdown:
        cfg = self.config
        vehicle = frame.vehicle

        raw_delta = projection.progress - prev_progress
        clipped_delta = float(
            np.clip(raw_delta, -cfg.max_backward_per_step, cfg.max_progress_per_step)
        )
        clamped = abs(raw_delta - clipped_delta) > 1e-9

        progress_term = cfg.progress_weight * clipped_delta

        forward_speed = max(0.0, vehicle.speed_forward)
        speed_term = cfg.speed_weight * min(1.0, forward_speed / max(cfg.speed_ref, 1e-6))

        half_width = float(self.track.corridor_half_width[projection.index])
        excursion = max(0.0, abs(projection.lateral_offset) - half_width - cfg.off_track_margin)
        off_track_term = -cfg.off_track_weight * excursion

        heading_term = -cfg.heading_weight * abs(
            self.track.heading_error(projection, vehicle.yaw())
        )

        slip_term = -cfg.slip_weight * min(
            1.0, abs(vehicle.speed_sideward) / max(cfg.speed_ref, 1e-6)
        )

        slide_term = -cfg.slide_penalty * (1.0 if vehicle.is_sliding else 0.0)
        step_term = -cfg.step_penalty
        finish_term = cfg.finish_bonus if finished else 0.0

        total = (
            progress_term
            + speed_term
            + off_track_term
            + heading_term
            + slip_term
            + slide_term
            + step_term
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
            finish=float(finish_term),
            progress_metres=clipped_delta,
            clamped=clamped,
        )


__all__ = ["ProgressReward", "RewardBreakdown", "RewardConfig"]
