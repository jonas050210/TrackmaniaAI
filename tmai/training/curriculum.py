"""Curriculum learning: reveal the track suite progressively instead of all at once.

Why a curriculum: from step one the agent is dropped onto the hardest track in the suite
alongside the easiest. Early gradients are then dominated by whatever the hardest map
throws at it, and the policy can spend a long time learning nothing transferable. A
curriculum starts with the gentlest tracks and short episodes, and widens both as the
policy improves -- the standard remedy, and cheap to implement because the multi-track
environment already samples a track per episode.

Two independent levers, both driven by training step:

* **Track reveal.** Tracks are ordered by a difficulty score computed from their geometry
  statistics (mean curvature first -- gentle, flowing tracks before tight technical ones)
  and only the easiest ``reveal_tracks`` are sampled until the next stage.
* **Episode length.** Early episodes are a fraction of the full lap length, so the agent
  learns to drive before it has to learn to *survive*.

Both are deterministic functions of the step count, so a run is reproducible, and the
active stage is logged as an event and a metric so a long run shows when it advanced.

The curriculum applies to **training only**. Held-out evaluation always uses every track
at full length: measuring generalisation against a curriculum-filtered suite would be
measuring the filter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from tmai.tracks.library import TrackEntry

logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """One curriculum stage, active up to ``until_step`` training steps."""

    #: Training step at which this stage ends (exclusive). Steps beyond the last stage
    #: run the final stage: every track, full-length episodes.
    until_step: int
    #: How many of the easiest training tracks are sampled. ``None`` means all of them.
    reveal_tracks: int | None = None
    #: Episode length as a fraction of the configured maximum, in ``(0, 1]``.
    episode_length_fraction: float = 1.0

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.until_step <= 0:
            problems.append(f"until_step must be positive, got {self.until_step}")
        if self.reveal_tracks is not None and self.reveal_tracks < 1:
            problems.append(f"reveal_tracks must be >= 1 or null, got {self.reveal_tracks}")
        if not 0.0 < self.episode_length_fraction <= 1.0:
            problems.append(
                f"episode_length_fraction must be in (0, 1], got {self.episode_length_fraction}"
            )
        return problems


@dataclass
class CurriculumSpec:
    """Configuration of the curriculum. Empty stages means "no curriculum"."""

    enabled: bool = False
    stages: list[CurriculumStage] = field(default_factory=list)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.enabled:
            return problems
        if not self.stages:
            problems.append("curriculum.enabled is true but no stages are configured")
            return problems
        previous = 0
        for i, stage in enumerate(self.stages):
            for problem in stage.validate():
                problems.append(f"curriculum.stages[{i}]: {problem}")
            if stage.until_step <= previous:
                problems.append(
                    f"curriculum.stages[{i}].until_step ({stage.until_step}) must be "
                    f"greater than the previous stage's ({previous})"
                )
            previous = stage.until_step
        return problems


def difficulty_score(entry: TrackEntry) -> float:
    """A deterministic difficulty score for one track, lower being easier.

    Mean absolute curvature dominates: a gentle flowing track has low mean curvature, a
    tight technical track high. Corner density breaks ties so that a long track with one
    hairpin is not ranked easier than a short track that is all hairpins. Ties break on
    name so the ordering is total and stable across runs and machines.
    """
    stats = entry.stats
    if stats is None:
        return 0.0
    length = max(stats.length, 1e-6)
    corner_density = stats.corner_count / length
    return stats.curvature_mean + 0.01 * corner_density


class Curriculum:
    """Resolves the active stage for a training step and the tracks/length it implies.

    Args:
        spec: the curriculum configuration.
        entries: the training tracks, with statistics (used for the difficulty order).
    """

    def __init__(self, spec: CurriculumSpec, entries: list[TrackEntry]) -> None:
        self.spec = spec
        self._stages = sorted(spec.stages, key=lambda s: s.until_step)
        # Difficulty order over the training tracks: easiest first, name as tie-break.
        self._ordered = sorted(entries, key=lambda e: (difficulty_score(e), e.track.name))
        self._final = CurriculumStage(
            until_step=2**62, reveal_tracks=None, episode_length_fraction=1.0
        )

    @property
    def num_stages(self) -> int:
        """Configured stages plus the implicit final stage (everything revealed)."""
        return len(self._stages) + 1

    def stage_index_at(self, step: int) -> int:
        """Index of the active stage: ``0..len(stages)``, the last being the final stage."""
        for i, stage in enumerate(self._stages):
            if step <= stage.until_step:
                return i
        return len(self._stages)

    def stage_at(self, step: int) -> CurriculumStage:
        index = self.stage_index_at(step)
        return self._stages[index] if index < len(self._stages) else self._final

    def active_track_names(self, step: int) -> list[str] | None:
        """Names of the tracks sampled at ``step``, or ``None`` for "all of them"."""
        stage = self.stage_at(step)
        if stage.reveal_tracks is None:
            return None
        revealed = self._ordered[: max(1, min(stage.reveal_tracks, len(self._ordered)))]
        return [e.track.name for e in revealed]

    def episode_max_steps(self, step: int, base_max_steps: int) -> int | None:
        """The episode step cap at ``step``, or ``None`` when episodes run full length.

        ``base_max_steps * episode_length_fraction``, rounded; a fraction of 1.0 means no
        cap at all, which is reported as ``None`` so the environment keeps its configured
        limit rather than overriding it with the same value.
        """
        stage = self.stage_at(step)
        if stage.episode_length_fraction >= 1.0:
            return None
        return max(1, int(round(base_max_steps * stage.episode_length_fraction)))

    def difficulty_order(self) -> list[str]:
        """Training track names, easiest first (the reveal order)."""
        return [e.track.name for e in self._ordered]

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.spec.enabled,
            "num_stages": self.num_stages,
            "stages": [
                {
                    "until_step": s.until_step,
                    "reveal_tracks": s.reveal_tracks,
                    "episode_length_fraction": s.episode_length_fraction,
                }
                for s in self._stages
            ],
            "final_stage": {
                "until_step": None,
                "reveal_tracks": None,
                "episode_length_fraction": 1.0,
            },
            "difficulty_order": self.difficulty_order(),
            "num_train_tracks": len(self._ordered),
        }


__all__ = ["Curriculum", "CurriculumSpec", "CurriculumStage", "difficulty_score"]
