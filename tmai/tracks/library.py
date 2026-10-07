"""A library of tracks with train/validation/test separation.

Training on one map and testing on that same map measures memorisation, not driving. This
module is what makes the distinction enforceable rather than a matter of discipline:

* **Deterministic splits.** A track's split is a function of its identity, not of insertion
  order or a global shuffle. Adding a track to the library never moves another track between
  splits, so results stay comparable across runs.
* **Leakage prevention.** A track identity (its UID, falling back to a content hash of its
  geometry) can only ever belong to one split. The same map recorded twice is *the same
  track* for this purpose, which is the failure mode that silently inflates every
  generalisation number ever reported.
* **Explicit held-out sets.** ``test`` tracks are never sampled during training and are only
  reachable through the evaluation path.

Nothing here touches the game. A library can be built entirely from recorded centreline
files, which is what allows the whole generalisation pipeline to be developed and tested
without Trackmania.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tmai.tracks.centerline import CenterlineTrack
from tmai.tracks.stats import TrackStats, compute_stats

logger = logging.getLogger(__name__)

#: Recognised split names. ``test`` is reserved for held-out evaluation.
SPLITS: tuple[str, ...] = ("train", "validation", "test")

LIBRARY_SCHEMA_VERSION = 1


class TrackLibraryError(ValueError):
    """The library cannot be built or would leak tracks between splits."""


def track_identity(track: CenterlineTrack) -> str:
    """A stable identity for a track, used to keep splits disjoint.

    The map UID is preferred because it is what the game reports and survives re-recording.
    Without one, a hash of the geometry stands in: two recordings of the same line hash the
    same, which is exactly the property that stops a duplicate from being treated as a new
    track and quietly leaking into the validation set.
    """
    if track.uid and not str(track.uid).startswith("synthetic:"):
        return f"uid:{track.uid}"
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(np.round(track.points, 3)).tobytes())
    digest.update(str(track.closed).encode())
    return f"geom:{digest.hexdigest()[:32]}"


def split_for_identity(identity: str, *, weights: Mapping[str, float]) -> str:
    """Deterministically map an identity to a split.

    Uses a stable hash (SHA-256, not ``hash()``, which is salted per process) so the
    assignment is identical across runs, machines and Python versions.
    """
    total = sum(weights.values())
    if total <= 0:
        raise TrackLibraryError(f"split weights must sum to a positive value, got {weights!r}")
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    # 53 bits of the digest as a uniform float in [0, 1).
    value = int(digest[:13], 16) / float(16**13)
    cumulative = 0.0
    for name in SPLITS:
        cumulative += weights.get(name, 0.0) / total
        if value < cumulative:
            return name
    return SPLITS[-1]


@dataclass
class TrackEntry:
    """One track in the library, with its split and cached statistics."""

    track: CenterlineTrack
    split: str
    identity: str
    #: Path the track was loaded from, when it came from disk.
    source: str | None = None
    stats: TrackStats | None = None

    @property
    def name(self) -> str:
        return self.track.name

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.track.name,
            "identity": self.identity,
            "split": self.split,
            "source": self.source,
            "length": round(self.track.length, 2),
            "stats": self.stats.as_dict() if self.stats else None,
        }


@dataclass
class TrackLibrary:
    """An ordered collection of tracks partitioned into splits.

    Args:
        entries: the tracks. Order is preserved, which keeps sampling reproducible for a
            given seed.
        split_weights: relative size of each split when a track has no explicit assignment.
    """

    entries: list[TrackEntry] = field(default_factory=list)
    split_weights: dict[str, float] = field(
        default_factory=lambda: {"train": 0.7, "validation": 0.15, "test": 0.15}
    )

    # -- construction ---------------------------------------------------------------

    def add(
        self,
        track: CenterlineTrack,
        *,
        split: str | None = None,
        source: str | None = None,
        compute_track_stats: bool = True,
    ) -> TrackEntry:
        """Add a track, assigning it a split if one was not given.

        Raises:
            TrackLibraryError: if the track's identity is already present in a *different*
                split, which would leak it.
        """
        identity = track_identity(track)
        existing = self.find(identity)
        if existing is not None:
            wanted = split or existing.split
            if wanted != existing.split:
                raise TrackLibraryError(
                    f"track {track.name!r} (identity {identity}) is already in the "
                    f"{existing.split!r} split; refusing to also place it in {wanted!r}. "
                    "A track must belong to exactly one split or evaluation is not meaningful."
                )
            logger.debug("track %r already present in %r; skipping", track.name, existing.split)
            return existing

        if split is None:
            split = split_for_identity(identity, weights=self.split_weights)
        if split not in SPLITS:
            raise TrackLibraryError(f"unknown split {split!r}; expected one of {SPLITS}")

        entry = TrackEntry(
            track=track,
            split=split,
            identity=identity,
            source=source,
            stats=compute_stats(track) if compute_track_stats else None,
        )
        self.entries.append(entry)
        return entry

    def extend(
        self,
        tracks: Iterable[CenterlineTrack],
        *,
        split: str | None = None,
    ) -> list[TrackEntry]:
        return [self.add(track, split=split) for track in tracks]

    @staticmethod
    def from_directory(
        directory: str | Path,
        *,
        split_weights: Mapping[str, float] | None = None,
        pattern: str = "*.json",
        explicit_splits: Mapping[str, str] | None = None,
        recursive: bool = False,
    ) -> TrackLibrary:
        """Load every centreline file in ``directory``.

        ``explicit_splits`` maps a *filename stem* to a split, which is how an operator pins
        particular maps to ``test``. Anything unlisted is assigned deterministically.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise TrackLibraryError(f"not a directory: {directory}")

        library = TrackLibrary(split_weights=dict(split_weights or TrackLibrary().split_weights))
        globber = directory.rglob(pattern) if recursive else directory.glob(pattern)
        paths = sorted(globber)
        if not paths:
            raise TrackLibraryError(f"no track files matching {pattern!r} in {directory}")

        for path in paths:
            track = CenterlineTrack.load(path)
            split = (explicit_splits or {}).get(path.stem)
            library.add(track, split=split, source=str(path))
            logger.info(
                "loaded track %r -> %s (%.0f m, %d points)",
                track.name,
                library.entries[-1].split,
                track.length,
                track.num_points,
            )
        return library

    # -- access ---------------------------------------------------------------------

    def find(self, identity: str) -> TrackEntry | None:
        for entry in self.entries:
            if entry.identity == identity:
                return entry
        return None

    def by_split(self, split: str) -> list[TrackEntry]:
        if split not in SPLITS:
            raise TrackLibraryError(f"unknown split {split!r}; expected one of {SPLITS}")
        return [e for e in self.entries if e.split == split]

    def tracks(self, split: str) -> list[CenterlineTrack]:
        return [e.track for e in self.by_split(split)]

    @property
    def train(self) -> list[CenterlineTrack]:
        return self.tracks("train")

    @property
    def validation(self) -> list[CenterlineTrack]:
        return self.tracks("validation")

    @property
    def test(self) -> list[CenterlineTrack]:
        return self.tracks("test")

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[TrackEntry]:
        return iter(self.entries)

    def counts(self) -> dict[str, int]:
        out = {name: 0 for name in SPLITS}
        for entry in self.entries:
            out[entry.split] += 1
        return out

    # -- sampling -------------------------------------------------------------------

    def sampler(self, split: str, *, seed: int | None = None) -> TrackSampler:
        """A reproducible sampler over one split."""
        tracks = self.tracks(split)
        if not tracks:
            raise TrackLibraryError(
                f"split {split!r} is empty; the library has {self.counts()}. "
                "Check the track files and the configured split weights."
            )
        return TrackSampler(tracks, seed=seed)

    # -- diagnostics ----------------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """Split composition and geometry coverage, for the manifest and for humans.

        The per-split geometry summary is the cheap check on whether the held-out tracks are
        even comparable to the training tracks.
        """
        out: dict[str, Any] = {
            "num_tracks": len(self.entries),
            "counts": self.counts(),
            "split_weights": dict(self.split_weights),
            "tracks": [e.as_dict() for e in self.entries],
            "geometry_by_split": {},
        }
        for split in SPLITS:
            entries = self.by_split(split)
            if not entries or not all(e.stats for e in entries):
                continue
            lengths = np.array([e.stats.length for e in entries])  # type: ignore[union-attr]
            corners = np.array([e.stats.corner_count for e in entries])  # type: ignore[union-attr]
            curvatures = np.array([e.stats.curvature_mean for e in entries])  # type: ignore[union-attr]
            out["geometry_by_split"][split] = {
                "length_mean": round(float(lengths.mean()), 2),
                "length_min": round(float(lengths.min()), 2),
                "length_max": round(float(lengths.max()), 2),
                "corners_mean": round(float(corners.mean()), 2),
                "curvature_mean": round(float(curvatures.mean()), 6),
            }
        return out

    def summary(self) -> str:
        counts = self.counts()
        parts = ", ".join(f"{counts[s]} {s}" for s in SPLITS if counts[s])
        return f"{len(self.entries)} tracks ({parts})" if parts else "empty library"

    # -- serialisation --------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LIBRARY_SCHEMA_VERSION,
            "split_weights": dict(self.split_weights),
            "tracks": [
                {
                    "name": e.track.name,
                    "identity": e.identity,
                    "split": e.split,
                    "source": e.source,
                }
                for e in self.entries
            ],
        }

    def save_manifest(self, path: str | Path) -> Path:
        """Write the library index (not the geometry) so a run can record its track set."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


class TrackSampler:
    """Samples tracks from one split, reproducibly and without silent repetition.

    Draws are shuffled in blocks rather than i.i.d.: with a small library (say four maps)
    i.i.d. sampling can leave a map untouched for hundreds of episodes, and the policy
    quietly forgets it. Cycling through a permutation guarantees every training track is
    seen every ``n`` episodes, which matters far more than perfect uniformity.
    """

    def __init__(self, tracks: Sequence[CenterlineTrack], *, seed: int | None = None) -> None:
        if not tracks:
            raise TrackLibraryError("cannot sample from an empty track list")
        self._tracks = list(tracks)
        self._rng = np.random.default_rng(seed)
        self._order: list[int] = []
        self._draws = 0

    def __len__(self) -> int:
        return len(self._tracks)

    @property
    def draws(self) -> int:
        return self._draws

    def next(self) -> CenterlineTrack:
        if not self._order:
            self._order = list(self._rng.permutation(len(self._tracks)))
        index = int(self._order.pop())
        self._draws += 1
        return self._tracks[index]

    def peek_all(self) -> list[CenterlineTrack]:
        """Every track in the sampler, in library order (for exhaustive evaluation)."""
        return list(self._tracks)

    def state_dict(self) -> dict[str, Any]:
        return {"order": list(self._order), "draws": self._draws}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._order = [int(i) for i in state.get("order", [])]
        self._draws = int(state.get("draws", 0))


__all__ = [
    "LIBRARY_SCHEMA_VERSION",
    "SPLITS",
    "TrackEntry",
    "TrackLibrary",
    "TrackLibraryError",
    "TrackSampler",
    "split_for_identity",
    "track_identity",
]
