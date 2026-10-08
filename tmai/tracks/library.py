"""A library of tracks with train/validation/test separation.

Training on one map and testing on that same map measures memorisation, not driving. This
module is what makes the distinction enforceable rather than a matter of discipline:

* **Deterministic splits.** A track's split is a function of its identity, not of insertion
  order or a global shuffle. Adding a track to the library never moves another track between
  splits, so results stay comparable across runs.
* **Leakage prevention.** A track identity (its UID, falling back to a content hash of its
  geometry) can only ever belong to one split. Related maps can also share an explicit
  metadata ``family`` label; the whole family is then assigned as one group.
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
import unicodedata
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


def _clean_family_label(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def track_family(track: CenterlineTrack) -> str | None:
    """Return an optional operator-supplied family label.

    Families are declared in track metadata as ``{"family": "author-or-layout-family"}``.
    A reliable family cannot be inferred from a centreline alone, so unlabelled tracks retain
    identity-based splitting rather than being clustered by a guess.
    """
    value = track.metadata.get("family")
    if value is None:
        return None
    if not isinstance(value, str):
        raise TrackLibraryError(
            f"track {track.name!r} metadata family must be a string, got {type(value).__name__}"
        )
    family = _clean_family_label(value)
    return family or None


def _family_key(family: str) -> str:
    """Canonical, namespaced key for grouping explicitly related tracks."""
    return f"family:{_clean_family_label(family).casefold()}"


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
    #: Optional explicit group label. All tracks in one family share a split.
    family: str | None = None
    #: Path the track was loaded from, when it came from disk.
    source: str | None = None
    stats: TrackStats | None = None

    @property
    def name(self) -> str:
        return self.track.name

    @property
    def split_group(self) -> str:
        """Stable grouping key used to keep related layouts in one split."""
        return _family_key(self.family) if self.family else self.identity

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.track.name,
            "identity": self.identity,
            "family": self.family,
            "split_group": self.split_group,
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
            TrackLibraryError: if an identity or explicitly labelled family is assigned to
                more than one split, which would leak map geometry across evaluation sets.
        """
        identity = track_identity(track)
        family = track_family(track)
        if split is not None and split not in SPLITS:
            raise TrackLibraryError(f"unknown split {split!r}; expected one of {SPLITS}")

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

        group_key = _family_key(family) if family else identity
        family_entries = [entry for entry in self.entries if entry.split_group == group_key]
        group_splits = {entry.split for entry in family_entries}
        if len(group_splits) > 1:
            raise TrackLibraryError(
                f"track family {family!r} is already spread across splits "
                f"{sorted(group_splits)}; fix the existing assignments before adding more maps"
            )
        if family_entries:
            family_split = family_entries[0].split
            if split is not None and split != family_split:
                raise TrackLibraryError(
                    f"track family {family!r} is already in the {family_split!r} split; "
                    f"refusing to assign {track.name!r} to {split!r}. Related maps must "
                    "stay in one split to avoid family leakage."
                )
            split = family_split
        elif split is None:
            split = split_for_identity(group_key, weights=self.split_weights)

        # `split` is now either explicit or derived from the family/track grouping key.
        assert split is not None
        entry = TrackEntry(
            track=track,
            split=split,
            identity=identity,
            family=family,
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
        particular maps or labelled families to ``test``. A pin on any family member is
        propagated to the rest of that family. Anything unlisted is assigned deterministically.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise TrackLibraryError(f"not a directory: {directory}")

        library = TrackLibrary(split_weights=dict(split_weights or TrackLibrary().split_weights))
        globber = directory.rglob(pattern) if recursive else directory.glob(pattern)
        paths = sorted(globber)
        if not paths:
            raise TrackLibraryError(f"no track files matching {pattern!r} in {directory}")

        loaded: list[tuple[Path, CenterlineTrack, str | None]] = []
        family_splits: dict[str, str] = {}
        for path in paths:
            track = CenterlineTrack.load(path)
            requested_split = (explicit_splits or {}).get(path.stem)
            family = track_family(track)
            if family and requested_split:
                key = _family_key(family)
                prior = family_splits.get(key)
                if prior is not None and prior != requested_split:
                    raise TrackLibraryError(
                        f"explicit splits assign family {family!r} to both {prior!r} and "
                        f"{requested_split!r}; all related maps must stay in one split"
                    )
                family_splits[key] = requested_split
            loaded.append((path, track, requested_split))

        duplicates: list[tuple[str, str]] = []
        for path, track, requested_split in loaded:
            family = track_family(track)
            split = requested_split
            if split is None and family:
                split = family_splits.get(_family_key(family))
            entry = library.add(track, split=split, source=str(path))

            # `add` returns the *existing* entry when the geometry is already present, so the
            # file was read but not added. Logging that as "loaded" would claim something that
            # did not happen, and the library would then look smaller than the log implies.
            if entry.track is not track:
                duplicates.append((track.name, entry.track.name))
                continue

            logger.info(
                "loaded track %r -> %s (%.0f m, %d points)",
                track.name,
                entry.split,
                track.length,
                track.num_points,
            )

        if duplicates:
            # Silent here means an operator believes they are training on N maps when they are
            # training on fewer, with no way to notice from the log.
            logger.warning(
                "skipped %d duplicate track file(s) with geometry identical to one already "
                "loaded: %s. Each distinct map must have distinct geometry, or it is the same "
                "map under another name.",
                len(duplicates),
                ", ".join(f"{name!r} == {kept!r}" for name, kept in duplicates),
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
            "num_split_groups": len({entry.split_group for entry in self.entries}),
            "counts": self.counts(),
            "families_by_split": {
                split: len(
                    {_family_key(entry.family) for entry in self.by_split(split) if entry.family is not None}
                )
                for split in SPLITS
            },
            "split_groups_by_split": {
                split: len({entry.split_group for entry in self.by_split(split)}) for split in SPLITS
            },
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
                    "family": e.family,
                    "split_group": e.split_group,
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
    "track_family",
    "track_identity",
]
