"""``.Map.Gbx`` ingestion, deliberately isolated.

.. warning::
   **Nothing in this module has been verified against a real Trackmania map file.**

   The sandbox this was developed in has no Windows, no Trackmania and no ``.Map.Gbx`` sample,
   so a parser written here could not be tested against the thing it is supposed to parse.
   Shipping one would have been a guess presented as a feature -- exactly the failure this
   project is trying to avoid. So the *shape* of the integration is here, and the parsing
   itself raises until it has been verified on a real file.

What is implemented and tested here:

* :class:`MapBlock` / :class:`MapGeometry` -- the data model a parser must produce. This is
  the part that matters for the RL stack, and it is fully testable without the game.
* :func:`blocks_to_centerline` -- converts block geometry into a drivable centreline. Pure
  geometry, fully unit-tested, and the piece that will actually feed the agent.
* :class:`GbxSource` -- the isolated boundary. It raises :class:`GbxUnavailableError` with a
  precise account of what is needed.

Why the third-party option was not wired up
-------------------------------------------
``pygbx`` (PyPI, GPL-3, last released 2021-05) is the only maintained-looking Python GBX
parser. It was rejected for this phase on three concrete grounds:

1. It targets **TMNF/TMUF** and documents only partial TM2 support. Trackmania (2020) maps
   are a newer format version, so success is not assured.
2. It requires ``python-lzo``, a C extension that needs system LZO headers -- a fragile
   dependency for a project that otherwise installs cleanly.
3. It is GPL-3. ``tminterface`` already is, so this is not a new constraint, but it is one
   more.

The right time to revisit is with a real map file in hand: parse it, compare the extracted
blocks against the map in the editor, and only then trust the output.

What is needed to finish this (see docs/LIMITATIONS.md):

1. A real ``.Map.Gbx`` from Trackmania (2020), Stadium environment.
2. A verified block extraction: name, position, orientation, size.
3. Block-name -> surface metadata (drivable or not, width) for the map's block set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tmai.tracks.centerline import CenterlineTrack

logger = logging.getLogger(__name__)

GBX_SCHEMA_VERSION = 1


class GbxUnavailableError(NotImplementedError):
    """Raised instead of returning plausible-looking but unverified data.

    Subclasses :class:`NotImplementedError` so callers can distinguish "this integration is
    deliberately unfinished" from "this file is corrupt".
    """


@dataclass(frozen=True)
class MapBlock:
    """One placed block in a map, in the units Trackmania uses.

    Args:
        name: the block model name, e.g. ``"PlatformRoad"``.
        position: world position of the block origin.
        rotation: 3x3 orientation matrix.
        size: bounding-box extents along the block's local axes.
        drivable: whether the block provides a drivable surface. Unknown blocks should be
            ``None`` rather than guessed -- the conversion step treats ``None`` as "not
            part of the racing line".
    """

    name: str
    position: np.ndarray
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    size: np.ndarray = field(default_factory=lambda: np.ones(3))
    drivable: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", np.asarray(self.position, dtype=np.float64).reshape(3))
        object.__setattr__(self, "rotation", np.asarray(self.rotation, dtype=np.float64).reshape(3, 3))
        object.__setattr__(self, "size", np.asarray(self.size, dtype=np.float64).reshape(3))


@dataclass
class MapGeometry:
    """Everything a parser must hand over for the RL stack to work."""

    #: Every block in the map.
    blocks: list[MapBlock] = field(default_factory=list)
    #: Map display name, when the file records one.
    name: str = ""
    #: The map UID, which is the stable identity used for train/test split assignment.
    uid: str | None = None
    #: Author-declared environment/collection, e.g. ``"Stadium"``.
    environment: str = ""
    #: Provenance of this geometry, recorded into the track metadata.
    source: str = "gbx"

    @property
    def drivable_blocks(self) -> list[MapBlock]:
        return [b for b in self.blocks if b.drivable is True]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GBX_SCHEMA_VERSION,
            "name": self.name,
            "uid": self.uid,
            "environment": self.environment,
            "source": self.source,
            "num_blocks": len(self.blocks),
            "num_drivable_blocks": len(self.drivable_blocks),
        }


def blocks_to_centerline(
    geometry: MapGeometry,
    *,
    corridor_half_width: float = 5.0,
    spacing: float = 2.0,
) -> CenterlineTrack:
    """Derive a centreline from block geometry.

    This is the part of GBX support that is genuinely verifiable without the game, and it is
    where the real work lives. Given a set of drivable blocks it produces an ordered
    centreline by walking from block to block through nearest neighbours.

    The nearest-neighbour walk is a deliberate simplification: it produces a usable reference
    line for simple maps and a visibly wrong one for maps with branches or overlapping
    circuits. That is the right trade here -- a wrong line is obvious when plotted with
    ``tmai show-track``, whereas a plausible-looking wrong line is not.

    Args:
        geometry: parsed map geometry.
        corridor_half_width: metres of drivable corridor either side of the line. A constant
            until per-block widths are used.
        spacing: resampling distance along the line, metres.

    Raises:
        ValueError: if fewer than two drivable blocks are available.
    """
    drivable = geometry.drivable_blocks
    if len(drivable) < 2:
        raise ValueError(
            f"need at least 2 drivable blocks to build a centreline, got {len(drivable)}; "
            "check that the block set has drivable surfaces annotated"
        )

    points = np.array([b.position for b in drivable], dtype=np.float64)
    order = _nearest_neighbour_order(points)
    ordered = points[order]

    # Drop consecutive duplicates: two blocks at the same position produce a zero-length
    # segment, which CenterlineTrack rejects.
    keep = [ordered[0]]
    for point in ordered[1:]:
        if float(np.linalg.norm(point - keep[-1])) > 1e-6:
            keep.append(point)
    if len(keep) < 2:
        raise ValueError("all drivable blocks are at the same position")

    path = np.asarray(keep, dtype=np.float64)
    path = _resample(path, spacing)

    return CenterlineTrack(
        path,
        name=geometry.name or "gbx_track",
        uid=geometry.uid,
        corridor_half_width=corridor_half_width,
        closed=False,
        metadata={
            "source": geometry.source,
            "environment": geometry.environment,
            "num_blocks": len(geometry.blocks),
            "num_drivable_blocks": len(drivable),
            "centerline_method": "nearest_neighbour_walk",
            "spacing": spacing,
        },
    )


def _nearest_neighbour_order(points: np.ndarray) -> np.ndarray:
    """Greedy nearest-neighbour ordering, starting from the block furthest from the centroid.

    Starting at an extremity rather than at ``points[0]`` matters: block order in a map file
    is editor insertion order, which has no relationship to the racing line.
    """
    centroid = points.mean(axis=0)
    start = int(np.argmax(np.linalg.norm(points - centroid, axis=1)))

    remaining = set(range(len(points)))
    remaining.discard(start)
    order = [start]
    current = start
    while remaining:
        candidates = np.array(sorted(remaining))
        distances = np.linalg.norm(points[candidates] - points[current], axis=1)
        current = int(candidates[int(np.argmin(distances))])
        order.append(current)
        remaining.discard(current)
    return np.asarray(order, dtype=int)


def _resample(path: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline to roughly uniform arc-length spacing."""
    if spacing <= 0:
        return path
    deltas = np.diff(path, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    if total <= 0:
        return path
    count = max(2, int(total / spacing) + 1)
    targets = np.linspace(0.0, total, count)
    out = np.empty((count, 3), dtype=np.float64)
    for i, target in enumerate(targets):
        segment = int(np.clip(np.searchsorted(cumulative, target, side="right") - 1,
                              0, len(lengths) - 1))
        denom = max(lengths[segment], 1e-9)
        fraction = (target - cumulative[segment]) / denom
        out[i] = path[segment] + deltas[segment] * np.clip(fraction, 0.0, 1.0)
    return out


class GbxSource:
    """The isolated boundary for ``.Map.Gbx`` ingestion.

    Constructing one is cheap and safe. Calling :meth:`load` raises until the parser has been
    verified against real map files, with a message that says exactly what is missing.
    """

    def __init__(self, *, backend: str = "none") -> None:
        self.backend = backend

    def load(self, path: str) -> MapGeometry:
        """Parse a ``.Map.Gbx`` file into :class:`MapGeometry`.

        Raises:
            GbxUnavailableError: always, in this phase.
        """
        raise GbxUnavailableError(
            "GBX parsing is not implemented, and deliberately so: it has never been verified "
            f"against a real Trackmania map (attempted to load {path!r}).",
        )

    @staticmethod
    def requirements() -> list[str]:
        """What must be true before this can be implemented honestly."""
        return [
            "A real .Map.Gbx file from Trackmania (2020), Stadium environment, to test against.",
            "A verified block extraction: model name, position, orientation and size.",
            "A block-name -> surface table for the map's block set, marking drivable blocks.",
            "A cross-check of the derived centreline against the same map driven manually "
            "(tmai record-track), so parser errors are visible rather than assumed away.",
        ]


__all__ = [
    "GBX_SCHEMA_VERSION",
    "GbxSource",
    "GbxUnavailableError",
    "MapBlock",
    "MapGeometry",
    "blocks_to_centerline",
]
