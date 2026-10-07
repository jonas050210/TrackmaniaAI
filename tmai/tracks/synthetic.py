"""Procedurally generated centreline tracks for tests, CI and the simulated driver.

These are **not** Trackmania maps. They are smooth parametric curves used to give the
pipeline something to run against when no real map data has been recorded yet. Real track
data comes from :func:`tmai.tracks.recording.record_centerline` (drive the real map once and
save the telemetry) or, later, from a ``.Map.Gbx`` parser.
"""

from __future__ import annotations

import numpy as np

from tmai.tracks.centerline import CenterlineTrack


def _make(points: np.ndarray, name: str, closed: bool, corridor: float, metadata: dict) -> CenterlineTrack:
    return CenterlineTrack(
        points,
        name=name,
        uid=f"synthetic:{name}",
        corridor_half_width=corridor,
        closed=closed,
        metadata={"synthetic": True, "generator": "tmai.tracks.synthetic", **metadata},
    )


def straight(length: float = 200.0, spacing: float = 1.0, corridor: float = 6.0) -> CenterlineTrack:
    """A flat straight along +z. The minimal useful track."""
    n = max(2, int(length / spacing) + 1)
    pts = np.zeros((n, 3), dtype=np.float64)
    pts[:, 2] = np.linspace(0.0, length, n)
    return _make(pts, "straight", closed=False, corridor=corridor,
                 metadata={"kind": "straight", "length": length})


def oval(length: float = 90.0, width: float = 50.0, spacing: float = 1.0,
         corridor: float = 6.0) -> CenterlineTrack:
    """A closed oval circuit in the ground plane."""
    perimeter = np.pi * (1.5 * (length + width) / 2 - np.sqrt(length * width))
    n = max(16, int(perimeter / spacing))
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = np.zeros((n, 3), dtype=np.float64)
    pts[:, 0] = (length / 2.0) * np.cos(t)
    pts[:, 2] = (width / 2.0) * np.sin(t)
    return _make(pts, "oval", closed=True, corridor=corridor,
                 metadata={"kind": "oval", "length": length, "width": width})


def s_curve(length: float = 240.0, amplitude: float = 25.0, cycles: float = 2.0,
            spacing: float = 1.0, corridor: float = 6.0) -> CenterlineTrack:
    """A sinusoidal point-to-point track with alternating left/right corners."""
    n = max(8, int(length / spacing) + 1)
    pts = np.zeros((n, 3), dtype=np.float64)
    pts[:, 2] = np.linspace(0.0, length, n)
    pts[:, 0] = amplitude * np.sin(2.0 * np.pi * cycles * pts[:, 2] / length)
    return _make(pts, "s_curve", closed=False, corridor=corridor,
                 metadata={"kind": "s_curve", "length": length, "amplitude": amplitude})


def figure_eight(radius: float = 40.0, spacing: float = 1.0,
                 corridor: float = 6.0) -> CenterlineTrack:
    """A closed figure-eight (lemniscate) -- exercises a self-crossing centreline."""
    n = max(32, int(2.0 * np.pi * radius / spacing))
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    denom = 1.0 + np.sin(t) ** 2
    pts = np.zeros((n, 3), dtype=np.float64)
    pts[:, 0] = radius * np.cos(t) / denom
    pts[:, 2] = radius * np.sin(t) * np.cos(t) / denom
    return _make(pts, "figure_eight", closed=True, corridor=corridor,
                 metadata={"kind": "figure_eight", "radius": radius})


#: Named generators, used by ``tmai train --synthetic-track <name>`` and by the tests.
SYNTHETIC_TRACKS = {
    "straight": straight,
    "oval": oval,
    "s_curve": s_curve,
    "figure_eight": figure_eight,
}


def build_synthetic(name: str, **kwargs) -> CenterlineTrack:
    """Instantiate one of the synthetic tracks by name."""
    if name not in SYNTHETIC_TRACKS:
        raise KeyError(
            f"unknown synthetic track {name!r}; available: {sorted(SYNTHETIC_TRACKS)}"
        )
    return SYNTHETIC_TRACKS[name](**kwargs)


__all__ = ["SYNTHETIC_TRACKS", "build_synthetic", "figure_eight", "oval", "s_curve", "straight"]
