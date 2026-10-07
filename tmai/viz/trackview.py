"""Simplified 3D visualisation of the track, independent of the game's renderer.

The long-term requirement is a switchable view that shows the *track structure* rather than
Trackmania's normal rendering, so that what the agent perceives can be inspected. This module
is the phase-1 foundation for that: it renders the geometric model the agent actually uses --
centreline, drivable corridor, curvature and the car's position relative to it -- and it can
export the same geometry to Wavefront ``.obj`` for any external viewer or the future GUI.

Deliberately minimal and headless-safe:

* matplotlib with the ``Agg`` backend, so it works on a training host with no display.
* No game rendering, no screen capture, no window management.

What it cannot show yet: block-level structure (walls, ramps, obstacles). That needs map block
geometry from a ``.Map.Gbx`` parser -- see ``docs/ROADMAP.md``. Everything here is written
against :class:`~tmai.tracks.centerline.CenterlineTrack`, so adding block meshes later is
additive.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tmai.game.protocol import GameFrame
from tmai.tracks.centerline import CenterlineTrack, TrackProjection

logger = logging.getLogger(__name__)


@dataclass
class TrackViewConfig:
    """Rendering options."""

    show_corridor: bool = True
    show_curvature: bool = True
    show_car: bool = True
    show_trajectory: bool = True
    #: Metres of the ribbon drawn per side of the centreline.
    corridor_samples: int = 2
    figsize: tuple[float, float] = (9.0, 7.0)
    dpi: int = 110
    equal_aspect: bool = True


class TrackView:
    """Renders a track (and optionally the car and its trajectory) to an image or ``.obj``."""

    def __init__(self, track: CenterlineTrack, config: TrackViewConfig | None = None) -> None:
        self.track = track
        self.config = config or TrackViewConfig()
        self._corridor = self._build_corridor()

    # -- geometry -------------------------------------------------------------------

    def _build_corridor(self) -> tuple[np.ndarray, np.ndarray]:
        """Left and right corridor edges, each ``(N, 3)``, in that order."""
        points = self.track.points
        tangents = np.gradient(points, axis=0)
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / np.maximum(norms, 1e-9)
        # Must match CenterlineTrack.project(): Trackmania's world frame is left-handed with
        # +y up, so the right-hand side of a heading is cross(up, tangent).
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up[None, :], tangents)
        right_norm = np.linalg.norm(right, axis=1, keepdims=True)
        right = right / np.maximum(right_norm, 1e-9)
        half = self.track.corridor_half_width[:, None]
        return points - right * half, points + right * half

    @property
    def left_edge(self) -> np.ndarray:
        return self._corridor[0]

    @property
    def right_edge(self) -> np.ndarray:
        return self._corridor[1]

    def corridor_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Triangle-strip indices for the corridor surface, for ``.obj`` export."""
        n = len(self.track.points)
        left, right = self._corridor
        vertices = np.concatenate([left, right], axis=0)
        faces: list[tuple[int, int, int]] = []
        for i in range(n - 1):
            a, b, c, d = i, i + 1, n + i + 1, n + i
            faces.append((a, b, c))
            faces.append((a, c, d))
        return vertices, np.asarray(faces, dtype=np.int64)

    def to_obj(self, path: str | Path) -> Path:
        """Export the track surface as a Wavefront ``.obj`` file."""
        vertices, faces = self.corridor_mesh()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# TrackmaniaAI track surface: {self.track.name}",
            f"# points={self.track.num_points} length={self.track.length:.2f}m",
        ]
        lines += [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in vertices]
        # .obj faces are 1-indexed.
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
        # Centreline as a polyline for reference.
        offset = len(vertices)
        lines += [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in self.track.points]
        lines.append("l " + " ".join(str(offset + i + 1) for i in range(self.track.num_points)))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("wrote track mesh: %s (%d vertices, %d faces)", path, len(vertices), len(faces))
        return path

    # -- rendering ------------------------------------------------------------------

    def render(
        self,
        path: str | Path,
        *,
        frame: GameFrame | None = None,
        projection: TrackProjection | None = None,
        trajectory: Sequence[np.ndarray] | None = None,
        title: str | None = None,
    ) -> Path:
        """Render to a PNG. Safe headless: forces the non-interactive matplotlib backend."""
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        cfg = self.config
        fig = plt.figure(figsize=cfg.figsize, dpi=cfg.dpi)
        ax = fig.add_subplot(111, projection="3d")

        points = self.track.points
        ax.plot(points[:, 0], points[:, 2], points[:, 1], color="tab:blue", lw=1.6,
                label="centreline")

        if cfg.show_corridor:
            left, right = self._corridor
            verts = []
            for i in range(len(points) - 1):
                verts.append(
                    [
                        (left[i, 0], left[i, 2], left[i, 1]),
                        (left[i + 1, 0], left[i + 1, 2], left[i + 1, 1]),
                        (right[i + 1, 0], right[i + 1, 2], right[i + 1, 1]),
                        (right[i, 0], right[i, 2], right[i, 1]),
                    ]
                )
            surface = Poly3DCollection(verts, alpha=0.12, facecolor="tab:gray", edgecolor="none")
            ax.add_collection3d(surface)
            ax.plot(left[:, 0], left[:, 2], left[:, 1], color="tab:red", lw=0.7, label="corridor")
            ax.plot(right[:, 0], right[:, 2], right[:, 1], color="tab:red", lw=0.7)

        if cfg.show_curvature:
            curvature = self.track._curvature  # noqa: SLF001 - internal but stable geometry
            scaled = np.clip(curvature / max(np.abs(curvature).max(), 1e-9), -1.0, 1.0)
            ax.scatter(
                points[:-1, 0], points[:-1, 2], points[:-1, 1],
                c=scaled, cmap="coolwarm", s=6, vmin=-1, vmax=1, label="curvature",
            )

        if trajectory is not None and cfg.show_trajectory and len(trajectory) > 0:
            traj = np.asarray(trajectory, dtype=np.float64).reshape(-1, 3)
            ax.plot(traj[:, 0], traj[:, 2], traj[:, 1], color="tab:green", lw=1.0,
                    label="trajectory", alpha=0.8)

        if frame is not None and cfg.show_car:
            p = frame.vehicle.position
            ax.scatter([p[0]], [p[2]], [p[1]], color="black", s=45, label="car")
            fwd = frame.vehicle.forward_vector() * 4.0
            ax.quiver(p[0], p[2], p[1], fwd[0], fwd[2], fwd[1], color="black", linewidth=1.4)
            if projection is not None:
                c = projection.centre
                ax.plot(
                    [p[0], c[0]], [p[2], c[2]], [p[1], c[1]],
                    color="tab:orange", ls="--", lw=1.0, label="lateral offset",
                )

        if cfg.equal_aspect:
            _set_equal_aspect(ax, points)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_zlabel("y (m)")
        ax.set_title(title or f"track: {self.track.name}  ({self.track.length:.0f} m)")
        ax.legend(loc="upper left", fontsize=8)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("wrote track view: %s", path)
        return path


def _set_equal_aspect(ax: Any, points: np.ndarray) -> None:
    """Force equal scaling on all axes so the track shape is not distorted."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centre = (mins + maxs) / 2.0
    span = float((maxs - mins).max()) / 2.0 * 1.1 + 1e-3
    # Matplotlib's 3-D axes are (x, z, y) in the calls above: x, z horizontal, y vertical.
    ax.set_xlim(centre[0] - span, centre[0] + span)
    ax.set_ylim(centre[2] - span, centre[2] + span)
    ax.set_zlim(centre[1] - span, centre[1] + span)


__all__ = ["TrackView", "TrackViewConfig"]
