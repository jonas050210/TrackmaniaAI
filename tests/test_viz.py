"""Tests for the simplified track visualisation and mesh export."""

from __future__ import annotations

import numpy as np
import pytest

from tmai.game.simulated import SimulatedGameDriver
from tmai.viz.trackview import TrackView, TrackViewConfig


class TestCorridorGeometry:
    def test_edges_are_offset_by_the_half_width(self, straight_track):
        view = TrackView(straight_track)
        half = straight_track.corridor_half_width[0]
        # The straight runs along +z with +x to the right, so the left edge is at -x.
        np.testing.assert_allclose(view.left_edge[0, 0], -half, atol=1e-9)
        np.testing.assert_allclose(view.right_edge[0, 0], +half, atol=1e-9)

    def test_edges_match_the_track_model_lateral_convention(self, straight_track):
        """The renderer must agree with CenterlineTrack about which side is 'right'."""
        view = TrackView(straight_track)
        right_point = view.right_edge[50]
        projection = straight_track.project(right_point)
        # A point on the right edge has a positive lateral offset by definition.
        assert projection.lateral_offset > 0
        assert projection.lateral_offset == pytest.approx(
            straight_track.corridor_half_width_at(projection.progress), rel=1e-3
        )

        left_point = view.left_edge[50]
        assert straight_track.project(left_point).lateral_offset < 0

    def test_edges_are_the_right_shape(self, s_curve_track):
        view = TrackView(s_curve_track)
        assert view.left_edge.shape == (s_curve_track.num_points, 3)
        assert view.right_edge.shape == (s_curve_track.num_points, 3)
        assert np.all(np.isfinite(view.left_edge))


class TestMesh:
    def test_mesh_vertex_and_face_counts(self, straight_track):
        view = TrackView(straight_track)
        vertices, faces = view.corridor_mesh()
        assert vertices.shape == (2 * straight_track.num_points, 3)
        # Two triangles per quad, one quad per segment.
        assert faces.shape == (2 * (straight_track.num_points - 1), 3)

    def test_face_indices_are_in_range(self, oval_track):
        view = TrackView(oval_track)
        vertices, faces = view.corridor_mesh()
        assert faces.min() >= 0
        assert faces.max() < len(vertices)

    def test_obj_export(self, tmp_path, s_curve_track):
        view = TrackView(s_curve_track)
        path = view.to_obj(tmp_path / "track.obj")
        content = path.read_text(encoding="utf-8")
        assert content.startswith("# TrackmaniaAI track surface")
        assert "s_curve" in content

        vertices = [line for line in content.splitlines() if line.startswith("v ")]
        faces = [line for line in content.splitlines() if line.startswith("f ")]
        lines = [line for line in content.splitlines() if line.startswith("l ")]
        # Corridor vertices plus the centreline polyline.
        assert len(vertices) == 3 * s_curve_track.num_points
        assert len(faces) == 2 * (s_curve_track.num_points - 1)
        assert len(lines) == 1

    def test_obj_faces_are_one_indexed(self, tmp_path, straight_track):
        path = TrackView(straight_track).to_obj(tmp_path / "t.obj")
        faces = [
            [int(x) for x in line.split()[1:]]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("f ")
        ]
        assert min(min(f) for f in faces) >= 1

    def test_obj_creates_parent_directories(self, tmp_path, straight_track):
        path = TrackView(straight_track).to_obj(tmp_path / "nested" / "dir" / "t.obj")
        assert path.exists()


class TestRendering:
    def test_renders_a_png(self, tmp_path, s_curve_track):
        path = TrackView(s_curve_track).render(tmp_path / "view.png")
        assert path.exists()
        assert path.stat().st_size > 2000
        # PNG magic number.
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_renders_with_car_and_trajectory(self, tmp_path, straight_track):
        driver = SimulatedGameDriver(straight_track)
        driver.open()
        frame = driver.reset()
        trajectory = []
        for _ in range(20):
            from tmai.game.protocol import Action

            frame = driver.step(Action(throttle=1.0))
            trajectory.append(frame.vehicle.position.copy())
        driver.close()

        projection = straight_track.project(frame.vehicle.position)
        path = TrackView(straight_track).render(
            tmp_path / "car.png",
            frame=frame,
            projection=projection,
            trajectory=trajectory,
            title="with car",
        )
        assert path.exists()
        assert path.stat().st_size > 2000

    def test_options_can_disable_layers(self, tmp_path, oval_track):
        config = TrackViewConfig(show_corridor=False, show_curvature=False)
        path = TrackView(oval_track, config).render(tmp_path / "plain.png")
        assert path.exists()

    def test_creates_parent_directories(self, tmp_path, oval_track):
        path = TrackView(oval_track).render(tmp_path / "a" / "b" / "v.png")
        assert path.exists()

    def test_headless_backend_is_used(self, tmp_path, oval_track):
        """The renderer must not require a display."""
        import matplotlib

        TrackView(oval_track).render(tmp_path / "v.png")
        assert matplotlib.get_backend().lower() == "agg"
