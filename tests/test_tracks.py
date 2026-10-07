"""Tests for track geometry, synthetic generators and centreline recording."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tmai.tracks.centerline import SCHEMA_VERSION, CenterlineTrack
from tmai.tracks.recording import CenterlineRecorder, _deduplicate, record_track
from tmai.tracks.synthetic import (
    SYNTHETIC_TRACKS,
    build_synthetic,
    figure_eight,
    oval,
    s_curve,
    straight,
)


class TestConstruction:
    def test_length_and_segments(self, straight_track):
        assert straight_track.length == pytest.approx(200.0)
        assert straight_track.num_points == 201

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            CenterlineTrack(np.zeros((5, 2)))

    def test_rejects_too_few_points(self):
        with pytest.raises(ValueError, match="at least 2"):
            CenterlineTrack(np.zeros((1, 3)))

    def test_rejects_non_finite(self):
        pts = np.zeros((3, 3))
        pts[1, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            CenterlineTrack(pts)

    def test_rejects_duplicate_points(self):
        pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="zero-length segment"):
            CenterlineTrack(pts)

    def test_rejects_non_positive_corridor(self, ):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="positive"):
            CenterlineTrack(pts, corridor_half_width=0.0)

    def test_per_point_corridor_width_is_interpolated(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        track = CenterlineTrack(pts, corridor_half_width=[3.0, 4.0, 5.0])
        # Widths are per point, lookups are per arc length: the result must interpolate, and
        # the final sample must be reachable (it is not, if indexing by segment).
        assert track.corridor_half_width_at(0.0) == pytest.approx(3.0)
        assert track.corridor_half_width_at(0.5) == pytest.approx(3.5)
        assert track.corridor_half_width_at(1.5) == pytest.approx(4.5)
        assert track.corridor_half_width_at(2.0) == pytest.approx(5.0)

    def test_corridor_length_mismatch_rejected(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="one entry per point"):
            CenterlineTrack(pts, corridor_half_width=[3.0, 4.0])


class TestProjection:
    def test_point_on_centreline_has_zero_lateral(self, straight_track):
        projection = straight_track.project([0.0, 0.0, 50.0])
        assert projection.progress == pytest.approx(50.0)
        assert projection.lateral_offset == pytest.approx(0.0, abs=1e-9)
        assert projection.distance == pytest.approx(0.0, abs=1e-9)

    def test_lateral_sign_is_right_of_travel(self, straight_track):
        # The straight runs along +z, so +x is to the right of travel.
        right = straight_track.project([3.0, 0.0, 50.0])
        left = straight_track.project([-3.0, 0.0, 50.0])
        assert right.lateral_offset > 0
        assert left.lateral_offset < 0
        assert right.lateral_offset == pytest.approx(-left.lateral_offset)

    def test_progress_is_clamped_at_both_ends(self, straight_track):
        assert straight_track.project([0.0, 0.0, -500.0]).progress == 0.0
        assert straight_track.project([0.0, 0.0, 5000.0]).progress == pytest.approx(200.0)

    def test_hint_does_not_change_result_for_nearby_points(self, s_curve_track):
        target = s_curve_track.point_at(120.0) + np.array([2.0, 0.0, 0.0])
        global_projection = s_curve_track.project(target)
        hinted = s_curve_track.project(target, hint_s=115.0, search_window=50.0)
        assert hinted.progress == pytest.approx(global_projection.progress, abs=1e-6)
        assert hinted.lateral_offset == pytest.approx(global_projection.lateral_offset, abs=1e-6)

    def test_rejects_non_finite_position(self, straight_track):
        with pytest.raises(ValueError, match="non-finite"):
            straight_track.project([np.nan, 0.0, 0.0])

    def test_vertical_offset_tracks_height(self, straight_track):
        projection = straight_track.project([0.0, 7.5, 50.0])
        assert projection.vertical_offset == pytest.approx(7.5)

    def test_edge_distances_sum_to_corridor_width(self, straight_track):
        projection = straight_track.project([2.0, 0.0, 50.0])
        left, right = straight_track.edge_distances(projection)
        half = straight_track.corridor_half_width_at(50.0)
        assert left + right == pytest.approx(2 * half)
        # Offset to the right means less room on the right.
        assert right < left

    def test_is_on_track(self, straight_track):
        half = straight_track.corridor_half_width_at(50.0)
        assert straight_track.is_on_track(straight_track.project([0.0, 0.0, 50.0]))
        assert not straight_track.is_on_track(
            straight_track.project([half + 1.0, 0.0, 50.0])
        )


class TestCurvature:
    def test_straight_has_zero_curvature(self, straight_track):
        assert np.allclose(straight_track._curvature, 0.0, atol=1e-12)

    def test_oval_has_consistent_sign(self, oval_track):
        curvature = oval_track._curvature
        # A convex circuit turns the same way everywhere.
        assert np.all(curvature[:-1] >= -1e-12) or np.all(curvature[:-1] <= 1e-12)
        assert abs(curvature).mean() > 0

    def test_s_curve_alternates_sign(self, s_curve_track):
        curvature = s_curve_track._curvature
        assert (curvature > 0).any() and (curvature < 0).any()

    def test_lookahead_sampling(self, s_curve_track):
        values = s_curve_track.sample_lookahead(100.0, [5.0, 10.0, 20.0])
        assert values.shape == (3,)
        assert np.all(np.isfinite(values))

    def test_point_at_and_heading_at_agree(self, s_curve_track):
        s = 87.5
        point = s_curve_track.point_at(s)
        heading = s_curve_track.heading_at(s)
        np.testing.assert_allclose(np.linalg.norm(heading), 1.0, atol=1e-9)
        # Stepping forward along the heading must increase arc length.
        ahead = s_curve_track.project(point + heading * 1.0, hint_s=s, search_window=20.0)
        assert ahead.progress > s


class TestHeadingError:
    def test_aligned_heading_gives_zero_error(self, straight_track):
        projection = straight_track.project([0.0, 0.0, 50.0])
        tangent = projection.tangent
        yaw = float(np.arctan2(tangent[0], tangent[2]))
        assert straight_track.heading_error(projection, yaw) == pytest.approx(0.0, abs=1e-9)

    def test_error_wraps_into_pi(self, straight_track):
        projection = straight_track.project([0.0, 0.0, 50.0])
        tangent = projection.tangent
        yaw = float(np.arctan2(tangent[0], tangent[2]))
        # 3*pi beyond alignment must wrap to pi, not report 3*pi.
        error = straight_track.heading_error(projection, yaw + 3.0 * np.pi)
        assert abs(error) <= np.pi + 1e-9


class TestSerialisation:
    def test_roundtrip_preserves_everything(self, tmp_path, s_curve_track):
        path = tmp_path / "track.json"
        s_curve_track.save(path)
        loaded = CenterlineTrack.load(path)
        assert loaded.name == s_curve_track.name
        assert loaded.uid == s_curve_track.uid
        assert loaded.length == pytest.approx(s_curve_track.length)
        np.testing.assert_allclose(loaded.points, s_curve_track.points)
        np.testing.assert_allclose(
            loaded.corridor_half_width, s_curve_track.corridor_half_width
        )
        assert loaded.metadata == s_curve_track.metadata

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CenterlineTrack.load(tmp_path / "nope.json")

    def test_wrong_schema_version_rejected(self, tmp_path, straight_track):
        path = tmp_path / "bad.json"
        data = straight_track.to_dict()
        data["schema_version"] = SCHEMA_VERSION + 1
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="schema version"):
            CenterlineTrack.load(path)

    def test_wrong_kind_rejected(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "kind": "other"}), encoding="utf-8")
        with pytest.raises(ValueError, match="not a centerline track"):
            CenterlineTrack.load(path)


class TestSynthetic:
    @pytest.mark.parametrize("name", sorted(SYNTHETIC_TRACKS))
    def test_generators_produce_valid_tracks(self, name):
        track = build_synthetic(name)
        assert track.num_points >= 2
        assert track.length > 0
        assert np.all(np.isfinite(track.points))
        # Projection must work everywhere along the track.
        for fraction in (0.0, 0.25, 0.5, 0.75, 0.99):
            projection = track.project(track.point_at(track.length * fraction))
            assert np.isfinite(projection.progress)

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError, match="unknown synthetic track"):
            build_synthetic("nurburgring")

    def test_closed_flag_set_for_circuits(self):
        assert oval().closed is True
        assert figure_eight().closed is True
        assert straight().closed is False
        assert s_curve().closed is False

    def test_metadata_marks_them_synthetic(self):
        assert oval().metadata["synthetic"] is True


class TestRecorder:
    def test_decimates_by_distance(self):
        recorder = CenterlineRecorder(min_spacing=2.0)
        kept = [recorder.add([0.0, 0.0, float(i)]) for i in range(10)]
        assert sum(kept) == 5  # 0, 2, 4, 6, 8
        assert len(recorder) == 5

    def test_ignores_non_finite(self):
        recorder = CenterlineRecorder()
        assert recorder.add([np.nan, 0.0, 0.0]) is False
        assert len(recorder) == 0

    def test_needs_two_points_to_build(self):
        recorder = CenterlineRecorder(min_spacing=0.0)
        recorder.add([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="at least 2"):
            recorder.build_track(name="t")

    def test_build_track_length(self):
        recorder = CenterlineRecorder(min_spacing=1.0, smoothing_window=1)
        for i in range(101):
            recorder.add([0.0, 0.0, float(i)])
        track = recorder.build_track(name="recorded")
        assert track.length == pytest.approx(100.0, rel=1e-6)
        assert track.metadata["source"] == "recorded_telemetry"
        assert track.metadata["raw_samples"] == 101

    def test_smoothing_requires_odd_window(self):
        recorder = CenterlineRecorder(smoothing_window=4)
        for i in range(20):
            recorder.add([0.0, 0.0, float(i)])
        with pytest.raises(ValueError, match="odd"):
            recorder.build_track(name="t")

    def test_smoothing_reduces_jitter(self):
        rng = np.random.default_rng(0)
        clean = np.stack(
            [np.zeros(200), np.zeros(200), np.arange(200.0)], axis=1
        )
        noisy = clean + rng.normal(0.0, 0.4, size=clean.shape)
        noisy[:, 2] = np.arange(200.0)  # keep monotonic progress

        raw = CenterlineRecorder(min_spacing=1.0, smoothing_window=1)
        smooth = CenterlineRecorder(min_spacing=1.0, smoothing_window=9)
        for point in noisy:
            raw.add(point)
            smooth.add(point)

        raw_track = raw.build_track(name="raw")
        smooth_track = smooth.build_track(name="smooth")
        raw_lateral = np.abs(np.diff(raw_track.points[:, 0])).mean()
        smooth_lateral = np.abs(np.diff(smooth_track.points[:, 0])).mean()
        assert smooth_lateral < raw_lateral

    def test_deduplicate_removes_near_duplicates(self):
        pts = np.array([[0.0, 0.0, 0.0], [1e-6, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert len(_deduplicate(pts, eps=1e-3)) == 2


class TestRecordTrackFromDriver:
    """Exercises the recording path against the simulated driver."""

    def test_records_and_saves_a_track(self, tmp_path, s_curve_track):
        from tmai.game.protocol import Action
        from tmai.game.simulated import SimulatedGameDriver

        driver = SimulatedGameDriver(s_curve_track)
        driver.open()
        out = tmp_path / "rec.json"
        # Drive with a gentle correcting controller so the recording stays near the line.
        def controller(frame):
            projection = s_curve_track.project(frame.vehicle.position, hint_s=None)
            return Action(steer=float(np.clip(-projection.lateral_offset * 0.1, -1, 1)),
                          throttle=0.8)

        track = record_track(
            driver,
            out_path=str(out),
            name="recorded",
            min_spacing=2.0,
            action_provider=controller,
        )
        driver.close()
        assert out.exists()
        assert track.num_points >= 2
        assert track.length > 0
        reloaded = CenterlineTrack.load(out)
        assert reloaded.num_points == track.num_points

    def test_stops_on_should_stop(self, tmp_path, straight_track):
        from tmai.game.protocol import Action
        from tmai.game.simulated import SimulatedGameDriver

        driver = SimulatedGameDriver(straight_track)
        driver.open()
        calls = {"n": 0}

        def stop_after_20():
            calls["n"] += 1
            return calls["n"] > 20

        track = record_track(
            driver,
            out_path=str(tmp_path / "rec.json"),
            name="partial",
            min_spacing=1.0,
            should_stop=stop_after_20,
            # The car has to actually move, otherwise every sample is the same position and
            # the distance-based decimation keeps only one point.
            action_provider=lambda frame: Action(throttle=1.0),
        )
        driver.close()
        assert track.num_points >= 2
        assert calls["n"] == 21


class TestPointAtArcLength:
    """Regression: point_at() once advanced by the bare fraction of a segment.

    ``_tangents`` are unit vectors, so multiplying by ``frac`` moved the point ``frac``
    metres along the segment instead of ``frac`` of the way down it. That is only correct
    when every segment is exactly one metre -- true of the default synthetic tracks
    (``spacing=1.0``) and of nothing recorded from a real map, which is why the bug survived.
    """

    @staticmethod
    def _track(spacing: float, count: int = 5) -> CenterlineTrack:
        pts = np.zeros((count, 3))
        pts[:, 2] = np.arange(count) * spacing
        return CenterlineTrack(pts, name=f"spacing{spacing}")

    @pytest.mark.parametrize("spacing", [1.0, 2.0, 4.0, 7.5])
    def test_point_at_roundtrips_through_projection(self, spacing):
        track = self._track(spacing)
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            s = fraction * track.length
            position = track.point_at(s)
            assert track.project(position).progress == pytest.approx(s, abs=1e-9)

    @pytest.mark.parametrize("spacing", [1.0, 3.0, 8.0])
    def test_point_at_advances_by_metres_not_fractions(self, spacing):
        track = self._track(spacing, count=4)
        half = spacing / 2.0
        position = track.point_at(half)
        # Mid-way through the first segment must be half a segment along, in metres.
        assert position[2] == pytest.approx(half)

    def test_point_at_is_continuous_across_segment_boundaries(self):
        track = self._track(4.0, count=6)
        before = track.point_at(4.0 - 1e-9)
        after = track.point_at(4.0 + 1e-9)
        assert np.linalg.norm(before - after) < 1e-6

    def test_endpoints(self):
        track = self._track(5.0, count=4)
        assert track.point_at(0.0) == pytest.approx(track.points[0])
        assert track.point_at(track.length) == pytest.approx(track.points[-1])

    def test_clamped_beyond_the_ends(self):
        track = self._track(4.0, count=3)
        assert track.point_at(-50.0) == pytest.approx(track.points[0])
        assert track.point_at(1e6) == pytest.approx(track.points[-1])
