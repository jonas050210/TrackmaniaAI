"""Tests for the track library: identities, splits, leakage prevention, sampling, stats."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tmai.tracks.centerline import CenterlineTrack
from tmai.tracks.library import (
    SPLITS,
    TrackLibrary,
    TrackLibraryError,
    TrackSampler,
    split_for_identity,
    track_identity,
)
from tmai.tracks.stats import compute_stats
from tmai.tracks.synthetic import build_synthetic, figure_eight, oval, s_curve, straight


def _track(name: str = "t", length: float = 100.0, uid: str | None = None) -> CenterlineTrack:
    pts = np.zeros((int(length) + 1, 3))
    pts[:, 2] = np.arange(len(pts))
    return CenterlineTrack(pts, name=name, uid=uid, corridor_half_width=5.0)


# -- identity -----------------------------------------------------------------------


class TestTrackIdentity:
    def test_same_geometry_hashes_the_same(self):
        assert track_identity(_track("a")) == track_identity(_track("b"))

    def test_different_geometry_hashes_differently(self):
        assert track_identity(_track(length=100.0)) != track_identity(_track(length=120.0))

    def test_uid_is_preferred_over_geometry(self):
        track = _track(uid="ABC123")
        assert track_identity(track) == "uid:ABC123"

    def test_synthetic_uids_are_not_treated_as_map_uids(self):
        """Synthetic tracks get a generated uid; trusting it would collide across variants."""
        a = build_synthetic("oval", length=90.0)
        b = build_synthetic("oval", length=120.0)
        assert a.uid == b.uid  # both "synthetic:oval"
        assert track_identity(a) != track_identity(b)  # but identities differ

    def test_closed_flag_is_part_of_the_identity(self):
        pts = np.zeros((10, 3))
        pts[:, 2] = np.arange(10)
        open_track = CenterlineTrack(pts, name="o", closed=False)
        closed_track = CenterlineTrack(pts, name="c", closed=True)
        assert track_identity(open_track) != track_identity(closed_track)


class TestSplitAssignment:
    WEIGHTS = {"train": 0.7, "validation": 0.15, "test": 0.15}

    def test_is_deterministic(self):
        first = split_for_identity("uid:map42", weights=self.WEIGHTS)
        for _ in range(20):
            assert split_for_identity("uid:map42", weights=self.WEIGHTS) == first

    def test_returns_a_known_split(self):
        for i in range(200):
            assert split_for_identity(f"uid:m{i}", weights=self.WEIGHTS) in SPLITS

    def test_matches_the_requested_proportions(self):
        counts = {name: 0 for name in SPLITS}
        n = 4000
        for i in range(n):
            counts[split_for_identity(f"uid:map{i}", weights=self.WEIGHTS)] += 1
        assert counts["train"] / n == pytest.approx(0.70, abs=0.03)
        assert counts["validation"] / n == pytest.approx(0.15, abs=0.03)
        assert counts["test"] / n == pytest.approx(0.15, abs=0.03)

    def test_zero_weight_split_is_never_used(self):
        weights = {"train": 1.0, "validation": 0.0, "test": 0.0}
        assert {split_for_identity(f"uid:m{i}", weights=weights) for i in range(300)} == {"train"}

    def test_rejects_weights_that_sum_to_zero(self):
        with pytest.raises(TrackLibraryError):
            split_for_identity("uid:x", weights={"train": 0.0, "validation": 0.0, "test": 0.0})


# -- library ------------------------------------------------------------------------


class TestTrackLibrary:
    def test_add_assigns_a_split(self):
        library = TrackLibrary()
        entry = library.add(_track())
        assert entry.split in SPLITS
        assert len(library) == 1

    def test_explicit_split_is_honoured(self):
        library = TrackLibrary()
        entry = library.add(_track(), split="test")
        assert entry.split == "test"
        assert library.test and not library.train

    def test_unknown_split_is_rejected(self):
        with pytest.raises(TrackLibraryError, match="unknown split"):
            TrackLibrary().add(_track(), split="holdout")

    def test_same_track_twice_is_deduplicated(self):
        library = TrackLibrary()
        first = library.add(_track(), split="train")
        second = library.add(_track(), split="train")
        assert len(library) == 1
        assert first is second

    def test_duplicate_across_splits_is_refused(self):
        """The leakage this whole module exists to prevent."""
        library = TrackLibrary()
        library.add(_track(uid="SAME"), split="train")
        with pytest.raises(TrackLibraryError, match="already in the 'train' split"):
            library.add(_track(uid="SAME"), split="test")

    def test_counts_and_split_accessors(self):
        library = TrackLibrary()
        library.add(_track(uid="a"), split="train")
        library.add(_track(uid="b"), split="train")
        library.add(_track(uid="c"), split="test")
        assert library.counts() == {"train": 2, "validation": 0, "test": 1}
        assert len(library.train) == 2
        assert len(library.test) == 1
        assert library.validation == []

    def test_extend(self):
        library = TrackLibrary()
        library.extend([_track(uid="a"), _track(uid="b")], split="train")
        assert len(library) == 2

    def test_by_split_rejects_unknown(self):
        with pytest.raises(TrackLibraryError):
            TrackLibrary().by_split("nope")

    def test_summary_describes_composition(self):
        library = TrackLibrary()
        library.add(_track(uid="a"), split="train")
        assert "1 tracks" in library.summary()
        assert "1 train" in library.summary()
        assert TrackLibrary().summary() == "empty library"


class TestLibraryFromDirectory:
    def _write(self, tmp_path, names):
        for name in names:
            build_synthetic(name).save(tmp_path / f"{name}.json")

    def test_loads_every_track(self, tmp_path):
        self._write(tmp_path, ["straight", "oval", "s_curve"])
        library = TrackLibrary.from_directory(tmp_path)
        assert len(library) == 3
        assert all(e.source for e in library.entries)

    def test_explicit_splits_by_filename_stem(self, tmp_path):
        self._write(tmp_path, ["straight", "oval", "s_curve"])
        library = TrackLibrary.from_directory(
            tmp_path, explicit_splits={"oval": "test", "s_curve": "validation"}
        )
        assert [e.name for e in library.by_split("test")] == ["oval"]
        assert [e.name for e in library.by_split("validation")] == ["s_curve"]
        assert [e.name for e in library.by_split("train")] == ["straight"]

    def test_missing_directory(self, tmp_path):
        with pytest.raises(TrackLibraryError, match="not a directory"):
            TrackLibrary.from_directory(tmp_path / "nope")

    def test_empty_directory(self, tmp_path):
        with pytest.raises(TrackLibraryError, match="no track files"):
            TrackLibrary.from_directory(tmp_path)

    def test_pattern_filter(self, tmp_path):
        self._write(tmp_path, ["straight", "oval"])
        (tmp_path / "notes.txt").write_text("not a track")
        assert len(TrackLibrary.from_directory(tmp_path)) == 2

    def test_recursive(self, tmp_path):
        sub = tmp_path / "nested"
        sub.mkdir()
        build_synthetic("oval").save(sub / "oval.json")
        # Non-recursive finds nothing, and says so rather than returning an empty library.
        with pytest.raises(TrackLibraryError, match="no track files"):
            TrackLibrary.from_directory(tmp_path)
        assert len(TrackLibrary.from_directory(tmp_path, recursive=True)) == 1

    def test_report_and_manifest_roundtrip(self, tmp_path):
        self._write(tmp_path, ["straight", "oval", "s_curve", "figure_eight"])
        library = TrackLibrary.from_directory(tmp_path)
        report = library.report()
        assert report["num_tracks"] == 4
        assert sum(report["counts"].values()) == 4
        assert report["geometry_by_split"]
        # JSON-serialisable end to end.
        json.loads(json.dumps(report, default=str))

        path = library.save_manifest(tmp_path / "library.json")
        payload = json.loads(path.read_text())
        assert payload["schema_version"] == 1
        assert len(payload["tracks"]) == 4


# -- sampler ------------------------------------------------------------------------


class TestTrackSampler:
    def test_visits_every_track_within_one_block(self):
        tracks = [build_synthetic(n) for n in ("straight", "oval", "s_curve", "figure_eight")]
        sampler = TrackSampler(tracks, seed=0)
        drawn = [sampler.next().name for _ in range(len(tracks))]
        assert sorted(drawn) == sorted(t.name for t in tracks)

    def test_is_reproducible_for_a_seed(self):
        tracks = [build_synthetic(n) for n in ("straight", "oval", "s_curve")]
        a = [TrackSampler(tracks, seed=7).next().name for _ in range(6)]
        b = [TrackSampler(tracks, seed=7).next().name for _ in range(6)]
        assert a == b

    def test_different_seeds_differ(self):
        tracks = [build_synthetic(n) for n in ("straight", "oval", "s_curve", "figure_eight")]
        a = [TrackSampler(tracks, seed=1).next().name for _ in range(8)]
        b = [TrackSampler(tracks, seed=2).next().name for _ in range(8)]
        assert a != b

    def test_empty_is_rejected(self):
        with pytest.raises(TrackLibraryError):
            TrackSampler([])

    def test_state_roundtrip(self):
        tracks = [build_synthetic(n) for n in ("straight", "oval", "s_curve", "figure_eight")]
        sampler = TrackSampler(tracks, seed=3)
        sampler.next()
        state = sampler.state_dict()
        assert state["draws"] == 1

        restored = TrackSampler(tracks, seed=99)
        restored.load_state_dict(state)
        assert restored.draws == 1
        # The remaining order is restored, not re-drawn.
        assert sorted(restored.state_dict()["order"]) == sorted(state["order"])

    def test_peek_all_does_not_consume(self):
        tracks = [build_synthetic("oval")]
        sampler = TrackSampler(tracks, seed=0)
        assert len(sampler.peek_all()) == 1
        assert sampler.draws == 0

    def test_library_sampler_rejects_empty_split(self):
        library = TrackLibrary()
        library.add(_track(), split="train")
        with pytest.raises(TrackLibraryError, match="split 'test' is empty"):
            library.sampler("test")


# -- stats --------------------------------------------------------------------------


class TestTrackStats:
    def test_straight_has_no_corners(self):
        stats = compute_stats(straight(length=200.0))
        assert stats.corner_count == 0
        assert stats.straight_fraction == pytest.approx(1.0)
        assert not np.isfinite(stats.min_corner_radius)

    def test_oval_has_corners_and_turning(self):
        stats = compute_stats(oval())
        assert stats.corner_count >= 1
        assert stats.total_turning > 0
        assert stats.straight_fraction < 0.5

    def test_s_curve_counts_alternating_corners(self):
        stats = compute_stats(s_curve())
        assert stats.corner_count >= 2

    def test_closed_flag_and_span(self):
        """A closed circuit's chord is ~one sample spacing; an open track's is its length."""
        closed = compute_stats(oval())
        open_track = compute_stats(straight(length=100.0))
        assert closed.closed is True
        # Sampled with endpoint=False, so first and last are one sample apart along the
        # curve; the straight-line chord across that gap is therefore small.
        assert closed.span <= closed.sample_spacing + 1e-9
        assert closed.span < closed.length / 10
        assert open_track.closed is False
        assert open_track.span == pytest.approx(100.0, abs=1.0)

    def test_length_matches_the_track(self):
        track = s_curve()
        assert compute_stats(track).length == pytest.approx(track.length)

    def test_corridor_statistics(self):
        stats = compute_stats(straight(corridor=6.0))
        assert stats.corridor_mean == pytest.approx(12.0)
        assert stats.corridor_min == pytest.approx(12.0)

    def test_as_dict_is_json_safe(self):
        payload = compute_stats(straight()).as_dict()
        json.dumps(payload)
        # inf must not survive serialisation.
        assert payload["min_corner_radius"] is None

    def test_summary_is_readable(self):
        text = compute_stats(oval()).summary()
        assert "corners" in text
        assert "m," in text

    def test_stats_are_attached_to_library_entries(self):
        library = TrackLibrary()
        entry = library.add(oval())
        assert entry.stats is not None
        assert entry.stats.corner_count >= 1

    def test_stats_can_be_skipped(self):
        entry = TrackLibrary().add(oval(), compute_track_stats=False)
        assert entry.stats is None

    def test_figure_eight_geometry(self):
        stats = compute_stats(figure_eight())
        assert stats.total_turning > 0
        assert stats.num_points > 10
