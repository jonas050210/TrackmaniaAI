from __future__ import annotations

import numpy as np
import pytest

from tmai.replay import EpisodeReplay
from tmai.tracks.centerline import CenterlineTrack
from tmai.training.analysis import analyze_replays


def _straight_track() -> CenterlineTrack:
    return CenterlineTrack(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 100.0]],
        name="test-straight",
        corridor_half_width=5.0,
    )


def _replay(*, name: str = "test-straight", end_reason: str = "finished", lateral: float = 0.0):
    station = np.linspace(0.0, 100.0, 101)
    positions = np.column_stack(
        [np.full_like(station, lateral), np.zeros_like(station), station]
    )
    return EpisodeReplay(
        episode=1,
        step=100,
        track=name,
        end_reason=end_reason,
        finished=end_reason == "finished",
        race_time=5.0,
        positions=positions,
        speeds=np.full_like(station, 20.0),
        progress=station,
        race_times=station / 20.0,
    )


def test_analyze_replays_reports_sector_pace_and_failure_location():
    report = analyze_replays(
        [_replay(end_reason="crash", lateral=6.0)],
        _straight_track(),
        sector_count=4,
    )

    assert report["num_replays"] == 1
    assert report["num_samples"] == 101
    assert report["failure_reasons"] == {"crash": 1}
    assert report["failure_heatmap"]["events_with_location"] == 1
    assert sum(map(sum, report["failure_heatmap"]["counts"])) == 1
    assert report["sectors"][0]["mean_speed_mps"] == pytest.approx(20.0)
    assert report["sectors"][0]["mean_sector_time_s"] == pytest.approx(1.25)
    assert report["sectors"][3]["failure_count"] == 1
    assert report["sectors"][3]["mean_abs_lateral_m"] == pytest.approx(6.0)


def test_invalid_finish_is_counted_as_a_failure():
    report = analyze_replays(
        [_replay(end_reason="invalid_finish")],
        _straight_track(),
        sector_count=4,
    )

    assert report["failure_reasons"] == {"invalid_finish": 1}
    assert report["failure_heatmap"]["events_with_location"] == 1


def test_analyze_reprojects_human_ghosts_without_saved_progress():
    replay = _replay()
    replay.progress = np.zeros(0, dtype=np.float64)
    report = analyze_replays([replay], _straight_track(), sector_count=4)
    assert [s["speed_samples"] for s in report["sectors"]] == [25, 25, 25, 26]
    assert all(s["mean_sector_time_s"] == pytest.approx(1.25) for s in report["sectors"])


def test_analyze_ignores_replays_for_other_named_tracks():
    report = analyze_replays([_replay(name="other")], _straight_track(), sector_count=4)
    assert report["num_replays"] == 0
    assert report["num_samples"] == 0
    assert report["failure_reasons"] == {}


@pytest.mark.parametrize(
    ("sector_count", "lateral_bin_count"),
    [(0, 7), (4, 2), (4, 6)],
)
def test_analyze_rejects_invalid_grid_dimensions(sector_count: int, lateral_bin_count: int):
    with pytest.raises(ValueError):
        analyze_replays(
            [],
            _straight_track(),
            sector_count=sector_count,
            lateral_bin_count=lateral_bin_count,
        )
