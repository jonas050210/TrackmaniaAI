"""Episode replays and ghost comparison."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tmai.replay import (
    EpisodeReplay,
    ReplayComparison,
    ReplayRecorder,
    ReplayStore,
    compare_replays,
)
from tmai.tracks.synthetic import build_synthetic
from tmai.training.demos import Demonstration


def _replay(
    *,
    progress,
    times,
    finished=False,
    race_time=0.0,
    track="straight",
    source="training",
):
    progress = np.asarray(progress, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    n = len(progress)
    # The synthetic straight track runs along +z, so arc length == z here.
    positions = np.stack([np.zeros(n), np.zeros(n), progress], axis=1)
    return EpisodeReplay(
        episode=1,
        step=100,
        track=track,
        end_reason="finish" if finished else "time_limit",
        finished=finished,
        race_time=race_time or (float(times[-1]) if n else 0.0),
        positions=positions,
        speeds=np.full(n, 10.0),
        actions=np.zeros((n, 3)),
        rewards=np.zeros(n),
        progress=progress,
        race_times=times,
        source=source,
    )


class TestEpisodeReplay:
    def test_round_trip(self, tmp_path):
        replay = _replay(progress=[0, 5, 10], times=[0.0, 0.5, 1.0], finished=True, race_time=1.0)
        path = replay.save(tmp_path / "episode_000001.json")
        loaded = EpisodeReplay.load(path)
        assert loaded.episode == 1
        assert loaded.track == "straight"
        assert loaded.finished is True
        np.testing.assert_allclose(loaded.progress, [0, 5, 10])
        np.testing.assert_allclose(loaded.race_times, [0.0, 0.5, 1.0])
        assert loaded.num_samples == 3

    def test_schema_version_guard(self, tmp_path):
        replay = _replay(progress=[0, 5], times=[0.0, 0.5])
        path = replay.save(tmp_path / "r.json")
        data = json.loads(path.read_text())
        data["schema_version"] = 999
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="schema version"):
            EpisodeReplay.load(path)

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="positions has"):
            EpisodeReplay(
                episode=1,
                step=0,
                positions=np.zeros((3, 3)),
                speeds=np.zeros(2),
            )

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            EpisodeReplay.load(tmp_path / "nope.json")

    def test_from_demonstration(self):
        demo = Demonstration(
            observations=np.zeros((5, 4), dtype=np.float32),
            actions=np.tile([0.1, 0.9, 0.0], (5, 1)).astype(np.float32),
            positions=np.tile([1.0, 0.0, 2.0], (5, 1)),
            speeds=np.arange(5, dtype=np.float64),
            rewards=np.arange(5, dtype=np.float64),
            race_times=np.arange(5, dtype=np.float64) * 0.1,
            metadata={"track": "oval", "finished": True, "end_reason": "finish"},
        )
        ghost = EpisodeReplay.from_demonstration(demo)
        assert ghost.source == "human"
        assert ghost.track == "oval"
        assert ghost.finished is True
        assert ghost.num_samples == 5
        np.testing.assert_allclose(ghost.race_times, demo.race_times)


class TestReplayRecorder:
    def test_decimation(self):
        recorder = ReplayRecorder(decimation=3)
        for i in range(10):
            recorder.record(
                position=np.array([float(i), 0.0, 0.0]),
                speed=float(i),
                action=np.zeros(3),
                reward=0.0,
                progress=float(i),
                race_time=float(i) * 0.05,
            )
        assert recorder.steps == 10
        replay = recorder.build(episode=1, step=10)
        # Steps 3, 6, 9 are recorded (the first sample is kept so the trajectory starts
        # at the episode's beginning... actually step 1 is also kept).
        assert replay.num_samples == 4  # steps 1, 3, 6, 9
        np.testing.assert_allclose(replay.progress, [0.0, 2.0, 5.0, 8.0])

    def test_invalid_decimation(self):
        with pytest.raises(ValueError, match="decimation"):
            ReplayRecorder(decimation=0)

    def test_reset_clears(self):
        recorder = ReplayRecorder()
        recorder.record(
            position=np.zeros(3), speed=1.0, action=np.zeros(3),
            reward=0.0, progress=0.0, race_time=0.0,
        )
        recorder.reset()
        assert recorder.steps == 0
        assert recorder.build(episode=1, step=1).num_samples == 0


class TestReplayStore:
    def test_save_list_load(self, tmp_path):
        store = ReplayStore(tmp_path / "replays")
        for episode in (1, 2, 3):
            replay = _replay(progress=[0, 1], times=[0.0, 0.1], track=f"t{episode}")
            replay.episode = episode
            store.save(replay)
        rows = store.list()
        assert [r["episode"] for r in rows] == [1, 2, 3]
        assert rows[0]["track"] == "t1"
        latest = store.latest()
        assert latest.episode == 3

    def test_max_replays_prunes_oldest(self, tmp_path):
        store = ReplayStore(tmp_path / "replays")
        for episode in range(1, 6):
            replay = _replay(progress=[0, 1], times=[0.0, 0.1])
            replay.episode = episode
            store.save(replay, max_replays=2)
        rows = store.list()
        assert [r["episode"] for r in rows] == [4, 5]

    def test_list_missing_directory(self, tmp_path):
        assert ReplayStore(tmp_path / "nope").list() == []

    def test_load_by_name(self, tmp_path):
        store = ReplayStore(tmp_path / "replays")
        store.save(_replay(progress=[0, 1], times=[0.0, 0.1]))
        replay = store.load("episode_000001.json")
        assert replay.episode == 1


class TestCompareReplays:
    def test_station_and_segment_gaps(self):
        # Ghost: 10 m/s constant -> 1 s per 10 m. AI: same speed but 0.5 s slower per 10 m.
        stations = np.arange(0.0, 100.0, 10.0)
        ghost = _replay(progress=stations, times=stations / 10.0, finished=True, race_time=10.0)
        ai = _replay(
            progress=stations,
            times=stations / 10.0 + 0.05 * np.arange(len(stations)),
            finished=True,
            race_time=10.45,
        )
        comparison = compare_replays(ai, ghost, station_spacing=10.0)
        assert isinstance(comparison, ReplayComparison)
        np.testing.assert_allclose(comparison.stations, stations)
        # Station gap grows by 0.05 s per station.
        np.testing.assert_allclose(comparison.gaps, 0.05 * np.arange(len(stations)), atol=1e-9)
        # Segment gap is constant: the AI loses 0.05 s per 10 m.
        np.testing.assert_allclose(comparison.segment_gaps, 0.05, atol=1e-9)
        assert comparison.mean_gap == pytest.approx(0.05)
        assert comparison.max_gap == pytest.approx(0.05)
        assert comparison.race_time_delta == pytest.approx(0.45)

    def test_segment_gaps_are_start_station_invariant(self):
        """A partial lap (random start) must still report where time is lost."""
        # Ghost drives the whole 0..100 m at 10 m/s.
        ghost = _replay(
            progress=np.arange(0.0, 100.0, 10.0),
            times=np.arange(0.0, 100.0, 10.0) / 10.0,
        )
        # AI starts at station 40 (random start station) and drives 40..100 at 5 m/s.
        ai_progress = np.arange(40.0, 100.0, 10.0)
        ai = _replay(
            progress=ai_progress,
            times=(ai_progress - 40.0) / 5.0,  # 2 s per 10 m, clock starts at its own start
        )
        comparison = compare_replays(ai, ghost, station_spacing=10.0)
        # Overlap is 40..100 m; the AI takes 2 s per 10 m where the ghost takes 1 s.
        np.testing.assert_allclose(comparison.segment_gaps, 1.0, atol=1e-9)
        assert comparison.mean_gap == pytest.approx(1.0)

    def test_no_overlap_rejected(self):
        ai = _replay(progress=[0, 10], times=[0.0, 1.0])
        ghost = _replay(progress=[50, 60], times=[0.0, 1.0])
        with pytest.raises(ValueError, match="do not overlap"):
            compare_replays(ai, ghost)

    def test_missing_progress_requires_track(self):
        ai = _replay(progress=[0, 10], times=[0.0, 1.0])
        ghost = _replay(progress=[0, 10], times=[0.0, 1.0])
        ghost.progress = np.zeros(0)
        with pytest.raises(ValueError, match="arc-length progress"):
            compare_replays(ai, ghost)

    def test_progress_projected_from_positions_when_missing(self):
        track = build_synthetic("straight", length=200.0)
        # Replays whose progress array is empty but whose positions lie on the track.
        ghost_progress = np.arange(0.0, 100.0, 10.0)
        ghost_positions = np.stack(
            [track.point_at(s) for s in ghost_progress], axis=0
        )
        ghost = EpisodeReplay(
            episode=1,
            step=0,
            track="straight",
            positions=ghost_positions,
            speeds=np.full(len(ghost_progress), 10.0),
            actions=np.zeros((len(ghost_progress), 3)),
            rewards=np.zeros(len(ghost_progress)),
            progress=np.zeros(0),
            race_times=ghost_progress / 10.0,
        )
        ai = _replay(
            progress=ghost_progress,
            times=ghost_progress / 10.0 + 0.1,
            track="straight",
        )
        comparison = compare_replays(ai, ghost, track=track, station_spacing=10.0)
        np.testing.assert_allclose(comparison.stations, ghost_progress, atol=1e-6)
        assert comparison.mean_gap == pytest.approx(0.0, abs=0.05)

    def test_early_crash_shortens_the_comparison(self):
        """A crash early in the lap must not fake a comparison over the whole track."""
        ghost = _replay(
            progress=np.arange(0.0, 200.0, 10.0),
            times=np.arange(0.0, 200.0, 10.0) / 10.0,
        )
        ai = _replay(progress=[0.0, 10.0, 20.0], times=[0.0, 2.0, 4.0])  # crashed at 20 m
        comparison = compare_replays(ai, ghost, station_spacing=10.0)
        # Only the shared 0..20 m is compared; the ghost's remaining 180 m is not invented.
        assert comparison.stations[-1] == pytest.approx(20.0)
        assert len(comparison.stations) == 3
        # The AI took 2 s per 10 m where the ghost took 1 s.
        np.testing.assert_allclose(comparison.segment_gaps, 1.0, atol=1e-9)
        assert comparison.mean_gap == pytest.approx(1.0)

    def test_race_time_delta_only_when_both_finished(self):
        ghost = _replay(progress=[0, 10], times=[0.0, 1.0], finished=True, race_time=1.0)
        ai = _replay(progress=[0, 10], times=[0.0, 1.2], finished=False, race_time=1.2)
        comparison = compare_replays(ai, ghost, station_spacing=10.0)
        assert comparison.race_time_delta is None

    def test_summary_and_dict(self):
        ghost = _replay(progress=[0, 10], times=[0.0, 1.0], finished=True, race_time=1.0)
        ai = _replay(progress=[0, 10], times=[0.0, 1.5], finished=True, race_time=1.5)
        comparison = compare_replays(ai, ghost, station_spacing=10.0)
        text = comparison.summary()
        assert "stations" in text
        data = comparison.as_dict()
        assert data["segment_gaps"] == pytest.approx([0.5])
        assert data["race_time_delta"] == pytest.approx(0.5)
        assert json.dumps(data)  # serialisable
