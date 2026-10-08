"""Spatial analysis of saved episode replays.

This module turns replay trajectories into sector-level pace summaries and a failure heatmap.
It is deliberately descriptive: low speed or a large lateral offset is a diagnosis prompt, not
proof of a bad policy, and a failure can only be attributed to a location when the replay has
valid positions and a matching track geometry.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from tmai.replay import EpisodeReplay
from tmai.tracks.centerline import CenterlineTrack, TrackProjection

ANALYSIS_SCHEMA_VERSION = 1
FAILURE_REASONS = frozenset(
    {
        "crash",
        "off_track",
        "out_of_bounds",
        "fell_off",
        "no_ground_contact",
        "wrong_way",
        "stalled",
        "game_error",
        "invalid_finish",
    }
)


def _trajectory_progress(replay: EpisodeReplay, track: CenterlineTrack) -> np.ndarray:
    """Return a continuous arc-length coordinate for every replay position."""
    n = len(replay.positions)
    progress = np.asarray(replay.progress, dtype=np.float64).reshape(-1)
    if (
        progress.size == n
        and np.all(np.isfinite(progress))
        and (n < 2 or float(np.ptp(progress)) > 1e-6)
    ):
        # Demonstration ghosts created before progress was stored contain an all-zero array.
        # Reproject those positions instead of treating the whole lap as station zero.
        return progress

    result = np.empty(n, dtype=np.float64)
    hint: float | None = None
    for i, position in enumerate(replay.positions):
        projection = track.project(
            position,
            hint_s=hint,
            search_window=max(track.length, 1.0),
        )
        result[i] = projection.progress
        hint = projection.progress
    return result


def _crossing_times(
    progress: np.ndarray,
    race_times: np.ndarray,
    *,
    track_length: float,
    sector_count: int,
    closed: bool,
) -> list[list[float]]:
    """Interpolate elapsed time between consecutive fixed sector boundaries."""
    sector_times: list[list[float]] = [[] for _ in range(sector_count)]
    if len(progress) < 2 or race_times.shape != progress.shape or track_length <= 0:
        return sector_times
    if not np.all(np.isfinite(progress)) or not np.all(np.isfinite(race_times)):
        return sector_times

    # Arc-length projection can jitter slightly backwards. A monotone envelope avoids
    # manufacturing negative sector times while preserving the first forward crossing.
    p = np.maximum.accumulate(progress)
    t = np.maximum.accumulate(race_times)
    sector_length = track_length / sector_count
    start_boundary_index = int(round(float(p[0]) / sector_length))
    starts_on_boundary = abs(float(p[0]) - start_boundary_index * sector_length) <= 1e-8
    last_boundary_index: int | None = start_boundary_index if starts_on_boundary else None
    last_boundary_time: float | None = float(t[0]) if starts_on_boundary else None
    next_boundary_index = (
        start_boundary_index + 1
        if starts_on_boundary
        else int(np.floor(p[0] / sector_length)) + 1
    )

    for i in range(1, len(p)):
        p0, p1 = float(p[i - 1]), float(p[i])
        t0, t1 = float(t[i - 1]), float(t[i])
        if p1 <= p0:
            continue
        while next_boundary_index * sector_length <= p1 + 1e-9:
            boundary = next_boundary_index * sector_length
            fraction = float(np.clip((boundary - p0) / (p1 - p0), 0.0, 1.0))
            boundary_time = t0 + fraction * (t1 - t0)
            if (
                last_boundary_index is not None
                and last_boundary_time is not None
                and next_boundary_index == last_boundary_index + 1
            ):
                sector_index = (last_boundary_index % sector_count) if closed else last_boundary_index
                if 0 <= sector_index < sector_count:
                    duration = boundary_time - last_boundary_time
                    if np.isfinite(duration) and duration >= 0:
                        sector_times[sector_index].append(float(duration))
            last_boundary_index = next_boundary_index
            last_boundary_time = boundary_time
            next_boundary_index += 1
    return sector_times


def analyze_replays(
    replays: Sequence[EpisodeReplay],
    track: CenterlineTrack,
    *,
    sector_count: int = 20,
    lateral_bin_count: int = 7,
) -> dict[str, Any]:
    """Summarize replay pace, track-relative driving and failure locations.

    Args:
        replays: episode replays from one map. Replays for other map names are ignored when
            both the replay and track have a non-empty name.
        track: recorded centreline/corridor used to project positions.
        sector_count: number of equal-distance sectors around the track.
        lateral_bin_count: odd number of bins across the normalized corridor width. The center
            bin represents the centreline; outer bins indicate excursions beyond the corridor.

    Returns:
        JSON-serializable per-sector metrics, failure reasons and a station-by-lateral failure
        count grid. Empty or incomplete trajectories are counted as episodes but do not invent
        measurements.
    """
    if sector_count < 1:
        raise ValueError(f"sector_count must be >= 1, got {sector_count}")
    if lateral_bin_count < 3 or lateral_bin_count % 2 == 0:
        raise ValueError(f"lateral_bin_count must be an odd integer >= 3, got {lateral_bin_count}")
    if not np.isfinite(track.length) or track.length <= 0:
        raise ValueError("track length must be finite and positive")

    selected = [
        replay
        for replay in replays
        if not replay.track or not track.name or replay.track == track.name
    ]
    speeds: list[list[float]] = [[] for _ in range(sector_count)]
    offsets: list[list[float]] = [[] for _ in range(sector_count)]
    offset_ratios: list[list[float]] = [[] for _ in range(sector_count)]
    timing_samples: list[list[float]] = [[] for _ in range(sector_count)]
    sector_episode_ids: list[set[int]] = [set() for _ in range(sector_count)]
    failure_grid = np.zeros((sector_count, lateral_bin_count), dtype=np.int64)
    failure_reasons: Counter[str] = Counter()
    failure_locations = 0
    observed_samples = 0

    for replay_index, replay in enumerate(selected):
        positions = np.asarray(replay.positions, dtype=np.float64).reshape(-1, 3)
        n = len(positions)
        if n == 0:
            if replay.end_reason in FAILURE_REASONS:
                failure_reasons[replay.end_reason] += 1
            continue

        progress = _trajectory_progress(replay, track)
        if progress.shape != (n,):
            continue
        replay_speeds = np.asarray(replay.speeds, dtype=np.float64).reshape(-1)
        replay_times = np.asarray(replay.race_times, dtype=np.float64).reshape(-1)
        speeds_present = replay_speeds.shape == (n,)
        times_present = replay_times.shape == (n,)
        observed_samples += n

        projections: list[TrackProjection | None] = []
        for index, (position, station) in enumerate(zip(positions, progress, strict=True)):
            try:
                projection = track.project(
                    position,
                    hint_s=float(station),
                    search_window=max(track.length, 1.0),
                )
            except (ValueError, IndexError):
                projections.append(None)
                continue
            projections.append(projection)
            local_station = float(station % track.length) if track.closed else float(
                np.clip(station, 0.0, track.length)
            )
            sector = min(sector_count - 1, int(local_station / track.length * sector_count))
            if speeds_present and np.isfinite(replay_speeds[index]):
                speeds[sector].append(float(replay_speeds[index]))
                sector_episode_ids[sector].add(replay_index)
            if np.isfinite(projection.lateral_offset):
                width = max(track.corridor_half_width_at(local_station), 1e-6)
                offsets[sector].append(abs(float(projection.lateral_offset)))
                offset_ratios[sector].append(float(projection.lateral_offset) / width)
                sector_episode_ids[sector].add(replay_index)

        if times_present:
            clean_times = replay_times.copy()
            if np.all(np.isfinite(clean_times)):
                # Only use an episode-relative time axis; timestamps must increase enough to
                # be meaningful. This also avoids relying on host wall-clock time.
                clean_times -= clean_times[0]
                for sector, durations in enumerate(
                    _crossing_times(
                        progress,
                        clean_times,
                        track_length=track.length,
                        sector_count=sector_count,
                        closed=track.closed,
                    )
                ):
                    timing_samples[sector].extend(durations)

        if replay.end_reason in FAILURE_REASONS:
            failure_reasons[replay.end_reason] += 1
            last_projection = projections[-1]
            if last_projection is not None:
                last_station = float(
                    last_projection.progress % track.length
                    if track.closed
                    else np.clip(last_projection.progress, 0.0, track.length)
                )
                sector = min(sector_count - 1, int(last_station / track.length * sector_count))
                width = max(track.corridor_half_width_at(last_station), 1e-6)
                normalized_lateral = float(last_projection.lateral_offset) / width
                lateral_edges = np.linspace(-2.0, 2.0, lateral_bin_count + 1)
                lateral_bin = int(np.clip(
                    np.searchsorted(lateral_edges, normalized_lateral, side="right") - 1,
                    0,
                    lateral_bin_count - 1,
                ))
                failure_grid[sector, lateral_bin] += 1
                failure_locations += 1

    sectors: list[dict[str, Any]] = []
    sector_length = track.length / sector_count
    for index in range(sector_count):
        sector_speed = speeds[index]
        sector_offsets = offsets[index]
        sector_ratios = offset_ratios[index]
        sector_times = timing_samples[index]
        sectors.append(
            {
                "index": index,
                "start_m": round(index * sector_length, 3),
                "end_m": round((index + 1) * sector_length, 3),
                "episodes_with_samples": len(sector_episode_ids[index]),
                "speed_samples": len(sector_speed),
                "mean_speed_mps": round(float(np.mean(sector_speed)), 3) if sector_speed else None,
                "mean_abs_lateral_m": round(float(np.mean(sector_offsets)), 3) if sector_offsets else None,
                "mean_lateral_fraction_of_half_width": (
                    round(float(np.mean(sector_ratios)), 3) if sector_ratios else None
                ),
                "sector_time_samples": len(sector_times),
                "mean_sector_time_s": round(float(np.mean(sector_times)), 3) if sector_times else None,
                "median_sector_time_s": round(float(np.median(sector_times)), 3) if sector_times else None,
                "failure_count": int(failure_grid[index].sum()),
            }
        )

    slowest = sorted(
        (sector for sector in sectors if sector["mean_sector_time_s"] is not None),
        key=lambda sector: float(sector["mean_sector_time_s"]),
        reverse=True,
    )
    lateral_edges = np.linspace(-2.0, 2.0, lateral_bin_count + 1)
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "track": track.name,
        "track_uid": track.uid,
        "track_length_m": round(track.length, 3),
        "closed": track.closed,
        "num_replays": len(selected),
        "num_samples": observed_samples,
        "sector_count": sector_count,
        "sectors": sectors,
        "slowest_sectors": [sector["index"] for sector in slowest[: min(5, len(slowest))]],
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "failure_heatmap": {
            "station_edges_m": [round(i * sector_length, 3) for i in range(sector_count + 1)],
            "normalized_lateral_edges": [round(float(v), 3) for v in lateral_edges],
            "lateral_labels": [
                "far_left",
                "left",
                "left_center",
                "center",
                "right_center",
                "right",
                "far_right",
            ] if lateral_bin_count == 7 else [f"bin_{i}" for i in range(lateral_bin_count)],
            "counts": failure_grid.tolist(),
            "events_with_location": failure_locations,
        },
    }


__all__ = ["ANALYSIS_SCHEMA_VERSION", "FAILURE_REASONS", "analyze_replays"]
