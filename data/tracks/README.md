# Recorded track data

Centreline recordings produced by:

```bash
tmai record-track --out data/tracks/<map_name>.json --name <map_name>
```

Files here are **artefacts, not source** — `*.json` and `*.npz` are git-ignored (see
`.gitignore`). Keep them alongside the runs they were trained with; the run manifest records
the track name, UID and length, so a run can be matched back to its recording.

Each file is a `CenterlineTrack` document (schema version 1):

| Field | Meaning |
|---|---|
| `points` | ordered centreline samples, metres, shape `(N, 3)` |
| `corridor_half_width` | drivable half-width per point, metres |
| `length` | total centreline length, metres |
| `closed` | whether the last sample reconnects to the first |
| `metadata` | provenance: `source`, `raw_samples`, `min_spacing`, `smoothing_window`, `race_time_s` |

Inspect a recording before training on it:

```bash
tmai show-track --track data/tracks/<map_name>.json --out <map_name>.png
```

A sloppy recording lap gives a sloppy reference line, which biases the reward. Drive one clean
lap, or record several and keep the best.
