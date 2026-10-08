# Track representation and generalisation

The long-term goal is an agent that drives maps it has never seen. This document describes the
representation that makes that possible, the mechanisms that prevent memorisation, and how
generalisation is measured.

---

## 1. Track representation

The track is a **centreline**: an ordered 3-D polyline with arc-length parameterisation and a
drivable corridor (`tmai/tracks/centerline.py`).

```
progress s  ──▶  ●────●────●────●────●────●────▶   centreline points
                 │  ← lateral_offset (signed, + = right of travel)
                 ▼
                car
```

From it the environment derives everything the policy needs:

| Quantity | Method | Used for |
|---|---|---|
| Arc-length progress | `project()` | The primary reward |
| Signed lateral offset | `project()` | Off-track detection, corridor reward |
| Heading error | `heading_error()` | Alignment reward |
| Upcoming curvature | `sample_lookahead()` | Knowing a corner is coming |
| Distance to each edge | `edge_distances()` | "How much room do I have" |
| Corridor width | `corridor_half_width_at()` | What counts as off-track |

**Why a centreline rather than map geometry:** it is the minimum that makes track-relative
observations possible, and it is obtainable for *any* map by driving it once
(`tmai record-track`) — no parser, no format reverse-engineering. Block-level geometry would
improve the corridor and add obstacle awareness; it is a documented next step, not a
prerequisite.

**Two conventions that are easy to get wrong and are covered by tests:**

* The world frame is **left-handed with +y up** (+x east, +z south). "Right of travel" is
  `cross(up, tangent)`, not the right-handed `cross(tangent, up)` most 3-D maths assumes.
  Getting this backwards flips every steering correction the policy learns.
* `point_at(s)` must scale by the segment length, because `_tangents` are *unit* vectors.
  Multiplying by the bare fraction is correct only when every segment is exactly one metre —
  true of the default synthetic tracks and of nothing recorded from a real map. This was a
  real bug found by a test.

---

## 2. The library and its splits

`tmai/tracks/library.py` holds many tracks and partitions them into `train`, `validation` and
`test`.

**Splits are a function of track identity, not of insertion order.** A track's identity is its
map UID, or — when there is no UID — a SHA-256 hash of its geometry. The split is derived from
a stable hash of that identity, so:

* The assignment is identical across runs, machines and Python versions (it does *not* use
  `hash()`, which is salted per process).
* Adding a map to the library never moves another map between splits, so results stay
  comparable over time.
* Two recordings of the same line hash identically and are therefore *the same track*.

That last property is the point. The easiest way to publish an inflated generalisation number
is to record the same map twice, put one copy in `train` and the other in `test`, and report
the result. The library makes that structurally impossible: a track identity can only ever
belong to one split, and attempting otherwise raises `TrackLibraryError` rather than quietly
proceeding.

`track.explicit_splits` pins particular maps to a split by filename stem, which is how an
operator reserves specific maps for final held-out testing.

---

## 3. Mechanisms against memorisation

| Mechanism | What it defends against | Where |
|---|---|---|
| **Track sampling per episode** | Keying on track identity | `MultiTrackEnv` |
| **Random start station** | Treating the start line as a memorised cue | `MultiTrackEnv` |
| **Random lateral offset** | Never learning to recover from off the racing line | `MultiTrackEnv` |
| **No absolute coordinates** | Learning world positions instead of driving | `ObservationEncoder` |
| **No track identity in the observation** | Conditioning on which map it is | `ObservationSpec` |
| **Held-out evaluation** | Mistaking memorisation for skill | `Trainer` |

### The invariant that is actually tested

Translation invariance is the strongest available check that no absolute coordinate leaks in.
A test builds a straight track, translates it by `[750, 0, -420]`, drives both identically, and
asserts the observation vectors are equal to within 1e-6. They are equal to within **0.0**.

A second test asserts no observation feature is *named* like a world coordinate, so a future
feature addition cannot quietly reintroduce one.

### Curriculum over the training split only

With `curriculum.enabled`, early training reveals the easiest tracks first (ordered by mean
curvature plus corner density) and may shorten episodes (`episode_length_fraction`), which
concentrates early learning where the signal is cleanest. Two properties keep it honest:

* it applies **only to the training environment** — held-out evaluation always runs the full
  suite, so a curriculum can never flatter the number it is measured by;
* the active stage is logged (`curriculum_stage` events + a `curriculum/stage` metric), so a
  run's manifest says exactly what the policy was exposed to at every step.

### Sampling without replacement blocks

`TrackSampler` draws shuffled *blocks* rather than i.i.d. With a four-map library, i.i.d.
sampling can leave a map untouched for hundreds of episodes and the policy quietly forgets it.
Cycling through a permutation guarantees every training track is seen every `n` episodes —
which matters more here than perfect uniformity.

### The honest limitation on start randomisation

Random start stations require placing the car at an arbitrary point on the track. **The real
game cannot do this** — Trackmania respawns to the last checkpoint. So
`DriverCapabilities.supports_start_repositioning` is `True` for the simulated driver and
`False` for the real one, and:

* `TMInterfaceDriver.reposition()` raises `UnsupportedFeatureError` rather than silently
  returning a frame from the start line.
* `TrackmaniaEnv.reset(options={"start_station": ...})` raises if the driver cannot do it.
* `RunConfig.validate()` rejects a real-game config that asks for start randomisation.
* `tmai/configs/default.yaml` ships with it **disabled**.

Silently ignoring the request would produce a run reporting randomised starts that never
happened. Recording per-map `CheckpointData` states would make this genuinely available; it is
in the roadmap.

---

## 4. Observation

20 features (`OBSERVATION_VERSION = 1`), all ego-relative or track-relative:

```
speed_forward, speed_sideward          ego velocity
rpm, gear, is_sliding                  drivetrain state
yaw_rate                               ego rotation rate
lateral_offset, heading_error          position within the corridor
progress_rate                          how fast progress is being made
curvature_at_{5,10,20,40,80}m          what the track does next
edge_distance_left, edge_distance_right  how much room is left
checkpoint_progress                    the game's own validity counter
last_steer, last_throttle, last_brake  the previous action
```

Feature groups are individually switchable through `ObservationSpec`, which is how ablations
are run without touching encoder code. The layout is explicit and ordered so it can be logged,
diffed between runs, and validated against a checkpoint.

**Normalisation** (`tmai/env/normalization.py`) is applied *at use time*, inside a
`NormalizingLearner` decorator, not at collection time. The replay buffer keeps raw
observations. If normalisation were baked into stored data, every improvement in the running
statistics would silently invalidate the millions of transitions already in the buffer.
Statistics use Welford's algorithm (no catastrophic cancellation over a multi-day run) and are
part of the checkpoint, so a resumed run does not restart from unit statistics.

Because the decorator implements the same `Learner` protocol, adding it changed nothing in the
trainer — which is the concrete payoff of keeping the algorithm behind an interface.

---

## 5. Measuring generalisation

Reports (`tmai/training/evaluate.py`) are per-track and per-split, never a single blended
number:

```
track                  split       fin%  prog%    std  worst%  crash%  best lap
---------------------------------------------------------------------------
oval                   train         0%   52.3    4.1    48.2      0%        --
straight               train         0%   61.0    0.0    61.0      0%        --
s_curve                validation    0%    2.4    1.2     1.2    100%        --
figure_eight           validation    0%    0.8    0.0     0.8    100%        --
---------------------------------------------------------------------------
TOTAL                             0%   29.1    1.3             50%
```

Metrics that matter, all in the JSON output and the metrics stream:

| Metric | Meaning |
|---|---|
| `mean_progress_fraction` | Mean of *per-track* means, so every track counts equally |
| `crash_rate` | Fraction ending off-track, stalled or airborne — "drove badly" |
| `consistency_std` | Mean within-track spread. High means unreliable, not slow |
| `worst_progress_fraction` | The bad case, which means hide behind an average |
| `generalization_gap` | Train progress − held-out progress. **Positive means overfitting** |
| `best_race_time` | Valid finishes only — an `invalid_finish` never counts |

`generalization_gap` is `None` when either side is missing, which is the honest answer rather
than `0`.

Held-out evaluation runs on a **separate environment** over the validation split, so the
training episode and the training driver are untouched, and it is wrapped so a failure there
can never kill a multi-day run.

---

## 6. Track data pipeline

```
real map ──drive once──▶ tmai record-track ──▶ data/tracks/<map>.json
                                                    │
                            tmai list-tracks ◀──────┤  (geometry fingerprint, split)
                            tmai show-track  ◀──────┤  (PNG)
                            tmai export-obj  ◀──────┘  (Wavefront .obj)
```

Each file is a `CenterlineTrack` document (schema version 1) with points, per-point corridor
width, length, closed flag and provenance metadata. `tmai validate-config` and
`TrackLibrary.from_directory` both fail loudly on a malformed or empty directory rather than
proceeding with a partial library.

`tmai list-tracks` prints a geometry fingerprint per map — length, corner count, tightest
corner radius, straight fraction, corridor width — plus **geometry coverage by split**. That
last table is the cheap check on whether the held-out maps are even comparable to the training
maps: a suite that trains on ovals and tests on a technical track is measuring a distribution
shift, not generalisation.

---

## 7. `.Map.Gbx` — deliberately not implemented

Block-level map geometry would improve the corridor model and enable obstacle awareness. It is
**not implemented**, and that is a decision rather than an omission.

The environment this was developed in has no Windows, no Trackmania and no sample `.Map.Gbx`.
A parser written here could not have been tested against the thing it parses, so shipping one
would have been a guess presented as a feature.

What *does* exist (`tmai/tracks/gbx.py`), all fully unit-tested:

* `MapBlock` / `MapGeometry` — the data model a parser must produce.
* `blocks_to_centerline()` — the geometry that turns blocks into a drivable line. Tested,
  including that block *insertion order* does not affect the result.
* `GbxSource.load()` — the isolated boundary. It raises `GbxUnavailableError` (a subclass of
  `NotImplementedError`, so callers can distinguish "unfinished" from "corrupt file") with a
  precise account of what is needed.

The third-party option (`pygbx`, PyPI) was evaluated and rejected for this phase: it targets
TMNF/TMUF with only partial TM2 support, requires the `python-lzo` C extension, and is GPL-3.
Revisit it with a real map file in hand. `GbxSource.requirements()` lists exactly what is
needed; see [`LIMITATIONS.md`](LIMITATIONS.md).

---

## 8. What generalisation still needs

1. **Real maps.** Everything here is proven with synthetic centrelines. A policy that
   generalises across four generated curves may still fail across four real Trackmania maps.
2. **Block geometry**, for a corridor that reflects the actual road rather than a constant
   width.
3. **More maps than four.** Split statistics are only meaningful with enough maps per split.
4. **Held-out reporting as the headline metric.** Training-map performance should never be the
   number a run is judged by.
