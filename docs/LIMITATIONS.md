# Limitations and verification status

This document is the honest account. Every claim about the real Trackmania integration is
listed with its verification status and what it would take to verify it.

**The single most important fact: no code in this repository has ever talked to a running
Trackmania instance.** This project was built in a Linux container with no Windows, no game
and no GPU. Everything below is written to make that boundary impossible to miss.

Three categories are used throughout:

* **Verified** — executed and checked in this repository.
* **Offline-tested** — the logic is exercised by tests against realistic stand-ins (real
  `tminterface` structs, a scripted tick source), but not against the live game.
* **Requires the real PC** — cannot be checked anywhere but a Windows host with Trackmania.

---

## Verification status

### Game integration

| Claim | Status | How it was verified / what is missing |
|---|---|---|
| `tminterface` 1.0.2 API surface used by the driver | **Verified** | Read from the published source (`interface.py`, `structs.py`, `constants.py`, `client.py`). |
| Field mapping `SimStateData → VehicleState/RaceState` | **Offline-tested against real structs** | `tests/test_telemetry.py::TestAgainstRealStructs` decodes genuine `SimStateData` objects from byte buffers and checks the mapping. |
| Analog input range `[-65536, 65536]` | **Offline-tested (documented)** | Taken from `TMInterface.set_input_state`'s docstring; asserted in `test_analog_full_scale_matches_tminterface_documentation`. Not exercised against a real car. |
| Driver control logic (reset sequencing, decimation, speed ratio, connection loss) | **Offline-tested** | Tests in `tests/test_tminterface_driver.py` use a scripted tick source honouring the real `TickSource` contract. |
| `tmai play` CLI, checkpoint/heuristic selection, bounded rollout, replay recording and cleanup | **Offline-tested** | `tests/test_cli_new.py::TestPlayCli` runs both controller paths through the explicitly opted-in simulated driver. It does not verify a live TMInterface session or real-map steering. |
| TMInterface cannot connect on Linux | **Verified** | `mmap.mmap(-1, size, tagname=...)` raises `TypeError` on Linux; asserted in `test_mmap_tagname_is_windows_only`. |
| The game actually steers when `set_input_state` is called | **Requires the real PC** | — |
| `get_simulation_state()` returns sane values during a live race | **Requires the real PC** | — |
| Trackmania's forward axis convention | **Requires the real PC — must be calibrated** | `tmai doctor --calibrate` measures it. `driver.forward_axis` and `driver.forward_sign` now apply the measured result; the default is only a starting value. |
| Trackmania's position/velocity units are metres | **Requires the real PC — must be calibrated** | `tmai doctor --calibrate` measures `position_scale`. |
| Physics tick period is 100 Hz | **Requires the real PC** | The `tick_period` check measures it. |
| `iface.respawn()` restarts an episode usefully | **Requires the real PC** | `ResetStrategy.COMMAND` exists as the alternative if respawn is insufficient. |
| Finish/checkpoint detection matches the game's own race logic | **Requires the real PC** | Wired to `player_info.race_finished` and `on_checkpoint_count_changed`; needs an in-game check. |
| Game-speed multiplier gives a usable training speedup | **Requires the real PC** | Upstream documents the mechanism and warns that factors above ~100 may drop inputs. |
| The car can be placed at an arbitrary track station | **Verified as NOT possible** | `TMInterfaceDriver.reposition()` raises `UnsupportedFeatureError`; the capability reports `False`; `RunConfig.validate()` rejects a real-game config that asks for it. |
| `.Map.Gbx` parsing | **Deliberately not implemented** | `GbxSource.load()` raises `GbxUnavailableError`. See below. |

### RL stack

| Claim | Status | How it was verified |
|---|---|---|
| RL environment, reward, termination | **Verified** | Includes `gymnasium.utils.env_checker.check_env`. |
| SAC learner | **Verified** | Includes a Bellman fixed-point check and a numerical log-prob check. |
| Observation normalisation | **Verified** | Statistics, warmup, clipping, NaN resistance, checkpoint round-trip. |
| Training loop, checkpointing, resume | **Verified** | Full runs executed and their artefacts inspected. |
| Track geometry and centreline recording | **Verified offline** | Unit tests cover projection, closed-track wrapping, corridor interpolation, recording, and export. |
| Track library, splits, leakage prevention | **Verified offline** | Tests cover split determinism and refusal to place identical geometry in different splits. |
| Multi-track environment | **Verified offline** | Track switching, start randomisation, driver caching, episode context. |
| Translation invariance of the observation | **Verified offline** | A translated track produces a bit-identical observation (diff exactly 0.0). |
| Telemetry calibration detects wrong conventions | **Verified offline** | Deliberately fed wrong conventions and required failures. |
| Evaluation reports (per-track, per-split, gap) | **Verified offline** | Tests and CLI workflows use simulated run artefacts; no real-game scores exist. |
| Status/compare CLI and dashboard API | **Verified offline** | CLI tests and API tests cover snapshots, summaries, and split comparisons. |
| Simplified track visualisation | **Verified offline** | Headless PNG and `.obj` exports, including the closed-track seam. |
| Driving rules (crash/OOB/fall/wrong-way → immediate respawn + penalty) | **Verified offline** | Tests cover impact vs scrape vs parking, reward-exploit checks, and failure-reason recording. |
| Temporal observations (frame stacking) | **Verified offline** | Tests cover stack ordering/reset, environment dimensions, translation invariance, and checkpoint dimension guards. |
| Curriculum learning | **Verified offline** | Tests cover validation, difficulty order, stage resolution, reveal + episode caps, and trainer integration. |
| Human demonstrations + behaviour cloning | **Verified against the simulated driver** | The data path and BC fit are tested. **The recording path has never run against the real game** — it needs a human at the wheel of live Trackmania. |
| Replays, ghost comparison & replay analysis | **Verified offline** | Tests cover round-trips, decimation, store pruning, station-invariant segment gaps, overlap clamping, sector summaries and spatial failure bins; no real-game replays exist yet. |
| Model registry | **Verified offline** | Tests cover register/list/get/delete/tag, name validation, corrupt checkpoints, and wrapped learners. |
| Benchmarking | **Verified offline** | Tests cover ranking, run-directory resolution, serialization, and empty-split rejection. |
| Resource monitoring | **Verified offline** | Tests cover graceful `None` values when a counter is unavailable. |
| GUI backend (HTTP + WebSocket + jobs) | **Verified offline** | FastAPI tests include a real training subprocess started through `POST /api/train` and a WebSocket tick; no game is involved. |
| GUI frontend | **Built and served; not user-tested in a browser** | `npm run build` (TypeScript + Vite) passes; `tmai serve` returned the static app and API health/system endpoints over HTTP. No human click-through or Trackmania connection was tested. |
| **Total** | | **869 passed, 9 skipped; 91% overall coverage; Ruff and Mypy clean** |

Measured with `pytest --cov=tmai --cov-report=term-missing` (869 passed, 9 skipped; 91% overall; 7,995 statements):

| File | Coverage | Remaining gap |
|---|---:|---|
| `tmai/game/tminterface/session.py` | **74%** | Windows registration/callback transport and live-game synchronization are not exercised here. |
| `tmai/cli.py` | **80%** | Interactive commands and some failure paths remain untested; commands requiring a real game are intentionally not simulated as live validation. |
| `tmai/server/app.py` | **82%** | Optional job/error routes and host integrations remain partially uncovered. |
| `tmai/monitoring.py` | **77%** | Some platform-specific resource counters are unavailable in this container. |

The transport gap is an explicit real-game verification limitation, not a reason to fake the
external process.

### Things that have never happened

| Claim | Status |
|---|---|
| The reward function produces good driving in the real game | **Requires the real PC** — weights are reasoned defaults, never tuned against real physics. |
| A policy trained here can drive a real map | **Not done** — no real training has happened. |
| The agent generalises to unseen *real* maps | **Not done** — generalisation is proven only across four synthetic centrelines. |
| A human has driven a demonstration into `tmai record-demo` | **Not done** — the recording path is tested against the simulated driver only; it has never captured a real human's inputs from a live game. |
| The GUI has been used by a human in a real browser | **Not done** — the frontend is built, type-checked and served, and its API is exercised by tests and over HTTP; no human has clicked through it yet. |

---

## Bugs the test suite found

These were real defects in production code, found by tests and fixed. They are recorded here
because each one would have been expensive to discover during a long training run, and because
several are the kind that produce *plausible-looking but wrong* results rather than errors.

| Bug | Why it mattered |
|---|---|
| `CenterlineTrack.project()` computed arc length from a windowed slice using a global segment index | Wrong progress or `IndexError` on **every** hinted projection — i.e. every env step after reset. |
| Lateral sign used the right-handed `cross(tangent, up)` | Trackmania's frame is left-handed. This flipped every steering correction the policy would learn. |
| `point_at()` multiplied a **unit** tangent by the bare segment fraction | Returned a point `frac` metres along the segment instead of `frac` of the way down it. Correct only when every segment is exactly 1 m — true of the default synthetic tracks and of nothing recorded from a real map. |
| Corridor half-width indexed per sample instead of interpolated by arc length | A car near a segment boundary was off-track for the reward but on-track for termination. |
| Progress clamp was a fixed 6 m/step | At a 0.1 s control period this clamped legitimate driving above **60 m/s** — punishing exactly the speed the reward exists to encourage. Replaced with a speed-derived threshold. |
| A stationary car scored exactly 0.0 | Every early driving attempt scores negative, so "do nothing" was the optimal policy. Learning could never start. Fixed with `idle_penalty`. |
| Penalties charged per step rather than per second | The same lap scored differently at different control rates, and the effective discount horizon silently changed. |
| `TelemetryCalibrator.analyse()` never populated its recommendation fields; `calibrate_driver` faked them by parsing a formatted string | Never derive machine-readable values from human-readable output. |
| `TrackView._build_corridor()` returned `(right, left)` while index 0 is exposed as `left_edge` | The rendered and exported corridor was mirrored. |
| `env._info()` contained wall-clock timing | Violated gymnasium's determinism requirement. |
| `tmai eval --checkpoint <run>` fell back to `default.yaml` | Risked loading weights into a differently-shaped observation. Fixed with config discovery plus a guard that rejects a mismatched checkpoint. |
| `tmai eval` evaluated only the first train track | Silently reported a single map for a multi-track run, discarding the per-track comparison. Found by an end-to-end CLI run; fixed to iterate every split with `--split`. |
| **`train.seed` did not reproduce a run** | Two entropy sources escaped the seed: gymnasium spaces own a private generator that `np.random.seed` cannot reach (so warm-up actions varied), and `ReplayBuffer` owns a `default_rng` seeded from `replay.seed`, which every shipped config leaves `null` — i.e. fresh OS entropy. Found by measuring, not by reading code; fixed with `seed_everything()` and a derived buffer stream. |
| Checkpoint docstring claimed resume was "exact" | It is not: the buffer *contents* are not stored, so a resumed run refills and diverges. The docstring and `TRAINING.md` now state the trade-off explicitly, and a test pins it. |
| `tmai compare` always printed `--` in the `gap` column | It read `generalization_gap` off the *training* report, but training and held-out runs are logged as separate reports that each carry only their own split — so the value was always `None`. The one column that shows overfitting was silently blank; the gap is now computed across both reports. |
| **`tmai train --resume runs/<run>` did not work** | The documented command failed with "no track configured". Config auto-discovery was wired only to `--checkpoint`, so `--resume` silently fell back to `default.yaml` even though the run's own `config.yaml` was in the directory. Found by running the documented command; discovery now covers both flags. |
| `tmai doctor --calibrate` produced no output against the simulated driver | Skipping is correct — calibration measures real telemetry against the game's clock — but it skipped *silently* with exit 0, which reads exactly like a calibration that found nothing wrong. It now says why it was skipped. |
| Nested dicts were stringified in `events.jsonl` | Evaluation reports were written as the *string* `"{'a': 1}"`, breaking machine readability. |
| Episode events did not record which track they ran on | Per-track episode analysis was impossible for a multi-track run; a bad track looked like a bad policy. |
| `Any` used but not imported in `tmai/cli.py` | A `NameError` waiting on the `compare` code path; caught by `ruff`. |
| `evaluate_tracks` assumed a `MultiTrackEnv` | A single-track split builds a plain `TrackmaniaEnv` (no `select_track`), so evaluating/benchmarking a one-track library crashed with `AttributeError` instead of reporting. Found by an API-level evaluation job; fixed to pin only when the env supports it and to refuse a track the env does not hold. |
| Run links keyed by manifest `run_name` but endpoints keyed by directory name | With `--run-name`, the two differ, so a UI linking by run name 404'd. Found by exercising the live server; `list_runs` now reports both ids and the server resolves either. |
| Replay comparison was not start-station invariant | With `random_start_station`, comparing absolute race times at each station made a partial lap look like it was losing time it never had a chance to lose. Segment gaps (time per station pair) are now the headline metric. |
| The final curriculum stage re-applied the episode cap | A `1.0` length fraction overrode the env's own `max_steps` with the same value — harmless but wrong in principle, and it broke the "no cap" contract. `episode_max_steps` now returns `None` for full-length stages. |
| The model registry read dims off a normalisation-wrapped learner | `observation_dim`/`action_dim` came back `null` for every wrapped checkpoint. The registry now unwraps `inner`. |
| `tmai record-demo` dereferenced `.track` on a `CenterlineTrack` from `library.train` | The CLI crashed before recording when a track library was configured. It now passes the selected `CenterlineTrack` directly; an end-to-end simulated-recording test covers the wiring. |
| `tmai replay show --out` passed an unsupported `car_positions` keyword to `TrackView.render` | Rendering a replay failed at runtime. The CLI now passes the replay positions as its trajectory, covered by a PNG export regression test. |
| `tmai validate-config` printed the one-frame observation width for temporally stacked policies | With `history_length: 3`, it reported 20 although the policy receives 60 values. It now reports `stacked_dim`, with a CLI regression test. |
| Closed-track curvature rendering paired `N` curvature values with `N-1` points and omitted the closing corridor quad | `show-track`/PNG rendering failed on closed tracks and `.obj` meshes left a seam. Rendering, mesh export, and OBJ polylines now wrap over the seam. |
| The track-geometry API labelled the right-side offset as the left edge | The 3D viewer received swapped corridor-side labels even though the mesh shape looked plausible. API edge generation now matches `CenterlineTrack`/`TrackView` sign conventions, with a side-sign regression test. |

---

## Known limitations of the design

### The centreline is the only map knowledge

Phase 1 knows nothing about walls, ramps, obstacles or track width variation. The drivable
corridor is a single constant half-width around a recorded line. Consequences:

* The agent cannot learn to avoid an obstacle it cannot perceive.
* A map with wildly varying width is represented badly.
* "Off track" means "outside the corridor", which is an approximation of "off the road".

The fix is block-level geometry behind the same interface. See `.Map.Gbx` below.

### `.Map.Gbx` parsing is not implemented, deliberately

There is no Windows, no Trackmania and no sample map file here, so a parser could not have been
tested against the thing it parses. Shipping one would have been a guess presented as a feature.

What exists (`tmai/tracks/gbx.py`), all unit-tested: the `MapBlock`/`MapGeometry` data model,
`blocks_to_centerline()` (including that block insertion order does not affect the result), and
`GbxSource.load()` which raises `GbxUnavailableError` — a subclass of `NotImplementedError`, so
callers can distinguish "unfinished" from "corrupt file".

The third-party option (`pygbx`, PyPI) was evaluated and rejected for this phase: it targets
TMNF/TMUF with only partial TM2 support, requires the `python-lzo` C extension (which does not
build here without system headers), and is GPL-3.

`GbxSource.requirements()` lists what is needed:

1. A real `.Map.Gbx` from Trackmania (2020), Stadium environment.
2. A verified block extraction: model name, position, orientation, size.
3. A block-name → surface table marking drivable blocks.
4. A cross-check of the derived centreline against the same map driven manually
   (`tmai record-track`), so parser errors are visible rather than assumed away.

### Recorded centreline quality bounds reward quality

The centreline comes from driving the map once. A sloppy lap gives a sloppy reference line,
which biases the reward toward that line. Recording several laps and averaging, or using the
map author's medal ghost, is the obvious improvement and is not implemented.

### Generalisation is proven only on synthetic tracks

The library, splits, leakage prevention and translation invariance are all genuinely tested —
but with four procedurally generated centrelines. A policy that generalises across those may
still fail across real Trackmania maps, which have elevation, jumps, narrow sections and
surfaces the synthetic tracks do not model.

### Start-position randomisation is unavailable against the real game

The real game cannot place the car at an arbitrary station. This is handled honestly
(capability flag, raising driver, config validation, and disabled in `default.yaml`) rather
than silently ignored — but it does mean real-game training starts every episode at the start
line unless per-map `CheckpointData` states are recorded.

### One game instance

There is no parallel rollout. Throughput is capped by one game, mitigated only by the
game-speed multiplier. A vectorised environment and a learner service are the fix, and neither
exists.

### Latent state / partial observability

The observation is fully state-based. That is fine for a state-based policy but cannot support
a vision-based policy without adding an image term and an encoder.

### The simulated driver is not a physics model

`SimulatedGameDriver` is a short kinematic bicycle model with no tyres, no drift and no
collisions. It exists to test plumbing. **A policy trained against it has learned nothing about
Trackmania and will not transfer.** The trainer refuses to use it unless
`--allow-simulated-driver` is passed, every run records `driver: simulated` in its manifest,
the CLI prints a banner, and `tmai status` / `tmai compare` flag it.

### Seed repeats only create variety when the environment can randomise

With a deterministic policy and the simulated driver, episodes are repeatable for a fixed
seed; the default simulated configuration randomises start station/lateral offset, so distinct
seeds can still produce meaningfully different trials. If start randomisation is disabled,
repeated seeds may reproduce the same trajectory. The real game cannot reposition the car, so
its seed repeats do not guarantee varied starts. Benchmark bootstrap intervals are descriptive
and approximate, not a significance test; they resample track groups when possible and should
be interpreted cautiously with few maps.

### The GUI backend has no authentication

`tmai serve` can start training runs, delete models and read files, and it binds to
`0.0.0.0` by default (so it is reachable from preview proxies and other machines' browsers).
It is a local tool: **do not run it on an untrusted network**; use `--host 127.0.0.1` if in
doubt. See [`GUI.md`](GUI.md#security-model-read-this).

---

## What to do first on a real Windows host

In this order, because each step makes the next one trustworthy:

1. `tmai doctor` — confirm the toolchain and that TMInterface is reachable.
2. `tmai doctor --calibrate` — **all four checks must pass.** Apply the printed
   `position_scale`, `forward_axis` and `forward_sign` recommendations to the matching
   `driver.*` config keys before training. The forward-axis settings flow through telemetry,
   yaw and track-relative observations.
3. In offline/single-player mode, run `tmai record-track --out data/tracks/my_map.json` and
   inspect it with `tmai show-track`. Use `--family "author-pack"` when the map is related to
   another recorded layout; that prevents the family being split across train and test.
4. `tmai validate-config -c tmai/configs/default.yaml` — confirm the real-game config is
   coherent before committing to a long run.
5. Run a short `tmai train` (a few hundred or thousand steps) in offline/single-player mode;
   confirm the game responds, telemetry stays finite, and `reward/progress` increases. Stop
   immediately if control or reset behavior looks unsafe.
6. Close TMInterface/game during a disposable short run and verify the process exits with a
   useful error and preserves a checkpoint; then test Ctrl-C shutdown and checkpoint recovery.
7. Optionally use `tmai record-demo` and `tmai pretrain` to exercise behaviour cloning against
   real human input, then inspect the run and replays in `tmai serve`.
8. Add maps and family labels as available; audit split counts and geometry with
   `tmai list-tracks data/tracks --json`. Use `tmai benchmark --seed-repeats 3` for paired
   comparisons, remembering that Trackmania cannot randomise its start position.
9. Only then, attempt a long run.

Steps 1–6 are the missing live-game acceptance checks. They are intentionally a Windows
operator procedure rather than a claimed CI result: this repository has not controlled a
running Trackmania instance in the current environment.
