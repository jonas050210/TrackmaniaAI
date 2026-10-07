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
| Driver control logic (reset sequencing, decimation, speed ratio, connection loss) | **Offline-tested** | 31 tests in `tests/test_tminterface_driver.py` against a scripted tick source honouring the real `TickSource` contract. |
| TMInterface cannot connect on Linux | **Verified** | `mmap.mmap(-1, size, tagname=...)` raises `TypeError` on Linux; asserted in `test_mmap_tagname_is_windows_only`. |
| The game actually steers when `set_input_state` is called | **Requires the real PC** | — |
| `get_simulation_state()` returns sane values during a live race | **Requires the real PC** | — |
| Trackmania's forward axis is rotation column 0 | **Requires the real PC — must be calibrated** | `tmai doctor --calibrate` measures it. Do not trust the default. |
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
| Track geometry and centreline recording | **Verified** | 53 tests. |
| Track library, splits, leakage prevention | **Verified** | 44 tests, including split determinism and refusal to place one track in two splits. |
| Multi-track environment | **Verified** | Track switching, start randomisation, driver caching, episode context. |
| Translation invariance of the observation | **Verified** | A translated track produces a bit-identical observation (diff exactly 0.0). |
| Telemetry calibration detects wrong conventions | **Verified** | Deliberately fed wrong conventions and required failures. |
| Evaluation reports (per-track, per-split, gap) | **Verified** | 38 tests against artefacts from a real training run. |
| Status/compare CLI and dashboard API | **Verified** | 25 + 38 tests. |
| Simplified track visualisation | **Verified** | Headless PNG + `.obj`. |
| **Total** | | **543 tests, `ruff` clean, 93% statement coverage** |

Measured with `pytest --cov=tmai --cov-report=term-missing`. Coverage is high everywhere except
one file, deliberately:

| File | Coverage | Why |
|---|---|---|
| `tmai/game/tminterface/session.py` | **43%** | The Windows named-shared-memory transport. The untested lines are the ones that require a running game and cannot execute anywhere else. |
| `tmai/cli.py` | 89% | The remainder is `doctor --calibrate` and `record-track`, both of which drive the real game. |

A low number on the transport is the honest result, not a gap to paper over: testing it here
would mean faking the game.

### Things that have never happened

| Claim | Status |
|---|---|
| The reward function produces good driving in the real game | **Requires the real PC** — weights are reasoned defaults, never tuned against real physics. |
| A policy trained here can drive a real map | **Not done** — no real training has happened. |
| The agent generalises to unseen *real* maps | **Not done** — generalisation is proven only across four synthetic centrelines. |

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
| Nested dicts were stringified in `events.jsonl` | Evaluation reports were written as the *string* `"{'a': 1}"`, breaking machine readability. |
| Episode events did not record which track they ran on | Per-track episode analysis was impossible for a multi-track run; a bad track looked like a bad policy. |
| `Any` used but not imported in `tmai/cli.py` | A `NameError` waiting on the `compare` code path; caught by `ruff`. |

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

### Evaluation is deterministic and the sim is deterministic

With a deterministic policy and the simulated driver, repeated evaluation episodes are
identical, so `eval_episodes > 1` wastes time there. Against the real game there is
nondeterminism and repeated episodes are meaningful.

---

## What to do first on a real Windows host

In this order, because each step makes the next one trustworthy:

1. `tmai doctor` — confirm the toolchain and that TMInterface is reachable.
2. `tmai doctor --calibrate` — **all four checks must pass.** Apply the recommended
   `position_scale` and, if the forward axis is not column 0, that finding has to be fed into
   `VehicleState.forward_vector` (currently hard-coded to column 0 — this is the one place a
   calibration failure requires a code change rather than a config change).
3. `tmai record-track` — drive one clean lap and eyeball the result with `tmai show-track`.
4. `tmai validate-config -c tmai/configs/default.yaml` — confirm the real-game config is
   coherent before committing to a long run.
5. A short `tmai train` run (a few thousand steps) purely to confirm the loop runs against the
   real game and that `reward/progress` increases.
6. Record several more maps, then `tmai list-tracks data/tracks` to check the split geometry
   coverage is comparable.
7. Only then, a long run.

Steps 1–5 are the missing verification. Until they are done, nothing in this repository has
been shown to control Trackmania.
