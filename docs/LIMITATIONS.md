# Limitations and verification status

This document is the honest account. Every claim about the real Trackmania integration is
listed with its verification status and what it would take to verify it.

**The single most important fact: no code in this repository has ever talked to a running
Trackmania instance.** This project was built in a Linux container with no Windows, no game
and no GPU. Everything below is written to make that boundary impossible to miss.

## Verification status

| Claim | Status | How it was verified / what is missing |
|---|---|---|
| `tminterface` 1.0.2 API surface used by the driver | **Verified** | Read from the published source (`interface.py`, `structs.py`, `constants.py`, `client.py`). |
| Field mapping `SimStateData → VehicleState/RaceState` | **Verified against real structs** | `tests/test_telemetry.py::TestAgainstRealStructs` decodes genuine `SimStateData` objects from byte buffers and checks the mapping. 7 tests. |
| Analog input range `[-65536, 65536]` | **Verified (documented)** | Taken from `TMInterface.set_input_state`'s docstring; asserted in `test_analog_full_scale_matches_tminterface_documentation`. Not exercised against a real car. |
| Driver control logic (reset sequencing, decimation, speed ratio, connection loss) | **Verified** | 31 tests in `tests/test_tminterface_driver.py` against a scripted tick source honouring the real `TickSource` contract. |
| TMInterface cannot connect on Linux | **Verified** | `mmap.mmap(-1, size, tagname=...)` raises `TypeError` on Linux; asserted in `test_mmap_tagname_is_windows_only`. |
| The game actually steers when `set_input_state` is called | **NOT VERIFIED** | Requires a live game. |
| `get_simulation_state()` returns sane values during a live race | **NOT VERIFIED** | Requires a live game. |
| Trackmania's forward axis is rotation column 0 | **NOT VERIFIED — must be calibrated** | `tmai doctor --calibrate` measures it. Do not trust the default. |
| Trackmania's position/velocity units are metres | **NOT VERIFIED — must be calibrated** | `tmai doctor --calibrate` measures `position_scale`. |
| Physics tick period is 100 Hz | **NOT VERIFIED** | `tick_period` check measures it. |
| `iface.respawn()` restarts an episode usefully | **NOT VERIFIED** | `ResetStrategy.COMMAND` exists as the alternative if respawn is insufficient. |
| Finish/checkpoint detection matches the game's own race logic | **NOT VERIFIED** | Wired to `player_info.race_finished` and `on_checkpoint_count_changed`; needs an in-game check. |
| Game-speed multiplier gives a usable training speedup | **NOT VERIFIED** | Upstream documents the mechanism and warns about factors above ~100 dropping inputs. |
| The reward function produces good driving in the real game | **NOT VERIFIED** | Weights are a reasoned starting point, never tuned against real physics. |
| A policy trained here can drive a real map | **NOT DONE** | No real training has happened. |
| RL environment, reward, termination | **Verified** | 46 tests, including `gymnasium.utils.env_checker.check_env`. |
| SAC learner | **Verified** | 52 tests, including a Bellman fixed-point check and a numerical log-prob check. |
| Training loop, checkpointing, resume | **Verified** | 40 tests; a full run was executed and its artefacts inspected. |
| Track geometry and centreline recording | **Verified** | 43 tests. |
| Telemetry calibration detects wrong conventions | **Verified** | 18 tests; deliberately fed wrong conventions and required failures. |
| Simplified track visualisation | **Verified** | 13 tests (headless PNG + `.obj`). |

## Known limitations of the design

### The centreline is the only map knowledge

Phase 1 knows nothing about walls, ramps, obstacles or track width variation. The drivable
corridor is a single constant half-width around a recorded line. Consequences:

* The agent cannot learn to avoid an obstacle it cannot perceive.
* A map with wildly varying width is represented badly.
* "Off track" means "outside the corridor", which is an approximation of "off the road".

The fix is block-level geometry from a `.Map.Gbx` parser behind the same `TrackGeometry`
interface. See `docs/ROADMAP.md`.

### Recorded centreline quality bounds reward quality

The centreline comes from driving the map once. A sloppy lap gives a sloppy reference line,
which biases the reward toward that line. Recording several laps and averaging, or using the
map author's medal ghost, is the obvious improvement and is not implemented.

### One game instance

There is no parallel rollout. Throughput is capped by one game, mitigated only by the
game-speed multiplier. A vectorised environment and a learner service are the fix, and neither
exists.

### Latent state / partial observability

The observation is fully state-based (speed, gear, sliding, track-relative geometry). That is
fine for a state-based policy but means the current observation cannot support a vision-based
policy without adding an image term and an encoder.

### No observation normalisation layer

`ObservationScales` holds fixed normalisation constants. If the real game's ranges differ a lot
from the assumed ones, training will be slower than it should be. A running normaliser is a
small, additive change.

### The simulated driver is not a physics model

`SimulatedGameDriver` is a ~40-line kinematic bicycle model with no tyres, no drift and no
collisions. It exists to test plumbing. **A policy trained against it has learned nothing
about Trackmania and will not transfer.** The trainer refuses to use it unless
`--allow-simulated-driver` is passed, every run records `driver: simulated` in its manifest,
and the CLI prints a banner.

### Evaluation is deterministic and the sim is deterministic

With a deterministic policy and the simulated driver, repeated evaluation episodes are
identical, so `eval_episodes > 1` wastes time there. Against the real game there is
nondeterminism and repeated episodes are meaningful.

## What to do first on a real Windows host

In this order, because each step makes the next one trustworthy:

1. `tmai doctor` — confirm the toolchain and that TMInterface is reachable.
2. `tmai doctor --calibrate` — **all four checks must pass.** Apply the recommended
   `position_scale` and, if the forward axis is not column 0, that finding has to be fed into
   `VehicleState.forward_vector` (currently hard-coded to column 0 — this is the one place a
   calibration failure requires a code change rather than a config change).
3. `tmai record-track` — drive one clean lap and eyeball the result with `tmai show-track`.
4. A short `tmai train` run (a few thousand steps) purely to confirm the loop runs against the
   real game and that `reward/progress` increases.
5. Only then, a long run.

Steps 1–4 are the missing verification. Until they are done, nothing in this repository has
been shown to control Trackmania.
