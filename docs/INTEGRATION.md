# Real Trackmania integration

## The integration options, and which one this project uses

There is no official Nadeo API for controlling a car in Trackmania (2020). The realistic
options are:

| Option | Telemetry | Control | Platform | Verdict for this project |
|---|---|---|---|---|
| **TMInterface** | Full vehicle + race state via shared memory | Analog steer/gas injected into the input layer; **plus a game-speed multiplier** | Windows | **Chosen.** Everything needed, no extra drivers, and game-speed manipulation is what makes long training runs affordable. |
| **Openplanet plugin** | Telemetry over a local socket (plugin-dependent) | Needs a separate input path (virtual gamepad) | Windows / Proton | Viable fallback. Adds an AngelScript plugin and a virtual-gamepad driver to the dependency chain. |
| **ManiaPlanet / TMNF XML-RPC** | Race and player info | **None** — the protocol cannot steer a car | Cross-platform | Not usable for RL control. |
| Screen capture + input simulation | Indirect (OCR/vision) | Keyboard/gamepad emulation | Cross-platform | Rejected: slow, lossy, and cannot accelerate training. |

TMInterface was chosen because it is the only option that gives, from one documented Python
client: full telemetry, analog control, deterministic-ish frame-locked stepping, and a game
speed multiplier.

## What TMInterface is

[TMInterface](https://github.com/donadigo/TMInterface) is a third-party tool that loads into
the Trackmania process and exposes a synchronous, callback-driven message protocol over a
**Windows named memory-mapped file**. The official Python client is `tminterface` on PyPI
(this project pins `>=1.0.2`; the API described below was read from that version's source).

The game calls into the client once per physics tick. While it waits for the client's reply
the simulation is parked. That gives frame-locked control with no dropped or duplicated
actions — exactly what a real-time control loop needs.

> **Platform limit, verified.** The connection is `mmap.mmap(-1, size, tagname="TMInterface0")`.
> `tagname` is Windows-only; on Linux this raises
> `TypeError: 'tagname' is an invalid keyword argument for this function`. There is no
> workaround in this project's control. `TMInterfaceSession.start()` detects the platform and
> raises `UnsupportedPlatformError` with an actionable message. This is covered by
> `tests/test_tminterface_driver.py::TestSessionPlatformGuard`.

## Connection

```python
from tminterface.interface import TMInterface      # spawns its own daemon thread
iface = TMInterface("TMInterface0", 65535)
iface.register(client)                             # client is a tminterface.client.Client
```

`tminterface.run_client` is deliberately **not** used: it installs signal handlers and blocks,
which fails from a worker thread. `TMInterface.register` starts the same `_main_thread` loop
without touching signals.

On registration the session sets `set_timeout(-1)`, so the game waits indefinitely for our
reply instead of deregistering us after the default 2 s. That is what allows a slow learner to
think without losing the connection.

## Reading state

Inside `on_run_step`, `iface.get_simulation_state()` returns a `SimStateData` decoded from the
game's own memory. The fields this project uses:

| Accessor | Meaning |
|---|---|
| `flags` | bitmask (`SIM_HAS_TIMERS`, `SIM_HAS_DYNA`, `SIM_HAS_PLAYER_INFO`) — **checked before use** |
| `position` | world position, 3 floats (zeros when `SIM_HAS_DYNA` is unset) |
| `velocity` | world velocity, 3 floats |
| `rotation_matrix` | 3×3 vehicle-to-world matrix |
| `race_time` | race time in **milliseconds**, `-1` before the countdown finishes |
| `num_respawns` | respawn counter |
| `scene_mobil.sync_vehicle_state` | `speed_forward`, `speed_sideward`, `rpm`, `gearbox_state` |
| `scene_mobil.engine` | `gear`, `max_rpm` |
| `scene_mobil.is_sliding` | drift state |
| `player_info` | `race_finished`, `cur_cp_count`, `display_speed` |

`tmai/game/tminterface/telemetry.py` maps these onto the project's own `VehicleState` /
`RaceState`. The mapping is **pure** and raises `GameProtocolError` rather than returning
silently wrong data when the dynamics region is absent or a value is non-finite.

Checkpoint totals arrive separately through `on_checkpoint_count_changed(current, target)`.

## Sending control

```python
iface.set_input_state(steer=int, gas=int, brake=bool)
```

`steer` and `gas` are analog integers in **`[-65536, 65536]`**. `tmai` maps its `[-1, 1]`
action space onto that range (`ANALOG_FULL_SCALE = 65536`). Trackmania's standard bindings
have no analog brake, so `brake` is sent as a binary input, thresholded at 0.5.

Actions are clipped twice — once in `TMInterfaceDriver.step` and again in
`TMInterfaceSession.push_action` — so a policy emitting an out-of-range value can never reach
the game.

## Other game operations

| Call | Use here |
|---|---|
| `iface.set_speed(ratio)` | Game-speed multiplier. The main training-throughput lever. Upstream warns that factors above ~100 can make the game skip subsystems *including input processing*; `tmai` warns above 20 and the capability advertises `max_speed_ratio = 20.0`. This has not been measured in a live run. |
| `iface.give_up()` | Default `restart` strategy requests a new race attempt. A fresh running clock/checkpoint state is required before `reset()` returns. This reset sequence is unit-tested against a scripted source, not against Trackmania. |
| `iface.respawn()` | Checkpoint recovery, **not** a full episode reset after a checkpoint. It is available only as the explicit `respawn` strategy. |
| `iface.execute_command(cmd)` | Explicit `command` strategy for a host-specific reset command; the default is not this path. |
| `iface.get_context_mode()` | Distinguishes a normal race from replay validation. |

## Thread model

Every call into `TMInterface` must happen on its worker thread, because the request/response
handshake uses one shared buffer. Calls from the learner thread are therefore described as
small value objects (`RestartRaceOp`, `RespawnOp`, `SetSpeedOp`, `CommandOp`, `CallableOp`) and
executed at the top of the next callback. At each published control frame, the callback then
waits for the learner's next command before replying to TMInterface:

```
tminterface callback thread           learner / env thread
--------------------------           --------------------
on_run_step(iface, t):               next_frame(timeout)
  drain queued operations  <──────── request_op(op)
  apply current input
  read state and publish   ────────> next_frame returns GameFrame
  park callback at frame   <──────── push_action(a or None)
  apply new input / op
  return; game advances
```

`publish_every_n_ticks` controls the physics-tick decimation. The factory derives it from
`env.control_dt * driver.physics_hz` (defaults: 0.05 s and an estimated 100 Hz, or five ticks); `env.action_repeat` can hold each policy action across several such control intervals. The
configured rate is an estimate, not a measured property of the current game session.

This lock-step gate is intended to prevent the game advancing through unobserved control
frames while policy inference runs. If the learner stalls, the game callback also stalls until
`frame_timeout_s` causes the caller to fail and close the session. The behavior of the
upstream callback and game under this gate still requires a live Trackmania test.

## Timing semantics

The session uses the `on_run_step` callback as the frame boundary and submits input before the
callback returns; the installed TMInterface source shows that the protocol response follows
that callback. TMInterface documents that `set_input_state` affects a subsequent simulation
tick. Therefore a command can have a physics-tick actuation latency in addition to the
configured decimation interval. Race-clock deltas, when positive, measure actual simulated
time per environment transition; the configured interval is the fallback. Wall time is not
used for reward integration. All callback cadence, input timing, reset timing and game-speed
interactions remain **unverified against a running Trackmania instance**.

SAC now scales its per-transition discount as `gamma ** (elapsed_seconds / nominal_control_seconds)`.
This preserves the configured discount rate when real callback intervals differ from the
nominal step; old hand-built batches without elapsed-time metadata retain the ordinary `gamma`.

## Conventions that must be calibrated, not assumed

Nadeo does not document Trackmania's internal units or axis conventions. Rather than guess —
a wrong guess silently poisons every observation and the whole reward — `tmai` measures them.

```bash
tmai doctor --calibrate [--calibrate-steps 300] [--calibrate-out calib.json]
```

The calibrator drives the car in a straight line and checks physical invariants that must hold
regardless of the game's internal conventions:

| Check | Invariant |
|---|---|
| `forward_axis` | the forward column of the rotation matrix is parallel to velocity while driving forwards |
| `speed_forward_consistency` | `speed_forward == dot(velocity, forward)` |
| `position_scale` | `|Δposition| == |velocity| · dt` in the same unit; the ratio gives `position_scale` |
| `tick_period` | the observed frame period matches the assumed control `dt` |

It reports PASS/FAIL with numbers and prints the recommended `forward_axis`, sign and
`position_scale`. `tests/test_calibration.py` feeds it deliberately wrong conventions and
requires a failure — a calibrator that passed everything would be worthless.

Apply all three measured conventions directly, for example:

```bash
tmai train -c tmai/configs/default.yaml \
  --set driver.position_scale=0.01 \
  --set driver.forward_axis=2 \
  --set driver.forward_sign=-1
```

`forward_axis` and `forward_sign` are propagated into `VehicleState.forward_vector()` and
therefore affect yaw, heading error, and all track-relative observations. Calibration checks
`speed_forward` against the detected axis/sign rather than assuming column 0. These values
still need a live-game calibration; the default is not evidence that column 0 is correct.

## Legal and practical notes

* TMInterface is third-party software (GPL-3). Using it to modify a running game is against
  Nadeo's terms for **online** play. This project's use is local, single-player training on
  your own machine; do not use it on official online servers.
* `tminterface`'s own licence is GPL-3, which is why it is an *optional extra* and never a
  hard dependency: importing it is your choice, made on the game host.
* Nothing in this project reads protected assets, bypasses DRM, or communicates with
  Nadeo/Ubisoft services.

## Setting up the game host

1. Windows host with Trackmania (2020) installed.
2. Install TMInterface and launch Trackmania **through** it.
3. Load the map you want to train on and start a race (the car must be live — `tmai` waits for
   the `RUNNING` phase after each reset).
4. `pip install "trackmania-ai[game,learn]"`.
5. `tmai doctor --calibrate` and confirm every check passes.
6. `tmai record-track --out data/tracks/my_map.json --name my_map` — drive one clean lap.
7. Configure training to use the recorded maps and run `tmai validate-config`; start with a
   short `tmai train` before attempting a long run.
8. To drive a live map, set `track.path` to the centreline for the **currently loaded** map
   (and `track.directory: null`), then run `tmai play --checkpoint runs/<run> --record-replay`.
   `tmai play` runs at 1x by default, prints telemetry, progress and requested controls, and
   is bounded by the episode step limit; Ctrl+C requests neutral controls during shutdown.
   Actual live input delivery/release remains unverified. The printed controls are commands
   from the policy, not proof the game accepted them.
   Omitting `--checkpoint` selects
   the clearly labelled, untrained `CurvaturePilot` baseline instead.

The current TMInterface runtime cannot identify or switch maps. `tmai play` requires a
single configured track for real control and rejects a multi-map library rather than silently
selecting a centreline that might not match the game. These safeguards and the control loop
are tested offline; actual steering and map alignment still require the Windows game host.

## Adding a different backend

Implement the `GameDriver` protocol in `tmai/game/<name>/driver.py` and register it in
`tmai/training/factory.py::build_driver`. Nothing above `tmai.game` changes. An Openplanet
backend would need a telemetry socket client plus a control path (virtual gamepad or an
Openplanet input API) — both are outside the `GameDriver` contract's concerns.

## Recording a human demonstration

`tmai record-demo` drives the same `GameDriver` the trainer uses, but the inputs that get
recorded are the ones the **game reports** (`SceneVehicleCarState.input_steer` /
`input_gas` / `input_brake`, read back through `VehicleState.input_*`). With a human at the
wheel of the real game, that is the human's real control; the AI's own outputs are irrelevant
to the recording. The result is a JSONL dataset (`tmai/training/demos.py`) that
`tmai pretrain` / `bc:` turns into a warm start. See [TRAINING.md](TRAINING.md#demonstrations-and-behaviour-cloning).
