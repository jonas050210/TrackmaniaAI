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
| `iface.set_speed(ratio)` | Game-speed multiplier. The main training-throughput lever. Upstream warns that factors above ~100 can make the game skip subsystems *including input processing*; `tmai` warns above 20 and the capability advertises `max_speed_ratio = 20.0`. |
| `iface.respawn()` | Default episode restart. |
| `iface.execute_command(cmd)` | Alternative restart path (`ResetStrategy.COMMAND`), for hosts where a full race restart is needed. |
| `iface.get_context_mode()` | Distinguishes a normal race from replay validation. |

## Thread model

Every call into `TMInterface` must happen on its worker thread, because the request/response
handshake uses one shared buffer. Calls from the learner thread are therefore described as
small value objects (`RespawnOp`, `SetSpeedOp`, `CommandOp`, `CallableOp`) and executed at the
top of the next tick:

```
tminterface worker thread            learner / env thread
-------------------------            --------------------
on_run_step(iface, t):               push_action(a)      ──┐ mailbox, newest wins
  drain op queue       <──────────   request_op(op)        │
  apply pending input                                      │
  set_input_state(...)                                     │
  get_simulation_state()                                   │
  publish GameFrame  ────────────>   next_frame(timeout) <─┘
```

If the learner is slow, the newest action wins and the previous input is held, rather than
stalling the game. If no frame arrives within `frame_timeout_s`, `step()` raises
`GameTimeoutError` instead of hanging.

## Timing semantics

TMInterface applies an input set during tick *t* at tick *t+1* (documented on
`set_input_state`). `step()` therefore returns the frame observed at the tick after the input
was injected: a one-tick (≈10 ms) actuation latency inherent to this integration. This is the
standard real-time-gym trade-off and is not something the driver can remove.

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

Apply the result with `--set driver.position_scale=0.01`.

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
7. `tmai validate-config -c tmai/configs/default.yaml`, then `tmai train -c tmai/configs/default.yaml --set driver.speed_ratio=8`.

## Adding a different backend

Implement the `GameDriver` protocol in `tmai/game/<name>/driver.py` and register it in
`tmai/training/factory.py::build_driver`. Nothing above `tmai.game` changes. An Openplanet
backend would need a telemetry socket client plus a control path (virtual gamepad or an
Openplanet input API) — both are outside the `GameDriver` contract's concerns.
