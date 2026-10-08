# Architecture

## The layering

```
┌───────────────────────────────────────────────────────────────────────┐
│ tmai.cli              every command (train, eval, benchmark, models,  │
│                       replay, record-demo, pretrain, serve, ...)      │
├───────────────────────────────────────────────────────────────────────┤
│ tmai.server           GUI backend: FastAPI app · jobs · WebSocket      │
│ gui/                  web frontend (React + three.js), served static  │
├───────────────────────────────────────────────────────────────────────┤
│ tmai.training         Trainer · evaluate · benchmark · curriculum ·   │
│                       demos (recording) · checkpoint · factory        │
├──────────────────────────────┬────────────────────────────────────────┤
│ tmai.agents                  │ tmai.models                            │
│ SAC learner, replay buffer,  │ ActorCriticNetwork (policy + values)   │
│ behaviour cloning (bc.py)    │ (the neural network it trains)         │
├──────────────────────────────┴────────────────────────────────────────┤
│ tmai.registry         named, self-contained models                    │
│ tmai.replay           episode replays + ghost comparison             │
│ tmai.monitoring       system resources in the metrics stream         │
│ tmai.api.status       read-only run dashboard layer (shared by CLI    │
│                       `status`/`compare` and the GUI)                │
├───────────────────────────────────────────────────────────────────────┤
│ tmai.env              TrackmaniaEnv · MultiTrackEnv · observation ·  │
│                       reward · termination                            │
├───────────────────────────────────────────────────────────────────────┤
│ tmai.tracks           CenterlineTrack: progress, corridor, curvature  │
├───────────────────────────────────────────────────────────────────────┤
│ tmai.game             GameDriver protocol                             │
│   ├─ tminterface/     REAL game: session (IPC) + driver (logic)       │
│   ├─ calibration.py   measures telemetry conventions                  │
│   └─ simulated.py     labelled test double (NOT Trackmania)           │
└───────────────────────────────────────────────────────────────────────┘
```

Two rules hold everywhere:

1. **Nothing above `tmai.game` imports a game-specific module.** The environment, reward,
   learner and trainer see only the `GameDriver` protocol and plain dataclasses. This is what
   makes the whole stack testable without Windows or the game.
2. **The neural network and the RL algorithm are separate modules** joined by the `Learner`
   protocol. `tmai.models` contains no loss function and no optimiser; `tmai.agents` contains
   no network architecture.

## The `GameDriver` seam

`tmai/game/protocol.py` defines the entire surface the RL stack uses:

```python
class GameDriver(Protocol):
    name: str
    capabilities: DriverCapabilities
    def open(self) -> None
    def close(self) -> None
    def is_connected(self) -> bool
    def reset(self) -> GameFrame
    def step(self, action: Action) -> GameFrame
    def set_speed_ratio(self, ratio: float) -> float
    def describe(self) -> dict[str, Any]
```

with frozen dataclasses `Action`, `VehicleState`, `RaceState`, `GameFrame` and a
`DriverCapabilities` value object. Capabilities are *declared*, not assumed: a driver that
cannot manipulate game speed is still usable, and the environment degrades instead of
breaking.

Units are part of the contract, not an implementation detail: metres, m/s, seconds since race
start, `steer` in `[-1, 1]` negative-left.

## Why the TMInterface driver is split in two

```
tmai/game/tminterface/session.py    TMInterfaceSession   ← owns the IPC, Windows-only
tmai/game/tminterface/driver.py     TMInterfaceDriver    ← reset sequencing, decimation,
                                                            speed ratio, error handling
tmai/game/tminterface/telemetry.py  pure mapping functions
tmai/game/tminterface/ops.py        queued game operations
```

They are joined by the `TickSource` protocol (`tmai/game/ticksource.py`), which is the whole
surface the driver needs from the bridge.

This split is the single most important testability decision in the project. The IPC layer is
genuinely untestable off Windows (it uses a Windows named memory-mapped file). Everything
else — reset sequencing, action clipping, speed-ratio handling, connection-loss detection,
checkpoint counting — is ordinary logic that tests exercise against a scripted tick source
honouring the same contract. See `tests/test_tminterface_driver.py`: 31 tests over the exact
code that runs against the game.

The telemetry mapping is pure for the same reason. `tests/test_telemetry.py` builds genuine
`tminterface.structs.SimStateData` objects from byte buffers and runs the mapper over them,
validating the field mapping against the real upstream memory layout.

## Data flow for one step

```
trainer            env                    driver               session (game thread)
  │                 │                       │                        │
  ├─ act(obs) ─────>│                       │                        │
  ├─ env.step(a) ──>│                       │                        │
  │                 ├─ step(a) ────────────>│                        │
  │                 │                       ├─ push_action(a) ──────>│ (mailbox, newest wins)
  │                 │                       │                        │  on_run_step(iface, t):
  │                 │                       │                        │    drain op queue
  │                 │                       │                        │    set_input_state(...)
  │                 │                       │                        │    get_simulation_state()
  │                 │                       │<─ GameFrame ───────────┤    publish frame
  │                 │<─ GameFrame ──────────┤                        │
  │                 ├─ project → progress                            │
  │                 ├─ reward(Δprogress, ...)                        │
  │                 ├─ termination check                             │
  │                 ├─ encode observation                            │
  │<─ (obs, r, terminated, truncated, info) ─┤                       │
  ├─ buffer.add(transition)                                          │
  ├─ learner.update(batch)   (updates_per_step × per env step)       │
```

The environment is **lock-step with the game**, not with the wall clock. One `step()` equals
`action_repeat` physics ticks. Real-time pacing, when wanted (for human-vs-AI later), belongs
in a wrapper on top.

## Observation design

The policy never sees absolute world coordinates. Every feature is either an ego quantity
(speed, rpm, gear, sliding, yaw rate) or a track-relative quantity (progress rate, lateral
offset, heading error, upcoming curvature at 5/10/20/40/80 m, distance to the corridor edges),
plus the previous action. 20 floats per frame by default.

That is the generalisation argument: two different maps produce comparable observations, so a
policy has a chance of transferring. Absolute coordinates would make every map a new problem.

The layout is explicit, ordered and introspectable (`env.observation_names`, with a `[k]`
suffix per stacked frame), versioned by `OBSERVATION_VERSION`, and recorded in every run
manifest.

**Temporal stacking.** `env.observation.history_length` (default 1) stacks the last N frames
into one observation, so the policy sees velocity and steering *change*, not just position.
The environment owns the stack (`ObservationStacker`): reset fills it with the first frame
repeated (no zero-padded window mid-episode), and the observation space reports
`stacked_dim = dim × history_length`. A checkpoint's `observation_dim` guard rejects loading
weights trained against a different stack depth. Translation invariance holds under
stacking (tested).

## Reward design

Progress along the centreline is the dominant term. Since each control step lasts a fixed
`dt`, metres-of-progress-per-step and lap time are the same objective up to a constant — so
maximising progress rate minimises lap time. Dense progress also guides an agent that has
never finished a lap, which a sparse finish reward cannot.

The obvious exploits are closed explicitly:

| Exploit | Countermeasure |
|---|---|
| Cut across the map and rejoin far ahead | Credited progress is capped by what is physically achievable in one step (`max_speed_for_progress × dt × cut_margin`) |
| Oscillate across one station | backwards progress is credited negatively, bounded |
| Shortcut across grass | per-metre penalty outside the drivable corridor |
| Finish by an invalid route | the game's checkpoint counter is authoritative; `invalid_finish` is flagged |
| Farm reward by crashing into a wall | a significant wall collision terminates the episode immediately with a one-off `crash_penalty` |
| Farm reward by leaving the map | a confirmed out-of-bounds or fall terminates immediately with its own penalty |

Every component is returned in `RewardBreakdown` and logged, so a long run can be diagnosed
from its metrics alone.

## Configuration

`RunConfig` is a tree of dataclasses (`driver`, `track`, `multi`, `normalize`, `env`, `sac`,
`replay`, `curriculum`, `bc`, `train`) with YAML round-trip and `--set a.b.c=value` overrides
that coerce to the declared field type. The resolved config is written verbatim into the run
manifest and into `config.yaml`. Dataclasses rather than a schema library keeps this
dependency-free.

## Analysis and management layers

* **`tmai/training/curriculum.py`** — curriculum over the *training* split only: tracks are
  revealed easiest-first (mean curvature + corner density) and episodes can be shortened
  early. Held-out evaluation always runs the full suite.
* **`tmai/training/demos.py` + `tmai/agents/bc.py`** — human demonstrations (JSONL of
  observation + game-reported action) and supervised pretraining of the actor from them.
* **`tmai/replay.py`** — decimated episode replays, a replay store, and ghost comparison
  (station gaps and start-invariant segment gaps).
* **`tmai/registry.py`** — named, self-contained models (`models/<name>/{model.json,policy.pt}`).
* **`tmai/training/benchmark.py`** — one protocol, many models, ranked report.
* **`tmai/monitoring.py`** — stdlib-only resource snapshot, logged as `system/*` metrics.
* **`tmai/server/`** — the GUI backend (see [GUI.md](GUI.md)): FastAPI app, job manager,
  subprocess training jobs, WebSocket live tick.

## Where the boundaries would move

* **Multiple game instances** (parallel rollout): `GameDriver` is already per-instance; the
  missing piece is a vectorised env wrapper and a shared learner service.
* **Different algorithm**: implement `Learner`; nothing else changes.
* **Vision observations**: add an encoder in `tmai.models` and an image term to the
  observation spec. The `GameDriver` contract would need a screenshot method.
* **Block-level track geometry**: `CenterlineTrack` is the only map knowledge today. A
  `.Map.Gbx` parser would produce a richer `TrackGeometry` behind the same interface.
