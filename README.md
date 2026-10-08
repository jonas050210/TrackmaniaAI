# TrackmaniaAI

Reinforcement learning that drives the **real** Trackmania game. The agent starts with no
driving knowledge and learns steering, throttle, braking and drifting from a reward based on
track progress, with the long-term goal of driving maps it has never seen.

The system is complete end to end: real-game integration, SAC training with temporal
observations and curriculum learning, human demonstrations and behaviour cloning, evaluation
and benchmarking, a model registry, replay/ghost analysis, and a **local web command center**
(GUI) with an interactive 3D track viewer.

---

## Status: what actually works today

| | |
|---|---|
| Real Trackmania integration | Implemented against TMInterface's documented API. **Not yet verified on a live game** — it needs a Windows host with Trackmania, which this repository's CI does not have. See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md). |
| RL environment, reward, termination | Implemented and unit-tested (**746 tests**). Crash/out-of-bounds/fall/wrong-way terminate immediately with distinct penalties; crash detection distinguishes impact, scrape and parking. |
| SAC learner | Implemented from scratch, tested, including a Bellman fixed-point check. |
| Temporal observations | Frame stacking (`observation.history_length`) with an env-owned stacker; translation invariance holds under stacking. |
| Multi-track training + generalisation | Implemented and tested: track library, train/validation/test splits, leakage prevention, held-out evaluation. |
| Curriculum learning | Training-only track reveal by difficulty + episode-length caps; held-out evaluation is never filtered. |
| Human demonstrations + behaviour cloning | `tmai record-demo` records what the *game* reports the human typed; `tmai pretrain` (or `bc:` in config) warm-starts the policy; resumable checkpoints. |
| Observation normalisation | Implemented as a learner decorator, so replay data stays valid as statistics improve. |
| Training loop, checkpointing, resume, logging | Implemented and tested end to end, including resource monitoring in the metrics stream. |
| Replays & ghost analysis | Every training episode can be recorded (`train.record_replays`), replayed, and compared station-by-station against a human ghost. |
| Evaluation, benchmarking, model registry | Per-track/per-split reports, `tmai benchmark` ranks models under one protocol, `tmai models` manages named self-contained models. |
| Track model + recording | Implemented; recorded from real telemetry with `tmai record-track`. |
| `.Map.Gbx` parsing | **Deliberately not implemented.** Isolated behind an interface with a clear account of what verification it needs. See below. |
| Simplified 3D track view | Implemented (headless PNG + `.obj` export). |
| **GUI command center** | **Built.** Local backend (`tmai serve`: FastAPI + WebSocket) + web frontend (React + three.js): overview, runs, training, evaluation & benchmarking, models, tracks (3D), replays & ghosts, configuration, diagnostics. See [`docs/GUI.md`](docs/GUI.md). |

Read [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) before believing anything else on this page.
It states, claim by claim, what is verified and what is not.

---

## Why SAC and not PPO

The requirement was to justify the algorithm rather than assume it. In short: samples are the
scarce resource (they come from one real game running in real time), PPO is on-policy and
discards them, and SAC is off-policy and reuses them. The action space is continuous and
bounded, which is SAC's native setting. Full argument in [`docs/ALGORITHM.md`](docs/ALGORITHM.md).

The neural network (`tmai.models`) is the policy/value model; the RL algorithm
(`tmai.agents`) is what trains it. They are separate modules with a `Learner` protocol between
them, so swapping in PPO, TD3 or REDQ later is additive. Observation normalisation is a
decorator over that same protocol — evidence the seam works.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # everything, for development
```

Targeted extras:

| Extra | Purpose |
|---|---|
| `pip install -e ".[learn]"` | PyTorch — needed on the machine that trains |
| `pip install -e ".[game]"` | `tminterface` — needed on the Windows machine running the game |
| `pip install -e ".[viz]"` | matplotlib — simplified track visualisation |
| `pip install -e ".[gui]"` | FastAPI + uvicorn + websockets — the GUI backend (`tmai serve`) |

The game host and the training host are expected to be different machines; nothing requires
both on one box.

### The GUI frontend

```bash
cd gui && npm install && npm run build      # produces gui/dist, served by `tmai serve`
```

---

## Quick start

### 0. Open the command center (optional but recommended)

```bash
tmai serve          # http://localhost:8765 — runs, training, evaluation, models, tracks in 3D,
                    # replays & ghosts, configuration, diagnostics, all live
```

The GUI is a thin client over the same backend the CLI uses; the two can never disagree.
See [`docs/GUI.md`](docs/GUI.md).

### 1. Check the environment and your config

```bash
tmai doctor
tmai validate-config -c tmai/configs/default.yaml
```

`validate-config` catches a broken run before it costs you three hours: conflicting track
sources, warm-up longer than the run, a replay buffer smaller than the batch, and the
real-game/start-randomisation conflict described below.

### 2. Prove the pipeline runs (no game needed)

This uses the labelled toy model — **not** Trackmania — and exists only to validate plumbing:

```bash
tmai train -c tmai/configs/smoke.yaml
tmai eval  --checkpoint runs/<the-run-it-just-made> --split validation,test
```

`multitrack_smoke.yaml` exercises the whole generalisation path (four tracks, two splits,
held-out evaluation) in a few seconds. `pipeline_smoke.yaml` exercises curriculum +
behaviour cloning + replay recording on top. The trainer refuses `driver.kind: simulated`
unless you pass `--allow-simulated-driver`.

### 3. Teach it from your own driving (optional)

```bash
tmai record-demo -c tmai/configs/default.yaml --out data/demos/my_lap.jsonl   # drive one lap
tmai pretrain  -c tmai/configs/default.yaml --demo data/demos/my_lap.jsonl --out models/pretrained.pt
tmai train     -c tmai/configs/default.yaml --resume models/pretrained.pt
```

or set `bc.enabled: true` with `bc.demo_paths` in the config to pretrain automatically at
the start of a training run.

### 4. Drive the real game

On the Windows host running Trackmania through TMInterface:

```bash
tmai doctor --calibrate                  # verify + measure the integration
tmai record-track --out data/tracks/my_map.json --name my_map   # drive one clean lap
tmai train -c tmai/configs/default.yaml
```

`default.yaml` trains on every centreline in `data/tracks/` with a 70/15/15
train/validation/test split, and evaluates on the held-out maps every 20 000 steps.

### 5. Inspect, compare, promote what you have

```bash
tmai list-tracks data/tracks             # geometry fingerprint + split per map
tmai status --run runs/<run>             # live status, metrics, evaluations
tmai compare runs/<run-a> runs/<run-b>   # side-by-side
tmai show-track --track data/tracks/my_map.json --out track.png
tmai replay list --run runs/<run>        # recorded episode replays
tmai models register --name v1 --checkpoint runs/<run>/checkpoint_*.pt
tmai benchmark --model v1=runs/<run-a> --model v2=runs/<run-b> --split validation,test
```

---

## Commands

| Command | Purpose |
|---|---|
| `tmai doctor` | Environment and game-integration health report; `--calibrate` measures telemetry conventions |
| `tmai validate-config` | Check a configuration without running anything |
| `tmai train` | Run training. `--resume`, `--steps`, `--seed`, `--device`, `--set key=value` |
| `tmai eval` | Evaluate a checkpoint across every track in a split (`--split a,b`): per-track finish rate, progress, lap times, crashes |
| `tmai record-demo` | Record a human-driven lap as a behaviour-cloning demonstration (JSONL) |
| `tmai pretrain` | Behaviour-clone demonstrations into the policy; saves a resumable checkpoint |
| `tmai benchmark` | Evaluate several models across splits under one protocol and rank them |
| `tmai models` | Model registry: `list`, `register`, `info`, `delete`, `tag` |
| `tmai replay` | Replay analysis: `list`, `show`, `compare` (AI replay vs human ghost) |
| `tmai compare` | Compare several runs side by side (JSON with `--json`) |
| `tmai status` | Inspect a run: `--run <dir>`, `--list`, or `--json` for a full dashboard snapshot |
| `tmai list-tracks` | List a track directory with geometry statistics and split assignment |
| `tmai record-track` | Record a map centreline by driving it once |
| `tmai show-track` | Render the simplified 3D track view to PNG |
| `tmai export-obj` | Export the track mesh for an external viewer |
| `tmai serve` | Start the GUI backend + serve the built frontend at `/` |

Every command accepts `-c config.yaml` and repeatable `--set section.field=value` overrides.

---

## Generalisation: driving maps it has never seen

Training on one map and testing on that map measures memorisation, not driving. Four
mechanisms make the distinction enforceable rather than a matter of discipline:

* **Deterministic splits.** A track's split is a function of its identity (map UID, or a hash
  of its geometry), not of insertion order. Adding a map never moves another map between
  splits, so results stay comparable across runs.
* **Leakage prevention.** A track identity can only ever belong to one split; the same map
  recorded twice is *the same track* for this purpose.
* **Multi-track training from step one.** A different track is sampled at every episode reset,
  and the policy never sees track identity or absolute world coordinates. A test translates a
  track by 750 m and asserts the observation is bit-for-bit unchanged.
* **Curriculum over the training split only.** Early training reveals the easiest tracks
  first and shortens episodes; held-out evaluation always runs the full suite, so a
  curriculum can never flatter the numbers it is measured by.

Held-out evaluation runs on a separate environment over the validation split and reports a
**generalisation gap** (train progress minus held-out progress). A positive gap is overfitting,
and it is visible in the metrics stream rather than hidden inside a blended average.

`tmai list-tracks` reports geometry coverage per split, which is the cheap check on whether the
held-out maps are even comparable to the training maps.

---

## Run output

Each run writes a self-describing directory:

```
runs/2026-10-07T14-02-11Z_sac-multitrack/
    manifest.json     config + git sha + versions + driver/env + track library report + seed
    config.yaml       resolved config, verbatim
    metrics.jsonl     one JSON record per logged step (append-only, crash-safe)
    events.jsonl      episodes (with track), evaluations, held-out evals, curriculum stages, BC, errors
    run.log           human-readable log
    checkpoint_*.pt   weights + optimisers + RNG state (atomic writes, pruned)
    best.pt           best-scoring checkpoint
    replays/          decimated episode trajectories (train.record_replays: true)
```

A run directory alone is enough to know exactly what was run. `tmai.api.status` reads these
files into JSON for a dashboard, and `tmai serve` exposes the same data to the GUI.

---

## Tests

```bash
pytest                 # 746 tests, no game, no Windows, no display
ruff check tmai tests
pytest --cov=tmai --cov-report=term-missing
```

The suite deliberately covers what can be covered without the game, including the real
TMInterface memory layout (tests decode genuine `SimStateData` structs), the driver's control
logic (tested against a scripted tick source that honours the real contract), the driving
rules (crash vs scrape vs parking), the temporal stacking invariants, the curriculum, the
behaviour-cloning fit, replay/ghost comparison, the model registry, benchmarking, resource
monitoring, and the full GUI backend over HTTP and WebSocket (including a real training
subprocess started through the API).

Tests have found real defects rather than rubber-stamping code — see
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md#bugs-the-test-suite-found) for the list.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Layering, data flow, why the boundaries are where they are |
| [`docs/INTEGRATION.md`](docs/INTEGRATION.md) | How real Trackmania communication works, protocol-level |
| [`docs/ALGORITHM.md`](docs/ALGORITHM.md) | Algorithm selection: requirements → SAC, and why not PPO |
| [`docs/REWARD.md`](docs/REWARD.md) | Reward design, the loopholes it closes, and how to tune it |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Configuration, multi-track runs, curriculum, BC, replays, resume |
| [`docs/GENERALIZATION.md`](docs/GENERALIZATION.md) | Track representation, splits, anti-memorisation, evaluation |
| [`docs/GUI.md`](docs/GUI.md) | The command center: pages, API map, jobs, security model |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | **What is verified and what is not.** Read this. |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | What is left, in priority order |

## Licence

MIT. Trackmania is a trademark of Ubisoft/Nadeo; this project is not affiliated with or
endorsed by them. See [`docs/INTEGRATION.md`](docs/INTEGRATION.md) for the third-party tooling
this project depends on and the terms that apply to it.
