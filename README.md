# TrackmaniaAI

Reinforcement learning that drives the **real** Trackmania game. The agent starts with no
driving knowledge and learns steering, throttle, braking and drifting from a reward based on
track progress, with the long-term goal of driving maps it has never seen.

The software stack is implemented end to end: a real-game integration path, SAC training with
temporal observations and curriculum learning, demonstrations and behaviour cloning, evaluation
and benchmarking, a model registry, replay/ghost analysis, and a **local web command center**
(GUI) with an interactive 3D track viewer. Live Trackmania behavior remains unverified.

---

## Status: what actually works today

| | |
|---|---|
| Real Trackmania integration | Implemented against TMInterface's documented API. **Not yet verified on a live game** — it needs a Windows host with Trackmania, which this repository's CI does not have. See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md). |
| RL environment, reward, termination | Implemented and unit-tested offline. Crash/out-of-bounds/fall/wrong-way terminate with distinct penalties; crash detection distinguishes impact, scrape and parking. |
| SAC learner | Implemented from scratch, tested, including a Bellman fixed-point check. |
| Temporal observations | Frame stacking (`observation.history_length`) with an env-owned stacker; translation invariance holds under stacking. |
| Multi-track training + generalisation | Implemented and tested offline with the simulated driver: track library, splits, leakage prevention, held-out evaluation. The real driver cannot switch maps yet. |
| Curriculum learning | Training-only track reveal by difficulty + episode-length caps; held-out evaluation is never filtered. |
| Human demonstrations + behaviour cloning | `tmai record-demo` reads game-reported human inputs by design; `tmai pretrain` (or `bc:` in config) warm-starts the policy. The live human-recording path has not been verified in Trackmania. |
| Observation normalisation | Implemented as a learner decorator, so replay data stays valid as statistics improve. |
| Training loop, checkpointing, resume, logging | Implemented and tested end to end, including resource monitoring in the metrics stream. |
| Replays & racing analysis | Episodes can be recorded, compared against a human ghost, summarized by sector pace/lateral position, and mapped into a spatial failure heatmap. |
| Evaluation, benchmarking, model registry | Per-track/per-split reports, paired same-track model comparisons, `baseline:curvature` heuristic, and a self-contained model registry. |
| Track model + recording | `tmai record-track` is implemented; recording from live Trackmania telemetry has not been verified. |
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
cd gui && npm ci && npm run build      # requires Node 20.19+ or 22.12+, produces gui/dist
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
real-game/start-randomisation conflict described below. The real-game preset expects
`data/tracks/my_map.json` after you record that map; validation does not create the file.

### 2. Prove the pipeline runs (no game needed)

This uses the labelled toy model — **not** Trackmania — and exists only to validate plumbing:

```bash
tmai train -c tmai/configs/smoke.yaml
tmai eval  --checkpoint runs/<the-run-it-just-made> --split train
```

`multitrack_smoke.yaml` exercises the generalisation path (four synthetic tracks, train and
validation splits, held-out evaluation) in a few seconds. `pipeline_smoke.yaml` also tests
curriculum, behavior cloning, and replay recording; it needs a small generated demonstration
first:

```bash
tmai record-demo -c tmai/configs/pipeline_smoke.yaml --allow-simulated-driver \
  --max-steps 400 --out data/demos/smoke.jsonl
tmai train -c tmai/configs/pipeline_smoke.yaml
```

That demo is explicitly synthetic test data, **not human driving** and not a Trackmania
result. The trainer refuses `driver.kind: simulated` unless it is explicitly allowed.

### 3. Record the real map, then optionally teach it from your driving

On the Windows host with the matching map loaded in Trackmania, calibrate the connection
and record its centreline before any command that uses the default preset:

```bash
tmai doctor --calibrate
tmai record-track --out data/tracks/my_map.json --name my_map   # drive one clean lap
```

To warm-start from a human demonstration (optional):

```bash
tmai record-demo -c tmai/configs/default.yaml --out data/demos/my_lap.jsonl   # drive one lap
tmai pretrain  -c tmai/configs/default.yaml --demo data/demos/my_lap.jsonl --out models/pretrained.pt
tmai train     -c tmai/configs/default.yaml --resume models/pretrained.pt
```

or set `bc.enabled: true` with `bc.demo_paths` in the config to pretrain automatically at
the start of a training run.

### 4. Train on the recorded map

On that Windows host, with the same map loaded in Trackmania:

```bash
tmai train -c tmai/configs/default.yaml
```

To put a policy in the driver's seat, configure **only the centreline for the map currently
loaded in Trackmania** and run `tmai play`. It defaults to 1x game speed, has a finite
per-episode step cap, prints telemetry, progress and commanded controls, and requests neutral
controls on Ctrl+C or normal shutdown. Live input delivery/release is not verified here. The
displayed control values are policy commands, not proof that the game accepted them. Supply a
saved run/checkpoint to drive the learned policy; omit it to try the clearly labelled,
untrained curvature-pilot baseline:

```bash
tmai play --checkpoint runs/<run> \
  --set track.path=data/tracks/my_map.json --set track.directory=null \
  --record-replay
# Baseline only (not a trained AI):
tmai play -c tmai/configs/default.yaml \
  --set track.path=data/tracks/my_map.json --set track.directory=null
```

The real driver cannot identify/switch the map for you: do not run it with a multi-map
library or a centreline for a different map. Real-game training and `tmai play` both refuse
multi-track libraries rather than silently selecting one. The default config uses only
`data/tracks/my_map.json`, which you must record from the map currently loaded in the game.
It does not claim held-out results. Multi-track splits and held-out evaluation currently
exercise the simulated pipeline, not automatic switching between real maps.

### 5. Inspect, compare, promote what you have

```bash
tmai list-tracks data/tracks             # geometry fingerprint + split per map
tmai status --run runs/<run>             # live status, metrics, evaluations
tmai compare runs/<run-a> runs/<run-b>   # side-by-side
tmai show-track --track data/tracks/my_map.json --out track.png
tmai replay list --run runs/<run>        # recorded episode replays
tmai analyze runs/<run> --track data/tracks/my_map.json --json-out sectors.json
                                          # sector pace + spatial failure heatmap
tmai models register --name v1 --checkpoint runs/<run>/checkpoint_*.pt
tmai benchmark --model v1=runs/<run-a> --model v2=runs/<run-b> --split validation,test
tmai benchmark --model pilot=baseline:curvature --model v1=runs/<run-a> --split validation
```

`baseline:curvature` adds an explainable, track-relative heuristic to the same benchmark protocol. It is an evaluation reference only—not a learned model and not validated on a live Windows Trackmania/TMInterface session. Benchmarks also include paired wins/losses/ties for the same track and episode index.

---

## Commands

| Command | Purpose |
|---|---|
| `tmai doctor` | Environment and game-integration health report; `--calibrate` measures telemetry conventions |
| `tmai validate-config` | Check a configuration without running anything |
| `tmai train` | Run training. `--resume`, `--steps`, `--seed`, `--device`, `--set key=value` |
| `tmai eval` | Evaluate a checkpoint across every track in a split (`--split a,b`): per-track finish rate, progress, lap times, crashes |
| `tmai play` | Directly drive the one map loaded in Trackmania with a checkpoint or an explicitly labelled curvature heuristic; bounded, interruptible, optionally records a replay |
| `tmai record-demo` | Record a human-driven lap as a behaviour-cloning demonstration (JSONL) |
| `tmai pretrain` | Behaviour-clone demonstrations into the policy; saves a resumable checkpoint |
| `tmai benchmark` | Evaluate several models across splits under one protocol and rank them |
| `tmai models` | Model registry: `list`, `register`, `info`, `delete`, `tag` |
| `tmai replay` | Replay tools: `list`, `show`, `compare` (AI replay vs human ghost) |
| `tmai analyze` | Sector pace, lateral-position summaries and spatial failure heatmaps from replays |
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

Training on one map and testing on that map measures memorisation, not driving. The
following multi-map workflow is implemented and tested with the **simulated driver**, not
with automatic real-game map switching. Its mechanisms make the distinction enforceable:

* **Deterministic, family-aware splits.** A track's split is a stable function of its identity
  (map UID, or a hash of its geometry), unless related maps share an explicit `metadata.family`
  label. Family members stay in one split, preventing near-variant layouts from leaking between
  train and test. `tmai record-track --family "author-pack"` adds the label at capture time.
* **Leakage prevention.** A track identity can only ever belong to one split; the same map
  recorded twice is *the same track* for this purpose, and conflicting family splits are
  rejected.
* **Multi-track training from step one.** A different track is sampled at every episode reset,
  and the policy never sees track identity or absolute world coordinates. A test translates a
  track by 750 m and asserts the observation is bit-for-bit unchanged.
* **Curriculum over the training split only.** Early training reveals the easiest tracks
  first and shortens episodes; held-out evaluation always runs the full suite, so a
  curriculum can never flatter the numbers it is measured by.
* **Training Director (opt-in).** `director.enabled: true` uses a failure-aware coverage EMA to
  prioritize weak training maps within bounded weights. Every active track keeps a guaranteed
  share, and held-out evaluation is never sampled or reweighted.

Held-out evaluation runs on a separate environment over the validation split and reports a
**generalisation gap** (train progress minus held-out progress). A positive gap is overfitting,
and it is visible in the metrics stream rather than hidden inside a blended average.

`tmai list-tracks` reports family labels and geometry coverage per split, which helps audit
that the held-out suite is both family-disjoint and representative of the training maps.
Benchmarks repeat paired evaluations across seeds and report approximate, cluster-aware 95%
bootstrap intervals; these are descriptive intervals, not significance tests.

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

## Checks

Python checks run headlessly without Trackmania or Windows:

```bash
pytest -q
ruff check tmai tests
mypy tmai
pytest --cov=tmai --cov-report=term-missing
```

The tests cover what can be validated without the game, including genuine TMInterface
`SimStateData` decoding, driver control logic against a scripted tick source, crash/contact
rules, temporal observation stacking, curriculum, behavior cloning, replay/ghost comparison,
model registry, benchmarking, resource monitoring, and the GUI backend over HTTP and WebSocket
(including a real training subprocess started through the API). The Python suite does not
validate timing or behavior against a running Trackmania instance.

Build and audit the GUI with the supported Node version:

```bash
cd gui
npm ci
npm audit
npm run build
```

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
