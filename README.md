# TrackmaniaAI

Reinforcement learning that drives the **real** Trackmania game. The agent starts with no
driving knowledge and learns steering, throttle, braking and drifting from a reward based on
track progress.

This repository is the **foundation phase**: the architecture, the real game integration, the
RL environment, the training pipeline and the test suite. The GUI is deliberately not built
yet.

---

## Status: what actually works today

| | |
|---|---|
| Real Trackmania integration | Implemented against TMInterface's documented API. **Not yet verified on a live game** — it needs a Windows host with Trackmania, which this repository's CI does not have. See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md). |
| RL environment, reward, termination | Implemented and unit-tested (308 tests). |
| SAC learner | Implemented from scratch, tested, including a Bellman fixed-point check. |
| Training loop, checkpointing, resume, logging | Implemented and tested end to end. |
| Track model + recording | Implemented; recorded from real telemetry with `tmai record-track`. |
| Simplified 3D track view | Implemented (headless PNG + `.obj` export). Full block-level structure pending. |
| GUI / human-vs-AI evaluation | Not built (out of scope for this phase). |

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
them, so swapping in PPO, TD3 or REDQ later is additive.

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

The game host and the training host are expected to be different machines; nothing requires
both on one box.

---

## Quick start

### 1. Check the environment

```bash
tmai doctor
```

Reports platform, dependencies, CUDA, whether TMInterface is reachable, and how many
checkpoints the loaded map has. With `--calibrate` it *measures* the game's telemetry
conventions instead of assuming them (see below).

### 2. Prove the pipeline runs (no game needed)

This uses the labelled toy model — **not** Trackmania — and exists only to validate plumbing:

```bash
tmai train -c tmai/configs/smoke.yaml
tmai eval  -c tmai/configs/smoke.yaml
```

The trainer refuses `driver.kind: simulated` unless you pass `--allow-simulated-driver`.

### 3. Drive the real game

On the Windows host running Trackmania through TMInterface:

```bash
tmai doctor --calibrate                  # verify + measure the integration
tmai record-track --out data/tracks/my_map.json --name my_map   # drive one clean lap
tmai train -c tmai/configs/default.yaml --set track.path=data/tracks/my_map.json
```

### 4. Look at the track

```bash
tmai show-track  --track data/tracks/my_map.json --out track.png
tmai export-obj  --track data/tracks/my_map.json --out track.obj
```

---

## Commands

| Command | Purpose |
|---|---|
| `tmai doctor` | Environment and game-integration health report; `--calibrate` measures telemetry conventions |
| `tmai train` | Run training. `--resume`, `--steps`, `--seed`, `--device`, `--set key=value` |
| `tmai eval` | Evaluate a checkpoint: finish rate, progress, lap times |
| `tmai record-track` | Record a map centreline by driving it once |
| `tmai show-track` | Render the simplified 3D track view to PNG |
| `tmai export-obj` | Export the track mesh for an external viewer |

Every command accepts `-c config.yaml` and repeatable `--set section.field=value` overrides.

---

## Run output

Each run writes a self-describing directory:

```
runs/2026-10-07T14-02-11Z_sac-tminterface/
    manifest.json     config + git sha + versions + driver/env description + seed
    config.yaml       resolved config, verbatim
    metrics.jsonl     one JSON record per logged step (append-only, crash-safe)
    events.jsonl      episodes, evaluations, errors, checkpoints
    run.log           human-readable log
    checkpoint_*.pt   weights + optimisers + RNG state (atomic writes, pruned)
    best.pt           best-scoring checkpoint
```

A run directory alone is enough to know exactly what was run.

---

## Tests

```bash
pytest                 # 308 tests, no game, no Windows, no display
ruff check tmai tests
```

The suite deliberately covers what can be covered without the game, including the real
TMInterface memory layout (tests decode genuine `SimStateData` structs) and the driver's
control logic (tested against a scripted tick source that honours the real contract).

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Layering, data flow, why the boundaries are where they are |
| [`docs/INTEGRATION.md`](docs/INTEGRATION.md) | How real Trackmania communication works, protocol-level |
| [`docs/ALGORITHM.md`](docs/ALGORITHM.md) | Algorithm selection: requirements → SAC, and why not PPO |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Configuration, reward tuning, long runs, resume |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | **What is verified and what is not.** Read this. |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | What is left, in priority order |

## Licence

MIT. Trackmania is a trademark of Ubisoft/Nadeo; this project is not affiliated with or
endorsed by them. See [`docs/INTEGRATION.md`](docs/INTEGRATION.md) for the third-party tooling
this project depends on and the terms that apply to it.
