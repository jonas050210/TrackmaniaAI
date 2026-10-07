# Training

## Configuration

One `RunConfig` describes a whole run. Load it from YAML, override individual fields from the
command line, and the resolved result is written verbatim into the run manifest.

```bash
tmai train -c tmai/configs/default.yaml \
    --set track.path=data/tracks/my_map.json \
    --set driver.speed_ratio=8 \
    --set train.updates_per_step=2 \
    --set sac.gamma=0.98
```

`--set` coerces to the declared field type, so `--set train.total_steps=5000` yields an `int`
and `--set sac.learnable_temperature=false` yields a `bool`. An unknown path raises rather than
being silently ignored.

| Section | Contents |
|---|---|
| `driver` | `kind`, `server_name`, `position_scale`, `speed_ratio`, reset strategy, timeouts |
| `track` | `path` to a recorded centreline, or `synthetic` for a generated test track |
| `env` | `control_dt`, `action_repeat`, observation spec, reward weights, termination thresholds |
| `sac` | `gamma`, `tau`, learning rates, temperature, network shape |
| `replay` | `capacity`, `seed` |
| `train` | steps, warm-up, batch size, UTD ratio, intervals, seed, device, resume |

Two shipped configs:

* `default.yaml` — real game (`driver.kind: tminterface`), conservative settings for a long run.
* `smoke.yaml` — the labelled toy model, a few hundred steps, for validating plumbing and CI.

## The training loop

```
warm-up (random actions)  →  collect transition  →  store in replay buffer
                        →  updates_per_step gradient steps
                        →  log / evaluate / checkpoint on their intervals
```

Properties worth knowing:

* **Bootstrapping is correct.** `terminated` and `truncated` are tracked separately; only
  `terminated` stops bootstrapping, so time-limit truncations do not bias the value function.
* **Crash-resilient.** Metrics flush per record, checkpoints are written atomically (temp file
  + rename), and both `KeyboardInterrupt` and a lost game connection save a checkpoint before
  exiting. A multi-day run does not lose the last hours of work.
* **Bounded by wall clock.** `train.max_wall_seconds` lets CI and scheduled runs stop cleanly
  without a step-count hack.

## Resume

```bash
tmai train --resume runs/2026-10-07T14-02-11Z_sac-tminterface
tmai train --resume runs/.../checkpoint_000200000.pt
```

A checkpoint carries the network, both optimisers, the entropy temperature, the gradient-step
counter, RNG state (Python, NumPy, PyTorch and CUDA) and the config. A resumed run continues
the counters rather than restarting them, and records `resumed_from` in its manifest.

The replay buffer is **not** restored: it would dominate checkpoint size for a 1 M-transition
buffer. A resumed run refills it during warm-up. This is a deliberate trade-off.

## Reward tuning

Start from `default.yaml`. The terms, in the order they usually matter:

| Weight | Effect | Symptom if wrong |
|---|---|---|
| `progress_weight` | Metres of centreline progress. Dominant. | Too low relative to the penalties and the agent learns to be cautious. |
| `off_track_weight` | Per metre outside the corridor. | Too low → cuts across grass. Too high → the agent refuses to use the track's full width. |
| `max_progress_per_step` | Anti-cut clamp. | Too generous and cross-map shortcuts are rewarded. |
| `heading_weight` | Per radian of heading error. | Too high → the agent follows the line timidly instead of taking the racing line. |
| `slip_weight` / `slide_penalty` | Penalise uncontrolled sliding. | Any nonzero value fights drifting. **Start at 0** if you want the agent to learn to drift. |
| `speed_weight` | Normalised forward speed. | Mostly redundant with progress. Keep small; a large value rewards going fast in the wrong direction. |
| `finish_bonus` | One-off terminal bonus. | Useful once the agent can nearly finish a lap; noise before that. |

Every component is logged separately (`reward/progress`, `reward/off_track`, …) along with
`reward/clamped` and `reward/progress_metres`. Diagnose from those before changing weights.

`reward/clamped` staying high means the agent is making progress jumps larger than
`max_progress_per_step` — either it is cutting, or the clamp is too tight for the control step.

## Termination thresholds

Early termination exists to stop wasting the single game instance on dead time.

| Threshold | Meaning |
|---|---|
| `max_steps` | Hard truncation. |
| `off_track_limit` / `off_track_margin` | Consecutive steps beyond the corridor (plus margin) before giving up. |
| `stall_limit` / `min_progress_per_step` / `min_steps_before_stall` | Stuck detection, with a grace period for launch. |
| `no_ground_contact_limit` | Airborne/flipped detection. |

Every ending reports a machine-readable `end_reason` (`finished`, `off_track`, `stalled`,
`no_ground_contact`, `time_limit`), so a long run's failure modes are countable from
`events.jsonl`.

## Throughput

The game is the bottleneck. In order of leverage:

1. **`driver.speed_ratio`** — run the game faster than real time. The biggest lever. The
   capability advertises a recommended maximum of 20; upstream warns that very high factors can
   make the game skip subsystems including input processing. Watch `reward/clamped` and
   `env/nonfinite_observations` for signs of degradation.
2. **`train.updates_per_step`** — more gradient steps per expensive sample.
3. **`env.action_repeat`** — coarser control, fewer game round-trips per second of policy
   decisions.

`throughput/env_steps_per_second` is logged every `log_interval`. If it drops over a long run,
suspect the game host (thermal, background load) before suspecting the learner.

## Monitoring a run

```bash
# metrics are plain JSON Lines; any tool can read them
tail -f runs/<run>/metrics.jsonl | jq '{step, "reward/total", "env/progress_fraction"}'

# episode outcomes
jq -c 'select(.event=="episode_end") | {steps, end_reason, progress}' runs/<run>/events.jsonl

# evaluations
jq -c 'select(.event=="evaluation") | {step, finish_rate, mean_progress_fraction}' runs/<run>/events.jsonl
```

`metrics.jsonl` is append-only JSON Lines rather than a binary format so that it survives a
killed process, can be read line by line, and needs no special tooling.

## Evaluation

```bash
tmai eval --checkpoint runs/<run> --episodes 5
```

Reports finish rate, mean track fraction, best/mean lap time and a per-episode table. The
policy runs deterministically (no exploration noise) and no gradient steps occur.

`score` is track fraction completed with a bonus for finishing quickly, bounded in `[0, 2)`,
so `best.pt` selection cannot be gamed by finishing slowly. A "finish" that did not collect the
map's checkpoints is flagged `invalid_finish` and logged as a warning.

## Simplified track view

```bash
tmai show-track --track data/tracks/my_map.json --out track.png   # headless PNG
tmai export-obj --track data/tracks/my_map.json --out track.obj   # external viewer / future GUI
```

Shows the geometry the agent actually uses — centreline, drivable corridor, curvature, car
position and its lateral offset — rather than the game's rendering. Renders with matplotlib's
`Agg` backend, so it works on a training host with no display. Block-level structure needs the
map parser (see `docs/ROADMAP.md`).

## The simulated driver

`driver.kind: simulated` is a kinematic bicycle model, **not** Trackmania. It exists so the
pipeline can be developed and CI-tested without the game.

The trainer refuses it unless `driver.allow_simulated: true` / `--allow-simulated-driver`. Every
run records `driver: simulated` in its manifest and the CLI prints a banner. `tmai
record-track` refuses it outright, because recording a "map" from the toy model would be
actively misleading.
