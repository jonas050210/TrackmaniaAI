# Training

## Configuration

One `RunConfig` describes a whole run. Load it from YAML, override individual fields from the
command line, and the resolved result is written verbatim into the run manifest.

```bash
tmai validate-config -c tmai/configs/default.yaml      # check it before committing to a run
tmai train -c tmai/configs/default.yaml \
    --set track.directory=data/tracks \
    --set driver.speed_ratio=8 \
    --set train.updates_per_step=2 \
    --set sac.gamma=0.98
```

`validate-config` is worth running first: it catches conflicting track sources, warm-up longer
than the run, a replay buffer smaller than the batch, and the real-game / start-randomisation
conflict, all of which would otherwise cost hours.

`--set` coerces to the declared field type, so `--set train.total_steps=5000` yields an `int`
and `--set sac.learnable_temperature=false` yields a `bool`. An unknown path raises rather than
being silently ignored.

| Section | Contents |
|---|---|
| `driver` | `kind`, `server_name`, `position_scale`, `speed_ratio`, reset strategy, timeouts |
| `track` | `path` (one centreline), `directory` (a multi-track library), `synthetic`/`synthetic_suite`, plus `split_weights` and `explicit_splits` |
| `multi` | Track sampling and start-condition randomisation. See [GENERALIZATION.md](GENERALIZATION.md) |
| `normalize` | Online observation normalisation: `enabled`, `clip`, `warmup_steps` |
| `env` | `control_dt`, `action_repeat`, observation spec, reward weights, termination thresholds |
| `sac` | `gamma`, `tau`, learning rates, temperature, network shape |
| `replay` | `capacity`, `seed` |
| `train` | steps, warm-up, batch size, UTD ratio, intervals, held-out evaluation, seed, device, resume |

Three shipped configs, all of which pass `tmai validate-config`:

* `default.yaml` — real game (`driver.kind: tminterface`), trains on every centreline in
  `data/tracks/` with a 70/15/15 split, held-out evaluation on, start randomisation **off**
  (the real game cannot reposition the car).
* `smoke.yaml` — the labelled toy model, one synthetic track, a few hundred steps. For
  validating plumbing and CI.
* `multitrack_smoke.yaml` — the toy model across four synthetic tracks and two splits, with
  held-out evaluation. Exercises the whole generalisation path in a few seconds.

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
* **A different track every episode.** With `track.directory` or `track.synthetic_suite`, a new
  track is sampled at each reset. See [GENERALIZATION.md](GENERALIZATION.md).
* **Held-out evaluation cannot kill the run.** It executes on a separate environment and is
  wrapped, so a failure there is logged and training continues.

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
| `max_speed_for_progress` | Sets the cut threshold (× `dt` × `cut_margin`). | Too low and it clamps legitimate fast driving; too high and cuts are credited. |
| `idle_penalty` | Per second below `idle_speed_threshold`. | Zero makes "stand still" a stable optimum, because early driving scores negative. |
| `heading_weight` | Per radian of heading error. | Too high → the agent follows the line timidly instead of taking the racing line. |
| `slip_weight` / `slide_penalty` | Penalise uncontrolled sliding. | Any nonzero value fights drifting. **Start at 0** if you want the agent to learn to drift. |
| `speed_weight` | Normalised forward speed. | Mostly redundant with progress. Keep small; a large value rewards going fast in the wrong direction. |
| `finish_bonus` | One-off terminal bonus. | Useful once the agent can nearly finish a lap; noise before that. |

Every component is logged separately (`reward/progress`, `reward/off_track`, …) along with
`reward/clamped` and `reward/progress_metres`. Diagnose from those before changing weights.

`reward/clamped` staying high means the agent is making progress jumps beyond what
`max_speed_for_progress × dt × cut_margin` allows — either it is cutting, or that limit is set
below the car's real top speed.

The full rationale for the reward design, including why every shaping term is charged per
*second* rather than per step, is in [REWARD.md](REWARD.md).

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

# evaluations, with the held-out number alongside the training number
jq -c 'select(.event=="held_out_evaluation") | {step, report: .report.mean_progress_fraction}' runs/<run>/events.jsonl

# the generalisation gap: train progress minus held-out progress
jq -c 'select(.event=="evaluation") | .report.generalization_gap' runs/<run>/events.jsonl

# which track each episode ran on
jq -c 'select(.event=="episode_end") | {track, end_reason, progress}' runs/<run>/events.jsonl
```

`tmai status --run runs/<run>` renders all of this as a table, and `--json` emits the full
dashboard snapshot (status, downsampled curves, episodes, evaluations, checkpoints, manifest,
log tail) that a GUI can bind to.

`metrics.jsonl` is append-only JSON Lines rather than a binary format so that it survives a
killed process, can be read line by line, and needs no special tooling.

## Evaluation

```bash
tmai eval --checkpoint runs/<run> --episodes 5          # human-readable table
tmai eval --checkpoint runs/<run> --json-out report.json  # machine-readable
tmai compare runs/<run-a> runs/<run-b>                   # side by side
```

Reports are **per track and per split**, never one blended number — a single average hides the
only result that matters for generalisation. Each track reports finish rate, mean and worst
progress, the spread across episodes, crash rate and best valid lap time.

| Metric | Meaning |
|---|---|
| `mean_progress_fraction` | Mean of *per-track* means, so every track counts equally |
| `crash_rate` | Fraction ending off-track, stalled or airborne |
| `consistency_std` | Mean within-track spread. High means unreliable, not slow |
| `worst_progress_fraction` | The bad case, which means hide behind an average |
| `generalization_gap` | Train − held-out progress. **Positive means overfitting.** `None` when either side is missing |

The policy runs deterministically (no exploration noise) and no gradient steps occur.

`score` is track fraction completed with a bonus for finishing quickly, bounded in `[0, 2)`,
so `best.pt` selection cannot be gamed by finishing slowly. A "finish" that did not collect the
map's checkpoints is flagged `invalid_finish`, excluded from lap-time statistics and logged as
a warning — the game's own checkpoint counter is authoritative.

During training, set `train.held_out_eval_interval` to evaluate on the validation split
periodically. It runs on a separate environment so the training episode and driver are
untouched, is wrapped so a failure cannot kill a long run, and logs under the `heldout/`
prefix.

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
