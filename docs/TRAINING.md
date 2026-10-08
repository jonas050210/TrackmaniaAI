# Training

## Configuration

One `RunConfig` describes a whole run. Load it from YAML, override individual fields from the
command line, and the resolved result is written verbatim into the run manifest.

```bash
tmai validate-config -c tmai/configs/default.yaml      # check it before committing to a run
tmai train -c tmai/configs/default.yaml \
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
| `track` | `path` (one centreline), `directory` (a multi-track library), `synthetic`/`synthetic_suite`, plus `split_weights` and `explicit_splits`; related maps can share `metadata.family` |
| `multi` | Track sampling and start-condition randomisation. See [GENERALIZATION.md](GENERALIZATION.md) |
| `normalize` | Online observation normalisation: `enabled`, `clip`, `warmup_steps` |
| `env` | `control_dt`, `action_repeat`, observation spec, reward weights, termination thresholds |
| `sac` | `gamma`, `tau`, learning rates, temperature, network shape |
| `replay` | `capacity`, `seed` |
| `curriculum` | `enabled`, `stages` (track reveal + episode-length fraction per step range). Training split only |
| `bc` | `enabled`, `demo_paths`, `epochs`, `batch_size`, `lr`, `val_fraction`, `shuffle` — behaviour cloning from demonstrations |
| `train` | steps, warm-up, batch size, UTD ratio, intervals, held-out evaluation, seed, device, resume, `record_replays`/`replay_decimation`/`max_replays`, `model_store` |

Shipped configs (all pass `tmai validate-config`):

* `default.yaml` — real game (`driver.kind: tminterface`), one recorded centreline at
  `data/tracks/my_map.json`, held-out evaluation and start randomisation **off**. Record that
  map before training; validation checks the config but does not create its track file.
  The real driver cannot switch maps or reposition the car, so multi-map held-out results
  require the simulated pipeline until real-game map switching is implemented.
* `smoke.yaml` — the labelled toy model, one synthetic track, a few hundred steps. For
  validating plumbing and CI.
* `multitrack_smoke.yaml` — the toy model across four synthetic tracks and two splits, with
  held-out evaluation. Exercises the whole generalisation path in a few seconds.
* `pipeline_smoke.yaml` — the toy model with the full pipeline on: temporal observations
  (`history_length: 3`), curriculum (track reveal + episode caps), behavior cloning from a
  recorded demonstration, and replay recording. Generate its explicitly synthetic test demo
  first with `tmai record-demo -c tmai/configs/pipeline_smoke.yaml
  --allow-simulated-driver --max-steps 400 --out data/demos/smoke.jsonl`; it is not human
  driving data.

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
* **A different track every episode in simulation.** With `track.directory` or
  `track.synthetic_suite`, a new track is sampled at each reset by the simulated driver.
  The real driver rejects multi-map training because it cannot switch the game map.
  See [GENERALIZATION.md](GENERALIZATION.md).
* **Held-out evaluation cannot kill the run.** It executes on a separate environment and is
  wrapped, so a failure there is logged and training continues.
* **Curriculum advances with the step counter.** `curriculum.enabled` reveals tracks from
  easiest to hardest (by mean curvature + corner density) and can shorten early episodes;
  the trainer logs `curriculum_stage` events and a `curriculum/stage` metric. Held-out
  evaluation is **never** curriculum-filtered.
* **The Training Director can focus practice on weak maps.** Set `director.enabled: true` to
  update bounded per-track priorities from episode coverage and unfinished/failure outcomes.
  Every active training track keeps a guaranteed slot in each weighted sampling block; the
  director is attached only to the training environment, and its EMA state plus pending
  sampler schedule are stored in checkpoints for resume.
* **Behaviour cloning can warm-start the policy.** With `bc.enabled`, demonstrations are
  loaded and the actor is pretrained on them (logged as a `bc_pretrain` event) before RL
  begins. Skipped when resuming — the resumed checkpoint already carries its warm start.
* **Episodes can be recorded as replays.** `train.record_replays: true` writes a decimated
  trajectory per episode to `<run>/replays/` (capped by `max_replays`), ready for the Replays
  page and ghost comparison.
* **Resource usage is in the metrics stream.** `system/*` keys (load, memory, disk, CUDA when
  present) are logged alongside reward, so a dashboard can plot them without a second source.

## Resume

```bash
tmai train --resume runs/2026-10-07T14-02-11Z_sac-tminterface
tmai train --resume runs/.../checkpoint_000200000.pt
```

A checkpoint carries the network, both optimisers, the entropy temperature, the gradient-step
counter, RNG state (Python, NumPy, PyTorch and CUDA), the adaptive Training Director state and
multi-track sampling schedule when present, and the config. A resumed run continues the
counters rather than restarting them, and records `resumed_from` in its manifest.

The replay buffer is **not** restored: it would dominate checkpoint size for a 1 M-transition
buffer. A resumed run refills it during warm-up. This is a deliberate trade-off, and it means
**a resumed run does not reproduce the uninterrupted trajectory step for step.** It resumes the
*learning state* exactly; the data it learns from next is freshly collected. Both halves of
that sentence are pinned by `tests/test_reproducibility.py`.

## Reproducibility

`train.seed` makes a fresh run reproduce bit for bit. That was not true before it was measured:
two sources of entropy escaped the seed, and neither was visible from reading the code.

| Escape | Effect | Fix |
|---|---|---|
| Gymnasium spaces own a private generator that `np.random.seed` cannot reach | Warm-up actions (`env.action_space.sample()`) varied between runs | `seed_everything()` seeds the spaces explicitly |
| `ReplayBuffer` owns a `default_rng`, and every shipped config sets `replay.seed: null` | Minibatch sampling drew fresh OS entropy | `build_buffer()` derives a distinct stream from `train.seed`; an explicit `replay.seed` still wins |

So the invariant is: **same config, same seed, fresh run → identical metrics and identical final
weights.** A resumed run is reproducible *given the same checkpoint*, but is not expected to
match an uninterrupted run.

Note this is reproducibility of the training pipeline, not of the game. The real
Trackmania physics is not deterministic across processes, so a real-game run will not
reproduce exactly even with a fixed seed — the simulated driver will.

## Demonstrations and behaviour cloning

SAC has to discover "throttle drives the car" from reward alone. A human lap teaches it
directly, and the pipeline supports the whole loop. On the Windows game host, load the map,
then calibrate and record its centreline before using the default preset:

```bash
tmai doctor --calibrate
tmai record-track --out data/tracks/my_map.json --name my_map

# 1. drive one clean lap of the same map (the game reports YOUR inputs, not the AI's)
tmai record-demo -c tmai/configs/default.yaml --out data/demos/my_lap.jsonl

# 2a. pretrain a checkpoint from it (resumable like any other checkpoint)
tmai pretrain -c tmai/configs/default.yaml --demo data/demos/my_lap.jsonl --out models/pretrained.pt
tmai train -c tmai/configs/default.yaml --resume models/pretrained.pt

# 2b. ...or let a training run do it automatically
#     bc: {enabled: true, demo_paths: [data/demos/my_lap.jsonl], epochs: 20}
```

Demonstrations are JSON Lines — one transition per line: the observation, the
**game-reported** action (`input_steer`/`input_gas`/`input_brake` read back from the game, so a
human's real inputs are what get stored), position, speed, reward and race time. A torn
trailing line (Ctrl-C mid-recording) is skipped on load, so an interrupted recording still
leaves a usable dataset.

`pretrain_policy` fits the policy's deterministic action to the recorded actions (MSE in the
bounded action space). When the learner is normalisation-wrapped, statistics are fitted only
on the training partition and then applied to both training and validation data. Validation
holds out whole files when multiple demonstrations are supplied; a single file uses a
chronological holdout with a small purge gap to reduce leakage from adjacent frames. It is a
**warm start**, not an imitation objective: SAC takes over afterwards and can improve on the
demonstrations.

Devices and reproducibility: the pretraining runs on the learner's device (`train.device`,
CUDA when available), and the batch order is derived from `seed` alone, so the same seed gives
the same shuffle on CPU and GPU and is unaffected by other RNG draws. Inference
(`SACLearner.act`) moves inputs to the learner's device itself, so callers always pass host
arrays.

Two guards are deliberate: a demonstration whose observation/action dimensions do not match
the learner is rejected (a demo recorded against a different observation layout is a
configuration error, not data to truncate), and `record-demo` refuses the simulated driver
unless `--allow-simulated-driver` is passed, because the toy model just echoes the AI's own
outputs — a "human" demonstration recorded against it contains nothing a human did.

## Running the tests on a GPU

CI has no GPU, so the CUDA-only tests run on a self-hosted runner. To register one, install the
GitHub Actions runner on a machine with an NVIDIA GPU and CUDA-enabled PyTorch, and give it the
label `gpu`. Then start the **CI** workflow manually from the Actions tab (`workflow_dispatch`).
Locally, `pytest tests/test_bc.py tests/test_device.py -rs` shows which of those tests ran.

## Replays and ghosts

```bash
tmai replay list --run runs/<run>                       # every recorded episode
tmai replay show --run runs/<run> --replay episode_000042.json --track data/tracks/x.json --out lap.png
tmai replay compare --run runs/<run> --replay episode_000042.json \
        --other data/demos/my_lap.jsonl --track data/tracks/x.json --out gaps.json
```

A replay is a decimated trajectory (positions, speeds, actions, rewards, arc-length progress,
race times) plus the outcome. The trainer records them when `train.record_replays: true`.

Comparing an AI replay against a **ghost** (a human demonstration, or any other replay)
answers "where is the time lost?" over a grid of arc-length stations. Two gap series are
reported: the *station* gap (clock difference at each station — the racing-ghost view, only
meaningful when both replays start together) and the *segment* gap (time per station pair —
invariant to random start stations, so it is the honest answer for a partial lap). Only the
arc length both replays actually drove is compared, so an early crash shortens the comparison
instead of faking gaps over the rest of the track.

Sector pace and failure locations can be summarized from saved replays against a matching
centreline:

```bash
tmai analyze runs/<run> --track data/tracks/my_map.json --sectors 20 --json-out analysis.json
```

`analyze` reports observed speed and lateral offset per equal-distance sector, interpolated
sector times where consecutive boundary crossings are available, end-reason counts, and a
station-by-lateral failure grid. The GUI exposes the same summary on Replays & Ghosts. Treat
slow sectors and lateral excursions as review cues; they do not prove a particular line is
suboptimal without a valid ghost/baseline and calibrated real telemetry.

## The model registry

```bash
tmai models register --name v1 --checkpoint runs/<run>/checkpoint_00200000.pt --tag baseline
tmai models list
tmai models info --name v1
tmai models tag --name v1 --tag promoted
tmai models delete --name v1
```

A registered model is **self-contained**: the registry copies the checkpoint into
`models/<name>/` (`model.json` metadata + `policy.pt` weights), so deleting or renaming the
run that produced it never breaks the model. Benchmarks and the GUI's Models page bind to the
same store.

## Benchmarking

```bash
tmai benchmark --model v1=runs/<run-a> --model v2=runs/<run-b> \
        --split validation,test --episodes 3 --seed-repeats 3 \
        --name my-benchmark --out benchmarks/my-benchmark.json
```

Every model is evaluated on every split under one identical protocol — same tracks, same
seed schedule, same episode count and same determinism — so differences are the models, not the
measurement. `--seed-repeats 3` is paired across models and multiplies the episode count per
track; use `--seed-repeats 1` for a quick smoke check. Reports include the exact seeds,
approximate 95% bootstrap intervals for finish/progress/crash metrics and paired win-share /
progress-delta intervals. The bootstrap resamples explicit family clusters together, or
individual tracks when no family metadata is supplied, rather than treating every episode on a
map as independent. A single cluster falls back to episode-level intervals; one sample has no
interval.
These are descriptive percentile intervals, not formal significance tests, and the ranking
still sorts point scores. Random seeds only produce different starts if the driver supports
start randomisation; this is not available for the real game. Add `--model pilot=baseline:curvature` to include the
geometry-grounded heuristic controller. That baseline is not a learned policy and has not been
validated in a live Windows Trackmania/TMInterface session; simulated runs only test the
pipeline. Reports are JSON in `benchmarks/`; the GUI lists the intervals and can start one as a job.

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

# the generalisation gap: train progress minus held-out progress (simulated multi-map runs)
jq -c 'select(."eval/generalization_gap" != null) | {step, gap: ."eval/generalization_gap"}' runs/<run>/metrics.jsonl

# which track each episode ran on
jq -c 'select(.event=="episode_end") | {track, end_reason, progress}' runs/<run>/events.jsonl
```

`tmai status --run runs/<run>` renders all of this as a table, and `--json` emits the full
dashboard snapshot (status, downsampled curves, episodes, evaluations, checkpoints, manifest,
log tail) that a GUI can bind to.

The metrics stream also carries `system/*` keys (CPU load, memory, disk, CUDA when present),
logged every `train.log_interval` steps, and `curriculum/stage` when a curriculum is active.

The same data is what `tmai serve` exposes to the GUI — see [GUI.md](GUI.md).

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
map's checkpoints is flagged `invalid_finish`, earns no finish bonus during training, and is
excluded from lap-time statistics. Events and saved replays preserve the raw finish flag and
checkpoint counts; evaluation also emits a warning.

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
record-track` and `tmai record-demo` refuse it outright, because recording a "map" or a
"human lap" from the toy model would be actively misleading.
