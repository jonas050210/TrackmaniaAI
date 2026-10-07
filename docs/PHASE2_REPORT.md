# Phase 2 report — generalisation, reward rework, evaluation, observability

**Branch** `arena/f325538e-trackmaniaai` · **commit** `02ce5d0` (pushed) · **date** 2026-10-07
**Gate** `pytest tests/ -q` → **543 passed** · `ruff check tmai tests` → **All checks passed!**
**Cumulative diff vs `main`** 78 files, +18 088 · source 9 795 lines (`tmai/`), 5 805 lines (`tests/`)

---

## 1. The honest headline

**Nothing in this repository has ever controlled Trackmania.** The sandbox is Linux with no game
and no Windows. The `tminterface` transport needs a Windows named shared-memory segment that a
running `TMInterface.exe` creates; it cannot be opened here.

What *has* been built and verified is everything around that transport: the environment, reward,
termination, observations, track representation, multi-track training, evaluation, checkpointing,
the run-status API and the CLI. Those run against a clearly-labelled kinematic stand-in, and are
covered by 543 tests.

Three verification tiers are used consistently across the code and docs:

| Tier | Meaning |
|---|---|
| **Verified** | Exercised end to end in this sandbox |
| **Offline-tested** | Logic unit-tested against the simulated driver or pure data |
| **Requires real Trackmania** | Cannot be exercised here; isolated behind a protocol seam |

---

## 2. What was implemented

### 2.1 Track representation and generalisation

The project no longer knows about one track. A run trains over a *library*.

| Component | File | What it does |
|---|---|---|
| `TrackStats` / `compute_stats()` | `tmai/tracks/stats.py` | Geometry fingerprint: tightest radius, corner count, straight fraction, span, corridor stats |
| `TrackLibrary` / `TrackEntry` / `TrackSampler` | `tmai/tracks/library.py` | Deterministic 70/15/15 splits, permutation-block sampling, manifests, directory loading |
| `MultiTrackEnv` / `EpisodeContext` | `tmai/env/multi_track.py` | Samples a fresh track per reset; rebinds encoder, reward, termination and driver |
| `RunningNormalizer` | `tmai/env/normalization.py` | Online mean/var normalisation with warm-up, clipping, checkpointing |
| `NormalizingLearner` | `tmai/agents/normalize.py` | Wraps any `Learner`; buffer keeps **raw** observations |

Split assignment is a sha256 of the track identity, so it is **stable across processes** — measured
70.3 / 15.05 / 14.65 % over 2000 identities. A cross-split identity collision raises rather than
silently leaking a test track into training.

**Anti-memorisation mechanisms, all measured:**

- Per-episode track sampling.
- Random start station across the full lap span, random lateral offset (σ 1.5 m, clipped inside
  the corridor with a 1.0 m margin).
- **No absolute coordinates in the observation.** Translating a track by (750, 0, −420) yields an
  observation max-abs difference of **exactly 0.0**; a second test asserts no observation feature
  is even *named* like a world coordinate.
- Held-out evaluation with an explicit `generalization_gap` metric (train − held-out progress).

### 2.2 Reward rework

`RewardConfig` / `ProgressReward` were rewritten (`tmai/env/reward.py`). The old design had four
defects, each confirmed by probing the old implementation before changing it:

| Defect | Consequence | Fix |
|---|---|---|
| `max_progress_per_step = 6.0` hard clamp | **Legitimate high-speed driving was penalised as "cutting"** at coarser control rates | Limits derived from speed: `max_speed_for_progress · dt · cut_margin` |
| `step_penalty = 0.0` with no idle term | Standing still scored 0.0 — a **real local optimum** | `idle_penalty` 0.5/s below 1.0 m/s, suppressed when finished |
| Per-step penalties | Total return depended on `control_dt · action_repeat`, so tuning control rate silently re-tuned the reward | **Every shaping term is a rate per second, integrated by `dt`** |
| Corridor width read per point | Wrong corridor at indices that aren't the nearest point | Read per **arc length**, in both `reward.py` and `termination.py` |

`test_total_return_is_invariant_to_control_rate` pins the third one at dt 0.05 / 0.1 / 0.2.
The cut detector is asserted never to bind on honest driving at 70 m/s.

Weights: progress 1.0, speed 1.0 (÷ 70 m/s), off-track 6.0/m/s, heading 0.4/rad/s, slip 0.4/s,
finish bonus 20, step penalty **0**, slide penalty **0**. Full rationale in `docs/REWARD.md`.

### 2.3 Evaluation

`tmai/training/evaluate.py` was rewritten (`EVAL_SCHEMA_VERSION = 2`). Reports are **per track and
per split**, never one blended average.

| Metric | Meaning |
|---|---|
| `mean_progress_fraction` | Mean of *per-track* means, so every map counts equally |
| `crash_rate` | Off-track, stalled or airborne |
| `consistency_std` | Mean within-track spread |
| `worst_progress_fraction` | The bad case that hides behind an average |
| `generalization_gap` | Train − held-out progress. Positive means overfitting. `None` when either side is missing |

`score = progress + 1/(1 + best_lap/60)` ∈ [0, 2) drives `best.pt`, so selection cannot be gamed by
finishing slowly. A finish that did not collect the map's checkpoints is flagged `invalid_finish`,
excluded from lap-time statistics and logged as a warning — the game's own counter is authoritative.

### 2.4 Visualisation foundation (data only, no GUI)

Per the standing instruction, **no GUI was built**. `tmai/api/status.py` is a read-only data layer
a future GUI binds to:

`run_status` · `run_history` · `run_episodes` · `run_evaluations` · `run_checkpoints` ·
`run_snapshot` · `list_runs`, plus `RunStatus`, `RunHistory`, `Series`, `EpisodeRecord`,
`CheckpointRecord`. `STATUS_SCHEMA_VERSION = 1`.

It reads only the artefacts a run already writes (`metrics.jsonl`, `events.jsonl`,
`checkpoints.json`, `manifest.json`, `training.log`). `run_snapshot` returns downsampled curves,
episodes, evaluations, checkpoints and the log tail in one JSON-serialisable object.

`tmai status --run <dir>` renders it as a table today; the LearningView-style GUI is deferred.

### 2.5 Code quality and CLI

`RunConfig.validate()` returns a problem list covering 15 failure classes (duplicate track source,
split weights not summing to 1, unknown driver, `simulated` without opt-in, non-positive
`speed_ratio`/`control_dt`/`batch_size`, `action_repeat < 1`, empty observation spec, random start
with `tminterface`, `warmup_steps ≥ total_steps`, `eval_interval > total_steps`,
`capacity < batch_size`, out-of-range `gamma`/`tau`). `train_from_config()` calls
`validate_or_raise()` first.

CLI: `doctor` · `train` · `eval` · `record-track` · `show-track` · `export-obj` plus new `status` ·
`list-tracks` · `validate-config` · `compare`. All three shipped configs pass `validate-config`.

---

### 2.6 Reproducibility — measured, and found broken

`train.seed` **did not reproduce a run.** This was discovered by running the same config twice
and diffing the metrics, not by reading code; nothing in the source looked wrong.

Two independent sources of entropy escaped the seed:

| Escape | Effect | Fix |
|---|---|---|
| Gymnasium spaces own a private generator that `np.random.seed` and `torch.manual_seed` cannot reach | Warm-up actions from `env.action_space.sample()` varied run to run | `seed_everything()` / `seed_spaces()` in `tmai/training/checkpoint.py` |
| `ReplayBuffer` owns a `default_rng(config.seed)`, and every shipped config sets `replay.seed: null` | Minibatch sampling drew fresh OS entropy | `build_buffer()` derives a distinct stream from `train.seed`; an explicit `replay.seed` still wins |

After both fixes, same config + same seed reproduces **identical metrics and bit-identical final
weights** (asserted in `tests/test_reproducibility.py`). A companion test asserts that different
seeds *differ*, so the reproducibility test cannot pass merely because seeding is ignored.

**Resume is deliberately not trajectory-identical.** A checkpoint stores the buffer *size* but
not its contents — saving a 1 M-transition buffer into every checkpoint of a multi-day run is not
a trade worth making. A resumed run therefore refills the buffer and diverges from the
uninterrupted run. The checkpoint docstring previously claimed resume was "exact"; that claim was
false and has been corrected, and a test now pins the limitation so it cannot silently regress.

Scope matters here: this is reproducibility of the *training pipeline*. Real Trackmania physics is
not deterministic across processes, so a real-game run will not reproduce exactly even with a
fixed seed. The simulated driver will.

---

## 3. How real Trackmania communication is designed to work

Everything game-facing lives behind `tmai/game/`. **Only that package knows Trackmania exists.**

```
TrackmaniaEnv  →  GameDriver (Protocol)  →  TMInterfaceDriver  →  TMInterfaceSession  →  TMInterface.exe
                     ↑                            ↑                      ↑
              tested logic               tested logic              untestable OS IPC
```

- **All game interaction happens inside `on_run_step`.** The env thread uses a newest-wins
  `Queue(maxsize=1)` plus an ops queue and a `Condition`. `set_timeout(-1)` prevents
  deregistration during slow inference.
- `tminterface.run_client()` must **not** be called from a worker thread (it installs signal
  handlers); `TMInterface.register(client)` is the correct entry point.
- Analog control uses `ANALOG_FULL_SCALE = 65536`; brake is sent binary.
- Input is applied at the **next physics tick** — one tick of documented actuation latency.
- `SimulatedGameDriver` implements the same protocol, so every layer above it is genuinely tested.

**Start-position randomisation is capability-gated.** `DriverCapabilities` gained
`supports_start_repositioning`. The real game cannot teleport the car to an arbitrary station
without recorded `CheckpointData`, so requesting it against `tminterface` raises
`UnsupportedFeatureError` rather than silently doing nothing. The simulated driver advertises and
implements it.

**`.Map.Gbx` was deliberately not parsed.** `pygbx` (the only relevant PyPI package) is GPL-3,
last updated 2021, sdist-only, and requires `python-lzo`, which cannot be built here. More
decisively, there is no real map file to verify against. `tmai/tracks/gbx.py` therefore raises
`GbxUnavailableError` and lists exactly what is needed, while shipping a fully-tested
`blocks_to_centerline()` for when a real parser exists.

---

## 4. Bugs the test suite found and fixed

These are real production bugs, not test defects:

| Bug | Impact |
|---|---|
| `CenterlineTrack.project()` arc-length/window index mismatch | Wrong projection near segment boundaries |
| **`CenterlineTrack.point_at()` advanced along unit tangents by a *segment fraction* instead of metres** | Start-position randomisation placed cars at the wrong arc length |
| Lateral sign used a right-handed cross product | Left/right corridor edges swapped (world frame is left-handed, +y up) |
| `corridor_half_width` indexed per point, not per arc length | Wrong corridor width in reward *and* termination |
| `_info()` leaked wall-clock time into the observation | Non-stationary observations |
| `TelemetryCalibrator.analyse()` never populated its recommendation fields | `tmai doctor --calibrate` printed empty advice |
| `TrackView._build_corridor()` swapped edges | Misleading visualisation |
| `cmd_eval` ignored the run's own `config.yaml` | Risked loading weights into a differently-shaped observation |
| `_run_dir_config()` skipped the parent for `.pt` paths | Checkpoint-path config discovery failed |
| Multi-track episode context lost after `reset()` | Episode metadata wrong from episode 2 onward |
| `log_event("evaluation", step=…, **report.as_dict())` | `got multiple values for keyword argument 'step'` |
| `runlog._jsonable()` stringified nested reports | Evaluation reports unreadable in `events.jsonl` |
| **`tmai/cli.py` used `Any` without importing it** | `NameError` on the entire `compare` path |
| `cmd_compare` printed the timestamped run directory | Ignored the manifest `run_name` |
| **`tmai eval` evaluated only the first train track** | Silently discarded the per-track comparison for multi-track runs |
| **`train.seed` did not reproduce a run** | Gymnasium space generators and `replay.seed: null` both escaped the seed, so two runs of one config diverged. Found by measurement, not by reading code. |
| Checkpoint docstring claimed resume was "exact" | The buffer contents are not stored, so a resumed run refills and diverges. Claim corrected; the limitation is now pinned by a test. |
| `tmai compare` always printed `--` in the `gap` column | It read `generalization_gap` off the *training* report, but training and held-out runs are logged as separate reports that each carry only their own split, so the value was always `None`. The one column that shows overfitting was silently blank. |

The last one is worth calling out: it was found by running the CLI end to end, not by a unit test,
which is why the final verification pass drives the real commands rather than only the suite.

---

## 5. Tests

```
pytest tests/ -q        →  543 passed in ~21s
ruff check tmai tests   →  All checks passed!
```

| File | Tests | Coverage |
|---|---|---|
| `test_generalization.py` | 59 | library, splits, sampler, normalisation, translation invariance, held-out |
| `test_env.py` | 53 | reward (rewritten), termination, observation spec |
| `test_tracks.py` | 53 | centreline, geometry, stats, GBX block conversion |
| `test_agents.py` | 52 | SAC, replay buffer, normalising wrapper |
| `test_infra.py` | 50 | checkpointing, runlog, config validation |
| `test_tracks_library.py` | 44 | library loading, manifests, split stability |
| `test_trainer.py` | 40 | training loop, held-out eval, resume |
| `test_api_status.py` | 38 | status API, snapshots, history |
| `test_tminterface_driver.py` | 31 | driver logic over a scripted tick source |
| `test_cli_new.py` | 31 | new CLI commands, end to end |
| `test_evaluate_gaps.py` | 23 | generalisation gap, sampled evaluation, off-track accounting |
| `test_telemetry.py` | 21 | real `SimStateData` struct decoding |
| `test_calibration.py` | 18 | wrong-convention detection |
| `test_reproducibility.py` | 17 | seeding, run reproduction, resume guarantees |
| `test_viz.py` | 13 | headless PNG and `.obj` export |
| **Total** | **543** | |

Phase 1 gate was 314 passed; phase 2 adds 229.

**Verified end to end in this sandbox:** `validate-config` → `train` (multi-track, held-out
evaluation, 800 steps, best_score 0.5724) → `status` → `eval --split train,validation` (4 tracks
across 2 splits) → `compare`.

---

## 6. Remaining blockers — all require a Windows PC with Trackmania

| # | Blocker | Why it cannot be resolved here |
|---|---|---|
| 1 | **`TMInterfaceSession` has never connected** | Needs Windows named shared memory + a running game |
| 2 | **Telemetry scaling is unverified** | `position_scale` converts undocumented game units to metres; only `calibration.py` can settle it |
| 3 | **Reward weights are untested against real physics** | The bicycle model cannot produce real slip, jumps or wall contact |
| 4 | **No real map data** | Blocks `.Map.Gbx` work and every real-track evaluation |
| 5 | **Start repositioning on the real game** | Needs recorded `CheckpointData`; currently raises by design |
| 6 | **Throughput ceiling unknown** | `speed_ratio` limits and subsystem skipping are host-dependent |

---

## 7. Recommended next step once the PC is available

**Do not start a long training run first.** Work through `docs/ROADMAP.md` in order:

1. **`tmai doctor --calibrate`** on a real map. This is the single highest-value action: it settles
   `position_scale`, confirms the telemetry rate, and populates the recommendation fields. Everything
   downstream depends on it.
2. **Record one real track** with `tmai record-track`, then `tmai show-track` and compare the derived
   centreline against a manual drive. This validates the track pipeline against reality.
3. **A short real training run** (`smoke.yaml` switched to `tminterface`) to confirm the reward
   produces sane driving before committing to a long one. Watch `reward/clamped` and
   `env/nonfinite_observations`.
4. Only then scale up: real maps in `data/tracks/`, held-out evaluation on, long run with
   `train.max_wall_seconds` bounding it.

**Technical risks, ranked:** telemetry unit scaling being wrong (silent, corrupts every
observation) → reward weights tuned on a toy model transferring badly → game-side throttling at high
`speed_ratio` → single-game-instance throughput capping sample rate.

---

## 8. What is *not* here, deliberately

- **No GUI.** Data layer only, per instruction. The GUI comes after the core architecture is proven
  on the real game.
- **No PPO.** SAC is the right default for continuous bounded control at low sample rates. The
  `Learner` protocol is proven extensible — `NormalizingLearner` composes additively without
  touching SAC — so PPO can be added later without rewriting the env.
- **No GBX parser.** See §3.
- **No fake integration.** No stub claims to control the game, and the simulated driver is refused
  unless `driver.allow_simulated` is set explicitly.
