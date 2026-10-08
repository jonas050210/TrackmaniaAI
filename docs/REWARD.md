# Reward design

The objective is **fast, accurate driving**. This document records the reasoning behind the
current design, the specific loopholes it closes, and how to tune it. The implementation is
`tmai/env/reward.py`.

---

## 1. Why dense progress is the primary term

Every control step has a fixed duration, so metres of centreline progress per second and lap
time are the same objective up to a constant:

```
lap_time = track_length / mean_progress_rate
```

Maximising progress rate therefore minimises lap time. Progress is also *dense* — available at
every step — whereas a finish bonus cannot guide an agent that has never completed a lap. That
is why progress carries the dominant weight and the finish bonus is a small one-off.

---

## 2. Why every shaping term is a rate, integrated by `dt`

This was a real defect, not a design preference. The control period is configurable
(`env.control_dt × env.action_repeat`), and the game-speed multiplier changes how much
simulation time a step covers. If penalties are charged **per step**, then:

* Doubling the step rate doubles the penalty an identical lap receives.
* The same driving scores differently under different control rates.
* The effective discount horizon silently changes, so `gamma` means something different.

So every shaping term is expressed **per second** and multiplied by `dt`. Total return over a
lap is then invariant to the control rate, which is what makes results comparable across
configurations. There is a test for exactly this
(`test_total_return_is_invariant_to_control_rate`).

Progress is the deliberate exception: metres covered is already time-integrated, so a lap is
worth its own length regardless of how many steps it took.

---

## 3. Loopholes, and how each is closed

A reward function fails not through a wrong weight but through an exploit that scores well
numerically and drives badly. Each of these was identified and closed explicitly.

### 3.1 Cuts and teleports

**Exploit:** drive straight across the map, rejoin far ahead, collect a huge progress reward.

**Closure:** credited progress is capped by what is *physically achievable* in one step —
`max_speed_for_progress × dt × cut_margin` — rather than by a fixed metre count.

A fixed cap is a trap, and the original implementation fell into it. With
`max_progress_per_step = 6.0` and a 0.1 s control period, legitimate driving above **60 m/s**
(216 km/h) was clamped. The reward was punishing exactly the speed it was meant to encourage.
A speed-derived threshold detects a genuine cut (a teleport-scale jump) while never binding on
fast but honest driving — verified at three different control periods by
`test_fast_but_honest_driving_is_never_clamped`.

`max_speed_for_progress` should be set comfortably above the car's true top speed.

### 3.2 Oscillation

**Exploit:** shuttle back and forth across a station, farming progress reward each pass.

**Closure:** backward progress is credited *negatively*, bounded by `max_backward_speed`.
Net progress over an oscillation is therefore zero or worse, never positive.

### 3.3 Standing still

**Exploit — and the subtlest one:** do nothing.

With no idle penalty a stationary car scores exactly `0.0` per step, while every attempt to
drive early in training scores *negative* (off-track, heading error, slip). The optimal early
policy is therefore to sit still, and learning never starts. This is a genuine local optimum,
not a tuning inconvenience.

**Closure:** `idle_penalty` charges a rate while speed is below `idle_speed_threshold`. It is
targeted at idling specifically rather than a blanket `step_penalty`, so genuinely slow
cornering is not punished. It also does not fire once the car has finished — the car is
legitimately stationary then.

### 3.4 Off-track shortcuts

**Exploit:** cut across grass or through a gap, saving distance.

**Closure:** corridor excursion is charged per metre **per second**, so a shortcut costs more
the longer it lasts. The corridor width is looked up by interpolated arc length, matching what
the observation reports for edge distances — indexing by sample instead made a car near a
segment boundary off-track for the reward but on-track for termination.

### 3.5 Invalid finishes

**Exploit:** reach the finish line without collecting the map's checkpoints (a cut, or a bug).

**Closure:** the game's own checkpoint counter is authoritative. An episode that reports
`finished` with fewer checkpoints than the map defines is flagged `invalid_finish`, receives
no finish bonus during training, and is excluded from lap-time statistics. The training event
and any saved replay preserve the raw game flag and checkpoint counts for audit; it never counts
as a valid lap.

### 3.6 Going fast in the wrong direction

**Exploit:** maximise the speed term while pointing at a wall.

**Closure:** the speed weight is deliberately small (`speed_weight = 1.0` per second at
`speed_ref`, i.e. ~0.05 per step). Progress already encodes speed *along the line*, so a large
independent speed term would reward velocity without direction. Keep this term small.

---

## 4. Term reference

| Term | Default | Units | Effect |
|---|---|---|---|
| `progress_weight` | 1.0 | per metre | The dominant term. Must be positive. |
| `speed_weight` | 1.0 | per second | Normalised forward speed. Keep small. |
| `off_track_weight` | 6.0 | per metre·s | Corridor excursion. |
| `heading_weight` | 0.4 | per rad·s | Heading error vs the centreline tangent. |
| `slip_weight` | 0.4 | per second | Normalised lateral speed. |
| `idle_penalty` | 0.5 | per second | While below `idle_speed_threshold`. Removes the do-nothing optimum. |
| `slide_penalty` | 0.0 | per second | Flat while the car reports sliding. |
| `step_penalty` | 0.0 | per second | Blanket. Prefer `idle_penalty`. |
| `finish_bonus` | 20.0 | one-off | Crossing the finish line. |

Anti-exploit limits:

| Limit | Default | Meaning |
|---|---|---|
| `max_speed_for_progress` | 95.0 m/s | Fastest legitimate travel. Sets the cut threshold. |
| `cut_margin` | 1.15 | Tolerance for telemetry jitter. **Must be ≥ 1.0.** |
| `max_backward_speed` | 30.0 m/s | Reverse-progress credit bound. |
| `off_track_margin` | 0.0 m | Excursion tolerated before the penalty starts. |

Every component is returned in `RewardBreakdown` and logged as `reward/*`, so a long run can be
diagnosed from its metrics alone. `reward/clamped` in particular is worth watching: a
non-trivial rate means either the agent is cutting or `max_speed_for_progress` is too low.

`RewardConfig.validate()` rejects degenerate settings at startup — a zero progress weight, a
`cut_margin` below 1.0, negative penalties — rather than producing silently wrong training
signal three hours into a run.

---

## 5. Tuning guidance

Tune in this order, and change one thing at a time:

1. **Confirm progress dominates.** `reward/progress` should be the largest positive component
   on a good lap. If `reward/speed` is comparable, lower `speed_weight`.
2. **Watch `reward/clamped`.** Persistent clamping means a cut or a mis-set
   `max_speed_for_progress`. Never "fix" clamping by raising the cap without checking which.
3. **Check the agent is not idling.** If episodes end `stalled` with near-zero
   `reward/idle`, raise `idle_penalty`.
4. **Tighten the corridor slowly.** Raise `off_track_weight` only once the agent reliably
   completes laps, or it will spend its early training off-track and learning nothing.
5. **Decide about drifting last.** `slip_weight` and `slide_penalty` both discourage sliding.
   Trackmania rewards controlled drifts, so leaving both at their defaults is a real choice,
   not a neutral one. Set them to 0 to permit drifting.

The weights shipped in `tmai/configs/*.yaml` are reasoned defaults that have **never been
tuned against real Trackmania physics**. Expect to revise them on hardware.

---

## 6. What this reward does not yet do

* **No obstacle or wall awareness.** The corridor is a constant half-width around the
  centreline; block-level geometry would make `off_track_weight` far more meaningful.
* **No lookahead speed shaping.** A time-optimal driver brakes before a corner because it
  knows the corner is coming. The observation exposes upcoming curvature, so the policy can in
  principle learn this, but there is no explicit term rewarding it.
* **No lap-time-difference signal.** Racing against a ghost or a target lap time is not
  modelled.

These are in [`ROADMAP.md`](ROADMAP.md).
