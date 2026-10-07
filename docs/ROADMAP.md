# Roadmap

Ordered by what unblocks the most. Items 1–3 are verification, not features: until they are
done, nothing here has been shown to control Trackmania.

## 1. Verify the real integration (blocking)

On a Windows host with Trackmania + TMInterface:

- [ ] `tmai doctor` reports a connected driver and a non-zero checkpoint count.
- [ ] `tmai doctor --calibrate` passes all four checks; apply the measured `position_scale`.
- [ ] Confirm the forward axis. `VehicleState.forward_vector` currently reads rotation
      column 0. If calibration says otherwise, make the axis configurable rather than
      hard-coded, and cover it with a test.
- [ ] Drive the car manually and watch `tmai`'s telemetry agree with what the car is doing.
- [ ] Confirm `iface.respawn()` restarts an episode usefully. If it does not, switch to
      `ResetStrategy.COMMAND` with the right console command.

Everything else on this list is premature until this is done.

## 2. First real training run (blocking)

- [ ] Record a centreline with `tmai record-track` and inspect it with `tmai show-track`.
- [ ] A few thousand steps to confirm the loop runs against the real game and
      `reward/progress` increases.
- [ ] Measure achievable `env_steps_per_second` at several `driver.speed_ratio` values and
      pick the highest one that does not degrade control.
- [ ] Only then, a long run.

## 3. Reward tuning against real physics (blocking)

The shipped weights are reasoned defaults, never tuned. Expect to tune:
`off_track_weight`, `max_progress_per_step`, `heading_weight`, and the corridor half-width.
Keep `slip_weight`/`slide_penalty` at 0 until you have decided whether drifting is wanted.

## 4. Block-level track geometry

The centreline is the only map knowledge today; the drivable corridor is one constant
half-width. This caps what the agent can learn.

- [ ] Parse `.Map.Gbx` (blocks, positions, orientations) — the `gbx` package on PyPI is a
      starting point.
- [ ] Derive per-point corridor width from block geometry, replacing the constant.
- [ ] Expose obstacle/wall distances to the observation behind the existing
      `TrackGeometry`-style interface, so nothing above `tmai.tracks` changes.
- [ ] Cross-check the parsed geometry against a recorded centreline as a validation step.

## 5. Generalisation to unseen maps

Currently a policy is trained per map. To generalise:

- [ ] Record a set of maps and split them into train / held-out.
- [ ] Verify the observation is genuinely map-agnostic: no absolute coordinates, no
      map-specific scaling. (`ObservationSpec` is designed for this; it has not been tested
      across maps.)
- [ ] Train across the set with map sampling at reset.
- [ ] Report held-out map performance as the headline metric, not training-map performance.

## 6. Throughput: parallel rollout

One game instance caps everything.

- [ ] Vectorised environment wrapper over N drivers (N game instances, or N hosts).
- [ ] Learner service separate from rollout workers, so a GPU box trains while Windows boxes
      collect. `GameDriver` is already per-instance; the missing pieces are the vector env and
      the transport.
- [ ] Off-policy algorithms tolerate stale policy weights well, which is what makes this
      architecture viable for SAC.

## 7. GUI and human-vs-AI evaluation

Explicitly out of scope for the foundation phase; the data foundations exist:

- [ ] Switchable simplified 3D view showing track/block structure. `tmai.viz` already renders
      the centreline, corridor, curvature and car state to PNG and exports `.obj`; the GUI is a
      viewer over the same geometry plus block meshes from item 4.
- [ ] Live overlay of the agent's observation and reward during a run.
- [ ] Human-vs-AI: a real-time pacing wrapper on top of `GameDriver` (the environment is
      deliberately lock-step with the game, not the wall clock, so this is additive), plus
      ghost comparison against a recorded human lap.

## 8. Algorithm work

Only worth doing once the baseline is validated on the real game.

- [ ] REDQ / DroQ to raise the update-to-data ratio without divergence — the natural next step
      when samples are the bottleneck.
- [ ] Running observation normalisation if the real game's ranges differ much from the assumed
      `ObservationScales`.
- [ ] Recurrent policy when moving to vision (partial observability).
- [ ] PPO behind the `Learner` protocol, for comparison only.
- [ ] Vision observations: image term + CNN encoder; `GameDriver` would need a screenshot
      method.

## Deliberately not planned

- Automated map discovery or procedural map generation.
- Online/multiplayer play of any kind — the integration is for local, single-player training.
- Anything that bypasses DRM or communicates with Nadeo/Ubisoft services.
