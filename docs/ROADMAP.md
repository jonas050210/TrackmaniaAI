# Roadmap

Ordered by what unblocks the most. Items 1–3 are verification, not features: until they are
done, nothing here has been shown to control Trackmania.

---

## 1. Verify the real integration (blocking)

On a Windows host with Trackmania + TMInterface:

- [ ] `tmai doctor` reports a connected driver and a non-zero checkpoint count.
- [ ] `tmai doctor --calibrate` passes all four checks; apply the measured `position_scale`.
- [x] Make the rotation forward axis/sign configurable and propagate the calibrator's
      recommendation through telemetry, yaw and track-relative observations.
- [ ] Confirm the configured forward axis/sign on a real Trackmania session.
- [ ] Drive the car manually and watch `tmai`'s telemetry agree with what the car is doing.
- [x] Add a bounded `tmai play` path for a checkpoint or labelled baseline, with live progress,
      optional replay capture and shutdown cleanup (offline-tested only; live steering remains
      unverified here).
- [ ] Confirm `iface.respawn()` restarts an episode usefully. If it does not, switch to
      `ResetStrategy.COMMAND` with the right console command.
- [ ] Record a human lap with `tmai record-demo` and pretrain from it (`tmai pretrain`) — the
      demonstration path has only ever run against the simulated driver.

Everything else on this list is premature until this is done.

## 2. First real training run (blocking)

- [ ] Record a centreline with `tmai record-track`; inspect it with `tmai show-track`.
- [ ] `tmai validate-config` on the real-game config.
- [ ] A few thousand steps to confirm the loop runs against the real game and
      `reward/progress` increases.
- [ ] Measure achievable `env_steps_per_second` at several `driver.speed_ratio` values and
      pick the highest one that does not degrade control.
- [ ] Only then, a long run.

## 3. Reward tuning against real physics (blocking)

The shipped weights are reasoned defaults, never tuned. See
[`REWARD.md`](REWARD.md#5-tuning-guidance) for the order to change things in. Expect to tune
`off_track_weight`, `heading_weight`, `max_speed_for_progress` and the corridor half-width.
Decide explicitly whether drifting is wanted before touching `slip_weight`/`slide_penalty`.

## 4. Real maps, and more of them

The generalisation machinery is built and tested, but only against four synthetic centrelines.

- [ ] Record a set of real maps into `data/tracks/`.
- [ ] `tmai list-tracks data/tracks` — check the geometry coverage per split is comparable.
      A suite that trains on ovals and tests on a technical track measures a distribution
      shift, not generalisation.
- [ ] Pin a couple of maps to `test` via `track.explicit_splits` and never train on them.
- [ ] Report the held-out number as the headline metric, not training-map performance.

## 5. Block-level track geometry

The centreline is the only map knowledge today; the corridor is one constant half-width. This
caps what the agent can learn.

The data model and the geometry conversion already exist and are tested
(`tmai/tracks/gbx.py`: `MapBlock`, `MapGeometry`, `blocks_to_centerline`). What is missing is
the parser itself, which was deliberately not written because it could not be verified — see
[`LIMITATIONS.md`](LIMITATIONS.md#mapgbx-parsing-is-not-implemented-deliberately).

- [ ] Obtain a real `.Map.Gbx` (Trackmania 2020, Stadium).
- [ ] Verify a block extraction: name, position, orientation, size.
- [ ] Build a block-name → surface table marking drivable blocks.
- [ ] Cross-check the derived centreline against the same map driven manually.
- [ ] Derive per-point corridor width from block geometry, replacing the constant.
- [ ] Expose obstacle/wall distances to the observation behind the existing interface.

## 6. Start-position randomisation against the real game

Currently honest but unavailable: the real game respawns to the last checkpoint and cannot be
placed at an arbitrary station, so `supports_start_repositioning` is `False` and the
environment raises rather than pretending.

- [ ] Record per-map `CheckpointData` states and restore them with
      `TMInterface.set_checkpoint_state()`.
- [ ] Only then enable `multi.random_start_station` for real-game runs.

## 7. Throughput: parallel rollout

One game instance caps everything.

- [ ] Vectorised environment wrapper over N drivers (N game instances, or N hosts).
- [ ] Learner service separate from rollout workers, so a GPU box trains while Windows boxes
      collect. `GameDriver` is already per-instance; the missing pieces are the vector env and
      the transport.
- [ ] Off-policy algorithms tolerate stale policy weights well, which is what makes this
      architecture viable for SAC.

## 8. GUI and human-vs-AI evaluation — **done, with follow-ups**

The command center is built: `tmai serve` (FastAPI backend + WebSocket) with a React +
three.js frontend. See [`GUI.md`](GUI.md) for the full map.

- [x] Local backend exposing runs, metrics, tracks, models, replays, benchmarks,
      configuration and diagnostics as JSON, with jobs for long-running work.
- [x] Live dashboard over `tmai.api.status` with charts, hover states and smooth transitions.
- [x] Switchable 3D view showing the track: centreline, corridor ribbon, curvature
      colouring, trajectory playback with a car marker, checkpoints/start marker.
- [x] Human-vs-AI: ghost comparison of an AI replay against a recorded human lap
      (`tmai replay compare`, GUI Replays page), station-by-station and per-segment.
- [x] Sector pace/lateral-position summaries and station-by-lateral failure heatmaps from
      saved replays (`tmai analyze`, API and Replays GUI). These are descriptive and still
      need real-game replay data before they can reveal real racing weaknesses.
- [ ] Live overlay of the agent's observation and reward during a real run (needs the game).
- [ ] Block-level 3D meshes in the viewer (depends on item 5).
- [ ] Real-time pacing wrapper over `GameDriver` for a true race against the human.

## 9. Algorithm work

Only worth doing once the baseline is validated on the real game.

- [ ] REDQ / DroQ to raise the update-to-data ratio without divergence — the natural next step
      when samples are the bottleneck. The `Learner` protocol and the `NormalizingLearner`
      decorator show the seam works.
- [ ] Recurrent policy when moving to vision (partial observability).
- [ ] PPO behind the `Learner` protocol, for comparison only.
- [ ] Vision observations: image term + CNN encoder; `GameDriver` would need a screenshot
      method.

## Deliberately not planned

- Automated map discovery or procedural map generation.
- Online/multiplayer play of any kind — the integration is for local, single-player training.
- Anything that bypasses DRM or communicates with Nadeo/Ubisoft services.
