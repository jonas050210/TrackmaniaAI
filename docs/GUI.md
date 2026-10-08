# The GUI: local command center

`tmai serve` starts a **local hybrid application**: a small FastAPI backend that exposes
everything the CLI can do as JSON (plus a WebSocket for live updates), and a web frontend
(React + TypeScript + three.js) served as static files from `gui/dist`. It is a command
center for the whole system — training, monitoring, runs, evaluation, benchmarking, models,
tracks, replays/ghosts, configuration and diagnostics.

```bash
pip install -e ".[gui]"          # fastapi, uvicorn, websockets
cd gui && npm ci && npm run build
tmai serve                       # http://localhost:8765
```

The frontend uses Vite 8 and requires Node.js `^20.19.0` or `>=22.12.0` (the same range is
recorded in `gui/package.json`). Route pages and the three.js viewer are loaded on demand so
the initial dashboard bundle does not download the 3D renderer.

In development you can also run the Vite dev server (`npm run dev` in `gui/`), which proxies
`/api` (and the WebSocket) to `http://127.0.0.1:8765`.

## Design rules

* **One source of truth.** Every endpoint is a thin wrapper over the same modules the CLI
  uses (`tmai.api.status`, `tmai.registry`, `tmai.replay`, `tmai.training.*`). The GUI can
  never disagree with the command line, because there is no second implementation.
* **Long work is a job.** Training, evaluation, benchmarking, calibration and the doctor
  run as managed background jobs: a state machine (`queued → running → done | failed |
  cancelled`), a captured log, and a JSON result. The UI stays responsive, and a job's log
  survives a page reload. Jobs are persisted to `.tmai-server/jobs/jobs.jsonl`; a job that
  was running when the server restarted is reported as `interrupted`, not silently dropped.
* **Training runs as a subprocess.** A training job spawns `python -m tmai.cli train`, so a
  crash inside training cannot take the server (or the GUI) down with it.
* **Read-only by default.** GET endpoints only read. Mutations are limited to starting jobs,
  registering/deleting models, and validating config text.

## Security model (read this)

The backend has **no authentication and no authorisation**. It is a local tool: it can start
training runs, delete models and read every file the server process can read. It binds to
`0.0.0.0` by default so it is reachable from browsers on other machines (preview proxies,
VMs, a second monitor's browser), which means:

> **Do not run `tmai serve` on an untrusted network.** If you need it on a shared machine,
> bind it to the loopback interface: `tmai serve --host 127.0.0.1`.

Path safety: run names, replay files and model names are validated against path traversal
before they are joined onto a directory.

## Pages

| Page | What it does |
|---|---|
| **Overview** | Headline stats (runs, models, tracks, best progress), live system resources, recent runs, quick actions. |
| **Runs** | Every run with step/progress/driver/status; click through to the run detail. |
| **Run detail** | Metric charts (reward, progress, learner losses, system resources, Training Director), per-track episodes with valid/invalid finish outcomes, evaluations, checkpoints, log tail. Auto-refreshes. |
| **Training** | Start a run from a config preset or a full YAML editor (validated before starting), opt in to the training-only Training Director or explicitly allow the toy simulated driver, use run options (name, steps, resume), follow logs, cancel, and browse job history. |
| **Evaluate & Benchmark** | Evaluate checkpoints/runs on chosen splits; benchmark models across paired seed repeats; inspect per-model and head-to-head bootstrap intervals for finish rate, progress, crash rate and win share; browse report history. |
| **Models** | The model registry: register a checkpoint as a named model, inspect metadata and tags, delete. |
| **Tracks** | Library table (family label, split, geometry fingerprint) + interactive 3D viewer: centreline, corridor ribbon, curvature colouring, orbit/zoom/pan. |
| **Replays & Ghosts** | Pick a run and episode; replay the trajectory in 3D with playback; inspect speed/reward profiles and sector/failure analytics; compare against a human ghost or another AI replay from the same track, including across runs. |
| **Configuration** | View and edit the default config YAML with live validation. |
| **Diagnostics** | Live resources, environment table, paths, doctor report, telemetry calibration (game host only). |

## The 3D track viewer

`gui/src/components/TrackViewer3D.tsx` renders, with three.js:

* the **corridor ribbon** (left/right edges from the track model),
* the **centreline**, coloured by curvature (blue = straight, amber = corners),
* **trajectories** (a replay, or a replay + ghost at once) as coloured lines,
* a **car marker** that follows the trajectory with a playback slider,
* a start marker, a scale grid, fog, and full orbit/zoom/pan.

Track geometry comes from `GET /api/tracks/geometry`, which decimates the centreline to at
most 1500 samples and includes the corridor edges, per-sample curvature and the geometry
fingerprint.

## API map

```
GET  /api/health                    liveness + version
GET  /api/system                    resources + capabilities + paths
POST /api/doctor                    environment report (job; ?calibrate=1)
GET  /api/runs                      run list (both ids: directory name + manifest run_name)
GET  /api/runs/{name}               full run snapshot (metrics, episodes, evals, log tail)
GET  /api/runs/{name}/metrics       downsampled curves (?metrics=a,b&max_points=)
GET  /api/runs/{name}/log           log tail (?lines=)
GET  /api/runs/{name}/checkpoints
GET  /api/runs/{name}/analysis       ?track=&sectors= → sector pace + failure heatmap
GET  /api/tracks                    library report (+ synthetic list)
GET  /api/tracks/geometry           ?name= or ?synthetic= → 3D payload
GET  /api/models                    registry list
GET  /api/models/{name}
POST /api/models/register           {name, checkpoint, tags, notes, overwrite}
DELETE /api/models/{name}
GET  /api/benchmarks                report list
POST /api/benchmark                 benchmark (job)
POST /api/eval                      evaluation (job)
POST /api/train                     training (job, subprocess)
POST /api/calibrate                 telemetry calibration (job, game host)
GET  /api/replays?run={name}        replay list
GET  /api/replays/{run}/{file}      one replay (full trajectory)
POST /api/replays/compare           {run, a, b, track?} → station/segment gaps
GET  /api/demos                     demonstration files
GET  /api/config                    default config (path + YAML + parsed)
POST /api/config/validate           {yaml} → {ok, problems}
GET  /api/jobs                      job list (?kind=)
GET  /api/jobs/{id}                 job detail (state + log tail + result)
POST /api/jobs/{id}/cancel
WS   /api/ws                        tick every 1.5 s: system + runs + jobs
```

Runs resolve by **directory name** (canonical) *or* by the manifest's `run_name`, so links
keep working however a run was named.

The ghost in `/api/replays/compare` can be either another replay file **or a human
demonstration** (JSONL from `tmai record-demo`); demonstrations are converted to ghost
replays on the fly. When no `track` is passed, the track is resolved from the replay's own
track name (recorded library first, then the synthetic suite) so the arc-length projection
works out of the box.

## Server options

```
tmai serve --host 0.0.0.0 --port 8765
           --runs-dir runs --tracks-dir data/tracks --models-dir models
           --demos-dir data/demos --benchmarks-dir benchmarks
           --static-dir gui/dist --config tmai/configs/default.yaml --no-open
```

State (jobs, job logs, evaluations) lives in `.tmai-server/` and is git-ignored.

## Frontend development

```
cd gui
npm run dev        # Vite dev server on :5173, proxies /api → 127.0.0.1:8765
npm run build      # tsc type-check + production build into gui/dist
npm run typecheck  # tsc only
```

The frontend has no runtime dependencies beyond React, react-router and three.js; charts are
hand-rolled SVG (no chart library), so the bundle stays small and the design system is fully
in our control (CSS custom properties, dark theme, consistent spacing/radius/transition
tokens).
