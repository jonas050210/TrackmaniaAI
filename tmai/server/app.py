"""FastAPI backend for the TrackmaniaAI GUI.

Every endpoint is a thin wrapper over the same modules the CLI uses, so the GUI and the
command line can never disagree. Long-running work (training, evaluation, benchmarking,
calibration, doctor) runs as a managed background job (see :mod:`tmai.server.jobs`) so the
UI stays responsive and the log survives a page reload.

Route map::

    GET  /api/health                     liveness
    GET  /api/system                     resources + environment summary
    POST /api/doctor                     doctor report (job, optional --calibrate)
    GET  /api/runs                       run list
    GET  /api/runs/{name}                run snapshot (manifest + headline metrics)
    GET  /api/runs/{name}/metrics        downsampled metric curves
    GET  /api/runs/{name}/log            run log tail
    GET  /api/tracks                     track library report
    GET  /api/tracks/geometry            centreline + corridor + curvature payload
    GET  /api/models                     model registry
    GET  /api/models/{name}
    POST /api/models/register
    DELETE /api/models/{name}
    GET  /api/benchmarks
    POST /api/benchmark                  benchmark (job)
    POST /api/eval                       evaluation (job)
    POST /api/train                      training (job, subprocess)
    POST /api/calibrate                  telemetry calibration (job, game host only)
    GET  /api/replays?run={name}         replay list for a run
    GET  /api/replays/{run}/{file}       one replay (full trajectory)
    POST /api/replays/compare            AI replay vs ghost comparison
    GET  /api/demos                      demonstration files
    GET  /api/config                     default config (path + YAML + parsed)
    POST /api/config/validate            validate a config document
    GET  /api/jobs                       job list
    GET  /api/jobs/{id}                  job detail (state + log tail)
    POST /api/jobs/{id}/cancel
    WS   /api/ws                         live tick: system + runs + jobs

The built frontend (``gui/dist``) is served as static files at ``/``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from tmai import __version__
from tmai.api import status as status_api
from tmai.config import RunConfig
from tmai.monitoring import system_metrics
from tmai.server.jobs import Job, JobManager

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ServerConfig:
    """Where the server finds its data and what it serves."""

    #: No authentication: bind to loopback unless the operator explicitly chooses
    #: a trusted network interface (e.g. --host 0.0.0.0 behind an access-controlled proxy).
    host: str = "127.0.0.1"
    port: int = 8765
    runs_dir: str = "runs"
    tracks_dir: str = "data/tracks"
    models_dir: str = "models"
    demos_dir: str = "data/demos"
    benchmarks_dir: str = "benchmarks"
    static_dir: str = "gui/dist"
    config_path: str | None = None
    state_dir: str = ".tmai-server"

    def resolve(self, path: str | Path) -> Path:
        """Resolve a configured path against the project root (relative paths are cwd-based)."""
        path = Path(path)
        return path if path.is_absolute() else (PROJECT_ROOT / path)


@dataclass
class ServerState:
    """Runtime state shared by the routes."""

    config: ServerConfig
    jobs: JobManager
    started_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# -- helpers ----------------------------------------------------------------------------


def _safe_name(name: str, *, what: str = "name") -> str:
    """Reject path traversal in user-supplied names (run names, replay files, ...)."""
    if (
        not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or Path(name).name != name
        or name.startswith(".")
    ):
        raise HTTPException(status_code=400, detail=f"invalid {what}: {name!r}")
    return name


def _within(path: Path, base: Path, *, what: str) -> Path:
    """Resolve symlinks and reject paths outside an API-managed directory."""
    resolved = path.resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise HTTPException(status_code=400, detail=f"{what} must be inside {base}")
    return resolved


def _run_dir(state: ServerState, name: str) -> Path:
    """Resolve a run reference to its directory.

    The canonical id is the run *directory* name (``<timestamp>_<run_name>``), but operators
    think in run names, and ``--run-name`` can make the manifest label differ from the
    directory. Accept either: the directory name first, then a manifest ``run_name`` match.
    """
    base = state.config.resolve(state.config.runs_dir)
    path = base / _safe_name(name, what="run name")
    if path.is_dir():
        return _within(path, base, what="run")
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if not child.is_dir() or child.is_symlink():
                continue
            manifest = child / "manifest.json"
            if not manifest.is_file():
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if str(data.get("run_name", "")) == name:
                return child
    raise HTTPException(status_code=404, detail=f"no run named {name!r} in {base}")


def _track_payload(track) -> dict[str, Any]:
    """Centreline + corridor + curvature, decimated for the 3D view."""
    from tmai.tracks.stats import compute_stats

    stations = np.linspace(
        0.0,
        track.length,
        num=min(track.num_points, 1500),
        endpoint=not track.closed,
    )
    points = np.array([track.point_at(float(s)) for s in stations])
    headings = np.array([track.heading_at(float(s)) for s in stations])
    # Ground-plane left/right normals, matching TrackView._build_corridor: Trackmania's
    # world frame is left-handed with +y up, so the right-hand side of a heading is
    # cross(up, tangent).
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(np.broadcast_to(up, headings.shape), headings)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-9)
    widths = np.array([track.corridor_half_width_at(float(s)) for s in stations])
    curvature = np.array([track.curvature_at(float(s)) for s in stations])
    return {
        "name": track.name,
        "length": float(track.length),
        "closed": bool(track.closed),
        "num_points": int(track.num_points),
        "points": points.tolist(),
        "edges": {
            "left": (points - right * widths[:, None]).tolist(),
            "right": (points + right * widths[:, None]).tolist(),
        },
        "curvature": curvature.tolist(),
        "corridor_half_width": widths.tolist(),
        "stats": compute_stats(track).as_dict(),
    }


def _resolve_track(state: ServerState, name: str):
    """Resolve a track name to a CenterlineTrack: recorded file first, then synthetic.

    Lets endpoints that need geometry (replay comparison) work without the caller having
    to know where the track file lives.
    """
    from tmai.tracks.centerline import CenterlineTrack
    from tmai.tracks.synthetic import SYNTHETIC_TRACKS

    if not name:
        return None
    name = _safe_name(name, what="track name")
    base = state.config.resolve(state.config.tracks_dir)
    for candidate in (base / f"{name}.json", base / name):
        if candidate.is_file():
            return CenterlineTrack.load(_within(candidate, base, what="track"))
    # Files can be named independently of the embedded track name. Resolve the public name
    # from metadata as a fallback so replay analysis and the viewer agree with the library.
    if base.is_dir():
        for candidate in sorted(base.rglob("*.json")):
            try:
                track = CenterlineTrack.load(_within(candidate, base, what="track"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if track.name == name or track.uid == name:
                return track
    if name in SYNTHETIC_TRACKS:
        return SYNTHETIC_TRACKS[name]()
    return None


def _runs_light(state: ServerState) -> list[dict[str, Any]]:
    """Cheap run list for the WebSocket tick (no metric parsing)."""
    base = state.config.resolve(state.config.runs_dir)
    out: list[dict[str, Any]] = []
    if not base.is_dir():
        return out
    now = time.time()
    for child in sorted(base.iterdir(), reverse=True):
        if not child.is_dir() or child.is_symlink():
            continue
        metrics = child / "metrics.jsonl"
        updated = metrics.stat().st_mtime if metrics.is_file() else child.stat().st_mtime
        manifest = child / "manifest.json"
        step = 0
        if manifest.is_file():
            try:
                step = int(json.loads(manifest.read_text(encoding="utf-8")).get("step", 0))
            except (ValueError, json.JSONDecodeError):
                step = 0
        if not step and metrics.is_file():
            try:
                last = metrics.read_text(encoding="utf-8").strip().splitlines()[-1]
                step = int(json.loads(last).get("step", 0))
            except (ValueError, json.JSONDecodeError, IndexError):
                step = 0
        out.append(
            {
                "name": child.name,
                "step": step,
                "updated_utc": datetime.fromtimestamp(updated, timezone.utc).isoformat(),
                "running": (now - updated) < 90.0,
            }
        )
    return out[:50]


# -- job callables -----------------------------------------------------------------------


def _check_job_cancelled(state: ServerState, job: Job) -> None:
    if state.jobs.is_cancelled(job):
        raise RuntimeError("job cancelled by operator")


def _subprocess_job(
    state: ServerState,
    job: Job,
    argv: list[str],
    *,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run a CLI subprocess, streaming its output into the job log."""
    job.log(f"$ {' '.join(argv)}")
    log_path = state.config.resolve(state.config.state_dir) / "jobs" / f"{job.id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = None
    if extra_env:
        import os

        env = {**os.environ, **extra_env}
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            argv,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        def pump() -> None:
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    job.log(line.rstrip("\n"))
                    log_file.write(line)
                    log_file.flush()
            finally:
                # The reader owns the pipe and releases it when it finishes. Without this the
                # descriptor leaked once per job on a long-running server.
                proc.stdout.close()

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            while proc.poll() is None:
                if state.jobs.is_cancelled(job):
                    proc.terminate()
                    job.log("terminated by operator")
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
                        proc.kill()
                    break
                time.sleep(0.2)
        finally:
            if proc.poll() is None:
                # Something other than a clean exit or cancellation interrupted the wait; do
                # not leave an orphaned child running.
                proc.kill()
                proc.wait()
            reader.join(timeout=5)
    if proc.returncode != 0 and not state.jobs.is_cancelled(job):
        raise RuntimeError(
            f"command exited with status {proc.returncode}; see job log {log_path}"
        )
    return {
        "exit_code": proc.returncode,
        "log": str(log_path),
    }


def _train_job(state: ServerState, job: Job, params: dict[str, Any]) -> dict[str, Any]:
    """Start a training run as a subprocess (isolation: a crash cannot take the server down)."""
    from tmai.config import parse_overrides

    argv = [sys.executable, "-m", "tmai.cli", "train"]
    config_path = params.get("config_path")
    config_yaml = params.get("config_yaml")
    if config_yaml:
        cfg_file = state.config.resolve(state.config.state_dir) / "jobs" / f"{job.id}.yaml"
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(config_yaml, encoding="utf-8")
        config_path = str(cfg_file)
    if config_path:
        argv += ["-c", str(config_path)]
    overrides = dict(params.get("overrides") or {})
    if "allow_simulated_driver" in params:
        allow_simulated = params["allow_simulated_driver"]
        if not isinstance(allow_simulated, bool):
            raise ValueError("allow_simulated_driver must be a boolean")
        # The explicit GUI safety switch must override config YAML in either direction.
        overrides["driver.allow_simulated"] = str(allow_simulated).lower()
    if overrides:
        parsed = parse_overrides([f"{k}={v}" for k, v in overrides.items()])
        for key, value in parsed.items():
            argv += ["--set", f"{key}={value}"]
    if params.get("run_name"):
        argv += ["--run-name", str(params["run_name"])]
    if params.get("steps"):
        argv += ["--steps", str(int(params["steps"]))]
    if params.get("resume"):
        argv += ["--resume", str(params["resume"])]
    runs_dir = state.config.resolve(state.config.runs_dir)
    before = {p.name for p in runs_dir.iterdir()} if runs_dir.is_dir() else set()
    result = _subprocess_job(state, job, argv)
    after = {p.name for p in runs_dir.iterdir()} if runs_dir.is_dir() else set()
    new_runs = sorted(after - before)
    result["run"] = new_runs[-1] if new_runs else None
    result["runs_dir"] = str(runs_dir)
    return result


def _config_for_params(state: ServerState, params: dict[str, Any]) -> RunConfig:
    """Resolve the config a job should use: explicit path > run's config > server default."""
    config_path = params.get("config_path")
    if config_path:
        return RunConfig.from_yaml(str(config_path))
    run = params.get("run")
    if run:
        run_path = _run_dir(state, str(run))
        saved = run_path / "config.yaml"
        if saved.is_file():
            # Saved by an earlier build may carry retired keys; do not refuse to resume over them.
            return RunConfig.from_yaml(str(saved), strict=False)
    if state.config.config_path:
        return RunConfig.from_yaml(state.config.config_path)
    return RunConfig()


def _eval_job(state: ServerState, job: Job, params: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a checkpoint (in-process) and write the report JSON."""
    from tmai.training.checkpoint import latest_checkpoint, load_checkpoint
    from tmai.training.evaluate import evaluate_tracks
    from tmai.training.factory import build_learner, build_library, build_multi_track_env

    config = _config_for_params(state, params)
    config.validate_or_raise()
    library = build_library(config)

    checkpoint = params.get("checkpoint")
    if checkpoint:
        source = Path(str(checkpoint))
        path = source if source.is_file() else latest_checkpoint(source)
        if path is None:
            raise FileNotFoundError(f"no checkpoint found at {source}")
    else:
        run = str(params.get("run") or "")
        if not run:
            raise ValueError("eval needs a checkpoint or a run")
        path = latest_checkpoint(_run_dir(state, run))
        if path is None:
            raise FileNotFoundError(f"no checkpoint found in run {run!r}")
    job.log(f"evaluating {path}")

    splits = params.get("splits") or (["validation"] if library.by_split("validation") else ["train"])
    for split in splits:
        if not library.by_split(split):
            raise ValueError(
                f"requested split {split!r} is empty (library has {library.counts()}); "
                "select a populated split rather than substituting training results"
            )
    reports: list[dict[str, Any]] = []
    env = None
    try:
        for split in splits:
            _check_job_cancelled(state, job)
            job.log(f"split {split!r}: {len(library.by_split(split))} track(s)")
            env = build_multi_track_env(config, library, split=split, seed=config.train.seed)
            learner = build_learner(env, config)
            payload = load_checkpoint(path)
            learner.load_state_dict(payload["learner"])
            report = evaluate_tracks(
                env,
                learner,
                tracks=[(e.track.name, split) for e in library.by_split(split)],
                episodes_per_track=int(params.get("episodes", 3)),
                max_steps=config.env.termination.max_steps,
                deterministic=not bool(params.get("stochastic", False)),
                seed=config.train.seed,
                label=f"gui-eval:{split}",
                step=int(payload.get("step", 0)),
                cancel_check=lambda: _check_job_cancelled(state, job),
            )
            reports.append(report.as_dict())
            job.log(f"  {split}: {report.summary()}")
            env.close()
            env = None
    finally:
        if env is not None:
            env.close()

    _check_job_cancelled(state, job)
    out_dir = state.config.resolve(state.config.state_dir) / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{job.id}.json"
    out_path.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    return {
        "path": str(out_path),
        "checkpoint": str(path),
        "splits": splits,
        "reports": reports,
    }


def _benchmark_job(state: ServerState, job: Job, params: dict[str, Any]) -> dict[str, Any]:
    """Run a benchmark (in-process) and write the report into the benchmarks directory."""
    from tmai.training.benchmark import run_benchmark

    config = _config_for_params(state, params)
    models = [(str(m["label"]), str(m["checkpoint"])) for m in params.get("models") or []]
    if not models:
        raise ValueError("benchmark needs at least one model")
    splits = params.get("splits") or ["validation", "test"]
    name = str(params.get("name") or "benchmark")

    report = run_benchmark(
        config,
        models,
        splits=list(splits),
        episodes_per_track=int(params.get("episodes", 3)),
        name=name,
        seed_repeats=int(params.get("seed_repeats", 3)),
        cancel_check=lambda: _check_job_cancelled(state, job),
    )
    job.log(f"paired evaluation seeds: {report.evaluation_seeds}")
    for model in report.models:
        kind = f" ({model.baseline_kind} heuristic)" if model.baseline_kind else ""
        job.log(f"{model.label}{kind}: score {model.score:.3f}")
    job.log(f"ranking: {' > '.join(report.ranking)}")
    for comparison in report.head_to_head:
        job.log(
            f"{comparison.split} {comparison.model_a} vs {comparison.model_b}: "
            f"{comparison.wins_a}-{comparison.wins_b}-{comparison.ties} over "
            f"{comparison.episodes} paired episodes"
        )

    _check_job_cancelled(state, job)
    out_dir = state.config.resolve(state.config.benchmarks_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{name}_{stamp}.json"
    report.save(out_path)
    return {
        "path": str(out_path),
        "ranking": report.ranking,
        "table": report.table(),
        "report": report.as_dict(),
    }


def _calibrate_job(state: ServerState, job: Job, params: dict[str, Any]) -> dict[str, Any]:
    """Run telemetry calibration. Only meaningful on the game host; reports honestly if not."""
    argv = [sys.executable, "-m", "tmai.cli", "doctor", "--calibrate"]
    if params.get("config_path"):
        argv += ["-c", str(params["config_path"])]
    argv += ["--calibrate-steps", str(int(params.get("steps", 300)))]
    result = _subprocess_job(state, job, argv)
    result["note"] = (
        "calibration measures the real game's telemetry; without Trackmania connected "
        "the doctor reports that and skips the measurement"
    )
    return result


def _doctor_job(state: ServerState, job: Job, params: dict[str, Any]) -> dict[str, Any]:
    argv = [sys.executable, "-m", "tmai.cli", "doctor"]
    if params.get("config_path"):
        argv += ["-c", str(params["config_path"])]
    if params.get("calibrate"):
        argv += ["--calibrate", "--calibrate-steps", str(int(params.get("steps", 300)))]
    return _subprocess_job(state, job, argv)


# -- app factory --------------------------------------------------------------------------


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Build the FastAPI application. All state lives in ``app.state.tmai``."""
    config = config or ServerConfig()
    state_dir = config.resolve(config.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    jobs = JobManager(state_dir / "jobs")
    state = ServerState(config=config, jobs=jobs)

    app = FastAPI(
        title="TrackmaniaAI",
        version=__version__,
        description="Local command center for TrackmaniaAI training, evaluation and analysis.",
    )
    app.state.tmai = state

    # The frontend uses relative /api URLs (Vite proxies them in development).
    # Do not opt every website into reading this unauthenticated local API.

    # -- health & system ------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "started_utc": state.started_utc}

    @app.get("/api/system")
    def system() -> dict[str, Any]:
        import importlib.util
        import platform

        resources = system_metrics()
        torch_spec = importlib.util.find_spec("torch")
        tmi_spec = importlib.util.find_spec("tminterface")
        return {
            "version": __version__,
            "python": resources.get("python"),
            "platform": platform.platform(),
            "windows_host": sys.platform == "win32",
            "torch_installed": torch_spec is not None,
            "tminterface_installed": tmi_spec is not None,
            "game_integration_possible": sys.platform == "win32" and tmi_spec is not None,
            "resources": resources,
            "paths": {
                "runs": str(config.resolve(config.runs_dir)),
                "tracks": str(config.resolve(config.tracks_dir)),
                "models": str(config.resolve(config.models_dir)),
                "demos": str(config.resolve(config.demos_dir)),
                "benchmarks": str(config.resolve(config.benchmarks_dir)),
                "state": str(state_dir),
            },
        }

    # -- runs -----------------------------------------------------------------------

    @app.get("/api/runs")
    def runs() -> dict[str, Any]:
        base = config.resolve(config.runs_dir)
        return {"runs": status_api.list_runs(base), "runs_dir": str(base)}

    @app.get("/api/runs/{name}")
    def run_detail(name: str) -> dict[str, Any]:
        return status_api.run_snapshot(_run_dir(state, name))

    @app.get("/api/runs/{name}/metrics")
    def run_metrics(
        name: str,
        max_points: int = Query(400, ge=1, le=20000),
        metrics: str | None = None,
    ) -> dict[str, Any]:
        history = status_api.run_history(
            _run_dir(state, name),
            max_points=max_points,
            metrics_filter=([m.strip() for m in metrics.split(",") if m.strip()] if metrics else None),
        ).as_dict()
        return {"run": name, "history": history}

    @app.get("/api/runs/{name}/log")
    def run_log(name: str, lines: int = Query(200, ge=1, le=5000)) -> dict[str, Any]:
        run_path = _run_dir(state, name)
        log_path = run_path / "run.log"
        tail: list[str] = []
        if log_path.is_file():
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        return {"run": name, "log": tail}

    @app.get("/api/runs/{name}/checkpoints")
    def run_checkpoints(name: str) -> dict[str, Any]:
        from tmai.training.checkpoint import list_checkpoints

        run_path = _run_dir(state, name)
        files = list_checkpoints(run_path)
        best = run_path / "best.pt"
        return {
            "run": name,
            "checkpoints": [
                {"name": p.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime} for p in files
            ],
            "best": {"name": best.name, "size": best.stat().st_size} if best.is_file() else None,
        }

    # -- tracks ----------------------------------------------------------------------

    def _track_directory(directory: str | None) -> Path:
        root = config.resolve(config.tracks_dir)
        if not directory:
            return root
        path = Path(directory)
        path = path if path.is_absolute() else (PROJECT_ROOT / path)
        return _within(path, root, what="track directory")

    @app.get("/api/tracks")
    def tracks(directory: str | None = None) -> dict[str, Any]:
        from tmai.tracks.library import TrackLibrary, TrackLibraryError

        base = _track_directory(directory)
        if not base.is_dir():
            return {
                "directory": str(base),
                "report": None,
                "error": "not found",
                "synthetic": ["straight", "oval", "s_curve", "figure_eight"],
            }
        # The library loader accepts symlinks for CLI use; the HTTP API must not read
        # linked JSON files pointing outside the configured track directory.
        for candidate in base.glob("*.json"):
            _within(candidate, config.resolve(config.tracks_dir), what="track")
        try:
            library = TrackLibrary.from_directory(base)
        except TrackLibraryError as exc:
            return {
                "directory": str(base),
                "report": None,
                "error": str(exc),
                "synthetic": ["straight", "oval", "s_curve", "figure_eight"],
            }
        return {
            "directory": str(base),
            "report": library.report(),
            "synthetic": ["straight", "oval", "s_curve", "figure_eight"],
        }

    @app.get("/api/tracks/geometry")
    def track_geometry(
        name: str | None = None,
        synthetic: str | None = None,
        directory: str | None = None,
    ) -> dict[str, Any]:
        from tmai.tracks.centerline import CenterlineTrack
        from tmai.tracks.synthetic import SYNTHETIC_TRACKS

        if synthetic:
            if synthetic not in SYNTHETIC_TRACKS:
                raise HTTPException(404, f"unknown synthetic track {synthetic!r}")
            track = SYNTHETIC_TRACKS[synthetic]()
            return _track_payload(track)
        if not name:
            raise HTTPException(400, "pass ?name=<track> or ?synthetic=<name>")
        safe_track_name = _safe_name(name, what="track name")
        if not directory:
            track = _resolve_track(state, safe_track_name)
            if track is not None:
                return _track_payload(track)
        base = _track_directory(directory)
        path = base / safe_track_name
        candidates = [path] if path.suffix == ".json" else [path.with_suffix(".json"), path]
        for candidate in candidates:
            if candidate.is_file():
                return _track_payload(CenterlineTrack.load(_within(candidate, base, what="track")))
        raise HTTPException(404, f"track {name!r} not found in {base}")

    # -- models ----------------------------------------------------------------------

    def _store():
        from tmai.registry import ModelStore

        return ModelStore(config.resolve(config.models_dir))

    @app.get("/api/models")
    def models() -> dict[str, Any]:
        return {
            "models": [m.as_dict() for m in _store().list()],
            "directory": str(config.resolve(config.models_dir)),
        }

    @app.get("/api/models/{name}")
    def model_detail(name: str) -> dict[str, Any]:
        from tmai.registry import RegistryError

        try:
            return _store().get(_safe_name(name, what="model name")).as_dict()
        except RegistryError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/models/register")
    def model_register(payload: dict[str, Any]) -> dict[str, Any]:
        from tmai.registry import RegistryError

        name = str(payload.get("name") or "")
        checkpoint = str(payload.get("checkpoint") or "")
        if not name or not checkpoint:
            raise HTTPException(400, "name and checkpoint are required")
        try:
            info = _store().register(
                name,
                checkpoint,
                tags=list(payload.get("tags") or []),
                notes=str(payload.get("notes") or ""),
                overwrite=bool(payload.get("overwrite", False)),
            )
        except RegistryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return info.as_dict()

    @app.delete("/api/models/{name}")
    def model_delete(name: str) -> dict[str, Any]:
        from tmai.registry import RegistryError

        try:
            _store().delete(_safe_name(name, what="model name"))
        except RegistryError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"deleted": name}

    # -- benchmarks ------------------------------------------------------------------

    @app.get("/api/benchmarks")
    def benchmarks() -> dict[str, Any]:
        base = config.resolve(config.benchmarks_dir)
        out: list[dict[str, Any]] = []
        if base.is_dir():
            for path in sorted(base.glob("*.json"), reverse=True):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (ValueError, json.JSONDecodeError):
                    continue
                out.append(
                    {
                        "name": path.name,
                        "path": str(path),
                        "benchmark": data.get("name", path.stem),
                        "created_utc": data.get("created_utc", ""),
                        "ranking": data.get("ranking", []),
                        "splits": data.get("splits", []),
                        "seed_repeats": data.get("seed_repeats", 1),
                    }
                )
        return {"benchmarks": out, "directory": str(base)}

    @app.post("/api/benchmark")
    def benchmark(payload: dict[str, Any]) -> dict[str, Any]:
        job = jobs.submit(
            "benchmark",
            f"benchmark {payload.get('name') or 'benchmark'}",
            lambda j: _benchmark_job(state, j, payload),
            params=payload,
        )
        return {"job": job.as_dict()}

    # -- jobs: eval / train / calibrate / doctor --------------------------------------

    @app.post("/api/eval")
    def start_eval(payload: dict[str, Any]) -> dict[str, Any]:
        job = jobs.submit(
            "eval",
            f"eval {payload.get('checkpoint') or payload.get('run') or ''}".strip(),
            lambda j: _eval_job(state, j, payload),
            params=payload,
        )
        return {"job": job.as_dict()}

    @app.post("/api/train")
    def start_train(payload: dict[str, Any]) -> dict[str, Any]:
        job = jobs.submit(
            "train",
            f"train {payload.get('run_name') or ''}".strip() or "train",
            lambda j: _train_job(state, j, payload),
            params=payload,
        )
        return {"job": job.as_dict()}

    @app.post("/api/calibrate")
    def start_calibrate(payload: dict[str, Any]) -> dict[str, Any]:
        job = jobs.submit(
            "calibrate",
            "telemetry calibration",
            lambda j: _calibrate_job(state, j, payload),
            params=payload,
        )
        return {"job": job.as_dict()}

    @app.post("/api/doctor")
    def start_doctor(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        job = jobs.submit(
            "doctor",
            "environment & game-integration report",
            lambda j: _doctor_job(state, j, payload),
            params=payload,
        )
        return {"job": job.as_dict()}

    @app.get("/api/jobs")
    def job_list(kind: str | None = None, limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
        return {
            "jobs": [j.as_dict() for j in jobs.list(kind=kind, limit=limit)],
            "workers": 2,
        }

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        return job.as_dict()

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str) -> dict[str, Any]:
        if not jobs.cancel(job_id):
            raise HTTPException(status_code=404, detail=f"no cancellable job {job_id!r}")
        job = jobs.get(job_id)
        return {"job": job.as_dict() if job else None}

    # -- replays -----------------------------------------------------------------------

    def _replay_store(run: str):
        from tmai.replay import ReplayStore

        base = _run_dir(state, run)
        return ReplayStore(_within(base / "replays", base, what="replay directory"))

    def _replay_path(store, name: str) -> Path:
        path = store.path / _safe_name(name, what="replay file")
        return _within(path, store.path, what="replay file")

    def _user_file(value: str, *, roots: tuple[Path, ...], suffixes: tuple[str, ...]) -> Path:
        """Resolve a GUI-supplied filename only within its allowed data directories."""
        path = Path(value)
        if path.suffix.lower() not in suffixes:
            raise HTTPException(400, "unsupported file type")
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        resolved = path.resolve()
        if not any(resolved.is_relative_to(root.resolve()) for root in roots):
            raise HTTPException(400, "file must be inside the configured data directories")
        return resolved

    @app.get("/api/runs/{name}/analysis")
    def run_analysis(
        name: str,
        track: str = Query(..., min_length=1),
        sectors: int = Query(20, ge=1, le=200),
        lateral_bins: int = Query(7, ge=3, le=31),
    ) -> dict[str, Any]:
        from tmai.replay import ReplayStore
        from tmai.training.analysis import analyze_replays

        run_dir = _run_dir(state, name)
        track_name = _safe_name(track, what="track name")
        geometry = _resolve_track(state, track_name)
        if geometry is None:
            raise HTTPException(status_code=404, detail=f"track {track_name!r} not found")
        store = ReplayStore(run_dir / "replays")
        matching = [
            store.load(row["name"])
            for row in store.list()
            if not row.get("track") or row.get("track") == track_name
        ]
        try:
            return analyze_replays(
                matching,
                geometry,
                sector_count=sectors,
                lateral_bin_count=lateral_bins,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/replays")
    def replays(run: str = Query(...)) -> dict[str, Any]:
        store = _replay_store(run)
        return {"run": run, "replays": store.list(), "directory": str(store.path)}

    @app.get("/api/replays/{run}/{file}")
    def replay_detail(run: str, file: str) -> dict[str, Any]:

        try:
            store = _replay_store(run)
            replay = store.load(_replay_path(store, file))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return replay.as_dict()

    @app.post("/api/replays/compare")
    def replay_compare(payload: dict[str, Any]) -> dict[str, Any]:
        from tmai.replay import EpisodeReplay, compare_replays
        from tmai.tracks.centerline import CenterlineTrack
        from tmai.training.demos import Demonstration

        run = str(payload.get("run") or "")
        if not run:
            raise HTTPException(400, "run is required")
        store = _replay_store(run)
        try:
            ai = store.load(_replay_path(store, str(payload.get("a") or "")))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        # The ghost can come from this/another run's replay directory or the demos
        # directory, but not from an arbitrary server-side filesystem path.
        ghost_value = str(payload.get("b") or "")
        if Path(ghost_value).name == ghost_value and (store.path / ghost_value).is_file():
            ghost_path = _replay_path(store, ghost_value)
        else:
            runs_root = config.resolve(config.runs_dir)
            demos_root = config.resolve(config.demos_dir)
            ghost_path = _user_file(ghost_value, roots=(runs_root, demos_root), suffixes=(".json", ".jsonl"))
            if ghost_path.is_relative_to(runs_root.resolve()) and ghost_path.parent.name != "replays":
                raise HTTPException(400, "ghost replay must be in a run's replays directory")
        try:
            ghost = EpisodeReplay.load(ghost_path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            try:
                ghost = EpisodeReplay.from_demonstration(Demonstration.load(ghost_path))
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

        # Geometry for the progress projection: an explicit track file wins, then the
        # replay's own track name (recorded library or synthetic).
        track = None
        track_path = payload.get("track")
        if track_path:
            track = CenterlineTrack.load(
                _user_file(str(track_path), roots=(config.resolve(config.tracks_dir),), suffixes=(".json",))
            )
        else:
            track = _resolve_track(state, ai.track or ghost.track)
        try:
            comparison = compare_replays(ai, ghost, track=track)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return comparison.as_dict()

    # -- demonstrations ------------------------------------------------------------------

    @app.get("/api/demos")
    def demos() -> dict[str, Any]:
        base = config.resolve(config.demos_dir)
        out: list[dict[str, Any]] = []
        if base.is_dir():
            for path in sorted(base.rglob("*.jsonl")):
                steps = 0
                try:
                    with path.open(encoding="utf-8") as handle:
                        for line in handle:
                            if line.startswith('{"observation"'):
                                steps += 1
                except OSError:
                    continue
                out.append(
                    {
                        "name": str(path.relative_to(base)),
                        "path": str(path),
                        "steps": steps,
                        "size": path.stat().st_size,
                    }
                )
        return {"demos": out, "directory": str(base)}

    # -- config ---------------------------------------------------------------------------

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        path = (
            Path(state.config.config_path)
            if state.config.config_path
            else PROJECT_ROOT / "tmai" / "configs" / "default.yaml"
        )
        yaml_text = path.read_text(encoding="utf-8") if path.is_file() else ""
        parsed: dict[str, Any] = {}
        problems: list[str] = []
        if yaml_text:
            try:
                parsed = RunConfig.from_yaml_text(yaml_text).to_dict()
            except ValueError as exc:  # ConfigError and validation errors are ValueErrors
                problems = [str(exc)]
        return {
            "path": str(path),
            "yaml": yaml_text,
            "config": parsed,
            "problems": problems,
        }

    @app.post("/api/config/validate")
    def validate_config(payload: dict[str, Any]) -> dict[str, Any]:
        yaml_text = str(payload.get("yaml") or "")
        if not yaml_text.strip():
            raise HTTPException(400, "yaml is required")
        try:
            config = RunConfig.from_yaml_text(yaml_text)
            config.validate_or_raise()
        except ValueError as exc:  # ConfigError and validation errors are ValueErrors
            return {"ok": False, "problems": [str(exc)]}
        except Exception as exc:  # noqa: BLE001 - report any parse failure
            return {"ok": False, "problems": [f"{type(exc).__name__}: {exc}"]}
        return {"ok": True, "problems": [], "config": config.to_dict()}

    # -- websocket -------------------------------------------------------------------------

    @app.websocket("/api/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        # CORS does not apply to WebSockets. Browsers always send Origin; refuse
        # cross-site reads of job/system data even when bound to loopback.
        origin = websocket.headers.get("origin")
        if origin and (
            urlsplit(origin).scheme not in ("http", "https")
            or urlsplit(origin).netloc.lower() != websocket.headers.get("host", "").lower()
        ):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            while True:
                tick = {
                    "type": "tick",
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "system": system_metrics(),
                    "runs": _runs_light(state),
                    "jobs": [
                        {
                            "id": j.id,
                            "kind": j.kind,
                            "state": j.state.value,
                            "description": j.description,
                        }
                        for j in jobs.list(limit=20)
                    ],
                }
                await websocket.send_text(json.dumps(tick, default=str))
                await _sleep(1.5)
        except WebSocketDisconnect:
            return
        except Exception:  # noqa: BLE001 - a broken client must not kill the loop
            logger.debug("websocket closed", exc_info=True)
            return

    # -- static frontend --------------------------------------------------------------------

    static_dir = config.resolve(config.static_dir)
    if static_dir.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="gui")
    else:

        @app.get("/", include_in_schema=False)
        def frontend_missing() -> JSONResponse:
            return JSONResponse(
                {
                    "error": "frontend not built",
                    "detail": (
                        f"no built frontend at {static_dir}. Build it with "
                        "'cd gui && npm run build', then restart tmai serve."
                    ),
                    "api": "/api/health",
                },
                status_code=404,
            )

    return app


async def _sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def run_server(config: ServerConfig | None = None, *, open_browser: bool = True) -> None:
    """Run the server in the foreground (uvicorn)."""
    import uvicorn

    config = config or ServerConfig()
    app = create_app(config)
    url = f"http://{config.host}:{config.port}"
    print(f"TrackmaniaAI GUI backend listening on {url}")
    print(f"  runs       {config.resolve(config.runs_dir)}")
    print(f"  tracks     {config.resolve(config.tracks_dir)}")
    print(f"  models     {config.resolve(config.models_dir)}")
    print(f"  frontend   {config.resolve(config.static_dir)}")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


__all__ = ["ServerConfig", "ServerState", "create_app", "run_server"]
