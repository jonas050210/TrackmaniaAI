"""Command-line interface.

Sub-commands::

    tmai doctor          environment + real-game integration health report
    tmai train           run training
    tmai eval            evaluate a checkpoint
    tmai record-track    record a track centreline by driving the real map once
    tmai show-track      render the simplified 3D track view to a PNG
    tmai export-obj      export the track mesh for an external viewer

``doctor`` is the entry point for verifying the real integration: it reports platform,
dependency and connectivity status, and with ``--calibrate`` it measures the telemetry
conventions rather than assuming them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

from tmai import __version__
from tmai.api.status import HEADLINE_METRICS
from tmai.config import RunConfig, parse_overrides

logger = logging.getLogger("tmai")


# -- helpers ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _run_dir_config(checkpoint: str | None) -> Path | None:
    """Find the ``config.yaml`` saved in a checkpoint's run directory, if there is one.

    A checkpoint is only meaningful together with the configuration that produced it -- the
    observation layout, the track, and the normalisation all have to match. Pointing ``eval`` at
    a run directory is therefore also a request to use that run's configuration, and doing so
    implicitly avoids the silent mismatch of loading a checkpoint into a differently-configured
    observation. An explicitly passed ``-c`` still wins.
    """
    if not checkpoint:
        return None
    path = Path(checkpoint)
    # A .pt file lives inside the run directory, so start from its parent; a directory
    # argument *is* the run directory.
    start = path.parent if path.is_file() else path
    for candidate in (start, *start.parents):
        saved = candidate / "config.yaml"
        if saved.is_file():
            return saved
    return None


def _load_config(args: argparse.Namespace) -> RunConfig:
    source = getattr(args, "config", None)
    if not source:
        # Both `eval --checkpoint <run>` and `train --resume <run>` point at an existing run,
        # and in both cases the run's own config.yaml is the right one: the observation layout,
        # the track and the normalisation have to match what produced the checkpoint. Without
        # this, `tmai train --resume runs/<run>` silently fell back to default.yaml and failed
        # with "no track configured" even though the run directory says exactly what to use.
        for attr in ("checkpoint", "resume"):
            found = _run_dir_config(getattr(args, attr, None))
            if found is not None:
                source = str(found)
                logger.info("using the run's saved configuration: %s", found)
                break
    config = RunConfig.from_yaml(source) if source else RunConfig()
    overrides = parse_overrides(getattr(args, "set", None))
    if getattr(args, "allow_simulated_driver", False):
        overrides["driver.allow_simulated"] = True
    return config.apply_overrides(overrides) if overrides else config


def _print_report(title: str, rows: list[tuple[str, str, str]]) -> None:
    """Print a status table. Third element is one of ok / warn / fail."""
    marks = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]"}
    print(f"\n{title}")
    print("-" * max(len(title), 60))
    for name, value, status in rows:
        print(f"  {marks.get(status, '[ ? ]')} {name:<28} {value}")


# -- doctor ----------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report environment health and, optionally, calibrate real telemetry."""
    import platform

    import numpy as np

    rows: list[tuple[str, str, str]] = [
        ("trackmania-ai", __version__, "ok"),
        ("python", sys.version.split()[0], "ok"),
        ("platform", platform.platform(), "ok"),
        ("numpy", np.__version__, "ok"),
    ]

    windows = sys.platform == "win32"
    rows.append(
        (
            "windows host",
            "yes" if windows else f"no ({sys.platform})",
            "ok" if windows else "warn",
        )
    )
    rows.append(
        (
            "tminterface",
            "importable" if importlib.util.find_spec("tminterface") else "not installed",
            "ok" if importlib.util.find_spec("tminterface") else "warn",
        )
    )

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec:
        import torch

        rows.append(("torch", torch.__version__, "ok"))
        rows.append(
            ("cuda available", str(torch.cuda.is_available()),
             "ok" if torch.cuda.is_available() else "warn")
        )
    else:
        rows.append(("torch", "not installed (needed for training)", "warn"))

    _print_report("environment", rows)

    game_rows: list[tuple[str, str, str]] = []
    driver = None
    try:
        config = _load_config(args)
        from tmai.training.factory import build_driver, build_track

        track = None
        try:
            track = build_track(config)
            game_rows.append(
                ("track", f"{track.name} ({track.length:.1f} m, {track.num_points} pts)", "ok")
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            game_rows.append(("track", str(exc).splitlines()[0], "warn"))

        if config.driver.kind != "simulated":
            driver = build_driver(config, track) if track is not None else None
            if driver is None:
                game_rows.append(("driver", "not built (no track available)", "warn"))
            else:
                driver.open()
                game_rows.append(("driver", f"connected ({driver.name})", "ok"))
                info = driver.describe()
                game_rows.append(
                    ("checkpoints on map", str(info.get("checkpoint_total")), "ok")
                )
        else:
            game_rows.append(
                ("driver", "simulated (NOT the real game)", "warn")
            )
    except Exception as exc:  # noqa: BLE001 - the whole point is to report failures
        game_rows.append(("driver", str(exc).splitlines()[0], "fail"))

    _print_report("game integration", game_rows)

    exit_code = 0 if all(s != "fail" for _, _, s in game_rows) else 1

    if args.calibrate and driver is not None:
        from tmai.game.calibration import calibrate_driver

        print(f"\ncalibrating with {args.calibrate_steps} steps of full throttle ...")
        report = calibrate_driver(
            driver,
            steps=args.calibrate_steps,
            assumed_control_dt=_load_config(args).env.control_dt,
        )
        print(report.format())
        if args.calibrate_out:
            Path(args.calibrate_out).write_text(
                json.dumps(report.as_dict(), indent=2), encoding="utf-8"
            )
            print(f"\nwrote {args.calibrate_out}")
        if not report.ok:
            exit_code = 1

    if driver is not None:
        driver.close()
    print()
    return exit_code


# -- train -----------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    from tmai.training.trainer import train_from_config

    config = _load_config(args)
    for key, value in (
        ("total_steps", args.steps),
        ("run_name", args.run_name),
        ("output_dir", args.output_dir),
        ("resume", args.resume),
        ("seed", args.seed),
        ("device", args.device),
    ):
        if value is not None:
            setattr(config.train, key, value)

    if config.driver.kind == "simulated" and not config.driver.allow_simulated:
        from tmai.training.factory import ConfigError

        raise ConfigError(
            "driver.kind='simulated' is a toy model, not Trackmania. "
            "Pass --allow-simulated-driver if that is really what you want."
        )

    result = train_from_config(config)
    summary = result.as_dict()
    summary.pop("final_evaluation", None)
    print("\n=== training finished ===")
    for key, value in summary.items():
        print(f"  {key:<16} {value}")
    if result.final_evaluation is not None:
        print(f"  {'evaluation':<16} {result.final_evaluation.summary()}")
    print(f"  {'run directory':<16} {result.run_dir}")
    return 1 if result.failure else 0


# -- eval ------------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    from tmai.training.checkpoint import latest_checkpoint, load_checkpoint
    from tmai.training.evaluate import evaluate_policy, evaluate_tracks
    from tmai.training.factory import build_learner, build_library, build_multi_track_env

    config = _load_config(args)
    library = build_library(config)

    # Evaluating a multi-track run on one map would throw away the only number that matters,
    # so every split the operator asked for is covered. Default is "validation" when the run
    # has one, because that is the honest measure; fall back to train otherwise.
    splits = (
        [s.strip() for s in args.split.split(",") if s.strip()]
        if args.split
        else (["validation"] if library.by_split("validation") else ["train"])
    )

    env = None
    reports: list[Any] = []
    exit_code = 0
    try:
        for split in splits:
            if not library.by_split(split):
                print(
                    f"split {split!r} is empty (library has {library.counts()}); skipping",
                    file=sys.stderr,
                )
                continue
            env = build_multi_track_env(config, library, split=split, seed=config.train.seed)
            learner = build_learner(env, config)

            source = Path(args.checkpoint) if args.checkpoint else Path(config.train.output_dir)
            path = source if source.is_file() else latest_checkpoint(source)
            if path is None:
                print(f"no checkpoint found at {source}", file=sys.stderr)
                return 1
            payload = load_checkpoint(path)
            learner.load_state_dict(payload["learner"])
            print(
                f"loaded {path.name} (step {payload.get('step')}, "
                f"{payload.get('gradient_steps')} gradient steps) for split {split!r}"
            )

            if library.by_split(split) and len(library.by_split(split)) > 1:
                # Exhaustive: every track in the split, so nothing is left to sampling luck.
                report = evaluate_tracks(
                    env,
                    learner,
                    tracks=[(e.track.name, split) for e in library.by_split(split)],
                    episodes_per_track=args.episodes,
                    max_steps=config.env.termination.max_steps,
                    deterministic=not args.stochastic,
                    seed=config.train.seed,
                    label=f"eval:{split}",
                    step=int(payload.get("step", 0)),
                )
            else:
                report = evaluate_policy(
                    env,
                    learner,
                    episodes=args.episodes,
                    max_steps=config.env.termination.max_steps,
                    deterministic=not args.stochastic,
                    seed=config.train.seed,
                    label=f"eval:{split}",
                    step=int(payload.get("step", 0)),
                    split=split,
                )
            reports.append(report)
            env.close()
            env = None
    finally:
        if env is not None:
            env.close()

    if not reports:
        print("no split could be evaluated", file=sys.stderr)
        return 1

    for report in reports:
        print(f"\n=== evaluation: {report.label} ({report.num_tracks} tracks) ===")
        print(report.table())
        print(f"\n  finish rate     {report.finish_rate * 100:.1f}%")
        print(f"  mean progress   {report.mean_progress_fraction * 100:.2f}%")
        print(f"  crash rate      {report.crash_rate * 100:.1f}%")
        print(f"  consistency     ±{report.consistency * 100:.1f}%")
        best = report.best_race_time
        print(f"  best lap time   {best:.3f}s" if best is not None else "  best lap time   --")
        invalid = [e for e in report.episodes if e.invalid_finish]
        if invalid:
            print(f"  !! {len(invalid)} INVALID finish(es): crossed the line without all checkpoints")

    if args.json_out:
        payload_out = (
            reports[0].as_dict()
            if len(reports) == 1
            else [r.as_dict() for r in reports]
        )
        Path(args.json_out).write_text(
            json.dumps(payload_out, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nwrote {args.json_out}")
    return exit_code


# -- record-track ----------------------------------------------------------------------


def cmd_record_track(args: argparse.Namespace) -> int:
    from tmai.tracks.recording import record_track
    from tmai.training.factory import build_driver, build_track

    config = _load_config(args)
    if config.driver.kind == "simulated":
        print(
            "record-track is meant to capture a REAL map. Refusing to run against the "
            "simulated driver; remove --allow-simulated-driver.",
            file=sys.stderr,
        )
        return 2

    driver = build_driver(config, build_track(config) if config.track.path or config.track.synthetic else None)
    driver.open()
    print(
        "\nDrive one clean lap. The centreline is sampled from your telemetry.\n"
        "Recording stops automatically at the finish line, or press Ctrl-C to stop early.\n"
    )
    track = record_track(
        driver,
        out_path=args.out,
        name=args.name,
        min_spacing=args.min_spacing,
        corridor_half_width=args.corridor,
        smoothing_window=args.smoothing,
    )
    driver.close()
    print(f"\nrecorded {track.num_points} points over {track.length:.1f} m -> {args.out}")
    return 0


# -- visualisation ---------------------------------------------------------------------


def _resolve_track(args: argparse.Namespace):
    from tmai.tracks.centerline import CenterlineTrack
    from tmai.tracks.synthetic import build_synthetic

    if getattr(args, "track", None):
        return CenterlineTrack.load(args.track)
    config = _load_config(args)
    if config.track.path:
        return CenterlineTrack.load(config.track.path)
    if config.track.synthetic:
        return build_synthetic(config.track.synthetic, **config.track.synthetic_kwargs)
    raise SystemExit("no track given: pass --track <file.json> or set track.synthetic")


def cmd_show_track(args: argparse.Namespace) -> int:
    from tmai.viz.trackview import TrackView, TrackViewConfig

    track = _resolve_track(args)
    view = TrackView(
        track,
        TrackViewConfig(
            show_corridor=not args.no_corridor,
            show_curvature=not args.no_curvature,
            show_car=False,
            show_trajectory=False,
        ),
    )
    path = view.render(args.out, title=f"{track.name} ({track.length:.0f} m)")
    print(f"wrote {path}")
    return 0


def cmd_export_obj(args: argparse.Namespace) -> int:
    from tmai.viz.trackview import TrackView

    track = _resolve_track(args)
    path = TrackView(track).to_obj(args.out)
    print(f"wrote {path} ({track.num_points} centreline points)")
    return 0


# -- parser ----------------------------------------------------------------------------


# -- status / inspection -------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    """Print a dashboard-style status for one or all runs."""
    from tmai.api.status import list_runs, run_snapshot, run_status

    if args.list:
        runs = list_runs(args.runs_dir or ".")
        if not runs:
            print(f"no runs found in {args.runs_dir or '.'}")
            return 1
        header = f"{'run':<44} {'step':>10} {'prog%':>6} {'ckpts':>6} {'driver':<12} state"
        print(header)
        print("-" * len(header))
        for run in runs:
            step = f"{run['step']}/{run['total_steps'] or '?'}"
            state = "ended" if run["ended"] else ("stale?" if False else "running")
            print(
                f"{Path(run['run_dir']).name[:44]:<44} {step:>10} "
                f"{(run['progress_fraction'] or 0) * 100:>6.1f} {run['checkpoints']:>6} "
                f"{(run['driver'] or '?'):<12} {state}"
            )
        return 0

    if not args.run:
        print("give --run <run-dir>, or --list to see available runs", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(run_snapshot(args.run, max_points=args.max_points), default=str, indent=2))
        return 0

    status = run_status(args.run)
    if not status.exists:
        print(f"no such run directory: {status.run_dir}", file=sys.stderr)
        return 1

    rows: list[tuple[str, str, str]] = [
        ("run", status.run_name or Path(status.run_dir).name, "ok"),
        ("step", f"{status.step}" + (f" / {status.total_steps}" if status.total_steps else ""), "ok"),
    ]
    if status.progress_fraction is not None:
        rows.append(("progress", f"{status.progress_fraction * 100:.1f}%", "ok"))
    rows += [
        ("episodes", str(status.episodes), "ok"),
        ("evaluations", str(status.evaluations), "ok"),
        ("checkpoints", str(status.checkpoints), "ok"),
        ("driver", status.driver or "?", "warn" if status.simulated else "ok"),
        (
            "last update",
            f"{status.seconds_since_update:.0f}s ago" if status.seconds_since_update is not None else "?",
            "fail" if status.stale else "ok",
        ),
        ("state", status.end_reason or ("ended" if status.ended else "running"), "ok"),
    ]
    _print_report(f"run status: {Path(status.run_dir).name}", rows)

    if status.simulated:
        print("\n  !! this run used the SIMULATED driver (a toy model, not Trackmania)")

    if status.latest:
        print("\nlatest metrics")
        print("-" * 60)
        for key in HEADLINE_METRICS:
            if key in status.latest:
                print(f"  {key:<36} {status.latest[key]:.4f}")

    from tmai.api.status import run_evaluations

    evaluations = run_evaluations(args.run)
    if evaluations:
        print("\nevaluations")
        print("-" * 74)
        print(f"  {'step':>8} {'kind':<10} {'fin%':>5} {'prog%':>6} {'crash%':>7} {'gap':>8}")
        for item in evaluations[-10:]:
            report = item["report"]
            gap = report.get("generalization_gap")
            print(
                f"  {item['step']:>8} {item['kind']:<10} "
                f"{report.get('finish_rate', 0) * 100:>5.0f} "
                f"{report.get('mean_progress_fraction', 0) * 100:>6.1f} "
                f"{report.get('crash_rate', 0) * 100:>7.0f} "
                f"{(f'{gap:+.3f}' if gap is not None else '--'):>8}"
            )
    return 0


def cmd_list_tracks(args: argparse.Namespace) -> int:
    """List the tracks in a directory with their geometry fingerprint and split."""
    from tmai.tracks.library import TrackLibrary

    library = TrackLibrary.from_directory(
        args.directory, pattern=args.pattern, recursive=args.recursive
    )
    report = library.report()

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    header = (
        f"{'track':<26} {'split':<11} {'length':>8} {'corners':>8} "
        f"{'tightest':>9} {'straight%':>10} {'width':>7}"
    )
    print(header)
    print("-" * len(header))
    for entry in report["tracks"]:
        stats = entry.get("stats") or {}
        radius = stats.get("min_corner_radius")
        radius_text = "straight" if radius is None else f"{radius:.0f}m"
        print(
            f"{entry['name'][:26]:<26} {entry['split']:<11} "
            f"{entry['length']:>7.0f}m {stats.get('corner_count', 0):>8} "
            f"{radius_text:>9} {stats.get('straight_fraction', 0) * 100:>10.0f} "
            f"{stats.get('corridor_mean', 0):>6.1f}m"
        )
    print("-" * len(header))
    counts = report["counts"]
    print(f"{report['num_tracks']} tracks: " + ", ".join(f"{v} {k}" for k, v in counts.items() if v))

    geometry = report.get("geometry_by_split") or {}
    if len(geometry) > 1:
        print("\ngeometry coverage by split (are the held-out tracks comparable?)")
        for split, data in geometry.items():
            print(
                f"  {split:<11} length {data['length_mean']:>7.1f}m "
                f"(range {data['length_min']:.0f}-{data['length_max']:.0f}) "
                f"corners {data['corners_mean']:.1f} curvature {data['curvature_mean']:.5f}"
            )
    return 0


def cmd_validate_config(args: argparse.Namespace) -> int:
    """Check a configuration file without running anything."""
    config = _load_config(args)
    problems = config.validate()

    if args.json:
        print(json.dumps({"valid": not problems, "problems": problems}, indent=2))
        return 1 if problems else 0

    if not problems:
        print("configuration is valid")
        print(f"  driver          {config.driver.kind}")
        sources = [
            name
            for name, value in (
                ("track.path", config.track.path),
                ("track.directory", config.track.directory),
                ("track.synthetic", config.track.synthetic),
                ("track.synthetic_suite", config.track.synthetic_suite),
            )
            if value
        ]
        print(f"  track source    {sources[0] if sources else '(none)'}")
        print(f"  total steps     {config.train.total_steps}")
        print(f"  observation dim {config.env.observation.dim}")
        print(f"  normalisation   {'on' if config.normalize.enabled else 'off'}")
        return 0

    print("configuration has problems:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return 1


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare several runs or checkpoints side by side."""
    from tmai.api.status import run_evaluations, run_status

    rows: list[dict[str, Any]] = []
    for target in args.targets:
        status = run_status(target)
        if not status.exists:
            logger.warning("skipping %s: not a run directory", target)
            continue
        evaluations = run_evaluations(target)
        training = [e for e in evaluations if e["kind"] == "training"]
        held = [e for e in evaluations if e["kind"] == "held_out"]
        last_train = training[-1]["report"] if training else {}
        last_held = held[-1]["report"] if held else {}

        # The training and held-out runs are logged as *separate* reports, each carrying only
        # its own split, so neither one can produce the gap on its own. Reading it off the
        # training report alone always yielded None; it has to be computed across the two.
        gap = last_train.get("generalization_gap")
        train_progress = last_train.get("mean_progress_fraction")
        held_progress = last_held.get("mean_progress_fraction")
        if gap is None and train_progress is not None and held_progress is not None:
            gap = round(float(train_progress) - float(held_progress), 4)

        rows.append(
            {
                # The manifest's run_name, not the timestamped directory name: the directory
                # is what the operator typed, but the name is what identifies the experiment.
                "name": status.run_name or Path(target).name,
                "run_dir": str(Path(target)),
                "step": status.step,
                "driver": status.driver or "?",
                "progress": train_progress,
                "finish": last_train.get("finish_rate"),
                "crash": last_train.get("crash_rate"),
                "held_progress": held_progress,
                "gap": gap,
                "best_lap": last_train.get("best_race_time"),
                "evals": len(training),
            }
        )

    if not rows:
        print("nothing to compare: no valid run directories given", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0

    def cell(value: Any, fmt: str = "{:.1f}", scale: float = 1.0) -> str:
        if value is None:
            return "--"
        return fmt.format(value * scale)

    header = (
        f"{'run':<38} {'step':>8} {'prog%':>6} {'fin%':>5} {'crash%':>7} "
        f"{'held%':>6} {'gap':>7} {'lap':>8} {'driver':<11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['name'][:38]:<38} {row['step']:>8} "
            f"{cell(row['progress'], '{:.1f}', 100):>6} "
            f"{cell(row['finish'], '{:.0f}', 100):>5} "
            f"{cell(row['crash'], '{:.0f}', 100):>7} "
            f"{cell(row['held_progress'], '{:.1f}', 100):>6} "
            f"{cell(row['gap'], '{:+.3f}'):>7} "
            f"{cell(row['best_lap'], '{:.2f}s'):>8} {row['driver']:<11}"
        )
    if any(row["driver"] == "simulated" for row in rows):
        print("\n  !! at least one of these runs used the SIMULATED driver (not Trackmania)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmai",
        description="TrackmaniaAI - reinforcement learning that drives the real Trackmania game",
    )
    parser.add_argument("--version", action="version", version=f"trackmania-ai {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("-c", "--config", help="YAML config file")
        p.add_argument(
            "--set",
            action="append",
            metavar="KEY=VALUE",
            help="override a config value, e.g. --set sac.gamma=0.98 (repeatable)",
        )
        p.add_argument(
            "--allow-simulated-driver",
            action="store_true",
            help="permit the toy model driver (NOT the real game)",
        )

    doctor = sub.add_parser("doctor", help="environment and game-integration health report")
    add_config_args(doctor)
    doctor.add_argument("--calibrate", action="store_true",
                        help="drive the game and measure telemetry conventions")
    doctor.add_argument("--calibrate-steps", type=int, default=300)
    doctor.add_argument("--calibrate-out", help="write the calibration report as JSON")
    doctor.set_defaults(func=cmd_doctor)

    train = sub.add_parser("train", help="run training")
    add_config_args(train)
    train.add_argument("--steps", type=int, help="override train.total_steps")
    train.add_argument("--run-name", help="override train.run_name")
    train.add_argument("--output-dir", help="override train.output_dir")
    train.add_argument("--resume", help="run directory or checkpoint to resume from")
    train.add_argument("--seed", type=int, help="override train.seed")
    train.add_argument("--device", help="override train.device (cpu/cuda)")
    train.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="evaluate a checkpoint")
    add_config_args(ev)
    ev.add_argument("--checkpoint", help="checkpoint file or run directory (default: latest)")
    ev.add_argument("--episodes", type=int, default=3)
    ev.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    ev.add_argument("--json-out", help="write the evaluation report as JSON")
    ev.add_argument(
        "--split",
        help="comma-separated library splits to evaluate, e.g. 'validation' or 'validation,test' "
        "(default: validation when present, else train)",
    )
    ev.set_defaults(func=cmd_eval)

    rec = sub.add_parser("record-track", help="record a track centreline from the real game")
    add_config_args(rec)
    rec.add_argument("--out", required=True, help="output JSON path")
    rec.add_argument("--name", default="recorded_track")
    rec.add_argument("--min-spacing", type=float, default=1.0, help="metres between samples")
    rec.add_argument("--corridor", type=float, default=5.0, help="corridor half width, metres")
    rec.add_argument("--smoothing", type=int, default=5, help="odd moving-average window")
    rec.set_defaults(func=cmd_record_track)

    show = sub.add_parser("show-track", help="render the simplified 3D track view to a PNG")
    add_config_args(show)
    show.add_argument("--track", help="track JSON file")
    show.add_argument("--out", default="track.png")
    show.add_argument("--no-corridor", action="store_true")
    show.add_argument("--no-curvature", action="store_true")
    show.set_defaults(func=cmd_show_track)

    obj = sub.add_parser("export-obj", help="export the track mesh as Wavefront .obj")
    add_config_args(obj)
    obj.add_argument("--track", help="track JSON file")
    obj.add_argument("--out", default="track.obj")
    obj.set_defaults(func=cmd_export_obj)

    status = sub.add_parser("status", help="inspect a training run (dashboard data)")
    status.add_argument("--run", help="run directory")
    status.add_argument("--runs-dir", help="directory to scan with --list")
    status.add_argument("--list", action="store_true", help="list runs in --runs-dir")
    status.add_argument("--json", action="store_true", help="emit the full snapshot as JSON")
    status.add_argument("--max-points", type=int, default=400,
                        help="max points per curve with --json")
    status.set_defaults(func=cmd_status)

    tracks = sub.add_parser("list-tracks", help="list tracks with geometry and split assignment")
    tracks.add_argument("directory", help="directory of centreline JSON files")
    tracks.add_argument("--pattern", default="*.json")
    tracks.add_argument("--recursive", action="store_true")
    tracks.add_argument("--json", action="store_true")
    tracks.set_defaults(func=cmd_list_tracks)

    check = sub.add_parser("validate-config", help="check a config without running anything")
    add_config_args(check)
    check.add_argument("--json", action="store_true")
    check.set_defaults(func=cmd_validate_config)

    compare = sub.add_parser("compare", help="compare several runs side by side")
    compare.add_argument("targets", nargs="+", help="run directories to compare")
    compare.add_argument("--json", action="store_true")
    compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level reporting
        logger.error("%s: %s", type(exc).__name__, exc)
        if getattr(args, "verbose", False):
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
