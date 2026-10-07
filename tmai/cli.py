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

from tmai import __version__
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
    if not source and getattr(args, "checkpoint", None):
        found = _run_dir_config(args.checkpoint)
        if found is not None:
            source = str(found)
            logger.info("using the run's saved configuration: %s", found)
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
    from tmai.training.evaluate import evaluate_policy
    from tmai.training.factory import build_env, build_learner, build_track

    config = _load_config(args)
    track = build_track(config)

    from tmai.training.factory import build_driver

    driver = build_driver(config, track)
    env = build_env(driver, track, config)
    learner = build_learner(env, config)

    source = Path(args.checkpoint) if args.checkpoint else Path(config.train.output_dir)
    path = source if source.is_file() else latest_checkpoint(source)
    if path is None:
        print(f"no checkpoint found at {source}", file=sys.stderr)
        env.close()
        return 1
    payload = load_checkpoint(path)
    learner.load_state_dict(payload["learner"])
    print(f"loaded {path} (step {payload.get('step')})")

    report = evaluate_policy(
        env,
        learner,
        episodes=args.episodes,
        max_steps=config.env.termination.max_steps,
        deterministic=not args.stochastic,
    )
    env.close()

    print(f"\n=== evaluation: {path.name} ===")
    print(f"  episodes        {report.num_episodes}")
    print(f"  finish rate     {report.finish_rate * 100:.1f}%")
    print(f"  mean progress   {report.mean_progress_fraction * 100:.2f}%")
    best = report.best_race_time
    print(f"  best lap time   {best:.3f}s" if best is not None else "  best lap time   --")
    for i, episode in enumerate(report.episodes, start=1):
        print(
            f"   #{i} {episode.end_reason:<16} progress {episode.progress_fraction * 100:5.1f}%  "
            f"time {episode.race_time:7.2f}s  mean speed {episode.mean_speed:5.1f} m/s"
            + ("  INVALID FINISH" if episode.invalid_finish else "")
        )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


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
