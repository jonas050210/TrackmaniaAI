"""Run logging: reproducible manifests plus streaming metrics.

A multi-day training run has to be diagnosable after the fact and comparable across runs, so
every run directory is self-describing::

    runs/2026-10-07T14-02-11_sac-scurve/
        manifest.json    config + git sha + versions + driver/env description + seed
        config.yaml      the resolved config, verbatim
        metrics.jsonl    one JSON object per logged step (append-only, crash-safe)
        events.jsonl     non-metric events: episodes finished, evals, errors, checkpoints
        run.log          human-readable log

``metrics.jsonl`` is append-only JSON Lines rather than a binary format on purpose: it can be
read line by line with any tool, survives a killed process, and can be resumed into.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
METRICS_NAME = "metrics.jsonl"
EVENTS_NAME = "events.jsonl"
CONFIG_NAME = "config.yaml"
LOG_NAME = "run.log"


def _git_sha() -> str | None:
    """Best-effort current commit; ``None`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for module in ("numpy", "torch", "gymnasium", "tminterface"):
        try:
            mod = __import__(module)
            versions[module] = str(getattr(mod, "__version__", "unknown"))
        except Exception:  # noqa: BLE001 - optional dependency
            versions[module] = "absent"
    return versions


def make_run_dir(base_dir: str | Path, name: str = "run") -> Path:
    """Create a timestamped run directory: ``<base>/<UTC timestamp>_<name>``."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    path = Path(base_dir) / f"{stamp}_{safe}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class RunLogger:
    """Writes the manifest, metrics stream and log file for one training run.

    Args:
        run_dir: directory to write into (created if missing).
        run_name: human-readable name recorded in the manifest.
        config: the resolved run configuration (must be JSON-serialisable).
        extra_manifest: additional manifest fields (driver/env descriptions).
        seed: the run seed, echoed into the manifest.
    """

    def __init__(
        self,
        run_dir: str | Path,
        run_name: str = "run",
        config: dict[str, Any] | None = None,
        extra_manifest: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self._t0 = time.time()

        self._metrics_path = self.run_dir / METRICS_NAME
        self._events_path = self.run_dir / EVENTS_NAME
        self._metrics_file = self._metrics_path.open("a", encoding="utf-8")
        self._events_file = self._events_path.open("a", encoding="utf-8")

        self._file_handler = logging.FileHandler(self.run_dir / LOG_NAME, encoding="utf-8")
        self._file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
        root = logging.getLogger()
        root.addHandler(self._file_handler)
        if root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)

        self._closed = False
        self._metrics_written = 0
        self._manifest = self._build_manifest(config or {}, extra_manifest or {}, seed)
        self.write_manifest()
        if config:
            (self.run_dir / CONFIG_NAME).write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
        logger.info("run directory: %s", self.run_dir)

    # -- manifest -------------------------------------------------------------------

    def _build_manifest(
        self, config: dict[str, Any], extra: dict[str, Any], seed: int | None
    ) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "host": {
                "hostname": socket.gethostname(),
                "user": getpass.getuser(),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "cpu_count": os.cpu_count(),
            },
            "versions": _package_versions(),
            "git": {"sha": _git_sha(), "dirty": _git_dirty()},
            "config": config,
            **extra,
        }

    def write_manifest(self) -> Path:
        path = self.run_dir / MANIFEST_NAME
        path.write_text(json.dumps(self._manifest, indent=2, default=str), encoding="utf-8")
        return path

    def update_manifest(self, **fields: Any) -> None:
        """Merge fields into the manifest and rewrite it (used for resume bookkeeping)."""
        self._manifest.update(fields)
        self.write_manifest()

    @property
    def manifest(self) -> dict[str, Any]:
        return self._manifest

    # -- streaming output -----------------------------------------------------------

    def log_metrics(self, step: int, metrics: Mapping[str, float | int | str | bool]) -> None:
        """Append one metrics record. Flushes so a crash loses at most one record."""
        record = {
            "step": int(step),
            "t": round(time.time() - self._t0, 3),
            **{k: _jsonable(v) for k, v in metrics.items()},
        }
        self._metrics_file.write(json.dumps(record, default=str) + "\n")
        self._metrics_file.flush()
        self._metrics_written += 1

    def log_event(self, event: str, **data: Any) -> None:
        record = {
            "event": event,
            "t": round(time.time() - self._t0, 3),
            **{k: _jsonable(v) for k, v in data.items()},
        }
        self._events_file.write(json.dumps(record, default=str) + "\n")
        self._events_file.flush()
        logger.info("event %s %s", event, {k: v for k, v in data.items() if k != "event"})

    @property
    def metrics_written(self) -> int:
        return self._metrics_written

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._metrics_file.close()
            self._events_file.close()
        finally:
            logging.getLogger().removeHandler(self._file_handler)
            self._file_handler.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _jsonable(value: Any) -> Any:
    """Coerce a value into something ``json`` can serialise, preserving structure.

    Containers are handled recursively rather than falling through to ``str()``: an event
    payload is meant to be machine-readable, and a nested report turned into the *string*
    ``"{'a': 1}"`` is not. Anything genuinely unserialisable still degrades to ``str``, so
    logging can never raise on an odd value.
    """
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001 - fall through to list conversion
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def read_metrics(run_dir: str | Path) -> list[dict[str, Any]]:
    """Read ``metrics.jsonl`` back, skipping torn trailing lines."""
    path = Path(run_dir) / METRICS_NAME
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("skipping malformed metrics line in %s", path)
    return out


def read_manifest(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"no manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "CONFIG_NAME",
    "EVENTS_NAME",
    "LOG_NAME",
    "MANIFEST_NAME",
    "METRICS_NAME",
    "RunLogger",
    "make_run_dir",
    "read_manifest",
    "read_metrics",
]
