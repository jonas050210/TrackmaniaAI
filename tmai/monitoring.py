"""System resource monitoring for training runs and the GUI's diagnostics page.

Stdlib-only by design: the training host may be a bare container, so nothing here requires
psutil. Every probe degrades to ``None`` when the information is unavailable (a platform
without ``/proc``, no CUDA, ...), and the caller renders "n/a" rather than guessing.

Two consumers:

* the trainer, which logs ``system/*`` metrics into the run's metrics stream so resource
  usage is visible alongside reward curves in any dashboard;
* the server, which exposes the same snapshot for a live diagnostics view.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from typing import Any


def _meminfo() -> dict[str, float]:
    """Memory counters (bytes) from ``/proc/meminfo``; empty when unavailable."""
    try:
        values: dict[str, float] = {}
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":"):
                    try:
                        values[parts[0][:-1]] = float(parts[1]) * 1024.0
                    except ValueError:
                        continue
        return values
    except OSError:
        return {}


def _process_memory_bytes() -> float | None:
    """This process's resident set size, from ``/proc/self/status``."""
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) * 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def _torch_cuda() -> dict[str, Any]:
    """CUDA memory counters, when torch and a GPU are both present."""
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch is an optional dependency
        return {}
    if not torch.cuda.is_available():
        return {}
    try:
        return {
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_memory_allocated": float(torch.cuda.memory_allocated(0)),
            "cuda_memory_reserved": float(torch.cuda.memory_reserved(0)),
        }
    except Exception:  # noqa: BLE001 - never let monitoring break training
        return {}


def system_metrics() -> dict[str, Any]:
    """A snapshot of host resources. Values are ``None`` when unavailable."""
    mem = _meminfo()
    total = mem.get("MemTotal")
    available = mem.get("MemAvailable")
    metrics: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "memory_total_bytes": total,
        "memory_available_bytes": available,
        "memory_used_fraction": (
            round(1.0 - available / total, 4) if total and available is not None else None
        ),
        "process_memory_bytes": _process_memory_bytes(),
        "disk_free_bytes": None,
        "disk_total_bytes": None,
    }
    try:
        usage = shutil.disk_usage(os.getcwd())
        metrics["disk_free_bytes"] = usage.free
        metrics["disk_total_bytes"] = usage.total
    except OSError:
        pass
    metrics.update(_torch_cuda())
    return metrics


def flat_system_metrics(prefix: str = "system") -> dict[str, float]:
    """The snapshot flattened to scalar floats for the metrics stream.

    ``None`` values are skipped: a missing counter must not appear as a fake 0.
    """
    out: dict[str, float] = {}
    snapshot = system_metrics()
    for key, value in snapshot.items():
        if value is None or isinstance(value, (str, list, dict)):
            continue
        out[f"{prefix}/{key}"] = float(value)
    if snapshot.get("load_average"):
        for i, load in enumerate(snapshot["load_average"]):
            out[f"{prefix}/load{i + 1}"] = float(load)
    return out


__all__ = ["flat_system_metrics", "system_metrics"]
