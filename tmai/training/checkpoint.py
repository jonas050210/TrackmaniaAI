"""Checkpointing for long training runs.

A run that takes days will be interrupted, so checkpoints must be complete enough to resume a
run without losing its place: network weights, both optimisers, the entropy temperature, the
gradient-step counter, the replay buffer size, the RNG states and the config. Everything is
written to a temporary file and atomically renamed, so a crash mid-write cannot leave a corrupt
"latest" checkpoint behind.

One thing is deliberately *not* stored: the replay buffer contents. Only its size is recorded,
so a resumed run refills the buffer from fresh interaction and therefore does not reproduce the
uninterrupted trajectory step for step. Saving a million-transition buffer into every
checkpoint of a multi-day run is not a trade worth making; the learner state and the RNG are
what actually matter for continuing to learn. ``tests/test_reproducibility.py`` pins both the
guarantee and this limitation.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tmai.agents.base import Learner

logger = logging.getLogger(__name__)

CHECKPOINT_EXTENSION = ".pt"
CHECKPOINT_PREFIX = "checkpoint"
BEST_PREFIX = "best"


class CheckpointError(ValueError):
    """A checkpoint file exists but cannot be read as a payload of this build.

    Raised for a truncated, zero-byte or otherwise damaged file. ``torch.load`` would surface
    these as a bare ``OSError``/``EOFError``/``UnpicklingError`` naming no file and suggesting no
    remedy, which is exactly the wrong report to hand someone resuming a multi-day run.
    """


@dataclass
class CheckpointMeta:
    path: Path
    step: int
    gradient_steps: int
    episode: int
    best_score: float | None
    created_utc: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "step": self.step,
            "gradient_steps": self.gradient_steps,
            "episode": self.episode,
            "best_score": self.best_score,
            "created_utc": self.created_utc,
        }


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def seed_spaces(env: Any) -> None:
    """Re-seed an environment's gymnasium spaces from the *current* global RNG state.

    Gymnasium spaces own a private generator that ``np.random.seed`` and ``torch.manual_seed``
    do not reach, so ``action_space.sample()`` (used for warm-up exploration) stays
    non-reproducible unless the spaces are seeded explicitly. Deriving the seed from the global
    state rather than taking one as an argument is what makes this work identically for a fresh
    run and for a resume: both call it right after the global state is established.
    """
    if env is None:
        return
    for name in ("action_space", "observation_space"):
        space = getattr(env, name, None)
        if space is not None and hasattr(space, "seed"):
            space.seed(int(np.random.randint(0, 2**31 - 1)))


def seed_everything(seed: int | None, *, env: Any = None) -> None:
    """Seed every RNG a training run touches, so a rerun of a seed reproduces bit for bit.

    Seeding the learner alone is not enough: warm-up actions come from
    ``env.action_space.sample()``, and the track sampler and normaliser draw from numpy.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    seed_spaces(env)


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except RuntimeError:  # pragma: no cover - device count changed
            logger.warning("could not restore CUDA RNG state (device count changed?)")


def save_checkpoint(
    directory: str | Path,
    *,
    step: int,
    learner: Learner,
    episode: int = 0,
    buffer_size: int = 0,
    config: dict[str, Any] | None = None,
    best_score: float | None = None,
    extra: dict[str, Any] | None = None,
    keep: int = 3,
    rng_state: bool = True,
) -> Path:
    """Write ``checkpoint_<step>.pt`` and prune older ones down to ``keep``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    from datetime import datetime, timezone

    payload: dict[str, Any] = {
        "schema_version": 1,
        "step": int(step),
        "episode": int(episode),
        "gradient_steps": int(getattr(learner, "gradient_steps", 0)),
        "buffer_size": int(buffer_size),
        "best_score": best_score,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "learner": learner.state_dict(),
        "config": config or {},
        "extra": extra or {},
    }
    if rng_state:
        payload["rng"] = _rng_state()

    path = directory / f"{CHECKPOINT_PREFIX}_{step:09d}{CHECKPOINT_EXTENSION}"
    _atomic_torch_save(payload, path)
    _prune(directory, keep=keep, prefix=CHECKPOINT_PREFIX)
    logger.info(
        "saved checkpoint %s (gradient_steps=%d, buffer=%d)",
        path.name,
        payload["gradient_steps"],
        buffer_size,
    )
    return path


def save_best(
    directory: str | Path,
    *,
    step: int,
    learner: Learner,
    score: float,
    episode: int = 0,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save (overwrite) the best-scoring checkpoint so far."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    payload = {
        "schema_version": 1,
        "step": int(step),
        "episode": int(episode),
        "gradient_steps": int(getattr(learner, "gradient_steps", 0)),
        "best_score": float(score),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "learner": learner.state_dict(),
        "config": config or {},
        "extra": {"reason": "best_score", **(extra or {})},
    }
    path = directory / f"{BEST_PREFIX}{CHECKPOINT_EXTENSION}"
    _atomic_torch_save(payload, path)
    return path


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint payload.

    A file that is present but unreadable is reported as a :class:`CheckpointError` naming the
    file, its size and the underlying cause, rather than letting ``torch.load`` escape with a
    bare ``OSError``/``EOFError`` that points at neither.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except (OSError, EOFError, pickle.UnpicklingError, RuntimeError, TypeError) as exc:
        size = path.stat().st_size if path.exists() else -1
        raise CheckpointError(
            f"checkpoint is unreadable and cannot be resumed: {path} ({size} bytes). "
            f"Cause: {type(exc).__name__}: {exc}. "
            "The file is truncated or damaged; delete it and resume from an earlier "
            f"{CHECKPOINT_PREFIX}_N{CHECKPOINT_EXTENSION}, or start a fresh run."
        ) from exc
    if not isinstance(payload, dict):
        raise CheckpointError(
            f"checkpoint does not contain a payload mapping: {path} "
            f"(got {type(payload).__name__}); expected a dict written by this build."
        )
    version = int(payload.get("schema_version", 0))
    if version != 1:
        raise ValueError(f"unsupported checkpoint schema version {version}; this build reads 1")
    return payload


def restore_rng(payload: dict[str, Any]) -> None:
    _restore_rng_state(payload.get("rng"))


def latest_checkpoint(directory: str | Path, *, prefix: str = CHECKPOINT_PREFIX) -> Path | None:
    """Highest-numbered checkpoint in ``directory``, or ``None``."""
    directory = Path(directory)
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(f"{prefix}_*{CHECKPOINT_EXTENSION}"))
    return candidates[-1] if candidates else None


def list_checkpoints(directory: str | Path, *, prefix: str = CHECKPOINT_PREFIX) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.glob(f"{prefix}_*{CHECKPOINT_EXTENSION}"))


def read_meta(path: str | Path) -> CheckpointMeta:
    payload = load_checkpoint(path)
    return CheckpointMeta(
        path=Path(path),
        step=int(payload.get("step", 0)),
        gradient_steps=int(payload.get("gradient_steps", 0)),
        episode=int(payload.get("episode", 0)),
        best_score=payload.get("best_score"),
        created_utc=str(payload.get("created_utc", "")),
    )


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _prune(directory: Path, *, keep: int, prefix: str) -> None:
    if keep <= 0:
        return
    for path in list_checkpoints(directory, prefix=prefix)[:-keep]:
        try:
            path.unlink()
        except OSError:  # pragma: no cover - best effort
            logger.warning("could not prune old checkpoint %s", path, exc_info=True)


def write_index(directory: str | Path, entries: list[dict[str, Any]]) -> Path:
    """Write a small JSON index of checkpoints (used by the future GUI)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "checkpoints.json"
    path.write_text(json.dumps(entries, indent=2, default=str), encoding="utf-8")
    return path


__all__ = [
    "BEST_PREFIX",
    "CHECKPOINT_EXTENSION",
    "CHECKPOINT_PREFIX",
    "CheckpointMeta",
    "latest_checkpoint",
    "list_checkpoints",
    "load_checkpoint",
    "read_meta",
    "restore_rng",
    "save_best",
    "save_checkpoint",
    "write_index",
]
