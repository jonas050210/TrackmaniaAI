"""Model registry: named, versioned, self-contained AI models.

A checkpoint inside a run directory is tied to that run: prune the run and the model is
gone, rename the directory and its provenance is muddied. The registry gives trained
policies a *name* and a home of their own::

    models/<name>/
        model.json    name, source run/checkpoint, step, scores, tags, notes, created
        policy.pt     a copy of the checkpoint payload (self-contained)

Registering copies the checkpoint, so a model survives the run that produced it and can
be evaluated, compared, exported or deleted without touching run history. Everything is
plain files, so the registry is inspectable with any tool.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tmai.training.checkpoint import CheckpointError, load_checkpoint

logger = logging.getLogger(__name__)

REGISTRY_SCHEMA_VERSION = 1
MODEL_META_NAME = "model.json"
MODEL_WEIGHTS_NAME = "policy.pt"

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class RegistryError(ValueError):
    """The registry cannot perform the requested operation."""


def _validate_name(name: str) -> str:
    if not _NAME_PATTERN.match(name or ""):
        raise RegistryError(
            f"invalid model name {name!r}: use letters, digits, '.', '_' or '-', starting "
            "with a letter or digit, at most 64 characters"
        )
    return name


@dataclass
class ModelInfo:
    """One registered model."""

    name: str
    directory: Path
    source_checkpoint: str = ""
    source_run: str = ""
    step: int = 0
    gradient_steps: int = 0
    best_score: float | None = None
    created_utc: str = ""
    algorithm: str = ""
    observation_dim: int | None = None
    action_dim: int | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    evaluation: dict[str, Any] = field(default_factory=dict)

    @property
    def weights_path(self) -> Path:
        return self.directory / MODEL_WEIGHTS_NAME

    @property
    def meta_path(self) -> Path:
        return self.directory / MODEL_META_NAME

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "name": self.name,
            "directory": str(self.directory),
            "source_checkpoint": self.source_checkpoint,
            "source_run": self.source_run,
            "step": self.step,
            "gradient_steps": self.gradient_steps,
            "best_score": self.best_score,
            "created_utc": self.created_utc,
            "algorithm": self.algorithm,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "tags": list(self.tags),
            "notes": self.notes,
            "evaluation": self.evaluation,
            "weights": str(self.weights_path),
        }


class ModelStore:
    """A directory-backed registry of named models."""

    def __init__(self, root: str | Path = "models") -> None:
        self.root = Path(root)

    def _dir(self, name: str) -> Path:
        return self.root / _validate_name(name)

    # -- operations -----------------------------------------------------------------

    def register(
        self,
        name: str,
        checkpoint: str | Path,
        *,
        tags: list[str] | None = None,
        notes: str = "",
        evaluation: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> ModelInfo:
        """Register ``checkpoint`` under ``name``, copying it into the registry.

        The checkpoint is read (so a corrupt file fails here, not at eval time) and its
        payload copied, making the model self-contained.
        """
        _validate_name(name)
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise RegistryError(f"checkpoint not found: {checkpoint}")
        try:
            payload = load_checkpoint(checkpoint)
        except CheckpointError as exc:
            # A corrupt or truncated checkpoint is a registration failure, reported in the
            # registry's own error type so callers have one exception to catch.
            raise RegistryError(f"cannot read checkpoint {checkpoint}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - any read failure is a registration failure
            raise RegistryError(f"cannot read checkpoint {checkpoint}: {exc}") from exc

        directory = self._dir(name)
        if directory.exists() and not overwrite:
            raise RegistryError(
                f"a model named {name!r} already exists at {directory}; pass "
                "overwrite=True (or delete it first)"
            )
        directory.mkdir(parents=True, exist_ok=True)

        weights = directory / MODEL_WEIGHTS_NAME
        shutil.copyfile(checkpoint, weights)

        learner_state = payload.get("learner") or {}
        # A normalisation-wrapped learner stores its SAC state under "inner"; unwrap so the
        # registry always records the policy's own dimensions.
        if isinstance(learner_state.get("inner"), dict):
            learner_state = learner_state["inner"]
        config = payload.get("config") or {}
        sac = config.get("sac") or {}
        info = ModelInfo(
            name=name,
            directory=directory,
            source_checkpoint=str(checkpoint),
            source_run=str(checkpoint.parent) if checkpoint.parent.name else "",
            step=int(payload.get("step", 0)),
            gradient_steps=int(payload.get("gradient_steps", 0)),
            best_score=payload.get("best_score"),
            created_utc=datetime.now(timezone.utc).isoformat(),
            algorithm=str(learner_state.get("algorithm", sac.get("algorithm", "sac"))),
            observation_dim=learner_state.get("observation_dim"),
            action_dim=learner_state.get("action_dim"),
            tags=list(tags or []),
            notes=notes,
            evaluation=dict(evaluation or {}),
        )
        info.meta_path.write_text(
            json.dumps(info.as_dict(), indent=2, default=str), encoding="utf-8"
        )
        logger.info(
            "registered model %r (step %d, gradient steps %d) -> %s",
            name, info.step, info.gradient_steps, directory,
        )
        return info

    def list(self) -> list[ModelInfo]:
        """Every registered model, sorted by name."""
        if not self.root.is_dir():
            return []
        out: list[ModelInfo] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir() or not (child / MODEL_META_NAME).is_file():
                continue
            try:
                out.append(self._read(child))
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning("skipping unreadable model entry %s: %s", child, exc)
        return out

    def get(self, name: str) -> ModelInfo:
        directory = self._dir(name)
        if not (directory / MODEL_META_NAME).is_file():
            raise RegistryError(f"no model named {name!r} in {self.root}")
        return self._read(directory)

    def exists(self, name: str) -> bool:
        try:
            return (self._dir(name) / MODEL_META_NAME).is_file()
        except RegistryError:
            return False

    def delete(self, name: str) -> None:
        """Remove a model and its weights. Raises if the name is unknown."""
        info = self.get(name)
        shutil.rmtree(info.directory)
        logger.info("deleted model %r (%s)", name, info.directory)

    def add_tags(self, name: str, tags: Sequence[str]) -> ModelInfo:
        info = self.get(name)
        merged = [tag for tag in dict.fromkeys([*info.tags, *tags])]
        return self._update(name, tags=merged)

    def set_evaluation(self, name: str, evaluation: dict[str, Any]) -> ModelInfo:
        """Attach (or replace) the model's latest evaluation summary."""
        return self._update(name, evaluation=dict(evaluation))

    def set_notes(self, name: str, notes: str) -> ModelInfo:
        return self._update(name, notes=notes)

    # -- internals ------------------------------------------------------------------

    def _read(self, directory: Path) -> ModelInfo:
        data = json.loads((directory / MODEL_META_NAME).read_text(encoding="utf-8"))
        return ModelInfo(
            name=str(data.get("name", directory.name)),
            directory=directory,
            source_checkpoint=str(data.get("source_checkpoint", "")),
            source_run=str(data.get("source_run", "")),
            step=int(data.get("step", 0)),
            gradient_steps=int(data.get("gradient_steps", 0)),
            best_score=data.get("best_score"),
            created_utc=str(data.get("created_utc", "")),
            algorithm=str(data.get("algorithm", "")),
            observation_dim=data.get("observation_dim"),
            action_dim=data.get("action_dim"),
            tags=list(data.get("tags") or []),
            notes=str(data.get("notes", "")),
            evaluation=dict(data.get("evaluation") or {}),
        )

    def _update(self, name: str, **fields: Any) -> ModelInfo:
        info = self.get(name)
        for key, value in fields.items():
            setattr(info, key, value)
        info.meta_path.write_text(
            json.dumps(info.as_dict(), indent=2, default=str), encoding="utf-8"
        )
        return info


__all__ = ["ModelInfo", "ModelStore", "RegistryError"]
