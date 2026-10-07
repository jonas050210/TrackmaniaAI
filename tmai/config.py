"""Typed run configuration.

Everything that can change a run's outcome lives in one :class:`RunConfig`, which is loaded
from YAML, overridable from the command line (``--set sac.gamma=0.98``) and written verbatim
into the run manifest. That is the reproducibility contract: a run directory alone is enough
to know exactly what was run.

Dataclasses (rather than a schema library) keep this dependency-free and make the config
introspectable and testable.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

from tmai.agents.replay import ReplayBufferConfig
from tmai.agents.sac import SACConfig
from tmai.env.tm_env import EnvConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class DriverSpec:
    """Which game driver to use and how to connect to it."""

    #: ``"tminterface"`` for the real game, ``"simulated"`` for the labelled test double.
    kind: str = "tminterface"
    server_name: str = "TMInterface0"
    #: Game position units -> metres. Calibrate with ``tmai doctor --calibrate``.
    position_scale: float = 1.0
    #: Requested game-speed multiplier (training throughput).
    speed_ratio: float = 1.0
    reset_strategy: str = "respawn"
    reset_command: str = "respawn"
    settle_ticks: int = 5
    connect_timeout_s: float = 20.0
    frame_timeout_s: float = 30.0
    #: Refuse to start unless the operator explicitly allows the simulated driver.
    allow_simulated: bool = False


@dataclass
class TrackSpec:
    """Where the track centreline comes from."""

    #: Path to a recorded centreline JSON (produced by ``tmai record-track``).
    path: str | None = None
    #: Name of a procedurally generated track, used when ``path`` is not set. Only useful
    #: with the simulated driver or for smoke tests.
    synthetic: str | None = None
    synthetic_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainSpec:
    """Training loop settings."""

    total_steps: int = 100_000
    #: Random actions collected before the first gradient step.
    warmup_steps: int = 1_000
    batch_size: int = 256
    #: Gradient steps per environment step (the update-to-data ratio).
    updates_per_step: float = 1.0
    #: Perform updates every ``update_every`` environment steps.
    update_every: int = 1
    log_interval: int = 100
    eval_interval: int = 5_000
    eval_episodes: int = 3
    checkpoint_interval: int = 5_000
    keep_checkpoints: int = 3
    seed: int = 0
    #: ``"cpu"``, ``"cuda"`` or ``None`` for auto.
    device: str | None = None
    run_name: str = "tmai"
    output_dir: str = "runs"
    #: Run directory to resume from (``None`` for a fresh run).
    resume: str | None = None
    #: Stop after this many wall-clock seconds (useful for CI and for scheduled runs).
    max_wall_seconds: float | None = None


@dataclass
class RunConfig:
    """The complete configuration of one training run."""

    driver: DriverSpec = field(default_factory=DriverSpec)
    track: TrackSpec = field(default_factory=TrackSpec)
    env: EnvConfig = field(default_factory=EnvConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    replay: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    train: TrainSpec = field(default_factory=TrainSpec)

    # -- serialisation --------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")
        return path

    @staticmethod
    def from_dict(data: dict[str, Any]) -> RunConfig:
        return _from_dict(RunConfig, data)  # type: ignore[return-value]

    @staticmethod
    def from_yaml(path: str | Path) -> RunConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"config root must be a mapping, got {type(data).__name__}")
        return RunConfig.from_dict(data)

    def apply_overrides(self, overrides: dict[str, Any]) -> RunConfig:
        """Return a copy with ``{"sac.gamma": 0.98}``-style overrides applied."""
        config = copy.deepcopy(self)
        for dotted, value in overrides.items():
            _set_dotted(config, dotted, value)
        return config


# -- generic dataclass <-> dict conversion -------------------------------------------


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_to_dict(v) for v in obj]
        return converted if isinstance(obj, list) else tuple(converted)
    return obj


def _from_dict(cls: type[T], data: Any) -> T:
    """Recursively build ``cls`` from a plain mapping, ignoring unknown keys with a warning."""
    if not is_dataclass(cls):
        return data  # type: ignore[return-value]
    if data is None:
        return cls()  # type: ignore[call-arg]
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} expects a mapping, got {type(data).__name__}")

    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    for key in data:
        if key not in known:
            logger.warning("ignoring unknown config key %s.%s", cls.__name__, key)

    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            continue
        kwargs[key] = _coerce(hints[key], value, path=f"{cls.__name__}.{key}")
    return cls(**kwargs)  # type: ignore[call-arg]


def _coerce(hint: Any, value: Any, *, path: str) -> Any:
    """Convert a YAML value into the type declared by the dataclass field."""
    if value is None:
        return None

    origin = get_origin(hint)
    args = get_args(hint)

    # Optional[X] / X | None
    if origin is not None and type(None) in args:
        inner = [a for a in args if a is not type(None)]  # noqa: E721
        if len(inner) == 1:
            return _coerce(inner[0], value, path=path)
    if hint is type(None):  # noqa: E721
        return None
    if origin is None and hasattr(hint, "__origin__") is False and isinstance(hint, type):
        if is_dataclass(hint):
            return _from_dict(hint, value)
        if issubclass(hint, Enum):
            return hint(value)
        if hint in (int, float, str, bool):
            if isinstance(value, str):
                return _parse_scalar(hint, value, path=path)
            return hint(value)
        if hint in (dict, list):
            return value
    if origin in (list, tuple):
        item = args[0] if args else Any
        items = [_coerce(item, v, path=path) for v in value]
        return tuple(items) if origin is tuple else items
    if origin is dict:
        return dict(value)
    return value


def _parse_scalar(target: type, text: str, *, path: str) -> Any:
    lowered = text.strip().lower()
    if target is bool:
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{path}: cannot interpret {text!r} as a boolean")
    if target is int:
        try:
            return int(text, 0)
        except ValueError as exc:
            raise ValueError(f"{path}: cannot interpret {text!r} as an integer") from exc
    if target is float:
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{path}: cannot interpret {text!r} as a float") from exc
    return text


def _set_dotted(config: Any, dotted: str, value: Any) -> None:
    """Apply one ``a.b.c=value`` override, coercing to the declared field type."""
    parts = dotted.split(".")
    node = config
    for part in parts[:-1]:
        if not is_dataclass(node):
            raise KeyError(f"cannot descend into {part!r} of {dotted}: not a config object")
        if not hasattr(node, part):
            raise KeyError(f"unknown config path {dotted!r} (no field {part!r})")
        node = getattr(node, part)
    leaf = parts[-1]
    if not is_dataclass(node) or not hasattr(node, leaf):
        raise KeyError(f"unknown config path {dotted!r}")

    declared = get_type_hints(type(node)).get(leaf)
    coerced = _coerce(declared, value, path=dotted) if declared is not None else value
    if (
        isinstance(coerced, str)
        and isinstance(declared, type)
        and get_origin(declared) is None
        and declared in (int, float, bool)
    ):
        coerced = _parse_scalar(declared, coerced, path=dotted)
    setattr(node, leaf, coerced)
    logger.debug("override %s = %r", dotted, coerced)


def parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """Parse ``["sac.gamma=0.98", ...]`` into a mapping."""
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"--set expects a non-empty key, got {pair!r}")
        out[key] = yaml.safe_load(raw) if raw.strip() else None
    return out


__all__ = [
    "DriverSpec",
    "RunConfig",
    "TrackSpec",
    "TrainSpec",
    "parse_overrides",
]
