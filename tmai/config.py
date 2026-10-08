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
from tmai.training.curriculum import CurriculumSpec

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
    """Where the track centreline(s) come from.

    Exactly one of ``path``, ``directory`` or ``synthetic`` should be set. ``directory`` is
    the multi-track case: every centreline file in it becomes one track in a
    :class:`~tmai.tracks.library.TrackLibrary`, and the split weights below decide which
    tracks are trained on and which are held out.
    """

    #: Path to a recorded centreline JSON (produced by ``tmai record-track``).
    path: str | None = None
    #: Directory of centreline JSON files. Builds a multi-track library.
    directory: str | None = None
    #: Glob used inside ``directory``.
    pattern: str = "*.json"
    #: Search ``directory`` recursively.
    recursive: bool = False
    #: Name of a procedurally generated track, used when neither path nor directory is set.
    #: Only useful with the simulated driver or for smoke tests.
    synthetic: str | None = None
    synthetic_kwargs: dict[str, Any] = field(default_factory=dict)
    #: Several synthetic tracks at once, for generalisation tests without recorded maps.
    #: Each entry is ``{"name": "oval", "kwargs": {...}}``.
    synthetic_suite: list[dict[str, Any]] = field(default_factory=list)

    #: Relative split sizes for tracks that have no explicit assignment. Must sum to 1.
    split_weights: dict[str, float] = field(
        default_factory=lambda: {"train": 0.7, "validation": 0.15, "test": 0.15}
    )
    #: Filename stem -> split, for pinning particular maps to ``test``.
    explicit_splits: dict[str, str] = field(default_factory=dict)


@dataclass
class NormalizeSpec:
    """Online observation normalisation.

    Applied inside a :class:`~tmai.agents.normalize.NormalizingLearner`, so the replay buffer
    keeps raw observations and stored transitions stay valid as the statistics improve.
    """

    enabled: bool = True
    #: Symmetric clip after normalisation. Bounds the damage from a single outlier frame
    #: (a respawn teleport) without discarding ordinary variation.
    clip: float = 10.0
    epsilon: float = 1e-4
    #: Samples before the running standard deviation is trusted; until then observations are
    #: centred only. Dividing by a provisional near-zero std would inject huge inputs.
    warmup_steps: int = 100


@dataclass
class MultiTrackSpec:
    """How tracks and episode start conditions are sampled during training."""

    #: Sample a different track on every episode reset. This is the primary defence against
    #: memorising a single map; turning it off trains on the first track only.
    sample_tracks: bool = True
    #: Randomise the arc-length position each episode starts from.
    #:
    #: Requires a driver that supports start repositioning. The real game does not (see
    #: ``DriverCapabilities.supports_start_repositioning``), so real-game runs must set this
    #: to false -- the environment raises rather than silently starting at the start line.
    random_start_station: bool = True
    #: Fraction of the lap the random start may cover.
    start_station_fraction: float = 1.0
    #: Standard deviation of the random lateral start offset, metres.
    start_lateral_std: float = 1.5
    #: Metres of corridor edge kept clear of the random start offset.
    start_edge_margin: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_tracks": self.sample_tracks,
            "random_start_station": self.random_start_station,
            "start_station_fraction": self.start_station_fraction,
            "start_lateral_std": self.start_lateral_std,
            "start_edge_margin": self.start_edge_margin,
        }


@dataclass
class BCSpec:
    """Behaviour-cloning pretraining from recorded human demonstrations.

    SAC starts from a random policy. When demonstrations exist (recorded with
    ``tmai record-demo``), a supervised pretraining pass teaches the policy network the
    human's action mapping first, which typically shortens the aimless early phase
    considerably. Purely additive: with no demonstrations configured, nothing changes.
    """

    #: Master switch.
    enabled: bool = False
    #: Demonstration files (JSONL, one ``{"observation": [...], "action": [...]}`` per line).
    demo_paths: list[str] = field(default_factory=list)
    #: Supervised epochs over the demonstrations before RL starts.
    epochs: int = 10
    #: Minibatch size for the supervised updates.
    batch_size: int = 256
    #: Learning rate for the supervised updates (separate from the SAC actor LR).
    lr: float = 1e-3
    #: Fraction of demonstrations held out for a validation loss report, in ``[0, 1)``.
    val_fraction: float = 0.1
    #: Shuffle the demonstrations each epoch (recommended; seeded, so reproducible).
    shuffle: bool = True

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.enabled:
            return problems
        if not self.demo_paths:
            problems.append("bc.enabled is true but no demo_paths are configured")
        if self.epochs < 1:
            problems.append(f"bc.epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            problems.append(f"bc.batch_size must be >= 1, got {self.batch_size}")
        if self.lr <= 0:
            problems.append(f"bc.lr must be positive, got {self.lr}")
        if not 0.0 <= self.val_fraction < 1.0:
            problems.append(f"bc.val_fraction must be in [0, 1), got {self.val_fraction}")
        return problems


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
    #: Evaluate on held-out tracks every this many steps. ``0`` disables it. This is the
    #: number that distinguishes learning to drive from memorising one map.
    held_out_eval_interval: int = 0
    #: Episodes per held-out evaluation.
    held_out_eval_episodes: int = 2
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
    #: Record per-episode trajectories (positions, speeds, actions, rewards) as replays
    #: under ``<run>/replays/``. Powers replay/ghost analysis in the GUI.
    record_replays: bool = False
    #: Keep only every Nth step of a recorded replay (storage/decimation control).
    replay_decimation: int = 2
    #: Maximum number of replay files kept per run (the newest are kept).
    max_replays: int = 200
    #: Where registered models live (see ``tmai models``).
    model_store: str = "models"


@dataclass
class RunConfig:
    """The complete configuration of one training run."""

    driver: DriverSpec = field(default_factory=DriverSpec)
    track: TrackSpec = field(default_factory=TrackSpec)
    multi: MultiTrackSpec = field(default_factory=MultiTrackSpec)
    normalize: NormalizeSpec = field(default_factory=NormalizeSpec)
    env: EnvConfig = field(default_factory=EnvConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    replay: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    train: TrainSpec = field(default_factory=TrainSpec)
    #: Progressive track reveal / episode-length curriculum over training steps.
    curriculum: CurriculumSpec = field(default_factory=CurriculumSpec)
    #: Behaviour-cloning pretraining from recorded human demonstrations.
    bc: BCSpec = field(default_factory=BCSpec)

    # -- validation -----------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of human-readable configuration problems (empty means valid).

        Called before a run starts. Catching a bad configuration here costs nothing; catching
        it three hours into a training run costs three hours.
        """
        problems: list[str] = []

        sources = [
            bool(self.track.path),
            bool(self.track.directory),
            bool(self.track.synthetic),
            bool(self.track.synthetic_suite),
        ]
        if sum(sources) == 0:
            problems.append(
                "no track configured: set track.path, track.directory, track.synthetic "
                "or track.synthetic_suite"
            )
        elif sum(sources) > 1:
            problems.append(
                "exactly one track source must be set; got "
                + ", ".join(
                    name
                    for name, set_ in zip(
                        ("track.path", "track.directory", "track.synthetic",
                         "track.synthetic_suite"),
                        sources,
                        strict=True,
                    )
                    if set_
                )
            )

        total = sum(self.track.split_weights.values())
        if abs(total - 1.0) > 1e-6:
            problems.append(f"track.split_weights must sum to 1.0, got {total:.4f}")
        for name, weight in self.track.split_weights.items():
            if weight < 0:
                problems.append(f"track.split_weights[{name!r}] must be non-negative")

        if self.driver.kind not in ("tminterface", "simulated"):
            problems.append(
                f"driver.kind must be 'tminterface' or 'simulated', got {self.driver.kind!r}"
            )
        if self.driver.kind == "simulated" and not self.driver.allow_simulated:
            problems.append(
                "driver.kind='simulated' is a toy model, not Trackmania: set "
                "driver.allow_simulated=true (CLI: --allow-simulated-driver) to proceed"
            )
        if self.driver.speed_ratio <= 0:
            problems.append(f"driver.speed_ratio must be positive, got {self.driver.speed_ratio}")
        if self.driver.position_scale <= 0:
            problems.append(
                f"driver.position_scale must be positive, got {self.driver.position_scale}"
            )

        # Start randomisation needs a driver that can actually move the car.
        wants_random_start = (
            self.multi.random_start_station or self.multi.start_lateral_std > 0
        )
        if self.driver.kind == "tminterface" and wants_random_start:
            problems.append(
                "multi.random_start_station / start_lateral_std require a driver that can "
                "reposition the car, and the real game cannot. Set "
                "multi.random_start_station=false and multi.start_lateral_std=0 for "
                "real-game runs (see docs/LIMITATIONS.md)."
            )

        if self.env.control_dt <= 0:
            problems.append(f"env.control_dt must be positive, got {self.env.control_dt}")
        if self.env.action_repeat < 1:
            problems.append(f"env.action_repeat must be >= 1, got {self.env.action_repeat}")
        if self.env.observation.dim <= 0:
            problems.append(
                "the observation spec produced an empty vector; enable at least one feature"
            )
        for problem in self.env.observation.validate():
            problems.append(f"env.observation: {problem}")
        for problem in self.env.termination.validate():
            problems.append(f"env.termination: {problem}")
        for problem in self.curriculum.validate():
            problems.append(problem)
        for problem in self.bc.validate():
            problems.append(problem)

        t = self.train
        if t.total_steps <= 0:
            problems.append(f"train.total_steps must be positive, got {t.total_steps}")
        if t.batch_size <= 0:
            problems.append(f"train.batch_size must be positive, got {t.batch_size}")
        if t.warmup_steps < 0:
            problems.append(f"train.warmup_steps must be non-negative, got {t.warmup_steps}")
        if t.warmup_steps >= t.total_steps:
            problems.append(
                f"train.warmup_steps ({t.warmup_steps}) must be below train.total_steps "
                f"({t.total_steps}) or no gradient step ever happens"
            )
        if t.updates_per_step <= 0:
            problems.append(f"train.updates_per_step must be positive, got {t.updates_per_step}")
        if t.keep_checkpoints < 1:
            problems.append(f"train.keep_checkpoints must be >= 1, got {t.keep_checkpoints}")
        if t.eval_interval and t.eval_interval > t.total_steps:
            problems.append(
                f"train.eval_interval ({t.eval_interval}) exceeds train.total_steps "
                f"({t.total_steps}); the run would never evaluate"
            )
        if self.replay.capacity < t.batch_size:
            problems.append(
                f"replay.capacity ({self.replay.capacity}) is below train.batch_size "
                f"({t.batch_size}); no minibatch could ever be drawn"
            )
        if t.replay_decimation < 1:
            problems.append(f"train.replay_decimation must be >= 1, got {t.replay_decimation}")
        if t.max_replays < 0:
            problems.append(f"train.max_replays must be non-negative, got {t.max_replays}")
        if self.sac.gamma <= 0 or self.sac.gamma >= 1:
            problems.append(f"sac.gamma must be in (0, 1), got {self.sac.gamma}")
        if self.sac.tau <= 0 or self.sac.tau > 1:
            problems.append(f"sac.tau must be in (0, 1], got {self.sac.tau}")

        return problems

    def validate_or_raise(self) -> None:
        problems = self.validate()
        if problems:
            raise ValueError(
                "invalid configuration:\n  - " + "\n  - ".join(problems)
            )

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
        return RunConfig.from_yaml_text(path.read_text(encoding="utf-8"))

    @staticmethod
    def from_yaml_text(text: str) -> RunConfig:
        """Parse a config from a YAML document (the GUI's config editor posts text)."""
        data = yaml.safe_load(text) or {}
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
    "MultiTrackSpec",
    "NormalizeSpec",
    "RunConfig",
    "TrackSpec",
    "TrainSpec",
    "parse_overrides",
]
