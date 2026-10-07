"""``GameDriver`` implementation for the real Trackmania game via TMInterface.

This module contains the *logic* of driving the real game (reset sequencing, tick
decimation, finish handling, capability reporting) and depends only on the
:class:`~tmai.game.ticksource.TickSource` protocol. The Windows/IPC specifics live in
:mod:`tmai.game.tminterface.session`.

Real-game behaviour worth knowing before reading the code:

* TMInterface applies an input set during tick *t* at tick *t+1* (documented in
  ``TMInterface.set_input_state``). ``step()`` therefore returns the frame observed at the
  tick after the input was injected, which is the standard one-tick (10 ms) actuation
  latency of this integration.
* The game parks its simulation while it waits for our tick handler to reply, so a slow
  policy slows the game down instead of losing actions. This is a feature for training.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from tmai.game.errors import GameConnectionError, GameTimeoutError, UnsupportedFeatureError
from tmai.game.protocol import (
    Action,
    DriverCapabilities,
    GameDriver,
    GameFrame,
    RacePhase,
)
from tmai.game.ticksource import TickSource
from tmai.game.tminterface.ops import CommandOp, RespawnOp, SetSpeedOp

logger = logging.getLogger(__name__)


class ResetStrategy(str, Enum):
    """How an episode is restarted in the real game."""

    #: ``TMInterface.respawn()`` -- respawns at the last checkpoint. Reliable, verified API.
    RESPAWN = "respawn"
    #: An arbitrary TMInterface console command, for hosts where a full race restart is
    #: needed. The command string is deployment-specific and must be validated in-game.
    COMMAND = "command"


@dataclass
class TMInterfaceDriverConfig:
    """Driver-level settings (connection settings live on the session config)."""

    reset_strategy: ResetStrategy = ResetStrategy.RESPAWN
    reset_command: str = "respawn"
    #: Consecutive ``RUNNING`` ticks required before ``reset()`` returns, so that the
    #: countdown has actually finished and the car is settled on the start pad.
    settle_ticks: int = 5
    frame_timeout_s: float = 30.0
    #: Game-speed multiplier requested on ``open()``.
    default_speed_ratio: float = 1.0


class TMInterfaceDriver:
    """Drives the real Trackmania instance through a :class:`TickSource`.

    Args:
        session: the tick source (the real TMInterface session in production).
        config: driver settings.
        own_session: when true (default), ``open()``/``close()`` start and stop the
            session. Set to false to drive a session owned by someone else.
    """

    name = "tminterface"

    def __init__(
        self,
        session: TickSource,
        config: TMInterfaceDriverConfig | None = None,
        *,
        own_session: bool = True,
    ) -> None:
        self.session = session
        self.config = config or TMInterfaceDriverConfig()
        self._own_session = own_session
        self._opened = False
        self._applied_speed_ratio = 1.0

    # -- GameDriver -----------------------------------------------------------------

    @property
    def capabilities(self) -> DriverCapabilities:
        return DriverCapabilities(
            analog_control=True,
            game_speed_control=True,
            deterministic_reset=False,
            reports_checkpoints=True,
            reports_finish=True,
            reports_sliding=True,
            headless_capable=False,
            max_speed_ratio=20.0,
            notes=(
                "TMInterface injects analog steer/gas into the game's input layer; brake is "
                "binary on the standard bindings.",
                "Requires Windows + Trackmania launched through TMInterface.",
                "Game speed >~20x risks dropped inputs per upstream documentation.",
            ),
        )

    def open(self) -> None:
        if self._opened:
            return
        self.session.start()
        if not self.session.is_alive:
            raise GameConnectionError("TMInterface session is not alive after start()")
        # Make sure the tick stream actually flows before we claim success.
        first = self.session.next_frame(self.config.frame_timeout_s)
        logger.info(
            "TMInterface connected; first frame: race_time=%.3fs phase=%s cp=%d/%d",
            first.race.race_time,
            first.race.phase.value,
            first.race.checkpoint_index,
            first.race.checkpoint_total,
        )
        # Mark the driver open before applying the default speed ratio: set_speed_ratio()
        # requires an open driver, and this is part of opening.
        self._opened = True
        if self.config.default_speed_ratio != 1.0:
            self.set_speed_ratio(self.config.default_speed_ratio)

    def close(self) -> None:
        self._opened = False
        if self._own_session:
            self.session.stop()

    def is_connected(self) -> bool:
        return self._opened and self.session.is_alive

    def reset(self) -> GameFrame:
        """Restart the attempt and wait until the car is live and controllable."""
        self._require_open()

        if self.config.reset_strategy is ResetStrategy.COMMAND:
            self.session.request_op(CommandOp(self.config.reset_command))
        else:
            self.session.request_op(RespawnOp())

        deadline = time.monotonic() + self.config.frame_timeout_s
        settled = 0
        while True:
            if time.monotonic() > deadline:
                raise GameTimeoutError(
                    f"car did not reach a RUNNING state within {self.config.frame_timeout_s:.0f}s "
                    "after reset (is a map loaded and a race started?)"
                )
            frame = self.session.next_frame(self.config.frame_timeout_s)
            if frame.race.phase is RacePhase.RUNNING:
                settled += 1
                if settled >= max(1, self.config.settle_ticks):
                    logger.debug(
                        "reset complete after %d settled ticks (race_time=%.3fs)",
                        settled,
                        frame.race.race_time,
                    )
                    return frame
            else:
                settled = 0

    def reposition(self, station: float, lateral: float = 0.0) -> GameFrame:
        """Not supported by the real game integration.

        Trackmania respawns the car to the last checkpoint; it cannot be placed at an
        arbitrary point on the centreline without a recorded ``CheckpointData`` state for that
        point. Reporting ``supports_start_repositioning=False`` and raising here is
        deliberate: a caller that asked for a randomised start must find out, rather than
        quietly training on start-line episodes it believes were randomised.

        Recording checkpoint states per map would make this possible; see docs/ROADMAP.md.
        """
        raise UnsupportedFeatureError(
            "the TMInterface driver cannot place the car at an arbitrary track station",
            remedy=(
                "Disable env start randomisation for real-game runs "
                "(multi.random_start_station=false), or record per-map checkpoint states."
            ),
        )

    def step(self, action: Action) -> GameFrame:
        """Send ``action`` to the game and return the frame produced by it.

        The action is clipped here rather than only in the session: the ``GameDriver``
        contract promises the game never sees an out-of-range input, and a policy that emits
        a slightly out-of-bounds value must not be able to violate that.
        """
        self._require_open()
        self.session.push_action(action.clipped())
        return self.session.next_frame(self.config.frame_timeout_s)

    def set_speed_ratio(self, ratio: float) -> float:
        """Request a game-speed multiplier; returns the ratio actually requested."""
        self._require_open()
        if ratio <= 0:
            raise ValueError(f"speed ratio must be positive, got {ratio}")
        if ratio > self.capabilities.max_speed_ratio:
            logger.warning(
                "speed ratio %.1f exceeds the recommended maximum %.1f; inputs may be dropped",
                ratio,
                self.capabilities.max_speed_ratio,
            )
        self.session.request_op(SetSpeedOp(ratio))
        self._applied_speed_ratio = float(ratio)
        return float(ratio)

    def describe(self) -> dict[str, Any]:
        return {
            "driver": self.name,
            "connected": self.is_connected(),
            "speed_ratio": self.session.speed_ratio,
            "checkpoint_total": self.session.checkpoint_total,
            "reset_strategy": self.config.reset_strategy.value,
            "settle_ticks": self.config.settle_ticks,
            "capabilities": self.capabilities.describe(),
        }

    # -- internals ------------------------------------------------------------------

    def _require_open(self) -> None:
        if not self._opened:
            raise GameConnectionError(
                "TMInterfaceDriver.open() has not been called (or the driver was closed)"
            )
        if not self.session.is_alive:
            raise GameConnectionError(
                "lost the TMInterface connection (game closed, or client deregistered)",
                remedy="restart Trackmania through TMInterface and retry",
            )


def build_tminterface_driver(
    server_name: str = "TMInterface0",
    *,
    speed_ratio: float = 1.0,
    position_scale: float = 1.0,
    reset_strategy: str = "respawn",
    reset_command: str = "respawn",
    settle_ticks: int = 5,
    frame_timeout_s: float = 30.0,
    connect_timeout_s: float = 20.0,
) -> GameDriver:
    """Construct the production driver from plain configuration values.

    This is the single place where the real :class:`TMInterfaceSession` is instantiated,
    which keeps the Windows-only import path out of module import time.
    """
    from tmai.game.tminterface.session import (
        TMInterfaceSession,
        TMInterfaceSessionConfig,
    )

    session = TMInterfaceSession(
        TMInterfaceSessionConfig(
            server_name=server_name,
            position_scale=position_scale,
            connect_timeout_s=connect_timeout_s,
            frame_timeout_s=frame_timeout_s,
        )
    )
    return TMInterfaceDriver(
        session,
        TMInterfaceDriverConfig(
            reset_strategy=ResetStrategy(reset_strategy),
            reset_command=reset_command,
            settle_ticks=settle_ticks,
            frame_timeout_s=frame_timeout_s,
            default_speed_ratio=speed_ratio,
        ),
    )


__all__ = [
    "ResetStrategy",
    "TMInterfaceDriver",
    "TMInterfaceDriverConfig",
    "build_tminterface_driver",
]
