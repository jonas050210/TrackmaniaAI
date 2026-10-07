"""Control operations that must be executed on the game thread.

TMInterface's request/response handshake shares a single memory-mapped buffer, so game
calls cannot be issued from two threads at once. The learner thread therefore *describes*
what it wants with one of these objects and the tick loop executes it at the top of the
next physics tick. Keeping them as small value objects makes the driver's control flow
inspectable and testable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RespawnOp:
    """Respawn the car at the last checkpoint (``TMInterface.respawn``)."""

    def execute(self, iface: Any) -> None:
        iface.respawn()


@dataclass(frozen=True)
class SetSpeedOp:
    """Change the global game speed multiplier (``TMInterface.set_speed``).

    Running the game faster than real time is what makes long training runs affordable.
    TMInterface's own documentation warns that very high factors (>100) can make the game
    skip subsystems including input processing, so callers should stay well below that.
    """

    ratio: float

    def execute(self, iface: Any) -> None:
        iface.set_speed(float(self.ratio))


@dataclass(frozen=True)
class CommandOp:
    """Run a raw TMInterface console command (``TMInterface.execute_command``)."""

    command: str

    def execute(self, iface: Any) -> None:
        iface.execute_command(self.command)


@dataclass(frozen=True)
class CallableOp:
    """Escape hatch for operations that do not deserve their own class yet."""

    description: str
    callback: Callable[[Any], None]

    def execute(self, iface: Any) -> None:
        self.callback(iface)


class _OpProtocol(Protocol):
    def execute(self, iface: Any) -> None: ...


#: Any object exposing ``execute(iface)``.
GameOp = _OpProtocol

__all__ = ["CallableOp", "CommandOp", "GameOp", "RespawnOp", "SetSpeedOp"]
