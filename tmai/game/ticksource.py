"""The seam between the real game IPC and the rest of the driver logic.

:class:`TickSource` is the whole surface that :class:`~tmai.game.tminterface.driver.TMInterfaceDriver`
needs from the game bridge. Splitting it out is what makes the driver's logic (reset
sequencing, tick decimation, finish detection, capability reporting) testable without
Windows, without the game and without the ``tminterface`` package: tests inject a scripted
tick source, and the exact same driver code runs in production.

Only :class:`tmai.game.tminterface.session.TMInterfaceSession` implements this against the
real game.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tmai.game.protocol import Action, GameFrame
from tmai.game.tminterface.ops import GameOp


@runtime_checkable
class TickSource(Protocol):
    """A source of physics ticks plus a mailbox for control operations."""

    @property
    def is_alive(self) -> bool:
        """Whether the bridge is connected and healthy."""

    @property
    def checkpoint_total(self) -> int:
        """Total checkpoints of the loaded map, ``0`` when unknown."""

    @property
    def speed_ratio(self) -> float:
        """Currently applied game speed multiplier."""

    def start(self) -> None:
        """Connect to the game. Raises on failure."""

    def stop(self) -> None:
        """Disconnect. Must be idempotent and never raise."""

    def request_op(self, op: GameOp) -> None:
        """Queue a control operation for execution on the next physics tick."""

    def push_action(self, action: Action | None) -> None:
        """Hand over the newest control command; ``None`` enables passive observation."""

    def next_frame(self, timeout: float | None = None) -> GameFrame:
        """Block until the game produces a new frame."""


__all__ = ["TickSource"]
