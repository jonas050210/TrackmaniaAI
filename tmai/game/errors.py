"""Typed failures for the game integration layer.

The distinction between these types matters: :class:`GameNotInstalledError` means the
support software is missing, :class:`GameConnectionError` means it is present but the game
is not reachable, and :class:`GameTimeoutError` means the game stopped answering. Callers
(``tmai doctor``, the trainer) turn them into actionable messages instead of tracebacks.
"""

from __future__ import annotations


class GameError(Exception):
    """Base class for every failure coming from the game integration layer."""


class GameNotInstalledError(GameError):
    """The support library / plugin that bridges Python and the game is not available."""

    def __init__(self, message: str, *, remedy: str | None = None) -> None:
        super().__init__(message if remedy is None else f"{message}\n  -> {remedy}")
        self.remedy = remedy


class GameConnectionError(GameError):
    """The support library is available but no reachable game instance was found."""

    def __init__(self, message: str, *, remedy: str | None = None) -> None:
        super().__init__(message if remedy is None else f"{message}\n  -> {remedy}")
        self.remedy = remedy


class GameTimeoutError(GameError):
    """The game stopped responding within the configured deadline."""


class GameProtocolError(GameError):
    """The game answered, but with something that violates the expected protocol."""


class UnsupportedPlatformError(GameError):
    """The requested integration cannot work on this operating system at all."""

    def __init__(self, message: str, *, remedy: str | None = None) -> None:
        super().__init__(message if remedy is None else f"{message}\n  -> {remedy}")
        self.remedy = remedy


class UnsupportedFeatureError(GameError):
    """The driver does not implement an optional capability.

    Raised rather than silently ignored: a caller that asked for a randomised start and got
    the start line instead would believe its data was randomised when it was not.
    """

    def __init__(self, message: str, *, remedy: str | None = None) -> None:
        super().__init__(message if remedy is None else f"{message}\n  -> {remedy}")
        self.remedy = remedy


__all__ = [
    "GameConnectionError",
    "GameError",
    "GameNotInstalledError",
    "GameProtocolError",
    "GameTimeoutError",
    "UnsupportedFeatureError",
    "UnsupportedPlatformError",
]
