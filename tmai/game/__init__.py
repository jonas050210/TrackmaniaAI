"""Game integration layer.

The only place in the project that knows anything about Trackmania itself. Everything above
this package talks to the :class:`~tmai.game.protocol.GameDriver` protocol and to plain
dataclasses, which is what allows the rest of the stack to be tested without Windows or the
game.

Submodules:

* :mod:`tmai.game.protocol` -- the backend-agnostic contract (``GameDriver``, ``Action``,
  ``VehicleState``, ``RaceState``, ``GameFrame``, ``DriverCapabilities``).
* :mod:`tmai.game.tminterface` -- the real integration: IPC session, driver logic, telemetry
  mapping and queued game operations.
* :mod:`tmai.game.calibration` -- measures the game's telemetry conventions instead of
  assuming them.
* :mod:`tmai.game.simulated` -- a clearly-labelled test double. **Not Trackmania.**
"""
