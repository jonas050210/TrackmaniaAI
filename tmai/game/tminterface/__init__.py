"""Real Trackmania integration via TMInterface.

Split into four modules so that the untestable part stays as small as possible:

* :mod:`~tmai.game.tminterface.session` -- owns the connection. The only module that imports
  ``tminterface``, and the only one that is Windows-only.
* :mod:`~tmai.game.tminterface.driver` -- the :class:`~tmai.game.protocol.GameDriver`
  implementation: reset sequencing, action clipping, speed ratio, connection-loss handling.
  Fully unit-tested against a scripted tick source.
* :mod:`~tmai.game.tminterface.telemetry` -- pure mapping from TMInterface's simulation state
  onto this project's dataclasses.
* :mod:`~tmai.game.tminterface.ops` -- value objects describing game operations to run on the
  game thread.

See ``docs/INTEGRATION.md`` for the protocol-level description.
"""
