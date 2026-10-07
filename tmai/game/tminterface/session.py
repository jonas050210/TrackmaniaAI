"""Real TMInterface session: the only module that touches the ``tminterface`` package.

TMInterface (https://github.com/donadigo/TMInterface) is a Windows-only tool that injects
into the Trackmania process and exposes a synchronous, callback-driven message protocol
over a **Windows named memory-mapped file** (``mmap.mmap(-1, size, tagname="TMInterface0")``).
The game calls into our process once per physics tick; while it is waiting for our reply
the simulation is parked. That gives us frame-locked control, which is what a real-time RL
control loop needs: no dropped or duplicated actions.

Consequences that shape this module:

* **Every** call into ``TMInterface`` (reading state, injecting input, respawn, game speed)
  must happen on the tminterface worker thread, because the request/response handshake uses
  one shared buffer. Calls issued from the learner thread are therefore queued as
  :class:`GameOp` objects and executed at the top of the next tick.
* ``tminterface.run_client`` installs signal handlers and blocks, so we cannot use it from
  a worker thread. We call ``TMInterface.register`` instead, which starts the same
  ``_main_thread`` loop without touching signals.
* ``mmap(..., tagname=...)`` raises ``TypeError`` on non-Windows platforms, so ``open()``
  fails with :class:`UnsupportedPlatformError` there. That is a hard platform limit of the
  upstream tool, not something this project can work around.

The class below is deliberately thin. All policy-relevant logic (decimation, reset
sequencing, capabilities) lives in :class:`tmai.game.tminterface.driver.TMInterfaceDriver`,
which is testable against a scripted tick source.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

from tmai.game.errors import (
    GameConnectionError,
    GameTimeoutError,
    UnsupportedPlatformError,
)
from tmai.game.protocol import Action
from tmai.game.ticksource import TickSource
from tmai.game.tminterface.ops import GameOp, RespawnOp, SetSpeedOp
from tmai.game.tminterface.telemetry import frame_from_sim_state

logger = logging.getLogger(__name__)

#: Analog inputs in TMInterface span ``[-65536, 65536]``; see ``TMInterface.set_input_state``.
ANALOG_FULL_SCALE = 65536


def _require_tminterface() -> Any:
    """Import the real client, raising an actionable error when it is unavailable."""
    try:
        import tminterface  # noqa: F401  (import side-effect is the point)
        from tminterface import interface as tm_interface
        from tminterface.client import Client as _ClientBase
    except ImportError as exc:  # pragma: no cover - depends on the host machine
        raise GameConnectionError(
            f"the 'tminterface' package is not importable ({exc})",
            remedy="pip install tminterface  (or: pip install 'trackmania-ai[game]')",
        ) from exc
    return tm_interface, _ClientBase


@dataclass
class TMInterfaceSessionConfig:
    """Connection settings for the TMInterface bridge."""

    server_name: str = "TMInterface0"
    #: Must match the server's buffer size (``TMInterface.exe /serversize=...``).
    buffer_size: int = 65535
    #: ``-1`` makes TMInterface wait for us forever instead of deregistering us after 2 s.
    client_timeout_ms: int = -1
    connect_timeout_s: float = 20.0
    #: How long ``next_frame`` waits for the game to produce a tick before giving up.
    frame_timeout_s: float = 30.0
    #: Game position units -> metres. Calibrate with ``tmai doctor --measure-units``.
    position_scale: float = 1.0
    #: Ticks to skip between publishing frames (thinning a 100 Hz stream).
    publish_every_n_ticks: int = 1


class TMInterfaceSession(TickSource):
    """Owns the tminterface connection and turns its callbacks into a tick stream.

    Thread model::

        tminterface worker thread            learner / env thread
        -------------------------            --------------------
        on_run_step(iface, t):               push_action(a)
          drain op queue      <---------     request_op(op)
          apply pending input                next_frame(timeout)
          publish GameFrame   ---------->
    """

    def __init__(self, config: TMInterfaceSessionConfig | None = None) -> None:
        self.config = config or TMInterfaceSessionConfig()
        self._iface: Any = None
        self._client: Any = None
        self._thread: threading.Thread | None = None

        self._registered = threading.Event()
        self._shutdown = threading.Event()
        self._stopped = threading.Event()

        self._lock = threading.Lock()
        self._frame_available = threading.Condition(self._lock)
        self._latest_frame: Any = None
        self._frame_seq = 0

        self._ops: queue.Queue[GameOp] = queue.Queue()
        self._action_slot: queue.Queue[Action] = queue.Queue(maxsize=1)
        self._last_action = Action()
        self._checkpoint_total = 0

        self._tick_count = 0
        self._exception: BaseException | None = None
        self._speed_ratio = 1.0

    # -- TickSource ----------------------------------------------------------------

    @property
    def is_alive(self) -> bool:
        return (
            self._iface is not None
            and self._registered.is_set()
            and not self._shutdown.is_set()
            and self._exception is None
        )

    def start(self) -> None:
        """Connect to the game and block until registration succeeds."""
        if sys.platform != "win32":
            raise UnsupportedPlatformError(
                "TMInterface communicates with Trackmania through a Windows named "
                f"memory-mapped file; this host is '{sys.platform}'.",
                remedy=(
                    "run the game host on Windows, or use a different GameDriver "
                    "(see docs/INTEGRATION.md for the Openplanet option)"
                ),
            )

        tm_interface, client_base = _require_tminterface()
        session = self

        class _Client(client_base):  # type: ignore[misc, valid-type]
            def on_registered(self, iface):
                try:
                    iface.set_timeout(session.config.client_timeout_ms)
                except Exception:  # noqa: BLE001 - report, never kill the tick loop
                    logger.warning("set_timeout failed", exc_info=True)
                session._registered.set()

            def on_deregistered(self, iface):
                logger.warning("TMInterface deregistered our client")
                session._shutdown.set()

            def on_shutdown(self, iface):
                logger.warning("TMInterface server shut down (game closed?)")
                session._shutdown.set()

            def on_run_step(self, iface, race_time_ms):
                session._on_tick(iface, int(race_time_ms))

            def on_checkpoint_count_changed(self, iface, current, target):
                session._checkpoint_total = int(target)

            def on_client_exception(self, iface, exception):
                logger.error("tminterface client exception: %s", exception)
                session._exception = exception
                session._shutdown.set()

        self._client = _Client()
        self._iface = tm_interface.TMInterface(
            self.config.server_name, self.config.buffer_size
        )

        # register() spawns tminterface's own daemon thread, which retries the connection
        # until the game (with TMInterface loaded) shows up.
        self._iface.register(self._client)
        if not self._registered.wait(self.config.connect_timeout_s):
            self.stop()
            raise GameConnectionError(
                f"no TMInterface server named '{self.config.server_name}' registered "
                f"within {self.config.connect_timeout_s:.0f}s",
                remedy=(
                    "launch Trackmania through TMInterface, load a map and start a race; "
                    "check the server name in TMInterface's console"
                ),
            )
        logger.info("registered with TMInterface server '%s'", self.config.server_name)

    def stop(self) -> None:
        """Deregister from the game and stop the worker thread. Safe to call repeatedly."""
        self._shutdown.set()
        # Unblock anyone parked in next_frame().
        with self._frame_available:
            self._frame_available.notify_all()
        iface = self._iface
        if iface is not None:
            try:
                iface.close()
            except Exception:  # noqa: BLE001 - shutdown must never raise
                logger.debug("error while closing tminterface", exc_info=True)
        self._iface = None

    def request_op(self, op: GameOp) -> None:
        self._raise_if_dead()
        self._ops.put(op)

    def push_action(self, action: Action) -> None:
        self._raise_if_dead()
        try:
            self._action_slot.put_nowait(action.clipped())
        except queue.Full:
            # The game has not consumed the previous action yet; the newest one wins.
            with contextlib.suppress(queue.Empty):  # race with the tick thread
                self._action_slot.get_nowait()
            self._action_slot.put_nowait(action.clipped())

    def next_frame(self, timeout: float | None = None):
        """Block until the game publishes a new frame and return it."""
        deadline = time.monotonic() + (
            timeout if timeout is not None else self.config.frame_timeout_s
        )
        with self._frame_available:
            seen = self._frame_seq
            while True:
                self._raise_if_dead()
                if self._latest_frame is not None and self._frame_seq != seen:
                    return self._latest_frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GameTimeoutError(
                        f"no physics tick from Trackmania for "
                        f"{self.config.frame_timeout_s:.1f}s"
                    )
                self._frame_available.wait(min(remaining, 0.1))

    @property
    def checkpoint_total(self) -> int:
        return self._checkpoint_total

    @property
    def speed_ratio(self) -> float:
        return self._speed_ratio

    # -- internals (run on the tminterface thread) ----------------------------------

    def _raise_if_dead(self) -> None:
        if self._exception is not None:
            raise GameConnectionError(f"TMInterface session died: {self._exception!r}")
        if self._shutdown.is_set():
            raise GameConnectionError("TMInterface session is shut down")

    def _on_tick(self, iface: Any, race_time_ms: int) -> None:
        try:
            self._tick_count += 1
            self._drain_ops(iface)

            # Newest action wins; if the learner is slow we hold the previous input rather
            # than stalling the game (TMInterface parks the simulation while we reply).
            with contextlib.suppress(queue.Empty):
                self._last_action = self._action_slot.get_nowait()
            self._apply_action(iface, self._last_action)

            if self._tick_count % max(1, self.config.publish_every_n_ticks) != 0:
                return

            sim_state = iface.get_simulation_state()
            frame = frame_from_sim_state(
                sim_state,
                position_scale=self.config.position_scale,
                race_time_ms=race_time_ms,
                checkpoint_total=self._checkpoint_total,
                wall_time=time.monotonic(),
            )
            with self._frame_available:
                self._latest_frame = frame
                self._frame_seq += 1
                self._frame_available.notify_all()
        except Exception as exc:  # noqa: BLE001 - a dead tick loop would hang the game
            self._exception = exc
            self._shutdown.set()
            logger.exception("tick loop failed; ending session")
            with self._frame_available:
                self._frame_available.notify_all()

    def _drain_ops(self, iface: Any) -> None:
        while True:
            try:
                op = self._ops.get_nowait()
            except queue.Empty:
                return
            self._execute_op(iface, op)

    def _execute_op(self, iface: Any, op: GameOp) -> None:
        if isinstance(op, RespawnOp):
            iface.respawn()
        elif isinstance(op, SetSpeedOp):
            iface.set_speed(op.ratio)
            self._speed_ratio = op.ratio
        else:
            op.execute(iface)

    def _apply_action(self, iface: Any, action: Action) -> None:
        """Inject an :class:`Action` as TMInterface analog inputs.

        ``steer`` and ``gas`` are analog (``[-65536, 65536]``). Trackmania has no analog
        brake on the standard bindings, so ``brake`` is sent as the binary ``brake`` input
        and thresholded; see ``docs/INTEGRATION.md``.
        """
        steer = int(round(float(action.steer) * ANALOG_FULL_SCALE))
        gas = int(round(float(action.throttle) * ANALOG_FULL_SCALE))
        iface.set_input_state(
            steer=steer,
            gas=gas,
            brake=bool(action.brake > 0.5),
        )


__all__ = ["ANALOG_FULL_SCALE", "TMInterfaceSession", "TMInterfaceSessionConfig"]
