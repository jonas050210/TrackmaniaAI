"""Direct tests for the callback/learner handshake at the TMInterface boundary."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

from tmai.game.protocol import Action
from tmai.game.tminterface.ops import CallableOp
from tmai.game.tminterface.session import TMInterfaceSession, TMInterfaceSessionConfig


class _State:
    def __init__(self):
        self.flags = 0x2 | 0x80
        self.position = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]
        self.rotation_matrix = np.eye(3)
        self.race_time = 100
        self.num_respawns = 0
        self.player_info = SimpleNamespace(
            race_time=100, race_finished=False, cur_cp_count=0
        )
        self.scene_mobil = SimpleNamespace(
            sync_vehicle_state=SimpleNamespace(
                speed_forward=0.0,
                speed_sideward=0.0,
                rpm=1000.0,
                gearbox_state=1,
                input_steer=0.0,
                input_gas=0.0,
                input_brake=0.0,
            ),
            engine=SimpleNamespace(gear=1, max_rpm=10000.0),
            is_sliding=False,
            has_any_lateral_contact=False,
        )


class _FakeInterface:
    def __init__(self):
        self.inputs: list[dict] = []
        self.operations: list[str] = []
        self.state = _State()

    def get_simulation_state(self):
        return self.state

    def set_input_state(self, **kwargs):
        self.inputs.append(kwargs)

    def set_speed(self, ratio):
        self.operations.append(f"speed:{ratio}")

    def close(self):
        pass


def _session(*, publish_every_n_ticks: int = 1):
    session = TMInterfaceSession(
        TMInterfaceSessionConfig(
            frame_timeout_s=1.0,
            publish_every_n_ticks=publish_every_n_ticks,
        )
    )
    session._iface = _FakeInterface()
    session._registered.set()
    return session


def _run_tick(session: TMInterfaceSession, race_time_ms: int = 100):
    thread = threading.Thread(target=session._on_tick, args=(session._iface, race_time_ms))
    thread.start()
    return thread


def test_published_tick_waits_for_policy_action_before_releasing_game():
    session = _session()
    thread = _run_tick(session)
    frame = session.next_frame(timeout=1.0)

    assert frame.race.race_time == 0.1
    thread.join(timeout=0.05)
    assert thread.is_alive(), "the published callback must park until the next command arrives"
    assert session._iface.inputs == []

    session.push_action(Action(steer=0.25, throttle=0.75))
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert session._iface.inputs == [
        {"steer": 16384, "gas": 49152, "brake": False}
    ]


def test_passive_observation_does_not_inject_neutral_or_override_human_input():
    session = _session()
    thread = _run_tick(session)
    session.next_frame(timeout=1.0)
    session.push_action(None)
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert session._iface.inputs == []


def test_tick_decimation_publishes_only_after_configured_physics_ticks():
    session = _session(publish_every_n_ticks=3)
    session._on_tick(session._iface, 10)
    session._on_tick(session._iface, 20)
    assert session._frame_seq == 0

    thread = _run_tick(session, race_time_ms=30)
    session.next_frame(timeout=1.0)
    assert session._frame_seq == 1
    session.push_action(Action())
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_request_op_releases_parked_tick_and_executes_on_callback_thread():
    session = _session()
    thread = _run_tick(session)
    session.next_frame(timeout=1.0)
    session.request_op(CallableOp("sentinel", lambda iface: iface.operations.append("op")))
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert session._iface.operations == ["op"]
    # Reset/operation wakeups explicitly apply neutral input, not a stale held throttle.
    assert session._iface.inputs[-1] == {"steer": 0, "gas": 0, "brake": False}


def test_orderly_shutdown_releases_controls_on_the_callback_thread():
    session = _session()
    session._control_enabled = True
    session._last_action = Action(steer=0.5, throttle=1.0)
    iface = session._iface
    thread = _run_tick(session)
    session.next_frame(timeout=1.0)

    session.stop()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert session._iface is None
    assert session._shutdown_neutralized.is_set()
    assert iface.inputs[-1] == {"steer": 0, "gas": 0, "brake": False}


def test_tick_failure_attempts_to_neutralize_controls():
    session = _session()
    session._control_enabled = True
    session._last_action = Action(throttle=1.0)

    def fail_read():
        raise RuntimeError("broken game state")

    session._iface.get_simulation_state = fail_read
    session._on_tick(session._iface, 100)

    assert session._exception is not None
    assert session._shutdown.is_set()
    assert session._iface.inputs[-1] == {"steer": 0, "gas": 0, "brake": False}
