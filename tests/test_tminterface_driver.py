"""Tests for the TMInterface driver logic.

These exercise the exact code that runs against the real game -- reset sequencing, op
dispatch, speed-ratio handling, connection loss -- through a scripted tick source that
honours the same contract as the real session. The only thing not covered here is the
Windows named-memory IPC itself, which is what :mod:`tmai.game.tminterface.session` isolates.
"""

from __future__ import annotations

import pytest

from tests.conftest import ScriptedTickSource, make_frame
from tmai.game.errors import GameConnectionError, GameTimeoutError
from tmai.game.protocol import Action, RacePhase
from tmai.game.simulated import SimulatedGameDriver
from tmai.game.ticksource import TickSource
from tmai.game.tminterface.driver import (
    ResetStrategy,
    TMInterfaceDriver,
    TMInterfaceDriverConfig,
)
from tmai.game.tminterface.ops import CommandOp, RespawnOp, SetSpeedOp
from tmai.game.tminterface.session import TMInterfaceSession, TMInterfaceSessionConfig


def _running_frames(count: int, start: int = 0):
    return [
        make_frame(
            position=(0.0, 0.0, float(i)),
            race_time=0.01 * (i + 1),
            phase=RacePhase.RUNNING,
        )
        for i in range(start, start + count)
    ]


class TestCapabilities:
    def test_reports_analog_control_and_speed_control(self):
        driver = TMInterfaceDriver(ScriptedTickSource([]))
        caps = driver.capabilities
        assert caps.analog_control is True
        assert caps.game_speed_control is True
        assert caps.reports_checkpoints is True
        assert caps.reports_finish is True
        assert caps.headless_capable is False

    def test_describe_is_json_serialisable(self):
        import json

        driver = TMInterfaceDriver(ScriptedTickSource(_running_frames(1)))
        driver.open()
        payload = json.dumps(driver.describe())
        assert "tminterface" in payload
        driver.close()


class TestOpenClose:
    def test_open_starts_session_and_reads_first_frame(self):
        source = ScriptedTickSource(_running_frames(3))
        driver = TMInterfaceDriver(source)
        driver.open()
        assert source.started is True
        assert driver.is_connected() is True
        assert source.consumed == 1
        driver.close()
        assert source.stopped is True
        assert driver.is_connected() is False

    def test_open_is_idempotent(self):
        source = ScriptedTickSource(_running_frames(3))
        driver = TMInterfaceDriver(source)
        driver.open()
        driver.open()
        assert source.consumed == 1
        driver.close()

    def test_step_before_open_raises(self):
        driver = TMInterfaceDriver(ScriptedTickSource(_running_frames(2)))
        with pytest.raises(GameConnectionError, match="open\\(\\) has not been called"):
            driver.step(Action())

    def test_step_after_close_raises(self):
        driver = TMInterfaceDriver(ScriptedTickSource(_running_frames(2)))
        driver.open()
        driver.close()
        with pytest.raises(GameConnectionError):
            driver.step(Action())

    def test_dead_session_is_reported_as_connection_error(self):
        source = ScriptedTickSource(_running_frames(5))
        driver = TMInterfaceDriver(source)
        driver.open()
        source.alive = False  # the game closed under us
        with pytest.raises(GameConnectionError, match="lost the TMInterface connection"):
            driver.step(Action())
        driver.close()

    def test_close_is_safe_when_not_owning_session(self):
        source = ScriptedTickSource(_running_frames(2))
        driver = TMInterfaceDriver(source, own_session=False)
        driver.open()
        driver.close()
        assert source.stopped is False

    def test_close_twice_does_not_raise(self):
        driver = TMInterfaceDriver(ScriptedTickSource(_running_frames(1)))
        driver.open()
        driver.close()
        driver.close()


class TestReset:
    def test_waits_for_settle_ticks_in_running_phase(self):
        source = ScriptedTickSource(_running_frames(10))
        driver = TMInterfaceDriver(source, TMInterfaceDriverConfig(settle_ticks=4))
        driver.open()
        frame = driver.reset()
        assert frame.race.phase is RacePhase.RUNNING
        # 1 frame for open() + 4 settling frames.
        assert source.consumed == 5
        # A respawn must have been queued before the first settling frame.
        assert any(isinstance(op, RespawnOp) for op in source.ops)
        driver.close()

    def test_resets_settle_counter_when_phase_is_not_running(self):
        # open() consumes the first frame, so 3 RUNNING frames are needed up front to get
        # two settling ticks before the interruption.
        frames = (
            _running_frames(3)
            + [make_frame(phase=RacePhase.NOT_RACING), make_frame(phase=RacePhase.NOT_RACING)]
            + _running_frames(3)
        )
        source = ScriptedTickSource(frames)
        driver = TMInterfaceDriver(source, TMInterfaceDriverConfig(settle_ticks=3))
        driver.open()
        frame = driver.reset()
        assert frame.race.phase is RacePhase.RUNNING
        # 1 (open) + 2 running + 2 not-running (counter resets) + 3 running
        assert source.consumed == 8
        driver.close()

    def test_command_reset_strategy_sends_console_command(self):
        source = ScriptedTickSource(_running_frames(5))
        driver = TMInterfaceDriver(
            source,
            TMInterfaceDriverConfig(
                reset_strategy=ResetStrategy.COMMAND, reset_command="restart_race", settle_ticks=1
            ),
        )
        driver.open()
        driver.reset()
        assert any(
            isinstance(op, CommandOp) and op.command == "restart_race" for op in source.ops
        )
        assert not any(isinstance(op, RespawnOp) for op in source.ops)
        driver.close()

    def test_timeout_when_car_never_starts(self):
        # repeat_last emulates a game that keeps ticking in the same state forever, so the
        # driver deadline is what fires rather than the scripted source running dry.
        source = ScriptedTickSource([make_frame(phase=RacePhase.NOT_RACING)], repeat_last=True)
        driver = TMInterfaceDriver(
            source, TMInterfaceDriverConfig(settle_ticks=2, frame_timeout_s=0.05)
        )
        driver.open()
        with pytest.raises(GameTimeoutError, match="RUNNING"):
            driver.reset()
        driver.close()


class TestStep:
    def test_step_forwards_action_and_returns_next_frame(self):
        source = ScriptedTickSource(_running_frames(5))
        driver = TMInterfaceDriver(source)
        driver.open()
        action = Action(steer=-0.25, throttle=0.75, brake=0.1)
        frame = driver.step(action)
        assert source.actions[-1] == action
        assert frame.race.phase is RacePhase.RUNNING
        driver.close()

    def test_action_is_clipped_before_reaching_the_game(self):
        source = ScriptedTickSource(_running_frames(5))
        driver = TMInterfaceDriver(source)
        driver.open()
        driver.step(Action(steer=5.0, throttle=-2.0, brake=9.0))
        sent = source.actions[-1]
        assert sent.steer == 1.0
        assert sent.throttle == 0.0
        assert sent.brake == 1.0
        driver.close()


class TestSpeedRatio:
    def test_requests_speed_op(self):
        source = ScriptedTickSource(_running_frames(5))
        driver = TMInterfaceDriver(source)
        driver.open()
        applied = driver.set_speed_ratio(8.0)
        assert applied == 8.0
        assert any(isinstance(op, SetSpeedOp) and op.ratio == 8.0 for op in source.ops)
        driver.close()

    def test_rejects_non_positive_ratio(self):
        driver = TMInterfaceDriver(ScriptedTickSource(_running_frames(2)))
        driver.open()
        with pytest.raises(ValueError, match="positive"):
            driver.set_speed_ratio(0.0)
        driver.close()

    def test_warns_above_recommended_maximum(self, caplog):
        source = ScriptedTickSource(_running_frames(5))
        driver = TMInterfaceDriver(source)
        driver.open()
        with caplog.at_level("WARNING"):
            driver.set_speed_ratio(150.0)
        assert "exceeds the recommended maximum" in caplog.text
        driver.close()

    def test_default_speed_ratio_applied_on_open(self):
        source = ScriptedTickSource(_running_frames(5))
        driver = TMInterfaceDriver(source, TMInterfaceDriverConfig(default_speed_ratio=4.0))
        driver.open()
        assert any(isinstance(op, SetSpeedOp) and op.ratio == 4.0 for op in source.ops)
        driver.close()


class TestAnalogInputEncoding:
    """The analog scaling happens in the session; verify it against the documented range."""

    def test_analog_full_scale_matches_tminterface_documentation(self):
        from tmai.game.tminterface.session import ANALOG_FULL_SCALE

        # TMInterface.set_input_state documents steer/gas in [-65536, 65536].
        assert ANALOG_FULL_SCALE == 65536

    def test_full_scale_maps_action_bounds_to_analog_range(self):
        from tmai.game.tminterface.session import ANALOG_FULL_SCALE

        for value, expected in ((1.0, ANALOG_FULL_SCALE), (-1.0, -ANALOG_FULL_SCALE), (0.0, 0)):
            assert int(round(value * ANALOG_FULL_SCALE)) == expected


class TestSessionPlatformGuard:
    """The real session must fail loudly and helpfully off Windows."""

    def test_start_raises_unsupported_platform_on_linux(self):
        import sys

        from tmai.game.errors import UnsupportedPlatformError

        session = TMInterfaceSession(TMInterfaceSessionConfig(connect_timeout_s=0.1))
        if sys.platform == "win32":
            pytest.skip("this check only applies to non-Windows hosts")
        with pytest.raises(UnsupportedPlatformError, match="Windows"):
            session.start()

    def test_stop_without_start_is_safe(self):
        session = TMInterfaceSession()
        session.stop()
        assert session.is_alive is False

    def test_mmap_tagname_is_windows_only(self):
        """Documents the hard platform constraint this driver works around."""
        import mmap
        import sys

        if sys.platform == "win32":
            pytest.skip("tagname is supported on Windows")
        with pytest.raises(TypeError, match="tagname"):
            mmap.mmap(-1, 4096, tagname="TMInterface0")

    def test_session_implements_tick_source_protocol(self):
        assert isinstance(TMInterfaceSession(), TickSource)


class TestFactory:
    def test_build_produces_tminterface_driver(self):
        from tmai.game.tminterface.driver import build_tminterface_driver

        driver = build_tminterface_driver(server_name="TMInterface0", speed_ratio=2.0)
        assert driver.name == "tminterface"
        assert driver.capabilities.analog_control is True

    def test_build_rejects_unknown_reset_strategy(self):
        from tmai.game.tminterface.driver import build_tminterface_driver

        with pytest.raises(ValueError):
            build_tminterface_driver(reset_strategy="teleport")


class TestSimulatedDriverContract:
    """The test double must satisfy the same contract as the real driver."""

    def test_satisfies_game_driver_protocol(self):
        from tmai.game.protocol import GameDriver
        from tmai.tracks.synthetic import straight

        driver = SimulatedGameDriver(straight())
        assert isinstance(driver, GameDriver)

    def test_full_straight_reaches_finish(self):
        from tmai.tracks.synthetic import straight

        track = straight(length=100.0)
        driver = SimulatedGameDriver(track)
        driver.open()
        frame = driver.reset()
        for _ in range(2000):
            frame = driver.step(Action(throttle=1.0))
            if frame.race.finished:
                break
        assert frame.race.finished is True
        assert frame.vehicle.speed_forward > 5.0
        driver.close()

    def test_off_track_slows_the_car(self):
        from tmai.tracks.synthetic import straight

        track = straight(length=300.0)
        driver = SimulatedGameDriver(track)
        driver.open()
        driver.reset()
        # Steer hard off the road and hold it there.
        for _ in range(200):
            frame = driver.step(Action(steer=1.0, throttle=1.0))
        off_track_speed = frame.vehicle.speed_forward
        driver.close()

        driver2 = SimulatedGameDriver(straight(length=300.0))
        driver2.open()
        driver2.reset()
        for _ in range(200):
            frame2 = driver2.step(Action(steer=0.0, throttle=1.0))
        on_track_speed = frame2.vehicle.speed_forward
        driver2.close()

        assert off_track_speed < on_track_speed

    def test_describe_marks_it_as_toy(self):
        from tmai.tracks.synthetic import straight

        driver = SimulatedGameDriver(straight())
        description = driver.describe()
        assert description["driver"] == "simulated"
        assert "toy" in description["model"].lower()
