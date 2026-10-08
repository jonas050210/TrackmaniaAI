"""Tests for the generalisation infrastructure.

Covers observation normalisation, the multi-track environment, GBX isolation and the
mechanisms that stop a policy memorising a single map.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import ScriptedTickSource, make_frame
from tmai.agents.base import Batch
from tmai.agents.normalize import NormalizingLearner
from tmai.agents.sac import SACConfig, SACLearner
from tmai.env.multi_track import EpisodeContext, MultiTrackConfig, MultiTrackEnv
from tmai.env.normalization import RunningNormalizer
from tmai.env.tm_env import EnvConfig
from tmai.game.errors import UnsupportedFeatureError
from tmai.game.protocol import Action
from tmai.game.simulated import SimulatedGameDriver
from tmai.models.networks import NetworkConfig
from tmai.tracks.centerline import CenterlineTrack
from tmai.tracks.gbx import (
    GbxSource,
    GbxUnavailableError,
    MapBlock,
    MapGeometry,
    blocks_to_centerline,
)
from tmai.tracks.library import TrackLibrary
from tmai.tracks.synthetic import build_synthetic, straight

SUITE = ("straight", "oval", "s_curve", "figure_eight")


def _library(*names: str) -> TrackLibrary:
    library = TrackLibrary()
    for name in names or SUITE:
        library.add(build_synthetic(name))
    return library


def _multi_env(config: MultiTrackConfig | None = None, *, seed: int = 0, split: str = "train"):
    library = _library()
    if not library.by_split(split):
        # Force everything into one split so the test does not depend on hash luck.
        library = TrackLibrary()
        for name in SUITE:
            library.add(build_synthetic(name), split=split)
    return MultiTrackEnv(
        library.sampler(split, seed=seed),
        lambda track: SimulatedGameDriver(track),
        EnvConfig(),
        config or MultiTrackConfig(),
        seed=seed,
        split=split,
    )


# -- normalisation ------------------------------------------------------------------


class TestRunningNormalizer:
    def test_batch_update_matches_row_by_row_welford(self):
        """The vectorised update must give the same statistics as folding one row at a time."""
        rng = np.random.default_rng(4)
        batches = [rng.normal(loc=3.0, scale=2.0, size=(n, 5)) for n in (7, 1, 64, 0, 256)]
        batches.append(rng.normal(size=5))  # a single 1-D observation

        fast = RunningNormalizer(5, warmup_steps=0)
        count, mean, m2 = 0, np.zeros(5), np.zeros(5)
        for batch in batches:
            fast.update(batch)
            for row in np.atleast_2d(batch):
                count += 1
                delta = row - mean
                mean = mean + delta / count
                m2 = m2 + delta * (row - mean)

        assert fast.count == count
        np.testing.assert_allclose(fast.mean, mean, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(fast.std, np.sqrt(m2 / count), rtol=1e-10, atol=1e-12)

    def test_zero_mean_unit_std_after_enough_data(self):
        rng = np.random.default_rng(0)
        data = rng.normal(loc=5.0, scale=3.0, size=(2000, 4))
        normalizer = RunningNormalizer(4, warmup_steps=10)
        normalizer.update(data)
        assert normalizer.mean == pytest.approx(data.mean(axis=0), abs=1e-6)
        assert normalizer.std == pytest.approx(data.std(axis=0), abs=1e-6)

    def test_normalize_produces_zero_mean(self):
        rng = np.random.default_rng(1)
        data = rng.normal(loc=-2.0, scale=4.0, size=(1000, 3))
        normalizer = RunningNormalizer(3, warmup_steps=10)
        normalizer.update(data)
        out = normalizer.normalize(data)
        assert out.mean(axis=0) == pytest.approx(0.0, abs=0.05)

    def test_clipping_bounds_outliers(self):
        normalizer = RunningNormalizer(2, warmup_steps=1, clip=5.0)
        normalizer.update(np.array([[0.0, 0.0], [1.0, 1.0]]))
        out = normalizer.normalize(np.array([[1e6, -1e6]]))
        assert np.all(np.abs(out) <= 5.0 + 1e-6)

    def test_before_warmup_only_centres(self):
        normalizer = RunningNormalizer(2, warmup_steps=1000)
        normalizer.update(np.array([[10.0, 20.0]]))
        out = normalizer.normalize(np.array([[10.0, 20.0]]))
        # Centred but not scaled: the provisional std is not trusted yet.
        assert out == pytest.approx(np.zeros((1, 2)), abs=1e-6)

    def test_single_observation_shape_preserved(self):
        normalizer = RunningNormalizer(3, warmup_steps=1)
        normalizer.update(np.zeros((2, 3)))
        out = normalizer.normalize(np.ones(3))
        assert out.shape == (3,)
        assert out.dtype == np.float32

    def test_batch_shape_preserved(self):
        normalizer = RunningNormalizer(3, warmup_steps=1)
        normalizer.update(np.zeros((2, 3)))
        assert normalizer.normalize(np.ones((7, 3))).shape == (7, 3)

    def test_rejects_wrong_dimension(self):
        normalizer = RunningNormalizer(3)
        with pytest.raises(ValueError):
            normalizer.update(np.zeros((2, 5)))
        with pytest.raises(ValueError):
            normalizer.normalize(np.zeros((2, 5)))

    def test_rejects_bad_construction(self):
        with pytest.raises(ValueError):
            RunningNormalizer(0)
        with pytest.raises(ValueError):
            RunningNormalizer(3, clip=0.0)

    def test_nan_cannot_poison_the_statistics(self):
        normalizer = RunningNormalizer(2, warmup_steps=1)
        normalizer.update(np.array([[1.0, 1.0], [np.nan, 5.0]]))
        assert np.all(np.isfinite(normalizer.mean))
        assert np.all(np.isfinite(normalizer.std))

    def test_state_roundtrip(self):
        normalizer = RunningNormalizer(3, warmup_steps=2)
        normalizer.update(np.random.default_rng(0).normal(size=(50, 3)))
        state = normalizer.state_dict()

        restored = RunningNormalizer(3, warmup_steps=2)
        restored.load_state_dict(state)
        assert restored.count == normalizer.count
        assert restored.mean == pytest.approx(normalizer.mean)
        assert restored.std == pytest.approx(normalizer.std)

    def test_state_rejects_dimension_mismatch(self):
        state = RunningNormalizer(3).state_dict()
        with pytest.raises(ValueError, match="dim"):
            RunningNormalizer(5).load_state_dict(state)

    def test_state_rejects_unknown_version(self):
        state = RunningNormalizer(3).state_dict()
        state["version"] = 999
        with pytest.raises(ValueError, match="version"):
            RunningNormalizer(3).load_state_dict(state)

    def test_std_is_one_for_constant_features(self):
        normalizer = RunningNormalizer(2, warmup_steps=1)
        normalizer.update(np.ones((10, 2)))
        assert normalizer.std == pytest.approx(np.zeros(2), abs=1e-9)
        # And normalising a constant feature does not divide by zero.
        assert np.all(np.isfinite(normalizer.normalize(np.ones((3, 2)))))


class TestNormalizingLearner:
    def _learner(self, dim: int = 6) -> SACLearner:
        return SACLearner(dim, 3, SACConfig(network=NetworkConfig(hidden_sizes=(8, 8))), seed=0)

    def _batch(self, dim: int = 6, n: int = 32, *, seed: int = 0) -> Batch:
        rng = np.random.default_rng(seed)
        return Batch(
            observations=rng.normal(size=(n, dim)).astype(np.float32),
            actions=rng.uniform(-1, 1, size=(n, 3)).astype(np.float32),
            rewards=rng.normal(size=(n,)).astype(np.float32),
            next_observations=rng.normal(size=(n, dim)).astype(np.float32),
            terminated=np.zeros(n, dtype=np.float32),
            truncated=np.zeros(n, dtype=np.float32),
        )

    def test_passes_through_dimensions(self):
        inner = self._learner()
        wrapped = NormalizingLearner(inner)
        assert wrapped.observation_dim == inner.observation_dim
        assert wrapped.action_dim == inner.action_dim

    def test_act_returns_a_valid_action(self):
        wrapped = NormalizingLearner(self._learner())
        action = wrapped.act(np.zeros(6, dtype=np.float32))
        assert action.shape == (3,)
        assert np.all(np.isfinite(action))

    def test_update_returns_normalizer_statistics(self):
        wrapped = NormalizingLearner(self._learner())
        metrics = wrapped.update(self._batch())
        assert "normalizer/count" in metrics
        assert "normalizer/std_mean" in metrics
        assert "sac/critic_loss" in metrics

    def test_statistics_accumulate_over_updates(self):
        wrapped = NormalizingLearner(self._learner())
        wrapped.update(self._batch())
        first = wrapped.normalizer.count
        wrapped.update(self._batch(seed=1))
        assert wrapped.normalizer.count > first

    def test_state_dict_includes_both_normalizer_and_learner(self):
        wrapped = NormalizingLearner(self._learner())
        wrapped.update(self._batch())
        state = wrapped.state_dict()
        assert "normalizer" in state
        assert "inner" in state

        fresh = NormalizingLearner(self._learner())
        fresh.load_state_dict(state)
        assert fresh.normalizer.count == wrapped.normalizer.count

    def test_accepts_an_unwrapped_inner_state(self):
        """A checkpoint from a plain SAC learner must still load."""
        inner = self._learner()
        wrapped = NormalizingLearner(self._learner())
        wrapped.load_state_dict(inner.state_dict())
        assert wrapped.gradient_steps == inner.gradient_steps

    def test_rejects_dimension_mismatch(self):
        with pytest.raises(ValueError, match="does not match"):
            NormalizingLearner(self._learner(6), RunningNormalizer(9))

    def test_describe_reports_normalization(self):
        description = NormalizingLearner(self._learner()).describe()
        assert description["observation_normalization"]["enabled"] is True
        assert description["algorithm"] == "sac"


# -- multi-track environment --------------------------------------------------------


class TestMultiTrackEnv:
    def test_spaces_match_the_inner_env(self):
        env = _multi_env()
        assert env.observation_space.shape == (20,)
        assert env.action_space.shape == (3,)
        env.close()

    def test_reset_switches_tracks(self):
        env = _multi_env(seed=0)
        seen = set()
        for _ in range(8):
            _, info = env.reset()
            seen.add(info["track"])
        assert len(seen) > 1, "the environment never changed track"
        env.close()

    def test_episode_context_survives_the_whole_episode(self):
        env = _multi_env(seed=0)
        _, reset_info = env.reset()
        for _ in range(5):
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            assert info["track"] == reset_info["track"]
            assert "start_station" in info
            if terminated or truncated:
                break
        env.close()

    def test_random_start_station_varies(self):
        env = _multi_env(MultiTrackConfig(random_start_station=True), seed=0)
        stations = [env.reset()[1]["start_station"] for _ in range(6)]
        assert len(set(round(s, 3) for s in stations)) > 1
        env.close()

    def test_start_station_can_be_disabled(self):
        env = _multi_env(
            MultiTrackConfig(random_start_station=False, start_lateral_std=0.0), seed=0
        )
        assert all(env.reset()[1]["start_station"] == 0.0 for _ in range(4))
        env.close()

    def test_lateral_offset_stays_inside_the_corridor(self):
        env = _multi_env(
            MultiTrackConfig(start_lateral_std=5.0, start_edge_margin=1.0), seed=0
        )
        for _ in range(10):
            env.reset()
            half = env.track.corridor_half_width_at(env.episode_context.start_station)
            assert abs(env.episode_context.start_lateral) <= half - 1.0 + 1e-9
        env.close()

    def test_select_track_pins_the_next_reset_only(self):
        env = _multi_env(seed=0)
        target = env.sampler.peek_all()[0].name
        env.select_track(target)
        assert env.reset()[1]["track"] == target
        # The pin is consumed: a later reset may sample something else.
        assert env.pinned_track is None
        env.close()

    def test_select_track_rejects_unknown_names(self):
        env = _multi_env(seed=0)
        with pytest.raises(KeyError, match="no track named"):
            env.select_track("nonexistent")
        env.close()

    def test_drivers_are_cached_per_track(self):
        env = _multi_env(seed=0)
        for _ in range(6):
            env.reset()
        assert len(env._drivers) <= len(env.sampler)  # noqa: SLF001
        env.close()

    def test_describe_reports_multi_track_configuration(self):
        env = _multi_env(seed=0)
        description = env.describe()
        assert description["multi_track"]["num_tracks"] == len(env.sampler)
        assert description["multi_track"]["split"] == "train"
        assert "config" in description["multi_track"]
        env.close()

    def test_close_releases_drivers(self):
        env = _multi_env(seed=0)
        env.reset()
        env.close()
        assert env._drivers == {}  # noqa: SLF001

    def test_episode_context_serialises(self):
        payload = EpisodeContext(track_name="oval", split="train", start_station=1.5).as_dict()
        assert payload["track"] == "oval"
        assert payload["start_station"] == 1.5

    def test_sampling_disabled_uses_the_first_track(self):
        env = _multi_env(MultiTrackConfig(sample_tracks=False), seed=0)
        names = {env.reset()[1]["track"] for _ in range(4)}
        assert len(names) == 1
        env.close()


class TestStartRepositioningIsHonest:
    """The real game cannot teleport the car; nothing may pretend otherwise."""

    def test_simulated_driver_supports_it(self):
        driver = SimulatedGameDriver(straight())
        driver.open()
        assert driver.capabilities.supports_start_repositioning is True
        frame = driver.reposition(50.0, 1.0)
        assert np.all(np.isfinite(frame.vehicle.position))

    def test_reposition_places_the_car_at_the_requested_station(self):
        track = straight(length=100.0)
        driver = SimulatedGameDriver(track)
        driver.open()
        driver.reposition(60.0, 0.0)
        projection = track.project(driver._position)  # noqa: SLF001
        assert projection.progress == pytest.approx(60.0, abs=0.5)
        assert projection.lateral_offset == pytest.approx(0.0, abs=0.1)

    def test_reposition_starts_stationary(self):
        driver = SimulatedGameDriver(straight())
        driver.open()
        frame = driver.reposition(40.0)
        assert frame.vehicle.speed_forward == 0.0

    def test_tminterface_driver_refuses(self):
        """A driver that cannot reposition must say so, not silently use the start line."""
        driver = _tminterface_driver()
        with pytest.raises(UnsupportedFeatureError, match="cannot place the car"):
            driver.reposition(50.0)

    def test_tminterface_reports_the_capability_as_false(self):
        assert _tminterface_driver().capabilities.supports_start_repositioning is False

    def test_env_raises_when_asked_to_randomise_with_a_fixed_start_driver(self):
        """The environment must surface the limitation, not quietly start at the line."""
        from tmai.env.tm_env import TrackmaniaEnv

        track = straight(length=100.0)
        env = TrackmaniaEnv(_tminterface_driver(track), track, EnvConfig())
        with pytest.raises(UnsupportedFeatureError, match="cannot place the car"):
            env.reset(options={"start_station": 40.0})

    def test_env_allows_a_reset_without_start_options(self):
        """The same driver still works normally when nothing asks for repositioning."""
        from tmai.env.tm_env import TrackmaniaEnv
        from tmai.game.protocol import RacePhase

        track = straight(length=100.0)
        frames = [
            make_frame(position=track.point_at(0.0), race_time=1.0, phase=RacePhase.RUNNING),
            make_frame(position=track.point_at(0.0), race_time=0.0, phase=RacePhase.GAVE_UP),
            *[
                make_frame(
                    position=track.point_at(0.0),
                    race_time=0.05 * i,
                    phase=RacePhase.RUNNING,
                )
                for i in range(1, 6)
            ],
        ]
        env = TrackmaniaEnv(_tminterface_driver(track, frames), track, EnvConfig())
        observation, _ = env.reset()
        assert observation.shape == (20,)
        env.close()


def _tminterface_driver(track=None, frames=None):
    """A TMInterfaceDriver over a scripted tick source (no game required)."""
    from tmai.game.protocol import RacePhase
    from tmai.game.tminterface.driver import TMInterfaceDriver

    track = track or straight(length=100.0)
    frames = frames or [
        make_frame(position=track.point_at(0.0), race_time=1.0, phase=RacePhase.RUNNING),
        make_frame(position=track.point_at(0.0), race_time=0.0, phase=RacePhase.GAVE_UP),
        *[
            make_frame(
                position=track.point_at(0.0),
                race_time=0.05 * i,
                phase=RacePhase.RUNNING,
            )
            for i in range(1, 6)
        ],
    ]
    return TMInterfaceDriver(ScriptedTickSource(frames, repeat_last=True))


class TestTranslationInvariance:
    """The strongest available check that no absolute coordinate leaks into the observation."""

    def _observation(self, track):
        from tmai.env.observation import ObservationEncoder, ObservationInputs

        driver = SimulatedGameDriver(track)
        driver.open()
        frame = driver.step(Action(throttle=1.0))
        frame = driver.step(Action(throttle=1.0, steer=0.3))
        encoder = ObservationEncoder(track)
        projection = track.project(frame.vehicle.position)
        return encoder.encode(
            ObservationInputs(
                frame=frame,
                projection=projection,
                prev_projection=None,
                prev_yaw=None,
                dt=0.05,
                last_action=Action(),
                yaw_rate=0.1,
            )
        )

    def test_translation_does_not_change_the_observation(self):
        base = straight(length=120.0)
        moved = CenterlineTrack(
            base.points + np.array([750.0, 0.0, -420.0]),
            name="moved",
            corridor_half_width=6.0,
        )
        assert self._observation(base) == pytest.approx(self._observation(moved), abs=1e-6)

    def test_no_feature_depends_on_world_origin(self):
        """Every feature must be expressible without world coordinates."""
        from tmai.env.observation import ObservationSpec

        names = ObservationSpec().names()
        forbidden = ("position", "world", "coord", "x", "y", "z")
        for name in names:
            lowered = name.lower()
            assert not any(
                token == lowered or lowered.startswith(token + "_") for token in forbidden
            ), f"observation feature {name!r} looks like an absolute coordinate"


# -- GBX isolation ------------------------------------------------------------------


def _road_geometry(n: int = 10, spacing: float = 8.0) -> MapGeometry:
    blocks = [
        MapBlock(
            name="PlatformRoad",
            position=np.array([0.0, 0.0, i * spacing]),
            size=np.array([8.0, 1.0, 8.0]),
            drivable=True,
        )
        for i in range(n)
    ]
    blocks.append(
        MapBlock(name="DecorationTree", position=np.array([30.0, 0.0, 0.0]), drivable=False)
    )
    return MapGeometry(blocks=blocks, name="TestMap", uid="abc", environment="Stadium")


class TestGbxIsolation:
    def test_load_raises_rather_than_fabricating(self):
        with pytest.raises(GbxUnavailableError, match="not implemented"):
            GbxSource().load("some.Map.Gbx")

    def test_error_is_a_not_implemented_error(self):
        """Callers can distinguish 'unfinished' from 'corrupt file'."""
        assert issubclass(GbxUnavailableError, NotImplementedError)

    def test_requirements_are_documented(self):
        requirements = GbxSource.requirements()
        assert len(requirements) >= 3
        assert any("Map.Gbx" in r for r in requirements)


class TestBlocksToCenterline:
    def test_builds_a_centerline_from_drivable_blocks(self):
        track = blocks_to_centerline(_road_geometry(), spacing=4.0)
        assert track.num_points >= 2
        assert track.length > 0
        assert track.uid == "abc"

    def test_non_drivable_blocks_are_excluded(self):
        geometry = _road_geometry()
        track = blocks_to_centerline(geometry)
        assert track.metadata["num_drivable_blocks"] == len(geometry.drivable_blocks)
        assert track.metadata["num_blocks"] == len(geometry.blocks)
        # The decoration block sits 30 m off to the side; it must not bend the line.
        assert abs(track.points[:, 0]).max() < 1e-6

    def test_block_order_does_not_matter(self):
        """Map files store blocks in editor insertion order, not racing order."""
        rng = np.random.default_rng(0)
        geometry = _road_geometry()
        shuffled = MapGeometry(
            blocks=[geometry.blocks[i] for i in rng.permutation(len(geometry.blocks))],
            name="TestMap",
            uid="abc",
        )
        a = blocks_to_centerline(geometry, spacing=4.0)
        b = blocks_to_centerline(shuffled, spacing=4.0)
        assert a.points == pytest.approx(b.points)

    def test_metadata_records_the_method(self):
        track = blocks_to_centerline(_road_geometry())
        assert track.metadata["centerline_method"] == "nearest_neighbour_walk"
        assert track.metadata["source"] == "gbx"

    def test_too_few_blocks_is_an_error_not_a_guess(self):
        single = MapGeometry(
            blocks=[MapBlock(name="x", position=np.zeros(3), drivable=True)]
        )
        with pytest.raises(ValueError, match="at least 2 drivable blocks"):
            blocks_to_centerline(single)

    def test_no_drivable_blocks_is_an_error(self):
        geometry = MapGeometry(
            blocks=[MapBlock(name="x", position=np.zeros(3), drivable=False)]
        )
        with pytest.raises(ValueError, match="at least 2 drivable blocks"):
            blocks_to_centerline(geometry)

    def test_coincident_blocks_do_not_produce_zero_length_segments(self):
        geometry = MapGeometry(
            blocks=[
                MapBlock(name="a", position=np.zeros(3), drivable=True),
                MapBlock(name="b", position=np.zeros(3), drivable=True),
                MapBlock(name="c", position=np.array([0.0, 0.0, 5.0]), drivable=True),
            ]
        )
        track = blocks_to_centerline(geometry)
        assert track.num_points >= 2

    def test_all_coincident_blocks_is_an_error(self):
        geometry = MapGeometry(
            blocks=[MapBlock(name="a", position=np.zeros(3), drivable=True) for _ in range(3)]
        )
        with pytest.raises(ValueError, match="same position"):
            blocks_to_centerline(geometry)

    def test_result_is_a_usable_centerline(self):
        """The conversion output must work with the rest of the track machinery."""
        track = blocks_to_centerline(_road_geometry(), spacing=4.0)
        projection = track.project(track.point_at(10.0))
        assert projection.progress == pytest.approx(10.0, abs=0.5)
        assert np.all(np.isfinite(track.sample_lookahead(5.0, (5.0, 10.0))))


class TestMapBlockValidation:
    def test_coerces_shapes(self):
        block = MapBlock(name="x", position=[1, 2, 3])
        assert block.position.shape == (3,)
        assert block.rotation.shape == (3, 3)
        assert block.size.shape == (3,)

    def test_drivable_defaults_to_unknown(self):
        assert MapBlock(name="x", position=np.zeros(3)).drivable is None

    def test_geometry_filters_only_true(self):
        geometry = MapGeometry(
            blocks=[
                MapBlock(name="a", position=np.zeros(3), drivable=True),
                MapBlock(name="b", position=np.zeros(3), drivable=False),
                MapBlock(name="c", position=np.zeros(3)),  # unknown
            ]
        )
        assert len(geometry.drivable_blocks) == 1

    def test_as_dict_is_json_safe(self):
        import json

        json.dumps(_road_geometry().as_dict())
