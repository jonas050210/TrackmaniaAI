"""Tests for the replay buffer, the policy/value network and the SAC learner."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tmai.agents.base import Batch, Transition
from tmai.agents.replay import ReplayBuffer, ReplayBufferConfig
from tmai.agents.sac import SACConfig, SACLearner
from tmai.models.networks import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    ActorCriticNetwork,
    GaussianPolicy,
    NetworkConfig,
    TwinCritic,
)


def make_transition(i: int, obs_dim: int = 4, action_dim: int = 3) -> Transition:
    rng = np.random.default_rng(i)
    return Transition(
        observation=rng.normal(size=obs_dim).astype(np.float32),
        action=rng.uniform(-1, 1, size=action_dim).astype(np.float32),
        reward=float(i),
        next_observation=rng.normal(size=obs_dim).astype(np.float32),
        terminated=(i % 7 == 0),
        truncated=(i % 11 == 0),
    )


class TestReplayBuffer:
    def test_add_and_len(self):
        buffer = ReplayBuffer(4, 3, ReplayBufferConfig(capacity=100, seed=0))
        for i in range(10):
            buffer.add(make_transition(i))
        assert len(buffer) == 10
        assert buffer.total_added == 10

    def test_wraps_around_at_capacity(self):
        buffer = ReplayBuffer(4, 3, ReplayBufferConfig(capacity=5, seed=0))
        for i in range(12):
            buffer.add(make_transition(i))
        assert len(buffer) == 5
        assert buffer.total_added == 12
        assert buffer.is_full is True

    def test_oldest_entries_are_evicted(self):
        buffer = ReplayBuffer(4, 3, ReplayBufferConfig(capacity=3, seed=0))
        for i in range(5):
            buffer.add(make_transition(i))
        # sample() draws with replacement and refuses batches larger than the buffer, so read
        # the surviving contents deterministically instead.
        assert buffer.snapshot().rewards.tolist() == [2.0, 3.0, 4.0]

    def test_snapshot_is_in_insertion_order_after_wrapping(self):
        buffer = ReplayBuffer(4, 3, ReplayBufferConfig(capacity=4, seed=0))
        for i in range(10):
            buffer.add(make_transition(i))
        # Oldest survivors are 6, 7, 8, 9 -- and oldest-first must survive the wrap.
        assert buffer.snapshot().rewards.tolist() == [6.0, 7.0, 8.0, 9.0]

    def test_snapshot_of_empty_buffer(self):
        buffer = ReplayBuffer(4, 3, ReplayBufferConfig(capacity=10, seed=0))
        snapshot = buffer.snapshot()
        assert len(snapshot) == 0
        assert snapshot.observations.shape == (0, 4)
        assert snapshot.actions.shape == (0, 3)

    def test_sample_shapes_and_dtypes(self):
        buffer = ReplayBuffer(4, 3, ReplayBufferConfig(capacity=100, seed=0))
        for i in range(50):
            buffer.add(make_transition(i))
        batch = buffer.sample(16)
        assert batch.observations.shape == (16, 4)
        assert batch.actions.shape == (16, 3)
        assert batch.rewards.shape == (16,)
        assert batch.next_observations.shape == (16, 4)
        assert batch.terminated.shape == (16,)
        assert batch.observations.dtype == np.float32
        assert batch.size == 16

    def test_flags_are_stored_separately(self):
        """terminated must not be conflated with truncated: SAC bootstraps through truncation."""
        buffer = ReplayBuffer(4, 3, ReplayBufferConfig(capacity=100, seed=0))
        buffer.add(
            Transition(
                observation=np.zeros(4, dtype=np.float32),
                action=np.zeros(3, dtype=np.float32),
                reward=0.0,
                next_observation=np.zeros(4, dtype=np.float32),
                terminated=False,
                truncated=True,
            )
        )
        batch = buffer.sample(1)
        assert batch.terminated[0] == 0.0
        assert batch.truncated[0] == 1.0

    def test_elapsed_time_discount_exponent_roundtrips(self):
        buffer = ReplayBuffer(4, 3, ReplayBufferConfig(capacity=10, seed=0))
        transition = make_transition(0)
        transition.discount_exponent = 0.25
        buffer.add(transition)
        assert buffer.snapshot().discount_exponents.tolist() == pytest.approx([0.25])
        assert buffer.sample(1).discount_exponents[0] == pytest.approx(0.25)

    def test_invalid_discount_exponent_rejected(self):
        buffer = ReplayBuffer(4, 3, ReplayBufferConfig(capacity=10, seed=0))
        transition = make_transition(0)
        transition.discount_exponent = float("nan")
        with pytest.raises(ValueError, match="discount_exponent"):
            buffer.add(transition)

    def test_sampling_is_seeded(self):
        def fill():
            buffer = ReplayBuffer(4, 3, ReplayBufferConfig(capacity=100, seed=1234))
            for i in range(50):
                buffer.add(make_transition(i))
            return buffer.sample(8).rewards

        np.testing.assert_allclose(fill(), fill())

    def test_oversized_sample_raises(self):
        buffer = ReplayBuffer(4, 3, ReplayBufferConfig(capacity=100, seed=0))
        buffer.add(make_transition(0))
        with pytest.raises(ValueError, match="cannot sample"):
            buffer.sample(5)

    def test_zero_batch_size_raises(self):
        buffer = ReplayBuffer(4, 3, ReplayBufferConfig(capacity=10, seed=0))
        buffer.add(make_transition(0))
        with pytest.raises(ValueError, match="positive"):
            buffer.sample(0)

    def test_wrong_observation_dim_raises(self):
        buffer = ReplayBuffer(4, 3, ReplayBufferConfig(capacity=10, seed=0))
        with pytest.raises(ValueError, match="features"):
            buffer.add(make_transition(0, obs_dim=5))

    def test_wrong_action_dim_raises(self):
        buffer = ReplayBuffer(4, 3, ReplayBufferConfig(capacity=10, seed=0))
        with pytest.raises(ValueError, match="features"):
            buffer.add(make_transition(0, action_dim=2))

    def test_next_observation_shape_is_validated(self):
        buffer = ReplayBuffer(4, 3, ReplayBufferConfig(capacity=10, seed=0))
        transition = make_transition(0)
        transition.next_observation = np.zeros((1, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="one-dimensional"):
            buffer.add(transition)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("observation", np.array([0.0, np.nan, 0.0, 0.0], dtype=np.float32)),
            ("action", np.array([0.0, np.inf, 0.0], dtype=np.float32)),
            ("next_observation", np.array([0.0, 0.0, 0.0, np.nan], dtype=np.float32)),
            ("reward", float("nan")),
        ],
    )
    def test_nonfinite_transition_fields_are_rejected_atomically(self, field, value):
        buffer = ReplayBuffer(4, 3, ReplayBufferConfig(capacity=10, seed=0))
        buffer.add(make_transition(1))
        before = buffer.snapshot()
        transition = make_transition(2)
        setattr(transition, field, value)
        with pytest.raises(ValueError):
            buffer.add(transition)
        after = buffer.snapshot()
        assert buffer.total_added == 1
        np.testing.assert_array_equal(after.observations, before.observations)
        np.testing.assert_array_equal(after.actions, before.actions)
        np.testing.assert_array_equal(after.rewards, before.rewards)

    def test_invalid_construction_rejected(self):
        with pytest.raises(ValueError):
            ReplayBuffer(0, 3)
        with pytest.raises(ValueError):
            ReplayBuffer(4, 0)
        with pytest.raises(ValueError):
            ReplayBuffer(4, 3, ReplayBufferConfig(capacity=0))

    def test_statistics(self):
        buffer = ReplayBuffer(4, 3, ReplayBufferConfig(capacity=100, seed=0))
        assert buffer.statistics()["buffer/size"] == 0.0
        for i in range(10):
            buffer.add(make_transition(i))
        stats = buffer.statistics()
        assert stats["buffer/size"] == 10.0
        assert stats["buffer/mean_reward"] == pytest.approx(np.mean(range(10)))
        assert 0.0 <= stats["buffer/terminal_rate"] <= 1.0

    def test_extend(self):
        buffer = ReplayBuffer(4, 3, ReplayBufferConfig(capacity=100, seed=0))
        buffer.extend([make_transition(i) for i in range(5)])
        assert len(buffer) == 5


class TestNetwork:
    def test_shapes(self):
        net = ActorCriticNetwork(20, 3, NetworkConfig(hidden_sizes=(32, 32)))
        obs = torch.randn(8, 20)
        action, log_prob = net.act(obs)
        assert action.shape == (8, 3)
        assert log_prob.shape == (8, 1)
        q1, q2 = net.q_values(obs, action)
        assert q1.shape == (8, 1)
        assert q2.shape == (8, 1)

    def test_actions_respect_bounds(self):
        low = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        net = ActorCriticNetwork(20, 3, action_low=low, action_high=high)
        obs = torch.randn(512, 20) * 3.0
        action, _ = net.act(obs)
        assert torch.all(action[:, 0] >= -1.0) and torch.all(action[:, 0] <= 1.0)
        # Throttle and brake must never go negative: tanh output is rescaled into [0, 1].
        assert torch.all(action[:, 1] >= 0.0) and torch.all(action[:, 1] <= 1.0)
        assert torch.all(action[:, 2] >= 0.0) and torch.all(action[:, 2] <= 1.0)

    def test_invalid_bounds_rejected(self):
        with pytest.raises(ValueError, match="strictly greater"):
            GaussianPolicy(4, 2, action_low=[1.0, 1.0], action_high=[-1.0, -1.0])

    def test_log_std_is_clamped(self):
        policy = GaussianPolicy(4, 2, NetworkConfig(hidden_sizes=(8,)))
        obs = torch.randn(16, 4) * 1000.0
        _, log_std = policy(obs)
        assert torch.all(log_std >= LOG_STD_MIN - 1e-6)
        assert torch.all(log_std <= LOG_STD_MAX + 1e-6)

    def test_log_prob_matches_numerical_density(self):
        """The tanh-squashed log-prob must be a correct density, including the Jacobian term.

        Compared as a *function of the same action*: drawing fresh noise on each side would
        compare two unrelated samples and prove nothing.
        """
        torch.manual_seed(0)
        # Default bounds are [-1, 1], so the affine rescale contributes log(1) = 0.
        policy = GaussianPolicy(3, 1, NetworkConfig(hidden_sizes=(16, 16)))
        obs = torch.randn(1, 3)

        mean, log_std = policy(obs)
        std = log_std.exp()

        # Actions spread over the open interval, away from the tanh saturation ends.
        actions = torch.linspace(-0.95, 0.95, 500).reshape(-1, 1)
        u = torch.atanh(actions)

        # log p(a) = log N(u; mean, std) - log|da/du|, with da/du = 1 - tanh(u)^2 = 1 - a^2.
        expected = (
            -0.5 * (((u - mean) / std) ** 2)
            - log_std
            - 0.5 * np.log(2 * np.pi)
            - torch.log(1.0 - actions**2)
        )

        analytic = policy.log_prob(obs.expand(actions.shape[0], -1), actions)
        np.testing.assert_allclose(
            analytic.detach().numpy(),
            expected.detach().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_log_prob_roundtrip(self):
        """log_prob(action) must agree with the log-prob of the action sample() produced."""
        torch.manual_seed(1)
        policy = GaussianPolicy(5, 2, NetworkConfig(hidden_sizes=(16,)))
        obs = torch.randn(64, 5)
        action, log_prob = policy.sample(obs)
        recomputed = policy.log_prob(obs, action)
        np.testing.assert_allclose(
            recomputed.detach().numpy(), log_prob.detach().numpy(), rtol=1e-3, atol=1e-3
        )

    def test_deterministic_action_is_stable(self):
        net = ActorCriticNetwork(8, 2)
        obs = torch.randn(4, 8)
        first, _ = net.act(obs, deterministic=True)
        second, _ = net.act(obs, deterministic=True)
        torch.testing.assert_close(first, second)

    def test_stochastic_action_varies(self):
        net = ActorCriticNetwork(8, 2)
        obs = torch.randn(64, 8)
        first, _ = net.act(obs, deterministic=False)
        second, _ = net.act(obs, deterministic=False)
        assert not torch.allclose(first, second)

    def test_target_network_starts_as_a_copy(self):
        net = ActorCriticNetwork(6, 2)
        for target, online in zip(
            net.critic_target.state_dict().values(), net.critic.state_dict().values(), strict=True
        ):
            torch.testing.assert_close(target, online)

    def test_target_network_does_not_track_immediately(self):
        net = ActorCriticNetwork(6, 2)
        before = {k: v.clone() for k, v in net.critic_target.state_dict().items()}
        with torch.no_grad():
            for param in net.critic.parameters():
                param.add_(1.0)
        # Without an update call the target must be untouched.
        for key, value in net.critic_target.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_polyak_update_moves_targets_towards_online(self):
        net = ActorCriticNetwork(6, 2)
        before = {k: v.clone() for k, v in net.critic_target.state_dict().items()}
        with torch.no_grad():
            for param in net.critic.parameters():
                param.add_(10.0)
        net.update_targets(0.5)
        moved = False
        for key, value in net.critic_target.state_dict().items():
            if not torch.allclose(value, before[key]):
                moved = True
                expected = before[key] * 0.5 + (before[key] + 10.0) * 0.5
                torch.testing.assert_close(value, expected)
        assert moved

    def test_tau_validation(self):
        net = ActorCriticNetwork(6, 2)
        with pytest.raises(ValueError, match="tau"):
            net.update_targets(0.0)
        with pytest.raises(ValueError, match="tau"):
            net.update_targets(1.5)

    def test_target_parameters_are_frozen(self):
        net = ActorCriticNetwork(6, 2)
        assert all(not p.requires_grad for p in net.critic_target.parameters())

    def test_parameter_groups_are_disjoint(self):
        net = ActorCriticNetwork(6, 2)
        actor, critic = net.parameter_groups()
        actor_ids = {id(p) for p in actor}
        critic_ids = {id(p) for p in critic}
        assert actor_ids.isdisjoint(critic_ids)
        assert len(actor) > 0 and len(critic) > 0

    def test_invalid_dimensions_rejected(self):
        with pytest.raises(ValueError, match="obs_dim"):
            ActorCriticNetwork(0, 2)
        with pytest.raises(ValueError, match="action_dim"):
            ActorCriticNetwork(4, 0)

    def test_unknown_activation_rejected(self):
        with pytest.raises(ValueError, match="unknown activation"):
            ActorCriticNetwork(4, 2, NetworkConfig(activation="swish_max"))

    def test_state_snapshot_reports_parameter_count(self):
        net = ActorCriticNetwork(4, 2, NetworkConfig(hidden_sizes=(8,)))
        snapshot = net.state_snapshot()
        assert snapshot["num_parameters"] == sum(p.numel() for p in net.parameters())
        assert snapshot["obs_dim"] == 4

    def test_twin_critic_heads_differ(self):
        critic = TwinCritic(4, 2)
        obs = torch.randn(16, 4)
        action = torch.randn(16, 2)
        q1, q2 = critic(obs, action)
        assert not torch.allclose(q1, q2)


class TestSACLearner:
    @staticmethod
    def make_learner(seed=0, **kwargs) -> SACLearner:
        config = SACConfig(network=NetworkConfig(hidden_sizes=(32, 32)), **kwargs)
        return SACLearner(
            4,
            3,
            config,
            action_low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            action_high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            device="cpu",
            seed=seed,
        )

    @staticmethod
    def make_batch(n=64, seed=0) -> Batch:
        rng = np.random.default_rng(seed)
        return Batch(
            observations=rng.normal(size=(n, 4)).astype(np.float32),
            actions=rng.uniform(
                low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                size=(n, 3),
            ).astype(np.float32),
            rewards=rng.normal(size=(n,)).astype(np.float32),
            next_observations=rng.normal(size=(n, 4)).astype(np.float32),
            terminated=np.zeros(n, dtype=np.float32),
            truncated=np.zeros(n, dtype=np.float32),
        )

    def test_act_single_and_batched(self):
        learner = self.make_learner()
        single = learner.act(np.zeros(4, dtype=np.float32))
        assert single.shape == (3,)
        batched = learner.act(np.zeros((7, 4), dtype=np.float32))
        assert batched.shape == (7, 3)

    def test_act_respects_action_bounds(self):
        learner = self.make_learner()
        actions = learner.act(np.random.default_rng(0).normal(size=(256, 4)).astype(np.float32) * 5)
        assert np.all(actions[:, 0] >= -1.0) and np.all(actions[:, 0] <= 1.0)
        assert np.all(actions[:, 1] >= 0.0)
        assert np.all(actions[:, 2] >= 0.0)

    def test_act_rejects_wrong_dim(self):
        learner = self.make_learner()
        with pytest.raises(ValueError, match="features"):
            learner.act(np.zeros(5, dtype=np.float32))

    def test_act_rejects_nonfinite_and_wrong_rank(self):
        learner = self.make_learner()
        bad = np.zeros(4, dtype=np.float32)
        bad[2] = np.nan
        with pytest.raises(ValueError, match="finite"):
            learner.act(bad)
        with pytest.raises(ValueError, match="shape"):
            learner.act(np.zeros((2, 2, 1), dtype=np.float32))

    def test_deterministic_act_is_reproducible(self):
        learner = self.make_learner()
        obs = np.random.default_rng(0).normal(size=(8, 4)).astype(np.float32)
        np.testing.assert_allclose(
            learner.act(obs, deterministic=True), learner.act(obs, deterministic=True)
        )

    def test_update_returns_expected_metrics(self):
        learner = self.make_learner()
        metrics = learner.update(self.make_batch())
        for key in (
            "sac/critic_loss",
            "sac/actor_loss",
            "sac/temperature",
            "sac/q1_mean",
            "sac/q_target_mean",
            "sac/discount_mean",
            "sac/gradient_steps",
        ):
            assert key in metrics, f"missing metric {key}"
        assert np.isfinite(list(metrics.values())).all()
        assert metrics["sac/gradient_steps"] == 1.0

    def test_discount_factor_scales_with_elapsed_time(self):
        learner = self.make_learner(gamma=0.9)
        batch = self.make_batch(n=3)
        batch.discount_exponents = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        metrics = learner.update(batch)
        expected = np.mean([1.0, 0.9**0.5, 0.9])
        assert metrics["sac/discount_mean"] == pytest.approx(expected, rel=1e-6)

    def test_invalid_batch_discount_exponents_are_rejected(self):
        learner = self.make_learner()
        batch = self.make_batch(n=2)
        batch.discount_exponents = np.array([1.0, float("nan")], dtype=np.float32)
        with pytest.raises(ValueError, match="discount_exponents"):
            learner.update(batch)

    @pytest.mark.parametrize(
        "field,value,match",
        [
            ("observations", np.zeros((2, 3), dtype=np.float32), "observations"),
            ("actions", np.zeros((2, 2), dtype=np.float32), "actions"),
            ("rewards", np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32), "rewards"),
            ("next_observations", np.zeros((1, 4), dtype=np.float32), "next_observations"),
            ("terminated", np.array([0.0, 0.5], dtype=np.float32), "terminated"),
            ("truncated", np.array([0.0, np.nan], dtype=np.float32), "truncated"),
            ("discount_exponents", np.array([1.0, -0.1], dtype=np.float32), "discount_exponents"),
        ],
    )
    def test_malformed_batch_is_rejected_before_optimizer_updates(self, field, value, match):
        learner = self.make_learner()
        batch = self.make_batch(n=2)
        setattr(batch, field, value)
        before = [parameter.detach().clone() for parameter in learner.network.parameters()]
        with pytest.raises(ValueError, match=match):
            learner.update(batch)
        assert learner.gradient_steps == 0
        assert all(
            torch.equal(previous, current)
            for previous, current in zip(before, learner.network.parameters(), strict=True)
        )

    def test_action_outside_bounds_is_rejected(self):
        learner = self.make_learner()
        batch = self.make_batch(n=2)
        batch.actions[0, 1] = -0.1
        with pytest.raises(ValueError, match="action bounds"):
            learner.update(batch)

    def test_gradient_steps_increment(self):
        learner = self.make_learner()
        for _ in range(5):
            learner.update(self.make_batch())
        assert learner.gradient_steps == 5

    def test_losses_are_finite_and_gradients_flow(self):
        learner = self.make_learner()
        batch = self.make_batch()
        before = [p.detach().clone() for p in learner.network.policy.parameters()]
        learner.update(batch)
        after = list(learner.network.policy.parameters())
        assert any(not torch.allclose(b, a) for b, a in zip(before, after, strict=True))

    @pytest.mark.slow
    def test_critic_learns_a_constant_reward(self):
        """With r=1, gamma=0.9 and no terminals, Q must converge to 1/(1-gamma) = 10.

        This is the property that proves the Bellman target is wired up correctly: a critic
        that did not bootstrap would stay near 1, and one that double-counted would diverge.
        The entropy temperature is set to ~0 so the fixed point is exactly 1/(1-gamma)
        rather than shifted by the entropy bonus.
        """
        learner = self.make_learner(
            gamma=0.9,
            tau=0.05,
            critic_lr=1e-3,
            actor_lr=1e-3,
            initial_temperature=1e-4,
            learnable_temperature=False,
        )
        rng = np.random.default_rng(0)
        for _ in range(1200):
            n = 128
            batch = Batch(
                observations=rng.normal(size=(n, 4)).astype(np.float32),
                actions=rng.uniform(
                    low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                    high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                    size=(n, 3),
                ).astype(np.float32),
                rewards=np.ones(n, dtype=np.float32),
                next_observations=rng.normal(size=(n, 4)).astype(np.float32),
                terminated=np.zeros(n, dtype=np.float32),
                truncated=np.zeros(n, dtype=np.float32),
            )
            learner.update(batch)
        q1, q2 = learner.network.q_values(torch.zeros(64, 4), torch.zeros(64, 3))
        assert q1.mean().item() == pytest.approx(10.0, rel=0.1)
        assert q2.mean().item() == pytest.approx(10.0, rel=0.1)

    def test_terminal_states_do_not_bootstrap(self):
        """A terminal transition must target r, not r + gamma * Q(s')."""
        learner = self.make_learner(gamma=0.99, learnable_temperature=False)
        obs = torch.zeros(1, 4)
        action = torch.zeros(1, 3)
        with torch.no_grad():
            q_target, _ = learner.network.target_q_values(obs, action)
        reward = 3.0
        batch = Batch(
            observations=np.zeros((1, 4), dtype=np.float32),
            actions=np.zeros((1, 3), dtype=np.float32),
            rewards=np.array([reward], dtype=np.float32),
            next_observations=np.zeros((1, 4), dtype=np.float32),
            terminated=np.ones(1, dtype=np.float32),
            truncated=np.zeros(1, dtype=np.float32),
        )
        metrics = learner.update(batch)
        # With terminated=1 the target collapses to the reward regardless of Q(s').
        assert metrics["sac/q_target_mean"] == pytest.approx(reward, abs=1e-4)

    def test_truncated_states_do_bootstrap(self):
        """Same transition but truncated: the target must include the bootstrapped value."""
        learner = self.make_learner(gamma=0.99, learnable_temperature=False)
        reward = 3.0
        batch = Batch(
            observations=np.zeros((1, 4), dtype=np.float32),
            actions=np.zeros((1, 3), dtype=np.float32),
            rewards=np.array([reward], dtype=np.float32),
            next_observations=np.zeros((1, 4), dtype=np.float32),
            terminated=np.zeros(1, dtype=np.float32),
            truncated=np.ones(1, dtype=np.float32),
        )
        metrics = learner.update(batch)
        # Not equal to the bare reward: the value of the next state was added.
        assert metrics["sac/q_target_mean"] != pytest.approx(reward, abs=1e-4)

    def test_temperature_moves_towards_target_entropy(self):
        learner = self.make_learner(learnable_temperature=True, initial_temperature=0.2)
        start = learner.temperature
        for _ in range(50):
            learner.update(self.make_batch())
        # With a fresh policy the log-prob is far from the target entropy, so alpha must move.
        assert learner.temperature != start

    def test_fixed_temperature_when_disabled(self):
        learner = self.make_learner(learnable_temperature=False, initial_temperature=0.35)
        for _ in range(20):
            learner.update(self.make_batch())
        assert learner.temperature == pytest.approx(0.35, rel=1e-4)

    def test_target_entropy_default_is_negative_action_dim(self):
        learner = self.make_learner()
        assert learner.target_entropy == pytest.approx(-3.0)

    def test_gradient_clipping_changes_grad_norm_reporting(self):
        learner = self.make_learner(max_grad_norm=1.0)
        metrics = learner.update(self.make_batch())
        assert np.isfinite(metrics["sac/actor_grad_norm"])
        assert np.isfinite(metrics["sac/critic_grad_norm"])

    def test_state_dict_roundtrip_preserves_behaviour(self):
        learner = self.make_learner(seed=7)
        for _ in range(10):
            learner.update(self.make_batch())
        obs = np.random.default_rng(3).normal(size=(16, 4)).astype(np.float32)
        expected = learner.act(obs, deterministic=True)
        saved = learner.state_dict()

        restored = self.make_learner(seed=99)
        restored.load_state_dict(saved)
        np.testing.assert_allclose(
            restored.act(obs, deterministic=True), expected, rtol=1e-5, atol=1e-6
        )
        assert restored.gradient_steps == learner.gradient_steps

    def test_load_rejects_mismatched_observation_dim(self):
        learner = self.make_learner()
        state = learner.state_dict()
        state["observation_dim"] = 99
        with pytest.raises(ValueError, match="observation_dim"):
            self.make_learner().load_state_dict(state)

    def test_load_rejects_mismatched_action_dim(self):
        learner = self.make_learner()
        state = learner.state_dict()
        state["action_dim"] = 99
        with pytest.raises(ValueError, match="action_dim"):
            self.make_learner().load_state_dict(state)

    def test_describe_is_json_serialisable(self):
        import json

        learner = self.make_learner()
        payload = json.dumps(learner.describe())
        assert '"algorithm": "sac"' in payload

    def test_seeded_learners_start_identically(self):
        obs = np.zeros((4, 4), dtype=np.float32)
        first = self.make_learner(seed=42).act(obs, deterministic=True)
        second = self.make_learner(seed=42).act(obs, deterministic=True)
        np.testing.assert_allclose(first, second)
