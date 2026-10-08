# Algorithm selection

The requirement was explicit: do not assume PPO. This is the reasoning that was actually
applied, and where it landed.

## Requirements that constrain the choice

| Requirement | Consequence for the algorithm |
|---|---|
| Samples come from **one real game** running in real time (accelerated at best) | Sample efficiency dominates everything. Transitions are the expensive resource. |
| Learning **from scratch**, no demonstrations, no pre-trained policy | Needs principled exploration, not a hand-tuned epsilon schedule. |
| **Analog** steering and throttle (TMInterface injects `[-65536, 65536]`) | Continuous action space; bounded. |
| Learn steering, acceleration, braking **and drifting** | High stochasticity, contact-rich, multimodal good behaviour. |
| Reward is dense (track progress) but **hackable** | The algorithm must tolerate shaped rewards without collapsing onto a degenerate policy. |
| **Long training runs** over days | Must be resumable and stable under slow, irregular sample arrival. |
| Eventually generalise across maps | Representation matters more than the specific optimiser; the network must be swappable. |

## Why not PPO

PPO is on-policy: it collects a batch, takes a few epochs of gradient steps, then **discards
the batch**. On a simulator generating millions of steps per minute that is affordable. Here
the bottleneck is a single game instance producing tens to low hundreds of control steps per
second even with the game-speed multiplier. Throwing away every transition after a handful of
reuses is the one thing this project can least afford — empirically, on-policy methods need
one to two orders of magnitude more environment interaction than off-policy ones for
comparable continuous-control performance.

Two further mismatches: PPO's Gaussian policy ignores action bounds and relies on clipping,
while our actions are genuinely bounded; and PPO's advantage estimation wants full episodes,
which interacts badly with episodes that end on a wall-clock limit.

PPO is not excluded — it is a legitimate future option for stability-critical fine-tuning, and
the `Learner` protocol exists so it can be added without touching anything else.

## Why SAC

Soft Actor-Critic is the natural fit:

* **Off-policy.** A replay buffer reuses every expensive transition many times. This is the
  decisive property here.
* **Continuous, bounded actions.** The squashed-Gaussian policy (`a = tanh(μ(s) + σ(s)·ε)`,
  rescaled into the action bounds) is designed for exactly this space.
* **Maximum entropy gives principled exploration.** The policy stays as random as possible
  while maximising reward, which keeps it from collapsing onto a single line early — important
  when learning drifting, where several distinct strategies can be good.
* **Twin critics** (clipped double-Q) limit the over-estimation that makes value-based methods
  diverge on shaped rewards.
* **Automatic temperature tuning** removes one hand-tuned hyper-parameter, which matters when
  a run lasts days and nobody is watching to adjust it.

It is also the algorithm the existing Trackmania RL work converged on independently, which is
weak but real evidence that it suits this domain.

## The network/algorithm split

The requirement was that a neural network serves as the policy/value model while the RL
algorithm trains it. That is enforced structurally, not by convention:

```
tmai/models/networks.py     ActorCriticNetwork
                            ├─ GaussianPolicy      μ(s), σ(s) → tanh-squashed action
                            ├─ TwinCritic          Q1(s,a), Q2(s,a)
                            └─ critic_target       slowly-updated copy
                            No loss. No optimiser. No RL.

tmai/agents/sac.py          SACLearner
                            ├─ owns both optimisers and the entropy temperature
                            ├─ owns the losses and the Polyak update
                            └─ exposes act() / update() / state_dict()
```

`tmai.models` imports nothing from `tmai.agents`. Replacing SAC means implementing the
`Learner` protocol (`tmai/agents/base.py`): `act`, `update`, `state_dict`, `load_state_dict`,
`describe`. The network can be reused unchanged.

## Warm starts: temporal observations and behaviour cloning

Two mechanisms shorten the distance from a random policy to a driving one, and both are
orthogonal to the algorithm choice:

* **Temporal observations.** `env.observation.history_length` stacks the last N frames, so
  the policy sees *changes* (acceleration, steering rate) rather than a single snapshot. The
  stack is environment-owned and reset-filled, so no zero-padded window appears mid-episode.
* **Behaviour cloning.** `tmai pretrain` (or `bc:` in the config) fits the actor to recorded
  human demonstrations before RL begins. It is a warm start, not the training objective:
  SAC continues afterwards and can improve on the demonstrations. Skipped on resume, where
  the checkpoint already carries its warm start.

## Implementation notes that matter

* **Truncation vs termination.** `terminated` and `truncated` are stored separately and only
  `terminated` stops bootstrapping: `y = r + γ(1 − terminated)(Q_target − α·logπ)`. Treating a
  time limit as terminal biases the value function downward. Covered by
  `test_terminal_states_do_not_bootstrap` and `test_truncated_states_do_bootstrap`.
* **Log-probability correctness.** The tanh-squashed log-prob includes the Jacobian term
  `−log(1 − tanh(u)²)` and the affine rescale into `[low, high]`. A wrong log-prob silently
  corrupts both the actor loss and the entropy temperature. It is verified against a
  numerically computed density in `test_log_prob_matches_numerical_density`.
* **Update-to-data ratio.** `train.updates_per_step` decouples gradient steps from environment
  steps. Above 1 it makes expensive samples pay for themselves; too high over-fits the buffer.
  It is configurable rather than fixed.
* **Verified fixed point.** `test_critic_learns_a_constant_reward` trains on `r = 1`,
  `γ = 0.9` with no terminals and asserts `Q → 1/(1−γ) = 10`. A critic that did not bootstrap
  would sit near 1; one that double-counted would diverge. This is the test that proves the
  Bellman target is wired up correctly.

## What is deliberately not done yet

| Idea | Why deferred |
|---|---|
| REDQ / DroQ (higher UTD without divergence) | Real value, but adds complexity before the baseline is validated on the real game. |
| Recurrent policy (LSTM/GRU) | Needed for partial observability (vision). Premature for a state-based observation. |
| Observation normalisation | The fixed normalisation constants in `ObservationScales` are sufficient for a state-based observation; a running normaliser is a small addition when it is needed. |
| Distributional critics | No evidence it is the bottleneck. |
| Multi-instance rollout | A throughput optimisation; needs a vectorised env and a learner service first. |

## Tuning starting points

`tmai/configs/default.yaml` is a defensible starting point, not a tuned result:

```yaml
sac:
  gamma: 0.99            # ~1 s effective horizon at a 50 ms control step
  tau: 0.005             # slow target updates; raise if the critic lags badly
  actor_lr / critic_lr: 3e-4
  initial_temperature: 0.2
  learnable_temperature: true
train:
  updates_per_step: 1.0  # first thing to raise if the game is the bottleneck
  warmup_steps: 5000     # random actions before the first gradient step
```

If samples are the bottleneck (they will be), the highest-leverage changes are, in order:
raise `driver.speed_ratio`, raise `train.updates_per_step`, then raise `env.action_repeat`
(cheaper control, coarser policy).
