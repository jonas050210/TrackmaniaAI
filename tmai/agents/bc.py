"""Behaviour cloning: supervised pretraining of the policy from human demonstrations.

SAC starts from a randomly initialised policy and has to discover "throttle drives the car"
from reward alone. When demonstrations exist, a supervised pass teaches the actor the
human's observation -> action mapping first, which typically removes the longest, most
boring phase of training. This is warm-starting, not imitation learning as the training
objective: SAC takes over afterwards and can improve on the demonstrations.

Design notes:

* The pretraining mutates the SAC learner **in place** (its actor's weights), so the very
  next gradient step of the RL loop continues from the pretrained policy. There is no
  separate BC policy to keep in sync.
* When the learner is wrapped in :class:`~tmai.agents.normalize.NormalizingLearner`, the
  normaliser's statistics are first fitted on the demonstration observations and the demos
  are then normalised with them -- the same representation the network will see during RL.
* The loss is MSE between the policy's deterministic (mean) action and the recorded
  action, in the *bounded* action space, so the squashing transform is part of the fit.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from tmai.agents.base import Learner
from tmai.training.demos import Demonstration

logger = logging.getLogger(__name__)


def _unwrap_sac(learner: Learner):
    """Return ``(sac_learner, normalizer_or_None)`` for a (possibly wrapped) learner."""
    from tmai.agents.normalize import NormalizingLearner
    from tmai.agents.sac import SACLearner

    if isinstance(learner, NormalizingLearner):
        return learner.inner, learner.normalizer
    if isinstance(learner, SACLearner):
        return learner, None
    raise TypeError(
        f"behaviour cloning pretrains a SAC learner, got {type(learner).__name__}; "
        "wrap or replace it with SAC first"
    )


def pretrain_policy(
    learner: Learner,
    demos: Demonstration,
    *,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    shuffle: bool = True,
    seed: int | None = None,
    log_every: int = 0,
) -> dict[str, float]:
    """Supervised-pretrain ``learner``'s actor on ``demos``; returns scalar metrics.

    Args:
        learner: the (possibly normalisation-wrapped) SAC learner to warm-start.
        demos: the demonstration dataset; its dimensions must match the learner.
        epochs / batch_size / lr: supervised optimisation settings.
        val_fraction: fraction of the data held out for a validation loss.
        shuffle: shuffle each epoch (seeded, so reproducible).
        seed: RNG seed for the shuffle and the split.
        log_every: log every N epochs (0 = silent).
    """
    sac, normalizer = _unwrap_sac(learner)

    if demos.observation_dim != sac.observation_dim:
        raise ValueError(
            f"demonstrations have observation_dim {demos.observation_dim}, learner expects "
            f"{sac.observation_dim}; record the demos against the same observation layout"
        )
    if demos.action_dim != sac.action_dim:
        raise ValueError(
            f"demonstrations have action_dim {demos.action_dim}, learner expects "
            f"{sac.action_dim}"
        )
    if len(demos) < 2:
        raise ValueError(f"need at least 2 demonstration steps to pretrain, got {len(demos)}")

    device = sac.device
    rng = np.random.default_rng(seed)

    observations = np.asarray(demos.observations, dtype=np.float32)
    actions = np.asarray(demos.actions, dtype=np.float32)
    # Clip to the learner's action bounds: the game's input read-back should already be
    # inside them, but a stray out-of-range value must not poison the fit.
    actions = np.clip(actions, sac._action_low, sac._action_high)

    if normalizer is not None:
        # Fit the normaliser on the demonstrations so the network trains on the same
        # representation it will see during RL.
        normalizer.update(observations)
        observations = normalizer.normalize(observations)

    # Deterministic train/val split.
    order = rng.permutation(len(demos)) if shuffle else np.arange(len(demos))
    num_val = int(len(demos) * val_fraction)
    val_idx = order[:num_val]
    train_idx = order[num_val:]
    if len(train_idx) == 0:  # tiny dataset: train on everything, no val
        train_idx = order
        val_idx = np.asarray([], dtype=int)

    obs_t = torch.as_tensor(observations, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(actions, dtype=torch.float32, device=device)

    policy = sac.network.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    train_loss = float("nan")
    val_loss = float("nan")
    for epoch in range(epochs):
        if shuffle:
            perm = torch.randperm(len(train_idx), device=device)
            train_idx_t = torch.as_tensor(train_idx, device=device)[perm]
        else:
            train_idx_t = torch.as_tensor(train_idx, device=device)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(train_idx_t), batch_size):
            idx = train_idx_t[start:start + batch_size]
            if len(idx) == 0:
                continue
            mean_action, _ = policy.sample(obs_t[idx], deterministic=True)
            loss = F.mse_loss(mean_action, act_t[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        train_loss = epoch_loss / max(batches, 1)
        if len(val_idx):
            with torch.no_grad():
                val_action, _ = policy.sample(obs_t[torch.as_tensor(val_idx, device=device)],
                                               deterministic=True)
                val_loss = float(F.mse_loss(
                    val_action, act_t[torch.as_tensor(val_idx, device=device)]
                ).item())
        if log_every and (epoch + 1) % log_every == 0:
            logger.info(
                "bc epoch %d/%d: train_loss=%.5f val_loss=%.5f",
                epoch + 1, epochs, train_loss, val_loss,
            )

    metrics = {
        "bc/samples": float(len(demos)),
        "bc/train_samples": float(len(train_idx)),
        "bc/epochs": float(epochs),
        "bc/final_train_loss": train_loss,
        "bc/val_loss": val_loss,
    }
    logger.info(
        "behaviour cloning done: %d samples, %d epochs, final train loss %.5f, val %.5f",
        len(demos), epochs, train_loss, val_loss,
    )
    return metrics


__all__ = ["pretrain_policy"]
