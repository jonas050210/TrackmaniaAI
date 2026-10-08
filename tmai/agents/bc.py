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
  normaliser's statistics are fitted only on training observations, then used for both train
  and validation data -- the same representation the network will see during RL.
* Validation is split by whole demonstration file when multiple files are loaded; a single
  recording instead uses a time-ordered holdout with a purge gap to reduce adjacent-frame
  leakage.
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


def _split_demonstrations(
    demos: Demonstration,
    *,
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Split at lap/file boundaries when known, otherwise use a purged time-ordered holdout.

    Random row-wise splits are misleading for driving data: adjacent frames are nearly
    duplicates, so a validation loss can look excellent while the network is effectively
    evaluated on observations from the training samples immediately beside them.
    """
    count = len(demos)
    if val_fraction <= 0.0:
        return np.arange(count), np.asarray([], dtype=int), "none"

    raw_lengths = demos.metadata.get("steps_per_file")
    try:
        lengths = np.asarray(raw_lengths, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError):
        lengths = np.asarray([], dtype=np.int64)

    if (
        lengths.size >= 2
        and np.all(lengths > 0)
        and int(lengths.sum()) == count
    ):
        target = max(1, int(count * val_fraction))
        order = rng.permutation(lengths.size)
        validation_groups: list[int] = []
        validation_count = 0
        for group in order:
            if len(validation_groups) >= lengths.size - 1:
                break
            if validation_count < target or not validation_groups:
                validation_groups.append(int(group))
                validation_count += int(lengths[group])
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        val_idx = np.concatenate(
            [np.arange(offsets[g], offsets[g + 1]) for g in validation_groups]
        ).astype(int)
        train_mask = np.ones(count, dtype=bool)
        train_mask[val_idx] = False
        return np.flatnonzero(train_mask), val_idx, "whole_demo_files"

    num_val = int(count * val_fraction)
    if num_val < 1:
        return np.arange(count), np.asarray([], dtype=int), "none_too_small"
    val_start = count - num_val
    # Keep a small purge window between train and validation so stacked/nearby frames do not
    # leak across the boundary. For tiny datasets, prefer a usable train set over a purge.
    purge = min(max(1, num_val // 20), max(0, val_start - 1))
    train_end = val_start - purge
    if train_end < 1:
        train_end = val_start
    train_idx = np.arange(train_end, dtype=int)
    val_idx = np.arange(val_start, count, dtype=int)
    return train_idx, val_idx, "purged_temporal_holdout"


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

    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if lr <= 0 or not np.isfinite(lr):
        raise ValueError(f"lr must be finite and positive, got {lr}")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

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
    if not np.all(np.isfinite(observations)) or not np.all(np.isfinite(actions)):
        raise ValueError("demonstration observations and actions must be finite")
    # Clip to the learner's action bounds: the game's input read-back should already be
    # inside them, but a stray out-of-range value must not poison the fit.
    actions = np.clip(actions, sac._action_low, sac._action_high)

    train_idx, val_idx, split_kind = _split_demonstrations(
        demos,
        val_fraction=val_fraction,
        rng=rng,
    )
    if len(train_idx) == 0:  # tiny dataset: train on everything, no validation
        train_idx = np.arange(len(demos))
        val_idx = np.asarray([], dtype=int)
        split_kind = "none_too_small"

    if normalizer is not None:
        # Validation observations must not influence normalization statistics. Fit only on the
        # training subset, then transform both partitions with those frozen running moments.
        normalizer.update(observations[train_idx])
        observations = normalizer.normalize(observations)

    # Tensors that feed the network live on the learner's device (CPU or CUDA). Index
    # bookkeeping stays on the host, so the batch order is a property of ``seed`` alone:
    # the same seed yields the same shuffle on either device, and the ambient global torch
    # RNG state (which differs between CPU and CUDA draws) plays no part in it.
    obs_t = torch.as_tensor(observations, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    val_idx_t = torch.as_tensor(val_idx, dtype=torch.long, device=device)
    shuffle_gen = torch.Generator(device="cpu")
    shuffle_gen.manual_seed(int(rng.integers(0, 2**63 - 1)))
    train_order = np.asarray(train_idx, dtype=np.int64)

    policy = sac.network.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    train_loss = float("nan")
    val_loss = float("nan")
    for epoch in range(epochs):
        if shuffle:
            perm = torch.randperm(len(train_order), generator=shuffle_gen).numpy()
            epoch_order = train_order[perm]
        else:
            epoch_order = train_order
        order_t = torch.as_tensor(epoch_order, dtype=torch.long, device=device)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(order_t), batch_size):
            idx = order_t[start:start + batch_size]
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
                val_action, _ = policy.sample(obs_t[val_idx_t], deterministic=True)
                val_loss = float(F.mse_loss(val_action, act_t[val_idx_t]).item())
        if log_every and (epoch + 1) % log_every == 0:
            logger.info(
                "bc epoch %d/%d: train_loss=%.5f val_loss=%.5f",
                epoch + 1, epochs, train_loss, val_loss,
            )

    metrics = {
        "bc/samples": float(len(demos)),
        "bc/train_samples": float(len(train_idx)),
        "bc/val_samples": float(len(val_idx)),
        "bc/whole_file_validation": float(split_kind == "whole_demo_files"),
        "bc/epochs": float(epochs),
        "bc/final_train_loss": train_loss,
        "bc/val_loss": val_loss,
    }
    logger.info(
        "behaviour cloning done: %d samples (%d train, %d validation; %s split), "
        "%d epochs, final train loss %.5f, val %.5f",
        len(demos), len(train_idx), len(val_idx), split_kind, epochs, train_loss, val_loss,
    )
    return metrics


__all__ = ["pretrain_policy"]
