"""Device handling across CPU and CUDA.

These tests need a CUDA device and are skipped otherwise. The CPU suite never runs them (CI
hides CUDA with ``TMAI_TEST_DEVICE=cpu``); a GPU host runs them with the default device setting.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.test_bc import _demo_from_policy, _make_learner
from tmai.agents.bc import pretrain_policy

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


@requires_cuda
def test_cuda_checkpoint_loads_on_cpu(tmp_path):
    from tmai.training.checkpoint import load_checkpoint, save_checkpoint

    gpu = _make_learner(device="cuda")
    observations = np.random.default_rng(3).normal(size=(16, 4)).astype(np.float32)
    expected = np.stack([gpu.act(o, deterministic=True) for o in observations])

    path = save_checkpoint(tmp_path / "gpu", step=1, learner=gpu, config={})
    payload = load_checkpoint(path)  # map_location defaults to CPU

    cpu = _make_learner(device="cpu")
    cpu.load_state_dict(payload["learner"])
    assert cpu.device.type == "cpu"
    got = np.stack([cpu.act(o, deterministic=True) for o in observations])
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


@requires_cuda
def test_cpu_and_cuda_pretraining_agree():
    """Same seed, same batch order on both devices, so the losses agree up to float noise."""
    demos = _demo_from_policy()
    cpu = pretrain_policy(_make_learner(device="cpu"), demos, epochs=3, seed=7)
    gpu = pretrain_policy(_make_learner(device="cuda"), demos, epochs=3, seed=7)
    assert gpu["bc/final_train_loss"] == pytest.approx(cpu["bc/final_train_loss"], rel=1e-2)
    assert gpu["bc/val_loss"] == pytest.approx(cpu["bc/val_loss"], rel=1e-2)
