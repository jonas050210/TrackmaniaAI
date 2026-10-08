"""The model registry: named, self-contained, versioned models."""

from __future__ import annotations

import json

import pytest

from tmai.registry import ModelInfo, ModelStore, RegistryError
from tmai.training.checkpoint import save_checkpoint


def _learner():
    from tmai.agents.sac import SACLearner
    from tmai.config import RunConfig
    from tmai.models.networks import NetworkConfig

    config = RunConfig().sac
    config.network = NetworkConfig(hidden_sizes=[16, 16])
    import numpy as np

    return SACLearner(
        observation_dim=4,
        action_dim=3,
        config=config,
        action_low=np.array([-1.0, 0.0, 0.0]),
        action_high=np.array([1.0, 1.0, 1.0]),
        seed=0,
    )


@pytest.fixture
def checkpoint(tmp_path):
    learner = _learner()
    return save_checkpoint(
        tmp_path / "run", step=123, learner=learner, config={"sac": {}}, best_score=0.5
    )


class TestRegister:
    def test_register_copies_checkpoint(self, tmp_path, checkpoint):
        store = ModelStore(tmp_path / "models")
        info = store.register("my-model", checkpoint, tags=["a"], notes="hello")
        assert isinstance(info, ModelInfo)
        assert (info.directory / "model.json").is_file()
        assert (info.directory / "policy.pt").is_file()
        assert info.step == 123
        assert info.best_score == pytest.approx(0.5)
        assert info.observation_dim == 4
        assert info.action_dim == 3
        assert info.tags == ["a"]
        assert info.notes == "hello"
        # The registry holds its own copy: deleting the source changes nothing.
        source = info.weights_path.read_bytes()
        checkpoint.unlink()
        assert info.weights_path.read_bytes() == source

    def test_register_unwraps_normalising_learner(self, tmp_path):
        from tmai.agents.normalize import NormalizingLearner
        from tmai.env.normalization import RunningNormalizer

        learner = NormalizingLearner(_learner(), RunningNormalizer(4, warmup_steps=0))
        ckpt = save_checkpoint(tmp_path / "run", step=1, learner=learner)
        store = ModelStore(tmp_path / "models")
        info = store.register("wrapped", ckpt)
        assert info.observation_dim == 4
        assert info.action_dim == 3

    def test_duplicate_name_rejected(self, tmp_path, checkpoint):
        store = ModelStore(tmp_path / "models")
        store.register("dup", checkpoint)
        with pytest.raises(RegistryError, match="already exists"):
            store.register("dup", checkpoint)
        # ... unless overwrite is requested.
        info = store.register("dup", checkpoint, overwrite=True)
        assert info.name == "dup"

    def test_invalid_names_rejected(self, tmp_path, checkpoint):
        store = ModelStore(tmp_path / "models")
        for bad in ("../escape", "a/b", "", ".hidden", "with space", "x" * 100):
            with pytest.raises(RegistryError, match="invalid model name"):
                store.register(bad, checkpoint)

    def test_missing_checkpoint(self, tmp_path):
        store = ModelStore(tmp_path / "models")
        with pytest.raises(RegistryError, match="checkpoint not found"):
            store.register("m", tmp_path / "nope.pt")

    def test_corrupt_checkpoint(self, tmp_path):
        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"not a checkpoint")
        store = ModelStore(tmp_path / "models")
        with pytest.raises(RegistryError, match="cannot read checkpoint"):
            store.register("m", bad)


class TestListGetDelete:
    def test_list_sorted_and_skips_junk(self, tmp_path, checkpoint):
        store = ModelStore(tmp_path / "models")
        store.register("b-model", checkpoint)
        store.register("a-model", checkpoint)
        (tmp_path / "models" / "not-a-model").mkdir()  # no model.json inside
        (tmp_path / "models" / "broken").mkdir()
        (tmp_path / "models" / "broken" / "model.json").write_text("{not json")
        names = [m.name for m in store.list()]
        assert names == ["a-model", "b-model"]

    def test_get_unknown(self, tmp_path):
        store = ModelStore(tmp_path / "models")
        with pytest.raises(RegistryError, match="no model named"):
            store.get("ghost")

    def test_delete(self, tmp_path, checkpoint):
        store = ModelStore(tmp_path / "models")
        store.register("bye", checkpoint)
        store.delete("bye")
        assert store.list() == []
        with pytest.raises(RegistryError):
            store.delete("bye")

    def test_add_tags_merges_without_duplicates(self, tmp_path, checkpoint):
        store = ModelStore(tmp_path / "models")
        store.register("m", checkpoint, tags=["a", "b"])
        info = store.add_tags("m", ["b", "c"])
        assert info.tags == ["a", "b", "c"]
        meta = json.loads((info.directory / "model.json").read_text())
        assert meta["tags"] == ["a", "b", "c"]

    def test_set_evaluation_and_notes(self, tmp_path, checkpoint):
        store = ModelStore(tmp_path / "models")
        store.register("m", checkpoint)
        info = store.set_evaluation("m", {"finish_rate": 0.9})
        assert info.evaluation == {"finish_rate": 0.9}
        info = store.set_notes("m", "trained overnight")
        assert info.notes == "trained overnight"
        # Both survive a reload from disk.
        reloaded = ModelStore(tmp_path / "models").get("m")
        assert reloaded.evaluation == {"finish_rate": 0.9}
        assert reloaded.notes == "trained overnight"

    def test_empty_store(self, tmp_path):
        assert ModelStore(tmp_path / "nothing").list() == []
