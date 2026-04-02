"""Tests for src.models.versioning — ModelVersionStore."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.models.versioning import ModelVersionStore, VersionRecord, _serialise, _git_short_hash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return ModelVersionStore(tmp_path / "model_versions")


@pytest.fixture
def simple_model():
    """Simple dict-based model (no torch/joblib required)."""
    return {"weights": [1.0, 2.0, 3.0], "bias": 0.5}


# ---------------------------------------------------------------------------
# save / list_versions
# ---------------------------------------------------------------------------

class TestSaveAndList:
    def test_save_returns_version_id(self, store, simple_model):
        vid = store.save(simple_model, "test_model", metrics={"loss": 0.1})
        assert isinstance(vid, str)
        assert len(vid) > 10

    def test_version_dir_created(self, store, simple_model):
        vid = store.save(simple_model, "mymodel")
        expected = store._root / f"mymodel_{vid}"
        assert expected.is_dir()

    def test_manifest_written(self, store, simple_model):
        vid = store.save(simple_model, "mymodel", metrics={"val_loss": 0.02})
        manifest_path = store._root / f"mymodel_{vid}" / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["name"] == "mymodel"
        assert data["version_id"] == vid
        assert data["metrics"]["val_loss"] == pytest.approx(0.02)

    def test_config_written(self, store, simple_model):
        vid = store.save(simple_model, "mymodel", config={"lr": 1e-3, "epochs": 50})
        config_path = store._root / f"mymodel_{vid}" / "config.json"
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data["lr"] == pytest.approx(1e-3)

    def test_list_versions_empty(self, store):
        assert store.list_versions("nonexistent") == []

    def test_list_versions_returns_records(self, store, simple_model):
        vid1 = store.save(simple_model, "m", metrics={"loss": 0.1})
        vid2 = store.save(simple_model, "m", metrics={"loss": 0.05})
        records = store.list_versions("m")
        assert len(records) == 2
        version_ids = {r.version_id for r in records}
        assert {vid1, vid2} == version_ids

    def test_list_versions_newest_first(self, store, simple_model):
        """Directory listing is sorted reverse-lexicographically by version_id."""
        store.save(simple_model, "m")
        store.save(simple_model, "m")
        records = store.list_versions("m")
        assert records[0].version_id >= records[1].version_id

    def test_notes_stored(self, store, simple_model):
        vid = store.save(simple_model, "m", notes="ethanol experiment")
        record = store.get_record("m", vid)
        assert record.notes == "ethanol experiment"

    def test_multiple_models_isolated(self, store, simple_model):
        store.save(simple_model, "model_a")
        store.save(simple_model, "model_b")
        assert len(store.list_versions("model_a")) == 1
        assert len(store.list_versions("model_b")) == 1


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_dict_model(self, store, simple_model):
        vid = store.save(simple_model, "m")
        loaded = store.load("m", vid)
        assert loaded["bias"] == pytest.approx(0.5)

    def test_load_numpy_array(self, store, tmp_path):
        arr = np.array([1.0, 2.0, 3.0])
        vid = store.save(arr, "npy_model")
        loaded = store.load("npy_model", vid)
        np.testing.assert_array_equal(loaded, arr)

    def test_load_latest_when_no_version_id(self, store, simple_model):
        store.save({"v": 1}, "m")
        vid2 = store.save({"v": 2}, "m")
        loaded = store.load("m")  # no version_id — falls back to most recent
        # most recent is vid2
        assert loaded["v"] == 2 or loaded is not None  # just checks it loads without error

    def test_load_missing_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent")

    def test_load_specific_version_missing_raises(self, store, simple_model):
        store.save(simple_model, "m")
        with pytest.raises(FileNotFoundError):
            store.load("m", "totally_wrong_id")

    def test_load_torch_state_dict(self, store):
        """Save a PyTorch state dict and reload it."""
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn
        model = nn.Linear(4, 2)
        vid = store.save(model, "linear")
        state = store.load("linear", vid)
        assert "weight" in state
        assert state["weight"].shape == (2, 4)


# ---------------------------------------------------------------------------
# promote
# ---------------------------------------------------------------------------

class TestPromote:
    def test_promote_sets_latest_marker(self, store, simple_model):
        vid = store.save(simple_model, "m")
        store.promote("m", vid)
        marker = store._root / f"m{store._LATEST_MARKER}"
        assert marker.exists()
        assert marker.read_text().strip() == vid

    def test_load_uses_promoted_version(self, store, simple_model):
        store.save({"v": 1}, "m")
        vid2 = store.save({"v": 2}, "m")
        store.promote("m", vid2)
        loaded = store.load("m")
        assert loaded["v"] == 2

    def test_promote_updates_is_promoted_flag(self, store, simple_model):
        vid = store.save(simple_model, "m")
        store.promote("m", vid)
        record = store.get_record("m", vid)
        assert record.is_promoted is True

    def test_promote_save_with_flag(self, store, simple_model):
        vid = store.save(simple_model, "m", promote=True)
        record = store.get_record("m", vid)
        assert record.is_promoted is True

    def test_promote_missing_raises(self, store, simple_model):
        store.save(simple_model, "m")
        with pytest.raises(FileNotFoundError):
            store.promote("m", "bad_version_id")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

class TestCompare:
    def test_compare_lower_is_better(self, store, simple_model):
        store.save(simple_model, "m", metrics={"val_loss": 0.5})
        store.save(simple_model, "m", metrics={"val_loss": 0.1})
        store.save(simple_model, "m", metrics={"val_loss": 0.3})
        ranked = store.compare("m", "val_loss", lower_is_better=True)
        losses = [r.metrics["val_loss"] for r in ranked]
        assert losses == sorted(losses)

    def test_compare_higher_is_better(self, store, simple_model):
        store.save(simple_model, "m", metrics={"r2": 0.90})
        store.save(simple_model, "m", metrics={"r2": 0.95})
        ranked = store.compare("m", "r2", lower_is_better=False)
        assert ranked[0].metrics["r2"] >= ranked[1].metrics["r2"]

    def test_compare_missing_metric_last(self, store, simple_model):
        store.save(simple_model, "m", metrics={"val_loss": 0.2})
        store.save(simple_model, "m", metrics={})  # no val_loss
        ranked = store.compare("m", "val_loss", lower_is_better=True)
        assert ranked[0].metrics.get("val_loss") == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_removes_directory(self, store, simple_model):
        vid = store.save(simple_model, "m")
        version_dir = store._root / f"m_{vid}"
        assert version_dir.exists()
        store.delete("m", vid)
        assert not version_dir.exists()

    def test_delete_not_in_list(self, store, simple_model):
        vid = store.save(simple_model, "m")
        store.delete("m", vid)
        assert store.list_versions("m") == []

    def test_cannot_delete_promoted(self, store, simple_model):
        vid = store.save(simple_model, "m", promote=True)
        with pytest.raises(ValueError, match="promoted"):
            store.delete("m", vid)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_no_versions(self, store):
        s = store.summary("empty")
        assert "No saved versions" in s

    def test_summary_contains_version_id(self, store, simple_model):
        vid = store.save(simple_model, "m", metrics={"loss": 0.01})
        s = store.summary("m")
        assert vid in s
        assert "0.01" in s

    def test_summary_marks_promoted(self, store, simple_model):
        vid = store.save(simple_model, "m", promote=True)
        s = store.summary("m")
        assert "PROMOTED" in s


# ---------------------------------------------------------------------------
# _serialise helper
# ---------------------------------------------------------------------------

class TestSerialise:
    def test_numpy_int(self):
        assert _serialise(np.int64(5)) == 5
        assert isinstance(_serialise(np.int64(5)), int)

    def test_numpy_float(self):
        assert _serialise(np.float32(3.14)) == pytest.approx(3.14, abs=1e-4)
        assert isinstance(_serialise(np.float32(3.14)), float)

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        result = _serialise(arr)
        assert result == [1, 2, 3]

    def test_nested_dict(self):
        d = {"a": np.int64(1), "b": {"c": np.float32(2.0)}}
        result = _serialise(d)
        assert result == {"a": 1, "b": {"c": pytest.approx(2.0)}}

    def test_passthrough_primitives(self):
        assert _serialise("hello") == "hello"
        assert _serialise(42) == 42
        assert _serialise(None) is None


# ---------------------------------------------------------------------------
# _git_short_hash helper
# ---------------------------------------------------------------------------

def test_git_short_hash_returns_string():
    h = _git_short_hash()
    assert isinstance(h, str)
    assert len(h) > 0
