"""Tests for src.io.dataset_registry — unified multi-dataset registry."""
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.io.dataset_registry import DatasetEntry, DatasetRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spectrum_dir(tmp_path) -> Path:
    """Directory with two concentration CSVs."""
    wl = np.linspace(400, 900, 128)
    for conc in [1.0, 5.0]:
        df = pd.DataFrame({"wavelength": wl, "intensity": np.ones(128) * conc * 0.1})
        (tmp_path / f"{conc}_stable.csv").write_text(df.to_csv(index=False))
    return tmp_path


@pytest.fixture
def registry(spectrum_dir) -> DatasetRegistry:
    r = DatasetRegistry()
    r.register(
        name="Ethanol_A",
        path=str(spectrum_dir),
        analyte="Ethanol",
        config_id="CCS200_v1",
        normalisation="snv",
        description="Ethanol test dataset",
        tags=["calibration", "real"],
    )
    return r


# ---------------------------------------------------------------------------
# DatasetEntry
# ---------------------------------------------------------------------------

class TestDatasetEntry:
    def test_to_dict_roundtrip(self):
        entry = DatasetEntry(name="Test", path="/data/test",
                             analyte="Ethanol", config_id="CCS200")
        d = entry.to_dict()
        entry2 = DatasetEntry.from_dict(d)
        assert entry2.name == entry.name
        assert entry2.analyte == entry.analyte

    def test_default_tags_empty(self):
        entry = DatasetEntry(name="X", path="/x")
        assert entry.tags == []


# ---------------------------------------------------------------------------
# DatasetRegistry — registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_and_contains(self, spectrum_dir):
        r = DatasetRegistry()
        r.register("DS1", spectrum_dir)
        assert "DS1" in r

    def test_duplicate_raises(self, spectrum_dir):
        r = DatasetRegistry()
        r.register("DS1", spectrum_dir)
        with pytest.raises(ValueError, match="already registered"):
            r.register("DS1", spectrum_dir)

    def test_overwrite_allowed(self, spectrum_dir):
        r = DatasetRegistry()
        r.register("DS1", spectrum_dir, description="v1")
        r.register("DS1", spectrum_dir, description="v2", overwrite=True)
        assert r.get_entry("DS1").description == "v2"

    def test_unregister(self, registry):
        registry.unregister("Ethanol_A")
        assert "Ethanol_A" not in registry

    def test_unregister_missing_raises(self, registry):
        with pytest.raises(KeyError):
            registry.unregister("NonExistent")

    def test_len(self, registry):
        assert len(registry) == 1

    def test_chaining(self, spectrum_dir, tmp_path):
        r = DatasetRegistry()
        wl = np.linspace(400, 900, 50)
        df = pd.DataFrame({"wavelength": wl, "intensity": np.ones(50)})
        p2 = tmp_path / "other"
        p2.mkdir()
        (p2 / "1_stable.csv").write_text(df.to_csv(index=False))
        result = r.register("A", spectrum_dir).register("B", p2)
        assert result is r
        assert len(r) == 2


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestLoading:
    def test_load_returns_dataset(self, registry):
        ds = registry.load("Ethanol_A")
        assert ds.n_samples == 2
        assert ds.analyte == "Ethanol"

    def test_load_missing_raises(self, registry):
        with pytest.raises(KeyError):
            registry.load("NotRegistered")

    def test_load_normalisation_override(self, registry):
        ds = registry.load("Ethanol_A", normalisation="minmax")
        assert ds.spectra.min() >= 0.0 - 1e-9
        assert ds.spectra.max() <= 1.0 + 1e-9

    def test_load_merged(self, tmp_path):
        # Use isolated subdirs so recursive loading of A doesn't reach B's files
        wl = np.linspace(400, 900, 128)
        ds1_dir = tmp_path / "ds1"
        ds1_dir.mkdir()
        for conc in [1.0, 5.0]:
            df = pd.DataFrame({"wavelength": wl, "intensity": np.ones(128) * conc * 0.1})
            (ds1_dir / f"{conc}_stable.csv").write_text(df.to_csv(index=False))

        ds2_dir = tmp_path / "ds2"
        ds2_dir.mkdir()
        df = pd.DataFrame({"wavelength": wl, "intensity": np.ones(128) * 0.3})
        (ds2_dir / "2_stable.csv").write_text(df.to_csv(index=False))

        r = DatasetRegistry()
        r.register("A", ds1_dir, analyte="Ethanol")
        r.register("B", ds2_dir, analyte="Ethanol")
        merged = r.load_merged(["A", "B"])
        assert merged.n_samples == 3  # 2 from A + 1 from B

    def test_load_by_analyte(self, registry):
        ds = registry.load_by_analyte("Ethanol")
        assert ds.n_samples == 2

    def test_load_by_analyte_missing_raises(self, registry):
        with pytest.raises(KeyError):
            registry.load_by_analyte("Acetone")

    def test_load_by_config(self, registry):
        ds = registry.load_by_config("CCS200_v1")
        assert ds.n_samples == 2

    def test_load_by_tag(self, registry):
        ds = registry.load_by_tag("calibration")
        assert ds.n_samples == 2

    def test_load_by_tag_missing_raises(self, registry):
        with pytest.raises(KeyError):
            registry.load_by_tag("simulation")


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

class TestIntrospection:
    def test_list_names(self, registry):
        assert "Ethanol_A" in registry.list_names()

    def test_list_analytes(self, registry):
        assert "Ethanol" in registry.list_analytes()

    def test_list_configs(self, registry):
        assert "CCS200_v1" in registry.list_configs()

    def test_summary_contains_name(self, registry):
        s = registry.summary()
        assert "Ethanol_A" in s

    def test_repr(self, registry):
        r = repr(registry)
        assert "DatasetRegistry" in r


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_file(self, registry, tmp_path):
        p = tmp_path / "reg.json"
        registry.save(p)
        assert p.exists()

    def test_save_load_roundtrip(self, registry, tmp_path):
        p = tmp_path / "reg.json"
        registry.save(p)
        r2 = DatasetRegistry.from_file(p)
        assert "Ethanol_A" in r2
        entry = r2.get_entry("Ethanol_A")
        assert entry.analyte == "Ethanol"
        assert entry.config_id == "CCS200_v1"
        assert "calibration" in entry.tags

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DatasetRegistry.from_file(tmp_path / "nonexistent.json")

    def test_json_is_human_readable(self, registry, tmp_path):
        p = tmp_path / "reg.json"
        registry.save(p)
        data = json.loads(p.read_text())
        assert "entries" in data
        assert "version" in data
