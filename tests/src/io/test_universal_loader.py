"""Tests for src.io.universal_loader — physics-agnostic spectral dataset loading."""
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.io.universal_loader import (
    SpectralDataset,
    load_dataset,
    load_timeseries_features,
    load_session_csv,
    list_sessions,
    merge_datasets,
    _snv,
    _msc,
    _infer_analyte,
    _infer_concentration,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def long_format_csv(tmp_path) -> Path:
    """Long-format spectrum CSV: (N_pixels, 4) with wavelength + signals."""
    wl = np.linspace(400, 900, 512)
    intensity = np.exp(-0.5 * ((wl - 700) / 30) ** 2) + 0.01 * np.random.default_rng(0).normal(size=512)
    df = pd.DataFrame({
        "wavelength": wl,
        "intensity": intensity,
        "transmittance": intensity / intensity.max(),
        "absorbance": -np.log10(np.clip(intensity / intensity.max(), 1e-9, 1)),
    })
    p = tmp_path / "1_stable.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def wide_format_csv(tmp_path) -> Path:
    """Wide-format CSV: rows=frames, columns=wavelengths."""
    wl = np.linspace(400, 900, 200)
    spectra = np.random.default_rng(1).normal(0.5, 0.1, (5, 200)).clip(0)
    df = pd.DataFrame(spectra, columns=[f"{w:.2f}" for w in wl])
    p = tmp_path / "wide_spectra.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def timeseries_csv(tmp_path) -> Path:
    """Time-series feature CSV with frame_index and delta_lambda_nm columns."""
    n = 50
    df = pd.DataFrame({
        "frame_index": np.arange(n),
        "delta_lambda_nm": np.sin(np.linspace(0, 3, n)) * 0.5,
        "peak_wavelength_nm": 865.0 + np.sin(np.linspace(0, 3, n)) * 0.5,
        "mean_signal": np.random.default_rng(2).normal(0.5, 0.05, n),
    })
    p = tmp_path / "dataset_1ppm_run1.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def spectrum_directory(tmp_path) -> Path:
    """Directory of 3 long-format CSVs at different concentrations."""
    wl = np.linspace(400, 900, 256)
    for conc in [0.5, 1.0, 5.0]:
        intensity = 0.3 + 0.1 * conc + 0.01 * np.random.default_rng(int(conc*10)).normal(size=256)
        df = pd.DataFrame({"wavelength": wl, "intensity": intensity})
        (tmp_path / f"{conc}_stable.csv").write_text(
            df.to_csv(index=False), encoding="utf-8"
        )
    return tmp_path


# ---------------------------------------------------------------------------
# SpectralDataset
# ---------------------------------------------------------------------------

class TestSpectralDataset:
    def test_repr(self):
        ds = SpectralDataset(
            wavelengths=np.linspace(400, 900, 100),
            spectra=np.random.default_rng(0).normal(size=(10, 100)),
            signal_type="intensity",
            normalisation="snv",
            analyte="Ethanol",
        )
        r = repr(ds)
        assert "n_samples=10" in r
        assert "Ethanol" in r

    def test_n_samples_n_wavelengths(self):
        ds = SpectralDataset(
            wavelengths=np.arange(50, dtype=float),
            spectra=np.ones((7, 50)),
            signal_type="intensity",
            normalisation="none",
        )
        assert ds.n_samples == 7
        assert ds.n_wavelengths == 50

    def test_wl_range(self):
        wl = np.linspace(500, 850, 100)
        ds = SpectralDataset(wavelengths=wl, spectra=np.ones((3, 100)),
                             signal_type="intensity", normalisation="none")
        assert abs(ds.wl_range[0] - 500) < 1
        assert abs(ds.wl_range[1] - 850) < 1

    def test_subset(self):
        ds = SpectralDataset(
            wavelengths=np.arange(20, dtype=float),
            spectra=np.random.default_rng(0).normal(size=(6, 20)),
            signal_type="intensity",
            normalisation="none",
            labels=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            metadata=[{"i": i} for i in range(6)],
            source_paths=[f"f{i}.csv" for i in range(6)],
        )
        mask = np.array([True, False, True, False, True, False])
        sub = ds.subset(mask)
        assert sub.n_samples == 3
        np.testing.assert_array_equal(sub.labels, [1.0, 3.0, 5.0])

    def test_split_sizes(self):
        ds = SpectralDataset(
            wavelengths=np.arange(10, dtype=float),
            spectra=np.random.default_rng(0).normal(size=(20, 10)),
            signal_type="intensity",
            normalisation="none",
        )
        train, test = ds.split(test_fraction=0.2)
        assert train.n_samples + test.n_samples == 20
        assert test.n_samples >= 1


# ---------------------------------------------------------------------------
# load_dataset — single file
# ---------------------------------------------------------------------------

class TestLoadSingleFile:
    def test_long_format_shape(self, long_format_csv):
        ds = load_dataset(long_format_csv)
        assert ds.n_samples == 1
        assert ds.n_wavelengths == 512

    def test_long_format_signal_type_auto(self, long_format_csv):
        ds = load_dataset(long_format_csv, signal_type="auto")
        assert ds.signal_type == "intensity"

    def test_long_format_signal_type_explicit(self, long_format_csv):
        ds = load_dataset(long_format_csv, signal_type="absorbance")
        assert ds.signal_type == "absorbance"

    def test_wide_format_loads(self, wide_format_csv):
        ds = load_dataset(wide_format_csv)
        assert ds.n_samples == 1
        assert ds.n_wavelengths == 200

    def test_timeseries_raises(self, timeseries_csv):
        with pytest.raises(ValueError, match="time-series feature file"):
            load_dataset(timeseries_csv)

    def test_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/spec.csv")

    def test_concentration_inferred_from_filename(self, tmp_path):
        wl = np.linspace(400, 900, 100)
        df = pd.DataFrame({"wavelength": wl, "intensity": np.ones(100)})
        p = tmp_path / "5_stable.csv"
        df.to_csv(p, index=False)
        ds = load_dataset(p)
        assert ds.labels is not None
        assert abs(ds.labels[0] - 5.0) < 0.01


# ---------------------------------------------------------------------------
# load_dataset — directory
# ---------------------------------------------------------------------------

class TestLoadDirectory:
    def test_loads_all_csvs(self, spectrum_directory):
        ds = load_dataset(spectrum_directory)
        assert ds.n_samples == 3

    def test_labels_inferred(self, spectrum_directory):
        ds = load_dataset(spectrum_directory)
        assert ds.labels is not None
        assert len(ds.labels) == 3
        assert set(ds.labels).issubset({0.5, 1.0, 5.0})

    def test_analyte_inferred_from_dir(self, tmp_path):
        ethanol_dir = tmp_path / "Ethanol"
        ethanol_dir.mkdir()
        wl = np.linspace(400, 900, 50)
        df = pd.DataFrame({"wavelength": wl, "intensity": np.ones(50)})
        (ethanol_dir / "1_stable.csv").write_text(df.to_csv(index=False))
        ds = load_dataset(ethanol_dir)
        assert ds.analyte == "Ethanol"

    def test_normalisation_snv(self, spectrum_directory):
        ds = load_dataset(spectrum_directory, normalisation="snv")
        # SNV: each row should have mean ≈ 0 and std ≈ 1
        means = ds.spectra.mean(axis=1)
        stds = ds.spectra.std(axis=1)
        np.testing.assert_allclose(means, 0.0, atol=1e-10)
        np.testing.assert_allclose(stds, 1.0, atol=1e-10)

    def test_normalisation_area(self, spectrum_directory):
        ds = load_dataset(spectrum_directory, normalisation="area")
        # Each spectrum's trapezoid integral ≈ 1.0
        from numpy import trapz
        areas = [trapz(ds.spectra[i], ds.wavelengths) for i in range(ds.n_samples)]
        np.testing.assert_allclose(areas, 1.0, rtol=0.01)

    def test_normalisation_minmax(self, spectrum_directory):
        ds = load_dataset(spectrum_directory, normalisation="minmax")
        assert ds.spectra.min() >= 0.0 - 1e-9
        assert ds.spectra.max() <= 1.0 + 1e-9

    def test_common_wavelength_grid(self, tmp_path):
        """Two files with slightly different wavelength ranges → interpolated."""
        for i, n_wl in enumerate([200, 250]):
            wl = np.linspace(400, 900, n_wl)
            df = pd.DataFrame({"wavelength": wl, "intensity": np.ones(n_wl)})
            (tmp_path / f"{i+1}_stable.csv").write_text(df.to_csv(index=False))
        ds = load_dataset(tmp_path)
        # Both spectra should have the same number of wavelength points
        assert ds.spectra.shape[0] == 2
        assert ds.spectra.shape[1] in {200, 250}  # common grid from longest


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

class TestNormalisation:
    def test_snv_zero_mean(self):
        X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                       [10.0, 20.0, 30.0, 40.0, 50.0]])
        out = _snv(X)
        np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-10)
        np.testing.assert_allclose(out.std(axis=1), 1.0, atol=1e-10)

    def test_snv_constant_row_handled(self):
        X = np.array([[5.0, 5.0, 5.0]])
        out = _snv(X)
        # std=0 → std clamped to 1; result = (5-5)/1 = 0
        np.testing.assert_allclose(out, 0.0, atol=1e-10)

    def test_msc_preserves_shape(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0.5, 0.1, (10, 100)).clip(0.01)
        out = _msc(X)
        assert out.shape == X.shape


# ---------------------------------------------------------------------------
# Metadata inference
# ---------------------------------------------------------------------------

class TestMetadataInference:
    def test_analyte_ethanol(self, tmp_path):
        assert _infer_analyte(tmp_path / "Ethanol") == "Ethanol"

    def test_analyte_ipa(self, tmp_path):
        assert _infer_analyte(tmp_path / "IPA_sensor") is not None

    def test_analyte_not_found(self, tmp_path):
        assert _infer_analyte(tmp_path / "unknown_stuff") is None

    def test_concentration_from_filename(self, tmp_path):
        assert abs(_infer_concentration(tmp_path / "5_stable.csv") - 5.0) < 0.01
        assert abs(_infer_concentration(tmp_path / "0.1_stable.csv") - 0.1) < 0.001

    def test_concentration_from_dir(self, tmp_path):
        p = (tmp_path / "10" / "run1.csv")
        assert abs(_infer_concentration(p) - 10.0) < 0.01


# ---------------------------------------------------------------------------
# load_timeseries_features
# ---------------------------------------------------------------------------

class TestLoadTimeseriesFeatures:
    def test_loads_timeseries(self, timeseries_csv):
        df = load_timeseries_features(timeseries_csv)
        assert "delta_lambda_nm" in df.columns
        assert len(df) == 50

    def test_directory_concatenates(self, tmp_path):
        for i in range(3):
            df = pd.DataFrame({
                "frame_index": np.arange(10),
                "delta_lambda_nm": np.zeros(10),
            })
            (tmp_path / f"ts_{i}.csv").write_text(df.to_csv(index=False))
        result = load_timeseries_features(tmp_path)
        assert len(result) == 30


# ---------------------------------------------------------------------------
# merge_datasets
# ---------------------------------------------------------------------------

class TestMergeDatasets:
    def test_merge_two_datasets(self, spectrum_directory, tmp_path):
        ds1 = load_dataset(spectrum_directory)
        # Create second dataset with same wavelength range
        wl = ds1.wavelengths
        spectra2 = np.random.default_rng(99).normal(size=(2, len(wl)))
        ds2 = SpectralDataset(wavelengths=wl, spectra=spectra2,
                              signal_type="intensity", normalisation="none",
                              analyte="IPA")
        merged = merge_datasets(ds1, ds2)
        assert merged.n_samples == ds1.n_samples + 2
        assert merged.analyte == "merged"

    def test_merge_same_analyte(self, spectrum_directory):
        # Pass explicit analyte so both datasets carry the same value
        ds = load_dataset(spectrum_directory, analyte="Ethanol")
        merged = merge_datasets(ds, ds)
        assert merged.n_samples == ds.n_samples * 2
        assert merged.analyte == "Ethanol"


# ---------------------------------------------------------------------------
# load_session_csv + list_sessions (C1 bridge)
# ---------------------------------------------------------------------------

@pytest.fixture
def session_dir(tmp_path) -> Path:
    """Minimal SpectraAgent session directory with 5 frames of data."""
    d = tmp_path / "20260401_120000"
    d.mkdir()
    meta = {
        "gas_label": "Ethanol",
        "session_id": "20260401_120000",
        "frame_count": 5,
        "started_at": "2026-04-01T12:00:00+00:00",
    }
    (d / "session_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    df = pd.DataFrame({
        "frame": [0, 1, 2, 3, 4],
        "timestamp": ["2026-04-01T12:00:00"] * 5,
        "peak_wavelength": [531.5, 531.4, 531.3, 531.2, 531.1],
        "wavelength_shift": [-0.11, -0.22, -0.33, -0.44, -0.55],
        "concentration_ppm": [1.0, 2.0, 3.0, 4.0, 5.0],
        "ci_low": [-0.15, -0.26, -0.37, -0.48, -0.59],
        "ci_high": [-0.07, -0.18, -0.29, -0.40, -0.51],
        "snr": [15.0, 14.5, 16.0, 15.5, 14.8],
        "gas_type": ["Ethanol"] * 5,
        "confidence_score": [0.92, 0.91, 0.93, 0.90, 0.89],
    })
    df.to_csv(d / "pipeline_results.csv", index=False)
    return d


@pytest.fixture
def empty_session_dir(tmp_path) -> Path:
    """Session directory with an empty pipeline_results.csv (header only)."""
    d = tmp_path / "20260401_130000"
    d.mkdir()
    pd.DataFrame(columns=[
        "frame", "timestamp", "peak_wavelength", "wavelength_shift",
        "concentration_ppm", "snr", "confidence_score",
    ]).to_csv(d / "pipeline_results.csv", index=False)
    return d


class TestLoadSessionCsv:
    def test_returns_spectral_dataset(self, session_dir):
        ds = load_session_csv(session_dir)
        assert isinstance(ds, SpectralDataset)

    def test_correct_shape(self, session_dir):
        ds = load_session_csv(session_dir)
        # 5 frames × 4 feature columns
        assert ds.n_samples == 5
        assert ds.n_wavelengths == 4

    def test_signal_type_is_session_features(self, session_dir):
        ds = load_session_csv(session_dir)
        assert ds.signal_type == "session_features"

    def test_analyte_from_meta(self, session_dir):
        ds = load_session_csv(session_dir)
        assert ds.analyte == "Ethanol"

    def test_config_id_is_session_id(self, session_dir):
        ds = load_session_csv(session_dir)
        assert ds.config_id == "20260401_120000"

    def test_labels_are_concentrations(self, session_dir):
        ds = load_session_csv(session_dir)
        assert ds.labels is not None
        np.testing.assert_allclose(ds.labels, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_wavelength_shift_is_first_feature(self, session_dir):
        ds = load_session_csv(session_dir)
        # feature index 0 = wavelength_shift
        np.testing.assert_allclose(
            ds.spectra[:, 0], [-0.11, -0.22, -0.33, -0.44, -0.55]
        )

    def test_feature_names_in_metadata(self, session_dir):
        ds = load_session_csv(session_dir)
        names = ds.metadata[0]["feature_names"]
        assert "wavelength_shift" in names
        assert "peak_wavelength" in names

    def test_synthetic_wavelength_axis(self, session_dir):
        ds = load_session_csv(session_dir)
        np.testing.assert_array_equal(ds.wavelengths, [0.0, 1.0, 2.0, 3.0])

    def test_normalisation_minmax(self, session_dir):
        ds = load_session_csv(session_dir, normalisation="minmax")
        # Each row should be in [0, 1] after minmax normalisation
        assert ds.spectra.min() >= -1e-9
        assert ds.spectra.max() <= 1.0 + 1e-9

    def test_empty_session_raises(self, empty_session_dir):
        with pytest.raises(ValueError, match="empty"):
            load_session_csv(empty_session_dir)

    def test_missing_csv_raises(self, tmp_path):
        d = tmp_path / "nosession"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            load_session_csv(d)

    def test_partial_columns_still_loads(self, tmp_path):
        """Session with only wavelength_shift and no other feature columns."""
        d = tmp_path / "partial"
        d.mkdir()
        pd.DataFrame({
            "frame": [0, 1],
            "wavelength_shift": [-0.1, -0.2],
            "concentration_ppm": [1.0, 2.0],
        }).to_csv(d / "pipeline_results.csv", index=False)
        ds = load_session_csv(d)
        assert ds.n_wavelengths == 1
        assert ds.signal_type == "session_features"

    def test_unknown_gas_label_gives_none_analyte(self, tmp_path):
        d = tmp_path / "unknown_gas"
        d.mkdir()
        meta = {"gas_label": "unknown", "session_id": "x", "frame_count": 2}
        (d / "session_meta.json").write_text(json.dumps(meta))
        pd.DataFrame({
            "wavelength_shift": [-0.1, -0.2],
            "concentration_ppm": [1.0, 2.0],
        }).to_csv(d / "pipeline_results.csv", index=False)
        ds = load_session_csv(d)
        assert ds.analyte is None

    def test_no_meta_file_still_loads(self, tmp_path):
        """No session_meta.json — should load without error."""
        d = tmp_path / "nometa"
        d.mkdir()
        pd.DataFrame({
            "wavelength_shift": [-0.1, -0.2],
            "concentration_ppm": [1.0, 2.0],
        }).to_csv(d / "pipeline_results.csv", index=False)
        ds = load_session_csv(d)
        assert ds.n_samples == 2
        assert ds.analyte is None

    def test_no_concentration_column_gives_nan_labels(self, tmp_path):
        d = tmp_path / "noconc"
        d.mkdir()
        pd.DataFrame({
            "wavelength_shift": [-0.1, -0.2, -0.3],
            "snr": [10.0, 11.0, 12.0],
        }).to_csv(d / "pipeline_results.csv", index=False)
        ds = load_session_csv(d)
        assert ds.labels is None or np.all(np.isnan(ds.labels))


class TestListSessions:
    def test_returns_list(self, tmp_path):
        result = list_sessions(tmp_path)
        assert isinstance(result, list)

    def test_empty_when_no_sessions(self, tmp_path):
        assert list_sessions(tmp_path) == []

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        assert list_sessions(tmp_path / "nonexistent") == []

    def test_sorted_newest_first(self, tmp_path):
        for name in ["20260101_000000", "20260301_000000", "20260201_000000"]:
            d = tmp_path / name
            d.mkdir()
            pd.DataFrame({"wavelength_shift": [0.1]}).to_csv(
                d / "pipeline_results.csv", index=False
            )
        result = list_sessions(tmp_path)
        names = [d.name for d in result]
        assert names == sorted(names, reverse=True)

    def test_excludes_dirs_without_csv(self, tmp_path):
        empty = tmp_path / "20260101_120000"
        empty.mkdir()
        # No pipeline_results.csv
        assert list_sessions(tmp_path) == []
