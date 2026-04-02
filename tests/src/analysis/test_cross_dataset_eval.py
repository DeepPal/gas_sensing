"""Tests for src.analysis.cross_dataset_eval — cross-config benchmark."""
import numpy as np
import pytest

from src.analysis.cross_dataset_eval import (
    BenchmarkConfig,
    BenchmarkResult,
    ConfigResult,
    CrossDatasetBenchmark,
    run_benchmark,
)
from src.io.universal_loader import SpectralDataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dataset(n_samples: int, n_wl: int, n_classes: int,
                  wl_start: float = 400.0, seed: int = 0) -> SpectralDataset:
    """Create a synthetic labelled SpectralDataset."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(wl_start, wl_start + 500, n_wl)
    labels = np.tile(np.arange(n_classes), n_samples // n_classes + 1)[:n_samples]
    spectra = rng.normal(size=(n_samples, n_wl)).astype(np.float32)
    # Add class-specific signal so classification is learnable
    for c in range(n_classes):
        mask = labels == c
        spectra[mask, c * (n_wl // n_classes)] += 3.0
    return SpectralDataset(
        wavelengths=wl,
        spectra=spectra,
        signal_type="intensity",
        normalisation="none",
        labels=labels.astype(float),
    )


@pytest.fixture
def two_configs():
    ds_a = _make_dataset(24, 64, 3, wl_start=400.0, seed=0)
    ds_b = _make_dataset(18, 64, 3, wl_start=400.0, seed=1)
    return {"config_A": ds_a, "config_B": ds_b}


@pytest.fixture
def three_configs():
    return {
        "cfg_A": _make_dataset(24, 64, 3, wl_start=400.0, seed=0),
        "cfg_B": _make_dataset(18, 64, 3, wl_start=400.0, seed=1),
        "cfg_C": _make_dataset(20, 64, 3, wl_start=400.0, seed=2),
    }


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------

class TestBenchmarkResult:
    def test_mean_accuracy(self):
        r = BenchmarkResult(task="classification")
        r.config_results = [
            ConfigResult("A", [], "A", 10, 5, accuracy=0.8),
            ConfigResult("B", [], "B", 10, 5, accuracy=0.6),
        ]
        assert abs(r.mean_accuracy - 0.7) < 1e-9

    def test_mean_accuracy_no_results(self):
        r = BenchmarkResult(task="classification")
        assert np.isnan(r.mean_accuracy)

    def test_summary_contains_config_names(self, two_configs):
        cfg = BenchmarkConfig(task="classification", n_components_pca=8)
        bench = CrossDatasetBenchmark(cfg)
        for name, ds in two_configs.items():
            bench.add_config(name, ds)
        result = bench.run()
        s = result.summary()
        assert "config_A" in s or "config_B" in s


# ---------------------------------------------------------------------------
# CrossDatasetBenchmark
# ---------------------------------------------------------------------------

class TestCrossDatasetBenchmark:
    def test_requires_two_configs(self):
        bench = CrossDatasetBenchmark()
        bench.add_config("only_one", _make_dataset(10, 32, 2))
        with pytest.raises(ValueError, match="At least 2"):
            bench.run()

    def test_chaining(self, two_configs):
        bench = CrossDatasetBenchmark()
        result_ref = bench
        for name, ds in two_configs.items():
            result_ref = result_ref.add_config(name, ds)
        assert result_ref is bench

    def test_two_config_loo(self, two_configs):
        cfg = BenchmarkConfig(task="classification", n_components_pca=8)
        bench = CrossDatasetBenchmark(cfg)
        for name, ds in two_configs.items():
            bench.add_config(name, ds)
        result = bench.run()
        # 2 configs → 2 LOO folds
        assert len(result.config_results) == 2

    def test_three_config_loo(self, three_configs):
        cfg = BenchmarkConfig(task="classification", n_components_pca=8)
        bench = CrossDatasetBenchmark(cfg)
        for name, ds in three_configs.items():
            bench.add_config(name, ds)
        result = bench.run()
        assert len(result.config_results) == 3

    def test_accuracy_is_valid_float(self, two_configs):
        cfg = BenchmarkConfig(task="classification", n_components_pca=8)
        bench = CrossDatasetBenchmark(cfg)
        for name, ds in two_configs.items():
            bench.add_config(name, ds)
        result = bench.run()
        for r in result.config_results:
            if r.accuracy is not None and not np.isnan(r.accuracy):
                assert 0.0 <= r.accuracy <= 1.0

    def test_regression_task(self):
        rng = np.random.default_rng(5)
        wl = np.linspace(400, 900, 64)
        # Two configs with concentration labels
        def _make_reg(seed):
            sp = rng.normal(size=(20, 64)).astype(np.float32)
            labels = rng.uniform(0, 10, 20)
            return SpectralDataset(wl, sp, "intensity", "none", labels=labels)

        cfg = BenchmarkConfig(task="regression", n_components_pca=8)
        bench = CrossDatasetBenchmark(cfg)
        bench.add_config("A", _make_reg(0))
        bench.add_config("B", _make_reg(1))
        result = bench.run()
        assert len(result.config_results) == 2

    def test_different_wavelength_ranges_interpolated(self):
        """Configs with different wl ranges are interpolated to a common grid."""
        ds_a = _make_dataset(20, 100, 2, wl_start=400.0, seed=0)
        ds_b = _make_dataset(16, 80, 2, wl_start=420.0, seed=1)
        cfg = BenchmarkConfig(n_components_pca=8, interpolate_wavelengths=True)
        bench = CrossDatasetBenchmark(cfg)
        bench.add_config("A", ds_a)
        bench.add_config("B", ds_b)
        result = bench.run()   # should not raise
        assert len(result.config_results) == 2

    def test_custom_encoder(self, two_configs):
        """A custom encoder function is used instead of PCA."""
        def mock_encoder(X: np.ndarray) -> np.ndarray:
            # Simple mean-pooling as a trivial encoder
            return X.reshape(X.shape[0], -1, 8).mean(axis=-1)

        cfg = BenchmarkConfig(n_components_pca=8)
        bench = CrossDatasetBenchmark(cfg)
        bench.set_encoder(mock_encoder)
        for name, ds in two_configs.items():
            bench.add_config(name, ds)
        result = bench.run()
        assert len(result.config_results) == 2


# ---------------------------------------------------------------------------
# run_benchmark convenience function
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    def test_returns_result(self, two_configs):
        result = run_benchmark(two_configs, task="classification",
                               n_components_pca=8)
        assert isinstance(result, BenchmarkResult)
        assert len(result.config_results) == 2

    def test_task_stored_in_result(self, two_configs):
        result = run_benchmark(two_configs, task="classification",
                               n_components_pca=8)
        assert result.task == "classification"
