"""
tests.test_training_scripts
============================
Smoke tests for the training CLI scripts:
  - src.training.train_gpr  (GPR concentration calibration)
  - src.training.train_cnn  (CNN gas classifier)
  - src.training.ablation   (preprocessing ablation study)
  - src.training.cross_gas_eval (LOGO cross-validation)

These are smoke tests — they verify the entry points are importable, argument
parsers work, and the core logic runs on tiny synthetic datasets without error.
They do NOT require real gas-sensor data or a fitted model on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_csv_dir(tmp_path: Path, gas: str = "Ethanol", n_conc: int = 3, n_per: int = 5) -> Path:
    """Create a minimal Joy-data directory layout for one gas."""
    import csv

    gas_dir = tmp_path / gas
    for c_idx in range(1, n_conc + 1):
        conc_ppm = c_idx * 1.0
        sub = gas_dir / f"{conc_ppm} ppm {gas}-1"
        sub.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(c_idx)
        for i in range(n_per):
            fname = sub / f"spectrum_{i:03d}.csv"
            # Two-column: wavelength, intensity
            wl = np.linspace(500, 800, 50)
            intensity = 1000 + conc_ppm * 10 + rng.normal(0, 5, 50)
            with open(fname, "w", newline="") as f:
                writer = csv.writer(f)
                for w, v in zip(wl, intensity):
                    writer.writerow([w, v])
    return gas_dir


# ---------------------------------------------------------------------------
# src.training.train_gpr
# ---------------------------------------------------------------------------


class TestTrainGPR:
    def test_module_importable(self):
        import src.training.train_gpr  # noqa: F401

    def test_main_help_exits_cleanly(self):
        from src.training.train_gpr import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# src.training.train_cnn
# ---------------------------------------------------------------------------


class TestTrainCNN:
    def test_module_importable(self):
        import src.training.train_cnn  # noqa: F401

    def test_main_help_exits_cleanly(self):
        from src.training.train_cnn import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# src.training.ablation — AblationConfig dataclass
# ---------------------------------------------------------------------------


class TestAblationConfig:
    def test_default_config_all_on(self):
        from src.training.ablation import AblationConfig

        cfg = AblationConfig(name="all_on")
        assert cfg.use_baseline
        assert cfg.use_smoothing
        assert cfg.use_normalization

    def test_describe_all_on(self):
        from src.training.ablation import AblationConfig

        cfg = AblationConfig(name="all_on")
        assert cfg.describe() == "all steps active"

    def test_describe_no_baseline(self):
        from src.training.ablation import AblationConfig

        cfg = AblationConfig(name="no_baseline", use_baseline=False)
        assert "no_baseline" in cfg.describe()

    def test_describe_multiple_disabled(self):
        from src.training.ablation import AblationConfig

        cfg = AblationConfig(
            name="raw", use_baseline=False, use_smoothing=False, use_normalization=False
        )
        desc = cfg.describe()
        assert "no_baseline" in desc
        assert "no_smoothing" in desc
        assert "no_normalization" in desc

    def test_ablation_configs_list_has_six_entries(self):
        from src.training.ablation import ABLATION_CONFIGS

        assert len(ABLATION_CONFIGS) == 6

    def test_ablation_configs_first_is_all_on(self):
        from src.training.ablation import ABLATION_CONFIGS

        assert ABLATION_CONFIGS[0].name == "all_on"


class TestAblationPreprocess:
    def test_preprocess_returns_array(self):
        from src.training.ablation import AblationConfig, preprocess_spectrum

        cfg = AblationConfig(name="all_on")
        arr = np.ones(100, dtype=float)
        wl = np.linspace(500, 800, 100)
        result = preprocess_spectrum(arr, wl, cfg)
        assert result.shape == (100,)

    def test_preprocess_raw_only_returns_unchanged(self):
        from src.training.ablation import AblationConfig, preprocess_spectrum

        cfg = AblationConfig(
            name="raw_only", use_baseline=False, use_smoothing=False, use_normalization=False
        )
        arr = np.ones(50, dtype=float) * 3.14
        result = preprocess_spectrum(arr, None, cfg)
        np.testing.assert_array_almost_equal(result, arr)

    def test_preprocess_finite_output(self):
        from src.training.ablation import AblationConfig, preprocess_spectrum

        for cfg_name in ["all_on", "no_baseline", "no_smoothing", "no_normalization"]:
            cfg = AblationConfig(
                name=cfg_name,
                use_baseline=cfg_name != "no_baseline",
                use_smoothing=cfg_name != "no_smoothing",
                use_normalization=cfg_name != "no_normalization",
            )
            arr = np.random.default_rng(0).standard_normal(80) + 500
            wl = np.linspace(400, 900, 80)
            result = preprocess_spectrum(arr, wl, cfg)
            assert np.all(np.isfinite(result)), f"Non-finite output for config {cfg_name}"


class TestAblationCLI:
    def test_main_help_exits(self):
        from src.training.ablation import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_run_ablation_on_synthetic_data(self, tmp_path):
        """Full ablation run on tiny synthetic dataset."""
        from src.training.ablation import ABLATION_CONFIGS, run_ablation

        gas_dir = _make_csv_dir(tmp_path, gas="Ethanol", n_conc=3, n_per=4)
        # Only run 2 configs to keep test fast
        results = run_ablation(gas_dir, configs=ABLATION_CONFIGS[:2])
        assert "configs" in results
        assert "all_on" in results["configs"]
        assert results["n_spectra"] > 0

    def test_ablation_results_have_r2(self, tmp_path):
        from src.training.ablation import ABLATION_CONFIGS, run_ablation

        gas_dir = _make_csv_dir(tmp_path, gas="Ethanol", n_conc=3, n_per=4)
        results = run_ablation(gas_dir, configs=ABLATION_CONFIGS[:1])
        cfg_results = results["configs"]["all_on"]
        assert "r2" in cfg_results
        assert "rmse_ppm" in cfg_results

    def test_ablation_baseline_delta_is_zero(self, tmp_path):
        from src.training.ablation import ABLATION_CONFIGS, run_ablation

        gas_dir = _make_csv_dir(tmp_path, gas="Ethanol", n_conc=3, n_per=4)
        results = run_ablation(gas_dir, configs=ABLATION_CONFIGS[:1])
        assert results["configs"]["all_on"]["delta_r2"] == 0.0
        assert results["configs"]["all_on"]["delta_rmse_ppm"] == 0.0


# ---------------------------------------------------------------------------
# src.training.cross_gas_eval — data loading helpers
# ---------------------------------------------------------------------------


class TestCrossGasEvalHelpers:
    def test_parse_concentration_from_path(self, tmp_path):
        from src.training.cross_gas_eval import _parse_concentration

        p = tmp_path / "1 ppm Ethanol-1" / "spec.csv"
        assert _parse_concentration(p) == pytest.approx(1.0)

    def test_parse_concentration_0_5_ppm(self, tmp_path):
        from src.training.cross_gas_eval import _parse_concentration

        p = tmp_path / "0.5 ppm IPA-2" / "spec.csv"
        assert _parse_concentration(p) == pytest.approx(0.5)

    def test_parse_concentration_no_match_returns_zero(self, tmp_path):
        from src.training.cross_gas_eval import _parse_concentration

        p = tmp_path / "no_concentration_here" / "spec.csv"
        assert _parse_concentration(p) == pytest.approx(0.0)

    def test_load_spectra_from_dir(self, tmp_path):
        from src.training.cross_gas_eval import _load_spectra_from_dir

        # Create two gas dirs
        _make_csv_dir(tmp_path, gas="Ethanol", n_conc=2, n_per=3)
        _make_csv_dir(tmp_path, gas="IPA", n_conc=2, n_per=3)
        X, y_label, y_conc, class_names = _load_spectra_from_dir(tmp_path)
        assert X.ndim == 2
        assert len(y_label) == len(y_conc) == len(X)
        assert set(class_names) == {"Ethanol", "IPA"}

    def test_load_spectra_raises_on_empty_dir(self, tmp_path):
        from src.training.cross_gas_eval import _load_spectra_from_dir

        empty = tmp_path / "empty_gas"
        empty.mkdir()
        with pytest.raises(ValueError, match="No spectra"):
            _load_spectra_from_dir(tmp_path)


class TestCrossGasEvalCLI:
    def test_main_help_exits(self):
        from src.training.cross_gas_eval import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
