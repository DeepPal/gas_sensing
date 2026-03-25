"""
tests.test_models_registry
==========================
Unit tests for src.models.registry (ModelRegistry).

CNN tests are skipped when torch is not available.
GPR tests use real GPRCalibration instances to exercise the full path.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.models.registry import ModelRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry() -> ModelRegistry:
    return ModelRegistry()


def _write_calibration_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestModelRegistryInit:
    def test_fresh_registry_has_no_models(self):
        reg = _make_registry()
        assert not reg.has_cnn()
        assert not reg.has_gpr()
        assert not reg.has_calibration()

    def test_models_loaded_returns_all_false(self):
        reg = _make_registry()
        status = reg.models_loaded()
        assert status == {"cnn": False, "gpr": False, "calibration": False}


# ---------------------------------------------------------------------------
# load_all — calibration JSON formats
# ---------------------------------------------------------------------------


class TestModelRegistryLoadAll:
    def test_load_all_empty_dir_cnn_gpr_false(self, tmp_path):
        """CNN and GPR must be False for an empty dir (no .pt or .joblib files).
        Calibration may be True if legacy output/calibration_memory_*.json exist."""
        reg = _make_registry()
        status = reg.load_all(str(tmp_path))
        assert status["cnn"] is False
        assert status["gpr"] is False
        # calibration could be True due to legacy fallback — that's fine

    def test_load_calibration_normalised_format(self, tmp_path):
        data = {"slope": 0.116, "intercept": 0.0, "reference_wavelength": 717.9, "r_squared": 0.99}
        _write_calibration_json(tmp_path / "calibration_params.json", data)
        reg = _make_registry()
        status = reg.load_all(str(tmp_path))
        assert status["calibration"] is True
        assert reg.has_calibration()

    def test_load_calibration_current_calibration_wrapper(self, tmp_path):
        data = {
            "current_calibration": {
                "slope": 0.12,
                "intercept": 0.5,
                "reference_wavelength": 717.8,
                "r_squared": 0.97,
            }
        }
        _write_calibration_json(tmp_path / "calibration_params.json", data)
        reg = _make_registry()
        reg.load_all(str(tmp_path))
        assert reg.calibration_params["slope"] == pytest.approx(0.12)
        assert reg.calibration_params["reference_wavelength"] == pytest.approx(717.8)

    def test_load_calibration_batch_pipeline_format(self, tmp_path):
        data = {
            "wavelength_shift_slope": 0.118,
            "intercept": 0.1,
            "reference_wavelength": 718.0,
            "r_squared": 0.95,
        }
        _write_calibration_json(tmp_path / "calibration_params.json", data)
        reg = _make_registry()
        reg.load_all(str(tmp_path))
        assert reg.calibration_params["slope"] == pytest.approx(0.118)

    def test_load_all_returns_status_dict_shape(self, tmp_path):
        reg = _make_registry()
        status = reg.load_all(str(tmp_path))
        assert set(status.keys()) == {"cnn", "gpr", "calibration"}
        for v in status.values():
            assert isinstance(v, bool)


# ---------------------------------------------------------------------------
# Calibration accessors
# ---------------------------------------------------------------------------


class TestModelRegistryCalibrationAccessors:
    def _reg_with_cal(self, tmp_path, slope=0.116, ref_wl=717.9):
        data = {"slope": slope, "intercept": 0.0, "reference_wavelength": ref_wl, "r_squared": 0.99}
        _write_calibration_json(tmp_path / "calibration_params.json", data)
        reg = _make_registry()
        reg.load_all(str(tmp_path))
        return reg

    def test_get_calibration_slope(self, tmp_path):
        reg = self._reg_with_cal(tmp_path, slope=0.123)
        assert reg.get_calibration_slope() == pytest.approx(0.123)

    def test_get_reference_wavelength(self, tmp_path):
        reg = self._reg_with_cal(tmp_path, ref_wl=718.5)
        assert reg.get_reference_wavelength() == pytest.approx(718.5)

    def test_default_slope_when_no_calibration(self):
        reg = _make_registry()
        assert reg.get_calibration_slope() == pytest.approx(0.116)

    def test_default_ref_wl_when_no_calibration(self):
        reg = _make_registry()
        assert reg.get_reference_wavelength() == pytest.approx(531.5)


# ---------------------------------------------------------------------------
# Predictions without models
# ---------------------------------------------------------------------------


class TestModelRegistryPredictNoModels:
    def test_predict_gas_type_no_cnn_returns_unknown(self):
        reg = _make_registry()
        gas, conf = reg.predict_gas_type(np.ones(1000))
        assert gas == "unknown"
        assert conf == pytest.approx(0.0)

    def test_predict_concentration_gpr_no_gpr_returns_none(self):
        reg = _make_registry()
        conc, unc = reg.predict_concentration_gpr(-1.0)
        assert conc is None
        assert unc is None


# ---------------------------------------------------------------------------
# GPR integration (does not require torch)
# ---------------------------------------------------------------------------


class TestModelRegistryWithGPR:
    def _fitted_gpr(self):
        from src.calibration.gpr import GPRCalibration

        gpr = GPRCalibration(n_restarts_optimizer=1)
        shifts = np.linspace(-0.5, -4.0, 8).reshape(-1, 1)
        concs = np.linspace(0.5, 4.0, 8)
        gpr.fit(shifts, concs)
        return gpr

    def test_load_gpr_via_load_all(self, tmp_path):
        gpr = self._fitted_gpr()
        gpr.save(str(tmp_path / "gpr_calibration.joblib"))
        reg = _make_registry()
        status = reg.load_all(str(tmp_path))
        assert status["gpr"] is True
        assert reg.has_gpr()

    def test_predict_concentration_gpr_returns_float(self, tmp_path):
        gpr = self._fitted_gpr()
        gpr.save(str(tmp_path / "gpr_calibration.joblib"))
        reg = _make_registry()
        reg.load_all(str(tmp_path))
        conc, unc = reg.predict_concentration_gpr(-1.0)
        assert isinstance(conc, float)
        assert isinstance(unc, float)
        assert conc > 0
        assert unc >= 0


# ---------------------------------------------------------------------------
# CNN integration (skip if torch absent)
# ---------------------------------------------------------------------------


torch_skip = pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)


@torch_skip
class TestModelRegistryWithCNN:
    def _fitted_cnn(self, tmp_path):
        from src.models.cnn import CNNGasClassifier

        clf = CNNGasClassifier(input_length=100, num_classes=2)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 100)).astype(np.float32)
        y_label = np.array([0] * 10 + [1] * 10)
        y_conc = np.ones(20)
        clf.fit(X, y_label, y_conc, class_names=["A", "B"], epochs=2, batch_size=4)
        path = str(tmp_path / "cnn_classifier.pt")
        clf.save(path)
        return path

    def test_load_cnn_via_load_all(self, tmp_path):
        self._fitted_cnn(tmp_path)
        reg = _make_registry()
        status = reg.load_all(str(tmp_path))
        assert status["cnn"] is True
        assert reg.has_cnn()

    def test_predict_gas_type_returns_string(self, tmp_path):
        self._fitted_cnn(tmp_path)
        reg = _make_registry()
        reg.load_all(str(tmp_path))
        gas, conf = reg.predict_gas_type(np.ones(100))
        assert isinstance(gas, str)
        assert 0.0 <= conf <= 1.0
