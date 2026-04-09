"""Tests for src.calibration.sensitivity_matrix."""
import numpy as np
import pytest

from src.calibration.sensitivity_matrix import SensitivityEntry, SensitivityMatrix

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_analyte_sm():
    """1 analyte, 1 peak — simplest case."""
    sm = SensitivityMatrix(["Ethanol"], 1)
    sm.fit_analyte(
        "Ethanol", 0,
        conc_ppm=[0.1, 0.5, 1.0, 2.0, 5.0],
        shifts_nm=[-0.05, -0.25, -0.50, -1.00, -2.50],
    )
    return sm


@pytest.fixture
def two_analyte_two_peak_sm():
    """2 analytes × 2 peaks — full matrix."""
    sm = SensitivityMatrix(["Ethanol", "Acetone"], 2)
    # Ethanol: peak 0 has higher sensitivity; peak 1 lower
    sm.fit_analyte("Ethanol", 0, [0.5, 1.0, 2.0, 5.0], [-0.25, -0.50, -1.00, -2.50])
    sm.fit_analyte("Ethanol", 1, [0.5, 1.0, 2.0, 5.0], [-0.10, -0.20, -0.40, -1.00])
    # Acetone: peak 0 lower; peak 1 higher (orthogonal pattern)
    sm.fit_analyte("Acetone", 0, [0.5, 1.0, 2.0, 5.0], [-0.15, -0.30, -0.60, -1.50])
    sm.fit_analyte("Acetone", 1, [0.5, 1.0, 2.0, 5.0], [-0.20, -0.40, -0.80, -2.00])
    return sm


# ---------------------------------------------------------------------------
# Fit tests
# ---------------------------------------------------------------------------

class TestFitAnalyte:
    def test_fit_returns_sensitivity_entry(self, single_analyte_sm):
        sm = SensitivityMatrix(["X"], 1)
        entry = sm.fit_analyte("X", 0, [1.0, 2.0, 5.0], [-0.5, -1.0, -2.5])
        assert isinstance(entry, SensitivityEntry)

    def test_slope_is_correct(self, single_analyte_sm):
        # Slope should be -0.5 nm/ppm (perfect linear data)
        entry = single_analyte_sm._entries[("Ethanol", 0)]
        assert abs(entry.slope_nm_per_ppm - (-0.5)) < 0.01

    def test_r_squared_near_unity_for_perfect_data(self):
        sm = SensitivityMatrix(["X"], 1)
        entry = sm.fit_analyte("X", 0, [0.5, 1.0, 2.0, 5.0], [-0.25, -0.50, -1.00, -2.50])
        assert entry.r_squared > 0.999

    def test_wrong_analyte_raises(self):
        sm = SensitivityMatrix(["X"], 1)
        with pytest.raises(ValueError):
            sm.fit_analyte("Y", 0, [1.0], [-0.5])  # "Y" not in analytes

    def test_wrong_peak_idx_raises(self):
        sm = SensitivityMatrix(["X"], 1)
        with pytest.raises(IndexError):
            sm.fit_analyte("X", 5, [1.0], [-0.5])  # peak 5 doesn't exist

    def test_too_few_points_raises(self):
        sm = SensitivityMatrix(["X"], 1)
        with pytest.raises(ValueError):
            sm.fit_analyte("X", 0, [1.0], [-0.5])  # only 1 point

    def test_fit_from_dataframe(self):
        import pandas as pd
        sm = SensitivityMatrix(["A"], 1)
        df = pd.DataFrame({
            "analyte": ["A", "A", "A"],
            "concentration_ppm": [0.5, 1.0, 2.0],
            "peak_shift_0": [-0.25, -0.50, -1.00],
        })
        sm.fit_from_dataframe(df)
        assert sm._is_fully_fitted()


# ---------------------------------------------------------------------------
# Concentration estimation
# ---------------------------------------------------------------------------

class TestEstimateConcentrations:
    def test_single_analyte_correct(self, single_analyte_sm):
        # At Δλ = -1.0 nm, slope = -0.5 → c = 2.0 ppm
        concs, _ = single_analyte_sm.estimate_concentrations(np.array([-1.0]))
        assert abs(concs["Ethanol"] - 2.0) < 0.1

    def test_two_analyte_recovers_mixture(self, two_analyte_two_peak_sm):
        sm = two_analyte_two_peak_sm
        # True: Ethanol=1.0, Acetone=0.5 ppm
        # Predicted Δλ = S.T @ c
        c_true = np.array([1.0, 0.5])
        dl_pred = sm.matrix.T @ c_true
        concs, _ = sm.estimate_concentrations(dl_pred)
        assert abs(concs["Ethanol"] - 1.0) < 0.15
        assert abs(concs["Acetone"] - 0.5) < 0.15

    def test_raises_when_not_fully_fitted(self):
        sm = SensitivityMatrix(["X", "Y"], 1)
        sm.fit_analyte("X", 0, [1.0, 2.0], [-0.5, -1.0])
        # Y not fitted
        with pytest.raises(RuntimeError):
            sm.estimate_concentrations(np.array([-0.5]))

    def test_wrong_shape_raises(self, single_analyte_sm):
        with pytest.raises(ValueError):
            single_analyte_sm.estimate_concentrations(np.array([-0.5, -0.3]))  # 2 peaks, expect 1


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

class TestQualityMetrics:
    def test_condition_number_is_finite(self, two_analyte_two_peak_sm):
        assert np.isfinite(two_analyte_two_peak_sm.condition_number)

    def test_rank_two_for_two_analytes(self, two_analyte_two_peak_sm):
        assert two_analyte_two_peak_sm.rank == 2

    def test_lod_mixture_returns_dict(self, single_analyte_sm):
        lod = single_analyte_sm.compute_lod_mixture(noise_nm=0.05)
        assert "Ethanol" in lod
        assert lod["Ethanol"] > 0

    def test_summary_keys_present(self, two_analyte_two_peak_sm):
        s = two_analyte_two_peak_sm.summary()
        for key in ("analytes", "n_peaks", "S_matrix", "condition_number", "rank"):
            assert key in s


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestSensitivityMatrixPersistence:
    def test_save_load_roundtrip(self, single_analyte_sm, tmp_path):
        path = str(tmp_path / "sm.joblib")
        single_analyte_sm.save(path)
        loaded = SensitivityMatrix.load(path)
        assert np.allclose(loaded.matrix, single_analyte_sm.matrix)
        concs_orig, _ = single_analyte_sm.estimate_concentrations(np.array([-1.0]))
        concs_load, _ = loaded.estimate_concentrations(np.array([-1.0]))
        assert abs(concs_orig["Ethanol"] - concs_load["Ethanol"]) < 1e-9
