"""Tests for src.calibration.selectivity — cross-reactivity analysis."""
import numpy as np
import pytest

from src.calibration.selectivity import SelectivityAnalyzer, SelectivityReport


def _ethanol_data(n: int = 6) -> tuple[np.ndarray, np.ndarray]:
    concs = np.linspace(0.5, 3.0, n)
    shifts = -10.0 * concs / (1.0 + concs) + np.random.default_rng(0).normal(0, 0.02, n)
    return concs, shifts


def _interferent_data(
    slope_fraction: float, n: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Interferent with sensitivity = slope_fraction * analyte sensitivity."""
    concs = np.linspace(1.0, 5.0, n)
    # Use a simple linear response scaled by slope_fraction
    shifts = slope_fraction * (-3.5) * concs
    return concs, shifts


def test_returns_selectivity_report():
    a_concs, a_shifts = _ethanol_data()
    analyzer = SelectivityAnalyzer(analyte="Ethanol")
    report = analyzer.analyze(a_concs, a_shifts, {})
    assert isinstance(report, SelectivityReport)
    assert report.analyte == "Ethanol"


def test_analyte_sensitivity_is_negative_for_lspr():
    """LSPR adsorption produces negative wavelength shifts → negative sensitivity."""
    a_concs, a_shifts = _ethanol_data()
    analyzer = SelectivityAnalyzer("Ethanol")
    report = analyzer.analyze(a_concs, a_shifts, {})
    assert report.analyte_sensitivity_nm_per_ppm < 0


def test_cross_reactivity_coefficient_correct():
    """K_AB = m_B / m_A must be computed correctly."""
    a_concs, a_shifts = _ethanol_data()
    # Interferent with exactly 10% of the analyte sensitivity
    b_concs, b_shifts = _interferent_data(slope_fraction=0.1)
    analyzer = SelectivityAnalyzer("Ethanol")
    report = analyzer.analyze(
        a_concs, a_shifts, {"CO2": (b_concs, b_shifts)}
    )
    k = report.cross_reactivity_coefficients["CO2"]
    assert np.isfinite(k)
    # 10% cross-reactivity → |K_AB| ≈ 0.10
    assert abs(abs(k) - 0.10) < 0.05


def test_selectivity_flag_excellent():
    """K_AB < 0.01 must be flagged as 'excellent'."""
    a_concs, a_shifts = _ethanol_data()
    b_concs, b_shifts = _interferent_data(slope_fraction=0.005)
    analyzer = SelectivityAnalyzer("Ethanol")
    report = analyzer.analyze(a_concs, a_shifts, {"trace": (b_concs, b_shifts)})
    assert report.selectivity_flags.get("trace") == "excellent"


def test_selectivity_flag_poor():
    """K_AB > 0.20 must be flagged as 'poor'."""
    a_concs, a_shifts = _ethanol_data()
    b_concs, b_shifts = _interferent_data(slope_fraction=0.5)
    analyzer = SelectivityAnalyzer("Ethanol")
    report = analyzer.analyze(a_concs, a_shifts, {"water": (b_concs, b_shifts)})
    assert report.selectivity_flags.get("water") == "poor"


def test_ppm_equivalent_is_reciprocal_of_k():
    """ppm_equivalent must equal 1/|K_AB|."""
    a_concs, a_shifts = _ethanol_data()
    b_concs, b_shifts = _interferent_data(slope_fraction=0.2)
    analyzer = SelectivityAnalyzer("Ethanol")
    report = analyzer.analyze(a_concs, a_shifts, {"gas": (b_concs, b_shifts)})
    k = abs(report.cross_reactivity_coefficients["gas"])
    ppm_eq = report.cross_reactivity_ppm_equivalent["gas"]
    if np.isfinite(k) and k > 1e-9:
        assert abs(ppm_eq - 1.0 / k) < 0.01


def test_insufficient_analyte_points_warns():
    """Only 1 analyte point must emit UserWarning."""
    analyzer = SelectivityAnalyzer("Ethanol")
    with pytest.warns(UserWarning, match="at least 2"):
        report = analyzer.analyze(
            np.array([1.0]), np.array([-5.0]), {}
        )
    assert report.analyte_sensitivity_nm_per_ppm != report.analyte_sensitivity_nm_per_ppm  # nan


def test_r2_values_populated():
    """R² must be populated for analyte and each interferent."""
    a_concs, a_shifts = _ethanol_data()
    b_concs, b_shifts = _interferent_data(0.1)
    analyzer = SelectivityAnalyzer("Ethanol")
    report = analyzer.analyze(a_concs, a_shifts, {"CO2": (b_concs, b_shifts)})
    assert report.r2_analyte is not None
    assert 0.0 <= report.r2_analyte <= 1.0
    assert "CO2" in report.r2_interferents


def test_summary_contains_analyte_name():
    a_concs, a_shifts = _ethanol_data()
    analyzer = SelectivityAnalyzer("Ethanol")
    report = analyzer.analyze(a_concs, a_shifts, {})
    assert "Ethanol" in report.summary
