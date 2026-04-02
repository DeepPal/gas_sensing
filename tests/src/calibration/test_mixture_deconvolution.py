"""Tests for src.calibration.mixture_deconvolution."""
import numpy as np
import pytest
from src.calibration.mixture_deconvolution import (
    DeconvolutionResult,
    LangmuirDeconvolver,
    LinearDeconvolver,
    deconvolve_mixture,
    langmuir_predicted_shifts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_analyte_S():
    """2 analytes, 2 peaks. Analytes have orthogonal response patterns."""
    return np.array([
        [-0.50, -0.10],   # Ethanol: high on peak 0, low on peak 1
        [-0.15, -0.40],   # Acetone: low on peak 0, high on peak 1
    ])


@pytest.fixture
def kd_matrix():
    """K_d values — high values (linear regime)."""
    return np.array([
        [100.0, 100.0],
        [80.0, 80.0],
    ])


# ---------------------------------------------------------------------------
# langmuir_predicted_shifts
# ---------------------------------------------------------------------------

class TestLangmuirPredictedShifts:
    def test_linear_regime_matches_S_times_c(self, two_analyte_S):
        """At low conc relative to Kd, Langmuir ≈ linear."""
        S = two_analyte_S
        Kd = np.full((2, 2), 1000.0)  # very large Kd → linear
        c = np.array([1.0, 0.5])
        shifts = langmuir_predicted_shifts(c, S, Kd)
        expected = S.T @ c
        np.testing.assert_allclose(shifts, expected, rtol=0.01)

    def test_saturation_at_high_conc(self, two_analyte_S):
        S = two_analyte_S
        Kd = np.ones((2, 2)) * 1.0  # very low Kd → fast saturation
        c_low = np.array([0.1, 0.0])
        c_high = np.array([100.0, 0.0])
        s_low = langmuir_predicted_shifts(c_low, S, Kd)
        s_high = langmuir_predicted_shifts(c_high, S, Kd)
        # High conc should NOT be 1000× low conc (saturation)
        ratio = abs(s_high[0]) / abs(s_low[0])
        assert ratio < 20  # saturated

    def test_superposition_two_analytes(self, two_analyte_S, kd_matrix):
        S = two_analyte_S
        c_A = np.array([1.0, 0.0])
        c_B = np.array([0.0, 1.0])
        c_AB = np.array([1.0, 1.0])
        s_A = langmuir_predicted_shifts(c_A, S, kd_matrix)
        s_B = langmuir_predicted_shifts(c_B, S, kd_matrix)
        s_AB = langmuir_predicted_shifts(c_AB, S, kd_matrix)
        # In linear regime: superposition holds
        np.testing.assert_allclose(s_A + s_B, s_AB, rtol=0.05)


# ---------------------------------------------------------------------------
# LinearDeconvolver
# ---------------------------------------------------------------------------

class TestLinearDeconvolver:
    def test_single_analyte_exact(self):
        S = np.array([[-0.5]])
        solver = LinearDeconvolver(["X"], S)
        result = solver.solve(np.array([-1.0]))
        assert abs(result.concentrations["X"] - 2.0) < 0.01

    def test_two_analyte_recovers_concentrations(self, two_analyte_S):
        S = two_analyte_S
        c_true = np.array([1.0, 0.5])
        dl = S.T @ c_true
        solver = LinearDeconvolver(["Ethanol", "Acetone"], S)
        result = solver.solve(dl)
        assert abs(result.concentrations["Ethanol"] - 1.0) < 0.05
        assert abs(result.concentrations["Acetone"] - 0.5) < 0.05

    def test_returns_deconvolution_result(self, two_analyte_S):
        solver = LinearDeconvolver(["E", "A"], two_analyte_S)
        result = solver.solve(np.array([-0.5, -0.2]))
        assert isinstance(result, DeconvolutionResult)
        assert result.solver == "linear"
        assert result.success is True
        assert result.iterations is None

    def test_residual_near_zero_for_exact_solution(self, two_analyte_S):
        S = two_analyte_S
        c_true = np.array([1.0, 0.5])
        dl = S.T @ c_true
        solver = LinearDeconvolver(["E", "A"], S)
        result = solver.solve(dl)
        assert result.residual_nm < 1e-9

    def test_negative_concentration_clipped_to_zero(self, two_analyte_S):
        # A shift that implies negative concentration → clipped to 0
        solver = LinearDeconvolver(["E", "A"], two_analyte_S)
        # Shifts that imply very large negative Acetone → clip
        result = solver.solve(np.array([-5.0, 0.0]))
        assert all(v >= 0 for v in result.concentrations.values())


# ---------------------------------------------------------------------------
# LangmuirDeconvolver
# ---------------------------------------------------------------------------

class TestLangmuirDeconvolver:
    def test_recovers_linear_regime(self, two_analyte_S, kd_matrix):
        S = two_analyte_S
        c_true = np.array([0.5, 0.3])
        dl = langmuir_predicted_shifts(c_true, S, kd_matrix)

        solver = LangmuirDeconvolver(["E", "A"], S, kd_matrix)
        result = solver.solve(dl)
        assert abs(result.concentrations["E"] - 0.5) < 0.1
        assert abs(result.concentrations["A"] - 0.3) < 0.1

    def test_handles_saturation_regime(self, two_analyte_S):
        S = two_analyte_S
        Kd = np.ones((2, 2)) * 5.0  # low Kd → saturation at c=5+
        c_true = np.array([10.0, 0.0])
        dl = langmuir_predicted_shifts(c_true, S, Kd)

        solver = LangmuirDeconvolver(["E", "A"], S, Kd, conc_bounds=(0.0, 50.0))
        result = solver.solve(dl)
        # Should recover approx 10 ppm (may not be perfect due to saturation degeneracy)
        assert result.concentrations["E"] > 5.0  # at least knows it's high

    def test_returns_correct_solver_type(self, two_analyte_S, kd_matrix):
        solver = LangmuirDeconvolver(["E", "A"], two_analyte_S, kd_matrix)
        result = solver.solve(np.array([-0.5, -0.2]))
        assert result.solver == "langmuir"
        assert result.iterations is not None

    def test_batch_solve_returns_list(self, two_analyte_S, kd_matrix):
        S = two_analyte_S
        c_seq = np.array([[0.5, 0.0], [1.0, 0.0], [2.0, 0.0]])
        dl_seq = np.array([langmuir_predicted_shifts(c, S, kd_matrix) for c in c_seq])

        solver = LangmuirDeconvolver(["E", "A"], S, kd_matrix)
        results = solver.solve_batch(dl_seq)
        assert len(results) == 3
        assert all(isinstance(r, DeconvolutionResult) for r in results)

    def test_residual_near_zero_for_exact_model(self, two_analyte_S, kd_matrix):
        S = two_analyte_S
        c_true = np.array([1.0, 0.5])
        dl = langmuir_predicted_shifts(c_true, S, kd_matrix)
        solver = LangmuirDeconvolver(["E", "A"], S, kd_matrix)
        result = solver.solve(dl)
        assert result.residual_nm < 0.1  # non-linear solver residual should be small


# ---------------------------------------------------------------------------
# deconvolve_mixture
# ---------------------------------------------------------------------------

class TestDeconvolveMixture:
    def test_without_kd_uses_linear(self, two_analyte_S):
        S = two_analyte_S
        c_true = np.array([1.0, 0.5])
        dl = S.T @ c_true
        result = deconvolve_mixture(dl, ["E", "A"], S)
        assert result.solver == "linear"

    def test_with_kd_and_high_conc_uses_langmuir(self, two_analyte_S):
        S = two_analyte_S
        Kd = np.ones((2, 2)) * 1.0  # very low Kd
        c_true = np.array([5.0, 0.0])  # >> Kd → non-linear
        dl = langmuir_predicted_shifts(c_true, S, Kd)
        result = deconvolve_mixture(dl, ["E", "A"], S, Kd=Kd, use_nonlinear=True)
        assert result.solver == "langmuir"

    def test_force_linear_with_use_nonlinear_false(self, two_analyte_S, kd_matrix):
        S = two_analyte_S
        c_true = np.array([1.0, 0.5])
        dl = langmuir_predicted_shifts(c_true, S, kd_matrix)
        result = deconvolve_mixture(dl, ["E", "A"], S, Kd=kd_matrix, use_nonlinear=False)
        assert result.solver == "linear"
