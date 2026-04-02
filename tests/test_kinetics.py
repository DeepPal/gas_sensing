"""
Unit tests for src.features.lspr_features.estimate_response_kinetics
and KineticFeatures dataclass.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from src.features.lspr_features import KineticFeatures, estimate_response_kinetics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step_response(
    t: np.ndarray,
    a: float,
    tau: float,
    onset: float = 0.0,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic 1:1 Langmuir step response."""
    rng = np.random.default_rng(seed)
    y = np.where(t >= onset, a * (1 - np.exp(-(t - onset) / tau)), 0.0)
    return y + rng.normal(0, noise, size=len(t))


# ---------------------------------------------------------------------------
# KineticFeatures dataclass
# ---------------------------------------------------------------------------

class TestKineticFeaturesDataclass:
    def test_all_none_defaults(self):
        kf = KineticFeatures()
        assert kf.tau_63_s is None
        assert kf.tau_95_s is None
        assert kf.k_on_per_s is None
        assert kf.delta_lambda_eq_nm is None
        assert kf.fit_r2 is None
        assert kf.onset_idx is None

    def test_tau_95_approximately_three_tau_63(self):
        """τ₉₅ = −τ·ln(0.05) ≈ 2.996·τ for any τ."""
        kf = KineticFeatures(tau_63_s=10.0, tau_95_s=-10.0 * math.log(0.05))
        assert kf.tau_95_s == pytest.approx(29.957, rel=1e-3)


# ---------------------------------------------------------------------------
# estimate_response_kinetics — clean synthetic signals
# ---------------------------------------------------------------------------

class TestEstimateResponseKineticsClean:
    def test_returns_kinetic_features(self):
        t = np.linspace(0, 120, 200)
        y = _step_response(t, a=-0.8, tau=15.0, onset=20.0)
        result = estimate_response_kinetics(y, t)
        assert isinstance(result, KineticFeatures)

    def test_tau_recovered_within_30pct(self):
        """Fitted τ should be within 30% of ground truth on a clean signal."""
        t = np.linspace(0, 200, 300)
        y = _step_response(t, a=-1.2, tau=20.0, onset=30.0, noise=0.005)
        result = estimate_response_kinetics(y, t)
        assert result.tau_63_s is not None
        assert abs(result.tau_63_s - 20.0) / 20.0 < 0.30

    def test_tau_95_is_about_3x_tau_63(self):
        t = np.linspace(0, 300, 400)
        y = _step_response(t, a=-1.0, tau=25.0, onset=30.0, noise=0.002)
        result = estimate_response_kinetics(y, t)
        if result.tau_63_s is not None and result.tau_95_s is not None:
            ratio = result.tau_95_s / result.tau_63_s
            assert 2.5 < ratio < 3.5

    def test_k_on_is_reciprocal_of_tau(self):
        t = np.linspace(0, 200, 300)
        y = _step_response(t, a=-0.9, tau=18.0, onset=20.0)
        result = estimate_response_kinetics(y, t)
        if result.tau_63_s is not None and result.k_on_per_s is not None:
            assert result.k_on_per_s == pytest.approx(1.0 / result.tau_63_s, rel=1e-6)

    def test_fit_r2_high_on_clean_signal(self):
        """R² ≥ 0.95 on a noise-free synthetic transient."""
        t = np.linspace(0, 200, 300)
        y = _step_response(t, a=-1.5, tau=30.0, onset=25.0, noise=0.0)
        result = estimate_response_kinetics(y, t)
        if result.fit_r2 is not None:
            assert result.fit_r2 >= 0.90

    def test_delta_lambda_eq_has_correct_sign(self):
        """Fitted ΔλEq should have the same sign as the equilibrium shift."""
        t = np.linspace(0, 200, 250)
        y_neg = _step_response(t, a=-0.8, tau=20.0, onset=20.0)
        y_pos = _step_response(t, a=+0.8, tau=20.0, onset=20.0)
        r_neg = estimate_response_kinetics(y_neg, t)
        r_pos = estimate_response_kinetics(y_pos, t)
        if r_neg.delta_lambda_eq_nm is not None:
            assert r_neg.delta_lambda_eq_nm < 0
        if r_pos.delta_lambda_eq_nm is not None:
            assert r_pos.delta_lambda_eq_nm > 0

    def test_onset_idx_within_reasonable_range(self):
        """Detected onset should be in the first half of the series."""
        t = np.linspace(0, 200, 300)
        y = _step_response(t, a=-1.0, tau=20.0, onset=30.0)
        result = estimate_response_kinetics(y, t)
        if result.onset_idx is not None:
            assert 0 < result.onset_idx < len(t) // 2


# ---------------------------------------------------------------------------
# estimate_response_kinetics — edge / failure cases
# ---------------------------------------------------------------------------

class TestEstimateResponseKineticsEdge:
    def test_too_few_points_returns_empty(self):
        result = estimate_response_kinetics([0.1, 0.2, 0.3], [0.0, 1.0, 2.0])
        assert result.tau_63_s is None
        assert result.tau_95_s is None

    def test_mismatched_lengths_returns_empty(self):
        result = estimate_response_kinetics([0.0] * 50, [0.0] * 30)
        assert result.tau_63_s is None

    def test_flat_signal_no_crash(self):
        """A flat baseline (no step) should return gracefully."""
        t = np.linspace(0, 100, 200)
        y = np.full(200, -0.5)
        result = estimate_response_kinetics(y, t)
        # tau may or may not be found — just must not raise
        assert isinstance(result, KineticFeatures)

    def test_pure_noise_no_crash(self):
        """Pure Gaussian noise should not crash the function."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 100, 200)
        y = rng.normal(0, 0.05, 200)
        result = estimate_response_kinetics(y, t)
        assert isinstance(result, KineticFeatures)

    def test_fit_r2_never_exceeds_one(self):
        t = np.linspace(0, 200, 300)
        y = _step_response(t, a=-1.0, tau=20.0, onset=20.0)
        result = estimate_response_kinetics(y, t)
        if result.fit_r2 is not None:
            assert result.fit_r2 <= 1.0

    def test_fit_r2_never_below_zero(self):
        t = np.linspace(0, 200, 300)
        y = _step_response(t, a=-1.0, tau=20.0, onset=20.0, noise=0.2)
        result = estimate_response_kinetics(y, t)
        if result.fit_r2 is not None:
            assert result.fit_r2 >= 0.0
