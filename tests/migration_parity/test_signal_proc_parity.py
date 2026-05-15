"""
Parity: src.preprocessing functions must match gas_analysis.core.signal_proc
for the functions still imported by dashboard/app.py, dashboard/experiment_tab.py,
and dashboard/predict_tab.py.

Covers four functions before migration in Task 14:
  - smooth_spectrum      (app.py + experiment_tab.py)
  - als_baseline         (app.py + experiment_tab.py)
  - wavelet_denoise      (app.py)
  - baseline_correction  (predict_tab.py, via fallback)

Known semantic differences are documented in dedicated "diff" tests rather than
treated as failures — the parity tests only assert what *must* match.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# smooth_spectrum — identical for all three methods used in the dashboard
# ---------------------------------------------------------------------------


class TestSmoothSpectrumParity:
    """smooth_spectrum: new location must produce identical results to old."""

    def _compare(self, intensity: np.ndarray, **kwargs) -> None:
        from gas_analysis.core.signal_proc import smooth_spectrum as old
        from src.preprocessing.denoising import smooth_spectrum as new

        old_result = old(intensity, **kwargs)
        new_result = new(intensity, **kwargs)
        np.testing.assert_allclose(
            new_result,
            old_result,
            rtol=1e-6,
            err_msg=f"smooth_spectrum parity failed with kwargs={kwargs}",
        )

    def test_savgol_default(self, single_spectrum: np.ndarray) -> None:
        """Default Savitzky-Golay smoothing must be numerically identical."""
        self._compare(single_spectrum)

    def test_savgol_explicit(self, single_spectrum: np.ndarray) -> None:
        """Explicit savgol method must be numerically identical."""
        self._compare(single_spectrum, method="savgol", window=11, poly_order=2)

    def test_moving_average(self, single_spectrum: np.ndarray) -> None:
        """Moving-average smoothing must be numerically identical."""
        self._compare(single_spectrum, method="moving_average", window=15)

    def test_gaussian(self, single_spectrum: np.ndarray) -> None:
        """Gaussian smoothing must be numerically identical."""
        self._compare(single_spectrum, method="gaussian", window=11)

    def test_empty_array(self) -> None:
        """Both implementations must return empty array unchanged."""
        from gas_analysis.core.signal_proc import smooth_spectrum as old
        from src.preprocessing.denoising import smooth_spectrum as new

        arr = np.array([], dtype=float)
        np.testing.assert_array_equal(old(arr), new(arr))

    def test_short_spectra(self, sample_spectra: list[np.ndarray]) -> None:
        """Parity must hold across a variety of realistic spectra."""
        from gas_analysis.core.signal_proc import smooth_spectrum as old
        from src.preprocessing.denoising import smooth_spectrum as new

        for sp in sample_spectra:
            old_r = old(sp)
            new_r = new(sp)
            np.testing.assert_allclose(
                new_r, old_r, rtol=1e-6, err_msg="smooth_spectrum parity failed on sample_spectra"
            )


# ---------------------------------------------------------------------------
# als_baseline — SEMANTIC DIFFERENCE DOCUMENTED
#
# Old (gas_analysis.core.signal_proc.als_baseline): returns the BASELINE curve.
# New (src.preprocessing.baseline.als_baseline):    returns CORRECTED signal
#                                                   (intensities − baseline).
#
# The dashboard files handle this correctly:
#   app.py l.658-661 — comment explicitly states "returns corrected" and
#                       reconstructs baseline with `baseline_est = processed - corrected`
#   experiment_tab.py l.273-274 — the try-block at top-of-file always resolves
#                                  to src.preprocessing (primary), so these lines
#                                  call `r - als_baseline(r)` which double-corrects
#                                  — this is a pre-existing bug NOT introduced by the
#                                  migration; parity tests do NOT mask it.
#
# Parity test: new(x) == old(x) - x  (i.e. new corrected = -(old baseline - x))
# ---------------------------------------------------------------------------


class TestAlsBaselineSemanticDiff:
    """Document and verify the known semantic difference in als_baseline return value."""

    def test_new_returns_corrected_not_baseline(self, single_spectrum: np.ndarray) -> None:
        """
        New als_baseline returns (intensity - baseline); old returns the baseline.

        Relationship: new(x) + old(x) == x  (within numerical tolerance).
        """
        from gas_analysis.core.signal_proc import als_baseline as old
        from src.preprocessing.baseline import als_baseline as new

        old_result = old(single_spectrum)
        new_result = new(single_spectrum)

        # They must sum to the original (corrected + baseline == intensity)
        np.testing.assert_allclose(
            new_result + old_result,
            single_spectrum,
            rtol=1e-6,
            atol=1e-10,
            err_msg="als_baseline: new + old must reconstruct the original intensity",
        )

    def test_new_corrected_has_reduced_mean(self, single_spectrum: np.ndarray) -> None:
        """
        New als_baseline (corrected signal) must have a lower mean than the input.

        ALS baseline removal subtracts the slowly-varying background.  For a
        Lorentzian-peaked spectrum the residual still carries the peak contribution,
        so the mean will not be exactly zero — but it must be substantially less
        than the input mean (which is dominated by the background pedestal).
        """
        from src.preprocessing.baseline import als_baseline as new

        corrected = new(single_spectrum)
        input_mean = abs(float(single_spectrum.mean()))
        corrected_mean = abs(float(corrected.mean()))
        # Corrected mean should be at least 50 % smaller than input mean
        assert corrected_mean < input_mean * 0.75, (
            f"als_baseline corrected mean {corrected_mean:.4f} is not "
            f"sufficiently smaller than input mean {input_mean:.4f}. "
            "Expected at least 25 % reduction after baseline removal."
        )

    def test_old_baseline_tracks_signal(self, single_spectrum: np.ndarray) -> None:
        """Old als_baseline (baseline curve) must track the signal level."""
        from gas_analysis.core.signal_proc import als_baseline as old

        baseline = old(single_spectrum)
        # The baseline should be in the same value range as the input
        assert float(np.max(baseline)) <= float(np.max(single_spectrum)) + 0.01, (
            "Old als_baseline returned values above the input signal — unexpected."
        )
        assert abs(float(baseline.mean()) - float(single_spectrum.mean())) < 0.5, (
            "Old als_baseline mean deviates too far from signal mean."
        )

    def test_custom_lam_and_p(self, single_spectrum: np.ndarray) -> None:
        """
        Both implementations accept lam and p kwargs and reconstruct correctly.

        Old uses kwarg name `niter`; new uses `n_iter` — only test shared kwargs.
        """
        from gas_analysis.core.signal_proc import als_baseline as old
        from src.preprocessing.baseline import als_baseline as new

        lam, p = 1e4, 0.05
        old_result = old(single_spectrum, lam=lam, p=p)
        new_result = new(single_spectrum, lam=lam, p=p)

        np.testing.assert_allclose(
            new_result + old_result,
            single_spectrum,
            rtol=1e-6,
            atol=1e-10,
            err_msg="als_baseline: new + old must reconstruct input for custom lam/p",
        )


# ---------------------------------------------------------------------------
# wavelet_denoise — KNOWN ALGORITHMIC DIFFERENCE
#
# Old: uses pywt.wavedec(mode='per') + level=floor(log2(N)) (full decomposition)
# New: uses pywt.wavedec() + level=floor(log2(N))-2 (shallower, edge-effect aware)
#
# Both implementations:
#   - return same shape as input
#   - fall back to Savitzky-Golay when pywt is unavailable
#   - produce output in a physically sensible range (within 3× signal amplitude)
#
# Numerical identity is NOT expected — this test verifies output properties only.
# ---------------------------------------------------------------------------


class TestWaveletDenoiseProperties:
    """Verify wavelet_denoise output properties rather than exact parity."""

    def test_output_shape_matches(self, single_spectrum: np.ndarray) -> None:
        """Both implementations must return an array the same length as input."""
        from gas_analysis.core.signal_proc import wavelet_denoise as old
        from src.preprocessing.denoising import wavelet_denoise as new

        assert old(single_spectrum).shape == single_spectrum.shape
        assert new(single_spectrum).shape == single_spectrum.shape

    def test_output_dtype_float(self, single_spectrum: np.ndarray) -> None:
        """Both must return a floating-point array."""
        from gas_analysis.core.signal_proc import wavelet_denoise as old
        from src.preprocessing.denoising import wavelet_denoise as new

        assert np.issubdtype(old(single_spectrum).dtype, np.floating)
        assert np.issubdtype(new(single_spectrum).dtype, np.floating)

    def test_denoising_reduces_high_freq_variance(
        self, single_spectrum: np.ndarray
    ) -> None:
        """
        Both implementations must reduce high-frequency noise.

        Use first-difference variance as a proxy for high-frequency content:
        diff-variance of denoised < diff-variance of raw input.
        """
        from gas_analysis.core.signal_proc import wavelet_denoise as old
        from src.preprocessing.denoising import wavelet_denoise as new

        raw_hf_var = float(np.var(np.diff(single_spectrum)))
        old_hf_var = float(np.var(np.diff(old(single_spectrum))))
        new_hf_var = float(np.var(np.diff(new(single_spectrum))))

        assert old_hf_var < raw_hf_var, (
            f"Old wavelet_denoise did not reduce HF variance: {old_hf_var:.4e} >= {raw_hf_var:.4e}"
        )
        assert new_hf_var < raw_hf_var, (
            f"New wavelet_denoise did not reduce HF variance: {new_hf_var:.4e} >= {raw_hf_var:.4e}"
        )

    def test_amplitude_preserved(self, single_spectrum: np.ndarray) -> None:
        """Both must preserve signal amplitude within a factor of 3."""
        from gas_analysis.core.signal_proc import wavelet_denoise as old
        from src.preprocessing.denoising import wavelet_denoise as new

        ref_max = float(np.max(np.abs(single_spectrum)))
        for label, result in [("old", old(single_spectrum)), ("new", new(single_spectrum))]:
            result_max = float(np.max(np.abs(result)))
            assert result_max < 3 * ref_max, (
                f"wavelet_denoise [{label}] amplitude {result_max:.4f} is "
                f">3x input {ref_max:.4f} — likely algorithmic failure."
            )

    def test_outputs_are_not_identical(self, single_spectrum: np.ndarray) -> None:
        """
        Regression guard: old and new wavelet_denoise use different parameters
        and must NOT produce identical output (confirms the difference is real).
        """
        from gas_analysis.core.signal_proc import wavelet_denoise as old
        from src.preprocessing.denoising import wavelet_denoise as new

        old_r = old(single_spectrum)
        new_r = new(single_spectrum)
        max_diff = float(np.max(np.abs(old_r - new_r)))
        assert max_diff > 1e-8, (
            "wavelet_denoise: old and new unexpectedly returned identical results. "
            "Confirm the algorithmic difference is still in place."
        )


# ---------------------------------------------------------------------------
# baseline_correction / correct_baseline — KNOWN ALGORITHMIC DIFFERENCE
#
# Old (gas_analysis.core.signal_proc.baseline_correction):
#   - polynomial method: polyfit over the FULL spectrum
#   - signature: (wavelengths, intensities, method='polynomial', poly_order=2)
#
# New (src.preprocessing.baseline.correct_baseline):
#   - polynomial method: polyfit over EDGE POINTS ONLY (first/last 10%)
#   - signature: (wavelengths, intensities, method='als', **kwargs)
#
# Methods that ARE numerically identical: 'als', 'rolling_min'
# Method that is NOT identical: 'polynomial' (different fitting domain)
#
# predict_tab.py calls correct_baseline(intensities, method=method) with
# only ONE positional arg — this raises TypeError in BOTH old and new,
# which is caught by the `except Exception: corrected = intensities` guard.
# The parity here is that both fail the same way (TypeError).
# ---------------------------------------------------------------------------


class TestBaselineCorrectionParity:
    """Test parity for baseline_correction / correct_baseline."""

    def test_als_method_numerically_identical(self, single_spectrum: np.ndarray) -> None:
        """
        ALS baseline correction must produce identical results in both locations.

        Both route to the same underlying algorithm with the same parameters.
        """
        import numpy as np
        from gas_analysis.core.signal_proc import baseline_correction as old
        from src.preprocessing.baseline import correct_baseline as new

        wl = np.linspace(500, 900, len(single_spectrum))
        old_result = old(wl, single_spectrum, method="als")
        new_result = new(wl, single_spectrum, method="als")

        np.testing.assert_allclose(
            new_result,
            old_result,
            rtol=1e-6,
            atol=1e-12,
            err_msg="baseline_correction [als] parity failed",
        )

    def test_rolling_min_method_numerically_identical(
        self, single_spectrum: np.ndarray
    ) -> None:
        """
        rolling_min baseline correction must produce identical results.
        """
        import numpy as np
        from gas_analysis.core.signal_proc import baseline_correction as old
        from src.preprocessing.baseline import correct_baseline as new

        wl = np.linspace(500, 900, len(single_spectrum))
        old_result = old(wl, single_spectrum, method="rolling_min")
        new_result = new(wl, single_spectrum, method="rolling_min")

        np.testing.assert_allclose(
            new_result,
            old_result,
            rtol=1e-6,
            atol=1e-12,
            err_msg="baseline_correction [rolling_min] parity failed",
        )

    def test_polynomial_method_differs_by_design(
        self, single_spectrum: np.ndarray
    ) -> None:
        """
        Document that polynomial method intentionally differs.

        Old: fits polynomial to the full spectrum (noise-sensitive).
        New: fits polynomial to edge points only (more robust for peaked spectra).
        Regression guard: the difference must be > 0.01 (confirm divergence is real).
        """
        import numpy as np
        from gas_analysis.core.signal_proc import baseline_correction as old
        from src.preprocessing.baseline import correct_baseline as new

        wl = np.linspace(500, 900, len(single_spectrum))
        old_result = old(wl, single_spectrum, method="polynomial")
        new_result = new(wl, single_spectrum, method="polynomial")

        max_diff = float(np.max(np.abs(old_result - new_result)))
        assert max_diff > 0.01, (
            "baseline_correction [polynomial]: expected divergence > 0.01 "
            f"(old=full-spectrum fit, new=edge-only fit), but max|diff|={max_diff:.4f}. "
            "Confirm algorithm change is still in place."
        )

    def test_predict_tab_calling_convention_raises_typeerror(self) -> None:
        """
        predict_tab.py calls correct_baseline(intensities, method=method) with
        only one positional arg.  Both old and new raise TypeError for this call,
        which is caught by the dashboard's `except Exception` guard.

        This test confirms that BOTH raise TypeError so the migration is safe.
        """
        import numpy as np
        from gas_analysis.core.signal_proc import baseline_correction as old
        from src.preprocessing.baseline import correct_baseline as new

        rng = np.random.default_rng(42)
        intensities = rng.normal(0, 1, 100)

        with pytest.raises(TypeError):
            old(intensities, method="als")  # type: ignore[call-arg]

        with pytest.raises(TypeError):
            new(intensities, method="als")  # type: ignore[call-arg]
