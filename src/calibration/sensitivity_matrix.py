"""
src.calibration.sensitivity_matrix
=====================================
Sensitivity matrix calibration for multi-peak, multi-analyte optical sensors.

Mathematical framework
----------------------
The sensitivity matrix S ∈ ℝ^{N×M} relates analyte concentrations to peak shifts:

    Δλ = S^T × c  (linear regime)

where:
  - Δλ ∈ ℝ^M  — observed peak shift vector (one entry per spectral peak)
  - c ∈ ℝ^N   — analyte concentration vector (ppm)
  - S[i,j]    — sensitivity of peak j to analyte i (nm/ppm)
  - N          — number of analytes
  - M          — number of spectral peaks

Concentration estimation (deconvolution):

    ĉ = S⁺ × Δλ_observed

where S⁺ is the Moore-Penrose pseudoinverse.  For square systems (N=M),
this is exact (if S is invertible); for overdetermined (M>N), it is the
least-squares minimum-norm solution.

Calibration procedure
----------------------
1. Expose the sensor to each pure analyte independently at multiple
   concentrations (selectivity protocol).
2. For each pure-analyte run, extract steady-state Δλ per peak.
3. Fit a linear slope (or GPR for nonlinear sensors) for each S[i,j].
4. Optionally fit cross-terms by running mixture exposures.

Quality metrics
---------------
- Condition number κ(S): low = analytes well-discriminable; high = ill-posed
- R² per S[i,j] fit
- LOD per analyte in mixture context (accounting for cross-interference)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.linalg import lstsq, matrix_rank, norm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class SensitivityEntry:
    """Calibration fit result for one (analyte i, peak j) pair."""
    analyte: str
    peak_idx: int
    slope_nm_per_ppm: float         # S[i,j]: linear sensitivity
    intercept_nm: float             # offset at zero concentration (should ≈ 0)
    r_squared: float                # fit quality
    n_points: int                   # calibration points used
    conc_range_ppm: tuple[float, float]  # (min, max) calibration range


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SensitivityMatrix:
    """Multi-peak, multi-analyte sensitivity matrix calibration.

    Build the S matrix from single-analyte calibration runs, then use it
    to estimate concentrations from observed peak shift vectors.

    Example
    -------
    ::

        sm = SensitivityMatrix(analytes=['Ethanol', 'Acetone'], n_peaks=2)
        # Pure Ethanol calibration
        sm.fit_analyte('Ethanol', 0,
                       conc_ppm=[0.1, 0.5, 1.0, 2.0, 5.0],
                       shifts_nm=[-0.05, -0.25, -0.50, -1.00, -2.50])
        sm.fit_analyte('Ethanol', 1,
                       conc_ppm=[0.1, 0.5, 1.0, 2.0, 5.0],
                       shifts_nm=[-0.03, -0.15, -0.32, -0.65, -1.60])
        # Pure Acetone calibration
        sm.fit_analyte('Acetone', 0, ...)
        sm.fit_analyte('Acetone', 1, ...)

        # Estimate concentrations in a mixture
        delta_lambda = np.array([-1.20, -0.82])
        c_est, c_std = sm.estimate_concentrations(delta_lambda)
        # {'Ethanol': 2.15 ppm, 'Acetone': 0.93 ppm}
    """

    def __init__(self, analytes: list[str], n_peaks: int) -> None:
        self._analytes = list(analytes)
        self._n_peaks = n_peaks
        self._n_analytes = len(analytes)
        # S[i, j] = sensitivity of peak j to analyte i (nm/ppm)
        self._S: np.ndarray = np.zeros((self._n_analytes, n_peaks))
        self._entries: dict[tuple[str, int], SensitivityEntry] = {}
        self._fitted: set[tuple[str, int]] = set()

    # ── Fitting ───────────────────────────────────────────────────────────

    def fit_analyte(
        self,
        analyte: str,
        peak_idx: int,
        conc_ppm: list[float] | np.ndarray,
        shifts_nm: list[float] | np.ndarray,
        force_zero_intercept: bool = False,
    ) -> SensitivityEntry:
        """Fit S[analyte, peak] from single-analyte calibration data.

        Parameters
        ----------
        analyte:
            Analyte name (must be in ``analytes`` list passed to constructor).
        peak_idx:
            Which spectral peak (0-indexed).
        conc_ppm:
            Calibration concentrations (ppm).
        shifts_nm:
            Measured peak shifts (nm) at each concentration.
            Sign: blue-shift = negative.
        force_zero_intercept:
            If True, fit passes through (0, 0) — theoretically correct but
            may reduce R² if there is a small measurement offset.

        Returns
        -------
        SensitivityEntry
            Fit result including slope, intercept, R².
        """
        if analyte not in self._analytes:
            raise ValueError(
                f"Analyte '{analyte}' not in registered analytes {self._analytes}"
            )
        if not 0 <= peak_idx < self._n_peaks:
            raise IndexError(f"peak_idx {peak_idx} out of range [0, {self._n_peaks})")

        c = np.asarray(conc_ppm, dtype=float)
        dl = np.asarray(shifts_nm, dtype=float)

        if len(c) != len(dl):
            raise ValueError("conc_ppm and shifts_nm must have the same length")
        if len(c) < 2:
            raise ValueError("Need at least 2 calibration points")

        if force_zero_intercept:
            # OLS through origin: slope = (c·Δλ) / (c·c)
            slope = float(np.dot(c, dl) / np.dot(c, c))
            intercept = 0.0
            y_hat = slope * c
        else:
            A = np.column_stack([c, np.ones(len(c))])
            coeffs, _, _, _ = lstsq(A, dl, rcond=None)
            slope, intercept = float(coeffs[0]), float(coeffs[1])
            y_hat = slope * c + intercept

        ss_res = float(np.sum((dl - y_hat) ** 2))
        ss_tot = float(np.sum((dl - dl.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

        entry = SensitivityEntry(
            analyte=analyte,
            peak_idx=peak_idx,
            slope_nm_per_ppm=slope,
            intercept_nm=intercept,
            r_squared=r2,
            n_points=len(c),
            conc_range_ppm=(float(c.min()), float(c.max())),
        )
        i = self._analytes.index(analyte)
        self._S[i, peak_idx] = slope
        self._entries[(analyte, peak_idx)] = entry
        self._fitted.add((analyte, peak_idx))

        log.debug(
            "S[%s, peak %d] = %.4f nm/ppm  R²=%.4f",
            analyte, peak_idx, slope, r2,
        )
        return entry

    def fit_from_dataframe(self, df: "pd.DataFrame") -> None:  # noqa: F821
        """Fit the full S matrix from a calibration DataFrame.

        Expects columns: ``analyte``, ``concentration_ppm``,
        ``peak_shift_0``, ``peak_shift_1``, …, ``peak_shift_{M-1}``.
        """
        for analyte in df["analyte"].unique():
            sub = df[df["analyte"] == analyte].copy()
            for j in range(self._n_peaks):
                col = f"peak_shift_{j}"
                if col not in sub.columns:
                    continue
                valid = sub.dropna(subset=[col, "concentration_ppm"])
                if len(valid) < 2:
                    continue
                self.fit_analyte(
                    analyte=analyte,
                    peak_idx=j,
                    conc_ppm=valid["concentration_ppm"].values,
                    shifts_nm=valid[col].values,
                )

    # ── Concentration estimation ──────────────────────────────────────────

    def estimate_concentrations(
        self,
        delta_lambda: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Estimate analyte concentrations from observed peak shifts.

        Solves: ĉ = S⁺ × Δλ  (least-squares / pseudoinverse).

        Parameters
        ----------
        delta_lambda:
            Observed peak shift vector, shape (n_peaks,), in nm.

        Returns
        -------
        concentrations:
            Dict mapping analyte name → estimated concentration (ppm).
        residuals:
            Dict mapping analyte name → residual uncertainty proxy.
        """
        if not self._is_fully_fitted():
            raise RuntimeError(
                "Not all S[i,j] entries are fitted. Call fit_analyte() for each "
                "(analyte, peak) pair before estimating concentrations."
            )

        dl = np.asarray(delta_lambda, dtype=float)
        if dl.shape != (self._n_peaks,):
            raise ValueError(
                f"delta_lambda shape {dl.shape} does not match n_peaks={self._n_peaks}"
            )

        # Pseudoinverse solution: ĉ = S⁺ × Δλ
        # S^T × ĉ = Δλ  →  S_T = S.T  shape (M, N)
        # ĉ = pinv(S.T) × Δλ
        S_T = self._S.T  # (M, N)
        c_est, _, _, _ = lstsq(S_T, dl, rcond=None)

        # Residual: how well S^T × ĉ reconstructs Δλ
        dl_reconstructed = S_T @ c_est
        residual_norm = float(norm(dl - dl_reconstructed))

        concentrations = {
            name: float(c_est[i]) for i, name in enumerate(self._analytes)
        }
        residuals = {
            name: residual_norm / self._n_peaks for name in self._analytes
        }
        return concentrations, residuals

    # ── Quality metrics ───────────────────────────────────────────────────

    @property
    def condition_number(self) -> float:
        """Condition number κ(S).

        Indicates how well analytes can be discriminated from peak shifts:
        - κ close to 1: analytes produce orthogonal peak shift patterns → easy
        - κ > 100: ill-conditioned → large concentration estimation errors
        """
        if not self._is_fully_fitted():
            return float("nan")
        return float(np.linalg.cond(self._S))

    @property
    def matrix(self) -> np.ndarray:
        """Raw S matrix, shape (n_analytes, n_peaks)."""
        return self._S.copy()

    @property
    def rank(self) -> int:
        """Rank of S. Must equal n_analytes for full identifiability."""
        return int(matrix_rank(self._S))

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of the calibration quality."""
        entries_summary = [
            {
                "analyte": e.analyte,
                "peak": e.peak_idx,
                "slope_nm_per_ppm": e.slope_nm_per_ppm,
                "r_squared": e.r_squared,
                "n_points": e.n_points,
                "conc_range_ppm": e.conc_range_ppm,
            }
            for e in self._entries.values()
        ]
        return {
            "analytes": self._analytes,
            "n_peaks": self._n_peaks,
            "S_matrix": self._S.tolist(),
            "condition_number": self.condition_number,
            "rank": self.rank,
            "is_fully_fitted": self._is_fully_fitted(),
            "entries": entries_summary,
        }

    def compute_lod_mixture(
        self,
        noise_nm: float = 0.05,
        k: float = 3.0,
    ) -> dict[str, float]:
        """LOD for each analyte in mixture context (accounting for cross-interference).

        LOD_i = k × σ_noise / |effective_sensitivity_i|

        where effective_sensitivity_i is the diagonal of (S⁺)^T — the row
        of the pseudoinverse corresponding to analyte i — dotted with the
        noise vector.

        Parameters
        ----------
        noise_nm:
            Estimated 1-sigma peak detection noise (nm). Typical: 0.02–0.1 nm.
        k:
            LOD factor (3 for ~99.7% confidence under Gaussian noise).
        """
        if not self._is_fully_fitted():
            return {name: float("nan") for name in self._analytes}

        S_pinv = np.linalg.pinv(self._S.T)  # shape (N, M)
        lod = {}
        for i, name in enumerate(self._analytes):
            row = S_pinv[i, :]
            # Propagated noise: σ_c = k × σ_Δλ × ‖row‖
            effective_noise = k * noise_nm * float(norm(row))
            lod[name] = effective_noise
        return lod

    # ── Serialisation ─────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist to a .joblib file."""
        import joblib
        joblib.dump({
            "analytes": self._analytes,
            "n_peaks": self._n_peaks,
            "S": self._S,
            "entries": self._entries,
            "fitted": list(self._fitted),
        }, path)

    @classmethod
    def load(cls, path: str) -> "SensitivityMatrix":
        """Load from a .joblib file created by :meth:`save`."""
        import joblib
        state = joblib.load(path)
        obj = cls(state["analytes"], state["n_peaks"])
        obj._S = state["S"]
        obj._entries = state["entries"]
        obj._fitted = set(map(tuple, state["fitted"]))
        return obj

    # ── Helpers ───────────────────────────────────────────────────────────

    def _is_fully_fitted(self) -> bool:
        """True if every (analyte, peak) combination has been fitted."""
        for i, name in enumerate(self._analytes):
            for j in range(self._n_peaks):
                if (name, j) not in self._fitted:
                    return False
        return True
