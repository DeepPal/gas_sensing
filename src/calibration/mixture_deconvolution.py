"""
src.calibration.mixture_deconvolution
=======================================
Multi-analyte concentration estimation from observed peak-shift vectors.

Two complementary solvers are provided:

1. **LinearDeconvolver** — fast, analytic pseudoinverse solution.
   Valid in the linear (low-concentration) regime: c << K_d.
   ĉ = S⁺ × Δλ_observed.

2. **LangmuirDeconvolver** — non-linear iterative solver.
   Handles the full Langmuir isotherm including saturation and (optionally)
   competitive binding.  Uses scipy.optimize.minimize with the Nelder-Mead
   simplex or L-BFGS-B gradient method.

   Minimises:
       L(c) = ||Δλ_observed − Δλ_predicted(c)||² + λ × regularisation(c)

   where:
       Δλ_predicted_j(c) = Σᵢ S_ij × cᵢ / (1 + cᵢ / K_d_ij)   (superposition)

Physical assumptions
--------------------
- Superposition holds: each analyte's contribution to each peak is independent
  (valid in the linear regime; breaks down in competitive binding at high conc).
- All analytes have been calibrated (S matrix and K_d values are known).
- The dominant noise source is Gaussian on the peak shift estimate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize, OptimizeResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class DeconvolutionResult:
    """Result of one mixture deconvolution inference.

    Attributes
    ----------
    concentrations:
        Estimated analyte concentrations (ppm). Dict: analyte name → ppm.
    residual_nm:
        RMS residual in nm: sqrt(mean((Δλ_obs − Δλ_pred)²)).
        Small residual = good fit; large = out-of-calibration or wrong analytes.
    success:
        Whether the optimiser converged (non-linear solver only).
    solver:
        Which solver was used: 'linear' or 'langmuir'.
    iterations:
        Number of solver iterations (non-linear only; None for linear).
    predicted_shifts:
        Δλ_predicted from the solution — for residual diagnostics.
    """
    concentrations: dict[str, float]
    residual_nm: float
    success: bool
    solver: str
    iterations: int | None
    predicted_shifts: np.ndarray


# ---------------------------------------------------------------------------
# Langmuir forward model
# ---------------------------------------------------------------------------


def langmuir_predicted_shifts(
    c: np.ndarray,
    S: np.ndarray,
    Kd: np.ndarray,
) -> np.ndarray:
    """Compute predicted peak shifts from concentration vector.

    Parameters
    ----------
    c:
        Concentration vector (ppm), shape (N,).
    S:
        Sensitivity matrix (nm/ppm), shape (N, M).
    Kd:
        Langmuir K_d matrix (ppm), shape (N, M).
        Set to np.inf for linear (no saturation).

    Returns
    -------
    np.ndarray
        Predicted peak shifts (nm), shape (M,).
    """
    N, M = S.shape
    shifts = np.zeros(M)
    for i in range(N):
        for j in range(M):
            kd = Kd[i, j]
            if kd > 0 and np.isfinite(kd):
                shifts[j] += S[i, j] * c[i] / (1.0 + c[i] / kd)
            else:
                shifts[j] += S[i, j] * c[i]
    return shifts


# ---------------------------------------------------------------------------
# Linear solver
# ---------------------------------------------------------------------------


class LinearDeconvolver:
    """Fast analytic pseudoinverse deconvolution (linear regime).

    Valid when concentrations are well below K_d (saturation threshold).
    Approximately linear for c < K_d / 5.

    Parameters
    ----------
    analytes:
        List of analyte names.
    S:
        Sensitivity matrix, shape (N, M).
    """

    def __init__(self, analytes: list[str], S: np.ndarray) -> None:
        self._analytes = list(analytes)
        self._S = np.asarray(S, dtype=float)  # (N, M)
        self._S_pinv = np.linalg.pinv(self._S.T)  # (N, M)

    @classmethod
    def from_sensitivity_matrix(cls, sm: object) -> "LinearDeconvolver":
        """Construct from a fitted :class:`~src.calibration.sensitivity_matrix.SensitivityMatrix`."""
        return cls(sm._analytes, sm.matrix)  # type: ignore[attr-defined]

    def solve(self, delta_lambda: np.ndarray) -> DeconvolutionResult:
        """Estimate concentrations from observed peak shifts.

        Parameters
        ----------
        delta_lambda:
            Observed peak shifts (nm), shape (M,).

        Returns
        -------
        DeconvolutionResult
        """
        dl = np.asarray(delta_lambda, dtype=float)
        # Least-squares solve: S^T × ĉ = Δλ
        c_est, _, _, _ = np.linalg.lstsq(self._S.T, dl, rcond=None)

        dl_pred = self._S.T @ c_est
        residual = float(np.sqrt(np.mean((dl - dl_pred) ** 2)))

        return DeconvolutionResult(
            concentrations={
                name: max(0.0, float(c_est[i]))  # clip negatives
                for i, name in enumerate(self._analytes)
            },
            residual_nm=residual,
            success=True,
            solver="linear",
            iterations=None,
            predicted_shifts=dl_pred,
        )


# ---------------------------------------------------------------------------
# Non-linear Langmuir solver
# ---------------------------------------------------------------------------


class LangmuirDeconvolver:
    """Non-linear deconvolution using the full Langmuir isotherm.

    Handles saturation regime (c ≈ K_d) and optionally regularisation
    for ill-conditioned sensitivity matrices.

    Parameters
    ----------
    analytes:
        List of analyte names.
    S:
        Sensitivity matrix (nm/ppm), shape (N, M).
    Kd:
        Dissociation constant matrix (ppm), shape (N, M).
        Use ``np.full_like(S, np.inf)`` for no saturation (linear).
    conc_bounds:
        (min, max) concentration bounds in ppm applied to all analytes.
        Default: (0.0, 1000.0).
    regularisation:
        L2 regularisation weight. Helps when S is ill-conditioned (κ >> 10).
        0 = no regularisation; 0.01–0.1 typical for ill-conditioned cases.
    method:
        scipy optimisation method. 'L-BFGS-B' (gradient, fast) or
        'Nelder-Mead' (gradient-free, robust to noise in cost function).
    """

    def __init__(
        self,
        analytes: list[str],
        S: np.ndarray,
        Kd: np.ndarray | None = None,
        conc_bounds: tuple[float, float] = (0.0, 1000.0),
        regularisation: float = 0.0,
        method: str = "L-BFGS-B",
    ) -> None:
        self._analytes = list(analytes)
        self._S = np.asarray(S, dtype=float)
        N, M = self._S.shape
        if Kd is None:
            self._Kd = np.full((N, M), np.inf)
        else:
            self._Kd = np.asarray(Kd, dtype=float)
        self._bounds = [conc_bounds] * len(analytes)
        self._reg = regularisation
        self._method = method
        # Warm start: use linear solution as initial guess
        self._linear = LinearDeconvolver(analytes, S)

    @classmethod
    def from_sensitivity_matrix(
        cls,
        sm: object,
        Kd: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "LangmuirDeconvolver":
        """Construct from a fitted SensitivityMatrix."""
        return cls(sm._analytes, sm.matrix, Kd=Kd, **kwargs)  # type: ignore[attr-defined]

    def _cost(self, c: np.ndarray, dl_obs: np.ndarray) -> float:
        """Objective: ||Δλ_obs − Δλ_pred(c)||² + λ ||c||²"""
        dl_pred = langmuir_predicted_shifts(c, self._S, self._Kd)
        residual_sq = float(np.sum((dl_obs - dl_pred) ** 2))
        reg_term = self._reg * float(np.sum(c ** 2))
        return residual_sq + reg_term

    def solve(
        self,
        delta_lambda: np.ndarray,
        x0: np.ndarray | None = None,
    ) -> DeconvolutionResult:
        """Estimate concentrations from observed peak shifts.

        Parameters
        ----------
        delta_lambda:
            Observed peak shifts (nm), shape (M,).
        x0:
            Initial concentration guess (ppm). If None, uses the linear
            solution as a warm start.

        Returns
        -------
        DeconvolutionResult
        """
        dl = np.asarray(delta_lambda, dtype=float)

        # Warm start from linear solution
        if x0 is None:
            lin = self._linear.solve(dl)
            x0 = np.array([lin.concentrations[name] for name in self._analytes])
            x0 = np.maximum(x0, 0.0)

        result: OptimizeResult = minimize(
            self._cost,
            x0,
            args=(dl,),
            method=self._method,
            bounds=self._bounds if self._method in ("L-BFGS-B", "SLSQP") else None,
            options={"maxiter": 500, "ftol": 1e-10} if self._method == "L-BFGS-B" else {"maxiter": 1000},
        )

        c_est = np.maximum(result.x, 0.0)
        dl_pred = langmuir_predicted_shifts(c_est, self._S, self._Kd)
        residual = float(np.sqrt(np.mean((dl - dl_pred) ** 2)))

        if not result.success:
            log.warning(
                "LangmuirDeconvolver did not converge: %s (residual=%.4f nm)",
                result.message,
                residual,
            )

        return DeconvolutionResult(
            concentrations={
                name: float(c_est[i]) for i, name in enumerate(self._analytes)
            },
            residual_nm=residual,
            success=bool(result.success),
            solver="langmuir",
            iterations=int(result.nit),
            predicted_shifts=dl_pred,
        )

    def solve_batch(
        self,
        delta_lambda_matrix: np.ndarray,
    ) -> list[DeconvolutionResult]:
        """Solve a batch of frames.

        Parameters
        ----------
        delta_lambda_matrix:
            Shape (n_frames, n_peaks).

        Returns
        -------
        List of DeconvolutionResult, one per frame.
        """
        results = []
        x0 = None
        for dl in delta_lambda_matrix:
            res = self.solve(dl, x0=x0)
            # Use previous solution as warm start for next frame (time-series continuity)
            x0 = np.array([res.concentrations[name] for name in self._analytes])
            results.append(res)
        return results


# ---------------------------------------------------------------------------
# Automatic solver selection
# ---------------------------------------------------------------------------


def deconvolve_mixture(
    delta_lambda: np.ndarray,
    analytes: list[str],
    S: np.ndarray,
    Kd: np.ndarray | None = None,
    use_nonlinear: bool = True,
    regularisation: float = 0.0,
) -> DeconvolutionResult:
    """Convenience wrapper: automatically selects linear or Langmuir solver.

    Uses the linear solver when all concentrations are expected to be well
    below K_d (linear regime check after initial linear solution). Switches
    to the non-linear Langmuir solver if any concentration approaches K_d.

    Parameters
    ----------
    delta_lambda:
        Observed peak shifts (nm), shape (M,).
    analytes, S, Kd:
        Sensor calibration parameters (same as LangmuirDeconvolver).
    use_nonlinear:
        Force Langmuir solver even in the linear regime.
    regularisation:
        L2 regularisation for ill-conditioned S.
    """
    # Always compute a fast linear solution first
    lin_solver = LinearDeconvolver(analytes, S)
    lin_result = lin_solver.solve(delta_lambda)

    if not use_nonlinear or Kd is None:
        return lin_result

    # Check if any concentration is in the non-linear regime (c > Kd/5)
    c_est_lin = np.array([lin_result.concentrations[name] for name in analytes])
    if np.all(Kd == np.inf) or np.all(c_est_lin < Kd.min(axis=1) / 5.0):
        # Safe to use linear result
        return lin_result

    # Concentrate in non-linear regime → use Langmuir solver
    solver = LangmuirDeconvolver(
        analytes, S, Kd,
        regularisation=regularisation,
    )
    return solver.solve(delta_lambda, x0=c_est_lin)
