"""
src.calibration.physics_kernel
================================
Physics-informed GPR using a Langmuir isotherm as the GP mean function.

The Langmuir model   Δλ(c) = Δλ_max · c / (K_D + c)
captures the sub-linear saturation behaviour of LSPR sensors; the GPR
residual then only has to model deviations from this physically correct
trend, leading to better extrapolation and tighter uncertainty bands.

Public API
----------
- ``LangmuirMeanFunction``   — callable mean function (sklearn GP compatible)
- ``fit_langmuir_params``    — least-squares fit of Δλ_max, K_D from data
- ``PhysicsInformedGPR``     — drop-in replacement for GPRCalibration
"""
from __future__ import annotations

import logging
from typing import Any, cast
import warnings

import numpy as np
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


class LangmuirMeanFunction:
    """Langmuir isotherm Δλ(c) = Δλ_max · c / (K_D + c).

    Parameters
    ----------
    delta_lambda_max:
        Saturation shift (nm). Typically negative for LSPR adsorption.
    k_d:
        Dissociation constant (ppm). The concentration at half-saturation.
    """

    def __init__(self, delta_lambda_max: float = -10.0, k_d: float = 1.0) -> None:
        self.delta_lambda_max = delta_lambda_max
        self.k_d = k_d

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the Langmuir isotherm.

        Parameters
        ----------
        X : shape (n, 1) — concentration column vector (ppm)

        Returns
        -------
        shape (n, 1) — predicted Δλ values (nm)
        """
        c = X.ravel()
        c = np.maximum(c, 0.0)  # concentrations are non-negative
        result = self.delta_lambda_max * c / (self.k_d + c)
        return cast(np.ndarray, np.asarray(result.reshape(-1, 1), dtype=float))


def fit_langmuir_params(
    concentrations: np.ndarray,
    shifts: np.ndarray,
) -> dict[str, float]:
    """Fit Langmuir isotherm parameters by non-linear least squares.

    Parameters
    ----------
    concentrations : 1-D array of concentration values (ppm)
    shifts         : 1-D array of Δλ values (nm), same length

    Returns
    -------
    dict with keys ``delta_lambda_max`` and ``k_d``
    """

    def _langmuir(c: np.ndarray, delta_max: float, k_d: float) -> np.ndarray:
        return cast(np.ndarray, np.asarray(delta_max * c / (k_d + c), dtype=float))

    # Initial guess: max shift from data, K_D = median concentration
    p0 = [float(np.min(shifts)), float(np.median(concentrations))]
    bounds = ([-np.inf, 1e-6], [0.0, np.inf])  # delta_max <= 0, k_d > 0

    try:
        popt, _ = curve_fit(
            _langmuir,
            concentrations,
            shifts,
            p0=p0,
            bounds=bounds,
            maxfev=10_000,
        )
        return {"delta_lambda_max": float(popt[0]), "k_d": float(popt[1])}
    except Exception:
        # Fallback: use the initial guesses
        return {"delta_lambda_max": float(p0[0]), "k_d": float(p0[1])}


class PhysicsInformedGPR:
    """Drop-in replacement for GPRCalibration that uses a Langmuir prior mean.

    Two operating modes, detected automatically from training data:

    * ``_fit_on_shifts=True``  (X=shifts, y=concentrations):
      Plain GP trained directly on (shifts → concentrations).  No Langmuir
      residual subtraction — the Langmuir parameters are still fitted for
      potential forward-direction diagnostics, but the GP sees raw targets.

    * ``_fit_on_shifts=False`` (X=concentrations, y=shifts):
      GP trained on Langmuir residuals: y_fit = shifts − Langmuir(concs).
      ``predict()`` adds the Langmuir mean back to GP output.

    Usage (same contract as GPRCalibration)
    ----------------------------------------
    ::

        model = PhysicsInformedGPR()
        model.fit(shifts.reshape(-1, 1), concentrations)
        mean, std = model.predict(np.array([[-0.75]]))
    """

    def __init__(
        self,
        random_state: int = 42,
        n_restarts_optimizer: int = 10,
        mode: str = "auto",
    ) -> None:
        """
        Parameters
        ----------
        mode : "auto" | "shift_to_conc" | "conc_to_shift"
            Controls how X/y are interpreted:

            * ``"auto"``          — infer from sign of median(X): negative → shift_to_conc.
            * ``"shift_to_conc"`` — X = wavelength shifts (nm, typically negative),
                                    y = concentrations (ppm).  Plain GP, no residual.
            * ``"conc_to_shift"`` — X = concentrations (ppm), y = wavelength shifts (nm).
                                    GP trained on Langmuir residuals.
        """
        if mode not in ("auto", "shift_to_conc", "conc_to_shift"):
            raise ValueError(
                f"mode must be 'auto', 'shift_to_conc', or 'conc_to_shift', got {mode!r}"
            )
        self.random_state = random_state
        self.n_restarts_optimizer = n_restarts_optimizer
        self._mode = mode
        self._langmuir: LangmuirMeanFunction | None = None
        self._gpr: GaussianProcessRegressor | None = None
        self._scaler_X = StandardScaler()
        self._fitted = False
        self._fit_on_shifts: bool = True  # resolved in fit()
        self._linearity_result: dict[str, Any] | None = None  # Mandel test result

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict[str, object]:
        """Fit the physics-informed GPR.

        Parameters
        ----------
        X : (n, 1) — feature column (Δλ shifts or concentrations)
        y : (n,)   — targets

        Returns
        -------
        dict with keys ``log_marginal_likelihood``, ``kernel_params``,
        ``n_samples`` — same contract as ``GPRCalibration.fit()``.
        """
        X_2d = np.atleast_2d(X)
        y_1d = np.ravel(y)

        # Resolve operating mode
        if self._mode == "auto":
            # Heuristic: shifts are typically negative for LSPR adsorption.
            # Use explicit mode="shift_to_conc" or "conc_to_shift" to avoid ambiguity.
            self._fit_on_shifts = bool(np.median(X_2d.ravel()) <= 0)
        else:
            self._fit_on_shifts = self._mode == "shift_to_conc"

        # Always fit Langmuir for diagnostics / forward-direction use
        if self._fit_on_shifts:
            # X = shifts, y = concentrations
            concs = y_1d
            shifts = X_2d.ravel()
        else:
            concs = X_2d.ravel()
            shifts = y_1d

        # Gate Langmuir prior on Mandel's F-test (ICH Q2(R1) §4.2):
        # only apply nonlinear prior when calibration data show statistically
        # significant curvature (p < 0.05).  On linear data the Langmuir prior
        # biases GP residuals and degrades low-concentration extrapolation.
        apply_langmuir = False
        if len(concs) >= 4:
            try:
                from src.scientific.lod import mandel_linearity_test
                linearity = mandel_linearity_test(concs, shifts)
                self._linearity_result = linearity  # type: ignore[assignment]
                is_linear = bool(linearity.get("is_linear", True))
                if is_linear:
                    p_value = float(cast(Any, linearity.get("p_value", 1.0)))
                    log.info(
                        "Mandel's test: linear model sufficient (p=%.3f); "
                        "Langmuir prior not applied.",
                        p_value,
                    )
                else:
                    apply_langmuir = True
                    p_value = float(cast(Any, linearity.get("p_value", 0.0)))
                    log.info(
                        "Mandel's test: significant nonlinearity (p=%.3f); "
                        "Langmuir prior applied.",
                        p_value,
                    )
            except Exception as exc:
                warnings.warn(
                    f"Mandel linearity test failed ({exc}); Langmuir prior applied by default.",
                    UserWarning,
                    stacklevel=3,
                )
                apply_langmuir = True
        elif len(concs) >= 3:
            # Too few points for F-test — apply Langmuir conservatively
            apply_langmuir = True

        if apply_langmuir:
            params = fit_langmuir_params(concs, shifts)
            self._langmuir = LangmuirMeanFunction(**params)
        else:
            self._langmuir = None

        # Build GP training arrays
        if self._fit_on_shifts:
            # Plain GP: shifts → concentrations (no residual subtraction)
            X_fit = X_2d
            y_fit = y_1d
        else:
            # Langmuir-residual GP: concs → (shifts − Langmuir(concs))
            if self._langmuir is not None:
                langmuir_pred = self._langmuir(X_2d).ravel()
                y_fit = y_1d - langmuir_pred
            else:
                y_fit = y_1d
            X_fit = X_2d

        X_scaled = self._scaler_X.fit_transform(X_fit)

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5
        ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
        self._gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
        )
        self._gpr.fit(X_scaled, y_fit)
        self._fitted = True

        out: dict[str, Any] = {
            "log_marginal_likelihood": float(
                self._gpr.log_marginal_likelihood(self._gpr.kernel_.theta)
            ),
            "kernel_params": {
                k: float(v)
                for k, v in self._gpr.kernel_.get_params().items()
                if isinstance(v, (int, float))
            },
            "n_samples": len(y_fit),
            "langmuir_applied": self._langmuir is not None,
        }
        if self._linearity_result is not None:
            out["mandel_linearity"] = self._linearity_result
        return out

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and std (1-sigma).

        Returns
        -------
        (mean, std) both shape (n,)
        """
        if not self._fitted or self._gpr is None:
            raise RuntimeError("PhysicsInformedGPR.fit() must be called first.")

        X_2d = np.atleast_2d(X)
        X_scaled = self._scaler_X.transform(X_2d)

        gpr_mean, gpr_std = self._gpr.predict(X_scaled, return_std=True)

        if self._langmuir is not None and not self._fit_on_shifts:
            # Add Langmuir mean back to recover predicted shifts
            langmuir_pred = self._langmuir(X_2d).ravel()
            mean = gpr_mean + langmuir_pred
        else:
            mean = gpr_mean

        if not return_std:
            return mean.ravel(), np.zeros_like(mean.ravel())
        return mean.ravel(), gpr_std.ravel()
