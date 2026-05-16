"""
src.calibration.isotherms
==========================
Nonlinear calibration models (binding isotherms) for LSPR gas sensors.

Background
----------
Linear calibration (OLS) is valid only within the Henry-law regime
(low concentration, unsaturated binding sites).  At higher concentrations
the sensor response follows Langmuir kinetics because the number of MIP
binding sites is finite:

    Langmuir:   R(c) = R_max · K · c / (1 + K · c)

where *R_max* is the saturation response and *K* is the affinity constant.

When binding sites have a heterogeneous energy distribution (typical in
polymer-based sensors), the Freundlich isotherm applies:

    Freundlich:  R(c) = K · c^n   (0 < n < 1 for sub-linear response)

The Hill model generalises Langmuir with a cooperativity exponent *n*:

    Hill:        R(c) = R_max · c^n / (K_d^n + c^n)

All models are fit by nonlinear least squares (scipy.optimize.curve_fit)
rather than linearisation, which is the statistically correct approach
and required for ICH Q2(R1) accuracy claims.

Public API
----------
- ``fit_langmuir``     — Langmuir isotherm fit
- ``fit_freundlich``   — Freundlich power-law fit
- ``fit_hill``         — Hill model (cooperative binding)
- ``select_isotherm``  — AIC-based model selection
- ``IsothermResult``   — result dataclass returned by all fit functions
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np
from scipy.optimize import curve_fit

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class IsothermResult:
    """Result of a nonlinear isotherm fit.

    Attributes
    ----------
    model:
        Model name (``'langmuir'``, ``'freundlich'``, or ``'hill'``).
    params:
        Fitted parameters (model-specific; see individual fit functions).
    param_stderrs:
        1-σ parameter uncertainties from the covariance matrix.
    r_squared:
        Coefficient of determination (R²) of the fit.
    rmse:
        Root-mean-squared error in the same units as *responses*.
    aic:
        Akaike Information Criterion — lower is better for model selection.
    n_params:
        Number of free parameters in the model.
    concentrations_fit:
        Dense concentration grid for plotting the fitted curve.
    responses_fit:
        Predicted response on ``concentrations_fit``.
    """

    model: str
    params: dict[str, float]
    param_stderrs: dict[str, float]
    r_squared: float
    rmse: float
    aic: float
    n_params: int
    concentrations_fit: np.ndarray = field(repr=False)
    responses_fit: np.ndarray = field(repr=False)

    def predict(self, concentrations: np.ndarray) -> np.ndarray:
        """Predict sensor response at arbitrary concentrations."""
        c = np.asarray(concentrations, dtype=float)
        sign = float(self.params.get("sign", 1.0))  # stored by fit functions
        if self.model == "langmuir":
            R_max = self.params["R_max"]
            K = self.params["K"]
            return np.asarray(sign * R_max * K * c / (1.0 + K * c))
        if self.model == "freundlich":
            K = self.params["K"]
            n = self.params["n"]
            return np.asarray(sign * K * np.abs(c) ** n)
        if self.model == "hill":
            R_max = self.params["R_max"]
            K_d = self.params["K_d"]
            n = self.params["n"]
            return np.asarray(sign * R_max * c**n / (K_d**n + c**n))
        if self.model == "linear":
            slope = self.params["slope"]
            intercept = self.params["intercept"]
            return np.asarray(slope * c + intercept)
        raise ValueError(f"Unknown model: {self.model!r}")

    def __str__(self) -> str:
        param_str = ", ".join(
            f"{k}={v:.4g}±{self.param_stderrs.get(k, float('nan')):.2g}"
            for k, v in self.params.items()
        )
        return (
            f"IsothermResult({self.model}: {param_str}, "
            f"R²={self.r_squared:.4f}, RMSE={self.rmse:.4g}, AIC={self.aic:.2f})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _r_squared(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res: float = float(np.sum((y_obs - y_pred) ** 2))
    ss_tot: float = float(np.sum((y_obs - y_obs.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _rmse(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_obs - y_pred) ** 2)))


def _aic(n_obs: int, n_params: int, rss: float) -> float:
    """AICc (AIC with small-sample correction) for Gaussian errors.

    AICc = n·ln(RSS/n) + 2k + 2k(k+1)/(n−k−1)

    The correction term is essential for calibration datasets (n = 3–10);
    at n=5, k=3 the correction adds 24 — dwarfing the bare ΔAIC threshold.
    Falls back to bare AIC when n−k−1 ≤ 0 (degenerate case).
    """
    if n_obs <= n_params:
        return float("inf")
    rss_safe = max(rss, 1e-300)  # avoid log(0); perfect fit → AIC → −∞
    aic = float(n_obs * np.log(rss_safe / n_obs) + 2.0 * n_params)
    denom = n_obs - n_params - 1
    if denom > 0:
        aic += 2.0 * n_params * (n_params + 1) / denom
    return aic


def _safe_perr(pcov: np.ndarray) -> np.ndarray:
    """Extract parameter SEs from covariance matrix, guarding inf/NaN diagonals."""
    if not np.all(np.isfinite(pcov)):
        return np.full(pcov.shape[0], float("nan"))
    diag = np.diag(pcov)
    return np.where(diag >= 0, np.sqrt(np.clip(diag, 0.0, None)), float("nan"))


def _build_fit_grid(c: np.ndarray, n_pts: int = 200) -> np.ndarray:
    """Dense grid from min(c) to 1.2×max(c) for plotting fitted curves.

    Starts at min(c) rather than 0 to avoid 0^n artefacts in Freundlich/Hill.
    """
    c_start = max(float(np.min(c)), 1e-6)
    return np.asarray(np.linspace(c_start, float(np.max(c)) * 1.2, n_pts))


# ---------------------------------------------------------------------------
# Langmuir isotherm
# ---------------------------------------------------------------------------


def fit_langmuir(
    concentrations: np.ndarray,
    responses: np.ndarray,
    R_max_init: float | None = None,
    K_init: float | None = None,
) -> IsothermResult:
    """Fit a Langmuir adsorption isotherm to calibration data.

    Model
    -----
    .. math::

        R(c) = \\frac{R_{\\max} \\cdot K \\cdot c}{1 + K \\cdot c}

    - *R_max*: maximum (saturation) response.
    - *K*: affinity constant (ppm⁻¹); higher → higher affinity.

    The Langmuir model is appropriate when:

    - Binding sites are finite and equivalent.
    - Response saturates at high concentration.
    - R² > 0.99 across the full measured range.

    Parameters
    ----------
    concentrations:
        Analyte concentrations in ppm, shape ``(n,)``.  Must be positive.
    responses:
        Sensor responses (e.g. |Δλ| in nm), shape ``(n,)``.
        Signs are preserved; pass absolute values if working with
        magnitude-only calibration.
    R_max_init, K_init:
        Initial parameter guesses.  If ``None``, auto-estimated from data.

    Returns
    -------
    :class:`IsothermResult`
    """
    c = np.asarray(concentrations, dtype=float).ravel()
    r = np.asarray(responses, dtype=float).ravel()

    if len(c) < 3:
        raise ValueError("Langmuir fit requires at least 3 calibration points.")

    if R_max_init is None:
        R_max_init = float(np.max(np.abs(r)) * 1.5)
    if K_init is None:
        # Estimate K from half-saturation: R(c_half) ≈ R_max/2 → K ≈ 1/c_half
        r_half = R_max_init / 2.0
        idx_half = np.argmin(np.abs(np.abs(r) - r_half))
        K_init = float(1.0 / max(c[idx_half], 1e-6))

    sign_r = -1.0 if np.mean(r) < 0 else 1.0

    def _langmuir(c_arr: np.ndarray, R_max: float, K: float) -> np.ndarray:
        return np.asarray(sign_r * R_max * K * c_arr / (1.0 + K * c_arr))

    try:
        popt, pcov = curve_fit(
            _langmuir,
            c,
            r,
            p0=[abs(R_max_init), K_init],
            bounds=([0, 1e-6], [np.inf, np.inf]),
            maxfev=5000,
        )
        perr = _safe_perr(pcov)
        R_max_fit, K_fit = popt
        R_max_se, K_se = perr
    except RuntimeError as exc:
        log.warning("Langmuir fit failed: %s", exc)
        raise

    r_pred = _langmuir(c, R_max_fit, K_fit)
    rss = float(np.sum((r - r_pred) ** 2))
    r2 = _r_squared(r, r_pred)
    rmse = _rmse(r, r_pred)
    aic = _aic(len(c), 2, rss)

    c_grid = _build_fit_grid(c)
    r_grid = _langmuir(c_grid, R_max_fit, K_fit)

    log.info(
        "Langmuir fit: R_max=%.4f±%.4f, K=%.4f±%.4f ppm⁻¹, R²=%.4f",
        R_max_fit,
        R_max_se,
        K_fit,
        K_se,
        r2,
    )

    return IsothermResult(
        model="langmuir",
        params={"R_max": float(R_max_fit), "K": float(K_fit), "sign": float(sign_r)},
        param_stderrs={"R_max": float(R_max_se), "K": float(K_se), "sign": 0.0},
        r_squared=round(r2, 6),
        rmse=round(rmse, 8),
        aic=round(aic, 4),
        n_params=2,
        concentrations_fit=c_grid,
        responses_fit=r_grid,
    )


# ---------------------------------------------------------------------------
# Freundlich isotherm
# ---------------------------------------------------------------------------


def fit_freundlich(
    concentrations: np.ndarray,
    responses: np.ndarray,
) -> IsothermResult:
    """Fit a Freundlich power-law isotherm to calibration data.

    Model
    -----
    .. math::

        R(c) = K \\cdot c^n

    Linearised as ``log|R| = log|K| + n·log(c)`` for initial estimates,
    then refined by nonlinear least squares on the original form.

    - *K*: Freundlich coefficient (includes sign of response).
    - *n*: empirical exponent; 0 < n < 1 → sub-linear (typical for MIP);
      n = 1 → linear (reduces to Henry's law); n > 1 → cooperative.

    Parameters
    ----------
    concentrations:
        Positive concentrations in ppm.
    responses:
        Sensor responses (sign-preserving).

    Returns
    -------
    :class:`IsothermResult`
    """
    c = np.asarray(concentrations, dtype=float).ravel()
    r = np.asarray(responses, dtype=float).ravel()

    if len(c) < 3:
        raise ValueError("Freundlich fit requires at least 3 calibration points.")
    if np.any(c <= 0):
        raise ValueError("Freundlich model requires all concentrations > 0.")

    sign_r = -1.0 if np.mean(r) < 0 else 1.0
    r_abs = np.abs(r)

    # Log-linear initial estimate
    log_c = np.log(c)
    log_r = np.log(np.maximum(r_abs, 1e-12))
    p = np.polyfit(log_c, log_r, 1)
    n_init = float(p[0])
    K_init = float(np.exp(p[1]))

    def _freundlich(c_arr: np.ndarray, K: float, n: float) -> np.ndarray:
        return np.asarray(sign_r * K * c_arr**n)

    try:
        popt, pcov = curve_fit(
            _freundlich,
            c,
            r,
            p0=[K_init, max(n_init, 0.1)],
            bounds=([1e-9, 1e-3], [np.inf, 5.0]),
            maxfev=5000,
        )
        perr = _safe_perr(pcov)
        K_fit, n_fit = popt
        K_se, n_se = perr
    except RuntimeError as exc:
        log.warning("Freundlich fit failed: %s", exc)
        raise

    r_pred = _freundlich(c, K_fit, n_fit)
    rss = float(np.sum((r - r_pred) ** 2))
    r2 = _r_squared(r, r_pred)
    rmse = _rmse(r, r_pred)
    aic = _aic(len(c), 2, rss)

    c_grid = _build_fit_grid(c)
    r_grid = _freundlich(c_grid, K_fit, n_fit)

    log.info(
        "Freundlich fit: K=%.4f±%.4f, n=%.4f±%.4f, R²=%.4f",
        K_fit,
        K_se,
        n_fit,
        n_se,
        r2,
    )

    return IsothermResult(
        model="freundlich",
        params={"K": float(K_fit), "n": float(n_fit), "sign": float(sign_r)},
        param_stderrs={"K": float(K_se), "n": float(n_se), "sign": 0.0},
        r_squared=round(r2, 6),
        rmse=round(rmse, 8),
        aic=round(aic, 4),
        n_params=2,
        concentrations_fit=c_grid,
        responses_fit=r_grid,
    )


# ---------------------------------------------------------------------------
# Hill model
# ---------------------------------------------------------------------------


def fit_hill(
    concentrations: np.ndarray,
    responses: np.ndarray,
) -> IsothermResult:
    """Fit a Hill isotherm (cooperative binding) to calibration data.

    Model
    -----
    .. math::

        R(c) = \\frac{R_{\\max} \\cdot c^n}{K_d^n + c^n}

    The Hill model reduces to Langmuir when n = 1.  n > 1 → positive
    cooperativity; n < 1 → anti-cooperative (heterogeneous sites).

    Parameters
    ----------
    concentrations, responses:
        Calibration data, shape ``(n,)``.  Requires ``n ≥ 4``.

    Returns
    -------
    :class:`IsothermResult`
    """
    c = np.asarray(concentrations, dtype=float).ravel()
    r = np.asarray(responses, dtype=float).ravel()

    if len(c) < 4:
        raise ValueError("Hill fit requires at least 4 calibration points.")

    sign_r = -1.0 if np.mean(r) < 0 else 1.0
    R_max_init = float(np.max(np.abs(r)) * 1.5)
    K_d_init = float(np.median(c))

    def _hill(c_arr: np.ndarray, R_max: float, K_d: float, n: float) -> np.ndarray:
        return np.asarray(sign_r * R_max * c_arr**n / (K_d**n + c_arr**n))

    try:
        popt, pcov = curve_fit(
            _hill,
            c,
            r,
            p0=[abs(R_max_init), K_d_init, 1.0],
            bounds=([0, 1e-6, 0.1], [np.inf, np.inf, 10.0]),
            maxfev=10000,
        )
        perr = _safe_perr(pcov)
        R_max_fit, K_d_fit, n_fit = popt
        R_max_se, K_d_se, n_se = perr
    except RuntimeError as exc:
        log.warning("Hill fit failed: %s", exc)
        raise

    r_pred = _hill(c, R_max_fit, K_d_fit, n_fit)
    rss = float(np.sum((r - r_pred) ** 2))
    r2 = _r_squared(r, r_pred)
    rmse = _rmse(r, r_pred)
    aic = _aic(len(c), 3, rss)

    c_grid = _build_fit_grid(c)
    r_grid = _hill(c_grid, R_max_fit, K_d_fit, n_fit)

    log.info(
        "Hill fit: R_max=%.4f±%.4f, K_d=%.4f±%.4f, n=%.4f±%.4f, R²=%.4f",
        R_max_fit,
        R_max_se,
        K_d_fit,
        K_d_se,
        n_fit,
        n_se,
        r2,
    )

    return IsothermResult(
        model="hill",
        params={
            "R_max": float(R_max_fit),
            "K_d": float(K_d_fit),
            "n": float(n_fit),
            "sign": float(sign_r),
        },
        param_stderrs={
            "R_max": float(R_max_se),
            "K_d": float(K_d_se),
            "n": float(n_se),
            "sign": 0.0,
        },
        r_squared=round(r2, 6),
        rmse=round(rmse, 8),
        aic=round(aic, 4),
        n_params=3,
        concentrations_fit=c_grid,
        responses_fit=r_grid,
    )


# ---------------------------------------------------------------------------
# AIC-based model selection
# ---------------------------------------------------------------------------


def select_isotherm(
    concentrations: np.ndarray,
    responses: np.ndarray,
    models: list[str] | None = None,
) -> dict[str, object]:
    """Fit multiple isotherm models and select the best by AIC.

    AIC penalises model complexity, so a 3-parameter Hill model is only
    preferred if it fits *substantially* better than a 2-parameter Langmuir
    (ΔAIC < −2 rule of thumb).

    Parameters
    ----------
    concentrations, responses:
        Calibration data.
    models:
        Subset of ``['linear', 'langmuir', 'freundlich', 'hill']`` to
        evaluate.  Default: all four.

    Returns
    -------
    dict with keys:

    - ``best_model`` — name of the AIC-selected model
    - ``best_result`` — :class:`IsothermResult` for the best model
    - ``all_results`` — dict mapping model name → IsothermResult or error str
    - ``aic_table`` — list of (model, aic, r2, rmse) sorted by AIC
    - ``recommendation`` — interpretation string for reports
    """
    from src.scientific.lod import calculate_sensitivity

    if models is None:
        models = ["linear", "langmuir", "freundlich", "hill"]

    c = np.asarray(concentrations, dtype=float).ravel()
    r = np.asarray(responses, dtype=float).ravel()

    all_results: dict[str, object] = {}
    aic_table: list[tuple[str, float, float, float]] = []

    for model_name in models:
        try:
            if model_name == "linear":
                slope, intercept, r2, _ = calculate_sensitivity(c, r)
                r_pred = slope * c + intercept
                rss = float(np.sum((r - r_pred) ** 2))
                rmse = float(np.sqrt(np.mean((r - r_pred) ** 2)))
                aic = _aic(len(c), 2, rss)
                c_grid = _build_fit_grid(c)
                result = IsothermResult(
                    model="linear",
                    params={"slope": float(slope), "intercept": float(intercept)},
                    param_stderrs={"slope": float("nan"), "intercept": float("nan")},
                    r_squared=round(r2, 6),
                    rmse=round(rmse, 8),
                    aic=round(aic, 4),
                    n_params=2,
                    concentrations_fit=c_grid,
                    responses_fit=slope * c_grid + intercept,
                )
            elif model_name == "langmuir":
                result = fit_langmuir(c, r)
            elif model_name == "freundlich":
                result = fit_freundlich(c, r)
            elif model_name == "hill":
                result = fit_hill(c, r)
            else:
                continue

            all_results[model_name] = result
            aic_table.append((model_name, result.aic, result.r_squared, result.rmse))

        except Exception as exc:
            log.warning("Isotherm model '%s' failed: %s", model_name, exc)
            all_results[model_name] = str(exc)

    if not aic_table:
        raise RuntimeError("All isotherm models failed to converge.")

    aic_table.sort(key=lambda x: x[1])
    best_name, best_aic, best_r2, best_rmse = aic_table[0]
    best_result = all_results[best_name]

    # Build recommendation string
    if len(aic_table) > 1:
        second_name, second_aic = aic_table[1][0], aic_table[1][1]
        delta_aic = second_aic - best_aic
        rec = (
            f"Best model: {best_name} (AIC={best_aic:.2f}, R²={best_r2:.4f}). "
            f"ΔAIC vs {second_name}={delta_aic:.1f}"
        )
    else:
        rec = f"Best (only) model: {best_name} (AIC={best_aic:.2f}, R²={best_r2:.4f})."

    log.info("Isotherm selection: %s", rec)

    return {
        "best_model": best_name,
        "best_result": best_result,
        "all_results": all_results,
        "aic_table": aic_table,
        "recommendation": rec,
    }
