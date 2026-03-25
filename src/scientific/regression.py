"""
src.scientific.regression
==========================
Robust linear regression utilities for calibration curve fitting.

Background
----------
Simple OLS (ordinary least squares) is fragile when a calibration dataset
contains outlier concentration points (e.g., a gas leak during a reference
run, or a saturated sensor response at the highest concentration).  Three
robust alternatives are provided:

* **Weighted OLS** — reduces the influence of low-reliability points via
  explicit sample weights (e.g., derived from replicate standard deviations).

* **Theil-Sen** — slope is the median of all pairwise slopes; breakdown
  point of 29 %, insensitive to a single bad data point.
  Confidence interval via Sen's rank-based method (``scipy.stats.theilslopes``).

* **RANSAC** — random-sample consensus; explicitly separates inliers from
  outliers and fits OLS only on the inlier set.

All three return a uniform ``dict[str, float]`` so callers can pick the
best model by R² / RMSE without special-casing.

All functions now populate ``slope_stderr``, ``slope_ci_low``, and
``slope_ci_high`` (previously always NaN), enabling full uncertainty
propagation into LOD/LOQ and predicted concentration bands.

Public API
----------
- ``weighted_linear(x, y, w)``
- ``theil_sen(x, y)``
- ``ransac(x, y)``
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import r2_score


def weighted_linear(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> dict[str, object] | None:
    """Weighted ordinary least squares regression.

    Args:
        x: Predictor array (concentration, 1-D).
        y: Response array (signal, 1-D).
        w: Non-negative sample weights (same length as ``x``).
            Points with weight ≤ 0 are excluded.

    Returns:
        Dict with keys ``model``, ``slope``, ``intercept``, ``r2``, ``rmse``,
        ``slope_stderr``, ``slope_ci_low``, ``slope_ci_high``,
        ``preds``, ``residuals``.
        Returns ``None`` if fewer than 3 valid weighted points exist.

    Notes
    -----
    Slope standard error uses the WLS formula for one predictor::

        SE(β₁) = √(σ² / Σwᵢ(xᵢ − x̄_w)²)

    where σ² = Σwᵢ(yᵢ − ŷᵢ)² / (n − 2) is the weighted MSE.
    The 95 % CI uses the t-distribution with n − 2 degrees of freedom.
    RMSE is computed on the fitted (non-zero-weight) subset only.
    """
    if w.size != x.size:
        return None
    mask = np.isfinite(w) & (w > 0)
    n_fit = int(mask.sum())
    if n_fit < 3:
        return None
    xw, yw, ww = x[mask], y[mask], w[mask]
    ww = ww / float(np.nanmax(ww))  # normalise to [0, 1]

    lr = LinearRegression()
    lr.fit(xw.reshape(-1, 1), yw, sample_weight=ww)

    preds_fit = np.asarray(lr.predict(xw.reshape(-1, 1))).ravel()
    residuals_fit = yw - preds_fit
    preds_all = np.asarray(lr.predict(x.reshape(-1, 1)))
    residuals_all = y - preds_all

    r2_val = float(r2_score(yw, preds_fit)) if n_fit > 1 else float("nan")
    if np.isnan(r2_val):
        return None

    # RMSE on fitted subset only (excluded points shouldn't inflate error)
    rmse = float(np.sqrt(float(np.nanmean(residuals_fit ** 2))))

    # Slope standard error (WLS analytical formula, 1-predictor)
    xbar_w = float(np.average(xw, weights=ww))
    ss_xx = float(np.sum(ww * (xw - xbar_w) ** 2))
    sigma2 = float(np.sum(ww * residuals_fit ** 2) / max(n_fit - 2, 1))

    if ss_xx > 1e-15 and sigma2 >= 0.0:
        slope_se = float(np.sqrt(sigma2 / ss_xx))
        t_crit = float(stats.t.ppf(0.975, df=max(n_fit - 2, 1)))
        slope_ci_low = float(lr.coef_[0]) - t_crit * slope_se
        slope_ci_high = float(lr.coef_[0]) + t_crit * slope_se
    else:
        slope_se = float("nan")
        slope_ci_low = float("nan")
        slope_ci_high = float("nan")

    return {
        "model": "weighted_ols",
        "slope": float(lr.coef_[0]),
        "intercept": float(lr.intercept_),
        "r2": r2_val,
        "rmse": rmse,
        "slope_stderr": slope_se,
        "slope_ci_low": slope_ci_low,
        "slope_ci_high": slope_ci_high,
        "preds": preds_all,
        "residuals": residuals_all,
    }


def theil_sen(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, object] | None:
    """Theil-Sen robust regression with analytical confidence intervals.

    Uses the Sen (1968) rank-based method via ``scipy.stats.theilslopes``,
    which provides exact 95 % confidence bounds on the slope without
    bootstrapping.  Breakdown point ~29 % — robust against a minority of
    outlier data points.

    Args:
        x: Predictor array (1-D).
        y: Response array (1-D, same length as ``x``).

    Returns:
        Dict with ``slope``, ``intercept``, ``r2``, ``rmse``,
        ``slope_stderr`` (approximated as half-CI-width / 1.96),
        ``slope_ci_low``, ``slope_ci_high``.
        Returns ``None`` if ``x`` has fewer than 2 elements or fitting fails.
    """
    if x.size < 2:
        return None
    try:
        res = stats.theilslopes(y, x, alpha=0.05)
        slope = float(res.slope)
        intercept = float(res.intercept)

        preds = slope * x + intercept
        residuals = y - preds
        r2_val = float(r2_score(y, preds))
        rmse = float(np.sqrt(float(np.mean(residuals ** 2))))

        ci_low = float(res.low_slope)
        ci_high = float(res.high_slope)
        # Approximate SE from half-CI-width / z_{0.975}
        slope_se = float((ci_high - ci_low) / (2.0 * 1.96)) if np.isfinite(ci_high - ci_low) else float("nan")

        return {
            "model": "theil_sen",
            "slope": slope,
            "intercept": intercept,
            "r2": r2_val,
            "rmse": rmse,
            "slope_stderr": slope_se,
            "slope_ci_low": ci_low,
            "slope_ci_high": ci_high,
            "preds": preds,
            "residuals": residuals,
        }
    except Exception:
        return None


def ransac(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, object] | None:
    """RANSAC (random-sample consensus) robust regression.

    Explicitly separates inliers from outliers; fits OLS only on inliers.
    Slope standard error and confidence intervals are computed from an
    OLS fit on the inlier subset via ``scipy.stats.linregress``.
    RMSE is computed on inliers only (outliers by definition are excluded
    from the model quality estimate).

    Args:
        x: Predictor array (1-D).
        y: Response array (1-D, same length as ``x``).

    Returns:
        Dict with ``slope``, ``intercept``, ``r2``, ``rmse``,
        ``slope_stderr``, ``slope_ci_low``, ``slope_ci_high``,
        ``n_inliers``, ``inlier_mask``.
        Returns ``None`` if ``x`` has fewer than 3 elements or fitting fails.
    """
    if x.size < 3:
        return None
    try:
        model = RANSACRegressor(estimator=LinearRegression(), random_state=0)
        model.fit(x.reshape(-1, 1), y)

        slope = (
            float(model.estimator_.coef_[0])
            if hasattr(model.estimator_, "coef_")
            else float("nan")
        )
        intercept = (
            float(model.estimator_.intercept_)
            if hasattr(model.estimator_, "intercept_")
            else float("nan")
        )
        preds_all = np.asarray(model.predict(x.reshape(-1, 1)))
        residuals_all = y - preds_all

        inlier_mask: np.ndarray = model.inlier_mask_  # type: ignore[assignment]
        x_in, y_in = x[inlier_mask], y[inlier_mask]
        n_inliers = int(inlier_mask.sum())

        # R² on all data (RANSAC's model should still generalise)
        r2_val = float(r2_score(y, preds_all))

        # RMSE and SE on inliers only
        if n_inliers >= 3:
            preds_in = slope * x_in + intercept
            rmse = float(np.sqrt(float(np.mean((y_in - preds_in) ** 2))))
            lr_res = stats.linregress(x_in, y_in)
            slope_se = float(lr_res.stderr) if lr_res.stderr is not None else float("nan")
            t_crit = float(stats.t.ppf(0.975, df=max(n_inliers - 2, 1)))
            slope_ci_low = slope - t_crit * slope_se
            slope_ci_high = slope + t_crit * slope_se
        else:
            preds_in_all = slope * x + intercept
            rmse = float(np.sqrt(float(np.mean((y - preds_in_all) ** 2))))
            slope_se = float("nan")
            slope_ci_low = float("nan")
            slope_ci_high = float("nan")

        return {
            "model": "ransac",
            "slope": slope,
            "intercept": intercept,
            "r2": r2_val,
            "rmse": rmse,
            "slope_stderr": slope_se,
            "slope_ci_low": slope_ci_low,
            "slope_ci_high": slope_ci_high,
            "n_inliers": n_inliers,
            "inlier_mask": inlier_mask,
            "preds": preds_all,
            "residuals": residuals_all,
        }
    except Exception:
        return None
