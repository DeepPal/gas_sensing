"""
src.scientific.lod
==================
Limit of Detection (LOD), Limit of Quantification (LOQ), and calibration
sensitivity functions for LSPR gas sensor characterisation.

Standard methodology (IUPAC)
-----------------------------
- **LOD** = 3 σ_noise / sensitivity     (3-sigma criterion)
- **LOQ** = 10 σ_noise / sensitivity    (10-sigma criterion)
- **Sensitivity** = slope of the linear calibration curve (Δλ/ppm or
  ΔIntensity/ppm), obtained by linear regression over the calibration data.

These metrics are required for any publication or regulatory submission
characterising the sensor's detection capability.

Public API
----------
- ``calculate_lod_3sigma``       — LOD from noise std + slope
- ``calculate_loq_10sigma``      — LOQ from noise std + slope
- ``calculate_sensitivity``      — linear regression slope, intercept, R²
- ``sensor_performance_summary`` — complete performance dict for one gas
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def calculate_lod_3sigma(noise_std: float, sensitivity_slope: float) -> float:
    """Limit of Detection using the IUPAC 3σ criterion.

    .. math::

        \\text{LOD} = \\frac{3 \\, \\sigma_{\\text{noise}}}{|m|}

    Parameters
    ----------
    noise_std:
        Standard deviation of the blank / baseline signal (σ_noise).
        Compute from the first few frames of a zero-gas measurement.
    sensitivity_slope:
        Slope *m* of the calibration curve (signal per ppm).
        Use :func:`calculate_sensitivity` to obtain this from calibration data.

    Returns
    -------
    float
        LOD in concentration units (ppm).  Returns ``inf`` if slope is zero.
    """
    # Accept a bare float, a 1-element array, or a (slope, intercept, r2, se) tuple
    _s = sensitivity_slope
    if isinstance(_s, (tuple, list)):
        _s = _s[0]
    slope = float(np.squeeze(_s))
    if slope == 0 or not np.isfinite(slope):
        return float("inf")
    return (3.0 * abs(noise_std)) / abs(slope)


def calculate_loq_10sigma(noise_std: float, sensitivity_slope: float) -> float:
    """Limit of Quantification using the IUPAC 10σ criterion.

    .. math::

        \\text{LOQ} = \\frac{10 \\, \\sigma_{\\text{noise}}}{|m|}

    Below the LOQ you can *detect* the analyte but cannot reliably *quantify*
    its concentration.  LOQ is always ≥ LOD.

    Parameters
    ----------
    noise_std, sensitivity_slope:
        Same as :func:`calculate_lod_3sigma`.

    Returns
    -------
    float
        LOQ in ppm.
    """
    _s = sensitivity_slope
    if isinstance(_s, (tuple, list)):
        _s = _s[0]
    slope = float(np.squeeze(_s))
    if slope == 0 or not np.isfinite(slope):
        return float("inf")
    return (10.0 * abs(noise_std)) / abs(slope)


def calculate_sensitivity(
    concentrations: np.ndarray,
    responses: np.ndarray,
) -> tuple[float, float, float, float]:
    """Calibration sensitivity via ordinary least-squares linear regression.

    Fits the model: ``response = m * concentration + b``

    Parameters
    ----------
    concentrations:
        Known analyte concentrations in ppm, shape ``(n,)``.
    responses:
        Measured sensor responses (e.g. peak wavelength shift in nm, or
        peak intensity), shape ``(n,)``.

    Returns
    -------
    slope : float
        Calibration sensitivity *m* — change in response per ppm.
        Negative for LSPR sensors (Δλ < 0 with increasing concentration).
    intercept : float
        Regression intercept *b*.
    r_squared : float
        Coefficient of determination R² ∈ [0, 1].
    slope_se : float
        Standard error of the slope (1-σ) from OLS covariance matrix.
        Used for uncertainty propagation: σ_LOD = LOD × √((σ_noise/noise)² + (SE_slope/slope)²).
    """
    from scipy.stats import linregress

    c = np.asarray(concentrations, dtype=float).ravel()
    r = np.asarray(responses, dtype=float).ravel()
    if len(c) < 2:
        raise ValueError("At least 2 calibration points are required.")

    slope, intercept, r_value, _p, slope_se = linregress(c, r)
    return float(slope), float(intercept), float(r_value**2), float(slope_se)


def lod_bootstrap_ci(
    concentrations: np.ndarray,
    responses: np.ndarray,
    baseline_noise_std: float | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the LOD (ICH Q2(R1) compliant).

    Resamples (concentration, response) pairs *with replacement* and computes
    LOD for each resample, giving an empirical distribution under sampling
    variability — the preferred method when the LOD distribution is non-normal
    (ICH Q2(R2) Appendix B, 2022).

    Parameters
    ----------
    concentrations, responses:
        Calibration data.
    baseline_noise_std:
        Blank noise σ.  If ``None``, estimated from OLS residuals (acceptable
        for screening; blank measurement preferred for regulatory submissions).
    n_bootstrap:
        Number of resamples (≥ 1 000 for 95 % CI stability).
    confidence:
        Confidence level (default 0.95).
    random_state:
        Seed for reproducibility.

    Returns
    -------
    tuple ``(lod_point, ci_lower, ci_upper)`` in ppm.
    """
    rng = np.random.default_rng(random_state)
    c = np.asarray(concentrations, dtype=float).ravel()
    r = np.asarray(responses, dtype=float).ravel()
    n = len(c)

    slope_full, intercept_full, _, _ = calculate_sensitivity(c, r)
    if baseline_noise_std is None:
        residuals = r - (slope_full * c + intercept_full)
        baseline_noise_std = float(np.std(residuals))

    lod_point = calculate_lod_3sigma(baseline_noise_std, slope_full)

    lod_boot: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        cb, rb = c[idx], r[idx]
        if len(np.unique(cb)) < 2:
            continue
        try:
            sb, ib, _, _ = calculate_sensitivity(cb, rb)
            noise_b = float(np.std(rb - (sb * cb + ib)))
            lod_b = calculate_lod_3sigma(noise_b, sb)
            if np.isfinite(lod_b) and lod_b > 0:
                lod_boot.append(lod_b)
        except Exception:
            continue

    alpha = 1.0 - confidence
    if len(lod_boot) >= 10:
        ci_lower = float(np.percentile(lod_boot, 100 * alpha / 2))
        ci_upper = float(np.percentile(lod_boot, 100 * (1 - alpha / 2)))
    else:
        ci_lower = ci_upper = lod_point

    return lod_point, ci_lower, ci_upper


def robust_sensitivity(
    concentrations: np.ndarray,
    responses: np.ndarray,
    method: str = "huber",
    huber_epsilon: float = 1.35,
    ransac_residual_threshold: float | None = None,
    random_state: int = 42,
) -> dict[str, object]:
    """Outlier-resistant calibration sensitivity estimation.

    OLS is the standard for clean calibration data, but a single missed
    equilibration frame or spike can shift the slope by >20 %.  Robust
    regression down-weights or excludes outliers automatically.

    Methods
    -------
    ``'huber'``
        Huber regression (M-estimator): quadratic loss for residuals
        |r| ≤ ε, linear for |r| > ε.  ``epsilon=1.35`` gives 95 %
        efficiency vs OLS on Gaussian data (default per scikit-learn docs).
        No data points are discarded — all contribute continuously.
    ``'ransac'``
        Random Sample Consensus: fits line to random minimal subset,
        classifies inliers, iterates.  Best when gross outliers may
        represent >30 % of data.  Inlier mask returned for inspection.
    ``'theilsen'``
        Theil-Sen estimator: slope = median of all pairwise slopes.
        Breakdown point 29 % — robust to nearly a third of outliers.
        Used for drift rate estimation elsewhere in this pipeline.

    Parameters
    ----------
    concentrations, responses:
        Calibration data, shape ``(n,)``.
    method:
        One of ``'huber'``, ``'ransac'``, ``'theilsen'``.
    huber_epsilon:
        Huber threshold (in units of σ_residuals).  1.35 is the standard
        choice for 95 % asymptotic efficiency.
    ransac_residual_threshold:
        RANSAC inlier threshold in response units.  If ``None``, auto-set
        to ``1.5 × MAD(OLS residuals)``.
    random_state:
        Seed for RANSAC reproducibility.

    Returns
    -------
    dict with keys:

    - ``slope`` — robust sensitivity estimate
    - ``intercept`` — robust intercept
    - ``r_squared`` — R² computed on all data (not just inliers)
    - ``ols_slope`` — OLS slope for comparison
    - ``outlier_mask`` — boolean array, True = suspected outlier (RANSAC/Theil)
    - ``method`` — method name used
    - ``n_outliers`` — count of flagged outliers
    - ``recommendation`` — interpretation string
    """
    from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor

    c = np.asarray(concentrations, dtype=float).ravel()
    r = np.asarray(responses, dtype=float).ravel()
    n = len(c)

    if n < 3:
        raise ValueError("Robust regression requires at least 3 calibration points.")

    X = c.reshape(-1, 1)

    # OLS baseline for comparison
    from scipy.stats import linregress

    ols_slope, ols_intercept, ols_r, _, _ = linregress(c, r)
    ols_r2 = float(ols_r**2)

    outlier_mask: np.ndarray = np.zeros(n, dtype=bool)

    if method == "huber":
        reg = HuberRegressor(epsilon=huber_epsilon, max_iter=500)
        reg.fit(X, r)
        slope = float(reg.coef_[0])
        intercept = float(reg.intercept_)
        # Huber outlier mask: samples where |residual| > epsilon × scale
        resid = np.abs(r - (slope * c + intercept))
        scale = float(reg.scale_) if hasattr(reg, "scale_") else float(np.std(resid))
        outlier_mask = resid > huber_epsilon * scale

    elif method == "ransac":
        if ransac_residual_threshold is None:
            ols_resid = np.abs(r - (ols_slope * c + ols_intercept))
            mad = float(np.median(ols_resid))
            ransac_residual_threshold = max(mad * 1.5, 1e-6)
        from sklearn.linear_model import LinearRegression

        base_estimator = LinearRegression()
        reg = RANSACRegressor(
            estimator=base_estimator,
            residual_threshold=ransac_residual_threshold,
            random_state=random_state,
            max_trials=500,
        )
        reg.fit(X, r)
        slope = float(reg.estimator_.coef_[0])
        intercept = float(reg.estimator_.intercept_)
        outlier_mask = ~reg.inlier_mask_

    elif method == "theilsen":
        reg = TheilSenRegressor(random_state=random_state, max_iter=300)
        reg.fit(X, r)
        slope = float(reg.coef_[0])
        intercept = float(reg.intercept_)
        # Flag points with |residual| > 2.5 × MAD as outliers
        resid = np.abs(r - (slope * c + intercept))
        mad = float(np.median(resid))
        outlier_mask = resid > 2.5 * max(mad, 1e-12)

    else:
        raise ValueError(
            f"Unknown robust method: {method!r}. Choose from: huber, ransac, theilsen."
        )

    # R² computed on all points (not just inliers)
    r_pred = slope * c + intercept
    ss_res: float = float(np.sum((r - r_pred) ** 2))
    ss_tot: float = float(np.sum((r - r.mean()) ** 2))
    r2_all = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    n_out = int(outlier_mask.sum())
    slope_shift_pct = abs(slope - ols_slope) / max(abs(ols_slope), 1e-12) * 100.0

    if n_out == 0:
        rec = f"{method.capitalize()}: no outliers detected. OLS and {method} agree."
    else:
        rec = (
            f"{method.capitalize()}: {n_out}/{n} points flagged as outliers. "
            f"Slope shift vs OLS: {slope_shift_pct:.1f} %. "
            "Inspect flagged points before finalising calibration."
        )
        if slope_shift_pct > 10.0:
            rec += " WARNING: >10% slope shift — outliers are influential."

    log.info(
        "Robust sensitivity [%s]: slope=%.4f (OLS=%.4f), outliers=%d/%d",
        method,
        slope,
        ols_slope,
        n_out,
        n,
    )

    return {
        "slope": round(slope, 8),
        "intercept": round(intercept, 8),
        "r_squared": round(r2_all, 6),
        "ols_slope": round(float(ols_slope), 8),
        "ols_intercept": round(float(ols_intercept), 8),
        "ols_r_squared": round(ols_r2, 6),
        "slope_shift_pct": round(slope_shift_pct, 2),
        "outlier_mask": outlier_mask,
        "n_outliers": n_out,
        "method": method,
        "recommendation": rec,
    }


def mandel_linearity_test(
    concentrations: np.ndarray,
    responses: np.ndarray,
) -> dict[str, object]:
    """Mandel's fitting test (lack-of-fit F-test) for linearity assessment.

    Compares a linear calibration model against a quadratic (second-order)
    model using an F-test.  If the quadratic fit is significantly better
    (p < 0.05), linearity is rejected and the calibration range should be
    restricted or a nonlinear model used.

    This test is required by ICH Q2(R1) §4.2 (Linearity) and ISO 11843-3.

    Reference
    ---------
    Mandel, J. (1964). The Statistical Analysis of Experimental Data.
    Interscience, New York.  ICH Q2(R1) §4.2, 2005.

    Parameters
    ----------
    concentrations, responses:
        Calibration data, shape ``(n,)``.  Requires ``n ≥ 4``.

    Returns
    -------
    dict with keys:

    - ``f_statistic`` — observed F value (larger → more nonlinear)
    - ``p_value`` — probability under H₀ (linearity); p < 0.05 → reject
    - ``is_linear`` — ``True`` if linearity is not rejected at p = 0.05
    - ``r2_linear`` — R² of linear fit
    - ``r2_quadratic`` — R² of quadratic fit
    - ``delta_r2`` — improvement in R² from adding quadratic term
    - ``rss_linear``, ``rss_quadratic`` — residual sum of squares
    - ``recommendation`` — human-readable string

    Examples
    --------
    >>> result = mandel_linearity_test(concs, responses)
    >>> if not result['is_linear']:
    ...     print(f"Linearity rejected: F={result['f_statistic']:.2f}, p={result['p_value']:.4f}")
    """
    from scipy.stats import f as f_dist

    c = np.asarray(concentrations, dtype=float).ravel()
    r = np.asarray(responses, dtype=float).ravel()
    n = len(c)

    if n < 4:
        raise ValueError(f"Mandel's test requires at least 4 calibration points, got {n}.")

    # Linear fit (degree 1)
    coef1 = np.polyfit(c, r, 1)
    r_fit1 = np.polyval(coef1, c)
    rss1 = float(np.sum((r - r_fit1) ** 2))

    # Quadratic fit (degree 2)
    coef2 = np.polyfit(c, r, 2)
    r_fit2 = np.polyval(coef2, c)
    rss2 = float(np.sum((r - r_fit2) ** 2))

    # Coefficient of determination
    ss_tot = float(np.sum((r - r.mean()) ** 2))
    r2_lin = 1.0 - rss1 / ss_tot if ss_tot > 0 else 0.0
    r2_quad = 1.0 - rss2 / ss_tot if ss_tot > 0 else 0.0

    # F-statistic: (reduction in RSS per df gained) / (residual variance of quadratic)
    # df_linear = n - 2, df_quadratic = n - 3  →  df_extra = 1
    df_extra = 1
    df_quad = n - 3

    if df_quad <= 0 or rss2 <= 0:
        return {
            "f_statistic": float("nan"),
            "p_value": float("nan"),
            "is_linear": True,
            "r2_linear": round(r2_lin, 6),
            "r2_quadratic": round(r2_quad, 6),
            "delta_r2": round(r2_quad - r2_lin, 6),
            "rss_linear": round(rss1, 8),
            "rss_quadratic": round(rss2, 8),
            "recommendation": "Insufficient data for F-test (n < 5); assume linear.",
        }

    # Clamp to 0: floating-point can make rss2 slightly larger than rss1
    # for nearly identical fits, yielding a trivially negative F-statistic.
    f_stat = max(0.0, ((rss1 - rss2) / df_extra) / (rss2 / df_quad))
    p_val = float(1.0 - f_dist.cdf(f_stat, df_extra, df_quad))
    is_linear = p_val >= 0.05

    if is_linear:
        rec = f"Linearity confirmed (F={f_stat:.2f}, p={p_val:.4f} ≥ 0.05)."
    else:
        rec = (
            f"Linearity REJECTED (F={f_stat:.2f}, p={p_val:.4f} < 0.05). "
            "Consider restricting concentration range or using Langmuir/Freundlich calibration."
        )

    return {
        "f_statistic": round(float(f_stat), 4),
        "p_value": round(p_val, 6),
        "is_linear": bool(is_linear),
        "r2_linear": round(r2_lin, 6),
        "r2_quadratic": round(r2_quad, 6),
        "delta_r2": round(r2_quad - r2_lin, 6),
        "rss_linear": round(rss1, 8),
        "rss_quadratic": round(rss2, 8),
        "recommendation": rec,
    }


def sensor_performance_summary(
    concentrations: np.ndarray,
    responses: np.ndarray,
    baseline_noise_std: float | None = None,
    gas_name: str = "unknown",
) -> dict[str, object]:
    """Full sensor performance characterisation for one gas analyte.

    Computes sensitivity, R², LOD, LOQ, and (if provided) a signal-to-noise
    estimate at each calibration point.

    Parameters
    ----------
    concentrations:
        Calibration concentrations in ppm.
    responses:
        Corresponding sensor responses (Δλ in nm, or peak intensity).
    baseline_noise_std:
        Noise standard deviation of the blank measurement.  If None,
        estimated as the std of the residuals from the linear fit.
    gas_name:
        Gas analyte label (used only for the returned dict key).

    Returns
    -------
    dict
        Keys: ``gas``, ``sensitivity``, ``intercept``, ``r_squared``,
        ``lod_ppm``, ``loq_ppm``, ``noise_std``, ``n_calibration_points``.

    Example
    -------
    >>> from src.scientific.lod import sensor_performance_summary
    >>> import numpy as np
    >>> concs = np.array([0.5, 1.0, 2.0, 5.0])
    >>> responses = np.array([-1.1, -2.0, -4.1, -10.0])
    >>> summary = sensor_performance_summary(concs, responses, gas_name="Ethanol")
    >>> print(f"LOD = {summary['lod_ppm']:.3f} ppm, R² = {summary['r_squared']:.4f}")
    """
    c = np.asarray(concentrations, dtype=float).ravel()
    r = np.asarray(responses, dtype=float).ravel()

    slope, intercept, r2, slope_se = calculate_sensitivity(c, r)

    if baseline_noise_std is None:
        # Estimate noise from linear fit residuals (ICH Q2(R1) §5.1)
        residuals = r - (slope * c + intercept)
        baseline_noise_std = float(np.std(residuals))

    lod = calculate_lod_3sigma(baseline_noise_std, slope)
    loq = calculate_loq_10sigma(baseline_noise_std, slope)

    # Bootstrap 95 % CI for LOD and LOQ (ICH Q2(R1) Appendix)
    lod_point, lod_ci_lo, lod_ci_hi = lod_bootstrap_ci(c, r, baseline_noise_std)
    loq_ci_lo = lod_ci_lo * (10.0 / 3.0)  # LOQ = 10σ/S, LOD = 3σ/S → ratio = 10/3
    loq_ci_hi = lod_ci_hi * (10.0 / 3.0)

    # Propagated LOD uncertainty from slope SE only (Taylor first-order):
    #   σ_LOD = LOD × (SE_slope / |slope|)
    # The noise uncertainty term is omitted here because baseline_noise_std is
    # treated as a fixed measured value (not itself uncertain). If you have
    # replicate blank measurements, add that term in quadrature manually.
    if abs(slope) > 1e-12 and np.isfinite(lod):
        lod_propagated_std = float(lod * slope_se / abs(slope))
    else:
        lod_propagated_std = float("nan")

    # LOB (Limit of Blank): highest signal expected from a blank sample
    # = μ_blank + 1.645·σ_blank (IUPAC 2012, one-sided 95th percentile)
    # In concentration units: (|μ_blank| + 1.645·σ_blank) / |slope|
    # Since blank mean ≈ 0 for a well-zeroed sensor, this simplifies to
    # 1.645·σ_blank / |slope|, but we keep the general form.
    blank_mean_signal = 0.0  # LSPR shifts are reference-subtracted → blank mean = 0
    if abs(slope) > 1e-12:
        lob = max(
            (abs(blank_mean_signal) + 1.645 * baseline_noise_std) / abs(slope), 1e-7
        )
    else:
        lob = float("inf")

    # Mandel's linearity test on the full calibration range
    linearity_result: dict | None = None
    lol_ppm: float | None = None
    if len(c) >= 5:
        # LOL = highest conc where Mandel p ≥ 0.05 (progressive truncation)
        sort_idx = np.argsort(c)
        sc = c[sort_idx]
        sr = r[sort_idx]
        for n_keep in range(len(sc), 3, -1):
            try:
                lin = mandel_linearity_test(sc[:n_keep], sr[:n_keep])
                if lin.get("is_linear", False):
                    lol_ppm = float(sc[n_keep - 1])
                    linearity_result = lin
                    break
            except Exception:
                continue
    elif len(c) >= 4:
        try:
            linearity_result = mandel_linearity_test(c, r)
            if linearity_result.get("is_linear", False):
                lol_ppm = float(np.max(c))
        except Exception:
            pass

    summary = {
        # ICH Q2(R1) mandatory fields
        "gas": gas_name,
        "sensitivity": round(slope, 6),
        "sensitivity_se": round(slope_se, 8),  # slope standard error
        "intercept": round(intercept, 6),
        "r_squared": round(r2, 6),
        "noise_std": round(baseline_noise_std, 8),
        "n_calibration_points": int(len(c)),
        # LOB (IUPAC 2012 mandatory for publication)
        "lob_ppm": round(lob, 6) if np.isfinite(lob) else None,
        # LOD
        "lod_ppm": round(lod, 6) if np.isfinite(lod) else None,
        "lod_ppm_ci_lower": round(lod_ci_lo, 6) if np.isfinite(lod_ci_lo) else None,
        "lod_ppm_ci_upper": round(lod_ci_hi, 6) if np.isfinite(lod_ci_hi) else None,
        "lod_ppm_propagated_std": round(lod_propagated_std, 8)
        if np.isfinite(lod_propagated_std)
        else None,
        # LOQ
        "loq_ppm": round(loq, 6) if np.isfinite(loq) else None,
        "loq_ppm_ci_lower": round(loq_ci_lo, 6) if np.isfinite(loq_ci_lo) else None,
        "loq_ppm_ci_upper": round(loq_ci_hi, 6) if np.isfinite(loq_ci_hi) else None,
        # LOL (Limit of Linearity, ICH Q2(R1) §4.2)
        "lol_ppm": round(lol_ppm, 6) if lol_ppm is not None else None,
        # Mandel's linearity test result (full range)
        "mandel_linearity": linearity_result,
        # Methodology tag for audit trail
        "lod_method": "ICH Q2(R1) 3.3σ/S with bootstrap 95% CI (n=1000)",
        "loq_method": "ICH Q2(R1) 10σ/S with bootstrap 95% CI (n=1000)",
        "lob_method": "IUPAC 2012: (|μ_blank| + 1.645·σ_blank) / |S|",
        "lol_method": "ICH Q2(R1) §4.2 Mandel F-test progressive truncation",
    }

    log.info(
        "Sensor performance [%s]: S=%.4f/ppm (SE=%.4f), R²=%.4f, "
        "LOD=%.4f [%.4f–%.4f] ppm, LOQ=%.4f ppm",
        gas_name,
        slope,
        slope_se,
        r2,
        lod if np.isfinite(lod) else float("nan"),
        lod_ci_lo if np.isfinite(lod_ci_lo) else float("nan"),
        lod_ci_hi if np.isfinite(lod_ci_hi) else float("nan"),
        loq if np.isfinite(loq) else float("nan"),
    )

    return summary
