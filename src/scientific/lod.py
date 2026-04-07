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
    blank_measurements: np.ndarray | None = None,
    fix_noise_std: bool = False,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the LOD (ICH Q2(R1) compliant).

    Resamples (concentration, response) pairs *with replacement* and computes
    LOD for each resample, giving an empirical distribution under sampling
    variability — the preferred method when the LOD distribution is non-normal
    (ICH Q2(R2) Appendix B, 2022).

    When ``blank_measurements`` is supplied the blank noise is held fixed at
    the measured σ_blank during bootstrap (only the calibration slope varies).
    This is the correct procedure: blank noise is a *separate* measurement and
    should not be re-estimated from calibration residuals on each resample.
    When ``blank_measurements`` is None the noise is re-estimated from OLS
    residuals on each resample — acceptable for screening but will produce
    artificially tight CIs.

    Parameters
    ----------
    concentrations, responses:
        Calibration data.
    baseline_noise_std:
        Blank noise σ.  If ``None`` and ``blank_measurements`` is also None,
        estimated from OLS residuals (acceptable for screening; blank
        measurement preferred for regulatory submissions).
    n_bootstrap:
        Number of resamples (≥ 1 000 for 95 % CI stability).
    confidence:
        Confidence level (default 0.95).
    random_state:
        Seed for reproducibility.
    blank_measurements:
        Raw blank-signal array (same units as ``responses``).  When provided,
        σ_blank is computed from this array and held fixed during bootstrap —
        making the CI capture only slope uncertainty, as intended.
    fix_noise_std:
        When ``True``, ``baseline_noise_std`` is treated as a known constant
        and held fixed during every bootstrap iteration (only the calibration
        slope varies).  Use this when ``baseline_noise_std`` comes from an
        independent experiment (e.g., Allan deviation σ_min computed from a
        separate baseline time series).  Has no effect when
        ``blank_measurements`` is also provided (blank σ always takes precedence).

    Returns
    -------
    tuple ``(lod_point, ci_lower, ci_upper)`` in ppm.
    """
    rng = np.random.default_rng(random_state)
    c = np.asarray(concentrations, dtype=float).ravel()
    r = np.asarray(responses, dtype=float).ravel()
    n = len(c)

    slope_full, intercept_full, _, _ = calculate_sensitivity(c, r)

    # Determine the fixed blank noise to use throughout bootstrap.
    # Priority: blank_measurements σ > fix_noise_std > re-estimate each resample.
    if blank_measurements is not None and len(blank_measurements) >= 2:
        _bm = np.asarray(blank_measurements, dtype=float).ravel()
        fixed_sigma_blank: float | None = float(np.std(_bm, ddof=1))
    elif fix_noise_std and baseline_noise_std is not None:
        # Caller guarantees this σ came from an independent measurement —
        # hold it fixed so CI captures only calibration slope uncertainty.
        fixed_sigma_blank = float(baseline_noise_std)
    else:
        fixed_sigma_blank = None  # will re-estimate from residuals each resample

    if baseline_noise_std is None:
        if fixed_sigma_blank is not None:
            baseline_noise_std = fixed_sigma_blank
        else:
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
            if fixed_sigma_blank is not None:
                # Hold blank noise fixed: CI captures only slope uncertainty
                # (blank/Allan noise is a separate, independent measurement)
                noise_b = fixed_sigma_blank
            else:
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
    baseline_time_series: np.ndarray | None = None,
    dt_s: float = 0.05,
    blank_measurements: np.ndarray | None = None,
) -> dict[str, object]:
    """Full sensor performance characterisation for one gas analyte.

    Computes sensitivity, R², LOD, LOQ with bootstrap CI, LOB from
    dedicated blank measurements, NEC, residual diagnostics
    (Durbin-Watson / Shapiro-Wilk / Breusch-Pagan), and optionally
    Allan deviation noise characterisation when a baseline time series
    is provided.

    Parameters
    ----------
    concentrations:
        Calibration concentrations in ppm.
    responses:
        Corresponding sensor responses (Δλ in nm, or peak intensity).
    baseline_noise_std:
        Noise standard deviation of the blank measurement.  If None,
        estimated as the std of the residuals from the linear fit.
        If ``baseline_time_series`` is also provided, the Allan deviation
        σ_min is used instead (more physically meaningful).
    gas_name:
        Gas analyte label (used only for the returned dict key).
    baseline_time_series:
        Optional 1-D array of blank / zero-gas measurements used for
        Allan deviation noise analysis.  If provided, ``sigma_min``
        (optimal-τ noise floor) replaces the OLS residual σ for LOD.
    dt_s:
        Sample interval for ``baseline_time_series`` in seconds.
    blank_measurements:
        Optional 1-D array of replicate blank / carrier-gas responses
        (Δλ at zero analyte concentration).  When provided, LOB is
        computed from the actual blank distribution per IUPAC 2012:
        LOB = μ_blank + 1.645 · σ_blank  (one-sided 95th percentile).
        NEC (Noise Equivalent Concentration) = σ_blank / |sensitivity|.
        If None, blank_mean = 0 is assumed (reference-subtracted sensor).

    Returns
    -------
    dict
        Keys: ``gas``, ``sensitivity``, ``intercept``, ``r_squared``,
        ``lod_ppm``, ``loq_ppm``, ``lob_ppm``, ``nec_ppm``,
        ``noise_std``, ``n_calibration_points``,
        ``residual_diagnostics``, and optionally ``allan_deviation``.

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

    # ── Determine authoritative σ_blank ──────────────────────────────────────
    # Correct σ_blank hierarchy (IUPAC 2012):
    #   1. Explicit blank_measurements array   → σ_blank = std(blanks)   [preferred]
    #   2. Explicit baseline_noise_std argument → use as-is              [user override]
    #   3. Fallback → std of OLS calibration residuals                    [screening only]
    #
    # CRITICAL: OLS residuals measure how well the linear model fits the
    # calibration data — they are NOT a measurement of the sensor noise floor
    # at zero analyte concentration.  Using OLS σ for LOD systematically
    # under- or over-estimates the true detection limit depending on
    # calibration data density.  Blank measurements are always preferred.
    _user_provided_noise = baseline_noise_std  # remember if user explicitly supplied
    if blank_measurements is not None and len(blank_measurements) >= 2:
        _bm = np.asarray(blank_measurements, dtype=float).ravel()
        blank_mean_signal = float(np.mean(_bm))
        blank_std_signal = float(np.std(_bm, ddof=1))
        _lob_method = "blank_measurements"
        if baseline_noise_std is None:
            # Use blank σ as authoritative noise estimate for LOD/LOQ/NEC
            baseline_noise_std = blank_std_signal
    else:
        # Reference-subtracted sensor: blank mean ≈ 0; σ_blank ≈ OLS residuals
        blank_mean_signal = 0.0
        _lob_method = "residual_std"
        if baseline_noise_std is None:
            residuals = r - (slope * c + intercept)
            baseline_noise_std = float(np.std(residuals))
        blank_std_signal = baseline_noise_std

    lod = calculate_lod_3sigma(baseline_noise_std, slope)
    loq = calculate_loq_10sigma(baseline_noise_std, slope)

    # Bootstrap 95 % CI for LOD and LOQ (ICH Q2(R1) Appendix).
    # When blank_measurements are provided, σ_blank is held fixed so the CI
    # captures only calibration slope uncertainty (not noise uncertainty) —
    # the correct treatment when blanks are an independent experiment.
    # When no blanks: fix_noise_std=False → re-estimate σ from each resample.
    lod_point, lod_ci_lo, lod_ci_hi = lod_bootstrap_ci(
        c, r, baseline_noise_std,
        blank_measurements=blank_measurements,
        fix_noise_std=(_user_provided_noise is not None),
    )
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
    if abs(slope) > 1e-12:
        lob = max(
            (abs(blank_mean_signal) + 1.645 * blank_std_signal) / abs(slope), 1e-7
        )
    else:
        lob = float("inf")

    # NEC (Noise Equivalent Concentration) = σ_blank / |sensitivity|
    # The theoretical minimum detectable concentration (LOD = 3 × NEC by definition)
    nec_ppm: float | None = None
    nec_ci_lower: float | None = None
    nec_ci_upper: float | None = None
    if abs(slope) > 1e-12 and blank_std_signal > 0:
        nec_ppm = float(blank_std_signal / abs(slope))
        # Bootstrap CI on NEC (when actual blank measurements are available)
        # Resample blank measurements to capture σ_blank uncertainty; bootstrap
        # the calibration slope to capture slope uncertainty; propagate both.
        if blank_measurements is not None and len(blank_measurements) >= 4:
            rng_nec = np.random.default_rng(42)
            _bm_arr = np.asarray(blank_measurements, dtype=float).ravel()
            n_bm = len(_bm_arr)
            nec_boot: list[float] = []
            for _ in range(1000):
                # Resample blank measurements
                bm_b = _bm_arr[rng_nec.integers(0, n_bm, size=n_bm)]
                sigma_b = float(np.std(bm_b, ddof=1))
                # Resample calibration data
                idx = rng_nec.integers(0, len(c), size=len(c))
                cb, rb = c[idx], r[idx]
                if len(np.unique(cb)) < 2:
                    continue
                try:
                    sb, _, _, _ = calculate_sensitivity(cb, rb)
                    if abs(sb) > 1e-12 and sigma_b > 0:
                        nec_boot.append(sigma_b / abs(sb))
                except Exception:
                    continue
            if len(nec_boot) >= 10:
                nec_ci_lower = float(np.percentile(nec_boot, 2.5))
                nec_ci_upper = float(np.percentile(nec_boot, 97.5))

    # Mandel's linearity test on the full calibration range.
    # IMPORTANT: With n ≤ 5 calibration points the F-test has very low statistical
    # power — it will almost always accept linearity not because the data IS linear
    # but because there are too few points to detect curvature. LOL is unreliable
    # and is flagged accordingly.
    linearity_result: dict | None = None
    lol_ppm: float | None = None
    _mandel_low_power = False
    if len(c) >= 5:
        _mandel_low_power = len(c) <= 5  # n=5 is borderline; flag it
        # LOL = highest conc where Mandel p ≥ 0.05 (progressive truncation)
        sort_idx = np.argsort(c)
        sc = c[sort_idx]
        sr = r[sort_idx]
        for n_keep in range(len(sc), 3, -1):
            try:
                lin = mandel_linearity_test(sc[:n_keep], sr[:n_keep])
                if lin.get("is_linear", False):
                    lol_ppm = float(sc[n_keep - 1]) if not _mandel_low_power else None
                    linearity_result = lin
                    if _mandel_low_power:
                        linearity_result["low_power_warning"] = (
                            f"n={len(c)} calibration points: F-test has low power. "
                            "LOL is not reported. Add ≥6 points (ideally ≥8) for reliable LOL."
                        )
                    break
            except Exception:
                continue
    elif len(c) >= 4:
        _mandel_low_power = True
        try:
            linearity_result = mandel_linearity_test(c, r)
            linearity_result["low_power_warning"] = (
                f"n={len(c)} calibration points: F-test has low power (n ≤ 5). "
                "LOL is not reported. Add ≥6 points for reliable linearity assessment."
            )
            # Do NOT set lol_ppm — unreliable at n=4
        except Exception:
            pass

    # ── Allan deviation: if a baseline time series is provided, use sigma_min
    # as the noise estimate — it is more conservative and physically meaningful
    # than the OLS-residual estimate because it captures the actual sensor noise
    # floor at the optimal averaging time.
    allan_result_dict: dict | None = None
    if baseline_time_series is not None:
        try:
            from src.scientific.allan_deviation import allan_deviation as _adev
            bts = np.asarray(baseline_time_series, dtype=float).ravel()
            if len(bts) >= 10:
                adev_result = _adev(bts, dt=dt_s)
                allan_result_dict = {
                    "tau_opt_s": round(float(adev_result.tau_opt), 4),
                    "sigma_min": round(float(adev_result.sigma_min), 8),
                    "noise_type": str(adev_result.noise_type),
                    "white_noise_coeff": round(float(adev_result.white_noise_coeff), 8)
                    if not np.isnan(adev_result.white_noise_coeff) else None,
                    "n_samples": int(adev_result.n_samples),
                }
                # Use sigma_min as a more rigorous noise estimate for LOD
                sigma_for_lod = float(adev_result.sigma_min)
                lod = calculate_lod_3sigma(sigma_for_lod, slope)
                loq = calculate_loq_10sigma(sigma_for_lod, slope)
                # Allan σ_min comes from an independent baseline time series —
                # hold it fixed in bootstrap so CI captures only slope
                # uncertainty (not Allan noise uncertainty).
                lod_point, lod_ci_lo, lod_ci_hi = lod_bootstrap_ci(
                    c, r, sigma_for_lod, fix_noise_std=True
                )
                loq_ci_lo = lod_ci_lo * (10.0 / 3.0)
                loq_ci_hi = lod_ci_hi * (10.0 / 3.0)
                if abs(slope) > 1e-12 and np.isfinite(lod):
                    lod_propagated_std = float(lod * slope_se / abs(slope))
                if abs(slope) > 1e-12:
                    lob = max(
                        (1.645 * sigma_for_lod) / abs(slope), 1e-7
                    )
                log.info(
                    "Allan deviation [%s]: σ_min=%.4g at τ_opt=%.3g s "
                    "(noise_type=%s) → LOD updated to %.4f ppm",
                    gas_name,
                    adev_result.sigma_min,
                    adev_result.tau_opt,
                    adev_result.noise_type,
                    lod if np.isfinite(lod) else float("nan"),
                )
        except Exception as _e:
            log.error(
                "Allan deviation computation failed (%s); falling back to OLS noise estimate. "
                "The lod_method tag in the summary dict reflects this fallback. "
                "If baseline_time_series data is available, investigate and re-run.",
                _e,
            )
            allan_result_dict = {"error": str(_e), "fallback": "OLS_residual_sigma"}

    # ── Detection-limit hierarchy validation ─────────────────────────────────
    # IUPAC hierarchy: NEC ≤ LOB ≤ LOD ≤ LOQ (all in concentration units).
    # NEC = σ_blank / |S|          (fundamental noise-equivalent concentration)
    # LOB = (|μ_blank| + 1.645·σ) / |S|   (one-sided 95th percentile of blank)
    # LOD = 3·σ / |S|              (3-sigma detection criterion)
    # LOQ = 10·σ / |S|             (10-sigma quantification criterion)
    #
    # Hierarchy should hold by construction when μ_blank ≈ 0 (reference-
    # subtracted sensor). LOB ≥ LOD occurs when |μ_blank| > 1.355·σ_blank,
    # indicating significant blank offset (drift, incomplete reference
    # subtraction, or wrong reference spectrum).  This is not a code error —
    # it is a data-quality finding that the researcher must investigate.
    _hier_warnings: list[str] = []
    _lob_for_hier = lob if np.isfinite(lob) else None
    _lod_for_hier = lod if np.isfinite(lod) else None
    _loq_for_hier = loq if np.isfinite(loq) else None

    _nec_lt_lob: bool | None = None
    _lob_lt_lod: bool | None = None
    _lod_lt_loq: bool | None = None

    if nec_ppm is not None and _lob_for_hier is not None:
        _nec_lt_lob = bool(nec_ppm <= _lob_for_hier + 1e-9)
        if not _nec_lt_lob:
            _hier_warnings.append(
                f"NEC ({nec_ppm:.4g} ppm) > LOB ({_lob_for_hier:.4g} ppm): "
                "unexpected — check blank measurements and reference subtraction."
            )

    if _lob_for_hier is not None and _lod_for_hier is not None:
        _lob_lt_lod = bool(_lob_for_hier <= _lod_for_hier + 1e-9)
        if not _lob_lt_lod:
            _hier_warnings.append(
                f"LOB ({_lob_for_hier:.4g} ppm) > LOD ({_lod_for_hier:.4g} ppm): "
                "blank mean signal is large relative to blank std "
                f"(|μ_blank|={abs(blank_mean_signal):.4g}, σ_blank={blank_std_signal:.4g}). "
                "Check: (1) reference subtraction applied? (2) sensor drift corrected? "
                "(3) correct carrier gas used as blank?"
            )

    if _lod_for_hier is not None and _loq_for_hier is not None:
        _lod_lt_loq = bool(_lod_for_hier <= _loq_for_hier + 1e-9)
        # LOD < LOQ is guaranteed by the 3σ vs 10σ definition — violations
        # should not occur unless floating-point issues arise.

    if _hier_warnings:
        for _w in _hier_warnings:
            log.warning("Hierarchy check [%s]: %s", gas_name, _w)

    hierarchy_check: dict[str, object] = {
        "nec_lt_lob": _nec_lt_lob,
        "lob_lt_lod": _lob_lt_lod,
        "lod_lt_loq": _lod_lt_loq,
        "hierarchy_ok": (
            (_nec_lt_lob is None or _nec_lt_lob)
            and (_lob_lt_lod is None or _lob_lt_lod)
            and (_lod_lt_loq is None or _lod_lt_loq)
        ),
        "warnings": _hier_warnings,
    }

    # ── Residual diagnostics: Durbin-Watson, Shapiro-Wilk, Breusch-Pagan
    residual_diag_dict: dict | None = None
    try:
        from src.scientific.residual_diagnostics import residual_diagnostics as _rdiag
        rdiag = _rdiag(c, r, slope=slope, intercept=intercept)
        residual_diag_dict = rdiag.as_dict()
    except Exception as _e:
        log.warning("Residual diagnostics failed (%s); skipping.", _e)

    summary = {
        # ICH Q2(R1) mandatory fields
        "gas": gas_name,
        "sensitivity": round(slope, 6),
        "sensitivity_se": round(slope_se, 8),  # slope standard error
        "intercept": round(intercept, 6),
        "r_squared": round(r2, 6),
        "noise_std": round(baseline_noise_std, 8),
        "n_calibration_points": int(len(c)),
        # LOB (IUPAC 2012 mandatory for publication) — from blank measurements when available
        "lob_ppm": round(lob, 6) if np.isfinite(lob) else None,
        "lob_from_blank_measurements": blank_measurements is not None and len(blank_measurements) >= 2,
        # NEC (Noise Equivalent Concentration) = σ_blank / |S| — fundamental detection limit
        "nec_ppm": round(nec_ppm, 6) if nec_ppm is not None and np.isfinite(nec_ppm) else None,
        "nec_ppm_ci_lower": round(nec_ci_lower, 6) if nec_ci_lower is not None else None,
        "nec_ppm_ci_upper": round(nec_ci_upper, 6) if nec_ci_upper is not None else None,
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
        # Residual diagnostics (Durbin-Watson, Shapiro-Wilk, Breusch-Pagan)
        "residual_diagnostics": residual_diag_dict,
        # Allan deviation noise characterisation (populated if baseline_time_series given)
        "allan_deviation": allan_result_dict,
        # Detection-limit hierarchy: NEC ≤ LOB ≤ LOD ≤ LOQ
        # hierarchy_ok=False signals a data-quality issue (usually blank offset)
        "hierarchy_check": hierarchy_check,
        # Audit trail flags (important for publication reproducibility)
        "lod_ci_from_blank": blank_measurements is not None and len(blank_measurements) >= 2,
        "mandel_low_power_warning": _mandel_low_power,
        # Methodology tag for audit trail
        "lod_method": (
            "ICH Q2(R1) 3σ/S from Allan σ_min (τ_opt) with bootstrap 95% CI (n=1000, σ_blank fixed)"
            if (allan_result_dict is not None and "error" not in (allan_result_dict or {}))
            else "ICH Q2(R1) 3σ/S (OLS residuals — no blank measurements provided) with bootstrap 95% CI (n=1000)"
            if blank_measurements is None
            else "ICH Q2(R1) 3σ/S from measured σ_blank with bootstrap 95% CI (n=1000, σ_blank fixed)"
        ),
        "loq_method": "ICH Q2(R1) 10σ/S with bootstrap 95% CI (n=1000)",
        "lob_method": (
            f"IUPAC 2012: μ_blank + 1.645·σ_blank / |S| from {len(blank_measurements)} blank measurements"
            if blank_measurements is not None and len(blank_measurements) >= 2
            else "IUPAC 2012: 1.645·σ_noise / |S| (no blank measurements; assumed μ_blank=0)"
        ),
        "nec_method": "σ_blank / |S| (Noise Equivalent Concentration, IUPAC)",
        "lol_method": "ICH Q2(R1) §4.2 Mandel F-test progressive truncation",
    }

    log.info(
        "Sensor performance [%s]: S=%.4f/ppm (SE=%.4f), R²=%.4f, "
        "LOD=%.4f [%.4f–%.4f] ppm, LOQ=%.4f ppm  diag=%s",
        gas_name,
        slope,
        slope_se,
        r2,
        lod if np.isfinite(lod) else float("nan"),
        lod_ci_lo if np.isfinite(lod_ci_lo) else float("nan"),
        lod_ci_hi if np.isfinite(lod_ci_hi) else float("nan"),
        loq if np.isfinite(loq) else float("nan"),
        "PASS" if (residual_diag_dict or {}).get("overall_pass", True) else "FAIL",
    )

    return summary
