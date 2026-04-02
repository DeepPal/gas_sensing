"""
src.scientific.cross_session
==============================
Cross-session statistical comparison for LSPR sensor characterisation.

Provides formal statistical tests comparing two or more measurement sessions,
which are required for publication-grade reproducibility analysis.

Tests implemented
-----------------
- Paired t-test (Δλ means across sessions at matched concentrations)
- Bland-Altman analysis (method agreement, limits of agreement)
- F-test for variance equality (reproducibility)
- Mann-Whitney U (non-parametric, for small or non-normal samples)

Reference
---------
Bland JM, Altman DG (1986). "Statistical methods for assessing agreement
between two methods of clinical measurement." *The Lancet*, 327(8476), 307–310.

ISO 5725-2:2019 Accuracy (trueness and precision) of measurement methods —
Part 2: Basic method for the determination of repeatability and reproducibility.

Usage
-----
::

    from src.scientific.cross_session import compare_sessions, SessionData

    a = SessionData(concentrations=[0.1, 0.5, 1.0], delta_lambdas=[-0.12, -0.58, -1.15])
    b = SessionData(concentrations=[0.1, 0.5, 1.0], delta_lambdas=[-0.13, -0.55, -1.18])
    result = compare_sessions(a, b)
    print(result.bland_altman_bias, result.bland_altman_loa_lower)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Input data model
# ---------------------------------------------------------------------------

@dataclass
class SessionData:
    """Minimal summary of one LSPR measurement session for cross-session comparison.

    Attributes
    ----------
    concentrations   : Analyte concentrations (ppm) at which Δλ was measured.
    delta_lambdas    : Corresponding wavelength shifts (nm).  Must be the same
                      length as ``concentrations``.
    session_id       : Optional identifier for logging/reporting.
    lod_ppm          : Session LOD (ppm), if available.
    sensitivity_nm_per_ppm : Session sensitivity, if available.
    """
    concentrations: list[float]
    delta_lambdas: list[float]
    session_id: str = ""
    lod_ppm: Optional[float] = None
    sensitivity_nm_per_ppm: Optional[float] = None


# ---------------------------------------------------------------------------
# Result data model
# ---------------------------------------------------------------------------

@dataclass
class CrossSessionComparison:
    """Statistical comparison between two LSPR measurement sessions.

    All p-values follow the conventional two-sided significance threshold
    (α = 0.05).  Fields are ``None`` when insufficient data prevented the
    test from running.

    Paired t-test
    -------------
    Tests H₀: mean(Δλ_A − Δλ_B) = 0 at matched concentrations.
    Significant (p < 0.05) → systematic bias between sessions.

    Bland-Altman analysis
    ---------------------
    ``bland_altman_bias``        : mean of (A − B) differences (nm)
    ``bland_altman_loa_lower/upper`` : 95% limits of agreement = bias ± 1.96·σ_diff
    A bias near 0 + narrow LoA → sessions are interchangeable.

    F-test for variance equality
    ----------------------------
    Tests H₀: σ²_A = σ²_B (equal repeatability between sessions).
    Significant → one session has higher noise.

    Mann-Whitney U (non-parametric)
    --------------------------------
    Tests H₀: distributions of Δλ_A and Δλ_B are identical (no location shift).
    Preferred over t-test for n < 10 or when normality is not established.
    """

    # Paired t-test
    paired_t_statistic: Optional[float] = None
    paired_t_p_value: Optional[float] = None
    paired_t_significant: Optional[bool] = None     # p < 0.05
    paired_t_n: int = 0

    # Bland-Altman
    bland_altman_bias: Optional[float] = None       # mean difference A − B (nm)
    bland_altman_loa_lower: Optional[float] = None  # lower 95% limit of agreement
    bland_altman_loa_upper: Optional[float] = None  # upper 95% limit of agreement
    bland_altman_sd: Optional[float] = None         # std of differences
    bland_altman_n: int = 0

    # F-test (variance ratio)
    f_statistic: Optional[float] = None
    f_p_value: Optional[float] = None
    f_variances_equal: Optional[bool] = None        # True when p ≥ 0.05

    # Mann-Whitney U
    mw_u_statistic: Optional[float] = None
    mw_p_value: Optional[float] = None
    mw_significant: Optional[bool] = None

    # LOD / sensitivity comparison (if available in input)
    delta_lod_ppm: Optional[float] = None           # lod_B − lod_A (improvement if < 0)
    delta_sensitivity_nm_per_ppm: Optional[float] = None

    # Overall verdict
    sessions_reproducible: Optional[bool] = None
    """True when paired t-test is not significant (p ≥ 0.05) AND Bland-Altman
    bias ≤ 0.5·LOD AND F-test does not reject variance equality."""

    reproducibility_rsd_pct: Optional[float] = None
    """Relative standard deviation of all Δλ values pooled across sessions (%),
    as defined in ISO 5725-2 for intermediate precision."""

    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_sessions(
    session_a: SessionData,
    session_b: SessionData,
    *,
    alpha: float = 0.05,
    lod_ppm: Optional[float] = None,
) -> CrossSessionComparison:
    """Compare two LSPR measurement sessions statistically.

    Parameters
    ----------
    session_a, session_b : Sessions to compare.
    alpha                : Significance level (default 0.05).
    lod_ppm              : Sensor LOD used for reproducibility verdict.
                           Falls back to ``session_a.lod_ppm`` if not provided.

    Returns
    -------
    CrossSessionComparison
        All statistical test results.
    """
    result = CrossSessionComparison()
    effective_lod = lod_ppm or session_a.lod_ppm

    a_dl = np.asarray(session_a.delta_lambdas, dtype=float)
    b_dl = np.asarray(session_b.delta_lambdas, dtype=float)

    # ── Paired t-test and Bland-Altman (require matched pairs) ───────────
    n_paired = min(len(a_dl), len(b_dl))
    if n_paired >= 3:
        a_p = a_dl[:n_paired]
        b_p = b_dl[:n_paired]
        diffs = b_p - a_p  # B − A: positive = B drifted higher vs A (Bland-Altman convention)

        # Paired t-test
        try:
            t_stat, t_pval = stats.ttest_rel(a_p, b_p)
            result.paired_t_statistic = float(t_stat)
            result.paired_t_p_value = float(t_pval)
            result.paired_t_significant = bool(t_pval < alpha)
            result.paired_t_n = n_paired
        except Exception:
            pass

        # Bland-Altman
        bias = float(np.mean(diffs))
        sd_diff = float(np.std(diffs, ddof=1))
        result.bland_altman_bias = bias
        result.bland_altman_sd = sd_diff
        result.bland_altman_loa_lower = bias - 1.96 * sd_diff
        result.bland_altman_loa_upper = bias + 1.96 * sd_diff
        result.bland_altman_n = n_paired

        if sd_diff > 0 and n_paired < 10:
            result.warnings.append(
                f"Bland-Altman LoA based on n={n_paired} pairs — "
                "LoA are unreliable below n=30 (ISO 5725)."
            )
    else:
        result.warnings.append(
            f"Insufficient paired points (n={n_paired}) for paired t-test / Bland-Altman."
        )

    # ── F-test for variance equality ─────────────────────────────────────
    if len(a_dl) >= 3 and len(b_dl) >= 3:
        var_a = float(np.var(a_dl, ddof=1))
        var_b = float(np.var(b_dl, ddof=1))
        if var_b > 1e-12 and var_a > 1e-12:
            f_stat = var_a / var_b
            df_a = len(a_dl) - 1
            df_b = len(b_dl) - 1
            # Two-sided F-test
            p_right = float(stats.f.sf(f_stat, df_a, df_b))
            p_left = float(stats.f.cdf(f_stat, df_a, df_b))
            f_pval = 2.0 * min(p_right, p_left)
            result.f_statistic = f_stat
            result.f_p_value = float(f_pval)
            result.f_variances_equal = bool(f_pval >= alpha)

    # ── Mann-Whitney U (non-parametric) ──────────────────────────────────
    if len(a_dl) >= 3 and len(b_dl) >= 3:
        try:
            u_stat, mw_pval = stats.mannwhitneyu(a_dl, b_dl, alternative="two-sided")
            result.mw_u_statistic = float(u_stat)
            result.mw_p_value = float(mw_pval)
            result.mw_significant = bool(mw_pval < alpha)
        except Exception:
            pass

    # ── LOD / sensitivity delta ───────────────────────────────────────────
    if session_a.lod_ppm is not None and session_b.lod_ppm is not None:
        result.delta_lod_ppm = session_b.lod_ppm - session_a.lod_ppm
    if (
        session_a.sensitivity_nm_per_ppm is not None
        and session_b.sensitivity_nm_per_ppm is not None
    ):
        result.delta_sensitivity_nm_per_ppm = (
            session_b.sensitivity_nm_per_ppm - session_a.sensitivity_nm_per_ppm
        )

    # ── Pooled RSD (intermediate precision, ISO 5725-2) ───────────────────
    # Use std of pair-wise differences / |grand mean|.  For identical sessions
    # the differences are all zero → RSD = 0, which correctly indicates perfect
    # reproducibility.  Using std(all_dl)/mean(all_dl) would give CV of the
    # concentration-response range — a different and misleading quantity here.
    all_dl = np.concatenate([a_dl, b_dl])
    grand_mean = abs(float(np.mean(all_dl)))
    if n_paired >= 4 and grand_mean > 1e-6 and result.bland_altman_sd is not None:
        result.reproducibility_rsd_pct = float(
            100.0 * result.bland_altman_sd / grand_mean
        )

    # ── Overall reproducibility verdict ──────────────────────────────────
    t_ok = (result.paired_t_significant is False)  # not significant = reproducible
    ba_ok = (
        result.bland_altman_bias is not None
        and effective_lod is not None
        and abs(result.bland_altman_bias) <= 0.5 * effective_lod
    ) or (
        result.bland_altman_bias is not None
        and effective_lod is None
    )
    f_ok = result.f_variances_equal is not False  # None = not tested → don't penalise
    result.sessions_reproducible = t_ok and ba_ok and f_ok

    return result


def compare_lod_series(
    lod_values: list[float],
    *,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Compute trend statistics for a series of LOD values across sessions.

    Returns a dict with keys:
      ``mean_lod``, ``std_lod``, ``rsd_pct``, ``trend_slope_ppm_per_session``,
      ``trend_p_value``, ``drifting`` (bool, True if slope is significant).
    """
    n = len(lod_values)
    if n < 3:
        return {"error": f"Need ≥ 3 sessions, got {n}"}

    arr = np.asarray(lod_values, dtype=float)
    sessions = np.arange(n, dtype=float)

    slope, intercept, r_val, p_val, se = stats.linregress(sessions, arr)
    mean_lod = float(np.mean(arr))
    std_lod = float(np.std(arr, ddof=1))
    rsd = 100.0 * std_lod / mean_lod if mean_lod > 1e-9 else math.nan

    return {
        "mean_lod": mean_lod,
        "std_lod": std_lod,
        "rsd_pct": rsd,
        "trend_slope_ppm_per_session": float(slope),
        "trend_r2": float(r_val ** 2),
        "trend_p_value": float(p_val),
        "drifting": bool(p_val < alpha and slope > 0),
        "n_sessions": n,
    }
