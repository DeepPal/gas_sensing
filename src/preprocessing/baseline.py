"""
src.preprocessing.baseline
===========================
Baseline correction algorithms for spectral data.

All functions are pure — no side effects, no project imports beyond numpy/scipy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve


def als_baseline(
    intensities: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    n_iter: int = 10,
) -> np.ndarray:
    """Asymmetric Least Squares (ALS) baseline correction.

    Fits a smooth baseline to the spectrum by iteratively re-weighting
    residuals so that points below the estimated baseline receive higher
    weight (``p`` ≈ 0.01 forces the baseline to lie at/below the signal).

    Parameters
    ----------
    intensities:
        1-D raw intensity array.
    lam:
        Smoothness penalty (λ).  Larger → smoother baseline.
        Typical range: 10² – 10⁷.
    p:
        Asymmetry parameter.  Smaller → baseline hugs the bottom of the signal.
        Typical range: 0.001 – 0.1.
    n_iter:
        Maximum number of IRLS iterations (usually converges in < 10).

    Returns
    -------
    np.ndarray
        Baseline-corrected intensities (``intensities − estimated_baseline``).
    """
    if len(intensities) < 4:
        return intensities.copy()

    L = len(intensities)
    # Second-order difference matrix
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L)).T
    w = np.ones(L)
    z: np.ndarray = intensities.copy().astype(float)

    for _ in range(n_iter):
        W = sparse.diags(w)
        Z = (W + lam * D.dot(D.T)).tocsc()
        z = np.asarray(spsolve(Z, w * intensities))
        w = p * (intensities > z) + (1 - p) * (intensities <= z)

    return np.asarray(intensities - z)


def polynomial_baseline(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    order: int = 2,
) -> np.ndarray:
    """Fit a polynomial to the spectrum edges and subtract it.

    Parameters
    ----------
    wavelengths:
        1-D wavelength axis.
    intensities:
        1-D intensity array.
    order:
        Polynomial degree.  Clamped to ``min(order, len(wavelengths)//10 - 1)``.

    Returns
    -------
    np.ndarray
        Baseline-corrected intensities.
    """
    if len(wavelengths) < 4:
        return intensities.copy()

    n_edge = max(1, len(wavelengths) // 10)
    edge_wl = np.concatenate([wavelengths[:n_edge], wavelengths[-n_edge:]])
    edge_int = np.concatenate([intensities[:n_edge], intensities[-n_edge:]])

    order = max(1, min(order, len(edge_wl) - 1))
    coef = np.polyfit(edge_wl, edge_int, order)
    baseline: np.ndarray = np.asarray(np.polyval(coef, wavelengths))
    return np.asarray(intensities - baseline)


def rolling_min_baseline(
    intensities: np.ndarray,
    window_fraction: float = 0.05,
) -> np.ndarray:
    """Rolling-minimum baseline correction.

    Computes a rolling minimum with a window equal to ``window_fraction`` of
    the array length, then subtracts it.

    Parameters
    ----------
    intensities:
        1-D intensity array.
    window_fraction:
        Window size as fraction of total length (default 5%).

    Returns
    -------
    np.ndarray
        Baseline-corrected intensities.
    """
    if len(intensities) < 4:
        return intensities.copy()

    window = max(3, int(len(intensities) * window_fraction))
    if window % 2 == 0:
        window += 1

    baseline: np.ndarray = (
        pd.Series(intensities).rolling(window=window, center=True).min().bfill().ffill().to_numpy()
    )
    return np.asarray(intensities - baseline)


def airpls_baseline(
    intensities: np.ndarray,
    lam: float = 1e5,
    n_iter: int = 15,
    tol: float = 0.001,
) -> np.ndarray:
    """Adaptive iteratively reweighted Penalized Least Squares (airPLS).

    Improves on standard ALS by adaptively setting weights: points *above*
    the estimated baseline receive weight 0 (ignored), while points *below*
    receive exponential weights proportional to their deviation.  This
    eliminates the need to manually tune the ``p`` asymmetry parameter.

    Reference
    ---------
    Zhang, Z.M., Chen, S., & Liang, Y.Z. (2010). Baseline correction using
    adaptive iteratively reweighted penalized least squares.
    *Analyst*, 135(5), 1138–1146. https://doi.org/10.1039/b922045c

    Parameters
    ----------
    intensities:
        1-D raw intensity array.
    lam:
        Smoothness penalty (λ).  Same interpretation as :func:`als_baseline`.
        Typical range: 10² – 10⁷.
    n_iter:
        Maximum IRLS iterations (usually converges in < 10).
    tol:
        Convergence threshold: stop when
        ``Σ|negative residuals| / Σ|signal| < tol``.

    Returns
    -------
    np.ndarray
        Baseline-corrected intensities (``intensities − estimated_baseline``).
    """
    if len(intensities) < 4:
        return intensities.copy()

    L = len(intensities)
    y: np.ndarray = intensities.astype(float)

    # Second-order difference matrix for smoothness penalty
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    H = lam * D.T.dot(D)

    w = np.ones(L)
    z: np.ndarray = y.copy()

    for t in range(1, n_iter + 1):
        W = sparse.diags(w)
        C = (W + H).tocsc()
        z = np.asarray(spsolve(C, w * y))

        d = y - z  # residuals: positive = above baseline, negative = below

        # Convergence check: relative sum of negative residuals
        neg_mask = d < 0
        if not neg_mask.any():
            break
        sum_neg = float(np.abs(d[neg_mask]).sum())
        sum_y = float(np.abs(y).sum())
        if sum_y > 0 and sum_neg / sum_y < tol:
            break

        # Adaptive weights: 0 for points above baseline (d ≥ 0),
        # exponential for points below (d < 0) — smaller weight for
        # deeper deviation (prevents baseline from chasing signal valleys).
        w = np.zeros(L)
        if sum_neg > 0:
            w[neg_mask] = np.exp(t * d[neg_mask] / sum_neg)

    return np.asarray(y - z)


def correct_baseline(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    method: str = "als",
    **kwargs: Any,
) -> np.ndarray:
    """Dispatch to the requested baseline correction method.

    Parameters
    ----------
    wavelengths:
        1-D wavelength axis (required for polynomial method; ignored by ALS).
    intensities:
        1-D intensity array.
    method:
        One of ``'als'``, ``'airpls'``, ``'polynomial'``, ``'rolling_min'``.
    **kwargs:
        Forwarded to the specific method (e.g. ``lam`` for ALS/airPLS).

    Returns
    -------
    np.ndarray
        Baseline-corrected intensities.
    """
    if method == "als":
        return als_baseline(intensities, **kwargs)
    if method == "airpls":
        return airpls_baseline(intensities, **kwargs)
    if method == "polynomial":
        return polynomial_baseline(wavelengths, intensities, **kwargs)
    if method == "rolling_min":
        return rolling_min_baseline(intensities, **kwargs)

    raise ValueError(
        f"Unknown baseline method: {method!r}. Choose from: als, airpls, polynomial, rolling_min."
    )
