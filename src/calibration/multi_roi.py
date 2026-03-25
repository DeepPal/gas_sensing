"""Multi-ROI fusion calibration.

Combines delta-wavelength signals from multiple spectral regions into a
multivariate linear model for concentration prediction.  Designed as the
next step after :func:`~src.signal.roi.find_monotonic_wavelengths` identifies
candidate wavelength regions.
"""
from __future__ import annotations
from typing import Any

import logging
import math

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

log = logging.getLogger(__name__)


def select_multi_roi_candidates(
    discovered_roi: dict[str, Any],
    max_features: int = 4,
) -> list[dict[str, Any]]:
    """Select the best quality-passing ROI candidates from a discovery result.

    Filters to candidates where ``quality_ok=True``, deduplicates by
    ``center_nm``, and ranks by ``(|slope_nm_per_ppm|, R²)`` descending.

    Args:
        discovered_roi: Dict containing a ``"candidates"`` list and optionally a
            ``"selected"`` entry (both from the calibration discovery pipeline).
            Each candidate must have ``"quality_ok"``, ``"center_nm"``,
            ``"concentrations_ppm"``, and either ``"deltas_valid_nm"`` or
            ``"deltas_nm"``.
        max_features: Maximum number of candidates to return.

    Returns:
        Sorted list of candidate dicts (up to *max_features*).  Empty list if
        *discovered_roi* is not a valid dict or no candidates pass.
    """
    if not isinstance(discovered_roi, dict):
        return []

    candidates = discovered_roi.get("candidates") or []
    if not isinstance(candidates, list):
        candidates = []

    quality: list[dict[str, Any]] = []
    seen_centers: set[float] = set()

    def _maybe_add(cand: dict[str, Any]) -> None:
        if not isinstance(cand, dict):
            return
        if not bool(cand.get("quality_ok", False)):
            return
        center = cand.get("center_nm")
        deltas = cand.get("deltas_valid_nm") or cand.get("deltas_nm")
        concs = cand.get("concentrations_ppm")
        if (
            center is None
            or not isinstance(deltas, (list, tuple))
            or not isinstance(concs, (list, tuple))
        ):
            return
        if len(deltas) != len(concs) or len(deltas) < 2:
            return
        center_val = float(center)
        if center_val in seen_centers:
            return
        seen_centers.add(center_val)
        quality.append(cand)

    _maybe_add(discovered_roi.get("selected"))  # type: ignore[arg-type]
    for cand in candidates:
        _maybe_add(cand)  # type: ignore[arg-type]

    def _score(cand: dict[str, Any]) -> tuple[float, float]:
        slope = cand.get("slope_nm_per_ppm")
        r2 = cand.get("r2")
        return (
            abs(float(slope)) if isinstance(slope, (int, float)) else 0.0,
            float(r2) if isinstance(r2, (int, float)) else 0.0,
        )

    quality.sort(key=_score, reverse=True)
    return quality[:max_features]


def fit_multi_roi_fusion(
    discovered_roi: dict[str, Any],
    concentrations: list[float] | np.ndarray,
    max_features: int = 4,
) -> dict[str, Any] | None:
    """Fit a multi-ROI linear fusion model for concentration prediction.

    Stacks delta-wavelength signals from the top ROI candidates as columns of a
    feature matrix *X*, then fits ``LinearRegression(y = X @ coef + intercept)``.
    Leave-one-out cross-validation (LOOCV) is performed when ``n ≥ 4``.

    This is a **pure computation** function — no file I/O, no plotting.
    Callers are responsible for persisting metrics and generating figures.

    Args:
        discovered_roi: Candidate ROI dict — passed directly to
            :func:`select_multi_roi_candidates`.
        concentrations: Known concentrations for each calibration spectrum (ppm).
            Must have ≥ 3 finite values.
        max_features: Maximum number of ROI features to include in the model.

    Returns:
        Metrics dict, or ``None`` if inputs are insufficient or degenerate
        (fewer than 2 valid feature vectors, or fewer than 3 concentrations).

        Keys:

        - ``"n_points"`` / ``"n_features"``
        - ``"feature_centers_nm"`` – list of center wavelengths used
        - ``"coefficients"`` / ``"intercept_ppm"``
        - ``"r2"`` / ``"rmse_ppm"``
        - ``"lod_ppm"`` – 3σ limit of detection estimate (nan if n ≤ 1)
        - ``"r2_cv"`` / ``"rmse_cv_ppm"`` – LOOCV metrics (nan if n < 4)
        - ``"actual_concentrations_ppm"`` / ``"predicted_concentrations_ppm"``
        - ``"cv_predictions_ppm"`` – LOOCV predictions (None if n < 4)
        - ``"residuals_ppm"`` / ``"features"``
    """
    y = np.array(concentrations, dtype=float)
    if y.shape[0] < 3 or not np.all(np.isfinite(y)):
        return None

    selected = select_multi_roi_candidates(discovered_roi, max_features=max_features)
    if len(selected) < 2:
        return None

    feature_vectors: list[np.ndarray] = []
    feature_details: list[dict[str, Any]] = []
    for cand in selected:
        deltas = cand.get("deltas_valid_nm") or cand.get("deltas_nm")
        vec = np.array(deltas, dtype=float)
        if vec.shape[0] != y.shape[0]:
            continue
        if not np.all(np.isfinite(vec)):
            continue
        if np.std(vec) < 1e-6:
            continue
        feature_vectors.append(vec)
        feature_details.append(
            {
                "center_nm": float(cand.get("center_nm", float("nan"))),
                "slope_nm_per_ppm": float(cand.get("slope_nm_per_ppm", float("nan"))),
                "r2": float(cand.get("r2", float("nan"))),
                "snr": float(cand.get("snr", float("nan"))),
            }
        )

    if len(feature_vectors) < 2:
        return None

    X = np.column_stack(feature_vectors)
    n = int(y.shape[0])
    n_feat = int(X.shape[1])

    # Ridge regression guards against overfitting when n/p < 5.
    # alpha=1.0 provides mild L2 regularisation; LOOCV R² is the primary metric.
    if n <= n_feat + 2:
        log.warning(
            "Multi-ROI fusion: n=%d samples, p=%d features — underdetermined. "
            "Using Ridge(alpha=1.0); rely on LOOCV R² not training R².",
            n, n_feat,
        )
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    y_pred: np.ndarray = model.predict(X)
    r2 = float(r2_score(y, y_pred))
    rmse = math.sqrt(float(mean_squared_error(y, y_pred)))
    residuals: np.ndarray = y_pred - y

    lod_ppm = float("nan")
    if residuals.size > 1:
        sigma = float(np.std(residuals, ddof=1))
        if np.isfinite(sigma):
            lod_ppm = 3.0 * sigma

    r2_cv = float("nan")
    rmse_cv = float("nan")
    cv_preds: np.ndarray | None = None
    if n >= 4:
        cv_arr = np.empty(n, dtype=float)
        valid = True
        for i in range(n):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            try:
                m_cv = Ridge(alpha=1.0)
                m_cv.fit(X_train, y_train)
                cv_arr[i] = float(m_cv.predict(X[i : i + 1])[0])
            except Exception:
                valid = False
                break
        if valid and np.all(np.isfinite(cv_arr)):
            cv_preds = cv_arr
            r2_cv = float(r2_score(y, cv_preds))
            rmse_cv = math.sqrt(float(mean_squared_error(y, cv_preds)))

    return {
        "n_points": n,
        "n_features": int(X.shape[1]),
        "feature_centers_nm": [fd["center_nm"] for fd in feature_details],
        "coefficients": model.coef_.tolist(),
        "intercept_ppm": float(model.intercept_),
        "r2": r2,
        "rmse_ppm": rmse,
        "lod_ppm": lod_ppm,
        "r2_cv": r2_cv,
        "rmse_cv_ppm": rmse_cv,
        "actual_concentrations_ppm": y.tolist(),
        "predicted_concentrations_ppm": y_pred.tolist(),
        "cv_predictions_ppm": cv_preds.tolist() if cv_preds is not None else None,
        "residuals_ppm": residuals.tolist(),
        "features": feature_details,
    }
