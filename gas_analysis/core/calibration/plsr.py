from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score


def build_feature_matrix_from_canonical(
    canonical: dict[float, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build X (spectral matrix) and y (concentrations) from canonical dict.

    Returns (X, y, wavelengths). X has shape (n_samples, n_wavelengths).
    """
    if not canonical:
        raise ValueError("No canonical spectra provided for PLSR")

    items = sorted(canonical.items(), key=lambda kv: kv[0])
    concs = np.array([float(k) for k, _ in items], dtype=float)

    # Use the first spectrum as base grid
    base_wl = items[0][1]["wavelength"].to_numpy()

    # Determine signal column
    first_df = items[0][1]
    if (
        "absorbance" in first_df.columns
        or "transmittance" in first_df.columns
        or "intensity" in first_df.columns
    ):
        pass
    else:
        # Fallback to second column
        cols = [c for c in first_df.columns if c != "wavelength"]
        if cols:
            cols[0]

    X_list: list[np.ndarray] = []
    for _, df in items:
        wl = df["wavelength"].to_numpy()

        # Determine signal column for this dataframe
        current_sig_col = "intensity"
        if "absorbance" in df.columns:
            current_sig_col = "absorbance"
        elif "transmittance" in df.columns:
            current_sig_col = "transmittance"
        elif "intensity" in df.columns:
            current_sig_col = "intensity"
        else:
            cols = [c for c in df.columns if c != "wavelength"]
            if cols:
                current_sig_col = cols[0]

        ysig = df[current_sig_col].to_numpy()
        if not np.array_equal(wl, base_wl):
            ysig = np.interp(base_wl, wl, ysig)
        X_list.append(ysig)

    X = np.vstack(X_list)
    return X, concs, base_wl


def fit_plsr_calibration(
    canonical: dict[float, pd.DataFrame],
    cfg: dict[str, Any],
    matrix_cache: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Optional[dict[str, Any]]:
    """Fit PLSR to predict concentration from full spectrum, with simple CV model selection."""
    try:
        if matrix_cache is None:
            X, y, wl = build_feature_matrix_from_canonical(canonical)
        else:
            X, y, wl = matrix_cache
    except ValueError:
        return None

    # Optional absorbance transform
    if bool(cfg.get("absorbance", False)):
        x_max = np.nanmax(X)
        x_min = np.nanmin(X)
        if np.isfinite(x_max) and np.isfinite(x_min) and x_min >= 0.0 and x_max <= 1.2:
            with np.errstate(divide="ignore", invalid="ignore"):
                X = -np.log10(np.clip(X, 1e-6, None))

    # Optional wavelength limits
    wl_min = cfg.get("wl_min")
    wl_max = cfg.get("wl_max")
    if wl_min is not None or wl_max is not None:
        mask = np.ones_like(wl, dtype=bool)
        if wl_min is not None:
            mask &= wl >= float(wl_min)
        if wl_max is not None:
            mask &= wl <= float(wl_max)
        if np.any(mask):
            X = X[:, mask]
            wl = wl[mask]

    # Optional feature preprocessing
    prep = str(cfg.get("feature_prep", "raw")).lower()
    if prep in ("derivative", "first_derivative") or "derivative" in prep:
        X = np.gradient(X, wl, axis=1)
    if prep == "snv" or "snv" in prep:
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        sd[sd < 1e-9] = 1.0
        X = (X - mu) / sd
    elif prep == "mean_center":
        mu = X.mean(axis=0, keepdims=True)
        X = X - mu

    n_samples, n_features = X.shape
    if n_samples < 3:
        return None

    max_comp_cfg = int(cfg.get("max_components", 5))
    max_components = max(1, min(max_comp_cfg, n_samples - 1))
    best_cv_r2 = float("-inf")
    best_cv_rmse = float("inf")
    best_n = 1
    preds_cv_best = None
    r2_cv_curve: list[float] = []
    rmse_cv_curve: list[float] = []
    r2_in_curve: list[float] = []

    # Leave-one-out CV
    for n_comp in range(1, max_components + 1):
        cv_preds = np.zeros_like(y)
        for i in range(n_samples):
            mask = np.ones(n_samples, dtype=bool)
            mask[i] = False
            pls = PLSRegression(n_components=n_comp, scale=bool(cfg.get("scale", True)))
            pls.fit(X[mask], y[mask])
            cv_preds[i] = float(pls.predict(X[i : i + 1]).ravel()[0])
        cv_r2 = float(r2_score(y, cv_preds))
        cv_rmse = float(np.sqrt(mean_squared_error(y, cv_preds)))
        r2_cv_curve.append(cv_r2)
        rmse_cv_curve.append(cv_rmse)

        # In-sample diagnostic
        pls_full = PLSRegression(n_components=n_comp, scale=bool(cfg.get("scale", True)))
        pls_full.fit(X, y)
        preds_full = pls_full.predict(X).ravel()
        r2_in_n = float(r2_score(y, preds_full)) if np.isfinite(np.var(y)) else float("nan")
        r2_in_curve.append(r2_in_n)

        if np.isfinite(cv_r2) and cv_r2 > best_cv_r2:
            best_cv_r2 = cv_r2
            best_cv_rmse = cv_rmse
            best_n = n_comp
            preds_cv_best = cv_preds.copy()

    # Fit final model
    pls_final = PLSRegression(n_components=best_n, scale=bool(cfg.get("scale", True)))
    pls_final.fit(X, y)
    preds_in = pls_final.predict(X).ravel()
    in_r2 = float(r2_score(y, preds_in)) if np.isfinite(np.var(y)) else float("nan")
    in_rmse = float(np.sqrt(mean_squared_error(y, preds_in)))

    # Feature selection (optional)
    selected_mask = None
    fs_cfg = cfg.get("feature_selection", {}) if isinstance(cfg, dict) else {}
    if fs_cfg.get("enabled", False):
        method = str(fs_cfg.get("method", "coef_top_fraction")).lower()
        if method == "coef_top_fraction":
            frac = float(fs_cfg.get("top_fraction", 0.2))
            frac = float(np.clip(frac, 0.01, 0.8))
            coef = np.abs(pls_final.coef_.ravel())
            if coef.size > 0 and np.any(coef > 0):
                k = max(1, int(round(frac * coef.size)))
                top_idx = np.argpartition(coef, -k)[-k:]
                selected_mask = np.zeros_like(coef, dtype=bool)
                selected_mask[top_idx] = True

        if selected_mask is not None and np.any(selected_mask):
            X_sel = X[:, selected_mask]
            cv_preds_sel = np.zeros_like(y)
            for i in range(n_samples):
                mask = np.ones(n_samples, dtype=bool)
                mask[i] = False
                pls_sel = PLSRegression(
                    n_components=min(best_n, X_sel.shape[1]),
                    scale=bool(cfg.get("scale", True)),
                )
                pls_sel.fit(X_sel[mask], y[mask])
                cv_preds_sel[i] = float(pls_sel.predict(X_sel[i : i + 1]).ravel()[0])
            cv_r2_sel = float(r2_score(y, cv_preds_sel))
            cv_rmse_sel = float(np.sqrt(mean_squared_error(y, cv_preds_sel)))

            if np.isfinite(cv_r2_sel) and cv_r2_sel > best_cv_r2:
                best_cv_r2 = cv_r2_sel
                best_cv_rmse = cv_rmse_sel
                pls_final = PLSRegression(
                    n_components=min(best_n, X_sel.shape[1]),
                    scale=bool(cfg.get("scale", True)),
                )
                pls_final.fit(X_sel, y)
                preds_in = pls_final.predict(X_sel).ravel()
                in_r2 = float(r2_score(y, preds_in)) if np.isfinite(np.var(y)) else float("nan")
                in_rmse = float(np.sqrt(mean_squared_error(y, preds_in)))
                wl = wl[selected_mask]

    result: dict[str, Any] = {
        "model": "plsr",
        "n_components": int(best_n),
        "wavelengths": wl.tolist(),
        "concentrations": y.tolist(),
        "predictions_in": preds_in.tolist(),
        "r2_in": in_r2,
        "rmse_in": in_rmse,
        "predictions_cv": preds_cv_best.tolist() if preds_cv_best is not None else None,
        "r2_cv": best_cv_r2,
        "rmse_cv": best_cv_rmse,
        "r2_cv_curve": r2_cv_curve,
        "rmse_cv_curve": rmse_cv_curve,
        "r2_in_curve": r2_in_curve,
        "n_components_candidates": list(range(1, max_components + 1)),
        "coef_": pls_final.coef_.ravel().tolist(),
    }

    # Simple Bootstrap (omitted complex config logic for brevity/purity)
    bootstrap_cfg = cfg.get("bootstrap", {})
    if bootstrap_cfg.get("enabled", False) and n_samples >= 3:
        # Simplified bootstrap
        rng = np.random.default_rng(bootstrap_cfg.get("random_seed", None))
        iterations = 100
        pred_samples = []
        for _ in range(iterations):
            idx = rng.integers(0, n_samples, size=n_samples)
            try:
                pls_bs = PLSRegression(
                    n_components=min(best_n, n_samples - 1),
                    scale=bool(cfg.get("scale", True)),
                )
                pls_bs.fit(X[idx], y[idx])
                pred_samples.append(pls_bs.predict(X).ravel())
            except Exception:
                pass

        if pred_samples:
            arr = np.vstack(pred_samples)
            result["bootstrap"] = {
                "predictions": {
                    "ci_low": np.percentile(arr, 2.5, axis=0).tolist(),
                    "ci_high": np.percentile(arr, 97.5, axis=0).tolist(),
                    "median": np.median(arr, axis=0).tolist(),
                }
            }

    return result
