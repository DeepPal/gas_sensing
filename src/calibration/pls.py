"""
src.calibration.pls
====================
Partial Least Squares (PLS) calibration for spectroscopic quantification.

PLS is the gold-standard multivariate calibration method for spectroscopy
(NIR, Raman, UV-Vis, LSPR) because it:

1. Handles collinear predictors (spectral channels are highly correlated).
2. Works well with small N (typical in research: 10–50 samples).
3. Provides VIP scores to identify informative spectral regions.
4. Q² (cross-validated R²) gives an honest predictive-ability estimate.

References
----------
Wold, S. et al. (2001). PLS-regression: a basic tool of chemometrics.
    Chemometrics and Intelligent Laboratory Systems, 58, 109–130.

Chong, I. & Jun, C. (2005). Performance of some variable selection methods
    when multicollinearity is present.
    Chemometrics and Intelligent Laboratory Systems, 78, 103–112.

Mehmood, T. et al. (2012). A review of variable selection methods in PLS
    regression.  Chemometrics and Intelligent Laboratory Systems, 118, 62–69.

Standard compliance
-------------------
- ICH Q2(R1) §5.2 (Linearity), §5.3 (Range), §5.4 (Accuracy), §5.5 (Precision)
- ASTM E1655-17: Standard Practices for Infrared Multivariate Quantitative Analysis

Public API
----------
- ``PLSCalibration``  — fit, predict, VIP scores, diagnostics
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from scipy.stats import pearsonr

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PLSFitResult:
    """Calibration diagnostics returned by :meth:`PLSCalibration.fit`.

    All metrics follow standard chemometrics conventions.

    Attributes
    ----------
    n_components :
        Number of PLS latent variables used.
    n_samples, n_features :
        Training set dimensions.
    r2_calibration :
        R² on the training set (in-sample fit quality).
    rmsec :
        Root Mean Squared Error of Calibration (same units as y).
    q2 :
        Cross-validated R² (leave-one-out or K-fold).
        Q² > 0.90 is excellent; Q² < 0.50 suggests over-fitting.
    rmsecv :
        Root Mean Squared Error of Cross-Validation.
        Use RMSECV to compare models; it generalises to new samples.
    rmsecv_per_component :
        RMSECV curve used for component selection (shape ``(max_components,)``).
    optimal_n_components :
        Component index that minimises RMSECV (may differ from fitted).
    vip_scores :
        Variable Importance in Projection, shape ``(n_features,)``.
        VIP > 1.0 → variable is above-average important.
    x_loadings :
        P matrix: loadings of X on each component.
        Shape ``(n_features, n_components)``.
    x_scores :
        T matrix: scores of X on each component.
        Shape ``(n_samples, n_components)``.
    y_loadings :
        Q vector / matrix of y on each component.
    x_weights :
        W matrix (raw weights before deflation).
    explained_variance_x :
        Fraction of X variance explained by each component.
    explained_variance_y :
        Fraction of y variance explained by each component.
    pearson_r :
        Pearson correlation between y and ŷ on the training set.
    bias :
        Mean residual (predicted − actual) on the training set.
    """

    n_components: int = 0
    n_samples: int = 0
    n_features: int = 0

    # Calibration (training set)
    r2_calibration: float = float("nan")
    rmsec: float = float("nan")
    pearson_r: float = float("nan")
    bias: float = float("nan")

    # Cross-validation
    q2: float = float("nan")
    rmsecv: float = float("nan")
    rmsecv_per_component: list[float] = field(default_factory=list)
    optimal_n_components: int = 0

    # Feature importance
    vip_scores: np.ndarray = field(default_factory=lambda: np.array([]))

    # Spectral decomposition
    x_loadings: np.ndarray = field(default_factory=lambda: np.array([]))
    x_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    y_loadings: np.ndarray = field(default_factory=lambda: np.array([]))
    x_weights: np.ndarray = field(default_factory=lambda: np.array([]))

    # Variance explained
    explained_variance_x: list[float] = field(default_factory=list)
    explained_variance_y: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PLSCalibration
# ---------------------------------------------------------------------------


class PLSCalibration:
    """PLS-1 calibration for single-analyte spectroscopic quantification.

    Uses sklearn's :class:`~sklearn.cross_decomposition.PLSRegression` as the
    numerical backend, adding:

    * Automatic component-count selection via RMSECV minimisation.
    * VIP (Variable Importance in Projection) scores.
    * LOO or K-fold cross-validation diagnostics (Q², RMSECV).
    * Prediction uncertainty via jack-knife residual estimation.

    Parameters
    ----------
    n_components :
        Number of PLS latent variables.  Use :meth:`optimize_components`
        to find the optimal value data-adaptively.
    scale :
        If True, each spectral channel is divided by its standard deviation
        before fitting (mean-centring is always applied).  Usually True for
        spectra with heterogeneous signal magnitudes across the spectral axis.
    cv_folds :
        Number of cross-validation folds.  Set to ``-1`` for
        leave-one-out (LOO) CV, which is recommended for small datasets
        (N < 30).

    Examples
    --------
    ::

        import numpy as np
        from src.calibration.pls import PLSCalibration

        # X: spectral matrix (n_samples × n_wavelengths)
        # y: concentration vector (n_samples,)
        pls = PLSCalibration(n_components=3)
        result = pls.fit(X, y)
        print(f"Q² = {result.q2:.3f},  RMSECV = {result.rmsecv:.4f} ppm")

        # Feature importance
        import matplotlib.pyplot as plt
        plt.plot(wavelengths, result.vip_scores)
        plt.axhline(1.0, ls='--', color='red', label='VIP = 1 threshold')

        # Predict new samples
        y_pred = pls.predict(X_new)
    """

    def __init__(
        self,
        n_components: int = 3,
        scale: bool = True,
        cv_folds: int = -1,
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.cv_folds = cv_folds

        self._model: Any | None = None
        self._fit_result: PLSFitResult | None = None
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Component optimisation
    # ------------------------------------------------------------------

    def optimize_components(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_components: int = 10,
    ) -> tuple[int, list[float]]:
        """Find the optimal number of PLS components using RMSECV.

        Fits PLS models with 1 … ``max_components`` components, evaluating
        each with cross-validation.  Returns the component count that
        minimises RMSECV (the "one-standard-error" rule is *not* applied
        here; the global minimum is returned for simplicity).

        Parameters
        ----------
        X :
            Spectral matrix, shape ``(n_samples, n_features)``.
        y :
            Concentration vector, shape ``(n_samples,)``.
        max_components :
            Upper bound on the component search.  Capped at
            ``min(n_samples − 1, n_features)``.

        Returns
        -------
        optimal_n : int
            Optimal number of components.
        rmsecv_curve : list[float]
            RMSECV at each component count [1 … max_components].
        """
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        n = X_arr.shape[0]
        max_components = min(max_components, n - 1, X_arr.shape[1])

        cv = LeaveOneOut() if (self.cv_folds == -1 or n <= 10) else KFold(
            n_splits=min(self.cv_folds, n), shuffle=True, random_state=42
        )

        rmsecv_curve: list[float] = []
        for nc in range(1, max_components + 1):
            pls_tmp = PLSRegression(n_components=nc, scale=self.scale)
            y_cv = cross_val_predict(pls_tmp, X_arr, y_arr, cv=cv).ravel()
            rmsecv_curve.append(float(np.sqrt(np.mean((y_arr - y_cv) ** 2))))

        optimal_n = int(np.argmin(rmsecv_curve)) + 1  # 1-indexed
        log.info(
            "PLSCalibration.optimize_components: optimal n_components=%d "
            "(RMSECV=%.4f)",
            optimal_n,
            rmsecv_curve[optimal_n - 1],
        )
        return optimal_n, rmsecv_curve

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        wavelengths: np.ndarray | None = None,
    ) -> PLSFitResult:
        """Fit the PLS model and compute full calibration diagnostics.

        Parameters
        ----------
        X :
            Spectral matrix, shape ``(n_samples, n_features)``.
            Rows = samples, columns = wavelength channels.
        y :
            Known concentration vector, shape ``(n_samples,)``.
        wavelengths :
            Wavelength axis in nm, shape ``(n_features,)``.  Only used
            for logging; does not affect the fit.

        Returns
        -------
        PLSFitResult
            Full diagnostic result (VIP scores, Q², RMSECV, etc.).
        """
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X_arr.shape

        if n_samples < 3:
            raise ValueError(
                f"PLSCalibration.fit requires at least 3 samples, got {n_samples}."
            )

        n_comp = min(self.n_components, n_samples - 1, n_features)
        if n_comp != self.n_components:
            warnings.warn(
                f"PLSCalibration: n_components clamped from {self.n_components} "
                f"to {n_comp} (limited by dataset size).",
                UserWarning,
                stacklevel=2,
            )

        # ── Fit ────────────────────────────────────────────────────────
        pls = PLSRegression(n_components=n_comp, scale=self.scale)
        pls.fit(X_arr, y_arr)
        self._model = pls
        self._X_train = X_arr
        self._y_train = y_arr

        # ── Training metrics ───────────────────────────────────────────
        y_pred_cal = pls.predict(X_arr).ravel()
        ss_res = float(np.sum((y_arr - y_pred_cal) ** 2))
        ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
        r2_cal = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmsec = float(np.sqrt(np.mean((y_arr - y_pred_cal) ** 2)))
        bias = float(np.mean(y_pred_cal - y_arr))
        try:
            pearson_r = float(pearsonr(y_arr, y_pred_cal).statistic)
        except Exception:
            pearson_r = float("nan")

        # ── Cross-validation (Q², RMSECV) ─────────────────────────────
        cv = LeaveOneOut() if (self.cv_folds == -1 or n_samples <= 10) else KFold(
            n_splits=min(self.cv_folds, n_samples), shuffle=True, random_state=42
        )
        y_cv = cross_val_predict(
            PLSRegression(n_components=n_comp, scale=self.scale),
            X_arr, y_arr, cv=cv
        ).ravel()
        ss_res_cv = float(np.sum((y_arr - y_cv) ** 2))
        q2 = 1.0 - ss_res_cv / ss_tot if ss_tot > 0 else 0.0
        rmsecv = float(np.sqrt(np.mean((y_arr - y_cv) ** 2)))

        # ── RMSECV curve for all component counts ──────────────────────
        max_nc = min(n_samples - 1, n_features, 15)
        rmsecv_curve: list[float] = []
        for nc in range(1, max_nc + 1):
            pls_tmp = PLSRegression(n_components=nc, scale=self.scale)
            yc = cross_val_predict(pls_tmp, X_arr, y_arr, cv=cv).ravel()
            rmsecv_curve.append(float(np.sqrt(np.mean((y_arr - yc) ** 2))))
        optimal_nc = int(np.argmin(rmsecv_curve)) + 1

        # ── VIP scores ─────────────────────────────────────────────────
        vip = self._compute_vip(pls, X_arr, y_arr)

        # ── Explained variance per component ──────────────────────────
        ev_x: list[float] = []
        ev_y: list[float] = []
        X_var = float(np.var(X_arr, axis=0).sum())
        y_var = float(np.var(y_arr))
        for h in range(n_comp):
            t_h = pls.x_scores_[:, h]
            # X variance explained by component h
            p_h = pls.x_loadings_[:, h]
            x_approx = np.outer(t_h, p_h)
            ev_x.append(float(np.var(x_approx.sum(axis=0)) / X_var) if X_var > 0 else 0.0)
            # y variance explained by component h
            q_h = float(pls.y_loadings_[0, h]) if pls.y_loadings_.ndim == 2 else float(pls.y_loadings_[h])
            y_approx = t_h * q_h
            ev_y.append(float(np.var(y_approx) / y_var) if y_var > 0 else 0.0)

        result = PLSFitResult(
            n_components=n_comp,
            n_samples=n_samples,
            n_features=n_features,
            r2_calibration=round(r2_cal, 6),
            rmsec=round(rmsec, 6),
            pearson_r=round(pearson_r, 6),
            bias=round(bias, 8),
            q2=round(q2, 6),
            rmsecv=round(rmsecv, 6),
            rmsecv_per_component=rmsecv_curve,
            optimal_n_components=optimal_nc,
            vip_scores=vip,
            x_loadings=pls.x_loadings_.copy(),
            x_scores=pls.x_scores_.copy(),
            y_loadings=pls.y_loadings_.copy(),
            x_weights=pls.x_weights_.copy(),
            explained_variance_x=ev_x,
            explained_variance_y=ev_y,
        )
        self._fit_result = result

        log.info(
            "PLSCalibration.fit: n=%d, features=%d, n_comp=%d, "
            "R²_cal=%.4f, Q²=%.4f, RMSECV=%.4f",
            n_samples, n_features, n_comp, r2_cal, q2, rmsecv,
        )
        return result

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict concentrations from spectra.

        Parameters
        ----------
        X :
            Spectral matrix, shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Predicted concentrations, shape ``(n_samples,)``.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return cast(np.ndarray, self._model.predict(X_arr).ravel())  # type: ignore[union-attr]

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        method: str = "jackknife",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict concentrations with jack-knife uncertainty estimates.

        Parameters
        ----------
        X :
            Spectral matrix, shape ``(n_samples, n_features)``.
        method :
            ``'jackknife'`` uses leave-one-out residuals from training to
            estimate local prediction variance.

        Returns
        -------
        y_pred : np.ndarray
            Point predictions, shape ``(n_samples,)``.
        y_std : np.ndarray
            Estimated standard deviation (same units as y), shape ``(n_samples,)``.
        """
        from sklearn.cross_decomposition import PLSRegression

        self._check_fitted()
        X_train: np.ndarray = self._X_train  # type: ignore[assignment]
        y_train: np.ndarray = self._y_train  # type: ignore[assignment]
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        y_pred = self._model.predict(X_arr).ravel()  # type: ignore[union-attr]

        if method == "jackknife":
            # Jack-knife: RMSECV gives a conservative LOO uncertainty estimate
            n = len(X_train)
            loo_resids: list[float] = []
            nc = self._model.n_components  # type: ignore[union-attr]
            for i in range(n):
                idx = [j for j in range(n) if j != i]
                pls_loo = PLSRegression(n_components=nc, scale=self.scale)
                pls_loo.fit(X_train[idx], y_train[idx])
                pred_i = float(pls_loo.predict(X_train[[i]]).ravel()[0])
                loo_resids.append(pred_i - float(y_train[i]))
            rmsecv = float(np.sqrt(np.mean(np.array(loo_resids) ** 2)))
            # Broadcast: same uncertainty for all new predictions
            y_std = np.full(len(y_pred), rmsecv)
        else:
            raise ValueError(f"Unknown uncertainty method: {method!r}. Use 'jackknife'.")

        return y_pred, y_std

    # ------------------------------------------------------------------
    # Results / diagnostics
    # ------------------------------------------------------------------

    @property
    def fit_result(self) -> PLSFitResult:
        """The :class:`PLSFitResult` from the last :meth:`fit` call."""
        self._check_fitted()
        return self._fit_result  # type: ignore[return-value]

    @property
    def vip_scores(self) -> np.ndarray:
        """VIP scores from the last fit, shape ``(n_features,)``."""
        return self.fit_result.vip_scores

    @property
    def informative_wavelength_mask(self, threshold: float = 1.0) -> np.ndarray:
        """Boolean mask: True where VIP ≥ ``threshold`` (default 1.0)."""
        return self.vip_scores >= threshold

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a flat dict of all calibration metrics (for reports/export)."""
        r = self.fit_result
        return {
            "method": "PLS-1 (sklearn PLSRegression)",
            "n_components": r.n_components,
            "optimal_n_components": r.optimal_n_components,
            "n_samples": r.n_samples,
            "n_features": r.n_features,
            "r2_calibration": r.r2_calibration,
            "rmsec": r.rmsec,
            "q2_crossvalidated": r.q2,
            "rmsecv": r.rmsecv,
            "pearson_r": r.pearson_r,
            "bias": r.bias,
            "n_vip_above_1": int((r.vip_scores > 1.0).sum()),
            "cv_strategy": "LOO" if self.cv_folds == -1 else f"{self.cv_folds}-fold",
            "scale": self.scale,
            "references": [
                "Wold et al. (2001) Chemom. Intell. Lab. Syst. 58, 109–130",
                "ICH Q2(R1) §5.2–5.5 (multivariate calibration)",
                "ASTM E1655-17 (NIR multivariate quantitative analysis)",
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_vip(pls: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute VIP (Variable Importance in Projection) scores.

        VIP_j = sqrt( p * sum_h(w_hj^2 * SS_h) / sum_h(SS_h) )

        where ``p`` = number of features, ``w_hj`` = weight of variable j in
        component h (normalised), and ``SS_h`` = sum of squares of ŷ explained
        by component h.

        Reference
        ---------
        Chong, I. & Jun, C. (2005). Performance of some variable selection
        methods when multicollinearity is present. Chemometrics and
        Intelligent Laboratory Systems, 78(1–2), 103–112.
        """
        T = pls.x_scores_          # (n_samples, n_comp)
        W = pls.x_weights_         # (n_features, n_comp)
        Q = pls.y_loadings_        # (n_targets, n_comp) or (n_comp,)

        n_features = W.shape[0]
        n_comp = W.shape[1]

        # Normalize weights per component
        W_norm = W / np.linalg.norm(W, axis=0, keepdims=True)

        # Sum of squares of y explained by each component
        if Q.ndim == 2:
            q = Q[0]  # PLS-1 has one y target
        else:
            q = Q.ravel()

        SS = np.array(
            [float(np.dot(T[:, h], T[:, h]) * q[h] ** 2) for h in range(n_comp)]
        )
        total_SS = float(SS.sum())
        if total_SS <= 0:
            return np.ones(n_features)

        vip = np.sqrt(n_features * np.dot(W_norm ** 2, SS) / total_SS)
        return cast(np.ndarray, vip)

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "PLSCalibration has not been fitted yet. Call .fit(X, y) first."
            )
