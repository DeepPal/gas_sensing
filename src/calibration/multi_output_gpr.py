"""
src.calibration.multi_output_gpr
==================================
Multi-output Gaussian Process Regression for simultaneous multi-analyte
concentration estimation.

Architecture
------------
Two complementary approaches are provided:

1. **IndependentMultiOutputGPR** — fits one GPR per analyte independently.
   Each GPR maps the full peak-shift feature vector Δλ ∈ ℝ^M to a single
   analyte concentration.  Simple, interpretable, and often sufficient when
   cross-interference is handled at the sensitivity matrix level.

2. **JointMultiOutputGPR** — wraps sklearn's MultiOutputRegressor around a
   shared GPR kernel. Jointly predicts all analyte concentrations from the
   same feature vector in one call.  Enables consistent uncertainty estimates
   across analytes.

Both expose the same ``predict()`` API returning
``(concentrations_dict, uncertainties_dict)``.

Input features
--------------
The input X to both models is the peak-shift vector:

    X = [Δλ_0, Δλ_1, …, Δλ_{M-1}]   shape (n_samples, M)

Optionally augmented with kinetic features (τ₆₃, k_on) if available:

    X = [Δλ_0, …, Δλ_{M-1}, τ₆₃_0, …, k_on_0, …]

This fusion is the core novelty claim of the platform.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def _default_kernel() -> object:
    """Physics-motivated kernel for optical sensor calibration.

    Matérn 5/2: smooth but not infinitely differentiable — appropriate for
    Langmuir calibration curves which are C² but not analytic.
    WhiteKernel absorbs aleatoric noise (shot noise + fit residuals).
    """
    return ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))


# ---------------------------------------------------------------------------
# Independent GPR per analyte
# ---------------------------------------------------------------------------


class IndependentMultiOutputGPR:
    """One independent GPR per analyte, all sharing the same kernel type.

    This is the baseline multi-analyte GPR. It provides per-analyte
    posterior uncertainty estimates independently.

    Parameters
    ----------
    analytes:
        List of analyte names.
    n_restarts_optimizer:
        Number of kernel hyperparameter restarts (trade-off: accuracy vs time).
    random_state:
        Global random seed.
    """

    def __init__(
        self,
        analytes: list[str],
        n_restarts_optimizer: int = 5,
        random_state: int = 42,
    ) -> None:
        self._analytes = list(analytes)
        self._n_analytes = len(analytes)
        self._scaler = StandardScaler()
        self._gprs: dict[str, GaussianProcessRegressor] = {
            name: GaussianProcessRegressor(
                kernel=_default_kernel(),
                n_restarts_optimizer=n_restarts_optimizer,
                random_state=random_state + i,
                normalize_y=True,
            )
            for i, name in enumerate(analytes)
        }
        self.is_fitted: bool = False
        self._feature_names: list[str] = []

    # ── Fitting ───────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fit all per-analyte GPRs.

        Parameters
        ----------
        X:
            Feature matrix, shape (n_samples, n_features).
            Typically [Δλ_0, …, Δλ_{M-1}] — one column per spectral peak.
            Optionally extended with kinetic features.
        Y:
            Target concentrations, shape (n_samples, n_analytes).
            Column order must match ``analytes`` list.
        feature_names:
            Optional list of feature names (for logging/inspection).

        Returns
        -------
        dict
            Per-analyte log-marginal-likelihood values.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        if Y.shape[1] != self._n_analytes:
            raise ValueError(
                f"Y has {Y.shape[1]} columns but {self._n_analytes} analytes registered."
            )

        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        X_s = self._scaler.fit_transform(X)

        results: dict[str, Any] = {}
        for i, (name, gpr) in enumerate(self._gprs.items()):
            gpr.fit(X_s, Y[:, i])
            lml = float(gpr.log_marginal_likelihood(gpr.kernel_.theta))
            log.debug("GPR[%s] fitted — log-marginal-likelihood = %.3f", name, lml)
            results[name] = {"log_marginal_likelihood": lml, "n_train": len(Y)}

        self.is_fitted = True
        return results

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(
        self,
        X: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Predict concentrations and uncertainties.

        Parameters
        ----------
        X:
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        means:
            Dict mapping analyte name → predicted concentration array (ppm).
        stds:
            Dict mapping analyte name → 1-sigma posterior std (ppm).
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_s = self._scaler.transform(X)

        means: dict[str, np.ndarray] = {}
        stds: dict[str, np.ndarray] = {}
        for name, gpr in self._gprs.items():
            mu, sigma = gpr.predict(X_s, return_std=True)
            means[name] = mu
            stds[name] = sigma

        return means, stds

    def predict_single(
        self,
        x: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Convenience: predict for one sample, return scalar dicts."""
        means, stds = self.predict(x.reshape(1, -1))
        return (
            {k: float(v[0]) for k, v in means.items()},
            {k: float(v[0]) for k, v in stds.items()},
        )

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "analytes": self._analytes,
            "gprs": self._gprs,
            "scaler": self._scaler,
            "is_fitted": self.is_fitted,
            "feature_names": self._feature_names,
        }, path)

    @classmethod
    def load(cls, path: str) -> IndependentMultiOutputGPR:
        import joblib
        state = joblib.load(path)
        obj = cls(state["analytes"])
        obj._gprs = state["gprs"]
        obj._scaler = state["scaler"]
        obj.is_fitted = state["is_fitted"]
        obj._feature_names = state.get("feature_names", [])
        return obj


# ---------------------------------------------------------------------------
# Joint multi-output GPR (sklearn MultiOutputRegressor wrapper)
# ---------------------------------------------------------------------------


class JointMultiOutputGPR:
    """Joint multi-output GPR using sklearn's MultiOutputRegressor.

    Predicts all analyte concentrations in a single call. Compared to
    IndependentMultiOutputGPR, this shares the same feature scaler and
    allows easy extension to cross-covariance kernels.

    Parameters
    ----------
    analytes:
        List of analyte names.
    n_restarts_optimizer:
        Kernel hyperparameter restarts per analyte.
    random_state:
        Global random seed.
    """

    def __init__(
        self,
        analytes: list[str],
        n_restarts_optimizer: int = 5,
        random_state: int = 42,
    ) -> None:
        self._analytes = list(analytes)
        self._scaler = StandardScaler()
        base_gpr = GaussianProcessRegressor(
            kernel=_default_kernel(),
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
            normalize_y=True,
        )
        self._model = MultiOutputRegressor(base_gpr, n_jobs=1)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, Y: np.ndarray) -> dict[str, Any]:
        """Fit the joint model.

        X: (n_samples, n_features)
        Y: (n_samples, n_analytes)  — column order matches ``analytes``
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        X_s = self._scaler.fit_transform(X)
        self._model.fit(X_s, Y)
        self.is_fitted = True
        return {"n_analytes": len(self._analytes), "n_train": len(Y)}

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Predict concentrations.

        Returns dicts mapping analyte name → array(n_samples).
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_s = self._scaler.transform(X)

        if return_std:
            # MultiOutputRegressor: iterate estimators_ for std
            means_list = []
            stds_list = []
            for est in self._model.estimators_:
                mu, sig = est.predict(X_s, return_std=True)
                means_list.append(mu)
                stds_list.append(sig)
            means_arr = np.column_stack(means_list)  # (n_samples, n_analytes)
            stds_arr = np.column_stack(stds_list)
        else:
            means_arr = self._model.predict(X_s)
            stds_arr = np.zeros_like(means_arr)

        means = {name: means_arr[:, i] for i, name in enumerate(self._analytes)}
        stds = {name: stds_arr[:, i] for i, name in enumerate(self._analytes)}
        return means, stds

    def predict_single(
        self,
        x: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Convenience: predict for one sample."""
        means, stds = self.predict(x.reshape(1, -1))
        return (
            {k: float(v[0]) for k, v in means.items()},
            {k: float(v[0]) for k, v in stds.items()},
        )

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "analytes": self._analytes,
            "model": self._model,
            "scaler": self._scaler,
            "is_fitted": self.is_fitted,
        }, path)

    @classmethod
    def load(cls, path: str) -> JointMultiOutputGPR:
        import joblib
        state = joblib.load(path)
        obj = cls(state["analytes"])
        obj._model = state["model"]
        obj._scaler = state["scaler"]
        obj.is_fitted = state["is_fitted"]
        return obj


# ---------------------------------------------------------------------------
# Feature vector builder (peak shifts + optional kinetics)
# ---------------------------------------------------------------------------


def build_feature_vector(
    peak_shifts_nm: list[float] | np.ndarray,
    tau_63: list[float] | None = None,
    tau_95: list[float] | None = None,
    k_on: list[float] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build the feature vector for GPR input.

    Core (always included):
      - Δλ per peak — equilibrium sensitivity feature

    Optional kinetic extensions (include if measured):
      - τ₆₃ per peak — time to 63% equilibrium (s); analyte-specific
      - τ₉₅ per peak — time to 95% equilibrium (s)
      - k_on per peak — apparent association rate (ppm⁻¹ s⁻¹)

    Adding kinetics is the key novelty: two analytes with the same Δλ
    can be distinguished by their different τ₆₃ at a given concentration.

    Returns
    -------
    (feature_vector, feature_names)
    """
    features: list[float] = []
    names: list[str] = []

    shifts = np.asarray(peak_shifts_nm, dtype=float)
    for j, dl in enumerate(shifts):
        features.append(float(dl))
        names.append(f"delta_lambda_{j}")

    if tau_63 is not None:
        for j, t in enumerate(tau_63):
            features.append(float(t))
            names.append(f"tau_63_{j}")

    if tau_95 is not None:
        for j, t in enumerate(tau_95):
            features.append(float(t))
            names.append(f"tau_95_{j}")

    if k_on is not None:
        for j, k in enumerate(k_on):
            features.append(float(k))
            names.append(f"k_on_{j}")

    return np.array(features, dtype=float), names
