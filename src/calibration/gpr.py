"""
src.calibration.gpr
===================
Gaussian Process Regression calibration for LSPR concentration estimation.

The GPR maps a 1-D feature (wavelength shift Δλ, or a feature vector) to
concentration in ppm, returning both a mean estimate and a 1-sigma
uncertainty — essential for LOD calculations and quality flagging.

Physics note
------------
LSPR calibration is monotone and approximately linear in the 0–5 ppm range,
but the GPR handles mild nonlinearity gracefully via its RBF kernel.  The
WhiteKernel term models aleatoric noise (shot noise + read noise).

Public API
----------
- ``GPRCalibration``   — fit / predict / save / load
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


class GPRCalibration:
    """Gaussian Process Regressor for LSPR concentration calibration.

    Wraps scikit-learn's :class:`~sklearn.gaussian_process.GaussianProcessRegressor`
    with automatic feature scaling and a physically motivated kernel.

    Example
    -------
    ::

        gpr = GPRCalibration()
        shifts = np.array([-0.1, -0.5, -1.0, -2.0])   # Δλ in nm
        concs  = np.array([0.25, 0.5,  1.0,  2.0])     # ppm
        gpr.fit(shifts.reshape(-1, 1), concs)
        mean, std = gpr.predict(np.array([[-0.75]]))
    """

    def __init__(
        self,
        random_state: int = 42,
        n_restarts_optimizer: int = 10,
    ) -> None:
        self.random_state = random_state
        self.n_restarts_optimizer = n_restarts_optimizer

        # Matérn 5/2 captures the smooth-but-not-analytic trend of Langmuir
        # calibration curves (twice differentiable, appropriate for physical
        # measurements). WhiteKernel handles observation noise.
        # length_scale=1.0 matches the StandardScaler output (unit variance).
        self.kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * Matern(
            length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5
        ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
            normalize_y=True,
        )
        self.scaler_X = StandardScaler()
        self.is_fitted: bool = False
        self._n_train: int = 0
        self._aleatoric_noise_std: float = 0.0
        """Residual noise σ from training data (aleatoric uncertainty).
        Added in quadrature to the GPR posterior std during prediction so that
        total_std = √(epistemic² + aleatoric²).  This prevents overconfident
        predictions in noisy concentration ranges where the GPR has seen many
        training points (epistemic → 0) but sensor noise remains.
        """

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, Any]:
        """Fit the GPR to training data.

        Parameters
        ----------
        X:
            Feature matrix, shape ``(n_samples, n_features)``.  For single-
            wavelength-shift calibration, shape ``(n, 1)``; pass
            ``shifts.reshape(-1, 1)``.
        y:
            Target concentrations in ppm, shape ``(n_samples,)``.

        Returns
        -------
        dict
            ``{"log_marginal_likelihood": float, "kernel_params": dict}``
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self.scaler_X.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        self._n_train = len(y)

        # Compute aleatoric (sensor) noise from training residuals.
        # This is distinct from the GPR epistemic uncertainty (posterior std):
        # epistemic → 0 near training points; aleatoric is fixed by sensor noise floor.
        # We combine both in predict() so CIs don't collapse to zero near training data.
        y_pred_train = self.model.predict(X_scaled, return_std=False)
        train_residuals = y - y_pred_train
        self._aleatoric_noise_std = float(np.std(train_residuals, ddof=1)) if len(y) > 1 else 0.0

        return {
            "log_marginal_likelihood": float(
                self.model.log_marginal_likelihood(self.model.kernel_.theta)
            ),
            "kernel_params": {
                k: float(v)
                for k, v in self.model.kernel_.get_params().items()
                if isinstance(v, (int, float))
            },
            "n_samples": self._n_train,
            "aleatoric_noise_std": round(self._aleatoric_noise_std, 6),
        }

    # Alias
    optimize_hyperparameters = fit

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict concentration (and optional uncertainty) for *X*.

        Parameters
        ----------
        X:
            Feature matrix, shape ``(n_samples, n_features)``.
        return_std:
            If ``True`` (default), also returns the posterior standard
            deviation (1-sigma uncertainty in ppm).

        Returns
        -------
        y_mean : ndarray, shape (n_samples,)
            Predicted concentrations in ppm.
        y_std : ndarray, shape (n_samples,)
            Posterior standard deviations (zeros if ``return_std=False``).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if not self.is_fitted:
            raise RuntimeError("GPRCalibration must be fitted before calling predict().")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self.scaler_X.transform(X)

        if return_std:
            y_mean, y_epistemic = self.model.predict(X_scaled, return_std=True)
            # Combine epistemic (model) and aleatoric (sensor noise) uncertainty:
            #   total_std = √(epistemic² + aleatoric²)
            # Without this, CI collapses near training points even when sensor is noisy.
            y_std = np.sqrt(y_epistemic ** 2 + self._aleatoric_noise_std ** 2)
            return y_mean, y_std
        else:
            y_mean = self.model.predict(X_scaled, return_std=False)
            return y_mean, np.zeros_like(y_mean)

    def predict_single(
        self,
        x: float,
    ) -> tuple[float | None, float | None]:
        """Convenience wrapper for a scalar input feature.

        Returns ``(concentration_ppm, uncertainty_ppm)`` or
        ``(None, None)`` if not fitted.
        """
        if not self.is_fitted:
            return None, None
        try:
            mean, std = self.predict(np.array([[x]]))
            return float(mean[0]), float(std[0])
        except Exception:
            return None, None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize this calibration to *path* using joblib.

        Parameters
        ----------
        path:
            Destination ``.joblib`` file path.
        """
        import joblib

        joblib.dump(
            {
                "model": self.model,
                "scaler_X": self.scaler_X,
                "random_state": self.random_state,
                "n_restarts_optimizer": self.n_restarts_optimizer,
                "is_fitted": self.is_fitted,
                "n_train": self._n_train,
                "aleatoric_noise_std": self._aleatoric_noise_std,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> GPRCalibration:
        """Load a serialized :class:`GPRCalibration` from *path*.

        Parameters
        ----------
        path:
            Path to a ``.joblib`` file created by :meth:`save`.

        Returns
        -------
        GPRCalibration
            Ready-to-predict instance (``is_fitted=True``).
        """
        import joblib

        state = joblib.load(path)
        obj = cls(
            random_state=state.get("random_state", 42),
            n_restarts_optimizer=state.get("n_restarts_optimizer", 10),
        )
        obj.model = state["model"]
        obj.scaler_X = state["scaler_X"]
        obj.is_fitted = state.get("is_fitted", True)
        obj._n_train = state.get("n_train", 0)
        obj._aleatoric_noise_std = state.get("aleatoric_noise_std", 0.0)
        # Sync self.kernel with the optimised kernel_ so inspecting obj.kernel
        # returns the fitted hyperparameters rather than the unfitted template.
        if obj.is_fitted and hasattr(obj.model, "kernel_"):
            obj.kernel = obj.model.kernel_
        return obj
