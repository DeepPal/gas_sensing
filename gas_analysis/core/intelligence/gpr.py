import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.preprocessing import StandardScaler


class GPRCalibration:
    """
    Gaussian Process Regressor for probabilistic gas concentration prediction.

    Attributes:
        kernel: The covariance function of the GP.
        optimizer: Optimizer for kernel hyperparameters.
        alpha: Noise level (if not using WhiteKernel).
        normalize_y: Whether to normalize target values.
    """

    def __init__(self, random_state: int = 42, n_restarts_optimizer: int = 2):
        # Kernel: Constant * RBF + WhiteNoise
        # RBF handles smooth trend, WhiteKernel handles aleatoric noise
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(
            length_scale=10.0, length_scale_bounds=(1e-2, 1e4)
        ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e1))

        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=True,
            random_state=random_state,
        )
        self.scaler_X = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict[str, object]:
        """
        Fit the GPR model.

        Args:
            X: Feature matrix (n_samples, n_features) - typically wavelengths
            y: Target concentrations (n_samples,)

        Returns:
            Dictionary with training metrics (log_marginal_likelihood)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Scale features
        X_scaled = self.scaler_X.fit_transform(X)

        self.model.fit(X_scaled, y)
        self.is_fitted = True

        return {
            "log_marginal_likelihood": float(
                self.model.log_marginal_likelihood(self.model.kernel_.theta)
            ),
            "kernel_params": self.model.kernel_.get_params(),
        }

    def predict(self, X: np.ndarray, return_std: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict concentrations with uncertainty.

        Args:
            X: Feature matrix (n_samples, n_features)
            return_std: Whether to return standard deviation of prediction.

        Returns:
            (mean_prediction, std_prediction)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.scaler_X.transform(X)

        if return_std:
            y_mean, y_std = self.model.predict(X_scaled, return_std=True)
            return y_mean, y_std
        else:
            y_mean = self.model.predict(X_scaled, return_std=False)
            return y_mean, np.zeros_like(y_mean)

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        """
        Re-fit/optimize (wrapped for clarity, same as fit in sklearn GPR).
        """
        return self.fit(X, y)
