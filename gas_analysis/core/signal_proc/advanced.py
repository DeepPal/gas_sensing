from typing import Optional

import numpy as np
import pywt
from scipy import sparse
from scipy.sparse.linalg import spsolve


def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: Optional[int] = None,
    mode: str = "soft",
    sigma_est: str = "mad",
) -> np.ndarray:
    """
    Denoise signal using Discrete Wavelet Transform (DWT).

    Args:
        signal: Input 1D array.
        wavelet: Wavelet name (e.g., 'db4', 'sym8').
        level: Decomposition level. If None, calculated from signal length.
        mode: Thresholding mode ('soft' or 'hard').
        sigma_est: Method to estimate noise sigma ('mad' or 'std').

    Returns:
        Denoised signal.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return signal

    # Calculate level if not provided
    if level is None:
        level = int(np.floor(np.log2(signal.size)))

    # Decompose
    coeff = pywt.wavedec(signal, wavelet, mode="per", level=level)

    # Estimate noise from detailed coefficients at the highest resolution (last level)
    # Median Absolute Deviation (MAD) is robust to outliers/signal
    sigma = 0.0
    if sigma_est == "mad":
        detail = coeff[-1]
        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
    elif sigma_est == "std":
        sigma = np.std(coeff[-1])

    # Universal threshold (VisuShrink)
    uthresh = sigma * np.sqrt(2 * np.log(signal.size))

    # Thresholding
    coeff_thresh = []
    coeff_thresh.append(coeff[0])  # Keep approximation coefficients as is
    for i in range(1, len(coeff)):
        coeff_thresh.append(pywt.threshold(coeff[i], uthresh, mode=mode))

    # Reconstruct
    return pywt.waverec(coeff_thresh, wavelet, mode="per")


def als_baseline(
    signal: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10
) -> np.ndarray:
    """
    Asymmetric Least Squares Smoothing for baseline correction.

    Args:
        signal: Input 1D signal.
        lam: Smoothness parameter (lambda). Larger = smoother baseline.
        p: Asymmetry parameter. 0 < p < 1. Usually 0.001-0.1 for baselines below signal.
        niter: Number of iterations.

    Returns:
        Estimated baseline.
    """
    signal = np.asarray(signal, dtype=float)
    L = len(signal)
    if L < 3:
        return np.zeros_like(signal)

    # Construct discrete difference operator matrix D (second order difference)
    # D is (L-2, L). D.T is (L, L-2).
    # Used in penalty term: lambda * ||D * z||^2
    # Normal equations: (W + lambda * D.T * D) * z = W * y

    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    # D.dot(D.T) yields the pentadiagonal matrix needed for smoothing penalty
    # But wait, we need (L,L) matrix.
    # The penalty matrix P = D.T @ D  where D is (L-2, L).
    # Let's construct D as (L-2, L).
    #   Row 0: [1, -2, 1, 0, ..., 0]
    #   Row 1: [0, 1, -2, 1, ..., 0]
    # scipy.sparse.diags creates matrix of shape (L, L-2) if we pass shape.
    # Actually, let's use the explicit difference structure.

    E = sparse.eye(L, format="csc")
    D = E[1:] - E[:-1]  # 1st difference (L-1, L)
    D = D[1:] - D[:-1]  # 2nd difference (L-2, L)

    # Precompute constant part
    DT_D = D.T.dot(D)

    w = np.ones(L)
    z = np.zeros(L)

    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * DT_D
        z = spsolve(Z, w * signal)
        w = p * (signal > z) + (1 - p) * (signal < z)

    return z
