import numpy as np


def calculate_lod_3sigma(noise_std: float, sensitivity_slope: float) -> float:
    """
    Calculate Limit of Detection (LOD) using the 3*sigma / slope method.

    LOD_theoretical = 3 * sigma_noise / slope

    Args:
        noise_std (sigma): Standard deviation of the baseline signal (noise).
        sensitivity_slope (m): Slope of the calibration curve (Signal/ppm).

    Returns:
        LOD in ppm (or concentration units).
    """
    if sensitivity_slope == 0:
        return np.inf
    return (3.0 * noise_std) / abs(sensitivity_slope)


def calculate_sensitivity(
    concentrations: np.ndarray, responses: np.ndarray
) -> tuple[float, float, float]:
    """
    Calculate sensitivity (slope) from calibration data using linear regression.

    Returns:
        (slope, intercept, r_squared)
    """
    from scipy.stats import linregress

    slope, intercept, r_value, p_value, std_err = linregress(concentrations, responses)
    return slope, intercept, r_value**2
