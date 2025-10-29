import numpy as np
import pandas as pd


def generate_synthetic_spectra(
    num_samples: int = 100,
    wavelength_range: tuple = (200, 800),
    num_points: int = 1000,
    noise_level: float = 0.02,
    baseline_drift: bool = True,
    peak_count: int = 3,
) -> pd.DataFrame:
    """Generate synthetic spectra with controllable features for ML training.

    Args:
        num_samples: Number of spectra to generate.
        wavelength_range: (min, max) wavelength in nm.
        num_points: Number of wavelength points per spectrum.
        noise_level: Standard deviation of Gaussian noise.
        baseline_drift: Add polynomial baseline drift.
        peak_count: Number of Gaussian peaks per spectrum.

    Returns:
        DataFrame with columns: sample_id, wavelength, intensity.
    """
    wl = np.linspace(wavelength_range[0], wavelength_range[1], num_points)
    rows = []

    for i in range(num_samples):
        # Base signal
        intensity = np.zeros_like(wl)

        # Add peaks
        for _ in range(peak_count):
            amp = np.random.uniform(0.5, 2.0)
            center = np.random.uniform(wl[100], wl[-100])
            width = np.random.uniform(5, 20)
            intensity += amp * np.exp(-0.5 * ((wl - center) / width) ** 2)

        # Add baseline drift
        if baseline_drift:
            poly_coeffs = np.random.uniform(-0.001, 0.001, 3)
            baseline = np.polyval(poly_coeffs, wl - wl.mean())
            intensity += baseline

        # Add noise
        noise = np.random.normal(0, noise_level, size=wl.shape)
        intensity += noise

        # Clip to positive
        intensity = np.clip(intensity, 0, None)

        for j in range(len(wl)):
            rows.append({
                'sample_id': i,
                'wavelength': wl[j],
                'intensity': intensity[j],
            })

    return pd.DataFrame(rows)
