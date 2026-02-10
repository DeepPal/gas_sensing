"""
Spectral Feature Engineering for Weak Absorber Detection

This module implements the spectral feature engineering methodology from:
"1-s2.0-S0925400525000607-main.pdf" - First-derivative convolution with composite spectra

Key Innovation:
    Convolution of first-derivative with original spectra enhances weak absorber 
    (acetone) detection by 10× while reducing dynamic range from [0, 2.7] to [-0.02, 0.06]

Mathematical Framework:
    1. Beer-Lambert Law: α_ν = -ln(I_t/I_0) = σ(T,P,ν)·n·L
    2. First Derivative: dα_ν/dν eliminates flat baseline
    3. Convolution: C(α_ν, dα_ν/dν) = ∫α_ν(τ)·dα_ν/dν(ν-τ)dτ

Reference benchmark (ZnO-coated NCF sensor):
    - Target: Improve LoD from 3.26 ppm to <1 ppm (77% reduction)
    - Target: Improve sensitivity from 0.116 nm/ppm to 0.156 nm/ppm (35% improvement)
    - Target: Achieve 96% clinical accuracy for diabetes screening

Author: ML-Enhanced Gas Sensing Pipeline
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, convolve
from scipy.stats import linregress, spearmanr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Optional, Tuple, Union
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class SpectralFeatureEngineering:
    """
    Implements spectral feature engineering for weak absorber detection.
    
    Based on first-derivative convolution methodology that:
    1. Eliminates 60-80% of unnecessary baseline data
    2. Reduces dynamic range by 34× for better CNN learning
    3. Enhances SNR for weak absorbers by ~10×
    
    Attributes:
        wavelength (np.ndarray): Wavelength array in nm
        spectra (np.ndarray): Absorbance spectra matrix (n_samples, n_points)
        scaler (StandardScaler): Fitted scaler for normalization
    """
    
    def __init__(self, wavelength: np.ndarray = None, spectra: np.ndarray = None):
        """
        Initialize spectral feature engineering processor.
        
        Parameters
        ----------
        wavelength : array-like, shape (n_points,)
            Wavelength array in nm
        spectra : array-like, shape (n_samples, n_points) or (n_points,)
            Absorbance spectra matrix (can be single spectrum)
        """
        self.wavelength = np.asarray(wavelength) if wavelength is not None else None
        self.spectra = self._ensure_2d(spectra) if spectra is not None else None
        self.scaler = StandardScaler()
        self._derivatives = None
        self._convolved = None
        self._convolved_normalized = None
        
    def _ensure_2d(self, arr: np.ndarray) -> np.ndarray:
        """Ensure array is 2D (n_samples, n_points)."""
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr
    
    @property
    def n_samples(self) -> int:
        """Number of spectral samples."""
        return self.spectra.shape[0] if self.spectra is not None else 0
    
    @property
    def n_points(self) -> int:
        """Number of wavelength points."""
        return self.spectra.shape[1] if self.spectra is not None else 0
    
    def set_data(self, wavelength: np.ndarray, spectra: np.ndarray):
        """
        Set new data for processing.
        
        Parameters
        ----------
        wavelength : array-like
            Wavelength array
        spectra : array-like
            Spectra matrix
        """
        self.wavelength = np.asarray(wavelength)
        self.spectra = self._ensure_2d(spectra)
        # Reset cached results
        self._derivatives = None
        self._convolved = None
        self._convolved_normalized = None
        
    def calculate_absorbance(self, I: np.ndarray, I0: np.ndarray) -> np.ndarray:
        """
        Calculate absorbance from raw intensity data.
        
        Beer-Lambert Law: A = -log10(I/I0)
        
        Parameters
        ----------
        I : np.ndarray
            Transmitted intensity
        I0 : np.ndarray
            Reference (incident) intensity
            
        Returns
        -------
        absorbance : np.ndarray
            Absorbance values
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            A = -np.log10(I / (I0 + 1e-10))
            A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        return A
    
    def calculate_first_derivative(
        self,
        spectra: np.ndarray = None,
        window_length: int = 7,
        polyorder: int = 2
    ) -> np.ndarray:
        """
        Calculate first-order derivative using Savitzky-Golay filter.
        
        Preserves spectral features better than simple finite differences.
        Key benefit: dα_ν/dν = 0 for flat baseline regions.
        
        Parameters
        ----------
        spectra : np.ndarray, optional
            Spectra to differentiate (uses self.spectra if None)
        window_length : int
            Window length for Savitzky-Golay filter (must be odd)
        polyorder : int
            Polynomial order (typically 2)
            
        Returns
        -------
        derivatives : np.ndarray
            First derivative spectra (n_samples, n_points)
        """
        if spectra is None:
            spectra = self.spectra
        spectra = self._ensure_2d(spectra)
        
        # Ensure window_length is odd and smaller than spectrum length
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, spectra.shape[1] - 1)
        if window_length < 3:
            window_length = 3
            
        derivatives = np.zeros_like(spectra)
        
        for i in range(spectra.shape[0]):
            derivatives[i] = savgol_filter(
                spectra[i],
                window_length=window_length,
                polyorder=polyorder,
                deriv=1
            )
        
        self._derivatives = derivatives
        return derivatives
    
    def convolution_feature_engineering(
        self,
        spectra: np.ndarray = None,
        derivative_window: int = 7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply convolution of first derivative with original spectra.
        
        Mathematical basis:
            C(α_ν, dα_ν/dν) = ∫α_ν(τ) · dα_ν/dν(ν-τ) dτ
        
        Physical interpretation:
            - Peaks occur where spectral magnitude × spectral slope is maximum
            - For weak absorbers: feature enhancement ≈ 10×
        
        Parameters
        ----------
        spectra : np.ndarray, optional
            Spectra to process (uses self.spectra if None)
        derivative_window : int
            Window for derivative calculation
            
        Returns
        -------
        convolved : np.ndarray
            Convolved spectra (n_samples, 2*n_points-1)
        derivatives : np.ndarray
            First derivative spectra (n_samples, n_points)
        """
        if spectra is None:
            spectra = self.spectra
        spectra = self._ensure_2d(spectra)
        
        # Calculate first derivative
        derivatives = self.calculate_first_derivative(spectra, window_length=derivative_window)
        
        n_samples, n_points = spectra.shape
        convolved = np.zeros((n_samples, 2 * n_points - 1))
        
        for i in range(n_samples):
            convolved[i] = convolve(spectra[i], derivatives[i], mode='full')
        
        self._convolved = convolved
        return convolved, derivatives
    
    def normalize_features(
        self,
        features: np.ndarray = None,
        method: str = 'standard'
    ) -> Tuple[np.ndarray, object]:
        """
        Normalize features for better CNN learning.
        
        Dynamic range reduction:
            - Raw spectra: [0, 2.7]
            - After convolution + normalization: [-0.02, 0.06]
            - Reduction factor: ~34×
        
        Parameters
        ----------
        features : np.ndarray, optional
            Features to normalize (uses self._convolved if None)
        method : str
            'standard' (z-score) or 'minmax' (0-1 scaling)
            
        Returns
        -------
        normalized : np.ndarray
            Normalized features
        scaler : object
            Fitted scaler for inverse transform
        """
        if features is None:
            features = self._convolved
        features = self._ensure_2d(features)
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        normalized = scaler.fit_transform(features)
        self._convolved_normalized = normalized
        self.scaler = scaler
        
        return normalized, scaler
    
    def extract_peak_features(
        self,
        spectra: np.ndarray = None,
        roi_range: Tuple[float, float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract peak-related features from spectra.
        
        Parameters
        ----------
        spectra : np.ndarray, optional
            Spectra to analyze
        roi_range : tuple, optional
            (min_wavelength, max_wavelength) for ROI
            
        Returns
        -------
        features : dict
            Dictionary with peak_wavelengths, peak_intensities, fwhm, etc.
        """
        if spectra is None:
            spectra = self.spectra
        spectra = self._ensure_2d(spectra)
        
        if roi_range is not None and self.wavelength is not None:
            roi_min, roi_max = roi_range
            mask = (self.wavelength >= roi_min) & (self.wavelength <= roi_max)
            wl_roi = self.wavelength[mask]
            spectra_roi = spectra[:, mask]
        else:
            wl_roi = self.wavelength
            spectra_roi = spectra
        
        n_samples = spectra_roi.shape[0]
        peak_wavelengths = np.zeros(n_samples)
        peak_intensities = np.zeros(n_samples)
        centroids = np.zeros(n_samples)
        
        for i in range(n_samples):
            signal = spectra_roi[i]
            
            # Peak position (maximum intensity)
            peak_idx = np.argmax(signal)
            peak_wavelengths[i] = wl_roi[peak_idx]
            peak_intensities[i] = signal[peak_idx]
            
            # Centroid (intensity-weighted center)
            weights = np.abs(signal - np.min(signal))
            total_weight = np.sum(weights)
            if total_weight > 1e-10:
                centroids[i] = np.sum(wl_roi * weights) / total_weight
            else:
                centroids[i] = wl_roi[peak_idx]
        
        return {
            'peak_wavelengths': peak_wavelengths,
            'peak_intensities': peak_intensities,
            'centroids': centroids
        }
    
    def calculate_wavelength_shift(
        self,
        spectra: np.ndarray,
        reference_spectrum: np.ndarray,
        roi_range: Tuple[float, float] = None,
        method: str = 'centroid'
    ) -> np.ndarray:
        """
        Calculate wavelength shift relative to reference.
        
        Parameters
        ----------
        spectra : np.ndarray
            Sample spectra
        reference_spectrum : np.ndarray
            Reference spectrum (air/baseline)
        roi_range : tuple
            ROI for peak detection
        method : str
            'centroid', 'peak', or 'xcorr'
            
        Returns
        -------
        shifts : np.ndarray
            Wavelength shifts (nm) relative to reference
        """
        spectra = self._ensure_2d(spectra)
        reference_spectrum = self._ensure_2d(reference_spectrum)
        
        # Get reference peak position
        ref_features = self.extract_peak_features(reference_spectrum, roi_range)
        if method == 'centroid':
            ref_pos = ref_features['centroids'][0]
        else:
            ref_pos = ref_features['peak_wavelengths'][0]
        
        # Get sample peak positions
        sample_features = self.extract_peak_features(spectra, roi_range)
        if method == 'centroid':
            sample_pos = sample_features['centroids']
        else:
            sample_pos = sample_features['peak_wavelengths']
        
        shifts = sample_pos - ref_pos
        return shifts
    
    def calculate_sensitivity(
        self,
        concentrations: np.ndarray,
        wavelength_shifts: np.ndarray,
        remove_outliers: bool = True,
        outlier_threshold: float = 2.5
    ) -> Dict:
        """
        Calculate linear sensitivity: S = ΔWavelength / ΔConcentration
        
        IUPAC definition for optical sensors.
        
        Parameters
        ----------
        concentrations : np.ndarray
            Concentration values (ppm)
        wavelength_shifts : np.ndarray
            Measured wavelength shifts (nm)
        remove_outliers : bool
            Whether to apply outlier rejection
        outlier_threshold : float
            Z-score threshold for outlier detection
            
        Returns
        -------
        result : dict
            sensitivity (nm/ppm), r_squared, slope, intercept, residuals
        """
        conc = np.asarray(concentrations).flatten()
        shifts = np.asarray(wavelength_shifts).flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(conc) | np.isnan(shifts))
        conc = conc[valid_mask]
        shifts = shifts[valid_mask]
        
        if len(conc) < 2:
            return {
                'sensitivity': np.nan,
                'r_squared': np.nan,
                'slope': np.nan,
                'intercept': np.nan,
                'n_points': 0
            }
        
        # Optional outlier rejection
        if remove_outliers and len(conc) >= 4:
            # Fit initial line
            z = np.polyfit(conc, shifts, 1)
            predicted = np.polyval(z, conc)
            residuals = shifts - predicted
            
            # Z-score based rejection
            z_scores = np.abs((residuals - np.mean(residuals)) / (np.std(residuals) + 1e-10))
            inlier_mask = z_scores < outlier_threshold
            
            if np.sum(inlier_mask) >= 2:
                conc = conc[inlier_mask]
                shifts = shifts[inlier_mask]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(conc, shifts)
        r_squared = r_value ** 2
        
        # Spearman correlation for monotonicity check
        spearman_r, spearman_p = spearmanr(conc, shifts)
        
        return {
            'sensitivity': slope,
            'r_squared': r_squared,
            'r_value': r_value,
            'slope': slope,
            'intercept': intercept,
            'std_error': std_err,
            'p_value': p_value,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_points': len(conc)
        }
    
    def full_feature_engineering_pipeline(
        self,
        spectra: np.ndarray = None,
        derivative_window: int = 7,
        normalize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Run complete feature engineering pipeline.
        
        Steps:
        1. Calculate first derivative
        2. Convolve with original spectra
        3. Normalize (optional)
        
        Parameters
        ----------
        spectra : np.ndarray, optional
            Input spectra
        derivative_window : int
            Window for derivative calculation
        normalize : bool
            Whether to normalize output
            
        Returns
        -------
        result : dict
            original, derivatives, convolved, normalized features
        """
        if spectra is None:
            spectra = self.spectra
        spectra = self._ensure_2d(spectra)
        
        # Step 1 & 2: Derivative and convolution
        convolved, derivatives = self.convolution_feature_engineering(
            spectra, derivative_window
        )
        
        result = {
            'original': spectra,
            'derivatives': derivatives,
            'convolved': convolved
        }
        
        # Step 3: Normalization
        if normalize:
            normalized, scaler = self.normalize_features(convolved)
            result['normalized'] = normalized
            result['scaler'] = scaler
        
        # Add statistics
        result['stats'] = {
            'original_range': (float(spectra.min()), float(spectra.max())),
            'convolved_range': (float(convolved.min()), float(convolved.max())),
            'dynamic_range_reduction': float((spectra.max() - spectra.min()) / 
                                             (convolved.max() - convolved.min() + 1e-10))
        }
        
        return result


class DetectionLimitCalculator:
    """
    Calculates detection limit using standard IUPAC methodology.
    
    LoD = 3.3 × σ / S
    LoQ = 10 × σ / S
    
    Where:
        σ = standard deviation of blank signal
        S = sensitivity (slope of calibration curve)
    
    Reference: IUPAC Compendium of Chemical Terminology
    """
    
    def __init__(self, blank_signals: np.ndarray = None, sensitivity: float = None):
        """
        Initialize detection limit calculator.
        
        Parameters
        ----------
        blank_signals : np.ndarray
            Baseline signal measurements (no analyte)
        sensitivity : float
            Sensitivity in nm/ppm (or appropriate units)
        """
        self.blank_signals = np.asarray(blank_signals) if blank_signals is not None else None
        self.sensitivity = sensitivity
        self.std_dev = None
        self.lod = None
        self.loq = None
        
    def set_data(self, blank_signals: np.ndarray, sensitivity: float):
        """Set new data for calculation."""
        self.blank_signals = np.asarray(blank_signals)
        self.sensitivity = sensitivity
        self.std_dev = None
        self.lod = None
        self.loq = None
        
    def calculate_noise_std_dev(self, method: str = 'standard') -> float:
        """
        Calculate noise standard deviation from blank measurements.
        
        Parameters
        ----------
        method : str
            'standard': Simple standard deviation
            'white_noise': RMS of first differences
            'allan': Allan deviation estimate
            
        Returns
        -------
        std_dev : float
            Noise standard deviation
        """
        if self.blank_signals is None or len(self.blank_signals) < 2:
            return np.nan
            
        if method == 'standard':
            self.std_dev = np.std(self.blank_signals, ddof=1)
            
        elif method == 'white_noise':
            # RMS of first differences (less sensitive to drift)
            diffs = np.diff(self.blank_signals)
            self.std_dev = np.sqrt(np.mean(diffs ** 2)) / np.sqrt(2)
            
        elif method == 'allan':
            # Simplified Allan deviation estimate
            self.std_dev = self._estimate_allan_deviation()
            
        return self.std_dev
    
    def _estimate_allan_deviation(self, tau: int = 1) -> float:
        """
        Estimate Allan deviation for noise characterization.
        
        Allan deviation at τ=1 sample is related to white noise level.
        """
        signals = self.blank_signals
        if len(signals) < 2 * tau:
            return np.std(signals, ddof=1)
            
        n = len(signals) - tau
        allan_var = np.sum((signals[tau:] - signals[:-tau]) ** 2) / (2 * n)
        return np.sqrt(allan_var)
    
    def calculate_lod_iupac(self) -> float:
        """
        Calculate LoD using IUPAC definition.
        
        LoD = 3.3 × σ / S
        
        Returns
        -------
        lod : float
            Detection limit in concentration units (ppm)
        """
        if self.std_dev is None:
            self.calculate_noise_std_dev()
            
        if self.sensitivity is None or abs(self.sensitivity) < 1e-10:
            return np.nan
            
        self.lod = (3.3 * self.std_dev) / abs(self.sensitivity)
        return self.lod
    
    def calculate_loq_iupac(self) -> float:
        """
        Calculate Limit of Quantification using IUPAC definition.
        
        LoQ = 10 × σ / S
        
        Returns
        -------
        loq : float
            Quantification limit in concentration units (ppm)
        """
        if self.std_dev is None:
            self.calculate_noise_std_dev()
            
        if self.sensitivity is None or abs(self.sensitivity) < 1e-10:
            return np.nan
            
        self.loq = (10 * self.std_dev) / abs(self.sensitivity)
        return self.loq
    
    def calculate_allan_deviation_curve(
        self,
        signals: np.ndarray = None,
        max_tau: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Allan deviation as function of integration time.
        
        Useful for identifying noise types:
            - τ^(-1/2) slope: white noise dominated
            - τ^0 (flat): flicker noise dominated
            - τ^(+1/2) slope: random walk
        
        Parameters
        ----------
        signals : np.ndarray
            Time series signals (uses blank_signals if None)
        max_tau : int
            Maximum averaging time (samples)
            
        Returns
        -------
        tau_values : np.ndarray
            Averaging times
        allan_dev : np.ndarray
            Allan deviation values
        """
        if signals is None:
            signals = self.blank_signals
        signals = np.asarray(signals)
        
        if len(signals) < 10:
            return np.array([]), np.array([])
            
        if max_tau is None:
            max_tau = len(signals) // 3
            
        tau_values = np.arange(1, max_tau + 1)
        allan_dev = []
        
        for tau in tau_values:
            n_clusters = len(signals) // tau
            if n_clusters < 2:
                allan_dev.append(np.nan)
                continue
                
            # Reshape into clusters
            trimmed = signals[:n_clusters * tau]
            clusters = trimmed.reshape(n_clusters, tau)
            cluster_means = np.mean(clusters, axis=1)
            
            # Allan variance
            if len(cluster_means) > 1:
                allan_var = np.mean(np.diff(cluster_means) ** 2) / 2
                allan_dev.append(np.sqrt(allan_var))
            else:
                allan_dev.append(np.nan)
        
        return tau_values, np.array(allan_dev)
    
    def full_analysis(self, noise_method: str = 'white_noise') -> Dict:
        """
        Run complete detection limit analysis.
        
        Returns
        -------
        result : dict
            Complete LoD/LoQ analysis with statistics
        """
        std_dev = self.calculate_noise_std_dev(method=noise_method)
        lod = self.calculate_lod_iupac()
        loq = self.calculate_loq_iupac()
        
        return {
            'noise_std_dev': float(std_dev) if not np.isnan(std_dev) else None,
            'sensitivity_nm_per_ppm': float(self.sensitivity) if self.sensitivity else None,
            'lod_ppm': float(lod) if not np.isnan(lod) else None,
            'loq_ppm': float(loq) if not np.isnan(loq) else None,
            'n_blank_samples': len(self.blank_signals) if self.blank_signals is not None else 0,
            'noise_method': noise_method,
            'mean_blank_signal': float(np.mean(self.blank_signals)) if self.blank_signals is not None else None
        }


def engineer_features_for_gas_spectra(
    wavelength: np.ndarray,
    spectra_dict: Dict[float, np.ndarray],
    reference_spectrum: np.ndarray,
    roi_range: Tuple[float, float] = None
) -> Dict:
    """
    Convenience function to apply feature engineering to gas sensing spectra.
    
    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array
    spectra_dict : dict
        Dictionary mapping concentration (ppm) to spectra
    reference_spectrum : np.ndarray
        Reference (air) spectrum
    roi_range : tuple
        (min_wavelength, max_wavelength) for analysis
        
    Returns
    -------
    result : dict
        Processed features, calibration metrics, and LoD analysis
    """
    # Initialize processor
    sfe = SpectralFeatureEngineering(wavelength)
    
    concentrations = sorted(spectra_dict.keys())
    all_spectra = np.array([spectra_dict[c] for c in concentrations])
    
    # Apply feature engineering
    sfe.set_data(wavelength, all_spectra)
    features = sfe.full_feature_engineering_pipeline()
    
    # Calculate wavelength shifts
    shifts = sfe.calculate_wavelength_shift(
        all_spectra, reference_spectrum.reshape(1, -1), roi_range
    )
    
    # Calculate sensitivity
    sensitivity_result = sfe.calculate_sensitivity(
        np.array(concentrations), shifts
    )
    
    # Calculate detection limit
    # Use reference spectrum variations as blank signal proxy
    blank_variations = np.diff(reference_spectrum)
    lod_calc = DetectionLimitCalculator(
        blank_signals=blank_variations[:100],  # First 100 variations
        sensitivity=sensitivity_result['sensitivity']
    )
    lod_result = lod_calc.full_analysis()
    
    return {
        'concentrations': concentrations,
        'wavelength_shifts': shifts.tolist(),
        'features': features,
        'sensitivity': sensitivity_result,
        'detection_limit': lod_result
    }


if __name__ == '__main__':
    # Example demonstration
    print("Spectral Feature Engineering Module")
    print("=" * 50)
    
    # Generate synthetic test data
    wavelength = np.linspace(675, 690, 100)  # nm (acetone ROI)
    n_samples = 10
    concentrations = np.linspace(1, 10, n_samples)  # ppm
    
    # Simulate acetone spectra with peak shift (0.116 nm/ppm as baseline)
    spectra = np.zeros((n_samples, len(wavelength)))
    for i, c in enumerate(concentrations):
        peak_pos = 682.0 + c * 0.116  # Baseline sensitivity
        spectra[i] = 0.5 + 0.3 * np.exp(-((wavelength - peak_pos) ** 2) / (2 * 1.5 ** 2))
        spectra[i] += np.random.normal(0, 0.01, len(wavelength))  # Add noise
    
    # Reference spectrum (air - no gas)
    reference = 0.5 + 0.3 * np.exp(-((wavelength - 682.0) ** 2) / (2 * 1.5 ** 2))
    
    # Apply feature engineering
    sfe = SpectralFeatureEngineering(wavelength, spectra)
    result = sfe.full_feature_engineering_pipeline()
    
    print(f"Original dynamic range: {result['stats']['original_range']}")
    print(f"Convolved dynamic range: {result['stats']['convolved_range']}")
    print(f"Dynamic range reduction: {result['stats']['dynamic_range_reduction']:.1f}×")
    
    # Calculate sensitivity
    shifts = sfe.calculate_wavelength_shift(spectra, reference.reshape(1, -1))
    sens = sfe.calculate_sensitivity(concentrations, shifts)
    
    print(f"\nSensitivity: {sens['sensitivity']:.4f} nm/ppm")
    print(f"R² coefficient: {sens['r_squared']:.4f}")
    
    # Calculate LoD
    lod_calc = DetectionLimitCalculator(
        blank_signals=np.diff(reference),
        sensitivity=sens['sensitivity']
    )
    lod = lod_calc.full_analysis()
    print(f"\nDetection Limit (LoD): {lod['lod_ppm']:.2f} ppm")
    print(f"Quantification Limit (LoQ): {lod['loq_ppm']:.2f} ppm")
