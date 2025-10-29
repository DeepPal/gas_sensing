"""
Visualization functions for gas analysis.
"""
from typing import List, Optional, Dict, Any, Union
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def plot_spectrum(wavelengths: np.ndarray, 
                 spectrum: np.ndarray, 
                 peaks: Optional[List[Dict[str, float]]] = None,
                 title: str = 'Gas Spectrum',
                 xlabel: str = 'Wavelength (nm)',
                 ylabel: str = 'Intensity (a.u.)',
                 save_path: Optional[Union[str, Path]] = None,
                 **kwargs) -> Figure:
    """
    Plot a spectrum with optional peak annotations.
    
    Args:
        wavelengths: Array of wavelength values
        spectrum: Array of intensity values
        peaks: List of peak dictionaries with 'position' and 'intensity' keys
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: If provided, save the figure to this path
        **kwargs: Additional arguments for matplotlib.pyplot.plot()
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the spectrum
    ax.plot(wavelengths, spectrum, **kwargs)
    
    # Mark peaks if provided
    if peaks is not None:
        for peak in peaks:
            ax.plot(peak['position'], peak['intensity'], 'ro')
            ax.text(peak['position'], peak['intensity'], 
                   f"{peak['position']:.1f} nm", 
                   ha='center', va='bottom')
    
    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Save the figure if path is provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig

def plot_spectra(
    spectra: Dict[str, Union[Dict[str, np.ndarray], "pd.DataFrame", tuple, list]],
    title: str = "Spectra",
    xlabel: str = "Wavelength (nm)",
    ylabel: str = "Intensity (a.u.)",
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Plot multiple spectra on a single figure.

    Args:
        spectra: Mapping of label -> spectrum. Spectrum can be a DataFrame with
                 'wavelength' and 'intensity' columns, a dict with those keys,
                 or a (wavelengths, intensities) tuple/list.
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: If provided, save the figure to this path

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, spec in spectra.items():
        try:
            x = None
            y = None
            # pandas DataFrame
            if hasattr(spec, "__class__") and spec.__class__.__name__ == "DataFrame":
                x = spec["wavelength"].values
                y = spec["intensity"].values
            # dict-like
            elif isinstance(spec, dict) and "wavelength" in spec and "intensity" in spec:
                x = np.asarray(spec["wavelength"])
                y = np.asarray(spec["intensity"])
            # tuple/list of arrays
            elif isinstance(spec, (tuple, list)) and len(spec) >= 2:
                x = np.asarray(spec[0])
                y = np.asarray(spec[1])

            if x is not None and y is not None:
                ax.plot(x, y, label=label)
        except Exception as e:
            logger.error(f"Failed to plot spectrum '{label}': {e}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig

def plot_calibration_curve(concentrations: List[float],
                          responses: List[float],
                          title: str = 'Calibration Curve',
                          xlabel: str = 'Concentration',
                          ylabel: str = 'Peak Area',
                          save_path: Optional[Union[str, Path]] = None) -> Figure:
    """
    Plot a calibration curve with linear regression.
    
    Args:
        concentrations: List of concentration values
        responses: List of corresponding response values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: If provided, save the figure to this path
        
    Returns:
        Matplotlib Figure object
    """
    # Convert to numpy arrays
    x = np.array(concentrations)
    y = np.array(responses)
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line = slope * x + intercept
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, color='blue', label='Data Points')
    ax.plot(x, line, 'r-', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}\nR² = {r_value**2:.4f}')
    
    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save the figure if path is provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Calibration curve saved to {save_path}")
    
    return fig

def save_figure(fig: Figure, path: Union[str, Path], 
                dpi: int = 300, **kwargs) -> None:
    """
    Save a matplotlib figure to a file.
    
    Args:
        fig: Matplotlib Figure object
        path: Path to save the figure
        dpi: Resolution in dots per inch
        **kwargs: Additional arguments for savefig()
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', **kwargs)
    logger.info(f"Figure saved to {path}")
