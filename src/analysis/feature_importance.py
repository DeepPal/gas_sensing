"""Spectral feature importance using SHAP and gradient-based attribution.

Answers the question: **which wavelength regions does the model rely on?**

Two complementary methods are provided:

1. **Gradient × Input** (fast, exact for differentiable models): computes
   element-wise product of input spectrum with its gradient w.r.t. the
   predicted output.  No additional dependencies beyond PyTorch.

2. **SHAP DeepExplainer / GradientExplainer** (requires ``shap`` package):
   computes Shapley values — a principled measure of each wavelength's
   marginal contribution across all possible subsets of features.

Both methods return a per-wavelength importance array that can be overlaid on
the original spectrum to identify which spectral regions drive predictions.

Usage
-----
::

    from src.analysis.feature_importance import (
        gradient_attribution,
        shap_attribution,
        plot_wavelength_importance,
        top_wavelength_bands,
    )

    # Gradient × Input (works with any PyTorch model)
    importance = gradient_attribution(model, spectra, target_class=0)
    # importance: ndarray, shape (N_wl,) — averaged over all query spectra

    # SHAP (requires `pip install shap`)
    shap_vals = shap_attribution(model, background_spectra, query_spectra)

    # Visualise
    fig = plot_wavelength_importance(wavelengths, importance, title="Ethanol")

    # Identify top contributing bands
    bands = top_wavelength_bands(wavelengths, importance, n_bands=3, width_nm=20)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects
    import torch

__all__ = [
    "gradient_attribution",
    "shap_attribution",
    "integrated_gradients",
    "plot_wavelength_importance",
    "top_wavelength_bands",
    "WavelengthBand",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class WavelengthBand:
    """A spectral band with high attribution importance."""
    center_nm: float
    start_nm: float
    end_nm: float
    mean_importance: float
    rank: int


# ---------------------------------------------------------------------------
# Gradient × Input attribution
# ---------------------------------------------------------------------------

def gradient_attribution(
    model: "torch.nn.Module",          # type: ignore[name-defined]
    spectra: "np.ndarray",             # (N, n_wl) or (n_wl,)  # type: ignore[name-defined]
    target_class: int | None = None,   # class index; None → regression output
    target_index: int = 0,             # output index for multi-output regression
    batch_size: int = 32,
    device: str | None = None,
    absolute_value: bool = True,
) -> np.ndarray:
    """Gradient × Input feature attribution.

    For each input spectrum, computes::

        attribution[i] = spectrum[i] * d(output) / d(spectrum[i])

    This is a first-order Taylor expansion of the model output around the input.
    Averaged over all provided spectra to get a population-level importance map.

    Parameters
    ----------
    model :
        PyTorch model.  Must accept (B, n_wl) tensors.
        Output can be logits (B, n_classes) or regression (B, 1).
    spectra :
        ndarray, shape (N, n_wl) — query spectra to attribute.
    target_class :
        For classification models: class index to attribute toward.
        If None, attributions are computed for the highest-confidence class
        for each sample individually.
    target_index :
        For regression models (when target_class=None): output index.
    batch_size :
        Number of spectra to process per forward pass.
    absolute_value :
        If True (default), return |attribution| (unsigned importance).

    Returns
    -------
    importance : ndarray, shape (n_wl,)
        Mean attribution over all query spectra.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    if spectra.ndim == 1:
        spectra = spectra[np.newaxis, :]

    all_attrs = []

    for start in range(0, len(spectra), batch_size):
        chunk = spectra[start: start + batch_size]
        x = torch.from_numpy(chunk.astype(np.float32)).to(device)
        x.requires_grad_(True)

        # Forward pass — handle both MultiTaskOutput and plain tensors
        output = _get_model_output(model, x)

        if output.ndim == 2 and output.shape[1] > 1:
            # Classification logits
            if target_class is not None:
                score = output[:, target_class].sum()
            else:
                score = output.max(dim=-1).values.sum()
        else:
            # Regression: use output[:, target_index]
            if output.ndim == 2:
                score = output[:, target_index].sum()
            else:
                score = output.sum()

        score.backward()
        grad = x.grad.detach().cpu().numpy()  # (B, n_wl)
        attr = grad * chunk  # element-wise
        if absolute_value:
            attr = np.abs(attr)
        all_attrs.append(attr)

    importance = np.concatenate(all_attrs, axis=0).mean(axis=0)
    return importance


# ---------------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------------

def integrated_gradients(
    model: "torch.nn.Module",      # type: ignore[name-defined]
    spectra: "np.ndarray",         # (N, n_wl)  # type: ignore[name-defined]
    baseline: "np.ndarray | None" = None,  # (n_wl,) or (N, n_wl)  # type: ignore[name-defined]
    target_class: int | None = None,
    n_steps: int = 50,
    batch_size: int = 16,
    device: str | None = None,
    absolute_value: bool = True,
) -> np.ndarray:
    """Integrated Gradients attribution (Sundararajan et al., 2017).

    More reliable than single-point gradient × input for non-linear models.
    Integrates gradients along the path from a baseline (zero or mean spectrum)
    to the actual input.

    Parameters
    ----------
    baseline :
        Reference spectrum.  Defaults to all-zeros.
        A better baseline for spectral data is the dark/background spectrum.

    Returns
    -------
    importance : ndarray, shape (n_wl,)
        Mean integrated gradient magnitude over all query spectra.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    if spectra.ndim == 1:
        spectra = spectra[np.newaxis, :]

    if baseline is None:
        baseline = np.zeros(spectra.shape[1], dtype=np.float32)
    baseline = np.asarray(baseline, dtype=np.float32)
    if baseline.ndim == 1:
        baseline = np.tile(baseline, (len(spectra), 1))

    all_ig = []

    for i in range(len(spectra)):
        x_in = spectra[i]         # (n_wl,)
        x_base = baseline[i]      # (n_wl,)
        delta = x_in - x_base

        # Interpolated inputs along the path
        alphas = np.linspace(0, 1, n_steps + 1, dtype=np.float32)
        interp = x_base[np.newaxis, :] + alphas[:, np.newaxis] * delta[np.newaxis, :]
        # interp: (n_steps+1, n_wl)

        grads = []
        for start in range(0, len(interp), batch_size):
            batch = interp[start: start + batch_size]
            x_t = torch.from_numpy(batch).to(device)
            x_t.requires_grad_(True)
            output = _get_model_output(model, x_t)

            if output.ndim == 2 and output.shape[1] > 1:
                if target_class is not None:
                    score = output[:, target_class].sum()
                else:
                    score = output.max(dim=-1).values.sum()
            else:
                score = output.sum()

            score.backward()
            grads.append(x_t.grad.detach().cpu().numpy())

        grads_arr = np.concatenate(grads, axis=0)  # (n_steps+1, n_wl)
        # Trapezoidal integration
        ig = delta * np.trapezoid(grads_arr, alphas, axis=0)
        if absolute_value:
            ig = np.abs(ig)
        all_ig.append(ig)

    return np.stack(all_ig).mean(axis=0)


# ---------------------------------------------------------------------------
# SHAP attribution
# ---------------------------------------------------------------------------

def shap_attribution(
    model: "torch.nn.Module",             # type: ignore[name-defined]
    background_spectra: "np.ndarray",     # (K, n_wl) — reference distribution  # type: ignore[name-defined]
    query_spectra: "np.ndarray",          # (N, n_wl) — spectra to explain  # type: ignore[name-defined]
    target_class: int | None = None,
    device: str | None = None,
    max_evals: int = 500,
) -> np.ndarray:
    """SHAP DeepExplainer attribution.

    Requires ``pip install shap``.

    Parameters
    ----------
    background_spectra :
        Reference spectra representing the data distribution (10–200 samples).
        Used by SHAP as the "expectation" baseline.
    query_spectra :
        Spectra to explain.
    target_class :
        Class index to explain (classification models).
        None → explains the first output.

    Returns
    -------
    shap_values : ndarray, shape (n_wl,)
        Mean |SHAP value| over all query spectra.
    """
    try:
        import shap
        import torch
    except ImportError as e:
        raise ImportError(
            "shap package required: pip install shap") from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Wrap model to return a simple tensor output
    class _ModelWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return _get_model_output(self.inner, x)

    wrapper = _ModelWrapper(model)

    bg = torch.from_numpy(background_spectra.astype(np.float32)).to(device)
    explainer = shap.DeepExplainer(wrapper, bg)

    qr = torch.from_numpy(query_spectra.astype(np.float32)).to(device)
    shap_vals = explainer.shap_values(qr)

    if isinstance(shap_vals, list):
        # Multi-class: list of (N, n_wl) arrays
        idx = target_class if target_class is not None else 0
        vals = shap_vals[idx]
    else:
        vals = shap_vals  # (N, n_wl)

    return np.abs(np.asarray(vals)).mean(axis=0)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_wavelength_importance(
    wavelengths: "np.ndarray",        # (n_wl,)  # type: ignore[name-defined]
    importance: "np.ndarray",         # (n_wl,)  # type: ignore[name-defined]
    title: str = "Wavelength Importance",
    spectrum: "np.ndarray | None" = None,  # (n_wl,) optional overlay  # type: ignore[name-defined]
    top_n_bands: int = 3,
    band_width_nm: float = 20.0,
    height: int = 400,
) -> "plotly.graph_objects.Figure":  # type: ignore[name-defined]
    """Interactive Plotly figure showing per-wavelength feature importance.

    Optionally overlays a representative spectrum (on a secondary y-axis)
    and highlights the top-N most important wavelength bands.

    Parameters
    ----------
    wavelengths : nm axis
    importance : attribution values per wavelength (should be non-negative)
    spectrum : optional spectrum to overlay (secondary y-axis)
    top_n_bands : number of high-importance bands to highlight
    band_width_nm : width of each highlighted band in nm
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise ImportError("plotly required: pip install plotly") from e

    # Smooth importance for display
    smoothed = _smooth(importance, window=5)

    # Identify top bands
    bands = top_wavelength_bands(wavelengths, importance,
                                 n_bands=top_n_bands, width_nm=band_width_nm)

    fig = go.Figure()

    # Importance trace
    fig.add_trace(go.Scatter(
        x=wavelengths, y=smoothed,
        name="Importance",
        fill="tozeroy",
        line=dict(color="steelblue", width=1.5),
        fillcolor="rgba(70, 130, 180, 0.3)",
    ))

    # Optional spectrum overlay
    if spectrum is not None:
        spec_norm = spectrum / (spectrum.max() + 1e-12)
        imp_max = smoothed.max() + 1e-12
        fig.add_trace(go.Scatter(
            x=wavelengths, y=spec_norm * imp_max * 0.8,
            name="Spectrum (normalised)",
            line=dict(color="crimson", width=1, dash="dot"),
        ))

    # Highlight top bands
    for band in bands:
        fig.add_vrect(
            x0=band.start_nm, x1=band.end_nm,
            fillcolor="orange", opacity=0.15,
            annotation_text=f"{band.center_nm:.0f} nm",
            annotation_position="top left",
            line_width=0,
        )

    fig.update_layout(
        title=title,
        xaxis_title="Wavelength (nm)",
        yaxis_title="Attribution importance",
        height=height,
        template="plotly_white",
    )
    return fig


def _smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average smoothing."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    padded = np.pad(x, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(x)]


# ---------------------------------------------------------------------------
# Band extraction
# ---------------------------------------------------------------------------

def top_wavelength_bands(
    wavelengths: np.ndarray,
    importance: np.ndarray,
    n_bands: int = 3,
    width_nm: float = 20.0,
) -> list[WavelengthBand]:
    """Extract the top-N most important wavelength bands.

    Uses a greedy non-overlapping peak-picking algorithm: iteratively selects
    the highest-importance region, suppresses a window around it, then repeats.

    Parameters
    ----------
    wavelengths : nm axis, shape (n_wl,)
    importance : attribution values per wavelength, shape (n_wl,)
    n_bands : number of bands to extract
    width_nm : width of each band in nm

    Returns
    -------
    list of WavelengthBand sorted by importance (highest first)
    """
    imp = importance.copy()
    wl_range = wavelengths[-1] - wavelengths[0]
    half_width_frac = (width_nm / 2.0) / wl_range
    n_wl = len(wavelengths)
    half_idx = max(1, int(half_width_frac * n_wl))

    bands = []
    for rank in range(1, n_bands + 1):
        if imp.max() <= 0:
            break
        peak_idx = int(np.argmax(imp))
        center = float(wavelengths[peak_idx])

        # Find band boundaries
        start_idx = max(0, peak_idx - half_idx)
        end_idx = min(n_wl - 1, peak_idx + half_idx)
        start_nm = float(wavelengths[start_idx])
        end_nm = float(wavelengths[end_idx])
        mean_imp = float(imp[start_idx:end_idx + 1].mean())

        bands.append(WavelengthBand(
            center_nm=center,
            start_nm=start_nm,
            end_nm=end_nm,
            mean_importance=mean_imp,
            rank=rank,
        ))

        # Suppress this region for next iteration
        imp[start_idx:end_idx + 1] = 0.0

    return bands


# ---------------------------------------------------------------------------
# Internal model output helper
# ---------------------------------------------------------------------------

def _get_model_output(
    model: "torch.nn.Module",  # type: ignore[name-defined]
    x: "torch.Tensor",         # type: ignore[name-defined]
) -> "torch.Tensor":           # type: ignore[name-defined]
    """Get a plain tensor from a model forward call.

    Handles both plain tensor outputs and dataclass outputs (MultiTaskOutput).
    """
    import torch

    output = model(x)

    # MultiTaskOutput or similar dataclass
    if hasattr(output, "class_logits") and output.class_logits is not None:
        return output.class_logits
    if hasattr(output, "concentration") and output.concentration is not None:
        return output.concentration
    if hasattr(output, "features"):
        return output.features

    # Dict output (DomainAdaptModel)
    if isinstance(output, dict):
        if "class_logits" in output and output["class_logits"] is not None:
            return output["class_logits"]
        if "concentration" in output and output["concentration"] is not None:
            return output["concentration"]

    # Tuple output (TemporalEncoder: (conc, features))
    if isinstance(output, tuple):
        if output[0] is not None and isinstance(output[0], torch.Tensor):
            return output[0]
        return output[1]

    if isinstance(output, torch.Tensor):
        return output

    raise TypeError(f"Unsupported model output type: {type(output)}")
