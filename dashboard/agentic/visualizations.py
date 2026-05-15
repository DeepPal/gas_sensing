"""
Plotly/Matplotlib chart builders for the agentic pipeline tab.

All public functions return a ``plotly.graph_objects.Figure`` so the caller
(``tab.py``) decides where and how to render it.  No ``st.*`` calls here.

Note: the majority of the inline plot code in the original monolithic
``render()`` function could not be extracted without refactoring function
signatures (per Task 17 boundary rules).  This module provides the thin
reusable chart helpers that *can* be cleanly isolated.  The remaining
inline plot blocks in ``tab.py`` are candidates for future extraction.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def make_live_acquisition_chart(
    wl: np.ndarray,
    current_mean: np.ndarray,
    frame_idx: int,
    n_frames: int,
) -> go.Figure:
    """Return a Plotly figure for the live acquisition running average.

    Parameters
    ----------
    wl:
        Wavelength array (nm).
    current_mean:
        Running mean of acquired frames so far.
    frame_idx:
        0-based index of the most recently acquired frame.
    n_frames:
        Total frames to acquire.
    """
    fig = go.Figure(
        go.Scatter(
            x=wl,
            y=current_mean,
            mode="lines",
            name=f"Live Avg ({frame_idx + 1}/{n_frames})",
            line=dict(color="#FF4B4B", width=1.5),
        )
    )
    fig.update_layout(
        title=f"Live Acquisition — Frame {frame_idx + 1}/{n_frames}",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity (a.u.)",
        height=320,
        margin=dict(t=40, b=30, l=40, r=20),
    )
    return fig


def make_spectral_overlay(
    wl_common: np.ndarray,
    Z: np.ndarray,
    labels: list[str],
    has_diff: bool,
    ref_peak_wl: float | None = None,
) -> go.Figure:
    """Return a rainbow spectral overlay figure.

    Parameters
    ----------
    wl_common:
        Common wavelength grid (interpolated).
    Z:
        (n_spectra, n_wavelengths) intensity / differential-signal matrix.
    labels:
        One label per spectrum row in ``Z``.
    has_diff:
        If True, y-axis is labelled as ΔI (differential intensity).
    ref_peak_wl:
        Reference peak wavelength — a vertical dashed line is added when provided.
    """
    y_lbl = "ΔI (Differential Intensity)" if has_diff else "Intensity"
    fig = go.Figure()
    palette = [f"hsl({h},70%,50%)" for h in np.linspace(0, 240, len(labels))]
    for idx, (spec, lbl) in enumerate(zip(Z, labels)):
        fig.add_trace(
            go.Scatter(
                x=wl_common,
                y=spec,
                mode="lines",
                name=lbl,
                line=dict(color=palette[idx]),
            )
        )
    if has_diff and ref_peak_wl is not None:
        fig.add_vline(
            x=ref_peak_wl,
            line_dash="dash",
            line_color="white",
            annotation_text=f"Ref λ={ref_peak_wl:.1f} nm",
        )
    fig.update_layout(
        title="Spectral Overlay — All Concentrations"
        + (" (Differential ΔI)" if has_diff else ""),
        xaxis_title="Wavelength (nm)",
        yaxis_title=y_lbl,
        height=380,
        margin=dict(t=40, b=30, l=40, r=20),
    )
    return fig


def make_3d_response_surface(
    wl_common: np.ndarray,
    Z: np.ndarray,
    concs_plot: list[float],
    has_diff: bool,
) -> go.Figure:
    """Return a 3-D response surface (Wavelength × Concentration × Signal).

    Parameters
    ----------
    wl_common:
        Common wavelength grid (nm).
    Z:
        (n_spectra, n_wavelengths) signal matrix.
    concs_plot:
        Concentration values for each row of Z (ppm).
    has_diff:
        True when Z contains differential signals (ΔI).
    """
    z_lbl = "ΔI" if has_diff else "Intensity"
    fig = go.Figure(
        data=[go.Surface(z=Z, x=wl_common, y=concs_plot, colorscale="Viridis")]
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Wavelength (nm)",
            yaxis_title="Concentration (ppm)",
            zaxis_title=z_lbl,
        ),
        height=500,
        margin=dict(l=0, r=0, b=0, t=10),
    )
    return fig


def make_sensitivity_heatmap(
    wl_common: np.ndarray,
    Z: np.ndarray,
    concs_plot: list[float],
    has_diff: bool,
) -> go.Figure:
    """Return a Concentration × Wavelength sensitivity heatmap.

    Parameters
    ----------
    wl_common:
        Common wavelength grid (nm).
    Z:
        (n_spectra, n_wavelengths) signal matrix.
    concs_plot:
        Concentration values for each row of Z (ppm).
    has_diff:
        True when Z contains differential signals (ΔI).
    """
    z_lbl = "ΔI" if has_diff else "Intensity"
    fig = go.Figure(
        data=go.Heatmap(
            z=Z,
            x=wl_common,
            y=concs_plot,
            colorscale="Viridis",
            colorbar=dict(title=z_lbl),
        )
    )
    fig.update_layout(
        title=f"Sensitivity Heatmap — {z_lbl} (Concentration × Wavelength)",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Concentration (ppm)",
        height=380,
        margin=dict(t=40, b=30, l=40, r=20),
    )
    return fig


def make_preview_chart(wl: np.ndarray, intensity: np.ndarray) -> go.Figure:
    """Return a single-frame spectral preview figure (live preview, not recorded)."""
    fig = go.Figure(
        go.Scatter(
            x=wl,
            y=intensity,
            mode="lines",
            name="Live Preview",
            line=dict(color="limegreen", width=1.5),
        )
    )
    fig.update_layout(
        title="Live Spectral Preview (read-only — not recorded)",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity (a.u.)",
        height=280,
        margin=dict(t=30, b=20, l=40, r=20),
    )
    return fig


def make_raw_vs_preprocessed_chart(
    wl: np.ndarray,
    raw: np.ndarray,
    processed: np.ndarray,
) -> go.Figure:
    """Return a raw-vs-preprocessed comparison figure for a single spectrum."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=wl,
            y=raw,
            name="Raw",
            line=dict(color="gray", dash="dot"),
            opacity=0.6,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=wl,
            y=processed,
            name="Preprocessed",
            line=dict(color="royalblue", width=2),
        )
    )
    fig.update_layout(
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity",
        height=350,
        margin=dict(t=20, b=30, l=40, r=20),
    )
    return fig
