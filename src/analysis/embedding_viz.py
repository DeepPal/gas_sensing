"""Latent space visualisation for spectral embeddings.

Projects high-dimensional latent vectors (from ``SpectralAutoencoder`` or
``TemporalEncoder``) into 2-D using UMAP or t-SNE, then produces interactive
Plotly scatter plots coloured by analyte, concentration, session, or any
metadata field.

This is the key visualisation for the journal paper: showing that the
data-driven latent space separates analytes and encodes concentration
without any physics supervision.

Usage
-----
::

    from src.io.universal_loader import load_dataset
    from src.models.spectral_autoencoder import SpectralAutoencoder, AutoencoderConfig
    from src.analysis.embedding_viz import plot_embedding, reduce_dimensions

    # Load real data
    ds = load_dataset("output/batch/Ethanol/stable_selected", normalisation="snv")

    # Encode
    model = SpectralAutoencoder(AutoencoderConfig(input_length=3648, latent_dim=64))
    # ... train model ...
    latents = model.encode_numpy(ds.spectra)   # (N, 64)

    # Visualise
    fig = plot_embedding(
        latents,
        colour_by=ds.labels,
        colour_label="Concentration (ppm)",
        title="Ethanol latent space (autoencoder)",
        method="umap",
    )
    fig.show()
    fig.write_html("embedding.html")
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np

__all__ = [
    "reduce_dimensions",
    "plot_embedding",
    "plot_reconstruction",
    "plot_training_curves",
    "EmbeddingResult",
]


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class EmbeddingResult:
    """2-D embedding coordinates with metadata.

    Attributes
    ----------
    coords : ndarray, shape (N, 2)
        2-D embedding coordinates.
    method : str
        Reduction method used ('umap' or 'tsne').
    latent_dim : int
        Dimensionality of the original latent space.
    metadata : dict
        Extra metadata passed through (analyte, labels, etc.).
    """
    coords: np.ndarray
    method: str
    latent_dim: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_dimensions(
    latents: np.ndarray,
    method: Literal["umap", "tsne", "pca"] = "umap",
    n_components: int = 2,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    tsne_perplexity: float = 30.0,
    random_state: int = 42,
) -> EmbeddingResult:
    """Reduce latent vectors to 2-D for visualisation.

    Parameters
    ----------
    latents : ndarray, shape (N, D)
        Latent feature vectors from encoder.
    method :
        ``'umap'`` (recommended), ``'tsne'``, or ``'pca'``.
    n_components :
        Output dimensionality (default 2 for scatter plot).
    umap_n_neighbors :
        UMAP local neighbourhood size.  Smaller = more local structure.
    umap_min_dist :
        UMAP minimum distance in embedding.  Smaller = tighter clusters.
    tsne_perplexity :
        t-SNE perplexity.  Typical range 5–50.
    random_state :
        Reproducibility seed.

    Returns
    -------
    EmbeddingResult
    """
    if method == "pca":
        coords = _pca(latents, n_components)
    elif method == "tsne":
        coords = _tsne(latents, n_components, tsne_perplexity, random_state)
    elif method == "umap":
        coords = _umap(latents, n_components, umap_n_neighbors,
                       umap_min_dist, random_state)
    else:
        raise ValueError(f"Unknown method {method!r}. "
                         "Choose 'umap', 'tsne', or 'pca'.")

    return EmbeddingResult(
        coords=coords,
        method=method,
        latent_dim=latents.shape[1],
    )


def _pca(X: np.ndarray, n_components: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components).fit_transform(X)


def _tsne(X: np.ndarray, n_components: int, perplexity: float,
          seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE
    perp = min(perplexity, len(X) - 1)
    return TSNE(n_components=n_components, perplexity=perp,
                random_state=seed, init="pca",
                learning_rate="auto").fit_transform(X)


def _umap(X: np.ndarray, n_components: int, n_neighbors: int,
          min_dist: float, seed: int) -> np.ndarray:
    try:
        import umap
    except ImportError as e:
        raise ImportError(
            "umap-learn required: pip install umap-learn"
        ) from e
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=seed)
    return reducer.fit_transform(X)


# ---------------------------------------------------------------------------
# Plotly scatter plots
# ---------------------------------------------------------------------------

def plot_embedding(
    latents: np.ndarray | EmbeddingResult,
    colour_by: np.ndarray | list | None = None,
    colour_label: str = "Label",
    hover_text: list[str] | None = None,
    title: str = "Spectral Embedding",
    method: Literal["umap", "tsne", "pca"] = "umap",
    marker_size: int = 8,
    colorscale: str = "Viridis",
    width: int = 800,
    height: int = 600,
    **reduce_kwargs: Any,
) -> Any:
    """Interactive 2-D scatter plot of spectral latent vectors.

    Parameters
    ----------
    latents :
        Either raw latent ndarray (N, D) or a pre-computed ``EmbeddingResult``.
    colour_by :
        Array of values to colour points by (concentrations, analyte labels,
        session IDs, …).  Numeric → continuous colour scale.
        String → discrete colour palette.
    colour_label :
        Legend/axis label for the colour variable.
    hover_text :
        Per-point hover text.  If None, index is shown.
    title :
        Plot title.
    method :
        Reduction method (if ``latents`` is a raw array).
    marker_size :
        Point size.
    colorscale :
        Plotly colorscale name for continuous colour.
    width, height :
        Figure dimensions in pixels.
    **reduce_kwargs :
        Passed to ``reduce_dimensions`` (e.g. ``umap_n_neighbors=10``).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError("plotly required: pip install plotly") from e

    # Reduce to 2-D if needed
    if isinstance(latents, np.ndarray):
        result = reduce_dimensions(latents, method=method, **reduce_kwargs)
    else:
        result = latents

    coords = result.coords
    x, y = coords[:, 0], coords[:, 1]

    axis_label = result.method.upper()

    if hover_text is None:
        hover_text = [str(i) for i in range(len(x))]

    # Build figure
    if colour_by is None:
        fig = go.Figure(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=marker_size, opacity=0.8),
            text=hover_text, hovertemplate="%{text}<extra></extra>",
        ))
    else:
        colour_arr = np.asarray(colour_by)
        is_numeric = np.issubdtype(colour_arr.dtype, np.number)

        if is_numeric:
            fig = go.Figure(go.Scatter(
                x=x, y=y, mode="markers",
                marker=dict(
                    size=marker_size, opacity=0.85,
                    color=colour_arr,
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(title=colour_label),
                ),
                text=[f"{colour_label}: {v:.3g}<br>{t}"
                      for v, t in zip(colour_arr, hover_text)],
                hovertemplate="%{text}<extra></extra>",
            ))
        else:
            # Discrete colours
            unique_labels = sorted(set(colour_arr.tolist()))
            colours = px.colors.qualitative.Plotly
            colour_map = {lbl: colours[i % len(colours)]
                          for i, lbl in enumerate(unique_labels)}
            traces = []
            for lbl in unique_labels:
                mask = colour_arr == lbl
                traces.append(go.Scatter(
                    x=x[mask], y=y[mask], mode="markers", name=str(lbl),
                    marker=dict(size=marker_size, opacity=0.85,
                                color=colour_map[lbl]),
                    text=[hover_text[i] for i in np.where(mask)[0]],
                    hovertemplate="%{text}<extra></extra>",
                ))
            fig = go.Figure(traces)

    fig.update_layout(
        title=title,
        xaxis_title=f"{axis_label} 1",
        yaxis_title=f"{axis_label} 2",
        width=width, height=height,
        template="plotly_white",
        legend_title=colour_label,
    )
    return fig


def plot_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    wavelengths: np.ndarray | None = None,
    indices: list[int] | None = None,
    title: str = "Spectral Reconstruction Quality",
    width: int = 900,
    height: int = 500,
) -> Any:
    """Overlay original and reconstructed spectra to assess autoencoder quality.

    Parameters
    ----------
    original : ndarray, shape (N, N_wl)
    reconstructed : ndarray, shape (N, N_wl)
    wavelengths : ndarray, shape (N_wl,) — wavelength axis in nm
    indices : which samples to plot (default: first 3)
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError("plotly required: pip install plotly") from e

    if indices is None:
        indices = list(range(min(3, len(original))))

    if wavelengths is None:
        wavelengths = np.arange(original.shape[1], dtype=float)

    fig = go.Figure()
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for j, i in enumerate(indices):
        col = colours[j % len(colours)]
        fig.add_trace(go.Scatter(
            x=wavelengths, y=original[i],
            name=f"Original #{i}", line=dict(color=col, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=wavelengths, y=reconstructed[i],
            name=f"Reconstructed #{i}",
            line=dict(color=col, width=1.5, dash="dash"),
        ))

    # Reconstruction error summary
    mse = float(np.mean((original[indices] - reconstructed[indices]) ** 2))
    fig.update_layout(
        title=f"{title}  (MSE={mse:.2e})",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Signal",
        width=width, height=height,
        template="plotly_white",
    )
    return fig


def plot_training_curves(
    history: dict[str, list[float]],
    title: str = "Training History",
    width: int = 700,
    height: int = 400,
) -> Any:
    """Plot train/val loss curves from training history dict."""
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError("plotly required: pip install plotly") from e

    fig = go.Figure()
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig.add_trace(go.Scatter(
        x=epochs, y=history["train_loss"],
        name="Train loss", line=dict(color="#1f77b4", width=2),
    ))
    if "val_loss" in history:
        fig.add_trace(go.Scatter(
            x=epochs, y=history["val_loss"],
            name="Val loss", line=dict(color="#ff7f0e", width=2),
        ))

    fig.update_layout(
        title=title, xaxis_title="Epoch", yaxis_title="Loss",
        width=width, height=height, template="plotly_white",
        yaxis_type="log",
    )
    return fig
