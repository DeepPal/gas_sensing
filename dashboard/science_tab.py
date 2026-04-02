"""Phase 5C — Data-Driven Science Tab for SpectraAgent Streamlit Dashboard.

Five sub-tabs exposing the physics-agnostic feature discovery pipeline:
  1. Dataset Explorer      — load, inspect, normalise spectral datasets
  2. Feature Discovery     — spectral autoencoder + UMAP/t-SNE/PCA visualisation
  3. Model Training        — multi-task / contrastive model training + save
  4. Cross-Dataset Analysis — leave-one-config-out generalisation benchmark
  5. Publication Figures   — export-ready SVG/PDF/PNG figures

Design principles
-----------------
- All heavy computation goes through dedicated src/ modules — the UI layer
  only orchestrates, it does not contain algorithmic logic.
- Session state keys are namespaced with a prefix per sub-tab to avoid
  collisions with other dashboard tabs.
- Every optional import is guarded so the tab degrades gracefully when
  optional packages (torch, plotly, sklearn) are absent.
- Input paths are validated to stay within the project root (path traversal
  protection).
"""
from __future__ import annotations

import io
import logging
import math
from pathlib import Path
import time
from typing import Any
import warnings

import numpy as np
import streamlit as st

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root (resolves relative paths)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Optional heavy imports — each failure is captured for a user-facing message
# ---------------------------------------------------------------------------

_IMPORT_ERRORS: dict[str, str] = {}

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError as _e:
    TORCH_AVAILABLE = False
    _IMPORT_ERRORS["torch"] = str(_e)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError as _e:
    PLOTLY_AVAILABLE = False
    _IMPORT_ERRORS["plotly"] = str(_e)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError as _e:
    PANDAS_AVAILABLE = False
    _IMPORT_ERRORS["pandas"] = str(_e)

try:
    from src.io.universal_loader import (
        SpectralDataset,
        list_sessions,
        load_dataset,
        load_session_csv,
        merge_datasets,
    )
    LOADER_AVAILABLE = True
except Exception as _e:
    LOADER_AVAILABLE = False
    _IMPORT_ERRORS["loader"] = str(_e)

try:
    from src.models.spectral_autoencoder import (
        AutoencoderConfig,
        SpectralAutoencoder,
        train_autoencoder,
    )
    AUTOENCODER_AVAILABLE = TORCH_AVAILABLE
except Exception as _e:
    AUTOENCODER_AVAILABLE = False
    _IMPORT_ERRORS["autoencoder"] = str(_e)

try:
    from src.analysis.embedding_viz import plot_embedding
    EMBED_VIZ_AVAILABLE = PLOTLY_AVAILABLE
except Exception as _e:
    EMBED_VIZ_AVAILABLE = False
    _IMPORT_ERRORS["embed_viz"] = str(_e)

try:
    from src.models.contrastive import ContrastiveConfig, ContrastiveEncoder, train_contrastive
    from src.models.multi_task import (
        MultiTaskConfig,
        MultiTaskModel,
        MultiTaskTargets,
        train_multi_task,
    )
    MODELS_AVAILABLE = TORCH_AVAILABLE
except Exception as _e:
    MODELS_AVAILABLE = False
    _IMPORT_ERRORS["models"] = str(_e)

try:
    from src.analysis.cross_dataset_eval import BenchmarkResult, run_benchmark
    BENCHMARK_AVAILABLE = LOADER_AVAILABLE
except Exception as _e:
    BENCHMARK_AVAILABLE = False
    _IMPORT_ERRORS["benchmark"] = str(_e)

try:
    from src.analysis.feature_importance import gradient_attribution, top_wavelength_bands
    FEAT_IMP_AVAILABLE = TORCH_AVAILABLE and PLOTLY_AVAILABLE
except Exception as _e:
    FEAT_IMP_AVAILABLE = False

try:
    from src.experiment_tracking import ExperimentTracker
    MLFLOW_AVAILABLE = True
except Exception as _e:
    MLFLOW_AVAILABLE = False
    _IMPORT_ERRORS["mlflow"] = str(_e)
    _IMPORT_ERRORS["feat_imp"] = str(_e)


# ===========================================================================
# Public entry point
# ===========================================================================

def render() -> None:
    """Render the Data-Driven Science tab (called from app.py)."""
    st.title("🔬 Data-Driven Science")
    st.caption(
        "Physics-agnostic feature discovery and cross-dataset generalisation — "
        "trains on spectral patterns directly, no sensor-specific assumptions."
    )

    with st.expander("ℹ️ How to use this tab — start here", expanded=False):
        st.markdown(
            """
**Recommended workflow (follow the sub-tabs in order):**

1. **📂 Dataset Explorer** — Load one or more spectral CSV datasets.
   Point it at a directory of raw spectra (e.g. `Joy_Data/Ethanol/`) or a
   single wide-format CSV.  The loader infers analyte and concentration from
   folder names automatically.

   > **Note:** If your data came from a live `spectraagent start` session,
   > the session files in `output/sessions/{id}/` contain processed results,
   > not raw spectra.  Use `Joy_Data/` exports or enable `save_raw=True` in
   > the acquisition config to get raw spectral CSVs suitable for this tab.

2. **🧬 Feature Discovery** — Train an autoencoder on your loaded dataset.
   The encoder compresses each spectrum to a low-dimensional embedding.
   The UMAP/t-SNE scatter shows whether analytes form distinct clusters —
   if they do, discriminative features exist in the spectra.

3. **🏋️ Model Training** — Train a multi-task or contrastive model.
   Models are saved to `output/model_versions/`.

   > **To deploy a trained model to live acquisition:** copy
   > `output/model_versions/{name}_{version}/model.pt` to
   > `models/registry/cnn_classifier.pt`, then restart `spectraagent start`.

4. **📊 Cross-Dataset Analysis** — Load 3+ datasets from different sensor
   configurations (different integration times, reference wavelengths, or
   sensor chips).  The leave-one-out benchmark answers: *"Does the model
   generalise to an unseen sensor without retraining?"*

   > **Tip:** If you only have 1 sensor, use `spectraagent robustness` to
   > generate 3 config variants automatically (sweeps integration time).

5. **📄 Publication Figures** — Export calibration curves, feature importance
   maps, and generalisation plots as SVG/PDF for your manuscript.
            """
        )

    (
        tab_explore,
        tab_features,
        tab_train,
        tab_bench,
        tab_figures,
    ) = st.tabs([
        "📂 Dataset Explorer",
        "🧬 Feature Discovery",
        "🏋️ Model Training",
        "📊 Cross-Dataset Analysis",
        "📄 Publication Figures",
    ])

    with tab_explore:
        _render_dataset_explorer()
    with tab_features:
        _render_feature_discovery()
    with tab_train:
        _render_model_training()
    with tab_bench:
        _render_cross_dataset()
    with tab_figures:
        _render_publication_figures()


# ===========================================================================
# Shared helpers
# ===========================================================================

def _safe_path(raw: str) -> Path | None:
    """Resolve and validate that a path is inside the project root."""
    p = Path(raw.strip()).expanduser()
    if not p.is_absolute():
        p = _REPO_ROOT / p
    p = p.resolve()
    try:
        p.relative_to(_REPO_ROOT)
    except ValueError:
        return None
    return p


def _missing_module_error(name: str) -> None:
    st.error(
        f"Required module `{name}` is not available. "
        f"Error: {_IMPORT_ERRORS.get(name, 'unknown')}"
    )


def _int_labels_from_dataset(ds: SpectralDataset) -> np.ndarray | None:
    """Map unique float concentration labels → integer class indices."""
    if ds.labels is None:
        return None
    valid_mask = np.isfinite(ds.labels)
    if not valid_mask.any():
        return None
    unique_vals = np.unique(ds.labels[valid_mask])
    val_to_idx = {float(v): i for i, v in enumerate(unique_vals)}
    return np.array(
        [val_to_idx.get(float(v), 0) for v in ds.labels], dtype=np.int64
    )


def _loss_curve_figure(
    history: dict[str, list[float]],
    title: str = "Training Loss",
) -> go.Figure:
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    fig = go.Figure()
    for key, color, dash in [
        ("train_loss", "#4682b4", "solid"),
        ("val_loss", "#dc143c", "dot"),
    ]:
        if key in history and history[key]:
            fig.add_trace(go.Scatter(
                x=epochs, y=history[key],
                name=key.replace("_", " ").title(),
                line=dict(color=color, dash=dash, width=2),
            ))
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis_type="log",
        height=300,
        template="plotly_white",
    )
    return fig


# ===========================================================================
# Sub-tab 1 — Dataset Explorer
# ===========================================================================

def _render_dataset_explorer() -> None:
    st.subheader("Dataset Explorer")
    st.markdown(
        "Load a spectral dataset — either raw spectral CSVs from a Joy_Data "
        "directory, or processed feature results from a live SpectraAgent session."
    )

    if not LOADER_AVAILABLE:
        _missing_module_error("loader")
        return

    # ── Data source selector ─────────────────────────────────────────────
    data_source = st.radio(
        "Data source",
        ["Spectral CSV files", "SpectraAgent session results"],
        horizontal=True,
        key="de_source",
        help=(
            "Use **Spectral CSV files** for raw Joy_Data spectra (enables "
            "autoencoder). Use **SpectraAgent session results** to load "
            "processed features from a live acquisition session."
        ),
    )

    if data_source == "Spectral CSV files":
        _render_csv_loader()
    else:
        _render_session_loader()


def _render_csv_loader() -> None:
    """Load controls for raw spectral CSV directories."""
    col_path, col_norm = st.columns([3, 1])
    with col_path:
        raw_path = st.text_input(
            "Dataset path (file or directory, relative to project root or absolute)",
            value="",
            placeholder="output/batch/Ethanol  or  Joy_Data/Ethanol/stable_selected",
            key="de_path",
        )
    with col_norm:
        norm = st.selectbox(
            "Normalisation",
            ["none", "snv", "msc", "area", "minmax"],
            index=0,
            key="de_norm",
            help="SNV recommended for spectra with baseline offsets.",
        )

    col_analyte, col_sig = st.columns(2)
    with col_analyte:
        analyte_override = st.text_input(
            "Analyte name (override, optional)",
            value="",
            key="de_analyte",
            help="Overrides path-inferred analyte name.",
        )
    with col_sig:
        sig_type = st.selectbox(
            "Signal column",
            ["auto", "intensity", "transmittance", "absorbance", "reflectance"],
            index=0,
            key="de_sig",
        )

    if st.button("Load Dataset", key="de_load_btn", type="primary"):
        if not raw_path.strip():
            st.warning("Enter a dataset path above.")
            return
        p = _safe_path(raw_path)
        if p is None:
            st.error("Path must be inside the project directory (no traversal allowed).")
            return
        if not p.exists():
            st.error(f"Path does not exist: `{p}`")
            return

        with st.spinner("Loading dataset…"):
            try:
                ds = load_dataset(
                    p,
                    signal_type=sig_type,  # type: ignore[arg-type]
                    normalisation=norm,    # type: ignore[arg-type]
                    analyte=analyte_override.strip() or None,
                )
            except Exception as exc:
                st.error(f"Load failed: {exc}")
                log.exception("Dataset load failed: %s", p)
                return

        st.session_state["de_dataset"] = ds
        st.success(
            f"Loaded {ds.n_samples} spectra × {ds.n_wavelengths} wavelength points."
        )


def _render_session_loader() -> None:
    """Load controls for SpectraAgent session pipeline_results.csv files."""
    import json

    sessions_root = _REPO_ROOT / "output" / "sessions"
    sessions = list_sessions(sessions_root)

    if not sessions:
        st.warning(
            "No sessions found in `output/sessions/`. "
            "Run a live acquisition with `spectraagent start` first."
        )
        return

    # Build display labels: "20260401_175916 — Ethanol (60 frames)"
    def _session_label(d: Path) -> str:
        meta_p = d / "session_meta.json"
        gas, n_frames = "unknown", 0
        if meta_p.exists():
            try:
                m = json.loads(meta_p.read_text(encoding="utf-8"))
                gas = m.get("gas_label", "unknown") or "unknown"
                n_frames = m.get("frame_count", 0) or 0
            except Exception:
                pass
        return f"{d.name}  —  {gas}  ({n_frames} frames)"

    session_labels = [_session_label(s) for s in sessions]
    selected_idx = st.selectbox(
        "Select session",
        range(len(sessions)),
        format_func=lambda i: session_labels[i],
        key="de_session_idx",
    )
    selected_session = sessions[selected_idx]

    col_norm, col_analyte = st.columns(2)
    with col_norm:
        norm = st.selectbox(
            "Normalisation",
            ["none", "snv", "minmax"],
            index=0,
            key="de_sess_norm",
        )
    with col_analyte:
        analyte_override = st.text_input(
            "Analyte override (optional)",
            value="",
            key="de_sess_analyte",
            help="Leave blank to use gas_label from session metadata.",
        )

    st.info(
        "**Feature mode:** Session results contain processed LSPR features "
        "(`wavelength_shift`, `peak_wavelength`, `snr`, `confidence_score`), "
        "not raw spectra. The **Feature Discovery** autoencoder sub-tab will "
        "not be physically meaningful in this mode. "
        "**Model Training** and **Cross-Dataset Analysis** work normally.",
        icon="ℹ️",
    )

    if st.button("Load Session", key="de_sess_load_btn", type="primary"):
        with st.spinner(f"Loading session {selected_session.name}…"):
            try:
                ds = load_session_csv(
                    selected_session,
                    normalisation=norm,  # type: ignore[arg-type]
                )
                if analyte_override.strip():
                    ds = SpectralDataset(
                        wavelengths=ds.wavelengths,
                        spectra=ds.spectra,
                        signal_type=ds.signal_type,
                        normalisation=ds.normalisation,
                        labels=ds.labels,
                        label_unit=ds.label_unit,
                        analyte=analyte_override.strip(),
                        config_id=ds.config_id,
                        metadata=ds.metadata,
                        source_paths=ds.source_paths,
                    )
            except Exception as exc:
                st.error(f"Load failed: {exc}")
                log.exception("Session load failed: %s", selected_session)
                return

        st.session_state["de_dataset"] = ds
        feature_names = ds.metadata[0].get("feature_names", []) if ds.metadata else []
        st.success(
            f"Loaded {ds.n_samples} frames from session `{selected_session.name}`. "
            f"Features: {feature_names}. "
            f"Analyte: {ds.analyte or 'unknown'}."
        )

    ds: SpectralDataset | None = st.session_state.get("de_dataset")
    if ds is None:
        st.info("No dataset loaded yet.")
        return

    # ── Summary metrics ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", ds.n_samples)
    c2.metric("Wavelength points", ds.n_wavelengths)
    c3.metric("WL range", f"{ds.wl_range[0]:.0f}–{ds.wl_range[1]:.0f} nm")
    c4.metric("Analyte", ds.analyte or "—")

    # ── Spectrum browser ─────────────────────────────────────────────────
    st.markdown("---")
    col_idx, col_mean = st.columns([3, 1])
    with col_idx:
        sample_idx = st.slider(
            "Sample index", 0, max(0, ds.n_samples - 1), 0, key="de_idx"
        )
    with col_mean:
        show_mean = st.checkbox("Overlay mean", value=True, key="de_mean")

    if PLOTLY_AVAILABLE:
        lbl_str = ""
        if ds.labels is not None:
            v = ds.labels[sample_idx]
            lbl_str = f" — label={v:.3g} {ds.label_unit}" if np.isfinite(v) else ""

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ds.wavelengths,
            y=ds.spectra[sample_idx],
            name=f"Sample {sample_idx}",
            line=dict(color="#4682b4"),
        ))
        if show_mean:
            fig.add_trace(go.Scatter(
                x=ds.wavelengths,
                y=ds.spectra.mean(axis=0),
                name="Mean",
                line=dict(color="#dc143c", dash="dot", width=1),
            ))
        fig.update_layout(
            title=f"Sample {sample_idx}{lbl_str}",
            xaxis_title="Wavelength (nm)",
            yaxis_title=f"{ds.signal_type} ({ds.normalisation})",
            height=350,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Label distribution ───────────────────────────────────────────────
    if ds.labels is not None and PLOTLY_AVAILABLE:
        valid_labels = ds.labels[np.isfinite(ds.labels)]
        if len(valid_labels) > 0:
            with st.expander("Label distribution"):
                fig2 = px.histogram(
                    x=valid_labels,
                    nbins=min(30, len(valid_labels)),
                    labels={"x": f"Label ({ds.label_unit})"},
                    title="Concentration / class label distribution",
                )
                fig2.update_layout(height=250, template="plotly_white")
                st.plotly_chart(fig2, use_container_width=True)

    # ── Metadata table ───────────────────────────────────────────────────
    if ds.metadata and PANDAS_AVAILABLE:
        with st.expander("Per-sample metadata"):
            st.dataframe(pd.DataFrame(ds.metadata), use_container_width=True)

    # ── Multi-dataset registry ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Multi-Dataset Registry")
    st.caption(
        "Register the loaded dataset under a name. "
        "Load multiple sensor configurations and register each one — "
        "the Cross-Dataset Analysis tab uses all registered datasets."
    )

    reg_name = st.text_input(
        "Registry name for this dataset",
        value=ds.analyte or "Dataset_1",
        key="de_reg_name",
        help="Use a descriptive name like 'CCS200_v1' or 'Ethanol_config_A'.",
    )

    if st.button("Register current dataset", key="de_register_btn"):
        if not reg_name.strip():
            st.warning("Enter a registry name.")
        else:
            registry: dict[str, SpectralDataset] = st.session_state.setdefault(
                "ds_registry", {}
            )
            registry[reg_name.strip()] = ds
            st.success(
                f"Registered '{reg_name}'. "
                f"Registry now contains {len(registry)} dataset(s)."
            )

    registry = st.session_state.get("ds_registry", {})
    if registry and PANDAS_AVAILABLE:
        rows = [
            {
                "Name": name,
                "Samples": d.n_samples,
                "WL points": d.n_wavelengths,
                "WL range": f"{d.wl_range[0]:.0f}–{d.wl_range[1]:.0f} nm",
                "Analyte": d.analyte or "—",
                "Normalisation": d.normalisation,
            }
            for name, d in registry.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if st.button("Clear registry", key="de_clear_registry"):
            st.session_state["ds_registry"] = {}
            st.rerun()


# ===========================================================================
# Sub-tab 2 — Feature Discovery
# ===========================================================================

def _render_feature_discovery() -> None:
    st.subheader("Feature Discovery")
    st.markdown(
        "Train a **1-D convolutional spectral autoencoder** to discover compact "
        "latent representations from raw spectra — with no physics assumptions. "
        "Visualise how samples cluster in the learned embedding space."
    )

    if not AUTOENCODER_AVAILABLE:
        _missing_module_error("autoencoder")
        return
    if not EMBED_VIZ_AVAILABLE:
        _missing_module_error("embed_viz")
        return

    ds: SpectralDataset | None = st.session_state.get("de_dataset")
    if ds is None:
        st.info("Load a dataset in the **Dataset Explorer** tab first.")
        return

    st.markdown(f"Dataset: **{ds.n_samples}** samples × **{ds.n_wavelengths}** wavelengths")

    # ── Autoencoder config ───────────────────────────────────────────────
    st.markdown("#### Autoencoder Configuration")
    c1, c2, c3 = st.columns(3)
    with c1:
        latent_dim = st.slider(
            "Latent dimension", 8, 256, 64, step=8, key="fd_latent",
            help="Bottleneck size. Smaller = more compressed but may lose detail.",
        )
        n_epochs = st.slider("Epochs", 10, 500, 100, step=10, key="fd_epochs")
    with c2:
        vae_mode = st.checkbox(
            "VAE (variational)", value=False, key="fd_vae",
            help="Adds KL-divergence for a smoother latent space.",
        )
        lr = st.select_slider(
            "Learning rate", [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
            value=1e-3, key="fd_lr",
        )
    with c3:
        viz_method = st.selectbox(
            "Embedding visualisation", ["pca", "tsne", "umap"],
            key="fd_viz",
            help="PCA is fast; UMAP/t-SNE better reveal cluster structure.",
        )
        batch_size = st.slider("Batch size", 8, 128, 32, step=8, key="fd_batch")

    if st.button("Train Autoencoder", key="fd_train_btn", type="primary"):
        spectra_f32 = ds.spectra.astype(np.float32)
        cfg = AutoencoderConfig(
            input_length=ds.n_wavelengths,
            latent_dim=latent_dim,
            vae=vae_mode,
        )
        model = SpectralAutoencoder(cfg)

        progress_bar = st.progress(0, text="Initialising…")
        epoch_status = st.empty()

        def _on_epoch(epoch: int, total: int,
                      train_loss: float, val_loss: float) -> None:
            pct = int(epoch / total * 100)
            progress_bar.progress(pct, text=f"Epoch {epoch}/{total}")
            if epoch == 1 or epoch % max(1, total // 5) == 0 or epoch == total:
                epoch_status.caption(
                    f"Epoch {epoch}/{total} — "
                    f"train={train_loss:.5f}  val={val_loss:.5f}"
                )

        with st.spinner("Training autoencoder…"):
            try:
                hist = train_autoencoder(
                    model,
                    spectra_f32,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    verbose=False,
                    epoch_callback=_on_epoch,
                )
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                log.exception("Autoencoder training failed")
                return

        st.session_state["fd_model"] = model
        st.session_state["fd_history"] = hist
        # Clear stale latents from a previous model
        st.session_state.pop("fd_latents", None)
        progress_bar.progress(100, text="Done!")
        st.success(
            f"Training complete — "
            f"final train loss: {hist['train_loss'][-1]:.5f}  "
            f"val loss: {hist['val_loss'][-1]:.5f}"
        )

    # ── Training curves ──────────────────────────────────────────────────
    if "fd_history" in st.session_state and PLOTLY_AVAILABLE:
        with st.expander("Training loss curves"):
            st.plotly_chart(
                _loss_curve_figure(st.session_state["fd_history"],
                                   title="Autoencoder Reconstruction Loss"),
                use_container_width=True,
            )

    # ── Embedding visualisation ──────────────────────────────────────────
    model: SpectralAutoencoder | None = st.session_state.get("fd_model")
    if model is None:
        return

    st.markdown("---")
    st.markdown("#### Embedding Visualisation")

    if st.button("Encode all spectra + visualise", key="fd_encode_btn"):
        with st.spinner("Encoding spectra…"):
            try:
                latents = model.encode_numpy(ds.spectra.astype(np.float32))
                st.session_state["fd_latents"] = latents
            except Exception as exc:
                st.error(f"Encoding failed: {exc}")
                log.exception("Autoencoder encode failed")
                return

    latents: np.ndarray | None = st.session_state.get("fd_latents")
    if latents is not None:
        colour_by = ds.labels if ds.labels is not None else None
        colour_label = f"Concentration ({ds.label_unit})" if colour_by is not None else "Index"

        with st.spinner(f"Computing {viz_method.upper()} projection…"):
            try:
                fig = plot_embedding(
                    latents,
                    colour_by=colour_by,
                    colour_label=colour_label,
                    title=f"Latent Space — {viz_method.upper()}",
                    method=viz_method,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Visualisation failed: {exc}")
                log.exception("Embedding visualisation failed")

        st.caption(
            f"Latent dim={latents.shape[1]} → {viz_method.upper()} 2-D projection. "
            "Well-separated clusters indicate discriminative learned representations."
        )

        # ── Reconstruction quality ────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Reconstruction Quality")
        recon_idx = st.slider(
            "Sample", 0, max(0, ds.n_samples - 1), 0, key="fd_recon_idx"
        )
        if st.button("Show original vs reconstructed", key="fd_recon_btn"):
            orig = ds.spectra[recon_idx: recon_idx + 1].astype(np.float32)
            try:
                recon = model.reconstruct_numpy(orig)
                mse = float(((orig - recon) ** 2).mean())
                if PLOTLY_AVAILABLE:
                    fig_r = go.Figure([
                        go.Scatter(
                            x=ds.wavelengths, y=orig[0],
                            name="Original", line=dict(color="#4682b4"),
                        ),
                        go.Scatter(
                            x=ds.wavelengths, y=recon[0],
                            name="Reconstructed",
                            line=dict(color="#dc143c", dash="dot"),
                        ),
                    ])
                    fig_r.update_layout(
                        title=f"Sample {recon_idx}: reconstruction MSE = {mse:.6f}",
                        xaxis_title="Wavelength (nm)",
                        yaxis_title=f"{ds.signal_type}",
                        height=320,
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_r, use_container_width=True)
                st.metric("Reconstruction MSE", f"{mse:.6f}")
            except Exception as exc:
                st.error(f"Reconstruction failed: {exc}")


# ===========================================================================
# Sub-tab 3 — Model Training
# ===========================================================================

def _render_model_training() -> None:
    st.subheader("Model Training")
    st.markdown(
        "Train a **multi-task model** (analyte classification + concentration "
        "regression) or a **contrastive encoder** (few-shot analyte "
        "fingerprinting) on the loaded dataset."
    )

    if not MODELS_AVAILABLE:
        _missing_module_error("models")
        return

    ds: SpectralDataset | None = st.session_state.get("de_dataset")
    if ds is None:
        st.info("Load a dataset in the **Dataset Explorer** tab first.")
        return

    # ── Model type ───────────────────────────────────────────────────────
    model_type = st.radio(
        "Model type",
        ["Multi-Task (classification + regression)", "Contrastive (analyte fingerprinting)"],
        horizontal=True,
        key="mt_type",
    )
    is_multitask = "Multi-Task" in model_type

    # ── Hyperparameters ──────────────────────────────────────────────────
    st.markdown("#### Hyperparameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        embed_dim = st.slider(
            "Embedding / latent dim", 8, 256, 64, step=8, key="mt_embed"
        )
        n_epochs = st.slider("Epochs", 10, 500, 100, step=10, key="mt_epochs")
    with c2:
        lr = st.select_slider(
            "Learning rate", [1e-4, 5e-4, 1e-3, 2e-3],
            value=1e-3, key="mt_lr",
        )
        batch_size = st.slider("Batch size", 8, 128, 32, key="mt_batch")
    with c3:
        backbone = st.selectbox(
            "Backbone", ["gru", "cnn", "transformer"], key="mt_backbone",
            help="GRU handles variable-length sequences well; CNN is fastest.",
        )
        if is_multitask:
            int_labels_preview = _int_labels_from_dataset(ds)
            default_n = (
                int(int_labels_preview.max() + 1)
                if int_labels_preview is not None else 2
            )
            n_analytes = st.number_input(
                "Analyte classes", min_value=2, max_value=50,
                value=max(2, default_n), key="mt_n_analytes",
                help="Number of analyte classes for the classification head.",
            )

    # ── Train ────────────────────────────────────────────────────────────
    if st.button("Train Model", key="mt_train_btn", type="primary"):
        spectra_f32 = ds.spectra.astype(np.float32)
        n_wl = ds.n_wavelengths
        int_labels = _int_labels_from_dataset(ds)
        concs_f32 = (
            ds.labels.astype(np.float32) if ds.labels is not None else None
        )

        with st.spinner("Training…"):
            try:
                if is_multitask:
                    cfg = MultiTaskConfig(
                        input_dim=n_wl,
                        embed_dim=embed_dim,
                        hidden_dim=max(64, embed_dim * 2),
                        n_analytes=int(n_analytes),
                        backbone=backbone,  # type: ignore[arg-type]
                        predict_concentration=(ds.labels is not None),
                        predict_qc=False,
                        n_layers=2,
                    )
                    trained_model: nn.Module = MultiTaskModel(cfg)
                    hist = train_multi_task(
                        trained_model,  # type: ignore[arg-type]
                        spectra_f32,
                        analyte_labels=int_labels,
                        concentrations=concs_f32,
                        n_epochs=n_epochs,
                        batch_size=batch_size,
                        lr=lr,
                        verbose=False,
                    )
                else:
                    if int_labels is None:
                        st.error(
                            "Contrastive training requires analyte class labels. "
                            "Ensure the dataset has numeric labels that map to classes."
                        )
                        return
                    cfg_c = ContrastiveConfig(
                        input_dim=n_wl,
                        embed_dim=embed_dim,
                        loss_type="supcon",
                    )
                    trained_model = ContrastiveEncoder(cfg_c)
                    hist = train_contrastive(
                        trained_model,  # type: ignore[arg-type]
                        spectra_f32,
                        int_labels,
                        n_epochs=n_epochs,
                        batch_size=batch_size,
                        lr=lr,
                        verbose=False,
                    )

                st.session_state["mt_model"] = trained_model
                st.session_state["mt_history"] = hist
                st.session_state["mt_model_type"] = model_type
                st.session_state["mt_n_wl"] = n_wl
                final_loss = hist.get("train_loss", [0.0])[-1]

                # ── Log to MLflow ────────────────────────────────────
                mlflow_run_id: str | None = None
                if MLFLOW_AVAILABLE:
                    try:
                        tracker = ExperimentTracker(
                            experiment_name="SpectraAgent_Science_Tab"
                        )
                        run_name = (
                            f"{ds.analyte or 'unknown'}_{model_type.split()[0]}_"
                            f"{time.strftime('%Y%m%d_%H%M%S')}"
                        )
                        with tracker.start_run(
                            run_name=run_name,
                            tags={"source": "Tab5_ModelTraining"},
                        ):
                            mlflow_run_id = tracker.log_nn_run(
                                model_type=model_type,
                                params={
                                    "embed_dim": embed_dim,
                                    "n_epochs": n_epochs,
                                    "lr": lr,
                                    "batch_size": batch_size,
                                    "backbone": backbone,
                                },
                                history=hist,
                                analyte=ds.analyte,
                                n_samples=ds.n_samples,
                                n_features=n_wl,
                                session_id=ds.config_id,
                            )
                    except Exception as _mlf_exc:
                        log.warning("MLflow logging failed (non-fatal): %s", _mlf_exc)

                run_msg = (
                    f" MLflow run: `{mlflow_run_id}`" if mlflow_run_id else ""
                )
                st.success(
                    f"Training complete — final train loss: {final_loss:.5f}.{run_msg}"
                )

            except Exception as exc:
                st.error(f"Training failed: {exc}")
                log.exception("Model training failed")

    # ── Loss curves ──────────────────────────────────────────────────────
    if "mt_history" in st.session_state and PLOTLY_AVAILABLE:
        with st.expander("Training loss curves", expanded=True):
            st.plotly_chart(
                _loss_curve_figure(st.session_state["mt_history"],
                                   title="Model Training Loss"),
                use_container_width=True,
            )

    # ── Save / download ──────────────────────────────────────────────────
    if "mt_model" not in st.session_state:
        return

    st.markdown("---")
    st.markdown("#### Save Trained Model")
    trained_model = st.session_state["mt_model"]

    save_name = st.text_input(
        "Filename prefix (without extension)",
        value="model",
        key="mt_save_name",
    )

    col_save, col_dl = st.columns(2)

    with col_save:
        if st.button("Save to output/models/", key="mt_save_btn"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_dir = _REPO_ROOT / "output" / "models"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{save_name}_{timestamp}.pt"
            checkpoint = {
                "model_state_dict": trained_model.state_dict(),
                "model_type": st.session_state.get("mt_model_type", "unknown"),
                "timestamp": timestamp,
                "n_wl": st.session_state.get("mt_n_wl"),
                "analyte": ds.analyte,
                "n_samples_trained_on": ds.n_samples,
            }
            if hasattr(trained_model, "config"):
                checkpoint["config"] = trained_model.config
            torch.save(checkpoint, save_path)
            st.success(f"Saved to `{save_path.relative_to(_REPO_ROOT)}`")

    with col_dl:
        buf = io.BytesIO()
        torch.save(trained_model.state_dict(), buf)
        buf.seek(0)
        st.download_button(
            "Download weights (.pt)",
            data=buf,
            file_name=f"{save_name}.pt",
            mime="application/octet-stream",
            key="mt_dl_btn",
        )

    # ── Promote to live system ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Deploy to Live Acquisition")
    st.caption(
        "Copy this model to the SpectraAgent registry so `spectraagent start` "
        "uses it for real-time inference."
    )

    with st.expander("⚠️ What this does", expanded=False):
        st.markdown(
            """
**Promote to live system** copies the trained model weights to
`models/registry/cnn_classifier.pt` and writes a `registry_meta.json`
tracking the version, analyte, and training metadata.

After promoting:
1. Stop SpectraAgent if running: close the terminal or Ctrl+C
2. Restart: `run_spectraagent.bat --hardware` (or `--simulate`)
3. The new model is active for all future acquisition sessions

The previous model is **not deleted** — it remains in `output/models/`
and is tracked by `ModelVersionStore` for rollback if needed.
            """
        )

    if st.button("🚀 Promote to live system", key="mt_promote_btn"):
        import json as _json

        registry_dir = _REPO_ROOT / "models" / "registry"
        registry_dir.mkdir(parents=True, exist_ok=True)
        dest_model = registry_dir / "cnn_classifier.pt"
        dest_meta = registry_dir / "registry_meta.json"

        try:
            # Save full checkpoint to registry
            checkpoint = {
                "model_state_dict": trained_model.state_dict(),
                "model_type": st.session_state.get("mt_model_type", "unknown"),
                "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_wl": st.session_state.get("mt_n_wl"),
                "analyte": ds.analyte,
                "n_samples_trained_on": ds.n_samples,
                "source_session": ds.config_id,
            }
            if hasattr(trained_model, "config"):
                checkpoint["config"] = trained_model.config
            torch.save(checkpoint, dest_model)

            # Write human-readable metadata alongside the weights
            meta_record = {
                "promoted_at": checkpoint["promoted_at"],
                "model_type": checkpoint["model_type"],
                "analyte": ds.analyte,
                "n_wl": checkpoint["n_wl"],
                "n_samples_trained_on": ds.n_samples,
                "source_session": ds.config_id,
                "final_train_loss": (
                    st.session_state.get("mt_history", {})
                    .get("train_loss", [None])[-1]
                ),
            }
            dest_meta.write_text(
                _json.dumps(meta_record, indent=2, default=str),
                encoding="utf-8",
            )

            st.success(
                "Model promoted to `models/registry/cnn_classifier.pt`. "
                "Restart SpectraAgent to activate it."
            )
            log.info("Model promoted to registry: %s", dest_model)

        except Exception as exc:
            st.error(f"Promotion failed: {exc}")
            log.exception("Model promotion failed")


# ===========================================================================
# Sub-tab 4 — Cross-Dataset Analysis
# ===========================================================================

def _render_cross_dataset() -> None:
    st.subheader("Cross-Dataset Analysis")
    st.markdown(
        "**Leave-one-config-out benchmark**: trains a classifier on all registered "
        "sensor configurations except one, then tests on the held-out config. "
        "This directly measures the platform's cross-sensor generalisation — "
        "the core publishable metric."
    )
    st.info(
        "**Minimum requirement:** At least 3 sensor configurations for a meaningful "
        "result.  If you only have 1 physical sensor, generate 3 variants with "
        "`spectraagent robustness --param integration_time --range 45:55 --steps 3` "
        "and load each output directory as a separate config below.",
        icon="💡",
    )

    with st.expander("📋 How to generate multi-config data (single-sensor lab)", expanded=False):
        st.markdown(
            """
The cross-dataset benchmark proves **cross-sensor generalisation** — the core
publication claim. You need data from ≥ 3 distinct sensor configurations.

**Option A — Robustness sweep (automated, recommended)**

Run the built-in robustness tool to auto-generate 3 config variants by sweeping
integration time ±10% (a standard ICH §4.8 robustness parameter):

```bash
spectraagent robustness --param integration_time --range 45:55 --steps 3 --gas Ethanol
```

This creates 3 session directories under `output/robustness/`. Load each one in
**Dataset Explorer → SpectraAgent session results**, register each as a separate config
(`Config_45ms`, `Config_50ms`, `Config_55ms`), then return here.

**Option B — Manual collection**

Collect calibration sessions under these conditions and treat each as a config:

| Config | Change from nominal |
|---|---|
| Config A | Integration time = 50 ms (normal) |
| Config B | Integration time = 100 ms (×2 slower) |
| Config C | Reference recaptured after 4 h (aged reference) |

Load each session in Dataset Explorer and register with distinct names.

**Option C — Use existing Joy_Data**

If you have Joy_Data for ≥ 3 different analytes or chip preparations, treat each
directory as a separate config. The benchmark generalises across analyte types
as well as sensor configurations.

**After registering ≥ 3 configs**, return here and click **Run Benchmark**.
The result is the leave-one-config-out accuracy — the number that goes in your
Methods section as evidence of sensor-agnostic generalisation.
            """
        )

    if not BENCHMARK_AVAILABLE:
        _missing_module_error("benchmark")
        return

    registry: dict[str, SpectralDataset] = st.session_state.get("ds_registry", {})

    if len(registry) < 2:
        st.info(
            "Register **at least 2 datasets** (one per sensor configuration) "
            "in the Dataset Explorer tab first."
        )
        if registry:
            st.caption(f"Currently registered: {list(registry.keys())}")
        return

    st.markdown(
        f"**{len(registry)} configurations registered:** "
        f"{', '.join(f'`{n}`' for n in registry)}"
    )

    # ── Benchmark settings ───────────────────────────────────────────────
    st.markdown("#### Benchmark Settings")
    c1, c2 = st.columns(2)
    with c1:
        task = st.radio(
            "Task", ["classification", "regression"],
            horizontal=True, key="ba_task",
            help="Classification: analyte ID accuracy. Regression: concentration MAE/R².",
        )
        classifier = st.selectbox(
            "Base classifier / regressor", ["knn", "svc", "rf"],
            key="ba_clf",
            help="kNN: interpretable baseline. SVC: strong with small data. RF: robust.",
        )
    with c2:
        n_pca = st.slider(
            "PCA components (feature reduction)", 2, 64, 16, key="ba_pca",
            help="Applied before classification when no model encoder is provided.",
        )
        interp_wl = st.checkbox(
            "Interpolate to common wavelength grid",
            value=True, key="ba_interp",
            help="Required when configs have different wavelength ranges.",
        )

    # Optional: use trained model encoder
    use_model_enc = False
    mt_model: nn.Module | None = st.session_state.get("mt_model")
    if mt_model is not None and MODELS_AVAILABLE:
        use_model_enc = st.checkbox(
            "Use trained model encoder instead of PCA",
            value=False,
            key="ba_use_enc",
            help=(
                "Replaces PCA with the multi-task / contrastive model's "
                "embed_numpy() method for richer feature extraction."
            ),
        )

    if st.button("Run Benchmark", key="ba_run_btn", type="primary"):
        encoder = None
        if use_model_enc and mt_model is not None:
            if hasattr(mt_model, "embed_numpy"):
                encoder = mt_model.embed_numpy
            else:
                st.warning(
                    "Trained model does not have embed_numpy(); falling back to PCA."
                )

        with st.spinner("Running leave-one-config-out benchmark…"):
            try:
                result = run_benchmark(
                    registry,
                    task=task,    # type: ignore[arg-type]
                    encoder=encoder,
                    n_components_pca=n_pca,
                    classifier=classifier,   # type: ignore[arg-type]
                    interpolate_wavelengths=interp_wl,
                )
                st.session_state["ba_result"] = result
            except Exception as exc:
                st.error(f"Benchmark failed: {exc}")
                log.exception("Cross-dataset benchmark failed")
                return

    result: BenchmarkResult | None = st.session_state.get("ba_result")
    if result is None:
        return

    # ── Summary metrics ──────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    def _fmt(v: float) -> str:
        return f"{v:.4f}" if not math.isnan(v) else "N/A"

    if result.task == "classification":
        c1.metric("Mean cross-config accuracy", _fmt(result.mean_accuracy))
    else:
        c1.metric("Mean MAE", _fmt(result.mean_mae))
        c2.metric("Mean R²", _fmt(result.mean_r2))
    c3.metric("Mean silhouette", _fmt(result.mean_silhouette))

    # ── Per-configuration table ──────────────────────────────────────────
    st.markdown("#### Per-Configuration Results")
    if PANDAS_AVAILABLE:
        rows: list[dict[str, Any]] = []
        for r in result.config_results:
            row: dict[str, Any] = {
                "Held-out config": r.test_config,
                "Trained on": ", ".join(r.train_configs),
                "N train": r.n_train,
                "N test": r.n_test,
            }
            if r.accuracy is not None:
                row["Accuracy"] = _fmt(r.accuracy)
            if r.balanced_accuracy is not None:
                row["Bal. Acc."] = _fmt(r.balanced_accuracy)
            if r.mae is not None:
                row["MAE"] = _fmt(r.mae)
            if r.r2 is not None:
                row["R²"] = _fmt(r.r2)
            if r.embedding_separation is not None:
                row["Silhouette"] = _fmt(r.embedding_separation)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── Bar chart ────────────────────────────────────────────────────────
    if PLOTLY_AVAILABLE and result.config_results:
        configs = [r.test_config for r in result.config_results]
        if result.task == "classification":
            values = [
                r.accuracy if (r.accuracy is not None and not math.isnan(r.accuracy))
                else 0.0
                for r in result.config_results
            ]
            ylabel, title = "Accuracy", "Cross-Config Classification Accuracy"
        else:
            values = [
                r.mae if (r.mae is not None and not math.isnan(r.mae))
                else 0.0
                for r in result.config_results
            ]
            ylabel, title = "MAE (ppm)", "Cross-Config Concentration MAE"

        fig = px.bar(
            x=configs, y=values,
            labels={"x": "Held-out Config", "y": ylabel},
            title=title,
            text_auto=".3f",
        )
        fig.update_layout(height=320, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # ── Full text report ─────────────────────────────────────────────────
    with st.expander("Full benchmark report (text)"):
        st.code(result.summary(), language="text")


# ===========================================================================
# Sub-tab 5 — Publication Figures
# ===========================================================================

def _render_publication_figures() -> None:
    st.subheader("Publication Figures")
    st.markdown(
        "Export-ready figures for journal submission. "
        "SVG is recommended for vector output; PNG at 2× scale for raster."
    )

    if not PLOTLY_AVAILABLE:
        _missing_module_error("plotly")
        return

    # Determine which figures are available based on session state
    available: dict[str, bool] = {
        "Mean ± SD spectral envelope": "de_dataset" in st.session_state,
        "Embedding scatter plot": "fd_latents" in st.session_state,
        "Wavelength importance heatmap": (
            "mt_model" in st.session_state and FEAT_IMP_AVAILABLE
        ),
        "Cross-dataset accuracy bar chart": "ba_result" in st.session_state,
        "Training loss curve": (
            "fd_history" in st.session_state
            or "mt_history" in st.session_state
        ),
    }
    ready = [name for name, ok in available.items() if ok]
    not_ready = [name for name, ok in available.items() if not ok]

    if not ready:
        st.info(
            "No figures are ready yet. "
            "Load a dataset, train a model, or run the benchmark first."
        )
        return

    if not_ready:
        st.caption(
            f"Unavailable (data not yet generated): {', '.join(not_ready)}"
        )

    selected = st.multiselect(
        "Select figures to generate",
        ready,
        default=ready[:1],
        key="pub_select",
    )
    if not selected:
        return

    # ── Export settings ──────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        fig_w = st.number_input("Width (px)", 400, 3000, 1400, step=100, key="pub_w")
    with c2:
        fig_h = st.number_input("Height (px)", 250, 2000, 700, step=50, key="pub_h")
    with c3:
        fmt = st.selectbox("Export format", ["svg", "png", "pdf"], key="pub_fmt")

    st.markdown("---")
    ds: SpectralDataset | None = st.session_state.get("de_dataset")

    for fig_name in selected:
        st.markdown(f"**{fig_name}**")
        fig = _build_pub_figure(fig_name, ds)
        if fig is None:
            st.warning(f"Could not build figure '{fig_name}'.")
            continue

        fig.update_layout(width=fig_w, height=fig_h, font=dict(size=14))
        st.plotly_chart(fig, use_container_width=True)

        safe_name = fig_name.lower().replace(" ", "_").replace("/", "-")
        try:
            img_bytes = fig.to_image(
                format=fmt, width=fig_w, height=fig_h, scale=2
            )
            mime_map = {"svg": "image/svg+xml", "png": "image/png",
                        "pdf": "application/pdf"}
            st.download_button(
                f"Download as {fmt.upper()}",
                data=img_bytes,
                file_name=f"{safe_name}.{fmt}",
                mime=mime_map.get(fmt, "application/octet-stream"),
                key=f"pub_dl_{safe_name}",
            )
        except Exception as exc:
            # kaleido not installed — HTML fallback
            st.warning(
                f"Image export unavailable (install kaleido: "
                f"`pip install kaleido`). "
                f"Falling back to interactive HTML. Error: {exc}"
            )
            html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
            st.download_button(
                "Download as interactive HTML",
                data=html_bytes,
                file_name=f"{safe_name}.html",
                mime="text/html",
                key=f"pub_html_{safe_name}",
            )

        st.markdown("---")


def _build_pub_figure(
    fig_name: str, ds: SpectralDataset | None
) -> go.Figure | None:
    """Build and return a publication-quality Plotly figure by name."""

    if fig_name == "Mean ± SD spectral envelope" and ds is not None:
        mean = ds.spectra.mean(axis=0)
        sd = ds.spectra.std(axis=0)
        fig = go.Figure([
            go.Scatter(
                x=np.concatenate([ds.wavelengths, ds.wavelengths[::-1]]),
                y=np.concatenate([mean + sd, (mean - sd)[::-1]]),
                fill="toself",
                fillcolor="rgba(70,130,180,0.2)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Mean ± 1 SD",
            ),
            go.Scatter(
                x=ds.wavelengths, y=mean,
                name="Mean spectrum",
                line=dict(color="#4682b4", width=2),
            ),
        ])
        fig.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title=f"Signal ({ds.signal_type}, norm={ds.normalisation})",
            template="plotly_white",
        )
        return fig

    if fig_name == "Embedding scatter plot" and "fd_latents" in st.session_state:
        latents: np.ndarray = st.session_state["fd_latents"]
        colour_by = ds.labels if (ds is not None and ds.labels is not None) else None
        return plot_embedding(
            latents,
            colour_by=colour_by,
            colour_label="Concentration" if colour_by is not None else "Index",
            title="Spectral Embedding (PCA)",
            method="pca",
        )

    if fig_name == "Wavelength importance heatmap" and ds is not None:
        mt_model = st.session_state.get("mt_model")
        if mt_model is None or not FEAT_IMP_AVAILABLE:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imp = gradient_attribution(
                    mt_model,
                    ds.spectra[:min(20, ds.n_samples)].astype(np.float32),
                    absolute_value=True,
                )
            bands = top_wavelength_bands(ds.wavelengths, imp, n_bands=5, width_nm=15)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ds.wavelengths, y=imp,
                fill="tozeroy",
                fillcolor="rgba(70,130,180,0.3)",
                line=dict(color="#4682b4"),
                name="Attribution",
            ))
            for band in bands:
                fig.add_vrect(
                    x0=band.start_nm, x1=band.end_nm,
                    fillcolor="orange", opacity=0.2,
                    annotation_text=f"{band.center_nm:.0f} nm",
                    annotation_position="top left",
                    line_width=0,
                )
            fig.update_layout(
                xaxis_title="Wavelength (nm)",
                yaxis_title="Feature attribution importance",
                template="plotly_white",
            )
            return fig
        except Exception as exc:
            log.warning("Importance figure failed: %s", exc)
            return None

    if fig_name == "Cross-dataset accuracy bar chart":
        result: BenchmarkResult | None = st.session_state.get("ba_result")
        if result is None:
            return None
        configs = [r.test_config for r in result.config_results]
        if result.task == "classification":
            values = [
                r.accuracy if (r.accuracy is not None and not math.isnan(r.accuracy))
                else 0.0
                for r in result.config_results
            ]
            ylabel = "Accuracy"
            title = "Cross-Dataset Classification Accuracy"
        else:
            values = [
                r.mae if (r.mae is not None and not math.isnan(r.mae))
                else 0.0
                for r in result.config_results
            ]
            ylabel = "MAE (ppm)"
            title = "Cross-Dataset Concentration MAE"
        fig = px.bar(
            x=configs, y=values,
            labels={"x": "Held-out configuration", "y": ylabel},
            title=title,
            text_auto=".3f",
            color=values,
            color_continuous_scale="Blues",
        )
        fig.update_layout(template="plotly_white", showlegend=False)
        return fig

    if fig_name == "Training loss curve":
        hist = (
            st.session_state.get("mt_history")
            or st.session_state.get("fd_history")
            or {}
        )
        return _loss_curve_figure(hist, title="Model Training Loss")

    return None
