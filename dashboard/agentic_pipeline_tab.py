"""
Agentic Automation Pipeline Tab
================================
Orchestrates Agents 01-05 in a single sequential Streamlit dashboard.

Session-state keys (all prefixed with "ap_"):
  ap_step           int 1-4     current active step
  ap_meta           dict        gas/concentration/trial metadata
  ap_buffer         list        list of intensity arrays recorded
  ap_wl             ndarray     wavelengths from hardware or simulation
  ap_recording      bool        True while recording is active
  ap_preprocessed   list        processed intensity arrays
  ap_pp_wl          ndarray     wavelength axis for preprocessed data
  ap_model_trained  bool        True after training completes
  ap_pred_history   list        [(conc_mean, conc_std), ...]
  ap_gas_labels     list        gas type labels for prediction history
"""

from __future__ import annotations

import contextlib
from datetime import datetime
import json
from pathlib import Path
import re

# ── project root ──────────────────────────────────────────────────────────────
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ── hardware ──────────────────────────────────────────────────────────────────
try:
    from src.acquisition import CCS200Spectrometer

    if CCS200Spectrometer is None:
        raise ImportError("CCS200Spectrometer unavailable")
    _HW_AVAILABLE = True
except Exception:
    _HW_AVAILABLE = False

# ── signal processing ─────────────────────────────────────────────────────────
try:
    from src.preprocessing.baseline import als_baseline
    from src.preprocessing.baseline import correct_baseline as baseline_correction
    from src.preprocessing.denoising import smooth_spectrum
    from src.preprocessing.normalization import normalize_spectrum
    from src.preprocessing.quality import compute_snr, estimate_noise_metrics

    # detect_outliers not yet in src.preprocessing — try legacy location
    try:
        from gas_analysis.core.signal_proc import detect_outliers
    except Exception:

        def detect_outliers(x, threshold=3.0):  # type: ignore[misc]
            return x

    _SP_AVAILABLE = True
except Exception:
    try:
        from gas_analysis.core.preprocessing import (
            baseline_correction,
            compute_snr,
            estimate_noise_metrics,
            normalize_spectrum,
        )
        from gas_analysis.core.signal_proc import als_baseline, smooth_spectrum

        _SP_AVAILABLE = True
    except Exception:
        _SP_AVAILABLE = False

# ── ML models ─────────────────────────────────────────────────────────────────
try:
    from gas_analysis.core.intelligence.classifier import CNNGasClassifier
    from gas_analysis.core.intelligence.gpr import GPRCalibration

    _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False

# ── LOD / kinetics ────────────────────────────────────────────────────────────
try:
    from src.scientific.lod import calculate_lod_3sigma, calculate_sensitivity

    _LOD_AVAILABLE = True
except Exception:
    try:
        from gas_analysis.core.scientific.lod import calculate_lod_3sigma, calculate_sensitivity

        _LOD_AVAILABLE = True
    except Exception:
        _LOD_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────


def _step_badge(n: int, label: str, done: bool, active: bool) -> str:
    colour = "#2ecc71" if done else ("#1f77b4" if active else "#888")
    icon = "✓" if done else str(n)
    return (
        f'<span style="background:{colour};color:white;border-radius:50%;'
        f'padding:2px 9px;font-weight:bold;margin-right:6px">{icon}</span>'
        f'<span style="color:{colour};font-weight:{"bold" if active else "normal"}">{label}</span>'
    )


def _simulate_frame(
    wl: np.ndarray,
    concentration_ppm: float = 100.0,
    sim_peak_nm: float = 532.0,
) -> np.ndarray:
    """Produce a realistic simulated frame with a Gaussian peak + noise.

    Parameters
    ----------
    wl:
        Wavelength array (nm).
    concentration_ppm:
        Analyte concentration used to scale peak amplitude.
    sim_peak_nm:
        Centre wavelength of the simulated Gaussian peak (nm).
        Override to match your sensor's expected peak position.
    """
    peak = np.exp(-0.5 * ((wl - sim_peak_nm) / 18) ** 2) * (concentration_ppm / 100) * 4.5
    noise = np.random.normal(0, 0.04, len(wl))
    return peak + noise


def _acquire_frames(
    integration_ms: int,
    n_frames: int,
    concentration_ppm: float,
    chart_ph,
    sim_peak_nm: float = 532.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Acquire n_frames from CCS200 (or simulation) and return
    (wavelengths, mean_intensities).  Updates chart_ph live.

    Handles stale VISA handle errors (-1073807339 = VI_ERROR_INV_OBJECT)
    by closing and reinitialising the device once on failure.
    """
    wl = np.linspace(200, 1000, 1000)

    def _open_spec():
        s = CCS200Spectrometer(integration_time_s=integration_ms / 1000.0)
        return s, s.get_wavelengths()

    spec = None
    if _HW_AVAILABLE:
        try:
            spec, wl = _open_spec()
        except Exception:
            spec = None  # fall back to simulation if init fails

    try:
        frames = []
        prog = st.progress(0, text="Acquiring…")
        for i in range(n_frames):
            # Safe default — replaced by hardware read when available
            frame = _simulate_frame(wl, concentration_ppm, sim_peak_nm)
            if spec is not None:
                # Retry once on stale VISA handle
                for attempt in range(2):
                    try:
                        frame = spec.get_data()
                        break
                    except RuntimeError as exc:
                        if attempt == 0 and "getScanData" in str(exc):
                            with contextlib.suppress(Exception):
                                spec.close()
                            import time as _time

                            _time.sleep(0.5)
                            try:
                                spec, wl = _open_spec()
                            except Exception:
                                spec = None
                                frame = _simulate_frame(wl, concentration_ppm, sim_peak_nm)
                                break
                        else:
                            raise

            frames.append(frame)
            current_mean = np.mean(frames, axis=0)

            # Live chart update every 3 frames
            if chart_ph is not None and (i % 3 == 0 or i == n_frames - 1):
                fig = go.Figure(
                    go.Scatter(
                        x=wl,
                        y=current_mean,
                        mode="lines",
                        name=f"Live Avg ({i + 1}/{n_frames})",
                        line=dict(color="#FF4B4B", width=1.5),
                    )
                )
                fig.update_layout(
                    title=f"Live Acquisition — Frame {i + 1}/{n_frames}",
                    xaxis_title="Wavelength (nm)",
                    yaxis_title="Intensity (a.u.)",
                    height=320,
                    margin=dict(t=40, b=30, l=40, r=20),
                )
                chart_ph.plotly_chart(fig, use_container_width=True)

            prog.progress((i + 1) / n_frames, text=f"Frame {i + 1}/{n_frames}")

        prog.empty()
        return wl, np.mean(frames, axis=0)

    finally:
        if spec is not None:
            with contextlib.suppress(Exception):
                spec.close()


def _preprocess(
    wl: np.ndarray, intensity: np.ndarray, denoise: str, baseline: str, norm: str
) -> np.ndarray:
    """Apply the chosen preprocessing chain."""
    sig = intensity.copy()

    if _SP_AVAILABLE:
        if denoise == "Savitzky-Golay":
            sig = smooth_spectrum(sig, window=11, poly_order=2)
        elif denoise == "Wavelet (DWT-db4)":
            sig = smooth_spectrum(sig, method="wavelet")

        if baseline == "ALS":
            try:
                sig = sig - als_baseline(sig, lam=1e5, p=0.01)
            except Exception:
                sig = baseline_correction(wl, sig, method="als")
        elif baseline == "Polynomial":
            sig = baseline_correction(wl, sig, method="polynomial", poly_order=2)

        if norm == "Min-Max [0,1]":
            sig = normalize_spectrum(sig, method="minmax")
        elif norm == "Z-score":
            sig = normalize_spectrum(sig, method="standard")

    return sig


def _scan_dataset_dir() -> list[Path]:
    """Return all CSVs inside data/automation_dataset/."""
    base = _REPO / "data" / "automation_dataset"
    if not base.exists():
        return []
    return sorted(base.rglob("*.csv"))


@st.cache_data(show_spinner=False)
def _load_csv_spectrum(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    """Load wavelength + intensity from a two-column CSV. Cached per file path."""
    df = pd.read_csv(path_str, header=None, names=["wl", "intensity"])
    return df["wl"].values, df["intensity"].values


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────


def render() -> None:
    st.header("🤖 Research-Grade Automation Pipeline")
    st.caption("Agents 01–05: Acquisition → Logging → Preprocessing → Training → Deployment")

    # ── init session state ────────────────────────────────────────────────
    ss = st.session_state
    ss.setdefault("ap_step", 1)
    ss.setdefault("ap_meta", {})
    ss.setdefault("ap_buffer", [])  # list of (wl, intensity) tuples
    ss.setdefault("ap_wl", None)
    ss.setdefault("ap_recording", False)
    ss.setdefault("ap_preprocessed", [])  # list of (wl, intensity, label)
    ss.setdefault("ap_model_trained", False)
    ss.setdefault("ap_pred_history", [])
    ss.setdefault("ap_inference_active", False)

    step = ss["ap_step"]

    # ── step progress banner ──────────────────────────────────────────────
    cols = st.columns(4)
    badges = [
        "1  Acquisition & Logging",
        "2  Preprocessing",
        "3  Training & Insights",
        "4  Deployment & Testing",
    ]
    for i, (col, lbl) in enumerate(zip(cols, badges), 1):
        col.markdown(_step_badge(i, lbl, step > i, step == i), unsafe_allow_html=True)
    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1 — Real-Time Acquisition & Logging  (Agents 01 + 02)
    # ═══════════════════════════════════════════════════════════════════════
    if step == 1:
        st.subheader("Step 1 — Acquisition & Logging")

        if not _HW_AVAILABLE:
            _sim_col1, _sim_col2 = st.columns([3, 1])
            _sim_col1.info(
                "🔵 **Simulation mode** — No ThorLabs CCS200 detected. Using synthetic Gaussian peaks."
            )
            ss.setdefault("ap_sim_peak_nm", 532.0)
            ss["ap_sim_peak_nm"] = _sim_col2.number_input(
                "Sim peak (nm)",
                min_value=200.0,
                max_value=1100.0,
                value=float(ss["ap_sim_peak_nm"]),
                step=1.0,
                help="Centre wavelength of the simulated Gaussian peak. "
                "Set to your sensor's expected peak position.",
                key="ap_sim_peak_input",
            )

        # ── Section A: Live Spectral Preview (Agent 01) ───────────────────
        st.markdown("#### 📡 Live Data View")
        st.caption(
            "Verify spectrometer connection and signal quality before logging. Preview frames are **not** recorded."
        )
        prev_left, prev_right = st.columns([3, 1])
        with prev_right:
            prev_int_ms = st.number_input(
                "Integration (ms)",
                10,
                5000,
                30,
                key="ap_prev_int",
                help="Integration time for preview",
            )
            preview_clicked = st.button(
                "👁️ Preview Spectrum", help="Single-shot preview — not saved to dataset"
            )
            if ss.get("ap_preview_frame") is not None:
                snr_prev = compute_snr(ss["ap_preview_frame"]) if _SP_AVAILABLE else 0.0
                noise_prev = (
                    estimate_noise_metrics(ss["ap_preview_wl"], ss["ap_preview_frame"]).rms
                    if _SP_AVAILABLE
                    else 0.0
                )
                st.metric("SNR", f"{snr_prev:.1f}", "✅ OK" if snr_prev >= 10 else "⚠️ Low")
                st.metric("RMS Noise", f"{noise_prev:.4f}")
        chart_preview_ph = prev_left.empty()
        if preview_clicked:
            with st.spinner("Acquiring preview…"):
                wl_p, fr_p = _acquire_frames(
                    prev_int_ms, 5, 100.0, chart_preview_ph,
                    sim_peak_nm=ss.get("ap_sim_peak_nm", 532.0),
                )
            ss["ap_preview_wl"] = wl_p
            ss["ap_preview_frame"] = fr_p
        if ss.get("ap_preview_frame") is not None:
            wl_p, fr_p = ss["ap_preview_wl"], ss["ap_preview_frame"]
            fig_prev = go.Figure(
                go.Scatter(
                    x=wl_p,
                    y=fr_p,
                    mode="lines",
                    name="Live Preview",
                    line=dict(color="limegreen", width=1.5),
                )
            )
            fig_prev.update_layout(
                title="Live Spectral Preview (read-only — not recorded)",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Intensity (a.u.)",
                height=280,
                margin=dict(t=30, b=20, l=40, r=20),
            )
            chart_preview_ph.plotly_chart(fig_prev, use_container_width=True)
        else:
            chart_preview_ph.info(
                "Press **👁️ Preview Spectrum** to display a live frame from the spectrometer."
            )

        st.markdown("---")
        # ── Section B: Pre-Acquisition Logging & Labeled Recording (Agent 02) ─
        st.markdown("#### 📋 Data Logging & Acquisition")
        st.caption(
            "Fill in metadata, then record labeled spectra. Each snapshot is auto-saved as a timestamped CSV."
        )

        # ── Metadata form ─────────────────────────────────────────────────
        with st.form("ap_meta_form"):
            c1, c2 = st.columns(2)
            with c1:
                gas = st.selectbox(
                    "Gas Type", ["Ethanol", "Methanol", "Isopropanol", "MixVOC", "Air", "Custom"]
                )
                if gas == "Custom":
                    gas = st.text_input("Custom gas name")
                conc = st.number_input("Concentration (ppm)", min_value=0.0, value=100.0, step=10.0)
            with c2:
                trial = st.number_input("Trial #", min_value=1, step=1, value=1)
                integration_ms = st.slider("Integration time (ms)", 10, 5000, 30, 10)
                n_frames = st.slider("Frames to average", 5, 200, 30)

            comments = st.text_input("Comments (optional)")
            save_meta = st.form_submit_button("✅ Confirm Metadata & Arm Recording", type="primary")

        if save_meta:
            import re as _re
            # Sanitize gas name: keep only alphanumeric, dash, and underscore chars
            gas = _re.sub(r"[^\w\-]", "_", gas.strip()) or "unknown"
            # Only reset the buffer when gas/concentration changes, not every save
            old_meta = ss.get("ap_meta", {})
            if old_meta.get("gas") != gas or old_meta.get("concentration_ppm") != conc:
                ss["ap_buffer"] = []
            ss["ap_meta"] = {
                "gas": gas,
                "concentration_ppm": conc,
                "trial": trial,
                "integration_ms": integration_ms,
                "n_frames": n_frames,
                "comments": comments,
            }
            ss["ap_recording"] = False
            st.success(f"Metadata saved — **{gas}** at **{conc} ppm**, trial {trial}.")

        meta = ss["ap_meta"]
        if not meta:
            st.warning("Please confirm metadata above before recording.")
            return

        # ── Recording controls ────────────────────────────────────────────
        st.markdown("### Recording Controls")
        rc1, rc2 = st.columns(2)
        start = rc1.button(
            "▶️ Record Snapshot",
            type="primary",
            help="Acquire one labelled snapshot from spectrometer",
        )
        go_next = rc2.button("➡️ Proceed to Preprocessing", disabled=len(ss["ap_buffer"]) == 0)

        # ── Live acquisition ──────────────────────────────────────────────
        chart_ph = st.empty()
        st.empty()

        if start:
            with st.spinner("Acquiring…"):
                wl, mean_int = _acquire_frames(
                    meta["integration_ms"],
                    meta["n_frames"],
                    meta["concentration_ppm"],
                    chart_ph,
                    sim_peak_nm=ss.get("ap_sim_peak_nm", 532.0),
                )

            # QC check
            snr_val = compute_snr(mean_int) if _SP_AVAILABLE else 99.0
            noise_rms = estimate_noise_metrics(wl, mean_int).rms if _SP_AVAILABLE else 0.0

            # QC: only hard-block on saturation — SNR is informational (dark/air ≈ 4 is normal)
            saturated = mean_int.max() > 60000
            snr_warn = snr_val < 3.0  # truly dead signal
            qc_pass = not saturated

            qc1, qc2, qc3 = st.columns(3)
            qc1.metric(
                "SNR",
                f"{snr_val:.1f}",
                "✅ Signal"
                if snr_val >= 10
                else ("⚠️ Low (dark/air?)" if snr_val >= 3 else "❌ Dead signal"),
            )
            qc2.metric("RMS Noise", f"{noise_rms:.4f}", "✅ OK" if noise_rms < 0.1 else "⚠️ High")
            qc3.metric("Saturation", "✅ None" if not saturated else "❌ SATURATED")
            if snr_warn:
                st.warning("SNR < 3 — signal may be dead. Check light source. Recording anyway.")

            if qc_pass:
                # Save to disk
                out_dir = (
                    _REPO
                    / "data"
                    / "automation_dataset"
                    / meta["gas"]
                    / f"{meta['concentration_ppm']}ppm"
                    / f"trial_{meta['trial']}"
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = (
                    out_dir
                    / f"{meta['gas']}_{meta['concentration_ppm']}ppm_T{meta['trial']}_{ts}.csv"
                )
                pd.DataFrame({"wavelength": wl, "intensity": mean_int}).to_csv(
                    csv_path, index=False
                )
                (out_dir / f"metadata_{ts}.json").write_text(json.dumps(meta, indent=2))

                ss["ap_buffer"].append(
                    {
                        "wl": wl,
                        "intensity": mean_int,
                        "label": f"{meta['gas']}_{meta['concentration_ppm']}ppm",
                        "path": str(csv_path),
                    }
                )
                ss["ap_wl"] = wl
                st.success(
                    f"✅ Snapshot saved — QC passed. Total: **{len(ss['ap_buffer'])} recording(s)**."
                )
            else:
                st.error("❌ QC failed. Check the spectrometer signal or adjust integration time.")

        # ── Session log ───────────────────────────────────────────────────
        if ss["ap_buffer"]:
            st.markdown("### Session Recording Log")
            log_rows = [
                {"Label": r["label"], "Path": Path(r["path"]).name} for r in ss["ap_buffer"]
            ]
            st.dataframe(pd.DataFrame(log_rows), use_container_width=True)

        if go_next:
            ss["ap_step"] = 2
            st.rerun()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2 — Preprocessing  (Agent 03)
    # ═══════════════════════════════════════════════════════════════════════
    elif step == 2:
        st.subheader("Step 2 — Preprocessing Pipeline")

        recorded = ss["ap_buffer"]
        disk_csvs = _scan_dataset_dir()
        items = []  # list of (label, wl_array, intensity_array)

        # ── Source selector ───────────────────────────────────────────────
        source_options = ["Session recordings (in memory)", "Load from automation_dataset/"]
        # Joy_Data is the primary research dataset — always offer it
        joy_root = _REPO / "data" / "JOY_Data"
        if not joy_root.exists():
            joy_root = _REPO / "Joy_Data"  # legacy fallback
        if joy_root.exists():
            source_options.append("Load from Joy_Data (existing research data)")

        source = st.radio("Data source", source_options, horizontal=True)

        if source == "Session recordings (in memory)":
            if not recorded:
                st.warning("No session recordings yet. Go to Step 1 to record spectra.")
            else:
                items = [(r["label"], r["wl"], r["intensity"]) for r in recorded]

        elif source == "Load from automation_dataset/":
            if not disk_csvs:
                st.warning("No CSVs found in `data/automation_dataset/`. Record some first.")
            else:
                selected = st.multiselect("Select CSVs", [p.name for p in disk_csvs])
                for p in disk_csvs:
                    if p.name in selected:
                        df = pd.read_csv(p)
                        wl_ = (
                            df["wavelength"].values
                            if "wavelength" in df.columns
                            else df.iloc[:, 0].values
                        )
                        it_ = (
                            df["intensity"].values
                            if "intensity" in df.columns
                            else df.iloc[:, 1].values
                        )
                        items.append((p.stem, wl_, it_))

        else:  # Joy_Data
            st.markdown("**Select gas(es) and concentration(s) to load into the pipeline.**")
            st.caption(
                "Each concentration folder's last 10 CSVs (steady-state) are averaged into one spectrum per group."
            )
            gas_dirs = sorted([d for d in joy_root.iterdir() if d.is_dir()])
            gas_names = [d.name for d in gas_dirs]
            chosen_gases = st.multiselect(
                "Gas types", gas_names, default=gas_names[:2] if len(gas_names) >= 2 else gas_names
            )

            # Collect all conc-dirs for chosen gases
            joy_items_available = {}
            for gd in gas_dirs:
                if gd.name not in chosen_gases:
                    continue
                for conc_dir in sorted(gd.iterdir()):
                    if not conc_dir.is_dir():
                        continue
                    m = re.search(r"[\-\s]([\d.]+)[\-\s]", conc_dir.name)
                    conc_val = float(m.group(1)) if m else None
                    if conc_val is None:
                        m2 = re.search(r"([\d.]+)\s*ppm", conc_dir.name, re.IGNORECASE)
                        conc_val = float(m2.group(1)) if m2 else 0.0
                    label = f"{gd.name}_{conc_val}ppm"
                    joy_items_available.setdefault(label, []).extend(sorted(conc_dir.glob("*.csv")))

            chosen_labels = st.multiselect(
                "Select groups (gas + concentration)",
                sorted(joy_items_available.keys()),
                default=sorted(joy_items_available.keys())[:6] if joy_items_available else [],
            )
            st.info(
                f"{len(chosen_labels)} groups selected — will average last 10 CSVs per group as steady-state representative spectrum."
            )

            if chosen_labels:
                for lbl in chosen_labels:
                    csv_list = joy_items_available[lbl]
                    steady_files = csv_list[-10:] if len(csv_list) >= 10 else csv_list
                    spectra_batch = []
                    wl_ref_arr = None
                    for f in steady_files:
                        wl_arr, int_arr = _load_csv_spectrum(str(f))
                        if wl_ref_arr is None:
                            wl_ref_arr = wl_arr
                        spectra_batch.append(int_arr)
                    if spectra_batch and wl_ref_arr is not None:
                        mean_spec = np.mean(spectra_batch, axis=0)
                        items.append((lbl, wl_ref_arr, mean_spec))

            if items:
                st.success(f"Loaded {len(items)} representative spectra from Joy_Data.")

        if not items:
            st.info("Select at least one data source above, then configure preprocessing below.")
            if st.button("⬅️ Back to Step 1", key="back1_from2"):
                ss["ap_step"] = 1
                st.rerun()
            return

        # ── Reference spectrum (for LSPR Δλ / differential signal) ──────
        st.markdown("---")
        st.markdown("##### Reference Spectrum (optional but recommended for LSPR sensors)")
        st.caption(
            "Provide an air/blank reference to compute differential signal and peak shift Δλ — the true LSPR sensing metric."
        )
        ref_col1, ref_col2 = st.columns([2, 1])
        ref_files_available = sorted(joy_root.glob("ref*.csv")) if joy_root.exists() else []
        use_ref = ref_col2.checkbox("Use reference spectrum", value=bool(ref_files_available))
        ss_ref = None
        if use_ref:
            if ref_files_available:
                chosen_ref = ref_col1.selectbox(
                    "Reference file", ref_files_available, format_func=lambda p: p.name
                )
                df_ref = pd.read_csv(chosen_ref, header=None, names=["wl", "intensity"])
                ss_ref = df_ref["intensity"].values
                ss["ap_ref_spectrum"] = ss_ref
                ss["ap_ref_wl"] = df_ref["wl"].values
                ref_col2.success(f"Loaded: {chosen_ref.name}")
            else:
                uploaded_ref = ref_col1.file_uploader("Upload reference CSV", type=["csv"])
                if uploaded_ref:
                    df_ref = pd.read_csv(uploaded_ref, header=None, names=["wl", "intensity"])
                    ss_ref = df_ref["intensity"].values
                    ss["ap_ref_spectrum"] = ss_ref
        elif "ap_ref_spectrum" in ss:
            del ss["ap_ref_spectrum"]

        # ── Preprocessing configuration ───────────────────────────────────
        st.markdown("---")
        pc1, pc2, pc3 = st.columns(3)
        denoise = pc1.selectbox("Denoising", ["Savitzky-Golay", "Wavelet (DWT-db4)", "None"])
        baseline_m = pc2.selectbox("Baseline Removal", ["ALS", "Polynomial", "None"])
        norm_m = pc3.selectbox("Normalization", ["Min-Max [0,1]", "Z-score", "None"])

        if st.button("⚙️ Run Preprocessing", type="primary"):
            ref_arr = ss.get("ap_ref_spectrum")
            # Clear stale reference peak so Step 3 re-derives it from the new data
            ss.pop("ap_ref_peak_wl", None)
            with st.spinner("Processing all spectra…"):
                ss["ap_preprocessed"] = []
                qc_rows = []
                for label, wl, raw in items:
                    proc = _preprocess(wl, raw, denoise, baseline_m, norm_m)
                    # Compute differential signal relative to reference if provided
                    diff_signal = None
                    if ref_arr is not None:
                        ref_interp = np.interp(wl, ss.get("ap_ref_wl", wl), ref_arr)
                        diff_signal = raw - ref_interp  # ΔI = I_sample − I_ref
                    entry = {"label": label, "wl": wl, "raw": raw, "processed": proc}
                    if diff_signal is not None:
                        entry["diff_signal"] = diff_signal
                    ss["ap_preprocessed"].append(entry)
                    snr_raw = compute_snr(raw) if _SP_AVAILABLE else 0.0
                    snr_proc = compute_snr(proc) if _SP_AVAILABLE else 0.0
                    noise_raw = estimate_noise_metrics(wl, raw).rms if _SP_AVAILABLE else 0.0
                    noise_proc = estimate_noise_metrics(wl, proc).rms if _SP_AVAILABLE else 0.0
                    qc_rows.append(
                        {
                            "Label": label,
                            "Raw SNR": f"{snr_raw:.1f}",
                            "Proc SNR": f"{snr_proc:.1f}",
                            "Noise (raw)": f"{noise_raw:.4f}",
                            "Noise (proc)": f"{noise_proc:.4f}",
                            "Has Δ-signal": "✅" if diff_signal is not None else "—",
                        }
                    )

            st.success(f"Preprocessed {len(ss['ap_preprocessed'])} spectra.")
            st.dataframe(pd.DataFrame(qc_rows), use_container_width=True)

        # ── Comparison plot ───────────────────────────────────────────────
        if ss["ap_preprocessed"]:
            sel_idx = st.selectbox(
                "Inspect",
                range(len(ss["ap_preprocessed"])),
                format_func=lambda i: ss["ap_preprocessed"][i]["label"],
            )
            item = ss["ap_preprocessed"][sel_idx]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=item["wl"],
                    y=item["raw"],
                    name="Raw",
                    line=dict(color="gray", dash="dot"),
                    opacity=0.6,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=item["wl"],
                    y=item["processed"],
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
            st.plotly_chart(fig, use_container_width=True)

        col_nav = st.columns(2)
        if col_nav[0].button("⬅️ Back to Step 1"):
            ss["ap_step"] = 1
            st.rerun()
        if col_nav[1].button("➡️ Proceed to Training", disabled=len(ss["ap_preprocessed"]) == 0):
            ss["ap_step"] = 3
            st.rerun()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3 — Feature Extraction & Training  (Agent 04)
    # ═══════════════════════════════════════════════════════════════════════
    elif step == 3:
        st.subheader("Step 3 — Insights, Feature Extraction & Model Training")

        pp = ss["ap_preprocessed"]
        if not pp:
            st.warning("No preprocessed data. Go back to Step 2.")
            if st.button("⬅️ Back to Step 2", key="back2_from3"):
                ss["ap_step"] = 2
                st.rerun()
            return

        # ── Feature extraction summary ────────────────────────────────────
        st.markdown("### Extracted Features")
        has_diff = any("diff_signal" in it for it in pp)
        ref_peak_wl = ss.get("ap_ref_peak_wl")
        feat_rows = []
        X_list, y_concs, y_labels, class_names = [], [], [], []
        for item in pp:
            wl = item["wl"]
            if has_diff and "diff_signal" in item:
                # LSPR differential features: peak shift Δλ is the primary sensing metric
                diff = item["diff_signal"]
                peak_idx_d = int(np.argmax(np.abs(diff)))
                peak_wl_d = float(wl[peak_idx_d])
                peak_int_d = float(diff[peak_idx_d])
                area_d = float(np.trapezoid(diff, wl))
                std_d = float(np.std(diff))
                if ref_peak_wl is None:
                    ref_peak_wl = float(wl[int(np.argmax(item["processed"]))])
                    ss["ap_ref_peak_wl"] = ref_peak_wl
                delta_lam = peak_wl_d - ref_peak_wl
                feat_vec = [delta_lam, peak_int_d, area_d, std_d]
                feat_row_extra = {
                    "Δλ (nm)": f"{delta_lam:.3f}",
                    "ΔI peak": f"{peak_int_d:.4f}",
                    "ΔI area": f"{area_d:.4f}",
                    "ΔI std": f"{std_d:.4f}",
                }
            else:
                # Fallback: standard spectral features
                sig = item["processed"]
                peak_idx = int(np.argmax(sig))
                feat_vec = [
                    float(sig[peak_idx]),
                    float(wl[peak_idx]),
                    float(np.trapezoid(sig, wl)),
                    float(np.std(sig)),
                ]
                feat_row_extra = {
                    "Peak (nm)": f"{wl[peak_idx]:.1f}",
                    "Peak Intensity": f"{sig[peak_idx]:.4f}",
                    "Area": f"{np.trapezoid(sig, wl):.2f}",
                    "Std": f"{np.std(sig):.4f}",
                }

            mc = re.search(r"([\d.]+)ppm", item["label"])
            conc = float(mc.group(1)) if mc else 0.0
            gas_name = item["label"].split("_")[0]
            if gas_name not in class_names:
                class_names.append(gas_name)
            y_labels.append(class_names.index(gas_name))
            y_concs.append(conc)
            X_list.append(feat_vec)
            row = {"Label": item["label"]}
            row.update(feat_row_extra)
            feat_rows.append(row)

        st.dataframe(pd.DataFrame(feat_rows), use_container_width=True)
        X = np.array(X_list)
        if has_diff:
            st.success(
                "🔬 LSPR differential features active — Δλ (peak shift), ΔI peak, ΔI area, ΔI std"
            )
        else:
            st.info(
                "ℹ️ Standard spectral features in use. Load a reference spectrum in Step 2 to enable LSPR Δλ features."
            )

        # ── Scientific metrics ────────────────────────────────────────────
        if len(y_concs) >= 3 and _LOD_AVAILABLE:
            st.markdown("### Scientific Metrics (LOD / Sensitivity / R²)")
            try:
                concs_arr = np.array(y_concs, dtype=float)
                peak_arr = X[:, 0]
                slope, intercept, r2 = calculate_sensitivity(concs_arr, peak_arr)
                noise_floor = float(np.std(peak_arr[: max(1, len(peak_arr) // 5)]))
                lod = calculate_lod_3sigma(noise_floor, slope)
                loq = 10.0 * noise_floor / abs(slope) if slope != 0 else float("inf")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Sensitivity (slope)", f"{slope:.4f} a.u./ppm")
                m2.metric("R² (Linearity)", f"{r2:.4f}")
                m3.metric("LOD (3σ/slope)", f"{lod:.3f} ppm")
                m4.metric("LOQ (10σ/slope)", f"{loq:.3f} ppm")
                ss["ap_lod"] = lod
                ss["ap_r2"] = r2
            except Exception as e:
                st.info(f"Metrics require ≥3 concentration points. ({e})")

        # ── Shared spectral arrays for all visualisations (hoisted) ──────
        wl_common = None
        Z = None
        concs_plot = None
        if len(pp) >= 2:
            wl_common = np.linspace(
                float(min(it["wl"].min() for it in pp)),
                float(max(it["wl"].max() for it in pp)),
                300,
            )
            # Use differential signal when available (LSPR), else processed spectrum
            sig_key = (
                "diff_signal"
                if (has_diff and all("diff_signal" in it for it in pp))
                else "processed"
            )
            Z = np.array([np.interp(wl_common, it["wl"], it[sig_key]) for it in pp])
            concs_plot = []
            for i, it in enumerate(pp):
                mc2 = re.search(r"([\d.]+)ppm", it["label"])
                concs_plot.append(float(mc2.group(1)) if mc2 else float(i))

        # ── Spectral Overlay (Rainbow Plot) ──────────────────────────────
        if len(pp) >= 2:
            st.markdown("### Spectral Overlay")
            y_lbl_ov = "ΔI (Differential Intensity)" if has_diff else "Intensity"
            fig_overlay = go.Figure()
            palette = [f"hsl({h},70%,50%)" for h in np.linspace(0, 240, len(pp))]
            for idx, (spec, item) in enumerate(zip(Z, pp)):
                fig_overlay.add_trace(
                    go.Scatter(
                        x=wl_common,
                        y=spec,
                        mode="lines",
                        name=item["label"],
                        line=dict(color=palette[idx]),
                    )
                )
            if has_diff and ref_peak_wl:
                fig_overlay.add_vline(
                    x=ref_peak_wl,
                    line_dash="dash",
                    line_color="white",
                    annotation_text=f"Ref λ={ref_peak_wl:.1f} nm",
                )
            fig_overlay.update_layout(
                title="Spectral Overlay — All Concentrations"
                + (" (Differential ΔI)" if has_diff else ""),
                xaxis_title="Wavelength (nm)",
                yaxis_title=y_lbl_ov,
                height=380,
                margin=dict(t=40, b=30, l=40, r=20),
            )
            st.plotly_chart(fig_overlay, use_container_width=True)

        # ── Calibration Curve ─────────────────────────────────────────────
        if len(pp) >= 3:
            st.markdown("### Calibration Curve")
            try:
                from scipy.stats import linregress

                concs_arr_plot = np.array(concs_plot, dtype=float)
                if has_diff:
                    # LSPR: Δλ (X[:,0]) vs concentration — the physically meaningful metric
                    responses = X[:, 0]
                    y_axis_lbl = "Δλ (nm)"
                    title_calib = "Calibration Curve — Peak Shift (Δλ) vs Concentration"
                else:
                    peak_idx_global = int(np.argmax(Z.mean(axis=0)))
                    responses = Z[:, peak_idx_global]
                    y_axis_lbl = "Peak Intensity"
                    title_calib = f"Calibration Curve at λ = {wl_common[peak_idx_global]:.1f} nm"
                slope_c, intercept_c, r_val_c, _, _ = linregress(concs_arr_plot, responses)
                x_fit = np.linspace(concs_arr_plot.min(), concs_arr_plot.max(), 80)
                y_fit = slope_c * x_fit + intercept_c

                fig_calib = go.Figure()
                fig_calib.add_trace(
                    go.Scatter(
                        x=concs_arr_plot,
                        y=responses,
                        mode="markers",
                        name="Measured",
                        marker=dict(size=12, color="crimson", symbol="circle"),
                    )
                )
                fig_calib.add_trace(
                    go.Scatter(
                        x=x_fit,
                        y=y_fit,
                        mode="lines",
                        name=f"Linear Fit (R²={r_val_c**2:.4f})",
                        line=dict(color="royalblue", dash="dash"),
                    )
                )
                fig_calib.update_layout(
                    title=title_calib,
                    xaxis_title="Concentration (ppm)",
                    yaxis_title=y_axis_lbl,
                    height=380,
                    margin=dict(t=40, b=30, l=40, r=20),
                )
                st.plotly_chart(fig_calib, use_container_width=True)
                cc1, cc2 = st.columns(2)
                cc1.metric("Sensitivity (slope)", f"{slope_c:.4f} {y_axis_lbl}/ppm")
                cc2.metric("Linearity R²", f"{r_val_c**2:.4f}")
            except Exception as _e:
                st.info(f"Calibration curve requires scipy. ({_e})")

        # ── Sensitivity Heatmap ───────────────────────────────────────────
        if len(pp) >= 2:
            st.markdown("### Sensitivity Heatmap")
            z_axis_lbl = "ΔI" if has_diff else "Intensity"
            fig_hm = go.Figure(
                data=go.Heatmap(
                    z=Z,
                    x=wl_common,
                    y=concs_plot,
                    colorscale="Viridis",
                    colorbar=dict(title=z_axis_lbl),
                )
            )
            fig_hm.update_layout(
                title=f"Sensitivity Heatmap — {z_axis_lbl} (Concentration × Wavelength)",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Concentration (ppm)",
                height=380,
                margin=dict(t=40, b=30, l=40, r=20),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

        # ── 3D Response Surface ───────────────────────────────────────────
        if len(pp) >= 2:
            st.markdown("### 3D Response Surface")
            z_axis_lbl = "ΔI" if has_diff else "Intensity"
            fig3d = go.Figure(
                data=[go.Surface(z=Z, x=wl_common, y=concs_plot, colorscale="Viridis")]
            )
            fig3d.update_layout(
                scene=dict(
                    xaxis_title="Wavelength (nm)",
                    yaxis_title="Concentration (ppm)",
                    zaxis_title=z_axis_lbl,
                ),
                height=500,
                margin=dict(l=0, r=0, b=0, t=10),
            )
            st.plotly_chart(fig3d, use_container_width=True)

        # ── Confusion Matrix (multi-class) ────────────────────────────────
        if len(class_names) >= 2:
            st.markdown("### Confusion Matrix (Classification Preview)")
            st.caption(
                "Placeholder confusion matrix from feature-based nearest-centroid classification."
            )
            try:
                from sklearn.metrics import confusion_matrix
                from sklearn.model_selection import cross_val_predict
                from sklearn.neighbors import NearestCentroid

                nc = NearestCentroid()
                if len(X_list) >= 4 and len(set(y_labels)) >= 2:
                    y_pred_cv = cross_val_predict(nc, X, np.array(y_labels), cv=min(3, len(X_list)))
                    cm = confusion_matrix(np.array(y_labels), y_pred_cv)
                    import plotly.figure_factory as ff

                    fig_cm = ff.create_annotated_heatmap(
                        cm.tolist(),
                        x=class_names,
                        y=class_names,
                        colorscale="Blues",
                        showscale=True,
                    )
                    fig_cm.update_layout(
                        title="Cross-validated Confusion Matrix",
                        xaxis_title="Predicted",
                        yaxis_title="Actual",
                        height=350,
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                    acc = float(np.diag(cm).sum()) / float(cm.sum())
                    st.metric("Cross-val Accuracy", f"{acc:.1%}")
                else:
                    st.info("Need ≥4 samples and ≥2 gas classes for confusion matrix.")
            except Exception as _e:
                st.info(f"Confusion matrix unavailable: {_e}")

        # ── Model training ────────────────────────────────────────────────
        st.markdown("### Model Training")
        model_type = st.selectbox(
            "Select Model",
            ["Gaussian Process Regression (GPR — Concentration)", "1D CNN Classifier (Gas Type)"],
        )

        train_btn = st.button("🚀 Train Model", type="primary", disabled=len(X_list) < 2)

        if train_btn:
            model_dir = _REPO / "models"
            model_dir.mkdir(exist_ok=True)
            trained_ok = False

            with st.spinner("Training…"):
                if "CNN" in model_type and _ML_AVAILABLE:
                    try:
                        wl_t = np.linspace(200, 1000, 1000)
                        X_raw = np.array([np.interp(wl_t, it["wl"], it["processed"]) for it in pp])
                        clf = CNNGasClassifier(input_length=1000, num_classes=len(class_names))
                        history = clf.fit(
                            X_raw,
                            np.array(y_labels),
                            np.array(y_concs),
                            class_names=class_names,
                            epochs=20,
                        )
                        clf.save(str(model_dir / "gas_cnn.pt"))
                        st.success("CNN trained and saved to `models/gas_cnn.pt`.")
                        if history and "loss" in history:
                            st.line_chart(pd.DataFrame({"Training Loss": history["loss"]}))
                        trained_ok = True
                    except Exception as ex:
                        st.warning(f"Custom CNN skipped ({ex}), falling back to sklearn GPR.")

                elif "GPR" in model_type and _ML_AVAILABLE:
                    try:
                        gpr = GPRCalibration()
                        gpr.fit(X, np.array(y_concs))
                        gpr.save(str(model_dir / "gpr_calibration.pkl"))
                        st.success("Custom GPR trained and saved.")
                        trained_ok = True
                    except Exception as ex:
                        st.warning(f"Custom GPR skipped ({ex}), falling back to sklearn GPR.")

                # Always-available sklearn GPR fallback
                if not trained_ok:
                    import pickle

                    from sklearn.gaussian_process import GaussianProcessRegressor
                    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

                    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
                    gpr_sk = GaussianProcessRegressor(
                        kernel=kernel, n_restarts_optimizer=2, normalize_y=True
                    )
                    gpr_sk.fit(X, np.array(y_concs, dtype=float))
                    with open(model_dir / "gpr_sklearn.pkl", "wb") as f:
                        pickle.dump(gpr_sk, f)
                    st.success("✅ sklearn GPR trained and saved to `models/gpr_sklearn.pkl`.")
                    ss["ap_gpr_sklearn"] = gpr_sk

            ss["ap_model_trained"] = True
            ss["ap_X_train"] = X
            ss["ap_y_concs"] = np.array(y_concs, dtype=float)
            ss["ap_class_names"] = class_names
            ss["ap_pp_items"] = pp
            ss["ap_has_diff"] = has_diff  # pin feature-extraction mode for inference

        col_nav = st.columns(2)
        if col_nav[0].button("⬅️ Back to Step 2"):
            ss["ap_step"] = 2
            st.rerun()
        if col_nav[1].button("➡️ Deploy & Test", disabled=not ss["ap_model_trained"]):
            ss["ap_step"] = 4
            st.rerun()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4 — Deployment & Real-Time Inference  (Agent 05)
    # ═══════════════════════════════════════════════════════════════════════
    elif step == 4:
        st.subheader("Step 4 — Deployment & Real-Time Inference")

        # ── Load model from disk if not already in session ────────────────
        if not ss["ap_model_trained"]:
            st.markdown("#### Load Existing Model from Disk")
            st.caption(
                "No model trained in this session. You can load a previously saved model from `models/`."
            )
            model_dir_load = _REPO / "models"
            pkl_files = sorted(model_dir_load.glob("*.pkl")) if model_dir_load.exists() else []
            if pkl_files:
                chosen_pkl = st.selectbox(
                    "Select model file", pkl_files, format_func=lambda p: p.name
                )
                if st.button("📂 Load Model from Disk"):
                    import joblib

                    loaded_m = joblib.load(chosen_pkl)
                    ss["ap_gpr_sklearn"] = loaded_m
                    ss["ap_model_trained"] = True
                    ss["ap_y_concs"] = np.array([100.0])
                    st.success(f"Loaded **{chosen_pkl.name}** from disk.")
                    st.rerun()
            else:
                st.warning("No .pkl models found in `models/`. Go back to Step 3 to train one.")
            if st.button("⬅️ Back to Step 3"):
                ss["ap_step"] = 3
                st.rerun()
            return

        meta = ss.get("ap_meta", {})
        if not meta:
            st.warning("No acquisition metadata found — concentration estimates may be missing.")

        # ── Model status ──────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Model Status", "🟢 Loaded & Ready")
        c2.metric(
            "LOD",
            f"{ss.get('ap_lod', 'N/A')} ppm" if isinstance(ss.get("ap_lod"), float) else "N/A",
        )
        c3.metric(
            "Training R²",
            f"{ss.get('ap_r2', 0):.4f}" if isinstance(ss.get("ap_r2"), float) else "N/A",
        )

        st.markdown("---")
        st.markdown("### Live Inference Loop")
        st.caption("Each press acquires one spectrum, preprocesses it, and predicts concentration.")

        # Single-shot inference (Streamlit-safe — no blocking while loop)
        if st.button("🔮 Run Single Prediction", type="primary"):
            chart_ph2 = st.empty()
            wl_live, frame = _acquire_frames(
                meta.get("integration_ms", 30),
                meta.get("n_frames", 10),
                meta.get("concentration_ppm", 100.0),
                chart_ph2,
                sim_peak_nm=ss.get("ap_sim_peak_nm", 532.0),
            )
            proc = _preprocess(wl_live, frame, "Savitzky-Golay", "ALS", "Min-Max [0,1]")

            # Build feature vector — must match the branch used during training
            has_diff_inf = ss.get("ap_has_diff", False)
            ref_peak_wl_inf = ss.get("ap_ref_peak_wl")
            if has_diff_inf and ss.get("ap_ref_spectrum") is not None:
                ref_arr_inf = ss["ap_ref_spectrum"]
                ref_wl_inf = ss.get("ap_ref_wl", wl_live)
                ref_interp_inf = np.interp(wl_live, ref_wl_inf, ref_arr_inf)
                diff_inf = frame - ref_interp_inf  # ΔI = raw − ref (matches Step 2 convention)
                peak_idx_d = int(np.argmax(np.abs(diff_inf)))
                peak_wl_d = float(wl_live[peak_idx_d])
                delta_lam = peak_wl_d - ref_peak_wl_inf if ref_peak_wl_inf else 0.0
                feat = np.array(
                    [
                        [
                            delta_lam,
                            float(diff_inf[peak_idx_d]),
                            float(np.trapezoid(diff_inf, wl_live)),
                            float(np.std(diff_inf)),
                        ]
                    ]
                )
            else:
                peak_idx = int(np.argmax(proc))
                feat = np.array(
                    [
                        [
                            float(proc[peak_idx]),
                            float(wl_live[peak_idx]),
                            float(np.trapezoid(proc, wl_live)),
                            float(np.std(proc)),
                        ]
                    ]
                )

            # Predict
            gpr_model = ss.get("ap_gpr_sklearn")
            if gpr_model:
                pred_conc, pred_std = gpr_model.predict(feat, return_std=True)
                conc_val = float(pred_conc[0])
                conc_std = float(pred_std[0])
            else:
                # fallback: linear interpolation from training data
                conc_val = float(np.mean(ss.get("ap_y_concs", [100.0])))
                conc_std = float(np.std(ss.get("ap_y_concs", [0.0])))

            confidence = max(0.0, 1.0 - min(1.0, conc_std / (abs(conc_val) + 1e-6)))
            ss["ap_pred_history"].append((conc_val, conc_std))

            # Gauge
            fig_g = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=conc_val,
                    title={"text": "Predicted Concentration (ppm)", "font": {"size": 16}},
                    delta={"reference": meta.get("concentration_ppm", conc_val)},
                    gauge={
                        "axis": {"range": [0, max(600, conc_val * 1.5)]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 50], "color": "lightcyan"},
                            {"range": [50, 300], "color": "cyan"},
                            {"range": [300, 600], "color": "royalblue"},
                        ],
                    },
                )
            )
            fig_g.update_layout(height=280, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_g, use_container_width=True)

            ca, cb, cc = st.columns(3)
            ca.metric("Predicted (ppm)", f"{conc_val:.2f}")
            cb.metric("Uncertainty (±1σ)", f"{conc_std:.2f} ppm")
            cc.metric("Confidence", f"{confidence:.1%}")

        # ── Timeseries of predictions ─────────────────────────────────────
        hist = ss["ap_pred_history"]
        if hist:
            st.markdown("### Prediction History")
            means = [h[0] for h in hist]
            stds = [h[1] for h in hist]
            true_conc = meta.get("concentration_ppm", 0)
            fig_ts = go.Figure()
            fig_ts.add_hline(
                y=true_conc,
                line_dash="dash",
                line_color="green",
                annotation_text=f"True: {true_conc} ppm",
            )
            fig_ts.add_trace(
                go.Scatter(
                    y=means, mode="lines+markers", name="Predicted", line=dict(color="orange")
                )
            )
            fig_ts.add_trace(
                go.Scatter(
                    y=[m + s for m, s in zip(means, stds)],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig_ts.add_trace(
                go.Scatter(
                    y=[m - s for m, s in zip(means, stds)],
                    fill="tonexty",
                    mode="lines",
                    fillcolor="rgba(255,165,0,0.2)",
                    line=dict(width=0),
                    name="±1σ",
                )
            )
            fig_ts.update_layout(
                xaxis_title="Prediction #",
                yaxis_title="Concentration (ppm)",
                height=320,
                margin=dict(t=20, b=30, l=40, r=20),
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            # Accuracy vs true
            if len(means) >= 3:
                true_arr = np.full(len(means), true_conc)
                mae = float(np.mean(np.abs(np.array(means) - true_arr)))
                st.metric("MAE vs true concentration", f"{mae:.2f} ppm")

        # ── Test CSV Upload for Accuracy Evaluation ───────────────────────
        st.markdown("---")
        st.markdown("### Accuracy Testing against Real Test Data")
        st.caption(
            "Upload labeled test CSVs (with known concentrations) to evaluate model accuracy on real data."
        )
        test_files = st.file_uploader(
            "Upload test CSVs (wavelength, intensity columns; filename must contain concentration as e.g. `100ppm`)",
            type=["csv"],
            accept_multiple_files=True,
            key="ap_test_upload",
        )
        if test_files and st.button("📊 Evaluate on Test Set"):
            gpr_model_test = ss.get("ap_gpr_sklearn")
            if gpr_model_test is None:
                st.warning("No GPR model in session.")
            else:
                test_rows = []
                for tf in test_files:
                    try:
                        df_t = pd.read_csv(tf)
                        wl_t = df_t.iloc[:, 0].values
                        it_t = df_t.iloc[:, 1].values
                        proc_t = _preprocess(wl_t, it_t, "Savitzky-Golay", "ALS", "Min-Max [0,1]")
                        pi = int(np.argmax(proc_t))
                        feat_t = np.array(
                            [
                                [
                                    float(proc_t[pi]),
                                    float(wl_t[pi]),
                                    float(np.trapezoid(proc_t, wl_t)),
                                    float(np.std(proc_t)),
                                ]
                            ]
                        )
                        pred_c, pred_s = gpr_model_test.predict(feat_t, return_std=True)
                        m_true = re.search(r"([\d.]+)ppm", tf.name)
                        true_c = float(m_true.group(1)) if m_true else float("nan")
                        err = (
                            abs(float(pred_c[0]) - true_c) if not np.isnan(true_c) else float("nan")
                        )
                        test_rows.append(
                            {
                                "File": tf.name,
                                "True (ppm)": true_c,
                                "Predicted (ppm)": f"{pred_c[0]:.2f}",
                                "±1σ": f"{pred_s[0]:.2f}",
                                "Abs Error": f"{err:.2f}" if not np.isnan(err) else "N/A",
                            }
                        )
                    except Exception as _te:
                        test_rows.append({"File": tf.name, "Error": str(_te)})
                st.dataframe(pd.DataFrame(test_rows), use_container_width=True)
                valid = [r for r in test_rows if "Abs Error" in r and r["Abs Error"] != "N/A"]
                if valid:
                    mean_mae = np.mean([float(r["Abs Error"]) for r in valid])
                    st.metric("Mean Absolute Error (Test Set)", f"{mean_mae:.2f} ppm")

        # ── Session report ────────────────────────────────────────────────
        st.markdown("---")
        if st.button("📄 Generate Session Report"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            report = "# Gas Sensing Session Report\n\n"
            report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report += f"**Gas:** {meta.get('gas', 'N/A')}  \n"
            report += f"**Concentration:** {meta.get('concentration_ppm', 'N/A')} ppm  \n"
            report += f"**LOD:** {ss.get('ap_lod', 'N/A')} ppm  \n"
            report += f"**R²:** {ss.get('ap_r2', 'N/A')}  \n"
            report += f"**Total Predictions:** {len(hist)}  \n"
            if hist:
                report += f"**Mean Predicted Conc.:** {np.mean([h[0] for h in hist]):.2f} ppm\n"

            rep_dir = _REPO / "reports"
            rep_dir.mkdir(exist_ok=True)
            rep_path = rep_dir / f"session_{ts}.md"
            rep_path.write_text(report, encoding="utf-8")
            st.download_button(
                "⬇️ Download Report", report, file_name=rep_path.name, mime="text/markdown"
            )
            st.success(f"Report saved to `{rep_path}`")

        if st.button("⬅️ Back to Step 3"):
            ss["ap_step"] = 3
            st.rerun()
