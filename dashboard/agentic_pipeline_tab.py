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
from typing import Any, cast

# ── project root ──────────────────────────────────────────────────────────────
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.backups import BackupManager
from dashboard.reproducibility import create_session_manifest

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
        from gas_analysis.core.preprocessing import (  # type: ignore[assignment]
            baseline_correction,
            normalize_spectrum,
        )
        from gas_analysis.core.signal_proc import als_baseline, smooth_spectrum  # type: ignore[assignment]
        from src.preprocessing.quality import compute_snr, estimate_noise_metrics

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

# ── Conformal prediction ──────────────────────────────────────────────────────
try:
    from src.calibration.conformal import ConformalCalibrator

    _CONFORMAL_AVAILABLE = True
except Exception:
    _CONFORMAL_AVAILABLE = False

# ── LOD / kinetics ────────────────────────────────────────────────────────────
try:
    from src.reporting.metrics import compute_comprehensive_sensor_characterization
    from src.scientific.lod import calculate_lod_3sigma, calculate_loq_10sigma, calculate_sensitivity

    _LOD_AVAILABLE = True
except Exception:
    try:
        from gas_analysis.core.scientific.lod import (  # type: ignore[assignment]
            calculate_lod_3sigma,
            calculate_loq_10sigma,
            calculate_sensitivity,
        )
        compute_comprehensive_sensor_characterization = None  # type: ignore[assignment]
        _LOD_AVAILABLE = True
    except Exception:
        compute_comprehensive_sensor_characterization = None  # type: ignore[assignment]
        _LOD_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Quality gate
# ─────────────────────────────────────────────────────────────────────────────


def _check_calibration_quality(
    sensor_metrics: dict,
    r2: float | None,
    min_r2: float = 0.95,
    max_lod_ppm: float | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Check calibration quality before allowing model save.

    Parameters
    ----------
    sensor_metrics : dict
        Output of ``compute_comprehensive_sensor_characterization`` (may be empty).
    r2 : float | None
        Calibration R² from training; None if not yet computed.
    min_r2 : float
        Minimum acceptable R².  Default 0.95 (ACS Sensors standard).
    max_lod_ppm : float | None
        Maximum acceptable LOD in ppm.  None = not checked.

    Returns
    -------
    tuple (ok: bool, errors: list[str], warnings: list[str])
        ``ok`` is False when any hard error fires (should block save).
        ``warnings`` are advisory — save is still allowed.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # R² check (hard gate)
    if r2 is not None:
        if r2 < min_r2:
            errors.append(
                f"R²={r2:.4f} is below the minimum threshold of {min_r2:.2f}. "
                "Retrain with more or better-quality calibration data."
            )
    else:
        warnings.append("R² not yet computed — train the model first.")

    # LOD check
    lod = sensor_metrics.get("lod_ppm")
    if lod is not None:
        if max_lod_ppm is not None and lod > max_lod_ppm:
            errors.append(
                f"LOD={lod:.4f} ppm exceeds the allowed maximum of {max_lod_ppm:.4f} ppm."
            )
        # Advisory: LOD > LOQ would indicate a logic error
        loq = sensor_metrics.get("loq_ppm")
        if loq is not None and lod >= loq:
            warnings.append(
                f"LOD ({lod:.4f}) ≥ LOQ ({loq:.4f}) — check noise estimate."
            )

    # Mandel linearity check (advisory)
    lin = sensor_metrics.get("mandel_linearity")
    if lin is not None and not lin.get("is_linear", True):
        warnings.append(
            f"Mandel's test rejects linearity (p={lin.get('p_value', 'N/A'):.4f} < 0.05). "
            "The calibration range may exceed the linear dynamic range. "
            "Consider restricting to LOL or using a Langmuir model."
        )

    ok = len(errors) == 0
    return ok, errors, warnings


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

    return cast(np.ndarray, sig)


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
    ss.setdefault("ap_buffer", [])
    ss.setdefault("ap_wl", None)
    ss.setdefault("ap_recording", False)
    ss.setdefault("ap_preprocessed", [])
    ss.setdefault("ap_model_trained", False)
    ss.setdefault("ap_pred_history", [])
    ss.setdefault("ap_inference_active", False)

    # ── Persistent session: save / resume ────────────────────────────────
    _SESSION_CACHE = _REPO / "output" / ".session_cache.json"

    def _save_session_to_disk() -> None:
        """Serialize recoverable session data to disk."""
        import json as _json
        _serialisable_keys = [
            "ap_step", "ap_meta", "ap_gas", "ap_model_trained",
            "ap_sensor_metrics", "ap_r2", "ap_y_concs",
            "ap_denoise", "ap_baseline_m", "ap_norm_m",
            "ap_ref_peak_wl", "ap_quick_start_chosen",
        ]
        payload: dict = {"_saved_at": datetime.now().isoformat()}
        for k in _serialisable_keys:
            v = ss.get(k)
            if v is not None:
                try:
                    _json.dumps(v)   # only keep JSON-serialisable values
                    payload[k] = v
                except (TypeError, ValueError):
                    pass
        # Persist preprocessed labels (not arrays — too large) so scientist sees what was loaded
        payload["_preprocessed_labels"] = [
            {"label": it.get("label", ""), "path": it.get("path", "")}
            for it in ss.get("ap_preprocessed", [])
        ]
        _SESSION_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _SESSION_CACHE.write_text(_json.dumps(payload, indent=2), encoding="utf-8")

    def _load_session_from_disk() -> dict:
        import json as _json
        if not _SESSION_CACHE.exists():
            return {}
        try:
            return _json.loads(_SESSION_CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}

    # ── Resume banner (shown once, top of tab, before step content) ──────
    if not ss.get("_resume_checked"):
        ss["_resume_checked"] = True
        _saved = _load_session_from_disk()
        if _saved and _saved.get("ap_step", 1) > 1:
            _saved_at = _saved.get("_saved_at", "")[:16].replace("T", " ")
            _saved_gas = _saved.get("ap_gas", _saved.get("ap_meta", {}).get("gas", "?"))
            _resume_col1, _resume_col2 = st.columns([4, 1])
            _resume_col1.info(
                f"💾 **Saved session found** — {_saved_gas}, "
                f"last at Step {_saved.get('ap_step', '?')} ({_saved_at}). "
                "Resume where you left off?"
            )
            if _resume_col2.button("▶ Resume", key="ap_resume_btn"):
                for k, v in _saved.items():
                    if not k.startswith("_") and k != "_preprocessed_labels":
                        ss[k] = v
                st.rerun()

    step = ss["ap_step"]

    # Auto-save session state on every step transition (done at end of each step block)
    # Explicit save button in sidebar
    with st.sidebar:
        if st.button("💾 Save session", key="ap_save_sidebar", help="Save current progress to disk"):
            _save_session_to_disk()
            st.success("Session saved.")

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

        # ── Quick Start routing ───────────────────────────────────────────
        if not ss.get("ap_quick_start_chosen"):
            st.markdown("#### How would you like to start?")
            _qs1, _qs2, _qs3 = st.columns(3)
            with _qs1:
                st.markdown(
                    """
                    <div style='border:1.5px solid #4CAF50;border-radius:10px;padding:16px;text-align:center'>
                    <h3>🔬 Live Measurement</h3>
                    <p style='font-size:0.9em;color:#888'>CCS200 connected.<br>Record new spectra in real time.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Start Live Session", key="qs_live", use_container_width=True):
                    ss["ap_quick_start_chosen"] = "live"
                    st.rerun()
            with _qs2:
                st.markdown(
                    """
                    <div style='border:1.5px solid #2196F3;border-radius:10px;padding:16px;text-align:center'>
                    <h3>📂 Load Existing Data</h3>
                    <p style='font-size:0.9em;color:#888'>Have Joy_Data or CSV files?<br>Skip to analysis instantly.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Load My Data", key="qs_load", use_container_width=True):
                    ss["ap_quick_start_chosen"] = "load"
                    ss["ap_step"] = 2
                    st.rerun()
            with _qs3:
                st.markdown(
                    """
                    <div style='border:1.5px solid #FF9800;border-radius:10px;padding:16px;text-align:center'>
                    <h3>🎲 Simulate & Explore</h3>
                    <p style='font-size:0.9em;color:#888'>No hardware needed.<br>Explore the full pipeline with synthetic data.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Use Simulation", key="qs_sim", use_container_width=True):
                    ss["ap_quick_start_chosen"] = "sim"
                    st.rerun()
            st.stop()

        # ── Show chosen mode badge ────────────────────────────────────────
        _mode_map = {
            "live": ("🔬 Live Measurement Mode", "green"),
            "load": ("📂 Historical Data Mode", "blue"),
            "sim":  ("🎲 Simulation Mode", "orange"),
        }
        _mode_label, _mode_color = _mode_map.get(
            ss.get("ap_quick_start_chosen", "sim"), ("🎲 Simulation Mode", "orange")
        )
        st.markdown(
            f"<span style='background:{_mode_color};color:white;padding:3px 10px;"
            f"border-radius:12px;font-size:0.85em'>{_mode_label}</span>"
            f"&nbsp;&nbsp;<a href='#' style='font-size:0.8em;color:#888' "
            f"onclick='void(0)'>change</a>",
            unsafe_allow_html=True,
        )
        if st.button("↩ Change start mode", key="qs_reset", help="Go back to Quick Start"):
            ss.pop("ap_quick_start_chosen", None)
            st.rerun()
        st.markdown("")

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

                # Save reproducibility manifest next to the snapshot artifacts.
                try:
                    manifest_id = (
                        f"{meta['gas']}_{meta['concentration_ppm']}ppm_"
                        f"T{meta['trial']}_{ts}"
                    )
                    create_session_manifest(session_id=manifest_id, output_dir=out_dir)
                except Exception as manifest_exc:
                    st.warning(f"Manifest creation skipped: {manifest_exc}")

                # Create an integrity-verified compressed backup for this trial folder.
                try:
                    backup_meta = BackupManager(app_root=_REPO).backup_session(out_dir)
                    ss["ap_last_backup"] = backup_meta["backup_name"]
                except Exception as backup_exc:
                    st.warning(f"Backup creation skipped: {backup_exc}")

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
                if ss.get("ap_last_backup"):
                    st.caption(f"Backup: {ss['ap_last_backup']}")
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
            _save_session_to_disk()
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
        joy_root = _REPO / "data" / "JOY_Data"
        if not joy_root.exists():
            joy_root = _REPO / "Joy_Data"  # legacy fallback

        source_options = ["Session recordings (in memory)", "Load from automation_dataset/"]
        if joy_root.exists():
            source_options.insert(1, "📂 Load from Joy_Data (existing research data)")

        # Quick-start "Load mode" pre-selects Joy_Data automatically
        _default_src_idx = 0
        if ss.get("ap_quick_start_chosen") == "load" and joy_root.exists():
            _default_src_idx = source_options.index("📂 Load from Joy_Data (existing research data)")

        source = st.radio(
            "Data source",
            source_options,
            index=_default_src_idx,
            horizontal=True,
            help="Joy_Data: your existing lab measurements organised as Gas/Concentration folders",
        )

        # Summary of what's available in Joy_Data
        if joy_root.exists():
            _joy_gases = [d.name for d in sorted(joy_root.iterdir()) if d.is_dir()]
            _joy_counts = {
                d.name: sum(1 for _ in d.rglob("*.csv"))
                for d in joy_root.iterdir() if d.is_dir()
            }
            st.caption(
                f"**Joy_Data** contains: "
                + " | ".join(f"{g} ({_joy_counts.get(g,0)} CSVs)" for g in _joy_gases[:6])
                + (" | ..." if len(_joy_gases) > 6 else "")
            )

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

        elif "Joy_Data" in source:  # Joy_Data
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
            joy_items_available: dict[str, list[Path]] = {}
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
            # Store preprocessing choices so Step 4 can replay the same pipeline
            ss["ap_denoise"] = denoise
            ss["ap_baseline_m"] = baseline_m
            ss["ap_norm_m"] = norm_m
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
            _save_session_to_disk()
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

        # ── Data sufficiency check ────────────────────────────────────────
        _n_samples = len(pp)
        _n_concs = len(set(
            re.search(r"([\d.]+)ppm", it.get("label", "")).group(1)
            for it in pp if re.search(r"([\d.]+)ppm", it.get("label", ""))
        ))
        _suff_col1, _suff_col2, _suff_col3 = st.columns(3)
        _suff_col1.metric("Total spectra", _n_samples,
                          "✅ Good" if _n_samples >= 20 else ("⚠️ Low" if _n_samples >= 8 else "🚫 Too few"))
        _suff_col2.metric("Concentration levels", _n_concs,
                          "✅ Good" if _n_concs >= 5 else ("⚠️ Minimum" if _n_concs >= 3 else "🚫 Need more"))
        _suff_col3.metric("Avg replicates/level",
                          f"{_n_samples / max(_n_concs, 1):.1f}",
                          "✅ Good" if _n_samples / max(_n_concs, 1) >= 3 else "⚠️ Add replicates")

        if _n_samples < 8:
            st.error(
                "🚫 **Too few spectra for reliable calibration.** "
                f"You have {_n_samples} spectra across {_n_concs} concentration level(s). "
                "Record at least 8 spectra (ideally 5+ concentration levels × 3+ replicates) "
                "before training. Go back to Step 1 or Step 2 to load more data."
            )
        elif _n_samples < 20:
            st.warning(
                f"⚠️ **{_n_samples} spectra loaded** across {_n_concs} concentration level(s). "
                "Model will train but LOD/LOQ confidence intervals may be wide. "
                "For publication-grade results, aim for ≥20 spectra across ≥5 concentration levels."
            )
        else:
            st.success(
                f"✅ **{_n_samples} spectra** across {_n_concs} concentration level(s) — "
                "sufficient for reliable calibration and IUPAC LOD/LOQ estimation."
            )

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
                    # Derive peak from the blank/air reference spectrum, not the analyte
                    _ref_arr_peak = ss.get("ap_ref_spectrum")
                    _ref_wl_peak = ss.get("ap_ref_wl", wl)
                    if _ref_arr_peak is not None:
                        _ref_interp_peak = np.interp(wl, _ref_wl_peak, _ref_arr_peak)
                        ref_peak_wl = float(wl[int(np.argmax(_ref_interp_peak))])
                    else:
                        # No reference loaded — fall back to first sample peak
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

        # ── Feature explanation panel ─────────────────────────────────────
        if has_diff:
            st.success("🔬 **LSPR differential features active**")
            with st.expander("What do these features mean?", expanded=False):
                st.markdown(
                    """
| Feature | Symbol | Physical meaning | Why it matters |
|---|---|---|---|
| **Peak wavelength shift** | Δλ (nm) | Change in LSPR resonance position vs blank | Primary sensing signal — shifts negative as analyte binds to Au-MIP surface |
| **Differential peak intensity** | ΔI peak | Intensity change at the shifted peak | Reflects binding-induced plasmon damping |
| **Differential spectral area** | ΔI area | Integrated area under difference spectrum | Captures broad spectral redistribution, not just peak tip |
| **Differential std** | ΔI std | Spread of difference signal across wavelengths | Sensitive to lineshape changes — increases with stronger binding |

**Why Δλ is the most important:** The LSPR peak wavelength shifts because analyte molecules in the MIP cavities change the local refractive index around the Au nanoparticles. This shift is directly proportional to surface coverage (and therefore concentration) in the linear range.

**Tip:** If Δλ values are near zero across all concentrations, check that your reference spectrum was recorded in clean air/blank — not in the presence of any analyte.
                    """
                )
        else:
            st.info("ℹ️ **Standard spectral features** — load a reference spectrum in Step 2 to unlock LSPR Δλ mode.")
            with st.expander("How to enable LSPR mode", expanded=False):
                st.markdown(
                    """
**LSPR Δλ mode requires a reference (blank) spectrum.**

1. Go back to **Step 2**
2. In the *Reference Spectrum* section, load a CSV recorded with **clean air** (no analyte present)
3. Return to Step 3 — features will automatically switch to `[Δλ, ΔI_peak, ΔI_area, ΔI_std]`

**Why this matters:** Without a reference, the model trains on absolute intensities which vary with lamp power, temperature, and fibre coupling — none of which are related to analyte concentration. The differential signal removes all these common-mode effects.
                    """
                )

        # ── Scientific metrics (full IUPAC / ICH Q2 triad) ──────────────
        if len(y_concs) >= 3 and _LOD_AVAILABLE:
            st.markdown("### Scientific Metrics (IUPAC / ICH Q2 Characterisation)")
            try:
                concs_arr = np.array(y_concs, dtype=float)
                peak_arr = X[:, 0]

                # Use the comprehensive function when available (preferred):
                # gives LOB, LOD + bootstrap CI, LOQ, LOL, Mandel linearity
                if compute_comprehensive_sensor_characterization is not None:
                    _sm = compute_comprehensive_sensor_characterization(
                        concs_arr, peak_arr, gas_name=ss.get("ap_gas", "analyte")
                    )
                else:
                    # Fallback for environments where src.reporting is unavailable
                    slope, intercept, r2, slope_se = calculate_sensitivity(concs_arr, peak_arr)
                    noise_floor = float(np.std(peak_arr[: max(1, len(peak_arr) // 5)]))
                    _sm = {
                        "sensitivity": slope,
                        "sensitivity_se": slope_se,
                        "r_squared": r2,
                        "lob_ppm": None,
                        "lod_ppm": calculate_lod_3sigma(noise_floor, slope),
                        "lod_ppm_ci_lower": None,
                        "lod_ppm_ci_upper": None,
                        "loq_ppm": calculate_loq_10sigma(noise_floor, slope),
                        "lol_ppm": None,
                        "mandel_linearity": None,
                    }

                # ── Row 1: Detection limits ───────────────────────────────
                st.markdown("**Detection limits (IUPAC 2012 / ICH Q2(R1))**")
                r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                lob_val = _sm.get("lob_ppm")
                lod_val = _sm.get("lod_ppm")
                loq_val = _sm.get("loq_ppm")
                lol_val = _sm.get("lol_ppm")
                lod_lo  = _sm.get("lod_ppm_ci_lower")
                lod_hi  = _sm.get("lod_ppm_ci_upper")

                r1c1.metric(
                    "LOB (IUPAC)",
                    f"{lob_val:.4f} ppm" if lob_val is not None else "N/A",
                    help="Limit of Blank: μ_blank + 1.645·σ_blank / |S|",
                )
                _lod_label = (
                    f"{lod_val:.4f} ppm" if lod_val is not None else "N/A"
                )
                _lod_delta = (
                    f"95% CI [{lod_lo:.4f} – {lod_hi:.4f}]"
                    if lod_lo is not None and lod_hi is not None
                    else None
                )
                r1c2.metric(
                    "LOD (3σ/S)",
                    _lod_label,
                    delta=_lod_delta,
                    delta_color="off",
                    help="Limit of Detection — 95% bootstrap CI shown as delta",
                )
                r1c3.metric(
                    "LOQ (10σ/S)",
                    f"{loq_val:.4f} ppm" if loq_val is not None else "N/A",
                    help="Limit of Quantification: smallest reliably quantified concentration",
                )
                r1c4.metric(
                    "LOL (Mandel)",
                    f"{lol_val:.4f} ppm" if lol_val is not None else "N/A",
                    help="Limit of Linearity: highest concentration with linear response (ICH Q2 §4.2)",
                )

                # ── Row 2: Calibration quality ────────────────────────────
                st.markdown("**Calibration quality**")
                r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                sens = _sm.get("sensitivity", float("nan"))
                se   = _sm.get("sensitivity_se", float("nan"))
                r2   = _sm.get("r_squared", float("nan"))
                rmse = _sm.get("rmse")

                r2c1.metric(
                    "Sensitivity (S)",
                    f"{sens:.4f} a.u./ppm",
                    delta=f"±{se:.4f} SE" if se is not None else None,
                    delta_color="off",
                )
                r2c2.metric("R² (Linearity)", f"{r2:.4f}")
                r2c3.metric(
                    "RMSE",
                    f"{rmse:.4f} a.u." if rmse is not None else "N/A",
                    help="Root-mean-square error of the calibration fit",
                )
                _lin = _sm.get("mandel_linearity")
                r2c4.metric(
                    "Mandel p-value",
                    f"{_lin['p_value']:.4f}" if _lin and _lin.get("p_value") is not None else "N/A",
                    help="Mandel's F-test p ≥ 0.05 confirms linearity (ICH Q2 §4.2)",
                )

                # ── Linearity details expander ────────────────────────────
                if _lin:
                    with st.expander("🔬 Mandel linearity test details"):
                        is_lin = _lin.get("is_linear", True)
                        st.markdown(
                            f"**Result:** {'✅ Linear' if is_lin else '⚠️ Non-linear'} "
                            f"(F={_lin.get('f_statistic', 'N/A')}, p={_lin.get('p_value', 'N/A')})"
                        )
                        st.markdown(f"R² linear: `{_lin.get('r2_linear', 'N/A')}` | "
                                    f"R² quadratic: `{_lin.get('r2_quadratic', 'N/A')}` | "
                                    f"ΔR²: `{_lin.get('delta_r2', 'N/A')}`")
                        st.caption(_lin.get("recommendation", ""))

                # store in session state for Step 4 status and report generation
                ss["ap_sensor_metrics"] = _sm
                ss["ap_lod"] = lod_val
                ss["ap_r2"] = r2

                # ── Interpretation & Next Steps ───────────────────────────
                st.markdown("---")
                st.markdown("#### 🧭 Interpretation & Recommended Next Steps")
                _interp_col, _next_col = st.columns(2)

                with _interp_col:
                    st.markdown("**What your results mean:**")
                    # LOD context vs published Au-MIP LSPR literature (0.5–5 ppm for VOCs)
                    if lod_val is not None:
                        if lod_val < 0.5:
                            st.success(f"🏆 LOD {lod_val:.2f} ppm — **excellent**, below typical Au-MIP LSPR range (0.5–5 ppm). Publishable.")
                        elif lod_val < 2.0:
                            st.success(f"✅ LOD {lod_val:.2f} ppm — **competitive** with published Au-MIP LSPR sensors (0.5–5 ppm).")
                        elif lod_val < 5.0:
                            st.info(f"📊 LOD {lod_val:.2f} ppm — **within range** for Au-MIP LSPR sensors. Consider more replicates or optimising MIP layer.")
                        else:
                            st.warning(f"⚠️ LOD {lod_val:.2f} ppm — **above typical range**. Check reference spectrum quality and MIP surface condition.")
                    # R² context
                    if not (isinstance(r2, float) and np.isnan(r2)):
                        if r2 >= 0.999:
                            st.success(f"🏆 R² {r2:.4f} — near-perfect linearity. ICH Q2(R1) requires ≥0.999 for pharmaceutical assays.")
                        elif r2 >= 0.995:
                            st.success(f"✅ R² {r2:.4f} — excellent linearity (ICH Q2 threshold: 0.999 for pharma, 0.99 for environmental).")
                        elif r2 >= 0.99:
                            st.info(f"📊 R² {r2:.4f} — good linearity for research applications. Pharmaceutical grade requires ≥0.999.")
                        else:
                            st.warning(f"⚠️ R² {r2:.4f} — linearity below research-grade threshold (0.99). Check for outliers or non-linear response.")
                    # Dynamic range
                    if lod_val is not None and lol_val is not None:
                        _dyn_range = lol_val / lod_val
                        _dr_label = f"{_dyn_range:.0f}×" if _dyn_range < 1000 else f"{_dyn_range/1000:.1f}k×"
                        st.info(f"📏 Dynamic range: **{_dr_label}** ({lod_val:.2f}–{lol_val:.2f} ppm) — "
                                + ("excellent" if _dyn_range > 100 else "moderate" if _dyn_range > 20 else "limited"))

                with _next_col:
                    st.markdown("**Recommended next steps:**")
                    _steps_done, _steps_todo = [], []
                    # Check model trained
                    if ss.get("ap_model_trained"):
                        _steps_done.append("Model trained ✅")
                    else:
                        _steps_todo.append("Train a calibration model (scroll down ↓)")
                    # Check data sufficiency
                    if _n_samples >= 20 and _n_concs >= 5:
                        _steps_done.append(f"Data sufficient ({_n_samples} spectra) ✅")
                    else:
                        _steps_todo.append(f"Add more spectra (currently {_n_samples} — aim for 20+)")
                    # Reference spectrum
                    if has_diff:
                        _steps_done.append("Reference spectrum loaded — LSPR mode ✅")
                    else:
                        _steps_todo.append("Load blank/air reference in Step 2 → enables Δλ LSPR features")
                    # Linearity
                    if _lin and not _lin.get("is_linear", True):
                        _steps_todo.append("Response is non-linear — reduce concentration range or use non-linear model")
                    elif lol_val is not None:
                        _steps_done.append(f"Linearity confirmed up to {lol_val:.1f} ppm ✅")
                    # Cross-sensitivity
                    _steps_todo.append("Test cross-sensitivity: record methanol/IPA spectra and compare LOD shift")
                    # Temperature
                    _steps_todo.append("Record 5 blank spectra at ±2°C to characterise thermal drift (α nm/°C)")

                    if _steps_done:
                        for s in _steps_done:
                            st.markdown(f"- {s}")
                    if _steps_todo:
                        st.markdown("**To do:**")
                        for s in _steps_todo:
                            st.markdown(f"- {s}")

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

        assert wl_common is not None
        assert Z is not None
        assert concs_plot is not None

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

        # ── Nonlinear Isotherm Fitting ────────────────────────────────────
        if len(pp) >= 3:
            with st.expander("🔁 Nonlinear Isotherm Fitting (AIC selection)", expanded=False):
                st.caption(
                    "Fits Langmuir, Freundlich, Hill, and linear models by AICc. "
                    "Preferred over linear regression when the sensor nears binding-site saturation."
                )
                if st.button("Fit Isotherms", key="ap_btn_isotherms"):
                    try:
                        from src.calibration.isotherms import select_isotherm as _ap_sel_iso

                        _ap_concs = np.array(concs_plot, dtype=float)
                        if has_diff:
                            _ap_resp = X[:, 0]  # Δλ
                            _ap_y_lbl = "Δλ (nm)"
                        else:
                            _ap_peak_idx = int(np.argmax(Z.mean(axis=0)))
                            _ap_resp = Z[:, _ap_peak_idx]
                            _ap_y_lbl = "Peak Intensity"

                        _ap_iso_sel = _ap_sel_iso(_ap_concs, _ap_resp)
                        from src.calibration.isotherms import IsothermResult

                        _ap_iso = cast(IsothermResult, _ap_iso_sel["best_result"])
                        _ap_aic_table = cast(
                            list[tuple[str, float, float, float]], _ap_iso_sel["aic_table"]
                        )

                        _ap_c1, _ap_c2 = st.columns([2, 1])
                        with _ap_c1:
                            _fig_ap_iso = go.Figure()
                            _fig_ap_iso.add_trace(
                                go.Scatter(
                                    x=_ap_concs,
                                    y=_ap_resp,
                                    mode="markers",
                                    name="Measured",
                                    marker=dict(size=12, color="crimson"),
                                )
                            )
                            _fig_ap_iso.add_trace(
                                go.Scatter(
                                    x=_ap_iso.concentrations_fit,
                                    y=_ap_iso.responses_fit,
                                    mode="lines",
                                    name=f"{_ap_iso.model.capitalize()} (AIC winner)",
                                    line=dict(color="royalblue", dash="dash", width=2),
                                )
                            )
                            _fig_ap_iso.update_layout(
                                title=(
                                    f"Isotherm — {_ap_iso.model.capitalize()} "
                                    f"(R²={_ap_iso.r_squared:.4f}, AIC={_ap_iso.aic:.2f})"
                                ),
                                xaxis_title="Concentration (ppm)",
                                yaxis_title=_ap_y_lbl,
                                height=340,
                                margin=dict(t=40, b=30),
                            )
                            st.plotly_chart(_fig_ap_iso, use_container_width=True)
                        with _ap_c2:
                            st.markdown(f"**Winner: {_ap_iso.model.capitalize()}**")
                            st.caption(f"AIC = {_ap_iso.aic:.2f}")
                            st.caption(f"R² = {_ap_iso.r_squared:.4f}")
                            st.caption(f"RMSE = {_ap_iso.rmse:.4g}")
                            st.caption(_ap_iso_sel["recommendation"])
                            st.markdown("**Parameters:**")
                            for _pn, _pv in _ap_iso.params.items():
                                if _pn == "sign":
                                    continue
                                _pe = _ap_iso.param_stderrs.get(_pn, float("nan"))
                                st.caption(f"`{_pn}` = {_pv:.4g} ± {_pe:.2g}")
                            st.markdown("**AIC table:**")
                            for _mn, _ma, _mr2, _ in _ap_aic_table:
                                st.caption(f"{_mn}: AIC={_ma:.2f}, R²={_mr2:.4f}")
                    except Exception as _ap_iso_exc:
                        st.error(f"Isotherm fit failed: {_ap_iso_exc}")

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
            [
                "Gaussian Process Regression (GPR — Concentration)",
                "PLS — Partial Least Squares (Multivariate Concentration)",
                "1D CNN Classifier (Gas Type)",
            ],
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

                elif "PLS" in model_type:
                    try:
                        import pickle
                        from src.calibration.pls import PLSCalibration

                        y_arr_pls = np.array(y_concs, dtype=float)
                        # PLS uses the full spectral matrix (Z), not the 4-feature summary
                        X_spec = Z  # (n_samples, 300) — interpolated to common grid
                        if X_spec is None or len(X_spec) < 3:
                            st.error("PLS requires ≥3 spectra and a valid spectral matrix.")
                        else:
                            # Auto-select optimal component count
                            max_comp = min(10, len(X_spec) - 1)
                            pls_probe = PLSCalibration(n_components=1)
                            with st.spinner("Optimising PLS components via cross-validation…"):
                                n_opt, rmsecv_curve = pls_probe.optimize_components(
                                    X_spec, y_arr_pls, max_components=max_comp
                                )

                            pls_model = PLSCalibration(n_components=n_opt)
                            with st.spinner(f"Fitting PLS with {n_opt} components…"):
                                pls_result = pls_model.fit(
                                    X_spec, y_arr_pls, wavelengths=wl_common
                                )

                            # ── PLS metrics display ───────────────────────
                            st.markdown("#### PLS Calibration Diagnostics")
                            pc1, pc2, pc3, pc4 = st.columns(4)
                            pc1.metric("Components (opt.)", str(n_opt))
                            pc2.metric("R² (calibration)", f"{pls_result.r2_calibration:.4f}")
                            pc3.metric("Q² (cross-val)", f"{pls_result.q2:.4f}",
                                       help="Q² > 0.9 = excellent predictive ability")
                            pc4.metric("RMSECV", f"{pls_result.rmsecv:.4f} ppm")

                            # RMSECV curve
                            if rmsecv_curve:
                                st.caption("RMSECV vs. number of PLS components")
                                st.line_chart(
                                    pd.DataFrame(
                                        {"RMSECV (ppm)": rmsecv_curve},
                                        index=range(1, len(rmsecv_curve) + 1),
                                    )
                                )

                            # VIP bar chart (top 20 wavelengths)
                            if pls_result.vip_scores is not None and len(pls_result.vip_scores):
                                st.caption("VIP scores — features above 1.0 are most informative")
                                top_n = min(20, len(pls_result.vip_scores))
                                top_idx = np.argsort(pls_result.vip_scores)[::-1][:top_n]
                                vip_df = pd.DataFrame({
                                    "Wavelength (nm)": (
                                        [f"{wl_common[i]:.0f}" for i in top_idx]
                                        if wl_common is not None else [str(i) for i in top_idx]
                                    ),
                                    "VIP Score": pls_result.vip_scores[top_idx],
                                })
                                st.bar_chart(vip_df.set_index("Wavelength (nm)"))

                            # Store in session
                            ss["ap_pls_model"] = pls_model
                            ss["ap_pls_result"] = pls_result
                            ss["ap_pls_wl_common"] = wl_common

                            # Quality gate
                            _qc_ok_pls, _qc_errs_pls, _qc_wrns_pls = _check_calibration_quality(
                                sensor_metrics=ss.get("ap_sensor_metrics", {}),
                                r2=pls_result.r2_calibration,
                            )
                            for _w in _qc_wrns_pls:
                                st.warning(f"⚠️ QC warning: {_w}")
                            if not _qc_ok_pls:
                                for _e in _qc_errs_pls:
                                    st.error(f"🚫 QC gate: {_e}")
                                st.error("PLS model not saved — quality below threshold.")
                            else:
                                with open(model_dir / "pls_calibration.pkl", "wb") as fh:
                                    pickle.dump(pls_model, fh)
                                st.success(
                                    f"✅ PLS model saved (`models/pls_calibration.pkl`) — "
                                    f"{n_opt} components, Q²={pls_result.q2:.4f}"
                                )
                                trained_ok = True
                                # ── MLflow experiment tracking ────────────────
                                try:
                                    from src.experiment_tracking import get_tracker
                                    _ts_mlf = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    _tracker = get_tracker(
                                        experiment_name=f"LSPR_{ss.get('ap_gas', 'Unknown')}"
                                    )
                                    with _tracker.start_run(
                                        run_name=f"PLS_{_ts_mlf}",
                                        tags={"gas": str(ss.get("ap_gas", "")), "model": "PLS"},
                                    ):
                                        _tracker.log_pls_run(
                                            pls_model=pls_model,
                                            pls_result=pls_result,
                                            sensor_metrics=ss.get("ap_sensor_metrics", {}),
                                            y_concs=np.array(y_concs, dtype=float),
                                            wavelengths=wl_common,
                                            gas_name=str(ss.get("ap_gas", "")),
                                        )
                                    st.info("MLflow run logged — view at http://localhost:5000")
                                except Exception as _mlf_e:
                                    st.caption(f"MLflow logging skipped: {_mlf_e}")
                    except Exception as ex:
                        st.warning(f"PLS training failed ({ex}), falling back to sklearn GPR.")

                elif "GPR" in model_type and _ML_AVAILABLE:
                    try:
                        import pickle

                        gpr = GPRCalibration()
                        gpr.fit(X, np.array(y_concs))
                        # Quality gate before save
                        _qc_ok2, _qc_errs2, _qc_wrns2 = _check_calibration_quality(
                            sensor_metrics=ss.get("ap_sensor_metrics", {}),
                            r2=ss.get("ap_r2"),
                        )
                        for _w2 in _qc_wrns2:
                            st.warning(f"⚠️ QC warning: {_w2}")
                        if not _qc_ok2:
                            for _e2 in _qc_errs2:
                                st.error(f"🚫 QC gate failed: {_e2}")
                            st.error("Model not saved — calibration quality below threshold.")
                        else:
                            with open(model_dir / "gpr_calibration.pkl", "wb") as fh:
                                pickle.dump(gpr, fh)
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

                    # ── Quality gate before save ──────────────────────────
                    _qc_ok, _qc_errors, _qc_warns = _check_calibration_quality(
                        sensor_metrics=ss.get("ap_sensor_metrics", {}),
                        r2=ss.get("ap_r2"),
                    )
                    for _w in _qc_warns:
                        st.warning(f"⚠️ QC warning: {_w}")
                    if not _qc_ok:
                        for _e in _qc_errors:
                            st.error(f"🚫 QC gate failed: {_e}")
                        st.error(
                            "Model **not saved** — calibration quality below threshold. "
                            "Fix the issues above and retrain."
                        )
                    else:
                        with open(model_dir / "gpr_sklearn.pkl", "wb") as fh:
                            pickle.dump(gpr_sk, fh)
                        st.success("✅ sklearn GPR trained and saved to `models/gpr_sklearn.pkl`.")
                        # ── MLflow experiment tracking ────────────────────
                        try:
                            from src.experiment_tracking import get_tracker
                            _ts_mlf = datetime.now().strftime("%Y%m%d_%H%M%S")
                            _tracker = get_tracker(
                                experiment_name=f"LSPR_{ss.get('ap_gas', 'Unknown')}"
                            )
                            with _tracker.start_run(
                                run_name=f"GPR_{_ts_mlf}",
                                tags={"gas": str(ss.get("ap_gas", "")), "model": "sklearn_GPR"},
                            ):
                                _tracker.log_gpr_run(
                                    model=gpr_sk,
                                    sensor_metrics=ss.get("ap_sensor_metrics", {}),
                                    y_concs=np.array(y_concs, dtype=float),
                                    X_features=X,
                                    gas_name=str(ss.get("ap_gas", "")),
                                )
                            st.info("MLflow run logged — view at http://localhost:5000")
                        except Exception as _mlf_e:
                            st.caption(f"MLflow logging skipped: {_mlf_e}")
                    ss["ap_gpr_sklearn"] = gpr_sk
                    # Conformal calibration on a held-out split for valid coverage
                    # (calibrating on the same data GPR was trained on gives optimistic scores)
                    if _CONFORMAL_AVAILABLE and len(X) >= 4:
                        try:
                            from sklearn.gaussian_process import GaussianProcessRegressor
                            from sklearn.gaussian_process.kernels import RBF, WhiteKernel

                            y_arr = np.array(y_concs, dtype=float)
                            n_cal = max(1, len(X) // 5)  # 20% hold-out, min 1
                            # Pick calibration points spanning the concentration range
                            sort_idx = np.argsort(y_arr)
                            cal_idx = sort_idx[:: max(1, len(X) // n_cal)][:n_cal]
                            train_idx = np.array([i for i in range(len(X)) if i not in set(cal_idx.tolist())])

                            _kernel_cal = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
                            _gpr_cal = GaussianProcessRegressor(
                                kernel=_kernel_cal, n_restarts_optimizer=2, normalize_y=True
                            )
                            _gpr_cal.fit(X[train_idx], y_arr[train_idx])
                            _cc = ConformalCalibrator()
                            _cc.calibrate(_gpr_cal, X[cal_idx], y_arr[cal_idx])
                            ss["ap_conformal"] = _cc
                        except Exception:
                            ss["ap_conformal"] = None
                    else:
                        ss["ap_conformal"] = None

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
            _save_session_to_disk()
            st.rerun()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4 — Deployment & Real-Time Inference  (Agent 05)
    # ═══════════════════════════════════════════════════════════════════════
    elif step == 4:
        st.subheader("Step 4 — Deployment & Real-Time Inference")

        # ── Training run comparison (MLflow) ──────────────────────────────
        with st.expander("📊 Compare Training Runs", expanded=False):
            try:
                from src.experiment_tracking import get_tracker
                _cmp_gas = str(ss.get("ap_gas", ss.get("ap_meta", {}).get("gas", "")))
                _cmp_tracker = get_tracker(experiment_name=f"LSPR_{_cmp_gas}" if _cmp_gas else "LSPR_Gas_Sensing")
                _cmp_df = _cmp_tracker.list_runs(max_results=10)
                if _cmp_df is not None and len(_cmp_df) > 0:
                    _show = [c for c in [
                        "run_name", "start_time",
                        "params.model_type", "metrics.r_squared",
                        "metrics.lod_ppm", "metrics.loq_ppm",
                        "metrics.q2_crossvalidated", "metrics.rmsecv",
                    ] if c in _cmp_df.columns]
                    _display_df = _cmp_df[_show].copy() if _show else _cmp_df
                    # Rename for readability
                    _display_df = _display_df.rename(columns={
                        "params.model_type": "Model",
                        "metrics.r_squared": "R²",
                        "metrics.lod_ppm": "LOD (ppm)",
                        "metrics.loq_ppm": "LOQ (ppm)",
                        "metrics.q2_crossvalidated": "Q²",
                        "metrics.rmsecv": "RMSECV",
                        "run_name": "Run",
                        "start_time": "Trained at",
                    })
                    st.dataframe(_display_df, use_container_width=True)
                    _best = _cmp_tracker.best_run(metric="metrics.r_squared")
                    if _best:
                        st.success(
                            f"🏆 Best run: **{_best.get('run_name', '')[:20]}** "
                            f"R²={_best.get('metrics.r_squared', 'N/A')} | "
                            f"LOD={_best.get('metrics.lod_ppm', 'N/A')} ppm"
                        )
                    st.caption("Launch MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlruns.db` → http://localhost:5000")
                else:
                    st.info("No training runs logged yet. Train a model in Step 3 to start tracking.")
            except Exception as _cmp_e:
                st.caption(f"Run comparison unavailable: {_cmp_e}")

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
        _sm4 = ss.get("ap_sensor_metrics", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Model Status", "🟢 Loaded & Ready")
        c2.metric(
            "LOB",
            f"{_sm4['lob_ppm']:.4f} ppm" if _sm4.get("lob_ppm") is not None else "N/A",
            help="Limit of Blank (IUPAC 2012)",
        )
        c3.metric(
            "LOD",
            f"{ss.get('ap_lod', 'N/A'):.4f} ppm" if isinstance(ss.get("ap_lod"), float) else "N/A",
            help="Limit of Detection — 3σ/S (IUPAC 2012)",
        )
        c4.metric(
            "LOL",
            f"{_sm4['lol_ppm']:.4f} ppm" if _sm4.get("lol_ppm") is not None else "N/A",
            help="Limit of Linearity — Mandel F-test (ICH Q2 §4.2)",
        )
        c5.metric(
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
            # Replay the same preprocessing pipeline chosen in Step 2
            proc = _preprocess(
                wl_live,
                frame,
                ss.get("ap_denoise", "Savitzky-Golay"),
                ss.get("ap_baseline_m", "ALS"),
                ss.get("ap_norm_m", "Min-Max [0,1]"),
            )

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

            # Conformal 90% interval (distribution-free coverage guarantee)
            conformal_lo: float | None = None
            conformal_hi: float | None = None
            _conformal = ss.get("ap_conformal")
            if _conformal is not None and gpr_model is not None:
                try:
                    _lo, _hi = _conformal.predict_interval(gpr_model, feat, alpha=0.10)
                    conformal_lo = float(_lo[0])
                    conformal_hi = float(_hi[0])
                except Exception:
                    pass

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
            if conformal_lo is not None and conformal_hi is not None:
                st.info(
                    f"**90% Conformal Interval:** [{conformal_lo:.2f}, {conformal_hi:.2f}] ppm"
                    f" — distribution-free coverage guarantee (n={_conformal.n_cal} cal. points)"
                )

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
                        proc_t = _preprocess(
                            wl_t,
                            it_t,
                            ss.get("ap_denoise", "Savitzky-Golay"),
                            ss.get("ap_baseline_m", "ALS"),
                            ss.get("ap_norm_m", "Min-Max [0,1]"),
                        )
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
        # QC advisory banner — warn if calibration quality is below threshold
        _sm_export = ss.get("ap_sensor_metrics", {})
        _r2_export = _sm_export.get("r_squared")
        if _r2_export is not None and float(_r2_export) < 0.95:
            st.warning(
                f"⚠️ **Export advisory:** Calibration R²={_r2_export:.4f} is below the 0.95 "
                "research-grade threshold. The report and archive will contain this warning. "
                "Consider retraining before submission."
            )
        elif not ss.get("ap_model_trained"):
            st.info("ℹ️ No model trained yet — report and archive will not contain calibration metrics.")

        if st.button("📄 Generate Session Report"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            _sm_rep = ss.get("ap_sensor_metrics", {})

            def _fmt(v: object, decimals: int = 4) -> str:
                return f"{v:.{decimals}f}" if isinstance(v, float) else str(v) if v is not None else "N/A"

            report = "# Gas Sensing Session Report\n\n"
            report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report += f"**Gas:** {meta.get('gas', 'N/A')}  \n"
            report += f"**Concentration:** {meta.get('concentration_ppm', 'N/A')} ppm  \n\n"
            # QC status line for traceability
            _r2_rpt = _sm_rep.get("r_squared")
            if _r2_rpt is not None:
                _qc_label = "PASS" if float(_r2_rpt) >= 0.95 else "ADVISORY — below 0.95 threshold"
                report += f"**Calibration QC:** {_qc_label} (R²={_r2_rpt:.4f})  \n\n"

            # ── IUPAC / ICH Q2 characterisation table ──────────────────
            report += "## Sensor Characterisation (IUPAC 2012 / ICH Q2(R1))\n\n"
            report += "| Metric | Value | Method |\n"
            report += "|--------|-------|--------|\n"
            report += f"| LOB | {_fmt(_sm_rep.get('lob_ppm'))} ppm | μ_blank + 1.645·σ_blank / S |\n"
            _lod_ci = (
                f" [{_fmt(_sm_rep.get('lod_ppm_ci_lower'))}–{_fmt(_sm_rep.get('lod_ppm_ci_upper'))}] 95% CI"
                if _sm_rep.get("lod_ppm_ci_lower") is not None else ""
            )
            report += f"| LOD | {_fmt(_sm_rep.get('lod_ppm'))} ppm{_lod_ci} | 3σ/S (IUPAC 2012) |\n"
            _loq_ci = (
                f" [{_fmt(_sm_rep.get('loq_ppm_ci_lower'))}–{_fmt(_sm_rep.get('loq_ppm_ci_upper'))}] 95% CI"
                if _sm_rep.get("loq_ppm_ci_lower") is not None else ""
            )
            report += f"| LOQ | {_fmt(_sm_rep.get('loq_ppm'))} ppm{_loq_ci} | 10σ/S (IUPAC 2012) |\n"
            report += f"| LOL | {_fmt(_sm_rep.get('lol_ppm'))} ppm | Mandel F-test (ICH Q2 §4.2) |\n"
            report += f"| Sensitivity | {_fmt(_sm_rep.get('sensitivity'))} a.u./ppm | OLS slope |\n"
            report += f"| R² | {_fmt(_sm_rep.get('r_squared'))} | Linear regression |\n"
            report += f"| RMSE | {_fmt(_sm_rep.get('rmse'))} a.u. | Calibration residuals |\n"
            _lin_rep = _sm_rep.get("mandel_linearity")
            if _lin_rep:
                is_lin = _lin_rep.get("is_linear", True)
                report += (
                    f"| Mandel linearity | {'Linear' if is_lin else 'Non-linear'} "
                    f"(p={_fmt(_lin_rep.get('p_value'), 4)}) | F-test vs quadratic |\n"
                )
            report += "\n"

            # ── Prediction statistics ───────────────────────────────────
            report += "## Prediction Statistics\n\n"
            report += f"**Total Predictions:** {len(hist)}  \n"
            if hist:
                preds = [h[0] for h in hist]
                report += f"**Mean Predicted Conc.:** {np.mean(preds):.2f} ppm  \n"
                report += f"**Std Predicted Conc.:** {np.std(preds):.2f} ppm  \n"

            rep_dir = _REPO / "reports"
            rep_dir.mkdir(exist_ok=True)
            rep_path = rep_dir / f"session_{ts}.md"
            rep_path.write_text(report, encoding="utf-8")
            st.download_button(
                "⬇️ Download Report", report, file_name=rep_path.name, mime="text/markdown"
            )
            st.success(f"Report saved to `{rep_path}`")

        # ── HDF5 Session Archive ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Archive Session (HDF5)")
        st.caption(
            "Save the complete session — spectra, calibration, and results — to a "
            "self-describing HDF5 file readable by MATLAB, Python h5py, and Origin."
        )
        if st.button("💾 Save HDF5 Archive"):
            try:
                from src.io.hdf5 import open_archive_writer

                arc_dir = _REPO / "output" / "archives"
                arc_dir.mkdir(parents=True, exist_ok=True)
                ts_arc = datetime.now().strftime("%Y%m%d_%H%M%S")
                arc_path = arc_dir / f"session_{ts_arc}.h5"

                _pp_arc = ss.get("ap_preprocessed", [])
                _sm_arc = ss.get("ap_sensor_metrics", {})
                _hist_arc = ss.get("ap_pred_history", [])
                _meta_arc = ss.get("ap_meta", {})

                with open_archive_writer(
                    arc_path,
                    gas_name=str(_meta_arc.get("gas", "unknown")),
                ) as aw:
                    # Store calibration spectra as frames
                    for _i, _it in enumerate(_pp_arc):
                        import datetime as _dt
                        from src.spectrometer.base import SpectralFrame
                        _mc = re.search(r"([\d.]+)ppm", _it.get("label", ""))
                        _c = float(_mc.group(1)) if _mc else None
                        _frame = SpectralFrame(
                            wavelengths=np.asarray(_it["wl"]),
                            intensities=np.asarray(_it["processed"]),
                            timestamp=_dt.datetime.now(_dt.timezone.utc),
                            integration_time_s=0.05,
                            model_name="agentic_pipeline",
                            metadata={"label": _it.get("label", ""), "step": "calibration"},
                        )
                        aw.add_frame(_frame, concentration_ppm=_c)

                    # Store calibration fit
                    if _sm_arc:
                        _cs_arc = np.array([
                            re.search(r"([\d.]+)ppm", it.get("label", "0ppm"))
                            for it in _pp_arc
                        ])
                        _y_arc = ss.get("ap_y_concs")
                        _resp_arc = ss.get("ap_X_train")
                        if _y_arc is not None and _resp_arc is not None:
                            aw.set_calibration(
                                concentrations=np.asarray(_y_arc),
                                responses=np.asarray(_resp_arc)[:, 0],
                                fit_result=_sm_arc,
                            )

                    # Store prediction history as results
                    for _ri, (_c_pred, _c_std) in enumerate(_hist_arc):
                        aw.add_result(
                            timestamp=datetime.now().isoformat(),
                            concentration_ppm=_c_pred,
                            uncertainty_ppm=_c_std,
                        )

                st.success(f"✅ HDF5 archive saved: `{arc_path.relative_to(_REPO)}`")
                with open(arc_path, "rb") as _fh:
                    st.download_button(
                        "⬇️ Download HDF5 Archive",
                        _fh.read(),
                        file_name=arc_path.name,
                        mime="application/x-hdf",
                    )
            except ImportError:
                st.warning("h5py not installed. Run `pip install h5py` to enable HDF5 export.")
            except Exception as _arc_e:
                st.error(f"HDF5 export failed: {_arc_e}")

        # ── Publication Figure Export ─────────────────────────────────────
        st.markdown("---")
        st.markdown("### Export Publication Figures")
        st.caption(
            "Generate journal-ready figures at the correct dimensions and DPI for ACS, "
            "Nature, RSC, or Elsevier submission."
        )
        _pub_col1, _pub_col2 = st.columns(2)
        _journal = _pub_col1.selectbox(
            "Journal preset",
            ["acs_single", "acs_double", "nature_s", "nature_d",
             "rsc_single", "rsc_double", "elsevier_s", "elsevier_d"],
            key="ap_pub_preset",
        )
        _fig_fmt = _pub_col2.selectbox(
            "Format", ["tiff", "pdf", "png", "svg", "eps"], key="ap_pub_fmt"
        )
        if st.button("📊 Generate Publication Figures"):
            try:
                from src.reporting.publication import (
                    save_calibration_figure,
                    save_spectral_overlay_figure,
                    save_pls_diagnostics_figure,
                )

                fig_dir = _REPO / "output" / "figures"
                fig_dir.mkdir(parents=True, exist_ok=True)
                ts_fig = datetime.now().strftime("%Y%m%d_%H%M%S")
                _sm_fig = ss.get("ap_sensor_metrics", {})
                _pp_fig = ss.get("ap_preprocessed", [])
                _y_fig = ss.get("ap_y_concs")
                _generated: list[str] = []

                # Figure 1 — Calibration curve
                if _sm_fig and _y_fig is not None and ss.get("ap_X_train") is not None:
                    try:
                        cal_path = fig_dir / f"fig_calibration_{ts_fig}.{_fig_fmt}"
                        save_calibration_figure(
                            concentrations=np.asarray(_y_fig),
                            responses=np.asarray(ss["ap_X_train"])[:, 0],
                            fit_result=_sm_fig,
                            out_path=cal_path,
                            preset=_journal,
                            gas_name=str(ss.get("ap_gas", "")),
                            panel_label="(a)",
                        )
                        _generated.append(str(cal_path.name))
                    except Exception as _fe:
                        st.warning(f"Calibration figure: {_fe}")

                # Figure 2 — Spectral overlay
                if _pp_fig and len(_pp_fig) >= 2:
                    try:
                        _wl_fig = wl_common if wl_common is not None else np.linspace(
                            float(min(it["wl"].min() for it in _pp_fig)),
                            float(max(it["wl"].max() for it in _pp_fig)),
                            300,
                        )
                        _Z_fig = np.array([
                            np.interp(_wl_fig, it["wl"], it["processed"]) for it in _pp_fig
                        ])
                        _c_fig = np.array([
                            float(re.search(r"([\d.]+)ppm", it["label"]).group(1))
                            if re.search(r"([\d.]+)ppm", it["label"]) else float(i)
                            for i, it in enumerate(_pp_fig)
                        ])
                        ov_path = fig_dir / f"fig_spectral_overlay_{ts_fig}.{_fig_fmt}"
                        save_spectral_overlay_figure(
                            wavelengths=_wl_fig,
                            spectra=_Z_fig,
                            concentrations=_c_fig,
                            out_path=ov_path,
                            preset=_journal,
                            panel_label="(b)",
                        )
                        _generated.append(str(ov_path.name))
                    except Exception as _fe:
                        st.warning(f"Spectral overlay figure: {_fe}")

                # Figure 3 — PLS diagnostics (only when PLS was trained)
                _pls_res = ss.get("ap_pls_result")
                _pls_mdl = ss.get("ap_pls_model")
                if _pls_res is not None and _pls_mdl is not None and _y_fig is not None:
                    try:
                        _X_pls_fig = Z if Z is not None else ss.get("ap_X_train")
                        if _X_pls_fig is not None:
                            _y_pred_pls = _pls_mdl.predict(np.asarray(_X_pls_fig))
                            pls_path = fig_dir / f"fig_pls_diagnostics_{ts_fig}.{_fig_fmt}"
                            save_pls_diagnostics_figure(
                                pls_result=_pls_res,
                                concentrations=np.asarray(_y_fig),
                                predicted=_y_pred_pls,
                                out_path=pls_path,
                                preset="acs_double",
                                wavelengths=ss.get("ap_pls_wl_common"),
                                panel_labels=("(a)", "(b)", "(c)", "(d)"),
                            )
                            _generated.append(str(pls_path.name))
                    except Exception as _fe:
                        st.warning(f"PLS diagnostics figure: {_fe}")

                if _generated:
                    st.success(
                        f"✅ {len(_generated)} figure(s) saved to `output/figures/`: "
                        + ", ".join(_generated)
                    )
                else:
                    st.warning(
                        "No figures generated — train a model and ensure calibration metrics are available."
                    )
            except ImportError as _imp_e:
                st.error(f"Publication module unavailable: {_imp_e}")
            except Exception as _pub_e:
                st.error(f"Figure export failed: {_pub_e}")

        # ── MLflow Experiment History ─────────────────────────────────────
        st.markdown("---")
        st.markdown("### Experiment History (MLflow)")
        st.caption(
            "Compare all training runs for this gas. "
            "Start the MLflow UI with `mlflow ui` in the project root to browse interactively."
        )
        with st.expander("View past runs", expanded=False):
            try:
                from src.experiment_tracking import get_tracker

                _exp_gas = str(ss.get("ap_gas", "Unknown"))
                _hist_tracker = get_tracker(experiment_name=f"LSPR_{_exp_gas}")
                _runs_df = _hist_tracker.list_runs(max_results=20)
                if _runs_df is not None and len(_runs_df) > 0:
                    # Surface the most useful columns
                    _show_cols = [
                        c for c in [
                            "run_name", "status", "start_time",
                            "metrics.r_squared", "metrics.lod_ppm",
                            "metrics.loq_ppm", "metrics.rmsecv",
                            "metrics.q2_crossvalidated", "metrics.n_components",
                            "params.model_type", "params.gas_name",
                        ]
                        if c in _runs_df.columns
                    ]
                    st.dataframe(
                        _runs_df[_show_cols] if _show_cols else _runs_df,
                        use_container_width=True,
                    )
                    # Best run highlight
                    try:
                        _best = _hist_tracker.best_run(metric="metrics.r_squared", higher_is_better=True)
                        if _best:
                            st.success(
                                f"Best run: **{_best.get('run_name', _best.get('run_id', '')[:8])}** "
                                f"— R²={_best.get('metrics.r_squared', 'N/A')}"
                            )
                    except Exception:
                        pass
                    st.caption(
                        "Launch MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlruns.db` "
                        "then open http://localhost:5000"
                    )
                else:
                    st.info("No runs logged yet for this gas. Train a model above to start tracking.")
            except ImportError:
                st.info("MLflow not installed — run `pip install mlflow` to enable experiment tracking.")
            except Exception as _mlf_hist_e:
                st.warning(f"Could not load experiment history: {_mlf_hist_e}")

        if st.button("⬅️ Back to Step 3"):
            ss["ap_step"] = 3
            st.rerun()
