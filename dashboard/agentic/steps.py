"""
Pipeline step business logic — pure functions and light session-state helpers.

These functions may read/write ``st.session_state`` (via the ``ss`` dict
argument) but must NOT call any Streamlit layout functions
(``st.columns``, ``st.expander``, ``st.write``, etc.).

The one exception is ``_acquire_frames``, which calls ``st.progress`` /
``chart_ph.plotly_chart`` for live feedback — these are injected by the
caller in ``tab.py`` rather than created here, keeping the coupling minimal.
"""
from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── project root ───────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ── hardware ───────────────────────────────────────────────────────────────────
try:
    from src.acquisition import CCS200Spectrometer

    if CCS200Spectrometer is None:
        raise ImportError("CCS200Spectrometer unavailable")
    _HW_AVAILABLE = True
except Exception:
    _HW_AVAILABLE = False

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# UI badge helper (HTML-only — no st.* call)
# ─────────────────────────────────────────────────────────────────────────────


def _step_badge(n: int, label: str, done: bool, active: bool) -> str:
    colour = "#2ecc71" if done else ("#1f77b4" if active else "#888")
    icon = "✓" if done else str(n)
    return (
        f'<span style="background:{colour};color:white;border-radius:50%;'
        f'padding:2px 9px;font-weight:bold;margin-right:6px">{icon}</span>'
        f'<span style="color:{colour};font-weight:{"bold" if active else "normal"}">{label}</span>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quality gate
# ─────────────────────────────────────────────────────────────────────────────


def _check_calibration_quality(
    sensor_metrics: dict,
    r2: float | None,
    r2_cv: float | None = None,
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

    # Prefer cross-validated R² when available.
    # This avoids accepting overfit calibrations on in-sample fit quality.
    eff_r2 = r2_cv if r2_cv is not None else r2
    if eff_r2 is not None:
        if eff_r2 < min_r2:
            metric_name = "R²_LOOCV" if r2_cv is not None else "R²"
            errors.append(
                f"{metric_name}={eff_r2:.4f} is below the minimum threshold of {min_r2:.2f}. "
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
        # Hard gate: LOD >= LOQ violates IUPAC ordering.
        loq = sensor_metrics.get("loq_ppm")
        if loq is not None and lod >= loq:
            errors.append(
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
# Acquisition
# ─────────────────────────────────────────────────────────────────────────────


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
    # Import here to avoid circular dependency at module level
    from dashboard.agentic.lspr_physics import _simulate_frame

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


# ─────────────────────────────────────────────────────────────────────────────
# Dataset scanning
# ─────────────────────────────────────────────────────────────────────────────


def _scan_dataset_dir() -> list[Path]:
    """Return all CSVs inside data/automation_dataset/."""
    base = _REPO / "data" / "automation_dataset"
    if not base.exists():
        return []
    return sorted(base.rglob("*.csv"))


@st.cache_data(show_spinner=False)
def _load_csv_spectrum(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    """Load wavelength + intensity from a CSV. Handles both headered and bare files.

    Accepts:
    - Two-column bare CSV (no header): wavelength, intensity
    - Multi-column headered CSV: columns named 'wavelength'/'intensity' (our standard format)
    """
    df = pd.read_csv(path_str)

    # Detect if first row is actually data (no header) or a named header
    try:
        float(df.columns[0])
        # First column name is a number → no header; re-read
        df = pd.read_csv(path_str, header=None, names=["wavelength", "intensity"])
    except (ValueError, TypeError):
        pass  # has header row

    wl_col = next((c for c in df.columns if str(c).lower().startswith("wavel")), df.columns[0])
    int_col = next((c for c in df.columns if str(c).lower().startswith("intens")), df.columns[1])
    return df[wl_col].astype(float).values, df[int_col].astype(float).values


# ─────────────────────────────────────────────────────────────────────────────
# Project store synchronisation
# ─────────────────────────────────────────────────────────────────────────────


def _sync_to_project_store(ss: dict) -> None:
    """Snapshot the agentic pipeline's current state into the shared ProjectStore.

    Called at each step transition so the Predict Unknown tab (and any other
    tab) can see the latest calibration, model, and performance data without
    the researcher re-uploading anything.
    """
    try:
        from dashboard.project_store import get_project
        proj = get_project()

        meta = ss.get("ap_meta") or {}
        gas = meta.get("gas") or ss.get("ap_gas") or "unknown"
        proj.gas_name = gas

        # Wavelengths
        wl = ss.get("ap_wl")
        if wl is not None:
            proj.set_wavelengths(wl)

        # Reference spectrum
        ref_spec = ss.get("ap_ref_spectrum")
        ref_wl = ss.get("ap_ref_wl")
        ref_peak = ss.get("ap_ref_peak_wl")
        if ref_spec is not None:
            if ref_wl is not None and wl is not None:
                ref_interp = np.interp(wl, ref_wl, ref_spec)
            else:
                ref_interp = ref_spec
            proj.set_reference(ref_interp, peak_nm=ref_peak)

        # Calibration concentrations + responses (from preprocessed entries)
        preprocessed = ss.get("ap_preprocessed") or ss.get("ap_pp_items") or []
        if preprocessed:
            concs, resps = [], []
            for entry in preprocessed:
                c = entry.get("concentration_ppm") or entry.get("conc_ppm")
                r = entry.get("peak_shift") or entry.get("peak_wl") or entry.get("delta_lambda")
                if c is not None and r is not None:
                    concs.append(float(c))
                    resps.append(float(r))
            if len(concs) >= 2:
                baseline_m = ss.get("ap_baseline_m", "als")
                norm_m = ss.get("ap_norm_m", "none")
                proj.set_calibration(
                    np.array(concs), np.array(resps),
                    preprocessing_cfg={"baseline_method": baseline_m, "norm_method": norm_m},
                )

        # Trained model
        sensor_metrics = ss.get("ap_sensor_metrics") or {}
        gpr_model = ss.get("ap_gpr_sklearn")
        pls_model = ss.get("ap_pls_model")
        model_path_ss = ss.get("ap_model_path") or ss.get("ap_model_save_path")

        if (gpr_model is not None or pls_model is not None) and sensor_metrics:
            model_type = "PLS" if pls_model is not None else "GPR"
            if model_path_ss:
                lod = ss.get("ap_lod")
                r2 = ss.get("ap_r2")
                perf = dict(sensor_metrics)
                if lod is not None:
                    perf["lod_ppm"] = float(lod)
                if r2 is not None:
                    perf["r_squared"] = float(r2)
                proj.set_model(str(model_path_ss), model_type, perf)
                proj.save()
    except Exception as _e:
        log.debug("ProjectStore sync skipped: %s", _e)
