"""
Live Sensor Dashboard
---------------------
Streamlit page for real-time CCS200 gas sensor visualization.

Reads data from LiveDataStore (in-process singleton — no file polling).
Embedded as Tab 4 inside dashboard/app.py.

The SensorOrchestrator is stored in ``st.session_state`` so it persists
across Streamlit reruns.  The acquisition thread runs inside the Streamlit
process; LiveDataStore is therefore shared in-process without IPC.

Layout:
  [Connect panel]  [Status metrics]
  [Live spectrum]
  [Concentration trend + GPR uncertainty]
  [Wavelength shift trend]
  [Recent measurements table]
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Make sure project root is importable when this file is run directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.live_state import LiveDataStore

# ---------------------------------------------------------------------------
# Gas options (matching config.yaml roi overrides)
# ---------------------------------------------------------------------------
_GAS_OPTIONS = ["Ethanol", "Methanol", "Isopropanol", "MixVOC", "unknown"]


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------


def render() -> None:
    """
    Called by dashboard/app.py inside the Live Sensor tab.
    Renders the full live sensor UI and auto-refreshes every N seconds.
    """
    st.header("Live CCS200 Gas Sensor")

    # ---- Connection control panel ----
    _render_connection_panel()

    st.markdown("---")

    # ---- Status metrics ----
    _render_status_metrics()

    st.markdown("---")

    # ---- Live spectrum ----
    _render_live_spectrum()

    # ---- Trend plots ----
    recent_data = LiveDataStore.get_latest(500)
    if recent_data:
        df = _build_dataframe(recent_data)
        for _render_fn, _label in [
            (_render_concentration_trend, "Concentration trend"),
            (_render_shift_trend, "Wavelength shift trend"),
            (_render_recent_table, "Recent results"),
        ]:
            try:
                _render_fn(df)
            except Exception as _exc:
                st.warning(f"{_label} unavailable: {_exc}")
    else:
        st.info(
            "No data yet.  Click **Connect & Start** above to begin acquisition,\n"
            "or run:  `python run.py --mode sensor --gas Ethanol --duration 3600`"
        )

    # ---- Agent status (drift + training) ----
    if st.session_state.get("orchestrator") is not None:
        st.markdown("---")
        _render_agent_status()

    # ---- Auto-refresh (only when sensor is actively streaming) ----
    # Sleeping unconditionally blocks the entire Streamlit server process;
    # gate on is_running() so idle tab visits cost nothing.
    refresh_rate = st.sidebar.slider(
        "Dashboard Refresh (s)",
        min_value=1,
        max_value=10,
        value=2,
        key="sensor_refresh_rate",
    )
    if LiveDataStore.is_running():
        time.sleep(refresh_rate)
        st.rerun()


# ---------------------------------------------------------------------------
# Sub-renderers
# ---------------------------------------------------------------------------


def _render_connection_panel() -> None:
    """Connect / Stop controls with gas selector."""
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        gas_label = st.selectbox(
            "Gas label",
            _GAS_OPTIONS,
            key="sensor_gas_label",
        )

    with col2:
        integration_ms = st.number_input(
            "Integration (ms)",
            min_value=10.0,
            max_value=500.0,
            value=50.0,
            step=10.0,
            key="sensor_integration_ms",
        )

    is_running = LiveDataStore.is_running()

    with col3:
        if st.button("Connect & Start", disabled=is_running, key="btn_connect"):
            _start_session(gas_label, integration_ms)

    with col4:
        if st.button("Stop", disabled=not is_running, key="btn_stop"):
            _stop_session()


def _start_session(gas_label: str, integration_ms: float) -> None:
    """Instantiate orchestrator and start acquisition."""
    try:
        from config.config_loader import load_config
        from src.inference.orchestrator import SensorOrchestrator

        try:
            cfg = load_config(str(REPO_ROOT / "config" / "config.yaml"))
        except Exception:
            cfg = {}

        orch = SensorOrchestrator.from_config(cfg)
        # Override with UI values
        orch.integration_time_ms = integration_ms

        orch.start_session(gas_label=gas_label)
        st.session_state["orchestrator"] = orch
        st.success(f"Connected. Acquiring {gas_label} spectra…")

    except Exception as exc:
        st.error(f"Connection failed: {exc}")
        import traceback

        st.code(traceback.format_exc())


def _stop_session() -> None:
    orch = st.session_state.get("orchestrator")
    if orch is not None:
        try:
            session_dir = orch.stop_session()
            if session_dir:
                st.success(f"Session saved to {session_dir}")
        except Exception as exc:
            st.warning(f"Stop error: {exc}")
        finally:
            st.session_state["orchestrator"] = None
    else:
        LiveDataStore.set_running(False)


def _render_status_metrics() -> None:
    """Four metric cards: status, samples, gas, last concentration."""
    meta = LiveDataStore.get_session_meta()
    is_running = LiveDataStore.is_running()
    sample_count = LiveDataStore.get_sample_count()
    last = LiveDataStore.get_last_result()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if is_running:
            st.success("ACQUIRING")
        else:
            st.error("STOPPED")

    with col2:
        st.metric("Samples", f"{sample_count:,}")

    with col3:
        gas = meta.get("gas_label", "N/A")
        st.metric("Gas", gas)

    with col4:
        if last and last.get("concentration_ppm") is not None:
            conc = last["concentration_ppm"]
            unc = last.get("gpr_uncertainty")
            label = f"{conc:.3f} ppm"
            if unc is not None:
                label += f" ±{unc:.3f}"
            st.metric("Concentration", label)
        else:
            st.metric("Concentration", "—")


def _render_live_spectrum() -> None:
    """Live spectrum chart with ROI band overlay."""
    spectrum = LiveDataStore.get_latest_spectrum()
    last = LiveDataStore.get_last_result()

    if spectrum is None:
        return

    wl, inten = spectrum
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=wl,
            y=inten,
            mode="lines",
            name="Spectrum",
            line=dict(color="steelblue", width=1.5),
        )
    )

    # Overlay ROI band if available
    if last and last.get("roi_start") is not None and last.get("roi_end") is not None:
        fig.add_vrect(
            x0=last["roi_start"],
            x1=last["roi_end"],
            fillcolor="orange",
            opacity=0.2,
            annotation_text="ROI",
            annotation_position="top left",
        )

    # Mark peak wavelength
    if last and last.get("peak_wavelength") is not None:
        peak_wl = last["peak_wavelength"]
        idx = np.argmin(np.abs(wl - peak_wl))
        fig.add_trace(
            go.Scatter(
                x=[peak_wl],
                y=[inten[idx]],
                mode="markers",
                marker=dict(color="red", size=10, symbol="x"),
                name=f"Peak {peak_wl:.2f} nm",
            )
        )

    fig.update_layout(
        title="Live Spectrum",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity (counts)",
        height=300,
        margin=dict(t=40, b=40, l=50, r=20),
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_concentration_trend(df: pd.DataFrame) -> None:
    """Concentration over time with optional GPR uncertainty band, plus SNR/confidence."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Concentration (ppm)", "SNR & Confidence"),
        vertical_spacing=0.10,
        row_heights=[0.65, 0.35],
    )

    # Concentration line
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["concentration_ppm"],
            mode="lines",
            name="Concentration",
            line=dict(color="green", width=2),
        ),
        row=1,
        col=1,
    )

    # GPR uncertainty band
    if "gpr_uncertainty" in df.columns and df["gpr_uncertainty"].notna().any():
        upper = df["concentration_ppm"] + df["gpr_uncertainty"]
        lower = df["concentration_ppm"] - df["gpr_uncertainty"]
        x_band = pd.concat([df["datetime"], df["datetime"].iloc[::-1]])
        y_band = pd.concat([upper, lower.iloc[::-1]])
        fig.add_trace(
            go.Scatter(
                x=x_band,
                y=y_band,
                fill="toself",
                fillcolor="rgba(0, 200, 0, 0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="GPR ±1σ",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # SNR
    if "snr" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["snr"],
                mode="lines",
                name="SNR",
                line=dict(color="royalblue", width=1.5),
            ),
            row=2,
            col=1,
        )

    # Confidence
    if "confidence_score" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["confidence_score"],
                mode="lines",
                name="Confidence",
                line=dict(color="orange", width=1.5),
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="ppm", row=1, col=1)
    fig.update_yaxes(title_text="score / dB", row=2, col=1)
    fig.update_layout(height=500, title_text="Concentration & Quality")
    st.plotly_chart(fig, use_container_width=True)

    # Gas type badge (CNN)
    if "gas_type" in df.columns and df["gas_type"].notna().any():
        last_gas = df["gas_type"].dropna().iloc[-1]
        conf_val = (
            df["confidence_score"].dropna().iloc[-1] if "confidence_score" in df.columns else None
        )
        badge = f"**CNN Predicted Gas:** {last_gas}"
        if conf_val is not None:
            badge += f"  (confidence: {conf_val:.1%})"
        st.info(badge)


def _render_shift_trend(df: pd.DataFrame) -> None:
    """Wavelength shift time series."""
    if "wavelength_shift" not in df.columns:
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["wavelength_shift"],
            mode="lines+markers",
            name="Wavelength Shift",
            line=dict(color="purple", width=2),
            marker=dict(size=3),
        )
    )
    fig.update_layout(
        title="Wavelength Shift over Time",
        xaxis_title="Time",
        yaxis_title="Shift (nm)",
        height=260,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_recent_table(df: pd.DataFrame) -> None:
    """Last 10 measurements as a formatted table."""
    st.subheader("Recent Measurements (last 10)")

    display_cols = [
        "datetime",
        "concentration_ppm",
        "wavelength_shift",
        "snr",
        "confidence_score",
        "gas_type",
        "quality_score",
        "processing_time_ms",
    ]
    cols_present = [c for c in display_cols if c in df.columns]
    tail = df.tail(10)[cols_present].copy()
    tail["datetime"] = tail["datetime"].dt.strftime("%H:%M:%S.%f").str[:-3]

    # Round floats for readability
    for col in [
        "concentration_ppm",
        "wavelength_shift",
        "snr",
        "confidence_score",
        "quality_score",
        "processing_time_ms",
    ]:
        if col in tail.columns:
            tail[col] = tail[col].round(4)

    st.dataframe(tail, use_container_width=True)

    # Export button
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        label="Export session CSV",
        data=csv_bytes,
        file_name="live_session_export.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_dataframe(recent_data: list) -> pd.DataFrame:
    df = pd.DataFrame(recent_data)
    if "timestamp" in df.columns:
        # Handles datetime objects (from orchestrator) and legacy unix floats.
        # utc=True normalises timezone-aware datetimes; tz_convert(None) strips
        # tz info so Plotly/Streamlit can display without UTC suffix clutter.
        df["datetime"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    else:
        # Guarantee "datetime" column exists so renderers never KeyError
        df["datetime"] = pd.Timestamp.now()
    return df


# ---------------------------------------------------------------------------
# Agent status panel
# ---------------------------------------------------------------------------


def _render_agent_status() -> None:
    """Show live drift detection and training agent status.

    Reads directly from the SensorOrchestrator stored in session state.
    Only rendered when a session is active (orchestrator is not None).
    """
    orch = st.session_state.get("orchestrator")
    if orch is None:
        return

    st.subheader("Agent Status")
    col_drift, col_train = st.columns(2)

    # ---- Drift Monitor --------------------------------------------------
    with col_drift:
        st.markdown("**Drift Monitor**")
        try:
            ds = orch.drift_agent.get_status()
            n = ds["n_samples"]
            if n < 20:
                st.info(f"Warming up ({n}/20 samples)…")
            elif ds["is_drifting"]:
                rate = ds["drift_rate_nm_per_min"]
                offset = ds["baseline_offset_nm"]
                st.error(f"DRIFT DETECTED  rate={rate:+.3f} nm/min  offset={offset:+.3f} nm")
            else:
                st.success(f"Stable  drift rate={ds['drift_rate_nm_per_min']:+.4f} nm/min")

            # Last 3 drift alerts
            alerts = orch.drift_agent.get_recent_alerts(3)
            for alert in reversed(alerts):
                icon = (
                    "🔴"
                    if alert.severity == "critical"
                    else "🟡"
                    if alert.severity == "warning"
                    else "🟢"
                )
                st.caption(
                    f"{icon} {alert.timestamp.strftime('%H:%M:%S')} "
                    f"[{alert.alert_type.value}] — {alert.message}"
                )

            if ds["is_drifting"]:
                if st.button("Reset Baseline", key="btn_reset_baseline"):
                    orch.drift_agent.reset_baseline()
                    st.success("Baseline reset to current window mean.")
        except Exception as exc:
            st.warning(f"Drift agent unavailable: {exc}")

    # ---- Training Agent -------------------------------------------------
    with col_train:
        st.markdown("**Training Agent**")
        try:
            ts = orch.training_agent.get_status()

            if ts["is_retraining"]:
                st.warning("Retraining in progress…")

            r2 = ts.get("avg_r2_last_window")
            if r2 is not None:
                delta = "OK" if r2 >= 0.9 else "LOW — retrain may trigger"
                st.metric("GPR R² (rolling 20-sample avg)", f"{r2:.4f}", delta=delta)
            else:
                st.caption("No R² data yet (need 20+ samples with GPR loaded)")

            st.caption(f"Samples since last retrain: {ts['samples_since_retrain']:,}")
            st.caption(f"Total retrain cycles: {ts['n_retrain_cycles']}")
            if ts.get("last_retrain"):
                st.caption(f"Last retrain: {ts['last_retrain']}")

            if st.button(
                "Trigger Manual Retrain",
                key="btn_manual_retrain",
                disabled=ts["is_retraining"],
                help="Queue an immediate retrain cycle regardless of triggers",
            ):
                orch.training_agent.trigger_manual_retrain()
                st.info("Retrain queued — running in background thread.")
        except Exception as exc:
            st.warning(f"Training agent unavailable: {exc}")


# ---------------------------------------------------------------------------
# Stand-alone entry point (for testing without app.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    st.set_page_config(page_title="Live Gas Sensor", layout="wide")
    render()
