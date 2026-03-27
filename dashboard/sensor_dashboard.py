"""
Live Sensor Dashboard
---------------------
Streamlit page for CCS200 gas sensor control and monitoring.

The live spectrum / trend charts are served by a FastAPI + WebSocket server
(``dashboard/live_server.py``) embedded as an iframe.  This file handles only:

- Connection control panel (Connect & Start / Stop)
- Sensor peak-window and calibration slope configuration
- Agent status (drift monitor + training agent)

Layout::

  [Connect panel]  [Sensor Peak Window expander]
  ─────────────────────────────────────────────
  [iframe → http://localhost:5006 (uPlot live view)]
  ─────────────────────────────────────────────
  [Agent status — drift / training]
"""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.live_state import LiveDataStore

_GAS_OPTIONS = ["Ethanol", "Methanol", "Isopropanol", "MixVOC", "unknown"]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def render(live_server_port: int = 5006) -> None:
    """
    Called by dashboard/app.py inside the Live Sensor tab.

    The live spectrum / trend view is served by the FastAPI + WebSocket server
    (dashboard/live_server.py) embedded as an iframe.  This Streamlit page
    handles only the connection controls and agent status — it never polls or
    sleeps, so it never blocks the Streamlit server.
    """
    st.header("Live CCS200 Gas Sensor")

    # ---- Connection control panel ----
    _render_connection_panel()

    st.markdown("---")

    # ---- Live view — iframe into the FastAPI/uPlot server ----
    live_url = f"http://localhost:{live_server_port}"
    st.components.v1.iframe(live_url, height=700, scrolling=False)

    st.caption(
        f"Live view served by WebSocket server at [{live_url}]({live_url})  "
        "— open that URL directly for a distraction-free full-window view."
    )

    # ---- Agent status (drift + training) ----
    if st.session_state.get("orchestrator") is not None:
        st.markdown("---")
        _render_agent_status()


# ---------------------------------------------------------------------------
# Connection panel
# ---------------------------------------------------------------------------


def _render_connection_panel() -> None:
    """Connect / Stop controls with gas selector and sensor configuration."""
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

    with st.expander("Sensor Peak Window", expanded=False):
        st.caption(
            "Wavelength window the pipeline searches for the sensor response peak. "
            "Set to the range containing your sensor peak — outside this window is ignored."
        )
        _sw1, _sw2, _sw3 = st.columns(3)
        _sw1.number_input(
            "Min (nm)", min_value=100.0, max_value=2000.0,
            value=350.0, step=10.0, key="live_peak_min",
        )
        _sw2.number_input(
            "Max (nm)", min_value=100.0, max_value=2000.0,
            value=950.0, step=10.0, key="live_peak_max",
        )
        _sw3.number_input(
            "Cal. slope (nm/ppm)", value=-0.116, format="%.4f",
            key="live_cal_slope",
            help="Negative = blue-shift sensor, Positive = red-shift sensor.",
        )


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


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
        orch.integration_time_ms = integration_ms

        _peak_min = float(st.session_state.get("live_peak_min", 350.0))
        _peak_max = float(st.session_state.get("live_peak_max", 950.0))
        _cal_slope = float(st.session_state.get("live_cal_slope", -0.116))
        if hasattr(orch, "pipeline") and hasattr(orch.pipeline, "config"):
            orch.pipeline.config.peak_search_min_nm = _peak_min
            orch.pipeline.config.peak_search_max_nm = _peak_max
            orch.pipeline.config.calibration_slope = _cal_slope
            orch.pipeline.config.reference_wavelength = (_peak_min + _peak_max) / 2

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


# ---------------------------------------------------------------------------
# Agent status panel
# ---------------------------------------------------------------------------


def _render_agent_status() -> None:
    """Drift monitor + training agent status."""
    orch = st.session_state.get("orchestrator")
    if orch is None:
        return

    st.subheader("Agent Status")
    col_drift, col_train = st.columns(2)

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

            for alert in reversed(orch.drift_agent.get_recent_alerts(3)):
                icon = (
                    "🔴" if alert.severity == "critical"
                    else "🟡" if alert.severity == "warning"
                    else "🟢"
                )
                st.caption(
                    f"{icon} {alert.timestamp.strftime('%H:%M:%S')} "
                    f"[{alert.alert_type.value}] — {alert.message}"
                )

            if ds["is_drifting"]:
                if st.button("Reset Baseline", key="btn_reset_baseline"):
                    orch.drift_agent.reset_baseline()
                    st.success("Baseline reset.")
        except Exception as exc:
            st.warning(f"Drift agent unavailable: {exc}")

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
            ):
                orch.training_agent.trigger_manual_retrain()
                st.info("Retrain queued.")
        except Exception as exc:
            st.warning(f"Training agent unavailable: {exc}")


# ---------------------------------------------------------------------------
# Stand-alone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    st.set_page_config(page_title="Live Gas Sensor", layout="wide")
    render()
