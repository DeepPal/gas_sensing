"""
dashboard.project_store
========================
Shared project state for the SpectraAgent Streamlit dashboard.

**The problem this solves:**
Each dashboard tab previously managed its own isolated state using namespaced
``st.session_state`` keys (``ap_*``, ``de_*``, etc.).  A researcher who built
a calibration in Tab 1 had to re-upload the same data in Tab 5.  Returning the
next day meant starting from scratch.

**How it works:**
A single :class:`ProjectStore` object lives in ``st.session_state["project"]``.
Every tab reads from and writes to it.  The store can be saved to
``output/sessions/{id}/project_state.json`` and restored next session.

Usage
-----
::

    from dashboard.project_store import get_project, init_project

    # Get or create the current project (call once per tab at top of render())
    proj = get_project()

    # After fitting calibration, write results to the store
    proj.set_calibration(concentrations, responses, model_path, performance)
    proj.save()   # persist to disk

    # In a different tab — read the calibration back
    proj = get_project()
    if proj.has_calibration:
        concs = proj.calibration_concentrations
        model_path = proj.model_path

"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SESSIONS_DIR = _REPO_ROOT / "output" / "sessions"

# ---------------------------------------------------------------------------
# ProjectStore dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProjectStore:
    """Single shared project context, persisted across all dashboard tabs.

    Attributes
    ----------
    session_id : str
        Unique identifier — ``YYYYMMDD_HHMMSS`` format.
    gas_name : str
        Current analyte label (e.g. "Ethanol").
    created_at : str
        ISO-8601 creation timestamp.
    notes : str
        Free-text researcher notes.

    Spectral data
    -------------
    wavelengths : np.ndarray | None
        Wavelength axis shared by all spectra in this session (nm).
    reference_spectrum : np.ndarray | None
        Blank / zero-gas reference intensity spectrum.
    reference_peak_nm : float | None
        Peak wavelength of the reference spectrum (nm).

    Calibration
    -----------
    calibration_concentrations : np.ndarray | None
        Known analyte concentrations (ppm) used to build the calibration curve.
    calibration_responses : np.ndarray | None
        Sensor responses (Δλ nm, or peak intensity) at each calibration point.
    preprocessing_config : dict
        Parameters used in preprocessing (baseline method, normalisation, etc.).

    Model
    -----
    model_path : str | None
        Path to the saved model file (.joblib or .pt) relative to repo root.
    model_type : str
        Model class name, e.g. "GPR", "PLS", "LinearOLS".
    performance : dict
        Result of ``sensor_performance_summary()``:
        LOD, LOQ, R², sensitivity, residual diagnostics, etc.

    Predictions
    -----------
    predictions : list[dict]
        Accumulated predictions:
        [{timestamp, spectrum_file, concentration, ci_lower, ci_upper, quality}]

    Status flags
    ------------
    step_complete : dict[str, bool]
        Which pipeline steps have been completed in this session.
    """

    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    gas_name: str = "unknown"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""

    # Spectral data (not JSON-serialised directly — stored as .npy)
    _wavelengths: np.ndarray | None = field(default=None, repr=False)
    _reference_spectrum: np.ndarray | None = field(default=None, repr=False)
    reference_peak_nm: float | None = None

    # Calibration
    _calibration_concentrations: np.ndarray | None = field(default=None, repr=False)
    _calibration_responses: np.ndarray | None = field(default=None, repr=False)
    preprocessing_config: dict = field(default_factory=dict)

    # Model
    model_path: str | None = None
    model_type: str = "GPR"
    performance: dict = field(default_factory=dict)

    # Predictions
    predictions: list[dict] = field(default_factory=list)

    # Step completion flags
    step_complete: dict = field(default_factory=lambda: {
        "acquisition": False,
        "preprocessing": False,
        "training": False,
        "validation": False,
    })

    # ---------------------------------------------------------------------------
    # Convenience properties
    # ---------------------------------------------------------------------------

    @property
    def wavelengths(self) -> np.ndarray | None:
        return self._wavelengths

    @property
    def reference_spectrum(self) -> np.ndarray | None:
        return self._reference_spectrum

    @property
    def calibration_concentrations(self) -> np.ndarray | None:
        return self._calibration_concentrations

    @property
    def calibration_responses(self) -> np.ndarray | None:
        return self._calibration_responses

    @property
    def has_calibration(self) -> bool:
        return (
            self._calibration_concentrations is not None
            and self._calibration_responses is not None
            and len(self._calibration_concentrations) >= 2
        )

    @property
    def has_model(self) -> bool:
        return self.model_path is not None and Path(self.model_path).exists()

    @property
    def has_reference(self) -> bool:
        return self._reference_spectrum is not None

    @property
    def session_dir(self) -> Path:
        return _SESSIONS_DIR / self.session_id

    # ---------------------------------------------------------------------------
    # Setters (update step flags automatically)
    # ---------------------------------------------------------------------------

    def set_wavelengths(self, wl: np.ndarray) -> None:
        self._wavelengths = np.asarray(wl, dtype=float)

    def set_reference(self, spectrum: np.ndarray, peak_nm: float | None = None) -> None:
        self._reference_spectrum = np.asarray(spectrum, dtype=float)
        self.reference_peak_nm = peak_nm
        self.step_complete["acquisition"] = True

    def set_calibration(
        self,
        concentrations: np.ndarray,
        responses: np.ndarray,
        preprocessing_cfg: dict | None = None,
    ) -> None:
        self._calibration_concentrations = np.asarray(concentrations, dtype=float)
        self._calibration_responses = np.asarray(responses, dtype=float)
        if preprocessing_cfg:
            self.preprocessing_config = preprocessing_cfg
        self.step_complete["preprocessing"] = True

    def set_model(
        self,
        model_path: str,
        model_type: str,
        performance: dict,
    ) -> None:
        self.model_path = model_path
        self.model_type = model_type
        self.performance = performance
        self.step_complete["training"] = True
        if performance.get("residual_diagnostics"):
            self.step_complete["validation"] = True

    def add_prediction(
        self,
        concentration: float,
        ci_lower: float,
        ci_upper: float,
        quality: str = "ok",
        spectrum_label: str = "",
    ) -> None:
        self.predictions.append({
            "timestamp": datetime.now().isoformat(),
            "spectrum_label": spectrum_label,
            "concentration_ppm": round(float(concentration), 6),
            "ci_lower_ppm": round(float(ci_lower), 6),
            "ci_upper_ppm": round(float(ci_upper), 6),
            "quality": quality,
        })

    # ---------------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------------

    def save(self) -> None:
        """Persist project state to ``output/sessions/{session_id}/``."""
        d = self.session_dir
        d.mkdir(parents=True, exist_ok=True)

        # JSON metadata
        meta = {
            "session_id": self.session_id,
            "gas_name": self.gas_name,
            "created_at": self.created_at,
            "notes": self.notes,
            "reference_peak_nm": self.reference_peak_nm,
            "preprocessing_config": self.preprocessing_config,
            "model_path": self.model_path,
            "model_type": self.model_type,
            "performance": self.performance,
            "predictions": self.predictions,
            "step_complete": self.step_complete,
        }
        with open(d / "project_state.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

        # Numpy arrays
        for name, arr in [
            ("wavelengths", self._wavelengths),
            ("reference_spectrum", self._reference_spectrum),
            ("calibration_concentrations", self._calibration_concentrations),
            ("calibration_responses", self._calibration_responses),
        ]:
            if arr is not None:
                np.save(d / f"{name}.npy", arr)

        log.info("ProjectStore saved to %s", d)

    @classmethod
    def load(cls, session_id: str) -> "ProjectStore":
        """Load a ProjectStore from a saved session directory.

        Parameters
        ----------
        session_id : str
            The session directory name (``YYYYMMDD_HHMMSS``).

        Returns
        -------
        ProjectStore
            Restored store with all arrays and metadata.

        Raises
        ------
        FileNotFoundError
            If ``project_state.json`` does not exist in the session directory.
        """
        d = _SESSIONS_DIR / session_id
        state_file = d / "project_state.json"
        if not state_file.exists():
            raise FileNotFoundError(f"No project_state.json found in {d}")

        with open(state_file, encoding="utf-8") as f:
            meta = json.load(f)

        store = cls(
            session_id=meta["session_id"],
            gas_name=meta.get("gas_name", "unknown"),
            created_at=meta.get("created_at", ""),
            notes=meta.get("notes", ""),
            reference_peak_nm=meta.get("reference_peak_nm"),
            preprocessing_config=meta.get("preprocessing_config", {}),
            model_path=meta.get("model_path"),
            model_type=meta.get("model_type", "GPR"),
            performance=meta.get("performance", {}),
            predictions=meta.get("predictions", []),
            step_complete=meta.get("step_complete", {
                "acquisition": False, "preprocessing": False,
                "training": False, "validation": False,
            }),
        )

        # Load numpy arrays
        for name, attr in [
            ("wavelengths", "_wavelengths"),
            ("reference_spectrum", "_reference_spectrum"),
            ("calibration_concentrations", "_calibration_concentrations"),
            ("calibration_responses", "_calibration_responses"),
        ]:
            npy_file = d / f"{name}.npy"
            if npy_file.exists():
                setattr(store, attr, np.load(npy_file))

        log.info("ProjectStore loaded from %s", d)
        return store

    def summary_dict(self) -> dict[str, Any]:
        """Return a compact summary for display in session browser."""
        perf = self.performance or {}
        return {
            "session_id": self.session_id,
            "gas_name": self.gas_name,
            "created_at": self.created_at[:16].replace("T", " "),
            "n_cal_points": len(self._calibration_concentrations)
            if self._calibration_concentrations is not None else 0,
            "n_predictions": len(self.predictions),
            "model_type": self.model_type,
            "r_squared": perf.get("r_squared"),
            "lod_ppm": perf.get("lod_ppm"),
            "steps_done": sum(self.step_complete.values()),
        }


# ---------------------------------------------------------------------------
# Streamlit helpers
# ---------------------------------------------------------------------------

def get_project() -> ProjectStore:
    """Return the current ProjectStore from session state, creating if needed."""
    if "project" not in st.session_state or not isinstance(
        st.session_state["project"], ProjectStore
    ):
        st.session_state["project"] = ProjectStore()
    return st.session_state["project"]


def init_project(gas_name: str = "unknown") -> ProjectStore:
    """Create a fresh ProjectStore and store it in session state."""
    proj = ProjectStore(gas_name=gas_name)
    st.session_state["project"] = proj
    return proj


def set_project(proj: ProjectStore) -> None:
    """Replace the current project with a loaded one."""
    st.session_state["project"] = proj


# ---------------------------------------------------------------------------
# Session browser helpers
# ---------------------------------------------------------------------------

def list_saved_sessions() -> list[dict[str, Any]]:
    """Scan output/sessions/ and return a list of session summaries.

    Returns
    -------
    list[dict]
        Each dict has keys from :meth:`ProjectStore.summary_dict` plus
        ``has_project_state`` (bool).  Sorted newest first.
    """
    sessions: list[dict] = []
    if not _SESSIONS_DIR.exists():
        return sessions

    for d in sorted(_SESSIONS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        state_file = d / "project_state.json"
        if state_file.exists():
            try:
                with open(state_file, encoding="utf-8") as f:
                    meta = json.load(f)
                entry = {
                    "session_id": meta.get("session_id", d.name),
                    "gas_name": meta.get("gas_name", "?"),
                    "created_at": meta.get("created_at", "")[:16].replace("T", " "),
                    "model_type": meta.get("model_type", "—"),
                    "n_predictions": len(meta.get("predictions", [])),
                    "steps_done": sum(meta.get("step_complete", {}).values()),
                    "r_squared": (meta.get("performance") or {}).get("r_squared"),
                    "lod_ppm": (meta.get("performance") or {}).get("lod_ppm"),
                    "has_project_state": True,
                    "path": str(d),
                }
                sessions.append(entry)
            except Exception:
                pass
        else:
            # Legacy session — no project_state.json but may have pipeline_results.csv
            results_csv = d / "pipeline_results.csv"
            meta_json = d / "session_meta.json"
            if results_csv.exists() or meta_json.exists():
                gas = "?"
                if meta_json.exists():
                    try:
                        with open(meta_json, encoding="utf-8") as f:
                            m = json.load(f)
                        gas = m.get("gas_label", m.get("gas_name", "?"))
                    except Exception:
                        pass
                sessions.append({
                    "session_id": d.name,
                    "gas_name": gas,
                    "created_at": d.name[:8] + " " + d.name[9:11] + ":" + d.name[11:13]
                    if len(d.name) >= 15 else d.name,
                    "model_type": "—",
                    "n_predictions": 0,
                    "steps_done": 0,
                    "r_squared": None,
                    "lod_ppm": None,
                    "has_project_state": False,
                    "path": str(d),
                })

    return sessions


def render_session_browser() -> ProjectStore | None:
    """Render the session browser sidebar widget.

    Returns
    -------
    ProjectStore | None
        The loaded store if the user clicked "Load", otherwise None.
    """
    sessions = list_saved_sessions()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 Session Browser")

    if not sessions:
        st.sidebar.caption("No saved sessions found in `output/sessions/`.")
        return None

    # Current session badge
    proj = get_project()
    st.sidebar.caption(
        f"**Active:** {proj.gas_name} · {proj.session_id[:8]}…  "
        f"({proj.step_complete.get('training', False) and 'trained' or 'in progress'})"
    )

    # Compact list
    options = [
        f"{s['created_at']}  {s['gas_name']}  "
        f"[{s['model_type']}  R²={s['r_squared']:.3f}]"
        if s.get("r_squared") else
        f"{s['created_at']}  {s['gas_name']}  [{s['steps_done']}/4 steps]"
        for s in sessions
    ]

    selected_idx = st.sidebar.selectbox(
        "Load past session",
        range(len(options)),
        format_func=lambda i: options[i],
        key="sb_selected_session",
        index=None,
        placeholder="Choose a session to load…",
    )

    if selected_idx is not None:
        chosen = sessions[selected_idx]
        col1, col2 = st.sidebar.columns(2)

        with col1:
            if st.button("Load", key="sb_load_btn", use_container_width=True):
                if chosen["has_project_state"]:
                    try:
                        loaded = ProjectStore.load(chosen["session_id"])
                        set_project(loaded)
                        st.sidebar.success(f"Loaded: {chosen['session_id']}")
                        st.rerun()
                        return loaded
                    except Exception as exc:
                        st.sidebar.error(f"Load failed: {exc}")
                else:
                    st.sidebar.warning(
                        "Legacy session — no project_state.json. "
                        "Load manually via Batch Analysis tab."
                    )

        with col2:
            r2_str = f"R²={chosen['r_squared']:.3f}" if chosen.get("r_squared") else "no model"
            lod_str = f"LOD={chosen['lod_ppm']:.3g} ppm" if chosen.get("lod_ppm") else ""
            st.sidebar.caption(f"{r2_str}  {lod_str}")

    # New session button
    if st.sidebar.button("＋ New Session", key="sb_new_session", use_container_width=True):
        new_proj = ProjectStore()
        set_project(new_proj)
        st.sidebar.success("Started new session.")
        st.rerun()

    return None
