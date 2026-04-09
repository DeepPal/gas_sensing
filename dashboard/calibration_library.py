"""
dashboard.calibration_library
===============================
Persistent calibration library for the SpectraAgent Streamlit dashboard.

Stores validated calibration models so researchers can reuse them across
measurement days without re-running the full calibration pipeline.

**Structure on disk** (``output/calibration_library/``)::

    output/calibration_library/
        index.json                        — metadata for all entries
        {id}/
            model.pkl                     — serialised sklearn model (GPR/PLS/OLS)
            calibration_concentrations.npy
            calibration_responses.npy
            performance.json              — LOD, LOQ, R², residual diagnostics

**Workflow**:
1. Researcher runs calibration in Tab 1 (Guided Calibration).
2. When calibration passes QC (R² ≥ 0.99, residuals OK), they click
   "Save to Calibration Library".
3. Library stores the model + metadata under a unique ID.
4. In Tab 2 (Predict Unknown), researcher clicks "Load from library",
   picks their calibration, and the ProjectStore is populated.

Public API
----------
- ``CalibrationLibrary``         — the library manager class
- ``get_library()``              — get or create the singleton library
- ``render_library_sidebar()``   — Streamlit sidebar widget
- ``render_save_to_library()``   — "Save to library" button widget
- ``render_load_from_library()`` — "Load from library" select + load widget
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LIB_DIR = _REPO_ROOT / "output" / "calibration_library"


@dataclass
class CalibrationEntry:
    """Metadata record for one calibration saved in the library.

    Attributes
    ----------
    entry_id : str
        Unique identifier (``YYYYMMDD_HHMMSS_<gas>``).
    gas_name : str
        Analyte label.
    model_type : str
        Fitting model used (GPR / PLS / LinearOLS / Langmuir).
    created_at : str
        ISO-8601 timestamp when this calibration was saved.
    r_squared : float | None
    lod_ppm : float | None
    loq_ppm : float | None
    sensitivity : float | None
    n_cal_points : int
    concentration_range : tuple[float, float] | None
        (min_ppm, max_ppm) of the calibration series.
    session_id : str
        The session this calibration came from.
    notes : str
        Researcher annotations.
    model_path : str
        Relative path to the model file.
    performance : dict
        Full performance dict from ``sensor_performance_summary()``.
    residual_pass : bool | None
        True if all residual diagnostics passed.
    """

    entry_id: str
    gas_name: str
    model_type: str
    created_at: str
    r_squared: float | None
    lod_ppm: float | None
    loq_ppm: float | None
    sensitivity: float | None
    n_cal_points: int
    concentration_range: list[float] | None   # [min, max]
    session_id: str
    notes: str = ""
    model_path: str = ""
    performance: dict = field(default_factory=dict)
    residual_pass: bool | None = None

    def display_label(self) -> str:
        r2_str = f"R²={self.r_squared:.4f}" if self.r_squared is not None else ""
        lod_str = f"LOD={self.lod_ppm:.3g} ppm" if self.lod_ppm is not None else ""
        rng_str = (
            f"[{self.concentration_range[0]:.3g}–{self.concentration_range[1]:.3g} ppm]"
            if self.concentration_range else ""
        )
        date_str = self.created_at[:10]
        return f"{date_str}  {self.gas_name}  {self.model_type}  {r2_str}  {lod_str}  {rng_str}"


class CalibrationLibrary:
    """Manages the persistent calibration library on disk.

    All methods are safe to call from any Streamlit tab — they operate on
    the file system and do not touch ``st.session_state``.
    """

    def __init__(self, lib_dir: Path = _LIB_DIR) -> None:
        self._dir = lib_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.json"

    # ------------------------------------------------------------------
    # Index I/O
    # ------------------------------------------------------------------

    def _load_index(self) -> list[dict[str, Any]]:
        if not self._index_path.exists():
            return []
        try:
            with open(self._index_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            log.warning("Could not read calibration library index: %s", exc)
            return []

    def _save_index(self, entries: list[dict[str, Any]]) -> None:
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_entries(self) -> list[CalibrationEntry]:
        """Return all library entries, newest first."""
        raw = self._load_index()
        entries = []
        for d in raw:
            try:
                entries.append(CalibrationEntry(**d))
            except Exception:
                pass
        return list(reversed(entries))

    def save(
        self,
        gas_name: str,
        model_type: str,
        model_obj: Any,
        calibration_concentrations: np.ndarray,
        calibration_responses: np.ndarray,
        performance: dict,
        session_id: str = "",
        notes: str = "",
    ) -> CalibrationEntry:
        """Save a calibration to the library.

        Parameters
        ----------
        gas_name
        model_type : str
        model_obj : Any
            Sklearn-compatible model with ``.predict()`` (or a dict for Langmuir).
        calibration_concentrations, calibration_responses : np.ndarray
        performance : dict
            Output of ``sensor_performance_summary()``.
        session_id, notes : str

        Returns
        -------
        CalibrationEntry
            The newly saved entry.
        """
        import pickle
        import sys
        import importlib.metadata as _meta

        entry_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{gas_name.replace(' ', '_')}"
        entry_dir = self._dir / entry_id
        entry_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = entry_dir / "model.pkl"
        with open(model_path, "wb") as fh:
            pickle.dump(model_obj, fh)

        # Save version manifest alongside the pickle so load_model() can
        # detect sklearn / Python version mismatches before returning a
        # potentially corrupt model. Mismatch ≠ guaranteed failure, but
        # unpinned sklearn upgrades (especially 1.x→1.x minor bumps) have
        # silently changed GPR hyperparameter shapes in the past.
        def _pkg_version(name: str) -> str:
            try:
                return _meta.version(name)
            except Exception:
                return "unknown"

        versions = {
            "python": sys.version,
            "sklearn": _pkg_version("scikit-learn"),
            "numpy": _pkg_version("numpy"),
            "scipy": _pkg_version("scipy"),
            "joblib": _pkg_version("joblib"),
            "saved_at": datetime.now().isoformat(),
        }
        with open(entry_dir / "_versions.json", "w", encoding="utf-8") as fh:
            json.dump(versions, fh, indent=2)

        # Save arrays
        np.save(entry_dir / "concentrations.npy", calibration_concentrations)
        np.save(entry_dir / "responses.npy", calibration_responses)

        # Save performance
        with open(entry_dir / "performance.json", "w", encoding="utf-8") as fh:
            json.dump(performance, fh, indent=2, default=str)

        rdiag = performance.get("residual_diagnostics") or {}
        c = calibration_concentrations
        entry = CalibrationEntry(
            entry_id=entry_id,
            gas_name=gas_name,
            model_type=model_type,
            created_at=datetime.now().isoformat(),
            r_squared=performance.get("r_squared"),
            lod_ppm=performance.get("lod_ppm"),
            loq_ppm=performance.get("loq_ppm"),
            sensitivity=performance.get("sensitivity"),
            n_cal_points=int(len(calibration_concentrations)),
            concentration_range=[float(c.min()), float(c.max())] if len(c) > 0 else None,
            session_id=session_id,
            notes=notes,
            model_path=str(model_path.relative_to(_REPO_ROOT)),
            performance=performance,
            residual_pass=bool(rdiag.get("overall_pass")) if rdiag else None,
        )

        # Append to index
        raw = self._load_index()
        raw.append(asdict(entry))
        self._save_index(raw)

        log.info("CalibrationLibrary: saved %s (%s) as %s", gas_name, model_type, entry_id)
        return entry

    def load_model(self, entry_id: str) -> tuple[Any, list[str]]:
        """Load the serialised model for an entry.

        Returns
        -------
        tuple (model_object, version_warnings)
            ``version_warnings`` is a list of human-readable strings describing
            any package version mismatches between save-time and load-time.
            An empty list means the environment matches. Warnings do NOT prevent
            loading — the caller decides whether to block or alert the user.
        """
        import pickle
        import sys
        import importlib.metadata as _meta

        model_path = self._dir / entry_id / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"No model.pkl for entry {entry_id}")

        # Check version manifest before unpickling
        version_warnings: list[str] = []
        versions_path = self._dir / entry_id / "_versions.json"
        if versions_path.exists():
            try:
                with open(versions_path, encoding="utf-8") as fh:
                    saved_versions: dict = json.load(fh)
                checks = [
                    ("sklearn", "scikit-learn"),
                    ("numpy", "numpy"),
                    ("scipy", "scipy"),
                ]
                for key, pkg in checks:
                    saved_v = saved_versions.get(key, "unknown")
                    try:
                        current_v = _meta.version(pkg)
                    except Exception:
                        current_v = "unknown"
                    if saved_v != "unknown" and current_v != "unknown" and saved_v != current_v:
                        version_warnings.append(
                            f"{pkg}: saved={saved_v}, current={current_v} — "
                            "model may behave differently. Re-save to update."
                        )
                py_saved = saved_versions.get("python", "")[:6]
                py_now = sys.version[:6]
                if py_saved and py_saved != py_now:
                    version_warnings.append(
                        f"Python: saved={py_saved}, current={py_now} — "
                        "pickle compatibility not guaranteed across major Python versions."
                    )
            except Exception as _ve:
                version_warnings.append(f"Could not read version manifest: {_ve}")
        else:
            version_warnings.append(
                "No _versions.json found — this model was saved before version pinning "
                "was added. Re-save to enable mismatch detection."
            )

        if version_warnings:
            for w in version_warnings:
                log.warning("CalibrationLibrary version mismatch [%s]: %s", entry_id, w)

        with open(model_path, "rb") as fh:
            model = pickle.load(fh)

        return model, version_warnings

    def load_arrays(self, entry_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Load (concentrations, responses) for an entry."""
        d = self._dir / entry_id
        return np.load(d / "concentrations.npy"), np.load(d / "responses.npy")

    def delete(self, entry_id: str) -> None:
        """Delete an entry from the library."""
        entry_dir = self._dir / entry_id
        if entry_dir.exists():
            shutil.rmtree(entry_dir)
        raw = [e for e in self._load_index() if e.get("entry_id") != entry_id]
        self._save_index(raw)
        log.info("CalibrationLibrary: deleted %s", entry_id)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_library_instance: CalibrationLibrary | None = None


def get_library() -> CalibrationLibrary:
    """Return the shared CalibrationLibrary instance (created once per process)."""
    global _library_instance
    if _library_instance is None:
        _library_instance = CalibrationLibrary()
    return _library_instance


# ---------------------------------------------------------------------------
# Streamlit UI helpers
# ---------------------------------------------------------------------------

def render_save_to_library(
    gas_name: str,
    model_type: str,
    model_obj: Any,
    concentrations: np.ndarray,
    responses: np.ndarray,
    performance: dict,
    session_id: str = "",
) -> bool:
    """Render a "Save to Calibration Library" button + notes field.

    Returns True if the calibration was saved.
    """
    import streamlit as st

    with st.expander("💾 Save to Calibration Library", expanded=False):
        r2 = performance.get("r_squared") or 0
        lod = performance.get("lod_ppm")
        rdiag = performance.get("residual_diagnostics") or {}
        diag_pass = rdiag.get("overall_pass", True)

        # Quality badge
        if r2 >= 0.99 and diag_pass:
            st.success(f"R²={r2:.4f} — calibration quality: PUBLICATION GRADE")
        elif r2 >= 0.95:
            st.warning(f"R²={r2:.4f} — acceptable for screening; target R²≥0.99 for publication")
        else:
            st.error(f"R²={r2:.4f} — low quality; retrain or collect more calibration points")

        notes = st.text_area(
            "Researcher notes (optional)",
            placeholder="e.g. Chip batch #3, day 2, fresh reference, lab temp 23°C",
            key="cal_lib_notes",
        )

        if st.button("💾 Save to Library", key="cal_lib_save_btn", type="primary"):
            try:
                lib = get_library()
                entry = lib.save(
                    gas_name=gas_name,
                    model_type=model_type,
                    model_obj=model_obj,
                    calibration_concentrations=concentrations,
                    calibration_responses=responses,
                    performance=performance,
                    session_id=session_id,
                    notes=notes,
                )
                st.success(
                    f"Saved to library as **{entry.entry_id}**  "
                    f"(R²={r2:.4f}, LOD={lod:.3g} ppm)" if lod else
                    f"Saved to library as **{entry.entry_id}**"
                )
                return True
            except Exception as exc:
                st.error(f"Save failed: {exc}")

    return False


def render_load_from_library() -> tuple[Any, np.ndarray, np.ndarray, dict] | None:
    """Render a "Load from Library" selector.

    Returns
    -------
    (model, concentrations, responses, performance) or None if nothing selected.
    """
    import streamlit as st

    lib = get_library()
    entries = lib.list_entries()

    if not entries:
        st.info(
            "Calibration library is empty. "
            "Train a calibration in the **Guided Calibration** tab and save it."
        )
        return None

    opts = [e.display_label() for e in entries]
    sel = st.selectbox(
        "Choose calibration",
        range(len(opts)),
        format_func=lambda i: opts[i],
        key="cal_lib_select",
    )
    chosen = entries[sel]

    # Show quick stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Analyte", chosen.gas_name)
    col2.metric("R²", f"{chosen.r_squared:.4f}" if chosen.r_squared else "—")
    col3.metric("LOD", f"{chosen.lod_ppm:.3g} ppm" if chosen.lod_ppm else "—")
    col4.metric("n points", chosen.n_cal_points)

    if chosen.notes:
        st.caption(f"Notes: {chosen.notes}")

    diag_icon = "" if chosen.residual_pass is None else ("✅" if chosen.residual_pass else "⚠️")
    st.caption(
        f"Model: {chosen.model_type}  |  Saved: {chosen.created_at[:16]}  "
        f"|  Residual diagnostics: {diag_icon + ('PASS' if chosen.residual_pass else 'FAIL') if chosen.residual_pass is not None else 'not run'}"
    )

    if st.button("Load this calibration", key="cal_lib_load_btn", type="primary"):
        try:
            model, version_warnings = lib.load_model(chosen.entry_id)
            concs, resps = lib.load_arrays(chosen.entry_id)
            # Show version mismatch warnings before returning so the researcher
            # can decide whether to retrain before making predictions.
            if version_warnings:
                for _vw in version_warnings:
                    st.warning(f"⚠️ Version mismatch: {_vw}")
                st.caption(
                    "These warnings do not prevent loading, but may indicate the model "
                    "behaves differently than when saved. Re-save after upgrading packages."
                )
            else:
                st.success(f"Loaded: {chosen.gas_name} ({chosen.model_type}), {chosen.n_cal_points} cal points")
            # Merge entry metadata into the performance dict for downstream consumers
            perf = dict(chosen.performance)
            perf.setdefault("model_type", chosen.model_type)
            perf.setdefault("gas_name", chosen.gas_name)
            return model, concs, resps, perf
        except Exception as exc:
            st.error(f"Load failed: {exc}")

    return None
