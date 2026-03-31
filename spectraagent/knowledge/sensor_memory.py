"""
spectraagent.knowledge.sensor_memory
======================================
Data-driven sensor knowledge base.

SensorMemory accumulates everything the platform has *observed* about a
specific sensor over its lifetime — sensitivity, achieved LOD, drift
patterns, failure events, model performance history.  It is populated
automatically at the end of every session and provides the primary context
for all agent prompts.

This is the key architectural decision that makes SpectraAgent generic: the
knowledge base is NOT hardcoded literature values.  It is the sensor's own
experimental history.  Agents reason about "what this sensor actually does"
rather than "what sensors of this type are supposed to do."

On-disk format
--------------
JSON file at ``memory_dir / "sensor_memory.json"``.  Human-readable, version-
controlled, and directly inspectable by researchers.

Schema version 2 adds per-analyte calibration history with model performance
tracking.  Older files are migrated automatically on first load.

Usage
-----
::

    from spectraagent.knowledge.sensor_memory import SensorMemory

    mem = SensorMemory(memory_dir=Path("output/memory"))

    # Record session outcome (called by session lifecycle at stop):
    mem.record_session(session_result)

    # Read for agent context:
    summary = mem.get_analyte_summary("Ethanol")
    health  = mem.get_sensor_health_summary()
"""
from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

_SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class CalibrationObservation:
    """One observed calibration outcome from a completed session."""
    session_id: str
    timestamp_utc: str
    analyte: str
    sensitivity_nm_per_ppm: Optional[float]   # from linear-range fit
    lod_ppm: Optional[float]
    loq_ppm: Optional[float]
    r_squared: Optional[float]
    rmse_ppm: Optional[float]
    calibration_model: str          # "langmuir", "linear", "freundlich", etc.
    n_calibration_points: int
    reference_peak_nm: Optional[float]
    conformal_coverage: Optional[float]  # empirical coverage of conformal PI
    notes: str = ""


@dataclass
class FailureEvent:
    """One observed failure / anomaly event recorded during a session."""
    session_id: str
    timestamp_utc: str
    event_type: str           # "thermal_drift", "saturation", "low_snr", etc.
    severity: str             # "warning", "critical"
    drift_rate_nm_per_min: Optional[float] = None
    max_intensity_counts: Optional[float] = None
    snr_at_event: Optional[float] = None
    resolution: str = ""      # what the operator did (if recorded)
    agent_diagnosis: str = "" # Claude's diagnosis text (if available)


@dataclass
class DriftObservation:
    """Per-session baseline drift statistics."""
    session_id: str
    timestamp_utc: str
    max_drift_rate_nm_per_min: float
    mean_drift_rate_nm_per_min: float
    equilibration_time_min: float    # frames until drift fell below threshold
    opening_peak_nm: float           # peak wavelength at session start
    closing_peak_nm: float           # peak wavelength at session end


@dataclass
class ModelPerformanceRecord:
    """ML model evaluation result persisted after training or inference."""
    session_id: str
    timestamp_utc: str
    model_type: str          # "gpr", "cnn_classifier", "pls", etc.
    analyte: str
    val_r_squared: Optional[float] = None
    val_rmse_ppm: Optional[float] = None
    val_mae_ppm: Optional[float] = None
    n_training_samples: int = 0
    n_val_samples: int = 0
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


# ---------------------------------------------------------------------------
# Main store
# ---------------------------------------------------------------------------

class SensorMemory:
    """Persistent, append-only knowledge store for one sensor unit.

    Parameters
    ----------
    memory_dir:
        Directory where ``sensor_memory.json`` is stored.  Created if absent.
    sensor_id:
        Optional sensor identifier (e.g. serial number).  Written to the
        file header so logs can be traced back to the instrument.
    """

    def __init__(self, memory_dir: Path, sensor_id: str = "unknown") -> None:
        self._path = Path(memory_dir) / "sensor_memory.json"
        self._sensor_id = sensor_id
        self._data: dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _empty(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "sensor_id": self._sensor_id,
            "created_utc": _utcnow(),
            "updated_utc": _utcnow(),
            "analytes": {},
            "failure_events": [],
            "drift_observations": [],
            "model_performance": [],
            "session_log": [],
        }

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return self._empty()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return self._migrate(raw)
        except Exception as exc:
            log.warning("SensorMemory: could not load %s (%s) — starting fresh", self._path, exc)
            return self._empty()

    def _migrate(self, raw: dict[str, Any]) -> dict[str, Any]:
        v = raw.get("schema_version", 1)
        if v < 2:
            raw.setdefault("model_performance", [])
            raw.setdefault("drift_observations", [])
            raw["schema_version"] = 2
        return raw

    def save(self) -> None:
        """Flush in-memory state to disk (atomic write via temp file)."""
        self._data["updated_utc"] = _utcnow()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, default=str), encoding="utf-8")
        tmp.replace(self._path)

    # ------------------------------------------------------------------
    # Writing — called at session end
    # ------------------------------------------------------------------

    def record_calibration(self, obs: CalibrationObservation) -> None:
        """Append one calibration observation for an analyte."""
        key = obs.analyte.lower()
        bucket = self._data["analytes"].setdefault(key, {
            "canonical_name": obs.analyte,
            "calibration_history": [],
            "session_count": 0,
        })
        bucket["calibration_history"].append(asdict(obs))
        bucket["session_count"] = len(bucket["calibration_history"])
        self.save()

    def record_failure(self, evt: FailureEvent) -> None:
        """Append a failure event."""
        self._data["failure_events"].append(asdict(evt))
        self.save()

    def record_drift(self, obs: DriftObservation) -> None:
        """Append per-session drift statistics."""
        self._data["drift_observations"].append(asdict(obs))
        self.save()

    def record_model_performance(self, rec: ModelPerformanceRecord) -> None:
        """Append ML model evaluation result."""
        self._data["model_performance"].append(asdict(rec))
        self.save()

    def record_session(self, session_id: str, analyte: str, frame_count: int,
                       stopped_at: str, notes: str = "") -> None:
        """Lightweight session log entry (called regardless of calibration)."""
        self._data["session_log"].append({
            "session_id": session_id,
            "analyte": analyte,
            "frame_count": frame_count,
            "stopped_at": stopped_at,
            "notes": notes,
        })
        self.save()

    # ------------------------------------------------------------------
    # Reading — called by context builders
    # ------------------------------------------------------------------

    def get_analyte_summary(self, analyte: str) -> Optional[dict[str, Any]]:
        """Return aggregated calibration statistics for one analyte.

        Returns None if no calibration history exists.
        """
        key = analyte.lower()
        bucket = self._data["analytes"].get(key)
        if not bucket or not bucket.get("calibration_history"):
            return None

        history = bucket["calibration_history"]
        n = len(history)

        def _valid(field: str) -> list[float]:
            return [h[field] for h in history if h.get(field) is not None]

        lods = _valid("lod_ppm")
        loqs = _valid("loq_ppm")
        sensitivities = _valid("sensitivity_nm_per_ppm")
        r2s = _valid("r_squared")
        coverages = _valid("conformal_coverage")

        models = [h["calibration_model"] for h in history if h.get("calibration_model")]
        model_counts: dict[str, int] = {}
        for m in models:
            model_counts[m] = model_counts.get(m, 0) + 1
        best_model = max(model_counts, key=lambda k: model_counts[k]) if model_counts else "unknown"

        return {
            "analyte": bucket["canonical_name"],
            "n_sessions": n,
            "lod_ppm": _stat_summary(lods),
            "loq_ppm": _stat_summary(loqs),
            "sensitivity_nm_per_ppm": _stat_summary(sensitivities),
            "r_squared": _stat_summary(r2s),
            "conformal_coverage": _stat_summary(coverages),
            "dominant_model": best_model,
            "model_counts": model_counts,
            "most_recent": history[-1],
            "trend": _compute_trend(lods),   # "improving", "stable", "degrading"
            "reference_peaks": _valid("reference_peak_nm"),
        }

    def get_sensor_health_summary(self) -> dict[str, Any]:
        """Return cross-session sensor health indicators."""
        drifts = self._data.get("drift_observations", [])
        failures = self._data.get("failure_events", [])
        sessions = self._data.get("session_log", [])
        model_perf = self._data.get("model_performance", [])

        failure_counts: dict[str, int] = {}
        for f in failures:
            t = f.get("event_type", "unknown")
            failure_counts[t] = failure_counts.get(t, 0) + 1

        drift_rates = [d["max_drift_rate_nm_per_min"] for d in drifts if d.get("max_drift_rate_nm_per_min") is not None]
        eq_times = [d["equilibration_time_min"] for d in drifts if d.get("equilibration_time_min") is not None]

        return {
            "total_sessions": len(sessions),
            "total_failure_events": len(failures),
            "failure_type_counts": failure_counts,
            "drift_nm_per_min": _stat_summary(drift_rates),
            "equilibration_time_min": _stat_summary(eq_times),
            "sensor_id": self._data.get("sensor_id", "unknown"),
            "first_seen": sessions[0]["stopped_at"] if sessions else None,
            "last_seen": sessions[-1]["stopped_at"] if sessions else None,
            "model_count": len(model_perf),
        }

    def get_all_analytes(self) -> list[str]:
        """Return list of analytes this sensor has been characterised for."""
        return [v["canonical_name"] for v in self._data["analytes"].values()]

    def get_recent_failures(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the n most recent failure events."""
        return self._data["failure_events"][-n:]

    def get_calibration_trend_text(self, analyte: str) -> str:
        """One-line text description of calibration trend for agent prompts."""
        summary = self.get_analyte_summary(analyte)
        if not summary:
            return f"No calibration history found for {analyte}."

        n = summary["n_sessions"]
        lod = summary["lod_ppm"]
        trend = summary["trend"]
        model = summary["dominant_model"]

        parts = [f"{n} session(s) for {analyte}."]
        if lod and lod.get("last") is not None:
            parts.append(f"Most recent LOD: {lod['last']:.4f} ppm.")
        if lod and lod.get("mean") is not None and n >= 2:
            parts.append(f"Mean LOD over {n} sessions: {lod['mean']:.4f} ± {lod.get('std', 0):.4f} ppm.")
        if trend != "insufficient_data":
            parts.append(f"LOD trend: {trend}.")
        parts.append(f"Dominant calibration model: {model}.")
        return " ".join(parts)

    def format_for_agent_prompt(self, analyte: Optional[str] = None) -> str:
        """Return a comprehensive Markdown summary for injection into agent prompts.

        Parameters
        ----------
        analyte:
            If provided, include per-analyte calibration history.
            If None, include only sensor health summary.
        """
        lines: list[str] = ["## Sensor Memory (observed from real sessions)\n"]

        health = self.get_sensor_health_summary()
        lines.append(f"**Sensor ID**: {health['sensor_id']}  ")
        lines.append(f"**Total sessions recorded**: {health['total_sessions']}  ")
        if health["last_seen"]:
            lines.append(f"**Last session**: {health['last_seen']}  ")
        lines.append("")

        drift = health.get("drift_nm_per_min", {})
        if drift and drift.get("n", 0) > 0:
            lines.append(
                f"**Observed drift** (across {drift['n']} sessions): "
                f"max avg {drift.get('mean', '?'):.3f} ± {drift.get('std', 0):.3f} nm/min  "
            )
        if health["total_failure_events"] > 0:
            lines.append(f"**Failure events recorded**: {health['total_failure_events']}  ")
            for etype, count in sorted(health["failure_type_counts"].items(), key=lambda x: -x[1]):
                lines.append(f"  - {etype}: {count}×  ")
        lines.append("")

        if analyte:
            summary = self.get_analyte_summary(analyte)
            if summary:
                lines.append(f"### {analyte} calibration history ({summary['n_sessions']} session(s))\n")
                lod = summary["lod_ppm"]
                loq = summary["loq_ppm"]
                sens = summary["sensitivity_nm_per_ppm"]
                r2 = summary["r_squared"]
                if lod and lod.get("n", 0) > 0:
                    lines.append(f"- **LOD**: last={lod.get('last', '?'):.4f} ppm, "
                                  f"mean={lod.get('mean', '?'):.4f} ± {lod.get('std', 0):.4f} ppm, "
                                  f"trend={summary['trend']}  ")
                if loq and loq.get("n", 0) > 0:
                    lines.append(f"- **LOQ**: last={loq.get('last', '?'):.4f} ppm, "
                                  f"mean={loq.get('mean', '?'):.4f} ppm  ")
                if sens and sens.get("n", 0) > 0:
                    lines.append(f"- **Sensitivity**: last={sens.get('last', '?'):.2f} nm/ppm, "
                                  f"mean={sens.get('mean', '?'):.2f} nm/ppm  ")
                if r2 and r2.get("n", 0) > 0:
                    lines.append(f"- **Calibration R²**: last={r2.get('last', '?'):.4f}, "
                                  f"mean={r2.get('mean', '?'):.4f}  ")
                lines.append(f"- **Dominant model**: {summary['dominant_model']}  ")
            else:
                lines.append(f"No calibration history found for {analyte}.")

        known = self.get_all_analytes()
        if known:
            lines.append(f"\n**Analytes characterised so far**: {', '.join(known)}  ")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _stat_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    n = len(values)
    mn = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    return {
        "n": n,
        "mean": round(mn, 6),
        "std": round(sd, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "last": round(values[-1], 6),
    }


def _compute_trend(values: list[float]) -> str:
    """Classify trend as improving/stable/degrading based on last 5 values."""
    if len(values) < 3:
        return "insufficient_data"
    recent = values[-5:]
    n = len(recent)
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(recent) / n
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, recent))
    den = sum((xi - x_mean) ** 2 for xi in x)
    if den == 0:
        return "stable"
    slope = num / den
    # Relative slope: normalize by mean value
    rel_slope = slope / y_mean if y_mean != 0 else 0.0
    if rel_slope < -0.05:
        return "improving"   # LOD decreasing = better
    if rel_slope > 0.05:
        return "degrading"   # LOD increasing = worse
    return "stable"
