"""
spectraagent.webapp.agents.sensor_health
=========================================
SensorHealthAgent — cross-session multi-metric sensor lifecycle tracker.

Listens to the AgentBus and builds a comprehensive picture of sensor health
from two complementary event streams:

  model_selected  — emitted by CalibrationAgent after each calibration fit.
                    Carries: best_model, r_squared, sensitivity_nm_per_ppm,
                    n_points, lod_ppm, loq_ppm, rmse_ppm.

  session_complete — emitted by SessionAnalyzer at session stop.
                    Carries: lod_ppm, calibration_r2, mean_snr,
                    drift_rate_nm_per_frame, calibration_rmse_ppm, etc.

On session_complete the agent:
  1. Merges both event payloads into a single session snapshot.
  2. Looks up the sensor's calibration history from SensorMemory.
  3. Computes a 5-metric health scorecard (0–100 scale each).
  4. Emits a ``sensor_health_report`` event with the full scorecard.
  5. Emits ``recalibration_required`` if any critical threshold is crossed.
  6. Optionally calls Claude (when ``auto_explain=True``) for a plain-language
     narrative that is grounded in the sensor's actual history — not generic
     advice about "sensors of this type".

Health scorecard
----------------
Each dimension is scored 0–100 (100 = best ever observed for this sensor):

  LOD score        — How close is today's LOD to the best LOD this sensor
                     has ever achieved?  LOD_best / LOD_current × 100.
                     Score < 67 → recalibration trigger.

  Sensitivity score — How close is today's sensitivity to the sensor's best?
                     sens_current / sens_best × 100 (abs values).
                     Score < 70 → recalibration trigger.

  R² score         — Calibration linearity quality.
                     (R² − 0.90) / (1.00 − 0.90) × 100, clamped 0–100.
                     Score < 50 (R² < 0.95) for 2 consecutive sessions
                     → recalibration trigger.

  Drift score      — How stable is the baseline drift vs. sensor typical?
                     drift_typical / drift_current × 100.
                     Score < 50 (drift > 2× typical) → warning.

  SNR score        — Signal quality relative to best observed.
                     snr_current / snr_best × 100.
                     Score < 50 → warning (data quality at risk).

Overall health = weighted mean: LOD×30% + Sensitivity×30% + R²×25% +
                                Drift×10% + SNR×5%.

Recalibration is required when:
  - Any single trigger threshold is crossed, OR
  - Overall health < 55 (consistent multi-metric degradation).

Why LOD alone is insufficient
------------------------------
A sensor with surface fouling shows FWHM broadening (tracked via ΔFWHM trends
in feature data) and reduced sensitivity **before** LOD degrades.  By the time
LOD degrades, the surface is significantly fouled.  Monitoring sensitivity and
R² gives 1–3 sessions of advance warning.  Monitoring FWHM (via the Lorentzian
feature channel) gives even earlier warning — see RESEARCH_HANDBOOK.md.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Optional

from spectraagent.webapp.agents.claude_agents import _BaseClaude, _DEFAULT_MODEL, _DEFAULT_TIMEOUT_S

log = logging.getLogger(__name__)

try:
    from spectraagent.knowledge.context_builders import build_sensor_physics_preamble
    _KB_AVAILABLE = True
except ImportError:
    _KB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Health thresholds
# ---------------------------------------------------------------------------

_LOD_SCORE_RECAL_THRESHOLD = 67      # LOD has degraded >50% from best
_SENS_SCORE_RECAL_THRESHOLD = 70     # Sensitivity dropped >30% from best
_R2_LOW_ABSOLUTE = 0.95              # R² below which we flag
_R2_CONSECUTIVE_SESSIONS = 2        # must be low for N sessions to trigger
_DRIFT_SCORE_WARNING = 50           # drift > 2× typical → warning
_SNR_SCORE_WARNING = 50             # SNR < 50% of best → data quality warning
_OVERALL_HEALTH_RECAL_THRESHOLD = 55

# Weights for overall health score
_WEIGHTS = {
    "lod": 0.30,
    "sensitivity": 0.30,
    "r2": 0.25,
    "drift": 0.10,
    "snr": 0.05,
}


class SensorHealthAgent(_BaseClaude):
    """Multi-metric sensor health monitor and recalibration advisor.

    Parameters
    ----------
    bus:
        AgentBus for emitting events.
    model:
        Claude model ID.
    timeout_s:
        Claude API timeout.
    memory:
        SensorMemory instance — required for historical comparison.
    get_analyte:
        Callable returning the current session's analyte name.
    auto_explain:
        When True, calls Claude when recalibration is required to generate
        a plain-language explanation grounded in the sensor's history.
    sensor_type:
        Sensor modality for physics preamble ("lspr", "spr", "optical", …).
    """

    source = "SensorHealthAgent"

    def __init__(
        self,
        bus: Any,
        model: str = _DEFAULT_MODEL,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        memory: Optional[Any] = None,
        get_analyte: Optional[Callable[[], Optional[str]]] = None,
        auto_explain: bool = False,
        sensor_type: str = "optical",
    ) -> None:
        super().__init__(bus, model, timeout_s, memory=memory)
        self._get_analyte = get_analyte
        self._auto_explain = auto_explain
        self._sensor_type = sensor_type
        # Accumulate calibration data from model_selected events so it is
        # available when session_complete fires later in the same session.
        self._pending_calibration: dict[str, Any] = {}
        # Track consecutive low-R² sessions for the R² trigger.
        self._consecutive_low_r2: int = 0

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def on_event(self, event: Any) -> None:
        """React to model_selected (accumulate) and session_complete (assess)."""
        if event.type == "model_selected":
            self._accumulate_calibration(event.data)
        elif event.type == "session_complete":
            await self._assess_session_health(event)

    def _accumulate_calibration(self, data: dict[str, Any]) -> None:
        """Cache fields from model_selected for use when session_complete fires."""
        self._pending_calibration.update({
            k: v for k, v in {
                "best_model": data.get("best_model"),
                "sensitivity_nm_per_ppm": data.get("sensitivity_nm_per_ppm"),
                "n_points": data.get("n_points"),
                "r2_from_cal": data.get("r_squared"),
                "lod_from_cal": data.get("lod_ppm"),
                "rmse_from_cal": data.get("rmse_ppm"),
            }.items()
            if v is not None
        })

    async def _assess_session_health(self, event: Any) -> None:
        """Compute health scorecard and emit events."""
        data = event.data

        # Merge session_complete data with accumulated calibration fields.
        merged: dict[str, Any] = {**data, **self._pending_calibration}
        self._pending_calibration = {}  # reset for next session

        analyte = self._get_analyte() if self._get_analyte is not None else None
        if not analyte:
            analyte = merged.get("gas_label", "unknown")

        # Extract metrics (handle float("nan") sentinel values from analysis).
        lod = _safe_float(merged.get("lod_ppm"))
        r2 = _safe_float(merged.get("calibration_r2") or merged.get("r2_from_cal"))
        sensitivity = _safe_float(merged.get("sensitivity_nm_per_ppm"))
        drift = _safe_float(merged.get("drift_rate_nm_per_frame"))
        snr = _safe_float(merged.get("mean_snr"))
        rmse = _safe_float(merged.get("calibration_rmse_ppm") or merged.get("rmse_from_cal"))

        # Compute scorecard
        scorecard = self._compute_scorecard(lod, r2, sensitivity, drift, snr, analyte)
        overall = scorecard["overall_health"]

        # R² consecutive-session trigger
        if r2 is not None and r2 < _R2_LOW_ABSOLUTE:
            self._consecutive_low_r2 += 1
        else:
            self._consecutive_low_r2 = 0

        # Determine whether recalibration is required
        recal_reasons: list[str] = []
        if scorecard["lod_score"] < _LOD_SCORE_RECAL_THRESHOLD and lod is not None:
            recal_reasons.append(
                f"LOD has degraded to {lod:.4f} ppm "
                f"(score {scorecard['lod_score']:.0f}/100 vs best)"
            )
        if scorecard["sensitivity_score"] < _SENS_SCORE_RECAL_THRESHOLD and sensitivity is not None:
            recal_reasons.append(
                f"Sensitivity dropped to {abs(sensitivity):.3f} nm/ppm "
                f"(score {scorecard['sensitivity_score']:.0f}/100 vs best)"
            )
        if self._consecutive_low_r2 >= _R2_CONSECUTIVE_SESSIONS:
            recal_reasons.append(
                f"R² has been below {_R2_LOW_ABSOLUTE} for "
                f"{self._consecutive_low_r2} consecutive sessions"
            )
        if overall < _OVERALL_HEALTH_RECAL_THRESHOLD:
            recal_reasons.append(
                f"Overall health score {overall:.0f}/100 below minimum threshold"
            )

        warnings: list[str] = []
        if scorecard["drift_score"] < _DRIFT_SCORE_WARNING and drift is not None:
            warnings.append(f"Drift rate elevated ({drift:.6f} nm/frame) — allow longer equilibration")
        if scorecard["snr_score"] < _SNR_SCORE_WARNING and snr is not None:
            warnings.append(f"SNR {snr:.1f} below typical — check light source and sample cleanliness")

        needs_recal = bool(recal_reasons)
        event_type = "recalibration_required" if needs_recal else "sensor_health_report"
        level = "warn" if needs_recal else "info"

        # Build summary text
        summary_parts = [
            f"Sensor health: {overall:.0f}/100 overall "
            f"(LOD={scorecard['lod_score']:.0f}, "
            f"Sensitivity={scorecard['sensitivity_score']:.0f}, "
            f"R²={scorecard['r2_score']:.0f}, "
            f"Drift={scorecard['drift_score']:.0f}, "
            f"SNR={scorecard['snr_score']:.0f})"
        ]
        if recal_reasons:
            summary_parts.append("RECALIBRATION REQUIRED: " + "; ".join(recal_reasons))
        if warnings:
            summary_parts.append("Warnings: " + "; ".join(warnings))
        summary_text = " | ".join(summary_parts)

        self._bus.emit(self._AgentEvent(
            source=self.source,
            level=level,
            type=event_type,
            data={
                "analyte": analyte,
                "scorecard": scorecard,
                "recalibration_reasons": recal_reasons,
                "warnings": warnings,
                "consecutive_low_r2_sessions": self._consecutive_low_r2,
                "lod_ppm": lod,
                "sensitivity_nm_per_ppm": sensitivity,
                "calibration_r2": r2,
                "calibration_rmse_ppm": rmse,
                "drift_rate_nm_per_frame": drift,
                "mean_snr": snr,
            },
            text=summary_text,
        ))
        log.info("SensorHealthAgent: %s", summary_text)

        # Optional Claude narrative for recalibration events
        if needs_recal and self._auto_explain:
            await self._narrate_recalibration(analyte, scorecard, recal_reasons, lod, sensitivity)

    async def _narrate_recalibration(
        self,
        analyte: Optional[str],
        scorecard: dict[str, Any],
        reasons: list[str],
        lod: Optional[float],
        sensitivity: Optional[float],
    ) -> None:
        """Call Claude for a plain-language recalibration recommendation."""
        history_text = ""
        if self._memory is not None and analyte:
            history_text = self._memory.format_for_agent_prompt(analyte)

        physics = ""
        if _KB_AVAILABLE:
            physics = build_sensor_physics_preamble(self._sensor_type) + "\n\n"

        prompt = (
            physics
            + "## Sensor Health Assessment\n\n"
            + history_text
            + "\n\n---\n"
            f"**Current session health scorecard:**\n"
            f"- Overall health: {scorecard['overall_health']:.0f}/100\n"
            f"- LOD score: {scorecard['lod_score']:.0f}/100 "
            f"(current LOD = {lod:.4f} ppm)\n"
            if lod is not None else f"- LOD score: {scorecard['lod_score']:.0f}/100\n"
            + f"- Sensitivity score: {scorecard['sensitivity_score']:.0f}/100 "
            f"(current sensitivity = {abs(sensitivity):.3f} nm/ppm)\n"
            if sensitivity is not None else ""
            + f"- R² score: {scorecard['r2_score']:.0f}/100\n\n"
            "**Recalibration triggers:**\n"
            + "\n".join(f"- {r}" for r in reasons)
            + "\n\n"
            "In exactly 3–4 sentences for a researcher:\n"
            "(1) Diagnose which component of the sensor system is most likely "
            "responsible for the detected degradation, citing the scorecard.\n"
            "(2) Specify the recalibration procedure — which steps, in which "
            "order, and what measurements to collect.\n"
            "(3) Note one preventive action to slow future degradation of this "
            "specific metric."
        )
        text = await self._call(prompt)
        if text:
            self._bus.emit(self._AgentEvent(
                source=self.source,
                level="claude",
                type="recalibration_advice",
                data={"scorecard": scorecard, "reasons": reasons, "advice": text},
                text=text,
            ))

    # ------------------------------------------------------------------
    # Health score computation
    # ------------------------------------------------------------------

    def _compute_scorecard(
        self,
        lod: Optional[float],
        r2: Optional[float],
        sensitivity: Optional[float],
        drift: Optional[float],
        snr: Optional[float],
        analyte: Optional[str],
    ) -> dict[str, Any]:
        """Return per-dimension scores (0–100) and weighted overall score."""
        # Pull historical best values from memory
        best_lod: Optional[float] = None
        best_sensitivity: Optional[float] = None
        typical_drift: Optional[float] = None
        best_snr: Optional[float] = None

        if self._memory is not None and analyte:
            summary = self._memory.get_analyte_summary(analyte)
            if summary:
                lod_stats = summary.get("lod_ppm", {})
                if lod_stats.get("n", 0) > 0:
                    best_lod = lod_stats.get("min")  # lowest LOD = best
                sens_stats = summary.get("sensitivity_nm_per_ppm", {})
                if sens_stats.get("n", 0) > 0:
                    # Best sensitivity = largest absolute value
                    best_sensitivity = max(abs(sens_stats.get("min", 0)),
                                           abs(sens_stats.get("max", 0))) or None

            health = self._memory.get_sensor_health_summary()
            drift_stats = health.get("drift_nm_per_min", {})
            if drift_stats.get("n", 0) > 0:
                typical_drift = drift_stats.get("mean")

        # --- LOD score ---
        if lod is not None and best_lod is not None and best_lod > 0:
            lod_score = min(best_lod / lod * 100, 100.0)
        elif lod is not None:
            lod_score = 80.0  # no history yet — assume reasonable
        else:
            lod_score = 50.0  # no data

        # --- Sensitivity score ---
        if sensitivity is not None and best_sensitivity is not None and best_sensitivity > 0:
            sens_score = min(abs(sensitivity) / best_sensitivity * 100, 100.0)
        elif sensitivity is not None:
            sens_score = 80.0
        else:
            sens_score = 50.0

        # --- R² score --- (0.90=0, 1.00=100)
        if r2 is not None:
            r2_score = max(0.0, min((r2 - 0.90) / 0.10 * 100, 100.0))
        else:
            r2_score = 50.0

        # --- Drift score ---
        # drift_rate_nm_per_frame → convert to nm/min for comparison
        # (assuming 50ms integration → 20 Hz → 1200 frames/min)
        if drift is not None and typical_drift is not None and typical_drift > 0:
            drift_nm_per_min = abs(drift) * 1200
            drift_score = min(typical_drift / (drift_nm_per_min + 1e-9) * 100, 100.0)
        elif drift is not None and drift == 0.0:
            drift_score = 100.0
        else:
            drift_score = 70.0  # no history

        # --- SNR score ---
        if snr is not None:
            # Use 3 as the absolute minimum (QualityAgent threshold) and 30 as "excellent"
            snr_score = max(0.0, min((snr - 3.0) / 27.0 * 100, 100.0))
            if best_snr is not None and best_snr > 0:
                snr_score = min(snr / best_snr * 100, snr_score)
        else:
            snr_score = 50.0

        # --- Overall weighted score ---
        overall = (
            lod_score * _WEIGHTS["lod"]
            + sens_score * _WEIGHTS["sensitivity"]
            + r2_score * _WEIGHTS["r2"]
            + drift_score * _WEIGHTS["drift"]
            + snr_score * _WEIGHTS["snr"]
        )

        return {
            "lod_score": round(lod_score, 1),
            "sensitivity_score": round(sens_score, 1),
            "r2_score": round(r2_score, 1),
            "drift_score": round(drift_score, 1),
            "snr_score": round(snr_score, 1),
            "overall_health": round(overall, 1),
            "has_history": best_lod is not None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v: Any) -> Optional[float]:
    """Return float, or None for None/NaN/inf."""
    try:
        f = float(v)  # type: ignore[arg-type]
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None
