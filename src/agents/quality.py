"""
src.agents.quality
==================
Data Quality Agent for the gas sensing pipeline.

Responsibility
--------------
The quality agent is the first gate every incoming spectrum passes through.
It performs rapid pre-checks before any expensive processing (denoising,
peak detection, ML inference):

1. **Saturation check** — any channel ≥ ``saturation_threshold`` counts
   (default 60 000 for a 16-bit detector) means the photodetector is
   clipped.  Affected features are unreliable.
2. **Low-signal check** — mean intensity below ``min_signal`` suggests the
   light source is blocked, fibre is disconnected, or integration time is
   too short.
3. **SNR check** — estimated SNR below ``min_snr`` flags noisy spectra.
4. **Finite-value check** — any NaN or Inf causes an immediate hard reject.
5. **Wavelength monotonicity check** — out-of-order wavelength axes indicate
   a broken driver or file-parsing issue.

Gates are ordered by severity; the first failed gate sets the result code.

Public API
----------
- ``QualityResult``     — typed result (passed/failed + details)
- ``DataQualityAgent``  — check(wavelengths, intensities) → QualityResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class QualityCode(str, Enum):
    """Outcome of the quality check."""

    OK = "OK"
    WARNING_LOW_SNR = "WARNING_LOW_SNR"
    WARNING_LOW_SIGNAL = "WARNING_LOW_SIGNAL"
    FAIL_SATURATED = "FAIL_SATURATED"
    FAIL_NON_FINITE = "FAIL_NON_FINITE"
    FAIL_MONOTONICITY = "FAIL_MONOTONICITY"
    FAIL_TOO_SHORT = "FAIL_TOO_SHORT"


@dataclass
class QualityResult:
    """Result of a single quality-check run."""

    code: QualityCode
    passed: bool
    is_hard_fail: bool  # True → skip processing entirely
    snr: float | None
    saturation_fraction: float  # fraction of channels at/above threshold
    mean_intensity: float
    messages: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def quality_score(self) -> float:
        """Heuristic score in [0, 1] for downstream use.

        Hard fails → 0.0; warnings → 0.5; OK with high SNR → up to 1.0.
        """
        if self.is_hard_fail:
            return 0.0
        if self.code == QualityCode.WARNING_LOW_SNR:
            return 0.35
        if self.code == QualityCode.WARNING_LOW_SIGNAL:
            return 0.25
        # OK: scale by SNR up to 1.0
        if self.snr is not None and np.isfinite(self.snr):
            return float(np.clip(self.snr / 50.0, 0.0, 1.0))
        return 0.5


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class DataQualityAgent:
    """Gate-based spectrum quality checker.

    Parameters
    ----------
    saturation_threshold:
        Detector counts above which a channel is considered saturated.
    saturation_fraction_hard:
        Fraction of channels saturated that triggers a HARD fail (default: 0.1).
        Below this, saturation raises only a warning through a future soft gate.
    min_signal:
        Mean intensity below which a LOW_SIGNAL warning is raised.
    min_snr:
        SNR below which a LOW_SNR warning is raised (soft gate only).
    min_points:
        Minimum number of spectral points; shorter spectra are hard-rejected.
    """

    def __init__(
        self,
        saturation_threshold: float = 60_000.0,
        saturation_fraction_hard: float = 0.1,
        min_signal: float = 10.0,
        min_snr: float = 3.0,
        min_points: int = 10,
    ) -> None:
        self.saturation_threshold = saturation_threshold
        self.saturation_fraction_hard = saturation_fraction_hard
        self.min_signal = min_signal
        self.min_snr = min_snr
        self.min_points = min_points

        # Running statistics for logging
        self._n_checked: int = 0
        self._n_failed: int = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def check(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
    ) -> QualityResult:
        """Run all quality gates on a spectrum.

        Parameters
        ----------
        wavelengths:
            Wavelength axis in nm, shape ``(N,)``.
        intensities:
            Measured intensities (same shape as *wavelengths*).

        Returns
        -------
        QualityResult
            Detailed quality check result.  Check ``result.is_hard_fail``
            to decide whether to skip downstream processing.
        """
        wl = np.asarray(wavelengths, dtype=float)
        it = np.asarray(intensities, dtype=float)
        self._n_checked += 1

        # ── Gate 0: minimum length ──────────────────────────────────────
        if len(it) < self.min_points or len(wl) < self.min_points:
            return self._result(
                QualityCode.FAIL_TOO_SHORT,
                is_hard_fail=True,
                snr=None,
                sat_frac=0.0,
                mean_int=0.0,
                messages=[f"Spectrum too short: {len(it)} points (min {self.min_points})"],
            )

        # ── Gate 1: finite values ───────────────────────────────────────
        if not (np.isfinite(wl).all() and np.isfinite(it).all()):
            n_nan = int(~np.isfinite(it).sum())
            return self._result(
                QualityCode.FAIL_NON_FINITE,
                is_hard_fail=True,
                snr=None,
                sat_frac=0.0,
                mean_int=float(np.nanmean(it)),
                messages=[f"{n_nan} non-finite values in intensities."],
            )

        # ── Gate 2: wavelength monotonicity ────────────────────────────
        if len(wl) > 1 and not (np.diff(wl) > 0).all():
            return self._result(
                QualityCode.FAIL_MONOTONICITY,
                is_hard_fail=True,
                snr=None,
                sat_frac=0.0,
                mean_int=float(it.mean()),
                messages=["Wavelength axis is not strictly monotonically increasing."],
            )

        # ── Compute descriptive statistics ──────────────────────────────
        mean_int = float(it.mean())
        sat_mask = it >= self.saturation_threshold
        sat_frac = float(sat_mask.mean())
        snr = self._estimate_snr(it)

        messages: list[str] = []

        # ── Gate 3: saturation (hard above fraction threshold) ──────────
        if sat_frac >= self.saturation_fraction_hard:
            messages.append(
                f"Detector saturated: {sat_frac:.1%} of channels ≥ "
                f"{self.saturation_threshold:.0f} counts."
            )
            return self._result(
                QualityCode.FAIL_SATURATED,
                is_hard_fail=True,
                snr=snr,
                sat_frac=sat_frac,
                mean_int=mean_int,
                messages=messages,
            )

        # ── Gate 4: low signal (warning) ────────────────────────────────
        if mean_int < self.min_signal:
            messages.append(f"Low signal: mean intensity={mean_int:.1f} (min {self.min_signal}).")
            return self._result(
                QualityCode.WARNING_LOW_SIGNAL,
                is_hard_fail=False,
                snr=snr,
                sat_frac=sat_frac,
                mean_int=mean_int,
                messages=messages,
            )

        # ── Gate 5: low SNR (warning) ────────────────────────────────────
        if snr is not None and snr < self.min_snr:
            messages.append(
                f"HARD FAIL: Low SNR {snr:.1f} < threshold {self.min_snr:.1f}. "
                f"Spectrum too noisy for reliable analysis."
            )
            return self._result(
                QualityCode.WARNING_LOW_SNR,  # Keep enum for backward compatibility
                is_hard_fail=True,  # FIX C6: Changed from False (2026-04-07)
                snr=snr,
                sat_frac=sat_frac,
                mean_int=mean_int,
                messages=messages,
            )

        # ── All gates passed ─────────────────────────────────────────────
        return self._result(
            QualityCode.OK,
            is_hard_fail=False,
            snr=snr,
            sat_frac=sat_frac,
            mean_int=mean_int,
            messages=[],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _result(
        self,
        code: QualityCode,
        *,
        is_hard_fail: bool,
        snr: float | None,
        sat_frac: float,
        mean_int: float,
        messages: list[str],
    ) -> QualityResult:
        passed = code == QualityCode.OK or (
            not is_hard_fail and code.startswith("WARNING")  # type: ignore[attr-defined]
        )
        passed = code == QualityCode.OK or (not is_hard_fail)
        if is_hard_fail:
            self._n_failed += 1
        return QualityResult(
            code=code,
            passed=passed,
            is_hard_fail=is_hard_fail,
            snr=snr,
            saturation_fraction=sat_frac,
            mean_intensity=mean_int,
            messages=messages,
        )

    @staticmethod
    def _estimate_snr(intensities: np.ndarray) -> float | None:
        """Estimate signal-to-noise via robust statistics.

        Uses the inter-quartile range divided by ``0.6745`` as a robust
        noise estimator (equivalent to the MAD for Gaussian noise).
        """
        if len(intensities) < 4:
            return None
        _pct = np.asarray(np.percentile(intensities, [75, 25]), dtype=float)
        q75, q25 = float(_pct[0]), float(_pct[1])
        iqr = q75 - q25
        robust_noise = iqr / 1.3490  # IQR → σ for Gaussian
        signal = float(np.median(intensities))
        if robust_noise < 1e-12:
            return None
        return float(signal / robust_noise)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, object]:
        """Return cumulative check statistics."""
        return {
            "total_checked": self._n_checked,
            "total_failed": self._n_failed,
            "pass_rate": (
                round(1.0 - self._n_failed / self._n_checked, 4) if self._n_checked > 0 else None
            ),
        }

    def reset_stats(self) -> None:
        """Reset cumulative counters."""
        self._n_checked = 0
        self._n_failed = 0
