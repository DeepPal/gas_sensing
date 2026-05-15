"""
Science regression: LOD and sensitivity must not drift.

Tolerance rationale:
  ±2% LOD — analytical chemistry reproducibility standard (ICH Q2 R1)
  ±1% sensitivity — tighter because it drives all concentration predictions
"""
from __future__ import annotations

import numpy as np
import pytest

from src.scientific.lod import calculate_lod_3sigma, calculate_sensitivity


def test_sensitivity_within_1pct(cal_fixture, baselines):
    """Sensitivity must not change by more than 1% from baseline."""
    conc = cal_fixture["concentrations"]
    dl = cal_fixture["delta_lambda_measured"]

    # Signature: calculate_sensitivity(concentrations, responses)
    slope, intercept, r2, slope_se = calculate_sensitivity(conc, dl)
    baseline = baselines["sensitivity_nm_per_ppm"]

    rel_err = abs(float(slope) - baseline) / abs(baseline)
    assert rel_err < 0.01, (
        f"Sensitivity drifted {rel_err:.1%} from baseline {baseline:.6f} nm/ppm. "
        f"Got {float(slope):.6f}. Update ADR-002 if this change is intentional."
    )


def test_lod_within_2pct(cal_fixture, baselines):
    """LOD must not change by more than 2% from baseline."""
    conc = cal_fixture["concentrations"]
    dl = cal_fixture["delta_lambda_measured"]

    # Signature: calculate_sensitivity(concentrations, responses)
    slope, intercept, r2, slope_se = calculate_sensitivity(conc, dl)
    sigma_blank = 0.012  # nm — matches fixture generation
    lod = calculate_lod_3sigma(sigma_blank, abs(float(slope)))
    baseline = baselines["lod_ppm"]

    rel_err = abs(lod - baseline) / baseline
    assert rel_err < 0.02, (
        f"LOD drifted {rel_err:.1%} from baseline {baseline:.6f} ppm. "
        f"Got {lod:.6f}. Update ADR-001 if this change is intentional."
    )


def test_delta_lambda_predicted_at_1ppm(cal_fixture, baselines):
    """Sensitivity-derived prediction at 1.0 ppm must match Langmuir ground truth within ±0.3 nm."""
    dl = cal_fixture["delta_lambda_measured"]
    conc = cal_fixture["concentrations"]

    slope, intercept, r2, _ = calculate_sensitivity(conc, dl)
    # Use the linear fit to predict Δλ at 1.0 ppm
    predicted_dl_at_1ppm = slope * 1.0 + intercept
    langmuir_dl_at_1ppm = baselines["delta_lambda_at_1ppm_true"]

    assert abs(predicted_dl_at_1ppm - langmuir_dl_at_1ppm) < 0.3, (
        f"Predicted Δλ at 1.0 ppm ({predicted_dl_at_1ppm:.4f} nm) deviates more than 0.3 nm "
        f"from Langmuir ground truth ({langmuir_dl_at_1ppm:.4f} nm). "
        "Check calculate_sensitivity or sign convention (ADR-003)."
    )
