# ADR-S003: LSPR Signal Convention — Δλ Primary, Redshift Negative

**Status:** Accepted
**Date:** 2026-05-14
**Immutable:** Yes

## Decision

The primary analytical signal is the peak wavelength shift:

    Δλ = λ_gas − λ_reference  (nm)

Sign convention: analyte adsorption causes a redshift (increase in peak
wavelength), so Δλ is **negative** when analyte is present. Blank gives Δλ ≈ 0.

ΔIntensity is a secondary diagnostic only, not used for calibration.

## Rationale

Peak wavelength is the physically meaningful LSPR quantity (refractive index
change → resonance shift). Peak intensity changes are dominated by scattering
and source drift, making them unreliable for absolute concentration estimation.

The negative-redshift convention matches LSPR literature (Mayer & Hafner 2011).

## Implementation

`src/inference/realtime_pipeline.py` — cross-correlation peak tracking
`src/features/lspr_features.py` — Δλ feature extraction

## Reference

Mayer & Hafner, Chem. Rev. 2011, 111(6), 3828–3857.
