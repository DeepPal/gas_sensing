# ADR-S004: Figure of Merit — |S|/FWHM (Willets & Van Duyne 2007)

**Status:** Accepted
**Date:** 2026-05-14
**Immutable:** Yes

## Decision

    FOM = |S| / FWHM   (units: ppm⁻¹)

where S is the sensitivity (nm/ppm) from the linear calibration region and
FWHM is the full-width at half-maximum of the reference spectrum Lorentzian fit (nm).

## Rationale

FOM normalises sensitivity by peak width, enabling comparison across sensors
with different plasmon resonances. Willets & Van Duyne (2007) is the canonical
reference for LSPR FOM and is required by most analytical chemistry reviewers.

## Implementation

`src/scientific/lod.py → sensor_performance_summary()` — FOM field
`dashboard/agentic_pipeline_tab.py` — Step 3 FOM display

## Reference

Willets & Van Duyne, Annu. Rev. Phys. Chem. 2007, 58, 267–297.
