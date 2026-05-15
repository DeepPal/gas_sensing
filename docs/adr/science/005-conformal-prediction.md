# ADR-S005: Prediction Intervals — Conformal Prediction (Split-CP)

**Status:** Accepted
**Date:** 2026-05-14
**Immutable:** Yes

## Decision

Use split conformal prediction (split-CP) for concentration prediction intervals.
Coverage level: 95% marginal coverage.

## Rationale

Split-CP provides a distribution-free, finite-sample coverage guarantee:
P(y ∈ Ĉ(x)) ≥ 1 − α for any calibration distribution.
Bayesian credible intervals require correct model specification; if the GPR
prior is misspecified, credible intervals do not have nominal coverage.
For sensor data where the noise distribution is unknown, CP is safer.

## Implementation

`src/calibration/conformal.py → ConformalCalibrator`

## Reference

Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction," 2022.
Vovk et al., Algorithmic Learning in a Random World, Springer 2005.
