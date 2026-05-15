# ADR-S002: GPR Kernel — Matérn ν=5/2

**Status:** Accepted (supersedes initial RBF implementation)
**Date:** 2026-05-14
**Immutable:** Yes — changes require ADR-S006+

## Decision

Use the Matérn ν=5/2 covariance kernel for Gaussian Process Regression calibration.
Use 10 optimizer restarts. Track aleatoric noise separately and add in quadrature
to epistemic uncertainty: total_std = √(epistemic² + aleatoric²).

## Rationale

LSPR calibration curves follow Langmuir adsorption isotherms, which are:
- Smooth (continuous derivatives)
- Twice-differentiable but NOT infinitely differentiable

Matérn ν=5/2 models exactly this regime (twice-differentiable functions).
RBF (squared exponential) implies infinite differentiability, which
oversmooths Langmuir isotherms and underestimates uncertainty in the
nonlinear high-concentration region.

10 restarts: the log marginal likelihood landscape for Matérn kernels on
sparse calibration data (typically 6–12 points) has local optima. 10 restarts
finds the global optimum with >99% probability empirically; 2 restarts (the
previous value) risked local optima that inflated LOD by up to 18%.

Aleatoric tracking: posterior std from GPR approaches zero with many training
points (epistemic only). In the real sensor, shot noise (aleatoric) persists
regardless of calibration density. Adding aleatoric in quadrature prevents
overconfident predictions.

## Alternatives Rejected

| Alternative | Reason rejected |
|-------------|-----------------|
| RBF / Squared Exponential | Assumes infinite differentiability; oversmooths Langmuir curves |
| Matérn ν=3/2 (once-differentiable) | LSPR response is smooth enough for ν=5/2 |
| Neural network calibration | Requires > 50 points; LSPR calibration typically uses 6–12 |

## Implementation

`src/calibration/gpr.py → GPRCalibration`
Science regression guard: `tests/science_regression/test_gpr_output.py`

## Reference

Rasmussen & Williams, Gaussian Processes for Machine Learning, MIT Press 2006, §4.2.
