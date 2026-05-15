# ADR-S001: LOD Definition — IUPAC 2011 (3σ blank)

**Status:** Accepted
**Date:** 2026-05-14
**Immutable:** Yes — changes require ADR-S006+

## Decision

Use the IUPAC 2011 definition for Limit of Detection:

    LOD = 3 × σ_blank / |S|

where σ_blank is the standard deviation of blank (zero-concentration) measurements
and S is the calibration sensitivity (nm/ppm).

Confidence interval: parametric bootstrap, n=2000 iterations.

## Rationale

- IUPAC 2011 is the accepted standard for analytical chemistry
- DIN 32645 and the 3×SNR definition produce different values and are not
  interchangeable; using multiple definitions in one system invites confusion
- The 3σ blank method is reproducible given a fixed σ_blank estimate

## Alternatives Rejected

| Alternative | Reason rejected |
|-------------|-----------------|
| 3 × SNR (signal domain) | Conflates measurement noise with detection; not consistent with IUPAC |
| DIN 32645 | German standard; IUPAC has broader international acceptance |
| 10σ blank (LOQ) | This is the LOQ definition; LOD uses 3σ |

## Implementation

`src/scientific/lod.py → calculate_lod_3sigma(noise_std, sensitivity_slope)`

## Reference

IUPAC, Pure Appl. Chem. 2011, 83(5), 1129–1143.
