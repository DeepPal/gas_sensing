# ADR-0002: Strangler-Fig Migration of gas_analysis/core/pipeline.py

**Status**: Active (Phase 7 complete — Phase 8 pending; deployment-readiness fixes applied)
**Date**: 2026-03-24

---

## Context

`gas_analysis/core/pipeline.py` is a 9000-line legacy monolith that grew organically
over the lifetime of the project. It contains:

- Spectrum signal processing (transmittance, smoothing, peak detection)
- Robust linear regression (Theil-Sen, RANSAC, weighted OLS)
- Frame aggregation and stable-block detection
- Calibration curve fitting and ROI discovery
- I/O: CSV loading, JSON serialisation, matplotlib plotting
- Config-coupled orchestration

The problems with this structure:
1. **Untestable in isolation** — functions are entangled with global `CONFIG` and I/O
2. **No type safety** — excluded from mypy (advisory only)
3. **Hidden dead code** — discovered 230+ lines of unreachable code inside `_ransac`
4. **Not reusable** — cannot import pure functions without pulling in the full config/I/O

## Decision

Adopt the **strangler-fig pattern**: incrementally extract pure functions into the
clean `src/` package (typed, tested, mypy-clean) without breaking `pipeline.py`.

### Migration phases

| Phase | Status | Description |
|---|---|---|
| 1 | Complete | Extract pure functions into `src/` with tests |
| 2 | Complete | Redirect `pipeline.py` to import from `src/` (remove duplicates) |
| 3 | Complete | Migrate ROI scanning and calibration utilities to `src/` |
| 4 | Complete | Migrate metric aggregation to `src/reporting/metrics.py` |
| 5 | Complete | Migrate CONFIG-dependent functions with parameter injection |
| 6 | Complete | Migrate I/O serialisers and plotting to `src/reporting/io.py` and `src/reporting/plots.py` |
| 7 | In Progress | Migrate batch computation functions to `src/batch/` |

### Phase 1–5 functions extracted

| `pipeline.py` name | `src/` location |
|---|---|
| `_smooth` | `src.signal.transforms.smooth` |
| `_ensure_odd_window` | `src.signal.transforms.ensure_odd_window` |
| `compute_transmittance` | `src.signal.transforms.compute_transmittance` |
| `_append_absorbance_column` | `src.signal.transforms.append_absorbance_column` |
| `_gaussian_peak_center` | `src.signal.peak.gaussian_peak_center` |
| `_estimate_shift_crosscorr` | `src.signal.peak.estimate_shift_crosscorr` |
| `_weighted_linear` | `src.scientific.regression.weighted_linear` |
| `_theil_sen` | `src.scientific.regression.theil_sen` |
| `_ransac` | `src.scientific.regression.ransac` |
| `find_stable_block` | `src.batch.aggregation.find_stable_block` |
| `average_stable_block` | `src.batch.aggregation.average_stable_block` |
| `average_top_frames` | `src.batch.aggregation.average_top_frames` |
| `_compute_band_ratio_matrix` | `src.signal.roi.compute_band_ratio_matrix` |
| `_find_monotonic_wavelengths` | `src.signal.roi.find_monotonic_wavelengths` |
| `_transform_concentrations` | `src.calibration.transforms.transform_concentrations` |
| `_select_multi_roi_candidates` | `src.calibration.multi_roi.select_multi_roi_candidates` |
| `_compute_multi_roi_fusion_calibration` (computation) | `src.calibration.multi_roi.fit_multi_roi_fusion` |
| `_select_common_signal` | `src.reporting.metrics.select_common_signal` |
| `_common_signal_columns` | `src.reporting.metrics.common_signal_columns` |
| `compute_noise_metrics_map` | `src.reporting.metrics.compute_noise_metrics_map` |
| `compute_roi_repeatability` | `src.reporting.metrics.compute_roi_repeatability` |
| `compute_roi_performance` | `src.reporting.metrics.compute_roi_performance` |
| `summarize_top_comparison` | `src.reporting.metrics.summarize_top_comparison` |
| `summarize_dynamics_metrics` | `src.reporting.metrics.summarize_dynamics_metrics` |
| `summarize_quality_control` | `src.reporting.metrics.summarize_quality_control` |
| `compute_environment_summary` | `src.reporting.environment.compute_environment_summary` |
| `compute_environment_coefficients` | `src.reporting.environment.compute_environment_coefficients` |
| `_stack_trials_for_response` | `src.calibration.roi_scan.stack_trials_for_response` |
| `compute_concentration_response` | `src.calibration.roi_scan.compute_concentration_response` |
| `save_canonical_spectra` | `src.reporting.io.save_canonical_spectra` |
| `save_aggregated_spectra` | `src.reporting.io.save_aggregated_spectra` |
| `save_noise_metrics` | `src.reporting.io.save_noise_metrics` |
| `save_quality_summary` | `src.reporting.io.save_quality_summary` |
| `save_aggregated_summary` | `src.reporting.io.save_aggregated_summary` |
| `save_roi_performance_metrics` | `src.reporting.io.save_roi_performance_metrics` |
| `save_dynamics_summary` | `src.reporting.io.save_dynamics_summary` |
| `save_dynamics_error` | `src.reporting.io.save_dynamics_error` |
| `save_concentration_response_metrics` | `src.reporting.io.save_concentration_response_metrics` |
| `save_environment_compensation_summary` | `src.reporting.io.save_environment_compensation_summary` |
| `save_roi_discovery_plot` | `src.reporting.plots.save_roi_discovery_plot` |
| `save_concentration_response_plot` | `src.reporting.plots.save_concentration_response_plot` |
| `save_wavelength_shift_visualization` | `src.reporting.plots.save_wavelength_shift_visualization` |
| `save_research_grade_calibration_plot` | `src.reporting.plots.save_research_grade_calibration_plot` |
| `save_spectral_response_diagnostic` | `src.reporting.plots.save_spectral_response_diagnostic` |
| `save_roi_repeatability_plot` | `src.reporting.plots.save_roi_repeatability_plot` |
| `save_aggregated_plots` | `src.reporting.plots.save_aggregated_plots` |
| `save_canonical_overlay` | `src.reporting.plots.save_canonical_overlay` |
| `save_calibration_outputs` | `src.reporting.plots.save_calibration_outputs` |

Lines removed from `pipeline.py` (Phase 1–7): **~3869** (original: 9056 → Phase 6: 5920 → Phase 7a: 5656 → Phase 7b: 5153)

**Phase 7a** (4 batch aggregation functions — already in `src/batch/aggregation.py`):

| `pipeline.py` name | `src/` location |
|---|---|
| `find_stable_block` | `src.batch.aggregation.find_stable_block` |
| `average_stable_block` | `src.batch.aggregation.average_stable_block` |
| `average_top_frames` | `src.batch.aggregation.average_top_frames` |
| `select_canonical_per_concentration` | `src.batch.aggregation.select_canonical_per_concentration` |

**Phase 7b** (5 CONFIG-free private functions extracted to new modules):

| `pipeline.py` name | `src/` location |
|---|---|
| `_sort_frame_paths` | `src.batch.preprocessing.sort_frame_paths` |
| `_scale_reference_to_baseline` | `src.batch.response.scale_reference_to_baseline` |
| `_score_trial_quality` | `src.batch.response.score_trial_quality` |
| `_summarize_responsive_delta` | `src.batch.response.summarize_responsive_delta` |
| `_aggregate_responsive_delta_maps` | `src.batch.response.aggregate_responsive_delta_maps` |

**Phase 3 notes:**
- `_find_monotonic_wavelengths` and `_transform_concentrations` were dead code in `pipeline.py` (defined but never called). Extracted to `src/` for the public API; no redirect alias needed in `pipeline.py`.
- `_compute_multi_roi_fusion_calibration` kept as a thin I/O + plotting wrapper; pure computation delegated to `fit_multi_roi_fusion`.
- Bug fix in `transform_concentrations`: original used `if min_positive and not np.isnan(...)` (falsy for zero); corrected to `np.isfinite(min_positive) and min_positive > 0`.
- Useless `np.array(concs, dtype=float)` call in fusion loop (result discarded) silently removed.

**Phase 4 notes:**
- `summarize_dynamics_metrics` was dead code in `pipeline.py` (defined but never called). Extracted for API completeness; no redirect alias needed in `pipeline.py`.
- `compute_roi_performance` improved: hardcoded `lod_sigma=3` / `loq_sigma=10` made explicit keyword arguments (ICH Q2 defaults retained), enabling custom sigma multipliers in unit tests and downstream callers.
- `select_signal_column` added as a new helper (extracted from scattered inline logic); priority: absorbance > transmittance > intensity.

**Phase 5 notes:**
- All 5 CONFIG-dependent functions replaced with thin wrappers in `pipeline.py` that read CONFIG and delegate to pure `src/` implementations.
- `compute_concentration_response` side-channel eliminated: the original read `CONFIG["_last_repeatability"]["global"]["std_transmittance"]` from a previous call's side-effect. Replaced by explicit `global_std: float = 0.0` field on `RoiScanConfig`.
- `RoiScanConfig` dataclass groups all ~20 `CONFIG["roi"]` parameters with self-documenting defaults matching the legacy CONFIG section.
- `compute_environment_coefficients` outer `try/except Exception: return {}` removed — errors now surface properly in the pure implementation; the pipeline.py wrapper inherits the same defensive behaviour through its CONFIG guard.
- `_stack_trials_for_response` body removed entirely (0 lines remaining) — its `src.calibration.roi_scan` alias already imported at the top of `pipeline.py`.

**Phase 6 notes:**
- `io.py` has zero matplotlib dependency; all 10 serialisers are pure JSON/CSV with a `_json_safe()` helper for numpy scalar/array serialisation.
- `plots.py` sets `matplotlib.use("Agg")` at module load for headless/CI safety; all figures closed in `finally` blocks.
- Atomic rename (`path.tmp → path`) used throughout `plots.py` to prevent downstream consumers reading partial PNGs.
- Two CONFIG-reading functions parameterised: `save_concentration_response_plot` gains `x_min/x_max`; `save_spectral_response_diagnostic` gains `step_nm/window_nm`. Pipeline wrappers read CONFIG and delegate.
- `_save_run_metadata` (L305) and `_save_response_series` (L1236) deliberately excluded — called from orchestration layer, deferred to Phase 7.
- Replacement script (`scripts/phase6_replace_bodies.py`) uses single-pass bottom-to-top AST replacement to avoid re-parse failures from non-ASCII characters in docstrings.

## Alternatives Considered

1. **Big-bang rewrite** — rejected; too much risk on production code with no test coverage
2. **Keep duplicate code** — rejected; divergence between `src/` and `pipeline.py` would compound over time
3. **Freeze pipeline.py, write all new features in `src/`** — valid but slower; chose active redirect pattern

## Consequences

### Positive
- `src/` functions are fully typed and tested (669 tests passing)
- `pipeline.py` shrinks each migration cycle (9056 → 5153 lines, ~43% reduction)
- Dead code discovered and removed (230-line dead code block inside `_ransac`)
- API docs auto-generate from `src/` modules via MkDocs
- `src/py.typed` PEP 561 marker: typed consumers see real types, not `Any`
- `src/public_api.py` stable import surface for commercial integrators
- `dict[str, Any]` fixed across all `src/` modules — 0 mypy errors in `src/`

### Negative / Risks
- `pipeline.py` now imports from `src/` — creates a `gas_analysis → src` dependency
  (acceptable since `src` has no dependency on `gas_analysis`)
- Some `src/` APIs diverge slightly from the `pipeline.py` originals
  (e.g., `align_on_grid` returns weights as third element; `pipeline.py` returned signal column name)
  — these cannot be redirected without further cleanup

## Migration Notes

- `pipeline.py` is excluded from coverage (`pyproject.toml: omit`) and mypy advisory
- Always run `pytest -q` after pipeline.py changes to catch import errors
- The `align_on_grid` API divergence must be resolved before aggregation migration (Phase 3 note, still open)
- Phase 7 targets: orchestration entry-point functions (`_save_run_metadata`, `_save_response_series`, and orchestration logic)
