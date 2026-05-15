# Advanced Research Protocol Pack

This guide defines a repeatable protocol set for advanced research campaigns.
It is intended for researchers and scientists who need publication-grade rigor.

## Scope

Use this protocol pack when results must be:

- Reproducible across sessions and operators.
- Defensible under peer-review and external technical review.
- Traceable to complete evidence artifacts.

## Protocol Set

### 1. Linearity and Range Protocol

Objective:
- Establish concentration-response behavior and valid operating range.

Minimum design:
- At least 5 concentration levels.
- At least 3 replicates per level.
- Include low-end near expected LOQ and high-end near expected nonlinearity onset.

Acceptance targets:
- R^2 >= 0.99 for the claimed linear region.
- LOL reported with method and threshold notes.
- Residual diagnostics included.

Required artifacts:
- Calibration fit summary.
- Residual diagnostics output.
- Session-level reproducibility manifest.

### 2. Detection-Limit Protocol (LOB/LOD/LOQ)

Objective:
- Quantify detectability and quantification boundaries with uncertainty.

Minimum design:
- Dedicated blank replicates per session.
- Replicates at low concentrations near expected detection threshold.

Acceptance targets:
- Hierarchy integrity: NEC <= LOB <= LOD <= LOQ.
- Confidence interval bounds included for LOD/LOQ.

Required artifacts:
- LOB/LOD/LOQ report with CI bounds.
- Summary of blank replicate quality.

### 3. Cross-Session Reproducibility Protocol

Objective:
- Demonstrate statistical consistency across independent sessions.

Minimum design:
- At least 3 sessions on separate runs.
- Same protocol template and instrument settings recorded.

Acceptance targets:
- Reproducibility summary reported (RSD, trend indicators).
- No unresolved session-level quality gate failures.

Required artifacts:
- Cross-session comparison report.
- Session manifests and calibration summaries.

### 4. Selectivity and Interference Protocol

Objective:
- Quantify interference risk from non-target analytes.

Minimum design:
- Target and interferent exposures under comparable conditions.
- At least one high-interference stress condition.

Acceptance targets:
- Selectivity coefficients reported with assumptions.
- Interference caveats documented when thresholds are exceeded.

Required artifacts:
- Selectivity matrix snapshot.
- Experiment metadata (temperature/humidity included).

### 5. Robustness Protocol (Method Sensitivity)

Objective:
- Quantify impact of controllable method variations.

Minimum design:
- Parameter sweep over critical settings (for example integration time).
- Multiple runs per setting.

Acceptance targets:
- Stability of key metrics under configured tolerance bands.
- Clear recommendation for operational parameter window.

Required artifacts:
- Robustness summary table.
- Parameter sweep configuration and output logs.

## Campaign Execution Checklist

Before campaign:
- Run `python scripts/research_preflight.py --self-check`.
- Verify environment metadata capture is enabled.
- Validate API compatibility guard and integrator smoke check.

During campaign:
- Use consistent acquisition configuration templates.
- Capture reference spectrum under stable conditions.
- Track any protocol deviations in session notes.

After campaign:
- Build evidence pack and qualification artifacts.
- Store signed or checksummed outputs for traceability.
- Record release-ready summary in changelog and campaign report.

## Publication-Grade Output Minimum

Do not claim publication readiness unless all are present:

1. Linearity/range outputs with diagnostics.
2. Detection-limit outputs with CI and blank context.
3. Cross-session reproducibility summary.
4. Selectivity/interference analysis.
5. Robustness sweep summary.
