"""
spectraagent.knowledge.protocols
==================================
ICH Q2(R1) analytical method validation protocol for optical chemical sensors.

This module provides:
1. A complete specification of each required validation test.
2. A stateful ``ValidationTracker`` that observes session calibration data
   and tells the operator which tests are done, which are missing, and
   what measurements are needed to fill each gap.

ICH Q2(R1) reference: International Conference on Harmonisation of Technical
Requirements for Registration of Pharmaceuticals for Human Use, "Validation
of Analytical Procedures: Text and Methodology" (2005).

Adapted for optical chemical sensors (LSPR, SPR, fluorescence, Raman) where
the analytical procedure is: sensor → spectral signal → calibration model →
concentration estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ValidationStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class ValidationRequirement:
    """Specification for one ICH Q2(R1) validation test.

    Attributes
    ----------
    id : str
        Short identifier, e.g. "linearity".
    ich_section : str
        Section reference in ICH Q2(R1), e.g. "§4.2".
    display_name : str
        Human-readable test name.
    description : str
        What this test proves and why it matters.
    min_concentration_levels : int
        Minimum number of distinct concentration levels required.
    min_replicates_per_level : int
        Minimum repeat measurements per concentration level.
    acceptance_criteria : str
        Quantitative pass/fail criteria.
    calculation_method : str
        How to calculate the metric (maps to SpectraAgent's existing functions).
    depends_on : list[str]
        Tests that must pass before this one is meaningful.
    measurement_protocol : str
        Step-by-step experimental instructions for the researcher.
    what_is_reported : str
        What must appear in the final validation report / methods paper.
    is_mandatory : bool
        True if required for full ICH Q2(R1) compliance.
    """

    id: str
    ich_section: str
    display_name: str
    description: str
    min_concentration_levels: int
    min_replicates_per_level: int
    acceptance_criteria: str
    calculation_method: str
    measurement_protocol: str
    what_is_reported: str
    depends_on: list[str] = field(default_factory=list)
    is_mandatory: bool = True


# ---------------------------------------------------------------------------
# Complete ICH Q2(R1) validation protocol
# ---------------------------------------------------------------------------

ICH_Q2_PROTOCOL: dict[str, ValidationRequirement] = {

    "specificity": ValidationRequirement(
        id="specificity",
        ich_section="§4.1",
        display_name="Specificity / Selectivity",
        description=(
            "Demonstrates that the sensor responds selectively to the target analyte "
            "in the presence of expected interferents (co-present gases, environmental "
            "humidity, other VOCs). Cross-sensitivity coefficients K_{B/A} are reported. "
            "This is the first validation test because a non-specific sensor cannot "
            "provide meaningful quantitative data."
        ),
        min_concentration_levels=1,    # analyte at mid-range + each interferent
        min_replicates_per_level=3,
        acceptance_criteria=(
            "Cross-sensitivity coefficient K_{B/A} = (apparent signal from interferent B "
            "as if it were analyte A) / (true signal from A at same ppm). "
            "Target: K < 0.1 for all critical interferents. "
            "K 0.1–0.3: minor interference — report and apply mathematical correction. "
            "K > 0.5: significant interference — must be addressed before validation proceeds."
        ),
        calculation_method="src.scientific.selectivity.compute_cross_sensitivity_matrix()",
        depends_on=[],
        measurement_protocol=(
            "1. Establish clean carrier gas baseline (0 ppm all analytes). "
            "2. Introduce target analyte at mid-range concentration (3 replicates). "
            "3. Return to clean baseline (≥5 min purge). "
            "4. Introduce each interferent gas at its expected environmental concentration "
            "   (3 replicates per interferent). "
            "5. Compute K_{B/A} = ΔSignal(interferent) / ΔSignal(analyte at same ppm). "
            "6. Test interferents in order: most chemically similar first."
        ),
        what_is_reported=(
            "Cross-sensitivity matrix (K_{B/A} for each interferent tested), "
            "interferent concentrations tested, carrier gas composition, "
            "statement of any applied mathematical correction."
        ),
        is_mandatory=True,
    ),

    "linearity": ValidationRequirement(
        id="linearity",
        ich_section="§4.2",
        display_name="Linearity",
        description=(
            "Demonstrates a linear relationship between sensor signal and analyte "
            "concentration across the validated measurement range. "
            "Mandel's F-test distinguishes true linearity from apparent linearity."
        ),
        min_concentration_levels=5,
        min_replicates_per_level=1,
        acceptance_criteria=(
            "R² ≥ 0.999 across validated range. "
            "Mandel's F-test: p > 0.05 (linear model not significantly worse than quadratic). "
            "Residuals: randomly distributed about zero (no systematic curvature). "
            "y-intercept not significantly different from zero (t-test, p > 0.05)."
        ),
        calculation_method=(
            "src.scientific.lod.test_linearity_mandel(); "
            "scipy.stats.linregress() on [concentration, signal] pairs."
        ),
        depends_on=[],
        measurement_protocol=(
            "1. Prepare at least 5 concentration standards spanning the full intended range. "
            "2. Measure each standard in random order (to avoid systematic bias). "
            "3. If concentration range spans >2 orders of magnitude: include at least "
            "   2 points in each decade. "
            "4. Plot signal vs concentration — look for curvature. "
            "5. Run Mandel's F-test and report result."
        ),
        what_is_reported=(
            "Regression equation (slope, intercept, R²), "
            "Mandel's F-test result and p-value, "
            "residual plot description, "
            "validated concentration range."
        ),
        is_mandatory=True,
    ),

    "range": ValidationRequirement(
        id="range",
        ich_section="§4.3",
        display_name="Range",
        description=(
            "Defines the concentration interval over which the method meets "
            "acceptable precision, accuracy, and linearity. "
            "The range is bounded by LOQ (lower end) and LOL (upper end, Limit of Linearity)."
        ),
        min_concentration_levels=3,
        min_replicates_per_level=3,
        acceptance_criteria=(
            "Lower bound: ≥ LOQ (concentration where CV ≤ 10% and accuracy ≥ 80%). "
            "Upper bound: highest concentration where Mandel linearity test still passes "
            "(LOL = Limit of Linearity). "
            "Range must span at least one order of magnitude for quantitative methods."
        ),
        calculation_method="Derived from linearity and LOQ tests.",
        depends_on=["linearity", "lod_loq"],
        measurement_protocol=(
            "1. Confirm the linearity test results define the upper bound (LOL). "
            "2. Confirm the LOQ from the precision/LOD test defines the lower bound. "
            "3. Verify: at LOQ concentration, CV ≤ 10% and signal is significantly "
            "   above blank (≥ 10σ_blank). "
            "4. Report range as [LOQ, LOL] with units."
        ),
        what_is_reported=(
            "Numerical range [LOQ ppm, LOL ppm], method used to establish each bound."
        ),
        is_mandatory=True,
    ),

    "precision_repeatability": ValidationRequirement(
        id="precision_repeatability",
        ich_section="§4.4.1",
        display_name="Precision — Repeatability (intra-day)",
        description=(
            "Demonstrates measurement reproducibility when the same analyst measures "
            "the same sample on the same day using the same equipment. "
            "Expressed as coefficient of variation (CV%) or relative standard deviation."
        ),
        min_concentration_levels=3,    # low, mid, high of validated range
        min_replicates_per_level=6,    # ICH Q2(R1) requires ≥6 replicates at each level
        acceptance_criteria=(
            "CV% ≤ 2% at each concentration level (typical for optical sensors). "
            "Report CV at low (~LOQ), mid-range, and high (~LOL) concentrations. "
            "If sensor shows higher inherent variability: justify and report."
        ),
        calculation_method="CV% = (std(replicates) / mean(replicates)) × 100",
        depends_on=["linearity"],
        measurement_protocol=(
            "1. Select 3 concentration levels: near LOQ, mid-range, near LOL. "
            "2. At each level: measure ≥6 successive replicates WITHOUT re-preparing sample. "
            "3. Allow adequate signal equilibration between replicates. "
            "4. Compute mean, std, CV% at each level."
        ),
        what_is_reported=(
            "CV% at each concentration level, number of replicates, "
            "conditions (same analyst, same day, same instrument)."
        ),
        is_mandatory=True,
    ),

    "precision_intermediate": ValidationRequirement(
        id="precision_intermediate",
        ich_section="§4.4.2",
        display_name="Precision — Intermediate (inter-day)",
        description=(
            "Demonstrates reproducibility across different days, analysts, or "
            "equipment (if applicable). Essential for methods to be used over time "
            "in the same lab or transferred to other labs."
        ),
        min_concentration_levels=3,
        min_replicates_per_level=3,
        acceptance_criteria=(
            "CV% ≤ 5% across measurement days for the same concentration levels. "
            "Compare results using F-test or ANOVA to assess day-to-day variability."
        ),
        calculation_method="ANOVA (between-day variance vs within-day variance).",
        depends_on=["precision_repeatability"],
        measurement_protocol=(
            "1. Repeat the repeatability measurements on ≥3 different days. "
            "2. Re-calibrate the sensor on each day (fresh reference spectrum). "
            "3. Compute inter-day CV% and compare with intra-day CV%."
        ),
        what_is_reported=(
            "Inter-day CV%, number of days, any observed trends, "
            "statement of conditions varied (day, re-calibration, etc.)."
        ),
        is_mandatory=False,    # important but often deferred for initial publication
    ),

    "accuracy": ValidationRequirement(
        id="accuracy",
        ich_section="§4.5",
        display_name="Accuracy",
        description=(
            "Demonstrates the closeness of the measured value to the true "
            "concentration. Uses certified gas standards or spiked samples "
            "with known analyte concentrations."
        ),
        min_concentration_levels=3,
        min_replicates_per_level=3,
        acceptance_criteria=(
            "Percent recovery (measured / true × 100%) between 90% and 110% "
            "across the validated range. "
            "Bias (systematic error): not significantly different from 0 (t-test, p > 0.05)."
        ),
        calculation_method="Recovery% = (C_measured / C_true) × 100",
        depends_on=["linearity", "precision_repeatability"],
        measurement_protocol=(
            "1. Use certified gas standards (NIST-traceable if available) at "
            "   low, mid, and high concentrations. "
            "2. Measure each standard ≥3 times after fresh calibration. "
            "3. Report recovery% and 95% CI at each concentration level."
        ),
        what_is_reported=(
            "Recovery% at each level, 95% CI, standard source and traceability, "
            "any systematic bias and its statistical significance."
        ),
        is_mandatory=True,
    ),

    "lod_loq": ValidationRequirement(
        id="lod_loq",
        ich_section="§4.6 / §4.7",
        display_name="LOD and LOQ",
        description=(
            "Limit of Detection (LOD): lowest concentration that can be reliably "
            "distinguished from the blank signal (3σ criterion). "
            "Limit of Quantification (LOQ): lowest concentration that can be "
            "quantified with acceptable precision and accuracy (10σ criterion). "
            "Bootstrap confidence intervals are computed to assess uncertainty."
        ),
        min_concentration_levels=8,    # blank + ≥7 calibration points for reliable LOD
        min_replicates_per_level=1,
        acceptance_criteria=(
            "LOD = 3 × σ_blank / sensitivity (IUPAC 3-sigma method). "
            "LOQ = 10 × σ_blank / sensitivity. "
            "Verification: ≥3 replicate measurements at 2×LOD must all give "
            "signal > LOD signal. "
            "Bootstrap 95% CI for LOD should be ≤ ±30% of the point estimate."
        ),
        calculation_method=(
            "src.scientific.lod.compute_lod(), compute_loq(), compute_lob(); "
            "Bootstrap CI via src.scientific.lod.bootstrap_lod_ci()."
        ),
        depends_on=["linearity"],
        measurement_protocol=(
            "1. Measure blank (0 ppm) ≥10 times to establish σ_blank. "
            "2. Measure ≥7 calibration points spanning the linear range. "
            "3. Compute LOD and LOQ from IUPAC 3-sigma/10-sigma methods. "
            "4. VERIFY: measure 3 replicates at 2×LOD — confirm detection. "
            "5. Compute bootstrap CI (n=1000 resamples) for LOD uncertainty."
        ),
        what_is_reported=(
            "LOD with 95% bootstrap CI, LOQ, LOB, number of blank replicates, "
            "sensitivity (slope), method used (IUPAC 3σ vs instrument noise method)."
        ),
        is_mandatory=True,
    ),

    "robustness": ValidationRequirement(
        id="robustness",
        ich_section="§4.8",
        display_name="Robustness",
        description=(
            "Demonstrates that the method is insensitive to small, deliberate "
            "variations in method parameters: integration time, carrier gas flow "
            "rate, temperature variation, humidity variation. "
            "Identifies which parameters are critical and must be controlled."
        ),
        min_concentration_levels=1,    # typically one mid-range concentration
        min_replicates_per_level=3,
        acceptance_criteria=(
            "Accuracy and precision remain within acceptance criteria (§4.5, §4.4) "
            "when parameters are varied within ±10% of nominal values. "
            "Parameters that cause failure when varied must be designated 'critical' "
            "and strict control limits must be specified."
        ),
        calculation_method="One-factor-at-a-time (OFAT) or fractional factorial design.",
        depends_on=["accuracy", "precision_repeatability"],
        measurement_protocol=(
            "1. Identify key parameters: integration time, temperature, flow rate, "
            "   humidity, reference capture timing. "
            "2. Vary each parameter individually by ±10% from nominal. "
            "3. Measure mid-range concentration × 3 replicates under each condition. "
            "4. Compare recovery% — flag if > ±5% deviation from nominal."
        ),
        what_is_reported=(
            "Parameters tested, variation range, impact on accuracy/precision, "
            "list of critical parameters and their control limits."
        ),
        is_mandatory=False,
    ),
}


# ---------------------------------------------------------------------------
# Stateful tracker
# ---------------------------------------------------------------------------

class ValidationTracker:
    """Tracks ICH Q2(R1) validation progress for one analyte/sensor combination.

    Called by the CalibrationValidationOrchestrator agent and by
    the context_builders to report gap status.

    Parameters
    ----------
    analyte : str
        Name of the analyte being validated.
    required_tests : list[str]
        Subset of ICH_Q2_PROTOCOL keys to require.
        Defaults to all mandatory tests.
    """

    def __init__(self, analyte: str, required_tests: Optional[list[str]] = None) -> None:
        self.analyte = analyte
        if required_tests is None:
            required_tests = [k for k, v in ICH_Q2_PROTOCOL.items() if v.is_mandatory]
        self._required = required_tests
        self._status: dict[str, ValidationStatus] = {
            k: ValidationStatus.NOT_STARTED for k in self._required
        }
        self._results: dict[str, dict[str, Any]] = {}

    def update(self, test_id: str, status: ValidationStatus,
               result: Optional[dict[str, Any]] = None) -> None:
        """Record the outcome of a test."""
        if test_id in self._status:
            self._status[test_id] = status
            if result:
                self._results[test_id] = result

    def infer_from_calibration_data(self, calib_data: dict[str, Any]) -> None:
        """Automatically update status from a calibration data dict.

        Parameters
        ----------
        calib_data:
            Keys expected (all optional):
            - n_points: int
            - r_squared: float
            - lod_ppm: float
            - loq_ppm: float
            - rmse_ppm: float
            - mandel_p: float (Mandel's F-test p-value)
            - n_replicates: int (replicates per level, if measured)
        """
        n = calib_data.get("n_points", 0)
        r2 = calib_data.get("r_squared")
        lod = calib_data.get("lod_ppm")
        loq = calib_data.get("loq_ppm")
        mandel_p = calib_data.get("mandel_p")

        # Linearity: need ≥5 points with R²≥0.999 and Mandel pass
        if n >= 5 and r2 is not None:
            req = ICH_Q2_PROTOCOL["linearity"]
            if r2 >= 0.999 and (mandel_p is None or mandel_p > 0.05):
                self.update("linearity", ValidationStatus.COMPLETE, {"r_squared": r2})
            elif r2 >= 0.99:
                self.update("linearity", ValidationStatus.IN_PROGRESS)

        # LOD/LOQ: need lod and loq computed
        if lod is not None and loq is not None:
            self.update("lod_loq", ValidationStatus.COMPLETE,
                        {"lod_ppm": lod, "loq_ppm": loq})

    def get_gaps(self) -> list[ValidationRequirement]:
        """Return list of tests that are NOT_STARTED, ordered by dependency."""
        gaps = []
        for test_id in self._required:
            if self._status.get(test_id) != ValidationStatus.COMPLETE:
                gaps.append(ICH_Q2_PROTOCOL[test_id])
        return gaps

    def get_next_test(self) -> Optional[ValidationRequirement]:
        """Return the next test to run (respecting dependencies)."""
        for test in self.get_gaps():
            deps_met = all(
                self._status.get(dep) == ValidationStatus.COMPLETE
                for dep in test.depends_on
            )
            if deps_met:
                return test
        return None

    def completion_pct(self) -> float:
        """Return completion percentage (0–100)."""
        if not self._required:
            return 100.0
        done = sum(1 for k in self._required if self._status.get(k) == ValidationStatus.COMPLETE)
        return 100.0 * done / len(self._required)

    def format_status_for_prompt(self) -> str:
        """Return an agent-prompt-ready Markdown status report."""
        pct = self.completion_pct()
        lines = [
            f"## ICH Q2(R1) Validation Status — {self.analyte}",
            f"**Completion**: {pct:.0f}%\n",
        ]

        for test_id in self._required:
            status = self._status.get(test_id, ValidationStatus.NOT_STARTED)
            req = ICH_Q2_PROTOCOL[test_id]
            icon = {"complete": "✅", "in_progress": "🔄",
                    "not_started": "⬜", "failed": "❌"}.get(status.value, "⬜")
            lines.append(f"{icon} **{req.display_name}** ({req.ich_section})")
            if status == ValidationStatus.NOT_STARTED:
                lines.append(f"   → *Not started. Requires: "
                              f"{req.min_concentration_levels} concentration levels × "
                              f"{req.min_replicates_per_level} replicates.*")
            elif status == ValidationStatus.COMPLETE and test_id in self._results:
                r = self._results[test_id]
                summary = ", ".join(f"{k}={v}" for k, v in list(r.items())[:3])
                lines.append(f"   → *Complete. {summary}*")

        next_test = self.get_next_test()
        if next_test:
            lines.append(f"\n### Next recommended test: {next_test.display_name}")
            lines.append(next_test.measurement_protocol)

        return "\n".join(lines)
