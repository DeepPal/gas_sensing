# Methods Section Template for Publication

**Platform version**: Phase 5E (2026-04-08). Copy the sections marked `[FILL]` with your own experimental details. Sections without `[FILL]` can be used verbatim or with minor adaptation.

---

## Suggested Title Options

1. "Automated AI-Assisted LSPR Gas Sensor Characterisation with Full IUPAC/ICH Q2(R1) Analytical Validation"
2. "Agentic Calibration Platform for LSPR-Based VOC Detection: From Raw Spectra to Publication-Ready Metrics"
3. "SpectraAgent: An Open-Source Pipeline for Reproducible LSPR Sensor Characterisation with Statistical Validation"

---

## 2. Materials and Methods

### 2.1 Sensor Fabrication and Experimental Setup

[FILL: your sensor description — chip material, functionalisation, flow cell geometry, gas delivery system]

Spectral acquisition was performed using a ThorLabs CCS200 compact spectrometer (range 200–1000 nm, 3648 pixels) connected via USB to a host computer running the SpectraAgent platform (commit `[FILL: git rev-parse --short HEAD]`). Integration time was set to [FILL] ms with [FILL]-frame averaging. The sensor was allowed to thermally equilibrate for ≥10 minutes before any measurement. Room temperature was [FILL: mean ± std] °C across all sessions.

### 2.2 Reference Spectrum Capture and Differential Signal

Prior to each calibration session, a reference spectrum was recorded under clean carrier gas (dry N₂ / laboratory air — [FILL]). The reference LSPR peak position (λ_ref) and linewidth (FWHM_ref) were determined by fitting a Lorentzian function to the peak region:

$$I(\lambda) = I_0 + \frac{A \cdot (\Gamma/2)^2}{(\lambda - \lambda_0)^2 + (\Gamma/2)^2}$$

where λ₀ is the centre wavelength and Γ is the FWHM. The differential signal for each measurement frame was computed as:

$$\Delta I(\lambda) = I_{\text{sample}}(\lambda) - I_{\text{ref}}(\lambda)$$

The primary sensing signal is the peak wavelength shift Δλ = λ₀,sample − λ_ref, which is proportional to the local refractive index change at the sensor surface.

### 2.3 Calibration Curve and Sensitivity

Calibration data consisted of [FILL: N] concentration levels spaced [FILL: logarithmically / uniformly] over the range [FILL: C_min–C_max] ppm, with [FILL: n] replicate measurements per level at steady state. The last [FILL: 10] frames of each equilibration plateau were averaged as the representative response.

A linear calibration model was fitted by ordinary least squares (OLS):

$$\Delta\lambda = S \cdot C + b$$

where S (nm ppm⁻¹) is the sensitivity, C is the analyte concentration (ppm), and b is the intercept. Sensitivity uncertainty was quantified as the OLS standard error of the slope SE(S), and a 95% confidence interval was derived using the t-distribution with n−2 degrees of freedom.

**Homoscedasticity check (Breusch-Pagan test)**: Prior to using OLS, residuals were tested for constant variance using the Breusch-Pagan Lagrange-Multiplier test at a Bonferroni-corrected significance level of α = 0.017 (family-wise α = 0.05 over three simultaneous tests). When heteroscedasticity was detected (p < 0.017), weighted least squares (WLS) was automatically applied using weights w_i = 1/C_i² (proportional error model), and the WLS sensitivity is reported in place of OLS.

**Linearity check (Mandel's F-test, ICH Q2 §4.2)**: Linearity was assessed by comparing the residual sum of squares of the linear fit against a second-degree polynomial using an F-test. p ≥ 0.05 confirms linearity. The limit of linearity (LOL) was the highest concentration for which linearity was not rejected.

**Residual normality (Shapiro-Wilk)** and **autocorrelation (Durbin-Watson)** were also tested as mandatory checks per ICH Q2(R1).

### 2.4 Detection Limits (IUPAC 2012 / ICH Q2(R1))

The following detection limit hierarchy was computed and verified (IUPAC 2012):

$$\text{NEC} \leq \text{LOB} \leq \text{LOD} \leq \text{LOQ}$$

**Blank measurements**: [FILL: N_blank ≥ 6] spectra measured in clean carrier gas (no analyte) provided the blank signal distribution. Peak wavelengths from these spectra gave μ_blank and σ_blank after conversion to Δλ units (subtracting λ_ref).

**Noise Equivalent Concentration (NEC)**:
$$\text{NEC} = \frac{\sigma_{\text{blank}}}{|S|}$$

**Limit of Blank (LOB)** (IUPAC 2012, one-sided 95th percentile of blank distribution):
$$\text{LOB} = \frac{|\mu_{\text{blank}}| + 1.645\,\sigma_{\text{blank}}}{|S|}$$

**Limit of Detection (LOD)** (IUPAC 3σ criterion):
$$\text{LOD} = \frac{3\,\sigma_{\text{blank}}}{|S|}$$

**Limit of Quantification (LOQ)** (IUPAC 10σ criterion):
$$\text{LOQ} = \frac{10\,\sigma_{\text{blank}}}{|S|}$$

**Bootstrap confidence intervals** (n = 1000 iterations, 95% CI): When blank measurements are provided, σ_blank is held fixed during bootstrap resampling of the calibration data so that the CI captures only slope uncertainty. This prevents artificially narrow CIs that would result from re-estimating σ from OLS residuals in each bootstrap iteration.

**Prediction interval at LOD** (EURACHEM/CITAC CG 4 §A3): The 95% prediction interval for a new single measurement at concentration x₀ = LOD is:

$$\hat{y} \pm t_{\alpha/2,\,n-2} \cdot s_e \cdot \sqrt{1 + \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{\sum(x_i - \bar{x})^2}}$$

This interval is wider than the OLS confidence band and is the formally correct uncertainty for LOD derivation.

### 2.5 Figure of Merit (FOM)

The sensor Figure of Merit was computed as:

$$\text{FOM} = \frac{|S|}{\text{FWHM}_{\text{ref}}} \quad \text{(ppm}^{-1}\text{)}$$

where FWHM_ref is the linewidth of the reference LSPR peak from the Lorentzian fit. FOM normalises sensitivity by peak sharpness, enabling direct comparison of sensing performance across sensor platforms and analytes (Willets & Van Duyne, *Annu. Rev. Phys. Chem.* 2007; Homola, *Chem. Rev.* 2008).

### 2.6 Cross-Session Reproducibility and Stability

Cross-session reproducibility was assessed using:

- **Paired t-test** (H₀: mean Δλ_session_A = mean Δλ_session_B at matched concentrations)
- **Bland-Altman analysis**: bias ± 1.96·σ_diff limits of agreement
- **F-test for variance equality** (H₀: σ²_A = σ²_B)
- **Mann-Whitney U test** (non-parametric, for n < 10 or non-normal distributions)
- **Mann-Kendall trend test** for monotonic sensitivity/LOD drift across sessions. The non-parametric Kendall τ statistic was used in preference to OLS regression for small session counts (n < 10) where linearity cannot be assumed. Significant increasing trend (p < 0.05) indicates sensor degradation.

Intra-day and inter-day relative standard deviations (RSD%) were computed per ICH Q2(R1) §4.4 (repeatability) and §4.5 (intermediate precision).

### 2.7 Software and Reproducibility

All statistical calculations were performed using the SpectraAgent platform ([FILL: repository URL], commit `[FILL: git rev-parse --short HEAD]`). Key dependencies: Python 3.10+, NumPy [FILL: version], SciPy [FILL: version], scikit-learn [FILL: version]. Session data, fitted models, and all metric outputs are archived in `output/sessions/` and linked to the exact git commit hash at time of acquisition.

---

## 3. Results (Template)

### 3.1 Calibration Performance

| Analyte | S (nm ppm⁻¹) | SE(S) | R² | R² (LOOCV) | FWHM_ref (nm) | FOM (ppm⁻¹) | Method |
|---------|-------------|-------|-----|-----------|--------------|-------------|--------|
| [FILL]  | [FILL]      | [FILL]| [FILL] | [FILL] | [FILL]    | [FILL]      | OLS / WLS |

Mandel linearity test: F = [FILL], p = [FILL] ([FILL: linear / nonlinear at α = 0.05]).

Residual diagnostics: Durbin-Watson = [FILL] ([FILL: no autocorrelation / autocorrelation detected]); Shapiro-Wilk W = [FILL], p = [FILL] ([FILL: normal / non-normal]); Breusch-Pagan p = [FILL] ([FILL: homoscedastic → OLS retained / heteroscedastic → WLS applied]).

### 3.2 Detection Limits

| Analyte | NEC (ppm) | LOB (ppm) | LOD (ppm) | 95% CI | LOQ (ppm) | LOL (ppm) | σ_blank (nm) | N_blank |
|---------|-----------|-----------|-----------|--------|-----------|-----------|-------------|---------|
| [FILL]  | [FILL]    | [FILL]    | [FILL]    | [[FILL], [FILL]] | [FILL] | [FILL] | [FILL] | [FILL] |

Hierarchy check: NEC ≤ LOB ≤ LOD ≤ LOQ — [FILL: ALL PASS / see note].

σ_blank source: [FILL: N_blank dedicated blank measurements / OLS residuals (note: blank measurements preferred)].

### 3.3 Cross-Session Reproducibility

| Sessions | Paired t p | Bland-Altman bias (nm) | LoA (nm) | F-test p | MK τ | MK trend | LOD RSD% |
|----------|-----------|----------------------|---------|---------|------|---------|---------|
| [FILL]   | [FILL]    | [FILL]               | [[FILL],[FILL]] | [FILL] | [FILL] | [FILL] | [FILL] |

Intra-day RSD (repeatability): [FILL]% (n = [FILL] replicates at [FILL] ppm).
Inter-day RSD (intermediate precision): [FILL]% over [FILL] days.

---

## Supplementary Information Checklist

For journal submission include:

- [ ] Session manifests (`output/sessions/*/session_meta.json`)
- [ ] Git commit hash at time of acquisition
- [ ] Raw calibration data CSV with concentration, Δλ, Δλ CI for each frame
- [ ] Blank measurement data (all N_blank Δλ values)
- [ ] Residual diagnostics report (Durbin-Watson, Shapiro-Wilk, Breusch-Pagan values)
- [ ] Bootstrap CI parameters (n_bootstrap, confidence level, fix_noise_std flag)
- [ ] Prediction interval at LOD (to distinguish from confidence band)
- [ ] Temperature and humidity per session
- [ ] FOM calculation inputs (S, FWHM_ref, source of FWHM fit)
- [ ] WLS weight specification (if WLS was applied)
- [ ] Mann-Kendall τ, p-value, n_sessions for any multi-session stability claim
- [ ] Integrity verification output (`research_integrity_gate.py --self-check`)
- [ ] Software version and dependency list (`pip freeze > requirements_snapshot.txt`)

---

## References to Cite

- IUPAC 2012: Long & Winefordner (1983) + IUPAC Commission recommendations
- ICH Q2(R1) (2005): *Validation of Analytical Procedures: Text and Methodology*
- EURACHEM/CITAC CG 4 (2019): *Quantifying Uncertainty in Analytical Measurement*
- Breusch & Pagan (1979): *Econometrica* — heteroscedasticity test
- Shapiro & Wilk (1965): *Biometrika* — normality test
- Mandel (1964): *The Statistical Analysis of Experimental Data*
- Bland & Altman (1986): *The Lancet* — method agreement
- Mann (1945) + Kendall (1975): non-parametric trend test
- Willets & Van Duyne (2007): *Annu. Rev. Phys. Chem.* 58, 267–297 — FOM definition
- Homola (2008): *Chem. Rev.* 108, 462–493 — SPR sensing fundamentals
- ISO 5725-2:2019 — accuracy (trueness and precision) methodology
