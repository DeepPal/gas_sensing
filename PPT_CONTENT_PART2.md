# PPT Content Part 2: Section D – Data & Pipeline (Slides 13-20)

---

# Section D: Data & Processing Pipeline (Slides 13–20)

---

## Slide 13 – What Raw Data Look Like

**Bullets:**
- **Format:** CSV (wavelength, intensity)
- **Range:** 198.8–1029.7 nm
- **Points:** 3,648 per spectrum
- **Resolution:** ~0.227 nm/point

**Directory Structure:**
```
Kevin_Data/
├── Acetone/
│   ├── air1.csv              # Reference
│   ├── 1ppm/T1/, T2/, T3/    # Trials
│   ├── 3ppm/...
│   ├── 5ppm/...
│   └── 10ppm/...
├── Ethanol/
└── ...
```

**Figure:** Example raw spectrum plot (200–1000 nm)

**Speaker Notes:**  
"Each spectrum is a simple CSV with wavelength and intensity. We have about 3,600 data points per spectrum covering 200 to 1000 nanometers."

---

## Slide 14 – Dataset Structure & Scale

**Bullets:**
- **Total:** ~40,000 spectrum files
- **Size:** ~3.6 GB

**Per-Gas Breakdown:**

| Gas | Files | Concentrations | Trials |
|-----|-------|----------------|--------|
| Acetone | 7,650 | 1, 3, 5, 10 ppm | 3 each |
| Ethanol | 7,980 | 1, 3, 5, 10 ppm | 3 each |
| Methanol | 6,240 | 1, 3, 5, 10 ppm | 3 each |
| Isopropanol | 7,039 | 1, 3, 5, 10 ppm | 3 each |
| Toluene | 2,268 | 1, 3, 5, 10 ppm | 3 each |
| Xylene | 5,954 | 1, 3, 5, 10 ppm | 3 each |

**Reference Files:**
- Acetone: `Acetone/air1.csv`
- Ethanol: `Ethanol/air for ethanol ref.csv`
- etc.

**Figure:** Table or bar chart of files per gas

---

## Slide 15 – Challenges in Raw Data

**Bullets:**
- **Detector noise:** Random intensity fluctuations
- **Lamp fluctuations:** Baseline drift over time
- **Coupling variations:** Intensity changes unrelated to gas
- **Transient responses:** Non-equilibrium during gas introduction/removal
- **Bad frames:** Occasional spikes or dropouts

**Consequence of Naive Analysis:**
- High noise → unstable slopes
- Fixed ROI → suboptimal sensitivity
- All frames → includes transients

**Solution:** Robust processing pipeline with temporal gating and data-driven ROI

**Figure:** Time series showing drift, noise, transients marked

**Speaker Notes:**  
"Raw data has several challenges. If we simply average all frames and use a fixed ROI, we get poor results. This is why we developed a careful processing pipeline."

---

## Slide 16 – Pre-processing: Transmittance & Absorbance

**Bullets:**

**Step 1: Transmittance**
```
T(λ) = I_sample(λ) / I_reference(λ)
```

**Step 2: Absorbance (Beer-Lambert)**
```
A(λ) = -log₁₀[T(λ)]
```

**Step 3: Smoothing (Savitzky-Golay)**
- Window: 7 points
- Polynomial order: 2

**Configuration:**
```yaml
analysis:
  primary_signal: absorbance
  signal_strategies:
    absorbance:
      smooth:
        enabled: true
        method: savgol
        window: 7
        poly_order: 2
```

**Figure:** Three-panel: Raw intensity → Transmittance → Absorbance

**Speaker Notes:**  
"We transform raw intensity to absorbance using Beer-Lambert law. This gives a signal linear with concentration. Light smoothing reduces noise while preserving features."

---

## Slide 17 – Temporal Gating & Frame Selection

**Bullets:**

**Problem:** Not all frames are equally informative

**Solution: Two-Stage Selection**

**Stage 1: Temporal Segmentation**
- Identify phases: baseline, rising, plateau, falling, recovery
- Keep only **plateau (stable) region**

**Stage 2: Top-K Selection**
- Rank frames by response strength
- Select top K frames (K=4)
- Average for final spectrum
- **Improves SNR**

**Configuration:**
```yaml
stability:
  top_k: 4
  min_block: 12
  diff_threshold: 0.05
```

**Figure:** Time series with stable region highlighted, selected frames marked

**Speaker Notes:**  
"We first identify the stable plateau region, then select the frames with strongest response. This top-K selection improves signal-to-noise ratio."

---

## Slide 18 – Wavelength-Window (ROI) Discovery

**Bullets:**

**Problem:** Which wavelength region gives best Δλ vs concentration?

**Our Approach: Systematic Scan**
- Range: 500–900 nm
- Window widths: 5, 8, 10, 15, 20 nm
- Step: 0.5 nm
- **~385 candidates evaluated**

**For Each Window:**
1. Extract spectra
2. Calculate Δλ (centroid method)
3. Fit linear regression
4. Calculate R², Spearman ρ, SNR

**Selection Criteria (Hierarchical):**
1. LOOCV R²
2. R²
3. Standard deviation
4. |Slope|

**Gates:**
```yaml
gates:
  min_r2: 0.7
  min_consistency: 0.75
  min_snr: 2.0
  min_abs_slope: 0.005
```

**Figure:** `output/publication_figures/Figure3_roi_discovery.png`

**Speaker Notes:**  
"Instead of assuming a fixed ROI, we scan the entire spectrum to find the optimal window. This data-driven approach consistently selects 595–625 nm (centroid mode) for the scientific pipeline, delivering higher slope and a 0.75 ppm LoD compared to the paper's 675–689 nm region." 

---

## Slide 19 – From ROI to Δλ Extraction

**Bullets:**

**Centroid Method:**
```
λ_centroid = Σ(λᵢ × Iᵢ) / Σ(Iᵢ)
```

**Wavelength Shift:**
```
Δλ = λ_centroid(sample) - λ_centroid(reference)
```

**Calibration Fitting:**
```
Δλ = S × C + b

where:
  S = Sensitivity (nm/ppm)
  C = Concentration (ppm)
  b = Intercept
```

**Metrics Calculated:**
- R², adjusted R²
- Standard error of slope
- Spearman ρ
- LoD = 3.3 × σ_residuals / |S|
- LoQ = 10 × σ_residuals / |S|

**Figure:** Spectrum with ROI highlighted, centroid marked, Δλ arrow

**Speaker Notes:**  
"Once we've selected the optimal ROI, we calculate wavelength shift using the centroid method. This gives sub-pixel resolution and is robust to noise."

---

## Slide 20 – Pipeline Summary (Flowchart)

**Bullets:**

**Pipeline Flow:**
```
RAW DATA (CSV)
    ↓
PRE-PROCESSING
• Transmittance: T = I_sample / I_ref
• Absorbance: A = -log₁₀(T)
• Smoothing: Savitzky-Golay
    ↓
TEMPORAL GATING
• Identify stable plateau
• Top-K frame selection (K=4)
• Average selected frames
    ↓
ROI DISCOVERY
• Scan 500-900 nm
• Evaluate R², ρ, SNR
• Select best window
    ↓
Δλ EXTRACTION
• Centroid calculation
• Δλ = λ_sample - λ_ref
    ↓
CALIBRATION
• Linear fit: Δλ = S×C + b
• Calculate R², LoD, ρ
• LOOCV validation
    ↓
OUTPUTS
• Calibration curve
• Metrics JSON
• Summary report
```

**Key Principles:**
1. **Data-driven:** ROI selected from data
2. **Robust:** Multiple gates and validation
3. **Reproducible:** All parameters in config
4. **Auditable:** Metadata logged

**Figure:** Clean block diagram of pipeline

**Speaker Notes:**  
"Here's the complete pipeline. Raw data goes through preprocessing, temporal gating, ROI discovery, wavelength shift extraction, and calibration. The key innovations are data-driven ROI discovery and top-K frame selection."
