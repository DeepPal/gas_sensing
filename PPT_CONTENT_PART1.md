# PPT Content Part 1: Sections A-C (Slides 1-12)

**Project:** Sensitivity-Optimized Auto-Selection for Optical Gas Sensing  
**Generated:** 2026-01-13 (Updated with Actual Results)

---

## Master Slide Map (34 Slides + Backup)

| Section | Slides | Content |
|---------|--------|---------|
| **A. Title & Roadmap** | 1–3 | Title, Motivation, Roadmap |
| **B. Background** | 4–8 | Gas sensing challenge, ZnO-NCF, benchmark, gap |
| **C. Experimental Setup** | 9–12 | Setup, fabrication, chamber, protocol |
| **D. Algorithm** | 13–20 | Sensitivity-first selection, implementation |
| **E. Results** | 21–28 | 4.9× improvement, multi-gas validation |
| **F. Novelty** | 29–31 | Performance-driven vs location-based |
| **G. Conclusion** | 32–34 | Impact, applications, future work |
| **Backup** | 35–38 | Technical details, validation |

---

# Section A: Title & Roadmap (Slides 1–3)

---

## Slide 1 – Title Slide

**Title:**  
Sensitivity-Optimized Auto-Selection: A Novel Performance-Driven Approach for Optical Gas Sensing

**Authors:**  
[Your Name], [Co-authors]

**Affiliation:**  
[Your Institution]

**Conference/Journal:**  
Sensors & Actuators B: Chemical

**Key Achievement:**  
2.3× Sensitivity Improvement (0.116 → 0.269 nm/ppm)

---

## Slide 2 – Motivation: Why Acetone Sensing?

**Bullets:**
- **537 million adults** have diabetes globally (IDF 2021); projected 783M by 2045
- Current monitoring: **invasive blood glucose testing** → low compliance
- **Breath acetone** is a validated biomarker:
  - Healthy: 0.2–1.8 ppm
  - Pre-diabetic: 1.0–1.5 ppm
  - Diabetic: 1.25–2.5 ppm
  - Ketoacidosis: >2.5 ppm
- **Challenge:** sensors must achieve **sub-ppm detection** to distinguish healthy vs diabetic

**Figure:** Diagram: Patient → Breath → Sensor → ppm reading → Diagnosis

**Speaker Notes:**  
"Diabetes affects over half a billion people worldwide. Current monitoring requires invasive blood testing. Breath acetone offers a non-invasive alternative, but we need sub-ppm detection to distinguish healthy from diabetic individuals."

---

## Slide 3 – Talk Roadmap

**Bullets:**
1. Background & State of the Art (Slides 4–8)
2. Experimental Setup (Slides 9–12)
3. Data Processing Pipeline (Slides 13–20)
4. Results (Slides 21–28)
5. Novelty & Interpretation (Slides 29–31)
6. Conclusions & Future Work (Slides 32–34)

**Figure:** Simple flowchart or numbered list with icons

---

# Section B: Background & State of the Art (Slides 4–8)

---

## Slide 4 – Gas Sensing Basics & Key Metrics

**Bullets:**
- **Sensitivity (S):** Δλ / ΔC (nm/ppm) – signal change per ppm
- **Detection Limit (LoD):** IUPAC: LoD = 3.3σ / S
- **R²:** Goodness of linear fit (target: >0.99)
- **Response Time (T₉₀):** Time to reach 90% of final signal
- **Recovery Time (T₁₀):** Time to return to 10% of baseline
- **Spearman ρ:** Monotonicity of response (|1| = perfect)

**Figure:** Annotated calibration curve showing slope, LoD, R²

**Speaker Notes:**  
"Sensitivity tells us signal change per ppm. Detection limit is the smallest measurable concentration. R-squared indicates calibration reliability."

---

## Slide 5 – Optical Fiber Sensing Principle

**Bullets:**
- **No-core fiber (NCF):** Multimode waveguide with strong evanescent field
- **Evanescent field:** Extends ~50–200 nm beyond fiber surface
- **Sensing mechanism:**
  1. Light travels through NCF
  2. Evanescent field interacts with ZnO coating
  3. VOC adsorbs on ZnO → refractive index change (Δn)
  4. Wavelength shift observed (Δλ)
- **Equation:** Δλ = (∂λ/∂n_eff) × Δn_eff

**Figure:** Cross-section of NCF with ZnO coating, evanescent field illustrated

**Speaker Notes:**  
"Our sensor uses a no-core fiber with ZnO coating. When acetone adsorbs on the ZnO surface, it changes the local refractive index, causing a measurable wavelength shift."

---

## Slide 6 – ZnO–Acetone Interaction Chemistry

**Bullets:**
- ZnO: Wide bandgap (3.37 eV), n-type semiconductor
- **Lewis acid-base interaction:** Zn²⁺ sites coordinate with carbonyl (C=O)
- Binding energy: ~0.8 eV (vs ~0.4 eV for alcohols, ~0.1 eV for aromatics)
- **Result:** Preferential acetone adsorption → higher sensitivity

**Binding Strength Comparison:**

| Functional Group | Binding Energy | Relative Response |
|-----------------|----------------|-------------------|
| **C=O (ketone)** | ~0.8 eV | **1.00** |
| -OH (alcohol) | ~0.4 eV | 0.3–0.5 |
| C-H (aromatic) | ~0.1 eV | 0.05–0.1 |

**Figure:** Molecular diagram: acetone binding to ZnO surface

**Speaker Notes:**  
"ZnO preferentially binds acetone because the carbonyl group forms a strong Lewis acid-base complex with zinc sites. This gives inherent selectivity."

---

## Slide 7 – Reference Paper: Baseline Performance

**Bullets:**
- **Sensor:** ZnO-coated NCF (85 nm ZnO layer)
- **ROI:** 675–689 nm
- **Performance:**

| Metric | Value |
|--------|-------|
| Sensitivity | 0.116 nm/ppm |
| LoD | 3.26 ppm |
| R² | 0.95 |
| T₉₀ | 26 s |
| Recovery | 32 s |
| Drift | 0.2% (30 days) |

- **Limitation:** LoD of 3.26 ppm insufficient for clinical use

**Figure:** Table or bar chart of reference paper metrics

**Speaker Notes:**  
"This is our baseline. The sensor works well but the detection limit of 3.26 ppm is the critical limitation for clinical applications."

---

## Slide 8 – Gap & Research Objectives

**Bullets:**
- **Gap:** LoD of 3.26 ppm insufficient for clinical diabetes screening (need <1 ppm)
- **Opportunity:** ML-enhanced spectral analysis can improve detection

**Objectives:**
1. Apply spectral feature engineering to ZnO-NCF data
2. Develop data-driven ROI optimization
3. Achieve sub-ppm detection limit
4. Validate selectivity against interfering VOCs
5. Demonstrate clinical relevance

**Figure:** "Before vs After" diagram: 3.26 ppm → <1 ppm

**Speaker Notes:**  
"Our hypothesis is that careful data processing can extract much better performance from the same sensor hardware."

---

# Section C: Experimental Setup (Slides 9–12)

---

## Slide 9 – Overall Experimental Setup

**Bullets:**
- **Light source:** Halogen lamp (Ocean Optics HL-2000)
- **Spectrometer:** Thorlabs CCS200/M (200–1000 nm)
- **VOC chamber:** Acrylic, 460 cm³
- **Sensor:** ZnO-coated NCF (3.4 cm length)
- **Conditions:** 23–25°C, 55% RH, 1 atm

**Figure:** Schematic: [Lamp] → [Fiber Sensor in Chamber] → [Spectrometer] → [PC]

**Speaker Notes:**  
"Light from a halogen lamp passes through the sensor in a gas chamber. The transmitted spectrum is recorded by a spectrometer."

---

## Slide 10 – Sensor Fabrication

**Bullets:**
- **ZnO synthesis:** Sol-gel (zinc acetate + KOH, 60°C), particle size ~10 nm
- **Coating:** Spray deposition at 250°C, 0.3 mL solution
- **Annealing:** 250°C for 2 hours
- **Thickness:** 85 nm (FESEM confirmed)
- **Assembly:** Fusion splice NCF between SMF, splice loss <0.01 dB

**Figure:** Fabrication flowchart or FESEM image of ZnO coating

**Speaker Notes:**  
"We synthesize ZnO nanoparticles and spray-deposit them onto the fiber. The resulting coating is about 85 nanometers thick."

---

## Slide 11 – Gas Chamber & Flow Control

**Bullets:**
- **Chamber:** Acrylic, 460 cm³, sealed
- **Gas delivery:** N₂ carrier + controlled VOC injection
- **Concentrations:** 1, 3, 5, 10 ppm
- **Purge:** N₂ for 120 s between measurements

**VOCs Tested:**

| VOC | Relevance |
|-----|-----------|
| **Acetone** | Diabetes biomarker |
| Methanol | Toxic exposure |
| Ethanol | Alcohol consumption |
| Isopropanol | Hand sanitizer |
| Toluene | Environmental |
| Xylene | Environmental |

**Figure:** Diagram of gas chamber with flow lines

---

## Slide 12 – Measurement Protocol

**Bullets:**
- **Per concentration:**
  - Baseline: ~200 frames (air/N₂)
  - Gas exposure: ~500 frames
  - Recovery: ~200 frames
- **Acquisition:** 10 ms/spectrum
- **Trials:** 3 per concentration
- **Total spectra per gas:** ~7,000–8,000

**Dataset Summary:**

| Gas | Files | Concentrations |
|-----|-------|----------------|
| Acetone | 7,650 | 1, 3, 5, 10 ppm |
| Ethanol | 7,980 | 1, 3, 5, 10 ppm |
| Methanol | 6,240 | 1, 3, 5, 10 ppm |
| Total | ~40,000 | |

**Figure:** Timeline: [Baseline] → [Gas ON] → [Plateau] → [Gas OFF] → [Recovery]

**Speaker Notes:**  
"Each measurement follows a standard protocol with baseline, exposure, and recovery phases. We have about 40,000 spectra total."
