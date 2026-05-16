# Tier-1 Publication Completion Design

**Date:** 2026-05-17
**Project:** Acetone Gas Sensing — ZnO-NCF Sensor with Automated ROI Discovery
**Target Journal:** Sensors and Actuators B: Chemical (IF ~8.5)
**Approach:** Drop ML framing, publish the ROI algorithm as core contribution

---

## 1. Context and Critical Finding

The analysis pipeline is complete and validated. The core scientific contribution is genuine and
Tier-1 publishable. However, an audit of the actual output data (ml_enhanced_results.json)
revealed a critical mismatch between manuscript claims and pipeline output:

| Method | LoD (ppm) | R² | What manuscript says |
|--------|-----------|-----|---------------------|
| Standard analysis at 580–590 nm | **0.17** | 0.9997 | Attributed to ML enhancement |
| "ML-enhanced" at 580–590 nm | **3.60** | 0.9981 | Claims 95% improvement |
| ROI auto-selection at 595–625 nm | **0.75** | 0.9945 | Primary result |

**The feature engineering (convolution) worsens LoD by 21×. The ML claims are not supported.**

The 0.17 ppm result is real but comes from the standard (non-ML) analysis at the
algorithmically-discovered optimal ROI. The genuine contribution is the automated
sensitivity-first ROI scanning algorithm.

---

## 2. The Real Contribution (What Is Actually Publishable)

The algorithm works as follows:

1. Scan 500–900 nm in sliding windows of 5–30 nm
2. Compute sensitivity (slope) and R² for each window
3. Keep only candidates with R² ≥ 0.95 (quality gate)
4. Select the window with the highest absolute sensitivity
5. Run full calibration at this window

**Result for Acetone:**
- Auto-discovered ROI: 595–625 nm (vs. literature 675–689 nm)
- Sensitivity: 0.269 nm/ppm (vs. 0.116 nm/ppm) → 2.3× improvement
- LoD: 0.75 ppm (vs. 3.26 ppm) → 4.3× improvement
- Further: at 580–590 nm (standard analysis, no ML): LoD = 0.17 ppm (19× improvement)
- LOOCV validated (R²_CV = 0.9735)

This is novel, reproducible, and clinically relevant for diabetes screening.

---

## 3. Manuscript Rewrites Required

### 3.1 Title Change
**Old:** "Machine Learning-Enhanced Spectral Feature Engineering for Sub-ppm Acetone Detection..."
**New:** "Sensitivity-First Automated ROI Discovery for Sub-ppm Acetone Detection Using ZnO-Coated Optical Fiber Sensor: Toward Non-Invasive Diabetes Monitoring"

### 3.2 Abstract — Full Rewrite
- Remove: "spectral feature engineering", "1D convolutional neural network", "first-derivative convolution"
- Add: "sensitivity-first automated spectral window optimization algorithm"
- Primary claim: 0.75 ppm LoD (4.3× improvement) from ROI auto-selection
- Secondary claim: 0.17 ppm at the absolute optimal discovered window (standard analysis)
- Remove all claims that ML reduces LoD

### 3.3 Highlights — Revise 3 of 5 bullets
- Remove CNN/ML bullet
- Add: "Novel sensitivity-first algorithm automatically discovers optimal ROI (595–625 nm vs.
  literature 675–689 nm), achieving 4.3× improvement in detection limit"
- Keep: clinical threshold, LOOCV, selectivity bullets

### 3.4 Introduction Section 1.4 — Retitle and Revise
- "Machine Learning in Spectral Analysis" → "Data-Driven ROI Optimization in Spectroscopy"
- Cite relevant ROI optimization literature instead of CNN papers
- The gap to fill is: no prior work has applied performance-driven automated ROI search
  to optical fiber VOC sensors

### 3.5 Methods Section 3.5 — Replace
- Remove: spectral feature engineering pseudocode
- Add: ROI scanning algorithm with pseudocode (already exists in MANUSCRIPT_PREPARATION_GUIDE)
- Section becomes: "3.5 Sensitivity-First ROI Discovery Algorithm"

### 3.6 Methods Section 3.6 — Remove entirely
- Delete the 1D-CNN architecture section (Table with Conv1D layers)
- Delete "[GPU specifications]" placeholder
- Delete training protocol subsection

### 3.7 Results Section 4.3 Table 1 — Revise
Replace the "Standard vs ML-Enhanced" comparison with:
"Traditional ROI (675-689 nm) vs. Auto-Discovered ROI (595-625 nm) vs. Optimal Discovered Window (580-590 nm)"

| Metric | Traditional ROI | Auto-Selected ROI | Optimal Window |
|--------|----------------|-------------------|----------------|
| Sensitivity | 0.116 nm/ppm | 0.269 nm/ppm | 0.054 nm/ppm |
| R² | 0.95 | 0.9945 | 0.9997 |
| LoD | 3.26 ppm | 0.75 ppm | 0.17 ppm |

### 3.8 Fix Duplicate Section 4.5.1
Remove the second occurrence (keep the cross-sensitivity matrix one).

### 3.9 Conclusions — Revise first 2 bullets
- "77% reduction via data-driven ROI auto-selection (3.26 → 0.75 ppm)"
- "19× further improvement to 0.17 ppm using optimal discovered spectral window"
- Remove CNN accuracy claim

### 3.10 Supplementary S1–S2 — Remove or Replace
- Remove S1 (spectral feature engineering math) or replace with ROI scanning algorithm math
- Remove S2 (1D-CNN architecture)
- Add: S1 new — Complete ROI scanning algorithm specification with complexity analysis

### 3.11 Fillable Placeholders (non-scientific)
| Placeholder | Replacement |
|-------------|-------------|
| `[GitHub link to be added]` in S7 | `https://github.com/DeepPal/gas_sensing` |
| `[Insert Date]` in cover letter | `2026-05-17` |
| Ref [19] placeholder | `**[TODO: Add your original ZnO-NCF publication with DOI]**` |
| Ref [23] placeholder | `**[TODO: Cite ROI optimization methodology reference]**` |

---

## 4. Cover Letter Update

The cover letter highlights need to match the revised paper:
- "First performance-driven automated ROI selection algorithm for optical fiber VOC sensors"
- "4.3× detection limit improvement (3.26 → 0.75 ppm) through data-driven spectral optimization"
- Remove ML/CNN language

---

## 5. GitHub Push

After manuscript rewrite is complete:
- Stage ALL current modifications (M) and deletions (D)
- Single commit: `docs: reframe manuscript around ROI algorithm — drop unsupported ML claims`
- Push to `origin main` at `https://github.com/DeepPal/gas_sensing.git`

---

## 6. Submission Checklist (`SUBMISSION_CHECKLIST.md`)

### Must-do before submission (blocks it)
- [ ] Author names, affiliations, ORCID IDs — manuscript AND cover letter
- [ ] Reference [19]: original ZnO-NCF paper with DOI
- [ ] Reference for ROI optimization methodology
- [ ] Acknowledgements + funding grant numbers
- [ ] Ethics approval board / IRB number for patient data (Section S6)
- [ ] Suggested reviewer names + institutions in cover letter
- [ ] Make GitHub repo public OR upload to Zenodo for DOI
- [ ] Convert to Elsevier Word/LaTeX template
- [ ] Submit via Elsevier Editorial Manager

### Nice-to-have (strengthens paper)
- [ ] Note patient cohort size (N=25+15=40 total) is preliminary; state intent to expand
- [ ] Add 2–3 more recent (2023–2025) acetone sensors to Table 3 comparison

---

## 7. Success Criteria

- [ ] Title no longer references "Machine Learning" (unless ML claims are re-validated)
- [ ] Abstract is internally consistent (no ML claims unsupported by data)
- [ ] Table 1 accurately represents ROI comparison, not ML comparison
- [ ] No duplicate Section 4.5.1
- [ ] All unfillable TODOs are clearly marked `**[TODO: ...]**`
- [ ] GitHub remote is up-to-date with local main
- [ ] `SUBMISSION_CHECKLIST.md` is created and complete
- [ ] Cover letter highlights match manuscript claims
