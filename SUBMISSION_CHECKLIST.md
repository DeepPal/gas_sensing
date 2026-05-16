# Submission Checklist — Sensors and Actuators B: Chemical

**Manuscript:** "Sensitivity-First Automated ROI Discovery for Sub-ppm Acetone Detection Using ZnO-Coated Optical Fiber Sensor: Toward Non-Invasive Diabetes Monitoring"

**Target journal:** Sensors and Actuators B: Chemical (IF ~8.5, Elsevier)

---

## MUST-DO before submission (blocks it)

### Author information

- [ ] Add all author names and affiliations (manuscript title page and cover letter)
- [ ] Add corresponding author email, phone, and institutional address to cover letter
- [ ] Add ORCID IDs for all authors (Elsevier Editorial Manager requires these)

### References

- [ ] **Reference [19]:** Replace `[TODO: Add your original ZnO-NCF publication with full citation and DOI]`
  with your prior paper: "Highly sensitive and real-time detection of acetone biomarker for
  diabetes using ZnO-coated no-core fiber sensor" — add journal, volume, pages, year, DOI
- [ ] **Reference [23]:** Replace `[TODO: Cite ROI/spectral window optimization methodology reference]`
  with a 2020–2025 paper applying data-driven spectral window selection to optical or NIR
  spectroscopy (search: "spectral region selection optimisation optical fiber" or similar)

### Ethics and clinical data

- [ ] Add IRB/ethics approval board name and approval number for patient breath data (Section S5,
  Clinical Validation Protocol — currently states "Appropriate ethics approval was obtained"
  without specifics; Elsevier requires the actual number)
- [ ] Confirm patient cohort size (N = 25 diabetic + 15 healthy = 40 total) is correct and matches
  institutional records; note in manuscript that expansion to larger cohort is planned

### Acknowledgements and funding

- [ ] Add acknowledgements section (currently missing): list funding grant numbers, institution,
  and any equipment/facility support
- [ ] Confirm no conflicts of interest (cover letter already states this — verify it is accurate)

### Code and data availability

- [ ] Make GitHub repository `https://github.com/DeepPal/gas_sensing` **public** before submission,
  OR upload to Zenodo to obtain a citable DOI (preferred for peer review reproducibility)
- [ ] Verify that `output/scientific/Acetone/` and all referenced pipeline scripts are present
  in the repository

### Suggested reviewers

- [ ] Add 3 suggested reviewer names and institutions to cover letter (currently placeholders):
  - Expert in optical fiber sensors for biomedical applications
  - Expert in spectral data analysis / chemometrics
  - Expert in breath analysis and diabetes biomarkers

### Journal formatting

- [ ] Convert manuscript to Elsevier Word template OR LaTeX (`elsarticle` class)
- [ ] Ensure figures are separate high-resolution files (300 DPI minimum, TIFF or EPS preferred)
- [ ] Submit via [Elsevier Editorial Manager](https://www.editorialmanager.com/snb/)

---

## NICE-TO-HAVE (strengthens paper but does not block submission)

### Scientific content

- [ ] Add 2–3 more recent (2023–2025) acetone sensors to Table 3 comparison
  (search: "optical fiber acetone sensor LoD 2023" or "ZnO gas sensor acetone 2024")
- [ ] Note in manuscript that the patient cohort (N = 40) is preliminary and that a larger
  prospective study is underway — this preempts reviewer concern about sample size
- [ ] Consider adding a ROI sensitivity map figure (heatmap of sensitivity across the full
  500–900 nm range) as a visual centrepiece for the algorithm contribution

### Statistical robustness

- [ ] Recalculate paired t-test values in S4.1 for the actual comparison:
  conventional ROI (675–689 nm) vs. auto-selected ROI (595–625 nm) using the
  per-replicate Δλ measurements — the current t-statistics were computed for an
  ML comparison and should be re-verified
- [ ] Confirm bootstrap CI bounds in S4.3 match the values computed by
  `run_scientific_pipeline.py` (point estimates are verified; CI widths are estimates)

---

## Scientific integrity — already resolved

These items were identified and corrected before submission:

- [x] ML/CNN claims removed — pipeline output showed feature engineering **worsened** LoD
  from 0.17 ppm to 3.60 ppm (21×); no CNN was deployed in the validated pipeline
- [x] Duplicate Section 4.5.1 fixed (renamed second occurrence to §4.5.2)
- [x] Response time corrected: 26 s T₉₀ (not 18 s; the 18 s figure was falsely attributed to ML)
- [x] Table 1 corrected: three-column comparison (Conventional / Algorithm ROI / Optimal Window)
- [x] Supplementary S1 replaced: ROI algorithm specification replaces Feature Engineering math
- [x] Supplementary S2 (1D-CNN architecture) removed entirely
- [x] Keywords updated: removed "machine learning; convolutional neural network"
- [x] LOOCV validated: R²_CV = 0.9735 at algorithm-selected ROI (595–625 nm)

---

*Last updated: 2026-05-17*
