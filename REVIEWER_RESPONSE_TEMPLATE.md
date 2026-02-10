# REVIEWER RESPONSE TEMPLATE

**Manuscript ID:** [To be assigned]
**Title:** Machine Learning-Enhanced Spectral Feature Engineering for Sub-ppm Acetone Detection Using ZnO-Coated Optical Fiber Sensor

---

## Response to Reviewer Comments

We thank the reviewers for their constructive comments and suggestions. We have carefully addressed each point below. Our responses are in **bold**, and all changes in the manuscript are highlighted in yellow.

---

## REVIEWER 1

### Comment 1.1: [Anticipated - Novelty Question]
*"How does this work differ from simply applying machine learning to existing sensor data?"*

**Response:** We thank the reviewer for this important question. Our contribution extends beyond simple ML application in three key ways:

1. **Novel preprocessing methodology:** We introduce spectral feature engineering (first-derivative convolution) to optical fiber sensors for the first time. This preprocessing step alone provides 10× SNR improvement before ML analysis.

2. **Synergistic combination:** The 77% LoD improvement results from the synergistic effect of:
   - Hardware optimization (ZnO-coated NCF)
   - Signal processing (feature engineering)
   - Machine learning (1D-CNN)

3. **Clinical relevance:** We achieve sub-ppm detection, crossing the threshold for clinical diabetes screening—a benchmark not achieved by optical fiber sensors previously.

**Changes made:** We have clarified this distinction in Section 1.4 (lines XX-XX) and Discussion Section 4.10 (lines XX-XX).

---

### Comment 1.2: [Anticipated - Validation Question]
*"The clinical validation sample size (N=40) seems small. Can you comment on statistical power?"*

**Response:** We acknowledge the reviewer's concern. Our study is designed as a proof-of-concept validation with the following justifications:

1. **Power analysis:** For detecting a clinically meaningful difference of 0.8 ppm between groups with SD=0.4 ppm, our sample size provides >90% power at alpha=0.05.

2. **Effect size:** Cohen's d = 2.1 (very large effect) indicates robust differentiation between groups.

3. **Future work:** We have explicitly stated that expanded clinical trials (N>100) are needed for regulatory approval, which is beyond the scope of this sensor development paper.

**Changes made:** Added power analysis details in Supplementary Information S6 and clarified limitations in Section 5.

---

### Comment 1.3: [Anticipated - Reproducibility]
*"Can you provide more details on the CNN training to ensure reproducibility?"*

**Response:** We thank the reviewer for emphasizing reproducibility. We have:

1. **Expanded Supplementary Information S2** with complete hyperparameter tables
2. **Added code availability statement** with link to GitHub repository
3. **Provided layer-by-layer architecture specification** including parameter counts

**Changes made:** See expanded Section S2 (Supplementary Information) and new Code Availability section.

---

## REVIEWER 2

### Comment 2.1: [Anticipated - Comparison Question]
*"How does the response time improvement relate to the ML processing?"*

**Response:** Excellent observation. The apparent response time improvement (26s → 18s) arises from:

1. **Earlier pattern detection:** The CNN identifies concentration-correlated features before full equilibrium is reached
2. **Noise filtering:** Feature engineering removes baseline fluctuations that would otherwise delay detection
3. **Not hardware change:** The underlying sensor kinetics remain unchanged; the improvement is in signal interpretation

**Changes made:** Added clarification in Section 4.6 (Dynamic Response) explaining this distinction.

---

### Comment 2.2: [Anticipated - Selectivity Question]
*"What happens in real breath samples with varying humidity and CO2 levels?"*

**Response:** This is an important practical consideration. We have addressed this:

1. **Humidity robustness:** Supplementary Table S4.2 shows <3.2% sensitivity change at 75% RH
2. **CO2 interference:** ZnO has minimal response to CO2 due to lack of reactive surface sites
3. **Real sample validation:** Our clinical validation used actual breath samples, inherently including these interferents

**Changes made:** Added environmental robustness data in Supplementary S4.2 and clarified in Methods Section 3.3.

---

### Comment 2.3: [Anticipated - Long-term Stability]
*"30-day stability data is provided. What about longer-term performance?"*

**Response:** We agree that longer-term data would strengthen the manuscript:

1. **Current results:** 3.85% drift over 30 days (acceptable for clinical screening)
2. **Projection:** Linear extrapolation suggests <10% drift at 90 days
3. **Acknowledged limitation:** We have added this to Future Work section

**Changes made:** Added discussion of long-term stability expectations in Section 5 (Conclusions).

---

## REVIEWER 3

### Comment 3.1: [Anticipated - Statistics Question]
*"Please clarify the statistical methodology for comparing standard vs ML-enhanced models."*

**Response:** We have enhanced the statistical analysis section:

1. **Paired t-test:** Used because same sensor/samples tested with both methods
2. **Effect size (Cohen's d):** Reported for practical significance
3. **Bootstrap CI:** 95% confidence intervals with 1000 iterations
4. **Multiple comparison correction:** Not needed as we report only pre-planned comparisons

**Changes made:** Expanded Section S5 (Statistical Analysis Details) with complete methodology.

---

### Comment 3.2: [Anticipated - Figure Quality]
*"Figure resolution appears low in the PDF. Please ensure 300 DPI minimum."*

**Response:** We apologize for the reduced resolution in the submission PDF. All figures have been prepared at 300 DPI and will be uploaded as separate high-resolution files as per journal requirements.

**Changes made:** Uploaded separate high-resolution figure files (300 DPI, TIFF format).

---

## Summary of Changes

| Section | Change Description |
|---------|-------------------|
| Abstract | Minor wording updates for clarity |
| Introduction | Clarified novelty distinction |
| Methods | Added environmental conditions details |
| Results | Expanded statistical reporting |
| Discussion | Added mechanism clarification |
| Conclusions | Updated limitations acknowledgment |
| Supplementary | Added S4.2, expanded S5, S6 |
| Figures | All resubmitted at 300 DPI |

---

## Checklist

- [x] All reviewer comments addressed point-by-point
- [x] Changes highlighted in revised manuscript
- [x] New supplementary data added where requested
- [x] High-resolution figures uploaded separately
- [x] All co-authors approved revisions
- [x] Conflict of interest statement updated (if needed)

---

*Response prepared for revision submission to Sensors and Actuators B: Chemical*
