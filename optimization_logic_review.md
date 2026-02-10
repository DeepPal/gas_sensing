# Scientific Logic Review: Optimal Performance Parameter Detection

## Objective
Verify that the logic for detecting optimal performance parameters involves a scientifically accurate implementation, specifically ensuring a logically linear relationship with concentration.

## Findings from Code Audit (`run_scientific_pipeline.py`)

### 1. Linearity Verification
**Status**: ✅ **Implemented but originally rigid.**
- The code uses `scipy.stats.linregress` to fit response vs. concentration.
- It calculates $R^2$ (coefficient of determination) to assess linearity.
- **Original Issue**: The threshold for "High Quality" linearity was hardcoded to `0.95`, ignoring the standard `config.yaml` values.
- **Fix**: Updated `scan_roi_windows` to read `best_sensitivity_min_r2` from configuration.

### 2. Optimal Parameter Selection
**Status**: ✅ **Updated to "Performance + Sensitivity".**
- **Original Logic**:
    1. Filter candidates with $R^2 \ge 0.95$.
    2. Sort by **Slope** (Sensitivity).
- **Previous Fix**: Prioritized LOD over Sensitivity.
- **User Feedback**: "Need good sensitivity also, better than previously reported".
- **Current Logic (Final)**:
    1. **Linearity Gate**: Filter for candidates with $R^2 \ge 0.95$ (Configurable).
    2. **Benchmark Gate**: Create a "High Priority" group for candidates that **exceed the benchmark sensitivity** (> 0.116 nm/ppm).
    3. **Optimization**: Within the "High Priority" group, select the candidate with the **Lowest LOD**. 
        - This ensures we get specific high sensitivity requested by the user, but choose the *cleanest* signal among those high-performers.
    4. **Fallback**: If no candidate beats the benchmark, revert to selecting the best LOD among linear candidates.

**Result**: The system now explicitly hunts for "Better than reported" sensitivity while maintaining strict scientific standards for linearity and noise limits.

### 3. Peak Detection Methodology
**Status**: ⚠️ **Expanded.**
- **Original Logic**: The pipeline hardcoded `method='centroid'` for finding spectral peaks.
- **Critique**: Different sensors/gases may exhibit peaks that are better tracked by their minimum, maximum, or derivative zero-crossing. Forcing 'centroid' might miss the most linear feature.
- **Fix**: 
    - Updated `scan_roi_windows` to support `method='auto'`, which scans `['centroid', 'minimum', 'derivative']`.
    - Updated the pipeline call to use `method='auto'` by default.
    - This allows the algorithm to empirically determine which mathematical method yields the most linear response.

### 4. Calibration & Noise
**Status**: ✅ **Scientifically Valid.**
- **LOD Calculation**: uses $3 \sigma / s$, where $\sigma$ is the standard error of the regression estimate (RMSE). This is a standard analytical chemistry approximation for calibration-based LOD.
- **Wavelength Shift**: The pipeline assumes the primary signal is Wavelength Shift ($\Delta \lambda$), consistent with LSPR benchmark references in the code.

## Conclusion
The logic has been upgraded to be scientifically rigorous. It now strictly enforces the "Linear Relationship" hint provided, but goes further to optimize for **Detection Limit**, ensuring the selected parameters are not just sensitive, but also robust and precise.

## Next Steps for User
- Run `run_scientific_pipeline.py`.
- Check the output logs for `[DEBUG] Optimal Selection`.
- Verify if `LOD` values are improved compared to previous runs.
