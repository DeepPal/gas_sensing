# Gas Analysis Run Summary

## Calibration Results

- **Slope**: 0.0009 nm/ppm
- **Intercept**: 0.1356 nm
- **R²**: 0.9642
- **RMSE**: 0.0006 nm
- **LOD**: 2.1801 ppm
- **R² (LOOCV)**: 0.9026
- **RMSE (LOOCV)**: 0.0010 nm
- **LOQ**: 7.2669 ppm
- **ROI Center**: 0.1365 nm

## Aggregated Spectra

- **Noise metrics**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\metrics\noise_metrics.json`
- **Aggregated summary CSV**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\metrics\aggregated_summary.csv`
- **Concentration response metrics**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\metrics\concentration_response.json`
- **Band-wise regressions CSV**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\metrics\debug_all_wavelength_regressions.csv`
- **Per-trial aggregated plots folder**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\plots\aggregated`

### ROI Selection Details
- Selection metric: hybrid
- r2_weight: 0.55
- band_half_width: 12
- band_window: 5
- min_corr: 0.5
- min_abs_slope: 0.0005
- adaptive_band: enabled=True, slope_fraction=0.75, max_half_width=50
- ROI: 606.57–613.94 nm
- Max R²: 0.2589
- Max slope @ λ: 609.33 nm
- Top candidates:
  - λ=500.07 nm, R²=0.0068, score=0.0000
  - λ=899.91 nm, R²=0.0767, score=0.0000
  - λ=899.67 nm, R²=0.1111, score=0.0000
  - λ=899.43 nm, R²=0.1433, score=0.0000
  - λ=899.18 nm, R²=0.1781, score=0.0000
- Response plot: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\plots\concentration_response.png`

#### Concentration Response
![](C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\plots\concentration_response.png)
- Full-scan response metrics: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\metrics\fullscan_concentration_response.json`
- Full-scan response plot: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\plots\fullscan_concentration_response.png`

#### Full-scan Concentration Response
![](C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\plots\fullscan_concentration_response.png)
- **Quality control summary**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\metrics\qc_summary.json`
- **Canonical overlay plot**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\plots\canonical_overlay.png`
![Canonical overlay](..\plots\canonical_overlay.png)
- **Deconvolution (MCR-ALS)**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\metrics\deconvolution_mcr_als.json`
![MCR-ALS components](..\plots\mcr_als_components.png)
![MCR-ALS predicted vs actual](..\plots\mcr_als_pred_vs_actual.png)
- **PLSR predicted vs actual**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\plots\plsr_pred_vs_actual.png`
![PLSR predicted vs actual](..\plots\plsr_pred_vs_actual.png)
- **PLSR coefficients**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\plots\plsr_coefficients.png`
![PLSR coefficients](..\plots\plsr_coefficients.png)
- **Concentration response plot**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\plots\concentration_response.png`
![Concentration response](..\plots\concentration_response.png)
- **ROI repeatability plot**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\plots\roi_repeatability.png`
![ROI repeatability](..\plots\roi_repeatability.png)
- **Multivariate model selection**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\metrics\multivariate_selection.json`

### Multivariate Selection
- Best by CV R²: plsr (0.3903)

| Model | CV R² | RMSE | Selected |
|---|---:|---:|:---:|
| PLSR | 0.3903 | 2.9454 |  |
| ICA | NA | NA |  |
| MCR-ALS | -0.1624 | 4.0670 |  |

- Selection policy: min_r2_cv=0.8, improve_margin=0.02, prefer_plsr_on_tie=True
- CV R² comparison plot: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\plots\multivariate_cv_r2.png`
![Multivariate CV R² Comparison](..\plots\multivariate_cv_r2.png)
- **ROI performance metrics**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\metrics\roi_performance.json`
- **Dynamics summary**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\metrics\dynamics_summary.json`
- **Dynamics plot**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\dynamics\response_recovery.png`
![Dynamics](..\dynamics\response_recovery.png)
- **Run metadata**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\metrics\run_metadata.json`

### ROI Performance Snapshot

- slope=0.002012 dT/ppm, R²=0.445, LOD=29.356 ppm, LOQ=97.854 ppm

### Dynamics Overview

- Overall: T90=678.71s ± 21.27s, T10=861.81s ± 1.02s
- 11.0 ppm: T90=688.46s ± 31.63s, T10=860.40s ± 2.73s
- 12.0 ppm: T90=689.04s ± 31.47s, T10=861.25s ± 1.20s
- 13.0 ppm: T90=665.98s ± 1.15s, T10=862.21s ± 0.51s
- 14.0 ppm: T90=688.46s ± 31.63s, T10=862.21s ± 0.85s
- 15.0 ppm: T90=688.58s ± 33.45s, T10=861.85s ± 0.00s
- 51.0 ppm: T90=688.70s ± 33.29s, T10=861.97s ± 1.54s
- 52.0 ppm: T90=665.16s ± 0.00s, T10=861.97s ± 0.17s
- 53.0 ppm: T90=690.11s ± 34.62s, T10=861.73s ± 1.54s
- 54.0 ppm: T90=688.93s ± 33.62s, T10=862.58s ± 0.34s
- 55.0 ppm: T90=666.79s ± 0.33s, T10=861.85s ± 0.00s
- 101.0 ppm: T90=664.93s ± nans, T10=863.06s ± nans
- 102.0 ppm: T90=666.56s ± nans, T10=861.85s ± nans
- 103.0 ppm: T90=666.09s ± nans, T10=862.09s ± nans
- 104.0 ppm: T90=664.46s ± nans, T10=860.40s ± nans
- 105.0 ppm: T90=665.39s ± nans, T10=861.85s ± nans

## Recommendations

- Keep analysis within ROI [500, 900] nm; treat outside as contextual only.
- Align prior: set expected_center≈609.3 nm (current 500.0±40.0).
- Reduce smoothing window (e.g., 11) to avoid shifting band apex visually.
- Selection gating: min_r2_cv=0.80, improve_margin=0.02. Relax slightly if no model is selected, or keep for conservatism.
- If CV R² is modest, add more concentration levels/replicates to stabilize LOOCV.
- Use Top-5 R² in `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\metrics\debug_all_wavelength_regressions.csv` to finalize per-gas ROI centers (expect ±5–10 nm stability).

### Files per Concentration

- **0.1**
  - `Multi mix vary-MeOH-0.1-1`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\0.1\Multi_mix_vary-MeOH-0.1-1.csv`
  - `Multi mix vary-MeOH-0.1-2`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\0.1\Multi_mix_vary-MeOH-0.1-2.csv`
  - `Multi mix vary-MeOH-0.1-3`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\0.1\Multi_mix_vary-MeOH-0.1-3.csv`
  - `Multi mix vary-MeOH-0.1-4`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\0.1\Multi_mix_vary-MeOH-0.1-4.csv`
  - `Multi mix vary-MeOH-0.1-5`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\0.1\Multi_mix_vary-MeOH-0.1-5.csv`

- **0.5**
  - `Multi mix vary-MeOH-0.5-1`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\0.5\Multi_mix_vary-MeOH-0.5-1.csv`
  - `Multi mix vary-MeOH-0.5-2`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\0.5\Multi_mix_vary-MeOH-0.5-2.csv`
  - `Multi mix vary-MeOH-0.5-3`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\0.5\Multi_mix_vary-MeOH-0.5-3.csv`
  - `Multi mix vary-MeOH-0.5-4`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\0.5\Multi_mix_vary-MeOH-0.5-4.csv`
  - `Multi mix vary-MeOH-0.5-5`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\0.5\Multi_mix_vary-MeOH-0.5-5.csv`

- **1**
  - `Multi mix vary-MeOH-1-1`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\1\Multi_mix_vary-MeOH-1-1.csv`
  - `Multi mix vary-MeOH-1-2`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\1\Multi_mix_vary-MeOH-1-2.csv`
  - `Multi mix vary-MeOH-1-3`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\1\Multi_mix_vary-MeOH-1-3.csv`
  - `Multi mix vary-MeOH-1-4`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\1\Multi_mix_vary-MeOH-1-4.csv`
  - `Multi mix vary-MeOH-1-5`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\1\Multi_mix_vary-MeOH-1-5.csv`

- **5**
  - `Multi mix vary-MeOH-5-1`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\5\Multi_mix_vary-MeOH-5-1.csv`
  - `Multi mix vary-MeOH-5-2`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\5\Multi_mix_vary-MeOH-5-2.csv`
  - `Multi mix vary-MeOH-5-3`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\5\Multi_mix_vary-MeOH-5-3.csv`
  - `Multi mix vary-MeOH-5-4`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\5\Multi_mix_vary-MeOH-5-4.csv`
  - `Multi mix vary-MeOH-5-5`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\5\Multi_mix_vary-MeOH-5-5.csv`

- **10**
  - `Multi mix vary-MeOH-10-1`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\10\Multi_mix_vary-MeOH-10-1.csv`
  - `Multi mix vary-MeOH-10-2`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\10\Multi_mix_vary-MeOH-10-2.csv`
  - `Multi mix vary-MeOH-10-3`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\10\Multi_mix_vary-MeOH-10-3.csv`
  - `Multi mix vary-MeOH-10-4`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\10\Multi_mix_vary-MeOH-10-4.csv`
  - `Multi mix vary-MeOH-10-5`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\meoh_topavg\aggregated\10\Multi_mix_vary-MeOH-10-5.csv`
