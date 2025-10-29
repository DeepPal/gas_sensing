# Gas Analysis Run Summary

## Calibration Results

- **Slope**: 0.0008 nm/ppm
- **Intercept**: 0.2396 nm
- **R²**: 0.9757
- **RMSE**: 0.0005 nm
- **LOD**: 1.8209 ppm
- **R² (LOOCV)**: 0.8499
- **RMSE (LOOCV)**: 0.0011 nm
- **LOQ**: 6.0695 ppm
- **ROI Center**: 0.2422 nm

## Aggregated Spectra

- **Noise metrics**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\metrics\noise_metrics.json`
- **Aggregated summary CSV**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\metrics\aggregated_summary.csv`
- **Concentration response metrics**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\metrics\concentration_response.json`
- **Band-wise regressions CSV**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\metrics\debug_all_wavelength_regressions.csv`
- **Per-trial aggregated plots folder**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\plots\aggregated`

### ROI Selection Details
- Selection metric: hybrid
- r2_weight: 0.55
- band_half_width: 12
- band_window: 5
- min_corr: 0.5
- min_abs_slope: 0.0005
- adaptive_band: enabled=True, slope_fraction=0.75, max_half_width=50
- ROI: 522.76–528.15 nm
- Max R²: 0.2610
- Max slope @ λ: 525.46 nm
- Top candidates:
  - λ=500.18 nm, R²=0.0117, score=0.0000
  - λ=899.79 nm, R²=0.2211, score=0.0000
  - λ=899.55 nm, R²=0.1787, score=0.0000
  - λ=899.31 nm, R²=0.3234, score=0.0000
  - λ=899.07 nm, R²=0.2291, score=0.0000
- Response plot: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\plots\concentration_response.png`

#### Concentration Response
![](C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\plots\concentration_response.png)
- Full-scan response metrics: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\metrics\fullscan_concentration_response.json`
- Full-scan response plot: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\plots\fullscan_concentration_response.png`

#### Full-scan Concentration Response
![](C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\plots\fullscan_concentration_response.png)
- **Quality control summary**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\metrics\qc_summary.json`
- **Canonical overlay plot**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\plots\canonical_overlay.png`
![Canonical overlay](..\plots\canonical_overlay.png)
- **Deconvolution (MCR-ALS)**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\metrics\deconvolution_mcr_als.json`
![MCR-ALS components](..\plots\mcr_als_components.png)
![MCR-ALS predicted vs actual](..\plots\mcr_als_pred_vs_actual.png)
- **PLSR predicted vs actual**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\plots\plsr_pred_vs_actual.png`
![PLSR predicted vs actual](..\plots\plsr_pred_vs_actual.png)
- **PLSR coefficients**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\plots\plsr_coefficients.png`
![PLSR coefficients](..\plots\plsr_coefficients.png)
- **Concentration response plot**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\plots\concentration_response.png`
![Concentration response](..\plots\concentration_response.png)
- **ROI repeatability plot**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\plots\roi_repeatability.png`
![ROI repeatability](..\plots\roi_repeatability.png)
- **Multivariate model selection**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\metrics\multivariate_selection.json`

### Multivariate Selection
- Best by CV R²: plsr (0.2832)

| Model | CV R² | RMSE | Selected |
|---|---:|---:|:---:|
| PLSR | 0.2832 | 3.2291 |  |
| ICA | NA | NA |  |
| MCR-ALS | -0.2606 | 4.2822 |  |

- Selection policy: min_r2_cv=0.8, improve_margin=0.02, prefer_plsr_on_tie=True
- CV R² comparison plot: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\plots\multivariate_cv_r2.png`
![Multivariate CV R² Comparison](..\plots\multivariate_cv_r2.png)
- **ROI performance metrics**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\metrics\roi_performance.json`
- **Dynamics summary**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\metrics\dynamics_summary.json`
- **Dynamics plot**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\dynamics\response_recovery.png`
![Dynamics](..\dynamics\response_recovery.png)
- **Run metadata**: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\metrics\run_metadata.json`

### ROI Performance Snapshot

- slope=-0.000885 dT/ppm, R²=0.083, LOD=51.617 ppm, LOQ=172.055 ppm

### Dynamics Overview

- Overall: T90=710.85s ± 7.30s, T10=942.65s ± 0.91s
- 11.0 ppm: T90=712.55s ± nans, T10=941.98s ± nans
- 12.0 ppm: T90=712.55s ± nans, T10=941.98s ± nans
- 13.0 ppm: T90=712.31s ± nans, T10=941.98s ± nans
- 14.0 ppm: T90=712.55s ± nans, T10=941.98s ± nans
- 15.0 ppm: T90=713.72s ± nans, T10=941.74s ± nans
- 51.0 ppm: T90=696.82s ± 23.90s, T10=941.98s ± 0.00s
- 52.0 ppm: T90=712.43s ± 0.17s, T10=942.34s ± 0.51s
- 53.0 ppm: T90=712.43s ± 0.17s, T10=942.46s ± 1.03s
- 54.0 ppm: T90=712.43s ± 0.17s, T10=942.34s ± 0.51s
- 55.0 ppm: T90=712.43s ± 0.17s, T10=942.58s ± 0.85s
- 101.0 ppm: T90=712.31s ± nans, T10=943.91s ± nans
- 102.0 ppm: T90=711.85s ± nans, T10=943.91s ± nans
- 103.0 ppm: T90=712.31s ± nans, T10=943.91s ± nans
- 104.0 ppm: T90=711.85s ± nans, T10=944.15s ± nans
- 105.0 ppm: T90=711.85s ± nans, T10=944.15s ± nans

## Recommendations

- Keep analysis within ROI [500, 900] nm; treat outside as contextual only.
- Reduce smoothing window (e.g., 11) to avoid shifting band apex visually.
- Selection gating: min_r2_cv=0.80, improve_margin=0.02. Relax slightly if no model is selected, or keep for conservatism.
- If CV R² is modest, add more concentration levels/replicates to stabilize LOOCV.
- Use Top-5 R² in `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\metrics\debug_all_wavelength_regressions.csv` to finalize per-gas ROI centers (expect ±5–10 nm stability).

### Files per Concentration

- **0.5**
  - `0.5 ppm EtOH IPA MeOH-1`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\0.5\0.5_ppm_EtOH_IPA_MeOH-1.csv`
  - `0.5 ppm EtOH IPA MeOH-2`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\0.5\0.5_ppm_EtOH_IPA_MeOH-2.csv`
  - `0.5 ppm EtOH IPA MeOH-3`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\0.5\0.5_ppm_EtOH_IPA_MeOH-3.csv`
  - `0.5 ppm EtOH IPA MeOH-4`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\0.5\0.5_ppm_EtOH_IPA_MeOH-4.csv`
  - `0.5 ppm EtOH IPA MeOH-5`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\0.5\0.5_ppm_EtOH_IPA_MeOH-5.csv`

- **1**
  - `1 ppm EtOH IPA MeOH-1`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\1\1_ppm_EtOH_IPA_MeOH-1.csv`
  - `1 ppm EtOH IPA MeOH-2`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\1\1_ppm_EtOH_IPA_MeOH-2.csv`
  - `1 ppm EtOH IPA MeOH-3`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\1\1_ppm_EtOH_IPA_MeOH-3.csv`
  - `1 ppm EtOH IPA MeOH-4`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\1\1_ppm_EtOH_IPA_MeOH-4.csv`
  - `1 ppm EtOH IPA MeOH-5`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\1\1_ppm_EtOH_IPA_MeOH-5.csv`

- **5**
  - `5 ppm EtOH IPA MeOH-1`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\5\5_ppm_EtOH_IPA_MeOH-1.csv`
  - `5 ppm EtOH IPA MeOH-2`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\5\5_ppm_EtOH_IPA_MeOH-2.csv`
  - `5 ppm EtOH IPA MeOH-3`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\5\5_ppm_EtOH_IPA_MeOH-3.csv`
  - `5 ppm EtOH IPA MeOH-4`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\5\5_ppm_EtOH_IPA_MeOH-4.csv`
  - `5 ppm EtOH IPA MeOH-5`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\5\5_ppm_EtOH_IPA_MeOH-5.csv`

- **10**
  - `10 ppm EtOH IPA MeOH-1`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\10\10_ppm_EtOH_IPA_MeOH-1.csv`
  - `10 ppm EtOH IPA MeOH-2`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\10\10_ppm_EtOH_IPA_MeOH-2.csv`
  - `10 ppm EtOH IPA MeOH-3`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\10\10_ppm_EtOH_IPA_MeOH-3.csv`
  - `10 ppm EtOH IPA MeOH-4`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\10\10_ppm_EtOH_IPA_MeOH-4.csv`
  - `10 ppm EtOH IPA MeOH-5`: `C:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis\output\mix_topavg\aggregated\10\10_ppm_EtOH_IPA_MeOH-5.csv`
