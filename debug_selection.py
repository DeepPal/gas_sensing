import json

with open('output/acetone_scientific/metrics/calibration_metrics.json', 'r') as f:
    data = json.load(f)

print('Debugging selection logic:')
print(f'best_sensitivity_min_r2 = 0.95')
print()

candidates = data['roi_scan']['top_10_candidates']
best_sens_candidate = None
best_abs_slope = float('-inf')

print('Checking candidates for sensitivity-first selection:')
for i, cand in enumerate(candidates[:5]):
    r2_val = cand['r2']
    slope_val = cand['slope']
    
    print(f'{i+1}. ROI {cand["roi_range"]} nm:')
    print(f'   R²: {r2_val:.4f} (≥0.95: {r2_val >= 0.95})')
    print(f'   Slope: {slope_val:.4f} nm/ppm')
    print(f'   |slope|: {abs(slope_val):.4f}')
    
    if r2_val >= 0.95:
        abs_slope = abs(slope_val)
        if abs_slope > best_abs_slope:
            best_abs_slope = abs_slope
            best_sens_candidate = cand
            print(f'   *** NEW BEST SENSITIVITY ***')
    print()

print(f'Final selection:')
if best_sens_candidate:
    print(f'Should select ROI {best_sens_candidate["roi_range"]} nm')
    print(f'Sensitivity: {best_sens_candidate["slope"]:.4f} nm/ppm')
else:
    print('No candidate met R² ≥ 0.95 threshold')

print(f'\nActual selection: ROI {data["roi_scan"]["best_roi"]} nm')
print(f'Actual sensitivity: {data["roi_scan"]["best_slope"]:.4f} nm/ppm')
