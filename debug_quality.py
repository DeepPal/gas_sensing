import json

with open('output/acetone_scientific/metrics/calibration_metrics.json', 'r') as f:
    data = json.load(f)

print('Checking quality_ok flag and candidate order:')
print('Candidates sorted by score (descending):')
print()

for i, cand in enumerate(data['roi_scan']['top_10_candidates']):
    print(f'{i+1}. ROI {cand["roi_range"]} nm:')
    print(f'   Score: {cand["score"]:.6f}')
    print(f'   R²: {cand["r2"]:.4f}')
    print(f'   Slope: {cand["slope"]:.4f} nm/ppm')
    
    # Check if this candidate would pass quality gates
    min_r2 = 0.6
    min_snr = 2.0
    min_abs_slope = 0.005
    
    quality_ok = (
        cand['r2'] >= min_r2 and
        abs(cand['slope']) >= min_abs_slope
        # SNR not available in JSON output
    )
    print(f'   quality_ok (estimated): {quality_ok}')
    print()
