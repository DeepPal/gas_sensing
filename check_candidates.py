import json

with open('output/acetone_scientific/metrics/calibration_metrics.json', 'r') as f:
    data = json.load(f)

print('Top 5 candidates with quality gates:')
for i, cand in enumerate(data['roi_scan']['top_10_candidates'][:5]):
    print(f'{i+1}. ROI {cand["roi_range"]} nm:')
    print(f'   R²: {cand["r2"]:.4f} (≥0.95: {cand["r2"] >= 0.95})')
    print(f'   Slope: {cand["slope"]:.4f} nm/ppm')
    print(f'   Score: {cand["score"]:.4f}')
    print()
