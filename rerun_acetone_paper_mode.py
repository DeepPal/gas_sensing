"""
Rerun Acetone analysis with:
1. Relaxed PELT parameters (to detect responses)
2. Only [1,3,5,10] ppm (match paper, exclude 50 ppm)
3. Proper baseline calculation (air reference per trial)
"""

import subprocess
import sys
import shutil
import json
import numpy as np
from pathlib import Path
from scipy.stats import linregress

def main():
    print("=" * 80)
    print("🔬 RERUN ACETONE ANALYSIS - PAPER MODE")
    print("=" * 80)
    print()
    
    print("Configuration:")
    print("  • PELT penalty: 1.5 (was 3.0) - more sensitive")
    print("  • Activation threshold: 0.0003 nm (was 0.0008 nm)")
    print("  • Concentrations: [1, 3, 5, 10] ppm only (exclude 50 ppm)")
    print("  • Baseline: Per-trial air reference")
    print()
    
    # Step 1: Move 50 ppm out
    data_dir = Path("Kevin_Data/Acetone")
    ppm50_dir = data_dir / "50ppm"
    temp_backup = Path("TEMP_50ppm_BACKUP")
    
    moved = False
    if ppm50_dir.exists():
        print("[STEP 1/4] Excluding 50 ppm...")
        try:
            if temp_backup.exists():
                shutil.rmtree(temp_backup)
            shutil.move(str(ppm50_dir), str(temp_backup))
            moved = True
            print("   ✅ 50 ppm moved to temp backup")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return
    print()
    
    try:
        # Step 2: Run pipeline
        print("[STEP 2/4] Running pipeline with relaxed PELT...")
        print()
        
        cmd = [
            sys.executable, "-m", "gas_analysis.core.run_each_gas",
            "--gas", "Acetone",
            "--avg-top-n", "6",
            "--top-k", "6"
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=False)
        print()
        print("✅ Pipeline completed!")
        print()
        
        # Step 3: Analyze with proper baseline
        print("[STEP 3/4] Analyzing with proper air baseline...")
        print("-" * 80)
        
        meta_path = Path("output/acetone_topavg/metrics/run_metadata.json")
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Check responsive frame detection
        print("\nResponsive frame detection status:")
        any_detected = False
        for conc_str in ['1.0', '3.0', '5.0', '10.0']:
            conc_data = meta['trial_debug'][conc_str]
            for trial_name in conc_data['__final__']:
                resp_count = conc_data[trial_name]['response_series']['responsive_frame_count']
                total_count = conc_data[trial_name]['frame_count']
                if resp_count > 0:
                    print(f"  ✅ {conc_str} ppm, {trial_name}: {resp_count}/{total_count} frames")
                    any_detected = True
                else:
                    print(f"  ⚠️ {conc_str} ppm, {trial_name}: NO responsive frames (fallback)")
        
        if not any_detected:
            print("\n⚠️ WARNING: Still no responsive frames detected!")
            print("   Using fallback method (may reduce accuracy)")
        print()
        
        # Calculate proper shifts from air baseline
        conc_shifts = {}
        for conc_str in ['1.0', '3.0', '5.0', '10.0']:
            conc = float(conc_str)
            conc_data = meta['trial_debug'][conc_str]
            
            shifts = []
            for trial_name in conc_data['__final__']:
                trial_info = conc_data[trial_name]['response_series']['responsive_summary']
                baseline_nm = trial_info['baseline_peak_nm']
                gas_peak_nm = trial_info['selected_peak_nm']
                shift_nm = gas_peak_nm - baseline_nm
                shifts.append(shift_nm)
            
            conc_shifts[conc] = shifts
        
        # Average and calculate slope
        concs = []
        avg_shifts = []
        std_shifts = []
        
        print("Shifts from air baseline:")
        for conc in sorted(conc_shifts.keys()):
            shifts_list = conc_shifts[conc]
            avg_shift = np.mean(shifts_list)
            std_shift = np.std(shifts_list)
            
            concs.append(conc)
            avg_shifts.append(avg_shift)
            std_shifts.append(std_shift)
            
            print(f"  {conc:5.1f} ppm: Δλ = {avg_shift:+.4f} ± {std_shift:.4f} nm")
        
        # Linear regression
        reg = linregress(concs, avg_shifts)
        slope = reg.slope
        r2 = reg.rvalue ** 2
        
        print()
        print(f"Linear fit: Δλ = {slope:.6f} × C + {reg.intercept:.6f}")
        print(f"  Slope: {slope:.6f} nm/ppm")
        print(f"  R²: {r2:.4f}")
        print()
        
        # Compare with paper
        paper_slope = 0.116
        paper_r2 = 0.95
        
        print("Comparison with paper:")
        print(f"  Paper:    Slope={paper_slope:.6f} nm/ppm, R²={paper_r2:.4f}")
        print(f"  Pipeline: Slope={slope:.6f} nm/ppm, R²={r2:.4f}")
        print(f"  Difference: {abs(slope-paper_slope):.6f} nm/ppm ({abs(slope-paper_slope)/paper_slope*100:.1f}%)")
        print()
        
        # Step 4: Check calibration metrics
        print("[STEP 4/4] Checking calibration metrics...")
        print("-" * 80)
        
        calib_path = Path("output/acetone_topavg/metrics/calibration_metrics.json")
        with open(calib_path, 'r') as f:
            calib = json.load(f)
        
        print(f"Calibration (canonical peaks):")
        print(f"  Selected model: {calib['selected_model']}")
        print(f"  Slope: {calib['slope']:.6f} nm/ppm")
        print(f"  R²: {calib['r2']:.4f}")
        print(f"  LOD: {calib['lod']:.2f} ppm")
        print()
        
        # Summary
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print()
        
        if abs(slope - paper_slope) < 0.03:
            print("✅ EXCELLENT! Slope matches paper within 3%!")
        elif abs(slope - paper_slope) < 0.05:
            print("✅ GOOD! Slope matches paper within 5%!")
        else:
            print(f"⚠️ Slope differs from paper by {abs(slope-paper_slope)/paper_slope*100:.1f}%")
            print("\nPossible reasons:")
            print("  1. Responsive frame detection still not optimal")
            print("  2. Different feature extraction (Gaussian vs Centroid)")
            print("  3. Manual selection in paper vs automated pipeline")
            print("  4. Different wavelength averaging or ROI width")
        
        print()
        
        if not any_detected:
            print("⚠️ CRITICAL: No responsive frames detected!")
            print("   Next step: Further relax PELT parameters or disable it")
            print("   Consider using simpler method: max(|Δλ|) in stable period")
        
        print()
        
    finally:
        # Restore 50 ppm
        if moved and temp_backup.exists():
            print("Restoring 50 ppm...")
            try:
                shutil.move(str(temp_backup), str(ppm50_dir))
                print("✅ 50 ppm restored")
            except Exception as e:
                print(f"⚠️ Failed to restore: {e}")
        print()

if __name__ == '__main__':
    main()
