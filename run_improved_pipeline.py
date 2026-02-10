"""
Run the improved gas sensing pipeline with all enhancements:
- T90/T10 response time calculation
- Langmuir saturation model
- Automatic summary table generation
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the complete improved pipeline."""
    
    print("=" * 80)
    print("🚀 RUNNING IMPROVED GAS SENSING PIPELINE")
    print("=" * 80)
    print()
    print("Improvements included:")
    print("  ✅ T90/T10 response time calculation")
    print("  ✅ Langmuir saturation model (auto-selected by AIC)")
    print("  ✅ Multivariate analysis disabled (2× speed-up)")
    print("  ✅ Increased baseline frames (20 → better LOD)")
    print("  ✅ Automatic summary table generation")
    print()
    print("=" * 80)
    print()
    
    # Step 1: Run the main pipeline
    print("[STEP 1/2] Running gas sensing pipeline...")
    print()
    
    cmd = [
        sys.executable, "-m", "gas_analysis.core.run_each_gas",
        "--gas", "Acetone",
        "--avg-top-n", "6",
        "--top-k", "6"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print()
        print("✅ Pipeline completed successfully!")
        print()
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline failed with error code {e.returncode}")
        sys.exit(1)
    
    # Step 2: Generate summary table
    print("[STEP 2/2] Generating results summary table...")
    print()
    
    output_dir = Path("output") / "acetone_topavg"
    
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        sys.exit(1)
    
    try:
        from gas_analysis.utils.generate_summary_table import generate_results_summary
        
        summary_path = generate_results_summary(str(output_dir))
        print()
        print(f"✅ Summary table generated: {summary_path}")
        print()
    except Exception as e:
        print(f"⚠️ Failed to generate summary table: {e}")
        print("   (Pipeline results are still available in output directory)")
    
    # Step 3: Display key results
    print("=" * 80)
    print("📊 KEY RESULTS")
    print("=" * 80)
    print()
    
    try:
        import json
        
        # Load calibration metrics
        calib_path = output_dir / "metrics" / "calibration_metrics.json"
        if calib_path.exists():
            with open(calib_path, 'r') as f:
                calib = json.load(f)
            
            print(f"Model: {calib.get('selected_model', 'N/A').upper()}")
            print(f"Slope: {calib.get('slope', 0):.4f} nm/ppm")
            print(f"R²: {calib.get('r2', 0):.4f}")
            print(f"LOD: {calib.get('lod', 0):.2f} ppm")
            print()
            
            # Show Langmuir info if selected
            if calib.get('selected_model') == 'langmuir':
                lang = calib.get('langmuir_model', {})
                print("Langmuir Parameters:")
                print(f"  a = {lang.get('parameter_a', 0):.4f} ± {lang.get('parameter_a_se', 0):.4f}")
                print(f"  b = {lang.get('parameter_b', 0):.6f} ± {lang.get('parameter_b_se', 0):.6f}")
                print(f"  AIC = {lang.get('aic', 0):.2f}")
                print()
        
        # Load dynamics metrics
        dynamics_path = output_dir / "metrics" / "dynamics_summary.json"
        if dynamics_path.exists():
            with open(dynamics_path, 'r') as f:
                dynamics = json.load(f)
            
            t90 = dynamics.get('T90_mean_s')
            t90_std = dynamics.get('T90_std_s')
            t10 = dynamics.get('T10_mean_s')
            t10_std = dynamics.get('T10_std_s')
            
            if t90:
                print(f"T90 (Response): {t90:.1f} ± {t90_std:.1f} s")
            if t10:
                print(f"T10 (Recovery): {t10:.1f} ± {t10_std:.1f} s")
            print()
    
    except Exception as e:
        print(f"⚠️ Could not load results: {e}")
    
    print("=" * 80)
    print("✅ ALL STEPS COMPLETED!")
    print("=" * 80)
    print()
    print("📁 Output files:")
    print(f"   - Calibration metrics: {output_dir}/metrics/calibration_metrics.json")
    print(f"   - Dynamics summary: {output_dir}/metrics/dynamics_summary.json")
    print(f"   - Results summary: {output_dir}/RESULTS_SUMMARY.md")
    print(f"   - Main plot: {output_dir}/plots/calibration.png")
    print()
    print("📖 Next steps:")
    print("   1. Review RESULTS_SUMMARY.md for detailed analysis")
    print("   2. Check plots/ directory for visualizations")
    print("   3. Examine time_series/ for raw data")
    print()


if __name__ == '__main__':
    main()
