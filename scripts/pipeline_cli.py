import argparse
import os
import sys
from pathlib import Path
import hashlib

# Ensure repo root is on sys.path when running from anywhere
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gas_analysis.core.pipeline import run_full_pipeline  # noqa: E402
from gas_analysis.core import pipeline as pipeline_mod
from config.config_loader import load_config
import itertools
import json
import copy


def _apply_config_overrides(overrides):
    cfg = pipeline_mod.CONFIG
    for key, value in overrides.items():
        if value is None:
            continue
        parts = key.split('.')
        node = cfg
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value


def _reset_config():
    pipeline_mod.CONFIG = load_config()


# Sweep functionality removed for minimal version


def main():
    parser = argparse.ArgumentParser(
        description="Headless gas analysis pipeline: transmittance → stability → canonical export → calibration"
    )
    parser.add_argument("--data", required=True, help="Experiment root directory (contains concentration folders)")
    parser.add_argument("--ref", required=True, help="Reference CSV path (wavelength,intensity)")
    parser.add_argument("--out", default=str(REPO_ROOT / "output"), help="Output root directory")
    parser.add_argument("--diff-threshold", type=float, default=0.01,
                        help="Frame-to-frame normalized difference threshold for stability")
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep instead of single pipeline")
    parser.add_argument("--avg-top-n", type=int, default=None,
                        help="If set, average the first N frames for both intensity-only and transmittance pipelines")
    parser.add_argument("--scan-full", action="store_true",
                        help="Also run a full-spectrum scan (ignores ROI min/max) and save metrics/plots as fullscan_*")
    parser.add_argument("--top-k", type=int, default=5, dest="top_k",
                        help="Number of top ROI candidates to record in metrics")
    parser.add_argument("--params", type=str, help="JSON string or path to JSON defining ROI parameter grid")
    parser.add_argument("--dataset-map", type=str,
                        help="Optional JSON file listing dataset/ref pairs for sweep runs")
    parser.add_argument("--predict-dir", type=str,
                        help="Optional directory containing CSV spectra for batch prediction using selected model")

    args = parser.parse_args()

    data_dir = os.path.abspath(args.data)
    ref_path = os.path.abspath(args.ref)
    out_root = os.path.abspath(args.out)

    if args.sweep:
        if not args.params:
            parser.error("--params is required when using --sweep")

        if os.path.isfile(args.params):
            with open(args.params, 'r') as f:
                param_grid = json.load(f)
        else:
            param_grid = json.loads(args.params)

        if args.dataset_map:
            with open(args.dataset_map, 'r') as f:
                datasets = json.load(f)
            if not isinstance(datasets, list):
                parser.error("--dataset-map must be a JSON list of {\"data\", \"ref\", optional \"label\"}")
        else:
            datasets = [{
                'data': data_dir,
                'ref': ref_path,
                'label': Path(data_dir).stem,
            }]

        best = sweep_hyperparameters(datasets, out_root, param_grid)
        print("\nHyperparameter sweep results")
        print("-----------------------------")
        for label, info in best.items():
            print(f"Dataset {label}:")
            print(f"  Params: {info['params']}")
            print(f"  R2: {info['r2']}")
            print(f"  LOD: {info['lod']}")
            print(f"  LOQ: {info['loq']}")
            print(f"  Output dir: {info['output_dir']}")
        return

    print("\n==============================")
    print(" Running spectral pipeline ✳ ")
    print("==============================")
    print(f"  data_dir        : {data_dir}")
    print(f"  reference_csv   : {ref_path}")
    print(f"  output_root     : {out_root}")
    print(f"  diff_threshold  : {args.diff_threshold:.4f}")

    result = run_full_pipeline(
        root_dir=data_dir,
        ref_path=ref_path,
        out_root=out_root,
        diff_threshold=args.diff_threshold,
        avg_top_n=args.avg_top_n,
        scan_full=args.scan_full,
        top_k_candidates=args.top_k,
    )

    metadata = result.get("metadata", {})
    multi_sig = metadata.get("multi_signal", {}) if isinstance(metadata, dict) else {}
    shift_meta = metadata.get("multi_signal", {}) if isinstance(metadata, dict) else {}
    minimal_outputs = bool(((metadata.get("roi_config", {}) or {}).get('shift', {}) or {}).get('minimal_outputs', False)) if isinstance(metadata, dict) else False
    best_shift_summary_path = None
    best_shift_audit_path = None
    try:
        outputs = metadata.get("outputs", {}) if isinstance(metadata, dict) else {}
        best_shift_summary_path = outputs.get("best_shift_summary")
        best_shift_audit_path = outputs.get("best_shift_audit")
    except Exception:
        pass

    print("\nPipeline mode")
    print("-------------")
    print(f"  minimal_outputs : {'ON' if minimal_outputs else 'OFF'}")
    print(f"  scan_full       : {'ON' if args.scan_full else 'OFF'}")
    print(f"  avg_top_n       : {args.avg_top_n if args.avg_top_n is not None else 'disabled'}")

    available_signals = []
    if multi_sig:
        print("\nPer-signal Δλ diagnostics")
        print("-------------------------")
        sig_map = multi_sig.get('signals', {}) if isinstance(multi_sig.get('signals', {}), dict) else {}
        available_signals = multi_sig.get('available_signals', []) if isinstance(multi_sig.get('available_signals', []), list) else []
        if not available_signals:
            available_signals = list(sig_map.keys())
        for sig, info in sig_map.items():
            info = info or {}
            best_row = info.get('best_window') or {}
            gate_flags = info.get('best_window_gate_flags') or {}
            boot = info.get('bootstrap') or {}
            slope = best_row.get('slope_nm_per_ppm')
            r2 = best_row.get('r2_w')
            qval = best_row.get('q_value')
            det_prob = boot.get('detection_probability_at_target')
            ci = boot.get('delta_lambda_nm_ci')
            print(f"  • {sig}")
            print(f"      window      : {best_row.get('start_nm', 'NA')}–{best_row.get('end_nm', 'NA')} nm")
            if slope is not None:
                print(f"      slope       : {slope:.6f} nm/ppm")
            if r2 is not None:
                print(f"      R² (LOOCV)  : {r2:.4f}")
            if qval is not None:
                print(f"      BH-q        : {qval:.3e}")
            if det_prob is not None:
                print(f"      P(Δλ≥target): {det_prob:.2%}")
            if ci is not None:
                try:
                    lo, hi = ci
                    print(f"      Δλ CI       : [{lo:.4f}, {hi:.4f}] nm")
                except Exception:
                    pass
            if gate_flags:
                passed = [k for k, v in gate_flags.items() if bool(v)]
                failed = [k for k, v in gate_flags.items() if not bool(v)]
                print(f"      gates pass  : {', '.join(passed) if passed else 'None'}")
                if failed:
                    print(f"      gates fail  : {', '.join(failed)}")

        cross = multi_sig.get('cross_signal', {}) if isinstance(multi_sig.get('cross_signal', {}), dict) else {}
        if cross:
            print("\nCross-signal agreement")
            print("----------------------")
            for pair, metrics in cross.items():
                if not isinstance(metrics, dict):
                    continue
                window_align = metrics.get('window_alignment', {})
                gate_flags = metrics.get('gate_flags', {})
                corr = metrics.get('pearson_r')
                rmse = metrics.get('rmse_delta_lambda_nm')
                bias = metrics.get('bias_nm')
                print(f"  • {pair}")
                if corr is not None:
                    print(f"      Δλ corr     : {corr:.4f}")
                if rmse is not None:
                    print(f"      Δλ RMSE     : {rmse:.4f} nm")
                if bias is not None:
                    print(f"      Δλ bias     : {bias:+.4f} nm")
                if window_align:
                    overlap = window_align.get('overlap_fraction')
                    center_delta = window_align.get('center_delta_nm')
                    if overlap is not None:
                        print(f"      window overlap : {overlap:.2%}")
                    if center_delta is not None:
                        print(f"      center Δλ       : {center_delta:+.3f} nm")
                misaligned = metrics.get('misalignment_flag')
                if misaligned is not None:
                    print(f"      misalignment   : {'YES' if misaligned else 'no'}")
                if isinstance(gate_flags, dict):
                    for sig, flags in gate_flags.items():
                        if not isinstance(flags, dict):
                            continue
                        failed = [k for k, v in flags.items() if not bool(v)]
                        if failed:
                            print(f"      {sig} gate fail : {', '.join(failed)}")

    calib = result.get("calibration", {})
    print("\nCalibration snapshot")
    print("---------------------")
    sel = str(calib.get("selected_model", ""))
    if sel:
        note = " (LOOCV)" if sel.endswith("_cv") else ""
        print(f"  model           : {sel}{note}")
    for key, label, fmt in [
        ("slope", "slope", "{:.6f}"),
        ("intercept", "intercept", "{:+.6f}"),
        ("r2", "R²", "{:.4f}"),
        ("rmse", "RMSE", "{:.4f} ppm"),
        ("lod", "LOD", "{:.4f} ppm"),
        ("loq", "LOQ", "{:.4f} ppm"),
        ("roi_center", "ROI center", "{:.3f} nm"),
    ]:
        if key in calib and calib[key] is not None:
            try:
                print(f"  {label:<14}: {fmt.format(float(calib[key]))}")
            except Exception:
                print(f"  {label:<14}: {calib[key]}")
    unc = calib.get("uncertainty", {}) if isinstance(calib.get("uncertainty", {}), dict) else {}
    if unc:
        try:
            r2cv = unc.get("r2_cv")
            rmse_cv = unc.get("rmse_cv")
            slope_ci = unc.get("slope_ci")
            if r2cv is not None:
                print(f"  R² (CV)        : {float(r2cv):.4f}")
            if rmse_cv is not None:
                print(f"  RMSE (CV)      : {float(rmse_cv):.4f} ppm")
            if slope_ci is not None and isinstance(slope_ci, (list, tuple)):
                lo, hi = slope_ci
                print(f"  slope CI       : [{float(lo):.6f}, {float(hi):.6f}] nm/ppm")
        except Exception:
            pass

    sig_map = multi_sig.get('signals', {}) if isinstance(multi_sig.get('signals', {}), dict) else {}
    best_primary = None
    primary_key = None
    def _passes_core_gates(flags: Dict[str, object]) -> bool:
        if not isinstance(flags, dict):
            return False
        required = ['fdr', 'min_abs_slope', 'min_r2_w', 'min_spearman_r', 'min_cv_r2_w']
        return all(bool(flags.get(name, False)) for name in required)

    for sig in available_signals:
        info = sig_map.get(sig)
        if not isinstance(info, dict):
            continue
        flags = info.get('best_window_gate_flags') or {}
        if _passes_core_gates(flags):
            best_primary = info
            primary_key = sig
            break
    if best_primary is None:
        # Fall back to any signal data if no candidate passes all gates
        for sig, info in sig_map.items():
            if isinstance(info, dict):
                best_primary = info
                primary_key = sig
                break
    if best_primary:
        best_row = best_primary.get('best_window') or {}
        print("\nBest-window recap")
        print("-----------------")
        print(f"  signal         : {best_primary.get('signal_label', primary_key or 'NA')}")
        print(f"  window         : {best_row.get('start_nm', 'NA')}–{best_row.get('end_nm', 'NA')} nm")
        slope = best_row.get('slope_nm_per_ppm')
        r2 = best_row.get('r2_w')
        qval = best_row.get('q_value')
        if slope is not None:
            print(f"  slope          : {slope:.6f} nm/ppm")
        if r2 is not None:
            print(f"  R² (LOOCV)     : {r2:.4f}")
        if qval is not None:
            print(f"  BH-q           : {qval:.3e}")
        boot = best_primary.get('bootstrap') or {}
        det_prob = boot.get('detection_probability_at_target')
        ci = boot.get('delta_lambda_nm_ci')
        if det_prob is not None:
            print(f"  P(Δλ≥target)   : {det_prob:.2%}")
        if ci is not None:
            try:
                lo, hi = ci
                print(f"  Δλ CI          : [{lo:.4f}, {hi:.4f}] nm")
            except Exception:
                pass
    else:
        print("\nBest-window recap")
        print("-----------------")
        print("  No multi-signal summary captured (check minimal mode or gating failures).")

    outputs = result.get("outputs", {})
    print("\nKey artifacts")
    print("-------------")
    best_plot = outputs.get("peak_shift_delta_best_plot") if isinstance(outputs, dict) else None
    if best_plot:
        print(f"  Δλ plot        : {best_plot}")
    if best_shift_summary_path:
        print(f"  Δλ summary     : {best_shift_summary_path}")
    if best_shift_audit_path:
        print(f"  Δλ audit JSON  : {best_shift_audit_path}")
    metadata_path = result.get("run_metadata") or metadata.get('run_metadata') if isinstance(metadata, dict) else None
    if metadata_path:
        print(f"  Run metadata   : {metadata_path}")
    report_path = result.get("report_artifacts", {}).get('summary_markdown') if isinstance(result.get("report_artifacts"), dict) else None
    if report_path:
        print(f"  Summary report : {report_path}")

    if args.predict_dir:
        print("\nBatch prediction")
        print("----------------")
        pred_path = pipeline_mod.predict_batch_with_selected_model(args.predict_dir, ref_path, out_root)
        if pred_path:
            print(f"  predictions_batch_csv: {pred_path}")
        else:
            print("  No predictions generated (check selection or inputs)")

    print("\nOutputs written under:", out_root)


if __name__ == "__main__":
    main()
