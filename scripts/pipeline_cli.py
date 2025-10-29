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

    print("Running full pipeline...")
    print(f"  data_dir: {data_dir}")
    print(f"  ref_path: {ref_path}")
    print(f"  out_root: {out_root}")
    print(f"  diff_threshold: {args.diff_threshold}")

    result = run_full_pipeline(
        root_dir=data_dir,
        ref_path=ref_path,
        out_root=out_root,
        diff_threshold=args.diff_threshold,
        avg_top_n=args.avg_top_n,
        scan_full=args.scan_full,
        top_k_candidates=args.top_k,
    )

    calib = result.get("calibration", {})
    outputs = result.get("outputs", {})
    print("\nCalibration summary")
    print("-------------------")
    sel = str(calib.get("selected_model", ""))
    if sel.endswith("_cv"):
        try:
            r2cv = float(calib.get("uncertainty", {}).get("r2_cv", float("nan")))
            print(f"selected_model: {sel} (CV R^2={r2cv:.4f})")
        except Exception:
            print(f"selected_model: {sel}")
    for k in ["slope", "intercept", "r2", "rmse", "lod", "loq", "roi_center"]:
        if k in calib:
            print(f"{k}: {calib[k]}")
    # Selected predictions artifacts
    try:
        sel_csv = outputs.get("selected_predictions_csv", None)
        sel_plot = outputs.get("selected_pred_vs_actual_plot", None)
        if sel_csv:
            print(f"selected_predictions_csv: {sel_csv}")
        if sel_plot:
            print(f"selected_pred_vs_actual_plot: {sel_plot}")
    except Exception:
        pass
    if args.predict_dir:
        print("\nBatch prediction")
        print("----------------")
        pred_path = pipeline_mod.predict_batch_with_selected_model(args.predict_dir, ref_path, out_root)
        if pred_path:
            print(f"predictions_batch_csv: {pred_path}")
        else:
            print("No predictions generated (check selection or inputs)")

    print("\nOutputs written under:", out_root)


if __name__ == "__main__":
    main()
