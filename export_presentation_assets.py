#!/usr/bin/env python3
"""Bundle canonical figures/metrics/text for the presentation toolkit."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

REFERENCE_LOD_PPM = 3.26  # Sensors & Actuators B baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export canonical assets for the Kevin_acetone_ppt project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gas",
        default="Acetone",
        help="Gas name whose scientific outputs should be exported (ignored if --gases is provided)",
    )
    parser.add_argument(
        "--gases",
        default=None,
        help="Comma-separated list of gases to export (e.g., 'Acetone,Ethanol'). Overrides --gas",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Override path to the gas-specific scientific output directory",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("dist/presentation_assets"),
        help="Destination folder that will contain the exported bundle",
    )
    parser.add_argument(
        "--sync-to",
        type=Path,
        help="Optional path to mirror the exported assets (e.g., Kevin_acetone_ppt/generated_assets/exported)",
    )
    parser.add_argument(
        "--unified-results",
        type=Path,
        default=Path("UNIFIED_RESULTS.md"),
        help="Path to the canonical unified results markdown file",
    )
    parser.add_argument(
        "--publication-figures",
        type=Path,
        default=Path("output/publication_figures"),
        help="Directory containing publication-ready figures",
    )
    parser.add_argument(
        "--reference-lod",
        type=float,
        default=REFERENCE_LOD_PPM,
        help="Reference LoD (ppm) used to compute improvement factors",
    )
    return parser.parse_args()


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dest: Path, manifest: List[Dict], source_label: str = None) -> None:
    if not src.exists():
        return
    for item in src.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(src)
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        manifest.append(_manifest_entry(item if source_label is None else Path(source_label), target))


def copy_file(src: Path, dest: Path, manifest: List[Dict]) -> None:
    if not src.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    manifest.append(_manifest_entry(src, dest))


def _manifest_entry(src: Path, dest: Path) -> Dict:
    return {
        "source": str(src.resolve()),
        "destination": str(dest.resolve()),
        "sha256": sha256(dest),
        "size_bytes": dest.stat().st_size,
    }


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_metrics(metrics_path: Path) -> Dict:
    with metrics_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_key_figures(metrics: Dict, reference_lod: float) -> Dict:
    centroid = metrics["calibration_wavelength_shift"]["centroid"]
    roi_range = metrics.get("roi_range", centroid.get("roi_range"))
    lod = centroid["lod_ppm"]
    improvement = reference_lod / lod if lod else None
    return {
        "gas": metrics.get("gas", "Unknown"),
        "timestamp": metrics.get("timestamp"),
        "pipeline_version": metrics.get("pipeline_version"),
        "roi_nm": roi_range,
        "sensitivity_nm_per_ppm": centroid["slope"],
        "r_squared": centroid["r2"],
        "spearman_r": centroid.get("spearman_r"),
        "lod_ppm": lod,
        "loq_ppm": centroid.get("loq_ppm"),
        "noise_std": centroid.get("noise_std"),
        "loocv_r2": centroid.get("r2_cv"),
        "loocv_rmse": centroid.get("rmse_cv"),
        "confidence_interval_nm_per_ppm": centroid.get("slope_ci_95"),
        "reference_lod_ppm": reference_lod,
        "lod_improvement_x": improvement,
    }


def write_json(data: Dict, dest: Path, manifest: List[Dict], source: Path | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    manifest.append(_manifest_entry(source if source else dest, dest))


def export_assets(args: argparse.Namespace) -> None:
    gases = (
        [g.strip() for g in args.gases.split(',') if g.strip()]
        if args.gases
        else [args.gas]
    )

    dest_root = args.dest
    ensure_clean_dir(dest_root)

    overall_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "destination": str(dest_root.resolve()),
        "gases": [],
    }

    for gas in gases:
        gas_slug = gas.replace(' ', '_')
        gas_dest = dest_root / gas_slug
        gas_dest.mkdir(parents=True, exist_ok=True)

        gas_output_dir = (
            args.results_dir
            if args.results_dir is not None and len(gases) == 1
            else Path("output") / "scientific" / gas
        )
        plots_dir = gas_output_dir / "plots"
        metrics_dir = gas_output_dir / "metrics"
        reports_dir = gas_output_dir / "reports"

        manifest_entries: List[Dict] = []

        copy_tree(plots_dir, gas_dest / "plots", manifest_entries)
        copy_tree(metrics_dir, gas_dest / "metrics", manifest_entries)
        copy_tree(args.publication_figures, gas_dest / "publication_figures", manifest_entries)
        copy_tree(reports_dir, gas_dest / "reports", manifest_entries)
        copy_file(args.unified_results, gas_dest / "text" / "unified_results.md", manifest_entries)

        calibration_metrics_path = metrics_dir / "calibration_metrics.json"
        if calibration_metrics_path.exists():
            metrics = load_metrics(calibration_metrics_path)
            key_figures = build_key_figures(metrics, args.reference_lod)
            write_json(key_figures, gas_dest / "text" / "key_figures.json", manifest_entries, calibration_metrics_path)
        else:
            print(f"Warning: {calibration_metrics_path} not found for {gas}; skipping key_figures.json")

        gas_manifest = {
            "gas": gas,
            "destination": str(gas_dest.resolve()),
            "assets": manifest_entries,
        }
        write_json(gas_manifest, gas_dest / "manifest.json", manifest_entries)
        overall_manifest["gases"].append(gas_manifest)

        if args.sync_to:
            target_root = args.sync_to
            gas_target = target_root / gas_slug
            ensure_clean_dir(gas_target)
            shutil.copytree(gas_dest, gas_target, dirs_exist_ok=True)

    write_json(overall_manifest, dest_root / "manifest.json", [])

    print(f"Export complete → {dest_root} ({len(gases)} gases)")


def main() -> None:
    args = parse_args()
    export_assets(args)


if __name__ == "__main__":
    main()
