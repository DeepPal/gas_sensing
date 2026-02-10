import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


GAS_OUTPUT_DIRS: Dict[str, str] = {
    "Acetone": "acetone_topavg",
    "Ethanol": "ethanol_topavg",
    "Isopropanol": "isopropanol_topavg",
    "Methanol": "methanol_topavg",
    "Toluene": "toluene_topavg",
    "Xylene": "xylene_topavg",
    "MixVOC": "mixvoc_topavg",
}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _compute_lod_from_candidate(candidate: Dict[str, Any]) -> Optional[float]:
    """Estimate LOD (ppm) from residual std and slope for a discovery candidate.

    LOD ≈ 3 * sigma_residual / |slope|.
    """
    slope = _safe_float(candidate.get("slope_nm_per_ppm"))
    residuals = candidate.get("residuals_nm") or []
    if slope is None or slope == 0 or not residuals:
        return None

    res = np.asarray(residuals, dtype=float)
    res = res[np.isfinite(res)]
    if res.size == 0:
        return None

    if res.size >= 2:
        sigma = float(res.std(ddof=1))
    else:
        sigma = float(res.std())
    if not math.isfinite(sigma) or sigma <= 0:
        return None

    lod = 3.0 * sigma / abs(slope)
    return float(lod) if math.isfinite(lod) else None


def _find_best_roi_candidate(discovery: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the ROI candidate with maximum |slope| among quality_ok=True entries."""
    best: Optional[Dict[str, Any]] = None
    best_abs_slope: float = -1.0

    for cand in discovery.get("candidates", []) or []:
        if not isinstance(cand, dict):
            continue
        if not bool(cand.get("quality_ok", False)):
            continue
        slope = _safe_float(cand.get("slope_nm_per_ppm"))
        r2 = _safe_float(cand.get("r2"))
        if slope is None or r2 is None or r2 < 0:
            continue
        if abs(slope) > best_abs_slope:
            best = cand
            best_abs_slope = abs(slope)

    return best


def _find_roi_at_center(discovery: Dict[str, Any], center_nm: float, tol_nm: float = 0.5) -> Optional[Dict[str, Any]]:
    """Find candidate whose center is within tol_nm of center_nm (closest if multiple)."""
    candidates = discovery.get("candidates", []) or []
    best: Optional[Dict[str, Any]] = None
    best_dist: float = float("inf")

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        c = _safe_float(cand.get("center_nm"))
        if c is None:
            continue
        dist = abs(c - center_nm)
        if dist <= tol_nm and dist < best_dist:
            best = cand
            best_dist = dist

    return best


def _base_row(gas: str) -> Dict[str, Any]:
    return {
        "gas": gas,
        "status": None,
        "canonical_center_nm": None,
        "canonical_slope_nm_per_ppm": None,
        "canonical_r2": None,
        "canonical_lod_ppm": None,
        "best_roi_center_nm": None,
        "best_roi_slope_nm_per_ppm": None,
        "best_roi_r2": None,
        "best_roi_lod_ppm": None,
        "best_roi_snr": None,
        "best_roi_consistency": None,
        "acetone_roi_center_nm": None,
        "slope_at_acetone_roi_nm_per_ppm": None,
        "r2_at_acetone_roi": None,
        "lod_at_acetone_roi_ppm": None,
    }


def _summarize_gas(
    gas: str,
    out_root: Path,
    acetone_roi_center: Optional[float],
    roi_match_tol: float,
) -> Dict[str, Any]:
    row = _base_row(gas)
    reasons: List[str] = []

    metrics_dir = out_root / "metrics"
    calib_path = metrics_dir / "calibration_metrics.json"
    roi_path = metrics_dir / "roi_discovery.json"

    if not out_root.is_dir():
        reasons.append("missing_output_dir")
        row["status"] = ";".join(reasons)
        return row

    if not calib_path.is_file():
        reasons.append("missing_calibration")
    if not roi_path.is_file():
        reasons.append("missing_roi_discovery")
    if reasons and ("missing_calibration" in reasons or "missing_roi_discovery" in reasons):
        row["status"] = ";".join(reasons)
        return row
    calib = _load_json(calib_path)
    discovery = _load_json(roi_path)

    canonical = calib.get("canonical_model") or {}
    can_center = None
    peaks = canonical.get("peak_wavelengths") or []
    if peaks:
        can_center = _safe_float(peaks[0])
    if can_center is None:
        can_center = _safe_float(calib.get("roi_center"))

    can_slope = _safe_float(canonical.get("slope_nm_per_ppm"))
    can_r2 = _safe_float(canonical.get("r2"))
    can_lod = _safe_float(canonical.get("lod_ppm"))

    # Best ROI among discovery candidates
    best_roi = _find_best_roi_candidate(discovery)
    if best_roi is not None:
        best_center = _safe_float(best_roi.get("center_nm"))
        best_slope = _safe_float(best_roi.get("slope_nm_per_ppm"))
        best_r2 = _safe_float(best_roi.get("r2"))
        best_snr = _safe_float(best_roi.get("snr"))
        best_consistency = _safe_float(best_roi.get("consistency"))
        best_lod = _compute_lod_from_candidate(best_roi)
    else:
        reasons.append("no_quality_roi")
        best_center = best_slope = best_r2 = best_snr = best_consistency = best_lod = None

    # Response at acetone-optimal ROI (if provided)
    slope_at_acetone = None
    r2_at_acetone = None
    lod_at_acetone = None
    if acetone_roi_center is not None:
        cand_ac = _find_roi_at_center(discovery, acetone_roi_center, tol_nm=roi_match_tol)
        if cand_ac is not None:
            slope_at_acetone = _safe_float(cand_ac.get("slope_nm_per_ppm"))
            r2_at_acetone = _safe_float(cand_ac.get("r2"))
            lod_at_acetone = _compute_lod_from_candidate(cand_ac)

    row.update(
        {
            "canonical_center_nm": can_center,
            "canonical_slope_nm_per_ppm": can_slope,
            "canonical_r2": can_r2,
            "canonical_lod_ppm": can_lod,
            "best_roi_center_nm": best_center,
            "best_roi_slope_nm_per_ppm": best_slope,
            "best_roi_r2": best_r2,
            "best_roi_lod_ppm": best_lod,
            "best_roi_snr": best_snr,
            "best_roi_consistency": best_consistency,
            "acetone_roi_center_nm": acetone_roi_center,
            "slope_at_acetone_roi_nm_per_ppm": slope_at_acetone,
            "r2_at_acetone_roi": r2_at_acetone,
            "lod_at_acetone_roi_ppm": lod_at_acetone,
        }
    )

    row["status"] = ";".join(reasons) if reasons else "ok"
    return row


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    import csv

    if not rows:
        return

    fieldnames = [
        "gas",
        "status",
        "canonical_center_nm",
        "canonical_slope_nm_per_ppm",
        "canonical_r2",
        "canonical_lod_ppm",
        "best_roi_center_nm",
        "best_roi_slope_nm_per_ppm",
        "best_roi_r2",
        "best_roi_lod_ppm",
        "best_roi_snr",
        "best_roi_consistency",
        "acetone_roi_center_nm",
        "slope_at_acetone_roi_nm_per_ppm",
        "r2_at_acetone_roi",
        "lod_at_acetone_roi_ppm",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _plot_best_roi_slopes(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    gases = []
    slopes = []
    lods = []

    for row in rows:
        s = _safe_float(row.get("best_roi_slope_nm_per_ppm"))
        if s is None:
            continue
        gases.append(row["gas"])
        slopes.append(s)
        lods.append(_safe_float(row.get("best_roi_lod_ppm")))

    if not gases:
        return

    x = np.arange(len(gases))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, slopes, color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(gases, rotation=45, ha="right")
    ax.set_ylabel("Slope (nm/ppm)")
    ax.set_title("Best-ROI Sensitivity per Gas")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "cross_gas_best_roi_slopes.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_acetone_roi_slopes(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    gases = []
    slopes = []

    for row in rows:
        s = _safe_float(row.get("slope_at_acetone_roi_nm_per_ppm"))
        if s is None:
            continue
        gases.append(row["gas"])
        slopes.append(s)

    if not gases:
        return

    x = np.arange(len(gases))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, slopes, color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(gases, rotation=45, ha="right")
    ax.set_ylabel("Slope at acetone ROI (nm/ppm)")
    ax.set_title("Response of Each Gas at Acetone-Optimal ROI")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "cross_gas_at_acetone_roi_slopes.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _parse_list_argument(entries: Optional[List[str]]) -> List[str]:
    tokens: List[str] = []
    if not entries:
        return tokens
    for entry in entries:
        for token in entry.split(','):
            token = token.strip()
            if token:
                tokens.append(token)
    return tokens


def _resolve_gases(include: List[str], exclude: List[str]) -> List[str]:
    available = list(GAS_OUTPUT_DIRS.keys())
    if include:
        selected: List[str] = []
        for name in include:
            if name not in GAS_OUTPUT_DIRS:
                raise ValueError(f"Unknown gas '{name}'. Valid options: {', '.join(available)}")
            if name not in selected:
                selected.append(name)
    else:
        selected = available.copy()

    for name in exclude:
        if name not in GAS_OUTPUT_DIRS:
            raise ValueError(f"Unknown gas '{name}'. Valid options: {', '.join(available)}")
        if name in selected:
            selected.remove(name)

    if not selected:
        raise ValueError("No gases selected after applying include/exclude filters")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-gas calibration and ROI discovery metrics into a selectivity summary",
    )
    parser.add_argument(
        "--gas",
        dest="gases",
        action="append",
        help="Gas label(s) to include (comma-separated or repeated). Defaults to all.",
    )
    parser.add_argument(
        "--exclude",
        dest="excludes",
        action="append",
        help="Gas label(s) to exclude (comma-separated or repeated).",
    )
    parser.add_argument(
        "--acetone-roi",
        type=float,
        default=None,
        help="Override acetone reference ROI center (nm). If omitted, auto-detected from Acetone output.",
    )
    parser.add_argument(
        "--roi-match-tol",
        type=float,
        default=0.6,
        help="Tolerance (nm) when matching other gases to the acetone ROI center (default: 0.6 nm).",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (defaults to repo root inferred from this script).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override output directory (defaults to <project_root>/output).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        include = _parse_list_argument(args.gases)
        exclude = _parse_list_argument(args.excludes)
        selected_gases = _resolve_gases(include, exclude)
    except ValueError as exc:
        print(f"[CROSS-GAS] {exc}")
        return

    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parent.parent
    output_root = Path(args.output_root).resolve() if args.output_root else project_root / "output"

    acetone_roi_center: Optional[float] = args.acetone_roi
    if acetone_roi_center is None:
        acetone_dir = output_root / GAS_OUTPUT_DIRS["Acetone"]
        acetone_roi_path = acetone_dir / "metrics" / "roi_discovery.json"
        if acetone_roi_path.is_file():
            acetone_discovery = _load_json(acetone_roi_path)
            selected = acetone_discovery.get("selected") or {}
            acetone_roi_center = _safe_float(selected.get("center_nm"))
        else:
            print("[CROSS-GAS] Warning: Acetone ROI discovery file not found; acetone ROI reference unavailable.")

    rows: List[Dict[str, Any]] = []
    for gas in selected_gases:
        rel_dir = GAS_OUTPUT_DIRS[gas]
        out_root = output_root / rel_dir
        summary = _summarize_gas(gas, out_root, acetone_roi_center, args.roi_match_tol)
        rows.append(summary)

    if not rows:
        print("[CROSS-GAS] No gas outputs found; run the pipeline first.")
        return

    os.makedirs(output_root, exist_ok=True)

    table_path = output_root / "cross_gas_selectivity.csv"
    _write_csv(rows, table_path)

    _plot_best_roi_slopes(rows, output_root)
    _plot_acetone_roi_slopes(rows, output_root)

    print(f"[CROSS-GAS] Wrote summary table: {table_path}")
    print(f"[CROSS-GAS] Plots saved in: {output_root}")


if __name__ == "__main__":
    main()
