import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def build_multi_roi_rows(calib: Dict[str, Any], discovery: Dict[str, Any], top_k: int = 4) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    canonical = calib.get("canonical_model") or {}
    roi_center = calib.get("roi_center")
    if roi_center is None:
        peaks = canonical.get("peak_wavelengths") or []
        if peaks:
            roi_center = peaks[0]

    # Canonical calibration row
    rows.append({
        "label": "canonical",
        "center_nm": _safe_float(roi_center),
        "slope_nm_per_ppm": _safe_float(canonical.get("slope_nm_per_ppm")),
        "r2": _safe_float(canonical.get("r2")),
        "rmse_nm": _safe_float(canonical.get("rmse_nm")),
        "lod_ppm": _safe_float(canonical.get("lod_ppm")),
        "snr": None,
        "consistency": None,
        "quality_ok": True,
        "source": "canonical",
    })

    candidates = discovery.get("candidates", []) or []

    # Always include the discovery "selected" candidate first (if present)
    selected = discovery.get("selected") or {}
    seen_centers: List[float] = []

    def _append_candidate_row(cand: Dict[str, Any], label_prefix: str = "roi") -> None:
        center = _safe_float(cand.get("center_nm"))
        if center is None:
            return
        # Avoid duplicate centers
        for c in seen_centers:
            if abs(c - center) < 1e-6:
                return
        seen_centers.append(center)

        slope = _safe_float(cand.get("slope_nm_per_ppm"))
        r2 = _safe_float(cand.get("r2"))
        rmse_nm = _safe_float(cand.get("rmse_nm"))
        snr = _safe_float(cand.get("snr"))
        consistency = _safe_float(cand.get("consistency"))
        quality_ok = bool(cand.get("quality_ok", False))
        lod_est = _compute_lod_from_candidate(cand)

        rows.append({
            "label": f"{label_prefix}_{center:.2f}nm",
            "center_nm": center,
            "slope_nm_per_ppm": slope,
            "r2": r2,
            "rmse_nm": rmse_nm,
            "lod_ppm": lod_est,
            "snr": snr,
            "consistency": consistency,
            "quality_ok": quality_ok,
            "source": "roi",
        })

    if isinstance(selected, dict) and selected:
        _append_candidate_row(selected, label_prefix="selected")

    # Sort remaining candidates by score and take top_k
    remaining: List[Dict[str, Any]] = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        center = _safe_float(cand.get("center_nm"))
        if center is None:
            continue
        # skip if identical to selected (already added)
        if selected and _safe_float(selected.get("center_nm")) is not None:
            if abs(center - float(selected["center_nm"])) < 1e-6:
                continue
        remaining.append(cand)

    remaining_sorted = sorted(
        remaining,
        key=lambda c: _safe_float(c.get("score")) if _safe_float(c.get("score")) is not None else float("-inf"),
        reverse=True,
    )

    if top_k is not None and top_k > 0:
        remaining_sorted = remaining_sorted[:top_k]

    for cand in remaining_sorted:
        _append_candidate_row(cand, label_prefix="roi")

    return rows


def write_multi_roi_table(rows: List[Dict[str, Any]], out_path: str) -> None:
    import csv

    if not rows:
        return

    fieldnames = [
        "label",
        "center_nm",
        "slope_nm_per_ppm",
        "r2",
        "rmse_nm",
        "lod_ppm",
        "snr",
        "consistency",
        "quality_ok",
        "source",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def plot_multi_roi(calib: Dict[str, Any], discovery: Dict[str, Any], rows: List[Dict[str, Any]], plots_dir: str) -> Optional[str]:
    abs_shift = calib.get("absolute_shift") or {}
    concs_c = abs_shift.get("concentrations") or []
    deltas_c = abs_shift.get("absolute_delta_wavelengths") or abs_shift.get("delta_wavelengths") or []

    concs_c = np.asarray(concs_c, dtype=float)
    deltas_c = np.asarray(deltas_c, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot canonical ROI if available
    if concs_c.size and deltas_c.size == concs_c.size:
        roi_center = calib.get("roi_center")
        if roi_center is None:
            cm = calib.get("canonical_model") or {}
            peaks = cm.get("peak_wavelengths") or []
            if peaks:
                roi_center = peaks[0]
        label = "canonical"
        rc_val = _safe_float(roi_center)
        if rc_val is not None:
            label = f"canonical ({rc_val:.1f} nm)"
        ax.plot(concs_c, deltas_c, "o-", label=label)

    # Build a lookup from center_nm to candidate for plotting
    cand_map: Dict[float, Dict[str, Any]] = {}
    for cand in discovery.get("candidates", []) or []:
        center = _safe_float(cand.get("center_nm"))
        if center is None:
            continue
        cand_map[center] = cand

    # Plot each ROI row based on candidates
    for row in rows:
        if row.get("source") != "roi":
            continue
        center = _safe_float(row.get("center_nm"))
        if center is None:
            continue
        # Match candidate by center
        cand = None
        for c_center, c in cand_map.items():
            if abs(c_center - center) < 1e-6:
                cand = c
                break
        if cand is None:
            continue

        concs = np.asarray(cand.get("concentrations_ppm") or [], dtype=float)
        deltas = np.asarray(cand.get("deltas_valid_nm") or cand.get("deltas_nm") or [], dtype=float)
        if concs.size == 0 or deltas.size != concs.size:
            continue

        slope = row.get("slope_nm_per_ppm")
        r2 = row.get("r2")
        label = f"{center:.1f} nm"
        if slope is not None and r2 is not None:
            label = f"{center:.1f} nm (s={slope:.3f}, R²={r2:.3f})"

        ax.plot(concs, deltas, "o-", label=label)

    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Concentration (ppm)")
    ax.set_ylabel("Δλ (nm)")
    ax.set_title("Multi-ROI Calibration (Δλ vs concentration)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    out_path = os.path.join(plots_dir, "calibration_multi_roi.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a multi-ROI summary table and Δλ vs concentration plot "
            "from calibration_metrics.json and roi_discovery.json."
        )
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Output root directory for a gas run (e.g. output/acetone_topavg)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Maximum number of additional ROI candidates (besides the selected one) to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = os.path.abspath(args.out_root)

    calib_path = os.path.join(out_root, "metrics", "calibration_metrics.json")
    roi_path = os.path.join(out_root, "metrics", "roi_discovery.json")

    if not os.path.isfile(calib_path):
        raise FileNotFoundError(f"Calibration metrics not found: {calib_path}")
    if not os.path.isfile(roi_path):
        raise FileNotFoundError(f"ROI discovery metrics not found: {roi_path}")

    calib = _load_json(calib_path)
    discovery = _load_json(roi_path)

    metrics_dir = os.path.join(out_root, "metrics")
    plots_dir = os.path.join(out_root, "plots")
    _ensure_dir(metrics_dir)
    _ensure_dir(plots_dir)

    rows = build_multi_roi_rows(calib, discovery, top_k=args.top_k)

    table_path = os.path.join(metrics_dir, "multi_roi_table.csv")
    write_multi_roi_table(rows, table_path)

    plot_path = plot_multi_roi(calib, discovery, rows, plots_dir)

    print(f"[MULTI-ROI] Wrote table: {table_path}")
    if plot_path:
        print(f"[MULTI-ROI] Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
