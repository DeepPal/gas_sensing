import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def _format_float(value: Optional[float], precision: int = 3, suffix: str = "") -> str:
    if value is None or not isinstance(value, (int, float)):
        return "—"
    return f"{value:.{precision}f}{suffix}"


def _build_markdown(
    rows: List[Dict[str, str]],
    csv_path: Path,
    output_path: Path,
    reference_gas: str,
) -> str:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    ok_rows = [row for row in rows if (row.get("status") or "ok").split(';')[0] == "ok"]
    best_candidates = []
    for row in ok_rows:
        slope = _safe_float(row.get("best_roi_slope_nm_per_ppm"))
        if slope is None:
            continue
        best_candidates.append((row, abs(slope)))

    best_candidates.sort(key=lambda item: item[1], reverse=True)
    reference_row = next((row for row in rows if row.get("gas") == reference_gas), None)
    reference_slope = None
    if reference_row:
        reference_slope = _safe_float(reference_row.get("best_roi_slope_nm_per_ppm"))
        if reference_slope is not None:
            reference_slope = abs(reference_slope)

    top_line = "No valid best-ROI slopes available."
    selectivity_line = "Selectivity ratio could not be computed."
    if best_candidates:
        leader_row, leader_slope = best_candidates[0]
        top_line = (
            f"Strongest best-ROI slope: **{leader_row['gas']}** at {leader_row.get('best_roi_center_nm', '—')} nm "
            f"with |slope| = {leader_slope:.3f} nm/ppm "
            f"(R² = {_format_float(_safe_float(leader_row.get('best_roi_r2')), 3)})."
        )
        if reference_slope and len(best_candidates) > 1:
            next_best = best_candidates[1][1]
            if next_best > 0:
                ratio = reference_slope / next_best
                selectivity_line = (
                    f"Selectivity ratio (|slope|_{reference_gas} / next best) = {ratio:.2f}."
                )
        elif reference_slope is not None:
            selectivity_line = f"Selectivity ratio relative to other gases unavailable (only one valid entry)."

    response_lines = []
    for row in rows:
        slope_at_ref = _safe_float(row.get("slope_at_acetone_roi_nm_per_ppm"))
        status = row.get("status", "ok")
        msg = (
            f"- {row.get('gas')}: status = {status}, "
            f"best ROI slope = {_format_float(_safe_float(row.get('best_roi_slope_nm_per_ppm')))} nm/ppm, "
            f"LOD = {_format_float(_safe_float(row.get('best_roi_lod_ppm')), 2, ' ppm')}, "
            f"response at acetone ROI = {_format_float(slope_at_ref)} nm/ppm"
        )
        response_lines.append(msg)

    # Build markdown table (top portion)
    header = (
        "| Gas | Status | Best ROI (nm) | |Slope| (nm/ppm) | R² | LOD (ppm) | Slope @ acetone ROI (nm/ppm) |\n"
        "|------|--------|---------------|------------------|----|-----------|------------------------------|"
    )
    table_lines = [header]
    for row in rows:
        slope_best = _safe_float(row.get("best_roi_slope_nm_per_ppm"))
        table_lines.append(
            "| {gas} | {status} | {center} | {slope} | {r2} | {lod} | {slope_ref} |".format(
                gas=row.get("gas", "—"),
                status=row.get("status", "—"),
                center=_format_float(_safe_float(row.get("best_roi_center_nm")), 2),
                slope=_format_float(abs(slope_best) if slope_best is not None else None),
                r2=_format_float(_safe_float(row.get("best_roi_r2"))),
                lod=_format_float(_safe_float(row.get("best_roi_lod_ppm")), 2),
                slope_ref=_format_float(_safe_float(row.get("slope_at_acetone_roi_nm_per_ppm"))),
            )
        )

    markdown = [
        "# Cross-Gas Selectivity Report",
        "",
        f"Generated: {timestamp}",
        f"Source CSV: `{csv_path}`",
        "",
        "## Highlights",
        top_line,
        selectivity_line,
        "",
        "## Response Summary",
        *response_lines,
        "",
        "## Detailed Table",
        *table_lines,
        "",
        f"Report saved to `{output_path}`",
    ]

    return "\n".join(markdown)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown summary from cross_gas_selectivity.csv",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="output/cross_gas_selectivity.csv",
        help="Path to cross_gas_selectivity.csv (default: output/cross_gas_selectivity.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the Markdown report (default: <output_dir>/selectivity_report.md)",
    )
    parser.add_argument(
        "--reference-gas",
        type=str,
        default="Acetone",
        help="Gas to use as reference for selectivity ratios (default: Acetone)",
    )
    args = parser.parse_args()

    csv_path = Path(args.input).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Selectivity CSV not found: {csv_path}")

    rows = _load_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    output_path = Path(args.output).resolve() if args.output else csv_path.parent / "selectivity_report.md"
    markdown = _build_markdown(rows, csv_path, output_path, args.reference_gas)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"[REPORT] Wrote selectivity summary: {output_path}")


if __name__ == "__main__":
    main()
