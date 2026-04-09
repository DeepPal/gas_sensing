"""Phase 6: Replace save_* / plot_* bodies in pipeline.py with thin wrappers.

Migrates 19 functions to src.reporting.io and src.reporting.plots.
Two functions (save_concentration_response_plot, save_spectral_response_diagnostic)
get CONFIG-reading wrappers; the rest become single-line delegates.

Run from project root:
    python scripts/phase6_replace_bodies.py
"""

import ast
import pathlib
import sys

PIPELINE = pathlib.Path("gas_analysis/core/pipeline.py")

# ---------------------------------------------------------------------------
# Import block to inject (inserted after the last existing src/ import group)
# ---------------------------------------------------------------------------
IMPORT_BLOCK = """\
from src.reporting.io import (
    save_aggregated_spectra as _save_aggregated_spectra_io,
    save_aggregated_summary as _save_aggregated_summary_io,
    save_canonical_spectra as _save_canonical_spectra_io,
    save_concentration_response_metrics as _save_concentration_response_metrics_io,
    save_dynamics_error as _save_dynamics_error_io,
    save_dynamics_summary as _save_dynamics_summary_io,
    save_environment_compensation_summary as _save_env_compensation_summary_io,
    save_noise_metrics as _save_noise_metrics_io,
    save_quality_summary as _save_quality_summary_io,
    save_roi_performance_metrics as _save_roi_performance_metrics_io,
)
from src.reporting.plots import (
    save_aggregated_plots as _save_aggregated_plots_src,
    save_calibration_outputs as _save_calibration_outputs_src,
    save_canonical_overlay as _save_canonical_overlay_src,
    save_concentration_response_plot as _save_concentration_response_plot_pure,
    save_research_grade_calibration_plot as _save_research_grade_calibration_plot_src,
    save_roi_discovery_plot as _save_roi_discovery_plot_src,
    save_roi_repeatability_plot as _save_roi_repeatability_plot_src,
    save_spectral_response_diagnostic as _save_spectral_response_diagnostic_pure,
    save_wavelength_shift_visualization as _save_wavelength_shift_visualization_src,
)
"""

ANCHOR = "from src.calibration.roi_scan import ("

# ---------------------------------------------------------------------------
# New function bodies (body only — indented 4 spaces, ends with newline)
# ---------------------------------------------------------------------------
BODIES: dict[str, str] = {
    "save_canonical_spectra": (
        "    return _save_canonical_spectra_io(canonical, out_root)\n"
    ),
    "save_aggregated_spectra": (
        "    return _save_aggregated_spectra_io(aggregated, out_root)\n"
    ),
    "save_noise_metrics": (
        "    return _save_noise_metrics_io(metrics, out_root)\n"
    ),
    "save_quality_summary": (
        "    return _save_quality_summary_io(qc, out_root)\n"
    ),
    "save_aggregated_summary": (
        "    return _save_aggregated_summary_io(aggregated, noise_metrics, out_root)\n"
    ),
    "save_roi_performance_metrics": (
        "    return _save_roi_performance_metrics_io(performance, out_root)\n"
    ),
    "save_roi_discovery_plot": (
        "    return _save_roi_discovery_plot_src(discovery, out_root)\n"
    ),
    "save_dynamics_summary": (
        "    return _save_dynamics_summary_io(summary, out_root)\n"
    ),
    "save_dynamics_error": (
        "    return _save_dynamics_error_io(message, out_root)\n"
    ),
    "save_concentration_response_metrics": (
        "    return _save_concentration_response_metrics_io(\n"
        "        response, repeatability, out_root, name=name\n"
        "    )\n"
    ),
    # CONFIG wrapper: reads roi.min_wavelength / roi.max_wavelength
    "save_concentration_response_plot": (
        '    roi_cfg = CONFIG.get("roi", {}) if isinstance(CONFIG, dict) else {}\n'
        '    x_min = roi_cfg.get("min_wavelength")\n'
        '    x_max = roi_cfg.get("max_wavelength")\n'
        "    return _save_concentration_response_plot_pure(\n"
        "        response, avg_by_conc, out_root, name=name, clamp_to_roi=clamp_to_roi,\n"
        "        x_min=float(x_min) if x_min is not None else None,\n"
        "        x_max=float(x_max) if x_max is not None else None,\n"
        "    )\n"
    ),
    "save_wavelength_shift_visualization": (
        "    return _save_wavelength_shift_visualization_src(\n"
        "        canonical, calib_result, out_root, dataset_label=dataset_label\n"
        "    )\n"
    ),
    "save_research_grade_calibration_plot": (
        "    return _save_research_grade_calibration_plot_src(\n"
        "        canonical, calib_result, out_root, dataset_label=dataset_label\n"
        "    )\n"
    ),
    # CONFIG wrapper: reads roi.shift.step_nm / roi.shift.window_nm
    "save_spectral_response_diagnostic": (
        '    roi_cfg = CONFIG.get("roi", {}) if isinstance(CONFIG, dict) else {}\n'
        '    shift_cfg = roi_cfg.get("shift", {}) if isinstance(roi_cfg.get("shift", {}), dict) else {}\n'
        '    step_nm = float(shift_cfg.get("step_nm", 2.0) or 2.0)\n'
        "    if not np.isfinite(step_nm) or step_nm <= 0:\n"
        "        step_nm = 2.0\n"
        '    window_nm = shift_cfg.get("window_nm", 10.0)\n'
        "    return _save_spectral_response_diagnostic_pure(\n"
        "        canonical, out_root, dataset_label=dataset_label,\n"
        "        wl_min=wl_min, wl_max=wl_max,\n"
        "        step_nm=step_nm, window_nm=window_nm,\n"
        "    )\n"
    ),
    "save_roi_repeatability_plot": (
        "    return _save_roi_repeatability_plot_src(stable_by_conc, response, out_root)\n"
    ),
    "save_aggregated_plots": (
        "    return _save_aggregated_plots_src(aggregated, out_root)\n"
    ),
    "save_canonical_overlay": (
        "    return _save_canonical_overlay_src(canonical, out_root)\n"
    ),
    "save_environment_compensation_summary": (
        "    return _save_env_compensation_summary_io(info, out_root)\n"
    ),
    "save_calibration_outputs": (
        "    _save_calibration_outputs_src(calib, out_root, name_suffix)\n"
    ),
}


def main() -> None:
    src = PIPELINE.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)

    # Guard: skip if already migrated
    if IMPORT_BLOCK.splitlines()[0] in src:
        print("Import block already present — already migrated, exiting.")
        return

    # -----------------------------------------------------------------------
    # 1. Parse ONCE from the original file
    # -----------------------------------------------------------------------
    tree = ast.parse(src)

    # -----------------------------------------------------------------------
    # 2. Collect replacement ranges (0-indexed)
    # -----------------------------------------------------------------------
    replacements: list[tuple[int, int, list[str], str, str]] = []
    for fname, new_body in BODIES.items():
        node = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fname),
            None,
        )
        if node is None:
            print(f"  WARNING: {fname} not found — skipping.")
            continue
        start = node.lineno - 1          # 0-indexed start (first decorator or def)
        end = node.end_lineno            # 0-indexed exclusive end
        body_start = node.body[0].lineno - 1  # 0-indexed first body line
        kept = lines[start:body_start]   # decorators + full def signature
        replacements.append((start, end, kept, new_body, fname))

    # -----------------------------------------------------------------------
    # 3. Apply replacements bottom-to-top so earlier indices stay valid
    # -----------------------------------------------------------------------
    replacements.sort(key=lambda x: x[0], reverse=True)
    total_removed = 0

    for start, end, kept, new_body, fname in replacements:
        old_len = end - start
        new_node_lines = kept + [new_body]
        delta = len(new_node_lines) - old_len
        total_removed -= delta
        lines = lines[:start] + new_node_lines + lines[end:]
        print(f"  {fname}: {old_len} -> {len(new_node_lines)} lines (delta={delta:+d})")

    # -----------------------------------------------------------------------
    # 4. Inject import block after the last existing src/ import group
    # -----------------------------------------------------------------------
    anchor_idx = next(
        (i for i, l in enumerate(lines) if l.strip().startswith(ANCHOR)), None
    )
    if anchor_idx is None:
        print(f"ERROR: anchor '{ANCHOR}' not found after replacements", file=sys.stderr)
        sys.exit(1)

    # Find the closing ')' of that import group
    close_idx = anchor_idx
    while close_idx < len(lines) and ")" not in lines[close_idx]:
        close_idx += 1

    insert_at = close_idx + 1
    lines = lines[:insert_at] + IMPORT_BLOCK.splitlines(keepends=True) + lines[insert_at:]
    print(f"  Injected Phase 6 import block after line {insert_at + 1}.")

    # -----------------------------------------------------------------------
    # 5. Sanity-check syntax and write
    # -----------------------------------------------------------------------
    new_src = "".join(lines)
    try:
        ast.parse(new_src)
    except SyntaxError as e:
        print(f"SYNTAX ERROR after edits: {e}", file=sys.stderr)
        sys.exit(1)

    PIPELINE.write_text(new_src, encoding="utf-8")
    old_count = src.count("\n")
    new_count = new_src.count("\n")
    print(f"\nDone. Lines: {old_count} -> {new_count} "
          f"(net -{old_count - new_count}, body savings: {total_removed}).")


if __name__ == "__main__":
    main()
