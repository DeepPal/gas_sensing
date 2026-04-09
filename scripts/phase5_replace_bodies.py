"""Replace Phase 5 function bodies in pipeline.py with thin CONFIG-coupled wrappers."""
import re

with open("gas_analysis/core/pipeline.py", encoding="utf-8") as f:
    lines = f.readlines()

total = len(lines)
print(f"Original lines: {total}")


def find_func_start(lines, name):
    for i, line in enumerate(lines):
        if line.rstrip() == f"def {name}(":
            return i
    return None


def find_func_end(lines, start_idx):
    i = start_idx + 1
    while i < len(lines):
        line = lines[i]
        if re.match(r"^(def |class |\Z)", line) and not line.startswith(" "):
            return i
        i += 1
    return len(lines)


replacements = []

# 1. compute_environment_summary
start = find_func_start(lines, "compute_environment_summary")
end = find_func_end(lines, start)
new_body = [
    'def compute_environment_summary(\n',
    '    stable_by_conc,\n',
    ') -> dict:\n',
    '    """Compute environment compensation summary (CONFIG-coupled wrapper)."""\n',
    '    env_cfg = CONFIG.get("environment", {}) if isinstance(CONFIG, dict) else {}\n',
    '    if not env_cfg:\n',
    '        return {}\n',
    '    ref = env_cfg.get("reference", {}) or {}\n',
    '    coeffs = env_cfg.get("coefficients", {}) or {}\n',
    '    override = env_cfg.get("override", {}) or {}\n',
    '    return _compute_environment_summary_pure(\n',
    '        stable_by_conc,\n',
    '        T_ref=float(ref.get("temperature", 25.0)),\n',
    '        H_ref=float(ref.get("humidity", 50.0)),\n',
    '        cT=coeffs.get("temperature", None),\n',
    '        cH=coeffs.get("humidity", None),\n',
    '        env_enabled=bool(env_cfg.get("enabled", False)),\n',
    '        apply_to_frames=bool(env_cfg.get("apply_to_frames", False)),\n',
    '        apply_to_transmittance=bool(env_cfg.get("apply_to_transmittance", True)),\n',
    '        override_temp=override.get("temperature", None),\n',
    '        override_humid=override.get("humidity", None),\n',
    '    )\n',
    '\n',
]
replacements.append((start, end, new_body))

# 2. compute_environment_coefficients
start = find_func_start(lines, "compute_environment_coefficients")
end = find_func_end(lines, start)
new_body = [
    'def compute_environment_coefficients(\n',
    '    stable_by_conc, calib\n',
    ') -> dict:\n',
    '    """Estimate environment coefficients (CONFIG-coupled wrapper)."""\n',
    '    env_cfg = CONFIG.get("environment", {}) if isinstance(CONFIG, dict) else {}\n',
    '    ref = env_cfg.get("reference", {}) or {}\n',
    '    return _compute_environment_coefficients_pure(\n',
    '        stable_by_conc,\n',
    '        calib,\n',
    '        T_ref=float(ref.get("temperature", 25.0)),\n',
    '        H_ref=float(ref.get("humidity", 50.0)),\n',
    '    )\n',
    '\n',
]
replacements.append((start, end, new_body))

# 3. summarize_quality_control
start = find_func_start(lines, "summarize_quality_control")
end = find_func_end(lines, start)
new_body = [
    'def summarize_quality_control(\n',
    '    stable_by_conc,\n',
    '    noise_metrics,\n',
    ') -> dict:\n',
    '    """Summarise QC metrics (CONFIG-coupled wrapper)."""\n',
    '    qcfg = CONFIG.get("quality", {}) if isinstance(CONFIG, dict) else {}\n',
    '    return _summarize_quality_control_pure(\n',
    '        stable_by_conc,\n',
    '        noise_metrics,\n',
    '        min_snr=float(qcfg.get("min_snr", 10.0)),\n',
    '        max_rsd=float(qcfg.get("max_rsd", 5.0)),\n',
    '    )\n',
    '\n',
]
replacements.append((start, end, new_body))

# 4. _stack_trials_for_response — remove body (alias already imported)
start = find_func_start(lines, "_stack_trials_for_response")
end = find_func_end(lines, start)
replacements.append((start, end, []))

# 5. compute_concentration_response
start = find_func_start(lines, "compute_concentration_response")
end = find_func_end(lines, start)
new_body = [
    'def compute_concentration_response(\n',
    '    stable_by_conc,\n',
    '    override_min_wavelength=None,\n',
    '    override_max_wavelength=None,\n',
    '    top_k_candidates: int = 0,\n',
    '    debug_out_root=None,\n',
    '):\n',
    '    """Scan every wavelength for the best ROI (CONFIG-coupled wrapper)."""\n',
    '    roi_cfg = CONFIG.get("roi", {}) if isinstance(CONFIG, dict) else {}\n',
    '    validation_cfg = roi_cfg.get("validation", {}) or {}\n',
    '    alt_cfg = roi_cfg.get("alternative_models", {}) or {}\n',
    '    adp_cfg = roi_cfg.get("adaptive_band", {}) or {}\n',
    '    bhw = roi_cfg.get("band_half_width", None)\n',
    '    repeatability = CONFIG.get("_last_repeatability", {}) or {}\n',
    '    g_std = float((repeatability.get("global", {}) or {}).get("std_transmittance", 0.0) or 0.0)\n',
    '    cfg = _RoiScanConfig(\n',
    '        selection_metric=str(roi_cfg.get("selection_metric", "r2")).lower(),\n',
    '        min_r2=float(roi_cfg.get("min_r2", 0.0)),\n',
    '        r2_weight=float(roi_cfg.get("r2_weight", 1.0)),\n',
    '        expected_trend=str(roi_cfg.get("expected_trend", "any")).lower(),\n',
    '        trend_modes=roi_cfg.get("trend_modes", None),\n',
    '        min_corr=float(roi_cfg.get("min_corr", 0.0)),\n',
    '        min_wavelength=override_min_wavelength if override_min_wavelength is not None else roi_cfg.get("min_wavelength", None),\n',
    '        max_wavelength=override_max_wavelength if override_max_wavelength is not None else roi_cfg.get("max_wavelength", None),\n',
    '        band_half_width=int(bhw) if bhw is not None else None,\n',
    '        band_window=int(roi_cfg.get("band_window", 0)),\n',
    '        derivative_weight=float(roi_cfg.get("derivative_weight", 0.0)),\n',
    '        ratio_weight=float(roi_cfg.get("ratio_weight", 0.0)),\n',
    '        ratio_half_width=int(max(1, roi_cfg.get("ratio_half_width", 5))),\n',
    '        slope_noise_weight=float(roi_cfg.get("slope_noise_weight", 0.0)),\n',
    '        min_slope_to_noise=float(roi_cfg.get("min_slope_to_noise", 0.0)),\n',
    '        global_std=g_std,\n',
    '        min_abs_slope=float(roi_cfg.get("min_abs_slope", 0.0)),\n',
    '        alt_models_enabled=bool(alt_cfg.get("enabled", False)),\n',
    '        poly_degree=int(max(1, alt_cfg.get("polynomial_degree", 2))),\n',
    '        adaptive_band_enabled=bool(adp_cfg.get("enabled", False)),\n',
    '        slope_fraction=float(adp_cfg.get("slope_fraction", 0.6)),\n',
    '        adaptive_max_half_width=int(adp_cfg.get("max_half_width", bhw if bhw is not None else 20)),\n',
    '        expected_center=validation_cfg.get("expected_center"),\n',
    '        center_tolerance=float(validation_cfg.get("tolerance", 0.0)),\n',
    '        validation_notes=str(validation_cfg.get("notes", "")),\n',
    '    )\n',
    '    return _compute_concentration_response_pure(\n',
    '        stable_by_conc,\n',
    '        cfg=cfg,\n',
    '        top_k_candidates=top_k_candidates,\n',
    '        debug_out_root=debug_out_root,\n',
    '    )\n',
    '\n',
]
replacements.append((start, end, new_body))

# Apply in reverse order to preserve indices
replacements.sort(key=lambda x: x[0], reverse=True)
for start, end, new_body in replacements:
    lines[start:end] = new_body
    print(f"  Replaced lines {start+1}-{end}: {end-start} -> {len(new_body)} lines (delta={len(new_body)-(end-start):+d})")

with open("gas_analysis/core/pipeline.py", "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"Done. New total: {len(lines)} lines (was {total})")
