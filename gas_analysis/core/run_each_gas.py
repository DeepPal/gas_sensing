import argparse
from pathlib import Path
from typing import Iterable, List

from config.config_loader import load_config
from gas_analysis.core import pipeline as pl

BASE = Path("Kevin_Data")
JOBS = {
    "Acetone": (
        BASE / "Acetone",
        BASE / "Acetone" / "air1.csv",
        Path("output") / "acetone_topavg",
    ),
    "Ethanol": (
        BASE / "Ethanol",
        BASE / "Ethanol" / "air for ethanol ref.csv",
        Path("output") / "ethanol_topavg",
    ),
    "Isopropanol": (
        BASE / "Isopropanol",
        BASE / "Isopropanol" / "Air_ref for IPA.csv",
        Path("output") / "isopropanol_topavg",
    ),
    "Methanol": (
        BASE / "Methanol",
        BASE / "Methanol" / "air ref after purging _N2.csv",
        Path("output") / "methanol_topavg",
    ),
    "Toluene": (
        BASE / "Toluene",
        BASE / "Toluene" / "toluene_ref air.csv",
        Path("output") / "toluene_topavg",
    ),
    "Xylene": (
        BASE / "Xylene",
        BASE / "Xylene" / "air ref xylene.csv",
        Path("output") / "xylene_topavg",
    ),
    "MixVOC": (
        BASE / "mix VOC",
        BASE / "mix VOC" / "air.csv",
        Path("output") / "mixvoc_topavg",
    ),
}


def _resolve_gases(gas_args: Iterable[str]) -> List[str]:
    if not gas_args:
        return list(JOBS.keys())
    selected: List[str] = []
    for entry in gas_args:
        # Allow comma-separated lists as a convenience
        tokens = [token.strip() for token in entry.split(',') if token.strip()]
        for token in tokens:
            if token not in JOBS:
                raise ValueError(f"Unknown gas label '{token}'. Valid options: {', '.join(JOBS.keys())}")
            if token not in selected:
                selected.append(token)
    return selected if selected else list(JOBS.keys())


def run_job(
    label: str,
    data_dir: Path,
    ref_path: Path,
    out_dir: Path,
    *,
    config_path: Path | None,
    diff_threshold: float | None,
    avg_top_n: int | None,
    scan_full: bool,
    top_k: int,
) -> None:
    config = load_config(str(config_path)) if config_path is not None else load_config()
    pl.CONFIG = config

    stability_cfg = (config.get('stability') or {}) if isinstance(config, dict) else {}
    diff_threshold_val = diff_threshold if diff_threshold is not None else float(stability_cfg.get('diff_threshold', 0.01))

    print(f"\n=== {label} ===")
    result = pl.run_full_pipeline(
        root_dir=str(data_dir.resolve()),
        ref_path=str(ref_path.resolve()),
        out_root=str(out_dir.resolve()),
        diff_threshold=diff_threshold_val,
        avg_top_n=avg_top_n,
        scan_full=scan_full,
        top_k_candidates=top_k,
        dataset_label=label,
    )

    if ((config.get('roi', {}) or {}).get('shift', {}) or {}).get('minimal_outputs', False):
        print("  (minimal_outputs=True — diagnostic plots may be suppressed)")

    print(f"Outputs → {out_dir}")
    print("Top full-scan candidates (main):", result.get("fullscan_concentration_response_metrics"))
    for mode, payload in (result.get("top_avg_results") or {}).items():
        print(f"Top full-scan candidates ({mode}): {payload.get('fullscan_metrics_path')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the gas analysis pipeline for one or more gases.",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        help="Optional YAML configuration to load instead of config/config.yaml.",
    )
    parser.add_argument(
        "--gas",
        dest="gases",
        action="append",
        help="Gas label to process (EtOH, IPA, MeOH, MIX). Can be repeated or comma-separated. Defaults to all.",
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=None,
        help="Frame-to-frame normalised difference threshold for stability filtering (default: uses config value)",
    )
    parser.add_argument(
        "--avg-top-n",
        type=int,
        default=10,
        help="Average the first N frames per concentration (default: 10). Use 0 to disable averaging.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top ROI candidates to record (default: 10)",
    )
    parser.add_argument(
        "--scan-full",
        dest="scan_full",
        action="store_true",
        help="Enable full-spectrum candidate scan (default)",
    )
    parser.add_argument(
        "--no-scan-full",
        dest="scan_full",
        action="store_false",
        help="Disable the full-spectrum candidate scan",
    )
    parser.set_defaults(scan_full=True)

    args = parser.parse_args()

    try:
        selected_gases = _resolve_gases(args.gases)
    except ValueError as exc:
        parser.error(str(exc))

    config_path = Path(args.config_path).resolve() if args.config_path else None
    if config_path and not config_path.is_file():
        parser.error(f"Configuration file not found: {config_path}")

    avg_top_n = None if args.avg_top_n is None else max(0, args.avg_top_n)

    for label in selected_gases:
        data_dir, ref_path, out_dir = JOBS[label]
        run_job(
            label,
            data_dir,
            ref_path,
            out_dir,
            config_path=config_path,
            diff_threshold=args.diff_threshold,
            avg_top_n=avg_top_n if avg_top_n > 0 else None,
            scan_full=args.scan_full,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()