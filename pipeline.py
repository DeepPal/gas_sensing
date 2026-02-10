#!/usr/bin/env python3
"""Unified pipeline CLI with subcommands: run, export, refresh, check."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
DEFAULT_GASES = ["Acetone", "Ethanol", "Methanol", "Isopropanol", "Toluene", "Xylene", "MixVOC"]
GAS_DATA_MAP = {
    "Acetone": "Acetone",
    "Ethanol": "Ethanol",
    "Methanol": "Methanol",
    "Isopropanol": "Isopropanol",
    "Toluene": "Toluene",
    "Xylene": "Xylene",
    "MixVOC": "mix VOC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified gas sensing pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Pipeline command")

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run a pipeline mode")
    run_parser.add_argument(
        "mode",
        choices=["scientific", "world-class", "ml-enhanced", "comparative", "debug", "validation"],
        help="Pipeline mode to run",
    )
    run_parser.add_argument("--gas", help="Target gas (required for non-comparative modes)")
    run_parser.add_argument("--frames", type=int, default=10, help="Frames to select per trial")
    run_parser.add_argument("--output", help="Custom output directory")
    run_parser.add_argument("--config", help="Custom config.yaml path")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    # export subcommand
    export_parser = subparsers.add_parser("export", help="Export presentation assets")
    export_parser.add_argument("--gases", default=",".join(DEFAULT_GASES), help="Comma-separated gas list")
    export_parser.add_argument("--dest", type=Path, default=ROOT / "dist" / "presentation_assets", help="Export destination")
    export_parser.add_argument("--sync-to", type=Path, default=ROOT / "Kevin_acetone_ppt" / "generated_assets" / "exported", help="Optional sync target")
    export_parser.add_argument("--reference-lod", type=float, default=3.26, help="Reference LoD (ppm) for improvement factor")
    export_parser.add_argument("--unified-results", type=Path, default=ROOT / "UNIFIED_RESULTS.md", help="Unified results markdown")
    export_parser.add_argument("--publication-figures", type=Path, default=ROOT / "output" / "publication_figures", help="Publication figures folder")

    # refresh subcommand
    refresh_parser = subparsers.add_parser("refresh", help="Run pipelines + export + optional PPT generation")
    refresh_parser.add_argument("--gases", default=",".join(DEFAULT_GASES), help="Comma-separated gas list")
    refresh_parser.add_argument("--skip-scientific", action="store_true", help="Skip scientific runs")
    refresh_parser.add_argument("--skip-world-class", action="store_true", help="Skip world-class analysis")
    refresh_parser.add_argument("--skip-export", action="store_true", help="Skip export")
    refresh_parser.add_argument("--skip-ppt", action="store_true", help="Skip PPT generation")
    refresh_parser.add_argument("--sync-to", type=Path, default=ROOT / "Kevin_acetone_ppt" / "generated_assets" / "exported", help="Sync target")
    refresh_parser.add_argument("--slides-config", type=Path, default=ROOT / "Kevin_acetone_ppt" / "config" / "presentation_scientific.yaml", help="Slides config")
    refresh_parser.add_argument("--slides-flags", default="--no-google", help="Extra flags for slides CLI (quoted)")

    # check subcommand
    check_parser = subparsers.add_parser("check", help="Validate project health")
    check_parser.add_argument("--gases", default=",".join(DEFAULT_GASES), help="Comma-separated gas list")
    check_parser.add_argument("--require-scientific", action="store_true", help="Require scientific outputs")
    check_parser.add_argument("--require-world-class", action="store_true", help="Require world-class outputs")
    check_parser.add_argument("--require-export", action="store_true", help="Require exported bundles")

    return parser.parse_args()


def run_cmd(label: str, cmd: List[str], cwd: Path) -> None:
    print(f"[RUN] {label}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def cmd_run(args: argparse.Namespace) -> None:
    if args.mode in ("scientific", "ml-enhanced", "world-class", "debug"):
        if not args.gas:
            sys.exit("Error: --gas is required for mode '{args.mode}'")
        script = {
            "scientific": "run_scientific_pipeline.py",
            "ml-enhanced": "run_ml_enhanced_pipeline.py",
            "world-class": "run_world_class_analysis.py",
            "debug": "run_debug.py",
        }[args.mode]
        cmd = ["python", script, "--gas", args.gas]
        if args.frames is not None:
            cmd += ["--frames", str(args.frames)]
        if args.output:
            cmd += ["--output", args.output]
        if args.config:
            cmd += ["--config", args.config]
        if args.verbose:
            cmd += ["--verbose"]
        run_cmd(f"run:{args.mode}", cmd, ROOT)
    elif args.mode == "comparative":
        run_cmd("run:comparative", ["python", "comparative_analysis.py"], ROOT)
    elif args.mode == "validation":
        run_cmd("run:validation", ["python", "validate_installation.py"], ROOT)


def cmd_export(args: argparse.Namespace) -> None:
    cmd = [
        "python",
        "export_presentation_assets.py",
        "--gases",
        args.gases,
        "--dest",
        str(args.dest),
        "--sync-to",
        str(args.sync_to),
        "--reference-lod",
        str(args.reference_lod),
        "--unified-results",
        str(args.unified_results),
        "--publication-figures",
        str(args.publication_figures),
    ]
    run_cmd("export", cmd, ROOT)


def cmd_refresh(args: argparse.Namespace) -> None:
    gases = [g.strip() for g in args.gases.split(",") if g.strip()]
    if not args.skip_scientific:
        for gas in gases:
            cmd_run(argparse.Namespace(mode="scientific", gas=gas, frames=10, output=None, config=None, verbose=False))
    if not args.skip_world_class:
        run_cmd("world-class", ["python", "run_world_class_analysis.py"], ROOT)
    if not args.skip_export:
        cmd_export(argparse.Namespace(
            gases=args.gases,
            dest=ROOT / "dist" / "presentation_assets",
            sync_to=args.sync_to,
            reference_lod=3.26,
            unified_results=ROOT / "UNIFIED_RESULTS.md",
            publication_figures=ROOT / "output" / "publication_figures",
        ))
    if not args.skip_ppt:
        cmd = ["python", "-m", "slides_automation.cli", "--config", str(args.slides_config)]
        cmd += args.slides_flags.split()
        run_cmd("slides_automation", cmd, ROOT / "Kevin_acetone_ppt")


def cmd_check(args: argparse.Namespace) -> None:
    gases = [g.strip() for g in args.gases.split(",") if g.strip()]
    errors = []

    # Check data paths
    for gas in gases:
        data_folder = GAS_DATA_MAP.get(gas, gas)
        gas_dir = ROOT / "Kevin_Data" / data_folder
        if not gas_dir.is_dir():
            errors.append(f"Missing data dir: {gas_dir}")

    # Check scientific outputs
    if args.require_scientific:
        for gas in gases:
            out_dir = ROOT / "output" / "scientific" / gas
            if not out_dir.is_dir():
                errors.append(f"Missing scientific output: {out_dir}")
            else:
                if not (out_dir / "plots").is_dir():
                    errors.append(f"Missing plots in: {out_dir}")
                if not (out_dir / "metrics" / "calibration_metrics.json").is_file():
                    errors.append(f"Missing calibration_metrics.json for {gas}")

    # Check world-class output
    if args.require_world_class:
        wc_dir = ROOT / "output" / "world_class"
        if not wc_dir.is_dir():
            errors.append(f"Missing world-class output: {wc_dir}")

    # Check export bundles
    if args.require_export:
        exp_dir = ROOT / "dist" / "presentation_assets"
        if not exp_dir.is_dir():
            errors.append(f"Missing export bundle: {exp_dir}")
        else:
            for gas in gases:
                gas_exp = exp_dir / gas
                if not gas_exp.is_dir():
                    errors.append(f"Missing export bundle for {gas}: {gas_exp}")

    if errors:
        print("\n".join(errors))
        sys.exit(1)
    else:
        print("All checks passed.")


def main() -> None:
    args = parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "refresh":
        cmd_refresh(args)
    elif args.command == "check":
        cmd_check(args)
    else:
        sys.exit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
