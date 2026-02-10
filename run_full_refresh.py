#!/usr/bin/env python3
"""End-to-end automation for scientific pipeline, exporter, and optional PPT generation."""
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import List

DEFAULT_GASES = [
    "Acetone",
    "Ethanol",
    "Methanol",
    "Isopropanol",
    "Toluene",
    "Xylene",
    "MixVOC",
]

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "pipeline_logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all scientific pipelines, world-class analysis, export assets, and optionally build PPTs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gases",
        default=",".join(DEFAULT_GASES),
        help="Comma-separated list of gases to process sequentially",
    )
    parser.add_argument(
        "--skip-scientific",
        action="store_true",
        help="Skip running the scientific pipeline stage",
    )
    parser.add_argument(
        "--skip-world-class",
        action="store_true",
        help="Skip running the world-class analysis stage",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip exporting presentation assets",
    )
    parser.add_argument(
        "--skip-ppt",
        action="store_true",
        help="Skip running the Kevin_acetone_ppt automation",
    )
    parser.add_argument(
        "--sync-to",
        type=Path,
        default=ROOT / "Kevin_acetone_ppt" / "generated_assets" / "exported",
        help="Destination folder to mirror exported assets",
    )
    parser.add_argument(
        "--slides-config",
        type=Path,
        default=ROOT / "Kevin_acetone_ppt" / "config" / "presentation_scientific.yaml",
        help="Slides config to pass to slides_automation CLI",
    )
    parser.add_argument(
        "--slides-flags",
        default="--no-google",
        help="Extra flags for slides_automation CLI (quoted string)",
    )
    return parser.parse_args()


def run_command(label: str, cmd: List[str], cwd: Path, log_path: Path | None = None) -> None:
    print(f"[RUN] {label}: {' '.join(cmd)}")
    kwargs = {
        "cwd": str(cwd),
        "check": True,
    }
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, **kwargs)
    else:
        subprocess.run(cmd, **kwargs)


def run_scientific_stage(gases: List[str]) -> None:
    for gas in gases:
        log_file = LOG_DIR / f"scientific_{gas}.log"
        run_command(
            f"scientific:{gas}",
            ["python", "run_scientific_pipeline.py", "--gas", gas],
            ROOT,
            log_file,
        )


def run_world_class_stage() -> None:
    log_file = LOG_DIR / "world_class.log"
    run_command(
        "world-class",
        ["python", "run_world_class_analysis.py"],
        ROOT,
        log_file,
    )


def run_export_stage(gases: List[str], sync_to: Path) -> None:
    cmd = [
        "python",
        "export_presentation_assets.py",
        "--gases",
        ",".join(gases),
        "--sync-to",
        str(sync_to),
    ]
    run_command("export", cmd, ROOT)


def run_ppt_stage(slides_config: Path, slides_flags: str) -> None:
    kevin_dir = ROOT / "Kevin_acetone_ppt"
    cmd = [
        "python",
        "-m",
        "slides_automation.cli",
        "--config",
        str(slides_config),
    ] + shlex.split(slides_flags)
    run_command("slides_automation", cmd, kevin_dir)


def main() -> None:
    args = parse_args()
    gases = [g.strip() for g in args.gases.split(",") if g.strip()]
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_scientific:
        run_scientific_stage(gases)
    else:
        print("[SKIP] scientific stage")

    if not args.skip_world_class:
        run_world_class_stage()
    else:
        print("[SKIP] world-class stage")

    if not args.skip_export:
        run_export_stage(gases, args.sync_to)
    else:
        print("[SKIP] export stage")

    if not args.skip_ppt:
        run_ppt_stage(args.slides_config, args.slides_flags)
    else:
        print("[SKIP] PPT stage")

    print("[DONE] Full refresh sequence completed")


if __name__ == "__main__":
    main()
